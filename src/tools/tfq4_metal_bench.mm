// Standalone Apple Silicon Metal TFQ4-E8P (2.001 bpw) Parity & Bandwidth Benchmark.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach-o/dyld.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include "../converters/tfq4/tfq4_format.hpp"

struct MetalTfq4Params {
    uint32_t rows, n, k, ldx, ldo, row_off, k_sign_off;
};

static inline double host_sgn(uint32_t seed, uint32_t i) {
    return (((seed ^ i) * 2654435761u) >> 31) ? -1.0 : 1.0;
}

static void fwht128_f64(double* x) {
    for (int len = 1; len < 128; len <<= 1) {
        for (int i = 0; i < 128; i += len * 2) {
            for (int j = 0; j < len; ++j) {
                double u = x[i + j], v = x[i + j + len];
                x[i + j] = u + v; x[i + j + len] = u - v;
            }
        }
    }
    const double inv = 1.0 / std::sqrt(128.0);
    for (int i = 0; i < 128; ++i) x[i] *= inv;
}

static std::vector<double> ref_dequant_e8p(
    const uint32_t* W, const uint16_t* sc, const tfq4::PwV4E8Tables& T, uint32_t N, uint32_t K
) {
    uint32_t n_bc = K / 128u, wpr = K / 16u;
    std::vector<double> Wrot((size_t)N * K, 0.0);
    for (uint32_t r = 0; r < N; ++r) {
        uint32_t br = r / 128u;
        for (uint32_t w = 0; w < wpr; ++w) {
            uint32_t bc = w / 8u;
            double fsc = (double)tfq4::f16_bits_to_f32(sc[(size_t)br * n_bc + bc]);
            uint32_t packed = W[(size_t)r * wpr + w];
            uint32_t c0 = packed & 0xFFFFu, c1 = packed >> 16;
            for (int half_idx = 0; half_idx < 2; ++half_idx) {
                uint32_t code = half_idx ? c1 : c0;
                uint32_t idx = code & 255u, sm = code >> 8;
                for (int d = 0; d < 8; ++d) {
                    double mag = (double)tfq4::f16_bits_to_f32(T.e8p_lut[idx][d]);
                    double sgn = ((sm >> d) & 1u) ? -1.0 : 1.0;
                    Wrot[(size_t)r * K + w * 16u + half_idx * 8u + d] = mag * sgn * fsc;
                }
            }
        }
    }
    std::vector<double> tmp(128);
    for (uint32_t br = 0; br < N / 128u; ++br) for (uint32_t c = 0; c < K; ++c) {
        for (int i = 0; i < 128; ++i) tmp[i] = Wrot[(size_t)(br * 128u + i) * K + c];
        fwht128_f64(tmp.data());
        for (int i = 0; i < 128; ++i) Wrot[(size_t)(br * 128u + i) * K + c] = tmp[i] * host_sgn(0x9E3779B9u, br * 128u + i);
    }
    for (uint32_t r = 0; r < N; ++r) for (uint32_t bc = 0; bc < K / 128u; ++bc) {
        for (int i = 0; i < 128; ++i) tmp[i] = Wrot[(size_t)r * K + bc * 128u + i];
        fwht128_f64(tmp.data());
        for (int i = 0; i < 128; ++i) Wrot[(size_t)r * K + bc * 128u + i] = tmp[i] * host_sgn(0x12345678u, bc * 128u + i);
    }
    return Wrot;
}

static std::string default_metallib(const char* name) {
    uint32_t sz = 0; _NSGetExecutablePath(nullptr, &sz);
    std::string exe(sz, '\0');
    if (_NSGetExecutablePath(&exe[0], &sz) == 0) {
        size_t sl = exe.find_last_of('/');
        if (sl != std::string::npos) return exe.substr(0, sl + 1) + name;
    }
    return name;
}

int main(int argc, char** argv) {
    @autoreleasepool {
        bool do_bench = false;
        std::string vec_path;
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "--bench") == 0) do_bench = true;
            else if (strcmp(argv[i], "--verify-vectors") == 0 && i + 1 < argc) vec_path = argv[++i];
        }

        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { fprintf(stderr, "No Apple Metal device\n"); return 1; }
        NSError* err = nil;
        std::string mpath = default_metallib("tfq4-metal-test.metallib");
        id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:mpath.c_str()]] error:&err];
        if (!lib) { fprintf(stderr, "metallib error: %s\n", [[err localizedDescription] UTF8String]); return 1; }
        auto mk = [&](const char* name) {
            return [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:[NSString stringWithUTF8String:name]] error:&err];
        };
        id<MTLComputePipelineState> pso_fx = mk("tfq4_fwht_x");
        id<MTLComputePipelineState> pso_e8 = mk("tfq4_e8p_dense");
        id<MTLCommandQueue> q = [dev newCommandQueue];

        if (!vec_path.empty()) {
            FILE* vf = fopen(vec_path.c_str(), "rb");
            if (!vf) { fprintf(stderr, "Cannot open VEC4 vector bundle: %s\n", vec_path.c_str()); return 1; }
            uint32_t hdr[2] = {};
            if (fread(hdr, 4, 2, vf) != 2 || hdr[0] != 0x56454334u) {
                fprintf(stderr, "Bad VEC4 header in %s\n", vec_path.c_str()); fclose(vf); return 1;
            }
            tfq4::PwV4E8Tables vtbl{};
            if (fread(&vtbl, 1, sizeof(vtbl), vf) != sizeof(vtbl)) { fprintf(stderr, "Truncated VEC4 tables\n"); fclose(vf); return 1; }
            id<MTLBuffer> blut = [dev newBufferWithBytes:vtbl.e8p_lut length:sizeof(vtbl.e8p_lut) options:MTLResourceStorageModeShared];
            for (uint32_t ci = 0; ci < hdr[1]; ++ci) {
                uint32_t ch[10] = {};
                if (fread(ch, 4, 10, vf) != 10) { fprintf(stderr, "Truncated VEC4 case %u\n", ci); fclose(vf); return 1; }
                uint32_t C = ch[2], rows = ch[3], r_off = ch[4], c_off = ch[5], n_loc = ch[6], k_loc = ch[7];
                (void)ch[1];
                uint32_t n_words = ch[8], n_scales = ch[9];
                std::vector<uint32_t> W(n_words);
                std::vector<uint16_t> sc(n_scales), X((size_t)rows * k_loc);
                std::vector<float> Y_ref((size_t)rows * n_loc);
                if (fread(W.data(), 4, n_words, vf) != n_words ||
                    fread(sc.data(), 2, n_scales, vf) != n_scales ||
                    fread(X.data(), 2, X.size(), vf) != X.size() ||
                    fread(Y_ref.data(), 4, Y_ref.size(), vf) != Y_ref.size()) {
                    fprintf(stderr, "Truncated VEC4 payload case %u\n", ci); fclose(vf); return 1;
                }
                // Extract contiguous [n_loc, k_loc/16] weight slice & [n_loc/128, k_loc/128] scale slice for offset case
                std::vector<uint32_t> W_loc((size_t)n_loc * (k_loc / 16u));
                for (uint32_t r = 0; r < n_loc; ++r) {
                    memcpy(&W_loc[(size_t)r * (k_loc / 16u)], &W[(size_t)(r_off + r) * (C / 16u) + (c_off / 16u)], (k_loc / 16u) * 4u);
                }
                std::vector<uint16_t> sc_loc((size_t)(n_loc / 128u) * (k_loc / 128u));
                for (uint32_t br = 0; br < n_loc / 128u; ++br) {
                    memcpy(&sc_loc[(size_t)br * (k_loc / 128u)], &sc[(size_t)((r_off / 128u) + br) * (C / 128u) + (c_off / 128u)], (k_loc / 128u) * 2u);
                }
                id<MTLBuffer> bW = [dev newBufferWithBytes:W_loc.data() length:W_loc.size() * 4 options:MTLResourceStorageModeShared];
                id<MTLBuffer> bS = [dev newBufferWithBytes:sc_loc.data() length:sc_loc.size() * 2 options:MTLResourceStorageModeShared];
                id<MTLBuffer> bX = [dev newBufferWithBytes:X.data() length:X.size() * 2 options:MTLResourceStorageModeShared];
                id<MTLBuffer> bXr = [dev newBufferWithLength:X.size() * 2 options:MTLResourceStorageModeShared];
                id<MTLBuffer> bY = [dev newBufferWithLength:Y_ref.size() * 4 options:MTLResourceStorageModeShared];
                MetalTfq4Params p{rows, n_loc, k_loc, k_loc, n_loc, r_off, c_off};
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pso_fx];
                [enc setBuffer:bX offset:0 atIndex:0]; [enc setBuffer:bXr offset:0 atIndex:1];
                [enc setBytes:&p length:sizeof(p) atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(k_loc / 128u, rows, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                [enc setComputePipelineState:pso_e8];
                [enc setBuffer:bW offset:0 atIndex:0]; [enc setBuffer:bS offset:0 atIndex:1];
                [enc setBuffer:blut offset:0 atIndex:2]; [enc setBuffer:bXr offset:0 atIndex:3];
                [enc setBuffer:bY offset:0 atIndex:4]; [enc setBytes:&p length:sizeof(p) atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake(n_loc / 128u, (rows + 15u) / 16u, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                const float* got = (const float*)[bY contents];
                double dot = 0.0, na = 0.0, nb = 0.0;
                for (size_t i = 0; i < Y_ref.size(); ++i) {
                    dot += (double)got[i] * Y_ref[i]; na += (double)got[i] * got[i]; nb += (double)Y_ref[i] * Y_ref[i];
                }
                double cos_v = dot / (std::sqrt(na * nb) + 1e-30);
                if (cos_v < 0.99998) {
                    fprintf(stderr, "FAIL Metal VEC4 case=%u r_off=%u c_off=%u cos=%.8f\n", ci, r_off, c_off, cos_v);
                    fclose(vf); return 1;
                }
            }
            fclose(vf);
            printf("PASS Metal TFQ4 golden vectors (%u cases, including r_off & c_off offsets)\n", hdr[1]);
            return 0;
        }

        tfq4::PwV4E8Tables h_tbl;
        tfq4::build_e8p_tables(h_tbl);
        id<MTLBuffer> b_lut = [dev newBufferWithBytes:h_tbl.e8p_lut length:sizeof(h_tbl.e8p_lut) options:MTLResourceStorageModeShared];

        // 1. Exact fp64 parity + bit-exact rows=1 == rows=16 check
        const uint32_t N = 512u, K = 1024u;
        std::mt19937 rng(777);
        std::vector<uint32_t> h_W((size_t)N * (K / 16u));
        std::vector<uint16_t> h_sc((size_t)(N / 128u) * (K / 128u)), h_X(16u * K);
        for (auto& w : h_W) w = rng();
        for (auto& s : h_sc) s = tfq4::f32_to_f16_bits(0.03f + 0.02f * ((float)(rng() & 255u) / 255.0f));
        for (auto& x : h_X) x = tfq4::f32_to_f16_bits(((int)(rng() % 201u) - 100) * 0.01f);
        auto W_ref = ref_dequant_e8p(h_W.data(), h_sc.data(), h_tbl, N, K);

        id<MTLBuffer> b_W = [dev newBufferWithBytes:h_W.data() length:h_W.size() * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_sc = [dev newBufferWithBytes:h_sc.data() length:h_sc.size() * 2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_X = [dev newBufferWithBytes:h_X.data() length:h_X.size() * 2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_xr = [dev newBufferWithLength:16u * K * 2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_Y = [dev newBufferWithLength:16u * N * 4 options:MTLResourceStorageModeShared];

        auto run = [&](uint32_t rows, std::vector<float>& out) {
            MetalTfq4Params p{rows, N, K, K, N, 0, 0};
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso_fx];
            [enc setBuffer:b_X offset:0 atIndex:0]; [enc setBuffer:b_xr offset:0 atIndex:1];
            [enc setBytes:&p length:sizeof(p) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(K / 128u, rows, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            [enc setComputePipelineState:pso_e8];
            [enc setBuffer:b_W offset:0 atIndex:0]; [enc setBuffer:b_sc offset:0 atIndex:1];
            [enc setBuffer:b_lut offset:0 atIndex:2]; [enc setBuffer:b_xr offset:0 atIndex:3];
            [enc setBuffer:b_Y offset:0 atIndex:4]; [enc setBytes:&p length:sizeof(p) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(N / 128u, (rows + 15u) / 16u, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
            out.resize((size_t)rows * N);
            memcpy(out.data(), [b_Y contents], out.size() * 4);
        };

        std::vector<float> y1, y16;
        run(1u, y1); run(16u, y16);
        for (uint32_t r = 0; r < N; ++r) {
            if (y1[r] != y16[r]) { fprintf(stderr, "FAIL TFQ4 Metal r1!=r16 @ %u (%.7f vs %.7f)\n", r, y1[r], y16[r]); return 1; }
        }
        double dot = 0.0, na = 0.0, nb = 0.0;
        for (uint32_t m = 0; m < 16u; ++m) for (uint32_t r = 0; r < N; ++r) {
            double ref = 0.0;
            for (uint32_t c = 0; c < K; ++c) ref += W_ref[(size_t)r * K + c] * (double)tfq4::f16_bits_to_f32(h_X[(size_t)m * K + c]);
            double g = y16[(size_t)m * N + r];
            dot += g * ref; na += g * g; nb += ref * ref;
        }
        double cos_v = dot / (std::sqrt(na * nb) + 1e-30);
        if (cos_v < 0.99999) { fprintf(stderr, "FAIL TFQ4 Metal fp64 parity cos=%.8f\n", cos_v); return 1; }
        // Issue #11: Verify fp16 saturation prevents Inf/NaN when rotated activation exceeds 65504
        {
            std::vector<uint16_t> big_x(K, tfq4::f32_to_f16_bits(10000.0f));
            memcpy([b_X contents], big_x.data(), K * 2);
            std::vector<float> y_big;
            run(1u, y_big);
            for (float v : y_big) {
                if (!std::isfinite(v)) { fprintf(stderr, "FAIL TFQ4 Metal fp16 overflow produced non-finite output\n"); return 1; }
            }
        }
        printf("PASS tfq4_metal_test (%s: 2.0b E8P fp64 parity cos=%.8f + bit-exact r1==r16 + fp16 sat)\n", [[dev name] UTF8String], cos_v);

        if (do_bench) {
            struct Shape { const char* name; uint32_t N, K; };
            Shape shapes[] = {{"MoE-8E-gu", 32768, 4096}, {"Qwen27-gu", 34816, 5120}, {"Qwen27-dn", 5120, 17408}};
            id<MTLBuffer> bx = [dev newBufferWithLength:16ULL * 17408ULL * 2 options:MTLResourceStorageModeShared];
            id<MTLBuffer> bxr = [dev newBufferWithLength:16ULL * 17408ULL * 2 options:MTLResourceStorageModePrivate];
            id<MTLBuffer> by = [dev newBufferWithLength:16ULL * 34816ULL * 4 options:MTLResourceStorageModePrivate];
            for (auto& s : shapes) {
                size_t nw = (size_t)s.N * (s.K / 16u);
                size_t ns = (size_t)(s.N / 128u) * (s.K / 128u);
                id<MTLBuffer> bw = [dev newBufferWithLength:nw * 4 options:MTLResourceStorageModeShared];
                id<MTLBuffer> bsc = [dev newBufferWithLength:ns * 2 options:MTLResourceStorageModeShared];
                double bytes = (double)(nw * 4 + ns * 2);
                for (uint32_t rows : {1u, 16u}) {
                    MetalTfq4Params p{rows, s.N, s.K, s.K, s.N, 0, 0};
                    auto run_n = [&](int reps) {
                        id<MTLCommandBuffer> cb = [q commandBuffer];
                        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                        for (int rep = 0; rep < reps; ++rep) {
                            [enc setComputePipelineState:pso_fx];
                            [enc setBuffer:bx offset:0 atIndex:0]; [enc setBuffer:bxr offset:0 atIndex:1];
                            [enc setBytes:&p length:sizeof(p) atIndex:2];
                            [enc dispatchThreadgroups:MTLSizeMake(s.K / 128u, rows, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                            [enc setComputePipelineState:pso_e8];
                            [enc setBuffer:bw offset:0 atIndex:0]; [enc setBuffer:bsc offset:0 atIndex:1];
                            [enc setBuffer:b_lut offset:0 atIndex:2]; [enc setBuffer:bxr offset:0 atIndex:3];
                            [enc setBuffer:by offset:0 atIndex:4]; [enc setBytes:&p length:sizeof(p) atIndex:5];
                            [enc dispatchThreadgroups:MTLSizeMake(s.N / 128u, (rows + 15u) / 16u, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                        }
                        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                    };
                    run_n(5);
                    auto t0 = std::chrono::steady_clock::now();
                    run_n(20);
                    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 20.0;
                    printf("  Metal TFQ4 %-10s [%5u x %5u] 2.0b rows=%2u | %5.1f MiB | %6.3f ms | %6.1f GB/s\n",
                           s.name, s.N, s.K, rows, bytes / (1024.0 * 1024.0), ms, (bytes / 1e9) / (ms / 1e3));
                }
            }
        }
    }
    return 0;
}
