// Standalone Apple Silicon Metal TFQ4-E8P (2.001 & 2.251 bpw) Parity & Bandwidth Benchmark.
// Verifies table_kind=0 (trained 2.0b), table_kind=2 (True E8P 2.0b), and table_kind=3 (True E8P-1024 2.25b).
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
    const uint32_t* W, const uint16_t* sc, const uint16_t (*lut)[8],
    uint32_t table_kind, uint32_t N, uint32_t K
) {
    const bool is_true = (table_kind == tfq4::PWV4_TABLE_TRUE_E8P || table_kind == tfq4::PWV4_TABLE_TRUE_E8P_1024);
    const bool is_1024 = (table_kind == tfq4::PWV4_TABLE_TRUE_E8P_1024);
    const uint32_t k_codes = is_1024 ? 1024u : 256u;
    std::vector<uint8_t> m_par(k_codes, 0u);
    if (is_true) {
        for (uint32_t k = 0; k < k_codes; ++k) m_par[k] = tfq4::true_e8p_abs_parity(lut[k]);
    }
    const uint32_t n_bc = K / 128u;
    const uint32_t wpb = is_1024 ? 9u : 8u;
    const uint32_t wpr = n_bc * wpb;
    std::vector<double> Wrot((size_t)N * K, 0.0);
    for (uint32_t r = 0; r < N; ++r) {
        uint32_t br = r / 128u;
        for (uint32_t bc = 0; bc < n_bc; ++bc) {
            double fsc = (double)tfq4::f16_bits_to_f32(sc[(size_t)br * n_bc + bc]);
            const uint32_t* blk = W + (size_t)r * wpr + (size_t)bc * wpb;
            uint32_t hi_word = is_1024 ? blk[8] : 0u;
            for (uint32_t sw = 0; sw < 8u; ++sw) {
                uint32_t packed = blk[sw];
                uint32_t c0 = packed & 0xFFFFu, c1 = packed >> 16;
                for (int half_idx = 0; half_idx < 2; ++half_idx) {
                    uint32_t code = half_idx ? c1 : c0;
                    uint32_t idx = code & 255u;
                    if (is_1024) {
                        uint32_t j = sw * 2u + (uint32_t)half_idx;
                        idx |= ((hi_word >> (2u * j)) & 3u) << 8;
                    }
                    if (is_true) {
                        uint32_t s7 = (code >> 8) & 0x7Fu;
                        uint32_t sm = s7 | (((static_cast<uint32_t>(__builtin_popcount(s7)) ^ m_par[idx]) & 1u) << 7);
                        double c_sgn = (code & 0x8000u) ? -1.0 : 1.0;
                        for (int d = 0; d < 8; ++d) {
                            double mag = (double)tfq4::f16_bits_to_f32(lut[idx][d]);
                            double sgn = ((sm >> d) & 1u) ? -1.0 : 1.0;
                            double q = c_sgn * (mag * sgn - 0.25);
                            Wrot[(size_t)r * K + bc * 128u + sw * 16u + half_idx * 8u + d] = q * fsc;
                        }
                    } else {
                        uint32_t sm = code >> 8;
                        for (int d = 0; d < 8; ++d) {
                            double mag = (double)tfq4::f16_bits_to_f32(lut[idx][d]);
                            double sgn = ((sm >> d) & 1u) ? -1.0 : 1.0;
                            Wrot[(size_t)r * K + bc * 128u + sw * 16u + half_idx * 8u + d] = mag * sgn * fsc;
                        }
                    }
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
        id<MTLComputePipelineState> pso_e8_true = mk("tfq4_true_e8p_dense");
        id<MTLComputePipelineState> pso_e8_1024 = mk("tfq4_true_e8p_1024_dense");
        if (!pso_fx || !pso_e8 || !pso_e8_true || !pso_e8_1024) {
            fprintf(stderr, "Failed to create Metal compute pipelines: %s\n", err ? [[err localizedDescription] UTF8String] : "unknown");
            return 1;
        }
        id<MTLCommandQueue> q = [dev newCommandQueue];

        if (!vec_path.empty()) {
            FILE* vf = fopen(vec_path.c_str(), "rb");
            if (!vf) { fprintf(stderr, "Cannot open VEC4 vector bundle: %s\n", vec_path.c_str()); return 1; }
            uint32_t hdr[2] = {};
            if (fread(hdr, 4, 2, vf) != 2 || hdr[0] != 0x56454334u) {
                fprintf(stderr, "Bad VEC4 header in %s\n", vec_path.c_str()); fclose(vf); return 1;
            }
            uint32_t thdr[16] = {};
            if (fread(thdr, 4, 16, vf) != 16) { fprintf(stderr, "Truncated VEC4 table header\n"); fclose(vf); return 1; }
            const uint32_t magic = thdr[0], ver = thdr[1], k_codes = thdr[2], dim = thdr[3], table_kind = thdr[4];
            if (magic != tfq4::MAGIC_PWV4 || ver != tfq4::FORMAT_VER || dim != 8u || table_kind > tfq4::PWV4_TABLE_MAX_KNOWN) {
                fprintf(stderr, "Unsupported PWV4 table in VEC4 bundle: magic=0x%08x ver=%u table_kind=%u\n", magic, ver, table_kind);
                fclose(vf); return 1;
            }
            const bool is_1024 = (table_kind == tfq4::PWV4_TABLE_TRUE_E8P_1024);
            if ((is_1024 && k_codes != 1024u) || (!is_1024 && k_codes != 256u)) {
                fprintf(stderr, "Mismatched k_codes=%u for table_kind=%u in VEC4 bundle\n", k_codes, table_kind);
                fclose(vf); return 1;
            }
            const size_t tbl_bytes = is_1024 ? sizeof(tfq4::PwV4E8Tables1024) : sizeof(tfq4::PwV4E8Tables);
            const size_t lut_bytes = (size_t)k_codes * 8u * 2u;
            std::vector<uint8_t> tbl_rest(tbl_bytes - 64u);
            if (fread(tbl_rest.data(), 1, tbl_rest.size(), vf) != tbl_rest.size()) {
                fprintf(stderr, "Truncated VEC4 tables\n"); fclose(vf); return 1;
            }
            id<MTLComputePipelineState> pso_vec = is_1024
                ? pso_e8_1024
                : ((table_kind == tfq4::PWV4_TABLE_TRUE_E8P) ? pso_e8_true : pso_e8);
            id<MTLBuffer> blut = [dev newBufferWithBytes:tbl_rest.data() length:lut_bytes options:MTLResourceStorageModeShared];
            const uint32_t wpb = is_1024 ? 9u : 8u;

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
                const uint32_t full_wpr = (C / 128u) * wpb;
                const uint32_t loc_wpr  = (k_loc / 128u) * wpb;
                const uint32_t c_w_off  = (c_off / 128u) * wpb;
                std::vector<uint32_t> W_loc((size_t)n_loc * loc_wpr);
                for (uint32_t r = 0; r < n_loc; ++r) {
                    memcpy(&W_loc[(size_t)r * loc_wpr], &W[(size_t)(r_off + r) * full_wpr + c_w_off], (size_t)loc_wpr * 4u);
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
                [enc setComputePipelineState:pso_vec];
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
                    fprintf(stderr, "FAIL Metal VEC4 table_kind=%u case=%u r_off=%u c_off=%u cos=%.8f\n",
                            table_kind, ci, r_off, c_off, cos_v);
                    fclose(vf); return 1;
                }
            }
            fclose(vf);
            printf("PASS Metal TFQ4 golden vectors (table_kind=%u, %u cases, including r_off & c_off offsets)\n",
                   table_kind, hdr[1]);
            return 0;
        }

        tfq4::PwV4E8Tables h_tbl_trained{}, h_tbl_true{};
        tfq4::PwV4E8Tables1024 h_tbl_1024{};
        tfq4::build_e8p_tables(h_tbl_trained);
        tfq4::build_true_e8p_tables(h_tbl_true);
        tfq4::build_true_e8p_1024_tables(h_tbl_1024);
        if (std::memcmp(h_tbl_true.e8p_lut, h_tbl_1024.e8p_lut, sizeof(h_tbl_true.e8p_lut)) != 0) {
            fprintf(stderr, "FAIL: first 256 entries of PwV4E8Tables1024 diverge from PwV4E8Tables (table_kind=2)\n");
            return 1;
        }
        id<MTLBuffer> b_lut_trained = [dev newBufferWithBytes:h_tbl_trained.e8p_lut length:sizeof(h_tbl_trained.e8p_lut) options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_lut_true = [dev newBufferWithBytes:h_tbl_true.e8p_lut length:sizeof(h_tbl_true.e8p_lut) options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_lut_1024 = [dev newBufferWithBytes:h_tbl_1024.e8p_lut length:sizeof(h_tbl_1024.e8p_lut) options:MTLResourceStorageModeShared];

        // 1. Exact fp64 parity + bit-exact rows=1 == rows=16 == rows=17 + fp16 saturation for kind=0, 2, and 3
        const uint32_t N = 512u, K = 1024u, MAX_R = 17u;
        std::mt19937 rng(777);
        std::vector<uint32_t> h_W20((size_t)N * (K / 16u));
        std::vector<uint32_t> h_W225((size_t)N * (K / 128u) * 9u);
        std::vector<uint16_t> h_sc((size_t)(N / 128u) * (K / 128u)), h_X((size_t)MAX_R * K);
        for (auto& w : h_W20) w = rng();
        for (auto& w : h_W225) w = rng();
        for (auto& s : h_sc) s = tfq4::f32_to_f16_bits(0.03f + 0.02f * ((float)(rng() & 255u) / 255.0f));
        for (auto& x : h_X) x = tfq4::f32_to_f16_bits(((int)(rng() % 201u) - 100) * 0.01f);

        id<MTLBuffer> b_W20  = [dev newBufferWithBytes:h_W20.data()  length:h_W20.size() * 4  options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_W225 = [dev newBufferWithBytes:h_W225.data() length:h_W225.size() * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_sc   = [dev newBufferWithBytes:h_sc.data()   length:h_sc.size() * 2   options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_X    = [dev newBufferWithBytes:h_X.data()    length:h_X.size() * 2    options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_xr   = [dev newBufferWithLength:(size_t)MAX_R * K * 2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_Y    = [dev newBufferWithLength:(size_t)MAX_R * N * 4 options:MTLResourceStorageModeShared];

        struct KindCfg {
            const char* name;
            const char* rate_lbl;
            uint32_t kind;
            uint32_t wpb;
            const uint16_t (*lut_host)[8];
            const uint32_t* w_host;
            id<MTLBuffer> bW;
            id<MTLBuffer> lut;
            id<MTLComputePipelineState> pso;
        };
        KindCfg kinds[] = {
            {"trained/kind=0",       "2.00b", tfq4::PWV4_TABLE_LEGACY_TRAINED, 8u, h_tbl_trained.e8p_lut, h_W20.data(),  b_W20,  b_lut_trained, pso_e8},
            {"true-E8P/kind=2",      "2.00b", tfq4::PWV4_TABLE_TRUE_E8P,       8u, h_tbl_true.e8p_lut,    h_W20.data(),  b_W20,  b_lut_true,    pso_e8_true},
            {"true-E8P-1024/kind=3", "2.25b", tfq4::PWV4_TABLE_TRUE_E8P_1024,  9u, h_tbl_1024.e8p_lut,    h_W225.data(), b_W225, b_lut_1024,    pso_e8_1024},
        };

        for (auto& kc : kinds) {
            memcpy([b_X contents], h_X.data(), h_X.size() * 2);
            auto W_ref = ref_dequant_e8p(kc.w_host, h_sc.data(), kc.lut_host, kc.kind, N, K);

            auto run = [&](uint32_t rows, std::vector<float>& out) {
                MetalTfq4Params p{rows, N, K, K, N, 0, 0};
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pso_fx];
                [enc setBuffer:b_X offset:0 atIndex:0]; [enc setBuffer:b_xr offset:0 atIndex:1];
                [enc setBytes:&p length:sizeof(p) atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(K / 128u, rows, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                [enc setComputePipelineState:kc.pso];
                [enc setBuffer:kc.bW offset:0 atIndex:0]; [enc setBuffer:b_sc offset:0 atIndex:1];
                [enc setBuffer:kc.lut offset:0 atIndex:2]; [enc setBuffer:b_xr offset:0 atIndex:3];
                [enc setBuffer:b_Y offset:0 atIndex:4]; [enc setBytes:&p length:sizeof(p) atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake(N / 128u, (rows + 15u) / 16u, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                out.resize((size_t)rows * N);
                memcpy(out.data(), [b_Y contents], out.size() * 4);
            };

            std::vector<float> y1, y16, y17;
            run(1u, y1); run(16u, y16); run(17u, y17);
            for (uint32_t r = 0; r < N; ++r) {
                if (y1[r] != y16[r]) {
                    fprintf(stderr, "FAIL TFQ4 Metal (%s) r1!=r16 @ %u (%.7f vs %.7f)\n", kc.name, r, y1[r], y16[r]);
                    return 1;
                }
                if (y1[r] != y17[r]) {
                    fprintf(stderr, "FAIL TFQ4 Metal (%s) r1!=r17 @ %u (%.7f vs %.7f)\n", kc.name, r, y1[r], y17[r]);
                    return 1;
                }
            }
            double dot = 0.0, na = 0.0, nb = 0.0;
            for (uint32_t m = 0; m < MAX_R; ++m) for (uint32_t r = 0; r < N; ++r) {
                double ref = 0.0;
                for (uint32_t c = 0; c < K; ++c) ref += W_ref[(size_t)r * K + c] * (double)tfq4::f16_bits_to_f32(h_X[(size_t)m * K + c]);
                double g = y17[(size_t)m * N + r];
                dot += g * ref; na += g * g; nb += ref * ref;
            }
            double cos_v = dot / (std::sqrt(na * nb) + 1e-30);
            if (cos_v < 0.9999999) {
                fprintf(stderr, "FAIL TFQ4 Metal (%s) fp64 parity cos=%.8f (< 0.9999999)\n", kc.name, cos_v);
                return 1;
            }
            // Issue #11: Verify fp16 saturation prevents Inf/NaN when rotated activation exceeds 65504
            {
                std::vector<uint16_t> big_x(K, tfq4::f32_to_f16_bits(10000.0f));
                memcpy([b_X contents], big_x.data(), K * 2);
                std::vector<float> y_big;
                run(1u, y_big);
                for (float v : y_big) {
                    if (!std::isfinite(v)) {
                        fprintf(stderr, "FAIL TFQ4 Metal (%s) fp16 overflow produced non-finite output\n", kc.name);
                        return 1;
                    }
                }
            }
            printf("PASS tfq4_metal_test (%s: %s %s fp64 parity cos=%.8f + bit-exact r1==r16==r17 + fp16 sat)\n",
                   [[dev name] UTF8String], kc.rate_lbl, kc.name, cos_v);
        }

        if (do_bench) {
            struct Shape { const char* name; uint32_t N, K; };
            Shape shapes[] = {{"MoE-8E-gu", 32768, 4096}, {"Qwen27-gu", 34816, 5120}, {"Qwen27-dn", 5120, 17408}};
            id<MTLBuffer> bx = [dev newBufferWithLength:16ULL * 17408ULL * 2 options:MTLResourceStorageModeShared];
            id<MTLBuffer> bxr = [dev newBufferWithLength:16ULL * 17408ULL * 2 options:MTLResourceStorageModePrivate];
            id<MTLBuffer> by = [dev newBufferWithLength:16ULL * 34816ULL * 4 options:MTLResourceStorageModePrivate];
            for (auto& kc : kinds) {
                for (auto& s : shapes) {
                    size_t nw = (size_t)s.N * (s.K / 128u) * kc.wpb;
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
                                [enc setComputePipelineState:kc.pso];
                                [enc setBuffer:bw offset:0 atIndex:0]; [enc setBuffer:bsc offset:0 atIndex:1];
                                [enc setBuffer:kc.lut offset:0 atIndex:2]; [enc setBuffer:bxr offset:0 atIndex:3];
                                [enc setBuffer:by offset:0 atIndex:4]; [enc setBytes:&p length:sizeof(p) atIndex:5];
                                [enc dispatchThreadgroups:MTLSizeMake(s.N / 128u, (rows + 15u) / 16u, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                            }
                            [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                        };
                        run_n(5);
                        auto t0 = std::chrono::steady_clock::now();
                        run_n(20);
                        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 20.0;
                        printf("  Metal TFQ4 %-20s %-10s [%5u x %5u] %s rows=%2u | %5.1f MiB | %6.3f ms | %6.1f GB/s\n",
                               kc.name, s.name, s.N, s.K, kc.rate_lbl, rows, bytes / (1024.0 * 1024.0), ms, (bytes / 1e9) / (ms / 1e3));
                    }
                }
            }
        }
    }
    return 0;
}
