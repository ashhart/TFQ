// Standalone Apple Silicon Metal TFQ3 Prefill + Zero-Repack Tensor-Parallel (Row-TP & Col-TP) Parity & Benchmark.
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

struct PwV3Tables {
    uint32_t magic, version;
    float a3, b3, a4, b4, a5, b5;
    uint32_t pad0[8];
    uint32_t lut3[256], lut4[1024], lut5[1024];
    uint16_t c64_3[64][4], c64_4[64][4], c64_5[64][4];
};

struct MetalDenseParams {
    uint32_t rows, n, k, ldx, ldo, row_off, k_sign_off, strip_off, sc_row_off, bc0, bc1, n_bc_total, ksplit;
};

static inline uint16_t f32_to_f16(float f) { __fp16 h = (__fp16)f; uint16_t u; memcpy(&u, &h, 2); return u; }
static inline float f16_to_f32(uint16_t u) { __fp16 h; memcpy(&h, &u, 2); return (float)h; }
static inline uint32_t pack_h2(float x, float y) { return (uint32_t)f32_to_f16(x) | ((uint32_t)f32_to_f16(y) << 16); }
static inline double host_sgn(uint32_t seed, uint32_t i) { return (((seed ^ i) * 2654435761u) >> 31) ? -1.0 : 1.0; }

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

static void init_synth_tables(PwV3Tables& t) {
    memset(&t, 0, sizeof(t));
    t.magic = 0x33565750u; t.version = 3u;
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (int i = 0; i < 256; ++i)  t.lut3[i] = pack_h2(nd(rng), nd(rng));
    for (int i = 0; i < 1024; ++i) t.lut4[i] = pack_h2(nd(rng), nd(rng));
    for (int i = 0; i < 1024; ++i) t.lut5[i] = pack_h2(nd(rng), nd(rng));
    auto fc = [](uint16_t dst[64][4], float a, float b) {
        for (int c = 0; c < 64; ++c) {
            float t0 = (float)(((c >> 0) & 3) * 2 - 3) / 3.0f;
            float t1 = (float)(((c >> 2) & 3) * 2 - 3) / 3.0f;
            float t2 = (float)(((c >> 4) & 1) * 2 - 1);
            float t3 = (float)(((c >> 5) & 1) * 2 - 1);
            dst[c][0] = f32_to_f16(1.0f + a * t2); dst[c][1] = f32_to_f16(1.0f + a * t3);
            dst[c][2] = f32_to_f16(b * t0);        dst[c][3] = f32_to_f16(b * t1);
        }
    };
    fc(t.c64_3, 0.22f, 0.38f); fc(t.c64_4, 0.18f, 0.30f); fc(t.c64_5, 0.10f, 0.18f);
}

static std::vector<double> ref_dequant(uint32_t bits, const uint32_t* W, const uint16_t* sc, const PwV3Tables& T, uint32_t N, uint32_t K) {
    uint32_t n_bc = K / 128u, wpb = bits * 2u;
    std::vector<double> Wrot((size_t)N * K, 0.0);
    for (uint32_t strip = 0; strip < N / 16u; ++strip) {
        for (uint32_t lane = 0; lane < 32u; ++lane) {
            uint32_t hx = (lane & 8u) + ((lane & 1u) << 2);
            uint32_t hy = ((lane & 16u) >> 2) | ((lane >> 1) & 3u);
            uint32_t st = 0x13579BDFu ^ (lane * 0x9E3779B1u);
            const uint32_t* ws = W + (size_t)strip * n_bc * wpb * 32ull;
            for (uint32_t bc = 0; bc < n_bc; ++bc) {
                double fsc = (double)f16_to_f32(sc[(size_t)(strip >> 3) * n_bc + bc]);
                const uint32_t* base = ws + (size_t)bc * wpb * 32ull;
                uint32_t words[10];
                if (bits == 4u) {
                    for (uint32_t k = 0; k < 4u; ++k) words[k] = base[lane * 4u + k];
                    for (uint32_t k = 0; k < 4u; ++k) words[k + 4u] = base[128u + lane * 4u + k];
                } else {
                    for (uint32_t w = 0; w < wpb; ++w) words[w] = base[w * 32u + lane];
                }
                uint32_t bp = 0;
                for (uint32_t kt = 0; kt < 8u; ++kt) {
                    uint32_t syms[4];
                    for (int p = 0; p < 4; ++p) {
                        uint32_t s = 0;
                        for (uint32_t bi = 0; bi < 2u * bits; ++bi) { s = (s << 1) | ((words[bp >> 5] >> (31u - (bp & 31u))) & 1u); bp++; }
                        syms[p] = s;
                    }
                    uint32_t h0 = st * 0x9E3779B1u;
                    uint32_t s1 = (st << (2u * bits)) | syms[0], h1 = s1 * 0x9E3779B1u;
                    uint32_t s2 = (s1 << (2u * bits)) | syms[1], h2 = s2 * 0x9E3779B1u;
                    uint32_t s3 = (s2 << (2u * bits)) | syms[2], h3 = s3 * 0x9E3779B1u;
                    st = (s3 << (2u * bits)) | syms[3];
                    const uint16_t* c0 = (bits == 3u) ? T.c64_3[(h0 >> 16) & 63u] : ((bits == 4u) ? T.c64_4[(h0 >> 16) & 63u] : T.c64_5[(h0 >> 16) & 63u]);
                    const uint16_t* c2 = (bits == 3u) ? T.c64_3[(h2 >> 16) & 63u] : ((bits == 4u) ? T.c64_4[(h2 >> 16) & 63u] : T.c64_5[(h2 >> 16) & 63u]);
                    auto dec = [&](uint32_t sym, uint32_t h, const uint16_t* c, double& ox, double& oy) {
                        uint32_t u = (bits == 3u) ? T.lut3[((sym << 2) ^ (h >> 24)) & 255u]
                                   : ((bits == 4u) ? T.lut4[((sym << 2) ^ (h >> 22)) & 1023u]
                                                   : T.lut5[(sym ^ (h >> 22)) & 1023u]);
                        float lx = f16_to_f32((uint16_t)(u & 0xFFFFu)), ly = f16_to_f32((uint16_t)(u >> 16));
                        ox = (double)(float)(__fp16)(lx * f16_to_f32(c[0]) + f16_to_f32(c[2])) * fsc;
                        oy = (double)(float)(__fp16)(ly * f16_to_f32(c[1]) + f16_to_f32(c[3])) * fsc;
                    };
                    double v[8];
                    dec(syms[0], h0, c0, v[0], v[1]); dec(syms[1], h1, c0, v[2], v[3]);
                    dec(syms[2], h2, c2, v[4], v[5]); dec(syms[3], h3, c2, v[6], v[7]);
                    uint32_t cc = bc * 128u + kt * 16u + hx;
                    uint32_t r0 = strip * 16u + hy, r1 = r0 + 8u;
                    for (int j = 0; j < 4; ++j) { Wrot[(size_t)r0 * K + cc + j] = v[j]; Wrot[(size_t)r1 * K + cc + j] = v[j + 4]; }
                }
            }
        }
    }
    // Undo 2D FWHT-128 -> spatial W
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
        std::string mpath = default_metallib("tfq3-metal-test.metallib");
        id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:mpath.c_str()]] error:&err];
        if (!lib) { fprintf(stderr, "metallib error: %s\n", [[err localizedDescription] UTF8String]); return 1; }
        auto mk = [&](const char* name) {
            id<MTLFunction> f = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
            return [dev newComputePipelineStateWithFunction:f error:&err];
        };
        id<MTLComputePipelineState> pso_fx = mk("tfq3_fwht_x");
        id<MTLComputePipelineState> pso_fy = mk("tfq3_fwht_y");
        id<MTLComputePipelineState> pso_dec[6] = {}, pso_mm[6] = {};
        pso_dec[3] = mk("tfq3_sk8_tp_3b");
        pso_dec[4] = mk("tfq3_sk8_tp_4b");
        pso_dec[5] = mk("tfq3_sk8_tp_5b");
        pso_mm[3]  = mk("tfq3_prefill_tp_3b");
        pso_mm[4]  = mk("tfq3_prefill_tp_4b");
        pso_mm[5]  = mk("tfq3_prefill_tp_5b");
        id<MTLCommandQueue> q = [dev newCommandQueue];

        if (!vec_path.empty()) {
            FILE* vf = fopen(vec_path.c_str(), "rb");
            if (!vf) { fprintf(stderr, "Cannot open vector bundle: %s\n", vec_path.c_str()); return 1; }
            uint32_t hdr[2] = {};
            if (fread(hdr, 4, 2, vf) != 2 || hdr[0] != 0x56454333u) {
                fprintf(stderr, "Bad VEC3 vector header in %s\n", vec_path.c_str()); fclose(vf); return 1;
            }
            PwV3Tables vtbl{};
            uint8_t pad_tbl[16384];
            if (fread(pad_tbl, 1, 16384, vf) != 16384) { fprintf(stderr, "Truncated VEC3 tables\n"); fclose(vf); return 1; }
            memcpy(&vtbl, pad_tbl, sizeof(vtbl));
            id<MTLBuffer> bt = [dev newBufferWithBytes:&vtbl length:sizeof(vtbl) options:MTLResourceStorageModeShared];
            for (uint32_t ci = 0; ci < hdr[1]; ++ci) {
                uint32_t ch[10] = {};
                if (fread(ch, 4, 10, vf) != 10) { fprintf(stderr, "Truncated VEC3 case %u\n", ci); fclose(vf); return 1; }
                uint32_t bits = ch[0], C = ch[2], rows = ch[3], r_off = ch[4], c_off = ch[5], n_loc = ch[6], k_loc = ch[7];
                (void)ch[1];
                uint32_t n_words = ch[8], n_scales = ch[9];
                std::vector<uint32_t> W(n_words);
                std::vector<uint16_t> sc(n_scales), X((size_t)rows * k_loc);
                std::vector<float> Y_ref((size_t)rows * n_loc);
                if (fread(W.data(), 4, n_words, vf) != n_words ||
                    fread(sc.data(), 2, n_scales, vf) != n_scales ||
                    fread(X.data(), 2, X.size(), vf) != X.size() ||
                    fread(Y_ref.data(), 4, Y_ref.size(), vf) != Y_ref.size()) {
                    fprintf(stderr, "Truncated VEC3 payload case %u\n", ci); fclose(vf); return 1;
                }
                id<MTLBuffer> bW = [dev newBufferWithBytes:W.data() length:W.size() * 4 options:MTLResourceStorageModeShared];
                id<MTLBuffer> bS = [dev newBufferWithBytes:sc.data() length:sc.size() * 2 options:MTLResourceStorageModeShared];
                id<MTLBuffer> bX = [dev newBufferWithBytes:X.data() length:X.size() * 2 options:MTLResourceStorageModeShared];
                id<MTLBuffer> bXr = [dev newBufferWithLength:X.size() * 2 options:MTLResourceStorageModeShared];
                id<MTLBuffer> bYr = [dev newBufferWithLength:8ULL * rows * n_loc * 4 options:MTLResourceStorageModePrivate];
                id<MTLBuffer> bY = [dev newBufferWithLength:Y_ref.size() * 4 options:MTLResourceStorageModeShared];
                for (bool decode : {true, false}) {
                    uint32_t ks = decode ? std::min<uint32_t>(8u, k_loc / 128u) : 1u;
                    MetalDenseParams p{rows, n_loc, k_loc, k_loc, n_loc, r_off, c_off, r_off >> 4, r_off >> 7, c_off >> 7, (c_off + k_loc) >> 7, C >> 7, ks};
                    id<MTLCommandBuffer> cb = [q commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc setComputePipelineState:pso_fx];
                    [enc setBuffer:bX offset:0 atIndex:0]; [enc setBuffer:bXr offset:0 atIndex:1];
                    [enc setBytes:&p length:sizeof(p) atIndex:2];
                    [enc dispatchThreadgroups:MTLSizeMake(k_loc / 128u, rows, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                    if (decode) {
                        [enc setComputePipelineState:pso_dec[bits]];
                        [enc setBuffer:bW offset:0 atIndex:0]; [enc setBuffer:bS offset:0 atIndex:1];
                        [enc setBuffer:bt offset:0 atIndex:2]; [enc setBuffer:bXr offset:0 atIndex:3];
                        [enc setBuffer:bYr offset:0 atIndex:4]; [enc setBytes:&p length:sizeof(p) atIndex:5];
                        [enc dispatchThreadgroups:MTLSizeMake(n_loc / 64u, ks, (rows + 15u) / 16u) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                        [enc setComputePipelineState:pso_fy];
                        [enc setBuffer:bYr offset:0 atIndex:0]; [enc setBuffer:bY offset:0 atIndex:1];
                        [enc setBytes:&p length:sizeof(p) atIndex:2];
                        [enc dispatchThreadgroups:MTLSizeMake(n_loc / 128u, rows, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                    } else {
                        [enc setComputePipelineState:pso_mm[bits]];
                        [enc setBuffer:bW offset:0 atIndex:0]; [enc setBuffer:bS offset:0 atIndex:1];
                        [enc setBuffer:bt offset:0 atIndex:2]; [enc setBuffer:bXr offset:0 atIndex:3];
                        [enc setBuffer:bY offset:0 atIndex:4]; [enc setBytes:&p length:sizeof(p) atIndex:5];
                        [enc dispatchThreadgroups:MTLSizeMake(n_loc / 128u, (rows + 31u) / 32u, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    }
                    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                    const float* got = (const float*)[bY contents];
                    double dot = 0.0, na = 0.0, nb = 0.0;
                    for (size_t i = 0; i < Y_ref.size(); ++i) {
                        dot += (double)got[i] * Y_ref[i]; na += (double)got[i] * got[i]; nb += (double)Y_ref[i] * Y_ref[i];
                    }
                    double cos_v = dot / (std::sqrt(na * nb) + 1e-30);
                    if (cos_v < 0.99998) {
                        fprintf(stderr, "FAIL Metal VEC3 case=%u bits=%u %s r_off=%u c_off=%u cos=%.8f\n",
                                ci, bits, decode ? "sk8" : "prefill", r_off, c_off, cos_v);
                        fclose(vf); return 1;
                    }
                }
            }
            fclose(vf);
            printf("PASS Metal TFQ3 golden vectors (%u cases across 3b/4b/5b sk8 + prefill, including r_off & bc>0 resume)\n", hdr[1]);
            return 0;
        }

        PwV3Tables h_tbl; init_synth_tables(h_tbl);
        id<MTLBuffer> b_tbl = [dev newBufferWithBytes:&h_tbl length:sizeof(h_tbl) options:MTLResourceStorageModeShared];

        std::mt19937 rng(1234);
        for (uint32_t bits : {3u, 4u, 5u}) {
            const uint32_t N = 512u, K = 1024u, rows = 64u;
            size_t nw = (size_t)(N / 16u) * (K / 128u) * (bits * 2u) * 32ull;
            size_t ns = (size_t)(N / 128u) * (K / 128u);
            std::vector<uint32_t> h_W(nw);
            std::vector<uint16_t> h_sc(ns), h_X(rows * K);
            for (auto& w : h_W) w = rng();
            for (auto& s : h_sc) s = f32_to_f16(0.03f + 0.02f * ((float)(rng() & 255u) / 255.0f));
            for (auto& x : h_X) x = f32_to_f16(((int)(rng() % 201u) - 100) * 0.01f);
            auto W_ref = ref_dequant(bits, h_W.data(), h_sc.data(), h_tbl, N, K);

            id<MTLBuffer> b_W = [dev newBufferWithBytes:h_W.data() length:nw * 4 options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_sc = [dev newBufferWithBytes:h_sc.data() length:ns * 2 options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_X = [dev newBufferWithBytes:h_X.data() length:rows * K * 2 options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_xr = [dev newBufferWithLength:rows * K * 2 options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_yr = [dev newBufferWithLength:8ULL * 32ULL * N * 4 options:MTLResourceStorageModePrivate];
            id<MTLBuffer> b_Y = [dev newBufferWithLength:rows * N * 4 options:MTLResourceStorageModeShared];

            auto run_tp = [&](bool decode, uint32_t r_cnt, uint32_t n_off, uint32_t n_loc, uint32_t k_off, uint32_t k_loc, std::vector<float>& out) {
                uint32_t ks = decode ? 8u : 1u;
                MetalDenseParams p{r_cnt, n_loc, k_loc, K, n_loc, n_off, k_off, n_off >> 4, n_off >> 7, k_off >> 7, (k_off + k_loc) >> 7, K >> 7, ks};
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pso_fx];
                [enc setBuffer:b_X offset:k_off * 2 atIndex:0]; [enc setBuffer:b_xr offset:0 atIndex:1];
                [enc setBytes:&p length:sizeof(p) atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(k_loc / 128u, r_cnt, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                if (decode) {
                    [enc setComputePipelineState:pso_dec[bits]];
                    [enc setBuffer:b_W offset:0 atIndex:0]; [enc setBuffer:b_sc offset:0 atIndex:1];
                    [enc setBuffer:b_tbl offset:0 atIndex:2]; [enc setBuffer:b_xr offset:0 atIndex:3];
                    [enc setBuffer:b_yr offset:0 atIndex:4]; [enc setBytes:&p length:sizeof(p) atIndex:5];
                    [enc dispatchThreadgroups:MTLSizeMake(n_loc / 64u, ks, (r_cnt + 15u) / 16u) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    [enc setComputePipelineState:pso_fy];
                    [enc setBuffer:b_yr offset:0 atIndex:0]; [enc setBuffer:b_Y offset:0 atIndex:1];
                    [enc setBytes:&p length:sizeof(p) atIndex:2];
                    [enc dispatchThreadgroups:MTLSizeMake(n_loc / 128u, r_cnt, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                } else {
                    [enc setComputePipelineState:pso_mm[bits]];
                    [enc setBuffer:b_W offset:0 atIndex:0]; [enc setBuffer:b_sc offset:0 atIndex:1];
                    [enc setBuffer:b_tbl offset:0 atIndex:2]; [enc setBuffer:b_xr offset:0 atIndex:3];
                    [enc setBuffer:b_Y offset:0 atIndex:4]; [enc setBytes:&p length:sizeof(p) atIndex:5];
                    [enc dispatchThreadgroups:MTLSizeMake(n_loc / 128u, (r_cnt + 31u) / 32u, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                }
                [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                out.resize((size_t)r_cnt * n_loc);
                memcpy(out.data(), [b_Y contents], (size_t)r_cnt * n_loc * 4);
            };

            std::vector<float> y_dec1, y_dec16, y_dec32, y_full;
            run_tp(true, 1u, 0, N, 0, K, y_dec1);
            run_tp(true, 16u, 0, N, 0, K, y_dec16);
            run_tp(true, 32u, 0, N, 0, K, y_dec32);
            // Bit-exact speculative verification parity: row 0 at rows=1 MUST equal row 0 at rows=16 and rows=32!
            for (uint32_t r = 0; r < N; ++r) {
                if (y_dec1[r] != y_dec16[r] || y_dec1[r] != y_dec32[r]) {
                    fprintf(stderr, "FAIL Metal decode bit-exact r1!=r16/r32 %ub @ %u\n", bits, r); return 1;
                }
            }
            run_tp(false, rows, 0, N, 0, K, y_full);
            double dot = 0.0, na = 0.0, nb = 0.0, dot_d = 0.0, nda = 0.0, ndb = 0.0;
            for (uint32_t m = 0; m < rows; ++m) for (uint32_t r = 0; r < N; ++r) {
                double ref = 0.0;
                for (uint32_t c = 0; c < K; ++c) ref += W_ref[(size_t)r * K + c] * (double)f16_to_f32(h_X[(size_t)m * K + c]);
                double g = y_full[(size_t)m * N + r];
                dot += g * ref; na += g * g; nb += ref * ref;
                if (m < 16u) { double gd = y_dec16[(size_t)m * N + r]; dot_d += gd * ref; nda += gd * gd; ndb += ref * ref; }
            }
            double cos_full = dot / (std::sqrt(na * nb) + 1e-30);
            double cos_dec = dot_d / (std::sqrt(nda * ndb) + 1e-30);
            if (cos_full < 0.99999 || cos_dec < 0.99999) { fprintf(stderr, "FAIL Metal %ub prefill=%.8f dec=%.8f\n", bits, cos_full, cos_dec); return 1; }

            // Verify Col-TP=4 (bit-exact concat) and Row-TP=4 (AllReduce SUM == full)
            uint32_t n_loc = N / 4u;
            for (uint32_t rk = 0; rk < 4u; ++rk) {
                std::vector<float> part;
                run_tp(false, rows, rk * n_loc, n_loc, 0, K, part);
                for (uint32_t m = 0; m < rows; ++m) for (uint32_t r = 0; r < n_loc; ++r) {
                    if (part[(size_t)m * n_loc + r] != y_full[(size_t)m * N + rk * n_loc + r]) {
                        fprintf(stderr, "FAIL Metal Col-TP=4 %ub mismatch\n", bits); return 1;
                    }
                }
            }
            uint32_t k_loc = K / 4u;
            std::vector<float> y_sum(rows * N, 0.0f);
            for (uint32_t rk = 0; rk < 4u; ++rk) {
                std::vector<float> part;
                run_tp(false, rows, 0, N, rk * k_loc, k_loc, part);
                for (size_t i = 0; i < y_sum.size(); ++i) y_sum[i] += part[i];
            }
            double dtp = 0.0, n1 = 0.0, n2 = 0.0;
            for (size_t i = 0; i < y_sum.size(); ++i) { dtp += (double)y_sum[i] * y_full[i]; n1 += (double)y_sum[i] * y_sum[i]; n2 += (double)y_full[i] * y_full[i]; }
            double cos_tp = dtp / (std::sqrt(n1 * n2) + 1e-30);
            if (cos_tp < 0.999999) { fprintf(stderr, "FAIL Metal Row-TP=4 %ub cos=%.8f\n", bits, cos_tp); return 1; }

            // Issue #11: Verify fp16 saturation prevents Inf/NaN when rotated activation exceeds 65504
            if (bits == 4u) {
                std::vector<uint16_t> big_x(K, f32_to_f16(10000.0f));
                memcpy([b_X contents], big_x.data(), K * 2);
                std::vector<float> y_big;
                run_tp(true, 1u, 0, N, 0, K, y_big);
                for (float v : y_big) {
                    if (!std::isfinite(v)) { fprintf(stderr, "FAIL Metal TFQ3 fp16 overflow produced non-finite output\n"); return 1; }
                }
            }
        }
        printf("PASS tfq3_metal_test (%s: 3b/4b/5b decode r1==r16==r32 bit-exact + simdgroup_matrix prefill32 vs fp64 + Col-TP=4 + Row-TP=4 + fp16 sat)\n", [[dev name] UTF8String]);

        if (do_bench) {
            struct Shape { const char* name; uint32_t N, K; };
            Shape shapes[] = {{"gate+up", 34816, 5120}, {"down", 5120, 17408}, {"qkv", 10240, 5120}};
            id<MTLBuffer> b_X = [dev newBufferWithLength:512ULL * 17408ULL * 2 options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_xr = [dev newBufferWithLength:512ULL * 17408ULL * 2 options:MTLResourceStorageModePrivate];
            id<MTLBuffer> b_yr = [dev newBufferWithLength:8ULL * 32ULL * 34816ULL * 4 options:MTLResourceStorageModePrivate];
            id<MTLBuffer> b_Y = [dev newBufferWithLength:512ULL * 34816ULL * 4 options:MTLResourceStorageModePrivate];
            for (auto& s : shapes) {
                size_t nw = (size_t)(s.N / 16u) * (s.K / 128u) * 8ull * 32ull;
                size_t ns = (size_t)(s.N / 128u) * (s.K / 128u);
                id<MTLBuffer> b_W = [dev newBufferWithLength:nw * 4 options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_sc = [dev newBufferWithLength:ns * 2 options:MTLResourceStorageModeShared];
                double bytes = (double)(nw * 4 + ns * 2);
                for (uint32_t rows : {1u, 16u, 32u, 256u, 512u}) {
                    bool dec = (rows <= 32u);
                    uint32_t ks = dec ? 8u : 1u;
                    MetalDenseParams p{rows, s.N, s.K, s.K, s.N, 0, 0, 0, 0, 0, s.K >> 7, s.K >> 7, ks};
                    auto run_n = [&](int reps) {
                        id<MTLCommandBuffer> cb = [q commandBuffer];
                        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                        for (int rep = 0; rep < reps; ++rep) {
                            [enc setComputePipelineState:pso_fx];
                            [enc setBuffer:b_X offset:0 atIndex:0]; [enc setBuffer:b_xr offset:0 atIndex:1];
                            [enc setBytes:&p length:sizeof(p) atIndex:2];
                            [enc dispatchThreadgroups:MTLSizeMake(s.K / 128u, rows, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                            if (dec) {
                                [enc setComputePipelineState:pso_dec[4]];
                                [enc setBuffer:b_W offset:0 atIndex:0]; [enc setBuffer:b_sc offset:0 atIndex:1];
                                [enc setBuffer:b_tbl offset:0 atIndex:2]; [enc setBuffer:b_xr offset:0 atIndex:3];
                                [enc setBuffer:b_yr offset:0 atIndex:4]; [enc setBytes:&p length:sizeof(p) atIndex:5];
                                [enc dispatchThreadgroups:MTLSizeMake(s.N / 64u, ks, (rows + 15u) / 16u) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                                [enc setComputePipelineState:pso_fy];
                                [enc setBuffer:b_yr offset:0 atIndex:0]; [enc setBuffer:b_Y offset:0 atIndex:1];
                                [enc setBytes:&p length:sizeof(p) atIndex:2];
                                [enc dispatchThreadgroups:MTLSizeMake(s.N / 128u, rows, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                            } else {
                                [enc setComputePipelineState:pso_mm[4]];
                                [enc setBuffer:b_W offset:0 atIndex:0]; [enc setBuffer:b_sc offset:0 atIndex:1];
                                [enc setBuffer:b_tbl offset:0 atIndex:2]; [enc setBuffer:b_xr offset:0 atIndex:3];
                                [enc setBuffer:b_Y offset:0 atIndex:4]; [enc setBytes:&p length:sizeof(p) atIndex:5];
                                [enc dispatchThreadgroups:MTLSizeMake(s.N / 128u, (rows + 31u) / 32u, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                            }
                        }
                        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                    };
                    run_n(5);
                    auto t0 = std::chrono::steady_clock::now();
                    run_n(20);
                    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 20.0;
                    printf("  Metal %-8s [%6u x %5u] 4b rows=%3u (%-3s): %7.3f ms | %6.1f GB/s | %5.2f TFLOP/s\n",
                           s.name, s.N, s.K, rows, dec ? "sk8" : "mm", ms, (bytes / 1e9) / (ms / 1e3), (2.0 * rows * s.N * s.K / 1e12) / (ms / 1e3));
                }
            }
        }
    }
    return 0;
}
