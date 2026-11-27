// Standalone NVIDIA CUDA TFQ4-E8P (2.001 & 2.251 bpw) Parity & Bandwidth Benchmark.
// Verifies table_kind=0 (trained 2.0b), table_kind=2 (True QuIP# E8P 2.0b), and table_kind=3 (True E8P-1024 2.25b).
#include "../kernels/cuda/tfq4_cuda.cuh"
#include "../converters/tfq4/tfq4_format.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CUDA_CHECK(x) do { \
    cudaError_t e__ = (x); \
    if (e__ != cudaSuccess) { \
        fprintf(stderr, "CUDA err %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e__)); \
        std::exit(1); \
    } \
} while (0)

static inline double host_sgn(uint32_t seed, uint32_t i) {
    return (((seed ^ i) * 2654435761u) >> 31) ? -1.0 : 1.0;
}

static inline float h_bf16_to_f32(uint16_t b) {
    uint32_t w = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &w, 4);
    return f;
}

static inline uint16_t h_f32_to_bf16(float f) {
    uint32_t w;
    std::memcpy(&w, &f, 4);
    return static_cast<uint16_t>((w + 0x7FFFu + ((w >> 16) & 1u)) >> 16);
}

static void fwht128_f64(double* x) {
    for (int len = 1; len < 128; len <<= 1) {
        for (int i = 0; i < 128; i += len * 2) {
            for (int j = 0; j < len; ++j) {
                double u = x[i + j], v = x[i + j + len];
                x[i + j] = u + v;
                x[i + j + len] = u - v;
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
    std::vector<double> Wrot(static_cast<size_t>(N) * K, 0.0);
    for (uint32_t r = 0; r < N; ++r) {
        const uint32_t br = r / 128u;
        for (uint32_t bc = 0; bc < n_bc; ++bc) {
            const double fsc = static_cast<double>(tfq4::f16_bits_to_f32(sc[static_cast<size_t>(br) * n_bc + bc]));
            const uint32_t* blk = W + static_cast<size_t>(r) * wpr + static_cast<size_t>(bc) * wpb;
            const uint32_t hi_word = is_1024 ? blk[8] : 0u;
            for (uint32_t sw = 0; sw < 8u; ++sw) {
                const uint32_t packed = blk[sw];
                const uint32_t c0 = packed & 0xFFFFu, c1 = packed >> 16;
                for (int half_idx = 0; half_idx < 2; ++half_idx) {
                    const uint32_t code = half_idx ? c1 : c0;
                    uint32_t idx = code & 255u;
                    if (is_1024) {
                        const uint32_t j = sw * 2u + static_cast<uint32_t>(half_idx);
                        idx |= ((hi_word >> (2u * j)) & 3u) << 8;
                    }
                    if (is_true) {
                        const uint32_t s7 = (code >> 8) & 0x7Fu;
                        const uint32_t sm = s7 | (((static_cast<uint32_t>(__builtin_popcount(s7)) ^ m_par[idx]) & 1u) << 7);
                        const double c_sgn = (code & 0x8000u) ? -1.0 : 1.0;
                        for (int d = 0; d < 8; ++d) {
                            const double mag = static_cast<double>(tfq4::f16_bits_to_f32(lut[idx][d]));
                            const double sgn = ((sm >> d) & 1u) ? -1.0 : 1.0;
                            const double q = c_sgn * (mag * sgn - 0.25);
                            Wrot[static_cast<size_t>(r) * K + bc * 128u + sw * 16u + half_idx * 8u + d] = q * fsc;
                        }
                    } else {
                        const uint32_t sm = code >> 8;
                        for (int d = 0; d < 8; ++d) {
                            const double mag = static_cast<double>(tfq4::f16_bits_to_f32(lut[idx][d]));
                            const double sgn = ((sm >> d) & 1u) ? -1.0 : 1.0;
                            Wrot[static_cast<size_t>(r) * K + bc * 128u + sw * 16u + half_idx * 8u + d] = mag * sgn * fsc;
                        }
                    }
                }
            }
        }
    }
    std::vector<double> tmp(128);
    for (uint32_t br = 0; br < N / 128u; ++br) {
        for (uint32_t c = 0; c < K; ++c) {
            for (int i = 0; i < 128; ++i) tmp[i] = Wrot[static_cast<size_t>(br * 128u + i) * K + c];
            fwht128_f64(tmp.data());
            for (int i = 0; i < 128; ++i) {
                Wrot[static_cast<size_t>(br * 128u + i) * K + c] = tmp[i] * host_sgn(0x9E3779B9u, br * 128u + i);
            }
        }
    }
    for (uint32_t r = 0; r < N; ++r) {
        for (uint32_t bc = 0; bc < K / 128u; ++bc) {
            for (int i = 0; i < 128; ++i) tmp[i] = Wrot[static_cast<size_t>(r) * K + bc * 128u + i];
            fwht128_f64(tmp.data());
            for (int i = 0; i < 128; ++i) {
                Wrot[static_cast<size_t>(r) * K + bc * 128u + i] = tmp[i] * host_sgn(0x12345678u, bc * 128u + i);
            }
        }
    }
    return Wrot;
}

static int run_verify_vectors(const std::string& vec_path) {
    FILE* vf = std::fopen(vec_path.c_str(), "rb");
    if (!vf) {
        fprintf(stderr, "Cannot open VEC4 vector bundle: %s\n", vec_path.c_str());
        return 1;
    }
    uint32_t hdr[2] = {};
    if (std::fread(hdr, 4, 2, vf) != 2 || hdr[0] != 0x56454334u) {
        fprintf(stderr, "Bad VEC4 header in %s\n", vec_path.c_str());
        std::fclose(vf);
        return 1;
    }
    uint32_t thdr[16] = {};
    if (std::fread(thdr, 4, 16, vf) != 16) {
        fprintf(stderr, "Truncated VEC4 table header\n");
        std::fclose(vf);
        return 1;
    }
    const uint32_t magic = thdr[0], ver = thdr[1], k_codes = thdr[2], dim = thdr[3], table_kind = thdr[4];
    if (magic != tfq4::MAGIC_PWV4 || ver != tfq4::FORMAT_VER || dim != 8u || table_kind > tfq4::PWV4_TABLE_MAX_KNOWN) {
        fprintf(stderr, "Unsupported PWV4 table in VEC4 bundle: magic=0x%08x ver=%u table_kind=%u\n",
                magic, ver, table_kind);
        std::fclose(vf);
        return 1;
    }
    const bool is_1024 = (table_kind == tfq4::PWV4_TABLE_TRUE_E8P_1024);
    if ((is_1024 && k_codes != 1024u) || (!is_1024 && k_codes != 256u)) {
        fprintf(stderr, "Mismatched k_codes=%u for table_kind=%u in VEC4 bundle\n", k_codes, table_kind);
        std::fclose(vf);
        return 1;
    }
    const size_t tbl_bytes = is_1024 ? sizeof(tfq4::PwV4E8Tables1024) : sizeof(tfq4::PwV4E8Tables);
    const size_t lut_bytes = static_cast<size_t>(k_codes) * 8u * 2u;
    std::vector<uint8_t> tbl_rest(tbl_bytes - 64u);
    if (std::fread(tbl_rest.data(), 1, tbl_rest.size(), vf) != tbl_rest.size()) {
        fprintf(stderr, "Truncated VEC4 tables\n");
        std::fclose(vf);
        return 1;
    }
    uint4* d_lut = nullptr;
    CUDA_CHECK(cudaMalloc(&d_lut, lut_bytes));
    CUDA_CHECK(cudaMemcpy(d_lut, tbl_rest.data(), lut_bytes, cudaMemcpyHostToDevice));
    const uint32_t wpb = is_1024 ? 9u : 8u;

    for (uint32_t ci = 0; ci < hdr[1]; ++ci) {
        uint32_t ch[10] = {};
        if (std::fread(ch, 4, 10, vf) != 10) {
            fprintf(stderr, "Truncated VEC4 case %u\n", ci);
            cudaFree(d_lut);
            std::fclose(vf);
            return 1;
        }
        const uint32_t C = ch[2], rows = ch[3], r_off = ch[4], c_off = ch[5], n_loc = ch[6], k_loc = ch[7];
        (void)ch[1];
        const uint32_t n_words = ch[8], n_scales = ch[9];
        std::vector<uint32_t> W(n_words);
        std::vector<uint16_t> sc(n_scales), X(static_cast<size_t>(rows) * k_loc);
        std::vector<float> Y_ref(static_cast<size_t>(rows) * n_loc);
        if (std::fread(W.data(), 4, n_words, vf) != n_words ||
            std::fread(sc.data(), 2, n_scales, vf) != n_scales ||
            std::fread(X.data(), 2, X.size(), vf) != X.size() ||
            std::fread(Y_ref.data(), 4, Y_ref.size(), vf) != Y_ref.size()) {
            fprintf(stderr, "Truncated VEC4 payload case %u\n", ci);
            cudaFree(d_lut);
            std::fclose(vf);
            return 1;
        }
        const uint32_t full_wpr = (C / 128u) * wpb;
        const uint32_t loc_wpr  = (k_loc / 128u) * wpb;
        const uint32_t c_w_off  = (c_off / 128u) * wpb;
        std::vector<uint32_t> W_loc(static_cast<size_t>(n_loc) * loc_wpr);
        for (uint32_t r = 0; r < n_loc; ++r) {
            std::memcpy(&W_loc[static_cast<size_t>(r) * loc_wpr],
                        &W[static_cast<size_t>(r_off + r) * full_wpr + c_w_off],
                        static_cast<size_t>(loc_wpr) * 4u);
        }
        std::vector<uint16_t> sc_loc(static_cast<size_t>(n_loc / 128u) * (k_loc / 128u));
        for (uint32_t br = 0; br < n_loc / 128u; ++br) {
            std::memcpy(&sc_loc[static_cast<size_t>(br) * (k_loc / 128u)],
                        &sc[static_cast<size_t>((r_off / 128u) + br) * (C / 128u) + (c_off / 128u)],
                        (k_loc / 128u) * 2u);
        }
        std::vector<uint16_t> X_bf16(X.size());
        for (size_t i = 0; i < X.size(); ++i) X_bf16[i] = h_f32_to_bf16(tfq4::f16_bits_to_f32(X[i]));

        uint32_t* d_W = nullptr;
        __half *d_S = nullptr, *d_X = nullptr, *d_Xr = nullptr;
        uint16_t* d_Xbf = nullptr;
        float* d_Y = nullptr;
        CUDA_CHECK(cudaMalloc(&d_W, W_loc.size() * 4));
        CUDA_CHECK(cudaMalloc(&d_S, sc_loc.size() * 2));
        CUDA_CHECK(cudaMalloc(&d_X, X.size() * 2));
        CUDA_CHECK(cudaMalloc(&d_Xbf, X_bf16.size() * 2));
        CUDA_CHECK(cudaMalloc(&d_Xr, X.size() * 2));
        CUDA_CHECK(cudaMalloc(&d_Y, Y_ref.size() * 4));
        CUDA_CHECK(cudaMemcpy(d_W, W_loc.data(), W_loc.size() * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_S, sc_loc.data(), sc_loc.size() * 2, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_X, X.data(), X.size() * 2, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Xbf, X_bf16.data(), X_bf16.size() * 2, cudaMemcpyHostToDevice));

        std::vector<float> got(Y_ref.size());
        // 1. FP16 input -> FP32 output
        CUDA_CHECK(tfq4_cuda::launch_dense_f16_f32(
            d_W, d_S, d_lut, d_X, d_Y, d_Xr,
            table_kind, rows, n_loc, k_loc, k_loc, n_loc, r_off, c_off
        ));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(got.data(), d_Y, got.size() * 4, cudaMemcpyDeviceToHost));
        double dot = 0.0, na = 0.0, nb = 0.0;
        for (size_t i = 0; i < Y_ref.size(); ++i) {
            dot += static_cast<double>(got[i]) * Y_ref[i];
            na += static_cast<double>(got[i]) * got[i];
            nb += static_cast<double>(Y_ref[i]) * Y_ref[i];
        }
        double cos_v = dot / (std::sqrt(na * nb) + 1e-30);
        if (cos_v < 0.99998) {
            fprintf(stderr, "FAIL CUDA VEC4 f16 table_kind=%u case=%u r_off=%u c_off=%u cos=%.8f\n",
                    table_kind, ci, r_off, c_off, cos_v);
            cudaFree(d_W); cudaFree(d_S); cudaFree(d_X); cudaFree(d_Xbf); cudaFree(d_Xr); cudaFree(d_Y); cudaFree(d_lut);
            std::fclose(vf);
            return 1;
        }
        // 2. BF16 input -> FP32 output
        CUDA_CHECK(tfq4_cuda::launch_dense_bf16(
            d_W, d_S, d_lut, d_Xbf, d_Y, d_Xr,
            table_kind, rows, n_loc, k_loc, k_loc, n_loc, r_off, c_off, true
        ));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(got.data(), d_Y, got.size() * 4, cudaMemcpyDeviceToHost));
        dot = 0.0; na = 0.0; nb = 0.0;
        for (size_t i = 0; i < Y_ref.size(); ++i) {
            dot += static_cast<double>(got[i]) * Y_ref[i];
            na += static_cast<double>(got[i]) * got[i];
            nb += static_cast<double>(Y_ref[i]) * Y_ref[i];
        }
        double cos_bf = dot / (std::sqrt(na * nb) + 1e-30);
        if (cos_bf < 0.9999) {
            fprintf(stderr, "FAIL CUDA VEC4 bf16 table_kind=%u case=%u r_off=%u c_off=%u cos=%.8f\n",
                    table_kind, ci, r_off, c_off, cos_bf);
            cudaFree(d_W); cudaFree(d_S); cudaFree(d_X); cudaFree(d_Xbf); cudaFree(d_Xr); cudaFree(d_Y); cudaFree(d_lut);
            std::fclose(vf);
            return 1;
        }

        cudaFree(d_W); cudaFree(d_S); cudaFree(d_X); cudaFree(d_Xbf); cudaFree(d_Xr); cudaFree(d_Y);
    }
    cudaFree(d_lut);
    std::fclose(vf);
    printf("PASS CUDA TFQ4 golden vectors (table_kind=%u, %u cases, including r_off & c_off offsets)\n",
           table_kind, hdr[1]);
    return 0;
}

int main(int argc, char** argv) {
    bool do_bench = false;
    std::string vec_path;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--bench") == 0) do_bench = true;
        else if (std::strcmp(argv[i], "--verify-vectors") == 0 && i + 1 < argc) vec_path = argv[++i];
    }

    if (!vec_path.empty()) {
        return run_verify_vectors(vec_path);
    }

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    tfq4::PwV4E8Tables h_tbl_trained{}, h_tbl_true{};
    tfq4::PwV4E8Tables1024 h_tbl_1024{};
    tfq4::build_e8p_tables(h_tbl_trained);
    tfq4::build_true_e8p_tables(h_tbl_true);
    tfq4::build_true_e8p_1024_tables(h_tbl_1024);
    if (std::memcmp(h_tbl_true.e8p_lut, h_tbl_1024.e8p_lut, sizeof(h_tbl_true.e8p_lut)) != 0) {
        fprintf(stderr, "FAIL: first 256 entries of PwV4E8Tables1024 diverge from PwV4E8Tables (table_kind=2)\n");
        return 1;
    }

    uint4 *d_lut_trained = nullptr, *d_lut_true = nullptr, *d_lut_1024 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_lut_trained, sizeof(h_tbl_trained.e8p_lut)));
    CUDA_CHECK(cudaMalloc(&d_lut_true, sizeof(h_tbl_true.e8p_lut)));
    CUDA_CHECK(cudaMalloc(&d_lut_1024, sizeof(h_tbl_1024.e8p_lut)));
    CUDA_CHECK(cudaMemcpy(d_lut_trained, h_tbl_trained.e8p_lut, sizeof(h_tbl_trained.e8p_lut), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lut_true, h_tbl_true.e8p_lut, sizeof(h_tbl_true.e8p_lut), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lut_1024, h_tbl_1024.e8p_lut, sizeof(h_tbl_1024.e8p_lut), cudaMemcpyHostToDevice));

    // 1. Exact fp64 parity + bit-exact rows=1 == rows=16 == rows=17 + bf16 + fp16 sat for kind=0, 2, and 3
    const uint32_t N = 512u, K = 1024u, MAX_R = 17u;
    const uint32_t LDX_BF = K + 128u, LDO_BF = N + 128u;
    std::mt19937 rng(777);
    std::vector<uint32_t> h_W20(static_cast<size_t>(N) * (K / 16u));
    std::vector<uint32_t> h_W225(static_cast<size_t>(N) * (K / 128u) * 9u);
    std::vector<uint16_t> h_sc(static_cast<size_t>(N / 128u) * (K / 128u));
    std::vector<uint16_t> h_X(static_cast<size_t>(MAX_R) * K);
    std::vector<uint16_t> h_X_bf(static_cast<size_t>(MAX_R) * LDX_BF, 0u);
    for (auto& w : h_W20) w = rng();
    for (auto& w : h_W225) w = rng();
    for (auto& s : h_sc) s = tfq4::f32_to_f16_bits(0.03f + 0.02f * (static_cast<float>(rng() & 255u) / 255.0f));
    for (uint32_t m = 0; m < MAX_R; ++m) {
        for (uint32_t c = 0; c < K; ++c) {
            float v = (static_cast<int>(rng() % 201u) - 100) * 0.01f;
            h_X[static_cast<size_t>(m) * K + c] = tfq4::f32_to_f16_bits(v);
            h_X_bf[static_cast<size_t>(m) * LDX_BF + c] = h_f32_to_bf16(v);
        }
    }

    uint32_t *d_W20 = nullptr, *d_W225 = nullptr;
    __half *d_sc = nullptr, *d_X = nullptr, *d_xr = nullptr;
    uint16_t *d_X_bf = nullptr, *d_Y_bf = nullptr;
    float* d_Y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_W20, h_W20.size() * 4));
    CUDA_CHECK(cudaMalloc(&d_W225, h_W225.size() * 4));
    CUDA_CHECK(cudaMalloc(&d_sc, h_sc.size() * 2));
    CUDA_CHECK(cudaMalloc(&d_X, h_X.size() * 2));
    CUDA_CHECK(cudaMalloc(&d_X_bf, h_X_bf.size() * 2));
    CUDA_CHECK(cudaMalloc(&d_xr, static_cast<size_t>(MAX_R) * K * 2));
    CUDA_CHECK(cudaMalloc(&d_Y, static_cast<size_t>(MAX_R) * N * 4));
    CUDA_CHECK(cudaMalloc(&d_Y_bf, static_cast<size_t>(MAX_R) * LDO_BF * 2));
    CUDA_CHECK(cudaMemcpy(d_W20, h_W20.data(), h_W20.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W225, h_W225.data(), h_W225.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sc, h_sc.data(), h_sc.size() * 2, cudaMemcpyHostToDevice));

    struct KindCfg {
        const char* name;
        const char* rate_lbl;
        uint32_t kind;
        uint32_t wpb;
        const uint16_t (*lut_host)[8];
        const uint32_t* w_host;
        const uint32_t* d_W;
        const uint4* d_lut;
    };
    KindCfg kinds[] = {
        {"trained/kind=0",       "2.00b", tfq4::PWV4_TABLE_LEGACY_TRAINED, 8u, h_tbl_trained.e8p_lut, h_W20.data(),  d_W20,  d_lut_trained},
        {"true-E8P/kind=2",      "2.00b", tfq4::PWV4_TABLE_TRUE_E8P,       8u, h_tbl_true.e8p_lut,    h_W20.data(),  d_W20,  d_lut_true},
        {"true-E8P-1024/kind=3", "2.25b", tfq4::PWV4_TABLE_TRUE_E8P_1024,  9u, h_tbl_1024.e8p_lut,    h_W225.data(), d_W225, d_lut_1024},
    };

    for (auto& kc : kinds) {
        CUDA_CHECK(cudaMemcpy(d_X, h_X.data(), h_X.size() * 2, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_X_bf, h_X_bf.data(), h_X_bf.size() * 2, cudaMemcpyHostToDevice));
        auto W_ref = ref_dequant_e8p(kc.w_host, h_sc.data(), kc.lut_host, kc.kind, N, K);

        auto run_f16 = [&](uint32_t rows, std::vector<float>& out) {
            CUDA_CHECK(tfq4_cuda::launch_dense_f16_f32(
                kc.d_W, d_sc, kc.d_lut, d_X, d_Y, d_xr,
                kc.kind, rows, N, K, K, N, 0u, 0u
            ));
            CUDA_CHECK(cudaDeviceSynchronize());
            out.resize(static_cast<size_t>(rows) * N);
            CUDA_CHECK(cudaMemcpy(out.data(), d_Y, out.size() * 4, cudaMemcpyDeviceToHost));
        };

        std::vector<float> y1, y16, y17;
        run_f16(1u, y1);
        run_f16(16u, y16);
        run_f16(17u, y17);
        for (uint32_t r = 0; r < N; ++r) {
            if (std::memcmp(&y1[r], &y16[r], 4) != 0) {
                fprintf(stderr, "FAIL TFQ4 CUDA (%s) r1!=r16 @ %u (%.7f vs %.7f)\n", kc.name, r, y1[r], y16[r]);
                return 1;
            }
            if (std::memcmp(&y1[r], &y17[r], 4) != 0) {
                fprintf(stderr, "FAIL TFQ4 CUDA (%s) r1!=r17 @ %u (%.7f vs %.7f)\n", kc.name, r, y1[r], y17[r]);
                return 1;
            }
        }
        double dot = 0.0, na = 0.0, nb = 0.0;
        for (uint32_t m = 0; m < MAX_R; ++m) {
            for (uint32_t r = 0; r < N; ++r) {
                double ref = 0.0;
                for (uint32_t c = 0; c < K; ++c) {
                    ref += W_ref[static_cast<size_t>(r) * K + c] *
                           static_cast<double>(tfq4::f16_bits_to_f32(h_X[static_cast<size_t>(m) * K + c]));
                }
                double g = y17[static_cast<size_t>(m) * N + r];
                dot += g * ref; na += g * g; nb += ref * ref;
            }
        }
        double cos_v = dot / (std::sqrt(na * nb) + 1e-30);
        if (cos_v < 0.9999999) {
            fprintf(stderr, "FAIL TFQ4 CUDA (%s) fp64 parity cos=%.8f (< 0.9999999)\n", kc.name, cos_v);
            return 1;
        }

        // Verify BF16 input/output launcher + non-trivial ldx/ldo strides + bit-exact r1==r16
        {
            std::vector<uint16_t> yb1(LDO_BF), yb16(static_cast<size_t>(16u) * LDO_BF);
            CUDA_CHECK(tfq4_cuda::launch_dense_bf16(
                kc.d_W, d_sc, kc.d_lut, d_X_bf, d_Y_bf, d_xr,
                kc.kind, 1u, N, K, LDX_BF, LDO_BF, 0u, 0u, false
            ));
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(yb1.data(), d_Y_bf, LDO_BF * 2, cudaMemcpyDeviceToHost));
            CUDA_CHECK(tfq4_cuda::launch_dense_bf16(
                kc.d_W, d_sc, kc.d_lut, d_X_bf, d_Y_bf, d_xr,
                kc.kind, 16u, N, K, LDX_BF, LDO_BF, 0u, 0u, false
            ));
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(yb16.data(), d_Y_bf, yb16.size() * 2, cudaMemcpyDeviceToHost));
            if (std::memcmp(yb1.data(), yb16.data(), N * 2) != 0) {
                fprintf(stderr, "FAIL TFQ4 CUDA (%s) bf16 r1!=r16\n", kc.name);
                return 1;
            }
            double bdot = 0.0, bna = 0.0, bnb = 0.0;
            for (uint32_t m = 0; m < 16u; ++m) {
                for (uint32_t r = 0; r < N; ++r) {
                    double ref = 0.0;
                    for (uint32_t c = 0; c < K; ++c) {
                        ref += W_ref[static_cast<size_t>(r) * K + c] *
                               static_cast<double>(h_bf16_to_f32(h_X_bf[static_cast<size_t>(m) * LDX_BF + c]));
                    }
                    double g = static_cast<double>(h_bf16_to_f32(yb16[static_cast<size_t>(m) * LDO_BF + r]));
                    bdot += g * ref; bna += g * g; bnb += ref * ref;
                }
            }
            double cos_bf = bdot / (std::sqrt(bna * bnb) + 1e-30);
            if (cos_bf < 0.9999) {
                fprintf(stderr, "FAIL TFQ4 CUDA (%s) bf16 fp64 parity cos=%.8f\n", kc.name, cos_bf);
                return 1;
            }
        }

        // Issue #11: Verify fp16 saturation prevents Inf/NaN when rotated activation exceeds 65504
        {
            std::vector<uint16_t> big_x(K, tfq4::f32_to_f16_bits(10000.0f));
            CUDA_CHECK(cudaMemcpy(d_X, big_x.data(), K * 2, cudaMemcpyHostToDevice));
            std::vector<float> y_big;
            run_f16(1u, y_big);
            for (float v : y_big) {
                if (!std::isfinite(v)) {
                    fprintf(stderr, "FAIL TFQ4 CUDA (%s) fp16 overflow produced non-finite output\n", kc.name);
                    return 1;
                }
            }
            std::vector<uint16_t> big_x_bf(LDX_BF, h_f32_to_bf16(800000.0f));
            CUDA_CHECK(cudaMemcpy(d_X_bf, big_x_bf.data(), LDX_BF * 2, cudaMemcpyHostToDevice));
            CUDA_CHECK(tfq4_cuda::launch_dense_bf16(
                kc.d_W, d_sc, kc.d_lut, d_X_bf, d_Y_bf, d_xr,
                kc.kind, 1u, N, K, LDX_BF, LDO_BF, 0u, 0u, false
            ));
            CUDA_CHECK(cudaDeviceSynchronize());
            std::vector<uint16_t> yb_big(LDO_BF);
            CUDA_CHECK(cudaMemcpy(yb_big.data(), d_Y_bf, LDO_BF * 2, cudaMemcpyDeviceToHost));
            for (uint32_t r = 0; r < N; ++r) {
                if (!std::isfinite(h_bf16_to_f32(yb_big[r]))) {
                    fprintf(stderr, "FAIL TFQ4 CUDA (%s) bf16 activation overflow produced non-finite output\n", kc.name);
                    return 1;
                }
            }
        }

        printf("PASS tfq4_cuda_test (%s SM %d.%d: %s %s fp64 parity cos=%.8f + bit-exact r1==r16==r17 + bf16 + fp16 sat)\n",
               prop.name, prop.major, prop.minor, kc.rate_lbl, kc.name, cos_v);
    }

    cudaFree(d_W20); cudaFree(d_W225); cudaFree(d_sc); cudaFree(d_X); cudaFree(d_X_bf); cudaFree(d_xr); cudaFree(d_Y); cudaFree(d_Y_bf);

    if (do_bench) {
        struct Shape { const char* name; uint32_t N, K; };
        Shape shapes[] = {{"MoE-8E-gu", 32768, 4096}, {"Qwen27-gu", 34816, 5120}, {"Qwen27-dn", 5120, 17408}};
        __half *bx = nullptr, *bxr = nullptr;
        float* by = nullptr;
        CUDA_CHECK(cudaMalloc(&bx, 16ULL * 17408ULL * 2ULL));
        CUDA_CHECK(cudaMalloc(&bxr, 16ULL * 17408ULL * 2ULL));
        CUDA_CHECK(cudaMalloc(&by, 16ULL * 34816ULL * 4ULL));
        CUDA_CHECK(cudaMemset(bx, 0x38, 16ULL * 17408ULL * 2ULL));
        cudaEvent_t ev0, ev1;
        CUDA_CHECK(cudaEventCreate(&ev0));
        CUDA_CHECK(cudaEventCreate(&ev1));
        for (auto& kc : kinds) {
            for (auto& s : shapes) {
                const size_t nw = static_cast<size_t>(s.N) * (s.K / 128u) * kc.wpb;
                const size_t ns = static_cast<size_t>(s.N / 128u) * (s.K / 128u);
                uint32_t* bw = nullptr;
                __half* bsc = nullptr;
                CUDA_CHECK(cudaMalloc(&bw, nw * 4));
                CUDA_CHECK(cudaMalloc(&bsc, ns * 2));
                CUDA_CHECK(cudaMemset(bw, 0x5A, nw * 4));
                CUDA_CHECK(cudaMemset(bsc, 0x2E, ns * 2));
                const double bytes = static_cast<double>(nw * 4 + ns * 2);
                for (uint32_t rows : {1u, 16u}) {
                    for (int rep = 0; rep < 5; ++rep) {
                        CUDA_CHECK(tfq4_cuda::launch_dense_f16_f32(
                            bw, bsc, kc.d_lut, bx, by, bxr, kc.kind, rows, s.N, s.K, s.K, s.N
                        ));
                    }
                    CUDA_CHECK(cudaDeviceSynchronize());
                    CUDA_CHECK(cudaEventRecord(ev0));
                    for (int rep = 0; rep < 20; ++rep) {
                        CUDA_CHECK(tfq4_cuda::launch_dense_f16_f32(
                            bw, bsc, kc.d_lut, bx, by, bxr, kc.kind, rows, s.N, s.K, s.K, s.N
                        ));
                    }
                    CUDA_CHECK(cudaEventRecord(ev1));
                    CUDA_CHECK(cudaEventSynchronize(ev1));
                    float total_ms = 0.0f;
                    CUDA_CHECK(cudaEventElapsedTime(&total_ms, ev0, ev1));
                    const double ms = static_cast<double>(total_ms) / 20.0;
                    printf("  CUDA  TFQ4 %-20s %-10s [%5u x %5u] %s rows=%2u | %5.1f MiB | %6.3f ms | %6.1f GB/s\n",
                           kc.name, s.name, s.N, s.K, kc.rate_lbl, rows, bytes / (1024.0 * 1024.0), ms, (bytes / 1e9) / (ms / 1e3));
                }
                cudaFree(bw);
                cudaFree(bsc);
            }
        }
        cudaEventDestroy(ev0);
        cudaEventDestroy(ev1);
        cudaFree(bx);
        cudaFree(bxr);
        cudaFree(by);
    }

    cudaFree(d_lut_trained);
    cudaFree(d_lut_true);
    cudaFree(d_lut_1024);
    return 0;
}
