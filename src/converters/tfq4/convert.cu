// ============================================================================
// TFQ4-E8P (2.001 bpw) Model-Agnostic NVIDIA CUDA GPU Converter
//   - Byte-identical PWV4 v4 + TFP4 v4 cross-platform output with Apple Metal
//   - Row-chunked GPU pipeline (4096 rows/chunk) for bounded VRAM on 70B+ models
//   - Automatic 3.4 GiB shard splitting (16 KiB-aligned cross-platform TFP4 v4)
// ============================================================================
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include "../generic/planner.hpp"
#include "tfq4_format.hpp"

namespace fs = std::filesystem;

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

__device__ __forceinline__ float sgn_bit(uint32_t seed, uint32_t i) {
    uint32_t h = (seed ^ i) * 2654435761u;
    return ((h >> 31) & 1u) ? -1.0f : 1.0f;
}

__device__ __forceinline__ void fwht128_warp4(float4& v, uint32_t lane) {
    #pragma unroll
    for (int d = 1; d <= 16; d <<= 1) {
        float4 o;
        o.x = __shfl_xor_sync(0xFFFFFFFFu, v.x, d);
        o.y = __shfl_xor_sync(0xFFFFFFFFu, v.y, d);
        o.z = __shfl_xor_sync(0xFFFFFFFFu, v.z, d);
        o.w = __shfl_xor_sync(0xFFFFFFFFu, v.w, d);
        bool keep = ((lane & d) == 0u);
        v.x = keep ? (v.x + o.x) : (o.x - v.x);
        v.y = keep ? (v.y + o.y) : (o.y - v.y);
        v.z = keep ? (v.z + o.z) : (o.z - v.z);
        v.w = keep ? (v.w + o.w) : (o.w - v.w);
    }
    float a0 = v.x + v.y, b0 = v.x - v.y, a1 = v.z + v.w, b1 = v.z - v.w;
    v = make_float4((a0 + a1) * 0.08838834764831845f, (b0 + b1) * 0.08838834764831845f,
                    (a0 - a1) * 0.08838834764831845f, (b0 - b1) * 0.08838834764831845f);
}

__global__ void f32_row_fwht_k(const float* __restrict__ W_in, float* __restrict__ W_tmp, uint32_t R, uint32_t C) {
    uint32_t r = blockIdx.y, bc = blockIdx.x, c0 = bc * 128u, lane = threadIdx.x;
    if (r >= R) return;
    const float* row = W_in + (uint64_t)r * C + c0;
    float4 v = make_float4(row[lane + 0u] * sgn_bit(0x12345678u, c0 + lane + 0u),
                           row[lane + 32u] * sgn_bit(0x12345678u, c0 + lane + 32u),
                           row[lane + 64u] * sgn_bit(0x12345678u, c0 + lane + 64u),
                           row[lane + 96u] * sgn_bit(0x12345678u, c0 + lane + 96u));
    fwht128_warp4(v, lane);
    float* dst = W_tmp + (uint64_t)r * C + c0;
    dst[lane + 0u] = v.x; dst[lane + 32u] = v.y; dst[lane + 64u] = v.z; dst[lane + 96u] = v.w;
}

__global__ void col_fwht_ss_k(
    const float* __restrict__ W_tmp, float* __restrict__ W_rot,
    float* __restrict__ col_ss, uint32_t R, uint32_t C, uint32_t r_off
) {
    uint32_t c = blockIdx.x, br = blockIdx.y, r0 = br * 128u, lane = threadIdx.x;
    if (c >= C || r0 >= R) return;
    uint32_t gr0 = r_off + r0;
    float4 v = make_float4(W_tmp[(uint64_t)(r0 + lane + 0u) * C + c] * sgn_bit(0x9E3779B9u, gr0 + lane + 0u),
                           W_tmp[(uint64_t)(r0 + lane + 32u) * C + c] * sgn_bit(0x9E3779B9u, gr0 + lane + 32u),
                           W_tmp[(uint64_t)(r0 + lane + 64u) * C + c] * sgn_bit(0x9E3779B9u, gr0 + lane + 64u),
                           W_tmp[(uint64_t)(r0 + lane + 96u) * C + c] * sgn_bit(0x9E3779B9u, gr0 + lane + 96u));
    fwht128_warp4(v, lane);
    W_rot[(uint64_t)(r0 + lane + 0u) * C + c] = v.x;
    W_rot[(uint64_t)(r0 + lane + 32u) * C + c] = v.y;
    W_rot[(uint64_t)(r0 + lane + 64u) * C + c] = v.z;
    W_rot[(uint64_t)(r0 + lane + 96u) * C + c] = v.w;
    float ss = (v.x * v.x + v.y * v.y) + (v.z * v.z + v.w * v.w);
    #pragma unroll
    for (int d = 16; d >= 1; d >>= 1) ss += __shfl_xor_sync(0xFFFFFFFFu, ss, d);
    uint32_t bi = br * (C >> 7) + (c >> 7);
    if (lane == 0) col_ss[(uint64_t)bi * 128u + (c & 127u)] = ss;
}

__global__ void finalize_scales_k(const float* __restrict__ col_ss, __half* __restrict__ sc, uint32_t n_blk, float scale_mult) {
    uint32_t bi = blockIdx.x, lane = threadIdx.x;
    if (bi >= n_blk) return;
    const float* cs = col_ss + (uint64_t)bi * 128u;
    float s = (cs[lane + 0u] + cs[lane + 32u]) + (cs[lane + 64u] + cs[lane + 96u]);
    #pragma unroll
    for (int d = 16; d >= 1; d >>= 1) s += __shfl_xor_sync(0xFFFFFFFFu, s, d);
    if (lane == 0) sc[bi] = __float2half_rn(sqrtf(s * (1.0f / 16384.0f) + 1e-12f) * scale_mult);
}

__global__ void tfq4_e8p_encode_k(
    const float* __restrict__ W_rot, const __half* __restrict__ scales,
    const __half* __restrict__ e8p_lut, uint32_t* __restrict__ out_words,
    float* __restrict__ metrics, uint32_t R, uint32_t C
) {
    __shared__ float sm_lut[256][8];
    uint32_t tid = threadIdx.x;
    #pragma unroll
    for (uint32_t d = 0u; d < 8u; ++d) sm_lut[tid][d] = __half2float(e8p_lut[tid * 8u + d]);
    __syncthreads();

    uint64_t gtid = (uint64_t)blockIdx.x * blockDim.x + tid;
    uint32_t wpr = C >> 4;
    uint64_t total_words = (uint64_t)R * wpr;
    float my_sig = 0.0f, my_err = 0.0f;

    if (gtid < total_words) {
        uint32_t r = (uint32_t)(gtid / wpr), wc = (uint32_t)(gtid - (uint64_t)r * wpr);
        uint32_t n_bc = C >> 7;
        float sc = __half2float(scales[(r >> 7) * n_bc + (wc >> 3)]);
        float inv_sc = (sc > 0.0f) ? (1.0f / sc) : 0.0f;
        const float* src = W_rot + (uint64_t)r * C + (uint64_t)wc * 16ull;

        uint32_t codes[2];
        #pragma unroll
        for (uint32_t h = 0u; h < 2u; ++h) {
            float v[8], a[8];
            uint32_t sm = 0u;
            #pragma unroll
            for (uint32_t d = 0u; d < 8u; ++d) {
                float x = src[h * 8u + d];
                v[d] = x;
                if (x < 0.0f) sm |= (1u << d);
                a[d] = fabsf(x) * inv_sc;
                my_sig += x * x;
            }
            uint32_t best_k = 0u;
            float best_d = 1e30f;
            for (uint32_t k = 0u; k < 256u; ++k) {
                float d0 = a[0] - sm_lut[k][0], d1 = a[1] - sm_lut[k][1];
                float d2 = a[2] - sm_lut[k][2], d3 = a[3] - sm_lut[k][3];
                float d4 = a[4] - sm_lut[k][4], d5 = a[5] - sm_lut[k][5];
                float d6 = a[6] - sm_lut[k][6], d7 = a[7] - sm_lut[k][7];
                float dist = ((d0*d0 + d1*d1) + (d2*d2 + d3*d3)) + ((d4*d4 + d5*d5) + (d6*d6 + d7*d7));
                if (dist < best_d) { best_d = dist; best_k = k; }
            }
            codes[h] = (sm << 8) | best_k;
            #pragma unroll
            for (uint32_t d = 0u; d < 8u; ++d) {
                float e = fabsf(v[d]) - sm_lut[best_k][d] * sc;
                my_err += e * e;
            }
        }
        out_words[gtid] = codes[0] | (codes[1] << 16);
    }

    #pragma unroll
    for (int d = 16; d >= 1; d >>= 1) {
        my_sig += __shfl_xor_sync(0xFFFFFFFFu, my_sig, d);
        my_err += __shfl_xor_sync(0xFFFFFFFFu, my_err, d);
    }
    __shared__ float sm_sig[8], sm_err[8];
    if ((tid & 31u) == 0u) {
        sm_sig[tid >> 5] = my_sig;
        sm_err[tid >> 5] = my_err;
    }
    __syncthreads();
    if (tid == 0u) {
        float bs = 0.0f, be = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) { bs += sm_sig[w]; be += sm_err[w]; }
        metrics[(uint64_t)blockIdx.x * 2ull + 0ull] = bs;
        metrics[(uint64_t)blockIdx.x * 2ull + 1ull] = be;
    }
}

__global__ void tfq4_true_e8p_encode_k(
    const float* __restrict__ W_rot, const __half* __restrict__ scales,
    const __half* __restrict__ e8p_lut, uint32_t* __restrict__ out_words,
    float* __restrict__ metrics, uint32_t R, uint32_t C
) {
    __shared__ float sm_lut[256][8];
    __shared__ float sm_n2[256];
    __shared__ uint32_t sm_par[256];
    uint32_t tid = threadIdx.x;
    {
        float a0 = __half2float(e8p_lut[tid * 8u + 0u]);
        float a1 = __half2float(e8p_lut[tid * 8u + 1u]);
        float a2 = __half2float(e8p_lut[tid * 8u + 2u]);
        float a3 = __half2float(e8p_lut[tid * 8u + 3u]);
        float a4 = __half2float(e8p_lut[tid * 8u + 4u]);
        float a5 = __half2float(e8p_lut[tid * 8u + 5u]);
        float a6 = __half2float(e8p_lut[tid * 8u + 6u]);
        float a7 = __half2float(e8p_lut[tid * 8u + 7u]);
        sm_lut[tid][0] = a0; sm_lut[tid][1] = a1; sm_lut[tid][2] = a2; sm_lut[tid][3] = a3;
        sm_lut[tid][4] = a4; sm_lut[tid][5] = a5; sm_lut[tid][6] = a6; sm_lut[tid][7] = a7;
        sm_n2[tid] = ((a0 * a0 + a1 * a1) + (a2 * a2 + a3 * a3))
                   + ((a4 * a4 + a5 * a5) + (a6 * a6 + a7 * a7));
        sm_par[tid] = ((uint32_t)(int)a0 + (uint32_t)(int)a1 + (uint32_t)(int)a2 + (uint32_t)(int)a3
                     + (uint32_t)(int)a4 + (uint32_t)(int)a5 + (uint32_t)(int)a6 + (uint32_t)(int)a7) & 1u;
    }
    __syncthreads();

    uint64_t gtid = (uint64_t)blockIdx.x * blockDim.x + tid;
    uint32_t wpr = C >> 4;
    uint64_t total_words = (uint64_t)R * wpr;
    float my_sig = 0.0f, my_err = 0.0f;

    if (gtid < total_words) {
        uint32_t r = (uint32_t)(gtid / wpr), wc = (uint32_t)(gtid - (uint64_t)r * wpr);
        uint32_t n_bc = C >> 7;
        float sc = __half2float(scales[(r >> 7) * n_bc + (wc >> 3)]);
        float inv_sc = (sc > 0.0f) ? (1.0f / sc) : 0.0f;
        const float* src = W_rot + (uint64_t)r * C + (uint64_t)wc * 16ull;

        uint32_t codes[2];
        #pragma unroll
        for (uint32_t h = 0u; h < 2u; ++h) {
            float x[8], v[8];
            #pragma unroll
            for (uint32_t d = 0u; d < 8u; ++d) {
                float xd = src[h * 8u + d];
                x[d] = xd;
                v[d] = xd * inv_sc;
                my_sig += xd * xd;
            }
            float best_d = 1e30f;
            uint32_t best_coset = 0u, best_k = 0u, best_nb = 0u;

            #pragma unroll
            for (uint32_t coset = 0u; coset < 2u; ++coset) {
                float z[8], az[8];
                uint32_t nb = 0u;
                #pragma unroll
                for (uint32_t d = 0u; d < 8u; ++d) {
                    float zd = ((coset == 0u) ? v[d] : -v[d]) + 0.25f;
                    z[d] = zd;
                    if (zd < 0.0f) nb |= (1u << d);
                    az[d] = fabsf(zd);
                }
                uint32_t z_par = (uint32_t)__popc(nb) & 1u;
                float z_n2 = ((z[0] * z[0] + z[1] * z[1]) + (z[2] * z[2] + z[3] * z[3]))
                           + ((z[4] * z[4] + z[5] * z[5]) + (z[6] * z[6] + z[7] * z[7]));

                for (uint32_t k = 0u; k < 256u; ++k) {
                    float p0 = az[0] * sm_lut[k][0], p1 = az[1] * sm_lut[k][1];
                    float p2 = az[2] * sm_lut[k][2], p3 = az[3] * sm_lut[k][3];
                    float p4 = az[4] * sm_lut[k][4], p5 = az[5] * sm_lut[k][5];
                    float p6 = az[6] * sm_lut[k][6], p7 = az[7] * sm_lut[k][7];
                    float dot = ((p0 + p1) + (p2 + p3)) + ((p4 + p5) + (p6 + p7));
                    float min_p = p0; uint32_t min_d = 0u;
                    if (p1 < min_p) { min_p = p1; min_d = 1u; }
                    if (p2 < min_p) { min_p = p2; min_d = 2u; }
                    if (p3 < min_p) { min_p = p3; min_d = 3u; }
                    if (p4 < min_p) { min_p = p4; min_d = 4u; }
                    if (p5 < min_p) { min_p = p5; min_d = 5u; }
                    if (p6 < min_p) { min_p = p6; min_d = 6u; }
                    if (p7 < min_p) { min_p = p7; min_d = 7u; }
                    bool flip = (z_par != sm_par[k]);
                    float adj = flip ? (dot - 2.0f * min_p) : dot;
                    float dist = (z_n2 + sm_n2[k]) - 2.0f * adj;
                    if (dist < best_d) {
                        best_d = dist;
                        best_coset = coset;
                        best_k = k;
                        best_nb = flip ? (nb ^ (1u << min_d)) : nb;
                    }
                }
            }
            codes[h] = (best_coset << 15) | ((best_nb & 0x7Fu) << 8) | best_k;
            #pragma unroll
            for (uint32_t d = 0u; d < 8u; ++d) {
                float sgn = ((best_nb >> d) & 1u) ? -1.0f : 1.0f;
                float q = sgn * sm_lut[best_k][d] - 0.25f;
                if (best_coset != 0u) q = -q;
                float e = x[d] - q * sc;
                my_err += e * e;
            }
        }
        out_words[gtid] = codes[0] | (codes[1] << 16);
    }

    #pragma unroll
    for (int d = 16; d >= 1; d >>= 1) {
        my_sig += __shfl_xor_sync(0xFFFFFFFFu, my_sig, d);
        my_err += __shfl_xor_sync(0xFFFFFFFFu, my_err, d);
    }
    __shared__ float sm_sig[8], sm_err[8];
    if ((tid & 31u) == 0u) {
        sm_sig[tid >> 5] = my_sig;
        sm_err[tid >> 5] = my_err;
    }
    __syncthreads();
    if (tid == 0u) {
        float bs = 0.0f, be = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) { bs += sm_sig[w]; be += sm_err[w]; }
        metrics[(uint64_t)blockIdx.x * 2ull + 0ull] = bs;
        metrics[(uint64_t)blockIdx.x * 2ull + 1ull] = be;
    }
}

static std::vector<std::vector<size_t>> partition_tfq4_shards(
    const std::vector<tfq_generic::QuantJob>& jobs, uint64_t max_shard_bytes
) {
    std::vector<std::vector<size_t>> shards;
    std::vector<size_t> cur;
    uint64_t cur_bytes = 0;
    for (size_t i = 0; i < jobs.size(); ++i) {
        uint64_t b = tfq4::tfp4_tensor_bytes(jobs[i].pad_R, jobs[i].pad_C);
        if (!cur.empty() && cur_bytes + b > max_shard_bytes) {
            shards.push_back(std::move(cur));
            cur.clear();
            cur_bytes = 0;
        }
        cur.push_back(i);
        cur_bytes += b;
    }
    if (!cur.empty()) shards.push_back(std::move(cur));
    return shards;
}

int main(int argc, char** argv) {
    tfq_generic::GenericOptions opt;
    opt.default_bits = 3;
    opt.head_bits = 0;
    double max_shard_gib = 3.4;
    uint32_t table_kind = tfq4::PWV4_TABLE_LEGACY_TRAINED;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "--model" || a == "-m") && i + 1 < argc) opt.model_dir = argv[++i];
        else if ((a == "--out" || a == "-o") && i + 1 < argc) opt.out_dir = argv[++i];
        else if (a == "--quantize-vision") opt.quantize_vision = true;
        else if (a == "--quantize-drafter") opt.quantize_drafter = true;
        else if (a == "--quantize-embed") opt.quantize_embed = true;
        else if (a == "--head-bits" && i + 1 < argc) opt.head_bits = (uint32_t)std::stoul(argv[++i]);
        else if (a == "--max-shard-gib" && i + 1 < argc) max_shard_gib = std::stod(argv[++i]);
        else if (a == "--table-kind" && i + 1 < argc) {
            std::string k = argv[++i];
            if (k == "trained" || k == "0") table_kind = tfq4::PWV4_TABLE_LEGACY_TRAINED;
            else if (k == "1") table_kind = tfq4::PWV4_TABLE_TRAINED;
            else if (k == "e8p" || k == "true-e8p" || k == "2") table_kind = tfq4::PWV4_TABLE_TRUE_E8P;
            else {
                fprintf(stderr, "tfq4-convert-cuda: invalid --table-kind '%s' (expected trained|e8p|0|1|2)\n", k.c_str());
                return 2;
            }
        }
        else if (a == "--dry-run") opt.dry_run = true;
        else if (a == "-h" || a == "--help") {
            printf("Usage: tfq4-convert-cuda --model DIR --out DIR [--head-bits 0|2] [--table-kind trained|e8p] [--max-shard-gib 3.4] [--quantize-vision] [--quantize-drafter] [--quantize-embed] [--dry-run]\n");
            return 0;
        }
    }
    if (opt.model_dir.empty() || opt.out_dir.empty()) {
        fprintf(stderr, "Usage: tfq4-convert-cuda --model DIR --out DIR [--head-bits 0|2] [--table-kind trained|e8p] [--max-shard-gib 3.4] [--quantize-vision] [--quantize-drafter] [--quantize-embed] [--dry-run]\n");
        return 2;
    }
    if (opt.head_bits != 0u && opt.head_bits != 2u) {
        fprintf(stderr, "tfq4-convert-cuda: invalid --head-bits %u (TFQ4 supports 0 [keep lm_head in residual, default] or 2 [quantize lm_head to 2-bit])\n", opt.head_bits);
        return 2;
    }
    uint64_t max_shard_bytes = (uint64_t)std::max(1.0 * (double)(1ULL << 20), max_shard_gib * (double)(1ULL << 30));

    tfq_generic::SafetensorsDir st;
    st.init(opt.model_dir);
    auto plan = tfq_generic::build_plan(st, opt);
    if (plan.jobs.empty()) { fprintf(stderr, "No quantizable 2D float tensors in %s\n", opt.model_dir.c_str()); return 1; }

    uint64_t est_tfq4_bytes = 0;
    for (auto& j : plan.jobs) {
        j.bits = 2u;
        est_tfq4_bytes += tfq4::tfp4_tensor_bytes(j.pad_R, j.pad_C);
    }
    plan.tfq_bytes = est_tfq4_bytes;
    auto shards = partition_tfq4_shards(plan.jobs, max_shard_bytes);

    printf("TFQ4-E8P (2.001 bpw) Plan: %zu quantized tensors (%.3f GiB across %zu shard(s)) + %zu residual (%.3f GiB) = %.3f GiB total\n",
           plan.jobs.size(), est_tfq4_bytes / (double)(1ULL << 30), shards.size(),
           plan.residual_names.size(), plan.residual_bytes / (double)(1ULL << 30),
           (est_tfq4_bytes + plan.residual_bytes) / (double)(1ULL << 30));
    if (opt.dry_run) return 0;

    const bool use_true_e8p = (table_kind == tfq4::PWV4_TABLE_TRUE_E8P);
    auto t_start = std::chrono::steady_clock::now();
    fs::create_directories(opt.out_dir);
    tfq4::PwV4E8Tables tbl;
    if (use_true_e8p) {
        tfq4::build_true_e8p_tables(tbl);
    } else {
        tfq4::build_e8p_tables(tbl);
        tbl.table_kind = table_kind;
    }
    {
        std::string tf_path = (fs::path(opt.out_dir) / "pw_v4_e8p_tables.bin").string();
        std::ofstream tf(tf_path, std::ios::binary);
        tf.write(reinterpret_cast<const char*>(&tbl), sizeof(tbl));
        tfq_generic::close_or_die(tf, tf_path);
    }

    uint32_t max_pad_C = 128u;
    for (const auto& j : plan.jobs) max_pad_C = std::max(max_pad_C, j.pad_C);
    const uint32_t MAX_CHUNK_R = 4096u;
    const size_t MAX_ELEMS = (size_t)MAX_CHUNK_R * max_pad_C;
    const size_t MAX_BLK   = (size_t)(MAX_CHUNK_R / 128u) * (max_pad_C / 128u);
    const size_t MAX_WORDS = (size_t)MAX_CHUNK_R * (max_pad_C / 16u);

    __half *d_lut = nullptr, *d_sc = nullptr;
    float *d_in = nullptr, *d_tmp = nullptr, *d_rot = nullptr, *d_ss = nullptr;
    uint32_t* d_out = nullptr;
    float* d_met = nullptr;
    const size_t MAX_MET_BLKS = (MAX_WORDS + 255ULL) / 256ULL;
    CUDA_CHECK(cudaMalloc((void**)&d_lut, sizeof(tbl.e8p_lut)));
    CUDA_CHECK(cudaMemcpy(d_lut, tbl.e8p_lut, sizeof(tbl.e8p_lut), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void**)&d_met, MAX_MET_BLKS * 2ULL * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_in,  MAX_ELEMS * 4ULL));
    CUDA_CHECK(cudaMalloc((void**)&d_tmp, MAX_ELEMS * 4ULL));
    CUDA_CHECK(cudaMalloc((void**)&d_rot, MAX_ELEMS * 4ULL));
    CUDA_CHECK(cudaMalloc((void**)&d_ss,  MAX_BLK * 128ULL * 4ULL));
    CUDA_CHECK(cudaMalloc((void**)&d_sc,  MAX_BLK   * 2ULL));
    CUDA_CHECK(cudaMalloc((void**)&d_out, MAX_WORDS * 4ULL));

    std::vector<float> host_chunk;
    std::vector<std::string> shard_names;
    double snr_sum = 0.0;
    uint64_t total_written = 0;

    for (size_t si = 0; si < shards.size(); ++si) {
        const auto& idxs = shards[si];
        std::vector<tfq4::Tfp4TensorMeta> metas;
        metas.reserve(idxs.size());
        std::vector<uint16_t> all_scales;
        std::vector<uint32_t> all_words;

        for (size_t ji : idxs) {
            auto& j = plan.jobs[ji];
            const uint32_t R = j.pad_R, C = j.pad_C;
            const uint32_t n_bc = C / 128u;
            const uint32_t n_sc_tot = (R / 128u) * n_bc;
            const uint64_t n_w_tot  = (uint64_t)R * (C / 16u);

            tfq4::Tfp4TensorMeta m{};
            std::strncpy(m.name, j.tfpn_name.c_str(), sizeof(m.name) - 1);
            m.R = R; m.C = C; m.b_bits = 2u;
            m.sc_off = (uint32_t)all_scales.size();
            m.word_off = (uint64_t)all_words.size();
            m.word_count = n_w_tot;
            m.pad[0] = j.orig_R; m.pad[1] = j.orig_C; m.pad[2] = j.flags;

            all_scales.resize(all_scales.size() + n_sc_tot);
            all_words.resize(all_words.size() + n_w_tot);

            double tot_sig = 0.0, tot_err = 0.0;
            uint32_t sc_cur = 0;
            uint64_t w_cur = 0;
            for (uint32_t r0 = 0; r0 < R; r0 += MAX_CHUNK_R) {
                uint32_t rc = std::min(MAX_CHUNK_R, R - r0);
                tfq_generic::decode_chunk_to_f32(st, j, r0, rc, host_chunk);
                uint32_t n_br = rc / 128u, n_blk = n_br * n_bc;
                uint64_t n_words = (uint64_t)rc * (C / 16u);

                uint32_t n_mblk = (uint32_t)((n_words + 255ull) / 256ull);
                CUDA_CHECK(cudaMemcpy(d_in, host_chunk.data(), host_chunk.size() * 4ULL, cudaMemcpyHostToDevice));

                f32_row_fwht_k<<<dim3(n_bc, rc), 32>>>(d_in, d_tmp, rc, C);
                col_fwht_ss_k<<<dim3(C, n_br), 32>>>(d_tmp, d_rot, d_ss, rc, C, r0);
                finalize_scales_k<<<n_blk, 32>>>(d_ss, d_sc, n_blk, use_true_e8p ? tfq4::TRUE_E8P_BASE_SCALE : 1.0f);
                if (use_true_e8p) {
                    tfq4_true_e8p_encode_k<<<n_mblk, 256>>>(d_rot, d_sc, d_lut, d_out, d_met, rc, C);
                } else {
                    tfq4_e8p_encode_k<<<n_mblk, 256>>>(d_rot, d_sc, d_lut, d_out, d_met, rc, C);
                }
                CUDA_CHECK(cudaDeviceSynchronize());

                std::vector<float> h_met((size_t)n_mblk * 2u);
                CUDA_CHECK(cudaMemcpy(h_met.data(), d_met, h_met.size() * sizeof(float), cudaMemcpyDeviceToHost));
                for (uint32_t mi = 0; mi < n_mblk; ++mi) {
                    tot_sig += (double)h_met[(size_t)mi * 2u + 0u];
                    tot_err += (double)h_met[(size_t)mi * 2u + 1u];
                }

                CUDA_CHECK(cudaMemcpy(all_scales.data() + m.sc_off + sc_cur, d_sc, (size_t)n_blk * 2ULL, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(all_words.data() + m.word_off + w_cur, d_out, (size_t)n_words * 4ULL, cudaMemcpyDeviceToHost));
                sc_cur += n_blk;
                w_cur += n_words;
            }

            float snr = (tot_sig > 0.0) ? (float)(10.0 * std::log10(tot_sig / std::max(tot_err, 1e-18))) : 0.0f;
            j.snr_db = snr;
            j.bits = 2u;
            m.snr_db = snr;
            snr_sum += snr;
            metas.push_back(m);
        }

        char fn[96];
        snprintf(fn, sizeof(fn), "tfq4-%05zu-of-%05zu.tffp4", si + 1, shards.size());
        shard_names.push_back(fn);
        uint64_t sz = tfq4::write_tfp4_shard((fs::path(opt.out_dir) / fn).string(), metas, all_scales, all_words);
        total_written += sz;
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
        printf("  wrote %s (%zu tensors, %.3f GiB) [%.1fs]\n", fn, metas.size(), sz / (double)(1ULL << 30), dt);
        fflush(stdout);
    }

    cudaFree(d_lut); cudaFree(d_met); cudaFree(d_in); cudaFree(d_tmp);
    cudaFree(d_rot); cudaFree(d_ss); cudaFree(d_sc); cudaFree(d_out);

    plan.tfq_bytes = total_written;
    tfq_generic::package_residuals_and_manifest(
        st, opt, plan, shard_names, "TFQ4-E8P", tfq4::FORMAT_VER, "pw_v4_e8p_tables.bin", "tfq4_manifest.json"
    );
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    printf("TFQ4-E8P CUDA DONE: %zu tensors -> %.3f GiB (%zu shard(s), 2.001 bpw) | mean SNR=%.2f dB | %.1fs -> %s\n",
           plan.jobs.size(), total_written / (double)(1ULL << 30), shards.size(), snr_sum / plan.jobs.size(), secs, opt.out_dir.c_str());
    return 0;
}
