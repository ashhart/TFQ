#pragma once
// ============================================================================
// Shared NVIDIA CUDA TFQ4 True QuIP# E8P (2.001 bpw & 2.251 bpw)
// Deterministic GPU Encoder Kernels — compiled with --fmad=false
// --prec-div=true --prec-sqrt=true for 100% SHA-256 bit-identical output
// with src/converters/tfq4/tfq4_encode_common.metal.
//
// Shared by tfq4/convert.cu, inkling2/convert.cu, and glm/convert.cu.
// ============================================================================
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace tfq4_enc {

__device__ __forceinline__ __half2 u32_to_h2(uint32_t u) {
    return *reinterpret_cast<const __half2*>(&u);
}

__global__ void tfq4_finalize_scales_k(
    const float* __restrict__ col_ss,
    __half* __restrict__ sc,
    uint32_t n_blk,
    float scale_mult
) {
    uint32_t bi = blockIdx.x, lane = threadIdx.x;
    if (bi >= n_blk) return;
    const float* cs = col_ss + (uint64_t)bi * 128u;
    float s = (cs[lane + 0u] + cs[lane + 32u]) + (cs[lane + 64u] + cs[lane + 96u]);
    #pragma unroll
    for (int d = 16; d >= 1; d >>= 1) s += __shfl_xor_sync(0xFFFFFFFFu, s, d);
    if (lane == 0u) sc[bi] = __float2half_rn(sqrtf(s * (1.0f / 16384.0f) + 1e-12f) * scale_mult);
}

// Exact nearest-point encoder for True QuIP# E8P (PWV4 table_kind=2, E8 + 1/4, 2.001 bpw).
// Also works seamlessly when pointed at the first 256 entries of a table_kind=3 (1024-entry) LUT.
__global__ void tfq4_true_e8p_encode_k(
    const float* __restrict__ W_rot,
    const __half* __restrict__ scales,
    const __half* __restrict__ e8p_lut,
    uint32_t* __restrict__ out_words,
    float* __restrict__ metrics,
    uint32_t R,
    uint32_t C
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

// Exact nearest-point encoder for True QuIP# E8P-1024 (PWV4 table_kind=3, E8 + 1/4, 2.251 bpw).
// Packing per 128-col row-block = 9 uint32 words (16 vectors v_0..v_15):
//   w[0..7]: low16(v_{2*sub_w}) | (low16(v_{2*sub_w+1}) << 16)
//   w[8]:    sum_{j=0..15} (((abs_idx_j >> 8) & 3u) << (2*j))
__global__ void tfq4_true_e8p_1024_encode_k(
    const float* __restrict__ W_rot,
    const __half* __restrict__ scales,
    const uint4* __restrict__ e8p_lut,
    uint32_t* __restrict__ out_words,
    float* __restrict__ metrics,
    uint32_t R,
    uint32_t C
) {
    __shared__ uint4 sm_lut[1024];
    __shared__ float sm_n2[1024];
    __shared__ uint8_t sm_par[1024];
    uint32_t tid = threadIdx.x;

    for (uint32_t i = tid; i < 1024u; i += 256u) {
        uint4 u = e8p_lut[i];
        sm_lut[i] = u;
        float2 f0 = __half22float2(u32_to_h2(u.x));
        float2 f1 = __half22float2(u32_to_h2(u.y));
        float2 f2 = __half22float2(u32_to_h2(u.z));
        float2 f3 = __half22float2(u32_to_h2(u.w));
        float a0 = f0.x, a1 = f0.y, a2 = f1.x, a3 = f1.y;
        float a4 = f2.x, a5 = f2.y, a6 = f3.x, a7 = f3.y;
        sm_n2[i] = ((a0 * a0 + a1 * a1) + (a2 * a2 + a3 * a3))
                 + ((a4 * a4 + a5 * a5) + (a6 * a6 + a7 * a7));
        sm_par[i] = static_cast<uint8_t>(
            ((uint32_t)(int)a0 + (uint32_t)(int)a1 + (uint32_t)(int)a2 + (uint32_t)(int)a3
           + (uint32_t)(int)a4 + (uint32_t)(int)a5 + (uint32_t)(int)a6 + (uint32_t)(int)a7) & 1u
        );
    }
    __syncthreads();

    uint64_t gtid = (uint64_t)blockIdx.x * blockDim.x + tid;
    uint32_t n_bc = C >> 7;
    uint64_t total_row_blks = (uint64_t)R * n_bc;
    uint64_t row_blk = gtid >> 3;
    uint32_t sub_w = (uint32_t)(gtid & 7ull);
    float my_sig = 0.0f, my_err = 0.0f;
    uint32_t my_hi4 = 0u;

    if (row_blk < total_row_blks) {
        uint32_t r = (uint32_t)(row_blk / n_bc);
        uint32_t bc = (uint32_t)(row_blk - (uint64_t)r * n_bc);
        float sc = __half2float(scales[(r >> 7) * n_bc + bc]);
        float inv_sc = (sc > 0.0f) ? (1.0f / sc) : 0.0f;
        const float* src = W_rot + (uint64_t)r * C + (uint64_t)bc * 128ull + (uint64_t)sub_w * 16ull;

        uint32_t low16[2];
        uint32_t hi2[2];
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

                for (uint32_t k = 0u; k < 1024u; ++k) {
                    uint4 u = sm_lut[k];
                    float2 f0 = __half22float2(u32_to_h2(u.x));
                    float2 f1 = __half22float2(u32_to_h2(u.y));
                    float2 f2 = __half22float2(u32_to_h2(u.z));
                    float2 f3 = __half22float2(u32_to_h2(u.w));
                    float p0 = az[0] * f0.x, p1 = az[1] * f0.y;
                    float p2 = az[2] * f1.x, p3 = az[3] * f1.y;
                    float p4 = az[4] * f2.x, p5 = az[5] * f2.y;
                    float p6 = az[6] * f3.x, p7 = az[7] * f3.y;
                    float dot = ((p0 + p1) + (p2 + p3)) + ((p4 + p5) + (p6 + p7));
                    float min_p = p0; uint32_t min_d = 0u;
                    if (p1 < min_p) { min_p = p1; min_d = 1u; }
                    if (p2 < min_p) { min_p = p2; min_d = 2u; }
                    if (p3 < min_p) { min_p = p3; min_d = 3u; }
                    if (p4 < min_p) { min_p = p4; min_d = 4u; }
                    if (p5 < min_p) { min_p = p5; min_d = 5u; }
                    if (p6 < min_p) { min_p = p6; min_d = 6u; }
                    if (p7 < min_p) { min_p = p7; min_d = 7u; }
                    bool flip = (z_par != static_cast<uint32_t>(sm_par[k]));
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
            low16[h] = (best_coset << 15) | ((best_nb & 0x7Fu) << 8) | (best_k & 0xFFu);
            hi2[h]   = (best_k >> 8) & 3u;

            uint4 ub = sm_lut[best_k];
            float2 bf0 = __half22float2(u32_to_h2(ub.x));
            float2 bf1 = __half22float2(u32_to_h2(ub.y));
            float2 bf2 = __half22float2(u32_to_h2(ub.z));
            float2 bf3 = __half22float2(u32_to_h2(ub.w));
            float ba[8] = {bf0.x, bf0.y, bf1.x, bf1.y, bf2.x, bf2.y, bf3.x, bf3.y};
            #pragma unroll
            for (uint32_t d = 0u; d < 8u; ++d) {
                float sgn = ((best_nb >> d) & 1u) ? -1.0f : 1.0f;
                float q = sgn * ba[d] - 0.25f;
                if (best_coset != 0u) q = -q;
                float e = x[d] - q * sc;
                my_err += e * e;
            }
        }
        out_words[row_blk * 9ull + sub_w] = low16[0] | (low16[1] << 16);
        my_hi4 = (hi2[0] | (hi2[1] << 2)) << (sub_w * 4u);
    }

    uint32_t hi_word = my_hi4;
    hi_word |= __shfl_xor_sync(0xFFFFFFFFu, hi_word, 1);
    hi_word |= __shfl_xor_sync(0xFFFFFFFFu, hi_word, 2);
    hi_word |= __shfl_xor_sync(0xFFFFFFFFu, hi_word, 4);
    if (row_blk < total_row_blks && sub_w == 0u) {
        out_words[row_blk * 9ull + 8ull] = hi_word;
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

} // namespace tfq4_enc
