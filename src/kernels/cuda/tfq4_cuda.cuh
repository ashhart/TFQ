#pragma once
// ============================================================================
// Standalone NVIDIA CUDA TFQ4-E8P (2.001 & 2.251 bpw) Inference Kernel Library
//   - Byte-identical PWV4 v4 + TFP4 v4 binary format with Apple Silicon Metal
//   - Supports table_kind=0/1 (trained 8-sign LUT, 2.001 bpw), table_kind=2
//     (True QuIP# E8P: E8 + 1/4, 2.001 bpw), AND table_kind=3 (True QuIP# E8P-1024:
//     E8 + 1/4, 1024-entry 16 KiB LUT, 2.251 bpw, 9 words/128 cols)
//   - Shared-memory LUT (4 KiB or 16 KiB) + register-resident 4-row-batch loop
//   - Bit-exact rows=1 == rows=16 (and arbitrary rows >= 1) + fused inv-FWHT-128
//   - Supports Col-TP (row_off) and Row-TP (k_sign_off)
//   - Supported architectures: sm_80+ (Ampere, Ada, Hopper, Blackwell / GB10)
// ============================================================================
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

#define TFQ4_HAS_E8P1024 1

namespace tfq4_cuda {

static constexpr uint32_t TABLE_KIND_LEGACY_TRAINED = 0u;
static constexpr uint32_t TABLE_KIND_TRAINED        = 1u;
static constexpr uint32_t TABLE_KIND_TRUE_E8P       = 2u;
static constexpr uint32_t TABLE_KIND_TRUE_E8P_1024  = 3u;
static constexpr uint32_t TABLE_KIND_MAX_KNOWN      = 3u;

__device__ __forceinline__ __half2 u32_to_h2(uint32_t u) {
    return *reinterpret_cast<const __half2*>(&u);
}

__device__ __forceinline__ float bf16_to_f32(uint16_t b) {
    return __uint_as_float(static_cast<uint32_t>(b) << 16);
}

__device__ __forceinline__ uint16_t f32_to_bf16_u16(float f) {
    __nv_bfloat16 b = __float2bfloat16(f);
    return *reinterpret_cast<const uint16_t*>(&b);
}

__device__ __forceinline__ __half float2half_sat(float v) {
    return __float2half(fminf(fmaxf(v, -65504.0f), 65504.0f));
}

__device__ __forceinline__ float pw_sgn(uint32_t seed, uint32_t i) {
    uint32_t h = (seed ^ i) * 2654435761u;
    return ((h >> 31) & 1u) ? -1.0f : 1.0f;
}

__device__ __forceinline__ void pw_fwht128_warp4(float4& v, uint32_t lane) {
    #pragma unroll
    for (int d = 1; d <= 16; d <<= 1) {
        float ox = __shfl_xor_sync(0xFFFFFFFFu, v.x, d);
        float oy = __shfl_xor_sync(0xFFFFFFFFu, v.y, d);
        float oz = __shfl_xor_sync(0xFFFFFFFFu, v.z, d);
        float ow = __shfl_xor_sync(0xFFFFFFFFu, v.w, d);
        bool lo = ((lane & static_cast<uint32_t>(d)) == 0u);
        v.x = lo ? (v.x + ox) : (ox - v.x);
        v.y = lo ? (v.y + oy) : (oy - v.y);
        v.z = lo ? (v.z + oz) : (oz - v.z);
        v.w = lo ? (v.w + ow) : (ow - v.w);
    }
    float a0 = v.x + v.y, b0 = v.x - v.y;
    float a1 = v.z + v.w, b1 = v.z - v.w;
    constexpr float inv = 0.08838834764831845f;
    v = make_float4((a0 + a1) * inv, (b0 + b1) * inv, (a0 - a1) * inv, (b0 - b1) * inv);
}

// ----------------------------------------------------------------------------
// 1. Input FWHT-128 Kernels (FP16 and BF16 inputs -> saturated FP16 X_rot)
// ----------------------------------------------------------------------------
__global__ void tfq4_fwht_x_f16_kernel(
    const __half* __restrict__ X,
    __half* __restrict__ X_rot,
    uint32_t rows,
    uint32_t k,
    uint32_t ldx,
    uint32_t k_sign_off = 0u
) {
    const uint32_t lane = threadIdx.x;
    const uint32_t c0 = blockIdx.x * 128u;
    if (c0 >= k) return;
    const uint32_t cs = c0 + k_sign_off;
    for (uint32_t row = blockIdx.y; row < rows; row += gridDim.y) {
        const __half* src = X + static_cast<uint64_t>(row) * ldx + c0;
        float4 v;
        v.x = __half2float(src[lane +  0u]) * pw_sgn(0x12345678u, cs + lane +  0u);
        v.y = __half2float(src[lane + 32u]) * pw_sgn(0x12345678u, cs + lane + 32u);
        v.z = __half2float(src[lane + 64u]) * pw_sgn(0x12345678u, cs + lane + 64u);
        v.w = __half2float(src[lane + 96u]) * pw_sgn(0x12345678u, cs + lane + 96u);
        pw_fwht128_warp4(v, lane);
        __half* dst = X_rot + static_cast<uint64_t>(row) * k + c0;
        dst[lane +  0u] = float2half_sat(v.x);
        dst[lane + 32u] = float2half_sat(v.y);
        dst[lane + 64u] = float2half_sat(v.z);
        dst[lane + 96u] = float2half_sat(v.w);
    }
}

__global__ void tfq4_fwht_x_bf16_kernel(
    const uint16_t* __restrict__ X,
    __half* __restrict__ X_rot,
    uint32_t rows,
    uint32_t k,
    uint32_t ldx,
    uint32_t k_sign_off = 0u
) {
    const uint32_t lane = threadIdx.x;
    const uint32_t c0 = blockIdx.x * 128u;
    if (c0 >= k) return;
    const uint32_t cs = c0 + k_sign_off;
    for (uint32_t row = blockIdx.y; row < rows; row += gridDim.y) {
        const uint16_t* src = X + static_cast<uint64_t>(row) * ldx + c0;
        float4 v;
        v.x = bf16_to_f32(src[lane +  0u]) * pw_sgn(0x12345678u, cs + lane +  0u);
        v.y = bf16_to_f32(src[lane + 32u]) * pw_sgn(0x12345678u, cs + lane + 32u);
        v.z = bf16_to_f32(src[lane + 64u]) * pw_sgn(0x12345678u, cs + lane + 64u);
        v.w = bf16_to_f32(src[lane + 96u]) * pw_sgn(0x12345678u, cs + lane + 96u);
        pw_fwht128_warp4(v, lane);
        __half* dst = X_rot + static_cast<uint64_t>(row) * k + c0;
        dst[lane +  0u] = float2half_sat(v.x);
        dst[lane + 32u] = float2half_sat(v.y);
        dst[lane + 64u] = float2half_sat(v.z);
        dst[lane + 96u] = float2half_sat(v.w);
    }
}

// ----------------------------------------------------------------------------
// 2. E8P 8D Codeword Decoders & Deterministic Dot-8 Helpers
// ----------------------------------------------------------------------------
__device__ __forceinline__ float e8p_xsum8_qtr(uint4 xv) {
    float2 x0 = __half22float2(u32_to_h2(xv.x));
    float2 x1 = __half22float2(u32_to_h2(xv.y));
    float2 x2 = __half22float2(u32_to_h2(xv.z));
    float2 x3 = __half22float2(u32_to_h2(xv.w));
    float s0 = __fadd_rn(x0.x, x0.y);
    float s1 = __fadd_rn(x1.x, x1.y);
    float s2 = __fadd_rn(x2.x, x2.y);
    float s3 = __fadd_rn(x3.x, x3.y);
    return __fmul_rn(__fadd_rn(__fadd_rn(s0, s1), __fadd_rn(s2, s3)), 0.25f);
}

template <int TABLE_KIND>
__device__ __forceinline__ void decode8_signed_mag(
    uint4 lut_u4, uint32_t m_par, uint32_t code, float4& wf0, float4& wf1
) {
    static_assert(TABLE_KIND == 0 || TABLE_KIND == 2 || TABLE_KIND == 3,
                  "TABLE_KIND must be 0 (trained), 2 (true-E8P), or 3 (true-E8P-1024)");
    uint32_t smask;
    if constexpr (TABLE_KIND == 2 || TABLE_KIND == 3) {
        uint32_t s7 = (code >> 8) & 0x7Fu;
        uint32_t s7_bit = (static_cast<uint32_t>(__popc(s7)) ^ m_par) & 1u;
        smask = s7 | (s7_bit << 7);
    } else {
        (void)m_par;
        smask = code >> 8;
    }
    uint32_t f0 = (((smask >> 0) & 1u) << 15) | (((smask >> 1) & 1u) << 31);
    uint32_t f1 = (((smask >> 2) & 1u) << 15) | (((smask >> 3) & 1u) << 31);
    uint32_t f2 = (((smask >> 4) & 1u) << 15) | (((smask >> 5) & 1u) << 31);
    uint32_t f3 = (((smask >> 6) & 1u) << 15) | (((smask >> 7) & 1u) << 31);
    float2 w0 = __half22float2(u32_to_h2(lut_u4.x ^ f0));
    float2 w1 = __half22float2(u32_to_h2(lut_u4.y ^ f1));
    float2 w2 = __half22float2(u32_to_h2(lut_u4.z ^ f2));
    float2 w3 = __half22float2(u32_to_h2(lut_u4.w ^ f3));
    wf0 = make_float4(w0.x, w0.y, w1.x, w1.y);
    wf1 = make_float4(w2.x, w2.y, w3.x, w3.y);
}

template <int TABLE_KIND>
__device__ __forceinline__ float dot8_decoded(
    float4 w0, float4 w1, uint4 xv, float xsum_qtr, uint32_t csign
) {
    float2 x0 = __half22float2(u32_to_h2(xv.x));
    float2 x1 = __half22float2(u32_to_h2(xv.y));
    float2 x2 = __half22float2(u32_to_h2(xv.z));
    float2 x3 = __half22float2(u32_to_h2(xv.w));
    float s0 = __fmaf_rn(w0.x, x0.x, __fmul_rn(w0.y, x0.y));
    float s1 = __fmaf_rn(w0.z, x1.x, __fmul_rn(w0.w, x1.y));
    float s2 = __fmaf_rn(w1.x, x2.x, __fmul_rn(w1.y, x2.y));
    float s3 = __fmaf_rn(w1.z, x3.x, __fmul_rn(w1.w, x3.y));
    float d = __fadd_rn(__fadd_rn(s0, s1), __fadd_rn(s2, s3));
    if constexpr (TABLE_KIND == 2 || TABLE_KIND == 3) {
        d = __fsub_rn(d, xsum_qtr);
        d = __uint_as_float(__float_as_uint(d) ^ csign);
    } else {
        (void)xsum_qtr;
        (void)csign;
    }
    return d;
}

__device__ __forceinline__ void warp_reduce8_store(
    float s, uint32_t sub_w, float* s_out, uint32_t row_m, uint32_t local_r
) {
    s = __fadd_rn(s, __shfl_xor_sync(0xFFFFFFFFu, s, 1));
    s = __fadd_rn(s, __shfl_xor_sync(0xFFFFFFFFu, s, 2));
    s = __fadd_rn(s, __shfl_xor_sync(0xFFFFFFFFu, s, 4));
    if (sub_w == 0u) s_out[row_m * 128u + local_r] = s;
}

// ----------------------------------------------------------------------------
// 3. Fused TFQ4 Dense Decode + Inv-FWHT-128 Kernel
//    grid = (N / 128, ceil(rows / 16), 1), block = 256 (8 warps).
//    Bit-exact across rows=1 == rows=16 (and any rows >= 1).
// ----------------------------------------------------------------------------
template <int TABLE_KIND, bool OUT_BF16>
__global__ void tfq4_dense_kernel(
    const uint32_t* __restrict__ W,
    const __half* __restrict__ scales,
    const uint4* __restrict__ e8p_lut,
    const __half* __restrict__ X_rot,
    void* __restrict__ OUT,
    uint32_t rows,
    uint32_t n,
    uint32_t k,
    uint32_t ldo,
    uint32_t row_off = 0u
) {
    static_assert(TABLE_KIND == 0 || TABLE_KIND == 2 || TABLE_KIND == 3,
                  "TABLE_KIND must be 0 (trained), 2 (true-E8P), or 3 (true-E8P-1024)");
    constexpr bool IS_TRUE_E8P = (TABLE_KIND == 2 || TABLE_KIND == 3);
    constexpr bool IS_E8P_1024 = (TABLE_KIND == 3);
    constexpr uint32_t LUT_SIZE = IS_E8P_1024 ? 1024u : 256u;
    constexpr uint32_t WORDS_PER_BLK = IS_E8P_1024 ? 9u : 8u;

    const uint32_t rb = blockIdx.x;
    const uint32_t r0 = rb * 128u;
    const uint32_t m0 = blockIdx.y * 16u;
    if (m0 >= rows || r0 >= n) return;

    const uint32_t nb = (rows - m0 < 16u) ? (rows - m0) : 16u;
    const uint32_t n_bc = k >> 7;
    const uint32_t words_per_row = n_bc * WORDS_PER_BLK;
    const uint32_t tid = threadIdx.x;
    const uint32_t warp = tid >> 5;
    const uint32_t lane = tid & 31u;
    const uint32_t sub_r = lane >> 3; // 0..3 (4 output rows per warp)
    const uint32_t sub_w = lane & 7u; // 0..7 (8 lanes = 128 cols)
    const uint32_t hi_shift = sub_w * 4u;

    // Shared memory: <= 13.5 KiB for kind=0/2, 26.25 KiB for kind=3 (<= 48 KiB default static SMEM).
    __shared__ uint4 s_lut[LUT_SIZE];
    __shared__ uint8_t s_par[IS_TRUE_E8P ? LUT_SIZE : 1];
    __shared__ uint4 s_x[4 * 16];
    __shared__ float s_xsum[IS_TRUE_E8P ? (4 * 16) : 1];
    __shared__ float s_out[16 * 128];

    for (uint32_t i = tid; i < LUT_SIZE; i += 256u) {
        uint4 u_lut = e8p_lut[i];
        s_lut[i] = u_lut;
        if constexpr (IS_TRUE_E8P) {
            float2 a0 = __half22float2(u32_to_h2(u_lut.x));
            float2 a1 = __half22float2(u32_to_h2(u_lut.y));
            float2 a2 = __half22float2(u32_to_h2(u_lut.z));
            float2 a3 = __half22float2(u32_to_h2(u_lut.w));
            uint32_t par = (static_cast<uint32_t>(static_cast<int>(a0.x))
                          + static_cast<uint32_t>(static_cast<int>(a0.y))
                          + static_cast<uint32_t>(static_cast<int>(a1.x))
                          + static_cast<uint32_t>(static_cast<int>(a1.y))
                          + static_cast<uint32_t>(static_cast<int>(a2.x))
                          + static_cast<uint32_t>(static_cast<int>(a2.y))
                          + static_cast<uint32_t>(static_cast<int>(a3.x))
                          + static_cast<uint32_t>(static_cast<int>(a3.y))) & 1u;
            s_par[i] = static_cast<uint8_t>(par);
        } else {
            (void)s_par;
            (void)s_xsum;
        }
    }
    __syncthreads();

    const __half* sc_row = scales + static_cast<uint64_t>(rb) * n_bc;

    if (nb == 1u) {
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint32_t bc = 0u; bc < n_bc; ++bc) {
            if (tid < 16u) {
                uint4 xv = *reinterpret_cast<const uint4*>(
                    X_rot + static_cast<uint64_t>(m0) * k + bc * 128u + tid * 8u
                );
                s_x[tid] = xv;
                if constexpr (IS_TRUE_E8P) {
                    s_xsum[tid] = e8p_xsum8_qtr(xv);
                }
            }
            __syncthreads();
            const float fsc = __half2float(__ldg(&sc_row[bc]));
            float xs0 = 0.0f, xs1 = 0.0f;
            if constexpr (IS_TRUE_E8P) {
                xs0 = s_xsum[sub_w * 2u];
                xs1 = s_xsum[sub_w * 2u + 1u];
            }
            #pragma unroll
            for (uint32_t pass = 0u; pass < 4u; ++pass) {
                const uint32_t local_r = warp * 16u + pass * 4u + sub_r;
                const uint32_t global_r = r0 + local_r;
                const uint64_t blk_off = static_cast<uint64_t>(global_r) * words_per_row + static_cast<uint64_t>(bc) * WORDS_PER_BLK;
                const uint32_t packed = __ldg(&W[blk_off + sub_w]);
                const uint32_t c0 = packed & 0xFFFFu;
                const uint32_t c1 = packed >> 16;
                uint32_t idx0 = c0 & 255u;
                uint32_t idx1 = c1 & 255u;
                if constexpr (IS_E8P_1024) {
                    const uint32_t hi_word = __ldg(&W[blk_off + 8u]);
                    idx0 |= ((hi_word >> hi_shift) & 3u) << 8;
                    idx1 |= ((hi_word >> (hi_shift + 2u)) & 3u) << 8;
                }
                uint32_t p0 = 0u, p1 = 0u;
                if constexpr (IS_TRUE_E8P) {
                    p0 = static_cast<uint32_t>(s_par[idx0]);
                    p1 = static_cast<uint32_t>(s_par[idx1]);
                }
                float4 w0a, w0b, w1a, w1b;
                decode8_signed_mag<TABLE_KIND>(s_lut[idx0], p0, c0, w0a, w0b);
                decode8_signed_mag<TABLE_KIND>(s_lut[idx1], p1, c1, w1a, w1b);
                const uint32_t cs0 = (c0 & 0x8000u) << 16;
                const uint32_t cs1 = (c1 & 0x8000u) << 16;
                float d = __fadd_rn(dot8_decoded<TABLE_KIND>(w0a, w0b, s_x[sub_w * 2u], xs0, cs0),
                                    dot8_decoded<TABLE_KIND>(w1a, w1b, s_x[sub_w * 2u + 1u], xs1, cs1));
                acc[pass] = __fmaf_rn(d, fsc, acc[pass]);
            }
            __syncthreads();
        }
        #pragma unroll
        for (uint32_t pass = 0u; pass < 4u; ++pass) {
            const uint32_t local_r = warp * 16u + pass * 4u + sub_r;
            warp_reduce8_store(acc[pass], sub_w, s_out, 0u, local_r);
        }
    } else {
        // Register-resident 4-row-batch loop (16 float accumulators, exact same order as nb==1 on row 0).
        for (uint32_t m_base = 0u; m_base < nb; m_base += 4u) {
            const uint32_t rem = (nb - m_base < 4u) ? (nb - m_base) : 4u;
            float a0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float a1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float a2[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float a3[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (uint32_t bc = 0u; bc < n_bc; ++bc) {
                const uint32_t n_u4 = rem * 16u;
                if (tid < n_u4) {
                    const uint32_t rl = tid >> 4;
                    const uint32_t cl = (tid & 15u) << 3;
                    uint4 xv = *reinterpret_cast<const uint4*>(
                        X_rot + static_cast<uint64_t>(m0 + m_base + rl) * k + bc * 128u + cl
                    );
                    s_x[tid] = xv;
                    if constexpr (IS_TRUE_E8P) {
                        s_xsum[tid] = e8p_xsum8_qtr(xv);
                    }
                }
                __syncthreads();
                const float fsc = __half2float(__ldg(&sc_row[bc]));
                float xs0a = 0.0f, xs0b = 0.0f;
                float xs1a = 0.0f, xs1b = 0.0f;
                float xs2a = 0.0f, xs2b = 0.0f;
                float xs3a = 0.0f, xs3b = 0.0f;
                if constexpr (IS_TRUE_E8P) {
                    xs0a = s_xsum[0u  + sub_w * 2u]; xs0b = s_xsum[0u  + sub_w * 2u + 1u];
                    if (rem > 1u) { xs1a = s_xsum[16u + sub_w * 2u]; xs1b = s_xsum[16u + sub_w * 2u + 1u]; }
                    if (rem > 2u) { xs2a = s_xsum[32u + sub_w * 2u]; xs2b = s_xsum[32u + sub_w * 2u + 1u]; }
                    if (rem > 3u) { xs3a = s_xsum[48u + sub_w * 2u]; xs3b = s_xsum[48u + sub_w * 2u + 1u]; }
                }
                #pragma unroll
                for (uint32_t pass = 0u; pass < 4u; ++pass) {
                    const uint32_t local_r = warp * 16u + pass * 4u + sub_r;
                    const uint32_t global_r = r0 + local_r;
                    const uint64_t blk_off = static_cast<uint64_t>(global_r) * words_per_row + static_cast<uint64_t>(bc) * WORDS_PER_BLK;
                    const uint32_t packed = __ldg(&W[blk_off + sub_w]);
                    const uint32_t c0 = packed & 0xFFFFu;
                    const uint32_t c1 = packed >> 16;
                    uint32_t idx0 = c0 & 255u;
                    uint32_t idx1 = c1 & 255u;
                    if constexpr (IS_E8P_1024) {
                        const uint32_t hi_word = __ldg(&W[blk_off + 8u]);
                        idx0 |= ((hi_word >> hi_shift) & 3u) << 8;
                        idx1 |= ((hi_word >> (hi_shift + 2u)) & 3u) << 8;
                    }
                    uint32_t p0 = 0u, p1 = 0u;
                    if constexpr (IS_TRUE_E8P) {
                        p0 = static_cast<uint32_t>(s_par[idx0]);
                        p1 = static_cast<uint32_t>(s_par[idx1]);
                    }
                    float4 w0a, w0b, w1a, w1b;
                    decode8_signed_mag<TABLE_KIND>(s_lut[idx0], p0, c0, w0a, w0b);
                    decode8_signed_mag<TABLE_KIND>(s_lut[idx1], p1, c1, w1a, w1b);
                    const uint32_t cs0 = (c0 & 0x8000u) << 16;
                    const uint32_t cs1 = (c1 & 0x8000u) << 16;
                    {
                        float d = __fadd_rn(dot8_decoded<TABLE_KIND>(w0a, w0b, s_x[0u + sub_w * 2u], xs0a, cs0),
                                            dot8_decoded<TABLE_KIND>(w1a, w1b, s_x[0u + sub_w * 2u + 1u], xs0b, cs1));
                        a0[pass] = __fmaf_rn(d, fsc, a0[pass]);
                    }
                    if (rem > 1u) {
                        float d = __fadd_rn(dot8_decoded<TABLE_KIND>(w0a, w0b, s_x[16u + sub_w * 2u], xs1a, cs0),
                                            dot8_decoded<TABLE_KIND>(w1a, w1b, s_x[16u + sub_w * 2u + 1u], xs1b, cs1));
                        a1[pass] = __fmaf_rn(d, fsc, a1[pass]);
                    }
                    if (rem > 2u) {
                        float d = __fadd_rn(dot8_decoded<TABLE_KIND>(w0a, w0b, s_x[32u + sub_w * 2u], xs2a, cs0),
                                            dot8_decoded<TABLE_KIND>(w1a, w1b, s_x[32u + sub_w * 2u + 1u], xs2b, cs1));
                        a2[pass] = __fmaf_rn(d, fsc, a2[pass]);
                    }
                    if (rem > 3u) {
                        float d = __fadd_rn(dot8_decoded<TABLE_KIND>(w0a, w0b, s_x[48u + sub_w * 2u], xs3a, cs0),
                                            dot8_decoded<TABLE_KIND>(w1a, w1b, s_x[48u + sub_w * 2u + 1u], xs3b, cs1));
                        a3[pass] = __fmaf_rn(d, fsc, a3[pass]);
                    }
                }
                __syncthreads();
            }
            #pragma unroll
            for (uint32_t pass = 0u; pass < 4u; ++pass) {
                const uint32_t local_r = warp * 16u + pass * 4u + sub_r;
                warp_reduce8_store(a0[pass], sub_w, s_out, m_base + 0u, local_r);
                if (rem > 1u) warp_reduce8_store(a1[pass], sub_w, s_out, m_base + 1u, local_r);
                if (rem > 2u) warp_reduce8_store(a2[pass], sub_w, s_out, m_base + 2u, local_r);
                if (rem > 3u) warp_reduce8_store(a3[pass], sub_w, s_out, m_base + 3u, local_r);
            }
        }
    }
    __syncthreads();

    const uint32_t sr0 = r0 + row_off;
    for (uint32_t m = warp; m < nb; m += 8u) {
        const float* ys = s_out + m * 128u;
        float4 v = make_float4(ys[lane], ys[lane + 32u], ys[lane + 64u], ys[lane + 96u]);
        pw_fwht128_warp4(v, lane);
        const uint64_t base = static_cast<uint64_t>(m0 + m) * ldo + r0;
        const float o0 = v.x * pw_sgn(0x9E3779B9u, sr0 + lane +  0u);
        const float o1 = v.y * pw_sgn(0x9E3779B9u, sr0 + lane + 32u);
        const float o2 = v.z * pw_sgn(0x9E3779B9u, sr0 + lane + 64u);
        const float o3 = v.w * pw_sgn(0x9E3779B9u, sr0 + lane + 96u);
        if constexpr (OUT_BF16) {
            uint16_t* dst = reinterpret_cast<uint16_t*>(OUT) + base;
            dst[lane +  0u] = f32_to_bf16_u16(o0);
            dst[lane + 32u] = f32_to_bf16_u16(o1);
            dst[lane + 64u] = f32_to_bf16_u16(o2);
            dst[lane + 96u] = f32_to_bf16_u16(o3);
        } else {
            float* dst = reinterpret_cast<float*>(OUT) + base;
            dst[lane +  0u] = o0;
            dst[lane + 32u] = o1;
            dst[lane + 64u] = o2;
            dst[lane + 96u] = o3;
        }
    }
}

// ----------------------------------------------------------------------------
// 4. Clean Host Launchers (FP16->FP32 and BF16->BF16/FP32)
// ----------------------------------------------------------------------------
inline cudaError_t launch_dense_f16_f32(
    const uint32_t* W,           // [n, (k/128)*(8 or 9)] packed TFQ4 words
    const __half* scales,        // [n / 128, k / 128] fp16 per-tile scales
    const uint4* e8p_lut,        // [256] (kind=0/1/2) or [1024] (kind=3) uint4 fp16 LUT
    const __half* X,             // [rows, ldx] fp16 input activations
    float* Y,                    // [rows, ldo] fp32 output activations
    __half* xrot_buf,            // [rows, k] fp16 scratch buffer for rotated X
    uint32_t table_kind,         // 0/1 = trained, 2 = true QuIP# E8P (2.0b), 3 = true QuIP# E8P-1024 (2.25b)
    uint32_t rows,               // batch / token rows (>= 1)
    uint32_t n,                  // output features (multiple of 128)
    uint32_t k,                  // input features (multiple of 128)
    uint32_t ldx,                // row stride of X in elements (>= k)
    uint32_t ldo,                // row stride of Y in elements (>= n)
    uint32_t row_off = 0u,       // absolute output row offset (multiple of 128, Col-TP)
    uint32_t k_sign_off = 0u,    // absolute input col offset (multiple of 128, Row-TP)
    cudaStream_t stream = nullptr
) {
    if (rows == 0u || n == 0u || k == 0u) return cudaSuccess;
    if ((n & 127u) != 0u || (k & 127u) != 0u || (row_off & 127u) != 0u || (k_sign_off & 127u) != 0u ||
        ldx < k || ldo < n || table_kind > TABLE_KIND_MAX_KNOWN) {
        return cudaErrorInvalidValue;
    }
    const uint32_t gy_x = (rows < 65535u) ? rows : 65535u;
    tfq4_fwht_x_f16_kernel<<<dim3(k >> 7, gy_x, 1u), 32u, 0, stream>>>(
        X, xrot_buf, rows, k, ldx, k_sign_off
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    const dim3 grid(n >> 7, (rows + 15u) >> 4, 1u);
    if (table_kind == TABLE_KIND_TRUE_E8P_1024) {
        tfq4_dense_kernel<3, false><<<grid, 256u, 0, stream>>>(
            W, scales, e8p_lut, xrot_buf, Y, rows, n, k, ldo, row_off
        );
    } else if (table_kind == TABLE_KIND_TRUE_E8P) {
        tfq4_dense_kernel<2, false><<<grid, 256u, 0, stream>>>(
            W, scales, e8p_lut, xrot_buf, Y, rows, n, k, ldo, row_off
        );
    } else {
        tfq4_dense_kernel<0, false><<<grid, 256u, 0, stream>>>(
            W, scales, e8p_lut, xrot_buf, Y, rows, n, k, ldo, row_off
        );
    }
    return cudaGetLastError();
}

inline cudaError_t launch_dense_bf16(
    const uint32_t* W,           // [n, (k/128)*(8 or 9)] packed TFQ4 words
    const __half* scales,        // [n / 128, k / 128] fp16 per-tile scales
    const uint4* e8p_lut,        // [256] (kind=0/1/2) or [1024] (kind=3) uint4 fp16 LUT
    const uint16_t* X,           // [rows, ldx] bf16 input activations
    void* Y,                     // [rows, ldo] bf16 (out_f32=false) or fp32 (out_f32=true)
    __half* xrot_buf,            // [rows, k] fp16 scratch buffer for rotated X
    uint32_t table_kind,         // 0/1 = trained, 2 = true QuIP# E8P (2.0b), 3 = true QuIP# E8P-1024 (2.25b)
    uint32_t rows,               // batch / token rows (>= 1)
    uint32_t n,                  // output features (multiple of 128)
    uint32_t k,                  // input features (multiple of 128)
    uint32_t ldx,                // row stride of X in elements (>= k)
    uint32_t ldo,                // row stride of Y in elements (>= n)
    uint32_t row_off = 0u,       // absolute output row offset (multiple of 128, Col-TP)
    uint32_t k_sign_off = 0u,    // absolute input col offset (multiple of 128, Row-TP)
    bool out_f32 = false,        // false -> uint16_t bf16 output, true -> float32 output
    cudaStream_t stream = nullptr
) {
    if (rows == 0u || n == 0u || k == 0u) return cudaSuccess;
    if ((n & 127u) != 0u || (k & 127u) != 0u || (row_off & 127u) != 0u || (k_sign_off & 127u) != 0u ||
        ldx < k || ldo < n || table_kind > TABLE_KIND_MAX_KNOWN) {
        return cudaErrorInvalidValue;
    }
    const uint32_t gy_x = (rows < 65535u) ? rows : 65535u;
    tfq4_fwht_x_bf16_kernel<<<dim3(k >> 7, gy_x, 1u), 32u, 0, stream>>>(
        X, xrot_buf, rows, k, ldx, k_sign_off
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    const dim3 grid(n >> 7, (rows + 15u) >> 4, 1u);
    if (table_kind == TABLE_KIND_TRUE_E8P_1024) {
        if (out_f32) {
            tfq4_dense_kernel<3, false><<<grid, 256u, 0, stream>>>(
                W, scales, e8p_lut, xrot_buf, Y, rows, n, k, ldo, row_off
            );
        } else {
            tfq4_dense_kernel<3, true><<<grid, 256u, 0, stream>>>(
                W, scales, e8p_lut, xrot_buf, Y, rows, n, k, ldo, row_off
            );
        }
    } else if (table_kind == TABLE_KIND_TRUE_E8P) {
        if (out_f32) {
            tfq4_dense_kernel<2, false><<<grid, 256u, 0, stream>>>(
                W, scales, e8p_lut, xrot_buf, Y, rows, n, k, ldo, row_off
            );
        } else {
            tfq4_dense_kernel<2, true><<<grid, 256u, 0, stream>>>(
                W, scales, e8p_lut, xrot_buf, Y, rows, n, k, ldo, row_off
            );
        }
    } else {
        if (out_f32) {
            tfq4_dense_kernel<0, false><<<grid, 256u, 0, stream>>>(
                W, scales, e8p_lut, xrot_buf, Y, rows, n, k, ldo, row_off
            );
        } else {
            tfq4_dense_kernel<0, true><<<grid, 256u, 0, stream>>>(
                W, scales, e8p_lut, xrot_buf, Y, rows, n, k, ldo, row_off
            );
        }
    }
    return cudaGetLastError();
}

} // namespace tfq4_cuda
