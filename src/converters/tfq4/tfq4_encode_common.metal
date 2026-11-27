#pragma once
// ============================================================================
// Shared Apple Silicon Metal TFQ4 True QuIP# E8P (2.001 bpw & 2.251 bpw)
// Deterministic GPU Encoder Kernels — compiled with -fno-fast-math for
// 100% SHA-256 bit-identical output with src/converters/tfq4/tfq4_encode_common.cuh.
//
// Shared by tfq4/convert.metal, inkling2/convert.metal, and glm/convert.metal.
// ============================================================================
#include <metal_stdlib>
using namespace metal;

#ifndef TF_UNROLL
#define TF_UNROLL _Pragma("clang loop unroll(full)")
#endif

struct Tfq4MatParams {
    uint R;
    uint C;
    uint r_off;
    uint pad0;
};

struct Tfq4ScaleParams {
    uint  n_blocks;
    float scale_mult;
};

static inline float tfq4_exact_fdiv(float a, float b) {
    float y = 1.0f / b;
    float e = fma(-b, y, 1.0f);
    y = fma(y, e, y);
    float q = a * y;
    float r = fma(-b, q, a);
    q = fma(r, y, q);
    float r2 = fma(-b, q, a);
    return fma(r2, y, q);
}

static inline float tfq4_det_simd_sum(float v) {
    v += simd_shuffle_xor(v, 16);
    v += simd_shuffle_xor(v, 8);
    v += simd_shuffle_xor(v, 4);
    v += simd_shuffle_xor(v, 2);
    v += simd_shuffle_xor(v, 1);
    return v;
}

kernel void tfq4_finalize_scales(
    const device float* col_ss       [[buffer(0)]],
    device half* scales_f16          [[buffer(1)]],
    constant Tfq4ScaleParams& SP     [[buffer(2)]],
    uint bi                          [[threadgroup_position_in_grid]],
    ushort lane                      [[thread_index_in_simdgroup]]
) {
    if (bi >= SP.n_blocks) return;
    const device float* cs = col_ss + ulong(bi) * 128ul;
    float s = tfq4_det_simd_sum((cs[lane + 0u] + cs[lane + 32u]) + (cs[lane + 64u] + cs[lane + 96u]));
    if (lane == 0) scales_f16[bi] = half(sqrt(s * (1.0f / 16384.0f) + 1e-12f) * SP.scale_mult);
}

// Exact nearest-point encoder for True QuIP# E8P (PWV4 table_kind=2, E8 + 1/4, 2.001 bpw):
//   - 256 half-integer base vectors in |D8 + 1/2| x 128 parity-constrained signs x 2 cosets = 65,536 points in E8 + 1/4.
//   - Codeword (u16): bits[7:0]=abs_idx, bits[14:8]=s[0..6], bit[15]=coset; s[7] derived from parity.
//   - Also works seamlessly when pointed at the first 256 entries of a table_kind=3 (1024-entry) LUT.
kernel void tfq4_true_e8p_encode(
    const device float* W_rot   [[buffer(0)]],
    const device half* scales   [[buffer(1)]],
    const device half* e8p_lut  [[buffer(2)]],
    device uint* out_words      [[buffer(3)]],
    device float* metrics       [[buffer(4)]],
    constant Tfq4MatParams& P   [[buffer(5)]],
    uint gtid                   [[thread_position_in_grid]],
    uint tgid                   [[threadgroup_position_in_grid]],
    ushort tid                  [[thread_index_in_threadgroup]],
    ushort lane                 [[thread_index_in_simdgroup]],
    ushort sg                   [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float sm_lut[256][8];
    threadgroup float sm_n2[256];
    threadgroup uint  sm_par[256];
    {
        float a0 = float(e8p_lut[uint(tid) * 8u + 0u]);
        float a1 = float(e8p_lut[uint(tid) * 8u + 1u]);
        float a2 = float(e8p_lut[uint(tid) * 8u + 2u]);
        float a3 = float(e8p_lut[uint(tid) * 8u + 3u]);
        float a4 = float(e8p_lut[uint(tid) * 8u + 4u]);
        float a5 = float(e8p_lut[uint(tid) * 8u + 5u]);
        float a6 = float(e8p_lut[uint(tid) * 8u + 6u]);
        float a7 = float(e8p_lut[uint(tid) * 8u + 7u]);
        sm_lut[tid][0] = a0; sm_lut[tid][1] = a1; sm_lut[tid][2] = a2; sm_lut[tid][3] = a3;
        sm_lut[tid][4] = a4; sm_lut[tid][5] = a5; sm_lut[tid][6] = a6; sm_lut[tid][7] = a7;
        sm_n2[tid] = ((a0 * a0 + a1 * a1) + (a2 * a2 + a3 * a3))
                   + ((a4 * a4 + a5 * a5) + (a6 * a6 + a7 * a7));
        sm_par[tid] = (uint(int(a0)) + uint(int(a1)) + uint(int(a2)) + uint(int(a3))
                     + uint(int(a4)) + uint(int(a5)) + uint(int(a6)) + uint(int(a7))) & 1u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint wpr = P.C >> 4;
    ulong total_words = ulong(P.R) * wpr;
    float my_sig = 0.0f, my_err = 0.0f;

    if (ulong(gtid) < total_words) {
        uint r = gtid / wpr;
        uint wc = gtid - r * wpr;
        uint n_bc = P.C >> 7;
        float sc = float(scales[(r >> 7) * n_bc + (wc >> 3)]);
        float inv_sc = (sc > 0.0f) ? tfq4_exact_fdiv(1.0f, sc) : 0.0f;
        const device float* src = W_rot + ulong(r) * P.C + ulong(wc) * 16ul;

        uint codes[2];
        TF_UNROLL for (uint h = 0u; h < 2u; ++h) {
            float x[8], v[8];
            TF_UNROLL for (uint d = 0u; d < 8u; ++d) {
                float xd = src[h * 8u + d];
                x[d] = xd;
                v[d] = xd * inv_sc;
                my_sig += xd * xd;
            }
            float best_d = 1e30f;
            uint best_coset = 0u, best_k = 0u, best_nb = 0u;

            TF_UNROLL for (uint coset = 0u; coset < 2u; ++coset) {
                float z[8], az[8];
                uint nb = 0u;
                TF_UNROLL for (uint d = 0u; d < 8u; ++d) {
                    float zd = ((coset == 0u) ? v[d] : -v[d]) + 0.25f;
                    z[d] = zd;
                    if (zd < 0.0f) nb |= (1u << d);
                    az[d] = fabs(zd);
                }
                uint z_par = popcount(nb) & 1u;
                float z_n2 = ((z[0] * z[0] + z[1] * z[1]) + (z[2] * z[2] + z[3] * z[3]))
                           + ((z[4] * z[4] + z[5] * z[5]) + (z[6] * z[6] + z[7] * z[7]));

                for (uint k = 0u; k < 256u; ++k) {
                    float p0 = az[0] * sm_lut[k][0], p1 = az[1] * sm_lut[k][1];
                    float p2 = az[2] * sm_lut[k][2], p3 = az[3] * sm_lut[k][3];
                    float p4 = az[4] * sm_lut[k][4], p5 = az[5] * sm_lut[k][5];
                    float p6 = az[6] * sm_lut[k][6], p7 = az[7] * sm_lut[k][7];
                    float dot = ((p0 + p1) + (p2 + p3)) + ((p4 + p5) + (p6 + p7));
                    float min_p = p0; uint min_d = 0u;
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
            TF_UNROLL for (uint d = 0u; d < 8u; ++d) {
                float sgn = ((best_nb >> d) & 1u) ? -1.0f : 1.0f;
                float q = sgn * sm_lut[best_k][d] - 0.25f;
                if (best_coset != 0u) q = -q;
                float e = x[d] - q * sc;
                my_err += e * e;
            }
        }
        out_words[gtid] = codes[0] | (codes[1] << 16);
    }

    float sg_sig = tfq4_det_simd_sum(my_sig), sg_err = tfq4_det_simd_sum(my_err);
    threadgroup float sm_sig[8], sm_err[8];
    if (lane == 0) {
        sm_sig[sg] = sg_sig;
        sm_err[sg] = sg_err;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float bs = 0.0f, be = 0.0f;
        for (uint w = 0u; w < 8u; ++w) { bs += sm_sig[w]; be += sm_err[w]; }
        metrics[tgid * 2u + 0u] = bs;
        metrics[tgid * 2u + 1u] = be;
    }
}

// Exact nearest-point encoder for True QuIP# E8P-1024 (PWV4 table_kind=3, E8 + 1/4, 2.251 bpw):
//   - 1024 half-integer base vectors in |D8 + 1/2| x 128 parity-constrained signs x 2 cosets = 262,144 points in E8 + 1/4.
//   - Packing per 128-col row-block = 9 uint32 words (16 vectors v_0..v_15):
//       w[0..7]: low16(v_{2*sub_w}) | (low16(v_{2*sub_w+1}) << 16)
//       w[8]:    sum_{j=0..15} (((abs_idx_j >> 8) & 3u) << (2*j))
//   - Threadgroup memory: 16 KiB sm_lut + 4 KiB sm_n2 + 1 KiB sm_par + 64 B metrics = 21.06 KiB <= 32 KiB.
kernel void tfq4_true_e8p_1024_encode(
    const device float* W_rot   [[buffer(0)]],
    const device half* scales   [[buffer(1)]],
    const device uint4* e8p_lut [[buffer(2)]], // 1024 x uint4 (16 KiB fp16 LUT)
    device uint* out_words      [[buffer(3)]], // R * (C / 128) * 9 uint32 words
    device float* metrics       [[buffer(4)]], // [tg * 2 + 0]=sig, [tg * 2 + 1]=err
    constant Tfq4MatParams& P   [[buffer(5)]],
    uint gtid                   [[thread_position_in_grid]],
    uint tgid                   [[threadgroup_position_in_grid]],
    ushort tid                  [[thread_index_in_threadgroup]],
    ushort lane                 [[thread_index_in_simdgroup]],
    ushort sg                   [[simdgroup_index_in_threadgroup]]
) {
    threadgroup uint4 sm_lut[1024];
    threadgroup float sm_n2[1024];
    threadgroup uchar sm_par[1024];

    for (uint i = uint(tid); i < 1024u; i += 256u) {
        uint4 u = e8p_lut[i];
        sm_lut[i] = u;
        half2 h0 = as_type<half2>(u.x), h1 = as_type<half2>(u.y);
        half2 h2 = as_type<half2>(u.z), h3 = as_type<half2>(u.w);
        float a0 = float(h0.x), a1 = float(h0.y);
        float a2 = float(h1.x), a3 = float(h1.y);
        float a4 = float(h2.x), a5 = float(h2.y);
        float a6 = float(h3.x), a7 = float(h3.y);
        sm_n2[i] = ((a0 * a0 + a1 * a1) + (a2 * a2 + a3 * a3))
                 + ((a4 * a4 + a5 * a5) + (a6 * a6 + a7 * a7));
        sm_par[i] = uchar((uint(int(a0)) + uint(int(a1)) + uint(int(a2)) + uint(int(a3))
                         + uint(int(a4)) + uint(int(a5)) + uint(int(a6)) + uint(int(a7))) & 1u);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint n_bc = P.C >> 7;
    ulong total_row_blks = ulong(P.R) * ulong(n_bc);
    ulong row_blk = ulong(gtid) >> 3;
    uint sub_w = gtid & 7u;
    float my_sig = 0.0f, my_err = 0.0f;
    uint my_hi4 = 0u;

    if (row_blk < total_row_blks) {
        uint r = uint(row_blk / ulong(n_bc));
        uint bc = uint(row_blk - ulong(r) * ulong(n_bc));
        float sc = float(scales[(r >> 7) * n_bc + bc]);
        float inv_sc = (sc > 0.0f) ? tfq4_exact_fdiv(1.0f, sc) : 0.0f;
        const device float* src = W_rot + ulong(r) * P.C + ulong(bc) * 128ul + ulong(sub_w) * 16ul;

        uint low16[2];
        uint hi2[2];
        TF_UNROLL for (uint h = 0u; h < 2u; ++h) {
            float x[8], v[8];
            TF_UNROLL for (uint d = 0u; d < 8u; ++d) {
                float xd = src[h * 8u + d];
                x[d] = xd;
                v[d] = xd * inv_sc;
                my_sig += xd * xd;
            }
            float best_d = 1e30f;
            uint best_coset = 0u, best_k = 0u, best_nb = 0u;

            TF_UNROLL for (uint coset = 0u; coset < 2u; ++coset) {
                float z[8], az[8];
                uint nb = 0u;
                TF_UNROLL for (uint d = 0u; d < 8u; ++d) {
                    float zd = ((coset == 0u) ? v[d] : -v[d]) + 0.25f;
                    z[d] = zd;
                    if (zd < 0.0f) nb |= (1u << d);
                    az[d] = fabs(zd);
                }
                uint z_par = popcount(nb) & 1u;
                float z_n2 = ((z[0] * z[0] + z[1] * z[1]) + (z[2] * z[2] + z[3] * z[3]))
                           + ((z[4] * z[4] + z[5] * z[5]) + (z[6] * z[6] + z[7] * z[7]));

                for (uint k = 0u; k < 1024u; ++k) {
                    uint4 u = sm_lut[k];
                    half2 h0 = as_type<half2>(u.x), h1 = as_type<half2>(u.y);
                    half2 h2 = as_type<half2>(u.z), h3 = as_type<half2>(u.w);
                    float p0 = az[0] * float(h0.x), p1 = az[1] * float(h0.y);
                    float p2 = az[2] * float(h1.x), p3 = az[3] * float(h1.y);
                    float p4 = az[4] * float(h2.x), p5 = az[5] * float(h2.y);
                    float p6 = az[6] * float(h3.x), p7 = az[7] * float(h3.y);
                    float dot = ((p0 + p1) + (p2 + p3)) + ((p4 + p5) + (p6 + p7));
                    float min_p = p0; uint min_d = 0u;
                    if (p1 < min_p) { min_p = p1; min_d = 1u; }
                    if (p2 < min_p) { min_p = p2; min_d = 2u; }
                    if (p3 < min_p) { min_p = p3; min_d = 3u; }
                    if (p4 < min_p) { min_p = p4; min_d = 4u; }
                    if (p5 < min_p) { min_p = p5; min_d = 5u; }
                    if (p6 < min_p) { min_p = p6; min_d = 6u; }
                    if (p7 < min_p) { min_p = p7; min_d = 7u; }
                    bool flip = (z_par != uint(sm_par[k]));
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
            half2 bh0 = as_type<half2>(ub.x), bh1 = as_type<half2>(ub.y);
            half2 bh2 = as_type<half2>(ub.z), bh3 = as_type<half2>(ub.w);
            float ba[8] = {
                float(bh0.x), float(bh0.y), float(bh1.x), float(bh1.y),
                float(bh2.x), float(bh2.y), float(bh3.x), float(bh3.y)
            };
            TF_UNROLL for (uint d = 0u; d < 8u; ++d) {
                float sgn = ((best_nb >> d) & 1u) ? -1.0f : 1.0f;
                float q = sgn * ba[d] - 0.25f;
                if (best_coset != 0u) q = -q;
                float e = x[d] - q * sc;
                my_err += e * e;
            }
        }
        out_words[row_blk * 9ul + ulong(sub_w)] = low16[0] | (low16[1] << 16);
        my_hi4 = (hi2[0] | (hi2[1] << 2)) << (sub_w * 4u);
    }

    uint hi_word = my_hi4;
    hi_word |= simd_shuffle_xor(hi_word, 1);
    hi_word |= simd_shuffle_xor(hi_word, 2);
    hi_word |= simd_shuffle_xor(hi_word, 4);
    if (row_blk < total_row_blks && sub_w == 0u) {
        out_words[row_blk * 9ul + 8ul] = hi_word;
    }

    float sg_sig = tfq4_det_simd_sum(my_sig), sg_err = tfq4_det_simd_sum(my_err);
    threadgroup float sm_sig[8], sm_err[8];
    if (lane == 0) {
        sm_sig[sg] = sg_sig;
        sm_err[sg] = sg_err;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float bs = 0.0f, be = 0.0f;
        for (uint w = 0u; w < 8u; ++w) { bs += sm_sig[w]; be += sm_err[w]; }
        metrics[tgid * 2u + 0u] = bs;
        metrics[tgid * 2u + 1u] = be;
    }
}
