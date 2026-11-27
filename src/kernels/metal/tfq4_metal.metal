#include <metal_stdlib>
using namespace metal;

#define TF_UNROLL _Pragma("clang loop unroll(full)")

// ============================================================================
// Standalone Apple Silicon Metal TFQ4-E8P (2.001 bpw) Inference Kernel Library
//   - Shares exact 2D FWHT-128 input/output rotation & 128x128 fp16 scales with TFQ3
//   - 4 KiB positive-orthant 8D E8P codebook loaded ONCE into threadgroup memory
//   - Branchless half2 sign XOR: 16 weights decoded per uint32_t word (2.0 bpw)
//   - Bit-exact rows=1 == rows=16 speculative verification + fused inv-FWHT-128
// ============================================================================
struct MetalTfq4Params {
    uint rows;
    uint n;
    uint k;
    uint ldx;
    uint ldo;
    uint row_off;
    uint k_sign_off;
};

inline float pw_sgn(uint seed, uint i) {
    uint h = (seed ^ i) * 2654435761u;
    return ((h >> 31) & 1u) ? -1.0f : 1.0f;
}

inline void pw_fwht128_simd4(thread float4& v, ushort lane) {
    TF_UNROLL
    for (ushort d = 1; d <= 16; d <<= 1) {
        float4 o = simd_shuffle_xor(v, d);
        v = ((lane & d) == 0u) ? (v + o) : (o - v);
    }
    float a0 = v.x + v.y, b0 = v.x - v.y;
    float a1 = v.z + v.w, b1 = v.z - v.w;
    v = float4(a0 + a1, b0 + b1, a0 - a1, b0 - b1) * 0.08838834764831845f;
}

kernel void tfq4_fwht_x(
    const device half* X        [[buffer(0)]],
    device half* X_rot          [[buffer(1)]],
    constant MetalTfq4Params& P [[buffer(2)]],
    uint2 gid                   [[threadgroup_position_in_grid]],
    ushort lane                 [[thread_index_in_simdgroup]]
) {
    uint row = gid.y, c0 = gid.x * 128u, cs = c0 + P.k_sign_off;
    if (row >= P.rows || c0 >= P.k) return;
    const device half* src = X + ulong(row) * P.ldx + c0;
    float4 v;
    v.x = float(src[lane +  0u]) * pw_sgn(0x12345678u, cs + lane +  0u);
    v.y = float(src[lane + 32u]) * pw_sgn(0x12345678u, cs + lane + 32u);
    v.z = float(src[lane + 64u]) * pw_sgn(0x12345678u, cs + lane + 64u);
    v.w = float(src[lane + 96u]) * pw_sgn(0x12345678u, cs + lane + 96u);
    pw_fwht128_simd4(v, lane);
    device half* dst = X_rot + ulong(row) * P.k + c0;
    dst[lane +  0u] = half(clamp(v.x, -65504.0f, 65504.0f));
    dst[lane + 32u] = half(clamp(v.y, -65504.0f, 65504.0f));
    dst[lane + 64u] = half(clamp(v.z, -65504.0f, 65504.0f));
    dst[lane + 96u] = half(clamp(v.w, -65504.0f, 65504.0f));
}

inline float e8p_dot8(uint4 lut_u4, uint smask, const threadgroup uint4* xv) {
    uint s0 = (((smask >> 0) & 1u) << 15) | (((smask >> 1) & 1u) << 31);
    uint s1 = (((smask >> 2) & 1u) << 15) | (((smask >> 3) & 1u) << 31);
    uint s2 = (((smask >> 4) & 1u) << 15) | (((smask >> 5) & 1u) << 31);
    uint s3 = (((smask >> 6) & 1u) << 15) | (((smask >> 7) & 1u) << 31);
    half2 w0 = as_type<half2>(lut_u4.x ^ s0), w1 = as_type<half2>(lut_u4.y ^ s1);
    half2 w2 = as_type<half2>(lut_u4.z ^ s2), w3 = as_type<half2>(lut_u4.w ^ s3);
    uint4 x = *xv;
    float4 wf0 = float4(float(w0.x), float(w0.y), float(w1.x), float(w1.y));
    float4 wf1 = float4(float(w2.x), float(w2.y), float(w3.x), float(w3.y));
    half2 x0 = as_type<half2>(x.x), x1 = as_type<half2>(x.y), x2 = as_type<half2>(x.z), x3 = as_type<half2>(x.w);
    float4 xf0 = float4(float(x0.x), float(x0.y), float(x1.x), float(x1.y));
    float4 xf1 = float4(float(x2.x), float(x2.y), float(x3.x), float(x3.y));
    return dot(wf0, xf0) + dot(wf1, xf1);
}

// High-bandwidth TFQ4-E8P dense decode (rows = 1..16, bit-exact rows=1 == rows=16) + fused inv-FWHT-128:
// grid = (N / 128, ceil(rows / 16)), block = 256 (8 simdgroups).
// Each simdgroup processes 4 output rows in parallel across 8-lane (128-col) strips -> 32 rows/pass x 4 passes = 128 rows.
kernel void tfq4_e8p_dense(
    const device uint* W        [[buffer(0)]],
    const device half* scales   [[buffer(1)]],
    const device uint4* e8p_lut [[buffer(2)]], // 256 x uint4 = 4 KiB (8 fp16 centroids per entry)
    const device half* X_rot    [[buffer(3)]],
    device float* OUT           [[buffer(4)]],
    constant MetalTfq4Params& P [[buffer(5)]],
    uint2 gid                   [[threadgroup_position_in_grid]],
    ushort tid                  [[thread_index_in_threadgroup]],
    ushort sg                   [[simdgroup_index_in_threadgroup]],
    ushort lane                 [[thread_index_in_simdgroup]]
) {
    uint rb = gid.x, r0 = rb * 128u, m0 = gid.y * 16u;
    if (m0 >= P.rows) return;
    uint nb = min(16u, P.rows - m0);
    uint n_bc = P.k >> 7;
    uint words_per_row = P.k >> 4; // K / 16

    // 4 KiB E8P LUT + 4 KiB X_rot tile (16x128 half) + 8 KiB output epilogue (16x128 float) = 16 KiB total!
    threadgroup uint4 sm_lut[256];
    threadgroup uint4 sm_x[16 * 16]; // 16 rows x 16 uint4 (= 128 halfs per row)
    threadgroup float sm_out[16 * 128];

    sm_lut[tid] = e8p_lut[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint sub_r = lane >> 3; // 0..3 (4 rows per simdgroup)
    uint sub_w = lane & 7u; // 0..7 (8 uint32 words = 128 cols)

    // Interleave 4 rows per simdgroup (sg*16 + pass*4 + sub_r) to avoid 32 KiB L1 set aliasing on power-of-2 K!
    if (nb == 1u) {
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        const device half* sc_row = scales + ulong(rb) * n_bc;
        for (uint bc = 0u; bc < n_bc; ++bc) {
            if (tid < 16u) {
                sm_x[tid] = *reinterpret_cast<const device uint4*>(X_rot + ulong(m0) * P.k + bc * 128u + tid * 8u);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float fsc = float(sc_row[bc]);
            TF_UNROLL for (uint pass = 0u; pass < 4u; ++pass) {
                uint local_r = uint(sg) * 16u + pass * 4u + sub_r;
                uint global_r = r0 + local_r;
                uint packed = W[ulong(global_r) * words_per_row + bc * 8u + sub_w];
                uint c0 = packed & 0xFFFFu, c1 = packed >> 16;
                float d = e8p_dot8(sm_lut[c0 & 255u], c0 >> 8, &sm_x[sub_w * 2u])
                        + e8p_dot8(sm_lut[c1 & 255u], c1 >> 8, &sm_x[sub_w * 2u + 1u]);
                acc[pass] = fma(d, fsc, acc[pass]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        TF_UNROLL for (uint pass = 0u; pass < 4u; ++pass) {
            float s = acc[pass];
            s += simd_shuffle_xor(s, 1);
            s += simd_shuffle_xor(s, 2);
            s += simd_shuffle_xor(s, 4);
            if (sub_w == 0u) sm_out[uint(sg) * 16u + pass * 4u + sub_r] = s;
        }
    } else {
        // Register-resident 4-row-batch loop (zero acc[16] stack spill, bit-exact r1==r16!).
        const device half* sc_row = scales + ulong(rb) * n_bc;
        for (uint pass = 0u; pass < 4u; ++pass) {
            uint local_r = uint(sg) * 16u + pass * 4u + sub_r;
            uint global_r = r0 + local_r;
            const device uint* w_row = W + ulong(global_r) * words_per_row;
            for (uint m_base = 0u; m_base < nb; m_base += 4u) {
                float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
                uint rem = min(4u, nb - m_base);
                for (uint bc = 0u; bc < n_bc; ++bc) {
                    uint n_u4 = rem * 16u;
                    if (tid < n_u4) {
                        uint rl = tid >> 4, cl = (tid & 15u) << 3;
                        sm_x[tid] = *reinterpret_cast<const device uint4*>(X_rot + ulong(m0 + m_base + rl) * P.k + bc * 128u + cl);
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    uint packed = w_row[bc * 8u + sub_w];
                    uint c0 = packed & 0xFFFFu, c1 = packed >> 16;
                    uint4 l0 = sm_lut[c0 & 255u], l1 = sm_lut[c1 & 255u];
                    uint sm0 = c0 >> 8, sm1 = c1 >> 8;
                    float fsc = float(sc_row[bc]);
                    a0 = fma(e8p_dot8(l0, sm0, &sm_x[0u  + sub_w * 2u]) + e8p_dot8(l1, sm1, &sm_x[0u  + sub_w * 2u + 1u]), fsc, a0);
                    if (rem > 1u) a1 = fma(e8p_dot8(l0, sm0, &sm_x[16u + sub_w * 2u]) + e8p_dot8(l1, sm1, &sm_x[16u + sub_w * 2u + 1u]), fsc, a1);
                    if (rem > 2u) a2 = fma(e8p_dot8(l0, sm0, &sm_x[32u + sub_w * 2u]) + e8p_dot8(l1, sm1, &sm_x[32u + sub_w * 2u + 1u]), fsc, a2);
                    if (rem > 3u) a3 = fma(e8p_dot8(l0, sm0, &sm_x[48u + sub_w * 2u]) + e8p_dot8(l1, sm1, &sm_x[48u + sub_w * 2u + 1u]), fsc, a3);
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                auto red_wr = [&](uint m_off, float s) {
                    s += simd_shuffle_xor(s, 1); s += simd_shuffle_xor(s, 2); s += simd_shuffle_xor(s, 4);
                    if (sub_w == 0u) sm_out[(m_base + m_off) * 128u + local_r] = s;
                };
                red_wr(0u, a0);
                if (rem > 1u) red_wr(1u, a1);
                if (rem > 2u) red_wr(2u, a2);
                if (rem > 3u) red_wr(3u, a3);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint sr0 = r0 + P.row_off;
    for (uint m = uint(sg); m < nb; m += 8u) {
        const threadgroup float* ys = sm_out + m * 128u;
        float4 v = float4(ys[lane], ys[lane + 32u], ys[lane + 64u], ys[lane + 96u]);
        pw_fwht128_simd4(v, lane);
        ulong base = ulong(m0 + m) * P.ldo + r0;
        OUT[base + lane +  0u] = v.x * pw_sgn(0x9E3779B9u, sr0 + lane +  0u);
        OUT[base + lane + 32u] = v.y * pw_sgn(0x9E3779B9u, sr0 + lane + 32u);
        OUT[base + lane + 64u] = v.z * pw_sgn(0x9E3779B9u, sr0 + lane + 64u);
        OUT[base + lane + 96u] = v.w * pw_sgn(0x9E3779B9u, sr0 + lane + 96u);
    }
}

// ============================================================================
// True QuIP# E8P (PWV4 table_kind=2, E8 + 1/4) Decode Helpers & Dense Kernel
//   - Codeword (u16): bits[7:0]=abs_idx, bits[14:8]=s[0..6], bit[15]=coset
//   - Derived s[7] = (popcount(s[0..6]) ^ m_par[abs_idx]) & 1 enforces s*a in D8 + 1/2
//   - q = ((-1)^coset) * (s * a - 1/4) -> q.x = ((-1)^coset) * ((s*a).x - 0.25*sum(x))
//   - Bit-exact rows=1 == rows=16 + fused inv-FWHT-128
// ============================================================================
inline float e8p_xsum8_qtr(uint4 x) {
    half2 x0 = as_type<half2>(x.x), x1 = as_type<half2>(x.y), x2 = as_type<half2>(x.z), x3 = as_type<half2>(x.w);
    return ((float(x0.x) + float(x0.y)) + (float(x1.x) + float(x1.y))
          + (float(x2.x) + float(x2.y)) + (float(x3.x) + float(x3.y))) * 0.25f;
}

inline uint e8p_true_smask(uint code, uint m_par) {
    uint s7 = (code >> 8) & 0x7Fu;
    return s7 | (((popcount(s7) ^ m_par) & 1u) << 7);
}

inline float e8p_true_dot8(uint4 lut_u4, uint smask, uint csign, const threadgroup uint4* xv, float xsum_qtr) {
    float d = e8p_dot8(lut_u4, smask, xv) - xsum_qtr;
    return as_type<float>(as_type<uint>(d) ^ csign);
}

// High-bandwidth TFQ4 True QuIP# E8P (table_kind=2) dense decode + fused inv-FWHT-128:
// grid = (N / 128, ceil(rows / 16)), block = 256 (8 simdgroups).
kernel void tfq4_true_e8p_dense(
    const device uint* W        [[buffer(0)]],
    const device half* scales   [[buffer(1)]],
    const device uint4* e8p_lut [[buffer(2)]], // 256 x uint4 = 4 KiB (8 fp16 odd half-integer centroids per entry)
    const device half* X_rot    [[buffer(3)]],
    device float* OUT           [[buffer(4)]],
    constant MetalTfq4Params& P [[buffer(5)]],
    uint2 gid                   [[threadgroup_position_in_grid]],
    ushort tid                  [[thread_index_in_threadgroup]],
    ushort sg                   [[simdgroup_index_in_threadgroup]],
    ushort lane                 [[thread_index_in_simdgroup]]
) {
    uint rb = gid.x, r0 = rb * 128u, m0 = gid.y * 16u;
    if (m0 >= P.rows || r0 >= P.n) return;
    uint nb = min(16u, P.rows - m0);
    uint n_bc = P.k >> 7;
    uint words_per_row = P.k >> 4; // K / 16

    // 4 KiB LUT + 256 B parities + 1 KiB X_rot tile + 256 B quarter-sums + 8 KiB output epilogue = 13.5 KiB!
    threadgroup uint4 sm_lut[256];
    threadgroup uchar sm_par[256];
    threadgroup uint4 sm_x[4 * 16];
    threadgroup float sm_xsum[4 * 16];
    threadgroup float sm_out[16 * 128];

    {
        uint4 u = e8p_lut[tid];
        sm_lut[tid] = u;
        half2 h0 = as_type<half2>(u.x), h1 = as_type<half2>(u.y);
        half2 h2 = as_type<half2>(u.z), h3 = as_type<half2>(u.w);
        uint par = (uint(int(float(h0.x))) + uint(int(float(h0.y)))
                  + uint(int(float(h1.x))) + uint(int(float(h1.y)))
                  + uint(int(float(h2.x))) + uint(int(float(h2.y)))
                  + uint(int(float(h3.x))) + uint(int(float(h3.y)))) & 1u;
        sm_par[tid] = uchar(par);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint sub_r = lane >> 3; // 0..3 (4 rows per simdgroup)
    uint sub_w = lane & 7u; // 0..7 (8 uint32 words = 128 cols)

    if (nb == 1u) {
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        const device half* sc_row = scales + ulong(rb) * n_bc;
        for (uint bc = 0u; bc < n_bc; ++bc) {
            if (tid < 16u) {
                uint4 xv = *reinterpret_cast<const device uint4*>(X_rot + ulong(m0) * P.k + bc * 128u + tid * 8u);
                sm_x[tid] = xv;
                sm_xsum[tid] = e8p_xsum8_qtr(xv);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float fsc = float(sc_row[bc]);
            float xs0 = sm_xsum[sub_w * 2u];
            float xs1 = sm_xsum[sub_w * 2u + 1u];
            TF_UNROLL for (uint pass = 0u; pass < 4u; ++pass) {
                uint local_r = uint(sg) * 16u + pass * 4u + sub_r;
                uint global_r = r0 + local_r;
                uint packed = W[ulong(global_r) * words_per_row + bc * 8u + sub_w];
                uint c0 = packed & 0xFFFFu, c1 = packed >> 16;
                uint idx0 = c0 & 255u, idx1 = c1 & 255u;
                float d = e8p_true_dot8(sm_lut[idx0], e8p_true_smask(c0, uint(sm_par[idx0])), (c0 & 0x8000u) << 16, &sm_x[sub_w * 2u], xs0)
                        + e8p_true_dot8(sm_lut[idx1], e8p_true_smask(c1, uint(sm_par[idx1])), (c1 & 0x8000u) << 16, &sm_x[sub_w * 2u + 1u], xs1);
                acc[pass] = fma(d, fsc, acc[pass]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        TF_UNROLL for (uint pass = 0u; pass < 4u; ++pass) {
            float s = acc[pass];
            s += simd_shuffle_xor(s, 1);
            s += simd_shuffle_xor(s, 2);
            s += simd_shuffle_xor(s, 4);
            if (sub_w == 0u) sm_out[uint(sg) * 16u + pass * 4u + sub_r] = s;
        }
    } else {
        // Register-resident 4-row-batch loop (zero acc[16] stack spill, bit-exact r1==r16!).
        const device half* sc_row = scales + ulong(rb) * n_bc;
        for (uint pass = 0u; pass < 4u; ++pass) {
            uint local_r = uint(sg) * 16u + pass * 4u + sub_r;
            uint global_r = r0 + local_r;
            const device uint* w_row = W + ulong(global_r) * words_per_row;
            for (uint m_base = 0u; m_base < nb; m_base += 4u) {
                float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
                uint rem = min(4u, nb - m_base);
                for (uint bc = 0u; bc < n_bc; ++bc) {
                    uint n_u4 = rem * 16u;
                    if (tid < n_u4) {
                        uint rl = tid >> 4, cl = (tid & 15u) << 3;
                        uint4 xv = *reinterpret_cast<const device uint4*>(X_rot + ulong(m0 + m_base + rl) * P.k + bc * 128u + cl);
                        sm_x[tid] = xv;
                        sm_xsum[tid] = e8p_xsum8_qtr(xv);
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    uint packed = w_row[bc * 8u + sub_w];
                    uint c0 = packed & 0xFFFFu, c1 = packed >> 16;
                    uint idx0 = c0 & 255u, idx1 = c1 & 255u;
                    uint4 l0 = sm_lut[idx0], l1 = sm_lut[idx1];
                    uint sm0 = e8p_true_smask(c0, uint(sm_par[idx0]));
                    uint sm1 = e8p_true_smask(c1, uint(sm_par[idx1]));
                    uint cs0 = (c0 & 0x8000u) << 16, cs1 = (c1 & 0x8000u) << 16;
                    float fsc = float(sc_row[bc]);
                    a0 = fma(e8p_true_dot8(l0, sm0, cs0, &sm_x[0u  + sub_w * 2u], sm_xsum[0u  + sub_w * 2u])
                           + e8p_true_dot8(l1, sm1, cs1, &sm_x[0u  + sub_w * 2u + 1u], sm_xsum[0u  + sub_w * 2u + 1u]), fsc, a0);
                    if (rem > 1u) a1 = fma(e8p_true_dot8(l0, sm0, cs0, &sm_x[16u + sub_w * 2u], sm_xsum[16u + sub_w * 2u])
                                         + e8p_true_dot8(l1, sm1, cs1, &sm_x[16u + sub_w * 2u + 1u], sm_xsum[16u + sub_w * 2u + 1u]), fsc, a1);
                    if (rem > 2u) a2 = fma(e8p_true_dot8(l0, sm0, cs0, &sm_x[32u + sub_w * 2u], sm_xsum[32u + sub_w * 2u])
                                         + e8p_true_dot8(l1, sm1, cs1, &sm_x[32u + sub_w * 2u + 1u], sm_xsum[32u + sub_w * 2u + 1u]), fsc, a2);
                    if (rem > 3u) a3 = fma(e8p_true_dot8(l0, sm0, cs0, &sm_x[48u + sub_w * 2u], sm_xsum[48u + sub_w * 2u])
                                         + e8p_true_dot8(l1, sm1, cs1, &sm_x[48u + sub_w * 2u + 1u], sm_xsum[48u + sub_w * 2u + 1u]), fsc, a3);
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                auto red_wr = [&](uint m_off, float s) {
                    s += simd_shuffle_xor(s, 1); s += simd_shuffle_xor(s, 2); s += simd_shuffle_xor(s, 4);
                    if (sub_w == 0u) sm_out[(m_base + m_off) * 128u + local_r] = s;
                };
                red_wr(0u, a0);
                if (rem > 1u) red_wr(1u, a1);
                if (rem > 2u) red_wr(2u, a2);
                if (rem > 3u) red_wr(3u, a3);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint sr0 = r0 + P.row_off;
    for (uint m = uint(sg); m < nb; m += 8u) {
        const threadgroup float* ys = sm_out + m * 128u;
        float4 v = float4(ys[lane], ys[lane + 32u], ys[lane + 64u], ys[lane + 96u]);
        pw_fwht128_simd4(v, lane);
        ulong base = ulong(m0 + m) * P.ldo + r0;
        OUT[base + lane +  0u] = v.x * pw_sgn(0x9E3779B9u, sr0 + lane +  0u);
        OUT[base + lane + 32u] = v.y * pw_sgn(0x9E3779B9u, sr0 + lane + 32u);
        OUT[base + lane + 64u] = v.z * pw_sgn(0x9E3779B9u, sr0 + lane + 64u);
        OUT[base + lane + 96u] = v.w * pw_sgn(0x9E3779B9u, sr0 + lane + 96u);
    }
}

// ============================================================================
// True QuIP# E8P-1024 (PWV4 table_kind=3, E8 + 1/4, 2.251 bpw) Dense Kernel
//   - 9 uint32 words per 128-col row-block:
//       w[0..7]: low16(v_{2*sub_w}) | (low16(v_{2*sub_w+1}) << 16)
//       w[8]:    hi2 packed across all 16 vectors: bits[2*j+1:2*j] = (abs_idx_j >> 8) & 3
//   - Threadgroup memory: 16 KiB sm_lut[1024] + 1 KiB sm_par[1024] + 1 KiB sm_x
//     + 256 B sm_xsum + 8 KiB sm_out = 26.25 KiB <= 32 KiB Apple Silicon TG limit!
//   - Bit-exact rows=1 == rows=16 (and arbitrary rows >= 1) + fused inv-FWHT-128.
// ============================================================================
kernel void tfq4_true_e8p_1024_dense(
    const device uint* W        [[buffer(0)]],
    const device half* scales   [[buffer(1)]],
    const device uint4* e8p_lut [[buffer(2)]], // 1024 x uint4 = 16 KiB
    const device half* X_rot    [[buffer(3)]],
    device float* OUT           [[buffer(4)]],
    constant MetalTfq4Params& P [[buffer(5)]],
    uint2 gid                   [[threadgroup_position_in_grid]],
    ushort tid                  [[thread_index_in_threadgroup]],
    ushort sg                   [[simdgroup_index_in_threadgroup]],
    ushort lane                 [[thread_index_in_simdgroup]]
) {
    uint rb = gid.x, r0 = rb * 128u, m0 = gid.y * 16u;
    if (m0 >= P.rows || r0 >= P.n) return;
    uint nb = min(16u, P.rows - m0);
    uint n_bc = P.k >> 7;
    uint words_per_row = n_bc * 9u; // 9 uint32 words per 128 cols (2.25 bpw)

    threadgroup uint4 sm_lut[1024];
    threadgroup uchar sm_par[1024];
    threadgroup uint4 sm_x[4 * 16];
    threadgroup float sm_xsum[4 * 16];
    threadgroup float sm_out[16 * 128];

    for (uint i = uint(tid); i < 1024u; i += 256u) {
        uint4 u = e8p_lut[i];
        sm_lut[i] = u;
        half2 h0 = as_type<half2>(u.x), h1 = as_type<half2>(u.y);
        half2 h2 = as_type<half2>(u.z), h3 = as_type<half2>(u.w);
        uint par = (uint(int(float(h0.x))) + uint(int(float(h0.y)))
                  + uint(int(float(h1.x))) + uint(int(float(h1.y)))
                  + uint(int(float(h2.x))) + uint(int(float(h2.y)))
                  + uint(int(float(h3.x))) + uint(int(float(h3.y)))) & 1u;
        sm_par[i] = uchar(par);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint sub_r = lane >> 3; // 0..3 (4 rows per simdgroup)
    uint sub_w = lane & 7u; // 0..7 (8 lanes = 128 cols)
    uint hi_shift = sub_w * 4u;

    if (nb == 1u) {
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        const device half* sc_row = scales + ulong(rb) * n_bc;
        for (uint bc = 0u; bc < n_bc; ++bc) {
            if (tid < 16u) {
                uint4 xv = *reinterpret_cast<const device uint4*>(X_rot + ulong(m0) * P.k + bc * 128u + tid * 8u);
                sm_x[tid] = xv;
                sm_xsum[tid] = e8p_xsum8_qtr(xv);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float fsc = float(sc_row[bc]);
            float xs0 = sm_xsum[sub_w * 2u];
            float xs1 = sm_xsum[sub_w * 2u + 1u];
            TF_UNROLL for (uint pass = 0u; pass < 4u; ++pass) {
                uint local_r = uint(sg) * 16u + pass * 4u + sub_r;
                uint global_r = r0 + local_r;
                ulong blk_off = ulong(global_r) * words_per_row + ulong(bc) * 9ul;
                uint packed = W[blk_off + sub_w];
                uint hi_word = W[blk_off + 8ul];
                uint c0 = packed & 0xFFFFu, c1 = packed >> 16;
                uint idx0 = (c0 & 255u) | (((hi_word >> hi_shift) & 3u) << 8);
                uint idx1 = (c1 & 255u) | (((hi_word >> (hi_shift + 2u)) & 3u) << 8);
                float d = e8p_true_dot8(sm_lut[idx0], e8p_true_smask(c0, uint(sm_par[idx0])), (c0 & 0x8000u) << 16, &sm_x[sub_w * 2u], xs0)
                        + e8p_true_dot8(sm_lut[idx1], e8p_true_smask(c1, uint(sm_par[idx1])), (c1 & 0x8000u) << 16, &sm_x[sub_w * 2u + 1u], xs1);
                acc[pass] = fma(d, fsc, acc[pass]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        TF_UNROLL for (uint pass = 0u; pass < 4u; ++pass) {
            float s = acc[pass];
            s += simd_shuffle_xor(s, 1);
            s += simd_shuffle_xor(s, 2);
            s += simd_shuffle_xor(s, 4);
            if (sub_w == 0u) sm_out[uint(sg) * 16u + pass * 4u + sub_r] = s;
        }
    } else {
        const device half* sc_row = scales + ulong(rb) * n_bc;
        for (uint pass = 0u; pass < 4u; ++pass) {
            uint local_r = uint(sg) * 16u + pass * 4u + sub_r;
            uint global_r = r0 + local_r;
            const device uint* w_row = W + ulong(global_r) * words_per_row;
            for (uint m_base = 0u; m_base < nb; m_base += 4u) {
                float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
                uint rem = min(4u, nb - m_base);
                for (uint bc = 0u; bc < n_bc; ++bc) {
                    uint n_u4 = rem * 16u;
                    if (tid < n_u4) {
                        uint rl = tid >> 4, cl = (tid & 15u) << 3;
                        uint4 xv = *reinterpret_cast<const device uint4*>(X_rot + ulong(m0 + m_base + rl) * P.k + bc * 128u + cl);
                        sm_x[tid] = xv;
                        sm_xsum[tid] = e8p_xsum8_qtr(xv);
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    uint packed = w_row[bc * 9u + sub_w];
                    uint hi_word = w_row[bc * 9u + 8u];
                    uint c0 = packed & 0xFFFFu, c1 = packed >> 16;
                    uint idx0 = (c0 & 255u) | (((hi_word >> hi_shift) & 3u) << 8);
                    uint idx1 = (c1 & 255u) | (((hi_word >> (hi_shift + 2u)) & 3u) << 8);
                    uint4 l0 = sm_lut[idx0], l1 = sm_lut[idx1];
                    uint sm0 = e8p_true_smask(c0, uint(sm_par[idx0]));
                    uint sm1 = e8p_true_smask(c1, uint(sm_par[idx1]));
                    uint cs0 = (c0 & 0x8000u) << 16, cs1 = (c1 & 0x8000u) << 16;
                    float fsc = float(sc_row[bc]);
                    a0 = fma(e8p_true_dot8(l0, sm0, cs0, &sm_x[0u  + sub_w * 2u], sm_xsum[0u  + sub_w * 2u])
                           + e8p_true_dot8(l1, sm1, cs1, &sm_x[0u  + sub_w * 2u + 1u], sm_xsum[0u  + sub_w * 2u + 1u]), fsc, a0);
                    if (rem > 1u) a1 = fma(e8p_true_dot8(l0, sm0, cs0, &sm_x[16u + sub_w * 2u], sm_xsum[16u + sub_w * 2u])
                                         + e8p_true_dot8(l1, sm1, cs1, &sm_x[16u + sub_w * 2u + 1u], sm_xsum[16u + sub_w * 2u + 1u]), fsc, a1);
                    if (rem > 2u) a2 = fma(e8p_true_dot8(l0, sm0, cs0, &sm_x[32u + sub_w * 2u], sm_xsum[32u + sub_w * 2u])
                                         + e8p_true_dot8(l1, sm1, cs1, &sm_x[32u + sub_w * 2u + 1u], sm_xsum[32u + sub_w * 2u + 1u]), fsc, a2);
                    if (rem > 3u) a3 = fma(e8p_true_dot8(l0, sm0, cs0, &sm_x[48u + sub_w * 2u], sm_xsum[48u + sub_w * 2u])
                                         + e8p_true_dot8(l1, sm1, cs1, &sm_x[48u + sub_w * 2u + 1u], sm_xsum[48u + sub_w * 2u + 1u]), fsc, a3);
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                auto red_wr = [&](uint m_off, float s) {
                    s += simd_shuffle_xor(s, 1); s += simd_shuffle_xor(s, 2); s += simd_shuffle_xor(s, 4);
                    if (sub_w == 0u) sm_out[(m_base + m_off) * 128u + local_r] = s;
                };
                red_wr(0u, a0);
                if (rem > 1u) red_wr(1u, a1);
                if (rem > 2u) red_wr(2u, a2);
                if (rem > 3u) red_wr(3u, a3);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint sr0 = r0 + P.row_off;
    for (uint m = uint(sg); m < nb; m += 8u) {
        const threadgroup float* ys = sm_out + m * 128u;
        float4 v = float4(ys[lane], ys[lane + 32u], ys[lane + 64u], ys[lane + 96u]);
        pw_fwht128_simd4(v, lane);
        ulong base = ulong(m0 + m) * P.ldo + r0;
        OUT[base + lane +  0u] = v.x * pw_sgn(0x9E3779B9u, sr0 + lane +  0u);
        OUT[base + lane + 32u] = v.y * pw_sgn(0x9E3779B9u, sr0 + lane + 32u);
        OUT[base + lane + 64u] = v.z * pw_sgn(0x9E3779B9u, sr0 + lane + 64u);
        OUT[base + lane + 96u] = v.w * pw_sgn(0x9E3779B9u, sr0 + lane + 96u);
    }
}
