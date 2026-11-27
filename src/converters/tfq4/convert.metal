#include <metal_stdlib>
using namespace metal;

#define TF_UNROLL _Pragma("clang loop unroll(full)")

struct Tfq4MatParams {
    uint R;
    uint C;
    uint r_off;
    uint pad0;
};

inline float sgn_bit(uint seed, uint i) {
    uint h = (seed ^ i) * 2654435761u;
    return ((h >> 31) & 1u) ? -1.0f : 1.0f;
}

inline void fwht128_simd4(thread float4& v, ushort lane) {
    TF_UNROLL
    for (ushort d = 1; d <= 16; d <<= 1) {
        float4 o = simd_shuffle_xor(v, d);
        v = ((lane & d) == 0u) ? (v + o) : (o - v);
    }
    float a0 = v.x + v.y, b0 = v.x - v.y;
    float a1 = v.z + v.w, b1 = v.z - v.w;
    v = float4(a0 + a1, b0 + b1, a0 - a1, b0 - b1) * 0.08838834764831845f;
}

kernel void f32_row_fwht(
    const device float* W_in  [[buffer(0)]],
    device float* W_tmp       [[buffer(1)]],
    constant Tfq4MatParams& P [[buffer(2)]],
    uint2 gid                 [[threadgroup_position_in_grid]],
    ushort lane               [[thread_index_in_simdgroup]]
) {
    uint r = gid.y, bc = gid.x, c0 = bc * 128u;
    const device float* row = W_in + ulong(r) * P.C + c0;
    float4 v;
    v.x = row[lane +  0u] * sgn_bit(0x12345678u, c0 + lane +  0u);
    v.y = row[lane + 32u] * sgn_bit(0x12345678u, c0 + lane + 32u);
    v.z = row[lane + 64u] * sgn_bit(0x12345678u, c0 + lane + 64u);
    v.w = row[lane + 96u] * sgn_bit(0x12345678u, c0 + lane + 96u);
    fwht128_simd4(v, lane);
    device float* dst = W_tmp + ulong(r) * P.C + c0;
    dst[lane +  0u] = v.x; dst[lane + 32u] = v.y; dst[lane + 64u] = v.z; dst[lane + 96u] = v.w;
}

inline float exact_fdiv(float a, float b) {
    float y = 1.0f / b;
    float e = fma(-b, y, 1.0f);
    y = fma(y, e, y);
    float q = a * y;
    float r = fma(-b, q, a);
    q = fma(r, y, q);
    float r2 = fma(-b, q, a);
    return fma(r2, y, q);
}

static inline float det_simd_sum(float v) {
    v += simd_shuffle_xor(v, 16);
    v += simd_shuffle_xor(v, 8);
    v += simd_shuffle_xor(v, 4);
    v += simd_shuffle_xor(v, 2);
    v += simd_shuffle_xor(v, 1);
    return v;
}

kernel void col_fwht_and_ss(
    const device float* W_tmp   [[buffer(0)]],
    device float* W_rot         [[buffer(1)]],
    device float* col_ss        [[buffer(2)]],
    constant Tfq4MatParams& P   [[buffer(3)]],
    uint2 gid                   [[threadgroup_position_in_grid]],
    ushort lane                 [[thread_index_in_simdgroup]]
) {
    uint c = gid.x, br = gid.y, r0 = br * 128u;
    uint gr0 = P.r_off + r0;
    float4 v;
    v.x = W_tmp[ulong(r0 + lane +  0u) * P.C + c] * sgn_bit(0x9E3779B9u, gr0 + lane +  0u);
    v.y = W_tmp[ulong(r0 + lane + 32u) * P.C + c] * sgn_bit(0x9E3779B9u, gr0 + lane + 32u);
    v.z = W_tmp[ulong(r0 + lane + 64u) * P.C + c] * sgn_bit(0x9E3779B9u, gr0 + lane + 64u);
    v.w = W_tmp[ulong(r0 + lane + 96u) * P.C + c] * sgn_bit(0x9E3779B9u, gr0 + lane + 96u);
    fwht128_simd4(v, lane);
    W_rot[ulong(r0 + lane +  0u) * P.C + c] = v.x;
    W_rot[ulong(r0 + lane + 32u) * P.C + c] = v.y;
    W_rot[ulong(r0 + lane + 64u) * P.C + c] = v.z;
    W_rot[ulong(r0 + lane + 96u) * P.C + c] = v.w;
    float ss = det_simd_sum((v.x * v.x + v.y * v.y) + (v.z * v.z + v.w * v.w));
    uint bi = br * (P.C >> 7) + (c >> 7);
    if (lane == 0) col_ss[ulong(bi) * 128ul + (c & 127u)] = ss;
}

struct Tfq4ScaleParams {
    uint n_blocks;
    float scale_mult;
};

kernel void finalize_scales(
    const device float* col_ss       [[buffer(0)]],
    device half* scales_f16          [[buffer(1)]],
    constant Tfq4ScaleParams& SP     [[buffer(2)]],
    uint bi                          [[threadgroup_position_in_grid]],
    ushort lane                      [[thread_index_in_simdgroup]]
) {
    if (bi >= SP.n_blocks) return;
    const device float* cs = col_ss + ulong(bi) * 128ul;
    float s = det_simd_sum((cs[lane + 0u] + cs[lane + 32u]) + (cs[lane + 64u] + cs[lane + 96u]));
    if (lane == 0) scales_f16[bi] = half(sqrt(s * (1.0f / 16384.0f) + 1e-12f) * SP.scale_mult);
}

// Exhaustive 256-entry positive-orthant 8D E8P encoder: 1 thread per uint32_t (16 weights = 2 x 8D).
// Loads 4 KiB E8P LUT into threadgroup float[256][8] once per 256-thread TG.
kernel void tfq4_e8p_encode(
    const device float* W_rot   [[buffer(0)]],
    const device half* scales   [[buffer(1)]],
    const device half* e8p_lut  [[buffer(2)]], // 256 * 8 halfs
    device uint* out_words      [[buffer(3)]],
    device float* metrics       [[buffer(4)]], // [tg * 2 + 0]=sig, [tg * 2 + 1]=err
    constant Tfq4MatParams& P   [[buffer(5)]],
    uint gtid                   [[thread_position_in_grid]],
    uint tgid                   [[threadgroup_position_in_grid]],
    ushort tid                  [[thread_index_in_threadgroup]],
    ushort lane                 [[thread_index_in_simdgroup]],
    ushort sg                   [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float sm_lut[256][8];
    for (uint d = 0u; d < 8u; ++d) sm_lut[tid][d] = float(e8p_lut[uint(tid) * 8u + d]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint wpr = P.C >> 4; // C / 16
    ulong total_words = ulong(P.R) * wpr;
    float my_sig = 0.0f, my_err = 0.0f;

    if (ulong(gtid) < total_words) {
        uint r = gtid / wpr;
        uint wc = gtid - r * wpr;
        uint n_bc = P.C >> 7;
        float sc = float(scales[(r >> 7) * n_bc + (wc >> 3)]);
        float inv_sc = (sc > 0.0f) ? exact_fdiv(1.0f, sc) : 0.0f;
        const device float* src = W_rot + ulong(r) * P.C + ulong(wc) * 16ul;

        uint codes[2];
        TF_UNROLL for (uint h = 0u; h < 2u; ++h) {
            float v[8], a[8];
            uint sm = 0u;
            TF_UNROLL for (uint d = 0u; d < 8u; ++d) {
                float x = src[h * 8u + d];
                v[d] = x;
                if (x < 0.0f) sm |= (1u << d);
                a[d] = fabs(x) * inv_sc;
                my_sig += x * x;
            }
            uint best_k = 0u;
            float best_d = 1e30f;
            for (uint k = 0u; k < 256u; ++k) {
                float d0 = a[0] - sm_lut[k][0], d1 = a[1] - sm_lut[k][1];
                float d2 = a[2] - sm_lut[k][2], d3 = a[3] - sm_lut[k][3];
                float d4 = a[4] - sm_lut[k][4], d5 = a[5] - sm_lut[k][5];
                float d6 = a[6] - sm_lut[k][6], d7 = a[7] - sm_lut[k][7];
                float dist = ((d0*d0 + d1*d1) + (d2*d2 + d3*d3)) + ((d4*d4 + d5*d5) + (d6*d6 + d7*d7));
                if (dist < best_d) { best_d = dist; best_k = k; }
            }
            codes[h] = (sm << 8) | best_k;
            TF_UNROLL for (uint d = 0u; d < 8u; ++d) {
                float e = fabs(v[d]) - sm_lut[best_k][d] * sc;
                my_err += e * e;
            }
        }
        out_words[gtid] = codes[0] | (codes[1] << 16);
    }

    float sg_sig = det_simd_sum(my_sig), sg_err = det_simd_sum(my_err);
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

// Exact nearest-point encoder for True QuIP# E8P (PWV4 table_kind=2, E8 + 1/4):
//   - 256 half-integer base vectors in |D8 + 1/2| x 128 parity-constrained signs x 2 cosets = 65,536 points in E8 + 1/4.
//   - Codeword (u16): bits[7:0]=abs_idx, bits[14:8]=s[0..6], bit[15]=coset; s[7] derived from parity.
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
        float inv_sc = (sc > 0.0f) ? exact_fdiv(1.0f, sc) : 0.0f;
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

    float sg_sig = det_simd_sum(my_sig), sg_err = det_simd_sum(my_err);
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
