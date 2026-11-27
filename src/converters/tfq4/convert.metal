#include "tfq4_encode_common.metal"

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
    float ss = tfq4_det_simd_sum((v.x * v.x + v.y * v.y) + (v.z * v.z + v.w * v.w));
    uint bi = br * (P.C >> 7) + (c >> 7);
    if (lane == 0) col_ss[ulong(bi) * 128ul + (c & 127u)] = ss;
}

// Exhaustive 256-entry positive-orthant 8D E8P encoder (PWV4 table_kind=0/1, 2.001 bpw).
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
        float inv_sc = (sc > 0.0f) ? tfq4_exact_fdiv(1.0f, sc) : 0.0f;
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
