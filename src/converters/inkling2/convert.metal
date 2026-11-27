// ============================================================================
// Inkling-2 (NVFP4 + MXFP8 3D-packed MoE + BF16 non-MoE) Metal TFQ3/TFQ4 Converter
// Compiled with -fno-fast-math for 100% bit-identical output with CUDA.
// ============================================================================
#include "../metal_encode_common.metal"
#include "../tfq4/tfq4_encode_common.metal"

struct Ink2Params {
    uint  R;        // output rows for this projection (inter for gate/up, hidden for down)
    uint  C;        // input cols (hidden for gate/up, inter for down)
    uint  row0;     // 0 for gate or down, 1 for up
    uint  row_step; // 2 for interleaved w13 (gate/up), 1 for w2 (down)
    float outer;    // NVFP4 outer scale = bf16(amax[e]) / bf16(den[0])
    uint  pad0;
};

inline float ink2_e2m1(uint nib) {
    constexpr float LUT[16] = {
         0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };
    return LUT[nib & 15u];
}

inline float ink2_e8m0(uint sb) {
    if (sb == 0u) return 5.8774717541114375e-39f; // 0x1p-127f
    return as_type<float>(sb << 23);
}

// Dequantize one NVFP4 expert projection (group=16 E4M3 scales + outer) with optional
// row deinterleaving (row_step=2 for w13 gate/up, row_step=1 for w2 down) + row FWHT-128.
kernel void nvfp4_dequant_row_fwht(
    const device uchar* codes   [[buffer(0)]],
    const device uchar* scales  [[buffer(1)]],
    constant float* fp8_lut     [[buffer(2)]],
    device float* W_tmp         [[buffer(3)]],
    constant Ink2Params& P      [[buffer(4)]],
    uint2 gid                   [[threadgroup_position_in_grid]],
    ushort lane                 [[thread_index_in_simdgroup]]
) {
    uint r = gid.y, bc = gid.x, c0 = bc * 128u;
    uint sr = P.row0 + r * P.row_step;
    const device uchar* crow = codes  + ulong(sr) * (P.C >> 1);
    const device uchar* srow = scales + ulong(sr) * (P.C >> 4);
    float outer = P.outer;

    uint c_0 = c0 + lane +  0u;
    uint c_1 = c0 + lane + 32u;
    uint c_2 = c0 + lane + 64u;
    uint c_3 = c0 + lane + 96u;

    uint b0 = crow[c_0 >> 1], n0 = (c_0 & 1u) ? (b0 >> 4) : (b0 & 15u);
    uint b1 = crow[c_1 >> 1], n1 = (c_1 & 1u) ? (b1 >> 4) : (b1 & 15u);
    uint b2 = crow[c_2 >> 1], n2 = (c_2 & 1u) ? (b2 >> 4) : (b2 & 15u);
    uint b3 = crow[c_3 >> 1], n3 = (c_3 & 1u) ? (b3 >> 4) : (b3 & 15u);

    float w0 = (ink2_e2m1(n0) * fp8_lut[srow[c_0 >> 4]]) * outer;
    float w1 = (ink2_e2m1(n1) * fp8_lut[srow[c_1 >> 4]]) * outer;
    float w2 = (ink2_e2m1(n2) * fp8_lut[srow[c_2 >> 4]]) * outer;
    float w3 = (ink2_e2m1(n3) * fp8_lut[srow[c_3 >> 4]]) * outer;

    float4 v;
    v.x = w0 * sgn_bit(0x12345678u, c_0);
    v.y = w1 * sgn_bit(0x12345678u, c_1);
    v.z = w2 * sgn_bit(0x12345678u, c_2);
    v.w = w3 * sgn_bit(0x12345678u, c_3);
    fwht128_simd4(v, lane);
    device float* dst = W_tmp + ulong(r) * P.C + c0;
    dst[lane +  0u] = v.x;
    dst[lane + 32u] = v.y;
    dst[lane + 64u] = v.z;
    dst[lane + 96u] = v.w;
}

// Dequantize one MXFP8 expert projection (F8_E4M3 codes + group=32 UE8M0 scales) with optional
// row deinterleaving (row_step=2 for w13 gate/up, row_step=1 for w2 down) + row FWHT-128.
kernel void mxfp8_dequant_row_fwht(
    const device uchar* codes   [[buffer(0)]],
    const device uchar* scales  [[buffer(1)]],
    constant float* fp8_lut     [[buffer(2)]],
    device float* W_tmp         [[buffer(3)]],
    constant Ink2Params& P      [[buffer(4)]],
    uint2 gid                   [[threadgroup_position_in_grid]],
    ushort lane                 [[thread_index_in_simdgroup]]
) {
    uint r = gid.y, bc = gid.x, c0 = bc * 128u;
    uint sr = P.row0 + r * P.row_step;
    const device uchar* crow = codes  + ulong(sr) * P.C;
    const device uchar* srow = scales + ulong(sr) * (P.C >> 5);

    uint c_0 = c0 + lane +  0u;
    uint c_1 = c0 + lane + 32u;
    uint c_2 = c0 + lane + 64u;
    uint c_3 = c0 + lane + 96u;

    float w0 = fp8_lut[crow[c_0]] * ink2_e8m0(srow[c_0 >> 5]);
    float w1 = fp8_lut[crow[c_1]] * ink2_e8m0(srow[c_1 >> 5]);
    float w2 = fp8_lut[crow[c_2]] * ink2_e8m0(srow[c_2 >> 5]);
    float w3 = fp8_lut[crow[c_3]] * ink2_e8m0(srow[c_3 >> 5]);

    float4 v;
    v.x = w0 * sgn_bit(0x12345678u, c_0);
    v.y = w1 * sgn_bit(0x12345678u, c_1);
    v.z = w2 * sgn_bit(0x12345678u, c_2);
    v.w = w3 * sgn_bit(0x12345678u, c_3);
    fwht128_simd4(v, lane);
    device float* dst = W_tmp + ulong(r) * P.C + c0;
    dst[lane +  0u] = v.x;
    dst[lane + 32u] = v.y;
    dst[lane + 64u] = v.z;
    dst[lane + 96u] = v.w;
}
