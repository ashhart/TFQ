"""
Zero-build Apple Silicon Metal / PyTorch MPS fused kernels for TFQ3 (3/4/5-bit) and
TFQ4-E8P (2.001 bpw table_kind 0/1/2 and 2.251 bpw E8P-1024 table_kind 3).

Uses PyTorch's built-in `torch.mps.compile_shader` runtime compiler directly against
`src/kernels/metal/tfq3_metal.metal` and `src/kernels/metal/tfq4_metal.metal` (with an
inline zero-build Metal fallback for `tfq4_true_e8p_1024_dense` when needed) — zero
Xcode, zero C++ extension build step, zero BF16 weight expansion in unified memory.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))

# Inline zero-build Metal fallback for TFQ4 2.251 bpw True QuIP# E8P-1024 (PWV4 table_kind=3).
# Used only if the bundled `tfq4_metal.metal` does not yet define `tfq4_true_e8p_1024_dense`.
_TFQ4_E8P1024_METAL_SRC = r"""
#include <metal_stdlib>
using namespace metal;

#define TF_UNROLL _Pragma("clang loop unroll(full)")

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

// TFQ4 2.251 bpw True QuIP# E8P-1024 (PWV4 table_kind=3, 9 uint32 words per 128-col block):
// grid = (N / 128, ceil(rows / 16)), block = 256 (8 simdgroups).
// Threadgroup memory: 16 KiB LUT + 1 KiB parity + 1 KiB X + 256 B xsum + 8 KiB out = 26.25 KiB.
kernel void tfq4_true_e8p_1024_dense(
    const device uint* W        [[buffer(0)]],
    const device half* scales   [[buffer(1)]],
    const device uint4* e8p_lut [[buffer(2)]], // 1024 x uint4 = 16 KiB fp16 odd half-integer centroids
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
    uint words_per_row = n_bc * 9u;

    threadgroup uint4 sm_lut[1024];
    threadgroup uchar sm_par[1024];
    threadgroup uint4 sm_x[4 * 16];
    threadgroup float sm_xsum[4 * 16];
    threadgroup float sm_out[16 * 128];

    TF_UNROLL for (uint k = 0u; k < 4u; ++k) {
        uint idx = k * 256u + uint(tid);
        uint4 u = e8p_lut[idx];
        sm_lut[idx] = u;
        half2 h0 = as_type<half2>(u.x), h1 = as_type<half2>(u.y);
        half2 h2 = as_type<half2>(u.z), h3 = as_type<half2>(u.w);
        uint par = (uint(int(float(h0.x))) + uint(int(float(h0.y)))
                  + uint(int(float(h1.x))) + uint(int(float(h1.y)))
                  + uint(int(float(h2.x))) + uint(int(float(h2.y)))
                  + uint(int(float(h3.x))) + uint(int(float(h3.y)))) & 1u;
        sm_par[idx] = uchar(par);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint sub_r = lane >> 3;
    uint sub_w = lane & 7u;

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
                ulong blk_off = ulong(global_r) * words_per_row + ulong(bc) * 9u;
                uint packed = W[blk_off + sub_w];
                uint hi_w = W[blk_off + 8u];
                uint c0 = packed & 0xFFFFu, c1 = packed >> 16;
                uint idx0 = (c0 & 255u) | (((hi_w >> (sub_w * 4u)) & 3u) << 8);
                uint idx1 = (c1 & 255u) | (((hi_w >> (sub_w * 4u + 2u)) & 3u) << 8);
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
                    uint packed = w_row[ulong(bc) * 9u + sub_w];
                    uint hi_w = w_row[ulong(bc) * 9u + 8u];
                    uint c0 = packed & 0xFFFFu, c1 = packed >> 16;
                    uint idx0 = (c0 & 255u) | (((hi_w >> (sub_w * 4u)) & 3u) << 8);
                    uint idx1 = (c1 & 255u) | (((hi_w >> (sub_w * 4u + 2u)) & 3u) << 8);
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
"""


def _find_metal_source(filename: str) -> Optional[str]:
    candidates = [
        os.path.join(_REPO_ROOT, "src", "kernels", "metal", filename),
        os.path.join(os.path.dirname(_HERE), "metal", filename),
        os.path.join(_HERE, "_metal", filename),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def mps_kernels_available() -> bool:
    return (
        torch.backends.mps.is_available()
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "compile_shader")
        and _find_metal_source("tfq3_metal.metal") is not None
    )


@lru_cache(maxsize=1)
def _get_tfq3_lib():
    if not mps_kernels_available():
        return None
    path = _find_metal_source("tfq3_metal.metal")
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    return torch.mps.compile_shader(src)


@lru_cache(maxsize=1)
def _get_tfq4_lib():
    if not (torch.backends.mps.is_available() and hasattr(torch, "mps") and hasattr(torch.mps, "compile_shader")):
        return None
    path = _find_metal_source("tfq4_metal.metal")
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    return torch.mps.compile_shader(src)


@lru_cache(maxsize=1)
def _get_tfq4_1024_lib():
    if not (torch.backends.mps.is_available() and hasattr(torch, "mps") and hasattr(torch.mps, "compile_shader")):
        return None
    lib = _get_tfq4_lib()
    if lib is not None and hasattr(lib, "tfq4_true_e8p_1024_dense"):
        return lib
    return torch.mps.compile_shader(_TFQ4_E8P1024_METAL_SRC)


def _as_zero_offset_mps(t: torch.Tensor) -> torch.Tensor:
    if not t.is_contiguous() or t.storage_offset() != 0:
        return t.contiguous().clone()
    return t


def tfq3_linear_mps(
    x: torch.Tensor,
    words_i32: torch.Tensor,
    scales_f16: torch.Tensor,
    tables_u8: torch.Tensor,
    bits: int,
    pad_r: int,
    pad_c: int,
    orig_r: int,
    row_off: int = 0,
    k_sign_off: int = 0,
    bc0_buf: int = 0,
    n_bc_buf: int = 0,
) -> torch.Tensor:
    """Fused TFQ3 3/4/5-bit 2D-FWHT + simdgroup_matrix linear on Apple Silicon MPS (Col-TP & Row-TP ready)."""
    lib = _get_tfq3_lib()
    if lib is None:
        raise RuntimeError("TFQ3 MPS Metal shaders are unavailable on this PyTorch build")
    if bits not in (3, 4, 5):
        raise ValueError(f"TFQ3 bits must be 3, 4, or 5, got {bits}")
    if (pad_r % 128) != 0 or (pad_c % 128) != 0:
        raise ValueError(f"pad_r ({pad_r}) and pad_c ({pad_c}) must be multiples of 128")
    if (row_off % 128) != 0 or row_off < 0 or (k_sign_off % 128) != 0 or k_sign_off < 0:
        raise ValueError("row_off and k_sign_off must be non-negative multiples of 128")

    orig_shape = list(x.shape)
    orig_c = orig_shape[-1]
    rows = x.numel() // orig_c
    if rows == 0:
        orig_shape[-1] = orig_r
        return torch.empty(orig_shape, device=x.device, dtype=x.dtype)

    n_bc_local = pad_c // 128
    n_bc_stored = n_bc_buf if n_bc_buf > 0 else n_bc_local
    if bc0_buf < 0 or bc0_buf + n_bc_local > n_bc_stored:
        raise ValueError(f"bc0_buf ({bc0_buf}) + pad_c/128 ({n_bc_local}) exceeds n_bc_buf ({n_bc_stored})")

    x2d = x.reshape(rows, orig_c).to(dtype=torch.float16)
    if orig_c < pad_c:
        x2d = F.pad(x2d, (0, pad_c - orig_c))
    x2d = _as_zero_offset_mps(x2d)

    w_buf = _as_zero_offset_mps(words_i32)
    s_buf = _as_zero_offset_mps(scales_f16)
    t_buf = _as_zero_offset_mps(tables_u8)

    x_rot = torch.empty((rows, pad_c), dtype=torch.float16, device=x.device)
    y_pad = torch.empty((rows, pad_r), dtype=torch.float32, device=x.device)

    use_decode = rows <= 32
    ks = min(8, n_bc_local) if use_decode else 1
    params = torch.tensor(
        [
            rows,
            pad_r,
            pad_c,
            pad_c,
            pad_r,
            row_off,
            k_sign_off,
            0,  # strip_off (words_i32 is already row-sliced to local partition)
            0,  # sc_row_off (scales_f16 is already row-sliced to local partition)
            bc0_buf,
            bc0_buf + n_bc_local,
            n_bc_stored,
            ks,
        ],
        dtype=torch.int32,
        device=x.device,
    )

    lib.tfq3_fwht_x(
        x2d,
        x_rot,
        params,
        threads=(n_bc_local * 32, rows, 1),
        group_size=(32, 1, 1),
    )

    if use_decode:
        yr = torch.empty((ks, rows, pad_r), dtype=torch.float32, device=x.device)
        sk8_fn = getattr(lib, f"tfq3_sk8_tp_{bits}b")
        sk8_fn(
            w_buf,
            s_buf,
            t_buf,
            x_rot,
            yr,
            params,
            threads=((pad_r // 64) * 128, ks, (rows + 15) // 16),
            group_size=(128, 1, 1),
        )
        lib.tfq3_fwht_y(
            yr,
            y_pad,
            params,
            threads=((pad_r // 128) * 32, rows, 1),
            group_size=(32, 1, 1),
        )
    else:
        pf_fn = getattr(lib, f"tfq3_prefill_tp_{bits}b")
        pf_fn(
            w_buf,
            s_buf,
            t_buf,
            x_rot,
            y_pad,
            params,
            threads=((pad_r // 128) * 256, (rows + 31) // 32, 1),
            group_size=(256, 1, 1),
        )

    y_out = y_pad[:, :orig_r] if orig_r < pad_r else y_pad
    orig_shape[-1] = orig_r
    return y_out.reshape(orig_shape).to(dtype=x.dtype)


def tfq4_linear_mps(
    x: torch.Tensor,
    words_i32: torch.Tensor,
    scales_f16: torch.Tensor,
    e8p_lut_f16: torch.Tensor,
    table_kind: int,
    pad_r: int,
    pad_c: int,
    orig_r: int,
    row_off: int = 0,
    k_sign_off: int = 0,
) -> torch.Tensor:
    """
    Fused TFQ4 2D-FWHT + E8P linear on Apple Silicon MPS:
      - table_kind 0/1: 2.001 bpw trained 8-sign E8P LUT (256 entries, 8 words / 128 cols)
      - table_kind 2:   2.001 bpw True QuIP# E8P-256 (256 entries, 8 words / 128 cols)
      - table_kind 3:   2.251 bpw True QuIP# E8P-1024 (1024 entries, 9 words / 128 cols)
    """
    if table_kind not in (0, 1, 2, 3):
        raise ValueError(f"Unsupported PWV4 table_kind={table_kind} (expected 0, 1, 2, or 3)")
    if (pad_r % 128) != 0 or (pad_c % 128) != 0:
        raise ValueError(f"pad_r ({pad_r}) and pad_c ({pad_c}) must be multiples of 128")
    if (row_off % 128) != 0 or row_off < 0 or (k_sign_off % 128) != 0 or k_sign_off < 0:
        raise ValueError("row_off and k_sign_off must be non-negative multiples of 128")

    if table_kind == 3:
        lib = _get_tfq4_1024_lib()
        kernel_name = "tfq4_true_e8p_1024_dense"
        need_words = pad_r * (pad_c // 128) * 9
        need_lut_elems = 1024 * 8
    else:
        lib = _get_tfq4_lib()
        kernel_name = "tfq4_true_e8p_dense" if table_kind == 2 else "tfq4_e8p_dense"
        need_words = pad_r * (pad_c // 16)
        need_lut_elems = 256 * 8

    if lib is None or not hasattr(lib, kernel_name):
        raise RuntimeError(f"TFQ4 MPS Metal kernel {kernel_name} is unavailable on this PyTorch build")
    if words_i32.numel() < need_words:
        raise ValueError(f"words_i32 too small for table_kind={table_kind}: {words_i32.numel()} < {need_words}")
    if e8p_lut_f16.numel() < need_lut_elems:
        raise ValueError(f"e8p_lut_f16 too small for table_kind={table_kind}: {e8p_lut_f16.numel()} < {need_lut_elems}")

    orig_shape = list(x.shape)
    orig_c = orig_shape[-1]
    rows = x.numel() // orig_c
    if rows == 0:
        orig_shape[-1] = orig_r
        return torch.empty(orig_shape, device=x.device, dtype=x.dtype)

    x2d = x.reshape(rows, orig_c).to(dtype=torch.float16)
    if orig_c < pad_c:
        x2d = F.pad(x2d, (0, pad_c - orig_c))
    x2d = _as_zero_offset_mps(x2d)

    w_buf = _as_zero_offset_mps(words_i32)
    s_buf = _as_zero_offset_mps(scales_f16)
    lut_src = e8p_lut_f16.to(dtype=torch.float16)
    if table_kind != 3 and lut_src.numel() > 256 * 8:
        lut_src = lut_src.reshape(-1)[: 256 * 8]
    lut_buf = _as_zero_offset_mps(lut_src)

    x_rot = torch.empty((rows, pad_c), dtype=torch.float16, device=x.device)
    y_pad = torch.empty((rows, pad_r), dtype=torch.float32, device=x.device)
    params = torch.tensor(
        [rows, pad_r, pad_c, pad_c, pad_r, row_off, k_sign_off],
        dtype=torch.int32,
        device=x.device,
    )

    lib.tfq4_fwht_x(
        x2d,
        x_rot,
        params,
        threads=((pad_c // 128) * 32, rows, 1),
        group_size=(32, 1, 1),
    )
    getattr(lib, kernel_name)(
        w_buf,
        s_buf,
        lut_buf,
        x_rot,
        y_pad,
        params,
        threads=((pad_r // 128) * 256, (rows + 15) // 16, 1),
        group_size=(256, 1, 1),
    )

    y_out = y_pad[:, :orig_r] if orig_r < pad_r else y_pad
    orig_shape[-1] = orig_r
    return y_out.reshape(orig_shape).to(dtype=x.dtype)
