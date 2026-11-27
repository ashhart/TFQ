"""
Zero-build Apple Silicon Metal / PyTorch MPS fused kernels for TFQ3 (3/4/5-bit) and TFQ4 (2-bit E8P).

Uses PyTorch's built-in `torch.mps.compile_shader` runtime compiler directly against
`src/kernels/metal/tfq3_metal.metal` and `src/kernels/metal/tfq4_metal.metal` — zero Xcode,
zero C++ extension build step, zero BF16 weight expansion in unified memory.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))


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
    """Fused TFQ4 2.001 bpw 2D-FWHT + E8P linear on Apple Silicon MPS (table_kind 0/1 trained & 2 true E8P)."""
    lib = _get_tfq4_lib()
    if lib is None:
        raise RuntimeError("TFQ4 MPS Metal shaders are unavailable on this PyTorch build")
    if table_kind not in (0, 1, 2):
        raise ValueError(f"Unsupported PWV4 table_kind={table_kind} (expected 0, 1, or 2)")
    if (pad_r % 128) != 0 or (pad_c % 128) != 0:
        raise ValueError(f"pad_r ({pad_r}) and pad_c ({pad_c}) must be multiples of 128")
    if (row_off % 128) != 0 or row_off < 0 or (k_sign_off % 128) != 0 or k_sign_off < 0:
        raise ValueError("row_off and k_sign_off must be non-negative multiples of 128")

    kernel_name = "tfq4_true_e8p_dense" if table_kind == 2 else "tfq4_e8p_dense"
    if not hasattr(lib, kernel_name):
        raise RuntimeError(f"Metal kernel {kernel_name} not found in tfq4_metal.metal")

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
    lut_buf = _as_zero_offset_mps(e8p_lut_f16.to(dtype=torch.float16))

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
