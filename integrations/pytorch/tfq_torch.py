"""
Standalone PyTorch & Hugging Face Transformers integration for TFQ3 (3/4/5-bit) and
TFQ4-E8P (2.001 bpw table_kind 0/1/2 and 2.251 bpw E8P-1024 table_kind 3).

Breaks TensorFold runtime lock-in: load and run ANY TFQ3 (.tffpw) or TFQ4 (.tffp4) checkpoint
directly in standard PyTorch / Hugging Face `transformers`:
  - NVIDIA CUDA (`sm_80+`): uses `tfq3_cuda_ext` (`src/kernels/cuda/tfq3_cuda.cuh` and
    `tfq4_cuda.cuh`) directly on packed weights with zero BF16 weight expansion in VRAM.
  - Apple Silicon MPS: zero-build `torch.mps.compile_shader` fused Metal kernels (`tfq_mps.py`)
    for TFQ3 3/4/5-bit and TFQ4 2.00/2.25 bpw with zero BF16 weight expansion.
  - CPU (or CUDA without compiled extension): portable exact vectorized NumPy dequantization
    fallback so every checkpoint runs out-of-the-box anywhere PyTorch runs.
"""
from __future__ import annotations

import glob
import json
import os
import struct
import sys
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ..reference import tfq3_reference as tfq3_ref  # type: ignore
    from ..reference import tfq4_lattice as tfq4_ref  # type: ignore
except ImportError:
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _REF_DIR = os.path.join(REPO_ROOT, "reference")
    if _REF_DIR not in sys.path:
        sys.path.insert(0, _REF_DIR)
    import tfq3_reference as tfq3_ref
    import tfq4_lattice as tfq4_ref

MAGIC_PWV4 = 0x34565750
MAGIC_TFP4 = 0x34504654
TFQ4_VER = 4
PWV4_TABLE_LEGACY_TRAINED = 0
PWV4_TABLE_TRAINED = 1
PWV4_TABLE_TRUE_E8P = 2
PWV4_TABLE_TRUE_E8P_1024 = 3
PWV4_TABLE_MAX_SUPPORTED = 3

_CUDA_EXT = None
_CUDA_EXT_TRIED = False
_MPS_MOD = None
_MPS_TRIED = False


def _get_cuda_ext():
    global _CUDA_EXT, _CUDA_EXT_TRIED
    if _CUDA_EXT_TRIED:
        return _CUDA_EXT
    _CUDA_EXT_TRIED = True
    try:
        import tfq3_cuda_ext  # type: ignore
        _CUDA_EXT = tfq3_cuda_ext
    except Exception:
        _CUDA_EXT = None
    return _CUDA_EXT


def _get_mps_mod():
    global _MPS_MOD, _MPS_TRIED
    if _MPS_TRIED:
        return _MPS_MOD
    _MPS_TRIED = True
    try:
        try:
            import tfq_mps  # type: ignore
        except ImportError:
            from . import tfq_mps  # type: ignore
        _MPS_MOD = tfq_mps if tfq_mps.mps_kernels_available() else None
    except Exception:
        _MPS_MOD = None
    return _MPS_MOD


def _decode_tfpn_words_np(words_u32: np.ndarray, scales_f16: np.ndarray, pad_r: int, pad_c: int, bits: int, tbl: dict) -> np.ndarray:
    """Fast exact vectorized TFPN v3 decoder (matches reference/tfq3_reference.py bit-for-bit)."""
    return tfq3_ref.decode_tfpn_fast_np(words_u32, scales_f16, pad_r, pad_c, bits, tbl)


def load_pwv4_tables_auto(path: str | os.PathLike[str]) -> tuple[np.ndarray, int, bytes]:
    """
    Load and validate a `PWV4` v4 codebook file (`pw_v4_e8p_tables.bin`).

    Supports both:
      - 16,384-byte `k_codes=256` files (`table_kind` 0/1 trained, 2 True QuIP# E8P-256)
      - 32,768-byte `k_codes=1024` files (`table_kind` 3 True QuIP# E8P-1024, 2.251 bpw)

    Returns `(lut_f32 [k_codes, 8], table_kind, raw_bytes)`.
    """
    p = os.fspath(path)
    with open(p, "rb") as f:
        raw = f.read()
    if len(raw) not in (16384, 32768):
        raise ValueError(f"Expected 16384- or 32768-byte PWV4 file, got {len(raw)}: {p}")
    magic, ver, k_codes, dim, table_kind = struct.unpack_from("<IIIII", raw, 0)
    if magic != MAGIC_PWV4 or ver != TFQ4_VER or dim != 8:
        raise ValueError(f"Bad PWV4 header in {p}: magic=0x{magic:08x} ver={ver} k={k_codes} dim={dim}")
    if table_kind < 0 or table_kind > PWV4_TABLE_MAX_SUPPORTED:
        raise ValueError(f"Unknown PWV4 table_kind={table_kind} in {p} (supported: 0, 1, 2, 3)")
    if table_kind == PWV4_TABLE_TRUE_E8P_1024:
        if len(raw) != 32768 or k_codes != 1024:
            raise ValueError(
                f"PWV4 table_kind=3 requires 32768-byte file with k_codes=1024, got {len(raw)} bytes k_codes={k_codes}"
            )
    else:
        if len(raw) != 16384 or k_codes != 256:
            raise ValueError(
                f"PWV4 table_kind={table_kind} requires 16384-byte file with k_codes=256, got {len(raw)} bytes k_codes={k_codes}"
            )
    lut = np.frombuffer(raw, dtype=np.float16, count=k_codes * 8, offset=64).reshape(k_codes, 8).astype(np.float32)
    if not np.all(np.isfinite(lut)) or np.any(lut < 0.0):
        raise ValueError(f"PWV4 e8p_lut contains non-finite or negative entries in {p}")
    if table_kind in (PWV4_TABLE_TRUE_E8P, PWV4_TABLE_TRUE_E8P_1024):
        two_v = lut * 2.0
        r = np.rint(two_v)
        if np.any(lut < 0.5) or np.any(np.abs(two_v - r) > 1e-3) or np.any((r.astype(np.int64) & 1) == 0):
            raise ValueError(f"PWV4 table_kind={table_kind} LUT entries must be positive odd half-integers")
    return lut, int(table_kind), raw


def pack_tfq4_e8p1024_codes(abs_idx: np.ndarray, s7: np.ndarray, coset: np.ndarray) -> np.ndarray:
    """
    Pack True QuIP# E8P-1024 (table_kind=3, 2.251 bpw) per-8D-vector fields into the
    canonical 9-word-per-128-col `uint32` layout.

    Inputs have shape `[R, C // 8]` (with `C % 128 == 0`):
      - `abs_idx`: integer indices in `[0, 1023]`
      - `s7`: 7-bit sign masks in `[0, 127]`
      - `coset`: 1-bit coset flags in `{0, 1}`
    Returns `words_u32` with shape `[R, (C // 128) * 9]`.
    """
    a = np.asarray(abs_idx, dtype=np.uint32)
    s = np.asarray(s7, dtype=np.uint32) & np.uint32(0x7F)
    c = np.asarray(coset, dtype=np.uint32) & np.uint32(1)
    if a.ndim != 2 or (a.shape[1] % 16) != 0:
        raise ValueError(f"Expected shape [R, 16*n_bc], got {a.shape}")
    R, n_vec = a.shape
    n_bc = n_vec // 16
    a = a.reshape(R, n_bc, 16)
    s = s.reshape(R, n_bc, 16)
    c = c.reshape(R, n_bc, 16)

    low16 = (a & np.uint32(0xFF)) | (s << np.uint32(8)) | (c << np.uint32(15))
    hi2 = (a >> np.uint32(8)) & np.uint32(3)

    out = np.zeros((R, n_bc, 9), dtype=np.uint32)
    out[:, :, :8] = low16[:, :, 0::2] | (low16[:, :, 1::2] << np.uint32(16))
    shifts = (np.arange(16, dtype=np.uint32) * 2)[None, None, :]
    out[:, :, 8] = np.bitwise_or.reduce(hi2 << shifts, axis=2)
    return out.reshape(R, n_bc * 9)


def _inv_fwht2d_128_offset(W_rot: np.ndarray, row_off: int = 0, k_sign_off: int = 0) -> np.ndarray:
    """Inverse 2D FWHT-128 with absolute row/column sign offsets (Col-TP / Row-TP)."""
    R, C = W_rot.shape
    sgn_r = tfq4_ref.sgn_vec(0x9E3779B9, row_off + R)[row_off:]
    sgn_c = tfq4_ref.sgn_vec(0x12345678, k_sign_off + C)[k_sign_off:]
    X = tfq4_ref.fwht128_rows(W_rot.T).T * sgn_r[:, None]
    X = tfq4_ref.fwht128_rows(X) * sgn_c[None, :]
    return X


def decode_tfq4_words_np(
    words_u32: np.ndarray,
    scales_f16: np.ndarray,
    pad_r: int,
    pad_c: int,
    e8p_lut: np.ndarray,
    table_kind: int,
    *,
    row_off: int = 0,
    k_sign_off: int = 0,
) -> np.ndarray:
    """
    Exact vectorized TFQ4-E8P decoder supporting all `PWV4` table kinds (0/1/2 at 2.001 bpw
    and 3 True QuIP# E8P-1024 at 2.251 bpw) plus Col-TP `row_off` and Row-TP `k_sign_off`.
    """
    lut = np.asarray(e8p_lut, dtype=np.float32)
    n_br, n_bc = pad_r // 128, pad_c // 128
    sc = np.asarray(scales_f16, dtype=np.float32).reshape(n_br, n_bc)

    if table_kind == PWV4_TABLE_TRUE_E8P_1024:
        if lut.shape != (1024, 8):
            raise ValueError(f"table_kind=3 requires LUT shape (1024, 8), got {lut.shape}")
        wd = np.asarray(words_u32, dtype=np.uint32).reshape(pad_r, n_bc, 9)
        w_lo = wd[:, :, :8]
        w_hi = wd[:, :, 8]
        low16 = np.empty((pad_r, n_bc, 16), dtype=np.uint32)
        low16[:, :, 0::2] = w_lo & np.uint32(0xFFFF)
        low16[:, :, 1::2] = w_lo >> np.uint32(16)
        shifts = (np.arange(16, dtype=np.uint32) * 2)[None, None, :]
        hi2 = (w_hi[:, :, None] >> shifts) & np.uint32(3)
        abs_idx = ((low16 & np.uint32(0xFF)) | (hi2 << np.uint32(8))).astype(np.int32)
        s7 = (low16 >> np.uint32(8)) & np.uint32(0x7F)
        coset = (low16 >> np.uint32(15)) & np.uint32(1)
        m_par = (np.sum(np.floor(lut).astype(np.int64), axis=1) & 1).astype(np.uint32)
        pc = np.zeros_like(s7)
        for b in range(7):
            pc ^= (s7 >> np.uint32(b)) & np.uint32(1)
        s7_bit = (pc ^ m_par[abs_idx]) & np.uint32(1)
        sm8 = s7 | (s7_bit << np.uint32(7))
        bits_mask = (sm8[..., None] >> np.arange(8, dtype=np.uint32)) & np.uint32(1)
        sgn = np.where(bits_mask != 0, -1.0, 1.0).astype(np.float32)
        q = lut[abs_idx] * sgn - np.float32(0.25)
        W_rot = np.where(coset[..., None] != 0, -q, q).astype(np.float32).reshape(pad_r, pad_c)
    else:
        lut256 = lut[:256]
        if lut256.shape != (256, 8):
            raise ValueError(f"table_kind={table_kind} requires >=256 LUT rows of dim 8, got {lut.shape}")
        wd = np.asarray(words_u32, dtype=np.uint32).reshape(pad_r, pad_c // 16)
        codes = np.empty((pad_r, pad_c // 8), dtype=np.uint16)
        codes[:, 0::2] = (wd & np.uint32(0xFFFF)).astype(np.uint16)
        codes[:, 1::2] = (wd >> np.uint32(16)).astype(np.uint16)
        if table_kind == PWV4_TABLE_TRUE_E8P:
            W_rot = tfq4_ref.decode_true_e8p_codes(codes, lut256).reshape(pad_r, pad_c)
        elif table_kind in (PWV4_TABLE_LEGACY_TRAINED, PWV4_TABLE_TRAINED):
            k_idx = (codes & np.uint16(0xFF)).astype(np.int32)
            sm = (codes >> np.uint16(8)).astype(np.uint32)
            bits_mask = (sm[..., None] >> np.arange(8, dtype=np.uint32)) & np.uint32(1)
            sgn = np.where(bits_mask != 0, -1.0, 1.0).astype(np.float32)
            W_rot = (lut256[k_idx] * sgn).reshape(pad_r, pad_c)
        else:
            raise ValueError(f"Unsupported PWV4 table_kind={table_kind}")

    sc_tile = np.repeat(np.repeat(sc, 128, axis=0), 128, axis=1)
    if row_off == 0 and k_sign_off == 0:
        return tfq4_ref.inv_fwht2d_128(W_rot * sc_tile)
    return _inv_fwht2d_128_offset(W_rot * sc_tile, row_off=row_off, k_sign_off=k_sign_off)


class Tfq3Linear(nn.Module):
    """
    Drop-in replacement for `torch.nn.Linear` backed by packed TFQ3 (`TFPN` v3) 3/4/5-bit buffers.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int,
        words_u32: np.ndarray,
        scales_f16: np.ndarray,
        tables_bytes: bytes,
        tables_dict: dict,
        bias: torch.Tensor | None = None,
        cache_decoded: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = int(bits)
        self.pad_r = ((out_features + 127) // 128) * 128
        self.pad_c = ((in_features + 127) // 128) * 128
        self.cache_decoded = cache_decoded
        self._tables_dict = tables_dict
        self._cached_w: torch.Tensor | None = None

        # Store packed uint32 words as int32 (torch.uint32 lacks some ops on MPS/older builds)
        self.register_buffer("words_i32", torch.from_numpy(words_u32.view(np.int32).copy()))
        self.register_buffer("scales_f16", torch.from_numpy(scales_f16.astype(np.float16).copy()))
        self.register_buffer("tables_u8", torch.from_numpy(np.frombuffer(tables_bytes, dtype=np.uint8).copy()))
        if bias is not None:
            self.register_buffer("bias", bias.detach().clone())
        else:
            self.register_buffer("bias", None)

    def dequantize(self, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        w_np = _decode_tfpn_words_np(
            self.words_i32.detach().cpu().numpy().view(np.uint32),
            self.scales_f16.detach().cpu().numpy(),
            self.pad_r,
            self.pad_c,
            self.bits,
            self._tables_dict,
        )[: self.out_features, : self.in_features]
        return torch.from_numpy(w_np).to(device=self.words_i32.device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ext = _get_cuda_ext() if x.is_cuda else None
        if ext is not None:
            y = ext.tfq3_linear(
                x.to(torch.bfloat16),
                self.words_i32,
                self.scales_f16,
                self.tables_u8,
                self.bits,
                self.pad_r,
                self.pad_c,
                self.out_features,
            )
            if self.bias is not None:
                y = y + self.bias.to(dtype=y.dtype)
            return y.to(dtype=x.dtype)

        if x.device.type == "mps":
            mps = _get_mps_mod()
            if mps is not None:
                y = mps.tfq3_linear_mps(
                    x,
                    self.words_i32,
                    self.scales_f16,
                    self.tables_u8,
                    self.bits,
                    self.pad_r,
                    self.pad_c,
                    self.out_features,
                )
                if self.bias is not None:
                    y = y + self.bias.to(device=y.device, dtype=y.dtype)
                return y

        if self.cache_decoded and self._cached_w is not None and self._cached_w.device == x.device:
            w = self._cached_w
        else:
            w = self.dequantize(dtype=x.dtype)
            if self.cache_decoded:
                self._cached_w = w
        b = self.bias.to(device=x.device, dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, w, b)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bits={self.bits}b, format=TFQ3-TFPN"


class Tfq4Linear(nn.Module):
    """
    Drop-in replacement for `torch.nn.Linear` backed by packed TFQ4-E8P (`TFP4` v4) buffers:
      - `table_kind` 0/1: 2.001 bpw trained 8-sign E8P LUT (`[256, 8]`, 8 words / 128 cols)
      - `table_kind` 2:   2.001 bpw True QuIP# E8P-256 (`[256, 8]`, 8 words / 128 cols)
      - `table_kind` 3:   2.251 bpw True QuIP# E8P-1024 (`[1024, 8]`, 9 words / 128 cols)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        words_u32: np.ndarray,
        scales_f16: np.ndarray,
        e8p_lut: np.ndarray,
        bias: torch.Tensor | None = None,
        cache_decoded: bool = False,
        table_kind: int = 0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pad_r = ((out_features + 127) // 128) * 128
        self.pad_c = ((in_features + 127) // 128) * 128
        self.cache_decoded = cache_decoded
        self.table_kind = int(table_kind)
        if self.table_kind not in (
            PWV4_TABLE_LEGACY_TRAINED,
            PWV4_TABLE_TRAINED,
            PWV4_TABLE_TRUE_E8P,
            PWV4_TABLE_TRUE_E8P_1024,
        ):
            raise ValueError(f"Unsupported PWV4 table_kind={self.table_kind} (expected 0, 1, 2, or 3)")

        lut_arr = np.asarray(e8p_lut, dtype=np.float32)
        wd_arr = np.asarray(words_u32, dtype=np.uint32)
        if self.table_kind == PWV4_TABLE_TRUE_E8P_1024:
            if lut_arr.shape != (1024, 8):
                raise ValueError(f"PWV4 table_kind=3 requires e8p_lut shape (1024, 8), got {lut_arr.shape}")
            exp_words = self.pad_r * (self.pad_c // 128) * 9
            if wd_arr.size != exp_words:
                raise ValueError(
                    f"PWV4 table_kind=3 requires {exp_words} uint32 words for [{self.pad_r}, {self.pad_c}], got {wd_arr.size}"
                )
            self.b_bits = 225
        else:
            if lut_arr.ndim != 2 or lut_arr.shape[1] != 8 or lut_arr.shape[0] < 256:
                raise ValueError(f"PWV4 table_kind={self.table_kind} requires e8p_lut shape (>=256, 8), got {lut_arr.shape}")
            lut_arr = lut_arr[:256]
            exp_words = self.pad_r * (self.pad_c // 16)
            if wd_arr.size != exp_words:
                raise ValueError(
                    f"PWV4 table_kind={self.table_kind} requires {exp_words} uint32 words for [{self.pad_r}, {self.pad_c}], got {wd_arr.size}"
                )
            self.b_bits = 2

        self._e8p_lut = lut_arr.copy()
        self._cached_w: torch.Tensor | None = None

        self.register_buffer("words_i32", torch.from_numpy(np.ascontiguousarray(wd_arr).view(np.int32).copy()))
        self.register_buffer("scales_f16", torch.from_numpy(scales_f16.astype(np.float16).copy()))
        self.register_buffer("e8p_lut_f16", torch.from_numpy(lut_arr.astype(np.float16).copy()))
        if bias is not None:
            self.register_buffer("bias", bias.detach().clone())
        else:
            self.register_buffer("bias", None)

    def dequantize(self, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        W_hat = decode_tfq4_words_np(
            self.words_i32.detach().cpu().numpy().view(np.uint32),
            self.scales_f16.detach().cpu().numpy(),
            self.pad_r,
            self.pad_c,
            self._e8p_lut,
            self.table_kind,
        )[: self.out_features, : self.in_features]
        return torch.from_numpy(W_hat).to(device=self.words_i32.device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            ext = _get_cuda_ext()
            if ext is not None and hasattr(ext, "tfq4_linear"):
                max_tk = int(getattr(ext, "TFQ4_MAX_TABLE_KIND", 2))
                if self.table_kind <= max_tk:
                    y = ext.tfq4_linear(
                        x.to(torch.bfloat16),
                        self.words_i32,
                        self.scales_f16,
                        self.e8p_lut_f16,
                        self.table_kind,
                        self.pad_r,
                        self.pad_c,
                        self.out_features,
                    )
                    if self.bias is not None:
                        y = y + self.bias.to(dtype=y.dtype)
                    return y.to(dtype=x.dtype)

        if x.device.type == "mps":
            mps = _get_mps_mod()
            if mps is not None:
                try:
                    y = mps.tfq4_linear_mps(
                        x,
                        self.words_i32,
                        self.scales_f16,
                        self.e8p_lut_f16,
                        self.table_kind,
                        self.pad_r,
                        self.pad_c,
                        self.out_features,
                    )
                    if self.bias is not None:
                        y = y + self.bias.to(device=y.device, dtype=y.dtype)
                    return y
                except RuntimeError:
                    pass

        if self.cache_decoded and self._cached_w is not None and self._cached_w.device == x.device:
            w = self._cached_w
        else:
            w = self.dequantize(dtype=x.dtype)
            if self.cache_decoded:
                self._cached_w = w
        b = self.bias.to(device=x.device, dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, w, b)

    def extra_repr(self) -> str:
        if self.table_kind == PWV4_TABLE_TRUE_E8P_1024:
            tag = "2.25b-E8P1024"
        elif self.table_kind == PWV4_TABLE_TRUE_E8P:
            tag = "2b-E8P-true"
        else:
            tag = "2b-E8P"
        return f"in_features={self.in_features}, out_features={self.out_features}, bits={tag}, format=TFQ4-TFP4"


def _set_submodule(root: nn.Module, dotted_name: str, new_mod: nn.Module) -> None:
    parts = dotted_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_mod)


def _read_safetensors_dict(path: str) -> dict[str, torch.Tensor]:
    """Minimal zero-dependency safetensors reader supporting BF16/F16/F32."""
    out = {}
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hlen))
        base = 8 + hlen
        for name, meta in hdr.items():
            if name == "__metadata__":
                continue
            dt, shape, (o0, o1) = meta["dtype"], meta["shape"], meta["data_offsets"]
            f.seek(base + o0)
            raw = f.read(o1 - o0)
            if dt == "BF16":
                arr = np.frombuffer(raw, dtype=np.uint16).copy()
                t = torch.from_numpy(arr).view(torch.bfloat16).reshape(shape)
            elif dt == "F16":
                t = torch.from_numpy(np.frombuffer(raw, dtype=np.float16).copy()).reshape(shape)
            elif dt == "F32":
                t = torch.from_numpy(np.frombuffer(raw, dtype=np.float32).copy()).reshape(shape)
            else:
                continue
            out[name] = t
    return out


def _load_tfp4_shards_into_model(
    model: nn.Module,
    mod_map: dict[str, nn.Module],
    res_state: dict[str, torch.Tensor],
    missing: set[str],
    skipped: list[str],
    tfq_dir: str,
    pw4_path: str,
    cache_decoded: bool,
) -> None:
    lut4, kind4, _ = load_pwv4_tables_auto(pw4_path)
    for sp in sorted(glob.glob(os.path.join(tfq_dir, "*.tffp4"))):
        with open(sp, "rb") as f:
            hdr = struct.unpack("<IIIIQQQQQQ", f.read(64))
            if hdr[0] != MAGIC_TFP4:
                skipped.append(f"{os.path.basename(sp)}: magic 0x{hdr[0]:08X} is not TFP4")
                continue
            nt, off_meta, off_scales, off_weights = hdr[2], hdr[4], hdr[5], hdr[6]
            f.seek(off_meta)
            recs = [f.read(128) for _ in range(nt)]
            for rec in recs:
                nm = rec[:80].split(b"\x00", 1)[0].decode("utf-8")
                if nm not in mod_map or not isinstance(mod_map[nm], nn.Linear):
                    skipped.append(
                        f"{os.path.basename(sp)}: {nm} ({type(mod_map[nm]).__name__ if nm in mod_map else 'no such module'}, only nn.Linear is supported)"
                    )
                    continue
                R, C, b_bits, sc_off, w_off, w_cnt, _, orig_r, orig_c, _ = struct.unpack("<IIIIQQfIII", rec[80:])
                orig_r = orig_r or R
                orig_c = orig_c or C
                if b_bits == 225:
                    if kind4 != PWV4_TABLE_TRUE_E8P_1024:
                        raise ValueError(
                            f"{os.path.basename(sp)}: tensor {nm} has b_bits=225 but pw_v4_e8p_tables.bin has table_kind={kind4} (expected 3)"
                        )
                    eff_kind = PWV4_TABLE_TRUE_E8P_1024
                    eff_lut = lut4
                elif b_bits == 2:
                    eff_kind = PWV4_TABLE_TRUE_E8P if kind4 == PWV4_TABLE_TRUE_E8P_1024 else kind4
                    eff_lut = lut4[:256]
                else:
                    skipped.append(f"{os.path.basename(sp)}: {nm} has unsupported TFP4 b_bits={b_bits}")
                    continue
                missing.discard(f"{nm}.weight")
                f.seek(off_scales + sc_off * 2)
                sc = np.frombuffer(f.read((R // 128) * (C // 128) * 2), dtype=np.float16).reshape(R // 128, C // 128).copy()
                f.seek(off_weights + w_off * 4)
                wd = np.frombuffer(f.read(w_cnt * 4), dtype=np.uint32).copy()
                old_mod = mod_map[nm]
                bias = res_state.get(f"{nm}.bias", old_mod.bias if old_mod.bias is not None else None)
                _set_submodule(
                    model,
                    nm,
                    Tfq4Linear(orig_c, orig_r, wd, sc, eff_lut, bias=bias, cache_decoded=cache_decoded, table_kind=eff_kind),
                )


def load_tfq_into_hf(
    tfq_dir: str,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    cache_decoded: bool = False,
) -> Any:
    """
    Load any TFQ3 (.tffpw), TFQ4 (.tffp4, 2.00 or 2.25 bpw), or hybrid TFQ3+TFQ4 dense
    checkpoint directory directly into a standard Hugging Face `AutoModelForCausalLM`
    without requiring TensorFold.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(tfq_dir, local_files_only=True)
    # Construct directly on CPU so non-persistent buffers (e.g. rotary_emb.inv_freq) are properly initialized.
    model = AutoModelForCausalLM.from_config(cfg, dtype=dtype)

    # 1. Load residual tensors (embeddings, norms, biases, and unquantized heads)
    res_state: dict[str, torch.Tensor] = {}
    for sf in sorted(glob.glob(os.path.join(tfq_dir, "*.safetensors"))):
        res_state.update(_read_safetensors_dict(sf))
    missing = set(model.load_state_dict(res_state, strict=False, assign=True).missing_keys)
    if getattr(cfg, "tie_word_embeddings", False) and hasattr(model, "tie_weights"):
        model.tie_weights()
    # A missing key that now shares its tensor with a loaded one (tied embeddings) is loaded.
    params = dict(model.named_parameters(remove_duplicate=False))
    loaded_ids = {id(t) for n, t in params.items() if n not in missing}
    missing = {k for k in missing if not (k in params and id(params[k]) in loaded_ids)}
    skipped: list[str] = []

    mod_map = dict(model.named_modules())

    # 2. Replace quantized nn.Linear layers with Tfq3Linear (TFPN) and/or Tfq4Linear (TFP4)
    pw3 = os.path.join(tfq_dir, "pw_v3_tables.bin")
    pw4 = os.path.join(tfq_dir, "pw_v4_e8p_tables.bin")
    has_pw3 = os.path.isfile(pw3)
    has_pw4 = os.path.isfile(pw4)

    if has_pw3:
        tbl_bytes = open(pw3, "rb").read()
        tbl_dict = tfq3_ref.load_tables(pw3)
        for sp in sorted(glob.glob(os.path.join(tfq_dir, "*.tffpw"))):
            with open(sp, "rb") as f:
                hdr = struct.unpack("<IIIIQQQQQQ", f.read(64))
                if hdr[0] != 0x4E504654:
                    skipped.append(f"{os.path.basename(sp)}: magic 0x{hdr[0]:08X} is not TFPN (TFPW MoE expert shards are not supported)")
                    continue
                nt, off_meta, off_scales, off_weights = hdr[2], hdr[4], hdr[5], hdr[6]
                f.seek(off_meta)
                recs = [f.read(128) for _ in range(nt)]
                for rec in recs:
                    nm = rec[:80].split(b"\x00", 1)[0].decode("utf-8")
                    if nm not in mod_map or not isinstance(mod_map[nm], nn.Linear):
                        skipped.append(f"{os.path.basename(sp)}: {nm} ({type(mod_map[nm]).__name__ if nm in mod_map else 'no such module'}, only nn.Linear is supported)")
                        continue
                    missing.discard(f"{nm}.weight")
                    R, C, bits, sc_off, w_off, w_cnt, _, orig_r, orig_c, _ = struct.unpack("<IIIIQQfIII", rec[80:])
                    orig_r = orig_r or R
                    orig_c = orig_c or C
                    f.seek(off_scales + sc_off * 2)
                    sc = np.frombuffer(f.read((R // 128) * (C // 128) * 2), dtype=np.float16).reshape(R // 128, C // 128).copy()
                    f.seek(off_weights + w_off * 4)
                    wd = np.frombuffer(f.read(w_cnt * 4), dtype=np.uint32).copy()
                    old_mod = mod_map[nm]
                    bias = res_state.get(f"{nm}.bias", old_mod.bias if old_mod.bias is not None else None)
                    _set_submodule(
                        model, nm,
                        Tfq3Linear(orig_c, orig_r, bits, wd, sc, tbl_bytes, tbl_dict, bias=bias, cache_decoded=cache_decoded),
                    )
        if not has_pw4:
            skipped += [
                f"{os.path.basename(sp)}: TFQ4 shard present without pw_v4_e8p_tables.bin"
                for sp in sorted(glob.glob(os.path.join(tfq_dir, "*.tffp4")))
            ]

    if has_pw4:
        _load_tfp4_shards_into_model(model, mod_map, res_state, missing, skipped, tfq_dir, pw4, cache_decoded)

    if missing:
        msg = [f"{tfq_dir}: {len(missing)} model tensors were not found in the checkpoint and would stay at random init:"]
        msg += [f"  {k}" for k in sorted(missing)]
        if skipped:
            msg.append("Records this loader skipped:")
            msg += [f"  {s}" for s in skipped]
        raise RuntimeError("\n".join(msg))
    return model.to(device=device)
