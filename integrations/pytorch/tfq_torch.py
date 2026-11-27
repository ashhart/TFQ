"""
Standalone PyTorch & Hugging Face Transformers integration for TFQ3 (3/4/5-bit) and TFQ4-E8P (2-bit).

Breaks TensorFold runtime lock-in: load and run ANY TFQ3 (.tffpw) or TFQ4 (.tffp4) checkpoint
directly in standard PyTorch / Hugging Face `transformers`:
  - NVIDIA CUDA (`sm_80+`): uses `tfq3_cuda_ext` (`src/kernels/cuda/tfq3_cuda.cuh`) directly on
    packed 3/4/5-bit weights with zero BF16 weight expansion in VRAM.
  - Apple Silicon MPS / CPU (or CUDA without compiled extension): portable exact on-the-fly
    dequantization fallback so every checkpoint runs out-of-the-box anywhere PyTorch runs.
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
    """Drop-in replacement for `torch.nn.Linear` backed by packed TFQ4-E8P (`TFP4` v4, 2.001 bpw) buffers."""

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
            tfq4_ref.PWV4_TABLE_LEGACY_TRAINED,
            tfq4_ref.PWV4_TABLE_TRAINED,
            tfq4_ref.PWV4_TABLE_TRUE_E8P,
        ):
            raise ValueError(f"Unsupported PWV4 table_kind={self.table_kind} (expected 0, 1, or 2)")
        self._e8p_lut = e8p_lut.astype(np.float32)
        self._cached_w: torch.Tensor | None = None

        self.register_buffer("words_i32", torch.from_numpy(words_u32.view(np.int32).copy()))
        self.register_buffer("scales_f16", torch.from_numpy(scales_f16.astype(np.float16).copy()))
        self.register_buffer("e8p_lut_f16", torch.from_numpy(e8p_lut.astype(np.float16).copy()))
        if bias is not None:
            self.register_buffer("bias", bias.detach().clone())
        else:
            self.register_buffer("bias", None)

    def dequantize(self, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        wd = self.words_i32.detach().cpu().numpy().view(np.uint32).reshape(self.pad_r, self.pad_c // 16)
        sc = self.scales_f16.detach().cpu().numpy().astype(np.float32).reshape(self.pad_r // 128, self.pad_c // 128)
        codes = np.empty((self.pad_r, self.pad_c // 8), dtype=np.uint16)
        codes[:, 0::2] = (wd & np.uint32(0xFFFF)).astype(np.uint16)
        codes[:, 1::2] = (wd >> np.uint32(16)).astype(np.uint16)
        if self.table_kind == tfq4_ref.PWV4_TABLE_TRUE_E8P:
            W_rot = tfq4_ref.decode_true_e8p_codes(codes, self._e8p_lut).reshape(self.pad_r, self.pad_c)
        else:
            k_idx = (codes & np.uint16(0xFF)).astype(np.int32)
            sm = (codes >> np.uint16(8)).astype(np.uint32)
            bits_mask = (sm[..., None] >> np.arange(8, dtype=np.uint32)) & np.uint32(1)
            sgn = np.where(bits_mask != 0, -1.0, 1.0).astype(np.float32)
            W_rot = (self._e8p_lut[k_idx] * sgn).reshape(self.pad_r, self.pad_c)
        sc_tile = np.repeat(np.repeat(sc, 128, axis=0), 128, axis=1)
        W_hat = tfq4_ref.inv_fwht2d_128(W_rot * sc_tile)[: self.out_features, : self.in_features]
        return torch.from_numpy(W_hat).to(device=self.words_i32.device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            ext = _get_cuda_ext()
            if ext is not None and hasattr(ext, "tfq4_linear"):
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
        return f"in_features={self.in_features}, out_features={self.out_features}, bits=2b-E8P, format=TFQ4-TFP4"


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


def load_tfq_into_hf(
    tfq_dir: str,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    cache_decoded: bool = False,
) -> Any:
    """
    Load any TFQ3 (.tffpw) or TFQ4 (.tffp4) checkpoint directory directly into a standard
    Hugging Face `AutoModelForCausalLM` without requiring TensorFold.
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

    # 2. Replace quantized nn.Linear layers with Tfq3Linear (TFPN) or Tfq4Linear (TFP4)
    pw3 = os.path.join(tfq_dir, "pw_v3_tables.bin")
    pw4 = os.path.join(tfq_dir, "pw_v4_e8p_tables.bin")

    if os.path.isfile(pw3):
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
        skipped += [f"{os.path.basename(sp)}: TFQ4 shard in a TFQ3 directory (mixed TFQ3+TFQ4 is not supported)"
                    for sp in sorted(glob.glob(os.path.join(tfq_dir, "*.tffp4")))]
    elif os.path.isfile(pw4):
        lut4, kind4 = tfq4_ref.load_pwv4_tables_with_kind(pw4)
        for sp in sorted(glob.glob(os.path.join(tfq_dir, "*.tffp4"))):
            with open(sp, "rb") as f:
                hdr = struct.unpack("<IIIIQQQQQQ", f.read(64))
                if hdr[0] != 0x34504654:
                    skipped.append(f"{os.path.basename(sp)}: magic 0x{hdr[0]:08X} is not TFP4")
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
                    R, C, _, sc_off, w_off, w_cnt, _, orig_r, orig_c, _ = struct.unpack("<IIIIQQfIII", rec[80:])
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
                        Tfq4Linear(orig_c, orig_r, wd, sc, lut4, bias=bias, cache_decoded=cache_decoded, table_kind=kind4),
                    )

    if missing:
        msg = [f"{tfq_dir}: {len(missing)} model tensors were not found in the checkpoint and would stay at random init:"]
        msg += [f"  {k}" for k in sorted(missing)]
        if skipped:
            msg.append("Records this loader skipped:")
            msg += [f"  {s}" for s in skipped]
        raise RuntimeError("\n".join(msg))
    return model.to(device=device)
