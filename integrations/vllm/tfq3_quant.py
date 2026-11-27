"""
vLLM QuantizationConfig, LinearMethodBase, FusedMoEMethodBase, and Shard Weight Loaders
for TFQ3 (`TFPN` v3 3/4/5-bit), TFQ4 (`TFP4` v4 2.00/2.25 bpw E8P), and `TFPW` v3 / Hybrid MoE.

Supports:
  - Automatic registration of `--quantization tfq3` (and `tfq4` alias) via vLLM's
    `register_quantization_config`
  - Per-tensor bit widths (`2`, `225`, `3`, `4`, `5`) parsed from `tfq3_manifest.json` /
    `tfq4_manifest.json` or auto-detected via `Tfq3Config.from_checkpoint_dir(...)`
  - Direct `.tffpw` (`TFPN`), `.tffp4` (`TFP4` v4 `table_kind=0..3`), and `layer_XX.tffpw`
    (`TFPW` v3 / hybrid MoE) shard loading with zero TensorFold runtime dependency
  - Merged projections (`qkv_proj`, `gate_up_proj`, arbitrary N-way) with independent per-partition
    128-row padding, mixed bit widths, and absolute row signs (`row_off`)
  - Zero-repack Tensor Parallelism (`TP > 1`): Column-Parallel (`row_off`) and Row-Parallel
    (`k_sign_off` + 1-block trellis state resume `bc0_buf` for TFQ3, stateless for TFQ4)
    across CUDA (`tfq3_cuda_ext`), Apple Silicon MPS (`tfq_mps`), and portable CPU fallback
  - Routed MoE (`Tfq3MoeMethod` + `load_tfq3_moe_weights`) wrapping `tfq_moe.py` and
    `tfq3_cuda_ext.tfq3_moe` (standard SwiGLU, GLM clamp, Inkling-2 tanh softcap, hybrid 2/225/3/4/5b)
"""
from __future__ import annotations

import glob
import json
import os
import re
import struct
import sys
from collections import Counter
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from ..reference import tfq3_reference as tfq3_ref  # type: ignore
except ImportError:
    _REF_DIR = os.path.join(_REPO_ROOT, "reference")
    if _REF_DIR not in sys.path:
        sys.path.insert(0, _REF_DIR)
    import tfq3_reference as tfq3_ref

try:
    from ..pytorch import tfq_torch as _tfq_torch  # type: ignore
    from ..pytorch import tfq_moe as _tfq_moe  # type: ignore
except ImportError:
    _PT_DIR = os.path.join(_REPO_ROOT, "integrations", "pytorch")
    if _PT_DIR not in sys.path:
        sys.path.insert(0, _PT_DIR)
    import tfq_torch as _tfq_torch  # type: ignore
    import tfq_moe as _tfq_moe  # type: ignore

MAGIC_TFPN = 0x4E504654
MAGIC_TFPW = 0x57504654
MAGIC_PWV3 = 0x33565750
MAGIC_PWV4 = 0x34565750
MAGIC_TFP4 = 0x34504654
FORMAT_VER = 3
TFQ4_VER = 4

_LAYER_IDX_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_LAYER_FILE_RE = re.compile(r"^layer_(\d+)\.tffpw$")

try:
    from vllm.model_executor.layers.linear import LinearMethodBase, set_weight_attrs  # type: ignore
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig  # type: ignore
    try:
        from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase  # type: ignore
    except Exception:
        class FusedMoEMethodBase:  # type: ignore
            pass
    HAS_VLLM = True
except Exception:
    HAS_VLLM = False

    class QuantizationConfig:  # type: ignore
        pass

    class LinearMethodBase:  # type: ignore
        pass

    class FusedMoEMethodBase:  # type: ignore
        pass

    def set_weight_attrs(weight: torch.Tensor, attrs: dict[str, Any]) -> None:
        for k, v in attrs.items():
            setattr(weight, k, v)


def _align128(n: int) -> int:
    return ((n + 127) // 128) * 128


def _strip_weight_suffix(name: str) -> str:
    return name[:-7] if name.endswith(".weight") else name


def _get_tp_rank_size(extra_weight_attrs: dict[str, Any]) -> tuple[int, int]:
    if "tp_rank" in extra_weight_attrs or "tp_size" in extra_weight_attrs:
        return int(extra_weight_attrs.get("tp_rank", 0)), int(extra_weight_attrs.get("tp_size", 1))
    if HAS_VLLM:
        try:
            from vllm.distributed import (  # type: ignore
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
            )
            return int(get_tensor_model_parallel_rank()), int(get_tensor_model_parallel_world_size())
        except Exception:
            pass
    return 0, 1


def _get_cuda_ext():
    try:
        import tfq3_cuda_ext  # type: ignore
        return tfq3_cuda_ext
    except Exception:
        return None


def _get_mps_mod():
    try:
        try:
            from ..pytorch import tfq_mps  # type: ignore
        except ImportError:
            _pt_dir = os.path.join(_REPO_ROOT, "integrations", "pytorch")
            if _pt_dir not in sys.path:
                sys.path.insert(0, _pt_dir)
            import tfq_mps  # type: ignore
        return tfq_mps if tfq_mps.mps_kernels_available() else None
    except Exception:
        return None


def _parse_pwv3_tables_bytes(raw: bytes) -> dict:
    if len(raw) < 10816:
        raise ValueError(f"PWV3 tables buffer too small: {len(raw)} bytes")
    magic, ver = struct.unpack_from("<II", raw, 0)
    if magic != MAGIC_PWV3 or ver != FORMAT_VER:
        raise ValueError(f"Bad PWV3 tables header: magic=0x{magic:08x} ver={ver}")
    a3, b3, a4, b4, a5, b5 = struct.unpack_from("<6f", raw, 8)
    lut3 = np.frombuffer(raw, dtype=np.float16, count=256 * 2, offset=64).reshape(256, 2).copy()
    lut4 = np.frombuffer(raw, dtype=np.float16, count=1024 * 2, offset=1088).reshape(1024, 2).copy()
    lut5 = np.frombuffer(raw, dtype=np.float16, count=1024 * 2, offset=5184).reshape(1024, 2).copy()
    c64_3 = np.frombuffer(raw, dtype=np.float16, count=64 * 4, offset=9280).reshape(64, 4).copy()
    c64_4 = np.frombuffer(raw, dtype=np.float16, count=64 * 4, offset=9792).reshape(64, 4).copy()
    c64_5 = np.frombuffer(raw, dtype=np.float16, count=64 * 4, offset=10304).reshape(64, 4).copy()
    return {
        "a": {3: a3, 4: a4, 5: a5},
        "b": {3: b3, 4: b4, 5: b5},
        "lut": {3: lut3, 4: lut4, 5: lut5},
        "c64": {3: c64_3, 4: c64_4, 5: c64_5},
    }


def _resolve_partition_names(
    prefix: str,
    output_partition_sizes: list[int],
    explicit_names: Optional[list[str]] = None,
) -> list[str]:
    if explicit_names is not None:
        return [_strip_weight_suffix(n) for n in explicit_names]
    clean = _strip_weight_suffix(prefix)
    n_parts = len(output_partition_sizes)
    if n_parts == 1:
        return [clean]
    if n_parts == 3 and clean.endswith("qkv_proj"):
        base = clean[: -len("qkv_proj")]
        return [base + "q_proj", base + "k_proj", base + "v_proj"]
    if n_parts == 2 and clean.endswith("gate_up_proj"):
        base = clean[: -len("gate_up_proj")]
        return [base + "gate_proj", base + "up_proj"]
    return [f"{clean}.{i}" for i in range(n_parts)]


def _infer_layer_idx(prefix: str, explicit: Optional[int] = None) -> Optional[int]:
    if explicit is not None:
        return int(explicit)
    m = _LAYER_IDX_RE.search(prefix or "")
    return int(m.group(1)) if m else None


class Tfq3CheckpointReader:
    """Random-access reader for TFQ3 (`TFPN` v3 `.tffpw`) checkpoints and `pw_v3_tables.bin`."""

    def __init__(self, checkpoint_dir: str) -> None:
        self.checkpoint_dir = checkpoint_dir
        mf_path = os.path.join(checkpoint_dir, "tfq3_manifest.json")
        self.manifest = json.load(open(mf_path, "r", encoding="utf-8")) if os.path.isfile(mf_path) else {}
        tables_name = self.manifest.get("tables", "pw_v3_tables.bin")
        if not str(tables_name).endswith("pw_v3_tables.bin") and os.path.isfile(
            os.path.join(checkpoint_dir, "pw_v3_tables.bin")
        ):
            tables_name = "pw_v3_tables.bin"
        tables_path = os.path.join(checkpoint_dir, tables_name)
        with open(tables_path, "rb") as f:
            self.tables_bytes = f.read()
        if len(self.tables_bytes) != 16384:
            raise ValueError(f"Expected 16384-byte {tables_path}, got {len(self.tables_bytes)}")
        self.tables_dict = _parse_pwv3_tables_bytes(self.tables_bytes)

        shard_names = [s for s in self.manifest.get("tfq_shards", []) if str(s).endswith(".tffpw")]
        if shard_names:
            shard_paths = [os.path.join(checkpoint_dir, s) for s in shard_names]
        else:
            shard_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "*.tffpw")))

        self.index: dict[str, dict[str, Any]] = {}
        for sp in shard_paths:
            with open(sp, "rb") as f:
                hdr = struct.unpack("<IIIIQQQQQQ", f.read(64))
                magic, ver, nt, _, off_meta, off_scales, off_weights, _, _, _ = hdr
                if magic != MAGIC_TFPN or ver != FORMAT_VER:
                    continue
                f.seek(off_meta)
                for _ in range(nt):
                    rec = f.read(128)
                    nm = rec[:80].split(b"\x00", 1)[0].decode("utf-8")
                    R, C, bits, sc_off, w_off, w_cnt, snr_db, orig_r, orig_c, flags = struct.unpack("<IIIIQQfIII", rec[80:])
                    self.index[nm] = {
                        "shard": sp,
                        "off_scales": off_scales,
                        "off_weights": off_weights,
                        "R": R,
                        "C": C,
                        "bits": bits,
                        "sc_off": sc_off,
                        "w_off": w_off,
                        "w_cnt": w_cnt,
                        "orig_r": orig_r if orig_r else R,
                        "orig_c": orig_c if orig_c else C,
                        "flags": flags,
                    }

    def has_tensor(self, name: str) -> bool:
        return _strip_weight_suffix(name) in self.index

    def read_tensor_slice(
        self,
        name: str,
        row_off: int = 0,
        pad_r: Optional[int] = None,
        bc_start: int = 0,
        n_bc_read: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Read packed `(words_u32, scales_f16, meta)` for a full tensor or TP slice."""
        key = _strip_weight_suffix(name)
        meta = self.index[key]
        R, C, bits = meta["R"], meta["C"], meta["bits"]
        n_bc_total = C // 128
        wpb = bits * 2
        if pad_r is None:
            pad_r = R - row_off
        if n_bc_read is None:
            n_bc_read = n_bc_total - bc_start
        if (row_off % 128) != 0 or (pad_r % 128) != 0 or row_off + pad_r > R:
            raise ValueError(f"{key}: invalid row slice [{row_off}:{row_off + pad_r}] for R={R}")
        if bc_start < 0 or bc_start + n_bc_read > n_bc_total:
            raise ValueError(f"{key}: invalid col-block slice [{bc_start}:{bc_start + n_bc_read}] for n_bc={n_bc_total}")

        with open(meta["shard"], "rb") as f:
            f.seek(meta["off_scales"] + meta["sc_off"] * 2)
            scales_all = np.frombuffer(f.read((R // 128) * n_bc_total * 2), dtype=np.float16).reshape(R // 128, n_bc_total)
            f.seek(meta["off_weights"] + meta["w_off"] * 4)
            words_all = np.frombuffer(f.read(meta["w_cnt"] * 4), dtype=np.uint32).reshape(R // 16, n_bc_total, wpb * 32)

        rb0, rb1 = row_off // 128, (row_off + pad_r) // 128
        st0, st1 = row_off // 16, (row_off + pad_r) // 16
        scales_sub = np.ascontiguousarray(scales_all[rb0:rb1, bc_start : bc_start + n_bc_read])
        words_sub = np.ascontiguousarray(words_all[st0:st1, bc_start : bc_start + n_bc_read, :]).reshape(-1)
        return words_sub, scales_sub.reshape(-1), meta


class Tfq4CheckpointReader:
    """
    Random-access reader for TFQ4 (`TFP4` v4 `.tffp4`) checkpoints and `pw_v4_e8p_tables.bin`.
    Supports both 2.001 bpw (`b_bits=2`, `table_kind=0/1/2`, 8 words/128 cols) and
    2.251 bpw (`b_bits=225`, `table_kind=3` True QuIP# E8P-1024, 9 words/128 cols).
    """

    def __init__(self, checkpoint_dir: str, *, verify_checksum: bool = True) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.manifest: dict[str, Any] = {}
        for mf_name in ("tfq4_manifest.json", "tfq3_manifest.json"):
            mf_path = os.path.join(checkpoint_dir, mf_name)
            if os.path.isfile(mf_path):
                with open(mf_path, "r", encoding="utf-8") as f:
                    self.manifest = json.load(f)
                break

        tables_path = os.path.join(checkpoint_dir, "pw_v4_e8p_tables.bin")
        if not os.path.isfile(tables_path):
            cand = self.manifest.get("tables_v4") or self.manifest.get("tables")
            if cand and os.path.isfile(os.path.join(checkpoint_dir, str(cand))):
                tables_path = os.path.join(checkpoint_dir, str(cand))
        self.tables_lut, self.table_kind, self.tables_bytes = _tfq_torch.load_pwv4_tables_auto(tables_path)

        shard_names = [s for s in self.manifest.get("tfq_shards", []) if str(s).endswith(".tffp4")]
        if shard_names:
            shard_paths = [os.path.join(checkpoint_dir, s) for s in shard_names]
        else:
            shard_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "*.tffp4")))

        self.index: dict[str, dict[str, Any]] = {}
        for sp in shard_paths:
            with open(sp, "rb") as f:
                hdr = struct.unpack("<IIIIQQQQQQ", f.read(64))
                magic, ver, nt, pad0, off_meta, off_scales, off_weights, tot_sc, tot_w, pad1 = hdr
                if magic != MAGIC_TFP4 or ver != TFQ4_VER:
                    continue
                if verify_checksum:
                    tfq3_ref.verify_tfpn_like_checksum(
                        f, off_meta, nt * 128, off_scales, tot_sc * 2, off_weights, tot_w * 4, pad0, pad1, f"TFP4 ({sp})"
                    )
                f.seek(off_meta)
                for _ in range(nt):
                    rec = f.read(128)
                    nm = rec[:80].split(b"\x00", 1)[0].decode("utf-8")
                    R, C, bits, sc_off, w_off, w_cnt, snr_db, orig_r, orig_c, flags = struct.unpack("<IIIIQQfIII", rec[80:])
                    if bits not in (2, 225):
                        raise ValueError(f"Unsupported TFP4 b_bits={bits} for '{nm}' in {sp}")
                    if bits == 225 and self.table_kind != _tfq_torch.PWV4_TABLE_TRUE_E8P_1024:
                        raise ValueError(
                            f"TFP4 tensor '{nm}' has b_bits=225 but PWV4 table_kind={self.table_kind} (expected 3)"
                        )
                    eff_kind = (
                        _tfq_torch.PWV4_TABLE_TRUE_E8P_1024
                        if bits == 225
                        else (_tfq_torch.PWV4_TABLE_TRUE_E8P if self.table_kind == 3 else self.table_kind)
                    )
                    self.index[nm] = {
                        "shard": sp,
                        "off_scales": off_scales,
                        "off_weights": off_weights,
                        "R": R,
                        "C": C,
                        "bits": bits,
                        "table_kind": eff_kind,
                        "sc_off": sc_off,
                        "w_off": w_off,
                        "w_cnt": w_cnt,
                        "orig_r": orig_r if orig_r else R,
                        "orig_c": orig_c if orig_c else C,
                        "flags": flags,
                    }

    def has_tensor(self, name: str) -> bool:
        return _strip_weight_suffix(name) in self.index

    def read_tensor_slice(
        self,
        name: str,
        row_off: int = 0,
        pad_r: Optional[int] = None,
        bc_start: int = 0,
        n_bc_read: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Read packed `(words_u32, scales_f16, meta)` for a full TFQ4 tensor or Col/Row TP slice."""
        key = _strip_weight_suffix(name)
        meta = self.index[key]
        R, C, bits = meta["R"], meta["C"], meta["bits"]
        n_bc_total = C // 128
        wpb_row = 9 if bits == 225 else 8
        if pad_r is None:
            pad_r = R - row_off
        if n_bc_read is None:
            n_bc_read = n_bc_total - bc_start
        if (row_off % 128) != 0 or (pad_r % 128) != 0 or row_off + pad_r > R:
            raise ValueError(f"{key}: invalid row slice [{row_off}:{row_off + pad_r}] for R={R}")
        if bc_start < 0 or bc_start + n_bc_read > n_bc_total:
            raise ValueError(f"{key}: invalid col-block slice [{bc_start}:{bc_start + n_bc_read}] for n_bc={n_bc_total}")

        with open(meta["shard"], "rb") as f:
            f.seek(meta["off_scales"] + meta["sc_off"] * 2)
            scales_all = np.frombuffer(f.read((R // 128) * n_bc_total * 2), dtype=np.float16).reshape(R // 128, n_bc_total)
            f.seek(meta["off_weights"] + meta["w_off"] * 4)
            words_all = np.frombuffer(f.read(meta["w_cnt"] * 4), dtype=np.uint32).reshape(R, n_bc_total, wpb_row)

        rb0, rb1 = row_off // 128, (row_off + pad_r) // 128
        scales_sub = np.ascontiguousarray(scales_all[rb0:rb1, bc_start : bc_start + n_bc_read])
        words_sub = np.ascontiguousarray(words_all[row_off : row_off + pad_r, bc_start : bc_start + n_bc_read, :]).reshape(-1)
        return words_sub, scales_sub.reshape(-1), meta


class Tfq3LinearMethod(LinearMethodBase):
    """vLLM LinearMethod supporting per-tensor TFQ3 bits (3/4/5b), merged projections, and Col/Row TP > 1."""

    def __init__(self, quant_config: "Tfq3Config", prefix: str = "") -> None:
        self.quant_config = quant_config
        self.prefix = prefix

    def _resolve_partition_names(
        self,
        prefix: str,
        output_partition_sizes: list[int],
        explicit_names: Optional[list[str]] = None,
    ) -> list[str]:
        return _resolve_partition_names(prefix, output_partition_sizes, explicit_names)

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        tp_rank, tp_size = _get_tp_rank_size(extra_weight_attrs)
        explicit_names = extra_weight_attrs.pop("partition_names", None)
        explicit_row_offs = extra_weight_attrs.pop("partition_row_offs", None)
        explicit_k_off = extra_weight_attrs.pop("k_sign_off", None)
        extra_weight_attrs.pop("tp_rank", None)
        extra_weight_attrs.pop("tp_size", None)

        prefix = getattr(layer, "prefix", "") or self.prefix
        part_names = self._resolve_partition_names(prefix, output_partition_sizes, explicit_names)

        # Row-parallel detection (input_size_per_partition < input_size)
        is_row_tp = input_size_per_partition < input_size
        if explicit_k_off is not None:
            k_sign_off = int(explicit_k_off)
        elif is_row_tp:
            k_sign_off = tp_rank * input_size_per_partition
        else:
            k_sign_off = 0
        if (k_sign_off % 128) != 0:
            raise ValueError(f"TFQ3 Row-TP requires k_sign_off to be a multiple of 128, got {k_sign_off}")

        pad_c = _align128(input_size_per_partition)
        n_bc_local = pad_c // 128
        bc0_abs = k_sign_off // 128
        bc_store_start = max(0, bc0_abs - 1) if k_sign_off > 0 else 0
        bc0_buf = bc0_abs - bc_store_start  # 0 or 1 (1 leading trellis-resume block)
        n_bc_buf = bc0_buf + n_bc_local

        # Column-parallel detection (sum(output_partition_sizes) < output_size)
        is_col_tp = sum(output_partition_sizes) < output_size

        part_bits: list[int] = []
        part_orig_r: list[int] = []
        part_pad_r: list[int] = []
        part_row_offs: list[int] = []
        part_w_offs: list[int] = []
        part_w_cnts: list[int] = []
        part_s_offs: list[int] = []
        part_s_cnts: list[int] = []

        w_cursor = 0
        s_cursor = 0
        for idx, (pname, orig_r_p) in enumerate(zip(part_names, output_partition_sizes)):
            bits_p = self.quant_config.get_tensor_bits(pname)
            pad_r_p = _align128(orig_r_p)
            if explicit_row_offs is not None:
                row_off_p = int(explicit_row_offs[idx])
            elif is_col_tp:
                row_off_p = tp_rank * orig_r_p
            else:
                row_off_p = 0
            if (row_off_p % 128) != 0:
                raise ValueError(
                    f"TFQ3 Col-TP requires per-partition row_off multiple of 128, got {row_off_p} for {pname}"
                )
            nw_p = (pad_r_p // 16) * n_bc_buf * (bits_p * 2) * 32
            ns_p = (pad_r_p // 128) * n_bc_buf

            part_bits.append(bits_p)
            part_orig_r.append(int(orig_r_p))
            part_pad_r.append(int(pad_r_p))
            part_row_offs.append(int(row_off_p))
            part_w_offs.append(w_cursor)
            part_w_cnts.append(nw_p)
            part_s_offs.append(s_cursor)
            part_s_cnts.append(ns_p)
            w_cursor += nw_p
            s_cursor += ns_p

        qweight = Parameter(torch.empty(w_cursor, dtype=torch.int32), requires_grad=False)
        scales = Parameter(torch.empty(s_cursor, dtype=torch.float16), requires_grad=False)
        tables = Parameter(torch.empty(16384, dtype=torch.uint8), requires_grad=False)

        def _weight_loader(param: Parameter, loaded_weight: torch.Tensor, loaded_shard_id: Optional[Any] = None) -> None:
            self.load_parameter(layer, param, loaded_weight, loaded_shard_id=loaded_shard_id)

        loader_attrs = dict(extra_weight_attrs)
        loader_attrs.setdefault("weight_loader", _weight_loader)
        set_weight_attrs(qweight, {"packed_dim": 0, "tfq3_param_kind": "words", **loader_attrs})
        set_weight_attrs(scales, {"tfq3_param_kind": "scales", **loader_attrs})
        set_weight_attrs(tables, {"tfq3_param_kind": "tables", **loader_attrs})

        layer.register_parameter("tfq3_words", qweight)
        layer.register_parameter("tfq3_scales", scales)
        layer.register_parameter("tfq3_tables", tables)

        # Per-partition metadata (exact padding, bit width, offsets, and TP signs per sub-tensor)
        layer.tfq3_partition_names = part_names
        layer.tfq3_partition_bits = part_bits
        layer.tfq3_partition_orig_r = part_orig_r
        layer.tfq3_partition_pad_r = part_pad_r
        layer.tfq3_partition_row_offs = part_row_offs
        layer.tfq3_partition_word_offs = part_w_offs
        layer.tfq3_partition_word_counts = part_w_cnts
        layer.tfq3_partition_scale_offs = part_s_offs
        layer.tfq3_partition_scale_counts = part_s_cnts
        layer.tfq3_k_sign_off = k_sign_off
        layer.tfq3_bc_store_start = bc_store_start
        layer.tfq3_bc0_buf = bc0_buf
        layer.tfq3_n_bc_buf = n_bc_buf
        layer.tfq3_orig_c = int(input_size_per_partition)
        layer.tfq3_tp_rank = tp_rank
        layer.tfq3_tp_size = tp_size
        layer._tfq3_tables_cache = None

        # Backwards-compatible summary attributes
        layer.tfq3_pad_r = sum(part_pad_r)
        layer.tfq3_pad_c = pad_c
        layer.tfq3_orig_r = sum(part_orig_r)

    def _resolve_shard_index(self, layer: torch.nn.Module, loaded_shard_id: Optional[Any]) -> int:
        if loaded_shard_id is None:
            return 0
        if isinstance(loaded_shard_id, int):
            return loaded_shard_id
        s = str(loaded_shard_id)
        alias = {"q": 0, "k": 1, "v": 2, "gate": 0, "up": 1}
        if s in alias and alias[s] < len(layer.tfq3_partition_names):
            return alias[s]
        clean = _strip_weight_suffix(s)
        for idx, pname in enumerate(layer.tfq3_partition_names):
            if pname == clean or pname.endswith("." + clean):
                return idx
        raise KeyError(f"Unknown TFQ3 shard id {loaded_shard_id!r} for partitions {layer.tfq3_partition_names}")

    def load_parameter(
        self,
        layer: torch.nn.Module,
        param: Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: Optional[Any] = None,
    ) -> None:
        kind = getattr(param, "tfq3_param_kind", "")
        if kind == "tables":
            assert loaded_weight.numel() == 16384
            param.data.copy_(loaded_weight.view(torch.uint8))
            layer._tfq3_tables_cache = None
            return
        idx = self._resolve_shard_index(layer, loaded_shard_id)
        if kind == "words":
            off = layer.tfq3_partition_word_offs[idx]
            cnt = layer.tfq3_partition_word_counts[idx]
            assert loaded_weight.numel() == cnt, f"words size mismatch for part {idx}: {loaded_weight.numel()} vs {cnt}"
            param.data[off : off + cnt].copy_(loaded_weight.view(torch.int32).reshape(-1))
        elif kind == "scales":
            off = layer.tfq3_partition_scale_offs[idx]
            cnt = layer.tfq3_partition_scale_counts[idx]
            assert loaded_weight.numel() == cnt, f"scales size mismatch for part {idx}: {loaded_weight.numel()} vs {cnt}"
            param.data[off : off + cnt].copy_(loaded_weight.to(torch.float16).reshape(-1))
        else:
            param.data.copy_(loaded_weight)

    def load_layer_from_reader(self, layer: torch.nn.Module, reader: Tfq3CheckpointReader) -> None:
        """Populate `tfq3_words`, `tfq3_scales`, and `tfq3_tables` directly from `.tffpw` shards."""
        tbl_t = torch.from_numpy(np.frombuffer(reader.tables_bytes, dtype=np.uint8).copy())
        layer.tfq3_tables.data.copy_(tbl_t)
        layer._tfq3_tables_cache = reader.tables_dict

        for idx, pname in enumerate(layer.tfq3_partition_names):
            w_u32, sc_f16, meta = reader.read_tensor_slice(
                pname,
                row_off=layer.tfq3_partition_row_offs[idx],
                pad_r=layer.tfq3_partition_pad_r[idx],
                bc_start=layer.tfq3_bc_store_start,
                n_bc_read=layer.tfq3_n_bc_buf,
            )
            if meta["bits"] != layer.tfq3_partition_bits[idx]:
                raise ValueError(
                    f"Bit-width mismatch for {pname}: shard has {meta['bits']}b, layer allocated {layer.tfq3_partition_bits[idx]}b"
                )
            w_off = layer.tfq3_partition_word_offs[idx]
            w_cnt = layer.tfq3_partition_word_counts[idx]
            s_off = layer.tfq3_partition_scale_offs[idx]
            s_cnt = layer.tfq3_partition_scale_counts[idx]
            layer.tfq3_words.data[w_off : w_off + w_cnt].copy_(torch.from_numpy(w_u32.view(np.int32).copy()))
            layer.tfq3_scales.data[s_off : s_off + s_cnt].copy_(torch.from_numpy(sc_f16.copy()))

    def _get_tables_dict(self, layer: torch.nn.Module) -> dict:
        if getattr(layer, "_tfq3_tables_cache", None) is None:
            raw = layer.tfq3_tables.detach().cpu().numpy().tobytes()
            layer._tfq3_tables_cache = _parse_pwv3_tables_bytes(raw)
        return layer._tfq3_tables_cache

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ext = _get_cuda_ext() if x.is_cuda else None
        mps = _get_mps_mod() if (ext is None and x.device.type == "mps") else None
        n_parts = len(layer.tfq3_partition_bits)
        pad_c = layer.tfq3_pad_c
        n_bc_local = pad_c // 128
        bc0_buf = layer.tfq3_bc0_buf
        n_bc_buf = layer.tfq3_n_bc_buf
        k_sign_off = layer.tfq3_k_sign_off
        orig_c = x.shape[-1]

        outs: list[torch.Tensor] = []
        for idx in range(n_parts):
            bits_p = layer.tfq3_partition_bits[idx]
            pad_r_p = layer.tfq3_partition_pad_r[idx]
            orig_r_p = layer.tfq3_partition_orig_r[idx]
            row_off_p = layer.tfq3_partition_row_offs[idx]
            w_off = layer.tfq3_partition_word_offs[idx]
            w_cnt = layer.tfq3_partition_word_counts[idx]
            s_off = layer.tfq3_partition_scale_offs[idx]
            s_cnt = layer.tfq3_partition_scale_counts[idx]

            w_p = layer.tfq3_words[w_off : w_off + w_cnt]
            s_p = layer.tfq3_scales[s_off : s_off + s_cnt]

            if ext is not None:
                y_p = ext.tfq3_linear(
                    x.to(torch.bfloat16),
                    w_p,
                    s_p,
                    layer.tfq3_tables,
                    bits_p,
                    pad_r_p,
                    pad_c,
                    orig_r_p,
                    row_off_p,
                    k_sign_off,
                    bc0_buf,
                    n_bc_buf,
                )
            elif mps is not None:
                y_p = mps.tfq3_linear_mps(
                    x,
                    w_p,
                    s_p,
                    layer.tfq3_tables,
                    bits_p,
                    pad_r_p,
                    pad_c,
                    orig_r_p,
                    row_off_p,
                    k_sign_off,
                    bc0_buf,
                    n_bc_buf,
                )
            else:
                tbl = self._get_tables_dict(layer)
                w_np = w_p.detach().cpu().numpy().view(np.uint32)
                s_np = s_p.detach().cpu().numpy().reshape(pad_r_p // 128, n_bc_buf)
                W_hat = tfq3_ref.decode_tfpn_fast_np(
                    w_np,
                    s_np,
                    pad_r_p,
                    n_bc_buf * 128,
                    bits_p,
                    tbl,
                    r_off=row_off_p,
                    bc0=bc0_buf,
                    bc1=bc0_buf + n_bc_local,
                    c_sign_off=k_sign_off,
                )[:orig_r_p, :orig_c]
                w_t = torch.from_numpy(W_hat).to(device=x.device, dtype=x.dtype)
                y_p = F.linear(x, w_t)
            outs.append(y_p)

        y = outs[0] if n_parts == 1 else torch.cat(outs, dim=-1)
        if bias is not None:
            y = y + bias.to(device=y.device, dtype=y.dtype)
        return y.to(dtype=x.dtype)


class Tfq4LinearMethod(LinearMethodBase):
    """
    vLLM LinearMethod for TFQ4 (`TFP4` v4, 2.001 bpw `table_kind=0/1/2` and 2.251 bpw
    `table_kind=3` True QuIP# E8P-1024), supporting merged projections and stateless Col/Row TP.
    """

    def __init__(self, quant_config: "Tfq3Config", prefix: str = "") -> None:
        self.quant_config = quant_config
        self.prefix = prefix

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        tp_rank, tp_size = _get_tp_rank_size(extra_weight_attrs)
        explicit_names = extra_weight_attrs.pop("partition_names", None)
        explicit_row_offs = extra_weight_attrs.pop("partition_row_offs", None)
        explicit_k_off = extra_weight_attrs.pop("k_sign_off", None)
        extra_weight_attrs.pop("tp_rank", None)
        extra_weight_attrs.pop("tp_size", None)

        prefix = getattr(layer, "prefix", "") or self.prefix
        part_names = _resolve_partition_names(prefix, output_partition_sizes, explicit_names)

        is_row_tp = input_size_per_partition < input_size
        if explicit_k_off is not None:
            k_sign_off = int(explicit_k_off)
        elif is_row_tp:
            k_sign_off = tp_rank * input_size_per_partition
        else:
            k_sign_off = 0
        if (k_sign_off % 128) != 0:
            raise ValueError(f"TFQ4 Row-TP requires k_sign_off to be a multiple of 128, got {k_sign_off}")

        pad_c = _align128(input_size_per_partition)
        n_bc_local = pad_c // 128
        bc_start = k_sign_off // 128
        is_col_tp = sum(output_partition_sizes) < output_size

        base_tk = int(self.quant_config.table_kind)
        part_bits: list[int] = []
        part_kinds: list[int] = []
        part_orig_r: list[int] = []
        part_pad_r: list[int] = []
        part_row_offs: list[int] = []
        part_w_offs: list[int] = []
        part_w_cnts: list[int] = []
        part_s_offs: list[int] = []
        part_s_cnts: list[int] = []

        w_cursor = 0
        s_cursor = 0
        for idx, (pname, orig_r_p) in enumerate(zip(part_names, output_partition_sizes)):
            bits_p = self.quant_config.get_tensor_bits(pname)
            if bits_p not in (2, 225):
                bits_p = 225 if base_tk == 3 else 2
            tk_p = 3 if bits_p == 225 else (2 if base_tk == 3 else base_tk)
            pad_r_p = _align128(orig_r_p)
            if explicit_row_offs is not None:
                row_off_p = int(explicit_row_offs[idx])
            elif is_col_tp:
                row_off_p = tp_rank * orig_r_p
            else:
                row_off_p = 0
            if (row_off_p % 128) != 0:
                raise ValueError(
                    f"TFQ4 Col-TP requires per-partition row_off multiple of 128, got {row_off_p} for {pname}"
                )
            wpb_row = 9 if bits_p == 225 else 8
            nw_p = pad_r_p * n_bc_local * wpb_row
            ns_p = (pad_r_p // 128) * n_bc_local

            part_bits.append(bits_p)
            part_kinds.append(tk_p)
            part_orig_r.append(int(orig_r_p))
            part_pad_r.append(int(pad_r_p))
            part_row_offs.append(int(row_off_p))
            part_w_offs.append(w_cursor)
            part_w_cnts.append(nw_p)
            part_s_offs.append(s_cursor)
            part_s_cnts.append(ns_p)
            w_cursor += nw_p
            s_cursor += ns_p

        lut_rows = 1024 if (3 in part_kinds or base_tk == 3) else 256
        qweight = Parameter(torch.empty(w_cursor, dtype=torch.int32), requires_grad=False)
        scales = Parameter(torch.empty(s_cursor, dtype=torch.float16), requires_grad=False)
        e8p_lut = Parameter(torch.empty((lut_rows, 8), dtype=torch.float16), requires_grad=False)

        def _weight_loader(param: Parameter, loaded_weight: torch.Tensor, loaded_shard_id: Optional[Any] = None) -> None:
            self.load_parameter(layer, param, loaded_weight, loaded_shard_id=loaded_shard_id)

        loader_attrs = dict(extra_weight_attrs)
        loader_attrs.setdefault("weight_loader", _weight_loader)
        set_weight_attrs(qweight, {"packed_dim": 0, "tfq4_param_kind": "words", **loader_attrs})
        set_weight_attrs(scales, {"tfq4_param_kind": "scales", **loader_attrs})
        set_weight_attrs(e8p_lut, {"tfq4_param_kind": "lut", **loader_attrs})

        layer.register_parameter("tfq4_words", qweight)
        layer.register_parameter("tfq4_scales", scales)
        layer.register_parameter("tfq4_e8p_lut", e8p_lut)

        layer.tfq4_partition_names = part_names
        layer.tfq4_partition_bits = part_bits
        layer.tfq4_partition_table_kinds = part_kinds
        layer.tfq4_partition_orig_r = part_orig_r
        layer.tfq4_partition_pad_r = part_pad_r
        layer.tfq4_partition_row_offs = part_row_offs
        layer.tfq4_partition_word_offs = part_w_offs
        layer.tfq4_partition_word_counts = part_w_cnts
        layer.tfq4_partition_scale_offs = part_s_offs
        layer.tfq4_partition_scale_counts = part_s_cnts
        layer.tfq4_k_sign_off = k_sign_off
        layer.tfq4_bc_start = bc_start
        layer.tfq4_n_bc_local = n_bc_local
        layer.tfq4_orig_c = int(input_size_per_partition)
        layer.tfq4_pad_c = pad_c
        layer.tfq4_pad_r = sum(part_pad_r)
        layer.tfq4_orig_r = sum(part_orig_r)
        layer.tfq4_table_kind = max(part_kinds)
        layer.tfq4_tp_rank = tp_rank
        layer.tfq4_tp_size = tp_size

    def _resolve_shard_index(self, layer: torch.nn.Module, loaded_shard_id: Optional[Any]) -> int:
        if loaded_shard_id is None:
            return 0
        if isinstance(loaded_shard_id, int):
            return loaded_shard_id
        s = str(loaded_shard_id)
        alias = {"q": 0, "k": 1, "v": 2, "gate": 0, "up": 1}
        if s in alias and alias[s] < len(layer.tfq4_partition_names):
            return alias[s]
        clean = _strip_weight_suffix(s)
        for idx, pname in enumerate(layer.tfq4_partition_names):
            if pname == clean or pname.endswith("." + clean):
                return idx
        raise KeyError(f"Unknown TFQ4 shard id {loaded_shard_id!r} for partitions {layer.tfq4_partition_names}")

    def load_parameter(
        self,
        layer: torch.nn.Module,
        param: Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: Optional[Any] = None,
    ) -> None:
        kind = getattr(param, "tfq4_param_kind", "")
        if kind == "lut":
            lut_t = loaded_weight.to(torch.float16)
            if lut_t.shape[0] > param.shape[0]:
                lut_t = lut_t[: param.shape[0]]
            param.data.copy_(lut_t)
            return
        idx = self._resolve_shard_index(layer, loaded_shard_id)
        if kind == "words":
            off = layer.tfq4_partition_word_offs[idx]
            cnt = layer.tfq4_partition_word_counts[idx]
            assert loaded_weight.numel() == cnt, f"TFQ4 words size mismatch for part {idx}: {loaded_weight.numel()} vs {cnt}"
            param.data[off : off + cnt].copy_(loaded_weight.view(torch.int32).reshape(-1))
        elif kind == "scales":
            off = layer.tfq4_partition_scale_offs[idx]
            cnt = layer.tfq4_partition_scale_counts[idx]
            assert loaded_weight.numel() == cnt, f"TFQ4 scales size mismatch for part {idx}: {loaded_weight.numel()} vs {cnt}"
            param.data[off : off + cnt].copy_(loaded_weight.to(torch.float16).reshape(-1))
        else:
            param.data.copy_(loaded_weight)

    def load_layer_from_reader(self, layer: torch.nn.Module, reader: Tfq4CheckpointReader) -> None:
        """Populate `tfq4_words`, `tfq4_scales`, and `tfq4_e8p_lut` directly from `.tffp4` shards."""
        lut_np = reader.tables_lut
        if lut_np.shape[0] > layer.tfq4_e8p_lut.shape[0]:
            lut_np = lut_np[: layer.tfq4_e8p_lut.shape[0]]
        layer.tfq4_e8p_lut.data.copy_(torch.from_numpy(lut_np.astype(np.float16).copy()))

        for idx, pname in enumerate(layer.tfq4_partition_names):
            w_u32, sc_f16, meta = reader.read_tensor_slice(
                pname,
                row_off=layer.tfq4_partition_row_offs[idx],
                pad_r=layer.tfq4_partition_pad_r[idx],
                bc_start=layer.tfq4_bc_start,
                n_bc_read=layer.tfq4_n_bc_local,
            )
            if meta["bits"] != layer.tfq4_partition_bits[idx]:
                raise ValueError(
                    f"TFQ4 bit-width mismatch for {pname}: shard has {meta['bits']}, layer allocated {layer.tfq4_partition_bits[idx]}"
                )
            layer.tfq4_partition_table_kinds[idx] = int(meta["table_kind"])
            w_off = layer.tfq4_partition_word_offs[idx]
            w_cnt = layer.tfq4_partition_word_counts[idx]
            s_off = layer.tfq4_partition_scale_offs[idx]
            s_cnt = layer.tfq4_partition_scale_counts[idx]
            layer.tfq4_words.data[w_off : w_off + w_cnt].copy_(torch.from_numpy(w_u32.view(np.int32).copy()))
            layer.tfq4_scales.data[s_off : s_off + s_cnt].copy_(torch.from_numpy(sc_f16.copy()))

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ext = _get_cuda_ext() if x.is_cuda else None
        mps = _get_mps_mod() if (ext is None and x.device.type == "mps") else None
        max_cuda_tk = int(getattr(ext, "TFQ4_MAX_TABLE_KIND", 2)) if ext is not None else -1

        n_parts = len(layer.tfq4_partition_bits)
        pad_c = layer.tfq4_pad_c
        k_sign_off = layer.tfq4_k_sign_off
        orig_c = x.shape[-1]

        outs: list[torch.Tensor] = []
        for idx in range(n_parts):
            tk_p = layer.tfq4_partition_table_kinds[idx]
            pad_r_p = layer.tfq4_partition_pad_r[idx]
            orig_r_p = layer.tfq4_partition_orig_r[idx]
            row_off_p = layer.tfq4_partition_row_offs[idx]
            w_off = layer.tfq4_partition_word_offs[idx]
            w_cnt = layer.tfq4_partition_word_counts[idx]
            s_off = layer.tfq4_partition_scale_offs[idx]
            s_cnt = layer.tfq4_partition_scale_counts[idx]

            w_p = layer.tfq4_words[w_off : w_off + w_cnt]
            s_p = layer.tfq4_scales[s_off : s_off + s_cnt]
            lut_p = layer.tfq4_e8p_lut if tk_p == 3 else layer.tfq4_e8p_lut[:256]

            if ext is not None and hasattr(ext, "tfq4_linear") and tk_p <= max_cuda_tk:
                y_p = ext.tfq4_linear(
                    x.to(torch.bfloat16),
                    w_p,
                    s_p,
                    lut_p,
                    tk_p,
                    pad_r_p,
                    pad_c,
                    orig_r_p,
                    row_off_p,
                    k_sign_off,
                )
            elif mps is not None:
                y_p = mps.tfq4_linear_mps(
                    x,
                    w_p,
                    s_p,
                    lut_p,
                    tk_p,
                    pad_r_p,
                    pad_c,
                    orig_r_p,
                    row_off_p,
                    k_sign_off,
                )
            else:
                w_np = w_p.detach().cpu().numpy().view(np.uint32)
                s_np = s_p.detach().cpu().numpy()
                lut_np = lut_p.detach().cpu().numpy().astype(np.float32)
                W_hat = _tfq_torch.decode_tfq4_words_np(
                    w_np,
                    s_np,
                    pad_r_p,
                    pad_c,
                    lut_np,
                    tk_p,
                    row_off=row_off_p,
                    k_sign_off=k_sign_off,
                )[:orig_r_p, :orig_c]
                w_t = torch.from_numpy(W_hat).to(device=x.device, dtype=x.dtype)
                y_p = F.linear(x, w_t)
            outs.append(y_p)

        y = outs[0] if n_parts == 1 else torch.cat(outs, dim=-1)
        if bias is not None:
            y = y + bias.to(device=y.device, dtype=y.dtype)
        return y.to(dtype=x.dtype)


class Tfq3MoeMethod(FusedMoEMethodBase):
    """
    vLLM FusedMoE method for `TFPW` v3 and hybrid TFQ3+TFQ4 MoE layers (`b_bits in {2, 225, 3, 4, 5}`).

    Wraps `integrations/pytorch/tfq_moe.py:TfpwLayer` and dispatches `{3, 4, 5}`-bit routed
    experts to `tfq3_cuda_ext.tfq3_moe` on CUDA (supporting standard SwiGLU, GLM `swiglu_limit`,
    and Inkling-2 `gate_softcap` / `up_softcap`) with seamless TFQ4 CUDA/MPS/CPU fallback for
    `{2, 225}`-bit experts.
    """

    def __init__(
        self,
        quant_config: "Tfq3Config",
        prefix: str = "",
        layer_idx: Optional[int] = None,
    ) -> None:
        self.quant_config = quant_config
        self.prefix = prefix
        self.layer_idx = _infer_layer_idx(prefix, layer_idx)

    def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> Optional[Any]:
        return None

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        layer_idx = extra_weight_attrs.pop("layer_idx", None)
        if layer_idx is None:
            layer_idx = getattr(layer, "layer_idx", None)
        if layer_idx is None:
            layer_idx = _infer_layer_idx(getattr(layer, "prefix", "") or self.prefix, self.layer_idx)
        layer.tfq3_moe_layer_idx = layer_idx
        layer.tfq3_moe_num_experts = int(num_experts)
        layer.tfq3_moe_hidden_size = int(hidden_size)
        layer.tfq3_moe_inter = int(intermediate_size_per_partition)
        layer.tfq3_moe_dtype = params_dtype
        layer.tfq3_moe_layer = None
        layer.tfq3_swiglu_limit = extra_weight_attrs.pop("swiglu_limit", self.quant_config.swiglu_limit)
        layer.tfq3_gate_softcap = extra_weight_attrs.pop("gate_softcap", self.quant_config.gate_softcap)
        layer.tfq3_up_softcap = extra_weight_attrs.pop("up_softcap", self.quant_config.up_softcap)

    def attach_tfpw_layer(self, layer: torch.nn.Module, tfpw_layer: _tfq_moe.TfpwLayer) -> None:
        if hasattr(layer, "tfq3_moe_num_experts") and tfpw_layer.num_experts != layer.tfq3_moe_num_experts:
            raise ValueError(
                f"MoE expert count mismatch on layer {tfpw_layer.layer_idx}: "
                f"shard has {tfpw_layer.num_experts}, module expected {layer.tfq3_moe_num_experts}"
            )
        if hasattr(layer, "tfq3_moe_hidden_size") and tfpw_layer.hidden_size != layer.tfq3_moe_hidden_size:
            raise ValueError(
                f"MoE hidden_size mismatch on layer {tfpw_layer.layer_idx}: "
                f"shard has {tfpw_layer.hidden_size}, module expected {layer.tfq3_moe_hidden_size}"
            )
        if hasattr(layer, "tfq3_moe_inter") and tfpw_layer.moe_inter != layer.tfq3_moe_inter:
            raise ValueError(
                f"MoE moe_inter mismatch on layer {tfpw_layer.layer_idx}: "
                f"shard has {tfpw_layer.moe_inter}, module expected {layer.tfq3_moe_inter}"
            )
        layer.tfq3_moe_layer = tfpw_layer

    def load_layer_from_checkpoint(
        self,
        layer: torch.nn.Module,
        ckpt: _tfq_moe.TfpwCheckpoint,
    ) -> None:
        lidx = getattr(layer, "tfq3_moe_layer_idx", None)
        if lidx is None:
            lidx = _infer_layer_idx(getattr(layer, "prefix", "") or self.prefix, self.layer_idx)
        if lidx is None or lidx not in ckpt.layers:
            raise KeyError(
                f"TFPW layer index {lidx!r} (prefix={getattr(layer, 'prefix', self.prefix)!r}) "
                f"not found in checkpoint layers {ckpt.layer_indices}"
            )
        self.attach_tfpw_layer(layer, ckpt.layers[lidx])

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: Optional[torch.Tensor] = None,
        top_k: Optional[int] = None,
        renormalize: bool = True,
        topk_weights: Optional[torch.Tensor] = None,
        topk_ids: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        tfpw: Optional[_tfq_moe.TfpwLayer] = getattr(layer, "tfq3_moe_layer", None)
        if tfpw is None:
            raise RuntimeError("Tfq3MoeMethod.apply() called before loading TFPW layer weights")

        orig_shape = list(x.shape)
        x2d = x.reshape(-1, x.shape[-1])
        if topk_ids is None or topk_weights is None:
            if router_logits is None or top_k is None:
                raise ValueError("Tfq3MoeMethod.apply() requires either (topk_ids, topk_weights) or (router_logits, top_k)")
            rl = router_logits.reshape(-1, router_logits.shape[-1]).to(torch.float32)
            probs = torch.softmax(rl, dim=-1)
            topk_weights, topk_ids = torch.topk(probs, k=int(top_k), dim=-1)
            if renormalize:
                topk_weights = topk_weights / torch.clamp(topk_weights.sum(dim=-1, keepdim=True), min=1e-12)
        else:
            topk_ids = topk_ids.reshape(x2d.shape[0], -1)
            topk_weights = topk_weights.reshape(x2d.shape[0], -1)

        swiglu_limit = kwargs.get("swiglu_limit", getattr(layer, "tfq3_swiglu_limit", None))
        gate_softcap = kwargs.get("gate_softcap", getattr(layer, "tfq3_gate_softcap", None))
        up_softcap = kwargs.get("up_softcap", getattr(layer, "tfq3_up_softcap", None))

        y = tfpw.forward_routed(
            x2d,
            topk_ids,
            topk_weights,
            swiglu_limit=swiglu_limit,
            gate_softcap=gate_softcap,
            up_softcap=up_softcap,
        )
        return y.reshape(orig_shape)


class Tfq3Config(QuantizationConfig):
    """
    vLLM QuantizationConfig for TFQ3 (`TFPN` 3/4/5b), TFQ4 (`TFP4` 2.00/2.25 bpw,
    `table_kind=0..3`), and `TFPW` / hybrid MoE checkpoints.
    """

    def __init__(
        self,
        bits: int = 4,
        tensor_bits: Optional[dict[str, int]] = None,
        tensor_orig_shapes: Optional[dict[str, tuple[int, int]]] = None,
        tfq_shards: Optional[list[str]] = None,
        tables: str = "pw_v3_tables.bin",
        *,
        table_kind: int = 0,
        format_type: Optional[str] = None,
        tables_v4: Optional[str] = None,
        moe_layers: Optional[list[int]] = None,
        swiglu_limit: Optional[float] = None,
        gate_softcap: Optional[float] = None,
        up_softcap: Optional[float] = None,
    ) -> None:
        if bits not in (2, 225, 3, 4, 5):
            raise ValueError(f"TFQ3/TFQ4 supports bits in {{2, 225, 3, 4, 5}}, got {bits}")
        self.bits = int(bits)
        self.tensor_bits: dict[str, int] = {}
        if tensor_bits:
            for k, v in tensor_bits.items():
                iv = int(v)
                if iv not in (2, 225, 3, 4, 5):
                    raise ValueError(f"TFQ tensor '{k}' has invalid bits={iv}")
                self.tensor_bits[_strip_weight_suffix(k)] = iv
        self.tensor_orig_shapes: dict[str, tuple[int, int]] = dict(tensor_orig_shapes or {})
        self.tfq_shards: list[str] = list(tfq_shards or [])
        self.tables: str = tables
        self.tables_v4: Optional[str] = tables_v4
        self.table_kind: int = int(table_kind)
        if format_type is None:
            self.format_type = "tfq4" if self.bits in (2, 225) else "tfq3"
        else:
            self.format_type = str(format_type).lower()
        self.moe_layers: list[int] = sorted(int(i) for i in (moe_layers or []))
        self.swiglu_limit = swiglu_limit
        self.gate_softcap = gate_softcap
        self.up_softcap = up_softcap

    def get_name(self) -> str:
        return "tfq3"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80  # NVIDIA Ampere (sm_80) and newer

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return ["tfq3_manifest.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Tfq3Config":
        tensor_bits: dict[str, int] = {}
        tensor_shapes: dict[str, tuple[int, int]] = {}
        raw_tensors = config.get("tensors")
        if isinstance(raw_tensors, list):
            for entry in raw_tensors:
                if not isinstance(entry, dict) or "name" not in entry or "bits" not in entry:
                    continue
                nm = _strip_weight_suffix(str(entry["name"]))
                tensor_bits[nm] = int(entry["bits"])
                if "orig_shape" in entry and len(entry["orig_shape"]) == 2:
                    tensor_shapes[nm] = (int(entry["orig_shape"][0]), int(entry["orig_shape"][1]))
        elif isinstance(raw_tensors, dict):
            for nm, entry in raw_tensors.items():
                key = _strip_weight_suffix(str(nm))
                if isinstance(entry, dict) and "bits" in entry:
                    tensor_bits[key] = int(entry["bits"])
                elif isinstance(entry, int):
                    tensor_bits[key] = int(entry)

        fmt = str(config.get("format", "")).lower()
        shards = list(config.get("tfq_shards", []))
        tables_name = str(config.get("tables", "pw_v3_tables.bin"))
        tables_v4 = config.get("tables_v4")
        table_kind = int(config.get("table_kind", 0))

        is_tfq4 = (
            "tfq4" in fmt
            or tables_name.endswith("pw_v4_e8p_tables.bin")
            or (bool(shards) and all(str(s).endswith(".tffp4") for s in shards))
        )

        if "bits" in config and config["bits"] is not None:
            default_bits = int(config["bits"])
        elif tensor_bits:
            default_bits = Counter(tensor_bits.values()).most_common(1)[0][0]
        elif is_tfq4:
            default_bits = 225 if table_kind == 3 else 2
        else:
            default_bits = 4

        if is_tfq4 and tables_v4 is None:
            tables_v4 = tables_name if tables_name.endswith(".bin") else "pw_v4_e8p_tables.bin"

        return cls(
            bits=default_bits,
            tensor_bits=tensor_bits,
            tensor_orig_shapes=tensor_shapes,
            tfq_shards=shards,
            tables=tables_name,
            table_kind=table_kind,
            format_type="tfq4" if is_tfq4 else "tfq3",
            tables_v4=tables_v4,
            moe_layers=config.get("moe_layers"),
            swiglu_limit=config.get("swiglu_limit"),
            gate_softcap=config.get("gate_softcap"),
            up_softcap=config.get("up_softcap"),
        )

    @classmethod
    def from_checkpoint_dir(cls, checkpoint_dir: str) -> "Tfq3Config":
        """Auto-detect TFQ3, TFQ4 (2.00/2.25 bpw), hybrid, and TFPW MoE layout from a checkpoint directory."""
        base_cfg: dict[str, Any] = {}
        for mf_name in ("tfq3_manifest.json", "tfq4_manifest.json"):
            mf_path = os.path.join(checkpoint_dir, mf_name)
            if os.path.isfile(mf_path):
                with open(mf_path, "r", encoding="utf-8") as f:
                    base_cfg = json.load(f)
                break

        tensor_bits: dict[str, int] = {}
        tensor_shapes: dict[str, tuple[int, int]] = {}
        if base_cfg:
            parsed = cls.from_config(base_cfg)
            tensor_bits.update(parsed.tensor_bits)
            tensor_shapes.update(parsed.tensor_orig_shapes)

        pw3_path = os.path.join(checkpoint_dir, "pw_v3_tables.bin")
        pw4_path = os.path.join(checkpoint_dir, "pw_v4_e8p_tables.bin")
        table_kind = int(base_cfg.get("table_kind", 0))
        if os.path.isfile(pw4_path):
            _, table_kind, _ = _tfq_torch.load_pwv4_tables_auto(pw4_path)

        tffp4_shards = sorted(glob.glob(os.path.join(checkpoint_dir, "*.tffp4")))
        tffpw_shards = sorted(glob.glob(os.path.join(checkpoint_dir, "*.tffpw")))
        moe_layers: list[int] = []
        tfq_shards: list[str] = []

        for sp in tffpw_shards:
            tfq_shards.append(os.path.basename(sp))
            m_layer = _LAYER_FILE_RE.match(os.path.basename(sp))
            with open(sp, "rb") as f:
                raw_hdr = f.read(64)
                if len(raw_hdr) < 64:
                    continue
                magic = struct.unpack_from("<I", raw_hdr, 0)[0]
                if magic == MAGIC_TFPW:
                    layer_idx = struct.unpack_from("<I", raw_hdr, 8)[0]
                    moe_layers.append(int(layer_idx))
                elif magic == MAGIC_TFPN:
                    _, ver, nt, _, off_meta, _, _, _, _, _ = struct.unpack_from("<IIIIQQQQQQ", raw_hdr, 0)
                    if ver == FORMAT_VER:
                        f.seek(off_meta)
                        for _ in range(nt):
                            rec = f.read(128)
                            nm = rec[:80].split(b"\x00", 1)[0].decode("utf-8")
                            R, C, bits, _, _, _, _, orig_r, orig_c, _ = struct.unpack("<IIIIQQfIII", rec[80:])
                            tensor_bits[nm] = int(bits)
                            tensor_shapes[nm] = (int(orig_r or R), int(orig_c or C))
            if m_layer and int(m_layer.group(1)) not in moe_layers:
                pass

        for sp in tffp4_shards:
            tfq_shards.append(os.path.basename(sp))
            with open(sp, "rb") as f:
                raw_hdr = f.read(64)
                if len(raw_hdr) < 64:
                    continue
                magic, ver, nt, _, off_meta, _, _, _, _, _ = struct.unpack_from("<IIIIQQQQQQ", raw_hdr, 0)
                if magic != MAGIC_TFP4 or ver != TFQ4_VER:
                    continue
                f.seek(off_meta)
                for _ in range(nt):
                    rec = f.read(128)
                    nm = rec[:80].split(b"\x00", 1)[0].decode("utf-8")
                    R, C, bits, _, _, _, _, orig_r, orig_c, _ = struct.unpack("<IIIIQQfIII", rec[80:])
                    tensor_bits[nm] = int(bits)
                    tensor_shapes[nm] = (int(orig_r or R), int(orig_c or C))

        has_tfq4_only = bool(tffp4_shards) and not bool(tffpw_shards)
        if tensor_bits:
            default_bits = Counter(tensor_bits.values()).most_common(1)[0][0]
        elif has_tfq4_only:
            default_bits = 225 if table_kind == 3 else 2
        else:
            default_bits = 4

        return cls(
            bits=default_bits,
            tensor_bits=tensor_bits,
            tensor_orig_shapes=tensor_shapes,
            tfq_shards=tfq_shards,
            tables="pw_v3_tables.bin" if os.path.isfile(pw3_path) else "pw_v4_e8p_tables.bin",
            table_kind=table_kind,
            format_type="tfq4" if has_tfq4_only else "tfq3",
            tables_v4="pw_v4_e8p_tables.bin" if os.path.isfile(pw4_path) else None,
            moe_layers=sorted(set(moe_layers)),
            swiglu_limit=base_cfg.get("swiglu_limit"),
            gate_softcap=base_cfg.get("gate_softcap"),
            up_softcap=base_cfg.get("up_softcap"),
        )

    def get_tensor_bits(self, name: str) -> int:
        key = _strip_weight_suffix(name)
        if key in self.tensor_bits:
            return self.tensor_bits[key]
        return self.bits

    def has_quantized_prefix(self, prefix: str) -> bool:
        if not self.tensor_bits:
            return True
        clean = _strip_weight_suffix(prefix)
        if clean in self.tensor_bits:
            return True
        if clean.endswith("qkv_proj"):
            base = clean[: -len("qkv_proj")]
            return any((base + s) in self.tensor_bits for s in ("q_proj", "k_proj", "v_proj"))
        if clean.endswith("gate_up_proj"):
            base = clean[: -len("gate_up_proj")]
            return any((base + s) in self.tensor_bits for s in ("gate_proj", "up_proj"))
        return False

    def _prefix_is_tfq4(self, prefix: str) -> bool:
        clean = _strip_weight_suffix(prefix)
        if clean in self.tensor_bits:
            return self.tensor_bits[clean] in (2, 225)
        if clean.endswith("qkv_proj"):
            base = clean[: -len("qkv_proj")]
            found = [self.tensor_bits[base + s] for s in ("q_proj", "k_proj", "v_proj") if (base + s) in self.tensor_bits]
            if found:
                return all(b in (2, 225) for b in found)
        if clean.endswith("gate_up_proj"):
            base = clean[: -len("gate_up_proj")]
            found = [self.tensor_bits[base + s] for s in ("gate_proj", "up_proj") if (base + s) in self.tensor_bits]
            if found:
                return all(b in (2, 225) for b in found)
        return self.format_type == "tfq4" or self.bits in (2, 225)

    def _is_moe_layer(self, layer: torch.nn.Module, prefix: str) -> bool:
        cname = layer.__class__.__name__
        if getattr(layer, "is_tfq3_moe", False) or "MoE" in cname or "FusedMoE" in cname or cname.endswith("Experts"):
            lidx = _infer_layer_idx(prefix, getattr(layer, "layer_idx", None))
            if self.moe_layers and lidx is not None:
                return lidx in self.moe_layers
            return True
        return False

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> Optional[Any]:
        if self._is_moe_layer(layer, prefix):
            return Tfq3MoeMethod(self, prefix=prefix, layer_idx=getattr(layer, "layer_idx", None))
        is_linear = (
            isinstance(layer, torch.nn.Linear)
            or layer.__class__.__name__.endswith("Linear")
            or hasattr(layer, "quant_method")
        )
        if not is_linear:
            return None
        if not self.has_quantized_prefix(prefix):
            return None
        if self._prefix_is_tfq4(prefix):
            return Tfq4LinearMethod(self, prefix=prefix)
        return Tfq3LinearMethod(self, prefix=prefix)

    def get_scaled_act_names(self) -> list[str]:
        return []


def load_tfq3_moe_weights(
    model: torch.nn.Module,
    checkpoint_dir: str,
    *,
    verify_checksums: bool = True,
    cache_experts: bool = False,
    source_dir: Optional[str] = None,
) -> int:
    """
    Open a `TFPW` v3 (or hybrid TFQ3+TFQ4) MoE checkpoint and attach every `TfpwLayer`
    to matching `Tfq3MoeMethod` layers in `model` (and also load any `Tfq3LinearMethod`
    or `Tfq4LinearMethod` layers present in the same directory).
    """
    ckpt = _tfq_moe.TfpwCheckpoint(
        checkpoint_dir,
        verify_checksums=verify_checksums,
        source_dir=source_dir,
        cache_experts=cache_experts,
    )
    # Attach checkpoint handle to model so mmap lifetime matches model lifetime
    setattr(model, "_tfq3_moe_checkpoint", ckpt)
    loaded = 0
    for mod in model.modules():
        qm = getattr(mod, "quant_method", None)
        if isinstance(qm, Tfq3MoeMethod):
            qm.load_layer_from_checkpoint(mod, ckpt)
            loaded += 1

    # Also load optional dense non-MoE TFQ3/TFQ4 shards if present
    has_tfq3_lin = any(
        isinstance(getattr(m, "quant_method", None), Tfq3LinearMethod) and hasattr(m, "tfq3_words")
        for m in model.modules()
    )
    if has_tfq3_lin and os.path.isfile(os.path.join(checkpoint_dir, "pw_v3_tables.bin")):
        reader3 = Tfq3CheckpointReader(checkpoint_dir)
        for mod in model.modules():
            qm = getattr(mod, "quant_method", None)
            if isinstance(qm, Tfq3LinearMethod) and hasattr(mod, "tfq3_words"):
                qm.load_layer_from_reader(mod, reader3)
                loaded += 1

    has_tfq4_lin = any(
        isinstance(getattr(m, "quant_method", None), Tfq4LinearMethod) and hasattr(m, "tfq4_words")
        for m in model.modules()
    )
    if has_tfq4_lin and os.path.isfile(os.path.join(checkpoint_dir, "pw_v4_e8p_tables.bin")):
        reader4 = Tfq4CheckpointReader(checkpoint_dir, verify_checksum=verify_checksums)
        for mod in model.modules():
            qm = getattr(mod, "quant_method", None)
            if isinstance(qm, Tfq4LinearMethod) and hasattr(mod, "tfq4_words"):
                qm.load_layer_from_reader(mod, reader4)
                loaded += 1

    return loaded


def load_tfq3_checkpoint_weights(model: torch.nn.Module, checkpoint_dir: str) -> int:
    """
    Load packed `.tffpw` (`TFPN`/`TFPW`) and/or `.tffp4` (`TFP4` 2.00/2.25 bpw) weights,
    scales, and tables into all TFQ3 / TFQ4 / MoE vLLM layers in `model`.
    """
    loaded = 0
    has_tfq3 = any(
        isinstance(getattr(m, "quant_method", None), Tfq3LinearMethod) and hasattr(m, "tfq3_words")
        for m in model.modules()
    )
    has_tfq4 = any(
        isinstance(getattr(m, "quant_method", None), Tfq4LinearMethod) and hasattr(m, "tfq4_words")
        for m in model.modules()
    )
    has_moe = any(isinstance(getattr(m, "quant_method", None), Tfq3MoeMethod) for m in model.modules())

    if has_tfq3:
        reader3 = Tfq3CheckpointReader(checkpoint_dir)
        for mod in model.modules():
            qm = getattr(mod, "quant_method", None)
            if isinstance(qm, Tfq3LinearMethod) and hasattr(mod, "tfq3_words"):
                qm.load_layer_from_reader(mod, reader3)
                loaded += 1

    if has_tfq4:
        reader4 = Tfq4CheckpointReader(checkpoint_dir)
        for mod in model.modules():
            qm = getattr(mod, "quant_method", None)
            if isinstance(qm, Tfq4LinearMethod) and hasattr(mod, "tfq4_words"):
                qm.load_layer_from_reader(mod, reader4)
                loaded += 1

    if has_moe:
        ckpt = _tfq_moe.TfpwCheckpoint(checkpoint_dir)
        setattr(model, "_tfq3_moe_checkpoint", ckpt)
        for mod in model.modules():
            qm = getattr(mod, "quant_method", None)
            if isinstance(qm, Tfq3MoeMethod):
                qm.load_layer_from_checkpoint(mod, ckpt)
                loaded += 1

    return loaded


def register_tfq3_quantization() -> bool:
    """Register `Tfq3Config` under `--quantization tfq3` (and `tfq4` alias) in vLLM's quantization registry."""
    if not HAS_VLLM:
        return False
    try:
        from vllm.model_executor.layers.quantization import (  # type: ignore
            QUANTIZATION_METHODS,
            register_quantization_config,
        )
        for name in ("tfq3", "tfq4"):
            if name not in QUANTIZATION_METHODS:
                register_quantization_config(name)(Tfq3Config)
        return True
    except Exception:
        try:
            from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS  # type: ignore
            QUANTIZATION_METHODS["tfq3"] = Tfq3Config
            QUANTIZATION_METHODS.setdefault("tfq4", Tfq3Config)
            return True
        except Exception:
            return False


register_tfq3_quantization()
