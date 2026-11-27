"""
vLLM QuantizationConfig, LinearMethodBase, and `.tffpw` Weight Loader for TFQ3 (`TFPN` v3, 3/4/5-bit).

Supports:
  - Automatic registration of `--quantization tfq3` via vLLM's `register_quantization_config("tfq3")`
  - Per-tensor bit widths (`3b`, `4b`, `5b`) parsed from `tfq3_manifest.json` (mixed-bit checkpoints)
  - Direct `.tffpw` (`TFPN`) shard + `pw_v3_tables.bin` (`PWV3`) weight loading (`Tfq3CheckpointReader`,
    `load_tfq3_checkpoint_weights`, and per-parameter `weight_loader`)
  - Merged projections (`qkv_proj`, `gate_up_proj`, arbitrary N-way) with independent per-partition
    128-row padding, per-partition bit widths, and per-partition absolute row signs (`row_off`)
  - Zero-repack Tensor Parallelism (`TP > 1`): Column-Parallel (`row_off`) and Row-Parallel
    (`k_sign_off` + 1-block trellis state resume `bc0_buf`) on both `tfq3_cuda_ext` and portable fallback
"""
from __future__ import annotations

import glob
import json
import os
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

MAGIC_TFPN = 0x4E504654
MAGIC_PWV3 = 0x33565750
FORMAT_VER = 3

try:
    from vllm.model_executor.layers.linear import LinearMethodBase, set_weight_attrs  # type: ignore
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig  # type: ignore
    HAS_VLLM = True
except Exception:
    HAS_VLLM = False

    class QuantizationConfig:  # type: ignore
        pass

    class LinearMethodBase:  # type: ignore
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


class Tfq3CheckpointReader:
    """Random-access reader for TFQ3 (`TFPN` v3 `.tffpw`) checkpoints and `pw_v3_tables.bin`."""

    def __init__(self, checkpoint_dir: str) -> None:
        self.checkpoint_dir = checkpoint_dir
        mf_path = os.path.join(checkpoint_dir, "tfq3_manifest.json")
        self.manifest = json.load(open(mf_path, "r", encoding="utf-8")) if os.path.isfile(mf_path) else {}
        tables_name = self.manifest.get("tables", "pw_v3_tables.bin")
        tables_path = os.path.join(checkpoint_dir, tables_name)
        with open(tables_path, "rb") as f:
            self.tables_bytes = f.read()
        if len(self.tables_bytes) != 16384:
            raise ValueError(f"Expected 16384-byte {tables_path}, got {len(self.tables_bytes)}")
        self.tables_dict = _parse_pwv3_tables_bytes(self.tables_bytes)

        shard_names = self.manifest.get("tfq_shards")
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


class Tfq3LinearMethod(LinearMethodBase):
    """vLLM LinearMethod supporting per-tensor bits, merged projections, and Col/Row TP > 1."""

    def __init__(self, quant_config: "Tfq3Config", prefix: str = "") -> None:
        self.quant_config = quant_config
        self.prefix = prefix

    def _resolve_partition_names(
        self,
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


class Tfq3Config(QuantizationConfig):
    """vLLM QuantizationConfig for TFQ3 checkpoints (`tfq3_manifest.json` + `.tffpw` + `pw_v3_tables.bin`)."""

    def __init__(
        self,
        bits: int = 4,
        tensor_bits: Optional[dict[str, int]] = None,
        tensor_orig_shapes: Optional[dict[str, tuple[int, int]]] = None,
        tfq_shards: Optional[list[str]] = None,
        tables: str = "pw_v3_tables.bin",
    ) -> None:
        if bits not in (3, 4, 5):
            raise ValueError(f"TFQ3 supports bits in {{3, 4, 5}}, got {bits}")
        self.bits = int(bits)
        self.tensor_bits: dict[str, int] = {}
        if tensor_bits:
            for k, v in tensor_bits.items():
                iv = int(v)
                if iv not in (3, 4, 5):
                    raise ValueError(f"TFQ3 tensor '{k}' has invalid bits={iv}")
                self.tensor_bits[_strip_weight_suffix(k)] = iv
        self.tensor_orig_shapes: dict[str, tuple[int, int]] = dict(tensor_orig_shapes or {})
        self.tfq_shards: list[str] = list(tfq_shards or [])
        self.tables: str = tables

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

        if "bits" in config and config["bits"] is not None:
            default_bits = int(config["bits"])
        elif tensor_bits:
            default_bits = Counter(tensor_bits.values()).most_common(1)[0][0]
        else:
            default_bits = 4

        return cls(
            bits=default_bits,
            tensor_bits=tensor_bits,
            tensor_orig_shapes=tensor_shapes,
            tfq_shards=list(config.get("tfq_shards", [])),
            tables=str(config.get("tables", "pw_v3_tables.bin")),
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

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> Optional[LinearMethodBase]:
        is_linear = (
            isinstance(layer, torch.nn.Linear)
            or layer.__class__.__name__.endswith("Linear")
            or hasattr(layer, "quant_method")
        )
        if not is_linear:
            return None
        if not self.has_quantized_prefix(prefix):
            return None
        return Tfq3LinearMethod(self, prefix=prefix)

    def get_scaled_act_names(self) -> list[str]:
        return []


def load_tfq3_checkpoint_weights(model: torch.nn.Module, checkpoint_dir: str) -> int:
    """Load packed `.tffpw` weights, scales, and `pw_v3_tables.bin` into all TFQ3 vLLM layers in `model`."""
    reader = Tfq3CheckpointReader(checkpoint_dir)
    loaded = 0
    for mod in model.modules():
        qm = getattr(mod, "quant_method", None)
        if isinstance(qm, Tfq3LinearMethod) and hasattr(mod, "tfq3_words"):
            qm.load_layer_from_reader(mod, reader)
            loaded += 1
    return loaded


def register_tfq3_quantization() -> bool:
    """Register `Tfq3Config` under `--quantization tfq3` in vLLM's quantization registry."""
    if not HAS_VLLM:
        return False
    try:
        from vllm.model_executor.layers.quantization import (  # type: ignore
            QUANTIZATION_METHODS,
            register_quantization_config,
        )
        if "tfq3" not in QUANTIZATION_METHODS:
            register_quantization_config("tfq3")(Tfq3Config)
        return True
    except Exception:
        try:
            from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS  # type: ignore
            QUANTIZATION_METHODS["tfq3"] = Tfq3Config
            return True
        except Exception:
            return False


register_tfq3_quantization()
