#!/usr/bin/env python3
"""
TFQ3 / TFQ4 Checkpoint Inspector, Quality Evaluator, Throughput Benchmarker &
Hugging Face Model Card Generator (`tfq3-eval`).

Capabilities:
  1. Offline checkpoint inspection (`--inspect-only`, zero HF downloads / zero GPU):
     - Inspects `PWV3` / `PWV4` (`table_kind=0..3`), `TFPN` v3, `TFPW` v3 / hybrid MoE,
       `TFP4` v4 (`2.00` & `2.25 bpw`), and residual `*.safetensors` headers
     - Reports exact effective bpw, disk footprint, parameter & bit-width histograms,
       table SHA-256s, and full `XXH64` shard integrity verification
  2. Language modeling evaluation (`--dataset wikitext2|c4` or `--text-file`):
     - Exact `fp64` Perplexity (PPL) and per-sequence NLL
     - Optional paired `--ref-model` comparison: exact `fp64` full-vocabulary forward KL
       divergence D_KL(P_ref || P_q) (mean/median/p95/p99/max), Top-1 token agreement %,
       and paired ΔNLL with 95% bootstrap CI
  3. Timed inference throughput (`--bench-throughput`):
     - Prefill tok/s and KV-cached autoregressive decode tok/s on CPU / MPS / CUDA
  4. Ready-to-paste Hugging Face Model Card output (`--model-card <out.md>`) and
     machine-readable JSON receipt (`--json <out.json>`).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import platform
import struct
import sys
import time
from typing import Any, Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)

try:
    from tfq3.reference import tfq3_reference as tfq3_ref  # type: ignore
except ImportError:
    _REF_DIR = os.path.join(_REPO_ROOT, "reference")
    if _REF_DIR not in sys.path:
        sys.path.insert(0, _REF_DIR)
    import tfq3_reference as tfq3_ref  # type: ignore

MAGIC_PWV3 = 0x33565750
MAGIC_PWV4 = 0x34565750
MAGIC_TFPN = 0x4E504654
MAGIC_TFPW = 0x57504654
MAGIC_TFP4 = 0x34504654
FLAG_XXH64_PRESENT = 0x1

_SAFETENSORS_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}

_TABLE_KIND_LABELS = {
    0: "legacy_trained_e8p_256",
    1: "trained_e8p_256",
    2: "true_quip_e8p_256",
    3: "true_quip_e8p_1024",
}


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _bit_label(bits: int) -> str:
    if bits == 225:
        return "2.25b"
    if bits == 2:
        return "2.00b"
    return f"{bits}b"


def _tfpw_mat_words(R: int, C: int, bits: int) -> int:
    if bits in (3, 4, 5):
        return (R // 16) * (C // 128) * (bits * 2) * 32
    if bits == 2:
        return R * (C // 16)
    if bits == 225:
        return R * (C // 128) * 9
    raise ValueError(f"Unsupported TFPW expert bit width: {bits}")


def _read_safetensors_meta(path: str) -> tuple[int, int, int]:
    """Return `(tensor_count, param_count, payload_bytes)` without reading tensor bodies."""
    with open(path, "rb") as f:
        raw_len = f.read(8)
        if len(raw_len) < 8:
            raise ValueError(f"Truncated safetensors file: {path}")
        hlen = struct.unpack("<Q", raw_len)[0]
        hdr = json.loads(f.read(hlen).decode("utf-8"))
    n_tensors = 0
    n_params = 0
    n_bytes = 0
    for k, meta in hdr.items():
        if k == "__metadata__":
            continue
        n_tensors += 1
        shape = meta.get("shape", [])
        elems = 1
        for d in shape:
            elems *= int(d)
        n_params += elems
        o0, o1 = meta.get("data_offsets", [0, 0])
        n_bytes += max(0, int(o1) - int(o0))
    return n_tensors, n_params, n_bytes


def inspect_checkpoint(ckpt_dir: str, *, verify_checksum: bool = True) -> dict[str, Any]:
    """
    Inspect any TFQ3 (`TFPN`/`TFPW`), TFQ4 (`TFP4` 2.00/2.25 bpw), or hybrid checkpoint
    directory completely offline (no HF downloads, no GPU required).
    """
    ckpt_abs = os.path.abspath(ckpt_dir)
    if not os.path.isdir(ckpt_abs):
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    config: dict[str, Any] = {}
    cfg_path = os.path.join(ckpt_abs, "config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            config = json.load(f)

    manifest: dict[str, Any] = {}
    manifest_name: Optional[str] = None
    for mf in ("tfq3_manifest.json", "tfq4_manifest.json"):
        mp = os.path.join(ckpt_abs, mf)
        if os.path.isfile(mp):
            with open(mp, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            manifest_name = mf
            break

    # 1. Inspect table binaries (PWV3 / PWV4)
    tables_info: dict[str, Any] = {}
    pw3_path = os.path.join(ckpt_abs, "pw_v3_tables.bin")
    if os.path.isfile(pw3_path):
        raw3 = open(pw3_path, "rb").read()
        if len(raw3) < 10816:
            raise ValueError(f"Truncated pw_v3_tables.bin ({len(raw3)} bytes)")
        magic3, ver3 = struct.unpack_from("<II", raw3, 0)
        if magic3 != MAGIC_PWV3 or ver3 != 3:
            raise ValueError(f"Bad PWV3 header: magic=0x{magic3:08x} ver={ver3}")
        tables_info["pw_v3"] = {
            "filename": "pw_v3_tables.bin",
            "version": int(ver3),
            "bytes": len(raw3),
            "sha256": hashlib.sha256(raw3).hexdigest(),
        }

    pw4_path = os.path.join(ckpt_abs, "pw_v4_e8p_tables.bin")
    if os.path.isfile(pw4_path):
        raw4 = open(pw4_path, "rb").read()
        if len(raw4) not in (16384, 32768):
            raise ValueError(f"Unexpected pw_v4_e8p_tables.bin size: {len(raw4)} bytes")
        magic4, ver4, k_codes, dim4, kind4 = struct.unpack_from("<IIIII", raw4, 0)
        if magic4 != MAGIC_PWV4 or ver4 != 4 or dim4 != 8:
            raise ValueError(f"Bad PWV4 header: magic=0x{magic4:08x} ver={ver4} k={k_codes} dim={dim4}")
        tables_info["pw_v4"] = {
            "filename": "pw_v4_e8p_tables.bin",
            "version": int(ver4),
            "k_codes": int(k_codes),
            "dim": int(dim4),
            "table_kind": int(kind4),
            "table_kind_label": _TABLE_KIND_LABELS.get(int(kind4), f"kind_{kind4}"),
            "bytes": len(raw4),
            "sha256": hashlib.sha256(raw4).hexdigest(),
        }

    # 2. Inspect .tffpw (TFPN / TFPW) and .tffp4 (TFP4) shards
    bit_hist_units: dict[str, int] = {}
    bit_hist_params: dict[str, int] = {}
    quantized_params = 0
    packed_payload_bytes = 0
    quantized_disk_bytes = 0
    tfpn_shards = 0
    tfpw_shards = 0
    tfp4_shards = 0
    tfpn_tensors = 0
    tfp4_tensors = 0
    moe_layer_indices: list[int] = []
    moe_experts_total = 0
    snr_vals: list[float] = []

    xxh_checked = 0
    xxh_verified = 0
    xxh_legacy = 0
    xxh_errors: list[str] = []

    for sp in sorted(glob.glob(os.path.join(ckpt_abs, "*.tffpw"))):
        base = os.path.basename(sp)
        fsize = os.path.getsize(sp)
        quantized_disk_bytes += fsize
        with open(sp, "rb") as f:
            raw_magic = f.read(4)
            if len(raw_magic) < 4:
                raise ValueError(f"Truncated shard: {base}")
            magic = struct.unpack("<I", raw_magic)[0]
            f.seek(0)

            if magic == MAGIC_TFPN:
                tfpn_shards += 1
                hdr = struct.unpack("<IIIIQQQQQQ", f.read(64))
                _, ver, nt, pad0, off_meta, off_scales, off_weights, tot_sc, tot_w, pad1 = hdr
                if ver != 3:
                    raise ValueError(f"Unsupported TFPN version {ver} in {base}")
                xxh_checked += 1
                if (pad0 & FLAG_XXH64_PRESENT) != 0:
                    if verify_checksum:
                        try:
                            tfq3_ref.verify_tfpn_like_checksum(
                                f, off_meta, nt * 128, off_scales, tot_sc * 2, off_weights, tot_w * 4, pad0, pad1, base
                            )
                            xxh_verified += 1
                        except ValueError as exc:
                            xxh_errors.append(str(exc))
                    else:
                        xxh_verified += 1
                else:
                    xxh_legacy += 1

                f.seek(off_meta)
                for _ in range(nt):
                    rec = f.read(128)
                    R, C, bits, _, _, w_cnt, snr_db, orig_r, orig_c, _ = struct.unpack("<IIIIQQfIII", rec[80:])
                    o_r = orig_r or R
                    o_c = orig_c or C
                    n_p = int(o_r) * int(o_c)
                    n_sc = (R // 128) * (C // 128)
                    quantized_params += n_p
                    packed_payload_bytes += int(w_cnt) * 4 + int(n_sc) * 2
                    tfpn_tensors += 1
                    lbl = _bit_label(int(bits))
                    bit_hist_units[lbl] = bit_hist_units.get(lbl, 0) + 1
                    bit_hist_params[lbl] = bit_hist_params.get(lbl, 0) + n_p
                    if math.isfinite(snr_db) and snr_db > 0.0:
                        snr_vals.append(float(snr_db))

            elif magic == MAGIC_TFPW:
                tfpw_shards += 1
                hdr_raw = f.read(144)
                if len(hdr_raw) < 144:
                    raise ValueError(f"Truncated TFPW header in {base}")
                _, ver, layer_idx, n_exp, _, H, I, flags = struct.unpack_from("<8I", hdr_raw, 0)
                _, stream_words, scale_halfs, off_meta, off_scales, off_weights = struct.unpack_from("<6Q", hdr_raw, 32)
                c_meta, c_scales, c_weights, _ = struct.unpack_from("<4Q", hdr_raw, 80)
                if ver != 3:
                    raise ValueError(f"Unsupported TFPW version {ver} in {base}")
                moe_layer_indices.append(int(layer_idx))
                moe_experts_total += int(n_exp)

                meta_bytes = n_exp * 64
                scales_bytes = scale_halfs * 2
                weights_bytes = stream_words * 4
                xxh_checked += 1
                if (flags & FLAG_XXH64_PRESENT) != 0:
                    if verify_checksum:
                        ok = True
                        for sec_name, off, length, exp in (
                            ("meta", off_meta, meta_bytes, c_meta),
                            ("scales", off_scales, scales_bytes, c_scales),
                            ("weights", off_weights, weights_bytes, c_weights),
                        ):
                            f.seek(off)
                            h = tfq3_ref.XXH64(0)
                            rem = length
                            while rem > 0:
                                chunk = f.read(min(rem, 4 << 20))
                                if not chunk:
                                    break
                                h.update(chunk)
                                rem -= len(chunk)
                            got = h.intdigest()
                            if got != exp:
                                xxh_errors.append(
                                    f"{base} {sec_name} XXH64 mismatch: got 0x{got:016x}, expected 0x{exp:016x}"
                                )
                                ok = False
                        if ok:
                            xxh_verified += 1
                    else:
                        xxh_verified += 1
                else:
                    xxh_legacy += 1

                n_sc_gu = (I // 128) * (H // 128)
                n_sc_dn = (H // 128) * (I // 128)
                f.seek(off_meta)
                for _ in range(n_exp):
                    em = f.read(64)
                    b_bits, _, _, _, _, _, _, _, gsnr, usnr, dsnr, _, _ = struct.unpack("<IIQQQIIIfffII", em)
                    gu_w = _tfpw_mat_words(I, H, int(b_bits))
                    dn_w = _tfpw_mat_words(H, I, int(b_bits))
                    n_p = 3 * int(H) * int(I)
                    quantized_params += n_p
                    packed_payload_bytes += (2 * gu_w + dn_w) * 4 + (2 * n_sc_gu + n_sc_dn) * 2
                    lbl = _bit_label(int(b_bits))
                    bit_hist_units[lbl] = bit_hist_units.get(lbl, 0) + 1
                    bit_hist_params[lbl] = bit_hist_params.get(lbl, 0) + n_p
                    for sv in (gsnr, usnr, dsnr):
                        if math.isfinite(sv) and sv > 0.0:
                            snr_vals.append(float(sv))
            else:
                raise ValueError(f"Unknown magic 0x{magic:08x} in {base}")

    for sp in sorted(glob.glob(os.path.join(ckpt_abs, "*.tffp4"))):
        base = os.path.basename(sp)
        fsize = os.path.getsize(sp)
        quantized_disk_bytes += fsize
        tfp4_shards += 1
        with open(sp, "rb") as f:
            hdr = struct.unpack("<IIIIQQQQQQ", f.read(64))
            magic, ver, nt, pad0, off_meta, off_scales, off_weights, tot_sc, tot_w, pad1 = hdr
            if magic != MAGIC_TFP4 or ver != 4:
                raise ValueError(f"Bad TFP4 header in {base}: magic=0x{magic:08x} ver={ver}")
            xxh_checked += 1
            if (pad0 & FLAG_XXH64_PRESENT) != 0:
                if verify_checksum:
                    try:
                        tfq3_ref.verify_tfpn_like_checksum(
                            f, off_meta, nt * 128, off_scales, tot_sc * 2, off_weights, tot_w * 4, pad0, pad1, base
                        )
                        xxh_verified += 1
                    except ValueError as exc:
                        xxh_errors.append(str(exc))
                else:
                    xxh_verified += 1
            else:
                xxh_legacy += 1

            f.seek(off_meta)
            for _ in range(nt):
                rec = f.read(128)
                R, C, bits, _, _, w_cnt, snr_db, orig_r, orig_c, _ = struct.unpack("<IIIIQQfIII", rec[80:])
                o_r = orig_r or R
                o_c = orig_c or C
                n_p = int(o_r) * int(o_c)
                n_sc = (R // 128) * (C // 128)
                quantized_params += n_p
                packed_payload_bytes += int(w_cnt) * 4 + int(n_sc) * 2
                tfp4_tensors += 1
                lbl = _bit_label(int(bits))
                bit_hist_units[lbl] = bit_hist_units.get(lbl, 0) + 1
                bit_hist_params[lbl] = bit_hist_params.get(lbl, 0) + n_p
                if math.isfinite(snr_db) and snr_db > 0.0:
                    snr_vals.append(float(snr_db))

    # 3. Inspect residual safetensors headers
    residual_shards = 0
    residual_tensors = 0
    residual_params = 0
    residual_disk_bytes = 0
    for sf in sorted(glob.glob(os.path.join(ckpt_abs, "*.safetensors"))):
        residual_shards += 1
        residual_disk_bytes += os.path.getsize(sf)
        nt_s, np_s, _ = _read_safetensors_meta(sf)
        residual_tensors += nt_s
        residual_params += np_s

    # 4. Total checkpoint disk footprint
    disk_bytes = 0
    for entry in os.listdir(ckpt_abs):
        ep = os.path.join(ckpt_abs, entry)
        if os.path.isfile(ep):
            disk_bytes += os.path.getsize(ep)

    total_params = quantized_params + residual_params
    quantized_bpw = (packed_payload_bytes * 8.0 / quantized_params) if quantized_params > 0 else 0.0
    table_bytes = sum(t["bytes"] for t in tables_info.values())
    total_weight_bytes = quantized_disk_bytes + residual_disk_bytes + table_bytes
    overall_bpw = (total_weight_bytes * 8.0 / total_params) if total_params > 0 else 0.0

    # Format classification
    has_tfq3_bits = any(k in bit_hist_units for k in ("3b", "4b", "5b"))
    has_tfq4_bits = any(k in bit_hist_units for k in ("2.00b", "2.25b"))
    if tfpw_shards > 0:
        fmt_label = "TFQ3+TFQ4-Hybrid-MoE" if (has_tfq3_bits and has_tfq4_bits) else ("TFQ4-MoE" if has_tfq4_bits else "TFQ3-MoE")
    elif has_tfq3_bits and has_tfq4_bits:
        fmt_label = "TFQ3+TFQ4-Hybrid-Dense"
    elif has_tfq4_bits:
        if "2.25b" in bit_hist_units and "2.00b" not in bit_hist_units:
            fmt_label = "TFQ4-2.25bpw"
        elif "2.00b" in bit_hist_units and "2.25b" not in bit_hist_units:
            fmt_label = "TFQ4-2.00bpw"
        else:
            fmt_label = "TFQ4-Mixed"
    elif tfpn_shards > 0:
        fmt_label = "TFQ3-Dense"
    else:
        fmt_label = "Unquantized"

    return {
        "checkpoint_name": os.path.basename(ckpt_abs),
        "format": fmt_label,
        "manifest": manifest_name,
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        "tables": tables_info,
        "shards": {
            "tfpn": tfpn_shards,
            "tfpw": tfpw_shards,
            "tfp4": tfp4_shards,
            "residual_safetensors": residual_shards,
        },
        "counts": {
            "tfpn_tensors": tfpn_tensors,
            "tfp4_tensors": tfp4_tensors,
            "moe_layers": len(moe_layer_indices),
            "moe_layer_indices": sorted(moe_layer_indices),
            "moe_experts_total": moe_experts_total,
            "residual_tensors": residual_tensors,
        },
        "params": {
            "quantized": quantized_params,
            "residual": residual_params,
            "total": total_params,
        },
        "bpw": {
            "quantized_effective": round(quantized_bpw, 4),
            "overall_effective": round(overall_bpw, 4),
        },
        "disk": {
            "total_bytes": disk_bytes,
            "total_gib": round(disk_bytes / float(1 << 30), 4),
            "quantized_bytes": quantized_disk_bytes,
            "residual_bytes": residual_disk_bytes,
            "packed_payload_bytes": packed_payload_bytes,
        },
        "bit_histogram_units": dict(sorted(bit_hist_units.items())),
        "bit_histogram_params": dict(sorted(bit_hist_params.items())),
        "mean_recorded_snr_db": round(float(np.mean(snr_vals)), 3) if snr_vals else None,
        "xxh64": {
            "shards_checked": xxh_checked,
            "shards_verified": xxh_verified,
            "shards_legacy_unchecksummed": xxh_legacy,
            "all_verified": (xxh_checked > 0 and xxh_verified == xxh_checked and not xxh_errors),
            "errors": xxh_errors,
        },
    }


def _select_device(device_arg: str):
    import torch

    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _select_dtype(dtype_arg: str):
    import torch

    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "f16": torch.float16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "f32": torch.float32,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if dtype_arg.lower() not in mapping:
        raise ValueError(f"Unsupported --dtype {dtype_arg!r} (expected bf16, f16, or f32)")
    return mapping[dtype_arg.lower()]


def _sync_device(device) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def _hardware_label(device) -> str:
    import torch

    if device.type == "cuda":
        try:
            return torch.cuda.get_device_name(device)
        except Exception:
            return "NVIDIA CUDA GPU"
    if device.type == "mps":
        return f"Apple Silicon MPS ({platform.machine()})"
    return f"CPU ({platform.machine()})"


def build_token_chunks_from_ids(token_ids: list[int], seq_len: int, eval_seqs: int):
    import torch

    max_seqs = len(token_ids) // seq_len
    use_seqs = max_seqs if eval_seqs <= 0 else min(eval_seqs, max_seqs)
    if use_seqs <= 0:
        raise RuntimeError(f"Not enough tokens ({len(token_ids)}) for seq_len={seq_len}")
    need = seq_len * use_seqs
    arr = torch.tensor(token_ids[:need], dtype=torch.long).reshape(use_seqs, seq_len)
    return [arr[i : i + 1] for i in range(use_seqs)]


def load_eval_chunks(
    tokenizer,
    *,
    dataset_name: Optional[str] = None,
    text_file: Optional[str] = None,
    seq_len: int = 2048,
    eval_seqs: int = 32,
) -> tuple[str, list]:
    if text_file:
        with open(text_file, "r", encoding="utf-8") as f:
            raw_text = f.read()
        ids = tokenizer.encode(raw_text, add_special_tokens=False)
        return f"local-text ({os.path.basename(text_file)})", build_token_chunks_from_ids(ids, seq_len, eval_seqs)

    ds_name = (dataset_name or "wikitext2").lower()
    from datasets import load_dataset  # type: ignore

    if ds_name == "wikitext2":
        texts = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"]
        text = "\n\n".join(t for t in texts if t.strip())
        ids = tokenizer.encode(text, add_special_tokens=False)
        return "Salesforce/wikitext (wikitext-2-raw-v1, test)", build_token_chunks_from_ids(ids, seq_len, eval_seqs)
    if ds_name == "c4":
        ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        texts = []
        tok_est = 0
        target_toks = (eval_seqs if eval_seqs > 0 else 64) * seq_len * 2
        for row in ds:
            t = row.get("text", "")
            if t.strip():
                texts.append(t)
                tok_est += max(1, len(t) // 3)
                if tok_est >= target_toks:
                    break
        text = "\n\n".join(texts)
        ids = tokenizer.encode(text, add_special_tokens=False)
        return "allenai/c4 (en, validation)", build_token_chunks_from_ids(ids, seq_len, eval_seqs if eval_seqs > 0 else 64)
    raise ValueError(f"Unsupported --dataset {dataset_name!r} (expected 'wikitext2' or 'c4')")


def _paired_bootstrap_ci95(deltas: np.ndarray, n_boot: int = 1000, seed: int = 42) -> tuple[float, float]:
    if deltas.size < 2:
        val = float(deltas[0]) if deltas.size == 1 else 0.0
        return val, val
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, deltas.size, size=(n_boot, deltas.size))
    means = deltas[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def evaluate_causal_lm(
    quant_model,
    chunks: list,
    *,
    ref_model=None,
    device=None,
    dataset_label: str = "custom",
) -> dict[str, Any]:
    """
    Exact `fp64` causal LM evaluation over tokenized chunks `[1, seq_len]`.

    When `ref_model` is provided, computes exact full-vocabulary `fp64` forward KL
    divergence `D_KL(P_ref || P_quant)`, Top-1 agreement %, and paired ΔNLL ± 95% CI.
    """
    import torch
    import torch.nn.functional as F

    if device is None:
        try:
            device = next(quant_model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    quant_model.eval()
    if ref_model is not None:
        ref_model.eval()

    q_seq_nlls: list[float] = []
    r_seq_nlls: list[float] = []
    q_nll_sum = 0.0
    r_nll_sum = 0.0
    tot_tokens = 0
    kl_chunks: list[np.ndarray] = []
    top1_hits = 0

    with torch.no_grad():
        for chunk in chunks:
            inp = chunk.to(device=device)
            targets = inp[:, 1:]  # [1, L-1]
            n_tok = int(targets.numel())
            if n_tok <= 0:
                continue

            q_out = quant_model(input_ids=inp)
            q_logits = q_out.logits[:, :-1, :].to(torch.float64)
            log_q = F.log_softmax(q_logits, dim=-1)
            q_tok_nll = -log_q.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            q_mean_seq = float(q_tok_nll.mean().item())
            q_seq_nlls.append(q_mean_seq)
            q_nll_sum += float(q_tok_nll.sum().item())
            tot_tokens += n_tok

            if ref_model is not None:
                r_out = ref_model(input_ids=inp)
                r_logits = r_out.logits[:, :-1, :].to(torch.float64)
                log_p = F.log_softmax(r_logits, dim=-1)
                r_tok_nll = -log_p.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                r_seq_nlls.append(float(r_tok_nll.mean().item()))
                r_nll_sum += float(r_tok_nll.sum().item())

                p = torch.exp(log_p)
                kl = torch.clamp(torch.sum(p * (log_p - log_q), dim=-1), min=0.0)
                kl_chunks.append(kl.reshape(-1).cpu().numpy())
                top1_hits += int((r_logits.argmax(dim=-1) == q_logits.argmax(dim=-1)).sum().item())

    if tot_tokens == 0:
        raise RuntimeError("No scored tokens in evaluation chunks")

    q_nll = q_nll_sum / float(tot_tokens)
    q_ppl = math.exp(min(q_nll, 100.0))
    seq_len = int(chunks[0].shape[-1]) if chunks else 0

    res: dict[str, Any] = {
        "dataset": dataset_label,
        "num_sequences": len(q_seq_nlls),
        "seq_len": seq_len,
        "scored_tokens": tot_tokens,
        "mean_nll": round(q_nll, 6),
        "ppl": round(q_ppl, 6),
        "per_seq_nll": [round(v, 6) for v in q_seq_nlls],
    }

    if ref_model is not None and kl_chunks:
        r_nll = r_nll_sum / float(tot_tokens)
        r_ppl = math.exp(min(r_nll, 100.0))
        deltas = np.asarray(q_seq_nlls, dtype=np.float64) - np.asarray(r_seq_nlls, dtype=np.float64)
        ci_lo, ci_hi = _paired_bootstrap_ci95(deltas)
        kl_all = np.concatenate(kl_chunks)
        res["reference"] = {
            "mean_nll": round(r_nll, 6),
            "ppl": round(r_ppl, 6),
            "delta_ppl": round(q_ppl - r_ppl, 6),
            "delta_ppl_pct": round(100.0 * (q_ppl - r_ppl) / max(r_ppl, 1e-12), 4),
            "delta_nll_mean": round(float(deltas.mean()), 6),
            "delta_nll_ci95": [round(ci_lo, 6), round(ci_hi, 6)],
            "kl_fp64_mean": round(float(np.mean(kl_all)), 6),
            "kl_fp64_median": round(float(np.median(kl_all)), 6),
            "kl_fp64_p95": round(float(np.percentile(kl_all, 95)), 6),
            "kl_fp64_p99": round(float(np.percentile(kl_all, 99)), 6),
            "kl_fp64_max": round(float(np.max(kl_all)), 6),
            "top1_agreement_pct": round(100.0 * top1_hits / float(tot_tokens), 4),
        }

    return res


def benchmark_throughput(
    model,
    *,
    vocab_size: int,
    device,
    prefill_len: int = 256,
    decode_steps: int = 32,
    warmup: int = 1,
) -> dict[str, Any]:
    """Timed prefill and KV-cached autoregressive decode throughput benchmark."""
    import torch

    model.eval()
    g = torch.Generator(device="cpu").manual_seed(1234)
    prompt = torch.randint(0, max(vocab_size, 2), (1, prefill_len), generator=g, dtype=torch.long).to(device)

    with torch.no_grad():
        for _ in range(max(1, warmup)):
            out = model(input_ids=prompt, use_cache=True)
            _sync_device(device)

        _sync_device(device)
        t0 = time.perf_counter()
        out = model(input_ids=prompt, use_cache=True)
        _sync_device(device)
        prefill_s = max(time.perf_counter() - t0, 1e-9)

        past = getattr(out, "past_key_values", None)
        next_tok = out.logits[:, -1:, :].argmax(dim=-1)
        use_kv = past is not None

        if use_kv:
            # 1 warmup decode step
            w_out = model(input_ids=next_tok, past_key_values=past, use_cache=True)
            _sync_device(device)
            past = w_out.past_key_values
            next_tok = w_out.logits[:, -1:, :].argmax(dim=-1)

            _sync_device(device)
            t1 = time.perf_counter()
            for _ in range(decode_steps):
                d_out = model(input_ids=next_tok, past_key_values=past, use_cache=True)
                past = d_out.past_key_values
                next_tok = d_out.logits[:, -1:, :].argmax(dim=-1)
            _sync_device(device)
            decode_s = max(time.perf_counter() - t1, 1e-9)
        else:
            _sync_device(device)
            t1 = time.perf_counter()
            cur = next_tok
            for _ in range(decode_steps):
                d_out = model(input_ids=cur, use_cache=False)
                cur = d_out.logits[:, -1:, :].argmax(dim=-1)
            _sync_device(device)
            decode_s = max(time.perf_counter() - t1, 1e-9)

    return {
        "device": str(device),
        "hardware": _hardware_label(device),
        "prefill_tokens": int(prefill_len),
        "prefill_ms": round(prefill_s * 1000.0, 3),
        "prefill_tok_per_sec": round(prefill_len / prefill_s, 2),
        "decode_steps": int(decode_steps),
        "decode_kv_cached": bool(use_kv),
        "decode_ms": round(decode_s * 1000.0, 3),
        "decode_ms_per_tok": round((decode_s * 1000.0) / max(decode_steps, 1), 3),
        "decode_tok_per_sec": round(decode_steps / decode_s, 2),
    }


def render_model_card(
    report: dict[str, Any],
    *,
    model_name: Optional[str] = None,
    cli_command: Optional[str] = None,
) -> str:
    """Render a ready-to-paste Hugging Face Model Card markdown section with exact provenance tags."""
    insp = report["inspection"]
    ev = report.get("evaluation")
    tp = report.get("throughput")
    prov = report.get("provenance", {})

    title = model_name or insp.get("checkpoint_name") or "TFQ-Checkpoint"
    tag = "[MEASURED]" if (ev is not None or tp is not None) else "[INSPECTED]"

    bit_parts = [f"`{k}`: {v}" for k, v in insp.get("bit_histogram_units", {}).items()]
    bit_str = ", ".join(bit_parts) if bit_parts else "none"

    tbl_parts = []
    if "pw_v3" in insp.get("tables", {}):
        t3 = insp["tables"]["pw_v3"]
        tbl_parts.append(f"`PWV3` v{t3['version']} (`sha256:{t3['sha256'][:12]}`)")
    if "pw_v4" in insp.get("tables", {}):
        t4 = insp["tables"]["pw_v4"]
        tbl_parts.append(
            f"`PWV4` v{t4['version']} `table_kind={t4['table_kind']}` ({t4['table_kind_label']}, `sha256:{t4['sha256'][:12]}`)"
        )
    tbl_str = "; ".join(tbl_parts) if tbl_parts else "none"

    xxh = insp.get("xxh64", {})
    if xxh.get("errors"):
        xxh_str = f"FAILED ({len(xxh['errors'])} error(s))"
    elif xxh.get("all_verified"):
        xxh_str = f"PASS ({xxh['shards_verified']}/{xxh['shards_checked']} shards verified)"
    elif xxh.get("shards_checked", 0) > 0:
        xxh_str = (
            f"PARTIAL ({xxh['shards_verified']}/{xxh['shards_checked']} verified, "
            f"{xxh['shards_legacy_unchecksummed']} legacy unchecksummed)"
        )
    else:
        xxh_str = "N/A"

    lines = [
        "## Quantization & Evaluation (TFQ3/TFQ4)",
        "",
        f"> **Checkpoint:** `{title}` | **Provenance:** `{tag}` | **Format:** `{insp['format']}`",
        "",
        "### Checkpoint Layout & Footprint",
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
        f"| **Format** | `{insp['format']}` |",
        f"| **Effective Quantized bpw** | `{insp['bpw']['quantized_effective']:.4f} bpw` |",
        f"| **Effective Overall bpw (incl. residuals)** | `{insp['bpw']['overall_effective']:.4f} bpw` |",
        f"| **Disk Footprint** | `{insp['disk']['total_gib']:.3f} GiB` ({insp['disk']['total_bytes']:,} bytes) |",
        f"| **Parameters (Quantized / Residual / Total)** | `{insp['params']['quantized']:,}` / `{insp['params']['residual']:,}` / `{insp['params']['total']:,}` |",
        f"| **Bit Distribution** | {bit_str} |",
        f"| **Codebook Tables** | {tbl_str} |",
        f"| **XXH64 Integrity** | `{xxh_str}` |",
    ]
    if insp.get("mean_recorded_snr_db") is not None:
        lines.append(f"| **Mean Recorded Weight SNR** | `{insp['mean_recorded_snr_db']:.2f} dB` |")

    if ev is not None:
        lines += [
            "",
            "### Quality Evaluation (`[MEASURED]`, exact `fp64` accumulation)",
            "",
            "| Metric | Value |",
            "| :--- | :--- |",
            f"| **Dataset** | `{ev['dataset']}` |",
            f"| **Sequences × Seq Len** | `{ev['num_sequences']} × {ev['seq_len']}` ({ev['scored_tokens']:,} scored tokens) |",
            f"| **Quantized Perplexity (PPL)** | `{ev['ppl']:.4f}` (mean NLL `{ev['mean_nll']:.6f}`) |",
        ]
        ref = ev.get("reference")
        if ref is not None:
            ci_lo, ci_hi = ref["delta_nll_ci95"]
            lines += [
                f"| **BF16 Reference PPL** | `{ref['ppl']:.4f}` (ΔPPL `{ref['delta_ppl']:+.4f}`, `{ref['delta_ppl_pct']:+.2f}%`) |",
                f"| **Paired ΔNLL (95% bootstrap CI)** | `{ref['delta_nll_mean']:+.6f}` (`[{ci_lo:+.6f}, {ci_hi:+.6f}]`) |",
                f"| **Full-Vocab fp64 KL (mean / median / p95 / p99)** | `{ref['kl_fp64_mean']:.6f}` / `{ref['kl_fp64_median']:.6f}` / `{ref['kl_fp64_p95']:.6f}` / `{ref['kl_fp64_p99']:.6f}` |",
                f"| **Top-1 Token Agreement vs Reference** | `{ref['top1_agreement_pct']:.2f}%` |",
            ]

    if tp is not None:
        lines += [
            "",
            "### Inference Throughput (`[MEASURED]`)",
            "",
            "| Metric | Value |",
            "| :--- | :--- |",
            f"| **Device / Hardware** | `{tp['device']}` — {tp['hardware']} |",
            f"| **Prefill ({tp['prefill_tokens']} tokens)** | `{tp['prefill_tok_per_sec']:.2f} tok/s` (`{tp['prefill_ms']:.2f} ms`) |",
            f"| **Decode ({tp['decode_steps']} steps, kv_cache={tp['decode_kv_cached']})** | `{tp['decode_tok_per_sec']:.2f} tok/s` (`{tp['decode_ms_per_tok']:.2f} ms/tok`) |",
        ]

    repro_cmd = cli_command or f"tfq3-eval --checkpoint ./{title} --inspect-only"
    lines += [
        "",
        "### Reproducibility & Provenance",
        "",
        f"- **Tooling:** `tfq3-eval` (tfq3 `{prov.get('tfq3_version', '3.1.0')}`, Python `{prov.get('python_version', platform.python_version())}`"
        + (f", PyTorch `{prov['torch_version']}`" if prov.get("torch_version") else "")
        + ")",
        "- **Command:**",
        "```bash",
        repro_cmd,
        "```",
        "",
    ]
    return "\n".join(lines)


def _sanitized_cli_command(args: argparse.Namespace) -> str:
    parts = ["tfq3-eval", "--checkpoint", f"./{os.path.basename(os.path.abspath(args.checkpoint))}"]
    if args.inspect_only:
        parts.append("--inspect-only")
    else:
        if args.ref_model:
            ref_lbl = os.path.basename(os.path.abspath(args.ref_model)) if os.path.isdir(args.ref_model) else args.ref_model
            parts += ["--ref-model", ref_lbl]
        if args.text_file:
            parts += ["--text-file", f"./{os.path.basename(args.text_file)}"]
        elif args.dataset:
            parts += ["--dataset", args.dataset]
        parts += ["--seq-len", str(args.seq_len), "--eval-seqs", str(args.eval_seqs)]
    if args.bench_throughput:
        parts += ["--bench-throughput", "--prefill-len", str(args.prefill_len), "--decode-steps", str(args.decode_steps)]
    if args.model_card:
        parts += ["--model-card", os.path.basename(args.model_card)]
    if args.json:
        parts += ["--json", os.path.basename(args.json)]
    return " ".join(parts)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="tfq3-eval",
        description="Inspect, evaluate, benchmark, and generate Hugging Face Model Cards for TFQ3/TFQ4 checkpoints.",
    )
    ap.add_argument("--checkpoint", required=True, help="Path to TFQ3/TFQ4/hybrid checkpoint directory")
    ap.add_argument("--ref-model", default=None, help="Optional BF16/FP16 reference HF model path or repo ID for paired KL & ΔNLL")
    ap.add_argument("--dataset", default="wikitext2", choices=["wikitext2", "c4"], help="Evaluation dataset (default: wikitext2)")
    ap.add_argument("--text-file", default=None, help="Optional local UTF-8 text file for offline PPL/KL evaluation")
    ap.add_argument("--seq-len", type=int, default=2048, help="Evaluation sequence length (default: 2048)")
    ap.add_argument("--eval-seqs", type=int, default=32, help="Number of evaluation sequences (0 = full split, default: 32)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"], help="Execution device (default: auto)")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "f16", "f32"], help="Compute dtype (default: bf16)")
    ap.add_argument("--cache-decoded", action="store_true", help="Cache decoded weights on CPU/fallback for repeated forward passes")
    ap.add_argument("--inspect-only", action="store_true", help="Offline shard/bpw/XXH64 inspection only (zero HF downloads, zero GPU)")
    ap.add_argument("--no-verify-checksum", action="store_true", help="Skip XXH64 shard checksum verification")
    ap.add_argument("--bench-throughput", action="store_true", help="Run timed prefill & KV-cached decode throughput benchmark")
    ap.add_argument("--prefill-len", type=int, default=256, help="Prompt length for throughput benchmark (default: 256)")
    ap.add_argument("--decode-steps", type=int, default=32, help="Autoregressive decode steps for throughput benchmark (default: 32)")
    ap.add_argument("--model-card", default=None, help="Write ready-to-paste Hugging Face Model Card markdown section to this path")
    ap.add_argument("--json", default=None, help="Write structured JSON report to this path")
    ap.add_argument("--model-name", default=None, help="Override display name in the Model Card header")
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    ap = build_arg_parser()
    args = ap.parse_args(argv)

    insp = inspect_checkpoint(args.checkpoint, verify_checksum=not args.no_verify_checksum)
    if insp["xxh64"]["errors"]:
        for err in insp["xxh64"]["errors"]:
            print(f"ERROR XXH64: {err}", file=sys.stderr)
        return 2

    provenance: dict[str, Any] = {
        "schema": "tfq-eval-v1",
        "tfq3_version": "3.1.0",
        "python_version": platform.python_version(),
        "machine": platform.machine(),
    }

    report: dict[str, Any] = {
        "inspection": insp,
        "evaluation": None,
        "throughput": None,
        "provenance": provenance,
    }

    if not args.inspect_only or args.bench_throughput:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        try:
            from tfq3.pytorch.tfq_torch import load_tfq_into_hf  # type: ignore
        except ImportError:
            _PT_DIR = os.path.join(_REPO_ROOT, "integrations", "pytorch")
            if _PT_DIR not in sys.path:
                sys.path.insert(0, _PT_DIR)
            from tfq_torch import load_tfq_into_hf  # type: ignore

        provenance["torch_version"] = torch.__version__
        device = _select_device(args.device)
        dtype = _select_dtype(args.dtype)
        quant_model = load_tfq_into_hf(
            args.checkpoint,
            device=device,
            dtype=dtype,
            cache_decoded=args.cache_decoded,
        )

        if not args.inspect_only:
            tok_src = args.checkpoint
            if not any(
                os.path.isfile(os.path.join(args.checkpoint, f))
                for f in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")
            ) and args.ref_model:
                tok_src = args.ref_model
            tok = AutoTokenizer.from_pretrained(tok_src, local_files_only=os.path.isdir(tok_src))
            ds_label, chunks = load_eval_chunks(
                tok,
                dataset_name=args.dataset,
                text_file=args.text_file,
                seq_len=args.seq_len,
                eval_seqs=args.eval_seqs,
            )
            ref_model = None
            if args.ref_model:
                ref_model = AutoModelForCausalLM.from_pretrained(
                    args.ref_model,
                    dtype=dtype,
                    local_files_only=os.path.isdir(args.ref_model),
                ).to(device=device)
            report["evaluation"] = evaluate_causal_lm(
                quant_model,
                chunks,
                ref_model=ref_model,
                device=device,
                dataset_label=ds_label,
            )
            if ref_model is not None:
                del ref_model

        if args.bench_throughput:
            vocab_sz = int(getattr(quant_model.config, "vocab_size", 32000))
            report["throughput"] = benchmark_throughput(
                quant_model,
                vocab_size=vocab_sz,
                device=device,
                prefill_len=args.prefill_len,
                decode_steps=args.decode_steps,
            )

    # Human-readable stdout summary
    print(f"Checkpoint: {insp['checkpoint_name']} ({insp['format']})")
    print(
        f"  Effective bpw: {insp['bpw']['quantized_effective']:.4f} quantized | "
        f"{insp['bpw']['overall_effective']:.4f} overall | {insp['disk']['total_gib']:.3f} GiB"
    )
    print(f"  Bits:          {insp['bit_histogram_units']}")
    print(
        f"  XXH64:         verified={insp['xxh64']['shards_verified']}/{insp['xxh64']['shards_checked']} "
        f"(legacy={insp['xxh64']['shards_legacy_unchecksummed']})"
    )
    if report["evaluation"] is not None:
        ev = report["evaluation"]
        print(f"  Eval ({ev['dataset']}): PPL={ev['ppl']:.4f} (NLL={ev['mean_nll']:.6f}, tokens={ev['scored_tokens']})")
        if "reference" in ev:
            ref = ev["reference"]
            print(
                f"  vs Reference:  ref_PPL={ref['ppl']:.4f} (ΔPPL={ref['delta_ppl']:+.4f}), "
                f"fp64_KL={ref['kl_fp64_mean']:.6f} (p99={ref['kl_fp64_p99']:.6f}), "
                f"Top-1={ref['top1_agreement_pct']:.2f}%"
            )
    if report["throughput"] is not None:
        tp = report["throughput"]
        print(
            f"  Throughput:    prefill={tp['prefill_tok_per_sec']:.2f} tok/s, "
            f"decode={tp['decode_tok_per_sec']:.2f} tok/s ({tp['decode_ms_per_tok']:.2f} ms/tok) on {tp['device']}"
        )

    if args.model_card:
        card_md = render_model_card(
            report,
            model_name=args.model_name,
            cli_command=_sanitized_cli_command(args),
        )
        out_dir = os.path.dirname(os.path.abspath(args.model_card))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.model_card, "w", encoding="utf-8") as f:
            f.write(card_md)
        print(f"Wrote Model Card -> {args.model_card}")

    if args.json:
        out_dir = os.path.dirname(os.path.abspath(args.json))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Wrote JSON report -> {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
