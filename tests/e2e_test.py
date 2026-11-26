#!/usr/bin/env python3
"""
End-to-end automated verification suite for TFQ3 (3/4/5-bit) and TFQ4-E8P (2-bit).

Verifies:
  1. Pure-Python reference codecs (`tfq3_reference.py`, `tfq4_lattice.py`)
  2. Strict `tfq3-inspect` validation on valid + corrupted PWV3 / PWV4 / TFPN / TFP4 files
     (bad magic, truncation, 16 KiB misalignment, section overlap, NaN/negative LUTs, duplicate names)
  3. CLI behavior (`bin/tfq3 --version`, `--help`, `doctor`, invalid `--bits` rejection, `--dry-run`, ERNIE detection)
  4. Full GPU converter round-trips (Metal on macOS / CUDA on Linux):
     - Multi-dtype Safetensors input (BF16, F16, F32 + 1D residuals)
     - Non-128-multiple zero-padding (`[192, 256]` -> `[256, 256]`)
     - Row-chunked encoding across the 4096-row boundary (`[4352, 512]`), checking SNR on rows [4096:4352]
     - TFQ3 3b / 4b / 5b + `--target-gib` exact decode parity vs `tfq3_reference.py`
     - TFQ4-E8P 2-bit (`--bits 2`) + multi-shard splitting + exact decode parity vs `tfq4_lattice.py`
"""
import argparse
import glob
import hashlib
import json
import math
import os
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "reference"))

import tfq3_reference as tfq3_ref
import tfq4_lattice as tfq4_ref

ALIGN_16K = 16384

GOLDEN_PWV3_SHA256 = "956c7abe5d400f302eaf8a2e9d0da50ee18f90d302de786cf6b2ec4905358bb7"
GOLDEN_PWV4_SHA256 = "ddf28171bd5ea33e2cddc560b5543f6890f2a179bb6dce7082a66243c04b7405"
GOLDEN_TFQ3_SHARD_SHA256 = {
    3: "8e6a166749b1d28e4cd20934c9b0d2113a2a846acdbfbaea6d0977e68915fa64",
    4: "c72b4d88b436404cbc5a21ea63d65419333d0f11fb14fcf5c03c7bb000f1a7d4",
    5: "cc09a1c7e46881470ea45dde41137cea564a3118d3eab1bfdb8e05f4948f2dac",
}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def align16k(n: int) -> int:
    return (n + ALIGN_16K - 1) & ~(ALIGN_16K - 1)


def run(cmd: list[str], expect_ok: bool | None = True, env: dict | None = None) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    if expect_ok and proc.returncode != 0:
        raise AssertionError(
            f"Command failed (rc={proc.returncode}): {' '.join(cmd)}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    if expect_ok is False and proc.returncode == 0:
        raise AssertionError(f"Command unexpectedly succeeded: {' '.join(cmd)}\nstdout:\n{proc.stdout}")
    return proc


def f32_to_bf16_bytes(arr: np.ndarray) -> bytes:
    u32 = arr.astype(np.float32, copy=False).view(np.uint32)
    # Round-to-nearest-even to bf16
    rounding = np.uint32(0x7FFF) + ((u32 >> np.uint32(16)) & np.uint32(1))
    bf16 = ((u32 + rounding) >> np.uint32(16)).astype(np.uint16)
    return bf16.tobytes()


def write_synthetic_model(model_dir: str) -> dict[str, np.ndarray]:
    """Create a small multi-dtype Safetensors model exercising padding, dtypes, and >4096 row-chunking."""
    os.makedirs(model_dir, exist_ok=True)
    rng = np.random.default_rng(12345)

    # BF16 [4352, 512]: crosses 4096-row chunk boundary + ~0.54 MiB in TFQ4
    w_chunk_a = (rng.standard_normal((4352, 512)).astype(np.float32) * 0.02)
    # F16 [4352, 512]: second chunked matrix so total TFQ4 > 1.0 MiB for multi-shard test
    w_chunk_b = (rng.standard_normal((4352, 512)).astype(np.float32) * 0.02)
    # F32 [192, 256]: non-128-multiple rows -> zero-padded to [256, 256]
    w_pad = (rng.standard_normal((192, 256)).astype(np.float32) * 0.03)
    # BF16 [256, 256]: standard square layer
    w_sq = (rng.standard_normal((256, 256)).astype(np.float32) * 0.025)
    # 1D residual norm weight (kept in residual.safetensors)
    w_norm = np.ones((512,), dtype=np.float32)

    tensors = [
        ("model.layers.0.mlp.gate_proj.weight", "BF16", w_chunk_a.shape, f32_to_bf16_bytes(w_chunk_a)),
        ("model.layers.0.mlp.up_proj.weight",   "F16",  w_chunk_b.shape, w_chunk_b.astype(np.float16).tobytes()),
        ("model.layers.0.self_attn.q_proj.weight", "F32", w_pad.shape,   w_pad.astype(np.float32).tobytes()),
        ("model.layers.0.self_attn.o_proj.weight", "BF16", w_sq.shape,   f32_to_bf16_bytes(w_sq)),
        ("model.layers.0.input_layernorm.weight",  "F32", w_norm.shape,  w_norm.tobytes()),
    ]

    header = {}
    payload = bytearray()
    for name, dtype, shape, raw in tensors:
        start = len(payload)
        payload.extend(raw)
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [start, len(payload)]}
    hdr_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with open(os.path.join(model_dir, "model.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", len(hdr_bytes)))
        f.write(hdr_bytes)
        f.write(payload)

    with open(os.path.join(model_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"model_type": "llama", "hidden_size": 512, "num_hidden_layers": 1}, f)
    with open(os.path.join(model_dir, "tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump({"version": "1.0"}, f)

    return {
        "model.layers.0.mlp.gate_proj": w_chunk_a,
        "model.layers.0.mlp.up_proj": w_chunk_b,
        "model.layers.0.self_attn.q_proj": w_pad,
        "model.layers.0.self_attn.o_proj": w_sq,
    }


def make_minimal_pwv4(path: str, bad_lut_u16: int | None = None) -> None:
    buf = bytearray(16384)
    struct.pack_into("<IIII", buf, 0, 0x34565750, 4, 256, 8)
    one_f16 = np.float16(0.5).view(np.uint16).item()
    for i in range(256 * 8):
        val = bad_lut_u16 if (bad_lut_u16 is not None and i == 17) else one_f16
        struct.pack_into("<H", buf, 64 + i * 2, val)
    with open(path, "wb") as f:
        f.write(buf)


def make_minimal_tfp4(
    path: str,
    names: list[str],
    corrupt_align: bool = False,
    corrupt_overlap: bool = False,
    truncate: bool = False,
) -> None:
    R, C = 128, 128
    n_sc = 1
    n_w = R * (C // 16)
    off_meta = 64
    off_scales = align16k(off_meta + len(names) * 128)
    off_weights = align16k(off_scales + len(names) * n_sc * 2)
    tot_sz = align16k(off_weights + len(names) * n_w * 4)

    hdr_off_scales = off_scales + 8 if corrupt_align else off_scales
    hdr_off_weights = off_scales if corrupt_overlap else off_weights

    buf = bytearray(tot_sz)
    struct.pack_into(
        "<IIIIQQQQQQ", buf, 0,
        0x34504654, 4, len(names), 0,
        off_meta, hdr_off_scales, hdr_off_weights,
        len(names) * n_sc, len(names) * n_w, 0,
    )
    for idx, nm in enumerate(names):
        rec = bytearray(128)
        nb = nm.encode("utf-8")[:79]
        rec[: len(nb)] = nb
        struct.pack_into(
            "<IIIIQQfIII", rec, 80,
            R, C, 2, idx * n_sc, idx * n_w, n_w, 10.1, R, C, 0,
        )
        buf[off_meta + idx * 128 : off_meta + (idx + 1) * 128] = rec

    if truncate:
        buf = buf[: tot_sz - 128]
    with open(path, "wb") as f:
        f.write(buf)


def test_inspector_negative_controls(inspect_bin: str, tmp: str) -> None:
    neg = os.path.join(tmp, "neg")
    os.makedirs(neg, exist_ok=True)

    # Valid PWV4 + valid TFP4 -> OK
    good_dir = os.path.join(neg, "good")
    os.makedirs(good_dir, exist_ok=True)
    make_minimal_pwv4(os.path.join(good_dir, "pw_v4_e8p_tables.bin"))
    make_minimal_tfp4(os.path.join(good_dir, "tfq4-00001-of-00001.tffp4"), ["layer.0.proj"])
    run([inspect_bin, "-v", good_dir], expect_ok=True)

    # Negative value in PWV4 e8p_lut -> FAIL
    neg_f16 = np.float16(-0.25).view(np.uint16).item()
    bad_pwv4_neg = os.path.join(neg, "bad_neg_pw_v4.bin")
    make_minimal_pwv4(bad_pwv4_neg, bad_lut_u16=neg_f16)
    run([inspect_bin, bad_pwv4_neg], expect_ok=False)

    # NaN in PWV4 e8p_lut -> FAIL
    bad_pwv4_nan = os.path.join(neg, "bad_nan_pw_v4.bin")
    make_minimal_pwv4(bad_pwv4_nan, bad_lut_u16=0x7E00)
    run([inspect_bin, bad_pwv4_nan], expect_ok=False)

    # Bad magic -> FAIL
    bad_magic = os.path.join(neg, "bad_magic.tffp4")
    with open(bad_magic, "wb") as f:
        f.write(b"NOPE" + b"\x00" * 16380)
    run([inspect_bin, bad_magic], expect_ok=False)

    # Misaligned off_scales -> FAIL
    bad_align = os.path.join(neg, "bad_align.tffp4")
    make_minimal_tfp4(bad_align, ["a"], corrupt_align=True)
    run([inspect_bin, bad_align], expect_ok=False)

    # Overlapping sections -> FAIL
    bad_overlap = os.path.join(neg, "bad_overlap.tffp4")
    make_minimal_tfp4(bad_overlap, ["a"], corrupt_overlap=True)
    run([inspect_bin, bad_overlap], expect_ok=False)

    # Truncated shard -> FAIL
    bad_trunc = os.path.join(neg, "bad_trunc.tffp4")
    make_minimal_tfp4(bad_trunc, ["a"], truncate=True)
    run([inspect_bin, bad_trunc], expect_ok=False)

    # Duplicate tensor name within one shard -> FAIL
    bad_dup_in = os.path.join(neg, "bad_dup_in.tffp4")
    make_minimal_tfp4(bad_dup_in, ["same.tensor", "same.tensor"])
    run([inspect_bin, bad_dup_in], expect_ok=False)

    # Duplicate tensor name across two shards in a directory -> FAIL
    dup_dir = os.path.join(neg, "dup_dir")
    os.makedirs(dup_dir, exist_ok=True)
    make_minimal_tfp4(os.path.join(dup_dir, "tfq4-00001-of-00002.tffp4"), ["shared.tensor"])
    make_minimal_tfp4(os.path.join(dup_dir, "tfq4-00002-of-00002.tffp4"), ["shared.tensor"])
    run([inspect_bin, dup_dir], expect_ok=False)

    print("PASS inspector strict validation & negative controls (9 cases)")


def make_minimal_tfpn(path: str, R: int = 128, C: int = 128, orig: tuple[int, int] | None = None,
                      words_extra: int = 0, pad_end: bool = True) -> None:
    bits = 3
    n_w = (R // 16) * (C // 128) * bits * 2 * 32
    n_sc = (R // 128) * (C // 128)
    off_meta, off_scales = 64, ALIGN_16K
    off_weights = align16k(off_scales + n_sc * 2)
    end = off_weights + n_w * 4
    buf = bytearray(align16k(end) if pad_end else end)
    struct.pack_into("<IIIIQQQQQQ", buf, 0, 0x4E504654, 3, 1, 0, off_meta, off_scales, off_weights, n_sc, n_w + words_extra, 0)
    o_r, o_c = orig if orig else (R, C)
    rec = bytearray(128)
    rec[:5] = b"layer"
    struct.pack_into("<IIIIQQfIII", rec, 80, R, C, bits, 0, 0, n_w, 20.0, o_r, o_c, 0)
    buf[off_meta:off_meta + 128] = rec
    with open(path, "wb") as f:
        f.write(buf)


def make_minimal_tfpw(path: str, num_tensors: int | None = None, words_extra: int = 0, pad_end: bool = True) -> None:
    E, H, I, bits = 2, 128, 128, 3
    per = (I // 16) * (H // 128) * bits * 2 * 32
    n_w, n_sc = E * 3 * per, E * 3
    off_meta, off_scales = 144, ALIGN_16K
    off_weights = align16k(off_scales + n_sc * 2)
    end = off_weights + n_w * 4
    buf = bytearray(align16k(end) if pad_end else end)
    struct.pack_into("<8I6Q", buf, 0, 0x57504654, 3, 0, E, 3 * E if num_tensors is None else num_tensors, H, I, 0,
                     0, n_w + words_extra, n_sc, off_meta, off_scales, off_weights)
    for e in range(E):
        w0, s0 = e * 3 * per, e * 3
        struct.pack_into("<IIQQQIIIfffII", buf, off_meta + e * 64, bits, e, w0, w0 + per, w0 + 2 * per,
                         s0, s0 + 1, s0 + 2, 20.0, 20.0, 20.0, 0, 0)
    with open(path, "wb") as f:
        f.write(buf)


def test_inspector_header_checks(inspect_bin: str, tmp: str) -> None:
    """Inspector must reject wrapped 64-bit sizes, bad pad fields, a wrong TFPW tensor count and unaligned file ends."""
    d = os.path.join(tmp, "hdr_checks")
    os.makedirs(d, exist_ok=True)
    wrap = 1 << 62  # x4 bytes wraps to 0 in 64 bits
    good = {
        "good.tffpw": lambda p: make_minimal_tfpn(p),
        "good_padded.tffpw": lambda p: make_minimal_tfpn(p, R=256, orig=(129, 100)),
        "good_moe.tffpw": lambda p: make_minimal_tfpw(p),
    }
    bad = {
        "tfpn_words_wrap.tffpw": lambda p: make_minimal_tfpn(p, words_extra=wrap),
        "tfpn_orig_r_over_r.tffpw": lambda p: make_minimal_tfpn(p, orig=(200, 128)),
        "tfpn_orig_c_too_small.tffpw": lambda p: make_minimal_tfpn(p, C=256, orig=(128, 100)),
        "tfpn_end_unaligned.tffpw": lambda p: make_minimal_tfpn(p, pad_end=False),
        "tfpw_words_wrap.tffpw": lambda p: make_minimal_tfpw(p, words_extra=wrap),
        "tfpw_num_tensors.tffpw": lambda p: make_minimal_tfpw(p, num_tensors=5),
        "tfpw_end_unaligned.tffpw": lambda p: make_minimal_tfpw(p, pad_end=False),
    }
    for name, make in good.items():
        make(os.path.join(d, name))
        run([inspect_bin, os.path.join(d, name)], expect_ok=True)
    tfp4_wrap = os.path.join(d, "tfp4_words_wrap.tffp4")
    make_minimal_tfp4(tfp4_wrap, ["a"])
    with open(tfp4_wrap, "r+b") as f:
        f.seek(48)
        n_w = struct.unpack("<Q", f.read(8))[0]
        f.seek(48)
        f.write(struct.pack("<Q", n_w + wrap))
    accepted = []
    for name, make in bad.items():
        make(os.path.join(d, name))
    for name in list(bad) + ["tfp4_words_wrap.tffp4"]:
        if run([inspect_bin, os.path.join(d, name)], expect_ok=None).returncode == 0:
            accepted.append(name)
    if accepted:
        raise AssertionError("tfq3-inspect accepted invalid shards: " + ", ".join(accepted))
    print(f"PASS inspector header checks ({len(good)} valid, {len(bad) + 1} invalid)")


def test_cli_basics(tfq3_cli: str, model_dir: str) -> None:
    ver = run([tfq3_cli, "--version"])
    assert "tfq3 3.0.0" in ver.stdout
    hlp = run([tfq3_cli, "--help"])
    assert "--bits" in hlp.stdout and "--target-gib" in hlp.stdout
    doc = run([tfq3_cli, "doctor"])
    assert "tfq3-inspect" in doc.stdout and "tfq4-convert" in doc.stdout
    # Reject invalid --bits
    run([tfq3_cli, model_dir, "--bits", "9", "--dry-run"], expect_ok=False)
    print("PASS CLI basics (--version, --help, doctor, invalid --bits rejection)")


def test_cli_ernie(tfq3_cli: str, tmp: str) -> None:
    # ERNIE detection and plan, against stub binaries (the converter is CUDA-only and not built here)
    src = os.path.join(tmp, "ernie_bf16")
    lib = os.path.join(tmp, "ernie_lib")
    os.makedirs(src)
    os.makedirs(lib)
    cfg = {"model_type": "ernie5_5_moe", "num_hidden_layers": 43}
    with open(os.path.join(src, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    for name in ("tfq3-ernie-convert-cuda", "tfq3-inspect"):
        p = os.path.join(lib, name)
        with open(p, "w", encoding="utf-8") as f:
            f.write("#!/bin/sh\nexit 0\n")
        os.chmod(p, 0o755)
    env = dict(os.environ, TFQ3_LIB_DIR=lib)
    if platform.system() == "Darwin":
        run([tfq3_cli, src, "--dry-run"], expect_ok=False, env=env)
        print("PASS CLI ERNIE (refused on Metal)")
        return
    out = run([tfq3_cli, src, "--dry-run"], env=env).stdout
    assert "family:   ernie" in out and "--layers 1 42 --bits 4" in out, out
    assert "--layers 1 42 --bits 3" in run([tfq3_cli, src, "--profile", "3b", "--dry-run"], env=env).stdout
    run([tfq3_cli, src, "--profile", "3.5b", "--dry-run"], expect_ok=False, env=env)
    run([tfq3_cli, src, "--family", "ernie", "--bits", "4", "--dry-run"], expect_ok=False, env=env)
    cfg["quantization_config"] = {"quant_method": "fp8", "expert_dtype": "mxfp4"}
    with open(os.path.join(src, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    run([tfq3_cli, src, "--dry-run"], expect_ok=False, env=env)
    print("PASS CLI ERNIE (detection, --profile widths, BF16-only and --bits refusals)")


def find_shard_for_tensor(out_dir: str, ext: str, tensor_name: str) -> str:
    for fp in sorted(glob.glob(os.path.join(out_dir, f"*{ext}"))):
        with open(fp, "rb") as f:
            hdr = struct.unpack("<IIIIQQQQQQ", f.read(64))
            nt, off_meta = hdr[2], hdr[4]
            f.seek(off_meta)
            for _ in range(nt):
                rec = f.read(128)
                nm = rec[:80].split(b"\x00", 1)[0].decode("utf-8")
                if nm == tensor_name:
                    return fp
    raise KeyError(f"{tensor_name} not found in any *{ext} in {out_dir}")


def test_gpu_roundtrips(tfq3_cli: str, inspect_bin: str, model_dir: str, refs: dict[str, np.ndarray], tmp: str) -> None:
    suffix = "" if platform.system() == "Darwin" else "-cuda"
    lib_dir = os.path.join(REPO_ROOT, "build")
    gen_bin = os.path.join(lib_dir, f"tfq3-generic-convert{suffix}")
    tfq4_bin = os.path.join(lib_dir, f"tfq4-convert{suffix}")

    if not (os.path.isfile(gen_bin) and os.path.isfile(tfq4_bin)):
        print(f"SKIP GPU round-trips (missing {gen_bin} or {tfq4_bin})")
        return

    # 1. TFQ3 generic round-trip across 3-bit, 4-bit, and 5-bit
    expected_min_snr = {3: 14.0, 4: 19.0, 5: 23.0}
    for bits, min_snr in expected_min_snr.items():
        out_dir = os.path.join(tmp, f"out_tfq3_{bits}b")
        run([tfq3_cli, model_dir, "--bits", str(bits), "--out", out_dir])
        run([inspect_bin, "-v", out_dir])

        mf = json.load(open(os.path.join(out_dir, "tfq3_manifest.json")))
        assert mf["format"] == "TFQ3" and mf["version"] == 3 and mf["tables"] == "pw_v3_tables.bin"
        assert mf["num_quantized_tensors"] == 4 and mf["num_residual_tensors"] == 1

        tables_path = os.path.join(out_dir, "pw_v3_tables.bin")
        assert sha256_file(tables_path) == GOLDEN_PWV3_SHA256, "pw_v3_tables.bin deviates from canonical SHA-256"
        shard_path = os.path.join(out_dir, "tfq3-00001-of-00001.tffpw")
        got_sha = sha256_file(shard_path)
        assert got_sha == GOLDEN_TFQ3_SHARD_SHA256[bits], (
            f"TFQ3 {bits}b cross-platform SHA-256 mismatch: got {got_sha}, expected {GOLDEN_TFQ3_SHARD_SHA256[bits]}"
        )

        # Check non-128-multiple padded tensor [192, 256] -> [256, 256]
        q_name = "model.layers.0.self_attn.q_proj"
        shard = find_shard_for_tensor(out_dir, ".tffpw", q_name)
        W_hat = tfq3_ref.dequantize_tfpn_tensor(shard, tables_path, q_name, max_blocks=4)
        assert W_hat.shape == (256, 256)
        snr_valid = tfq4_ref.snr_db(refs[q_name], W_hat[:192, :256])
        tail_rms = float(np.sqrt(np.mean(W_hat[192:, :] ** 2)))
        sig_rms = float(np.sqrt(np.mean(refs[q_name] ** 2)))
        assert snr_valid > min_snr, f"TFQ3 {bits}b padded tensor SNR too low: {snr_valid:.2f} dB"
        assert tail_rms < 0.25 * sig_rms, f"TFQ3 {bits}b zero-padded tail RMS too high: {tail_rms:.4f}"

        # Check >4096-row chunked tensor (especially rows 4096..4351 across r_off boundary!)
        g_name = "model.layers.0.mlp.gate_proj"
        shard_g = find_shard_for_tensor(out_dir, ".tffpw", g_name)
        W_g = tfq3_ref.dequantize_tfpn_tensor(shard_g, tables_path, g_name, max_blocks=64)
        assert W_g.shape == (4352, 512)
        snr_chunk0 = tfq4_ref.snr_db(refs[g_name][:4096, :], W_g[:4096, :])
        snr_chunk1 = tfq4_ref.snr_db(refs[g_name][4096:, :], W_g[4096:, :])
        assert snr_chunk0 > min_snr and snr_chunk1 > min_snr, (
            f"TFQ3 {bits}b row-chunked SNR mismatch: chunk0={snr_chunk0:.2f} dB, chunk1={snr_chunk1:.2f} dB"
        )
        print(f"PASS TFQ3 {bits}b GPU round-trip: padded={snr_valid:.2f} dB, chunk0={snr_chunk0:.2f} dB, chunk1(r>=4096)={snr_chunk1:.2f} dB")

    # 1b. A shard write on a full disk must fail the conversion
    if os.path.exists("/dev/full"):
        for conv, shard, extra in ((gen_bin, "tfq3-00001-of-00001.tffpw", ["--bits", "4"]),
                                   (tfq4_bin, "tfq4-00001-of-00002.tffp4", ["--max-shard-gib", "0.001"])):
            out_full = os.path.join(tmp, "out_full_" + shard.split("-")[0])
            os.makedirs(out_full)
            os.symlink("/dev/full", os.path.join(out_full, shard))
            proc = run([conv, "--model", model_dir, "--out", out_full] + extra, expect_ok=False)
            assert "Write error" in proc.stderr, proc.stderr
        print("PASS shard write on a full disk fails the TFQ3 and TFQ4 conversions")

    # 2. TFQ3 --target-gib dry-run plan check
    dry = run([tfq3_cli, model_dir, "--target-gib", "0.01", "--dry-run"])
    assert "TFQ3 Generic Plan" in dry.stdout

    # 3. TFQ4-E8P (--bits 2) GPU round-trip + multi-shard splitting + row-chunking parity
    out_tfq4 = os.path.join(tmp, "out_tfq4")
    run([tfq3_cli, model_dir, "--bits", "2", "--max-shard-gib", "0.001", "--out", out_tfq4])
    run([inspect_bin, "-v", out_tfq4])

    mf4 = json.load(open(os.path.join(out_tfq4, "tfq4_manifest.json")))
    assert mf4["format"] == "TFQ4-E8P" and mf4["version"] == 4 and mf4["tables"] == "pw_v4_e8p_tables.bin"
    assert len(mf4["tfq_shards"]) >= 2, f"Expected multi-shard TFP4 output, got {mf4['tfq_shards']}"
    assert mf4["num_quantized_tensors"] == 4 and mf4["num_residual_tensors"] == 1

    tbl4 = os.path.join(out_tfq4, "pw_v4_e8p_tables.bin")
    assert sha256_file(tbl4) == GOLDEN_PWV4_SHA256, "pw_v4_e8p_tables.bin deviates from canonical SHA-256"
    # Padded [192, 256] tensor
    q_name = "model.layers.0.self_attn.q_proj"
    shard_q = find_shard_for_tensor(out_tfq4, ".tffp4", q_name)
    W4_q = tfq4_ref.dequantize_tfp4_tensor(shard_q, tbl4, q_name)
    assert W4_q.shape == (256, 256)
    snr4_pad = tfq4_ref.snr_db(refs[q_name], W4_q[:192, :256])
    assert snr4_pad > 9.5, f"TFQ4 padded SNR too low: {snr4_pad:.2f} dB"

    # Row-chunked [4352, 512] BF16 & F16 tensors (verify r >= 4096 sign-vector parity!)
    for tname in ("model.layers.0.mlp.gate_proj", "model.layers.0.mlp.up_proj"):
        sh = find_shard_for_tensor(out_tfq4, ".tffp4", tname)
        W4 = tfq4_ref.dequantize_tfp4_tensor(sh, tbl4, tname)
        assert W4.shape == (4352, 512)
        s0 = tfq4_ref.snr_db(refs[tname][:4096, :], W4[:4096, :])
        s1 = tfq4_ref.snr_db(refs[tname][4096:, :], W4[4096:, :])
        assert s0 > 9.5 and s1 > 9.5, f"TFQ4 chunked SNR mismatch on {tname}: c0={s0:.2f} dB, c1={s1:.2f} dB"
    print(
        f"PASS TFQ4-E8P 2b GPU round-trip: {len(mf4['tfq_shards'])} shards, "
        f"padded={snr4_pad:.2f} dB, chunk0={s0:.2f} dB, chunk1(r>=4096)={s1:.2f} dB"
    )

    # 4. Bit-reproducibility check (Issue #2): repeated runs must produce byte-identical shards & manifests
    def assert_dirs_byte_identical(dir_a: str, dir_b: str, label: str) -> None:
        files_a = sorted(f for f in os.listdir(dir_a) if os.path.isfile(os.path.join(dir_a, f)))
        files_b = sorted(f for f in os.listdir(dir_b) if os.path.isfile(os.path.join(dir_b, f)))
        assert files_a == files_b, f"{label}: file list mismatch {files_a} vs {files_b}"
        for fn in files_a:
            ba = open(os.path.join(dir_a, fn), "rb").read()
            bb = open(os.path.join(dir_b, fn), "rb").read()
            assert ba == bb, f"{label}: non-deterministic bytes in {fn} ({len(ba)} vs {len(bb)} B)"

    out_tfq3_4b_r2 = os.path.join(tmp, "out_tfq3_4b_r2")
    run([tfq3_cli, model_dir, "--bits", "4", "--out", out_tfq3_4b_r2])
    assert_dirs_byte_identical(os.path.join(tmp, "out_tfq3_4b"), out_tfq3_4b_r2, "TFQ3-4b")

    out_tfq4_r2 = os.path.join(tmp, "out_tfq4_r2")
    run([tfq3_cli, model_dir, "--bits", "2", "--max-shard-gib", "0.001", "--out", out_tfq4_r2])
    assert_dirs_byte_identical(out_tfq4, out_tfq4_r2, "TFQ4-E8P")
    print("PASS bit-reproducibility: TFQ3-4b and TFQ4-E8P repeated runs are 100% byte-identical + cross-platform SHA-256 verified")

    # 5. Mismatched pre-existing pw_v3_tables.bin must be refused unless --tables is passed (Issue #18)
    out_bad_tbl = os.path.join(tmp, "out_bad_tbl")
    os.makedirs(out_bad_tbl, exist_ok=True)
    with open(os.path.join(out_bad_tbl, "pw_v3_tables.bin"), "wb") as f:
        bad_tbl = bytearray(open(os.path.join(REPO_ROOT, "reference", "tables", "pw_v3_tables.bin"), "rb").read())
        bad_tbl[128] ^= 0xFF
        f.write(bad_tbl)
    proc_bad_tbl = run([gen_bin, "--model", model_dir, "--out", out_bad_tbl, "--bits", "4"], expect_ok=False)
    assert "mismatch" in proc_bad_tbl.stderr.lower(), proc_bad_tbl.stderr
    print("PASS mismatched existing pw_v3_tables.bin refused without --tables")

    # 6. TFQ4 --head-bits semantics (Issue #10): default=0 keeps lm_head in residual; 2 quantizes; 4 rejected
    lm_dir = os.path.join(tmp, "lm_head_model")
    os.makedirs(lm_dir, exist_ok=True)
    rng = np.random.default_rng(77)
    w_proj = (rng.standard_normal((256, 256)).astype(np.float32) * 0.02)
    w_head = (rng.standard_normal((256, 256)).astype(np.float32) * 0.02)
    hdr, pay = {}, bytearray()
    for nm, arr in (("model.layers.0.self_attn.q_proj.weight", w_proj), ("lm_head.weight", w_head)):
        raw = f32_to_bf16_bytes(arr)
        hdr[nm] = {"dtype": "BF16", "shape": list(arr.shape), "data_offsets": [len(pay), len(pay) + len(raw)]}
        pay.extend(raw)
    hb = json.dumps(hdr, separators=(",", ":")).encode("utf-8")
    with open(os.path.join(lm_dir, "model.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", len(hb)) + hb + pay)
    with open(os.path.join(lm_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"model_type": "llama", "hidden_size": 256, "num_hidden_layers": 1}, f)

    out_h0 = os.path.join(tmp, "out_tfq4_h0")
    run([tfq3_cli, lm_dir, "--bits", "2", "--out", out_h0])
    mf_h0 = json.load(open(os.path.join(out_h0, "tfq4_manifest.json")))
    names_h0 = {t["name"] for t in mf_h0["tensors"]}
    assert mf_h0["num_quantized_tensors"] == 1 and mf_h0["num_residual_tensors"] == 1, mf_h0
    assert "lm_head" not in names_h0, mf_h0

    out_h2 = os.path.join(tmp, "out_tfq4_h2")
    run([tfq3_cli, lm_dir, "--bits", "2", "--head-bits", "2", "--out", out_h2])
    mf_h2 = json.load(open(os.path.join(out_h2, "tfq4_manifest.json")))
    names_h2 = {t["name"] for t in mf_h2["tensors"]}
    assert mf_h2["num_quantized_tensors"] == 2 and "lm_head" in names_h2, mf_h2

    run([tfq3_cli, lm_dir, "--bits", "2", "--head-bits", "4", "--out", os.path.join(tmp, "out_tfq4_h4")], expect_ok=False)
    run([tfq4_bin, "--model", lm_dir, "--head-bits", "4", "--out", os.path.join(tmp, "out_tfq4_h4b")], expect_ok=False)
    print("PASS TFQ4 --head-bits (default=0 keeps lm_head in residual, 2 quantizes, 4 rejected)")

    # 7. Non-finite weights (NaN / Inf) must fail both TFQ3 and TFQ4 converters nonzero (Issue #11)
    for bad_label, bad_val in (("nan", np.float32("nan")), ("inf", np.float32("inf"))):
        bad_dir = os.path.join(tmp, f"bad_{bad_label}_model")
        os.makedirs(bad_dir, exist_ok=True)
        arr = np.zeros((128, 128), dtype=np.float32)
        arr[7, 13] = bad_val
        raw = arr.tobytes()
        hdr_b = json.dumps({"model.layers.0.self_attn.q_proj.weight": {"dtype": "F32", "shape": [128, 128], "data_offsets": [0, len(raw)]}}).encode("utf-8")
        with open(os.path.join(bad_dir, "model.safetensors"), "wb") as f:
            f.write(struct.pack("<Q", len(hdr_b)) + hdr_b + raw)
        with open(os.path.join(bad_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump({"model_type": "llama", "hidden_size": 128, "num_hidden_layers": 1}, f)
        p3 = run([gen_bin, "--model", bad_dir, "--bits", "4", "--out", os.path.join(tmp, f"bad3_{bad_label}")], expect_ok=False)
        assert "non-finite" in p3.stderr.lower() and "[7, 13]" in p3.stderr, p3.stderr
        p4 = run([tfq4_bin, "--model", bad_dir, "--out", os.path.join(tmp, f"bad4_{bad_label}")], expect_ok=False)
        assert "non-finite" in p4.stderr.lower() and "[7, 13]" in p4.stderr, p4.stderr
    print("PASS NaN/Inf weights rejected nonzero by TFQ3 and TFQ4 converters with tensor + [row, col]")


def test_moe_datafree_ranking(tmp: str) -> None:
    """Compile & run a C++17 unit test for src/converters/moe_heat_rank.hpp proving non-iota sensitivity ranking."""
    cxx = "clang++" if platform.system() == "Darwin" else "g++"
    src = os.path.join(tmp, "test_moe_rank.cpp")
    exe = os.path.join(tmp, "test_moe_rank")
    hdr = os.path.join(REPO_ROOT, "src", "converters", "moe_heat_rank.hpp")
    with open(src, "w", encoding="utf-8") as f:
        f.write(
            f'#include "{hdr}"\n'
            '#include <cassert>\n'
            '#include <cstdio>\n'
            '#include <map>\n'
            'struct FakeCache {\n'
            '    std::map<std::string, std::vector<uint8_t>> m;\n'
            '    bool has_tensor(const std::string& k) const { return m.count(k) > 0; }\n'
            '    const uint8_t* get_tensor(const std::string& k, size_t* n = nullptr) {\n'
            '        const auto& v = m.at(k); if (n) *n = v.size(); return v.data();\n'
            '    }\n'
            '};\n'
            'int main() {\n'
            '    FakeCache c;\n'
            '    const int E = 8, H = 4;\n'
            '    std::vector<float> router(E * H, 0.1f);\n'
            '    // Make expert 6 have largest router norm, expert 3 second largest\n'
            '    for (int j = 0; j < H; ++j) { router[6 * H + j] = 5.0f; router[3 * H + j] = 2.5f; }\n'
            '    auto put_f32 = [&](const std::string& k, const std::vector<float>& v) {\n'
            '        const uint8_t* b = reinterpret_cast<const uint8_t*>(v.data());\n'
            '        c.m[k].assign(b, b + v.size() * sizeof(float));\n'
            '    };\n'
            '    put_f32("model.language_model.layers.3.mlp.gate.weight", router);\n'
            '    const char* projs[3] = {"gate_proj", "up_proj", "down_proj"};\n'
            '    for (int e = 0; e < E; ++e) {\n'
            '        float s = (e == 6) ? 4.0f : (e == 3 ? 2.0f : 0.2f);\n'
            '        for (int p = 0; p < 3; ++p) {\n'
            '            char nm[256];\n'
            '            std::snprintf(nm, sizeof(nm), "model.language_model.layers.3.mlp.experts.%d.%s.weight_scale_inv", e, projs[p]);\n'
            '            put_f32(nm, std::vector<float>(16, s));\n'
            '        }\n'
            '    }\n'
            '    bool used_router = false;\n'
            '    auto ranks = tfq_moe::rank_glm_experts_datafree(c, 3, E, H, &used_router);\n'
            '    assert(used_router);\n'
            '    assert(ranks[6] == 0 && ranks[3] == 1);\n'
            '    std::printf("PASS data-free MoE router+weight ranker (ranks[6]=%d, ranks[3]=%d)\\n", ranks[6], ranks[3]);\n'
            '    return 0;\n'
            '}\n'
        )
    run([cxx, "-O2", "-std=c++17", "-Wall", "-Wextra", "-Werror", src, "-o", exe])
    out = run([exe])
    print(out.stdout.strip())


def test_planner_shape_whitespace(tmp: str) -> None:
    """The generic planner must parse a Safetensors header with JSON whitespace inside a shape array."""
    cxx = "clang++" if platform.system() == "Darwin" else "g++"
    model_dir = os.path.join(tmp, "ws_model")
    os.makedirs(model_dir, exist_ok=True)
    nbytes = 128 * 256 * 4
    header = {"layer.weight": {"dtype": "F32", "shape": [128, 256], "data_offsets": [0, nbytes]}}
    hdr_bytes = json.dumps(header, indent=2).encode("utf-8")  # newline between the shape dims
    with open(os.path.join(model_dir, "model.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", len(hdr_bytes)))
        f.write(hdr_bytes)
        f.write(b"\x00" * nbytes)
    src = os.path.join(tmp, "test_planner_ws.cpp")
    exe = os.path.join(tmp, "test_planner_ws")
    hdr = os.path.join(REPO_ROOT, "src", "converters", "generic", "planner.hpp")
    with open(src, "w", encoding="utf-8") as f:
        f.write(
            f'#include "{hdr}"\n'
            '#include <cassert>\n'
            'int main(int, char** argv) {\n'
            '    tfq_generic::SafetensorsDir st;\n'
            '    st.init(argv[1]);\n'
            '    const auto& shape = st.tensors.at("layer.weight").shape;\n'
            '    assert(shape.size() == 2 && shape[0] == 128 && shape[1] == 256);\n'
            '    std::printf("PASS planner parses whitespace in Safetensors shapes ([%llu, %llu])\\n",\n'
            '                (unsigned long long)shape[0], (unsigned long long)shape[1]);\n'
            '    return 0;\n'
            '}\n'
        )
    run([cxx, "-O2", "-std=c++17", "-Wall", "-Wextra", "-Werror", src, "-o", exe])
    try:
        out = subprocess.run([exe, model_dir], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        raise AssertionError("planner hung parsing a Safetensors shape with a newline between dims")
    if out.returncode != 0:
        raise AssertionError(f"planner whitespace test failed (rc={out.returncode}):\n{out.stdout}\n{out.stderr}")
    print(out.stdout.strip())


def test_planner_fp8_scales(tmp: str) -> None:
    """FP8 weights with per-channel ([R,1] / [R]) and 128x128 block scales must decode with the right scale per row."""
    cxx = "clang++" if platform.system() == "Darwin" else "g++"
    model_dir = os.path.join(tmp, "fp8_model")
    os.makedirs(model_dir, exist_ok=True)
    R, C = 256, 128
    w = bytes([0x38]) * (R * C)  # F8_E4M3 1.0
    per_row = (np.arange(R, dtype=np.float32) + 1.0).tobytes()
    per_blk = np.array([[3.0], [5.0]], dtype=np.float32).tobytes()
    tensors = [
        ("a.weight", "F8_E4M3", [R, C], w),
        ("a.weight_scale", "F32", [R, 1], per_row),
        ("b.weight", "F8_E4M3", [R, C], w),
        ("b.weight_scale", "F32", [R], per_row),
        ("c.weight", "F8_E4M3", [R, C], w),
        ("c.weight_scale_inv", "F32", [2, 1], per_blk),
    ]
    header, payload = {}, bytearray()
    for name, dtype, shape, raw in tensors:
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [len(payload), len(payload) + len(raw)]}
        payload.extend(raw)
    hdr_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with open(os.path.join(model_dir, "model.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", len(hdr_bytes)))
        f.write(hdr_bytes)
        f.write(payload)
    src = os.path.join(tmp, "test_planner_fp8.cpp")
    exe = os.path.join(tmp, "test_planner_fp8")
    hdr = os.path.join(REPO_ROOT, "src", "converters", "generic", "planner.hpp")
    with open(src, "w", encoding="utf-8") as f:
        f.write(
            f'#include "{hdr}"\n'
            'int main(int, char** argv) {\n'
            '    using namespace tfq_generic;\n'
            '    SafetensorsDir st;\n'
            '    st.init(argv[1]);\n'
            '    GenericOptions opt;\n'
            '    opt.model_dir = argv[1];\n'
            '    ConversionPlan plan = build_plan(st, opt);\n'
            '    int bad = 0;\n'
            '    for (const auto& j : plan.jobs) {\n'
            '        std::vector<float> out;\n'
            '        decode_chunk_to_f32(st, j, 0, j.orig_R, out);\n'
            '        for (uint32_t r = 0; r < j.orig_R; ++r) {\n'
            '            float want = (j.tfpn_name == "c") ? ((r < 128) ? 3.0f : 5.0f) : (float)(r + 1);\n'
            '            float got = out[(size_t)r * j.pad_C + 5];\n'
            '            if (got != want) {\n'
            '                if (bad++ < 4) std::printf("%s row %u: got %g, want %g\\n", j.tfpn_name.c_str(), r, got, want);\n'
            '            }\n'
            '        }\n'
            '    }\n'
            '    if (plan.jobs.size() != 3 || bad) { std::printf("FAIL jobs=%zu bad_rows=%d\\n", plan.jobs.size(), bad); return 1; }\n'
            '    std::printf("PASS planner FP8 per-channel [R,1]/[R] and 128x128 block scales decode per row\\n");\n'
            '    return 0;\n'
            '}\n'
        )
    run([cxx, "-O2", "-std=c++17", "-Wall", "-Wextra", "-Werror", src, "-o", exe])
    out = run([exe, model_dir])
    print(out.stdout.strip())


def test_planner_budget_minimum(tmp: str) -> None:
    """A --target-gib below the all-3-bit footprint must be refused, not silently exceeded."""
    cxx = "clang++" if platform.system() == "Darwin" else "g++"
    model_dir = os.path.join(tmp, "budget_model")
    os.makedirs(model_dir, exist_ok=True)
    nbytes = 1024 * 1024 * 4
    header = {"model.layers.0.mlp.up_proj.weight": {"dtype": "F32", "shape": [1024, 1024], "data_offsets": [0, nbytes]}}
    hdr_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with open(os.path.join(model_dir, "model.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", len(hdr_bytes)))
        f.write(hdr_bytes)
        f.write(b"\x00" * nbytes)
    src = os.path.join(tmp, "test_planner_budget.cpp")
    exe = os.path.join(tmp, "test_planner_budget")
    hdr = os.path.join(REPO_ROOT, "src", "converters", "generic", "planner.hpp")
    with open(src, "w", encoding="utf-8") as f:
        f.write(
            f'#include "{hdr}"\n'
            'int main(int, char** argv) {\n'
            '    using namespace tfq_generic;\n'
            '    SafetensorsDir st;\n'
            '    st.init(argv[1]);\n'
            '    GenericOptions opt;\n'
            '    opt.model_dir = argv[1];\n'
            '    opt.out_dir = argv[1];\n'
            '    opt.target_gib = atof(argv[2]);\n'
            '    ConversionPlan plan = build_plan(st, opt);\n'
            '    print_plan_summary(opt, plan);\n'
            '    return 0;\n'
            '}\n'
        )
    run([cxx, "-O2", "-std=c++17", "-Wall", "-Wextra", "-Werror", src, "-o", exe])
    run([exe, model_dir, "0.01"])  # 3-bit minimum is ~0.0006 GiB: fits
    low = run([exe, model_dir, "0.0001"], expect_ok=False)
    assert "minimum" in low.stderr, low.stderr
    print("PASS planner refuses a --target-gib below the 3-bit minimum: " + low.stderr.strip())


def test_planner_write_errors(model_dir: str, tmp: str) -> None:
    """A failed residual/manifest write (disk full) must end with a nonzero exit, not a silent success."""
    if not os.path.exists("/dev/full"):
        print("SKIP planner write-error test (no /dev/full)")
        return
    cxx = "clang++" if platform.system() == "Darwin" else "g++"
    src = os.path.join(tmp, "test_planner_write.cpp")
    exe = os.path.join(tmp, "test_planner_write")
    hdr = os.path.join(REPO_ROOT, "src", "converters", "generic", "planner.hpp")
    with open(src, "w", encoding="utf-8") as f:
        f.write(
            f'#include "{hdr}"\n'
            'int main(int, char** argv) {\n'
            '    using namespace tfq_generic;\n'
            '    SafetensorsDir st;\n'
            '    st.init(argv[1]);\n'
            '    GenericOptions opt;\n'
            '    opt.model_dir = argv[1];\n'
            '    opt.out_dir = argv[2];\n'
            '    ConversionPlan plan = build_plan(st, opt);\n'
            '    package_residuals_and_manifest(st, opt, plan, {"tfq3-00001-of-00001.tffpw"});\n'
            '    return 0;\n'
            '}\n'
        )
    run([cxx, "-O2", "-std=c++17", "-Wall", "-Wextra", "-Werror", src, "-o", exe])
    ok_dir = os.path.join(tmp, "write_ok")
    os.makedirs(ok_dir)
    run([exe, model_dir, ok_dir])
    for target in ("residual.safetensors", "tfq3_manifest.json", "config.json"):
        out_dir = os.path.join(tmp, "write_full_" + target.split(".")[0])
        os.makedirs(out_dir)
        os.symlink("/dev/full", os.path.join(out_dir, target))
        proc = run([exe, model_dir, out_dir], expect_ok=False)
        assert "Write error" in proc.stderr, proc.stderr
    print("PASS planner write errors (residual, manifest, config copy on a full disk) exit nonzero")


def test_planner_mlx_bits(tmp: str) -> None:
    """MLX checkpoints quantized to other than 8 bits must be refused, not decoded as MLX 8-bit."""
    cxx = "clang++" if platform.system() == "Darwin" else "g++"

    def write_mlx(d: str, bits: int) -> None:
        os.makedirs(d, exist_ok=True)
        R, C, gs = 256, 256, 64
        cols = C * bits // 32
        tensors = [
            ("model.layers.0.mlp.up_proj.weight", "U32", [R, cols], bytes(R * cols * 4)),
            ("model.layers.0.mlp.up_proj.scales", "F16", [R, C // gs], bytes(R * (C // gs) * 2)),
            ("model.layers.0.mlp.up_proj.biases", "F16", [R, C // gs], bytes(R * (C // gs) * 2)),
        ]
        header, payload = {}, bytearray()
        for name, dtype, shape, raw in tensors:
            header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [len(payload), len(payload) + len(raw)]}
            payload.extend(raw)
        hdr_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
        with open(os.path.join(d, "model.safetensors"), "wb") as f:
            f.write(struct.pack("<Q", len(hdr_bytes)))
            f.write(hdr_bytes)
            f.write(payload)
        with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
            json.dump({"model_type": "llama", "quantization": {"group_size": gs, "bits": bits}}, f)

    src = os.path.join(tmp, "test_planner_mlx.cpp")
    exe = os.path.join(tmp, "test_planner_mlx")
    hdr = os.path.join(REPO_ROOT, "src", "converters", "generic", "planner.hpp")
    with open(src, "w", encoding="utf-8") as f:
        f.write(
            f'#include "{hdr}"\n'
            'int main(int, char** argv) {\n'
            '    using namespace tfq_generic;\n'
            '    SafetensorsDir st;\n'
            '    st.init(argv[1]);\n'
            '    GenericOptions opt;\n'
            '    opt.model_dir = argv[1];\n'
            '    ConversionPlan plan = build_plan(st, opt);\n'
            '    for (const auto& j : plan.jobs)\n'
            '        std::printf("%s %s [%u, %u] group=%u\\n", j.tfpn_name.c_str(), j.src_format.c_str(), j.orig_R, j.orig_C, j.group_size);\n'
            '    return 0;\n'
            '}\n'
        )
    run([cxx, "-O2", "-std=c++17", "-Wall", "-Wextra", "-Werror", src, "-o", exe])
    q8 = os.path.join(tmp, "mlx_q8")
    write_mlx(q8, 8)
    out = run([exe, q8])
    assert "MLX_Q8 [256, 256] group=64" in out.stdout, out.stdout
    q4 = os.path.join(tmp, "mlx_q4")
    write_mlx(q4, 4)
    bad = run([exe, q4], expect_ok=False)
    assert "4-bit" in bad.stderr, bad.stderr
    print("PASS planner reads MLX 8-bit and refuses MLX 4-bit: " + bad.stderr.strip())


def test_install_script(tmp: str) -> None:
    """install.sh with stub tools: failing nvidia-smi falls back, TFQ3_COMMIT is enforced, a truncated download runs nothing."""
    if platform.system() != "Linux":
        print("SKIP install.sh test (Linux CUDA path only)")
        return
    d = os.path.join(tmp, "install")
    stub = os.path.join(d, "bin")
    os.makedirs(stub)
    sha = "1" * 40
    stubs = {
        "nvidia-smi": "exit 9\n",
        "nvcc": "exit 0\n",
        "make": "exit 0\n",
        "git": f'if [ "${{1:-}}" = "-C" ]; then shift 2; fi\n[ "${{1:-}}" = "rev-parse" ] && echo {sha}\nexit 0\n',
    }
    for name, body in stubs.items():
        with open(os.path.join(stub, name), "w", encoding="utf-8") as f:
            f.write("#!/bin/sh\n" + body)
        os.chmod(os.path.join(stub, name), 0o755)
    env = dict(os.environ, PATH=stub + os.pathsep + os.environ.get("PATH", ""), NVCC=os.path.join(stub, "nvcc"),
               PREFIX=os.path.join(d, "prefix"), TFQ3_SRC_DIR=os.path.join(d, "src"))
    env.pop("CUDA_ARCH", None)
    script = os.path.join(REPO_ROOT, "install.sh")

    out = run(["bash", script], env=env)
    assert "CUDA_ARCH=native" in out.stdout, out.stdout + out.stderr
    run(["bash", script], env=dict(env, TFQ3_COMMIT=sha))
    bad = run(["bash", script], expect_ok=False, env=dict(env, TFQ3_COMMIT="2" * 40))
    assert "expected commit" in bad.stderr, bad.stderr

    with open(script, "rb") as f:
        body = f.read()
    cut = body.index(b'echo "==> Installing')  # download cut off before the install step
    trunc = subprocess.run(["bash"], input=body[:cut], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    assert trunc.returncode != 0 and b"==>" not in trunc.stdout, trunc.stdout.decode() + trunc.stderr.decode()
    print("PASS install.sh: nvidia-smi failure falls back to native, TFQ3_COMMIT enforced, truncated script runs nothing")


def test_canonical_tables(tmp: str) -> None:
    """Committed canonical tables match golden SHA-256s and tfq3-train-tables reproduces them byte-for-byte."""
    pwv3_ref = os.path.join(REPO_ROOT, "reference", "tables", "pw_v3_tables.bin")
    pwv4_ref = os.path.join(REPO_ROOT, "reference", "tables", "pw_v4_e8p_tables.bin")
    assert sha256_file(pwv3_ref) == GOLDEN_PWV3_SHA256, f"pw_v3_tables.bin hash mismatch: {sha256_file(pwv3_ref)}"
    assert sha256_file(pwv4_ref) == GOLDEN_PWV4_SHA256, f"pw_v4_e8p_tables.bin hash mismatch: {sha256_file(pwv4_ref)}"
    trainer = os.path.join(REPO_ROOT, "build", "tfq3-train-tables")
    if os.path.isfile(trainer):
        out_v3 = os.path.join(tmp, "regen_pw_v3.bin")
        out_v4 = os.path.join(tmp, "regen_pw_v4.bin")
        run([trainer, "--out-v3", out_v3, "--out-v4", out_v4])
        assert sha256_file(out_v3) == GOLDEN_PWV3_SHA256, "tfq3-train-tables PWV3 drifted from canonical SHA-256"
        assert sha256_file(out_v4) == GOLDEN_PWV4_SHA256, "tfq3-train-tables PWV4 drifted from canonical SHA-256"
        print("PASS canonical tables (PWV3 + PWV4 SHA-256 match committed files and tfq3-train-tables)")
    else:
        print("PASS canonical tables (committed PWV3 + PWV4 SHA-256 match golden hashes)")


def test_planner_nonfinite(tmp: str) -> None:
    """decode_chunk_to_f32 must refuse NaN/Inf in BF16, F16, F32, and FP8 weights with tensor name + [row, col]."""
    cxx = "clang++" if platform.system() == "Darwin" else "g++"
    src = os.path.join(tmp, "test_planner_nonfinite.cpp")
    exe = os.path.join(tmp, "test_planner_nonfinite")
    hdr = os.path.join(REPO_ROOT, "src", "converters", "generic", "planner.hpp")
    with open(src, "w", encoding="utf-8") as f:
        f.write(
            f'#include "{hdr}"\n'
            'int main(int, char** argv) {\n'
            '    using namespace tfq_generic;\n'
            '    SafetensorsDir st;\n'
            '    st.init(argv[1]);\n'
            '    GenericOptions opt;\n'
            '    opt.model_dir = argv[1];\n'
            '    ConversionPlan plan = build_plan(st, opt);\n'
            '    std::vector<float> out;\n'
            '    decode_chunk_to_f32(st, plan.jobs[0], 0, plan.jobs[0].orig_R, out);\n'
            '    return 0;\n'
            '}\n'
        )
    run([cxx, "-O2", "-std=c++17", "-Wall", "-Wextra", "-Werror", src, "-o", exe])
    cases = [
        ("f32_nan", "F32", np.float32("nan").tobytes()),
        ("f32_inf", "F32", np.float32("-inf").tobytes()),
        ("f16_nan", "F16", struct.pack("<H", 0x7E00)),
        ("bf16_inf", "BF16", struct.pack("<H", 0x7F80)),
        ("fp8_nan", "F8_E4M3", bytes([0x7F])),
    ]
    for label, dtype, bad_elem in cases:
        d = os.path.join(tmp, f"nf_{label}")
        os.makedirs(d, exist_ok=True)
        esz = len(bad_elem)
        raw = bytearray(128 * 128 * esz)
        pos = (5 * 128 + 9) * esz
        raw[pos : pos + esz] = bad_elem
        hdr_b = json.dumps({"layer.0.proj.weight": {"dtype": dtype, "shape": [128, 128], "data_offsets": [0, len(raw)]}}).encode("utf-8")
        with open(os.path.join(d, "model.safetensors"), "wb") as f:
            f.write(struct.pack("<Q", len(hdr_b)) + hdr_b + raw)
        proc = run([exe, d], expect_ok=False)
        assert "non-finite" in proc.stderr.lower() and "[5, 9]" in proc.stderr, f"{label}: {proc.stderr}"
    print(f"PASS planner rejects non-finite weights ({len(cases)} dtype/value cases)")


def test_flashnext_mlx_vlm_0712(tmp: str) -> None:
    """Verify FlashNext PR #26 fixes: package.py override preservation + shared Metal/CUDA PLE & --mtp helpers."""
    cxx = "clang++" if platform.system() == "Darwin" else "g++"
    src = os.path.join(tmp, "test_flashnext_common.cpp")
    exe = os.path.join(tmp, "test_flashnext_common")
    hdr = os.path.join(REPO_ROOT, "src", "converters", "flashnext", "flashnext_common.hpp")
    with open(src, "w", encoding="utf-8") as f:
        f.write(
            f'#include "{hdr}"\n'
            '#include <cassert>\n'
            '#include <set>\n'
            'struct FakeCache {\n'
            '    std::set<std::string> keys;\n'
            '    bool has_tensor(const std::string& k) const { return keys.count(k) > 0; }\n'
            '};\n'
            'int main() {\n'
            '    FakeCache c_old, c_new;\n'
            '    c_old.keys.insert("language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_7.weight");\n'
            '    c_new.keys.insert("language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.7.weight");\n'
            '    assert(tfq_flashnext::resolve_ple_src_stem(c_old, 7) ==\n'
            '           "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_7");\n'
            '    assert(tfq_flashnext::resolve_ple_src_stem(c_new, 7) ==\n'
            '           "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.7");\n'
            '    auto all48 = tfq_flashnext::build_layer_list(0, 47, false, false);\n'
            '    assert(all48.size() == 48 && all48.front() == 0 && all48.back() == 47);\n'
            '    auto mtp_only = tfq_flashnext::build_layer_list(0, 47, false, true);\n'
            '    assert(mtp_only.size() == 1 && mtp_only[0] == 48);\n'
            '    auto l47_mtp = tfq_flashnext::build_layer_list(47, 47, true, true);\n'
            '    assert(l47_mtp.size() == 2 && l47_mtp[0] == 47 && l47_mtp[1] == 48);\n'
            '    auto mid = tfq_flashnext::build_layer_list(6, 46, true, false);\n'
            '    assert(mid.size() == 41 && mid.front() == 6 && mid.back() == 46);\n'
            '    return 0;\n'
            '}\n'
        )
    run([cxx, "-O2", "-std=c++17", "-Wall", "-Wextra", "-Werror", src, "-o", exe])
    run([exe])

    # Test src/converters/flashnext/package.py per-tensor override preservation (mlx-vlm 0.7.12 layout)
    fn_src = os.path.join(tmp, "fn_src")
    fn_out = os.path.join(tmp, "fn_out")
    fn_ple = os.path.join(tmp, "fn_ple")
    os.makedirs(fn_src, exist_ok=True)
    os.makedirs(fn_ple, exist_ok=True)

    def write_st(path: str, items: list[tuple[str, str, list[int], bytes]]) -> None:
        h, pay = {}, bytearray()
        for nm, dt, sh, raw in items:
            s = len(pay)
            pay.extend(raw)
            h[nm] = {"dtype": dt, "shape": sh, "data_offsets": [s, len(pay)]}
        hb = json.dumps(h, separators=(",", ":")).encode("utf-8")
        with open(path, "wb") as fh:
            fh.write(struct.pack("<Q", len(hb)) + hb + pay)

    src_tensors = [
        ("language_model.model.layers.0.mlp.gate.weight", "U32", [4, 4], bytes(64)),
        ("language_model.model.layers.0.mlp.gate.scales", "BF16", [4, 1], bytes(8)),
        ("language_model.model.layers.0.mlp.gate.biases", "BF16", [4, 1], bytes(8)),
        ("language_model.model.layers.0.mlp.shared_expert_gate.weight", "U32", [4, 4], bytes(64)),
        ("language_model.model.layers.0.mlp.shared_expert_gate.scales", "BF16", [4, 1], bytes(8)),
        ("language_model.model.layers.0.mlp.shared_expert_gate.biases", "BF16", [4, 1], bytes(8)),
        ("language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight", "U32", [4, 4], bytes(64)),
        ("language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.0.weight", "U32", [4, 4], bytes(64)),
    ]
    write_st(os.path.join(fn_src, "model-00001-of-00001.safetensors"), src_tensors)
    with open(os.path.join(fn_src, "model.safetensors.index.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {"metadata": {"total_size": 0}, "weight_map": {k: "model-00001-of-00001.safetensors" for k, _, _, _ in src_tensors}},
            fh,
        )
    gate_ov = {"group_size": 64, "bits": 8, "mode": "affine"}
    src_q = {
        "group_size": 32,
        "bits": 8,
        "mode": "affine",
        "language_model.model.layers.0.mlp.gate": gate_ov,
        "language_model.model.layers.0.mlp.shared_expert_gate": gate_ov,
        "language_model.model.layers.0.mlp.switch_mlp.gate_proj": {"group_size": 64, "bits": 8, "mode": "affine"},
        "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.0": {"group_size": 32, "bits": 8, "mode": "affine"},
    }
    with open(os.path.join(fn_src, "config.json"), "w", encoding="utf-8") as fh:
        json.dump({"model_type": "qwen4_exp", "quantization": src_q, "text_config": {"quantization": src_q}}, fh)

    for grp in range(8):
        ple_items = []
        for s in range(grp * 16, grp * 16 + 16):
            st_nm = f"language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_{s}.weight"
            ple_items.append((st_nm, "U32", [2, 2], bytes(16)))
        write_st(os.path.join(fn_ple, f"ple_b3_{grp:02d}.safetensors"), ple_items)

    pkg_py = os.path.join(REPO_ROOT, "src", "converters", "flashnext", "package.py")
    run([sys.executable, pkg_py, "--src", fn_src, "--out", fn_out, "--extract-nonmoe", "--write-index",
         "--ple-bits", "3", "--ple-dir", fn_ple, "--move-ple"])

    out_cfg = json.load(open(os.path.join(fn_out, "config.json"), "r", encoding="utf-8"))
    out_q = out_cfg["quantization"]
    assert out_q["language_model.model.layers.0.mlp.gate"] == gate_ov, out_q
    assert out_q["language_model.model.layers.0.mlp.shared_expert_gate"] == gate_ov, out_q
    assert "language_model.model.layers.0.mlp.switch_mlp.gate_proj" not in out_q, out_q
    assert "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.0" not in out_q, out_q
    assert out_q["language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_0"] == {
        "group_size": 32, "bits": 3, "mode": "affine"
    }
    assert out_cfg["text_config"]["quantization"] == out_q
    out_idx = json.load(open(os.path.join(fn_out, "model.safetensors.index.json"), "r", encoding="utf-8"))["weight_map"]
    assert "language_model.model.layers.0.mlp.gate.weight" in out_idx
    assert "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight" not in out_idx
    assert "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.0.weight" not in out_idx
    assert "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_127.weight" in out_idx
    print("PASS FlashNext mlx-vlm 0.7.12 (PLE shards.N fallback, --layers+--mtp, package.py override preservation)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-gpu", action="store_true", help="Skip GPU converter round-trips (for headless CI runners)")
    args = ap.parse_args()

    inspect_bin = os.path.join(REPO_ROOT, "build", "tfq3-inspect")
    tfq3_cli = os.path.join(REPO_ROOT, "bin", "tfq3")
    if not os.path.isfile(inspect_bin):
        raise SystemExit(f"Missing {inspect_bin}; run 'make tools' or 'make' first")

    tfq3_ref.self_test()
    run([sys.executable, os.path.join(REPO_ROOT, "reference", "tfq4_lattice.py"), "--self-test"])

    with tempfile.TemporaryDirectory(prefix="tfq_e2e_") as tmp:
        model_dir = os.path.join(tmp, "synth_model")
        refs = write_synthetic_model(model_dir)
        test_canonical_tables(tmp)
        test_moe_datafree_ranking(tmp)
        test_flashnext_mlx_vlm_0712(tmp)
        test_planner_shape_whitespace(tmp)
        test_planner_fp8_scales(tmp)
        test_planner_nonfinite(tmp)
        test_planner_budget_minimum(tmp)
        test_planner_mlx_bits(tmp)
        test_inspector_negative_controls(inspect_bin, tmp)
        test_inspector_header_checks(inspect_bin, tmp)
        test_cli_basics(tfq3_cli, model_dir)
        test_planner_write_errors(model_dir, tmp)
        test_install_script(tmp)
        if not args.no_gpu:
            test_gpu_roundtrips(tfq3_cli, inspect_bin, model_dir, refs, tmp)
        test_cli_ernie(tfq3_cli, tmp)

    print("ALL E2E TESTS PASSED")


if __name__ == "__main__":
    main()
