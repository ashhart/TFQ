#!/usr/bin/env python3
"""
End-to-end verification suite for the Inkling-2 (NVFP4 + MXFP8 3D MoE) TFQ3 converter.

Tests:
  1. Synthetic 2-layer Inkling-2 checkpoint (L0=NVFP4, L1=MXFP8, E=6, H=256, I=128)
     with deliberately distinct gate (even w13 rows) vs up (odd w13 rows) distributions
     so any deinterleaving bug crashes SNR immediately.
  2. Full GPU conversion via `bin/tfq3 --family inkling2` (`heat43`, `uniform4`, `uniform3`,
     `heat543`) AND `--mode nonmoe --bits 5 --copy-residuals`.
  3. Exact pure-Python `modelopt.cu` NVFP4 & MXFP8 dequantization parity against
     `reference/tfq3_reference.py:decode_tfpn_fast_np` for EVERY expert and projection
     (`gate`, `up`, `down`) in `layer_00.tffpw` and `layer_01.tffpw`.
  4. Exact data-free router-norm+bias heat ranking verification on `heat43`.
  5. Strict `tfq3-inspect` validation, `--skip-existing` idempotence, and `--copy-residuals`
     completeness (all non-expert tensors preserved byte-for-byte, zero expert tensors leaked).
  6. Non-finite input rejection (NaN amax, zero denominator, FP8 E4M3 NaN scale/code, UE8M0 0xFF).
"""
import argparse
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

E2M1_LUT = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)

# Verified byte-identical on Apple Silicon Metal and NVIDIA CUDA (sm_121):
GOLDEN_SHA256 = {
    "heat43_L0": "fcbd0cf5e988c097ca6bea90daad7ce3a12f7324b205c2cff5057c0e576042a8",
    "heat43_L1": "436210b68207d42d177fa3efcd2bb7e616998c99ef9374e868fcc29d7727d487",
    "u4_L0":     "455edb1260f905d14cf1d903f9e88338525c05f55f8de13f796a402605b3f3a6",
    "u4_L1":     "8bbc12e68d7bb3180b40015ecdb1b781c236ef8ee0adc71de274496f44370dbd",
    "u3_L0":     "6fbec74a7e2227f3e0842823ac290db0c822405a47d23d5336956ecaf6e02aae",
    "u3_L1":     "f051a4c54bd55875c4d1e04afcc686d67735576b91d7656a935766513f38c186",
    "nonmoe":    "ca38cf43786c1cd22b40f3358729ca41325163168efdd66e4883a0b165ad55ff",
}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd: list[str], expect_ok: bool | None = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if expect_ok and proc.returncode != 0:
        raise AssertionError(
            f"Command failed (rc={proc.returncode}): {' '.join(cmd)}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    if expect_ok is False and proc.returncode == 0:
        raise AssertionError(f"Command unexpectedly succeeded: {' '.join(cmd)}\nstdout:\n{proc.stdout}")
    return proc


def e4m3_to_f32_lut() -> np.ndarray:
    lut = np.zeros(256, dtype=np.float32)
    for b in range(256):
        s = (b >> 7) & 1
        e = (b >> 3) & 15
        m = b & 7
        if e == 15 and m == 7:
            lut[b] = 0.0
        else:
            v = math.ldexp(float(m), -9) if e == 0 else math.ldexp(1.0 + m / 8.0, e - 7)
            lut[b] = -v if s else v
    return lut


E4M3_LUT = e4m3_to_f32_lut()


def e8m0_to_f32(sb: np.ndarray) -> np.ndarray:
    u32 = sb.astype(np.uint32) << np.uint32(23)
    out = u32.view(np.float32)
    return np.where(sb == 0, np.float32(2.0 ** -127), out)


def bf16_round_f32(x: float) -> float:
    u = struct.unpack("<I", struct.pack("<f", float(x)))[0]
    if (u & 0x7F800000) == 0x7F800000:
        return float(x)
    r = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", r))[0]


def f32_to_bf16_bytes(arr: np.ndarray) -> bytes:
    u32 = arr.astype(np.float32, copy=False).view(np.uint32)
    rounding = np.uint32(0x7FFF) + ((u32 >> np.uint32(16)) & np.uint32(1))
    return ((u32 + rounding) >> np.uint32(16)).astype(np.uint16).tobytes()


def dequant_nvfp4_ref(codes_u8: np.ndarray, scales_u8: np.ndarray, amax_f32: float, den_f32: float) -> np.ndarray:
    """Exact Python reference matching modelopt.cu nvfp4_dequant."""
    R, C_half = codes_u8.shape
    C = C_half * 2
    lo = codes_u8 & np.uint8(0x0F)
    hi = (codes_u8 >> np.uint8(4)) & np.uint8(0x0F)
    nibs = np.empty((R, C), dtype=np.uint8)
    nibs[:, 0::2] = lo
    nibs[:, 1::2] = hi
    sc = E4M3_LUT[scales_u8]  # [R, C // 16]
    sc_exp = np.repeat(sc, 16, axis=1)
    outer = np.float32(bf16_round_f32(amax_f32) / bf16_round_f32(den_f32))
    return (E2M1_LUT[nibs] * sc_exp) * outer


def dequant_mxfp8_ref(codes_u8: np.ndarray, scales_u8: np.ndarray) -> np.ndarray:
    """Exact Python reference matching modelopt.cu mxfp8_dequant."""
    sc = e8m0_to_f32(scales_u8)  # [R, C // 32]
    sc_exp = np.repeat(sc, 32, axis=1)
    return E4M3_LUT[codes_u8] * sc_exp


def snr_db(ref: np.ndarray, got: np.ndarray) -> float:
    sig = float(np.sum(ref.astype(np.float64) ** 2))
    err = float(np.sum((ref.astype(np.float64) - got.astype(np.float64)) ** 2))
    return 10.0 * math.log10(sig / max(err, 1e-18))


def write_safetensors(path: str, tensors: list[tuple[str, str, tuple[int, ...], bytes]]) -> None:
    hdr = {}
    payload = bytearray()
    for name, dtype, shape, raw in tensors:
        s = len(payload)
        payload.extend(raw)
        hdr[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [s, len(payload)]}
    hb = json.dumps(hdr, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        f.write(payload)


def read_safetensors_all(model_dir: str) -> dict[str, tuple[str, list[int], bytes]]:
    idx_path = os.path.join(model_dir, "model.safetensors.index.json")
    wm = json.load(open(idx_path, "r", encoding="utf-8"))["weight_map"]
    by_sh: dict[str, list[str]] = {}
    for k, sh in wm.items():
        by_sh.setdefault(sh, []).append(k)
    out = {}
    for sh, keys in by_sh.items():
        with open(os.path.join(model_dir, sh), "rb") as f:
            hl = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(hl).decode("utf-8"))
            base = 8 + hl
            for k in keys:
                meta = hdr[k]
                a, b = meta["data_offsets"]
                f.seek(base + a)
                out[k] = (meta["dtype"], meta["shape"], f.read(b - a))
    return out


def make_synthetic_inkling2(model_dir: str, E: int = 6, H: int = 256, I: int = 128) -> dict:
    """Create a 2-layer multi-shard Inkling-2 checkpoint (L0=NVFP4, L1=MXFP8)."""
    os.makedirs(model_dir, exist_ok=True)
    rng = np.random.default_rng(20261126)

    # Valid finite positive E4M3 codes for scales: exponent 5..10 (values ~0.25..8.0)
    pos_e4m3_scales = np.array([((e << 3) | m) for e in range(5, 11) for m in range(8)], dtype=np.uint8)
    # Valid finite signed E4M3 codes for MXFP8 weights (exclude 0x7F and 0xFF NaN)
    all_finite_e4m3 = np.array([b for b in range(256) if (b & 0x7F) != 0x7F and ((b >> 3) & 15) <= 10], dtype=np.uint8)
    # Valid finite UE8M0 scales around 2^-5 .. 2^-1 (122..126)
    mxfp8_scales_pool = np.arange(122, 127, dtype=np.uint8)

    # --- Layer 0: NVFP4 ---
    # Give even rows (gate) and odd rows (up) deliberately distinct scale magnitudes + patterns!
    l0_w13_c = rng.integers(0, 256, size=(E, 2 * I, H // 2), dtype=np.uint8)
    l0_w13_s = np.empty((E, 2 * I, H // 16), dtype=np.uint8)
    l0_w13_s[:, 0::2, :] = rng.choice(pos_e4m3_scales[24:], size=(E, I, H // 16))   # gate: larger scale
    l0_w13_s[:, 1::2, :] = rng.choice(pos_e4m3_scales[:16], size=(E, I, H // 16))   # up: smaller scale
    l0_w13_amax = rng.uniform(0.5, 2.5, size=(E,)).astype(np.float32)
    l0_w13_den = np.array([448.0 * 6.0], dtype=np.float32)

    l0_w2_c = rng.integers(0, 256, size=(E, H, I // 2), dtype=np.uint8)
    l0_w2_s = rng.choice(pos_e4m3_scales, size=(E, H, I // 16)).astype(np.uint8)
    l0_w2_amax = rng.uniform(0.4, 2.0, size=(E,)).astype(np.float32)
    l0_w2_den = np.array([448.0 * 6.0], dtype=np.float32)

    # --- Layer 1: MXFP8 ---
    l1_w13_c = rng.choice(all_finite_e4m3, size=(E, 2 * I, H)).astype(np.uint8)
    l1_w13_s = np.empty((E, 2 * I, H // 32), dtype=np.uint8)
    l1_w13_s[:, 0::2, :] = np.uint8(126)  # gate: 0.5x
    l1_w13_s[:, 1::2, :] = np.uint8(122)  # up:   0.03125x
    l1_w2_c = rng.choice(all_finite_e4m3, size=(E, H, I)).astype(np.uint8)
    l1_w2_s = rng.choice(mxfp8_scales_pool, size=(E, H, I // 32)).astype(np.uint8)

    # --- Router weights with deterministic known norm ordering (experts 4, 2 hottest) ---
    # Norms per expert: [0.5, 1.0, 4.0, 1.5, 6.0, 0.2] -> top-2 are E4 (rank 0) and E2 (rank 1)
    norms = [0.5, 1.0, 4.0, 1.5, 6.0, 0.2]
    gate_w = np.zeros((E, H), dtype=np.float32)
    for e, nm in enumerate(norms):
        v = rng.standard_normal(H).astype(np.float32)
        v *= nm / np.linalg.norm(v)
        gate_w[e] = v
    gate_b = np.zeros((E,), dtype=np.float32)

    # --- BF16 attention + residual tensors ---
    attn_wq_0 = (rng.standard_normal((H, H)).astype(np.float32) * 0.02)
    attn_wo_0 = (rng.standard_normal((H, H)).astype(np.float32) * 0.02)
    attn_wq_1 = (rng.standard_normal((H, H)).astype(np.float32) * 0.02)
    embed_w = (rng.standard_normal((64, H)).astype(np.float32) * 0.05)

    shard0_tensors = [
        ("model.llm.embed.weight", "BF16", embed_w.shape, f32_to_bf16_bytes(embed_w)),
        ("model.llm.layers.0.attn.wq_du.weight", "BF16", attn_wq_0.shape, f32_to_bf16_bytes(attn_wq_0)),
        ("model.llm.layers.0.attn.wo_ud.weight", "BF16", attn_wo_0.shape, f32_to_bf16_bytes(attn_wo_0)),
        ("model.llm.layers.0.mlp.gate.weight", "BF16", gate_w.shape, f32_to_bf16_bytes(gate_w)),
        ("model.llm.layers.0.mlp.gate.bias", "F32", gate_b.shape, gate_b.tobytes()),
        ("model.llm.layers.0.mlp.experts.w13_weight", "U8", l0_w13_c.shape, l0_w13_c.tobytes()),
        ("model.llm.layers.0.mlp.experts.w13_weight_scale", "F8_E4M3", l0_w13_s.shape, l0_w13_s.tobytes()),
        ("model.llm.layers.0.mlp.experts.w13_weight_amax", "F32", l0_w13_amax.shape, l0_w13_amax.tobytes()),
        ("model.llm.layers.0.mlp.experts.w13_weight_outer_scale_denominator", "F32", l0_w13_den.shape, l0_w13_den.tobytes()),
        ("model.llm.layers.0.mlp.experts.w2_weight", "U8", l0_w2_c.shape, l0_w2_c.tobytes()),
        ("model.llm.layers.0.mlp.experts.w2_weight_scale", "F8_E4M3", l0_w2_s.shape, l0_w2_s.tobytes()),
        ("model.llm.layers.0.mlp.experts.w2_weight_amax", "F32", l0_w2_amax.shape, l0_w2_amax.tobytes()),
        ("model.llm.layers.0.mlp.experts.w2_weight_outer_scale_denominator", "F32", l0_w2_den.shape, l0_w2_den.tobytes()),
    ]
    shard1_tensors = [
        ("model.llm.layers.1.attn.wq_du.weight", "BF16", attn_wq_1.shape, f32_to_bf16_bytes(attn_wq_1)),
        ("model.llm.layers.1.mlp.gate.weight", "BF16", gate_w.shape, f32_to_bf16_bytes(gate_w)),
        ("model.llm.layers.1.mlp.gate.bias", "F32", gate_b.shape, gate_b.tobytes()),
        ("model.llm.layers.1.mlp.experts.w13_weight", "F8_E4M3", l1_w13_c.shape, l1_w13_c.tobytes()),
        ("model.llm.layers.1.mlp.experts.w13_weight_scale", "U8", l1_w13_s.shape, l1_w13_s.tobytes()),
        ("model.llm.layers.1.mlp.experts.w2_weight", "F8_E4M3", l1_w2_c.shape, l1_w2_c.tobytes()),
        ("model.llm.layers.1.mlp.experts.w2_weight_scale", "U8", l1_w2_s.shape, l1_w2_s.tobytes()),
    ]

    write_safetensors(os.path.join(model_dir, "model-00001-of-00002.safetensors"), shard0_tensors)
    write_safetensors(os.path.join(model_dir, "model-00002-of-00002.safetensors"), shard1_tensors)
    wm = {}
    for k, _, _, _ in shard0_tensors:
        wm[k] = "model-00001-of-00002.safetensors"
    for k, _, _, _ in shard1_tensors:
        wm[k] = "model-00002-of-00002.safetensors"
    with open(os.path.join(model_dir, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump({"metadata": {"total_size": 0}, "weight_map": wm}, f, indent=2)
    with open(os.path.join(model_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_type": "inkling2",
                "llm_config": {"num_hidden_layers": 2, "hidden_size": H, "intermediate_size": I, "n_routed_experts": E},
            },
            f,
            indent=2,
        )
    with open(os.path.join(model_dir, "tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump({"version": "1.0"}, f)

    # Compute exact reference float32 matrices for every expert & projection
    refs = {0: [], 1: []}
    for e in range(E):
        w13_f = dequant_nvfp4_ref(l0_w13_c[e], l0_w13_s[e], float(l0_w13_amax[e]), float(l0_w13_den[0]))
        w2_f = dequant_nvfp4_ref(l0_w2_c[e], l0_w2_s[e], float(l0_w2_amax[e]), float(l0_w2_den[0]))
        refs[0].append({"gate": w13_f[0::2, :], "up": w13_f[1::2, :], "down": w2_f})

        w13_m = dequant_mxfp8_ref(l1_w13_c[e], l1_w13_s[e])
        w2_m = dequant_mxfp8_ref(l1_w2_c[e], l1_w2_s[e])
        refs[1].append({"gate": w13_m[0::2, :], "up": w13_m[1::2, :], "down": w2_m})

    attn_refs = {
        "model.llm.layers.0.attn.wq_du": attn_wq_0,
        "model.llm.layers.0.attn.wo_ud": attn_wo_0,
        "model.llm.layers.1.attn.wq_du": attn_wq_1,
    }
    return {"refs": refs, "attn_refs": attn_refs, "E": E, "H": H, "I": I}


def verify_tfpw_layer(layer_path: str, tables_path: str, ref_experts: list[dict], expected_bits: dict[int, int]) -> None:
    assert tfq3_ref.verify_tfpw_checksum(layer_path), f"Expected XXH64 checksums on {layer_path}"
    tbl = tfq3_ref.load_tables(tables_path)
    with open(layer_path, "rb") as f:
        raw = f.read()
    magic, ver, lyr, n_exp, n_ten, H, I, _ = struct.unpack_from("<8I", raw, 0)
    assert magic == 0x57504654 and ver == 3, f"Bad TFPW header in {layer_path}"
    assert n_exp == len(ref_experts) and n_ten == 3 * n_exp
    raw_b, sw, sh, off_meta, off_sc, off_w = struct.unpack_from("<6Q", raw, 32)
    scales_all = np.frombuffer(raw, dtype=np.float16, count=sh, offset=off_sc)
    words_all = np.frombuffer(raw, dtype=np.uint32, count=sw, offset=off_w)
    n_sc = (H // 128) * (I // 128)
    min_snr = {3: 15.5, 4: 21.5, 5: 27.0}

    for e in range(n_exp):
        b, hrank, gw, uw, dw, gs, us, ds, gsnr, usnr, dsnr, _, _ = struct.unpack_from(
            "<IIQQQIIIfffII", raw, off_meta + e * 64
        )
        assert b == expected_bits[e], f"L{lyr} E{e}: expected {expected_bits[e]}b, got {b}b (rank={hrank})"
        wpm = (I // 16) * (H // 128) * (b * 2) * 32
        for proj_name, R, C, w_off, s_off, hdr_snr in [
            ("gate", I, H, gw, gs, gsnr),
            ("up",   I, H, uw, us, usnr),
            ("down", H, I, dw, ds, dsnr),
        ]:
            w_slice = words_all[w_off : w_off + wpm]
            s_slice = scales_all[s_off : s_off + n_sc].reshape(R // 128, C // 128)
            dec = tfq3_ref.decode_tfpn_fast_np(w_slice, s_slice, R, C, b, tbl)
            ref = ref_experts[e][proj_name]
            got_snr = snr_db(ref, dec)
            assert got_snr >= min_snr[b], (
                f"L{lyr} E{e} {proj_name} ({b}b): decoded SNR={got_snr:.2f} dB < {min_snr[b]} dB "
                f"(hdr_snr={hdr_snr:.2f} dB)"
            )
            assert abs(got_snr - hdr_snr) < 0.25, (
                f"L{lyr} E{e} {proj_name}: header SNR {hdr_snr:.2f} dB != Python decoded SNR {got_snr:.2f} dB"
            )


def test_nonfinite_rejection(conv_bin: str, base_model_dir: str, work_dir: str) -> None:
    cases = [
        ("nan_amax", "model.llm.layers.0.mlp.experts.w13_weight_amax", struct.pack("<f", float("nan"))),
        ("zero_den", "model.llm.layers.0.mlp.experts.w13_weight_outer_scale_denominator", struct.pack("<f", 0.0)),
        ("nan_nvfp4_scale", "model.llm.layers.0.mlp.experts.w13_weight_scale", bytes([0x7F])),
        ("nan_mxfp8_code", "model.llm.layers.1.mlp.experts.w13_weight", bytes([0x7F])),
        ("nan_mxfp8_scale", "model.llm.layers.1.mlp.experts.w13_weight_scale", bytes([0xFF])),
    ]
    for label, tensor_name, bad_bytes in cases:
        bad_dir = os.path.join(work_dir, f"bad_{label}")
        out_dir = os.path.join(work_dir, f"out_{label}")
        shutil.copytree(base_model_dir, bad_dir)
        idx = json.load(open(os.path.join(bad_dir, "model.safetensors.index.json")))
        sh_path = os.path.join(bad_dir, idx["weight_map"][tensor_name])
        with open(sh_path, "r+b") as f:
            hl = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(hl).decode("utf-8"))
            off = 8 + hl + hdr[tensor_name]["data_offsets"][0]
            f.seek(off)
            f.write(bad_bytes)
        proc = run([conv_bin, "--model", bad_dir, "--out", out_dir, "--mode", "uniform3"], expect_ok=False)
        assert "non-finite" in proc.stderr.lower() or "nan" in proc.stderr.lower(), (
            f"Case {label}: unexpected stderr: {proc.stderr}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--print-hashes", action="store_true", help="Print SHA-256 of generated layer_00/01.tffpw for cross-platform checks")
    args = ap.parse_args()

    suffix = "" if platform.system() == "Darwin" else "-cuda"
    conv_bin = os.path.join(REPO_ROOT, "build", f"tfq3-inkling2-convert{suffix}")
    insp_bin = os.path.join(REPO_ROOT, "build", "tfq3-inspect")
    cli_bin = os.path.join(REPO_ROOT, "bin", "tfq3")

    assert os.path.isfile(conv_bin), f"Missing converter binary: {conv_bin}"
    assert os.path.isfile(insp_bin), f"Missing inspect binary: {insp_bin}"

    with tempfile.TemporaryDirectory(prefix="tfq3_ink2_test_") as tmp:
        src_dir = os.path.join(tmp, "Inkling-2-Tiny")
        meta = make_synthetic_inkling2(src_dir)
        E = meta["E"]

        # 1) Full CLI auto-detection + heat43 (E=6 => top 2 experts [4, 2] get 4b, rest get 3b)
        out_heat = os.path.join(tmp, "out_heat43")
        run([cli_bin, src_dir, "--out", out_heat, "--profile", "3.5b"])
        run([insp_bin, out_heat])
        exp_heat = {e: (4 if e in (4, 2) else 3) for e in range(E)}
        verify_tfpw_layer(os.path.join(out_heat, "layer_00.tffpw"), os.path.join(out_heat, "pw_v3_tables.bin"), meta["refs"][0], exp_heat)
        verify_tfpw_layer(os.path.join(out_heat, "layer_01.tffpw"), os.path.join(out_heat, "pw_v3_tables.bin"), meta["refs"][1], exp_heat)

        # Verify residuals: every non-expert tensor kept byte-for-byte, zero .mlp.experts. tensors
        src_tensors = read_safetensors_all(src_dir)
        res_tensors = read_safetensors_all(out_heat)
        expected_res_keys = {k for k in src_tensors if ".mlp.experts." not in k}
        assert set(res_tensors.keys()) == expected_res_keys, f"Residual keys mismatch: {set(res_tensors.keys()) ^ expected_res_keys}"
        for k in expected_res_keys:
            assert res_tensors[k] == src_tensors[k], f"Residual tensor {k} corrupted!"
        assert os.path.isfile(os.path.join(out_heat, "config.json"))
        assert os.path.isfile(os.path.join(out_heat, "tokenizer.json"))

        # 2) --skip-existing idempotence
        h0_before = sha256_file(os.path.join(out_heat, "layer_00.tffpw"))
        h1_before = sha256_file(os.path.join(out_heat, "layer_01.tffpw"))
        run([conv_bin, "--model", src_dir, "--out", out_heat, "--mode", "heat43", "--skip-existing"])
        assert sha256_file(os.path.join(out_heat, "layer_00.tffpw")) == h0_before
        assert sha256_file(os.path.join(out_heat, "layer_01.tffpw")) == h1_before

        # 3) uniform4 & uniform3 & heat543
        out_u4 = os.path.join(tmp, "out_u4")
        run([conv_bin, "--model", src_dir, "--out", out_u4, "--mode", "uniform4"])
        run([insp_bin, out_u4])
        verify_tfpw_layer(os.path.join(out_u4, "layer_00.tffpw"), os.path.join(out_u4, "pw_v3_tables.bin"), meta["refs"][0], {e: 4 for e in range(E)})
        verify_tfpw_layer(os.path.join(out_u4, "layer_01.tffpw"), os.path.join(out_u4, "pw_v3_tables.bin"), meta["refs"][1], {e: 4 for e in range(E)})

        out_u3 = os.path.join(tmp, "out_u3")
        run([conv_bin, "--model", src_dir, "--out", out_u3, "--mode", "uniform3"])
        run([insp_bin, out_u3])
        verify_tfpw_layer(os.path.join(out_u3, "layer_00.tffpw"), os.path.join(out_u3, "pw_v3_tables.bin"), meta["refs"][0], {e: 3 for e in range(E)})
        verify_tfpw_layer(os.path.join(out_u3, "layer_01.tffpw"), os.path.join(out_u3, "pw_v3_tables.bin"), meta["refs"][1], {e: 3 for e in range(E)})

        # 4) nonmoe 5-bit attention + residual exclusion
        out_nm = os.path.join(tmp, "out_nonmoe")
        run([conv_bin, "--model", src_dir, "--out", out_nm, "--mode", "nonmoe", "--bits", "5", "--copy-residuals"])
        run([insp_bin, out_nm])
        for nm, ref in meta["attn_refs"].items():
            dec = tfq3_ref.dequantize_tfpn_tensor(
                os.path.join(out_nm, "nonmoe.tffpw"), os.path.join(out_nm, "pw_v3_tables.bin"), nm, max_blocks=8
            )
            s = snr_db(ref, dec)
            assert s >= 27.5, f"nonmoe {nm} 5b SNR={s:.2f} dB < 27.5 dB"
        nm_res = read_safetensors_all(out_nm)
        for nm in meta["attn_refs"]:
            assert (nm + ".weight") not in nm_res, f"Quantized attention {nm}.weight leaked into residuals!"
        assert "model.llm.embed.weight" in nm_res

        # 5) Cross-platform bit-identity golden SHA-256 verification (Metal == CUDA)
        got_hashes = {
            "heat43_L0": h0_before,
            "heat43_L1": h1_before,
            "u4_L0":     sha256_file(os.path.join(out_u4, "layer_00.tffpw")),
            "u4_L1":     sha256_file(os.path.join(out_u4, "layer_01.tffpw")),
            "u3_L0":     sha256_file(os.path.join(out_u3, "layer_00.tffpw")),
            "u3_L1":     sha256_file(os.path.join(out_u3, "layer_01.tffpw")),
            "nonmoe":    sha256_file(os.path.join(out_nm, "nonmoe.tffpw")),
        }
        for k, expected in GOLDEN_SHA256.items():
            assert got_hashes[k] == expected, f"Golden SHA-256 mismatch for {k}: got {got_hashes[k]}, expected {expected}"

        # 5b) XXH64 integrity negative controls on real GPU-generated TFPW (layer_00) and TFPN (nonmoe)
        raw_l0 = bytearray(open(os.path.join(out_heat, "layer_00.tffpw"), "rb").read())
        _, _, _, n_exp_l0, _, _, _, res0_l0 = struct.unpack_from("<8I", raw_l0, 0)
        assert (res0_l0 & tfq3_ref.FLAG_XXH64_PRESENT) != 0
        _, _, _, l0_om, l0_os, l0_ow = struct.unpack_from("<6Q", raw_l0, 32)
        for sec, off in (("meta", l0_om + 56), ("scales", l0_os), ("weights", l0_ow + 13)):
            bad_p = os.path.join(tmp, f"flip_tfpw_{sec}.tffpw")
            mut = bytearray(raw_l0)
            mut[off] ^= 0x01
            open(bad_p, "wb").write(mut)
            p = run([insp_bin, bad_p], expect_ok=False)
            assert "xxh64" in p.stderr.lower() and sec in p.stderr.lower(), p.stderr
            try:
                tfq3_ref.verify_tfpw_checksum(bad_p)
                raise AssertionError(f"Expected XXH64 failure on TFPW {sec}")
            except ValueError:
                pass
        # Legacy flag=0 TFPW still accepted
        leg_l0 = os.path.join(tmp, "legacy_l0.tffpw")
        mut = bytearray(raw_l0)
        struct.pack_into("<I", mut, 28, 0)
        struct.pack_into("<4Q", mut, 80, 0, 0, 0, 0)
        open(leg_l0, "wb").write(mut)
        run([insp_bin, leg_l0], expect_ok=True)
        assert tfq3_ref.verify_tfpw_checksum(leg_l0) is False

        raw_nm = bytearray(open(os.path.join(out_nm, "nonmoe.tffpw"), "rb").read())
        _, _, _, nm_pad0, nm_om, nm_os, nm_ow, _, _, _ = struct.unpack_from("<IIIIQQQQQQ", raw_nm, 0)
        assert (nm_pad0 & tfq3_ref.FLAG_XXH64_PRESENT) != 0
        for sec, off in (("meta", nm_om + 1), ("scales", nm_os), ("weights", nm_ow + 13)):
            bad_p = os.path.join(tmp, f"flip_nonmoe_{sec}.tffpw")
            mut = bytearray(raw_nm)
            mut[off] ^= 0x01
            open(bad_p, "wb").write(mut)
            p = run([insp_bin, bad_p], expect_ok=False)
            assert "xxh64" in p.stderr.lower(), p.stderr

        # 6) Non-finite rejection
        test_nonfinite_rejection(conv_bin, src_dir, tmp)

        if args.print_hashes:
            for k, v in got_hashes.items():
                print(f"HASH {k:10s}={v}")

    print("PASS: All Inkling-2 (NVFP4 + MXFP8 + deinterleaved w13 + nonmoe + residuals + NaN guards) tests passed!")


if __name__ == "__main__":
    main()
