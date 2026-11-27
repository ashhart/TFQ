#!/usr/bin/env python3
"""
Verify the normative FORMAT_SPEC.md golden decoding test vectors under `reference/vectors/`
against:
  1. Pure-Python reference decoders (`reference/tfq3_reference.py` and `reference/tfq4_lattice.py`)
     - Micro-vectors: exact state_out + exact fp32 bit patterns
     - 128x128 block SHA-256
     - 256x384 full + sliced (r_off=128, bc0=1..3 trellis resume) SHA-256
  2. Apple Silicon Metal kernels (`build/tfq3-metal-test` and `build/tfq4-metal-test`) when present
  3. NVIDIA CUDA kernels (`build/tfq3-cuda-test` and `build/tfq4-cuda-test`) when present
"""
import argparse
import base64
import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
import zlib
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "reference"))

import tfq3_reference as tfq3_ref
import tfq4_lattice as tfq4_ref
from generate_vectors import decode_tfq4_array


def b64z(arr: np.ndarray) -> str:
    return base64.b64encode(zlib.compress(np.ascontiguousarray(arr).tobytes(), level=9)).decode("ascii")


def unb64z(s: str, dtype) -> np.ndarray:
    return np.frombuffer(zlib.decompress(base64.b64decode(s)), dtype=dtype).copy()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_f32(arr: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(arr, dtype="<f4").tobytes())


def verify_tfq3_reference(vec_path: str) -> tuple[dict, bytes]:
    doc = json.load(open(vec_path, "r", encoding="utf-8"))
    tbl_path = os.path.join(REPO_ROOT, doc["tables_file"])
    tbl_raw = open(tbl_path, "rb").read()
    assert sha256_bytes(tbl_raw) == doc["tables_sha256"], "PWV3 canonical table SHA-256 mismatch"
    tbl = tfq3_ref.load_tables(tbl_path)

    for b_str in ("3", "4", "5"):
        entry = doc["bit_widths"][b_str]
        bits = int(b_str)
        assert entry["bits"] == bits

        # 1. Micro-vectors: exact state_out and exact fp32 bits
        for i, mv in enumerate(entry["micro"]):
            if bits == 4:
                st_out, vals = tfq3_ref._decode_8_4b(mv["state_in"], mv["words"][0], tbl)
            elif bits == 3:
                st_out, vals = tfq3_ref._decode_8_3b(mv["state_in"], mv["m24"], tbl)
            else:
                st_out, vals = tfq3_ref._decode_8_5b(mv["state_in"], mv["hi8"], mv["lo32"], tbl)
            assert st_out == mv["state_out"], f"TFQ3 {bits}b micro[{i}] state_out mismatch"
            got_hex = [f"0x{int(u):08x}" for u in vals.astype("<f4").view(np.uint32)]
            assert got_hex == mv["values_u32_hex"], f"TFQ3 {bits}b micro[{i}] values mismatch: {got_hex} vs {mv['values_u32_hex']}"

        # 2. 128x128 full block SHA-256
        blk = entry["block_128x128"]
        w_blk = unb64z(blk["words_u32_z85_b64"], "<u4")
        sc_blk = np.array(blk["scales_f16_u16"], dtype=np.uint16).view(np.float16).reshape(1, 1)
        W_blk = tfq3_ref.decode_tfpn_fast_np(w_blk, sc_blk, 128, 128, bits, tbl)
        assert sha256_f32(W_blk) == blk["decoded_f32_sha256"], f"TFQ3 {bits}b 128x128 SHA-256 mismatch"

        # 3. 256x384 multi-block + r_off=128 + bc0=1..3 state resume SHA-256
        mt = entry["tensor_256x384_resume"]
        R, C = mt["R"], mt["C"]
        wpb = bits * 2
        w_m = unb64z(mt["words_u32_z85_b64"], "<u4")
        sc_m = np.array(mt["scales_f16_u16"], dtype=np.uint16).view(np.float16).reshape(R // 128, C // 128)
        W_full = tfq3_ref.decode_tfpn_fast_np(w_m, sc_m, R, C, bits, tbl)
        assert sha256_f32(W_full) == mt["full_decoded_f32_sha256"], f"TFQ3 {bits}b 256x384 full SHA-256 mismatch"

        strip_off = mt["r_off"] // 16
        w_sub_rows = w_m.reshape(R // 16, (C // 128) * wpb * 32)[strip_off:].reshape(-1)
        W_slice = tfq3_ref.decode_tfpn_fast_np(
            w_sub_rows, sc_m[mt["r_off"] // 128 :, :], R - mt["r_off"], C, bits, tbl,
            r_off=mt["r_off"], bc0=mt["bc0"], bc1=mt["bc1"],
        )
        assert sha256_f32(W_slice) == mt["slice_decoded_f32_sha256"], f"TFQ3 {bits}b 256x384 slice/resume SHA-256 mismatch"
        assert np.array_equal(W_full[mt["r_off"] :, mt["bc0"] * 128 : mt["bc1"] * 128], W_slice)

    print("PASS Python reference TFQ3 golden vectors (3b/4b/5b: 24 micro + 3x 128x128 + 3x 256x384 r_off & bc>0 resume)")
    return doc, tbl_raw


def verify_tfq4_reference(vec_path: str) -> tuple[dict, bytes]:
    doc = json.load(open(vec_path, "r", encoding="utf-8"))
    tbl_path = os.path.join(REPO_ROOT, doc["tables_file"])
    tbl_raw = open(tbl_path, "rb").read()
    assert sha256_bytes(tbl_raw) == doc["tables_sha256"], "PWV4 canonical table SHA-256 mismatch"
    lut = tfq4_ref.load_pwv4_tables(tbl_path)

    for i, mv in enumerate(doc["micro"]):
        w = mv["word_u32"]
        vals = []
        for code in (w & 0xFFFF, (w >> 16) & 0xFFFF):
            idx, sm = code & 0xFF, code >> 8
            for d in range(8):
                vals.append(float(lut[idx, d] * (-1.0 if ((sm >> d) & 1) else 1.0)))
        got_hex = [f"0x{int(u):08x}" for u in np.array(vals, dtype="<f4").view(np.uint32)]
        assert got_hex == mv["values_u32_hex"], f"TFQ4 micro[{i}] mismatch"

    blk = doc["block_128x128"]
    w_blk = unb64z(blk["words_u32_z85_b64"], "<u4").reshape(128, 128 // 16)
    sc_blk = np.array(blk["scales_f16_u16"], dtype=np.uint16).view(np.float16).reshape(1, 1)
    W_blk = decode_tfq4_array(w_blk, sc_blk, lut)
    assert sha256_f32(W_blk) == blk["decoded_f32_sha256"], "TFQ4 128x128 SHA-256 mismatch"

    mt = doc["tensor_256x384_offset"]
    R, C = mt["R"], mt["C"]
    w_m = unb64z(mt["words_u32_z85_b64"], "<u4").reshape(R, C // 16)
    sc_m = np.array(mt["scales_f16_u16"], dtype=np.uint16).view(np.float16).reshape(R // 128, C // 128)
    W_full = decode_tfq4_array(w_m, sc_m, lut)
    assert sha256_f32(W_full) == mt["full_decoded_f32_sha256"], "TFQ4 256x384 full SHA-256 mismatch"
    W_slice = decode_tfq4_array(
        w_m[mt["r_off"] :, mt["c_off"] // 16 :],
        sc_m[mt["r_off"] // 128 :, mt["c_off"] // 128 :],
        lut, r_off=mt["r_off"], c_off=mt["c_off"],
    )
    assert sha256_f32(W_slice) == mt["slice_decoded_f32_sha256"], "TFQ4 256x384 slice SHA-256 mismatch"

    print("PASS Python reference TFQ4 golden vectors (table_kind=0: 8 micro + 128x128 + 256x384 offset)")
    return doc, tbl_raw


TRUE_E8P_TABLE_SHA256 = "f1698d7e01bb0261f6b637c1a02b6965c7b21e6226c9790a5201283f44bf27e2"
TRUE_E8P_MICRO_SHA256 = "0e94cc8e315277ea14051c6d6e0c26f0c65202a0f3749e08d59b141139f99072"
TRUE_E8P_BLK128_SHA256 = "b9e961f1d4b80222cf11d78bef91beccc4969ed238074e2a05dca5d7b9a077c2"
TRUE_E8P_FULL256_SHA256 = "c06c5934c98c39871a90c6f142dd456ababb5b66c698f141c0235414921333a4"
TRUE_E8P_SLICE_SHA256 = "9cb890affd63243f15f21a16e99298dcee12dfb9595ad490c5ef11fd395a8329"


def decode_tfq4_true_e8p_array(
    words: np.ndarray, scales: np.ndarray, lut: np.ndarray, r_off: int = 0, c_off: int = 0
) -> np.ndarray:
    R, wpr = words.shape
    C = wpr * 16
    codes = np.empty((R, C // 8), dtype=np.uint16)
    codes[:, 0::2] = (words & np.uint32(0xFFFF)).astype(np.uint16)
    codes[:, 1::2] = (words >> np.uint32(16)).astype(np.uint16)
    W_rot = tfq4_ref.decode_true_e8p_codes(codes, lut).reshape(R, C)
    sc_tile = np.repeat(np.repeat(scales.astype(np.float32), 128, axis=0), 128, axis=1)
    scaled = W_rot * sc_tile
    sr = tfq3_ref.pw_sgn(tfq3_ref.SEED_ROW, np.arange(r_off, r_off + R, dtype=np.uint32))
    sc_v = tfq3_ref.pw_sgn(tfq3_ref.SEED_COL, np.arange(c_off, c_off + C, dtype=np.uint32))
    tmp = tfq3_ref.fwht_128(scaled.reshape(R // 128, 128, C).transpose(0, 2, 1)).transpose(0, 2, 1).reshape(R, C)
    tmp = tmp * sr[:, None]
    out = tfq3_ref.fwht_128(tmp.reshape(R, C // 128, 128)).reshape(R, C) * sc_v[None, :]
    return out.astype(np.float32, copy=False)


def verify_tfq4_true_e8p_reference() -> tuple[dict, bytes]:
    abs_cb, m_par = tfq4_ref.build_true_e8p_abs_codebook()
    buf = bytearray(16384)
    struct.pack_into("<IIIII", buf, 0, tfq4_ref.MAGIC_PWV4, tfq4_ref.TFQ4_VER, 256, 8, tfq4_ref.PWV4_TABLE_TRUE_E8P)
    lut_f16 = np.ascontiguousarray(abs_cb, dtype="<f2")
    buf[64 : 64 + lut_f16.nbytes] = lut_f16.tobytes()
    tbl_raw = bytes(buf)
    assert sha256_bytes(tbl_raw) == TRUE_E8P_TABLE_SHA256, "PWV4 table_kind=2 canonical table SHA-256 mismatch"

    with tempfile.TemporaryDirectory(prefix="tfq4_k2_tbl_") as tmp:
        tbl_path = os.path.join(tmp, "pw_v4_true_e8p_tables.bin")
        with open(tbl_path, "wb") as f:
            f.write(tbl_raw)
        lut, kind = tfq4_ref.load_pwv4_tables_with_kind(tbl_path)
    assert kind == tfq4_ref.PWV4_TABLE_TRUE_E8P, f"Expected table_kind=2, got {kind}"

    rng = np.random.default_rng(20260403)
    test_words = [0x00000000, 0xFFFFFFFF, 0xA53C5A81, 0x80007FFF, 0x7FFF8000] + [
        int(rng.integers(0, 1 << 32, dtype=np.uint64)) for _ in range(3)
    ]
    micro_vals = []
    for i, w in enumerate(test_words):
        scalar_vals = []
        for code in (w & 0xFFFF, (w >> 16) & 0xFFFF):
            idx = code & 0xFF
            s7 = (code >> 8) & 0x7F
            coset = (code >> 15) & 1
            s7_bit = (bin(s7).count("1") ^ int(m_par[idx])) & 1
            sm8 = s7 | (s7_bit << 7)
            c_sgn = -1.0 if coset else 1.0
            for d in range(8):
                sgn = -1.0 if ((sm8 >> d) & 1) else 1.0
                scalar_vals.append(float(c_sgn * (float(lut[idx, d]) * sgn - 0.25)))
        scalar_arr = np.array(scalar_vals, dtype="<f4")
        vec_arr = tfq4_ref.decode_true_e8p_codes(np.array([w & 0xFFFF, (w >> 16) & 0xFFFF], dtype=np.uint16), lut)
        assert np.array_equal(scalar_arr, vec_arr.reshape(-1)), f"TFQ4 true-E8P micro[{i}] scalar vs vec mismatch"
        assert float(np.max(np.abs(vec_arr - tfq4_ref.nearest_e8_quarter(vec_arr)))) == 0.0, (
            f"TFQ4 true-E8P micro[{i}] outside E8 + 1/4 lattice"
        )
        micro_vals.append(scalar_arr)
    assert sha256_f32(np.concatenate(micro_vals)) == TRUE_E8P_MICRO_SHA256, "TFQ4 true-E8P micro SHA-256 mismatch"

    w_blk = rng.integers(0, 1 << 32, size=(128, 128 // 16), dtype=np.uint32)
    sc_blk = np.array([[0.0375]], dtype=np.float16)
    W_blk = decode_tfq4_true_e8p_array(w_blk, sc_blk, lut)
    assert sha256_f32(W_blk) == TRUE_E8P_BLK128_SHA256, "TFQ4 true-E8P 128x128 SHA-256 mismatch"
    X_blk = (rng.standard_normal((4, 128)).astype(np.float32) * 0.25).astype(np.float16)
    Y_blk = (X_blk.astype(np.float64) @ W_blk.astype(np.float64).T).astype(np.float32)

    R_m, C_m, r_off, c_off = 256, 384, 128, 128
    w_multi = rng.integers(0, 1 << 32, size=(R_m, C_m // 16), dtype=np.uint32)
    sc_multi = np.array([[0.029, 0.041, 0.033], [0.048, 0.036, 0.051]], dtype=np.float16)
    W_full = decode_tfq4_true_e8p_array(w_multi, sc_multi, lut)
    assert sha256_f32(W_full) == TRUE_E8P_FULL256_SHA256, "TFQ4 true-E8P 256x384 full SHA-256 mismatch"
    W_slice = decode_tfq4_true_e8p_array(
        w_multi[r_off:, c_off // 16 :],
        sc_multi[r_off // 128 :, c_off // 128 :],
        lut, r_off=r_off, c_off=c_off,
    )
    assert sha256_f32(W_slice) == TRUE_E8P_SLICE_SHA256, "TFQ4 true-E8P 256x384 slice SHA-256 mismatch"
    assert np.array_equal(W_full[r_off:, c_off:], W_slice), "TFQ4 true-E8P 256x384 slice diverged from full decode"

    X_multi = (rng.standard_normal((4, C_m)).astype(np.float32) * 0.25).astype(np.float16)
    Y_full = (X_multi.astype(np.float64) @ W_full.astype(np.float64).T).astype(np.float32)
    Y_slice = (X_multi[:, c_off:].astype(np.float64) @ W_slice.astype(np.float64).T).astype(np.float32)

    doc = {
        "format": "TFQ4-TRUE-E8P",
        "version": 4,
        "table_kind": tfq4_ref.PWV4_TABLE_TRUE_E8P,
        "block_128x128": {
            "R": 128,
            "C": 128,
            "words_u32_z85_b64": b64z(w_blk.astype("<u4")),
            "scales_f16_u16": [int(x) for x in sc_blk.view(np.uint16).reshape(-1)],
            "probe_X_f16_b64": b64z(X_blk.astype("<f2")),
            "probe_Y_f32_b64": b64z(Y_blk.astype("<f4")),
        },
        "tensor_256x384_offset": {
            "R": R_m,
            "C": C_m,
            "r_off": r_off,
            "c_off": c_off,
            "words_u32_z85_b64": b64z(w_multi.astype("<u4")),
            "scales_f16_u16": [int(x) for x in sc_multi.view(np.uint16).reshape(-1)],
            "probe_X_f16_b64": b64z(X_multi.astype("<f2")),
            "probe_Y_full_f32_b64": b64z(Y_full.astype("<f4")),
            "probe_Y_slice_f32_b64": b64z(Y_slice.astype("<f4")),
        },
    }
    print("PASS Python reference TFQ4 golden vectors (table_kind=2 true-E8P: 8 micro + 128x128 + 256x384 offset)")
    return doc, tbl_raw


def write_vec3_bundle(path: str, doc: dict, tbl_raw: bytes) -> None:
    cases = []
    for b_str in ("3", "4", "5"):
        bits = int(b_str)
        entry = doc["bit_widths"][b_str]
        blk = entry["block_128x128"]
        w_blk = unb64z(blk["words_u32_z85_b64"], "<u4")
        sc_blk = np.array(blk["scales_f16_u16"], dtype="<u2")
        X_blk = unb64z(blk["probe_X_f16_b64"], "<u2")
        Y_blk = unb64z(blk["probe_Y_f32_b64"], "<f4")
        cases.append((bits, 128, 128, 4, 0, 0, 128, 128, w_blk, sc_blk, X_blk, Y_blk))

        mt = entry["tensor_256x384_resume"]
        R, C = mt["R"], mt["C"]
        w_m = unb64z(mt["words_u32_z85_b64"], "<u4")
        sc_m = np.array(mt["scales_f16_u16"], dtype="<u2")
        X_m = unb64z(mt["probe_X_f16_b64"], "<u2").reshape(4, C)
        Y_full = unb64z(mt["probe_Y_full_f32_b64"], "<f4")
        Y_slice = unb64z(mt["probe_Y_slice_f32_b64"], "<f4")
        cases.append((bits, R, C, 4, 0, 0, R, C, w_m, sc_m, X_m.reshape(-1), Y_full))
        r_off = mt["r_off"]
        c_off = mt["bc0"] * 128
        k_loc = (mt["bc1"] - mt["bc0"]) * 128
        X_sub = np.ascontiguousarray(X_m[:, c_off : c_off + k_loc]).reshape(-1)
        cases.append((bits, R, C, 4, r_off, c_off, R - r_off, k_loc, w_m, sc_m, X_sub, Y_slice))

    with open(path, "wb") as f:
        f.write(struct.pack("<II", 0x56454333, len(cases)))
        f.write(tbl_raw)
        for bits, R, C, rows, r_off, c_off, n_loc, k_loc, W, sc, X, Y in cases:
            f.write(struct.pack("<10I", bits, R, C, rows, r_off, c_off, n_loc, k_loc, len(W), len(sc)))
            f.write(W.tobytes())
            f.write(sc.tobytes())
            f.write(X.tobytes())
            f.write(Y.tobytes())


def write_vec4_bundle(path: str, doc: dict, tbl_raw: bytes) -> None:
    cases = []
    blk = doc["block_128x128"]
    cases.append((
        2, 128, 128, 4, 0, 0, 128, 128,
        unb64z(blk["words_u32_z85_b64"], "<u4"),
        np.array(blk["scales_f16_u16"], dtype="<u2"),
        unb64z(blk["probe_X_f16_b64"], "<u2"),
        unb64z(blk["probe_Y_f32_b64"], "<f4"),
    ))
    mt = doc["tensor_256x384_offset"]
    R, C = mt["R"], mt["C"]
    w_m = unb64z(mt["words_u32_z85_b64"], "<u4")
    sc_m = np.array(mt["scales_f16_u16"], dtype="<u2")
    X_m = unb64z(mt["probe_X_f16_b64"], "<u2").reshape(4, C)
    cases.append((2, R, C, 4, 0, 0, R, C, w_m, sc_m, X_m.reshape(-1), unb64z(mt["probe_Y_full_f32_b64"], "<f4")))
    r_off, c_off = mt["r_off"], mt["c_off"]
    X_sub = np.ascontiguousarray(X_m[:, c_off:]).reshape(-1)
    cases.append((2, R, C, 4, r_off, c_off, R - r_off, C - c_off, w_m, sc_m, X_sub, unb64z(mt["probe_Y_slice_f32_b64"], "<f4")))

    with open(path, "wb") as f:
        f.write(struct.pack("<II", 0x56454334, len(cases)))
        f.write(tbl_raw)
        for bits, R, C, rows, r_off, c_off, n_loc, k_loc, W, sc, X, Y in cases:
            f.write(struct.pack("<10I", bits, R, C, rows, r_off, c_off, n_loc, k_loc, len(W), len(sc)))
            f.write(W.tobytes())
            f.write(sc.tobytes())
            f.write(X.tobytes())
            f.write(Y.tobytes())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-gpu", action="store_true", help="Skip Metal/CUDA kernel vector verification")
    args = ap.parse_args()

    v3_doc, tbl3_raw = verify_tfq3_reference(os.path.join(REPO_ROOT, "reference", "vectors", "tfq3_vectors.json"))
    v4_doc, tbl4_raw = verify_tfq4_reference(os.path.join(REPO_ROOT, "reference", "vectors", "tfq4_vectors.json"))
    v4_k2_doc, tbl4_k2_raw = verify_tfq4_true_e8p_reference()

    if not args.no_gpu:
        with tempfile.TemporaryDirectory(prefix="tfq_vec_") as tmp:
            vec3_bin = os.path.join(tmp, "vec3.bin")
            vec4_bin = os.path.join(tmp, "vec4.bin")
            vec4_k2_bin = os.path.join(tmp, "vec4_k2.bin")
            write_vec3_bundle(vec3_bin, v3_doc, tbl3_raw)
            write_vec4_bundle(vec4_bin, v4_doc, tbl4_raw)
            write_vec4_bundle(vec4_k2_bin, v4_k2_doc, tbl4_k2_raw)

            for bin_name, bundle in (
                ("tfq3-metal-test", vec3_bin),
                ("tfq4-metal-test", vec4_bin),
                ("tfq4-metal-test", vec4_k2_bin),
                ("tfq3-cuda-test", vec3_bin),
                ("tfq4-cuda-test", vec4_bin),
                ("tfq4-cuda-test", vec4_k2_bin),
            ):
                exe = os.path.join(REPO_ROOT, "build", bin_name)
                if os.path.isfile(exe):
                    proc = subprocess.run([exe, "--verify-vectors", bundle], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    if proc.returncode != 0:
                        raise AssertionError(f"{bin_name} --verify-vectors failed:\n{proc.stdout}\n{proc.stderr}")
                    print(proc.stdout.strip())

    print("ALL GOLDEN VECTOR TESTS PASSED")


if __name__ == "__main__":
    main()
