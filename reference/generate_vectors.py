#!/usr/bin/env python3
"""
Generate canonical golden decoding test vectors for TFQ3 (3b/4b/5b) and TFQ4 (2b)
under `reference/vectors/`, using the committed canonical tables in `reference/tables/`.

Vectors cover:
  - TFQ3 (for each bit width 3, 4, 5):
      1. 8 micro-vectors: (state_in, code_words) -> (state_out, 8 fp32 values)
      2. One full 128x128 block: (words, scale) -> fp32 decoded block SHA-256 + GEMM Y_ref
      3. One 256x384 multi-block tensor exercising r_off=128 and bc0=1..3 state resume ->
         full [256, 384] SHA-256 + sliced [128, 256] (r_off=128, c_off=128) SHA-256 + GEMM Y_ref
  - TFQ4 (2-bit E8-initialized Voronoi):
      1. 8 micro-vectors: (word_u32) -> 16 fp32 rotated values
      2. One full 128x128 block -> fp32 decoded block SHA-256 + GEMM Y_ref
      3. One 256x384 tensor exercising r_off=128, c_off=128 -> full & sliced SHA-256 + GEMM Y_ref
"""
import base64
import hashlib
import json
import os
import struct
import sys
import zlib
import numpy as np

REF_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REF_DIR)

import tfq3_reference as tfq3_ref
import tfq4_lattice as tfq4_ref


def b64z(arr: np.ndarray) -> str:
    raw = np.ascontiguousarray(arr).tobytes()
    return base64.b64encode(zlib.compress(raw, level=9)).decode("ascii")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_f32(arr: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(arr, dtype="<f4").tobytes())


def generate_tfq3_vectors(tables_path: str) -> dict:
    tbl_raw = open(tables_path, "rb").read()
    tbl = tfq3_ref.load_tables(tables_path)
    rng = np.random.default_rng(20260401)

    per_bits = {}
    for bits in (3, 4, 5):
        micro = []
        init_states = [
            (tfq3_ref.INIT_ST_BASE ^ ((lane * tfq3_ref.HASH_MULT) & 0xFFFFFFFF)) & 0xFFFFFFFF
            for lane in (0, 7, 19, 31)
        ] + [0x00000000, 0xFFFFFFFF, 0xDEADBEEF, 0x12345678]
        for st_in in init_states:
            if bits == 4:
                wc = int(rng.integers(0, 1 << 32, dtype=np.uint64))
                st_out, vals = tfq3_ref._decode_8_4b(st_in, wc, tbl)
                micro.append({
                    "state_in": st_in,
                    "words": [wc],
                    "state_out": st_out,
                    "values_f32": [float(v) for v in vals],
                    "values_u32_hex": [f"0x{int(u):08x}" for u in vals.astype("<f4").view(np.uint32)],
                })
            elif bits == 3:
                m24 = int(rng.integers(0, 1 << 24, dtype=np.uint64))
                st_out, vals = tfq3_ref._decode_8_3b(st_in, m24, tbl)
                micro.append({
                    "state_in": st_in,
                    "m24": m24,
                    "state_out": st_out,
                    "values_f32": [float(v) for v in vals],
                    "values_u32_hex": [f"0x{int(u):08x}" for u in vals.astype("<f4").view(np.uint32)],
                })
            else:
                hi8 = int(rng.integers(0, 256, dtype=np.uint64))
                lo32 = int(rng.integers(0, 1 << 32, dtype=np.uint64))
                st_out, vals = tfq3_ref._decode_8_5b(st_in, hi8, lo32, tbl)
                micro.append({
                    "state_in": st_in,
                    "hi8": hi8,
                    "lo32": lo32,
                    "state_out": st_out,
                    "values_f32": [float(v) for v in vals],
                    "values_u32_hex": [f"0x{int(u):08x}" for u in vals.astype("<f4").view(np.uint32)],
                })

        wpb = bits * 2

        # 128x128 single-block vector
        w_blk = rng.integers(0, 1 << 32, size=(8 * 1 * wpb * 32,), dtype=np.uint32)
        sc_blk = np.array([[0.0425]], dtype=np.float16)
        W_blk = tfq3_ref.decode_tfpn_fast_np(w_blk, sc_blk, 128, 128, bits, tbl)
        X_blk = (rng.standard_normal((4, 128)).astype(np.float32) * 0.25).astype(np.float16)
        Y_blk = (X_blk.astype(np.float64) @ W_blk.astype(np.float64).T).astype(np.float32)

        # 256x384 multi-block tensor (tests r_off=128 and bc0=1..3 state resume!)
        R_m, C_m = 256, 384
        w_multi = rng.integers(0, 1 << 32, size=((R_m // 16) * (C_m // 128) * wpb * 32,), dtype=np.uint32)
        sc_multi = np.array([[0.031, 0.044, 0.027], [0.052, 0.038, 0.049]], dtype=np.float16)
        W_full = tfq3_ref.decode_tfpn_fast_np(w_multi, sc_multi, R_m, C_m, bits, tbl)
        # Verify slice (r_off=128, bc0=1..3 -> [128, 256]) directly via trellis resume
        strip_off = 128 // 16
        w_sub_rows = w_multi.reshape(R_m // 16, (C_m // 128) * wpb * 32)[strip_off:].reshape(-1)
        W_slice = tfq3_ref.decode_tfpn_fast_np(
            w_sub_rows, sc_multi[1:2, :], 128, C_m, bits, tbl, r_off=128, bc0=1, bc1=3
        )
        assert np.array_equal(W_full[128:256, 128:384], W_slice), "Slice decode diverged from full decode!"

        X_multi = (rng.standard_normal((4, C_m)).astype(np.float32) * 0.25).astype(np.float16)
        Y_full = (X_multi.astype(np.float64) @ W_full.astype(np.float64).T).astype(np.float32)
        Y_slice = (X_multi[:, 128:384].astype(np.float64) @ W_slice.astype(np.float64).T).astype(np.float32)

        per_bits[str(bits)] = {
            "bits": bits,
            "micro": micro,
            "block_128x128": {
                "R": 128,
                "C": 128,
                "words_u32_z85_b64": b64z(w_blk.astype("<u4")),
                "scales_f16_u16": [int(x) for x in sc_blk.view(np.uint16).reshape(-1)],
                "decoded_f32_sha256": sha256_f32(W_blk),
                "probe_X_f16_b64": b64z(X_blk.astype("<f2")),
                "probe_Y_f32_b64": b64z(Y_blk.astype("<f4")),
            },
            "tensor_256x384_resume": {
                "R": R_m,
                "C": C_m,
                "r_off": 128,
                "bc0": 1,
                "bc1": 3,
                "words_u32_z85_b64": b64z(w_multi.astype("<u4")),
                "scales_f16_u16": [int(x) for x in sc_multi.view(np.uint16).reshape(-1)],
                "full_decoded_f32_sha256": sha256_f32(W_full),
                "slice_decoded_f32_sha256": sha256_f32(W_slice),
                "probe_X_f16_b64": b64z(X_multi.astype("<f2")),
                "probe_Y_full_f32_b64": b64z(Y_full.astype("<f4")),
                "probe_Y_slice_f32_b64": b64z(Y_slice.astype("<f4")),
            },
        }

    return {
        "format": "TFQ3",
        "version": 3,
        "tables_file": "reference/tables/pw_v3_tables.bin",
        "tables_sha256": sha256_bytes(tbl_raw),
        "bit_widths": per_bits,
    }


def decode_tfq4_array(words: np.ndarray, scales: np.ndarray, lut: np.ndarray, r_off: int = 0, c_off: int = 0) -> np.ndarray:
    R, wpr = words.shape
    C = wpr * 16
    codes = np.empty((R, C // 8), dtype=np.uint16)
    codes[:, 0::2] = (words & np.uint32(0xFFFF)).astype(np.uint16)
    codes[:, 1::2] = (words >> np.uint32(16)).astype(np.uint16)
    k_idx = (codes & np.uint16(0xFF)).astype(np.int32)
    sm = (codes >> np.uint16(8)).astype(np.uint32)
    bits_mask = (sm[..., None] >> np.arange(8, dtype=np.uint32)) & np.uint32(1)
    sgn = np.where(bits_mask != 0, -1.0, 1.0).astype(np.float32)
    W_rot = (lut[k_idx] * sgn).reshape(R, C)
    sc_tile = np.repeat(np.repeat(scales.astype(np.float32), 128, axis=0), 128, axis=1)
    scaled = W_rot * sc_tile
    sr = tfq3_ref.pw_sgn(tfq3_ref.SEED_ROW, np.arange(r_off, r_off + R, dtype=np.uint32))
    sc_v = tfq3_ref.pw_sgn(tfq3_ref.SEED_COL, np.arange(c_off, c_off + C, dtype=np.uint32))
    tmp = tfq3_ref.fwht_128(scaled.reshape(R // 128, 128, C).transpose(0, 2, 1)).transpose(0, 2, 1).reshape(R, C)
    tmp = tmp * sr[:, None]
    out = tfq3_ref.fwht_128(tmp.reshape(R, C // 128, 128)).reshape(R, C) * sc_v[None, :]
    return out.astype(np.float32, copy=False)


def decode_tfq4_1024_array(
    words: np.ndarray, scales: np.ndarray, lut: np.ndarray, r_off: int = 0, c_off: int = 0
) -> np.ndarray:
    R, wpr = words.shape
    C = (wpr // 9) * 128
    W_rot = tfq4_ref.decode_true_e8p_1024_words(words, lut)
    sc_tile = np.repeat(np.repeat(scales.astype(np.float32), 128, axis=0), 128, axis=1)
    scaled = W_rot * sc_tile
    sr = tfq3_ref.pw_sgn(tfq3_ref.SEED_ROW, np.arange(r_off, r_off + R, dtype=np.uint32))
    sc_v = tfq3_ref.pw_sgn(tfq3_ref.SEED_COL, np.arange(c_off, c_off + C, dtype=np.uint32))
    tmp = tfq3_ref.fwht_128(scaled.reshape(R // 128, 128, C).transpose(0, 2, 1)).transpose(0, 2, 1).reshape(R, C)
    tmp = tmp * sr[:, None]
    out = tfq3_ref.fwht_128(tmp.reshape(R, C // 128, 128)).reshape(R, C) * sc_v[None, :]
    return out.astype(np.float32, copy=False)


def generate_tfq4_1024_section() -> dict:
    abs_cb, m_par = tfq4_ref.build_true_e8p_1024_abs_codebook()
    tbl_buf = bytearray(32768)
    struct.pack_into(
        "<IIIII", tbl_buf, 0, tfq4_ref.MAGIC_PWV4, tfq4_ref.TFQ4_VER, 1024, 8, tfq4_ref.PWV4_TABLE_TRUE_E8P_1024
    )
    lut_f16 = np.ascontiguousarray(abs_cb, dtype="<f2")
    tbl_buf[64 : 64 + lut_f16.nbytes] = lut_f16.tobytes()
    tbl_sha = sha256_bytes(bytes(tbl_buf))

    rng = np.random.default_rng(20260404)
    # 8 micro-vectors covering all 4 hi2 quadrants (0..255, 256..511, 512..767, 768..1023), both cosets & parities
    micro_cases = [
        (0, 0x00, 0),
        (255, 0x7F, 1),
        (256, 0x2A, 0),
        (511, 0x55, 1),
        (512, 0x01, 0),
        (767, 0x7E, 1),
        (1023, 0x3C, 0),
        (888, 0x63, 1),
    ]
    micro = []
    for abs_idx, s7, coset in micro_cases:
        low16 = (abs_idx & 0xFF) | ((s7 & 0x7F) << 8) | ((coset & 1) << 15)
        hi2 = (abs_idx >> 8) & 3
        s_bit7 = (bin(s7).count("1") ^ int(m_par[abs_idx])) & 1
        sm8 = s7 | (s_bit7 << 7)
        c_sgn = -1.0 if coset else 1.0
        vals = [float(c_sgn * (float(abs_cb[abs_idx, d]) * (-1.0 if ((sm8 >> d) & 1) else 1.0) - 0.25)) for d in range(8)]
        arr = np.array(vals, dtype="<f4")
        vec = tfq4_ref.decode_true_e8p_1024_codes(
            np.array([abs_idx], dtype=np.uint32),
            np.array([s7], dtype=np.uint32),
            np.array([coset], dtype=np.uint32),
            abs_cb,
        )[0]
        assert np.array_equal(arr, vec)
        assert float(np.max(np.abs(vec - tfq4_ref.nearest_e8_quarter(vec[None, :])[0]))) == 0.0
        micro.append({
            "abs_idx": abs_idx,
            "s7": s7,
            "coset": coset,
            "low16": low16,
            "hi2": hi2,
            "values_f32": vals,
            "values_u32_hex": [f"0x{int(u):08x}" for u in arr.view(np.uint32)],
        })

    w_blk = rng.integers(0, 1 << 32, size=(128, 9), dtype=np.uint32)
    sc_blk = np.array([[0.0375]], dtype=np.float16)
    W_blk = decode_tfq4_1024_array(w_blk, sc_blk, abs_cb)
    X_blk = (rng.standard_normal((4, 128)).astype(np.float32) * 0.25).astype(np.float16)
    Y_blk = (X_blk.astype(np.float64) @ W_blk.astype(np.float64).T).astype(np.float32)

    R_m, C_m, r_off, c_off = 256, 384, 128, 128
    w_multi = rng.integers(0, 1 << 32, size=(R_m, (C_m // 128) * 9), dtype=np.uint32)
    sc_multi = np.array([[0.029, 0.041, 0.033], [0.048, 0.036, 0.051]], dtype=np.float16)
    W_full = decode_tfq4_1024_array(w_multi, sc_multi, abs_cb)
    W_slice = decode_tfq4_1024_array(
        w_multi[r_off:, (c_off // 128) * 9 :],
        sc_multi[r_off // 128 :, c_off // 128 :],
        abs_cb,
        r_off=r_off,
        c_off=c_off,
    )
    assert np.array_equal(W_full[r_off:, c_off:], W_slice)

    X_multi = (rng.standard_normal((4, C_m)).astype(np.float32) * 0.25).astype(np.float16)
    Y_full = (X_multi.astype(np.float64) @ W_full.astype(np.float64).T).astype(np.float32)
    Y_slice = (X_multi[:, c_off:].astype(np.float64) @ W_slice.astype(np.float64).T).astype(np.float32)

    return {
        "format": "TFQ4-TRUE-E8P-1024",
        "version": 4,
        "table_kind": tfq4_ref.PWV4_TABLE_TRUE_E8P_1024,
        "k_codes": 1024,
        "b_bits": tfq4_ref.TFQ4_BITS_225,
        "tables_sha256": tbl_sha,
        "micro": micro,
        "block_128x128": {
            "R": 128,
            "C": 128,
            "words_u32_z85_b64": b64z(w_blk.astype("<u4")),
            "scales_f16_u16": [int(x) for x in sc_blk.view(np.uint16).reshape(-1)],
            "decoded_f32_sha256": sha256_f32(W_blk),
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
            "full_decoded_f32_sha256": sha256_f32(W_full),
            "slice_decoded_f32_sha256": sha256_f32(W_slice),
            "probe_X_f16_b64": b64z(X_multi.astype("<f2")),
            "probe_Y_full_f32_b64": b64z(Y_full.astype("<f4")),
            "probe_Y_slice_f32_b64": b64z(Y_slice.astype("<f4")),
        },
    }


def generate_tfq4_vectors(tables_path: str) -> dict:
    tbl_raw = open(tables_path, "rb").read()
    lut = tfq4_ref.load_pwv4_tables(tables_path)
    rng = np.random.default_rng(20260402)

    micro = []
    test_words = [0x00000000, 0xFFFFFFFF, 0xA53C5A81] + [
        int(rng.integers(0, 1 << 32, dtype=np.uint64)) for _ in range(5)
    ]
    for w in test_words:
        vals = []
        for code in (w & 0xFFFF, (w >> 16) & 0xFFFF):
            idx = code & 0xFF
            sm = code >> 8
            for d in range(8):
                sgn = -1.0 if ((sm >> d) & 1) else 1.0
                vals.append(float(lut[idx, d] * sgn))
        arr = np.array(vals, dtype="<f4")
        micro.append({
            "word_u32": w,
            "values_f32": vals,
            "values_u32_hex": [f"0x{int(u):08x}" for u in arr.view(np.uint32)],
        })

    w_blk = rng.integers(0, 1 << 32, size=(128, 128 // 16), dtype=np.uint32)
    sc_blk = np.array([[0.0375]], dtype=np.float16)
    W_blk = decode_tfq4_array(w_blk, sc_blk, lut)
    X_blk = (rng.standard_normal((4, 128)).astype(np.float32) * 0.25).astype(np.float16)
    Y_blk = (X_blk.astype(np.float64) @ W_blk.astype(np.float64).T).astype(np.float32)

    R_m, C_m = 256, 384
    w_multi = rng.integers(0, 1 << 32, size=(R_m, C_m // 16), dtype=np.uint32)
    sc_multi = np.array([[0.029, 0.041, 0.033], [0.048, 0.036, 0.051]], dtype=np.float16)
    W_full = decode_tfq4_array(w_multi, sc_multi, lut)
    W_slice = decode_tfq4_array(w_multi[128:256, 128 // 16 : 384 // 16], sc_multi[1:2, 1:3], lut, r_off=128, c_off=128)
    assert np.array_equal(W_full[128:256, 128:384], W_slice)

    X_multi = (rng.standard_normal((4, C_m)).astype(np.float32) * 0.25).astype(np.float16)
    Y_full = (X_multi.astype(np.float64) @ W_full.astype(np.float64).T).astype(np.float32)
    Y_slice = (X_multi[:, 128:384].astype(np.float64) @ W_slice.astype(np.float64).T).astype(np.float32)

    return {
        "format": "TFQ4-E8P",
        "version": 4,
        "tables_file": "reference/tables/pw_v4_e8p_tables.bin",
        "tables_sha256": sha256_bytes(tbl_raw),
        "micro": micro,
        "block_128x128": {
            "R": 128,
            "C": 128,
            "words_u32_z85_b64": b64z(w_blk.astype("<u4")),
            "scales_f16_u16": [int(x) for x in sc_blk.view(np.uint16).reshape(-1)],
            "decoded_f32_sha256": sha256_f32(W_blk),
            "probe_X_f16_b64": b64z(X_blk.astype("<f2")),
            "probe_Y_f32_b64": b64z(Y_blk.astype("<f4")),
        },
        "tensor_256x384_offset": {
            "R": R_m,
            "C": C_m,
            "r_off": 128,
            "c_off": 128,
            "words_u32_z85_b64": b64z(w_multi.astype("<u4")),
            "scales_f16_u16": [int(x) for x in sc_multi.view(np.uint16).reshape(-1)],
            "full_decoded_f32_sha256": sha256_f32(W_full),
            "slice_decoded_f32_sha256": sha256_f32(W_slice),
            "probe_X_f16_b64": b64z(X_multi.astype("<f2")),
            "probe_Y_full_f32_b64": b64z(Y_full.astype("<f4")),
            "probe_Y_slice_f32_b64": b64z(Y_slice.astype("<f4")),
        },
        "true_e8p_1024": generate_tfq4_1024_section(),
    }


def main() -> None:
    out_dir = os.path.join(REF_DIR, "vectors")
    os.makedirs(out_dir, exist_ok=True)
    v3 = generate_tfq3_vectors(os.path.join(REF_DIR, "tables", "pw_v3_tables.bin"))
    v4 = generate_tfq4_vectors(os.path.join(REF_DIR, "tables", "pw_v4_e8p_tables.bin"))

    p3 = os.path.join(out_dir, "tfq3_vectors.json")
    p4 = os.path.join(out_dir, "tfq4_vectors.json")
    with open(p3, "w", encoding="utf-8") as f:
        json.dump(v3, f, indent=2)
        f.write("\n")
    with open(p4, "w", encoding="utf-8") as f:
        json.dump(v4, f, indent=2)
        f.write("\n")
    print(f"Wrote {p3} ({os.path.getsize(p3)} B) and {p4} ({os.path.getsize(p4)} B)")


if __name__ == "__main__":
    main()
