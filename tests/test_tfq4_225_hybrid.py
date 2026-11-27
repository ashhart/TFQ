#!/usr/bin/env python3
"""
End-to-end verification suite for:
  1. TFQ4 2.25 bpw (True QuIP# E8P-1024, PWV4 table_kind=3, 32 KiB LUT, b_bits=225)
  2. Hybrid Hot/Cold MoE (4b/3b TFQ3 + 2.00b/2.25b TFQ4-E8P in TFPW) on both
     Inkling-2 (NVFP4 + MXFP8 3D MoE) and GLM (FP8 + BF16 MoE).

Run with `--print-hashes` to print deterministic SHA-256 digests for cross-platform
Apple Silicon Metal <-> NVIDIA CUDA bit-identity comparison.
"""
import argparse
import base64
import glob
import hashlib
import json
import math
import os
import platform
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

# Verified 100% byte-identical on Apple Silicon Metal (M5 Max) and NVIDIA CUDA (GB10 sm_121):
GOLDEN_CROSS_PLATFORM_SHA256 = {
    "pwv4_kind3_tables": "3b9c3f290bb15c5a8328cb4b19f589e2ac90c43bac0bd40943f6666a9884431f",
    "tfq4_225_tfq4-00001-of-00002.tffp4": "dc36374b8d18e53b17b3387af5c5374b11932a44befd574fb8cf9ba9e8fa58df",
    "tfq4_225_tfq4-00002-of-00002.tffp4": "c80858a5e323c8b908956a97d636943b279bf1c25d666f67ca0eb78c943bee68",
    "ink2_heat432_L0": "4db793e6cd815dd61b94f6fc955c8ac2ed955550ec7af22d8ccd487ec13e5a85",
    "ink2_heat432_L1": "a8bf0c0e50480f3cdbdf4d67d73b1699ede3cefb4f51200027c627725e9205f5",
    "ink2_heat43225_L0": "c73b0c41b9498924bd1b5012b0b8b150b8001f2bbf86780eba2d0dce0a5744cf",
    "ink2_heat43225_L1": "0ef79ba98e30de7f5611b9ff436a736a9678872418ee2070d78a0ab229986c77",
    "glm_fp8_heat432_L03": "aab221c9ce64bbd5d3604ae2f0508aeaa7fe8a8d6bd2e8ff32ae2bff3a21a8e2",
    "glm_fp8_heat43225_L03": "530bbccd704e087f9933d258d637df5e1fd455a50644bf5a2b3f67019e077d72",
    "glm_bf16_heat432_L03": "ba76747119e8b78ff37740f56b30f5a112be909620f9507423bd7e2fc5c84916",
    "glm_bf16_heat43225_L03": "24eabdcf128076f3bde2d065cd42530ca73e39faea4f619f8effd1dfc6b9fd3f",
}

E2M1_LUT = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)


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


def snr_db(ref: np.ndarray, got: np.ndarray) -> float:
    sig = float(np.sum(ref.astype(np.float64) ** 2))
    err = float(np.sum((ref.astype(np.float64) - got.astype(np.float64)) ** 2))
    return 10.0 * math.log10(sig / max(err, 1e-18))


def f32_to_bf16_bytes(arr: np.ndarray) -> bytes:
    u32 = arr.astype(np.float32, copy=False).view(np.uint32)
    rounding = np.uint32(0x7FFF) + ((u32 >> np.uint32(16)) & np.uint32(1))
    return ((u32 + rounding) >> np.uint32(16)).astype(np.uint16).tobytes()


def bf16_round_f32(x: float) -> float:
    u = struct.unpack("<I", struct.pack("<f", float(x)))[0]
    if (u & 0x7F800000) == 0x7F800000:
        return float(x)
    r = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", r))[0]


def e8m0_to_f32(sb: np.ndarray) -> np.ndarray:
    u32 = sb.astype(np.uint32) << np.uint32(23)
    return np.where(sb == 0, np.float32(2.0 ** -127), u32.view(np.float32))


def dequant_nvfp4_ref(codes_u8: np.ndarray, scales_u8: np.ndarray, amax_f32: float, den_f32: float) -> np.ndarray:
    R, C_half = codes_u8.shape
    C = C_half * 2
    nibs = np.empty((R, C), dtype=np.uint8)
    nibs[:, 0::2] = codes_u8 & np.uint8(0x0F)
    nibs[:, 1::2] = (codes_u8 >> np.uint8(4)) & np.uint8(0x0F)
    sc = np.repeat(E4M3_LUT[scales_u8], 16, axis=1)
    outer = np.float32(bf16_round_f32(amax_f32) / bf16_round_f32(den_f32))
    return (E2M1_LUT[nibs] * sc) * outer


def dequant_mxfp8_ref(codes_u8: np.ndarray, scales_u8: np.ndarray) -> np.ndarray:
    return E4M3_LUT[codes_u8] * np.repeat(e8m0_to_f32(scales_u8), 32, axis=1)


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


def read_tfp4_shard(path: str) -> tuple[list[dict], np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        data = f.read()
    magic, ver, n_t, pad0, off_m, off_s, off_w, n_sc, n_wd, pad1 = struct.unpack_from("<IIIIQQQQQQ", data, 0)
    assert magic == tfq4_ref.MAGIC_TFP4 and ver == tfq4_ref.TFQ4_VER
    metas = []
    for i in range(n_t):
        o = off_m + i * 128
        raw_name = data[o : o + 80].split(b"\x00", 1)[0].decode("utf-8")
        R, C, b_bits, sc_off, w_off, w_cnt, snr, orig_r, orig_c, flags = struct.unpack_from(
            "<IIIIQQfIII", data, o + 80
        )
        metas.append(
            {
                "name": raw_name,
                "R": R,
                "C": C,
                "b_bits": b_bits,
                "sc_off": sc_off,
                "word_off": w_off,
                "word_count": w_cnt,
                "snr_db": snr,
                "orig_R": orig_r,
                "orig_C": orig_c,
                "flags": flags,
            }
        )
    scales = np.frombuffer(data, dtype=np.float16, count=n_sc, offset=off_s).copy()
    words = np.frombuffer(data, dtype=np.uint32, count=n_wd, offset=off_w).copy()
    return metas, scales, words


def read_tfpw_layer(path: str) -> tuple[dict, list[dict], np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        data = f.read()
    hdr_vals = struct.unpack_from("<8I6Q4Q32s", data, 0)
    hdr = {
        "magic": hdr_vals[0],
        "version": hdr_vals[1],
        "layer_idx": hdr_vals[2],
        "num_experts": hdr_vals[3],
        "num_tensors": hdr_vals[4],
        "hidden_size": hdr_vals[5],
        "moe_inter": hdr_vals[6],
        "reserved0": hdr_vals[7],
        "raw_fp8_bytes": hdr_vals[8],
        "stream_words": hdr_vals[9],
        "scale_halfs": hdr_vals[10],
        "off_meta": hdr_vals[11],
        "off_scales": hdr_vals[12],
        "off_weights": hdr_vals[13],
        "meta_tag": hdr_vals[18].split(b"\x00", 1)[0].decode("utf-8"),
    }
    assert hdr["magic"] == 0x57504654 and hdr["version"] == 3
    metas = []
    for e in range(hdr["num_experts"]):
        o = hdr["off_meta"] + e * 64
        b, rank, gw, uw, dw, gs, us, ds, gsnr, usnr, dsnr, r0, r1 = struct.unpack_from("<IIQQQIIIfffII", data, o)
        metas.append(
            {
                "b_bits": b,
                "heat_rank": rank,
                "gate_word_off": gw,
                "up_word_off": uw,
                "down_word_off": dw,
                "gate_sc_off": gs,
                "up_sc_off": us,
                "down_sc_off": ds,
                "gate_snr_db": gsnr,
                "up_snr_db": usnr,
                "down_snr_db": dsnr,
            }
        )
    scales = np.frombuffer(data, dtype=np.float16, count=hdr["scale_halfs"], offset=hdr["off_scales"]).copy()
    words = np.frombuffer(data, dtype=np.uint32, count=hdr["stream_words"], offset=hdr["off_weights"]).copy()
    return hdr, metas, scales, words


def dequant_tfq4_arrays(
    words: np.ndarray,
    scales: np.ndarray,
    R: int,
    C: int,
    pwv4_lut: np.ndarray,
    pwv4_kind: int,
    b_bits: int,
    r_off: int = 0,
    c_off: int = 0,
) -> np.ndarray:
    n_br, n_bc = R // 128, C // 128
    sc = scales.reshape(n_br, n_bc).astype(np.float32)
    if b_bits == 225:
        assert pwv4_kind == 3
        w_rot = tfq4_ref.decode_true_e8p_1024_words(words.reshape(R, n_bc * 9), pwv4_lut)
    else:
        w2 = words.reshape(R, n_bc * 8)
        codes = np.empty((R, C // 8), dtype=np.uint16)
        codes[:, 0::2] = (w2 & np.uint32(0xFFFF)).astype(np.uint16)
        codes[:, 1::2] = (w2 >> np.uint32(16)).astype(np.uint16)
        w_rot = tfq4_ref.decode_true_e8p_codes(codes, pwv4_lut).reshape(R, C)
    scaled = w_rot * np.repeat(np.repeat(sc, 128, axis=0), 128, axis=1)
    sr = tfq3_ref.pw_sgn(tfq3_ref.SEED_ROW, np.arange(r_off, r_off + R, dtype=np.uint32))
    sc_v = tfq3_ref.pw_sgn(tfq3_ref.SEED_COL, np.arange(c_off, c_off + C, dtype=np.uint32))
    tmp = tfq3_ref.fwht_128(scaled.reshape(n_br, 128, C).transpose(0, 2, 1)).transpose(0, 2, 1).reshape(R, C) * sr[:, None]
    out = tfq3_ref.fwht_128(tmp.reshape(R, n_bc, 128)).reshape(R, C) * sc_v[None, :]
    return out.astype(np.float32, copy=False)


def decode_hybrid_proj(
    words: np.ndarray,
    scales: np.ndarray,
    R: int,
    C: int,
    b_bits: int,
    pwv3_tables: dict,
    pwv4_lut: np.ndarray | None,
    pwv4_kind: int,
) -> np.ndarray:
    if b_bits in (3, 4, 5):
        return tfq3_ref.decode_tfpn_fast_np(words, scales.reshape(R // 128, C // 128), R, C, b_bits, pwv3_tables)
    if b_bits in (2, 225):
        assert pwv4_lut is not None
        return dequant_tfq4_arrays(words, scales, R, C, pwv4_lut, pwv4_kind, b_bits)
    raise ValueError(f"Unsupported b_bits={b_bits}")


def test_pwv4_kind3_and_inspect(inspect_bin: str, tmp: str) -> None:
    lut2, _ = tfq4_ref.build_true_e8p_abs_codebook()
    lut3, _ = tfq4_ref.build_true_e8p_1024_abs_codebook()
    assert lut2.shape == (256, 8) and lut3.shape == (1024, 8)
    assert np.array_equal(lut3[:256], lut2), "First 256 rows of table_kind=3 must be 100% bit-identical to table_kind=2"

    # Verify algebraic norm bound sum((2*m+1)^2) <= 68
    m_vals = np.rint(lut3.astype(np.float64) - 0.5).astype(np.int64)
    assert np.all(np.abs(lut3.astype(np.float64) - (m_vals + 0.5)) == 0.0)
    assert np.all((m_vals >= 0) & (m_vals <= 3))
    n2_x4 = np.sum((2 * m_vals + 1) ** 2, axis=1)
    assert int(np.max(n2_x4)) <= 68

    tbl3_path = os.path.join(tmp, "pw_v4_kind3.bin")
    tfq4_ref.write_pwv4_tables(tbl3_path, lut3, table_kind=tfq4_ref.PWV4_TABLE_TRUE_E8P_1024)
    assert os.path.getsize(tbl3_path) == 32768
    loaded_lut, loaded_kind = tfq4_ref.load_pwv4_tables_with_kind(tbl3_path)
    assert loaded_kind == 3 and np.array_equal(loaded_lut, lut3)
    run([inspect_bin, tbl3_path])

    # Negative controls on PWV4 table_kind=3
    raw = bytearray(open(tbl3_path, "rb").read())

    # 1) Wrong table_kind=2 in 32 KiB file
    bad1 = bytearray(raw)
    struct.pack_into("<I", bad1, 16, 2)
    p1 = os.path.join(tmp, "bad_k2_32k.bin")
    open(p1, "wb").write(bad1)
    run([inspect_bin, p1], expect_ok=False)

    # 2) Unknown table_kind=4 in 32 KiB file
    bad2 = bytearray(raw)
    struct.pack_into("<I", bad2, 16, 4)
    p2 = os.path.join(tmp, "bad_k4_32k.bin")
    open(p2, "wb").write(bad2)
    run([inspect_bin, p2], expect_ok=False)

    # 3) Non-half-integer entry in table_kind=3 LUT
    bad3 = bytearray(raw)
    struct.pack_into("<H", bad3, 64, int(np.float16(1.0).view(np.uint16)))
    p3 = os.path.join(tmp, "bad_nonhalf_k3.bin")
    open(p3, "wb").write(bad3)
    run([inspect_bin, p3], expect_ok=False)

    # 4) Truncated 32 KiB file
    p4 = os.path.join(tmp, "bad_trunc_k3.bin")
    open(p4, "wb").write(raw[:-1])
    run([inspect_bin, p4], expect_ok=False)

    # 5) 16 KiB file patched to table_kind=3 must fail with "table_kind"
    tbl2_path = os.path.join(tmp, "pw_v4_kind2.bin")
    tfq4_ref.write_pwv4_tables(tbl2_path, lut2, table_kind=tfq4_ref.PWV4_TABLE_TRUE_E8P)
    bad5 = bytearray(open(tbl2_path, "rb").read())
    struct.pack_into("<I", bad5, 16, 3)
    p5 = os.path.join(tmp, "bad_k3_16k.bin")
    open(p5, "wb").write(bad5)
    proc = run([inspect_bin, p5], expect_ok=False)
    assert "table_kind" in proc.stderr

    # Verify single table_kind=3 LUT decodes 2.00 bpw words identically to table_kind=2 LUT
    rng = np.random.default_rng(77)
    w_test = rng.standard_normal((128, 128)).astype(np.float32) * 0.05
    w_rot = tfq4_ref.fwht2d_128(w_test)
    _, codes, scales2 = tfq4_ref.quant_true_e8p_e8_quarter(w_rot, lut2)
    c2 = codes.reshape(-1, 2).astype(np.uint32)
    words2 = c2[:, 0] | (c2[:, 1] << np.uint32(16))
    scales2 = scales2.reshape(-1)
    dec_k2 = dequant_tfq4_arrays(words2, scales2, 128, 128, lut2, 2, 2)
    dec_k3 = dequant_tfq4_arrays(words2, scales2, 128, 128, lut3, 3, 2)
    assert np.array_equal(dec_k2, dec_k3), "table_kind=3 must decode 2.00 bpw bit-identically to table_kind=2"
    print("PASS PWV4 table_kind=3 (32 KiB) format, negative controls, and kind=2/3 dual decode")


def test_vectors_kind3(gpu_bench_bin: str) -> None:
    def unb64z(s: str, dtype) -> np.ndarray:
        return np.frombuffer(zlib.decompress(base64.b64decode(s)), dtype=dtype).copy()

    vec_path = os.path.join(REPO_ROOT, "reference", "vectors", "tfq4_vectors.json")
    doc = json.load(open(vec_path, "r", encoding="utf-8"))
    assert "true_e8p_1024" in doc, "Missing true_e8p_1024 section in tfq4_vectors.json"
    sec = doc["true_e8p_1024"]
    assert sec["table_kind"] == 3 and sec["b_bits"] == 225 and sec["k_codes"] == 1024

    lut3, _ = tfq4_ref.build_true_e8p_1024_abs_codebook()
    tmp_bytes = bytearray(32768)
    struct.pack_into("<5I", tmp_bytes, 0, tfq4_ref.MAGIC_PWV4, 4, 1024, 8, 3)
    tmp_bytes[64 : 64 + 16384] = lut3.astype(np.float16).tobytes()
    assert hashlib.sha256(tmp_bytes).hexdigest() == sec["tables_sha256"]

    # 1) Micro vectors
    for mv in sec["micro"]:
        dec = tfq4_ref.decode_true_e8p_1024_codes(
            np.array([mv["abs_idx"]], dtype=np.uint32),
            np.array([mv["s7"]], dtype=np.uint32),
            np.array([mv["coset"]], dtype=np.uint32),
            lut3,
        )[0]
        assert np.array_equal(dec, np.array(mv["values_f32"], dtype=np.float32))

    # 2) 128x128 block
    v1 = sec["block_128x128"]
    w1 = unb64z(v1["words_u32_z85_b64"], "<u4").reshape(v1["R"], (v1["C"] // 128) * 9)
    sc1 = np.array(v1["scales_f16_u16"], dtype=np.uint16).view(np.float16)
    W1 = dequant_tfq4_arrays(w1, sc1, v1["R"], v1["C"], lut3, 3, 225)
    assert hashlib.sha256(np.ascontiguousarray(W1, dtype="<f4").tobytes()).hexdigest() == v1["decoded_f32_sha256"]
    X1 = unb64z(v1["probe_X_f16_b64"], "<f2").reshape(4, v1["C"])
    Y1_exp = unb64z(v1["probe_Y_f32_b64"], "<f4").reshape(4, v1["R"])
    Y1_got = (X1.astype(np.float64) @ W1.astype(np.float64).T).astype(np.float32)
    assert float(np.max(np.abs(Y1_got - Y1_exp))) < 1e-5

    # 3) 256x384 full + offset slice (r_off=128, c_off=128)
    v2 = sec["tensor_256x384_offset"]
    R2, C2, r_off, c_off = v2["R"], v2["C"], v2["r_off"], v2["c_off"]
    w2 = unb64z(v2["words_u32_z85_b64"], "<u4").reshape(R2, (C2 // 128) * 9)
    sc2 = np.array(v2["scales_f16_u16"], dtype=np.uint16).view(np.float16).reshape(R2 // 128, C2 // 128)
    W2_full = dequant_tfq4_arrays(w2, sc2, R2, C2, lut3, 3, 225)
    assert hashlib.sha256(np.ascontiguousarray(W2_full, dtype="<f4").tobytes()).hexdigest() == v2["full_decoded_f32_sha256"]
    # Decode offset slice directly
    w2_sl = w2[r_off:, (c_off // 128) * 9 :]
    sc2_sl = sc2[r_off // 128 :, c_off // 128 :]
    W2_sl = dequant_tfq4_arrays(w2_sl, sc2_sl, R2 - r_off, C2 - c_off, lut3, 3, 225, r_off=r_off, c_off=c_off)
    assert hashlib.sha256(np.ascontiguousarray(W2_sl, dtype="<f4").tobytes()).hexdigest() == v2["slice_decoded_f32_sha256"]

    # 4) Verify GPU kernel directly against true_e8p_1024 binary vector bundle
    if os.path.exists(gpu_bench_bin):
        X2 = unb64z(v2["probe_X_f16_b64"], "<f2").reshape(4, C2)
        Y2_full = unb64z(v2["probe_Y_full_f32_b64"], "<f4").reshape(4, R2)
        Y2_sl = unb64z(v2["probe_Y_slice_f32_b64"], "<f4").reshape(4, R2 - r_off)
        X2_sl = np.ascontiguousarray(X2[:, c_off:])
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tf:
            bundle_path = tf.name
            # VEC4 bundle format: magic=0x56454334, n_cases, 32768B PWV4 kind=3,
            # then per case: 10*u32 (0, R, C, rows, r_off, c_off, n_loc, k_loc, n_words, n_scales)
            # + full W (u32) + full sc (f16) + X_loc (f16) + Y_ref (f32)
            tf.write(struct.pack("<2I", 0x56454334, 3))
            tf.write(bytes(tmp_bytes))
            cases = [
                (v1["R"], v1["C"], 4, 0, 0, v1["R"], v1["C"], w1, sc1, X1, Y1_exp),
                (R2, C2, 4, 0, 0, R2, C2, w2, sc2, X2, Y2_full),
                (R2, C2, 4, r_off, c_off, R2 - r_off, C2 - c_off, w2, sc2, X2_sl, Y2_sl),
            ]
            for R, C, rows, ro, co, n_loc, k_loc, ww, ss, xx, yy in cases:
                ww_c = np.ascontiguousarray(ww, dtype="<u4")
                ss_c = np.ascontiguousarray(ss, dtype="<f2")
                tf.write(struct.pack("<10I", 0, R, C, rows, ro, co, n_loc, k_loc, ww_c.size, ss_c.size))
                tf.write(ww_c.tobytes())
                tf.write(ss_c.tobytes())
                tf.write(np.ascontiguousarray(xx, dtype="<f2").tobytes())
                tf.write(np.ascontiguousarray(yy, dtype="<f4").tobytes())
        try:
            run([gpu_bench_bin, "--verify-vectors", bundle_path])
        finally:
            os.unlink(bundle_path)
        print(f"PASS TFQ4 2.25 bpw golden vectors (Python + {os.path.basename(gpu_bench_bin)} kind=3)")
    else:
        print("PASS TFQ4 2.25 bpw golden vectors (Python)")


def test_tfq4_225_e2e(cli_bin: str, inspect_bin: str, tmp: str, hashes: dict[str, str]) -> None:
    model_dir = os.path.join(tmp, "dense_model")
    os.makedirs(model_dir, exist_ok=True)
    rng = np.random.default_rng(12345)
    w_chunk_a = rng.standard_normal((4352, 512)).astype(np.float32) * 0.02
    w_chunk_b = rng.standard_normal((4352, 512)).astype(np.float32) * 0.02
    w_pad = rng.standard_normal((192, 256)).astype(np.float32) * 0.03
    w_sq = rng.standard_normal((256, 256)).astype(np.float32) * 0.025
    w_norm = np.ones((512,), dtype=np.float32)
    refs = {
        "model.layers.0.mlp.gate_proj": w_chunk_a,
        "model.layers.0.mlp.up_proj": w_chunk_b,
        "model.layers.0.self_attn.q_proj": w_pad,
        "model.layers.0.self_attn.o_proj": w_sq,
    }
    write_safetensors(
        os.path.join(model_dir, "model.safetensors"),
        [
            ("model.layers.0.mlp.gate_proj.weight", "BF16", w_chunk_a.shape, f32_to_bf16_bytes(w_chunk_a)),
            ("model.layers.0.mlp.up_proj.weight", "F16", w_chunk_b.shape, w_chunk_b.astype(np.float16).tobytes()),
            ("model.layers.0.self_attn.q_proj.weight", "F32", w_pad.shape, w_pad.astype(np.float32).tobytes()),
            ("model.layers.0.self_attn.o_proj.weight", "BF16", w_sq.shape, f32_to_bf16_bytes(w_sq)),
            ("model.layers.0.input_layernorm.weight", "F32", w_norm.shape, w_norm.tobytes()),
        ],
    )
    open(os.path.join(model_dir, "config.json"), "w").write(json.dumps({"model_type": "qwen2"}))

    out_bits = os.path.join(tmp, "out_tfq4_225_bits")
    out_tk = os.path.join(tmp, "out_tfq4_225_tk")
    run([cli_bin, model_dir, "--out", out_bits, "--bits", "2.25", "--max-shard-gib", "0.001"])
    run([cli_bin, model_dir, "--out", out_tk, "--table-kind", "e8p1024", "--max-shard-gib", "0.001"])

    tbl_path = os.path.join(out_bits, "pw_v4_e8p_tables.bin")
    assert os.path.getsize(tbl_path) == 32768
    lut3, kind = tfq4_ref.load_pwv4_tables_with_kind(tbl_path)
    assert kind == 3
    ref_tbl = os.path.join(tmp, "ref_pwv4_k3.bin")
    lut3_ref, _ = tfq4_ref.build_true_e8p_1024_abs_codebook()
    tfq4_ref.write_pwv4_tables(ref_tbl, lut3_ref, table_kind=3)
    assert sha256_file(tbl_path) == sha256_file(ref_tbl), "C++ and Python PWV4 table_kind=3 must be byte-identical"
    assert sha256_file(tbl_path) == sha256_file(os.path.join(out_tk, "pw_v4_e8p_tables.bin"))
    hashes["pwv4_kind3_tables"] = sha256_file(tbl_path)

    shards = sorted(glob.glob(os.path.join(out_bits, "tfq4-*.tffp4")))
    assert len(shards) >= 2, f"Expected multi-shard split, got {shards}"
    run([inspect_bin, tbl_path] + shards)

    # --bits 2.25 and --table-kind e8p1024 must produce identical shards
    for sh in shards:
        bn = os.path.basename(sh)
        assert sha256_file(sh) == sha256_file(os.path.join(out_tk, bn))
        hashes[f"tfq4_225_{bn}"] = sha256_file(sh)

    seen = set()
    for sh in shards:
        metas, scales, words = read_tfp4_shard(sh)
        for m in metas:
            name = m["name"]
            seen.add(name)
            assert m["b_bits"] == 225, f"Expected b_bits=225 on {name}, got {m['b_bits']}"
            assert m["word_count"] == m["R"] * (m["C"] // 128) * 9
            dec = tfq4_ref.dequantize_tfp4_tensor(sh, tbl_path, name)
            ref = refs[name]
            orig_r, orig_c = ref.shape
            got_snr = snr_db(ref, dec[:orig_r, :orig_c])
            assert got_snr > 11.4, f"Low 2.25b SNR on {name}: {got_snr:.2f} dB"
            if orig_r > 4096:
                tail_snr = snr_db(ref[4096:], dec[4096:orig_r, :orig_c])
                assert tail_snr > 11.3, f"Low chunk-tail SNR on {name}: {tail_snr:.2f} dB"
            if m["R"] > orig_r:
                pad_rms = float(np.sqrt(np.mean(dec[orig_r:, :orig_c] ** 2)))
                assert pad_rms < 0.01, f"Padding rows not near zero on {name}: rms={pad_rms}"
    assert seen == set(refs.keys())

    # Bit-flip XXH64 rejection on 2.25b TFP4 shard
    corrupt_path = os.path.join(tmp, "corrupt_225.tffp4")
    corrupt = bytearray(open(shards[0], "rb").read())
    corrupt[-16384] ^= 0x01
    open(corrupt_path, "wb").write(corrupt)
    run([inspect_bin, corrupt_path], expect_ok=False)
    print(f"PASS TFQ4 2.25 bpw e2e ({len(shards)} shards, chunking, padding, Python decode SNR > 11.4 dB, XXH64)")


def make_synthetic_inkling2(model_dir: str, E: int = 10, H: int = 256, I: int = 128) -> dict:
    os.makedirs(model_dir, exist_ok=True)
    rng = np.random.default_rng(20261127)
    pos_e4m3_scales = np.array([((e << 3) | m) for e in range(5, 11) for m in range(8)], dtype=np.uint8)
    valid_e4m3_codes = np.array([b for b in range(256) if (b & 0x7F) != 0x7F], dtype=np.uint8)
    valid_e8m0_scales = np.arange(124, 130, dtype=np.uint8)

    refs = {"E": E, "H": H, "I": I, "layers": {}}
    tensors = []
    for lyr in (0, 1):
        # Distinct router norms so ranking is strictly deterministic (expert E-1 hottest, 0 coldest)
        router_w = np.zeros((E, H), dtype=np.float32)
        for e in range(E):
            router_w[e] = rng.standard_normal(H).astype(np.float32) * (0.2 + 0.3 * e)
        router_b = np.linspace(-0.5, 0.5, E, dtype=np.float32)
        tensors.append((f"model.llm.layers.{lyr}.mlp.gate.weight", "BF16", (E, H), f32_to_bf16_bytes(router_w)))
        tensors.append((f"model.llm.layers.{lyr}.mlp.gate.e_score_correction_bias", "F32", (E,), router_b.tobytes()))

        layer_ref = []
        if lyr == 0:
            w13_c = rng.integers(0, 256, size=(E, 2 * I, H // 2), dtype=np.uint8)
            w13_s = rng.choice(pos_e4m3_scales, size=(E, 2 * I, H // 16))
            w13_a = np.linspace(1.0, 2.0, E, dtype=np.float32)
            w13_d = np.array([6.0], dtype=np.float32)
            w2_c = rng.integers(0, 256, size=(E, H, I // 2), dtype=np.uint8)
            w2_s = rng.choice(pos_e4m3_scales, size=(E, H, I // 16))
            w2_a = np.linspace(1.2, 2.2, E, dtype=np.float32)
            w2_d = np.array([6.0], dtype=np.float32)
            for e in range(E):
                w13 = dequant_nvfp4_ref(w13_c[e], w13_s[e], float(w13_a[e]), float(w13_d[0]))
                w2 = dequant_nvfp4_ref(w2_c[e], w2_s[e], float(w2_a[e]), float(w2_d[0]))
                layer_ref.append({"gate": w13[0::2], "up": w13[1::2], "down": w2})
            tensors.extend(
                [
                    (f"model.llm.layers.{lyr}.mlp.experts.w13_weight", "U8", w13_c.shape, w13_c.tobytes()),
                    (f"model.llm.layers.{lyr}.mlp.experts.w13_weight_scale", "F8_E4M3", w13_s.shape, w13_s.tobytes()),
                    (f"model.llm.layers.{lyr}.mlp.experts.w13_weight_amax", "F32", w13_a.shape, w13_a.tobytes()),
                    (f"model.llm.layers.{lyr}.mlp.experts.w13_weight_outer_scale_denominator", "F32", w13_d.shape, w13_d.tobytes()),
                    (f"model.llm.layers.{lyr}.mlp.experts.w2_weight", "U8", w2_c.shape, w2_c.tobytes()),
                    (f"model.llm.layers.{lyr}.mlp.experts.w2_weight_scale", "F8_E4M3", w2_s.shape, w2_s.tobytes()),
                    (f"model.llm.layers.{lyr}.mlp.experts.w2_weight_amax", "F32", w2_a.shape, w2_a.tobytes()),
                    (f"model.llm.layers.{lyr}.mlp.experts.w2_weight_outer_scale_denominator", "F32", w2_d.shape, w2_d.tobytes()),
                ]
            )
        else:
            w13_c = rng.choice(valid_e4m3_codes, size=(E, 2 * I, H))
            w13_s = rng.choice(valid_e8m0_scales, size=(E, 2 * I, H // 32))
            w2_c = rng.choice(valid_e4m3_codes, size=(E, H, I))
            w2_s = rng.choice(valid_e8m0_scales, size=(E, H, I // 32))
            for e in range(E):
                w13 = dequant_mxfp8_ref(w13_c[e], w13_s[e])
                w2 = dequant_mxfp8_ref(w2_c[e], w2_s[e])
                layer_ref.append({"gate": w13[0::2], "up": w13[1::2], "down": w2})
            tensors.extend(
                [
                    (f"model.llm.layers.{lyr}.mlp.experts.w13_weight", "F8_E4M3", w13_c.shape, w13_c.tobytes()),
                    (f"model.llm.layers.{lyr}.mlp.experts.w13_weight_scale", "U8", w13_s.shape, w13_s.tobytes()),
                    (f"model.llm.layers.{lyr}.mlp.experts.w2_weight", "F8_E4M3", w2_c.shape, w2_c.tobytes()),
                    (f"model.llm.layers.{lyr}.mlp.experts.w2_weight_scale", "U8", w2_s.shape, w2_s.tobytes()),
                ]
            )
        refs["layers"][lyr] = layer_ref

    write_safetensors(os.path.join(model_dir, "model.safetensors"), tensors)
    open(os.path.join(model_dir, "config.json"), "w").write(json.dumps({"model_type": "inkling2"}))
    return refs


def check_hybrid_tfpw(
    out_dir: str,
    layer_idx: int,
    layer_refs: list[dict],
    H: int,
    I: int,
    expected_counts: dict[int, int],
    expected_pwv4_kind: int,
    inspect_bin: str,
) -> None:
    pwv3_path = os.path.join(out_dir, "pw_v3_tables.bin")
    pwv4_path = os.path.join(out_dir, "pw_v4_e8p_tables.bin")
    lyr_path = os.path.join(out_dir, f"layer_{layer_idx:02d}.tffpw")
    assert os.path.exists(pwv4_path), f"Missing {pwv4_path}"
    expected_sz = 32768 if expected_pwv4_kind == 3 else 16384
    assert os.path.getsize(pwv4_path) == expected_sz
    run([inspect_bin, pwv3_path, pwv4_path, lyr_path])

    pwv3_tables = tfq3_ref.load_tables(pwv3_path)
    pwv4_lut, pwv4_kind = tfq4_ref.load_pwv4_tables_with_kind(pwv4_path)
    assert pwv4_kind == expected_pwv4_kind

    hdr, metas, scales, words = read_tfpw_layer(lyr_path)
    assert hdr["hidden_size"] == H and hdr["moe_inter"] == I and hdr["num_experts"] == len(layer_refs)
    counts: dict[int, int] = {}
    for m in metas:
        counts[m["b_bits"]] = counts.get(m["b_bits"], 0) + 1
    assert counts == expected_counts, f"Tier counts mismatch: got {counts}, expected {expected_counts}"

    n_sc = (H // 128) * (I // 128)
    min_snr = {2: 9.8, 225: 11.2, 3: 15.0, 4: 21.0, 5: 27.0}
    for e, m in enumerate(metas):
        b = m["b_bits"]
        if b == 2:
            wpm = I * (H // 16)
        elif b == 225:
            wpm = I * (H // 128) * 9
        else:
            wpm = (I // 16) * (H // 128) * (2 * b) * 32
        for pname, R, C, w_off, s_off, rep_snr in (
            ("gate", I, H, m["gate_word_off"], m["gate_sc_off"], m["gate_snr_db"]),
            ("up", I, H, m["up_word_off"], m["up_sc_off"], m["up_snr_db"]),
            ("down", H, I, m["down_word_off"], m["down_sc_off"], m["down_snr_db"]),
        ):
            dec = decode_hybrid_proj(
                words[w_off : w_off + wpm],
                scales[s_off : s_off + n_sc],
                R,
                C,
                b,
                pwv3_tables,
                pwv4_lut,
                pwv4_kind,
            )
            got = snr_db(layer_refs[e][pname], dec)
            assert got > min_snr[b], f"L{layer_idx} E{e} {pname} ({b}b): SNR={got:.2f} < {min_snr[b]}"
            assert abs(got - rep_snr) < 0.05, f"L{layer_idx} E{e} {pname} ({b}b): ref={got:.3f} vs hdr={rep_snr:.3f}"


def test_inkling2_hybrid(cli_bin: str, inspect_bin: str, tmp: str, hashes: dict[str, str]) -> None:
    model_dir = os.path.join(tmp, "ink2_model")
    refs = make_synthetic_inkling2(model_dir, E=10, H=256, I=128)
    mode_specs = [
        ("heat432", {4: 2, 3: 2, 2: 6}, 2),
        ("heat42", {4: 2, 2: 8}, 2),
        ("heat43225", {4: 2, 3: 2, 225: 6}, 3),
        ("uniform2", {2: 10}, 2),
        ("uniform225", {225: 10}, 3),
    ]
    for mode, exp_counts, exp_kind in mode_specs:
        out_dir = os.path.join(tmp, f"ink2_{mode}")
        run([cli_bin, model_dir, "--family", "inkling2", "--inkling2-mode", mode, "--out", out_dir])
        for lyr in (0, 1):
            check_hybrid_tfpw(out_dir, lyr, refs["layers"][lyr], refs["H"], refs["I"], exp_counts, exp_kind, inspect_bin)
        if mode in ("heat432", "heat43225"):
            hashes[f"ink2_{mode}_L0"] = sha256_file(os.path.join(out_dir, "layer_00.tffpw"))
            hashes[f"ink2_{mode}_L1"] = sha256_file(os.path.join(out_dir, "layer_01.tffpw"))

    # Verify non-downgrade of existing 32 KiB table_kind=3 when writing a 2b layer into the same dir
    out_nodown = os.path.join(tmp, "ink2_heat43225")
    run([cli_bin, model_dir, "--family", "inkling2", "--inkling2-mode", "heat432", "--out", out_nodown])
    assert os.path.getsize(os.path.join(out_nodown, "pw_v4_e8p_tables.bin")) == 32768
    _, kind_after = tfq4_ref.load_pwv4_tables_with_kind(os.path.join(out_nodown, "pw_v4_e8p_tables.bin"))
    assert kind_after == 3, "Must never downgrade existing 32 KiB table_kind=3 PWV4 table"
    print("PASS Inkling-2 Hybrid MoE (NVFP4 + MXFP8: heat432, heat42, heat43225, uniform2, uniform225)")


def make_synthetic_glm(model_dir: str, src_bf16: bool, E: int = 10, H: int = 256, I: int = 128, lyr: int = 3) -> list[dict]:
    os.makedirs(model_dir, exist_ok=True)
    rng = np.random.default_rng(20261128 + (1 if src_bf16 else 0))
    valid_e4m3_codes = np.array([b for b in range(256) if (b & 0x7F) != 0x7F], dtype=np.uint8)
    tensors = []
    router_w = np.zeros((E, H), dtype=np.float32)
    for e in range(E):
        router_w[e] = rng.standard_normal(H).astype(np.float32) * (0.2 + 0.3 * e)
    tensors.append((f"model.layers.{lyr}.mlp.gate.weight", "BF16", (E, H), f32_to_bf16_bytes(router_w)))

    layer_ref = []
    for e in range(E):
        ep = {}
        for pname, R, C in (("gate_proj", I, H), ("up_proj", I, H), ("down_proj", H, I)):
            base = f"model.layers.{lyr}.mlp.experts.{e}.{pname}"
            if src_bf16:
                w = rng.standard_normal((R, C)).astype(np.float32) * 0.03
                raw = f32_to_bf16_bytes(w)
                # Exact bf16 value reference
                w_bf16 = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << np.uint32(16)).view(np.float32).reshape(R, C)
                tensors.append((base + ".weight", "BF16", (R, C), raw))
                ep[pname.replace("_proj", "")] = w_bf16
            else:
                codes = rng.choice(valid_e4m3_codes, size=(R, C))
                sc_inv = rng.uniform(0.01, 0.05, size=(R // 128, C // 128)).astype(np.float32)
                w_ref = E4M3_LUT[codes] * np.repeat(np.repeat(sc_inv, 128, axis=0), 128, axis=1)
                tensors.append((base + ".weight", "F8_E4M3", (R, C), codes.tobytes()))
                tensors.append((base + ".weight_scale_inv", "F32", sc_inv.shape, sc_inv.tobytes()))
                ep[pname.replace("_proj", "")] = w_ref
        layer_ref.append(ep)

    write_safetensors(os.path.join(model_dir, "model.safetensors"), tensors)
    cfg = {"model_type": "glm_moe_dsa"}
    if not src_bf16:
        cfg["quantization_config"] = {"quant_method": "fp8"}
    open(os.path.join(model_dir, "config.json"), "w").write(json.dumps(cfg))
    return layer_ref


def test_glm_hybrid(cli_bin: str, inspect_bin: str, tmp: str, hashes: dict[str, str]) -> None:
    glm_bin = os.path.join(
        REPO_ROOT, "build", "tfq3-glm-convert" if platform.system() == "Darwin" else "tfq3-glm-convert-cuda"
    )
    E, H, I, lyr = 10, 256, 128, 3
    for tag, src_bf16 in (("fp8", False), ("bf16", True)):
        model_dir = os.path.join(tmp, f"glm_{tag}_model")
        layer_ref = make_synthetic_glm(model_dir, src_bf16=src_bf16, E=E, H=H, I=I, lyr=lyr)
        mode_specs = [
            ("heat432", {4: 2, 3: 2, 2: 6}, 2),
            ("heat42", {4: 2, 2: 8}, 2),
            ("heat43225", {4: 2, 3: 2, 225: 6}, 3),
            ("uniform2", {2: 10}, 2),
            ("uniform225", {225: 10}, 3),
        ]
        for mode, exp_counts, exp_kind in mode_specs:
            out_dir = os.path.join(tmp, f"glm_{tag}_{mode}")
            # Verify CLI accepts --glm-mode cleanly via --dry-run
            dry_cmd = [cli_bin, model_dir, "--family", "glm", "--glm-mode", mode, "--out", out_dir, "--dry-run"]
            if src_bf16:
                dry_cmd.append("--keep-nonmoe-bf16")
            run(dry_cmd)
            # Run actual GPU conversion on layer 3
            run([glm_bin, "--model", model_dir, "--out", out_dir, "--mode", mode, "--layers", str(lyr), str(lyr)])
            check_hybrid_tfpw(out_dir, lyr, layer_ref, H, I, exp_counts, exp_kind, inspect_bin)
            if mode in ("heat432", "heat43225"):
                hashes[f"glm_{tag}_{mode}_L03"] = sha256_file(os.path.join(out_dir, f"layer_{lyr:02d}.tffpw"))
    print("PASS GLM Hybrid MoE (FP8 + BF16: heat432, heat42, heat43225, uniform2, uniform225)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--print-hashes", action="store_true", help="Print deterministic SHA-256 digests for Metal/CUDA parity")
    args = ap.parse_args()

    cli_bin = os.path.join(REPO_ROOT, "bin", "tfq3")
    inspect_bin = os.path.join(REPO_ROOT, "build", "tfq3-inspect")
    gpu_bench_bin = os.path.join(
        REPO_ROOT, "build", "tfq4-metal-test" if platform.system() == "Darwin" else "tfq4-cuda-test"
    )
    assert os.path.exists(inspect_bin), f"Missing {inspect_bin}; run `make all` first"

    hashes: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="tfq4_225_hybrid_") as tmp:
        test_pwv4_kind3_and_inspect(inspect_bin, tmp)
        test_vectors_kind3(gpu_bench_bin)
        test_tfq4_225_e2e(cli_bin, inspect_bin, tmp, hashes)
        test_inkling2_hybrid(cli_bin, inspect_bin, tmp, hashes)
        test_glm_hybrid(cli_bin, inspect_bin, tmp, hashes)

    for k, exp in GOLDEN_CROSS_PLATFORM_SHA256.items():
        got = hashes.get(k)
        assert got == exp, f"Cross-platform SHA-256 mismatch on {k}: got={got}, expected={exp}"
    print(f"PASS cross-platform Metal==CUDA golden SHA-256 parity ({len(GOLDEN_CROSS_PLATFORM_SHA256)} artifacts)")

    if args.print_hashes:
        print("SHA256_PARITY_JSON=" + json.dumps(hashes, sort_keys=True))
    print("ALL TFQ4 2.25 bpw & HYBRID MoE TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
