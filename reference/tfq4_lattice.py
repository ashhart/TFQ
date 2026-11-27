#!/usr/bin/env python3
"""
TFQ4 Research Prototype: Data-Free Sub-3-Bit (2.0-2.5 bpw) D4 & E8 Lattice Quantization
after TFQ3's exact 2D FWHT-128 rotation, benchmarked on synthetic Gaussian and real BF16 LLM weights.

Compares:
  - Uniform 2-bit & 3-bit affine (g64) baselines
  - TFQ4-D4 (2.0 bpw = 8b/4D, 2.25 bpw = 9b/4D): 4D checkerboard lattice D4 spherical codebook
  - TFQ4-E8 (2.0 bpw = 16b/8D via E8P 256-abs-codebook x 256 parity-signs, 2.25 bpw = 18b/8D):
    8D Gosset lattice E8 = D8 U (D8 + 1/2), exact O(1) Conway-Sloane nearest-point quantization
  - Exact disk footprint projections for GLM-5.3-Flash (288E) and Qwen3.8-Flash-Next (512E)
    cold-expert compression (fitting 200B-300B MoEs into 64-96 GB unified memory).
"""
import argparse
import glob
import json
import math
import os
import struct
import sys
import numpy as np

_REF_DIR = os.path.dirname(os.path.abspath(__file__))
if _REF_DIR not in sys.path:
    sys.path.insert(0, _REF_DIR)
import tfq3_reference as _tfq3_ref


def sgn_vec(seed: int, n: int) -> np.ndarray:
    idx = np.arange(n, dtype=np.uint32)
    h = ((np.uint32(seed) ^ idx) * np.uint32(2654435761)) >> np.uint32(31)
    return np.where(h == 1, -1.0, 1.0).astype(np.float32)


def fwht128_rows(x: np.ndarray) -> np.ndarray:
    orig_shape = x.shape
    a = x.reshape(-1, 128).astype(np.float32, copy=True)
    h = 1
    while h < 128:
        a = a.reshape(-1, 128 // (2 * h), 2, h)
        u = a[:, :, 0, :].copy()
        v = a[:, :, 1, :]
        a[:, :, 0, :] = u + v
        a[:, :, 1, :] = u - v
        a = a.reshape(-1, 128)
        h <<= 1
    a *= (1.0 / math.sqrt(128.0))
    return a.reshape(orig_shape)


def fwht2d_128(W: np.ndarray) -> np.ndarray:
    R, C = W.shape
    assert R % 128 == 0 and C % 128 == 0
    X = W.astype(np.float32) * sgn_vec(0x12345678, C)[None, :]
    X = fwht128_rows(X)
    X = X * sgn_vec(0x9E3779B9, R)[:, None]
    X = fwht128_rows(X.T).T
    return X


def inv_fwht2d_128(W_rot: np.ndarray) -> np.ndarray:
    R, C = W_rot.shape
    X = fwht128_rows(W_rot.T).T * sgn_vec(0x9E3779B9, R)[:, None]
    X = fwht128_rows(X) * sgn_vec(0x12345678, C)[None, :]
    return X


def snr_db(ref: np.ndarray, approx: np.ndarray) -> float:
    r = ref.astype(np.float64)
    d = r - approx.astype(np.float64)
    sig = float(np.sum(r * r))
    err = float(np.sum(d * d))
    return 10.0 * math.log10(sig / max(err, 1e-30))


def quant_uniform_affine(W: np.ndarray, bits: int, group: int = 64) -> np.ndarray:
    R, C = W.shape
    G = W.reshape(-1, group).astype(np.float32)
    qmax = float((1 << bits) - 1)
    lo = G.min(axis=1, keepdims=True)
    hi = G.max(axis=1, keepdims=True)
    sc = np.maximum((hi - lo) / qmax, 1e-12)
    q = np.clip(np.rint((G - lo) / sc), 0.0, qmax)
    return (q * sc + lo).reshape(R, C)


# ----------------------------------------------------------------------------
# Exact Conway-Sloane D_n & E_8 lattice nearest-point quantizers
# ----------------------------------------------------------------------------
def nearest_dn(x: np.ndarray) -> np.ndarray:
    """Nearest point in D_n = {z in Z^n : sum(z) is even}, vectorized over last axis."""
    f = np.rint(x)
    odd = (np.sum(f, axis=-1).astype(np.int64) & 1) != 0
    if not np.any(odd):
        return f
    diff = x - f
    worst = np.argmax(np.abs(diff), axis=-1)
    idx_odd = np.nonzero(odd)[0]
    w_odd = worst[idx_odd]
    step = np.where(diff[idx_odd, w_odd] >= 0.0, 1.0, -1.0)
    f[idx_odd, w_odd] += step
    return f


def nearest_e8(x: np.ndarray) -> np.ndarray:
    """Exact O(1) Conway-Sloane nearest point in E_8 = D_8 U (D_8 + 0.5)."""
    p0 = nearest_dn(x)
    p1 = nearest_dn(x - 0.5) + 0.5
    d0 = np.sum((x - p0) ** 2, axis=-1)
    d1 = np.sum((x - p1) ** 2, axis=-1)
    return np.where((d0 <= d1)[:, None], p0, p1)


def build_d4_codebook(n_codes: int = 256) -> np.ndarray:
    """Spherical truncated D4 + (1/2,1/2,1/2,1/2) shifted lattice codebook of size n_codes."""
    # Enumerate low-norm points in D4 + (0.25,0.25,0.25,0.25) offset to break origin tie
    pts = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            for c in range(-3, 4):
                for d in range(-3, 4):
                    if ((a + b + c + d) & 1) == 0:
                        v = np.array([a, b, c, d], dtype=np.float32) + 0.25
                        pts.append((float(np.dot(v, v)), v))
    pts.sort(key=lambda x: x[0])
    cb = np.stack([p[1] for p in pts[:n_codes]], axis=0)
    # Normalize variance to 1.0 per coordinate
    cb /= math.sqrt(float(np.mean(cb * cb)))
    return cb.astype(np.float32)


def build_e8p_abs_codebook(n_abs: int = 256) -> np.ndarray:
    """
    Build an E8P-style compact 256-entry positive-orthant E8 codebook (2.0 bpw total:
    8 bits abs-index + 8 bits sign pattern with E8 even/odd parity = 16 bits / 8 dims = 2.0 bpw!).
    Total GPU LUT size: 256 * 8 * 2B = 4 KiB (fits entirely in L1 / shared memory!).
    """
    # Positive orthant half-integer shifted E8 points: |v_i| in {0.25, 0.75, 1.25, 1.75, 2.25}
    # (equivalent to 0.5 * (odd integers) shifted by 0.25 -> every sign flip stays in shifted E8!)
    vals = [0.25, 0.75, 1.25, 1.75, 2.25]
    cands = []
    # Enumerate sorted 8-tuples by squared norm, then generate unique permutations up to n_abs
    # Using exact half-integer E8 + 0.25 shells gives exact zero-collision signed codebooks.
    rng = np.random.default_rng(2026)
    # Generate dense set of unique non-negative vectors from |E8 + 0.25|
    seen = set()
    # Analytical shells: all vectors with entries in {0.25, 0.75, 1.25, 1.75}
    # Count of 0.75, 1.25, 1.75 controls norm; prioritize lowest norms first.
    def add_vec(v):
        k = tuple(int(round(x * 4)) for x in v)
        if k not in seen:
            seen.add(k)
            cands.append((float(np.dot(v, v)), np.array(v, dtype=np.float32)))

    # Shell 0: all 0.25 (1 vec)
    add_vec([0.25] * 8)
    # Shell 1-4: 1..4 entries elevated to 0.75 / 1.25 / 1.75 (exact E8-like spherical shell coverage)
    for idx in range(8):
        v = [0.25] * 8; v[idx] = 0.75; add_vec(v)
        v = [0.25] * 8; v[idx] = 1.25; add_vec(v)
        v = [0.25] * 8; v[idx] = 1.75; add_vec(v)
    for i in range(8):
        for j in range(i + 1, 8):
            v = [0.25] * 8; v[i] = 0.75; v[j] = 0.75; add_vec(v)
            v = [0.25] * 8; v[i] = 1.25; v[j] = 0.75; add_vec(v)
            v = [0.25] * 8; v[i] = 0.75; v[j] = 1.25; add_vec(v)
            v = [0.25] * 8; v[i] = 1.25; v[j] = 1.25; add_vec(v)
    for i in range(8):
        for j in range(i + 1, 8):
            for k in range(j + 1, 8):
                v = [0.25] * 8; v[i] = 0.75; v[j] = 0.75; v[k] = 0.75; add_vec(v)
                v = [0.25] * 8; v[i] = 1.25; v[j] = 0.75; v[k] = 0.75; add_vec(v)
    # Initialize from sorted spherical E8 shells, topped up to n_abs from |N(0,1)^8|,
    # then refine with 25 Lloyd-Max steps (100% data-free synthetic Gaussian!).
    cands.sort(key=lambda x: x[0])
    pts = np.abs(rng.standard_normal((131072, 8)).astype(np.float32))
    init_list = [c[1] for c in cands[:n_abs]]
    if len(init_list) < n_abs:
        extra = pts[:n_abs - len(init_list)]
        cb = np.vstack([np.stack(init_list, axis=0), extra]).astype(np.float32)
    else:
        cb = np.stack(init_list, axis=0).astype(np.float32)
    for _ in range(25):
        # Assign in chunks
        sums = np.zeros_like(cb, dtype=np.float64)
        cnts = np.zeros(n_abs, dtype=np.int64)
        cb_n2 = np.sum(cb * cb, axis=1)[None, :]
        for s0 in range(0, len(pts), 16384):
            p = pts[s0:s0 + 16384]
            d = cb_n2 - 2.0 * (p @ cb.T)
            idx = np.argmin(d, axis=1)
            np.add.at(sums, idx, p)
            np.add.at(cnts, idx, 1)
        valid = cnts > 0
        cb[valid] = (sums[valid] / cnts[valid, None]).astype(np.float32)
    return cb


def quant_tfq4_d4(W_rot: np.ndarray, cb: np.ndarray, shrink: float = 0.90) -> np.ndarray:
    """Blockwise 128x128 RMS scaled 4D D4 codebook quantization (2.0 bpw @ 256, 2.25 bpw @ 512)."""
    R, C = W_rot.shape
    out = np.empty_like(W_rot)
    cb_n2 = np.sum(cb * cb, axis=1)[None, :]
    for br in range(R // 128):
        for bc in range(C // 128):
            blk = W_rot[br * 128:(br + 1) * 128, bc * 128:(bc + 1) * 128]
            rms = math.sqrt(float(np.mean(blk * blk)) + 1e-12) * shrink
            v = (blk / rms).reshape(-1, 4)
            idx = np.argmin(cb_n2 - 2.0 * (v @ cb.T), axis=1)
            out[br * 128:(br + 1) * 128, bc * 128:(bc + 1) * 128] = (cb[idx] * rms).reshape(128, 128)
    return out


def quant_tfq4_e8p(W_rot: np.ndarray, abs_cb: np.ndarray) -> np.ndarray:
    """
    2.0 bpw (256 abs x 256 signs = 65,536 effective 8D codewords) or 2.25 bpw (1024 abs x 256 signs).
    Each 128x128 tile stores ONE fp16 scale -> +0.001 bpw overhead!
    """
    R, C = W_rot.shape
    out = np.empty_like(W_rot)
    cb_n2 = np.sum(abs_cb * abs_cb, axis=1)[None, :]
    for br in range(R // 128):
        for bc in range(C // 128):
            blk = W_rot[br * 128:(br + 1) * 128, bc * 128:(bc + 1) * 128]
            rms = math.sqrt(float(np.mean(blk * blk)) + 1e-12)
            v = (blk / rms).reshape(-1, 8)
            sgn = np.where(v >= 0.0, 1.0, -1.0).astype(np.float32)
            av = np.abs(v)
            idx = np.argmin(cb_n2 - 2.0 * (av @ abs_cb.T), axis=1)
            q = abs_cb[idx] * sgn
            # Optimal per-tile least-squares scalar refinement (stored as the same single fp16 tile scale!)
            alpha = float(np.sum(v * q) / max(float(np.sum(q * q)), 1e-12))
            out[br * 128:(br + 1) * 128, bc * 128:(bc + 1) * 128] = (q * (rms * alpha)).reshape(128, 128)
    return out


def quant_tfq4_e8_exact_lattice(W_rot: np.ndarray, step: float) -> tuple[np.ndarray, float]:
    """Exact unbounded Conway-Sloane E8 lattice (measures information-theoretic E8 rate-distortion bound)."""
    R, C = W_rot.shape
    out = np.empty_like(W_rot)
    ent_bits = []
    for br in range(R // 128):
        for bc in range(C // 128):
            blk = W_rot[br * 128:(br + 1) * 128, bc * 128:(bc + 1) * 128]
            rms = math.sqrt(float(np.mean(blk * blk)) + 1e-12)
            v = (blk / rms).reshape(-1, 8)
            q = nearest_e8(v / step) * step
            out[br * 128:(br + 1) * 128, bc * 128:(bc + 1) * 128] = (q * rms).reshape(128, 128)
            # Empirical entropy estimate (bits/dim)
            codes = np.rint(q / step * 2.0).astype(np.int16)
            _, cnts = np.unique(codes, axis=0, return_counts=True)
            p = cnts / np.sum(cnts)
            ent_bits.append(float(-np.sum(p * np.log2(p))) / 8.0)
    return out, float(np.mean(ent_bits))


def load_real_bf16_tensor(model_dir: str, tensor_name: str, max_r: int = 1024, max_c: int = 1024) -> np.ndarray:
    idx_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.isfile(idx_path):
        wm = json.load(open(idx_path))["weight_map"]
        fn = wm[tensor_name]
    else:
        fn = "model.safetensors"
    fp = os.path.join(model_dir, fn)
    with open(fp, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hlen))
        meta = hdr[tensor_name]
        R, C = meta["shape"]
        o0, o1 = meta["data_offsets"]
        f.seek(8 + hlen + o0)
        raw = np.frombuffer(f.read(o1 - o0), dtype=np.uint16).reshape(R, C)
    rr = min(R, max_r) // 128 * 128
    cc = min(C, max_c) // 128 * 128
    u32 = raw[:rr, :cc].astype(np.uint32) << 16
    return u32.view(np.float32)


def eval_matrix(label: str, W: np.ndarray, d4_256, d4_512, e8_256, e8_1024):
    W_rot = fwht2d_128(W)
    rng = np.random.default_rng(99)
    X = rng.standard_normal((16, W.shape[1])).astype(np.float32)
    Y_ref = X @ W.T

    def report(name: str, bpw: float, W_hat: np.ndarray):
        Y_hat = X @ W_hat.T
        cos = float(np.sum(Y_ref * Y_hat) / (np.linalg.norm(Y_ref) * np.linalg.norm(Y_hat) + 1e-30))
        print(f"  {name:<26s} | {bpw:5.2f} bpw | SNR={snr_db(W, W_hat):6.2f} dB | Y-cos={cos:.6f}")

    print(f"\n=== {label} [{W.shape[0]} x {W.shape[1]}] ===")
    report("Uniform Affine 2b (g64)", 2.50, quant_uniform_affine(W, 2, 64))
    report("Uniform Affine 3b (g64)", 3.50, quant_uniform_affine(W, 3, 64))
    report("TFQ4-D4 (256 codes)",     2.00, inv_fwht2d_128(quant_tfq4_d4(W_rot, d4_256)))
    report("TFQ4-D4 (512 codes)",     2.25, inv_fwht2d_128(quant_tfq4_d4(W_rot, d4_512)))
    report("TFQ4-E8P (256x256, 4KiB)",2.00, inv_fwht2d_128(quant_tfq4_e8p(W_rot, e8_256)))
    report("TFQ4-E8P (1024x256,16KiB)",2.25, inv_fwht2d_128(quant_tfq4_e8p(W_rot, e8_1024)))
    q_e8, h_e8 = quant_tfq4_e8_exact_lattice(W_rot, step=1.18)
    report(f"Exact E8 bound (step=1.18)", h_e8, inv_fwht2d_128(q_e8))


def print_moe_projections():
    print("\n=== Projected Cold-Expert TFQ4-E8P (2.0 bpw) MoE Footprints ===")
    # GLM-5.3-Flash (288E, 42 MoE layers, H=4096, I=2048 -> 304.1B MoE params + ~11.2 GiB non-MoE/residuals)
    glm_moe_params = 42 * 288 * 3 * 4096 * 2048
    glm_nonmoe_gib = 11.2
    for name, f5, f4, f3, f2 in [
        ("TFQ3 Current (5/4/3b HeatAlloc)",      0.06, 0.24, 0.70, 0.00),
        ("TFQ4 Hybrid  (4b hot / 3b warm / 2b)", 0.00, 0.15, 0.25, 0.60),
        ("TFQ4 Ultra   (4b top-10% / 2b cold)",  0.00, 0.10, 0.00, 0.90),
    ]:
        avg_bpw = 5 * f5 + 4 * f4 + 3 * f3 + 2 * f2 + 0.002
        tot = glm_nonmoe_gib + (glm_moe_params * avg_bpw / 8.0) / (1 << 30)
        print(f"  GLM-5.3-Flash    | {name:<37s} | avg {avg_bpw:.2f} bpw | {tot:6.1f} GiB")

    # Qwen3.8-Flash-Next (512E, 48+1 MoE layers, H=3072, I=512 -> 118.4B MoE params + ~31.0 GiB non-MoE + 3b PLE)
    fn_moe_params = 49 * 512 * 3 * 3072 * 512
    fn_nonmoe_gib = 31.0
    for name, f5, f4, f3, f2 in [
        ("TFQ3 Current (5/4/3b HeatAlloc)",      0.02, 0.37, 0.61, 0.00),
        ("TFQ4 Hybrid  (4b hot / 2b cold)",      0.00, 0.20, 0.00, 0.80),
        ("TFQ4 64GB    (3b hot / 2b cold)",      0.00, 0.00, 0.20, 0.80),
    ]:
        avg_bpw = 5 * f5 + 4 * f4 + 3 * f3 + 2 * f2 + 0.002
        tot = fn_nonmoe_gib + (fn_moe_params * avg_bpw / 8.0) / (1 << 30)
        print(f"  Flash-Next 512E  | {name:<37s} | avg {avg_bpw:.2f} bpw | {tot:6.1f} GiB")


MAGIC_PWV4 = 0x34565750
MAGIC_TFP4 = 0x34504654
TFQ4_VER = 4
PWV4_TABLE_LEGACY_TRAINED = 0
PWV4_TABLE_TRAINED = 1
PWV4_TABLE_TRUE_E8P = 2
PWV4_TABLE_MAX_KNOWN = 2
TRUE_E8P_BASE_SCALE = 0.97


def nearest_e8_quarter(x: np.ndarray) -> np.ndarray:
    """Exact O(1) Conway-Sloane nearest point in the shifted lattice E_8 + (1/4)*1."""
    return nearest_e8(x - 0.25) + 0.25


def build_true_e8p_abs_codebook() -> tuple[np.ndarray, np.ndarray]:
    """Deterministic 256-entry |D_8 + 1/2| base codebook for PWV4 table_kind=2 (True QuIP# E8P).

    All 227 vectors with ||a||^2 <= 10 plus the 29 lowest-max-coordinate vectors at ||a||^2 = 12.
    Every entry is an exact odd half-integer in {0.5, 1.5, 2.5} (exact in fp16).
    Returns `(abs_cb [256, 8] float32, m_parity [256] uint8)` matching `tfq4::build_true_e8p_tables`.
    """
    cands = []
    for code in range(3**8):
        tmp = code
        m = [0] * 8
        n2_x4 = 0
        mx = 0
        for d in range(7, -1, -1):
            md = tmp % 3
            tmp //= 3
            m[d] = md
            v2 = 2 * md + 1
            n2_x4 += v2 * v2
            if md > mx:
                mx = md
        if n2_x4 <= 48:
            cands.append((n2_x4, mx, tuple(m)))
    cands.sort()
    top = cands[:256]
    abs_cb = np.array([[md + 0.5 for md in tup] for _, _, tup in top], dtype=np.float32)
    m_par = np.array([sum(tup) & 1 for _, _, tup in top], dtype=np.uint8)
    return abs_cb, m_par


def true_e8p_abs_parity(abs_cb: np.ndarray) -> np.ndarray:
    """Derive the intrinsic mod-2 parity bit of each row in a table_kind=2 half-integer LUT."""
    return (np.sum(np.rint(abs_cb - 0.5).astype(np.int64), axis=1) & 1).astype(np.uint8)


def write_pwv4_tables(path: str, lut: np.ndarray, table_kind: int = PWV4_TABLE_LEGACY_TRAINED) -> None:
    """Write a 16,384-byte PWV4 v4 codebook file with explicit `table_kind`."""
    if table_kind < 0 or table_kind > PWV4_TABLE_MAX_KNOWN:
        raise ValueError(f"Unsupported PWV4 table_kind={table_kind}")
    assert lut.shape == (256, 8)
    buf = bytearray(16384)
    struct.pack_into("<IIIII", buf, 0, MAGIC_PWV4, TFQ4_VER, 256, 8, int(table_kind))
    lut_f16 = np.ascontiguousarray(lut, dtype=np.float16)
    buf[64 : 64 + lut_f16.nbytes] = lut_f16.tobytes()
    with open(path, "wb") as f:
        f.write(buf)


def decode_true_e8p_codes(codes_u16: np.ndarray, abs_cb: np.ndarray) -> np.ndarray:
    """Decode packed 16-bit True QuIP# E8P (table_kind=2) codewords to points in E_8 + 1/4.

    Bit layout per uint16 codeword:
      - bits [7:0]  : abs_idx in [0, 255] -> abs_cb[abs_idx] in {0.5, 1.5, 2.5, ...}^8
      - bits [14:8] : 7 independent sign bits s[0..6] (1 = negative)
      - derived s[7]: parity bit = (popcount(s[0..6]) ^ m_parity[abs_idx]) & 1
      - bit  [15]   : coset bit (0 -> y - 1/4 in D_8 + 1/4, 1 -> -(y - 1/4) in D_8 + 3/4)
    """
    codes = np.asarray(codes_u16, dtype=np.uint32)
    idx = (codes & np.uint32(0xFF)).astype(np.int32)
    s7 = (codes >> np.uint32(8)) & np.uint32(0x7F)
    coset = (codes >> np.uint32(15)) & np.uint32(1)
    m_par = true_e8p_abs_parity(abs_cb).astype(np.uint32)
    pc = np.zeros_like(s7)
    for b in range(7):
        pc ^= (s7 >> np.uint32(b)) & np.uint32(1)
    s_bit7 = (pc ^ m_par[idx]) & np.uint32(1)
    sm8 = s7 | (s_bit7 << np.uint32(7))
    bits_mask = (sm8[..., None] >> np.arange(8, dtype=np.uint32)) & np.uint32(1)
    sgn = np.where(bits_mask != 0, -1.0, 1.0).astype(np.float32)
    q = abs_cb[idx] * sgn - np.float32(0.25)
    return np.where(coset[..., None] != 0, -q, q).astype(np.float32)


def quant_true_e8p_e8_quarter(
    W_rot: np.ndarray, abs_cb: np.ndarray, base_scale: float = TRUE_E8P_BASE_SCALE, ls_refine: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact nearest-point encoder for True QuIP# E8P (table_kind=2, 65,536 points in E_8 + 1/4).

    Returns `(W_rot_hat [R, C] f32, codes_u16 [R, C//8] u16, scales_f16 [R//128, C//128] f16)`.
    """
    R, C = W_rot.shape
    assert R % 128 == 0 and C % 128 == 0
    n_br, n_bc = R // 128, C // 128
    m_par = true_e8p_abs_parity(abs_cb).astype(np.int64)
    cb64 = abs_cb.astype(np.float64)
    cb_n2 = np.sum(cb64 * cb64, axis=1)

    out = np.empty((R, C), dtype=np.float32)
    codes_out = np.empty((R, C // 8), dtype=np.uint16)
    scales_out = np.empty((n_br, n_bc), dtype=np.float16)

    for br in range(n_br):
        for bc in range(n_bc):
            blk = W_rot[br * 128 : (br + 1) * 128, bc * 128 : (bc + 1) * 128].astype(np.float64)
            rms = math.sqrt(float(np.mean(blk * blk)) + 1e-12)
            sc_f16 = np.float16(rms * base_scale)
            sc_eff = float(sc_f16)
            V = (blk / max(sc_eff, 1e-12)).reshape(-1, 8)  # [2048, 8]
            N = V.shape[0]
            best_dist = np.full(N, np.inf, dtype=np.float64)
            best_code = np.zeros(N, dtype=np.uint16)

            for coset in (0, 1):
                Z = (V + 0.25) if coset == 0 else (-V + 0.25)
                aZ = np.abs(Z)
                nb = (Z < 0.0).astype(np.int64)
                zp = np.sum(nb, axis=1) & 1
                zn2 = np.sum(Z * Z, axis=1)
                prod = aZ[:, None, :] * cb64[None, :, :]
                dot = np.sum(prod, axis=2)
                min_p = np.min(prod, axis=2)
                nf = zp[:, None] != m_par[None, :]
                adj = dot - np.where(nf, 2.0 * min_p, 0.0)
                dist = zn2[:, None] + cb_n2[None, :] - 2.0 * adj
                kb = np.argmin(dist, axis=1)
                db = dist[np.arange(N), kb]
                better = db < best_dist
                if np.any(better):
                    bi = np.nonzero(better)[0]
                    kbi = kb[bi]
                    nb_sel = nb[bi].copy()
                    nfi = np.nonzero(nf[bi, kbi])[0]
                    if len(nfi):
                        fd = np.argmin(prod[bi[nfi], kbi[nfi]], axis=1)
                        nb_sel[nfi, fd] ^= 1
                    s7 = (
                        (nb_sel[:, 0] << 0) | (nb_sel[:, 1] << 1) | (nb_sel[:, 2] << 2) | (nb_sel[:, 3] << 3)
                        | (nb_sel[:, 4] << 4) | (nb_sel[:, 5] << 5) | (nb_sel[:, 6] << 6)
                    )
                    best_dist[bi] = db[bi]
                    best_code[bi] = ((coset << 15) | (s7 << 8) | kbi).astype(np.uint16)

            q_raw = decode_true_e8p_codes(best_code, abs_cb).reshape(128, 128)
            if ls_refine:
                alpha = float(np.sum(blk * q_raw.astype(np.float64)) / max(float(np.sum(q_raw.astype(np.float64) ** 2)), 1e-12))
                sc_f16 = np.float16(alpha)
                sc_eff = float(sc_f16)
            scales_out[br, bc] = sc_f16
            codes_out[br * 128 : (br + 1) * 128, bc * 16 : (bc + 1) * 16] = best_code.reshape(128, 16)
            out[br * 128 : (br + 1) * 128, bc * 128 : (bc + 1) * 128] = (q_raw * np.float32(sc_eff)).astype(np.float32)
    return out, codes_out, scales_out


def load_pwv4_tables_with_kind(path: str) -> tuple[np.ndarray, int]:
    """Load and validate a 16 KiB `pw_v4_e8p_tables.bin` codebook, returning `(lut [256, 8] f32, table_kind)`."""
    raw = open(path, "rb").read()
    if len(raw) != 16384:
        raise ValueError(f"Expected 16384-byte PWV4 file, got {len(raw)}")
    magic, ver, k_codes, dim, table_kind = struct.unpack_from("<IIIII", raw, 0)
    if magic != MAGIC_PWV4 or ver != TFQ4_VER or k_codes != 256 or dim != 8:
        raise ValueError(f"Bad PWV4 header: magic=0x{magic:08x} ver={ver} k={k_codes} dim={dim}")
    if table_kind > PWV4_TABLE_MAX_KNOWN:
        raise ValueError(f"Unknown PWV4 table_kind={table_kind} (supported: 0/1=trained, 2=true-E8P)")
    lut = np.frombuffer(raw, dtype=np.float16, count=256 * 8, offset=64).reshape(256, 8).astype(np.float32)
    if not np.all(np.isfinite(lut)) or np.any(lut < 0.0):
        raise ValueError("PWV4 e8p_lut contains non-finite or negative entries")
    if table_kind == PWV4_TABLE_TRUE_E8P:
        two_v = lut * 2.0
        r = np.rint(two_v)
        if np.any(lut < 0.5) or np.any(np.abs(two_v - r) > 1e-3) or np.any((r.astype(np.int64) & 1) == 0):
            raise ValueError("PWV4 table_kind=2 (true-E8P) LUT entries must be positive odd half-integers")
    return lut, int(table_kind)


def load_pwv4_tables(path: str) -> np.ndarray:
    """Load and validate a 16 KiB `pw_v4_e8p_tables.bin` codebook (`float32` [256, 8])."""
    lut, _ = load_pwv4_tables_with_kind(path)
    return lut


def dequantize_tfp4_tensor(
    shard_path: str, tables_path: str, tensor_name: str, max_blocks: int | None = None, verify_checksum: bool = True
) -> np.ndarray:
    """Dequantize a TFP4 v4 (2.001 bpw E8P) tensor (or its top-left tiles) to float32."""
    lut, table_kind = load_pwv4_tables_with_kind(tables_path)
    with open(shard_path, "rb") as f:
        hdr = struct.unpack("<IIIIQQQQQQ", f.read(64))
        magic, ver, nt, pad0, off_meta, off_scales, off_weights, tot_sc, tot_w, pad1 = hdr
        if magic != MAGIC_TFP4 or ver != TFQ4_VER:
            raise ValueError(f"Bad TFP4 header in {shard_path}")
        if verify_checksum:
            _tfq3_ref.verify_tfpn_like_checksum(
                f, off_meta, nt * 128, off_scales, tot_sc * 2, off_weights, tot_w * 4, pad0, pad1, f"TFP4 ({shard_path})"
            )
        f.seek(off_meta)
        found = None
        for _ in range(nt):
            rec = f.read(128)
            nm = rec[:80].split(b"\x00", 1)[0].decode("utf-8")
            if nm == tensor_name:
                found = struct.unpack("<IIIIQQfIII", rec[80:])
                break
        if found is None:
            raise KeyError(f"Tensor {tensor_name} not in {shard_path}")
        R, C, bits, sc_off, w_off, w_cnt, _, _, _, _ = found
        if bits != 2:
            raise ValueError(f"Unexpected TFP4 bits={bits} for {tensor_name}")
        n_br, n_bc = R // 128, C // 128
        rb_lim = n_br if max_blocks is None else min(max_blocks, n_br)
        cb_lim = n_bc if max_blocks is None else min(max_blocks, n_bc)
        f.seek(off_scales + sc_off * 2)
        scales = np.frombuffer(f.read(n_br * n_bc * 2), dtype=np.float16).reshape(n_br, n_bc).astype(np.float32)
        f.seek(off_weights + w_off * 4)
        words = np.frombuffer(f.read(w_cnt * 4), dtype=np.uint32).reshape(R, C // 16)

    rr, cc = rb_lim * 128, cb_lim * 128
    sub_w = words[:rr, : cc // 16]
    codes = np.empty((rr, cc // 8), dtype=np.uint16)
    codes[:, 0::2] = (sub_w & np.uint32(0xFFFF)).astype(np.uint16)
    codes[:, 1::2] = (sub_w >> np.uint32(16)).astype(np.uint16)

    if table_kind == PWV4_TABLE_TRUE_E8P:
        W_rot = decode_true_e8p_codes(codes, lut).reshape(rr, cc)
    else:
        k_idx = (codes & np.uint16(0xFF)).astype(np.int32)
        sm = (codes >> np.uint16(8)).astype(np.uint32)
        bits_mask = (sm[..., None] >> np.arange(8, dtype=np.uint32)) & np.uint32(1)
        sgn = np.where(bits_mask != 0, -1.0, 1.0).astype(np.float32)
        W_rot = (lut[k_idx] * sgn).reshape(rr, cc)
    sc_tile = np.repeat(np.repeat(scales[:rb_lim, :cb_lim], 128, axis=0), 128, axis=1)
    return inv_fwht2d_128(W_rot * sc_tile)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen25-dir", default="")
    ap.add_argument("--llama8b-dir", default="")
    ap.add_argument("--shard", default="", help="Decode a TFP4 .tffp4 shard tensor")
    ap.add_argument("--tables", default="", help="Path to pw_v4_e8p_tables.bin")
    ap.add_argument("--tensor", default="", help="Tensor name inside --shard")
    ap.add_argument("--max-blocks", type=int, default=2)
    ap.add_argument("--self-test", action="store_true", help="Fast invertibility & E8P roundtrip check")
    args = ap.parse_args()

    if args.shard:
        if not args.tables or not args.tensor:
            raise SystemExit("--shard requires --tables and --tensor")
        W = dequantize_tfp4_tensor(args.shard, args.tables, args.tensor, max_blocks=args.max_blocks)
        print(f"Decoded {args.tensor} {tuple(W.shape)}: mean={W.mean():.6f} std={W.std():.6f} rms={np.sqrt(np.mean(W**2)):.6f}")
        return

    if args.self_test:
        rng = np.random.default_rng(7)
        W = rng.standard_normal((256, 256)).astype(np.float32) * 0.02
        W_rot = fwht2d_128(W)
        err = float(np.max(np.abs(W - inv_fwht2d_128(W_rot))))
        assert err < 1e-5, f"2D FWHT inverse error: {err}"
        cb = build_e8p_abs_codebook(256)
        W_hat = inv_fwht2d_128(quant_tfq4_e8p(W_rot, cb))
        s = snr_db(W, W_hat)
        assert s > 9.5, f"Unexpectedly low E8P SNR: {s:.2f} dB"
        # True QuIP# E8P (table_kind=2, E8 + 1/4) exact pack/unpack + lattice membership check
        e8p_cb, _ = build_true_e8p_abs_codebook()
        W_rot_k2, codes_k2, sc_k2 = quant_true_e8p_e8_quarter(W_rot, e8p_cb)
        q_dec = decode_true_e8p_codes(codes_k2, e8p_cb).reshape(256, 256)
        sc_tile = np.repeat(np.repeat(sc_k2.astype(np.float32), 128, axis=0), 128, axis=1)
        assert float(np.max(np.abs(W_rot_k2 - q_dec * sc_tile))) < 1e-5
        # Verify decoded codewords lie strictly in E8 + 1/4
        pts = decode_true_e8p_codes(codes_k2.reshape(-1)[:512], e8p_cb)
        assert float(np.max(np.abs(pts - nearest_e8_quarter(pts)))) < 1e-5, "True E8P point outside E8 + 1/4!"
        s_k2 = snr_db(W, inv_fwht2d_128(W_rot_k2))
        assert s_k2 > 10.2, f"Unexpectedly low true-E8P SNR: {s_k2:.2f} dB"
        print(f"PASS TFQ4-E8P self-test: FWHT max_err={err:.2e}, trained SNR={s:.2f} dB, true-E8P(kind=2) SNR={s_k2:.2f} dB")
        return

    print("Building data-free D4 (256/512) and E8P (256/1024) universal Gaussian codebooks...")
    d4_256 = build_d4_codebook(256)
    d4_512 = build_d4_codebook(512)
    e8_256 = build_e8p_abs_codebook(256)
    e8_1024 = build_e8p_abs_codebook(1024)

    rng = np.random.default_rng(0)
    W_gauss = rng.standard_normal((1024, 1024)).astype(np.float32) * 0.02
    eval_matrix("Synthetic Unit Gaussian", W_gauss, d4_256, d4_512, e8_256, e8_1024)

    if args.qwen25_dir and os.path.isdir(args.qwen25_dir):
        W_q = load_real_bf16_tensor(args.qwen25_dir, "model.layers.0.mlp.down_proj.weight", 1024, 1024)
        eval_matrix("Real Qwen2.5-1.5B L0 mlp.down_proj", W_q, d4_256, d4_512, e8_256, e8_1024)
    if args.llama8b_dir and os.path.isdir(args.llama8b_dir):
        W_l = load_real_bf16_tensor(args.llama8b_dir, "model.layers.15.mlp.down_proj.weight", 1024, 1024)
        eval_matrix("Real Llama-3.1-8B L15 mlp.down_proj", W_l, d4_256, d4_512, e8_256, e8_1024)

    print_moe_projections()


if __name__ == "__main__":
    main()
