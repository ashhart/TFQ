#!/usr/bin/env python3
"""
Comprehensive ecosystem upgrade verification suite:
  1. TFQ4 2.00 bpw (table_kind=0/2) and 2.25 bpw E8P-1024 (table_kind=3, b_bits=225,
     9 uint32 words/128 cols) decode & linear parity (cos >= 0.999999) on CPU & Apple Silicon MPS
     (including r1 == r16 bit-exactness, Col-TP row_off, Row-TP k_sign_off, and HF load).
  2. Hybrid TFPW v3 MoE (b_bits in {5, 4, 3, 2, 225}) round-trip, XXH64 verification,
     auto-loading pw_v4_e8p_tables.bin, SwiGLU / GLM clamp / Inkling-2 tanh softcap parity
     on CPU & MPS (and CUDA when available).
  3. vLLM plugin upgrade: Tfq3Config.from_checkpoint_dir(), Tfq4CheckpointReader,
     Tfq4LinearMethod (2.00 & 2.25 bpw, merged gate_up_proj, Col-TP=2, Row-TP=2),
     Tfq3MoeMethod + load_tfq3_moe_weights (pure TFQ3 & hybrid MoE, router_logits & pre-routed).
  4. tfq3-eval CLI end-to-end: --inspect-only across TFQ3/TFQ4/Hybrid-MoE, XXH64 tamper
     rejection, synthetic HF model eval (--text-file, --ref-model, fp64 KL, Top-1, throughput),
     HF Model Card & JSON generation, and zero private-path/hostname leak check.
  5. tfq3 package surface (`import tfq3` version 3.1.0, submodule exports).
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in (
    REPO_ROOT,
    os.path.join(REPO_ROOT, "reference"),
    os.path.join(REPO_ROOT, "integrations", "pytorch"),
    os.path.join(REPO_ROOT, "integrations", "vllm"),
    os.path.join(REPO_ROOT, "tools"),
):
    if _sub not in sys.path:
        sys.path.insert(0, _sub)

import tfq3_reference as tfq3_ref
import tfq4_lattice as tfq4_ref
import tfq_mps
import tfq_torch
import tfq_moe
import tfq3_quant
import tfq3_eval

ALIGN_16K = 16384
PWV3_CANON = os.path.join(REPO_ROOT, "reference", "tables", "pw_v3_tables.bin")
PWV4_CANON = os.path.join(REPO_ROOT, "reference", "tables", "pw_v4_e8p_tables.bin")


def _align16k(n: int) -> int:
    return (n + ALIGN_16K - 1) & ~(ALIGN_16K - 1)


def _align128(n: int) -> int:
    return ((n + 127) // 128) * 128


def _cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.detach().float().reshape(-1).cpu()
    bf = b.detach().float().reshape(-1).cpu()
    return float(F.cosine_similarity(af.unsqueeze(0), bf.unsqueeze(0)).item())


def build_e8p1024_abs_codebook() -> np.ndarray:
    """
    Build a deterministic 1024-entry odd half-integer LUT whose first 256 entries are
    bit-identical to PWV4 table_kind=2 (`build_true_e8p_abs_codebook`).
    """
    base256, _ = tfq4_ref.build_true_e8p_abs_codebook()
    seen = {tuple(int(round(v - 0.5)) for v in row) for row in base256}
    cands = []
    # Enumerate positive odd half-integer 8D vectors with m_i in {0, 1, 2, 3} sorted by norm
    for code in range(4**8):
        tmp = code
        m = [0] * 8
        n2_x4 = 0
        mx = 0
        for d in range(7, -1, -1):
            md = tmp & 3
            tmp >>= 2
            m[d] = md
            v2 = 2 * md + 1
            n2_x4 += v2 * v2
            if md > mx:
                mx = md
        if n2_x4 <= 80:
            tup = tuple(m)
            if tup not in seen:
                cands.append((n2_x4, mx, tup))
    cands.sort()
    extra = np.array([[md + 0.5 for md in tup] for _, _, tup in cands[: 1024 - 256]], dtype=np.float32)
    lut1024 = np.vstack([base256.astype(np.float32), extra])
    assert lut1024.shape == (1024, 8)
    assert np.array_equal(lut1024[:256], base256)
    return lut1024


def write_pwv4_file(path: str, lut: np.ndarray, table_kind: int) -> None:
    if table_kind == 3:
        assert lut.shape == (1024, 8)
        buf = bytearray(32768)
        struct.pack_into("<IIIII", buf, 0, tfq_torch.MAGIC_PWV4, 4, 1024, 8, 3)
    else:
        assert lut.shape == (256, 8)
        buf = bytearray(16384)
        struct.pack_into("<IIIII", buf, 0, tfq_torch.MAGIC_PWV4, 4, 256, 8, int(table_kind))
    lut_f16 = np.ascontiguousarray(lut, dtype=np.float16)
    buf[64 : 64 + lut_f16.nbytes] = lut_f16.tobytes()
    with open(path, "wb") as f:
        f.write(buf)


def random_tfq4_words(R: int, C: int, bits: int, rng: np.random.Generator) -> np.ndarray:
    if bits == 225:
        abs_idx = rng.integers(0, 1024, size=(R, C // 8), dtype=np.uint32)
        s7 = rng.integers(0, 128, size=(R, C // 8), dtype=np.uint32)
        coset = rng.integers(0, 2, size=(R, C // 8), dtype=np.uint32)
        return tfq_torch.pack_tfq4_e8p1024_codes(abs_idx, s7, coset).reshape(-1)
    assert bits == 2
    return rng.integers(0, 1 << 32, size=(R * (C // 16),), dtype=np.uint64).astype(np.uint32)


def write_synthetic_tfp4_shard(
    path: str,
    specs: list[tuple[str, int, int, int]],
    lut: np.ndarray,
    table_kind: int,
    rng: np.random.Generator,
    *,
    with_xxh64: bool = True,
) -> dict[str, np.ndarray]:
    """Write a valid 16 KiB-aligned TFP4 v4 shard supporting b_bits in {2, 225}."""
    metas = bytearray()
    scales_list: list[np.ndarray] = []
    words_list: list[np.ndarray] = []
    ref_weights: dict[str, np.ndarray] = {}
    sc_off = 0
    w_off = 0

    for name, orig_r, orig_c, bits in specs:
        R, C = _align128(orig_r), _align128(orig_c)
        n_sc = (R // 128) * (C // 128)
        sc = (0.02 + 0.02 * rng.random((R // 128, C // 128))).astype(np.float16)
        wd = random_tfq4_words(R, C, bits, rng)
        eff_tk = 3 if bits == 225 else (2 if table_kind == 3 else table_kind)
        eff_lut = lut if eff_tk == 3 else lut[:256]
        W_hat = tfq_torch.decode_tfq4_words_np(wd, sc, R, C, eff_lut, eff_tk)[:orig_r, :orig_c].copy()
        ref_weights[name] = W_hat

        nb = name.encode("utf-8")[:79].ljust(80, b"\x00")
        metas.extend(
            nb
            + struct.pack(
                "<IIIIQQfIII",
                R,
                C,
                bits,
                sc_off,
                w_off,
                int(wd.size),
                12.5 if bits == 225 else 10.8,
                orig_r,
                orig_c,
                1 if (R != orig_r or C != orig_c) else 0,
            )
        )
        scales_list.append(sc.reshape(-1))
        words_list.append(wd)
        sc_off += n_sc
        w_off += int(wd.size)

    meta_b = bytes(metas)
    sc_b = np.concatenate(scales_list).astype("<f2").tobytes()
    wd_b = np.concatenate(words_list).astype("<u4").tobytes()

    off_meta = 64
    off_scales = _align16k(off_meta + len(meta_b))
    off_weights = _align16k(off_scales + len(sc_b))
    tot_sz = _align16k(off_weights + len(wd_b))

    flags = tfq3_ref.FLAG_XXH64_PRESENT if with_xxh64 else 0
    xxh = tfq3_ref.xxh64_sections((meta_b, sc_b, wd_b)) if with_xxh64 else 0
    hdr = struct.pack(
        "<IIIIQQQQQQ",
        tfq_torch.MAGIC_TFP4,
        4,
        len(specs),
        flags,
        off_meta,
        off_scales,
        off_weights,
        sc_off,
        w_off,
        xxh,
    )
    buf = bytearray(tot_sz)
    buf[:64] = hdr
    buf[off_meta : off_meta + len(meta_b)] = meta_b
    buf[off_scales : off_scales + len(sc_b)] = sc_b
    buf[off_weights : off_weights + len(wd_b)] = wd_b
    with open(path, "wb") as f:
        f.write(buf)
    return ref_weights


def write_synthetic_hybrid_tfpw(
    path: str,
    layer_idx: int,
    expert_bits: list[int],
    H: int,
    I: int,
    tbl3: dict,
    lut4: np.ndarray,
    table_kind4: int,
    rng: np.random.Generator,
    *,
    with_xxh64: bool = True,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Write a valid 16 KiB-aligned hybrid TFPW v3 shard supporting b_bits in {2, 225, 3, 4, 5}."""
    n_exp = len(expert_bits)
    metas = bytearray()
    scales_list: list[np.ndarray] = []
    words_list: list[np.ndarray] = []
    ref_experts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    sc_off = 0
    w_off = 0

    n_sc_gu = (I // 128) * (H // 128)
    n_sc_dn = (H // 128) * (I // 128)

    for e, bits in enumerate(expert_bits):
        if bits == 4 and (w_off % 4) != 0:
            pad_w = 4 - (w_off % 4)
            words_list.append(np.zeros(pad_w, dtype=np.uint32))
            w_off += pad_w

        sc_g = (0.02 + 0.02 * rng.random((I // 128, H // 128))).astype(np.float16)
        sc_u = (0.02 + 0.02 * rng.random((I // 128, H // 128))).astype(np.float16)
        sc_d = (0.02 + 0.02 * rng.random((H // 128, I // 128))).astype(np.float16)

        if bits in (3, 4, 5):
            gu_w = (I // 16) * (H // 128) * (bits * 2) * 32
            dn_w = (H // 16) * (I // 128) * (bits * 2) * 32
            wd_g = rng.integers(0, 1 << 32, size=gu_w, dtype=np.uint64).astype(np.uint32)
            wd_u = rng.integers(0, 1 << 32, size=gu_w, dtype=np.uint64).astype(np.uint32)
            wd_d = rng.integers(0, 1 << 32, size=dn_w, dtype=np.uint64).astype(np.uint32)
            ref_g = tfq3_ref.decode_tfpn_fast_np(wd_g, sc_g, I, H, bits, tbl3)
            ref_u = tfq3_ref.decode_tfpn_fast_np(wd_u, sc_u, I, H, bits, tbl3)
            ref_d = tfq3_ref.decode_tfpn_fast_np(wd_d, sc_d, H, I, bits, tbl3)
        else:
            wd_g = random_tfq4_words(I, H, bits, rng)
            wd_u = random_tfq4_words(I, H, bits, rng)
            wd_d = random_tfq4_words(H, I, bits, rng)
            eff_tk = 3 if bits == 225 else (2 if table_kind4 == 3 else table_kind4)
            eff_lut = lut4 if eff_tk == 3 else lut4[:256]
            ref_g = tfq_torch.decode_tfq4_words_np(wd_g, sc_g, I, H, eff_lut, eff_tk)
            ref_u = tfq_torch.decode_tfq4_words_np(wd_u, sc_u, I, H, eff_lut, eff_tk)
            ref_d = tfq_torch.decode_tfq4_words_np(wd_d, sc_d, H, I, eff_lut, eff_tk)

        gw_off = w_off
        uw_off = gw_off + int(wd_g.size)
        dw_off = uw_off + int(wd_u.size)
        w_off = dw_off + int(wd_d.size)

        gs_off = sc_off
        us_off = gs_off + n_sc_gu
        ds_off = us_off + n_sc_gu
        sc_off = ds_off + n_sc_dn

        metas.extend(
            struct.pack(
                "<IIQQQIIIfffII",
                bits,
                e,
                gw_off,
                uw_off,
                dw_off,
                gs_off,
                us_off,
                ds_off,
                18.0,
                18.5,
                19.0,
                0,
                0,
            )
        )
        scales_list.extend([sc_g.reshape(-1), sc_u.reshape(-1), sc_d.reshape(-1)])
        words_list.extend([wd_g, wd_u, wd_d])
        ref_experts.append((ref_g, ref_u, ref_d))

    meta_b = bytes(metas)
    sc_b = np.concatenate(scales_list).astype("<f2").tobytes()
    wd_b = np.concatenate(words_list).astype("<u4").tobytes()

    off_meta = 144
    off_scales = _align16k(off_meta + len(meta_b))
    off_weights = _align16k(off_scales + len(sc_b))
    tot_sz = _align16k(off_weights + len(wd_b))

    flags = tfq3_ref.FLAG_XXH64_PRESENT if with_xxh64 else 0
    c_meta = tfq3_ref.xxh64(meta_b, 0) if with_xxh64 else 0
    c_scales = tfq3_ref.xxh64(sc_b, 0) if with_xxh64 else 0
    c_weights = tfq3_ref.xxh64(wd_b, 0) if with_xxh64 else 0
    tag_b = b"hybrid_test".ljust(32, b"\x00")

    hdr = struct.pack(
        "<8I6Q4Q32s",
        tfq3_ref.MAGIC_TFPW,
        3,
        layer_idx,
        n_exp,
        3 * n_exp,
        H,
        I,
        flags,
        3 * n_exp * H * I,
        w_off,
        sc_off,
        off_meta,
        off_scales,
        off_weights,
        c_meta,
        c_scales,
        c_weights,
        0,
        tag_b,
    )
    buf = bytearray(tot_sz)
    buf[:144] = hdr
    buf[off_meta : off_meta + len(meta_b)] = meta_b
    buf[off_scales : off_scales + len(sc_b)] = sc_b
    buf[off_weights : off_weights + len(wd_b)] = wd_b
    with open(path, "wb") as f:
        f.write(buf)
    return ref_experts


def _write_safetensors_f32(path: str, tensors: dict[str, torch.Tensor]) -> None:
    hdr, blobs, off = {}, [], 0
    for k, t in sorted(tensors.items()):
        b = t.detach().to(torch.float32).contiguous().cpu().numpy().tobytes()
        hdr[k] = {"dtype": "F32", "shape": list(t.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    hb = json.dumps(hdr).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for b in blobs:
            f.write(b)


# ----------------------------------------------------------------------------
# Test 1: TFQ4 2.00 & 2.25 bpw (table_kind=0/2/3) CPU & MPS parity + bit-exact
# ----------------------------------------------------------------------------
def test_tfq4_200_and_225_cpu_and_mps() -> None:
    rng = np.random.default_rng(1001)
    lut0 = tfq4_ref.load_pwv4_tables(PWV4_CANON)
    lut1024 = build_e8p1024_abs_codebook()
    lut2 = lut1024[:256]

    # Validate 16 KiB & 32 KiB PWV4 roundtrip loader
    with tempfile.TemporaryDirectory(prefix="tfq_pwv4_") as tmp:
        p2 = os.path.join(tmp, "pw2.bin")
        p3 = os.path.join(tmp, "pw3.bin")
        write_pwv4_file(p2, lut2, 2)
        write_pwv4_file(p3, lut1024, 3)
        l2_r, k2_r, b2_r = tfq_torch.load_pwv4_tables_auto(p2)
        l3_r, k3_r, b3_r = tfq_torch.load_pwv4_tables_auto(p3)
        assert k2_r == 2 and len(b2_r) == 16384 and np.allclose(l2_r, lut2)
        assert k3_r == 3 and len(b3_r) == 32768 and np.allclose(l3_r, lut1024)
        assert np.array_equal(l3_r[:256], l2_r)

    # Exact bit-level check of table_kind=3 9-word packing vs reference formula
    R, C = 256, 384
    abs_idx = rng.integers(0, 1024, size=(R, C // 8), dtype=np.uint32)
    s7 = rng.integers(0, 128, size=(R, C // 8), dtype=np.uint32)
    coset = rng.integers(0, 2, size=(R, C // 8), dtype=np.uint32)
    wd225 = tfq_torch.pack_tfq4_e8p1024_codes(abs_idx, s7, coset)
    assert wd225.shape == (R, (C // 128) * 9)
    sc = (0.03 + 0.02 * rng.random((R // 128, C // 128))).astype(np.float16)

    # Manual scalar reference decode of first 32 vectors of row 7
    W_dec = tfq_torch.decode_tfq4_words_np(wd225, sc, R, C, lut1024, 3)
    W_rot_manual = np.zeros((R, C), dtype=np.float32)
    m_par = np.sum(np.floor(lut1024).astype(np.int64), axis=1) & 1
    for r in range(R):
        for blk in range(C // 128):
            w_blk = wd225[r, blk * 9 : blk * 9 + 9]
            for j in range(16):
                low16 = int((w_blk[j // 2] >> (16 * (j & 1))) & 0xFFFF)
                hi2 = int((w_blk[8] >> (2 * j)) & 3)
                idx = (low16 & 0xFF) | (hi2 << 8)
                s7_v = (low16 >> 8) & 0x7F
                c_v = (low16 >> 15) & 1
                s7_bit = (bin(s7_v).count("1") ^ int(m_par[idx])) & 1
                sm = s7_v | (s7_bit << 7)
                sgn = np.array([-1.0 if ((sm >> d) & 1) else 1.0 for d in range(8)], dtype=np.float32)
                q = lut1024[idx] * sgn - 0.25
                if c_v:
                    q = -q
                W_rot_manual[r, blk * 128 + j * 8 : blk * 128 + (j + 1) * 8] = q * float(sc[r // 128, blk])
    W_ref_manual = tfq4_ref.inv_fwht2d_128(W_rot_manual)
    assert float(np.max(np.abs(W_dec - W_ref_manual))) < 1e-5, "table_kind=3 vectorized vs scalar decode mismatch"

    # Tfq4Linear negative controls
    try:
        tfq_torch.Tfq4Linear(256, 256, random_tfq4_words(256, 256, 2, rng), sc[:2, :2], lut2, table_kind=3)
        raise AssertionError("Tfq4Linear accepted 256-row LUT for table_kind=3")
    except ValueError:
        pass
    try:
        tfq_torch.Tfq4Linear(256, 256, random_tfq4_words(256, 256, 2, rng), sc[:2, :2], lut1024, table_kind=3)
        raise AssertionError("Tfq4Linear accepted 8-word/block payload for table_kind=3")
    except ValueError:
        pass

    # CPU & Apple Silicon MPS parity across table_kind = 0, 2, 3 and non-128 orig_r/orig_c
    orig_r, orig_c = 210, 340
    pad_r, pad_c = _align128(orig_r), _align128(orig_c)
    sc_test = (0.03 + 0.02 * rng.random((pad_r // 128, pad_c // 128))).astype(np.float16)
    bias = torch.randn(orig_r, dtype=torch.float32)
    has_mps = tfq_mps.mps_kernels_available()

    for tk, bits, lut in ((0, 2, lut0), (2, 2, lut2), (3, 225, lut1024)):
        wd = random_tfq4_words(pad_r, pad_c, bits, rng)
        lin = tfq_torch.Tfq4Linear(orig_c, orig_r, wd, sc_test, lut, bias=bias, table_kind=tk)
        W_ref = torch.from_numpy(
            tfq_torch.decode_tfq4_words_np(wd, sc_test, pad_r, pad_c, lut, tk)[:orig_r, :orig_c]
        )
        torch.manual_seed(42 + tk)
        xs = [torch.randn(rows, orig_c, dtype=torch.float32) for rows in (1, 4, 16, 20)]
        for x in xs:
            y_cpu = lin(x)
            y_ref = F.linear(x, W_ref, bias)
            cos_cpu = _cos_sim(y_cpu, y_ref)
            assert cos_cpu >= 0.999999, f"CPU TFQ4 kind={tk} rows={x.shape[0]} cos={cos_cpu}"

        if has_mps:
            lin_mps = tfq_torch.Tfq4Linear(orig_c, orig_r, wd, sc_test, lut, bias=bias, table_kind=tk).to("mps")
            # Compare against fp16-input reference (since MPS fwht_x reads half inputs)
            y_r1 = None
            for x in xs:
                xh = x.to(torch.float16)
                y_ref_h = F.linear(xh.float(), W_ref, bias)
                y_mps = lin_mps(xh.to("mps")).cpu().float()
                cos_mps = _cos_sim(y_mps, y_ref_h)
                assert cos_mps >= 0.999999, f"MPS TFQ4 kind={tk} rows={x.shape[0]} cos={cos_mps:.8f}"
                if x.shape[0] == 16:
                    # Verify bit-exact row-0 parity between rows=1 and rows=16
                    y_r1 = lin_mps(xh[:1].to("mps")).cpu()
                    assert torch.equal(y_r1[0], lin_mps(xh.to("mps")).cpu()[0]), f"MPS TFQ4 kind={tk} r1 != r16 row 0"

    print(f"PASS Test 1: TFQ4 2.00 & 2.25 bpw (table_kind=0/2/3) CPU + MPS={'YES' if has_mps else 'SKIP'} cos >= 0.999999 & r1==r16")


# ----------------------------------------------------------------------------
# Test 2: Hybrid TFPW MoE (b_bits={5, 4, 3, 2, 225}) roundtrip + SwiGLU/softcap
# ----------------------------------------------------------------------------
def test_hybrid_tfpw_moe() -> None:
    rng = np.random.default_rng(2002)
    tbl3 = tfq_moe.load_and_validate_tables(PWV3_CANON)
    lut1024 = build_e8p1024_abs_codebook()
    H, I = 256, 128
    expert_bits = [5, 4, 3, 2, 225]
    has_mps = tfq_mps.mps_kernels_available()

    with tempfile.TemporaryDirectory(prefix="tfq_hybrid_moe_") as tmp:
        # Copy PWV3 + write PWV4 (table_kind=3, 32 KiB)
        with open(os.path.join(tmp, "pw_v3_tables.bin"), "wb") as f:
            f.write(open(PWV3_CANON, "rb").read())
        write_pwv4_file(os.path.join(tmp, "pw_v4_e8p_tables.bin"), lut1024, 3)
        with open(os.path.join(tmp, "config.json"), "w", encoding="utf-8") as f:
            json.dump({"model_type": "inkling2", "hidden_size": H, "moe_intermediate_size": I}, f)

        layer_path = os.path.join(tmp, "layer_00.tffpw")
        ref_experts = write_synthetic_hybrid_tfpw(
            layer_path, 0, expert_bits, H, I, tbl3, lut1024, 3, rng, with_xxh64=True
        )

        # Open via TfpwCheckpoint (auto-loads both pw_v3_tables.bin and pw_v4_e8p_tables.bin)
        with tfq_moe.TfpwCheckpoint(tmp, verify_checksums=True) as ckpt:
            summ = ckpt.summary()
            assert summ["is_hybrid"] is True
            assert summ["has_tfq4_tables"] is True
            assert summ["tfq4_table_kind"] == 3
            assert summ["all_checksums_verified"] is True
            assert summ["bit_histogram"] == {2: 1, 3: 1, 4: 1, 5: 1, 225: 1}
            exp_bpw = (5 + 4 + 3 + 2 + 2.25) / 5.0
            assert abs(summ["avg_bpw"] - exp_bpw) < 1e-9

            lyr = ckpt[0]
            assert lyr.is_hybrid is True
            torch.manual_seed(99)
            x = torch.randn(6, H, dtype=torch.float32) * 0.5
            picks = torch.tensor(
                [[0, 3], [1, 4], [2, 3], [4, 0], [3, 4], [2, 1]],
                dtype=torch.long,
            )
            weights = torch.tensor(
                [[0.6, 0.4], [0.7, 0.3], [0.5, 0.5], [0.25, 0.75], [0.8, 0.2], [0.35, 0.65]],
                dtype=torch.float32,
            )

            # Check per-expert decode parity for all 5 precisions
            for e in range(len(expert_bits)):
                wg, wu, wd = lyr.decode_expert(e)
                assert torch.allclose(wg, torch.from_numpy(ref_experts[e][0]), atol=1e-6)
                assert torch.allclose(wu, torch.from_numpy(ref_experts[e][1]), atol=1e-6)
                assert torch.allclose(wd, torch.from_numpy(ref_experts[e][2]), atol=1e-6)

            # Reference routed MoE for standard SwiGLU, GLM clamp, and Inkling-2 softcap
            for mode_name, kw in (
                ("swiglu", {}),
                ("glm_clamp", {"swiglu_limit": 7.0}),
                ("inkling2_softcap", {"gate_softcap": 4.0, "up_softcap": 25.0}),
            ):
                y_ref = torch.zeros_like(x)
                for t in range(x.shape[0]):
                    for s in range(picks.shape[1]):
                        e = int(picks[t, s].item())
                        w = float(weights[t, s].item())
                        rg = torch.from_numpy(ref_experts[e][0])
                        ru = torch.from_numpy(ref_experts[e][1])
                        rd = torch.from_numpy(ref_experts[e][2])
                        g = F.linear(x[t : t + 1], rg)
                        u = F.linear(x[t : t + 1], ru)
                        if "gate_softcap" in kw:
                            act = (4.0 * torch.tanh(g / 4.0)) * torch.sigmoid(g) * (25.0 * torch.tanh(u / 25.0))
                        elif "swiglu_limit" in kw:
                            act = F.silu(g.clamp(max=7.0)) * u.clamp(min=-7.0, max=7.0)
                        else:
                            act = F.silu(g) * u
                        y_ref[t : t + 1] += w * F.linear(act, rd)

                y_cpu = lyr.forward_routed(x, picks, weights, **kw)
                cos_c = _cos_sim(y_cpu, y_ref)
                assert cos_c >= 0.999999, f"Hybrid MoE CPU {mode_name} cos={cos_c}"

                if has_mps:
                    y_mps = lyr.forward_routed(x.to("mps"), picks.to("mps"), weights.to("mps"), **kw).cpu()
                    cos_m = _cos_sim(y_mps, y_ref)
                    assert cos_m >= 0.9999, f"Hybrid MoE MPS {mode_name} cos={cos_m}"

    print(f"PASS Test 2: Hybrid TFPW MoE {{5,4,3,2,225}}b roundtrip + SwiGLU/clamp/softcap CPU + MPS={'YES' if has_mps else 'SKIP'}")


# ----------------------------------------------------------------------------
# Test 3: vLLM plugin (TFQ4 2.00/2.25 bpw + Col/Row TP + TFPW/Hybrid MoE)
# ----------------------------------------------------------------------------
def test_vllm_tfq4_and_moe() -> None:
    rng = np.random.default_rng(3003)
    tbl3 = tfq_moe.load_and_validate_tables(PWV3_CANON)
    lut1024 = build_e8p1024_abs_codebook()
    has_mps = tfq_mps.mps_kernels_available()

    # 3a. TFQ4 checkpoint (mixed 2.00 & 2.25 bpw, merged gate_up_proj, Col-TP=2, Row-TP=2)
    with tempfile.TemporaryDirectory(prefix="tfq_vllm_tfq4_") as tmp:
        write_pwv4_file(os.path.join(tmp, "pw_v4_e8p_tables.bin"), lut1024, 3)
        specs = [
            ("model.layers.0.mlp.gate_proj", 192, 256, 225),
            ("model.layers.0.mlp.up_proj", 160, 256, 2),
            ("model.layers.0.mlp.col_tp_proj", 256, 256, 225),
            ("model.layers.0.mlp.down_proj", 256, 512, 225),
        ]
        refs = write_synthetic_tfp4_shard(
            os.path.join(tmp, "tfq4-00001-of-00001.tffp4"), specs, lut1024, 3, rng, with_xxh64=True
        )

        cfg = tfq3_quant.Tfq3Config.from_checkpoint_dir(tmp)
        assert cfg.table_kind == 3
        assert cfg.format_type == "tfq4"
        assert cfg.get_tensor_bits("model.layers.0.mlp.gate_proj") == 225
        assert cfg.get_tensor_bits("model.layers.0.mlp.up_proj") == 2

        # Merged 2-way mixed 2.25b + 2.00b gate_up_proj
        gu = torch.nn.Linear(256, 192 + 160, bias=False)
        gu.prefix = "model.layers.0.mlp.gate_up_proj"
        gu.quant_method = cfg.get_quant_method(gu, gu.prefix)
        assert isinstance(gu.quant_method, tfq3_quant.Tfq4LinearMethod)
        gu.quant_method.create_weights(gu, 256, [192, 160], 256, 192 + 160, torch.bfloat16)
        assert gu.tfq4_partition_bits == [225, 2]
        assert gu.tfq4_partition_table_kinds == [3, 2]

        class MiniTfq4Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gu = gu

        m = MiniTfq4Model()
        assert tfq3_quant.load_tfq3_checkpoint_weights(m, tmp) == 1
        torch.manual_seed(11)
        x256 = torch.randn(5, 256, dtype=torch.float32)
        y_gu = gu.quant_method.apply(gu, x256)
        y_gu_ref = torch.cat([
            F.linear(x256, torch.from_numpy(refs["model.layers.0.mlp.gate_proj"])),
            F.linear(x256, torch.from_numpy(refs["model.layers.0.mlp.up_proj"])),
        ], dim=-1)
        assert _cos_sim(y_gu, y_gu_ref) >= 0.999999
        if has_mps:
            y_gu_mps = gu.quant_method.apply(gu.to("mps"), x256.to(torch.float16).to("mps")).cpu().float()
            assert _cos_sim(y_gu_mps, y_gu_ref) >= 0.9999

        # Col-TP=2 on 2.25 bpw
        reader4 = tfq3_quant.Tfq4CheckpointReader(tmp)
        col_parts = []
        for rank in range(2):
            cl = torch.nn.Linear(256, 128, bias=False)
            cl.prefix = "model.layers.0.mlp.col_tp_proj"
            cm = cfg.get_quant_method(cl, cl.prefix)
            assert isinstance(cm, tfq3_quant.Tfq4LinearMethod)
            cm.create_weights(cl, 256, [128], 256, 256, torch.bfloat16, tp_rank=rank, tp_size=2)
            cm.load_layer_from_reader(cl, reader4)
            col_parts.append(cm.apply(cl, x256))
        y_col = torch.cat(col_parts, dim=-1)
        y_col_ref = F.linear(x256, torch.from_numpy(refs["model.layers.0.mlp.col_tp_proj"]))
        assert _cos_sim(y_col, y_col_ref) >= 0.999999

        # Row-TP=2 on 2.25 bpw down_proj [256, 512]
        x512 = torch.randn(4, 512, dtype=torch.float32)
        y_row_sum = torch.zeros(4, 256, dtype=torch.float32)
        for rank in range(2):
            rl = torch.nn.Linear(256, 256, bias=False)
            rl.prefix = "model.layers.0.mlp.down_proj"
            rm = cfg.get_quant_method(rl, rl.prefix)
            assert isinstance(rm, tfq3_quant.Tfq4LinearMethod)
            rm.create_weights(rl, 256, [256], 512, 256, torch.bfloat16, tp_rank=rank, tp_size=2)
            rm.load_layer_from_reader(rl, reader4)
            y_row_sum = y_row_sum + rm.apply(rl, x512[:, rank * 256 : (rank + 1) * 256])
        y_row_ref = F.linear(x512, torch.from_numpy(refs["model.layers.0.mlp.down_proj"]))
        assert _cos_sim(y_row_sum, y_row_ref) >= 0.999999

    # 3b. vLLM Tfq3MoeMethod + load_tfq3_moe_weights (hybrid {4, 3, 2, 225})
    with tempfile.TemporaryDirectory(prefix="tfq_vllm_moe_") as tmp:
        with open(os.path.join(tmp, "pw_v3_tables.bin"), "wb") as f:
            f.write(open(PWV3_CANON, "rb").read())
        write_pwv4_file(os.path.join(tmp, "pw_v4_e8p_tables.bin"), lut1024, 3)
        H, I = 256, 128
        ref_experts = write_synthetic_hybrid_tfpw(
            os.path.join(tmp, "layer_00.tffpw"), 0, [4, 3, 2, 225], H, I, tbl3, lut1024, 3, rng
        )
        cfg_moe = tfq3_quant.Tfq3Config.from_checkpoint_dir(tmp)
        assert cfg_moe.moe_layers == [0]

        class StubFusedMoE(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.prefix = "model.layers.0.mlp.experts"
                self.quant_method = cfg_moe.get_quant_method(self, self.prefix)
                assert isinstance(self.quant_method, tfq3_quant.Tfq3MoeMethod)
                self.quant_method.create_weights(
                    self, num_experts=4, hidden_size=H, intermediate_size_per_partition=I,
                    params_dtype=torch.bfloat16, gate_softcap=4.0, up_softcap=25.0,
                )

        class MiniMoeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.moe = StubFusedMoE()

        mm = MiniMoeModel()
        assert tfq3_quant.load_tfq3_moe_weights(mm, tmp) == 1
        torch.manual_seed(23)
        x_m = torch.randn(5, H, dtype=torch.float32) * 0.4
        r_logits = torch.randn(5, 4, dtype=torch.float32)
        y_moe = mm.moe.quant_method.apply(mm.moe, x_m, router_logits=r_logits, top_k=2, renormalize=True)

        probs = torch.softmax(r_logits, dim=-1)
        tw, tid = torch.topk(probs, k=2, dim=-1)
        tw = tw / tw.sum(dim=-1, keepdim=True)
        y_ref = torch.zeros_like(x_m)
        for t in range(5):
            for s in range(2):
                e = int(tid[t, s].item())
                rg, ru, rd = (torch.from_numpy(a) for a in ref_experts[e])
                g = F.linear(x_m[t : t + 1], rg)
                u = F.linear(x_m[t : t + 1], ru)
                act = (4.0 * torch.tanh(g / 4.0)) * torch.sigmoid(g) * (25.0 * torch.tanh(u / 25.0))
                y_ref[t : t + 1] += float(tw[t, s].item()) * F.linear(act, rd)
        assert _cos_sim(y_moe, y_ref) >= 0.999999
        mm._tfq3_moe_checkpoint.close()

    print("PASS Test 3: vLLM TFQ4 (2.00/2.25 bpw, merged, Col-TP=2, Row-TP=2) + Tfq3MoeMethod hybrid MoE")


# ----------------------------------------------------------------------------
# Test 4: tfq3-eval CLI (--inspect-only, synthetic HF eval + KL + Model Card)
# ----------------------------------------------------------------------------
def test_tfq3_eval_end_to_end() -> None:
    from transformers import AutoModelForCausalLM, LlamaConfig

    rng = np.random.default_rng(4004)
    tbl3 = tfq_moe.load_and_validate_tables(PWV3_CANON)
    lut1024 = build_e8p1024_abs_codebook()

    with tempfile.TemporaryDirectory(prefix="tfq_eval_e2e_") as tmp:
        # 4a. --inspect-only on Hybrid MoE + XXH64 tamper rejection
        moe_dir = os.path.join(tmp, "demo-hybrid-moe")
        os.makedirs(moe_dir)
        with open(os.path.join(moe_dir, "pw_v3_tables.bin"), "wb") as f:
            f.write(open(PWV3_CANON, "rb").read())
        write_pwv4_file(os.path.join(moe_dir, "pw_v4_e8p_tables.bin"), lut1024, 3)
        write_synthetic_hybrid_tfpw(
            os.path.join(moe_dir, "layer_00.tffpw"), 0, [4, 3, 2, 225], 256, 128, tbl3, lut1024, 3, rng
        )
        card_p = os.path.join(tmp, "moe_card.md")
        json_p = os.path.join(tmp, "moe_report.json")
        rc = tfq3_eval.main(["--checkpoint", moe_dir, "--inspect-only", "--model-card", card_p, "--json", json_p])
        assert rc == 0
        rep = json.load(open(json_p, "r", encoding="utf-8"))
        assert rep["inspection"]["format"] == "TFQ3+TFQ4-Hybrid-MoE"
        assert rep["inspection"]["xxh64"]["all_verified"] is True
        assert rep["inspection"]["tables"]["pw_v4"]["table_kind"] == 3
        assert rep["inspection"]["bit_histogram_units"] == {"2.00b": 1, "2.25b": 1, "3b": 1, "4b": 1}
        card_txt = open(card_p, "r", encoding="utf-8").read()
        assert "## Quantization & Evaluation (TFQ3/TFQ4)" in card_txt
        assert "[INSPECTED]" in card_txt
        assert "/Users/" not in card_txt and "/root/" not in card_txt

        # Corrupt 1 byte in layer_00.tffpw -> tfq3-eval MUST exit with code 2
        bad_p = os.path.join(moe_dir, "layer_00.tffpw")
        raw = bytearray(open(bad_p, "rb").read())
        raw[-128] ^= 0xFF
        # Flip a byte inside weights section (offset 32768 + 10)
        off_w = struct.unpack_from("<Q", raw, 72)[0]
        raw[off_w + 10] ^= 0x01
        open(bad_p, "wb").write(raw)
        assert tfq3_eval.main(["--checkpoint", moe_dir, "--inspect-only"]) == 2

        # 4b. Synthetic tiny Llama TFQ4 2.25 bpw HF checkpoint + paired ref-model eval + throughput + model card
        ref_dir = os.path.join(tmp, "tiny-llama-ref")
        q_dir = os.path.join(tmp, "tiny-llama-tfq4-225")
        os.makedirs(ref_dir)
        os.makedirs(q_dir)

        cfg = LlamaConfig(
            vocab_size=256,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            tie_word_embeddings=False,
        )
        cfg.save_pretrained(ref_dir)
        cfg.save_pretrained(q_dir)
        torch.manual_seed(0)
        ref_model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32)
        sd = ref_model.state_dict()
        _write_safetensors_f32(os.path.join(ref_dir, "model.safetensors"), sd)

        # Quantized TFQ4 2.25 bpw MLP shard + remaining residual tensors
        write_pwv4_file(os.path.join(q_dir, "pw_v4_e8p_tables.bin"), lut1024, 3)
        mlp_specs = [
            ("model.layers.0.mlp.gate_proj", 256, 128, 225),
            ("model.layers.0.mlp.up_proj", 256, 128, 225),
            ("model.layers.0.mlp.down_proj", 128, 256, 225),
        ]
        write_synthetic_tfp4_shard(
            os.path.join(q_dir, "tfq4-00001-of-00001.tffp4"), mlp_specs, lut1024, 3, rng, with_xxh64=True
        )
        packed = {n for n, *_ in mlp_specs}
        _write_safetensors_f32(
            os.path.join(q_dir, "model.safetensors"),
            {k: v for k, v in sd.items() if k.rsplit(".", 1)[0] not in packed},
        )

        # Load quantized TFQ4-2.25 model into HF and evaluate directly against ref_model
        q_model = tfq_torch.load_tfq_into_hf(q_dir, device="cpu", dtype=torch.float32, cache_decoded=True)
        assert sum(isinstance(m, tfq_torch.Tfq4Linear) and m.table_kind == 3 for m in q_model.modules()) == 3
        chunks = [torch.randint(0, 256, (1, 32), dtype=torch.long) for _ in range(4)]

        # Self-pair sanity check (KL == 0, Top-1 == 100%, ΔPPL == 0)
        ev_self = tfq3_eval.evaluate_causal_lm(q_model, chunks, ref_model=q_model, device=torch.device("cpu"), dataset_label="synthetic")
        assert ev_self["reference"]["kl_fp64_mean"] < 1e-10
        assert ev_self["reference"]["top1_agreement_pct"] == 100.0
        assert abs(ev_self["reference"]["delta_ppl"]) < 1e-6

        # Paired eval vs BF16/F32 ref model + throughput + Model Card
        ev_pair = tfq3_eval.evaluate_causal_lm(q_model, chunks, ref_model=ref_model, device=torch.device("cpu"), dataset_label="synthetic")
        tp = tfq3_eval.benchmark_throughput(q_model, vocab_size=256, device=torch.device("cpu"), prefill_len=32, decode_steps=8)
        assert tp["prefill_tok_per_sec"] > 0 and tp["decode_tok_per_sec"] > 0

        insp_q = tfq3_eval.inspect_checkpoint(q_dir)
        assert insp_q["format"] == "TFQ4-2.25bpw"
        assert abs(insp_q["bpw"]["quantized_effective"] - 2.251) < 0.002
        full_rep = {
            "inspection": insp_q,
            "evaluation": ev_pair,
            "throughput": tp,
            "provenance": {"schema": "tfq-eval-v1", "tfq3_version": "3.1.0", "torch_version": torch.__version__},
        }
        md = tfq3_eval.render_model_card(full_rep)
        assert "[MEASURED]" in md
        assert "Full-Vocab fp64 KL" in md
        assert "Top-1 Token Agreement" in md
        assert "Prefill" in md and "Decode" in md
        assert "/Users/" not in md and "/root/" not in md

    print("PASS Test 4: tfq3-eval (--inspect-only, XXH64 tamper reject, TFQ4-2.25 HF load, fp64 KL, throughput, Model Card)")


# ----------------------------------------------------------------------------
# Test 5: tfq3 package surface & entry point check
# ----------------------------------------------------------------------------
def test_package_exports() -> None:
    import tfq3

    assert tfq3.__version__ == "3.1.0"
    for attr in ("tfq3_eval", "tfq_moe", "tfq_mps", "tfq_torch", "tfq3_quant", "Tfq4LinearMethod", "Tfq3MoeMethod", "load_tfq3_moe_weights"):
        assert hasattr(tfq3, attr), f"tfq3 missing exported symbol {attr}"
    print("PASS Test 5: tfq3 v3.1.0 package exports verified")


def main() -> None:
    test_tfq4_200_and_225_cpu_and_mps()
    test_hybrid_tfpw_moe()
    test_vllm_tfq4_and_moe()
    test_tfq3_eval_end_to_end()
    test_package_exports()
    print("ALL ECOSYSTEM UPGRADE TESTS PASSED")


if __name__ == "__main__":
    main()
