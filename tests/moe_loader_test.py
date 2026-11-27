#!/usr/bin/env python3
"""
CPU unit tests for the standalone TFPW v3 MoE checkpoint loader (`integrations/pytorch/tfq_moe.py`).

Tests (zero GPU, zero TensorFold binary dependency):
  1. TFPW v3 header & per-section XXH64 validation (bit-flip rejection in meta, scales, weights).
  2. Legacy `flag == 0` TFPW acceptance & decode parity.
  3. Header/geometry/alignment/bounds negative controls (bad magic/ver/bits/alignment/oob).
  4. Lazy mmap `TfpwLayer.decode_expert(e)` & `forward_expert(x, e)` / `forward_routed` exact parity
     vs `reference/tfq3_reference.py:decode_tfpn_fast_np` across 3-bit, 4-bit, and 5-bit experts
     (standard SwiGLU, GLM clamped SwiGLU, and Inkling-2 softcapped SwiGLU).
  5. Inkling-2 checkpoint discovery (`layer_00..01.tffpw`, `residual-*.safetensors`,
     `model.safetensors.index.json`, standalone `mtp.safetensors`).
  6. GLM checkpoint discovery BOTH with `nonmoe.tffpw` (including TFPN decode + XXH64 rejection)
     AND without `nonmoe.tffpw` (`--keep-nonmoe-bf16` BF16-residual mode, colocated & `source_dir`).
"""
from __future__ import annotations

import json
import os
import struct
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "integrations", "pytorch"))
sys.path.insert(0, os.path.join(REPO_ROOT, "reference"))

import tfq3_reference as tfq3_ref
from tfq_moe import TfpwCheckpoint, TfpwLayer, load_and_validate_tables

ALIGN_16K = 16384
TABLES_PATH = os.path.join(REPO_ROOT, "reference", "tables", "pw_v3_tables.bin")


def _align16k(n: int) -> int:
    return (n + ALIGN_16K - 1) & ~(ALIGN_16K - 1)


def _mat_words(R: int, C: int, bits: int) -> int:
    return (R // 16) * (C // 128) * (bits * 2) * 32


def _mat_scales(R: int, C: int) -> int:
    return (R // 128) * (C // 128)


def _write_safetensors_bf16(path: str, tensors: dict[str, torch.Tensor]) -> None:
    hdr: dict[str, dict] = {}
    blobs: list[bytes] = []
    off = 0
    for name in sorted(tensors.keys()):
        t = tensors[name].detach().to(torch.bfloat16).contiguous()
        raw = t.view(torch.uint16).cpu().numpy().tobytes()
        hdr[name] = {"dtype": "BF16", "shape": list(t.shape), "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    hb = json.dumps(hdr, separators=(",", ":")).encode("utf-8")
    pad = (-len(hb)) & 7
    if pad:
        hb += b" " * pad
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for b in blobs:
            f.write(b)


def _write_synthetic_tfpw(
    path: str,
    layer_idx: int,
    expert_bits: list[int],
    H: int,
    I: int,
    rng: np.random.Generator,
    *,
    with_xxh64: bool = True,
    meta_tag: str = "test_tfpw",
) -> dict:
    """Create a valid 16 KiB-aligned TFPW v3 shard with known words/scales and optional XXH64."""
    n_exp = len(expert_bits)
    metas = bytearray()
    scales_list: list[np.ndarray] = []
    words_list: list[np.ndarray] = []
    sc_off = 0
    w_off = 0
    per_expert_raw: list[dict] = []

    n_sc_gu = _mat_scales(I, H)
    n_sc_dn = _mat_scales(H, I)

    for e, bits in enumerate(expert_bits):
        gu_w_cnt = _mat_words(I, H, bits)
        dn_w_cnt = _mat_words(H, I, bits)

        sc_g = (0.02 + 0.03 * rng.random(n_sc_gu)).astype(np.float16)
        sc_u = (0.02 + 0.03 * rng.random(n_sc_gu)).astype(np.float16)
        sc_d = (0.02 + 0.03 * rng.random(n_sc_dn)).astype(np.float16)

        wd_g = rng.integers(0, 2**32, size=gu_w_cnt, dtype=np.uint64).astype(np.uint32)
        wd_u = rng.integers(0, 2**32, size=gu_w_cnt, dtype=np.uint64).astype(np.uint32)
        wd_d = rng.integers(0, 2**32, size=dn_w_cnt, dtype=np.uint64).astype(np.uint32)

        gw_off = w_off
        uw_off = gw_off + gu_w_cnt
        dw_off = uw_off + gu_w_cnt
        w_off = dw_off + dn_w_cnt

        gs_off = sc_off
        us_off = gs_off + n_sc_gu
        ds_off = us_off + n_sc_gu
        sc_off = ds_off + n_sc_dn

        metas.extend(
            struct.pack(
                "<IIQQQIIIfffII",
                bits, e, gw_off, uw_off, dw_off,
                gs_off, us_off, ds_off,
                20.0 + bits, 20.5 + bits, 21.0 + bits,
                0, 0,
            )
        )
        scales_list.extend([sc_g, sc_u, sc_d])
        words_list.extend([wd_g, wd_u, wd_d])
        per_expert_raw.append({
            "bits": bits,
            "gate": (wd_g, sc_g.reshape(I // 128, H // 128)),
            "up": (wd_u, sc_u.reshape(I // 128, H // 128)),
            "down": (wd_d, sc_d.reshape(H // 128, I // 128)),
        })

    sc_bytes = np.concatenate(scales_list).tobytes()
    wd_bytes = np.concatenate(words_list).tobytes()
    meta_bytes = bytes(metas)

    off_meta = 144
    off_scales = _align16k(off_meta + len(meta_bytes))
    off_weights = _align16k(off_scales + len(sc_bytes))
    total_sz = _align16k(off_weights + len(wd_bytes))

    if with_xxh64:
        flags = tfq3_ref.FLAG_XXH64_PRESENT
        c_meta = tfq3_ref.xxh64(meta_bytes, 0)
        c_scales = tfq3_ref.xxh64(sc_bytes, 0)
        c_weights = tfq3_ref.xxh64(wd_bytes, 0)
    else:
        flags = 0
        c_meta = c_scales = c_weights = 0

    tag_b = meta_tag.encode("utf-8")[:31].ljust(32, b"\x00")
    hdr = struct.pack(
        "<8I6Q4Q32s",
        tfq3_ref.MAGIC_TFPW, tfq3_ref.FORMAT_VER, layer_idx, n_exp, 3 * n_exp, H, I, flags,
        3 * n_exp * H * I, w_off, sc_off, off_meta, off_scales, off_weights,
        c_meta, c_scales, c_weights, 0, tag_b,
    )
    assert len(hdr) == 144

    with open(path, "wb") as f:
        f.write(hdr)
        f.write(meta_bytes)
        f.write(b"\x00" * (off_scales - (off_meta + len(meta_bytes))))
        f.write(sc_bytes)
        f.write(b"\x00" * (off_weights - (off_scales + len(sc_bytes))))
        f.write(wd_bytes)
        f.write(b"\x00" * (total_sz - (off_weights + len(wd_bytes))))

    return {
        "off_meta": off_meta,
        "off_scales": off_scales,
        "off_weights": off_weights,
        "experts": per_expert_raw,
    }


def _write_synthetic_nonmoe(
    path: str,
    specs: list[tuple[str, int, int, int]],
    rng: np.random.Generator,
    *,
    with_xxh64: bool = True,
) -> dict:
    metas = bytearray()
    scales_list: list[np.ndarray] = []
    words_list: list[np.ndarray] = []
    sc_off = 0
    w_off = 0
    for name, R, C, bits in specs:
        n_sc = _mat_scales(R, C)
        n_wd = _mat_words(R, C, bits)
        sc = (0.02 + 0.02 * rng.random(n_sc)).astype(np.float16)
        wd = rng.integers(0, 2**32, size=n_wd, dtype=np.uint64).astype(np.uint32)
        nb = name.encode("utf-8")[:79].ljust(80, b"\x00")
        metas.extend(nb + struct.pack("<IIIIQQfIII", R, C, bits, sc_off, w_off, n_wd, 24.0, R, C, 0))
        scales_list.append(sc)
        words_list.append(wd)
        sc_off += n_sc
        w_off += n_wd

    meta_bytes = bytes(metas)
    sc_bytes = np.concatenate(scales_list).tobytes()
    wd_bytes = np.concatenate(words_list).tobytes()
    off_meta = 64
    off_scales = _align16k(off_meta + len(meta_bytes))
    off_weights = _align16k(off_scales + len(sc_bytes))
    tot_sz = _align16k(off_weights + len(wd_bytes))

    flags = tfq3_ref.FLAG_XXH64_PRESENT if with_xxh64 else 0
    xxh = tfq3_ref.xxh64_sections((meta_bytes, sc_bytes, wd_bytes)) if with_xxh64 else 0
    hdr = struct.pack(
        "<IIIIQQQQQQ",
        tfq3_ref.MAGIC_TFPN, tfq3_ref.FORMAT_VER, len(specs), flags,
        off_meta, off_scales, off_weights, sc_off, w_off, xxh,
    )
    with open(path, "wb") as f:
        f.write(hdr)
        f.write(meta_bytes)
        f.write(b"\x00" * (off_scales - (off_meta + len(meta_bytes))))
        f.write(sc_bytes)
        f.write(b"\x00" * (off_weights - (off_scales + len(sc_bytes))))
        f.write(wd_bytes)
        f.write(b"\x00" * (tot_sz - (off_weights + len(wd_bytes))))
    return {"off_meta": off_meta, "off_scales": off_scales, "off_weights": off_weights}


def test_tfpw_header_and_xxh64_validation() -> None:
    rng = np.random.default_rng(101)
    tbl = load_and_validate_tables(TABLES_PATH)
    with tempfile.TemporaryDirectory(prefix="tfq_moe_xxh_") as tmp:
        p = os.path.join(tmp, "layer_00.tffpw")
        info = _write_synthetic_tfpw(p, 0, [5, 4, 3, 4], H=256, I=128, rng=rng, with_xxh64=True)

        # Valid checksummed shard loads cleanly
        with TfpwLayer(p, tbl, verify_checksum=True) as lyr:
            assert lyr.checksum_verified is True
            assert lyr.header.has_xxh64 is True
            assert lyr.num_experts == 4
            assert lyr.bit_histogram() == {3: 1, 4: 2, 5: 1}
            assert abs(lyr.avg_bpw() - 4.0) < 1e-9

        # Bit-flip in each of meta, scales, weights MUST be rejected with section-specific error
        raw = bytearray(open(p, "rb").read())
        for sec, off in (
            ("meta", info["off_meta"] + 56),
            ("scales", info["off_scales"] + 3),
            ("weights", info["off_weights"] + 17),
        ):
            bad_p = os.path.join(tmp, f"bad_{sec}.tffpw")
            mut = bytearray(raw)
            mut[off] ^= 0x01
            with open(bad_p, "wb") as f:
                f.write(mut)
            try:
                TfpwLayer(bad_p, tbl, verify_checksum=True)
                raise AssertionError(f"Expected XXH64 checksum failure for corrupted {sec} section")
            except ValueError as exc:
                msg = str(exc).lower()
                assert "xxh64" in msg and sec in msg, f"Unexpected error message for {sec}: {exc}"

        # Legacy flag=0 shard MUST be accepted
        leg_p = os.path.join(tmp, "legacy_layer_00.tffpw")
        leg = bytearray(raw)
        struct.pack_into("<I", leg, 28, 0)
        struct.pack_into("<4Q", leg, 80, 0, 0, 0, 0)
        with open(leg_p, "wb") as f:
            f.write(leg)
        with TfpwLayer(p, tbl) as lyr_chk, TfpwLayer(leg_p, tbl, verify_checksum=True) as lyr_leg:
            assert lyr_leg.checksum_verified is False
            assert lyr_leg.header.has_xxh64 is False
            wg1, wu1, wd1 = lyr_chk.decode_expert(0)
            wg2, wu2, wd2 = lyr_leg.decode_expert(0)
            assert torch.equal(wg1, wg2) and torch.equal(wu1, wu2) and torch.equal(wd1, wd2)

        # Negative header/geometry/alignment controls (with flag=0 so structural checks fire directly)
        bad_cases: list[tuple[str, callable]] = [
            ("bad_magic", lambda b: struct.pack_into("<I", b, 0, 0xDEADBEEF)),
            ("bad_ver", lambda b: struct.pack_into("<I", b, 4, 2)),
            ("bad_hidden", lambda b: struct.pack_into("<I", b, 20, 200)),
            ("bad_off_scales_align", lambda b: struct.pack_into("<Q", b, 64, info["off_scales"] + 64)),
            ("bad_expert_bits", lambda b: struct.pack_into("<I", b, info["off_meta"], 2)),
            ("bad_4b_align", lambda b: struct.pack_into("<Q", b, info["off_meta"] + 64 + 8, 1)),
            ("oob_word_off", lambda b: struct.pack_into("<Q", b, info["off_meta"] + 8, 1 << 40)),
        ]
        for label, mutate in bad_cases:
            bp = os.path.join(tmp, f"struct_{label}.tffpw")
            mut = bytearray(leg)
            mutate(mut)
            with open(bp, "wb") as f:
                f.write(mut)
            try:
                TfpwLayer(bp, tbl, verify_checksum=True)
                raise AssertionError(f"Expected ValueError for structural corruption '{label}'")
            except ValueError:
                pass


def test_decode_expert_and_forward_expert_parity() -> None:
    rng = np.random.default_rng(202)
    tbl = load_and_validate_tables(TABLES_PATH)
    H, I = 256, 128
    expert_bits = [5, 4, 3, 4]
    with tempfile.TemporaryDirectory(prefix="tfq_moe_fwd_") as tmp:
        p = os.path.join(tmp, "layer_02.tffpw")
        info = _write_synthetic_tfpw(p, 2, expert_bits, H=H, I=I, rng=rng, with_xxh64=True)

        torch.manual_seed(7)
        x = torch.randn(5, H, dtype=torch.float32) * 0.5
        x_bf16 = x.to(torch.bfloat16)

        with TfpwLayer(p, tbl, cache_experts=True) as lyr:
            ref_weights: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
            for e, bits in enumerate(expert_bits):
                raw_e = info["experts"][e]
                ref_g = torch.from_numpy(
                    tfq3_ref.decode_tfpn_fast_np(raw_e["gate"][0], raw_e["gate"][1], I, H, bits, tbl)
                )
                ref_u = torch.from_numpy(
                    tfq3_ref.decode_tfpn_fast_np(raw_e["up"][0], raw_e["up"][1], I, H, bits, tbl)
                )
                ref_d = torch.from_numpy(
                    tfq3_ref.decode_tfpn_fast_np(raw_e["down"][0], raw_e["down"][1], H, I, bits, tbl)
                )
                ref_weights.append((ref_g, ref_u, ref_d))

                wg, wu, wd = lyr.decode_expert(e)
                assert wg.shape == (I, H) and wu.shape == (I, H) and wd.shape == (H, I)
                assert torch.equal(wg, ref_g), f"E{e} ({bits}b) gate decode mismatch"
                assert torch.equal(wu, ref_u), f"E{e} ({bits}b) up decode mismatch"
                assert torch.equal(wd, ref_d), f"E{e} ({bits}b) down decode mismatch"

                # Cached second decode matches identically
                wg2, wu2, wd2 = lyr.decode_expert(e)
                assert torch.equal(wg, wg2) and torch.equal(wu, wu2) and torch.equal(wd, wd2)

                # 1) Standard SwiGLU parity
                y_std = lyr.forward_expert(x, e)
                ref_std = F.linear(F.silu(F.linear(x, ref_g)) * F.linear(x, ref_u), ref_d)
                assert torch.allclose(y_std, ref_std, atol=1e-6), f"E{e} standard SwiGLU mismatch"

                # 2) GLM clamped SwiGLU (swiglu_limit=7.0) parity
                y_glm = lyr.forward_expert(x, e, swiglu_limit=7.0)
                g_c = F.linear(x, ref_g).clamp(max=7.0)
                u_c = F.linear(x, ref_u).clamp(min=-7.0, max=7.0)
                ref_glm = F.linear(F.silu(g_c) * u_c, ref_d)
                assert torch.allclose(y_glm, ref_glm, atol=1e-6), f"E{e} GLM clamped SwiGLU mismatch"

                # 3) Inkling-2 softcapped SwiGLU (gate_softcap=4.0, up_softcap=25.0) parity + BF16 dtype preservation
                y_ink = lyr.forward_expert(x_bf16, e, gate_softcap=4.0, up_softcap=25.0)
                assert y_ink.dtype == torch.bfloat16
                g_i = F.linear(x_bf16.float(), ref_g)
                u_i = F.linear(x_bf16.float(), ref_u)
                act_i = (4.0 * torch.tanh(g_i / 4.0)) * torch.sigmoid(g_i) * (25.0 * torch.tanh(u_i / 25.0))
                ref_ink = F.linear(act_i, ref_d).to(torch.bfloat16)
                assert torch.equal(y_ink, ref_ink), f"E{e} Inkling-2 softcapped SwiGLU mismatch"

            # 4) Routed top-2 forward parity
            eids = torch.tensor([[0, 2], [1, 3], [2, 0], [3, 1], [0, 1]], dtype=torch.long)
            ewts = torch.tensor([[0.6, 0.4], [0.7, 0.3], [0.5, 0.5], [0.2, 0.8], [0.9, 0.1]], dtype=torch.float32)
            y_routed = lyr.forward_routed(x, eids, ewts, gate_softcap=4.0, up_softcap=25.0)
            y_ref = torch.zeros_like(x)
            for t in range(x.shape[0]):
                for k in range(eids.shape[1]):
                    eid = int(eids[t, k])
                    y_ref[t] += ewts[t, k] * lyr.forward_expert(x[t : t + 1], eid, gate_softcap=4.0, up_softcap=25.0)[0]
            assert torch.allclose(y_routed, y_ref, atol=1e-5)


def test_inkling2_checkpoint_discovery_residuals_and_mtp() -> None:
    rng = np.random.default_rng(303)
    H, I = 256, 128
    with tempfile.TemporaryDirectory(prefix="tfq_ink2_ckpt_") as tmp:
        # Copy canonical pw_v3_tables.bin
        with open(TABLES_PATH, "rb") as sf, open(os.path.join(tmp, "pw_v3_tables.bin"), "wb") as df:
            df.write(sf.read())
        with open(os.path.join(tmp, "config.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_type": "inkling2_loop",
                    "llm_config": {"num_hidden_layers": 2, "hidden_size": H, "intermediate_size": I, "n_routed_experts": 4},
                },
                f,
            )

        _write_synthetic_tfpw(os.path.join(tmp, "layer_00.tffpw"), 0, [4, 4, 3, 3], H=H, I=I, rng=rng)
        _write_synthetic_tfpw(os.path.join(tmp, "layer_01.tffpw"), 1, [4, 3, 3, 3], H=H, I=I, rng=rng)

        torch.manual_seed(11)
        embed_t = torch.randn(32, H, dtype=torch.bfloat16)
        wq_t = torch.randn(H, H, dtype=torch.bfloat16)
        mtp_fuse = torch.randn(4, 4, dtype=torch.bfloat16)
        mtp_mask = torch.randn(H, dtype=torch.bfloat16)

        _write_safetensors_bf16(
            os.path.join(tmp, "residual-00001-of-00002.safetensors"),
            {"model.llm.embed.weight": embed_t},
        )
        _write_safetensors_bf16(
            os.path.join(tmp, "residual-00002-of-00002.safetensors"),
            {"model.llm.layers.0.attn.wq_du.weight": wq_t},
        )
        with open(os.path.join(tmp, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metadata": {"total_size": 0},
                    "weight_map": {
                        "model.llm.embed.weight": "residual-00001-of-00002.safetensors",
                        "model.llm.layers.0.attn.wq_du.weight": "residual-00002-of-00002.safetensors",
                    },
                },
                f,
            )
        _write_safetensors_bf16(
            os.path.join(tmp, "mtp.safetensors"),
            {"fuse_weight": mtp_fuse, "mask_embed": mtp_mask},
        )

        with TfpwCheckpoint(tmp, verify_checksums=True) as ckpt:
            s = ckpt.summary()
            assert ckpt.family == "inkling2"
            assert s["family"] == "inkling2"
            assert s["model_type"] == "inkling2_loop"
            assert ckpt.layer_indices == [0, 1]
            assert ckpt.has_nonmoe_tfq3 is False
            assert ckpt.nonmoe_mode == "bf16_residual"
            assert s["residual_shard_count"] == 2
            assert s["residual_tensor_count"] == 2
            assert ckpt.has_mtp_safetensors is True
            assert ckpt.mtp_tensor_names() == ["fuse_weight", "mask_embed"]
            assert all(os.path.basename(p) != "mtp.safetensors" for p in ckpt.residual_shards)
            assert s["all_checksums_verified"] is True

            assert torch.equal(ckpt.load_residual_tensor("model.llm.embed.weight"), embed_t)
            assert torch.equal(ckpt.load_residual_tensor("model.llm.layers.0.attn.wq_du.weight"), wq_t)
            assert torch.equal(ckpt.load_mtp_tensor("fuse_weight"), mtp_fuse)
            assert torch.equal(ckpt.load_mtp_tensor("mask_embed"), mtp_mask)

            y = ckpt.forward_expert(0, torch.randn(2, H), 1, gate_softcap=4.0, up_softcap=25.0)
            assert y.shape == (2, H) and torch.isfinite(y).all()
            assert "inkling2" in ckpt.format_summary()


def test_glm_checkpoint_with_and_without_nonmoe() -> None:
    rng = np.random.default_rng(404)
    H, I = 256, 128
    with tempfile.TemporaryDirectory(prefix="tfq_glm_ckpt_") as tmp:
        src_dir = os.path.join(tmp, "glm_source_bf16")
        tfq_dir = os.path.join(tmp, "glm_tfq3")
        os.makedirs(src_dir)
        os.makedirs(tfq_dir)

        with open(TABLES_PATH, "rb") as sf, open(os.path.join(tfq_dir, "pw_v3_tables.bin"), "wb") as df:
            df.write(sf.read())
        with open(os.path.join(src_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump({"model_type": "glm5_next", "hidden_size": H, "moe_intermediate_size": I}, f)

        torch.manual_seed(22)
        q_bf16 = torch.randn(H, H, dtype=torch.bfloat16)
        _write_safetensors_bf16(
            os.path.join(src_dir, "model.safetensors"),
            {"model.language_model.layers.0.self_attn.q_proj.weight": q_bf16},
        )

        # GLM MoE layers start at L3 (L0-L2 are dense!)
        _write_synthetic_tfpw(os.path.join(tfq_dir, "layer_03.tffpw"), 3, [5, 4, 3, 3], H=H, I=I, rng=rng)
        _write_synthetic_tfpw(os.path.join(tfq_dir, "layer_04.tffpw"), 4, [4, 4, 3, 3], H=H, I=I, rng=rng)

        # Case A: GLM --keep-nonmoe-bf16 (NO nonmoe.tffpw, BF16 residuals from source_dir)
        with TfpwCheckpoint(tfq_dir, verify_checksums=True, source_dir=src_dir) as ckpt_bf16:
            assert ckpt_bf16.family == "glm"
            assert ckpt_bf16.has_nonmoe_tfq3 is False
            assert ckpt_bf16.nonmoe_mode == "bf16_residual"
            assert ckpt_bf16.layer_indices == [3, 4]
            assert ckpt_bf16.has_mtp_safetensors is False
            got_q = ckpt_bf16.load_residual_tensor("model.language_model.layers.0.self_attn.q_proj.weight")
            assert torch.equal(got_q, q_bf16)
            try:
                ckpt_bf16.decode_nonmoe_tensor("lm_head")
                raise AssertionError("Expected FileNotFoundError when nonmoe.tffpw is absent")
            except FileNotFoundError:
                pass

        # Case B: GLM WITH nonmoe.tffpw (TFPN v3 + XXH64 verified)
        nm_path = os.path.join(tfq_dir, "nonmoe.tffpw")
        nm_info = _write_synthetic_nonmoe(
            nm_path,
            [
                ("model.language_model.layers.0.self_attn.q_proj", 256, 256, 4),
                ("lm_head", 256, 256, 5),
            ],
            rng,
            with_xxh64=True,
        )
        with TfpwCheckpoint(tfq_dir, verify_checksums=True, source_dir=src_dir) as ckpt_nm:
            assert ckpt_nm.family == "glm"
            assert ckpt_nm.has_nonmoe_tfq3 is True
            assert ckpt_nm.nonmoe_mode == "tfq3_tfpn"
            assert ckpt_nm.nonmoe is not None and ckpt_nm.nonmoe.checksum_verified is True
            assert ckpt_nm.nonmoe_tensor_names() == ["model.language_model.layers.0.self_attn.q_proj", "lm_head"]
            head_dec = ckpt_nm.decode_nonmoe_tensor("lm_head").numpy()
            head_ref = tfq3_ref.dequantize_tfpn_tensor(nm_path, TABLES_PATH, "lm_head", max_blocks=4)
            assert np.max(np.abs(head_dec - head_ref)) < 1e-4

        # Flip 1 bit in nonmoe.tffpw weights -> checkpoint load MUST reject via XXH64
        raw_nm = bytearray(open(nm_path, "rb").read())
        raw_nm[nm_info["off_weights"] + 9] ^= 0x01
        with open(nm_path, "wb") as f:
            f.write(raw_nm)
        try:
            TfpwCheckpoint(tfq_dir, verify_checksums=True, source_dir=src_dir)
            raise AssertionError("Expected ValueError on corrupted nonmoe.tffpw")
        except ValueError as exc:
            assert "xxh64" in str(exc).lower()

        # Legacy flag=0 nonmoe.tffpw is still accepted
        _write_synthetic_nonmoe(nm_path, [("lm_head", 256, 256, 4)], rng, with_xxh64=False)
        with TfpwCheckpoint(tfq_dir, verify_checksums=True, source_dir=src_dir) as ckpt_leg_nm:
            assert ckpt_leg_nm.has_nonmoe_tfq3 is True
            assert ckpt_leg_nm.nonmoe is not None and ckpt_leg_nm.nonmoe.checksum_verified is False

        # Case C: Colocated GLM --keep-nonmoe-bf16 (BF16 residual-*.safetensors directly inside tfq_dir, no nonmoe.tffpw)
        os.remove(nm_path)
        _write_safetensors_bf16(
            os.path.join(tfq_dir, "residual-00001-of-00001.safetensors"),
            {"model.language_model.layers.0.self_attn.q_proj.weight": q_bf16},
        )
        with TfpwCheckpoint(tfq_dir, verify_checksums=True) as ckpt_coloc:
            assert ckpt_coloc.family == "glm"
            assert ckpt_coloc.has_nonmoe_tfq3 is False
            assert ckpt_coloc.nonmoe_mode == "bf16_residual"
            assert torch.equal(
                ckpt_coloc.load_residual_tensor("model.language_model.layers.0.self_attn.q_proj.weight"),
                q_bf16,
            )


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:
                failed += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    sys.exit(1 if failed else 0)
