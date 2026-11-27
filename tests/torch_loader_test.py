#!/usr/bin/env python3
"""
CPU checks for `integrations/pytorch/tfq_torch.py` on tiny synthetic checkpoints (no GPU, no download,
no `datasets`): every model tensor must come from the checkpoint, or `load_tfq_into_hf` must raise.
"""
from __future__ import annotations

import json
import os
import struct
import sys
import tempfile

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "integrations", "pytorch"))
sys.path.insert(0, os.path.join(REPO_ROOT, "reference"))

import tfq3_reference as tfq3_ref
import tfq4_lattice as tfq4_ref
from tfq_torch import Tfq3Linear, Tfq4Linear, load_tfq_into_hf

MLP = ("gate_proj", "up_proj", "down_proj")


def write_pwv3(path: str, rng: np.random.Generator) -> None:
    raw = bytearray(16384)
    struct.pack_into("<II6f", raw, 0, tfq3_ref.MAGIC_PWV3, tfq3_ref.FORMAT_VER, 1, 0, 1, 0, 1, 0)
    for off, n in ((64, 256 * 2), (1088, 1024 * 2), (5184, 1024 * 2)):
        raw[off:off + n * 2] = rng.standard_normal(n).astype(np.float16).tobytes()
    for off in (9280, 9792, 10304):
        raw[off:off + 64 * 4 * 2] = np.tile(np.array([1, 1, 0, 0], dtype=np.float16), 64).tobytes()
    open(path, "wb").write(bytes(raw))


def write_shard(path: str, magic: int, recs: list[tuple[str, int, int, int]], rng: np.random.Generator) -> None:
    """recs: (name, R, C, bits); random words and scales, TFPN v3 layout (TFPW only differs in magic here)."""
    metas, scales, words = [], [], []
    n_sc = n_wd = 0
    for name, R, C, bits in recs:
        sc = (0.01 + 0.01 * rng.random((R // 128) * (C // 128))).astype(np.float16)
        wd = rng.integers(0, 2**32, (R // 16) * (C // 128) * bits * 2 * 32, dtype=np.uint64).astype(np.uint32)
        metas.append(name.encode().ljust(80, b"\0") + struct.pack("<IIIIQQfIII", R, C, bits, n_sc, n_wd, wd.size, 20.0, R, C, 0))
        scales.append(sc); words.append(wd)
        n_sc += sc.size; n_wd += wd.size
    off_meta = 64
    off_sc = -(-(off_meta + 128 * len(metas)) // 16384) * 16384
    off_wd = -(-(off_sc + 2 * n_sc) // 16384) * 16384
    with open(path, "wb") as f:
        f.write(struct.pack("<IIIIQQQQQQ", magic, tfq3_ref.FORMAT_VER, len(metas), 0, off_meta, off_sc, off_wd, n_sc, n_wd, 0))
        f.write(b"".join(metas))
        f.seek(off_sc); f.write(np.concatenate(scales).tobytes())
        f.seek(off_wd); f.write(np.concatenate(words).tobytes())


def write_residuals(path: str, tensors: dict[str, torch.Tensor]) -> None:
    hdr, blobs, off = {}, [], 0
    for k, t in tensors.items():
        b = t.detach().to(torch.float32).contiguous().numpy().tobytes()
        hdr[k] = {"dtype": "F32", "shape": list(t.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b); off += len(b)
    h = json.dumps(hdr).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h))); f.write(h); f.write(b"".join(blobs))


def make_ckpt(d: str, case: str) -> dict[str, torch.Tensor]:
    """A 1-layer Llama, MLP in .tffpw. Returns the reference state dict."""
    from transformers import AutoModelForCausalLM, LlamaConfig

    cfg = LlamaConfig(vocab_size=256, hidden_size=128, intermediate_size=256, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=2, tie_word_embeddings=False)
    cfg.save_pretrained(d)
    torch.manual_seed(0)
    ref = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32).state_dict()
    rng = np.random.default_rng(0)
    write_pwv3(os.path.join(d, "pw_v3_tables.bin"), rng)
    mlp = [(f"model.layers.0.mlp.{p}", 128 if p == "down_proj" else 256, 256 if p == "down_proj" else 128, 4) for p in MLP]
    packed = {n for n, *_ in mlp}
    TFPN, TFPW = 0x4E504654, 0x57504654
    if case == "ok":
        write_shard(os.path.join(d, "nonmoe.tffpw"), TFPN, mlp, rng)
    elif case == "embed":  # --quantize-embed: the embedding table is a packed record
        write_shard(os.path.join(d, "nonmoe.tffpw"), TFPN, mlp + [("model.embed_tokens", 256, 128, 4)], rng)
        packed.add("model.embed_tokens")
    elif case == "tfpw":  # specialized MoE converters write expert weights as TFPW shards
        write_shard(os.path.join(d, "layer_00.tffpw"), TFPW, mlp, rng)
    elif case == "hybrid":  # TFQ3 + TFQ4 in one directory: down_proj only in the .tffp4
        write_shard(os.path.join(d, "nonmoe.tffpw"), TFPN, mlp[:2], rng)
        open(os.path.join(d, "tfq4-00001-of-00001.tffp4"), "wb").write(b"\0" * 64)
    write_residuals(os.path.join(d, "model.safetensors"),
                    {k: v for k, v in ref.items() if k.rsplit(".", 1)[0] not in packed})
    return ref


def expect(case: str, missing: list[str]) -> None:
    with tempfile.TemporaryDirectory(prefix=f"tfq_torch_{case}_") as d:
        make_ckpt(d, case)
        try:
            load_tfq_into_hf(d, device="cpu", dtype=torch.float32)
        except RuntimeError as e:
            for k in missing:
                assert f"  {k}\n" in str(e) + "\n", f"{case}: {k} not named in: {e}"
            print(f"PASS {case}: load_tfq_into_hf raised and named {', '.join(missing)}")
            return
        raise AssertionError(f"{case}: load_tfq_into_hf returned a model with {', '.join(missing)} at random init")


def test_complete_checkpoint_loads() -> None:
    with tempfile.TemporaryDirectory(prefix="tfq_torch_ok_") as d:
        ref = make_ckpt(d, "ok")
        m = load_tfq_into_hf(d, device="cpu", dtype=torch.float32)
        sd = m.state_dict()
        assert torch.equal(sd["model.embed_tokens.weight"], ref["model.embed_tokens.weight"])
        assert sum(isinstance(x, Tfq3Linear) for x in m.modules()) == 3
        with torch.no_grad():
            y = m(torch.arange(8)[None]).logits
        assert torch.isfinite(y).all()
        print("PASS ok: complete checkpoint loads, residuals exact, 3 Tfq3Linear, CPU forward finite")


def test_cpu_decode_matches_reference() -> None:
    """The CPU/MPS fallback decoder needs only numpy (no datasets/transformers) and matches the slow reference."""
    with tempfile.TemporaryDirectory(prefix="tfq_torch_dec_") as d:
        rng = np.random.default_rng(1)
        tables = os.path.join(d, "pw_v3_tables.bin")
        write_pwv3(tables, rng)
        for bits in (3, 4, 5):
            sp = os.path.join(d, f"t{bits}.tffpw")
            write_shard(sp, 0x4E504654, [("w", 256, 256, bits)], rng)
            ref_w = tfq3_ref.dequantize_tfpn_tensor(sp, tables, "w", max_blocks=2)
            with open(sp, "rb") as f:
                hdr = struct.unpack("<IIIIQQQQQQ", f.read(64))
                f.seek(hdr[5]); sc = np.frombuffer(f.read(4 * 2), dtype=np.float16).reshape(2, 2)
                f.seek(hdr[6]); wd = np.frombuffer(f.read(hdr[8] * 4), dtype=np.uint32)
            lin = Tfq3Linear(256, 256, bits, wd, sc, open(tables, "rb").read(), tfq3_ref.load_tables(tables))
            w = lin.dequantize(dtype=torch.float32).numpy()
            err = float(np.max(np.abs(w - ref_w)))
            assert err < 1e-4, f"{bits}b CPU decode vs reference: {err}"
            x = torch.randn(3, 256)
            assert torch.allclose(lin(x), x @ torch.from_numpy(ref_w).T, atol=1e-3)
    print("PASS cpu: Tfq3Linear 3/4/5b CPU decode == tfq3_reference (max err < 1e-4)")


def test_tfq4_table_kind_decode() -> None:
    """Tfq4Linear decodes both PWV4 table_kind=0 (trained) and table_kind=2 (true E8P) identically to tfq4_lattice."""
    rng = np.random.default_rng(7)
    W = (rng.standard_normal((256, 256)) * 0.03).astype(np.float32)
    lut_trained = tfq4_ref.load_pwv4_tables(os.path.join(REPO_ROOT, "reference", "tables", "pw_v4_e8p_tables.bin"))
    lut_e8p, _ = tfq4_ref.build_true_e8p_abs_codebook()
    lut_e8p = lut_e8p.astype(np.float32)

    for kind, lut in ((0, lut_trained), (2, lut_e8p)):
        W_rot = tfq4_ref.fwht2d_128(W)
        if kind == 2:
            q_rot, codes, sc = tfq4_ref.quant_true_e8p_e8_quarter(
                W_rot, abs_cb=lut, base_scale=tfq4_ref.TRUE_E8P_BASE_SCALE, ls_refine=False
            )
            ref_w = tfq4_ref.inv_fwht2d_128(q_rot)
        else:
            blocks = W_rot.reshape(2, 128, 2, 128).transpose(0, 2, 1, 3)
            sc = np.sqrt(np.mean(blocks * blocks, axis=(2, 3))).astype(np.float16)
            sc_tile = np.repeat(np.repeat(sc.astype(np.float32), 128, axis=0), 128, axis=1)
            W_norm = W_rot / sc_tile
            v = W_norm.reshape(-1, 8)
            sgn = np.where(v < 0.0, -1.0, 1.0).astype(np.float32)
            av = np.abs(v)
            d2 = np.sum(av * av, axis=1, keepdims=True) + np.sum(lut * lut, axis=1)[None, :] - 2.0 * (av @ lut.T)
            k_idx = np.argmin(d2, axis=1).astype(np.uint16)
            sm = np.zeros(v.shape[0], dtype=np.uint16)
            for d_i in range(8):
                sm |= (v[:, d_i] < 0.0).astype(np.uint16) << np.uint16(d_i)
            codes = (k_idx | (sm << np.uint16(8))).reshape(256, 32)
            q_rot = (lut[k_idx] * sgn).reshape(256, 256) * sc_tile
            ref_w = tfq4_ref.inv_fwht2d_128(q_rot)
        c2 = codes.reshape(256, 16, 2).astype(np.uint32)
        wd = c2[:, :, 0] | (c2[:, :, 1] << np.uint32(16))
        lin = Tfq4Linear(256, 256, wd, sc, lut, table_kind=kind)
        w = lin.dequantize(dtype=torch.float32).numpy()
        err = float(np.max(np.abs(w - ref_w)))
        assert err < 1e-4, f"TFQ4 table_kind={kind} CPU decode vs reference: {err}"
    try:
        Tfq4Linear(256, 256, wd, sc, lut_e8p, table_kind=3)
        raise AssertionError("Tfq4Linear accepted unsupported table_kind=3")
    except ValueError:
        pass
    print("PASS cpu: Tfq4Linear table_kind=0 and table_kind=2 decode == tfq4_lattice (and rejects kind=3)")


def test_mps_fused_kernels() -> None:
    """Zero-build Apple Silicon MPS fused Metal kernels for Tfq3Linear (3/4/5b) and Tfq4Linear (kind=0 & kind=2)."""
    from tfq_mps import mps_kernels_available
    if not mps_kernels_available():
        print("SKIP mps: torch.mps.compile_shader unavailable")
        return

    rng = np.random.default_rng(19)
    tbl_path = os.path.join(REPO_ROOT, "reference", "tables", "pw_v3_tables.bin")
    tbl_b = open(tbl_path, "rb").read()
    tbl_d = tfq3_ref.load_tables(tbl_path)
    N, K = 384, 512

    for bits in (3, 4, 5):
        wd = rng.integers(0, 2**32, (N // 16) * (K // 128) * bits * 2 * 32, dtype=np.uint64).astype(np.uint32)
        sc = (0.02 + 0.01 * rng.random((N // 128, K // 128))).astype(np.float16)
        bias = torch.randn(300, dtype=torch.float32)
        lin = Tfq3Linear(450, 300, bits, wd, sc, tbl_b, tbl_d, bias=bias)
        xs = [torch.randn(rows, 450, dtype=torch.bfloat16) for rows in (1, 16, 40)]
        ys_cpu = [lin(x) for x in xs]
        lin.to("mps")
        for x, y_cpu in zip(xs, ys_cpu):
            y_mps = lin(x.to("mps")).cpu()
            cos = torch.nn.functional.cosine_similarity(y_cpu.float().flatten(), y_mps.float().flatten(), dim=0).item()
            assert cos > 0.9999, f"Tfq3Linear MPS {bits}b rows={x.shape[0]} cos={cos}"

    lut0 = tfq4_ref.load_pwv4_tables(os.path.join(REPO_ROOT, "reference", "tables", "pw_v4_e8p_tables.bin"))
    lut2, _ = tfq4_ref.build_true_e8p_abs_codebook()
    wd4 = rng.integers(0, 2**32, (N, K // 16), dtype=np.uint64).astype(np.uint32)
    sc4 = (0.02 + 0.01 * rng.random((N // 128, K // 128))).astype(np.float16)
    for kind, lut in ((0, lut0), (2, lut2.astype(np.float32))):
        lin4 = Tfq4Linear(450, 300, wd4, sc4, lut, table_kind=kind)
        x4 = torch.randn(4, 450, dtype=torch.float16)
        y_cpu = lin4(x4)
        y_mps = lin4.to("mps")(x4.to("mps")).cpu()
        cos = torch.nn.functional.cosine_similarity(y_cpu.float().flatten(), y_mps.float().flatten(), dim=0).item()
        assert cos > 0.9999, f"Tfq4Linear MPS kind={kind} cos={cos}"
    print("PASS mps: zero-build fused Metal kernels for Tfq3Linear (3/4/5b sk8+prefill) and Tfq4Linear (kind=0 & kind=2)")


if __name__ == "__main__":
    print("datasets importable:", __import__("importlib").util.find_spec("datasets") is not None)
    test_cpu_decode_matches_reference()
    test_tfq4_table_kind_decode()
    test_mps_fused_kernels()
    test_complete_checkpoint_loads()
    expect("embed", ["model.embed_tokens.weight"])
    expect("tfpw", [f"model.layers.0.mlp.{p}.weight" for p in MLP])
    expect("hybrid", ["model.layers.0.mlp.down_proj.weight"])
    print("ALL PASS")
