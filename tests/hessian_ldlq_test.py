#!/usr/bin/env python3
"""
CPU tests for the LDLQ error feedback in tools/tfq3_hessian.py (no GPU, no model, no compiled tools).

Correct LDLQ quantizes each block against the target that minimizes the remaining output error
given every earlier block already fixed: t_rest = W_rest + E_done H_done,rest H_rest,rest^-1.
"""
from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
import warnings
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))

import tfq3_hessian as tfq3_h


def _random_hessian(C: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(4 * C, C, generator=g) @ (torch.eye(C) + 0.3 * torch.randn(C, C, generator=g))
    return (X.T @ X) / X.shape[0]


def _conditional_targets(W_rot0: torch.Tensor, W_rot_hat: torch.Tensor, H: torch.Tensor, s: int) -> torch.Tensor:
    """Exact targets for columns [s, C) given columns [0, s) are fixed to W_rot_hat (float64)."""
    if s == 0:
        return W_rot0.clone()
    E_d = W_rot0[:, :s] - W_rot_hat[:, :s]
    return W_rot0[:, s:] + E_d @ torch.linalg.solve(H[s:, s:], H[:s, s:].T).T


def _setup(R: int, C: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(R, C, generator=g) * 0.02
    H = _random_hessian(C, seed + 1)
    H_rot = tfq3_h.rotate_hessian_torch(tfq3_h.pad_hessian_128(H, C))
    H_inv = tfq3_h.compute_h_inv_cholesky(H_rot, damp=0.01)
    H_eff = torch.linalg.inv(H_inv.to(torch.float64))  # damped H the feedback works against
    W_rot0 = tfq3_h.rotate_2d_torch(W).to(torch.float64)
    return W, H, H_eff, W_rot0


def _synthetic_tables(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "lut": {
            3: rng.standard_normal((256, 2)).astype(np.float16),
            4: rng.standard_normal((1024, 2)).astype(np.float16),
            5: rng.standard_normal((1024, 2)).astype(np.float16),
        },
        "c64": {
            3: rng.standard_normal((64, 4)).astype(np.float16),
            4: rng.standard_normal((64, 4)).astype(np.float16),
            5: rng.standard_normal((64, 4)).astype(np.float16),
        },
    }


def _tfq3_with_block_targets(W: torch.Tensor, **kwargs) -> tuple[dict, dict[int, torch.Tensor]]:
    """Run quantize_tfq3_matrix and record the W_rot columns each 128-column block is encoded against."""
    seen: dict[int, torch.Tensor] = {}
    encode = tfq3_h._encode_block_viterbi

    def recording_encode(*args, **kw):
        caller = sys._getframe(1).f_locals
        cb = caller["cb"]
        seen[cb] = caller["W_rot"][:, cb * 128 : (cb + 1) * 128].detach().to(torch.float64).clone()
        return encode(*args, **kw)

    tfq3_h._encode_block_viterbi = recording_encode
    try:
        res = tfq3_h.quantize_tfq3_matrix(W, 4, _synthetic_tables(4), beam=4, topk=4, refine_scales=False, **kwargs)
    finally:
        tfq3_h._encode_block_viterbi = encode
    return res, seen


def test_tfq3_gain_three_correlated_coordinates() -> None:
    # One coordinate per block (0, 128, 256) with H^-1 diagonal 1, off-diagonal 0.5, all else independent.
    # After block 0 is fixed, a unit error in block 1 must move block 2 by -1/3 (the full H^-1 gives -1/2).
    C = 384
    H_inv = torch.eye(C, dtype=torch.float64)
    idx = torch.tensor([0, 128, 256])
    H_inv[idx[:, None], idx[None, :]] = 0.5
    H_inv[idx, idx] = 1.0
    H_rot = torch.linalg.inv(H_inv).to(torch.float32)
    W = torch.randn(256, C, generator=torch.Generator().manual_seed(5)) * 0.02
    res, seen = _tfq3_with_block_targets(W, precomputed_h=(H_rot, H_inv.to(torch.float32)))
    W_rot0 = tfq3_h.rotate_2d_torch(W).to(torch.float64)
    W_rot_hat = tfq3_h.rotate_2d_torch(res["W_hat"]).to(torch.float64)
    e0 = W_rot0[:, 0] - W_rot_hat[:, 0]
    e1 = seen[1][:, 0] - W_rot_hat[:, 128]
    shift = seen[2][:, 0] - W_rot0[:, 256] + 0.5 * e0  # block 0's -e0/2 is the same on main and here
    gain = -float(torch.dot(shift, e1) / torch.dot(e1, e1))
    assert abs(gain - 1.0 / 3.0) < 1e-3, f"block 2 moved by {-gain:.4f} per unit error in block 1, expected -1/3"


def test_ldlq_gain_helper() -> None:
    H_inv = torch.full((3, 3), 0.5, dtype=torch.float64) + 0.5 * torch.eye(3, dtype=torch.float64)
    U = tfq3_h.ldlq_upper_factor(H_inv).to(torch.float64)
    assert abs(float(tfq3_h.ldlq_gain(U, 0, 1)[0, 0]) - 0.5) < 1e-6
    assert abs(float(tfq3_h.ldlq_gain(U, 1, 2)[0, 0]) - 1.0 / 3.0) < 1e-6


def test_tfq4_feedback_is_sequential_conditional() -> None:
    R, C = 128, 256
    W, H, H_eff, W_rot0 = _setup(R, C, seed=1)
    lut = np.abs(np.random.default_rng(2).standard_normal((256, 8))).astype(np.float32)
    res = tfq3_h.quantize_tfq4_matrix(W, lut, H=H, refine_scales=False)
    W_rot_hat = tfq3_h.rotate_2d_torch(res["W_hat"]).to(torch.float64)
    sc = torch.from_numpy(res["scales"].astype(np.float32)).to(torch.float64)
    lut_t = torch.from_numpy(lut).to(torch.float64)
    worst = 0.0
    for c0 in range(0, C, 8):
        sc_rows = sc[:, c0 // 128].repeat_interleave(128)[:, None]
        v = _conditional_targets(W_rot0, W_rot_hat, H_eff, c0)[:, :8] / sc_rows
        q = W_rot_hat[:, c0 : c0 + 8] / sc_rows
        d_best = torch.min(torch.cdist(v.abs(), lut_t) ** 2, dim=1).values
        d_chosen = torch.sum((v - q) ** 2, dim=1)
        worst = max(worst, float(torch.max(d_chosen - d_best)))
    # Every 8-column code must be the nearest codeword to its exact conditional target.
    assert worst < 1e-3, f"chosen code is {worst:.4f} farther than the nearest codeword to the conditional target"


def test_tfq3_feedback_is_sequential_conditional() -> None:
    R, C = 128, 384
    W, H, H_eff, W_rot0 = _setup(R, C, seed=3)
    res, seen = _tfq3_with_block_targets(W, H=H)
    W_rot_hat = tfq3_h.rotate_2d_torch(res["W_hat"]).to(torch.float64)
    rms = float(torch.sqrt(torch.mean(W_rot0 * W_rot0)))
    for cb in range(C // 128):
        want = _conditional_targets(W_rot0, W_rot_hat, H_eff, cb * 128)[:, :128]
        err = float(torch.max(torch.abs(seen[cb] - want))) / rms
        assert err < 1e-3, f"block {cb}: encoder target is off the conditional target by {err:.3f} x rms(W)"


def test_cholesky_failure_warns() -> None:
    H = torch.tensor([[1.0, 10.0], [10.0, 1.0]])  # indefinite even after the fallback damping
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        H_inv = tfq3_h.compute_h_inv_cholesky(H)
    assert torch.equal(H_inv, torch.eye(2))
    assert any(issubclass(w.category, RuntimeWarning) and "Cholesky failed" in str(w.message) for w in caught)


def test_hessian_safetensors_roundtrip_single_file_f32_and_f64() -> None:
    H_q = _random_hessian(64, seed=11).to(torch.float64)
    H_gu = _random_hessian(64, seed=12).to(torch.float64)
    H_dn = _random_hessian(32, seed=13).to(torch.float64)
    src = {
        "model.layers.0.self_attn.q_proj": H_q,
        "model.layers.3.mlp.experts.gate_up": H_gu,
        "model.layers.3.mlp.experts.0.down_proj": H_dn,
    }
    meta_in = {
        "tokens": 8192,
        "damp": 0.01,
        "model_id": "synthetic/moe-unit-test",
        "token_counts": {
            "model.layers.0.self_attn.q_proj": 8192,
            "model.layers.3.mlp.experts.gate_up": 8192,
            "model.layers.3.mlp.experts.0.down_proj": 512,
        },
    }
    with tempfile.TemporaryDirectory(prefix="tfq_h_rt_") as tmp:
        for dt_arg, st_dt, torch_dt in (("f32", "F32", torch.float32), ("f64", "F64", torch.float64)):
            out_path = os.path.join(tmp, f"hessians_{dt_arg}.safetensors")
            nbytes = tfq3_h.save_hessians_safetensors(out_path, src, metadata=meta_in, dtype=dt_arg)
            assert nbytes == os.path.getsize(out_path)

            # Inspect raw on-disk Safetensors header to ensure spec compliance
            with open(out_path, "rb") as f:
                hlen = struct.unpack("<Q", f.read(8))[0]
                assert hlen % 8 == 0, "Expected 8-byte aligned Safetensors JSON header"
                raw_hdr = json.loads(f.read(hlen).decode("utf-8"))
            raw_meta = raw_hdr["__metadata__"]
            assert all(isinstance(k, str) and isinstance(v, str) for k, v in raw_meta.items())
            assert raw_meta["format"] == tfq3_h.HESSIAN_FORMAT_V1
            assert raw_meta["dtype"] == st_dt
            for k in src:
                assert raw_hdr[k]["dtype"] == st_dt

            loaded, meta_out = tfq3_h.load_hessians_safetensors(out_path)
            assert set(loaded.keys()) == set(src.keys())
            for k, orig in src.items():
                assert loaded[k].dtype == torch_dt
                assert torch.equal(loaded[k], orig.to(torch_dt))
            assert meta_out["format"] == tfq3_h.HESSIAN_FORMAT_V1
            assert meta_out["dtype"] == st_dt
            assert meta_out["tokens"] == meta_in["tokens"]
            assert meta_out["damp"] == meta_in["damp"]
            assert meta_out["model_id"] == meta_in["model_id"]
            assert meta_out["token_counts"] == meta_in["token_counts"]

        # Negative validation checks
        bad_path = os.path.join(tmp, "bad.safetensors")
        for bad_h, bad_dt in (
            ({"rect": torch.zeros(4, 8)}, "f32"),
            ({"nan": torch.tensor([[1.0, float("nan")], [float("nan"), 1.0]])}, "f32"),
            ({"ok": torch.eye(4)}, "int8"),
        ):
            try:
                tfq3_h.save_hessians_safetensors(bad_path, bad_h, dtype=bad_dt)
                raise AssertionError(f"Expected ValueError for {bad_h.keys()} / {bad_dt}")
            except ValueError:
                pass


def test_hessian_safetensors_sharded_directories() -> None:
    h0 = _random_hessian(32, seed=21)
    h1 = _random_hessian(32, seed=22)
    h2 = _random_hessian(64, seed=23)
    with tempfile.TemporaryDirectory(prefix="tfq_h_shards_") as tmp:
        # 1) Directory with hessians.safetensors.index.json
        d_hidx = os.path.join(tmp, "by_hessians_index")
        os.makedirs(d_hidx)
        tfq3_h.save_hessians_safetensors(
            os.path.join(d_hidx, "hessians-00001-of-00002.safetensors"),
            {"model.layers.0.self_attn.q_proj": h0},
            metadata={"tokens": 2048, "damp": 0.01, "model_id": "sharded-test", "token_counts": {"model.layers.0.self_attn.q_proj": 2048}},
            dtype="f32",
        )
        tfq3_h.save_hessians_safetensors(
            os.path.join(d_hidx, "hessians-00002-of-00002.safetensors"),
            {"model.layers.3.mlp.experts.gate_up": h1, "model.layers.3.mlp.experts.0.down_proj": h2},
            metadata={
                "tokens": 2048,
                "damp": 0.01,
                "model_id": "sharded-test",
                "token_counts": {
                    "model.layers.3.mlp.experts.gate_up": 2048,
                    "model.layers.3.mlp.experts.0.down_proj": 128,
                },
            },
            dtype="f32",
        )
        with open(os.path.join(d_hidx, "hessians.safetensors.index.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metadata": {"format": tfq3_h.HESSIAN_FORMAT_V1},
                    "weight_map": {
                        "model.layers.0.self_attn.q_proj": "hessians-00001-of-00002.safetensors",
                        "model.layers.3.mlp.experts.gate_up": "hessians-00002-of-00002.safetensors",
                        "model.layers.3.mlp.experts.0.down_proj": "hessians-00002-of-00002.safetensors",
                    },
                },
                f,
            )
        got, meta = tfq3_h.load_hessians_safetensors(d_hidx)
        assert torch.equal(got["model.layers.0.self_attn.q_proj"], h0.to(torch.float32))
        assert torch.equal(got["model.layers.3.mlp.experts.gate_up"], h1.to(torch.float32))
        assert torch.equal(got["model.layers.3.mlp.experts.0.down_proj"], h2.to(torch.float32))
        assert meta["token_counts"] == {
            "model.layers.0.self_attn.q_proj": 2048,
            "model.layers.3.mlp.experts.gate_up": 2048,
            "model.layers.3.mlp.experts.0.down_proj": 128,
        }

        # 2) Directory with model.safetensors.index.json
        os.rename(
            os.path.join(d_hidx, "hessians.safetensors.index.json"),
            os.path.join(d_hidx, "model.safetensors.index.json"),
        )
        got2, _ = tfq3_h.load_hessians_safetensors(d_hidx)
        assert set(got2.keys()) == set(got.keys())
        for k in got:
            assert torch.equal(got2[k], got[k])

        # 3) Directory with plain *.safetensors shards (no index file)
        os.remove(os.path.join(d_hidx, "model.safetensors.index.json"))
        got3, meta3 = tfq3_h.load_hessians_safetensors(d_hidx)
        assert set(got3.keys()) == set(got.keys())
        assert meta3["token_counts"]["model.layers.3.mlp.experts.0.down_proj"] == 128

        # 4) Duplicate keys in unindexed directory rejected
        tfq3_h.save_hessians_safetensors(
            os.path.join(d_hidx, "dup.safetensors"),
            {"model.layers.0.self_attn.q_proj": h0},
            dtype="f32",
        )
        try:
            tfq3_h.load_hessians_safetensors(d_hidx)
            raise AssertionError("Expected ValueError on duplicate Hessian key across shards")
        except ValueError:
            pass


def test_resolve_hessian_exact_and_moe_gate_up_fallback() -> None:
    h_exact = torch.eye(16) * 2.0
    h_std = torch.eye(16) * 3.0
    h_glm = torch.eye(16) * 4.0
    h_ink = torch.eye(16) * 5.0
    h_down = torch.eye(8) * 6.0
    hessians = {
        "model.layers.3.mlp.experts.0.gate_proj": h_exact,
        "model.layers.3.mlp.experts.gate_up": h_std,
        "model.layers.3.mlp.experts.0.down_proj": h_down,
        "model.language_model.layers.12.mlp.experts.gate_up": h_glm,
        "model.llm.layers.5.mlp.experts.gate_up": h_ink,
    }

    # Exact match beats shared gate_up fallback
    assert torch.equal(tfq3_h.resolve_hessian(hessians, "model.layers.3.mlp.experts.0.gate_proj"), h_exact)
    assert torch.equal(tfq3_h.resolve_hessian(hessians, "model.layers.3.mlp.experts.0.gate_proj.weight"), h_exact)

    # Standard MoE fallback (gate_proj & up_proj, with or without .weight)
    assert torch.equal(tfq3_h.resolve_hessian(hessians, "model.layers.3.mlp.experts.17.gate_proj"), h_std)
    assert torch.equal(tfq3_h.resolve_hessian(hessians, "model.layers.3.mlp.experts.17.up_proj"), h_std)
    assert torch.equal(tfq3_h.resolve_hessian(hessians, "model.layers.3.mlp.experts.17.up_proj.weight"), h_std)

    # GLM MoE fallback
    assert torch.equal(tfq3_h.resolve_hessian(hessians, "model.language_model.layers.12.mlp.experts.42.gate_proj"), h_glm)
    assert torch.equal(tfq3_h.resolve_hessian(hessians, "model.language_model.layers.12.mlp.experts.42.up_proj"), h_glm)

    # Inkling-2 MoE fallback
    assert torch.equal(tfq3_h.resolve_hessian(hessians, "model.llm.layers.5.mlp.experts.99.gate"), h_ink)
    assert torch.equal(tfq3_h.resolve_hessian(hessians, "model.llm.layers.5.mlp.experts.99.up"), h_ink)
    assert torch.equal(tfq3_h.resolve_hessian(hessians, "model.llm.layers.5.mlp.experts.99.up.weight"), h_ink)

    # down_proj / down exact match works, and NEVER falls back to gate_up
    assert torch.equal(tfq3_h.resolve_hessian(hessians, "model.layers.3.mlp.experts.0.down_proj"), h_down)
    assert tfq3_h.resolve_hessian(hessians, "model.layers.3.mlp.experts.17.down_proj") is None
    assert tfq3_h.resolve_hessian(hessians, "model.language_model.layers.12.mlp.experts.42.down_proj") is None
    assert tfq3_h.resolve_hessian(hessians, "model.llm.layers.5.mlp.experts.99.down") is None


def test_ldlq_end_to_end_parity_in_memory_vs_f64_safetensors() -> None:
    """Encoding with in-memory Hessian vs reloaded F64 Safetensors Hessian must produce 100% identical TFQ3 words & scales."""
    R, C = 128, 384
    g = torch.Generator().manual_seed(77)
    W = torch.randn(R, C, generator=g) * 0.02
    H_mem = _random_hessian(C, seed=78).to(torch.float64)
    tables = _synthetic_tables(79)

    with tempfile.TemporaryDirectory(prefix="tfq_h_ldlq_") as tmp:
        h_file = os.path.join(tmp, "hessians_f64.safetensors")
        tfq3_h.save_hessians_safetensors(
            h_file,
            {"model.layers.3.mlp.experts.gate_up": H_mem},
            metadata={"tokens": 1536, "damp": 0.01, "model_id": "parity-test"},
            dtype="f64",
        )
        # Also verify CLI --load-hessians -> --export-hessians --capture-only round-trip
        h_reexport = os.path.join(tmp, "reexported_f64.safetensors")
        rc = tfq3_h.main([
            "--load-hessians", h_file,
            "--export-hessians", h_reexport,
            "--hessian-dtype", "f64",
            "--capture-only",
        ])
        assert rc == 0

        loaded, meta = tfq3_h.load_hessians_safetensors(h_reexport)
        assert meta["dtype"] == "F64"
        H_reloaded = tfq3_h.resolve_hessian(loaded, "model.layers.3.mlp.experts.9.gate_proj")
        assert H_reloaded is not None
        assert H_reloaded.dtype == torch.float64
        assert torch.equal(H_reloaded, H_mem)

        for bits in (3, 4, 5):
            r_mem = tfq3_h.quantize_tfq3_matrix(W, bits, tables, H=H_mem, damp=0.01, beam=4, topk=4, refine_scales=True)
            r_disk = tfq3_h.quantize_tfq3_matrix(W, bits, tables, H=H_reloaded, damp=0.01, beam=4, topk=4, refine_scales=True)
            assert np.array_equal(r_mem["words"], r_disk["words"]), f"{bits}b packed words differ after F64 reload"
            assert np.array_equal(r_mem["scales"], r_disk["scales"]), f"{bits}b fp16 scales differ after F64 reload"
            assert torch.equal(r_mem["W_hat"], r_disk["W_hat"]), f"{bits}b reconstructed W_hat differs after F64 reload"


if __name__ == "__main__":
    torch.manual_seed(0)
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # report every test, then exit non-zero
                failed += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(1 if failed else 0)
