#!/usr/bin/env python3
"""
Reproducible Head-to-Head Quantizer Quality Benchmark & Provenance Generator.

Compares TFQ3 (Data-Free & Optional Hessian-LDLQ) and TFQ4-E8P against in-house
reference baselines (plus optional external checkpoints such as GPTQModel or ExLlamaV3)
under strictly controlled, matched-scope conditions (identical BF16 source, identical
BF16 embeddings/norms/lm_head, disjoint calibration vs evaluation splits):

  In-house baselines (explicitly labeled as in-house approximations):
    1. BF16 Reference (gold baseline)
    2. RTN / Affine (in-house, g64 and g128)
    3. AWQ-style Salient-Scale (in-house, g128)
    4. GPTQ-LDLQ (in-house, no act-order, fixed min/max g128)
    5. Scalar Lloyd-Max + LDLQ (in-house ref, 2D-FWHT, per-col LDLQ)
    6. TFQ3 Data-Free (real compiled `tfq3-generic-convert` binary -> exact TFPN v3 decode)
    7. TFQ3 + Hessian-LDLQ (optional calibration mode -> same TFPN v3 binary format)
    8. TFQ4-E8P (2.001 bpw, real `tfq4-convert` binary & optional Hessian-LDLQ)

  Metrics reported for EVERY method (exact fp64 accumulation):
    - Exact effective bits/weight (bpw)
    - Mean Weight SNR (dB) and Held-Out Real-Activation Output SNR (dB)
    - Full-model Perplexity + **paired per-sequence delta-NLL vs BF16 with paired 95% bootstrap CI**
    - Full-vocabulary exact fp64 forward KL D_KL(P_BF16 || P_q) across ALL scored tokens
      (mean, median, p95, p99, max KL, and Top-1 token agreement %)
    - Wall-clock quantization time (seconds)

Emits a SHA-256 provenance receipt (`schema: tfq-provenance-v2`) pinning clean git commit,
dirty status, converter binary & output shard SHA-256s, script & token hashes, and raw
per-sequence NLL arrays.
"""
from __future__ import annotations

import argparse
import gc
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
import time
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "reference"))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))

import tfq3_reference as tfq3_ref
import tfq4_lattice as tfq4_ref
import tfq3_hessian as tfq3_h

decode_tfpn_fast_np = tfq3_ref.decode_tfpn_fast_np


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_token_chunks(tokenizer, split_texts: list[str], seq_len: int, n_seqs: int) -> list[torch.Tensor]:
    text = "\n\n".join(t for t in split_texts if t.strip())
    ids = tokenizer.encode(text, add_special_tokens=False)
    max_seqs = len(ids) // seq_len
    use_seqs = max_seqs if n_seqs <= 0 else min(n_seqs, max_seqs)
    if use_seqs <= 0:
        raise RuntimeError(f"Not enough tokens ({len(ids)}) for seq_len={seq_len}")
    need = seq_len * use_seqs
    arr = torch.tensor(ids[:need], dtype=torch.long).reshape(use_seqs, seq_len)
    return [arr[i : i + 1] for i in range(use_seqs)]


def load_eval_corpus(dataset_name: str, tokenizer, seq_len: int, eval_seqs: int) -> tuple[str, list[torch.Tensor]]:
    if dataset_name == "wikitext2":
        wt_test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"]
        return "Salesforce/wikitext (wikitext-2-raw-v1, test)", build_token_chunks(tokenizer, wt_test, seq_len, eval_seqs)
    if dataset_name == "c4":
        ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        texts = []
        tok_est = 0
        target_toks = (eval_seqs if eval_seqs > 0 else 128) * seq_len * 2
        for row in ds:
            t = row.get("text", "")
            if t.strip():
                texts.append(t)
                tok_est += len(t) // 3
                if tok_est >= target_toks:
                    break
        return "allenai/c4 (en, validation)", build_token_chunks(tokenizer, texts, seq_len, eval_seqs if eval_seqs > 0 else 128)
    raise ValueError(f"Unsupported --dataset {dataset_name!r}; expected 'wikitext2' or 'c4'")


# ----------------------------------------------------------------------------
# In-House Reference Quantizers (matched on exact same W [R, C] & H [C, C])
# ----------------------------------------------------------------------------
def quant_affine_rtn(W: torch.Tensor, bits: int, group: int = 64) -> torch.Tensor:
    """In-house asymmetric uniform affine RTN quantization (g64 / g128)."""
    R, C = W.shape
    pad_C = ((C + group - 1) // group) * group
    Wp = W.to(torch.float32)
    if pad_C != C:
        Wp = torch.nn.functional.pad(Wp, (0, pad_C - C))
    G = Wp.reshape(R, pad_C // group, group)
    qmax = float((1 << bits) - 1)
    lo = G.amin(dim=-1, keepdim=True)
    hi = G.amax(dim=-1, keepdim=True)
    sc = torch.clamp((hi - lo) / qmax, min=1e-12)
    q = torch.clamp(torch.round((G - lo) / sc), 0.0, qmax)
    return (q * sc + lo).reshape(R, pad_C)[:, :C]


def quant_awq_salient(W: torch.Tensor, H: torch.Tensor, bits: int, group: int = 128) -> torch.Tensor:
    """In-house AWQ-style salient-channel scaling + asymmetric affine quantization (g128)."""
    Wf = W.to(torch.float32)
    x_rms = torch.sqrt(torch.clamp(torch.diagonal(H.to(torch.float32)), min=1e-12))
    x_rms = x_rms / torch.clamp(x_rms.mean(), min=1e-12)
    best_err = float("inf")
    best_W = None
    Hf = H.to(torch.float32)
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        s = torch.clamp(x_rms.pow(alpha), min=1e-3, max=1e3)
        W_q = quant_affine_rtn(Wf * s[None, :], bits, group=group) / s[None, :]
        d = Wf - W_q
        err = float(torch.sum((d @ Hf) * d).item())
        if math.isfinite(err) and err < best_err:
            best_err = err
            best_W = W_q
    if best_W is None:
        best_W = quant_affine_rtn(Wf, bits, group=group)
    return best_W


def quant_gptq_ldlq(W: torch.Tensor, H: torch.Tensor, bits: int, group: int = 128, damp: float = 0.01, u_inv: torch.Tensor | None = None) -> torch.Tensor:
    """
    In-house GPTQ-LDLQ approximation: block-Cholesky Hessian error feedback with
    asymmetric per-group min/max scales fixed before feedback and NO act-order.
    """
    orig_dev = W.device
    Wf = W.detach().cpu().to(torch.float32).clone()
    R, C = Wf.shape
    if u_inv is None:
        H_inv = tfq3_h.compute_h_inv_cholesky(H.cpu(), damp=damp)
        U_inv = torch.linalg.cholesky(H_inv.to(torch.float64)).T.to(torch.float32)
    else:
        U_inv = u_inv.cpu()
    qmax = float((1 << bits) - 1)
    Q = torch.zeros_like(Wf)

    for c0 in range(0, C, group):
        c1 = min(c0 + group, C)
        blk = Wf[:, c0:c1].clone()
        lo = blk.amin(dim=1, keepdim=True)
        hi = blk.amax(dim=1, keepdim=True)
        sc = torch.clamp((hi - lo) / qmax, min=1e-12)
        U_blk = U_inv[c0:c1, c0:c1]
        Err = torch.zeros_like(blk)

        for i in range(c1 - c0):
            w = blk[:, i]
            d = torch.clamp(U_blk[i, i], min=1e-6)
            q = torch.clamp(torch.round((w - lo[:, 0]) / sc[:, 0]), 0.0, qmax) * sc[:, 0] + lo[:, 0]
            Q[:, c0 + i] = q
            err_i = (w - q) / d
            Err[:, i] = err_i
            if i + 1 < (c1 - c0):
                blk[:, i + 1 :] -= err_i.unsqueeze(1) * U_blk[i, i + 1 :].unsqueeze(0)

        if c1 < C:
            Wf[:, c1:] -= Err @ U_inv[c0:c1, c1:]

    return Q.to(orig_dev)


_GAUSS_LLOYD_CACHE: dict[int, np.ndarray] = {}


def _gauss_lloyd_1d(bits: int) -> np.ndarray:
    if bits in _GAUSS_LLOYD_CACHE:
        return _GAUSS_LLOYD_CACHE[bits]
    K = 1 << bits
    rng = np.random.default_rng(2026 + bits)
    x = rng.standard_normal(262144).astype(np.float64)
    qs = np.linspace(0.5 / K, 1.0 - 0.5 / K, K)
    cen = np.quantile(x, qs)
    for _ in range(40):
        mid = 0.5 * (cen[:-1] + cen[1:])
        idx = np.searchsorted(mid, x)
        for k in range(K):
            m = idx == k
            if np.any(m):
                cen[k] = x[m].mean()
    _GAUSS_LLOYD_CACHE[bits] = cen.astype(np.float32)
    return _GAUSS_LLOYD_CACHE[bits]


def quant_hadamard_ldlq_lloyd(
    W: torch.Tensor, H: torch.Tensor, bits: int, damp: float = 0.01,
    precomputed: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """
    In-house scalar Lloyd-Max + per-column LDLQ reference after 2D FWHT-128
    (matched bits + 0.001 bpw; scalar 1D Gaussian grid, NOT EXL3's trellis codebook).
    """
    orig_dev = W.device
    W_pad, orig_R, orig_C = tfq3_h.pad_weight_128(W.detach().cpu())
    R, C = W_pad.shape
    n_br, n_bc = R // 128, C // 128

    W_rot0 = tfq3_h.rotate_2d_torch(W_pad)
    W_rot = W_rot0.clone()
    if precomputed is not None:
        H_rot, U_inv = precomputed[0].cpu(), precomputed[1].cpu()
    else:
        H_pad = tfq3_h.pad_hessian_128(H.cpu(), C)
        H_rot = tfq3_h.rotate_hessian_torch(H_pad)
        H_inv = tfq3_h.compute_h_inv_cholesky(H_rot, damp=damp)
        U_inv = torch.linalg.cholesky(H_inv.to(torch.float64)).T.to(torch.float32)

    cen = torch.from_numpy(_gauss_lloyd_1d(bits))
    mids = 0.5 * (cen[:-1] + cen[1:])

    Q_unit = torch.zeros_like(W_rot)
    scales = torch.zeros((n_br, n_bc), dtype=torch.float32)

    for cb in range(n_bc):
        c0, c1 = cb * 128, (cb + 1) * 128
        blk = W_rot[:, c0:c1].clone()
        sc_b = torch.sqrt(torch.mean(blk.reshape(n_br, 128, 128) ** 2, dim=(1, 2)) + 1e-12).to(torch.float16).to(torch.float32)
        scales[:, cb] = sc_b
        sc_rows = sc_b.repeat_interleave(128)
        inv_sc = torch.where(sc_rows > 1e-12, 1.0 / sc_rows, torch.zeros_like(sc_rows))

        U_blk = U_inv[c0:c1, c0:c1]
        Err = torch.zeros_like(blk)
        for i in range(128):
            w = blk[:, i]
            d = torch.clamp(U_blk[i, i], min=1e-6)
            idx = torch.bucketize(w * inv_sc, mids)
            qu = cen[idx]
            q = qu * sc_rows
            Q_unit[:, c0 + i] = qu
            err_i = (w - q) / d
            Err[:, i] = err_i
            if i + 1 < 128:
                blk[:, i + 1 :] -= err_i.unsqueeze(1) * U_blk[i, i + 1 :].unsqueeze(0)

        if c1 < C:
            W_rot[:, c1:] -= Err @ U_inv[c0:c1, c1:]

    for cb in range(n_bc):
        c0, c1 = cb * 128, (cb + 1) * 128
        H_blk = H_rot[c0:c1, c0:c1]
        for rb in range(n_br):
            r0, r1 = rb * 128, (rb + 1) * 128
            W_b = W_rot0[r0:r1, c0:c1]
            Q_b = Q_unit[r0:r1, c0:c1]
            QH = Q_b @ H_blk
            num = float(torch.sum(W_b * QH).item())
            den = float(torch.sum(Q_b * QH).item())
            if den > 1e-12 and num > 0.0:
                scales[rb, cb] = float(torch.tensor(num / den, dtype=torch.float16).item())

    W_hat = tfq3_h.unrotate_2d_torch(
        Q_unit * scales.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    )[:orig_R, :orig_C]
    return W_hat.to(orig_dev)


def compute_layer_snrs(W: torch.Tensor, W_hat: torch.Tensor, H_eval: torch.Tensor) -> tuple[float, float]:
    """Exact Weight SNR (dB) and Held-Out Real-Activation Output SNR (dB) on device."""
    Wf = W.detach().to(torch.float32)
    Df = Wf - W_hat.detach().to(device=Wf.device, dtype=torch.float32)
    Hf = H_eval.detach().to(device=Wf.device, dtype=torch.float32)
    w_sig = float(torch.sum(Wf * Wf).item())
    w_err = float(torch.sum(Df * Df).item())
    w_snr = 10.0 * math.log10(w_sig / max(w_err, 1e-30))
    y_sig = float(torch.sum((Wf @ Hf) * Wf).item())
    y_err = float(torch.sum((Df @ Hf) * Df).item())
    out_snr = 10.0 * math.log10(y_sig / max(y_err, 1e-30))
    return w_snr, out_snr


@torch.no_grad()
def evaluate_model_ppl_and_kl(
    model: torch.nn.Module,
    eval_batches: list[torch.Tensor],
    kl_batches_count: int,
    ref_model: torch.nn.Module | None,
    gold_seq_nlls: list[float] | None,
    device: torch.device,
) -> dict:
    """
    Evaluate exact fp64 NLL/PPL over ALL eval_batches, paired per-sequence delta-NLL vs BF16
    with a 2,000-sample paired bootstrap 95% CI, and streaming exact fp64 full-vocab forward
    KL D_KL(P_BF16 || P_q) + Top-1 agreement over `kl_batches_count` (0 => ALL sequences).
    """
    model.eval()
    if ref_model is not None:
        ref_model.eval()
    use_kl_seqs = len(eval_batches) if kl_batches_count <= 0 else min(kl_batches_count, len(eval_batches))

    seq_nlls: list[float] = []
    tot_nll = 0.0
    tot_tok = 0
    kl_vals = []
    top1_match = 0
    top1_tot = 0

    for bi, ids in enumerate(eval_batches):
        ids_dev = ids.to(device)
        logits = model(ids_dev).logits[:, :-1, :].float()
        targets = ids_dev[:, 1:].to(torch.long)
        logp = torch.log_softmax(logits, dim=-1)
        tok_nll = -logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1).squeeze(0).detach().cpu().to(torch.float64)
        seq_nlls.append(float(tok_nll.mean().item()))
        tot_nll += float(tok_nll.sum().item())
        tot_tok += int(tok_nll.numel())

        if ref_model is not None and bi < use_kl_seqs:
            ref_logits = ref_model(ids_dev).logits[:, :-1, :].float()
            lp_ref = torch.log_softmax(ref_logits, dim=-1).squeeze(0)
            lp_q = logp.squeeze(0)
            p_ref = torch.exp(lp_ref)
            kl = torch.sum(p_ref * (lp_ref - lp_q), dim=-1).clamp(min=0.0).detach().cpu().to(torch.float64)
            kl_vals.append(kl.numpy())
            top1_match += int((lp_ref.argmax(dim=-1) == lp_q.argmax(dim=-1)).sum().item())
            top1_tot += int(lp_ref.shape[0])
            del ref_logits, lp_ref, lp_q, p_ref, kl
        del ids_dev, logits, targets, logp, tok_nll

    mean_nll = tot_nll / tot_tok
    ppl = math.exp(mean_nll)
    rng = np.random.default_rng(2026)
    arr = np.array(seq_nlls, dtype=np.float64)
    idx_boot = rng.integers(0, len(arr), size=(2000, len(arr)))
    boots_ppl = np.exp(arr[idx_boot].mean(axis=1))
    ci_lo, ci_hi = float(np.percentile(boots_ppl, 2.5)), float(np.percentile(boots_ppl, 97.5))

    res: dict = {
        "eval_tokens": tot_tok,
        "num_eval_seqs": len(seq_nlls),
        "mean_nll": mean_nll,
        "ppl": ppl,
        "ppl_ci95": [ci_lo, ci_hi],
        "seq_nlls": [round(x, 8) for x in seq_nlls],
    }

    if gold_seq_nlls is not None:
        gold_arr = np.array(gold_seq_nlls, dtype=np.float64)
        deltas = arr - gold_arr
        boot_dnll = deltas[idx_boot].mean(axis=1)
        dnll_mean = float(deltas.mean())
        dnll_lo, dnll_hi = float(np.percentile(boot_dnll, 2.5)), float(np.percentile(boot_dnll, 97.5))
        gold_mean_nll = float(gold_arr.mean())
        res["delta_nll_vs_bf16"] = dnll_mean
        res["delta_nll_ci95"] = [dnll_lo, dnll_hi]
        res["delta_ppl_vs_bf16"] = math.exp(gold_mean_nll + dnll_mean) - math.exp(gold_mean_nll)
        res["delta_ppl_ci95"] = [
            math.exp(gold_mean_nll + dnll_lo) - math.exp(gold_mean_nll),
            math.exp(gold_mean_nll + dnll_hi) - math.exp(gold_mean_nll),
        ]

    if kl_vals:
        all_kl = np.concatenate(kl_vals)
        res["kl_tokens"] = int(all_kl.size)
        res["kl_mean"] = float(np.mean(all_kl))
        res["kl_median"] = float(np.median(all_kl))
        res["kl_p95"] = float(np.percentile(all_kl, 95))
        res["kl_p99"] = float(np.percentile(all_kl, 99))
        res["kl_max"] = float(np.max(all_kl))
        res["same_top1_pct"] = 100.0 * top1_match / max(top1_tot, 1)
    return res


def load_real_tfq3_binary_weights(model_dir: str, bits: int, tmp_dir: str) -> tuple[dict[str, np.ndarray], float, dict]:
    """Run the compiled Metal/CUDA TFQ3 binary converter and return decoded weights + SHA-256 provenance."""
    suffix = "" if platform.system() == "Darwin" else "-cuda"
    conv_bin = os.path.join(REPO_ROOT, "build", f"tfq3-generic-convert{suffix}")
    insp_bin = os.path.join(REPO_ROOT, "build", "tfq3-inspect")
    out_dir = os.path.join(tmp_dir, f"real_tfq3_{bits}b")
    t0 = time.time()
    subprocess.run(
        [conv_bin, "--model", model_dir, "--out", out_dir, "--bits", str(bits), "--head-bits", "0"],
        check=True, stdout=subprocess.DEVNULL,
    )
    conv_sec = time.time() - t0
    subprocess.run([insp_bin, out_dir], check=True, stdout=subprocess.DEVNULL)

    tbl_path = os.path.join(out_dir, "pw_v3_tables.bin")
    tbl = tfq3_ref.load_tables(tbl_path)
    weights: dict[str, np.ndarray] = {}
    output_hashes: dict[str, str] = {
        "pw_v3_tables.bin": sha256_file(tbl_path),
        "tfq3_manifest.json": sha256_file(os.path.join(out_dir, "tfq3_manifest.json")),
    }

    verified_ref = False
    for shard_path in sorted(glob.glob(os.path.join(out_dir, "*.tffpw"))):
        output_hashes[os.path.basename(shard_path)] = sha256_file(shard_path)
        with open(shard_path, "rb") as f:
            hdr = struct.unpack("<IIIIQQQQQQ", f.read(64))
            nt, off_meta, off_scales, off_weights = hdr[2], hdr[4], hdr[5], hdr[6]
            f.seek(off_meta)
            recs = [f.read(128) for _ in range(nt)]
            for rec in recs:
                nm = rec[:80].split(b"\x00", 1)[0].decode("utf-8")
                R, C, b, sc_off, w_off, w_cnt, _, orig_r, orig_c, _ = struct.unpack("<IIIIQQfIII", rec[80:])
                orig_r = orig_r or R
                orig_c = orig_c or C
                f.seek(off_scales + sc_off * 2)
                sc = np.frombuffer(f.read((R // 128) * (C // 128) * 2), dtype=np.float16).reshape(R // 128, C // 128)
                f.seek(off_weights + w_off * 4)
                wd = np.frombuffer(f.read(w_cnt * 4), dtype=np.uint32)
                W_dec = np.ascontiguousarray(decode_tfpn_fast_np(wd, sc, R, C, b, tbl)[:orig_r, :orig_c])
                if not verified_ref:
                    W_ref = tfq3_ref.dequantize_tfpn_tensor(shard_path, tbl_path, nm, max_blocks=1)
                    err = float(np.max(np.abs(W_dec[:128, :128] - W_ref)))
                    assert err < 1e-4, f"Fast TFPN decode mismatch vs tfq3_reference: {err}"
                    verified_ref = True
                weights[nm] = W_dec
    prov = {
        "converter_binary": os.path.basename(conv_bin),
        "converter_binary_sha256": sha256_file(conv_bin),
        "output_sha256": output_hashes,
    }
    return weights, conv_sec, prov


def load_real_tfq4_binary_weights(model_dir: str, tmp_dir: str) -> tuple[dict[str, np.ndarray], np.ndarray, float, dict]:
    """Run the compiled Metal/CUDA TFQ4 converter and return decoded weights + SHA-256 provenance."""
    suffix = "" if platform.system() == "Darwin" else "-cuda"
    conv_bin = os.path.join(REPO_ROOT, "build", f"tfq4-convert{suffix}")
    insp_bin = os.path.join(REPO_ROOT, "build", "tfq3-inspect")
    out_dir = os.path.join(tmp_dir, "real_tfq4_2b")
    t0 = time.time()
    subprocess.run(
        [conv_bin, "--model", model_dir, "--out", out_dir, "--head-bits", "0"],
        check=True, stdout=subprocess.DEVNULL,
    )
    conv_sec = time.time() - t0
    subprocess.run([insp_bin, out_dir], check=True, stdout=subprocess.DEVNULL)

    tbl_path = os.path.join(out_dir, "pw_v4_e8p_tables.bin")
    lut = tfq4_ref.load_pwv4_tables(tbl_path)
    weights: dict[str, np.ndarray] = {}
    output_hashes: dict[str, str] = {
        "pw_v4_e8p_tables.bin": sha256_file(tbl_path),
        "tfq4_manifest.json": sha256_file(os.path.join(out_dir, "tfq4_manifest.json")),
    }
    for shard_path in sorted(glob.glob(os.path.join(out_dir, "*.tffp4"))):
        output_hashes[os.path.basename(shard_path)] = sha256_file(shard_path)
        with open(shard_path, "rb") as f:
            hdr = struct.unpack("<IIIIQQQQQQ", f.read(64))
            nt, off_meta = hdr[2], hdr[4]
            f.seek(off_meta)
            recs = [f.read(128) for _ in range(nt)]
        for rec in recs:
            nm = rec[:80].split(b"\x00", 1)[0].decode("utf-8")
            R, C, _, _, _, _, _, orig_r, orig_c, _ = struct.unpack("<IIIIQQfIII", rec[80:])
            orig_r = orig_r or R
            orig_c = orig_c or C
            weights[nm] = np.ascontiguousarray(
                tfq4_ref.dequantize_tfp4_tensor(shard_path, tbl_path, nm, verify_checksum=False)[:orig_r, :orig_c]
            )
    prov = {
        "converter_binary": os.path.basename(conv_bin),
        "converter_binary_sha256": sha256_file(conv_bin),
        "output_sha256": output_hashes,
    }
    return weights, lut, conv_sec, prov


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Local HF BF16 model directory (e.g. Qwen2.5-1.5B)")
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--dataset", default="wikitext2", choices=["wikitext2", "c4"], help="Evaluation corpus")
    ap.add_argument("--calib-seqs", type=int, default=64, help="Calibration sequences (seq_len=512 from WikiText-2 train)")
    ap.add_argument("--calib-seq-len", type=int, default=512)
    ap.add_argument("--seq-len", type=int, default=2048, help="Evaluation sequence length (default 2048)")
    ap.add_argument("--eval-seqs", type=int, default=0, help="Eval sequences (0 = full test split)")
    ap.add_argument("--kl-seqs", type=int, default=0, help="Full-vocab KL sequences (0 = ALL eval sequences)")
    ap.add_argument("--skip-hessian-tfq", action="store_true", help="Skip slower Python TFQ+Hessian-LDLQ rows")
    ap.add_argument("--out-json", default=os.path.join(REPO_ROOT, "benchmarks", "provenance_qwen25_1p5b.json"))
    args = ap.parse_args()

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    wt_train = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")["text"]
    calib_batches = build_token_chunks(tokenizer, wt_train, args.calib_seq_len, args.calib_seqs)
    eval_ds_label, eval_batches = load_eval_corpus(args.dataset, tokenizer, args.seq_len, args.eval_seqs)

    calib_sha = sha256_bytes(torch.cat(calib_batches, dim=0).numpy().tobytes())
    eval_sha = sha256_bytes(torch.cat(eval_batches, dim=0).numpy().tobytes())
    n_eval_toks = len(eval_batches) * (args.seq_len - 1)
    kl_n_seqs = len(eval_batches) if args.kl_seqs <= 0 else min(args.kl_seqs, len(eval_batches))

    print("=== TFQ Controlled Head-to-Head Quantizer Benchmark (v2 Paired Provenance) ===")
    print(f"  model:   {args.model_id}")
    print(f"  device:  {device} ({platform.system()} {platform.machine()})")
    print(f"  calib:   {len(calib_batches)} x {args.calib_seq_len} = {len(calib_batches) * args.calib_seq_len} WikiText-2 train tokens")
    print(f"  eval:    {len(eval_batches)} x {args.seq_len} = {n_eval_toks} scored tokens on {eval_ds_label} (KL on {kl_n_seqs} seqs)")
    print(f"  calib_sha256: {calib_sha[:16]}... | eval_sha256: {eval_sha[:16]}...")

    # Keep a read-only BF16 reference model for streaming exact fp64 KL across ALL tokens!
    ref_model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, local_files_only=True).to(device)
    ref_model.eval()
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, local_files_only=True).to(device)
    model.eval()

    target_linears = [
        nm for nm, mod in model.named_modules()
        if isinstance(mod, torch.nn.Linear) and nm.startswith("model.layers.") and mod.weight.shape[0] >= 128 and mod.weight.shape[1] >= 128
    ]
    print(f"  quantized linear layers: {len(target_linears)} (embed_tokens, norms, biases, lm_head kept BF16)")

    orig_weights = {nm: dict(model.named_modules())[nm].weight.detach().cpu().clone() for nm in target_linears}
    mod_map = {nm: mod for nm, mod in model.named_modules() if nm in orig_weights}

    # 1. Gold BF16 evaluation
    t0 = time.time()
    gold = evaluate_model_ppl_and_kl(ref_model, eval_batches, 0, None, None, device)
    gold_seq_nlls = gold["seq_nlls"]
    print(f"\n[GOLD BF16] PPL={gold['ppl']:.4f} (CI95 [{gold['ppl_ci95'][0]:.4f}, {gold['ppl_ci95'][1]:.4f}]) | NLL={gold['mean_nll']:.5f} ({time.time()-t0:.1f}s)")

    # 2. Collect strictly disjoint calibration Hessians (train) & held-out evaluation Hessians (test)
    t0 = time.time()
    H_calib = tfq3_h.collect_linear_hessians(ref_model, calib_batches, target_linears, device)
    H_eval = tfq3_h.collect_linear_hessians(ref_model, eval_batches[: min(8, len(eval_batches))], target_linears, device)
    print(f"Collected {len(H_calib)} train Hessians & {len(H_eval)} held-out test Hessians in {time.time()-t0:.1f}s")

    results = [{
        "name": "BF16 Reference (Gold)",
        "baseline_kind": "reference",
        "bits": 16,
        "eff_bpw": 16.000,
        "uses_calibration": False,
        "weight_snr_db": None,
        "output_snr_db": None,
        "conv_sec": 0.0,
        **gold,
    }]

    def run_variant(
        name: str, baseline_kind: str, bits: int, eff_bpw: float, uses_calib: bool,
        quant_fn, conv_sec_override: float | None = None, converter_prov: dict | None = None,
    ):
        w_snrs, o_snrs = [], []
        quant_time = 0.0
        for nm in target_linears:
            W_orig = orig_weights[nm].to(device=device, dtype=torch.float32)
            t_layer = time.time()
            W_hat = quant_fn(nm, W_orig)
            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()
            quant_time += time.time() - t_layer
            if not torch.isfinite(W_hat).all():
                raise RuntimeError(f"Non-finite W_hat produced by {name!r} on layer {nm}")
            ws, osnr = compute_layer_snrs(W_orig, W_hat, H_eval[nm])
            w_snrs.append(ws)
            o_snrs.append(osnr)
            mod_map[nm].weight.data.copy_(W_hat.to(device=device, dtype=torch.bfloat16))
            del W_hat, W_orig
        gc.collect()
        conv_s = conv_sec_override if conv_sec_override is not None else quant_time
        ev = evaluate_model_ppl_and_kl(model, eval_batches, args.kl_seqs, ref_model, gold_seq_nlls, device)
        row = {
            "name": name,
            "baseline_kind": baseline_kind,
            "bits": bits,
            "eff_bpw": eff_bpw,
            "uses_calibration": uses_calib,
            "weight_snr_db": float(np.mean(w_snrs)),
            "output_snr_db": float(np.mean(o_snrs)),
            "conv_sec": float(conv_s),
            **ev,
        }
        if converter_prov is not None:
            row["converter_provenance"] = converter_prov
        results.append(row)
        print(
            f"  {name:<48s} | {eff_bpw:5.3f} bpw | OutSNR={row['output_snr_db']:5.2f} dB | "
            f"PPL={row['ppl']:7.4f} (dNLL={row['delta_nll_vs_bf16']:+.5f} [{row['delta_nll_ci95'][0]:+.5f},{row['delta_nll_ci95'][1]:+.5f}]) | "
            f"KL={row['kl_mean']:.5f} (p99={row['kl_p99']:.4f}) | Top1={row['same_top1_pct']:5.2f}% | {conv_s:5.1f}s"
        )
        for nm in target_linears:
            mod_map[nm].weight.data.copy_(orig_weights[nm].to(device=device, dtype=torch.bfloat16))
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        gc.collect()

    # Precompute Cholesky factors ONCE per unique layer Hessian (deduplicating shared Q/K/V & gate/up inputs)
    t_ch = time.time()
    gptq_uinv: dict[str, torch.Tensor] = {}
    rot_h_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    rot_uinv_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    unique_ch: dict[tuple[tuple[int, ...], str], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for nm in target_linears:
        Hc = H_calib[nm]
        h_key = (tuple(Hc.shape), sha256_bytes(Hc.contiguous().numpy().tobytes()))
        if h_key not in unique_ch:
            pad_C = ((Hc.shape[0] + 127) // 128) * 128
            H_inv_raw = tfq3_h.compute_h_inv_cholesky(Hc, damp=0.01)
            u_raw = torch.linalg.cholesky(H_inv_raw.to(torch.float64)).T.to(torch.float32)

            H_pad = tfq3_h.pad_hessian_128(Hc.to(device), pad_C)
            H_rot = tfq3_h.rotate_hessian_torch(H_pad).cpu()
            del H_pad
            H_rot_inv = tfq3_h.compute_h_inv_cholesky(H_rot, damp=0.01)
            u_rot = torch.linalg.cholesky(H_rot_inv.to(torch.float64)).T.to(torch.float32)
            unique_ch[h_key] = (u_raw, H_rot, H_rot_inv, u_rot)
        u_raw, H_rot, H_rot_inv, u_rot = unique_ch[h_key]
        gptq_uinv[nm] = u_raw
        rot_h_cache[nm] = (H_rot, H_rot_inv)
        rot_uinv_cache[nm] = (H_rot, u_rot)
    del unique_ch
    gc.collect()
    print(f"Precomputed {len(target_linears)} Cholesky & rotated-Hessian factors in {time.time() - t_ch:.1f}s")

    with tempfile.TemporaryDirectory(prefix="tfq_bench_") as tmp:
        print("\n=== 4-BIT HEAD-TO-HEAD (All 196 Linear Layers, BF16 Embed/Head) ===")
        run_variant("RTN / Affine 4b (in-house, g64)", "in-house", 4, 4.500, False, lambda nm, W: quant_affine_rtn(W, 4, 64))
        run_variant("RTN / Affine 4b (in-house, g128)", "in-house", 4, 4.250, False, lambda nm, W: quant_affine_rtn(W, 4, 128))
        run_variant("AWQ-style Salient-Scale 4b (in-house, g128)", "in-house", 4, 4.250, True, lambda nm, W: quant_awq_salient(W, H_calib[nm].to(device), 4, 128))
        run_variant("GPTQ-LDLQ 4b (in-house, no act-order, g128)", "in-house", 4, 4.250, True, lambda nm, W: quant_gptq_ldlq(W, H_calib[nm], 4, 128, u_inv=gptq_uinv[nm]))
        tfq3_4b_w, tfq3_4b_s, tfq3_4b_prov = load_real_tfq3_binary_weights(args.model, 4, tmp)
        tables = tfq3_ref.load_tables(os.path.join(tmp, "real_tfq3_4b", "pw_v3_tables.bin"))
        run_variant("TFQ3-4b Data-Free (real .tffpw)", "tfq-binary", 4, 4.001, False, lambda nm, W: torch.from_numpy(tfq3_4b_w[nm]).to(device), conv_sec_override=tfq3_4b_s, converter_prov=tfq3_4b_prov)
        del tfq3_4b_w
        gc.collect()
        if not args.skip_hessian_tfq:
            run_variant(
                "TFQ3-4b + Hessian-LDLQ (same TFPN)", "tfq-hessian", 4, 4.001, True,
                lambda nm, W: tfq3_h.quantize_tfq3_matrix(W, 4, tables, H=H_calib[nm], beam=8, precomputed_h=rot_h_cache[nm])["W_hat"],
            )
        run_variant(
            "Scalar Lloyd-Max + LDLQ 4b (in-house ref)", "in-house", 4, 4.001, True,
            lambda nm, W: quant_hadamard_ldlq_lloyd(W, H_calib[nm], 4, precomputed=rot_uinv_cache[nm]),
        )

        print("\n=== 3-BIT HEAD-TO-HEAD (All 196 Linear Layers, BF16 Embed/Head) ===")
        run_variant("RTN / Affine 3b (in-house, g64)", "in-house", 3, 3.500, False, lambda nm, W: quant_affine_rtn(W, 3, 64))
        run_variant("AWQ-style Salient-Scale 3b (in-house, g128)", "in-house", 3, 3.250, True, lambda nm, W: quant_awq_salient(W, H_calib[nm].to(device), 3, 128))
        run_variant("GPTQ-LDLQ 3b (in-house, no act-order, g128)", "in-house", 3, 3.250, True, lambda nm, W: quant_gptq_ldlq(W, H_calib[nm], 3, 128, u_inv=gptq_uinv[nm]))
        tfq3_3b_w, tfq3_3b_s, tfq3_3b_prov = load_real_tfq3_binary_weights(args.model, 3, tmp)
        run_variant("TFQ3-3b Data-Free (real .tffpw)", "tfq-binary", 3, 3.001, False, lambda nm, W: torch.from_numpy(tfq3_3b_w[nm]).to(device), conv_sec_override=tfq3_3b_s, converter_prov=tfq3_3b_prov)
        del tfq3_3b_w
        gc.collect()
        if not args.skip_hessian_tfq:
            run_variant(
                "TFQ3-3b + Hessian-LDLQ (same TFPN)", "tfq-hessian", 3, 3.001, True,
                lambda nm, W: tfq3_h.quantize_tfq3_matrix(W, 3, tables, H=H_calib[nm], beam=8, precomputed_h=rot_h_cache[nm])["W_hat"],
            )
        run_variant(
            "Scalar Lloyd-Max + LDLQ 3b (in-house ref)", "in-house", 3, 3.001, True,
            lambda nm, W: quant_hadamard_ldlq_lloyd(W, H_calib[nm], 3, precomputed=rot_uinv_cache[nm]),
        )

        print("\n=== 2-BIT HEAD-TO-HEAD (All 196 Linear Layers, BF16 Embed/Head) ===")
        run_variant("RTN / Affine 2b (in-house, g64)", "in-house", 2, 2.500, False, lambda nm, W: quant_affine_rtn(W, 2, 64))
        run_variant("GPTQ-LDLQ 2b (in-house, no act-order, g128)", "in-house", 2, 2.250, True, lambda nm, W: quant_gptq_ldlq(W, H_calib[nm], 2, 128, u_inv=gptq_uinv[nm]))
        tfq4_2b_w, e8p_lut, tfq4_2b_s, tfq4_2b_prov = load_real_tfq4_binary_weights(args.model, tmp)
        run_variant("TFQ4-E8P 2b Data-Free (real .tffp4)", "tfq-binary", 2, 2.001, False, lambda nm, W: torch.from_numpy(tfq4_2b_w[nm]).to(device), conv_sec_override=tfq4_2b_s, converter_prov=tfq4_2b_prov)
        del tfq4_2b_w
        gc.collect()
        if not args.skip_hessian_tfq:
            run_variant(
                "TFQ4-E8P 2b + Hessian-LDLQ (same TFP4)", "tfq-hessian", 2, 2.001, True,
                lambda nm, W: tfq3_h.quantize_tfq4_matrix(W, e8p_lut, H=H_calib[nm], precomputed_h=rot_h_cache[nm])["W_hat"],
            )

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    snap_rev = os.path.basename(os.path.normpath(args.model))
    git_rev = subprocess.check_output(["git", "-C", REPO_ROOT, "rev-parse", "HEAD"], text=True).strip()
    git_dirty = bool(subprocess.check_output(["git", "-C", REPO_ROOT, "status", "--porcelain"], text=True).strip())
    prov = {
        "schema": "tfq-provenance-v2",
        "claim_status": "MEASURED",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo_commit": git_rev,
        "git_dirty": git_dirty,
        "benchmark_script": "tools/bench_quantizers.py",
        "benchmark_script_sha256": sha256_file(__file__),
        "hessian_module_sha256": sha256_file(os.path.join(REPO_ROOT, "tools", "tfq3_hessian.py")),
        "hardware": f"{platform.system()} {platform.machine()} (device={device})",
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "model": {
            "repo_id": args.model_id,
            "snapshot_revision": snap_rev,
            "quantized_linears": len(target_linears),
            "unquantized_preserved_bf16": ["embed_tokens", "lm_head", "rms_norms", "biases"],
        },
        "dataset": {
            "name": eval_ds_label,
            "calibration_split": "Salesforce/wikitext (wikitext-2-raw-v1, train)",
            "calibration_tokens": len(calib_batches) * args.calib_seq_len,
            "calibration_tokens_sha256": calib_sha,
            "eval_seq_len": args.seq_len,
            "eval_num_seqs": len(eval_batches),
            "eval_tokens": n_eval_toks,
            "eval_tokens_sha256": eval_sha,
            "kl_tokens": kl_n_seqs * (args.seq_len - 1),
        },
        "notes": [
            "Non-TFQ baselines in this script are in-house approximations: GPTQ-LDLQ has no act-order and fixes min/max group scales before error feedback; Scalar Lloyd-Max + LDLQ uses a 1D optimal Gaussian grid after 2D-FWHT (not EXL3's trellis codebook).",
            "delta_nll_vs_bf16 and delta_nll_ci95 are paired per-sequence NLL differences (method - BF16) bootstrapped over the identical evaluation sequences.",
        ],
        "results": results,
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2)
        f.write("\n")
    print(f"\nWrote SHA-256 provenance receipt -> {args.out_json} (git_dirty={git_dirty})")


if __name__ == "__main__":
    main()
