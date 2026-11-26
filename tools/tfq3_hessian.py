#!/usr/bin/env python3
"""
TFQ3 / TFQ4 Optional Calibration-Aware Hessian-Weighted Beam-Viterbi + LDLQ Encoder.

While TFQ3's default Metal/CUDA converters are 100% data-free (zero calibration data, minutes to
convert), this module adds an OPTIONAL calibration-aware mode that optimizes REAL ACTIVATION
OUTPUT ERROR directly inside TFQ3's exact binary format:

    min_{W_hat} || (W - W_hat) X^T ||_F^2 = tr((W_rot - W_rot_hat) H_rot (W_rot - W_rot_hat)^T)

where W_rot = Q_L W Q_R^T is TFQ3's exact orthonormal 2D FWHT-128 transform and
H_rot = Q_R H Q_R^T is the exact calibration input Hessian in TFQ3's rotated domain.

Algorithm:
  1. Per-128-column block Hessian-weighted Beam-Viterbi search over TFQ3's exact
     2D-AffineCoset/half4 state-transition trellis (weighting each pair's MSE by diag(H_rot)).
  2. Exact block-Cholesky LDLQ error feedback (Schur complement) propagating each 128-column
     block's quantization residual into all remaining columns before encoding them.
  3. Closed-form Hessian-optimal per-tile fp16 scale refinement:
         s* = <W_tile H_tile, Q_tile> / <Q_tile H_tile, Q_tile>
  4. 100% BYTE-COMPATIBLE with standard TFQ3 (`PWV3` + `TFPN` v3) and TFQ4 (`PWV4` + `TFP4` v4)
     binary formats — zero kernel changes, zero runtime overhead, same Metal & CUDA execution,
     and identical bit-exact speculative decoding.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import time
import warnings
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "reference"))
import tfq3_reference as tfq3_ref
import tfq4_lattice as tfq4_ref

ALIGN_16K = 16384
MAGIC_PWV3 = 0x33565750
MAGIC_TFPN = 0x4E504654
MAGIC_PWV4 = 0x34565750
MAGIC_TFP4 = 0x34504654
HASH_MULT = 0x9E3779B1
INIT_ST_BASE = 0x13579BDF


def align16k(n: int) -> int:
    return (n + ALIGN_16K - 1) & ~(ALIGN_16K - 1)


def fwht128_torch(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal 128-point Walsh-Hadamard transform along the last axis (exact match to Metal/CUDA)."""
    orig = x.shape
    y = x.reshape(-1, 128).to(torch.float32)
    h = 1
    while h < 128:
        y = y.reshape(-1, 128 // (2 * h), 2, h)
        a = y[:, :, 0, :]
        b = y[:, :, 1, :]
        y = torch.stack((a + b, a - b), dim=2).reshape(-1, 128)
        h <<= 1
    return (y * (1.0 / math.sqrt(128.0))).reshape(orig)


def sgn_torch(seed: int, n: int, device: torch.device) -> torch.Tensor:
    s = tfq3_ref.pw_sgn(seed, np.arange(n, dtype=np.uint32))
    return torch.from_numpy(s).to(device=device, dtype=torch.float32)


def rotate_2d_torch(W: torch.Tensor) -> torch.Tensor:
    """Exact TFQ3 2D FWHT-128 rotation: H_128 @ (D_L @ W @ D_R) @ H_128."""
    R, C = W.shape
    assert R % 128 == 0 and C % 128 == 0
    sr = sgn_torch(tfq3_ref.SEED_ROW, R, W.device)
    sc = sgn_torch(tfq3_ref.SEED_COL, C, W.device)
    tmp = fwht128_torch(W.to(torch.float32) * sc[None, :])
    return fwht128_torch((tmp * sr[:, None]).T).T


def unrotate_2d_torch(W_rot: torch.Tensor) -> torch.Tensor:
    """Exact inverse of rotate_2d_torch."""
    R, C = W_rot.shape
    sr = sgn_torch(tfq3_ref.SEED_ROW, R, W_rot.device)
    sc = sgn_torch(tfq3_ref.SEED_COL, C, W_rot.device)
    tmp = fwht128_torch(W_rot.T).T * sr[:, None]
    return fwht128_torch(tmp) * sc[None, :]


def rotate_hessian_torch(H: torch.Tensor) -> torch.Tensor:
    """Exact column-rotation of input Hessian: H_rot = Q_R H Q_R^T."""
    C = H.shape[0]
    sc = sgn_torch(tfq3_ref.SEED_COL, C, H.device)
    tmp = fwht128_torch(H.to(torch.float32) * sc[None, :])
    H_rot = fwht128_torch(tmp.T * sc[None, :]).T
    return 0.5 * (H_rot + H_rot.T)


def pad_weight_128(W: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    R, C = W.shape
    pad_R = ((R + 127) // 128) * 128
    pad_C = ((C + 127) // 128) * 128
    if pad_R == R and pad_C == C:
        return W.to(torch.float32), R, C
    out = torch.zeros((pad_R, pad_C), dtype=torch.float32, device=W.device)
    out[:R, :C] = W.to(torch.float32)
    return out, R, C


def pad_hessian_128(H: torch.Tensor, pad_C: int) -> torch.Tensor:
    C = H.shape[0]
    if C == pad_C:
        return H.to(torch.float32)
    out = torch.eye(pad_C, dtype=torch.float32, device=H.device) * 1e-4
    out[:C, :C] = H.to(torch.float32)
    return out


def compute_h_inv_cholesky(H_rot: torch.Tensor, damp: float = 0.01) -> torch.Tensor:
    """Compute damped H_rot^{-1} in float64 on CPU for numerical stability."""
    C = H_rot.shape[0]
    H64 = H_rot.detach().cpu().to(torch.float64)
    diag = torch.diagonal(H64)
    dead = diag <= 0
    if dead.any():
        H64[dead, dead] = 1.0
    mean_d = float(torch.mean(torch.clamp(diag, min=0.0)).item())
    lam = max(damp * mean_d, 1e-6)
    H64.diagonal().add_(lam)
    for mult in (1.0, 10.0, 100.0):
        try:
            L = torch.linalg.cholesky(H64)
            H_inv = torch.cholesky_inverse(L)
            return H_inv.to(device=H_rot.device, dtype=torch.float32)
        except RuntimeError:
            H64.diagonal().add_(lam * mult)
    warnings.warn(
        f"compute_h_inv_cholesky: Cholesky failed for a {C}x{C} Hessian even with 111x damping; "
        "using identity H^-1, so this matrix gets no Hessian weighting or LDLQ feedback",
        RuntimeWarning,
        stacklevel=2,
    )
    return torch.eye(C, dtype=torch.float32, device=H_rot.device)


def ldlq_upper_factor(H_inv: torch.Tensor) -> torch.Tensor:
    """Upper Cholesky factor U of H^-1 (H^-1 = U^T U); trailing blocks of U give the conditional inverse once earlier columns are fixed."""
    # Compute lower Cholesky L (H^-1 = L L^T) and transpose to U = L^T; avoids macOS PyTorch/Accelerate upper=True heap corruption.
    U = torch.linalg.cholesky(H_inv.detach().cpu().to(torch.float64)).T.contiguous()
    if H_inv.device.type == "mps":
        return U
    return U.to(device=H_inv.device, dtype=torch.float32)


def ldlq_gain(U: torch.Tensor, c0: int, c1: int, out_device: torch.device | None = None) -> torch.Tensor:
    """LDLQ feedback gain from columns [c0, c1) to columns [c1, C) given columns [0, c0) are fixed: U_bb^-1 U_b,rest."""
    if U.device.type == "mps":
        a = U[c0:c1, c0:c1].detach().cpu().to(torch.float64).contiguous()
        b = U[c0:c1, c1:].detach().cpu().to(torch.float64).contiguous()
        g = torch.linalg.solve_triangular(a, b, upper=True)
        return g.to(device=out_device or U.device, dtype=torch.float32)
    g = torch.linalg.solve_triangular(U[c0:c1, c0:c1].contiguous(), U[c0:c1, c1:].contiguous(), upper=True)
    if out_device is not None and (g.device != out_device or g.dtype != torch.float32):
        return g.to(device=out_device, dtype=torch.float32)
    return g


def _lane_maps(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    lanes = np.arange(32, dtype=np.int64)
    hc = (lanes & 1) * 4 + ((lanes >> 3) & 1) * 8
    hr = ((lanes >> 1) & 3) + ((lanes >> 4) & 1) * 4
    return torch.from_numpy(hc).to(device), torch.from_numpy(hr).to(device)


_GRID_CACHE: dict[tuple[int, int], torch.Tensor] = {}


def _build_spatial_grid(lut_by_c: torch.Tensor, bits: int) -> torch.Tensor:
    """
    Build exact spatial nearest-candidate lookup grid [N_cb, G, G, K] matching Metal/CUDA
    TFQ3 converters (RMAX=3.2): turns O(N_sym) exhaustive distance into O(K) O(1) gather!
    """
    key = (bits, lut_by_c.data_ptr())
    if key in _GRID_CACHE and _GRID_CACHE[key].device == lut_by_c.device:
        return _GRID_CACHE[key]
    G, K = (16, 8) if bits in (3, 4) else (32, 10)
    RMAX = 3.2
    n_cb = lut_by_c.shape[0]
    lut_cpu = lut_by_c.detach().cpu().numpy()  # [N_cb, N_sym, 2]
    xs = -RMAX + (np.arange(G, dtype=np.float32) + 0.5) * (2.0 * RMAX / G)
    gx, gy = np.meshgrid(xs, xs)  # [G, G]
    pts = np.stack([gx.reshape(-1), gy.reshape(-1)], axis=-1)  # [G*G, 2]
    grid = np.empty((n_cb, G * G, K), dtype=np.int64)
    for c in range(n_cb):
        cb = lut_cpu[c]  # [N_sym, 2]
        d = ((pts[:, None, 0] - cb[None, :, 0]) ** 2) + ((pts[:, None, 1] - cb[None, :, 1]) ** 2)
        grid[c] = np.argpartition(d, K, axis=1)[:, :K]
    gt = torch.from_numpy(grid.reshape(n_cb, G, G, K)).to(device=lut_by_c.device)
    _GRID_CACHE[key] = gt
    return gt


def _encode_block_viterbi(
    pairs_xy: torch.Tensor,
    pair_hw: torch.Tensor,
    init_st: torch.Tensor,
    bits: int,
    lut_by_c: torch.Tensor,
    c64_all: torch.Tensor,
    beam: int = 8,
    topk: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fast vectorized Hessian-weighted Beam-Viterbi trellis search over a 128-column TFQ3 block
    using TFQ3's exact spatial candidate grid (chunked over M for SLC cache locality).
    """
    M_tot = pairs_xy.shape[0]
    device = pairs_xy.device
    grid = _build_spatial_grid(lut_by_c, bits)  # [N_cb, G, G, K]
    G = grid.shape[1]
    K = grid.shape[3]
    RMAX = 3.2
    inv_step = G / (2.0 * RMAX)
    mask32 = 0xFFFFFFFF
    mult = HASH_MULT
    shift_bits = 2 * bits

    syms_all = torch.empty((M_tot, 32), dtype=torch.int64, device=device)
    q_all = torch.empty((M_tot, 32, 2), dtype=torch.float32, device=device)
    end_all = torch.empty((M_tot,), dtype=torch.int64, device=device)

    chunk_m = 4096
    for m0 in range(0, M_tot, chunk_m):
        m1 = min(m0 + chunk_m, M_tot)
        M = m1 - m0
        p_xy = pairs_xy[m0:m1]
        p_hw = pair_hw[m0:m1]
        cur_st = init_st[m0:m1].unsqueeze(1)
        cur_cost = torch.zeros((M, 1), dtype=torch.float32, device=device)
        cur_aff = torch.zeros((M, 1, 4), dtype=torch.float32, device=device)

        hist_prev = []
        hist_sym = []
        hist_q = []

        for t in range(32):
            B_cur = cur_st.shape[1]
            h_val = (cur_st * mult) & mask32
            if (t & 1) == 0:
                cur_aff = c64_all[(h_val >> 16) & 63]  # [M, B_cur, 4]

            s0 = cur_aff[..., 0]
            s1 = cur_aff[..., 1]
            t0 = cur_aff[..., 2]
            t1 = cur_aff[..., 3]
            inv_s0 = 1.0 / torch.where(s0.abs() > 1e-6, s0, torch.ones_like(s0))
            inv_s1 = 1.0 / torch.where(s1.abs() > 1e-6, s1, torch.ones_like(s1))

            px = p_xy[:, t, 0].unsqueeze(1)
            py = p_xy[:, t, 1].unsqueeze(1)
            hx = p_hw[:, t, 0].unsqueeze(1)
            hy = p_hw[:, t, 1].unsqueeze(1)

            tx = (px - t0) * inv_s0
            ty = (py - t1) * inv_s1
            gx_i = torch.clamp(((tx + RMAX) * inv_step).to(torch.int64), 0, G - 1)
            gy_i = torch.clamp(((ty + RMAX) * inv_step).to(torch.int64), 0, G - 1)

            if bits == 3:
                hl = h_val >> 24
                cb_id = hl & 3
                xor_m = (hl >> 2) & 63
            elif bits == 4:
                hl = h_val >> 22
                cb_id = hl & 3
                xor_m = (hl >> 2) & 255
            else:
                cb_id = torch.zeros_like(h_val)
                xor_m = (h_val >> 22) & 1023

            # O(1) spatial lookup of K nearest raw codebook entries!
            cand_raw = grid[cb_id, gy_i, gx_i]                      # [M, B_cur, K]
            cand_pt = lut_by_c[cb_id.unsqueeze(-1), cand_raw]       # [M, B_cur, K, 2]
            cand_qx = cand_pt[..., 0] * s0.unsqueeze(-1) + t0.unsqueeze(-1)
            cand_qy = cand_pt[..., 1] * s1.unsqueeze(-1) + t1.unsqueeze(-1)
            dx = px.unsqueeze(-1) - cand_qx
            dy = py.unsqueeze(-1) - cand_qy
            step_d = hx.unsqueeze(-1) * (dx * dx) + hy.unsqueeze(-1) * (dy * dy)

            cand_sym = cand_raw ^ xor_m.unsqueeze(-1)
            cand_cost = cur_cost.unsqueeze(-1) + step_d
            cand_q = torch.stack((cand_qx, cand_qy), dim=-1)
            cand_st = ((cur_st.unsqueeze(-1) << shift_bits) | cand_sym) & mask32

            flat_cost = cand_cost.reshape(M, B_cur * K)
            B_next = min(beam, B_cur * K)
            best_cost, best_flat = torch.topk(flat_cost, k=B_next, dim=-1, largest=False)

            parent_b = best_flat // K
            cur_cost = best_cost
            cur_st = torch.gather(cand_st.reshape(M, B_cur * K), 1, best_flat)
            chosen_sym = torch.gather(cand_sym.reshape(M, B_cur * K), 1, best_flat)
            chosen_q = torch.gather(
                cand_q.reshape(M, B_cur * K, 2), 1, best_flat.unsqueeze(-1).expand(-1, -1, 2)
            )
            if (t & 1) == 0:
                cur_aff = torch.gather(cur_aff, 1, parent_b.unsqueeze(-1).expand(-1, -1, 4))

            hist_prev.append(parent_b)
            hist_sym.append(chosen_sym)
            hist_q.append(chosen_q)

        best_idx = torch.argmin(cur_cost, dim=-1)
        end_all[m0:m1] = torch.gather(cur_st, 1, best_idx.unsqueeze(1)).squeeze(1)
        idx = best_idx
        for t in range(31, -1, -1):
            syms_all[m0:m1, t] = torch.gather(hist_sym[t], 1, idx.unsqueeze(1)).squeeze(1)
            q_all[m0:m1, t, :] = torch.gather(
                hist_q[t], 1, idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, 2)
            ).squeeze(1)
            idx = torch.gather(hist_prev[t], 1, idx.unsqueeze(1)).squeeze(1)

    return syms_all, q_all, end_all


def quantize_tfq3_matrix(
    W_orig: torch.Tensor,
    bits: int,
    tables: dict,
    H: torch.Tensor | None = None,
    damp: float = 0.01,
    beam: int = 8,
    topk: int = 8,
    refine_scales: bool = True,
    precomputed_h: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> dict:
    """
    Quantize a 2D weight matrix W [R, C] into exact TFQ3 (`TFPN` v3) packed uint32 words & fp16 scales.
    - If H is None: 100% data-free TFQ3 Beam-Viterbi encoding.
    - If H is provided ([C, C] input Hessian X^T X / N):
        Hessian-weighted Beam-Viterbi trellis + exact 128-col block-Cholesky LDLQ error feedback
        + closed-form Hessian-optimal per-tile fp16 scale refinement.
    """
    assert bits in (3, 4, 5)
    device = W_orig.device
    W_pad, orig_R, orig_C = pad_weight_128(W_orig)
    R, C = W_pad.shape
    n_br, n_bc = R // 128, C // 128
    n_strips = R // 16
    wpb = bits * 2

    W_rot0 = rotate_2d_torch(W_pad)
    W_rot = W_rot0.clone()

    H_rot = None
    H_inv = None
    h_diag = torch.ones(C, dtype=torch.float32, device=device)
    if precomputed_h is not None:
        H_rot, H_inv = precomputed_h[0].to(device), precomputed_h[1].to(device)
        hd = torch.clamp(torch.diagonal(H_rot), min=1e-8)
        h_diag = hd / torch.mean(hd)
    elif H is not None:
        H_pad = pad_hessian_128(H, C)
        H_rot = rotate_hessian_torch(H_pad)
        H_inv = compute_h_inv_cholesky(H_rot, damp=damp)
        hd = torch.clamp(torch.diagonal(H_rot), min=1e-8)
        h_diag = hd / torch.mean(hd)
    U_inv = ldlq_upper_factor(H_inv) if H_inv is not None else None

    shrink = {3: 0.92, 4: 0.94, 5: 0.96}[bits]
    blocks = W_rot0.reshape(n_br, 128, n_bc, 128).permute(0, 2, 1, 3)
    scales_f16 = (torch.sqrt(torch.mean(blocks * blocks, dim=(2, 3)) + 1e-12) * shrink).to(torch.float16)
    scales = scales_f16.to(torch.float32)

    lut_all = torch.from_numpy(tables["lut"][bits].astype(np.float32)).to(device)
    c64_all = torch.from_numpy(tables["c64"][bits].astype(np.float32)).to(device)
    if bits == 3:
        lut_by_c = lut_all.reshape(64, 4, 2).permute(1, 0, 2).contiguous()
    elif bits == 4:
        lut_by_c = lut_all.reshape(256, 4, 2).permute(1, 0, 2).contiguous()
    else:
        lut_by_c = lut_all.unsqueeze(0).contiguous()

    hc, hr = _lane_maps(device)
    strip_r0 = torch.arange(n_strips, device=device, dtype=torch.int64) * 16
    row_lo = strip_r0[:, None] + hr[None, :]  # [S, 32]
    row_hi = row_lo + 8
    strip_rb = torch.arange(n_strips, device=device, dtype=torch.int64) // 8

    # Build exact [S, 32, 32] (strip, lane, pair_t) -> (row, col_x, col_y) lookup inside a 128-col tile!
    # t = h*16 + k*4 + p, where p=0:(lo,hc+0,1), p=1:(lo,hc+2,3), p=2:(hi,hc+0,1), p=3:(hi,hc+2,3)
    pair_rows = torch.empty((n_strips, 32, 32), dtype=torch.int64, device=device)
    pair_cx = torch.empty((32, 32), dtype=torch.int64, device=device)  # [lane, t] offset inside 128
    for h in range(2):
        for k in range(4):
            t0 = h * 16 + k * 4
            c_sub = h * 64 + k * 16 + hc  # [32]
            pair_rows[:, :, t0 + 0] = row_lo
            pair_rows[:, :, t0 + 1] = row_lo
            pair_rows[:, :, t0 + 2] = row_hi
            pair_rows[:, :, t0 + 3] = row_hi
            pair_cx[:, t0 + 0] = c_sub + 0
            pair_cx[:, t0 + 1] = c_sub + 2
            pair_cx[:, t0 + 2] = c_sub + 0
            pair_cx[:, t0 + 3] = c_sub + 2

    words_out = torch.zeros((n_strips, n_bc * wpb * 32), dtype=torch.int64, device=device)
    Q_unit = torch.zeros_like(W_rot0)

    lane_arr = np.arange(32, dtype=np.uint32)
    init_st_np = (np.uint32(INIT_ST_BASE) ^ ((lane_arr * np.uint32(HASH_MULT)) & np.uint32(0xFFFFFFFF))).astype(np.int64)
    st = torch.from_numpy(init_st_np).to(device).unsqueeze(0).expand(n_strips, 32).reshape(-1).clone()

    mask32 = 0xFFFFFFFF
    M = n_strips * 32
    flat_rows = pair_rows.reshape(M, 32)
    lane_idx = torch.arange(32, device=device, dtype=torch.int64)

    for cb in range(n_bc):
        c_blk = cb * 128
        # Recompute current tile scale from LDLQ-updated W_rot[:, c_blk:c_blk+128] so dynamic range tracks feedback!
        if H_inv is not None and cb > 0:
            blk_now = W_rot[:, c_blk : c_blk + 128].reshape(n_br, 128, 128)
            sc_now = (torch.sqrt(torch.mean(blk_now * blk_now, dim=(1, 2)) + 1e-12) * shrink).to(torch.float16)
            scales_f16[:, cb] = sc_now
            scales[:, cb] = sc_now.to(torch.float32)

        sc_col = scales[strip_rb, cb].unsqueeze(1).expand(n_strips, 32).reshape(M, 1)  # [M, 1]
        inv_sc = torch.where(sc_col > 1e-12, 1.0 / sc_col, torch.zeros_like(sc_col))

        cx_abs = (c_blk + pair_cx).unsqueeze(0).expand(n_strips, 32, 32).reshape(M, 32)
        cy_abs = cx_abs + 1

        px = W_rot[flat_rows, cx_abs] * inv_sc
        py = W_rot[flat_rows, cy_abs] * inv_sc
        pairs_xy = torch.stack((px, py), dim=-1)  # [M, 32, 2]

        hx = h_diag[cx_abs]
        hy = h_diag[cy_abs]
        pair_hw = torch.stack((hx, hy), dim=-1)   # [M, 32, 2]

        syms, q_xy, st_end = _encode_block_viterbi(
            pairs_xy, pair_hw, st, bits, lut_by_c, c64_all, beam=beam, topk=topk
        )

        # Scatter reconstructed unit-scale weights into Q_unit
        Q_unit[flat_rows, cx_abs] = q_xy[..., 0]
        Q_unit[flat_rows, cy_abs] = q_xy[..., 1]

        # Pack symbols into exact TFPN uint32 words!
        syms_s = syms.reshape(n_strips, 32, 32)  # [S, lane, t]
        for h in range(2):
            sub_syms = [
                (
                    syms_s[:, :, h * 16 + k * 4 + 0],
                    syms_s[:, :, h * 16 + k * 4 + 1],
                    syms_s[:, :, h * 16 + k * 4 + 2],
                    syms_s[:, :, h * 16 + k * 4 + 3],
                )
                for k in range(4)
            ]
            if bits == 4:
                base_off = (cb * 2 + h) * 128
                for k, (s0, s1, s2, s3) in enumerate(sub_syms):
                    words_out[:, base_off + lane_idx * 4 + k] = (s0 << 24) | (s1 << 16) | (s2 << 8) | s3
                if h == 1:
                    st = (words_out[:, base_off + lane_idx * 4 + 3] & mask32).reshape(M)
            elif bits == 3:
                m24 = [(s0 << 18) | (s1 << 12) | (s2 << 6) | s3 for (s0, s1, s2, s3) in sub_syms]
                a0 = ((m24[0] << 8) | (m24[1] >> 16)) & mask32
                a1 = (((m24[1] & 0xFFFF) << 16) | (m24[2] >> 8)) & mask32
                a2 = (((m24[2] & 0xFF) << 24) | m24[3]) & mask32
                base_off = (cb * 6 + h * 3) * 32
                words_out[:, base_off + 0 : base_off + 32] = a0
                words_out[:, base_off + 32 : base_off + 64] = a1
                words_out[:, base_off + 64 : base_off + 96] = a2
                if h == 1:
                    st = (a2 & mask32).reshape(M)
            else:  # bits == 5
                hi8 = [(s0 >> 2) & 255 for (s0, _, _, _) in sub_syms]
                lo32 = [(((s0 & 3) << 30) | (s1 << 20) | (s2 << 10) | s3) & mask32 for (s0, s1, s2, s3) in sub_syms]
                w0 = ((hi8[0] << 24) | (lo32[0] >> 8)) & mask32
                w1 = (((lo32[0] & 0xFF) << 24) | (hi8[1] << 16) | (lo32[1] >> 16)) & mask32
                w2 = (((lo32[1] & 0xFFFF) << 16) | (hi8[2] << 8) | (lo32[2] >> 24)) & mask32
                w3 = (((lo32[2] & 0xFFFFFF) << 8) | hi8[3]) & mask32
                w4 = lo32[3] & mask32
                base_off = (cb * 10 + h * 5) * 32
                words_out[:, base_off + 0 : base_off + 32] = w0
                words_out[:, base_off + 32 : base_off + 64] = w1
                words_out[:, base_off + 64 : base_off + 96] = w2
                words_out[:, base_off + 96 : base_off + 128] = w3
                words_out[:, base_off + 128 : base_off + 160] = w4
                if h == 1:
                    st = (w4 & mask32).reshape(M)

        # Exact 128-column block-Cholesky LDLQ error feedback to all remaining columns!
        if H_inv is not None and (c_blk + 128) < C:
            sc_rows = scales[:, cb].repeat_interleave(128)[:, None]
            E_blk = W_rot[:, c_blk : c_blk + 128] - Q_unit[:, c_blk : c_blk + 128] * sc_rows
            if device.type == "mps":
                # Avoid compiling dozens of shrinking-width MPSGraph matmuls (MLIR StorageUniquer crash).
                gain = ldlq_gain(U_inv, c_blk, c_blk + 128).to(torch.float32)
                W_rot[:, c_blk + 128 :] -= (E_blk.detach().cpu() @ gain).to(device)
            else:
                W_rot[:, c_blk + 128 :] -= E_blk @ ldlq_gain(U_inv, c_blk, c_blk + 128, out_device=device)

    # Optional closed-form per-128x128-tile Hessian scale refinement (exact same single fp16 scale!)
    if refine_scales:
        for cb in range(n_bc):
            c0, c1 = cb * 128, (cb + 1) * 128
            H_blk = H_rot[c0:c1, c0:c1] if H_rot is not None else None
            for rb in range(n_br):
                r0, r1 = rb * 128, (rb + 1) * 128
                W_b = W_rot0[r0:r1, c0:c1]
                Q_b = Q_unit[r0:r1, c0:c1]
                if H_blk is not None:
                    QH = Q_b @ H_blk
                    num = float(torch.sum(W_b * QH).item())
                    den = float(torch.sum(Q_b * QH).item())
                else:
                    num = float(torch.sum(W_b * Q_b).item())
                    den = float(torch.sum(Q_b * Q_b).item())
                if den > 1e-12 and num > 0.0 and math.isfinite(num / den):
                    scales_f16[rb, cb] = torch.tensor(min(num / den, 65504.0), dtype=torch.float16, device=device)

    scales_final = scales_f16.to(torch.float32)
    W_rot_hat = Q_unit * scales_final.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    W_hat = unrotate_2d_torch(W_rot_hat)[:orig_R, :orig_C]

    diff = W_orig.detach().cpu().to(torch.float64) - W_hat.detach().cpu().to(torch.float64)
    W64 = W_orig.detach().cpu().to(torch.float64)
    w_snr = 10.0 * math.log10(float(torch.sum(W64 * W64).item()) / max(float(torch.sum(diff * diff).item()), 1e-30))
    out_snr = None
    if H is not None:
        H64 = H.detach().cpu().to(torch.float64)
        out_snr = 10.0 * math.log10(
            float(torch.sum((W64 @ H64) * W64).item()) / max(float(torch.sum((diff @ H64) * diff).item()), 1e-30)
        )

    return {
        "W_hat": W_hat,
        "scales": scales_f16.detach().cpu().numpy(),
        "words": words_out.reshape(-1).detach().cpu().numpy().astype(np.uint32),
        "weight_snr_db": w_snr,
        "output_snr_db": out_snr,
        "orig_R": orig_R,
        "orig_C": orig_C,
        "pad_R": R,
        "pad_C": C,
        "bits": bits,
    }


def quantize_tfq4_matrix(
    W_orig: torch.Tensor,
    e8p_lut: np.ndarray,
    H: torch.Tensor | None = None,
    damp: float = 0.01,
    refine_scales: bool = True,
    precomputed_h: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> dict:
    """
    Quantize a 2D weight matrix W [R, C] into exact TFQ4-E8P (`TFP4` v4, 2.001 bpw) packed words & fp16 scales.
    Supports both 100% data-free (H=None) and Hessian-LDLQ (H=[C, C] with 8-col subtile error feedback).
    """
    orig_device = W_orig.device
    # On Apple MPS, the 8-column LDLQ loop creates C/8 (~1,120) distinct shrinking matmul shapes,
    # triggering Apple MPSGraph MLIR StorageUniquer crashes; CPU Accelerate BLAS is both faster and crash-free.
    device = torch.device("cpu") if orig_device.type == "mps" else orig_device
    W_work = W_orig.detach().to(device=device, dtype=torch.float32)
    W_pad, orig_R, orig_C = pad_weight_128(W_work)
    R, C = W_pad.shape
    n_br, n_bc = R // 128, C // 128

    W_rot0 = rotate_2d_torch(W_pad)
    W_rot = W_rot0.clone()

    H_rot = None
    H_inv = None
    if precomputed_h is not None:
        H_rot, H_inv = precomputed_h[0].to(device), precomputed_h[1].to(device)
    elif H is not None:
        H_pad = pad_hessian_128(H.to(device), C)
        H_rot = rotate_hessian_torch(H_pad)
        H_inv = compute_h_inv_cholesky(H_rot, damp=damp)
    U_inv = ldlq_upper_factor(H_inv) if H_inv is not None else None

    blocks = W_rot0.reshape(n_br, 128, n_bc, 128).permute(0, 2, 1, 3)
    scales_f16 = torch.sqrt(torch.mean(blocks * blocks, dim=(2, 3)) + 1e-12).to(torch.float16)
    scales = scales_f16.to(torch.float32)

    lut_t = torch.from_numpy(e8p_lut.astype(np.float32)).to(device)
    lut_n2 = torch.sum(lut_t * lut_t, dim=1)[None, :]

    Q_unit = torch.zeros_like(W_rot0)
    codes_u16 = torch.zeros((R, C // 8), dtype=torch.int64, device=device)
    bit_shifts = torch.arange(8, device=device, dtype=torch.int64)

    for g in range(C // 8):
        c0 = g * 8
        cb = c0 // 128
        if H_inv is not None and (c0 % 128) == 0 and cb > 0:
            blk_now = W_rot[:, c0 : c0 + 128].reshape(n_br, 128, 128)
            sc_now = torch.sqrt(torch.mean(blk_now * blk_now, dim=(1, 2)) + 1e-12).to(torch.float16)
            scales_f16[:, cb] = sc_now
            scales[:, cb] = sc_now.to(torch.float32)

        sc_rows = scales[:, cb].repeat_interleave(128)[:, None]
        inv_sc = torch.where(sc_rows > 1e-12, 1.0 / sc_rows, torch.zeros_like(sc_rows))
        v = W_rot[:, c0 : c0 + 8] * inv_sc
        neg = (v < 0.0).to(torch.int64)
        sm = torch.sum(neg << bit_shifts[None, :], dim=1)
        sgn = torch.where(v < 0.0, -1.0, 1.0)
        av = v.abs()
        dist = lut_n2 - 2.0 * (av @ lut_t.T)
        best_k = torch.argmin(dist, dim=1)
        q_sub = lut_t[best_k] * sgn
        Q_unit[:, c0 : c0 + 8] = q_sub
        codes_u16[:, g] = (sm << 8) | best_k

        if H_inv is not None and (c0 + 8) < C:
            E_sub = W_rot[:, c0 : c0 + 8] - q_sub * sc_rows
            W_rot[:, c0 + 8 :] -= E_sub @ ldlq_gain(U_inv, c0, c0 + 8, out_device=device)

    if refine_scales:
        for cb in range(n_bc):
            c0, c1 = cb * 128, (cb + 1) * 128
            H_blk = H_rot[c0:c1, c0:c1] if H_rot is not None else None
            for rb in range(n_br):
                r0, r1 = rb * 128, (rb + 1) * 128
                W_b = W_rot0[r0:r1, c0:c1]
                Q_b = Q_unit[r0:r1, c0:c1]
                if H_blk is not None:
                    QH = Q_b @ H_blk
                    num = float(torch.sum(W_b * QH).item())
                    den = float(torch.sum(Q_b * QH).item())
                else:
                    num = float(torch.sum(W_b * Q_b).item())
                    den = float(torch.sum(Q_b * Q_b).item())
                if den > 1e-12 and num > 0.0 and math.isfinite(num / den):
                    scales_f16[rb, cb] = torch.tensor(min(num / den, 65504.0), dtype=torch.float16, device=device)

    scales_final = scales_f16.to(torch.float32)
    W_hat = unrotate_2d_torch(Q_unit * scales_final.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1))[:orig_R, :orig_C]

    lo = codes_u16[:, 0::2]
    hi = codes_u16[:, 1::2]
    words = (lo | (hi << 16)).reshape(-1).detach().cpu().numpy().astype(np.uint32)

    diff = W_orig.detach().cpu().to(torch.float64) - W_hat.detach().cpu().to(torch.float64)
    W64 = W_orig.detach().cpu().to(torch.float64)
    w_snr = 10.0 * math.log10(float(torch.sum(W64 * W64).item()) / max(float(torch.sum(diff * diff).item()), 1e-30))
    out_snr = None
    if H is not None:
        H64 = H.detach().cpu().to(torch.float64)
        out_snr = 10.0 * math.log10(
            float(torch.sum((W64 @ H64) * W64).item()) / max(float(torch.sum((diff @ H64) * diff).item()), 1e-30)
        )

    return {
        "W_hat": W_hat.to(orig_device),
        "scales": scales_f16.detach().cpu().numpy(),
        "words": words,
        "weight_snr_db": w_snr,
        "output_snr_db": out_snr,
        "orig_R": orig_R,
        "orig_C": orig_C,
        "pad_R": R,
        "pad_C": C,
        "bits": 2,
    }


def write_tfpn_shard(path: str, records: list[dict]) -> int:
    """Write a 16 KiB-aligned TFPN v3 binary shard compatible with Metal, CUDA, and tfq3-inspect."""
    all_scales = []
    all_words = []
    meta_bytes = bytearray()
    sc_off = 0
    w_off = 0
    for rec in records:
        sc = rec["scales"].astype(np.float16).reshape(-1)
        wd = rec["words"].astype(np.uint32).reshape(-1)
        nb = rec["name"].encode("utf-8")[:79]
        m = bytearray(128)
        m[: len(nb)] = nb
        struct.pack_into(
            "<IIIIQQfIII", m, 80,
            rec["pad_R"], rec["pad_C"], rec["bits"], sc_off,
            w_off, len(wd), float(rec["weight_snr_db"]),
            rec["orig_R"], rec["orig_C"], 0,
        )
        meta_bytes.extend(m)
        all_scales.append(sc)
        all_words.append(wd)
        sc_off += len(sc)
        w_off += len(wd)

    sc_arr = np.concatenate(all_scales) if all_scales else np.zeros(0, dtype=np.float16)
    wd_arr = np.concatenate(all_words) if all_words else np.zeros(0, dtype=np.uint32)
    off_meta = 64
    off_scales = align16k(off_meta + len(meta_bytes))
    off_weights = align16k(off_scales + sc_arr.nbytes)
    tot_sz = align16k(off_weights + wd_arr.nbytes)

    sc_bytes = sc_arr.tobytes()
    wd_bytes = wd_arr.tobytes()
    xxh = tfq3_ref.xxh64_sections((meta_bytes, sc_bytes, wd_bytes))
    hdr = struct.pack(
        "<IIIIQQQQQQ",
        MAGIC_TFPN, 3, len(records), tfq3_ref.FLAG_XXH64_PRESENT,
        off_meta, off_scales, off_weights,
        len(sc_arr), len(wd_arr), xxh,
    )
    with open(path, "wb") as f:
        f.write(hdr)
        f.write(meta_bytes)
        f.write(b"\x00" * (off_scales - (off_meta + len(meta_bytes))))
        f.write(sc_bytes)
        f.write(b"\x00" * (off_weights - (off_scales + len(sc_bytes))))
        f.write(wd_bytes)
        f.write(b"\x00" * (tot_sz - (off_weights + len(wd_bytes))))
    return tot_sz


def collect_linear_hessians(
    model: torch.nn.Module,
    input_ids_batches: list[torch.Tensor],
    module_names: list[str],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Collect exact input Hessians H = (1/N) sum X^T X for selected nn.Linear modules via forward hooks."""
    name_set = set(module_names)
    H_sum: dict[str, torch.Tensor] = {}
    H_cnt: dict[str, int] = {}
    hooks = []

    def make_hook(nm: str):
        def _hook(_mod, inp, _out):
            x = inp[0].detach()
            x2d = x.reshape(-1, x.shape[-1]).to(torch.float32)
            n = x2d.shape[0]
            h = (x2d.T @ x2d).cpu().to(torch.float64)
            if nm not in H_sum:
                H_sum[nm] = h
                H_cnt[nm] = n
            else:
                H_sum[nm] += h
                H_cnt[nm] += n
        return _hook

    for nm, mod in model.named_modules():
        if nm in name_set and isinstance(mod, torch.nn.Linear):
            hooks.append(mod.register_forward_hook(make_hook(nm)))

    model.eval()
    with torch.no_grad():
        for ids in input_ids_batches:
            model(ids.to(device))

    for hk in hooks:
        hk.remove()

    return {nm: (H_sum[nm] / max(H_cnt[nm], 1)).to(torch.float32) for nm in H_sum}


if __name__ == "__main__":
    import subprocess
    import tempfile

    print("Running TFQ3 Hessian-Weighted Beam-Viterbi + LDLQ self-test...")
    dev = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    with tempfile.TemporaryDirectory() as tmp:
        gen_bin = os.path.join(REPO_ROOT, "build", "tfq3-generic-convert")
        insp_bin = os.path.join(REPO_ROOT, "build", "tfq3-inspect")
        if not os.path.isfile(gen_bin):
            raise SystemExit("Build tfq3-generic-convert first via 'make'")

        torch.manual_seed(42)
        R, C, N = 256, 512, 2048
        eig = torch.ones(C, dtype=torch.float32)
        eig[:16] = 20.0
        eig[16:64] = 4.0
        Q_rand, _ = torch.linalg.qr(torch.randn(C, C))
        X = (torch.randn(N, C) * torch.sqrt(eig)[None, :]) @ Q_rand.T
        H = (X.T @ X) / N
        W = torch.randn(R, C) * 0.02

        m_dir, o_dir = os.path.join(tmp, "m"), os.path.join(tmp, "o")
        os.makedirs(m_dir)
        d = np.random.default_rng(0).standard_normal((128, 128)).astype(np.float32)
        hdr = json.dumps({"w.weight": {"dtype": "F32", "shape": [128, 128], "data_offsets": [0, d.nbytes]}}).encode()
        with open(os.path.join(m_dir, "model.safetensors"), "wb") as f:
            f.write(struct.pack("<Q", len(hdr)) + hdr + d.tobytes())
        open(os.path.join(m_dir, "config.json"), "w").write('{"model_type":"llama"}')
        subprocess.run([gen_bin, "--model", m_dir, "--out", o_dir, "--bits", "4"], check=True, stdout=subprocess.DEVNULL)
        tbl_path = os.path.join(o_dir, "pw_v3_tables.bin")
        tables = tfq3_ref.load_tables(tbl_path)

        for bits in (3, 4, 5):
            t0 = time.time()
            r_df = quantize_tfq3_matrix(W.to(dev), bits, tables, H=None, beam=16, topk=6, refine_scales=False)
            diff_df = (W - r_df["W_hat"].cpu()).to(torch.float64)
            H64, W64 = H.to(torch.float64), W.to(torch.float64)
            out_df = 10.0 * math.log10(float(torch.sum((W64 @ H64) * W64)) / float(torch.sum((diff_df @ H64) * diff_df)))

            r_hl = quantize_tfq3_matrix(W.to(dev), bits, tables, H=H.to(dev), beam=16, topk=6, refine_scales=True)
            dt = time.time() - t0
            shard = os.path.join(tmp, f"test_{bits}b.tffpw")
            write_tfpn_shard(shard, [{**r_hl, "name": "layer.0.proj"}])
            subprocess.run([insp_bin, shard], check=True, stdout=subprocess.DEVNULL)
            W_dec = tfq3_ref.dequantize_tfpn_tensor(shard, tbl_path, "layer.0.proj", max_blocks=8)
            max_diff = float(np.max(np.abs(W_dec - r_hl["W_hat"].cpu().numpy())))
            assert max_diff < 1e-4, f"Binary decode mismatch: {max_diff}"
            print(
                f"PASS {bits}b | W-SNR={r_df['weight_snr_db']:5.2f} dB | "
                f"Data-Free Out-SNR={out_df:5.2f} dB -> Hessian-Viterbi+LDLQ Out-SNR={r_hl['output_snr_db']:5.2f} dB "
                f"(+{r_hl['output_snr_db'] - out_df:.2f} dB) | TFPN max_err={max_diff:.2e} | {dt:.2f}s"
            )
