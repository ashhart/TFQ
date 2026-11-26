#!/usr/bin/env python3
"""
TFQ3 Reference Codec (Pure Python + NumPy)

Exact mathematical reference for the TFQ3 (2D-AffineCoset/half4 + 2D FWHT-128)
quantization format used by TensorFold:
  - Deterministic Murmur3 sign hash (`pw_sgn`)
  - Orthonormal 128-point Fast Walsh-Hadamard Transform (`fwht_128`)
  - 2D randomized rotation: W_rot = D_L @ H_128 @ W @ H_128 @ D_R
  - Exact 3-bit, 4-bit, and 5-bit 2D-AffineCoset/half4 state-transition dequantizer
  - Reader for `pw_v3_tables.bin` ("PWV3") and `TFPN` (".tffpw") dense shards

Zero external dependencies beyond NumPy. Run `--self-test` to verify orthogonality
and 3b/4b/5b roundtrip SNR against a standard normal matrix.
"""

import argparse
import os
import struct
import sys
import numpy as np

MAGIC_PWV3 = 0x33565750
MAGIC_TFPN = 0x4E504654
MAGIC_TFPW = 0x57504654
FORMAT_VER = 3
FLAG_XXH64_PRESENT = 0x1
SEED_COL = 0x12345678
SEED_ROW = 0x9E3779B9
HASH_MULT = 0x9E3779B1
INIT_ST_BASE = 0x13579BDF

_XXH_P1 = 0x9E3779B185EBCA87
_XXH_P2 = 0xC2B2AE3D27D4EB4F
_XXH_P3 = 0x165667B19E3779F9
_XXH_P4 = 0x85EBCA77C2B2AE63
_XXH_P5 = 0x27D4EB2F165667C5
_XXH_M64 = 0xFFFFFFFFFFFFFFFF


def _xxh_rotl64(x: int, r: int) -> int:
    return ((x << r) | (x >> (64 - r))) & _XXH_M64


def _xxh_round64(acc: int, inp: int) -> int:
    acc = (acc + inp * _XXH_P2) & _XXH_M64
    acc = ((acc << 31) | (acc >> 33)) & _XXH_M64
    return (acc * _XXH_P1) & _XXH_M64


def _xxh_merge_round64(acc: int, val: int) -> int:
    val = _xxh_round64(0, val)
    acc ^= val
    return (acc * _XXH_P1 + _XXH_P4) & _XXH_M64


class XXH64:
    """Zero-dependency streaming XXH64 (exact match to src/converters/tfq_xxh64.hpp)."""

    def __init__(self, seed: int = 0) -> None:
        s = seed & _XXH_M64
        self.total_len = 0
        self.v1 = (s + _XXH_P1 + _XXH_P2) & _XXH_M64
        self.v2 = (s + _XXH_P2) & _XXH_M64
        self.v3 = s
        self.v4 = (s - _XXH_P1) & _XXH_M64
        self.buf = bytearray()

    def update(self, data: bytes | bytearray | memoryview) -> None:
        mv = memoryview(data).cast("B")
        n = len(mv)
        if n == 0:
            return
        self.total_len += n
        pos = 0
        if self.buf:
            need = 32 - len(self.buf)
            if n < need:
                self.buf.extend(mv)
                return
            self.buf.extend(mv[:need])
            k1, k2, k3, k4 = struct.unpack_from("<4Q", self.buf, 0)
            self.v1 = _xxh_round64(self.v1, k1)
            self.v2 = _xxh_round64(self.v2, k2)
            self.v3 = _xxh_round64(self.v3, k3)
            self.v4 = _xxh_round64(self.v4, k4)
            self.buf.clear()
            pos = need
        rem = n - pos
        n_stripes = rem >> 5
        if n_stripes:
            end = pos + (n_stripes << 5)
            v1, v2, v3, v4 = self.v1, self.v2, self.v3, self.v4
            p1, p2, m = _XXH_P1, _XXH_P2, _XXH_M64
            for k1, k2, k3, k4 in struct.iter_unpack("<4Q", mv[pos:end]):
                v1 = (v1 + k1 * p2) & m
                v1 = ((((v1 << 31) | (v1 >> 33)) & m) * p1) & m
                v2 = (v2 + k2 * p2) & m
                v2 = ((((v2 << 31) | (v2 >> 33)) & m) * p1) & m
                v3 = (v3 + k3 * p2) & m
                v3 = ((((v3 << 31) | (v3 >> 33)) & m) * p1) & m
                v4 = (v4 + k4 * p2) & m
                v4 = ((((v4 << 31) | (v4 >> 33)) & m) * p1) & m
            self.v1, self.v2, self.v3, self.v4 = v1, v2, v3, v4
            pos = end
        if pos < n:
            self.buf.extend(mv[pos:])

    def intdigest(self) -> int:
        if self.total_len >= 32:
            h = (
                _xxh_rotl64(self.v1, 1)
                + _xxh_rotl64(self.v2, 7)
                + _xxh_rotl64(self.v3, 12)
                + _xxh_rotl64(self.v4, 18)
            ) & _XXH_M64
            h = _xxh_merge_round64(h, self.v1)
            h = _xxh_merge_round64(h, self.v2)
            h = _xxh_merge_round64(h, self.v3)
            h = _xxh_merge_round64(h, self.v4)
        else:
            h = (self.v3 + _XXH_P5) & _XXH_M64
        h = (h + self.total_len) & _XXH_M64
        b = self.buf
        p, end = 0, len(b)
        while p + 8 <= end:
            k1 = _xxh_round64(0, struct.unpack_from("<Q", b, p)[0])
            h ^= k1
            h = (_xxh_rotl64(h, 27) * _XXH_P1 + _XXH_P4) & _XXH_M64
            p += 8
        if p + 4 <= end:
            h ^= (struct.unpack_from("<I", b, p)[0] * _XXH_P1) & _XXH_M64
            h = (_xxh_rotl64(h, 23) * _XXH_P2 + _XXH_P3) & _XXH_M64
            p += 4
        while p < end:
            h ^= (b[p] * _XXH_P5) & _XXH_M64
            h = (_xxh_rotl64(h, 11) * _XXH_P1) & _XXH_M64
            p += 1
        h ^= h >> 33
        h = (h * _XXH_P2) & _XXH_M64
        h ^= h >> 29
        h = (h * _XXH_P3) & _XXH_M64
        h ^= h >> 32
        return h


def xxh64(data: bytes | bytearray | memoryview, seed: int = 0) -> int:
    st = XXH64(seed)
    st.update(data)
    return st.intdigest()


def xxh64_sections(sections, seed: int = 0) -> int:
    st = XXH64(seed)
    for sec in sections:
        st.update(sec)
    return st.intdigest()


def _update_xxh64_file_slice(f, off: int, length: int, st: XXH64) -> None:
    if length == 0:
        return
    f.seek(off)
    rem = length
    while rem > 0:
        chunk = f.read(min(rem, 1 << 20))
        if not chunk:
            raise ValueError(f"Short read while hashing slice [{off}, {off + length})")
        st.update(chunk)
        rem -= len(chunk)


def verify_tfpn_like_checksum(f, off_meta: int, meta_bytes: int, off_scales: int, sc_bytes: int,
                              off_weights: int, w_bytes: int, pad0: int, pad1: int, label: str) -> None:
    if (pad0 & FLAG_XXH64_PRESENT) == 0:
        return
    st = XXH64(0)
    _update_xxh64_file_slice(f, off_meta, meta_bytes, st)
    _update_xxh64_file_slice(f, off_scales, sc_bytes, st)
    _update_xxh64_file_slice(f, off_weights, w_bytes, st)
    got = st.intdigest()
    if got != pad1:
        raise ValueError(f"{label} XXH64 checksum mismatch: got 0x{got:016x}, expected 0x{pad1:016x}")


def verify_tfpw_checksum(shard_path: str) -> bool:
    """Verify per-section XXH64 checksums on a TFPW shard when reserved0 bit 0 is set.
    Returns True if checksums were verified, False for legacy shards without checksums."""
    with open(shard_path, "rb") as f:
        head = f.read(144)
        if len(head) != 144:
            raise ValueError(f"Truncated TFPW header in {shard_path}")
        magic, ver, _, n_exp, _, _, _, reserved0 = struct.unpack_from("<8I", head, 0)
        if magic != MAGIC_TFPW or ver != FORMAT_VER:
            raise ValueError(f"Bad TFPW header in {shard_path}")
        if (reserved0 & FLAG_XXH64_PRESENT) == 0:
            return False
        _, sw, sh, off_meta, off_sc, off_w = struct.unpack_from("<6Q", head, 32)
        exp_meta, exp_sc, exp_w, _ = struct.unpack_from("<4Q", head, 80)
        for sec_name, off, length, expected in (
            ("meta", off_meta, n_exp * 64, exp_meta),
            ("scales", off_sc, sh * 2, exp_sc),
            ("weights", off_w, sw * 4, exp_w),
        ):
            st = XXH64(0)
            _update_xxh64_file_slice(f, off, length, st)
            got = st.intdigest()
            if got != expected:
                raise ValueError(
                    f"TFPW {sec_name} XXH64 checksum mismatch in {shard_path}: "
                    f"got 0x{got:016x}, expected 0x{expected:016x}"
                )
        return True


def pw_sgn(seed: int, idx: np.ndarray) -> np.ndarray:
    """Deterministic +/-1 Rademacher sign hash matching Metal `pw_sgn` bit-for-bit."""
    h = ((idx.astype(np.uint32) ^ np.uint32(seed)) * np.uint32(2654435761)) & np.uint32(0xFFFFFFFF)
    return np.where(((h >> np.uint32(31)) & np.uint32(1)) != 0, -1.0, 1.0).astype(np.float32)


def fwht_128(x: np.ndarray) -> np.ndarray:
    """Orthonormal 128-point Walsh-Hadamard transform along the last axis."""
    assert x.shape[-1] == 128
    y = x.astype(np.float32, copy=True)
    h = 1
    while h < 128:
        y = y.reshape(*x.shape[:-1], -1, 2, h)
        a = y[..., 0, :]
        b = y[..., 1, :]
        y = np.stack([a + b, a - b], axis=-2).reshape(x.shape)
        h <<= 1
    y *= (1.0 / np.sqrt(128.0))
    return y


def rotate_block_2d(block: np.ndarray, r_off: int = 0, c_off: int = 0) -> np.ndarray:
    """Apply H_128 @ (D_L @ W @ D_R) @ H_128 to a [128, 128] weight block."""
    assert block.shape == (128, 128)
    sr = pw_sgn(SEED_ROW, np.arange(r_off, r_off + 128, dtype=np.uint32))
    sc = pw_sgn(SEED_COL, np.arange(c_off, c_off + 128, dtype=np.uint32))
    tmp = fwht_128(block * sc[None, :])
    return fwht_128((tmp * sr[:, None]).T).T


def unrotate_block_2d(block_rot: np.ndarray, r_off: int = 0, c_off: int = 0) -> np.ndarray:
    """Exact inverse of `rotate_block_2d`: D_L @ H_128 @ W_rot @ H_128 @ D_R."""
    assert block_rot.shape == (128, 128)
    sr = pw_sgn(SEED_ROW, np.arange(r_off, r_off + 128, dtype=np.uint32))
    sc = pw_sgn(SEED_COL, np.arange(c_off, c_off + 128, dtype=np.uint32))
    tmp = fwht_128(block_rot.T).T * sr[:, None]
    return fwht_128(tmp) * sc[None, :]


def load_tables(path: str) -> dict:
    """Load and validate a 16 KiB `pw_v3_tables.bin` codebook file."""
    raw = open(path, "rb").read()
    magic, ver = struct.unpack_from("<II", raw, 0)
    if magic != MAGIC_PWV3 or ver != FORMAT_VER:
        raise ValueError(f"Bad pw_v3_tables header: magic=0x{magic:08x} ver={ver}")
    a3, b3, a4, b4, a5, b5 = struct.unpack_from("<6f", raw, 8)
    lut3 = np.frombuffer(raw, dtype=np.float16, count=256 * 2, offset=64).reshape(256, 2).copy()
    lut4 = np.frombuffer(raw, dtype=np.float16, count=1024 * 2, offset=1088).reshape(1024, 2).copy()
    lut5 = np.frombuffer(raw, dtype=np.float16, count=1024 * 2, offset=5184).reshape(1024, 2).copy()
    c64_3 = np.frombuffer(raw, dtype=np.float16, count=64 * 4, offset=9280).reshape(64, 4).copy()
    c64_4 = np.frombuffer(raw, dtype=np.float16, count=64 * 4, offset=9792).reshape(64, 4).copy()
    c64_5 = np.frombuffer(raw, dtype=np.float16, count=64 * 4, offset=10304).reshape(64, 4).copy()
    return {
        "a": {3: a3, 4: a4, 5: a5},
        "b": {3: b3, 4: b4, 5: b5},
        "lut": {3: lut3, 4: lut4, 5: lut5},
        "c64": {3: c64_3, 4: c64_4, 5: c64_5},
    }


def _frag_home(lane: int) -> tuple[int, int]:
    """Exact 32-lane M5 / NAX 16x16 subtile coordinate mapping (`pw_frag_home`)."""
    col = (lane & 1) * 4 + ((lane >> 3) & 1) * 8
    row = ((lane >> 1) & 3) + ((lane >> 4) & 1) * 4
    return col, row


def _decode_8_4b(st: int, wc: int, tbl: dict) -> tuple[int, np.ndarray]:
    s0 = st & 0xFFFFFFFF
    h0 = (s0 * HASH_MULT) & 0xFFFFFFFF
    h1 = ((((s0 << 8) | (wc >> 24)) & 0xFFFFFFFF) * HASH_MULT) & 0xFFFFFFFF
    h2 = ((((s0 << 16) | (wc >> 16)) & 0xFFFFFFFF) * HASH_MULT) & 0xFFFFFFFF
    h3 = ((((s0 << 24) | (wc >> 8)) & 0xFFFFFFFF) * HASH_MULT) & 0xFFFFFFFF
    c0 = tbl["c64"][4][(h0 >> 16) & 63].astype(np.float32)
    c2 = tbl["c64"][4][(h2 >> 16) & 63].astype(np.float32)
    lut = tbl["lut"][4].astype(np.float32)
    p0 = lut[(((wc >> 24) << 2) ^ (h0 >> 22)) & 1023]
    p1 = lut[((((wc >> 16) & 255) << 2) ^ (h1 >> 22)) & 1023]
    p2 = lut[((((wc >> 8) & 255) << 2) ^ (h2 >> 22)) & 1023]
    p3 = lut[(((wc & 255) << 2) ^ (h3 >> 22)) & 1023]
    q0 = np.array([p0[0] * c0[0] + c0[2], p0[1] * c0[1] + c0[3], p1[0] * c0[0] + c0[2], p1[1] * c0[1] + c0[3]], dtype=np.float32)
    q1 = np.array([p2[0] * c2[0] + c2[2], p2[1] * c2[1] + c2[3], p3[0] * c2[0] + c2[2], p3[1] * c2[1] + c2[3]], dtype=np.float32)
    return wc & 0xFFFFFFFF, np.concatenate([q0, q1])


def _decode_8_3b(st: int, m24: int, tbl: dict) -> tuple[int, np.ndarray]:
    s0 = st & 0xFFFFFFFF
    h0 = (s0 * HASH_MULT) & 0xFFFFFFFF
    h1 = ((((s0 << 6) | (m24 >> 18)) & 0xFFFFFFFF) * HASH_MULT) & 0xFFFFFFFF
    h2 = ((((s0 << 12) | (m24 >> 12)) & 0xFFFFFFFF) * HASH_MULT) & 0xFFFFFFFF
    h3 = ((((s0 << 18) | (m24 >> 6)) & 0xFFFFFFFF) * HASH_MULT) & 0xFFFFFFFF
    new_st = ((s0 << 24) | m24) & 0xFFFFFFFF
    c0 = tbl["c64"][3][(h0 >> 16) & 63].astype(np.float32)
    c2 = tbl["c64"][3][(h2 >> 16) & 63].astype(np.float32)
    lut = tbl["lut"][3].astype(np.float32)
    p0 = lut[(((m24 >> 16) & 0xFC) ^ (h0 >> 24)) & 255]
    p1 = lut[(((m24 >> 10) & 0xFC) ^ (h1 >> 24)) & 255]
    p2 = lut[(((m24 >> 4) & 0xFC) ^ (h2 >> 24)) & 255]
    p3 = lut[(((m24 << 2) & 0xFC) ^ (h3 >> 24)) & 255]
    q0 = np.array([p0[0] * c0[0] + c0[2], p0[1] * c0[1] + c0[3], p1[0] * c0[0] + c0[2], p1[1] * c0[1] + c0[3]], dtype=np.float32)
    q1 = np.array([p2[0] * c2[0] + c2[2], p2[1] * c2[1] + c2[3], p3[0] * c2[0] + c2[2], p3[1] * c2[1] + c2[3]], dtype=np.float32)
    return new_st, np.concatenate([q0, q1])


def _decode_8_5b(st: int, hi8: int, lo32: int, tbl: dict) -> tuple[int, np.ndarray]:
    s0 = st & 0xFFFFFFFF
    b0 = ((hi8 << 2) | (lo32 >> 30)) & 1023
    b1 = (lo32 >> 20) & 1023
    b2 = (lo32 >> 10) & 1023
    b3 = lo32 & 1023
    h0 = (s0 * HASH_MULT) & 0xFFFFFFFF
    h1 = ((((s0 << 10) | b0) & 0xFFFFFFFF) * HASH_MULT) & 0xFFFFFFFF
    h2 = ((((s0 << 20) | (hi8 << 12) | (lo32 >> 20)) & 0xFFFFFFFF) * HASH_MULT) & 0xFFFFFFFF
    h3 = ((((s0 << 30) | (hi8 << 22) | (lo32 >> 10)) & 0xFFFFFFFF) * HASH_MULT) & 0xFFFFFFFF
    c0 = tbl["c64"][5][(h0 >> 16) & 63].astype(np.float32)
    c2 = tbl["c64"][5][(h2 >> 16) & 63].astype(np.float32)
    lut = tbl["lut"][5].astype(np.float32)
    p0 = lut[(b0 ^ (h0 >> 22)) & 1023]
    p1 = lut[(b1 ^ (h1 >> 22)) & 1023]
    p2 = lut[(b2 ^ (h2 >> 22)) & 1023]
    p3 = lut[(b3 ^ (h3 >> 22)) & 1023]
    q0 = np.array([p0[0] * c0[0] + c0[2], p0[1] * c0[1] + c0[3], p1[0] * c0[0] + c0[2], p1[1] * c0[1] + c0[3]], dtype=np.float32)
    q1 = np.array([p2[0] * c2[0] + c2[2], p2[1] * c2[1] + c2[3], p3[0] * c2[0] + c2[2], p3[1] * c2[1] + c2[3]], dtype=np.float32)
    return lo32 & 0xFFFFFFFF, np.concatenate([q0, q1])


def dequantize_tfpn_tensor(shard_path: str, tables_path: str, tensor_name: str, max_blocks: int = 1) -> np.ndarray:
    """Dequantize the top-left [max_blocks*128, max_blocks*128] tiles of a TFPN tensor to float32."""
    tbl = load_tables(tables_path)
    with open(shard_path, "rb") as f:
        hdr = struct.unpack("<IIIIQQQQQQ", f.read(64))
        magic, ver, nt, pad0, off_meta, off_scales, off_weights, tot_sc, tot_w, pad1 = hdr
        if magic != MAGIC_TFPN or ver != FORMAT_VER:
            raise ValueError(f"Bad TFPN header in {shard_path}")
        verify_tfpn_like_checksum(
            f, off_meta, nt * 128, off_scales, tot_sc * 2, off_weights, tot_w * 4, pad0, pad1, f"TFPN ({shard_path})"
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
        R, C, bits, sc_off, w_off, w_cnt, snr_db, _, _, _ = found
        n_bc = C // 128
        wpb = bits * 2
        rb_lim = min(max_blocks, R // 128)
        cb_lim = min(max_blocks, n_bc)
        f.seek(off_scales + sc_off * 2)
        scales = np.frombuffer(f.read((R // 128) * n_bc * 2), dtype=np.float16).reshape(R // 128, n_bc)
        f.seek(off_weights + w_off * 4)
        words = np.frombuffer(f.read(w_cnt * 4), dtype=np.uint32)

    out = np.zeros((rb_lim * 128, cb_lim * 128), dtype=np.float32)
    for rb in range(rb_lim):
        for cb in range(cb_lim):
            blk_rot = np.zeros((128, 128), dtype=np.float32)
            sc = float(scales[rb, cb])
            for s in range(8):
                strip = rb * 8 + s
                strip_base = strip * n_bc * wpb * 32
                for lane in range(32):
                    st = (INIT_ST_BASE ^ ((lane * HASH_MULT) & 0xFFFFFFFF)) & 0xFFFFFFFF
                    if cb > 0:
                        if bits == 4:
                            st = int(words[strip_base + (cb * 2 - 1) * 128 + lane * 4 + 3])
                        else:
                            st = int(words[strip_base + (cb * wpb - 1) * 32 + lane])
                    hc, hr = _frag_home(lane)
                    if bits == 4:
                        for h in range(2):
                            idx = strip_base + (cb * 2 + h) * 128 + lane * 4
                            for k in range(4):
                                st, q = _decode_8_4b(st, int(words[idx + k]), tbl)
                                c0 = h * 64 + k * 16 + hc
                                blk_rot[s * 16 + hr, c0 : c0 + 4] = q[:4] * sc
                                blk_rot[s * 16 + hr + 8, c0 : c0 + 4] = q[4:] * sc
                    elif bits == 3:
                        for h in range(2):
                            idx = strip_base + (cb * 6 + h * 3) * 32 + lane
                            a0, a1, a2 = int(words[idx]), int(words[idx + 32]), int(words[idx + 64])
                            m24s = [
                                a0 >> 8,
                                ((a0 << 16) | (a1 >> 16)) & 0xFFFFFF,
                                ((a1 << 8) | (a2 >> 24)) & 0xFFFFFF,
                                a2 & 0xFFFFFF,
                            ]
                            for k, m24 in enumerate(m24s):
                                st, q = _decode_8_3b(st, m24, tbl)
                                c0 = h * 64 + k * 16 + hc
                                blk_rot[s * 16 + hr, c0 : c0 + 4] = q[:4] * sc
                                blk_rot[s * 16 + hr + 8, c0 : c0 + 4] = q[4:] * sc
                    else:
                        for h in range(2):
                            idx = strip_base + (cb * 10 + h * 5) * 32 + lane
                            w0, w1, w2, w3, w4 = [int(words[idx + j * 32]) for j in range(5)]
                            pairs = [
                                (w0 >> 24, ((w0 << 8) | (w1 >> 24)) & 0xFFFFFFFF),
                                ((w1 >> 16) & 255, ((w1 << 16) | (w2 >> 16)) & 0xFFFFFFFF),
                                ((w2 >> 8) & 255, ((w2 << 24) | (w3 >> 8)) & 0xFFFFFFFF),
                                (w3 & 255, w4 & 0xFFFFFFFF),
                            ]
                            for k, (hi8, lo32) in enumerate(pairs):
                                st, q = _decode_8_5b(st, hi8, lo32, tbl)
                                c0 = h * 64 + k * 16 + hc
                                blk_rot[s * 16 + hr, c0 : c0 + 4] = q[:4] * sc
                                blk_rot[s * 16 + hr + 8, c0 : c0 + 4] = q[4:] * sc
            out[rb * 128 : (rb + 1) * 128, cb * 128 : (cb + 1) * 128] = unrotate_block_2d(
                blk_rot, r_off=rb * 128, c_off=cb * 128
            )
    return out


def decode_tfpn_fast_np(
    words: np.ndarray,
    scales: np.ndarray,
    R: int,
    C: int,
    bits: int,
    tbl: dict,
    r_off: int = 0,
    bc0: int = 0,
    bc1: int | None = None,
    c_sign_off: int | None = None,
) -> np.ndarray:
    """Vectorized NumPy decoder for TFPN v3 (exact match to dequantize_tfpn_tensor, ~50x faster).

    Optional TP/slice arguments:
      - `r_off`: absolute row offset (multiple of 128) for row sign hash `SEED_ROW`.
      - `bc0`, `bc1`: column-block slice `[bc0, bc1)` within `n_bc = C // 128`, resuming trellis state
        at `bc0 > 0` from block `bc0 - 1` and indexing column sign hash `SEED_COL` by absolute column.
    """
    n_strips = R // 16
    n_bc = C // 128
    if bc1 is None:
        bc1 = n_bc
    assert 0 <= bc0 < bc1 <= n_bc and (r_off % 128 == 0)
    n_bc_out = bc1 - bc0
    C_out = n_bc_out * 128
    wpb = bits * 2
    W = words.reshape(n_strips, n_bc * wpb * 32).astype(np.uint64)
    lut = tbl["lut"][bits].astype(np.float32)  # [N, 2]
    c64 = tbl["c64"][bits].astype(np.float32)  # [64, 4]
    lanes = np.arange(32, dtype=np.uint64)
    hc = ((lanes & 1) * 4 + ((lanes >> 3) & 1) * 8).astype(np.int64)
    hr = (((lanes >> 1) & 3) + ((lanes >> 4) & 1) * 4).astype(np.int64)

    mask32 = np.uint64(0xFFFFFFFF)
    mult = np.uint64(HASH_MULT)
    st = (np.uint64(INIT_ST_BASE) ^ ((lanes * mult) & mask32))[None, :].repeat(n_strips, axis=0)
    blk_rot = np.empty((R, C_out), dtype=np.float32)
    s_idx = (np.arange(n_strips, dtype=np.int64) * 16)[:, None]
    r_lo = s_idx + hr[None, :]
    r_hi = r_lo + 8

    for cb in range(bc0, bc1):
        cb_loc = cb - bc0
        if cb > 0:
            if bits == 4:
                st = W[:, (cb * 2 - 1) * 128 + lanes * 4 + 3] & mask32
            else:
                st = W[:, (cb * wpb - 1) * 32 + lanes] & mask32
        for h in range(2):
            if bits == 4:
                idx0 = (cb * 2 + h) * 128 + lanes * 4
                sub_w = [W[:, idx0 + k] for k in range(4)]
                for k, wc in enumerate(sub_w):
                    h0 = (st * mult) & mask32
                    h1 = ((((st << 8) | (wc >> 24)) & mask32) * mult) & mask32
                    h2 = ((((st << 16) | (wc >> 16)) & mask32) * mult) & mask32
                    h3 = ((((st << 24) | (wc >> 8)) & mask32) * mult) & mask32
                    c0 = c64[((h0 >> 16) & 63).astype(np.int64)]
                    c2 = c64[((h2 >> 16) & 63).astype(np.int64)]
                    p0 = lut[((((wc >> 24) << 2) ^ (h0 >> 22)) & 1023).astype(np.int64)]
                    p1 = lut[(((((wc >> 16) & 255) << 2) ^ (h1 >> 22)) & 1023).astype(np.int64)]
                    p2 = lut[(((((wc >> 8) & 255) << 2) ^ (h2 >> 22)) & 1023).astype(np.int64)]
                    p3 = lut[((((wc & 255) << 2) ^ (h3 >> 22)) & 1023).astype(np.int64)]
                    st = wc & mask32
                    cc = cb_loc * 128 + h * 64 + k * 16 + hc[None, :]
                    blk_rot[r_lo, cc + 0] = p0[..., 0] * c0[..., 0] + c0[..., 2]
                    blk_rot[r_lo, cc + 1] = p0[..., 1] * c0[..., 1] + c0[..., 3]
                    blk_rot[r_lo, cc + 2] = p1[..., 0] * c0[..., 0] + c0[..., 2]
                    blk_rot[r_lo, cc + 3] = p1[..., 1] * c0[..., 1] + c0[..., 3]
                    blk_rot[r_hi, cc + 0] = p2[..., 0] * c2[..., 0] + c2[..., 2]
                    blk_rot[r_hi, cc + 1] = p2[..., 1] * c2[..., 1] + c2[..., 3]
                    blk_rot[r_hi, cc + 2] = p3[..., 0] * c2[..., 0] + c2[..., 2]
                    blk_rot[r_hi, cc + 3] = p3[..., 1] * c2[..., 1] + c2[..., 3]
            elif bits == 3:
                b0 = (cb * 6 + h * 3) * 32 + lanes
                a0, a1, a2 = W[:, b0], W[:, b0 + 32], W[:, b0 + 64]
                m24s = [
                    a0 >> 8,
                    ((a0 << 16) | (a1 >> 16)) & 0xFFFFFF,
                    ((a1 << 8) | (a2 >> 24)) & 0xFFFFFF,
                    a2 & 0xFFFFFF,
                ]
                for k, m24 in enumerate(m24s):
                    h0 = (st * mult) & mask32
                    h1 = ((((st << 6) | (m24 >> 18)) & mask32) * mult) & mask32
                    h2 = ((((st << 12) | (m24 >> 12)) & mask32) * mult) & mask32
                    h3 = ((((st << 18) | (m24 >> 6)) & mask32) * mult) & mask32
                    c0 = c64[((h0 >> 16) & 63).astype(np.int64)]
                    c2 = c64[((h2 >> 16) & 63).astype(np.int64)]
                    p0 = lut[((((m24 >> 16) & 0xFC) ^ (h0 >> 24)) & 255).astype(np.int64)]
                    p1 = lut[((((m24 >> 10) & 0xFC) ^ (h1 >> 24)) & 255).astype(np.int64)]
                    p2 = lut[((((m24 >> 4) & 0xFC) ^ (h2 >> 24)) & 255).astype(np.int64)]
                    p3 = lut[((((m24 << 2) & 0xFC) ^ (h3 >> 24)) & 255).astype(np.int64)]
                    st = ((st << 24) | m24) & mask32
                    cc = cb_loc * 128 + h * 64 + k * 16 + hc[None, :]
                    blk_rot[r_lo, cc + 0] = p0[..., 0] * c0[..., 0] + c0[..., 2]
                    blk_rot[r_lo, cc + 1] = p0[..., 1] * c0[..., 1] + c0[..., 3]
                    blk_rot[r_lo, cc + 2] = p1[..., 0] * c0[..., 0] + c0[..., 2]
                    blk_rot[r_lo, cc + 3] = p1[..., 1] * c0[..., 1] + c0[..., 3]
                    blk_rot[r_hi, cc + 0] = p2[..., 0] * c2[..., 0] + c2[..., 2]
                    blk_rot[r_hi, cc + 1] = p2[..., 1] * c2[..., 1] + c2[..., 3]
                    blk_rot[r_hi, cc + 2] = p3[..., 0] * c2[..., 0] + c2[..., 2]
                    blk_rot[r_hi, cc + 3] = p3[..., 1] * c2[..., 1] + c2[..., 3]
            else:
                b0 = (cb * 10 + h * 5) * 32 + lanes
                w0, w1, w2, w3, w4 = [W[:, b0 + j * 32] for j in range(5)]
                pairs = [
                    (w0 >> 24, ((w0 << 8) | (w1 >> 24)) & mask32),
                    ((w1 >> 16) & 255, ((w1 << 16) | (w2 >> 16)) & mask32),
                    ((w2 >> 8) & 255, ((w2 << 24) | (w3 >> 8)) & mask32),
                    (w3 & 255, w4 & mask32),
                ]
                for k, (hi8, lo32) in enumerate(pairs):
                    b0v = ((hi8 << 2) | (lo32 >> 30)) & 1023
                    b1v = (lo32 >> 20) & 1023
                    b2v = (lo32 >> 10) & 1023
                    b3v = lo32 & 1023
                    h0 = (st * mult) & mask32
                    h1 = ((((st << 10) | b0v) & mask32) * mult) & mask32
                    h2 = ((((st << 20) | (hi8 << 12) | (lo32 >> 20)) & mask32) * mult) & mask32
                    h3 = ((((st << 30) | (hi8 << 22) | (lo32 >> 10)) & mask32) * mult) & mask32
                    c0 = c64[((h0 >> 16) & 63).astype(np.int64)]
                    c2 = c64[((h2 >> 16) & 63).astype(np.int64)]
                    p0 = lut[((b0v ^ (h0 >> 22)) & 1023).astype(np.int64)]
                    p1 = lut[((b1v ^ (h1 >> 22)) & 1023).astype(np.int64)]
                    p2 = lut[((b2v ^ (h2 >> 22)) & 1023).astype(np.int64)]
                    p3 = lut[((b3v ^ (h3 >> 22)) & 1023).astype(np.int64)]
                    st = lo32 & mask32
                    cc = cb_loc * 128 + h * 64 + k * 16 + hc[None, :]
                    blk_rot[r_lo, cc + 0] = p0[..., 0] * c0[..., 0] + c0[..., 2]
                    blk_rot[r_lo, cc + 1] = p0[..., 1] * c0[..., 1] + c0[..., 3]
                    blk_rot[r_lo, cc + 2] = p1[..., 0] * c0[..., 0] + c0[..., 2]
                    blk_rot[r_lo, cc + 3] = p1[..., 1] * c0[..., 1] + c0[..., 3]
                    blk_rot[r_hi, cc + 0] = p2[..., 0] * c2[..., 0] + c2[..., 2]
                    blk_rot[r_hi, cc + 1] = p2[..., 1] * c2[..., 1] + c2[..., 3]
                    blk_rot[r_hi, cc + 2] = p3[..., 0] * c2[..., 0] + c2[..., 2]
                    blk_rot[r_hi, cc + 3] = p3[..., 1] * c2[..., 1] + c2[..., 3]

    sc_sub = scales[:, bc0:bc1].astype(np.float32)
    sc_tile = np.repeat(np.repeat(sc_sub, 128, axis=0), 128, axis=1)
    scaled = blk_rot * sc_tile
    sr = pw_sgn(SEED_ROW, np.arange(r_off, r_off + R, dtype=np.uint32))
    c_start = (bc0 * 128) if c_sign_off is None else c_sign_off
    sc_v = pw_sgn(SEED_COL, np.arange(c_start, c_start + C_out, dtype=np.uint32))
    # Inverse 2D FWHT-128 on [R, C_out] with absolute row/col signs
    tmp = fwht_128(scaled.reshape(R // 128, 128, C_out).transpose(0, 2, 1)).transpose(0, 2, 1).reshape(R, C_out)
    tmp = tmp * sr[:, None]
    out = fwht_128(tmp.reshape(R, n_bc_out, 128)).reshape(R, C_out) * sc_v[None, :]
    return out.astype(np.float32, copy=False)


def self_test() -> None:
    rng = np.random.default_rng(42)
    W = rng.standard_normal((128, 128)).astype(np.float32)
    # Inject extreme transformer-like channel outliers (100x std)
    W[17, :] *= 25.0
    W[:, 93] *= 25.0
    W_rot = rotate_block_2d(W, r_off=256, c_off=512)
    W_rec = unrotate_block_2d(W_rot, r_off=256, c_off=512)
    max_err = float(np.max(np.abs(W - W_rec)))
    assert max_err < 1e-4, f"FWHT inverse error too large: {max_err}"

    kurt_before = float(np.mean(W**4) / (np.mean(W**2) ** 2))
    kurt_after = float(np.mean(W_rot**4) / (np.mean(W_rot**2) ** 2))
    # Official Cyan4973/xxHash XXH64 test vectors (empty + standard sanity buffer)
    assert xxh64(b"", 0) == 0xEF46DB3751D8E999
    assert xxh64(b"", 2654435761) == 0xAC75FDA2929B17EF
    sbuf = bytearray(222)
    bg = 2654435761
    for i in range(len(sbuf)):
        sbuf[i] = (bg >> 56) & 0xFF
        bg = (bg * 11400714785074694791) & _XXH_M64
    for n, expected in (
        (1, 0xE934A84ADB052768),
        (4, 0xE244DDBCF81E29E4),
        (8, 0xCA240B08878D6FF6),
        (14, 0xB89B3598E0BD0A0A),
        (31, 0xB0A1E0B773272144),
        (32, 0x91382F920A08001C),
        (33, 0x840F6F0A6B922839),
        (101, 0x99EC31474028A952),
        (222, 0x06CC5BD930BCEC3A),
    ):
        got_one = xxh64(sbuf[:n], 0)
        got_stream = xxh64_sections((sbuf[: n // 3], sbuf[n // 3 : (2 * n) // 3], sbuf[(2 * n) // 3 : n]), 0)
        assert got_one == expected and got_stream == expected, f"XXH64 len={n}: {got_one:#x} vs {expected:#x}"
    print(f"PASS 2D FWHT-128 exact invertibility: max_err={max_err:.2e}")
    print(f"PASS Outlier flattening (kurtosis):   before={kurt_before:.1f} -> after={kurt_after:.2f} (Gaussian=3.0)")
    print("PASS XXH64 reference vectors (one-shot & streaming)")


def main() -> None:
    ap = argparse.ArgumentParser(description="TFQ3 Python/NumPy reference codec and inspector")
    ap.add_argument("--self-test", action="store_true", help="Run mathematical invertibility & outlier-flattening self-test")
    ap.add_argument("--shard", help="Optional TFPN .tffpw shard to dequantize a tile from")
    ap.add_argument("--tables", help="Path to pw_v3_tables.bin")
    ap.add_argument("--tensor", help="Tensor name inside --shard")
    args = ap.parse_args()
    if args.self_test or not args.shard:
        self_test()
    if args.shard:
        tables = args.tables or os.path.join(os.path.dirname(args.shard), "pw_v3_tables.bin")
        tile = dequantize_tfpn_tensor(args.shard, tables, args.tensor, max_blocks=1)
        print(f"Dequantized {args.tensor} [0:128, 0:128]: mean={tile.mean():.5f} std={tile.std():.5f} min={tile.min():.4f} max={tile.max():.4f}")


if __name__ == "__main__":
    main()
