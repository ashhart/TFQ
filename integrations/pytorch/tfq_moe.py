"""
Standalone PyTorch loader for TFQ3 `TFPW` v3 MoE checkpoints (zero TensorFold runtime dependency).

Supports both major MoE checkpoint layouts produced by TFQ3 converters:
  - Inkling-2 (`inkling2` / `inkling2_loop`): `layer_00..23.tffpw` or `layer_00..47.tffpw`
    + `pw_v3_tables.bin` + `residual-*.safetensors` / `model.safetensors.index.json`
    + optional `nonmoe.tffpw` + optional standalone `mtp.safetensors`.
  - GLM (`glm5_next` / `glm_moe_dsa`): `layer_03..44.tffpw` (or higher) + `pw_v3_tables.bin`,
    either WITH `nonmoe.tffpw` OR WITHOUT `nonmoe.tffpw` (`--keep-nonmoe-bf16` BF16-residual mode).

Features:
  - Strict `TFPW` v3 / `TFPN` v3 / `PWV3` header, alignment, bounds, and `XXH64` integrity checks
    (accepts legacy `flag == 0` shards; rejects corrupted `meta`, `scales`, or `weights` bytes).
  - Lazy read-only `mmap` access: opening a checkpoint maps files with near-zero RAM footprint and
    decodes individual routed experts on demand via `reference/tfq3_reference.py:decode_tfpn_fast_np`.
  - Exact PyTorch SwiGLU routed expert forward (`forward_expert` & `forward_routed`), supporting
    standard SwiGLU, GLM clamped SwiGLU (`swiglu_limit`), and Inkling-2 softcapped SwiGLU
    (`gate_softcap`, `up_softcap`).
"""
from __future__ import annotations

from dataclasses import dataclass
import glob
import json
import mmap
import os
import re
import struct
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_REF_DIR = os.path.join(REPO_ROOT, "reference")
if _REF_DIR not in sys.path:
    sys.path.insert(0, _REF_DIR)

import tfq3_reference as tfq3_ref

try:
    from . import tfq_torch as _tfq_torch  # type: ignore
except ImportError:
    _PT_DIR = os.path.dirname(os.path.abspath(__file__))
    if _PT_DIR not in sys.path:
        sys.path.insert(0, _PT_DIR)
    import tfq_torch as _tfq_torch  # type: ignore

ALIGN_16K = 16384
TFPW_HEADER_BYTES = 144
TFPW_EXPERT_META_BYTES = 64
TFPN_HEADER_BYTES = 64
TFPN_TENSOR_META_BYTES = 128
PWV3_MIN_BYTES = 10816

_LAYER_FILE_RE = re.compile(r"^layer_(\d+)\.tffpw$")


def _get_cuda_ext():
    return _tfq_torch._get_cuda_ext()


def _get_mps_mod():
    return _tfq_torch._get_mps_mod()


def _bits_to_bpw(bits: int) -> float:
    return 2.25 if bits == 225 else float(bits)


def _pack_pwv3_tables_bytes(tbl: dict[str, Any]) -> bytes:
    raw = tbl.get("_raw_bytes")
    if isinstance(raw, (bytes, bytearray)) and len(raw) >= PWV3_MIN_BYTES:
        return bytes(raw)
    buf = bytearray(16384)
    struct.pack_into(
        "<II6f",
        buf,
        0,
        tfq3_ref.MAGIC_PWV3,
        tfq3_ref.FORMAT_VER,
        float(tbl["a"][3]),
        float(tbl["b"][3]),
        float(tbl["a"][4]),
        float(tbl["b"][4]),
        float(tbl["a"][5]),
        float(tbl["b"][5]),
    )
    buf[64 : 64 + 256 * 4] = np.ascontiguousarray(tbl["lut"][3], dtype=np.float16).tobytes()
    buf[1088 : 1088 + 1024 * 4] = np.ascontiguousarray(tbl["lut"][4], dtype=np.float16).tobytes()
    buf[5184 : 5184 + 1024 * 4] = np.ascontiguousarray(tbl["lut"][5], dtype=np.float16).tobytes()
    buf[9280 : 9280 + 64 * 8] = np.ascontiguousarray(tbl["c64"][3], dtype=np.float16).tobytes()
    buf[9792 : 9792 + 64 * 8] = np.ascontiguousarray(tbl["c64"][4], dtype=np.float16).tobytes()
    buf[10304 : 10304 + 64 * 8] = np.ascontiguousarray(tbl["c64"][5], dtype=np.float16).tobytes()
    return bytes(buf)


def load_and_validate_pwv4_tables(tables_path: str | os.PathLike[str]) -> tuple[np.ndarray, int, bytes]:
    """Load and validate `pw_v4_e8p_tables.bin` (16 KiB kind 0/1/2 or 32 KiB kind 3 E8P-1024)."""
    return _tfq_torch.load_pwv4_tables_auto(tables_path)


def _normalize_tables_v4(
    tables_v4: Any,
) -> tuple[np.ndarray, int, bytes | None] | None:
    if tables_v4 is None:
        return None
    if isinstance(tables_v4, (str, os.PathLike)):
        return load_and_validate_pwv4_tables(tables_v4)
    if isinstance(tables_v4, dict):
        lut = np.asarray(tables_v4["lut"], dtype=np.float32)
        kind = int(tables_v4["table_kind"])
        raw = tables_v4.get("raw_bytes")
        return lut, kind, bytes(raw) if raw is not None else None
    if isinstance(tables_v4, tuple):
        if len(tables_v4) == 2:
            return np.asarray(tables_v4[0], dtype=np.float32), int(tables_v4[1]), None
        if len(tables_v4) == 3:
            raw = tables_v4[2]
            return np.asarray(tables_v4[0], dtype=np.float32), int(tables_v4[1]), bytes(raw) if raw is not None else None
    raise TypeError(f"Unsupported tables_v4 value of type {type(tables_v4).__name__}")


@dataclass(frozen=True)
class TfpwHeader:
    magic: int
    version: int
    layer_idx: int
    num_experts: int
    num_tensors: int
    hidden_size: int
    moe_inter: int
    flags: int
    raw_fp8_bytes: int
    stream_words: int
    scale_halfs: int
    off_meta: int
    off_scales: int
    off_weights: int
    checksum_meta: int
    checksum_scales: int
    checksum_weights: int
    checksum_reserved: int
    meta_tag: str

    @property
    def has_xxh64(self) -> bool:
        return (self.flags & tfq3_ref.FLAG_XXH64_PRESENT) != 0


@dataclass(frozen=True)
class TfpwExpertMeta:
    expert_idx: int
    b_bits: int
    heat_rank: int
    gate_word_off: int
    up_word_off: int
    down_word_off: int
    gate_sc_off: int
    up_sc_off: int
    down_sc_off: int
    gate_snr_db: float
    up_snr_db: float
    down_snr_db: float


@dataclass(frozen=True)
class TfpnRecord:
    name: str
    R: int
    C: int
    bits: int
    sc_off: int
    word_off: int
    word_count: int
    snr_db: float
    orig_R: int
    orig_C: int
    flags: int


def _mat_words(R: int, C: int, bits: int) -> int:
    if bits in (3, 4, 5):
        return (R // 16) * (C // 128) * (bits * 2) * 32
    if bits == 2:
        return R * (C // 16)
    if bits == 225:
        return R * (C // 128) * 9
    raise ValueError(f"Unsupported expert bit width {bits} (expected 2, 225, 3, 4, or 5)")


def _mat_scales(R: int, C: int) -> int:
    return (R // 128) * (C // 128)


def load_and_validate_tables(tables_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load and validate `pw_v3_tables.bin` (magic, version, minimum active length, finite values)."""
    path = os.fspath(tables_path)
    sz = os.path.getsize(path)
    if sz < PWV3_MIN_BYTES:
        raise ValueError(f"Truncated pw_v3_tables.bin ({sz} bytes < {PWV3_MIN_BYTES}): {path}")
    with open(path, "rb") as f:
        raw_bytes = f.read()
    tbl = tfq3_ref.load_tables(path)
    for bits in (3, 4, 5):
        if not np.isfinite(tbl["a"][bits]) or not np.isfinite(tbl["b"][bits]):
            raise ValueError(f"Non-finite affine hyperparameter in {path} for {bits}b")
        if not np.isfinite(tbl["lut"][bits].astype(np.float32)).all():
            raise ValueError(f"Non-finite LUT entry in {path} for {bits}b")
        if not np.isfinite(tbl["c64"][bits].astype(np.float32)).all():
            raise ValueError(f"Non-finite c64 entry in {path} for {bits}b")
    tbl["_raw_bytes"] = raw_bytes
    return tbl


def _read_safetensors_header(path: str) -> tuple[int, dict[str, Any]]:
    file_size = os.path.getsize(path)
    if file_size < 8:
        raise ValueError(f"Truncated safetensors file: {path}")
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        if hlen == 0 or 8 + hlen > file_size:
            raise ValueError(f"Invalid safetensors header length ({hlen}) in {path}")
        hdr = json.loads(f.read(hlen).decode("utf-8"))
    if not isinstance(hdr, dict):
        raise ValueError(f"Invalid safetensors header object in {path}")
    return 8 + hlen, hdr


def _read_safetensors_tensor(path: str, name: str, device: str | torch.device = "cpu") -> torch.Tensor:
    """Lazily read a single tensor from a safetensors file without loading the rest of the shard."""
    file_size = os.path.getsize(path)
    base, hdr = _read_safetensors_header(path)
    if name not in hdr or name == "__metadata__":
        raise KeyError(f"Tensor '{name}' not found in {path}")
    meta = hdr[name]
    dt = meta["dtype"]
    shape = [int(d) for d in meta["shape"]]
    o0, o1 = int(meta["data_offsets"][0]), int(meta["data_offsets"][1])
    if o0 < 0 or o1 < o0 or base + o1 > file_size:
        raise ValueError(f"Out-of-bounds safetensors slice [{o0}, {o1}) for '{name}' in {path}")
    with open(path, "rb") as f:
        f.seek(base + o0)
        raw = f.read(o1 - o0)

    if dt == "BF16":
        t = torch.from_numpy(np.frombuffer(raw, dtype=np.uint16).copy()).view(torch.bfloat16).reshape(shape)
    elif dt == "F16":
        t = torch.from_numpy(np.frombuffer(raw, dtype=np.float16).copy()).reshape(shape)
    elif dt == "F32":
        t = torch.from_numpy(np.frombuffer(raw, dtype=np.float32).copy()).reshape(shape)
    elif dt == "F64":
        t = torch.from_numpy(np.frombuffer(raw, dtype=np.float64).copy()).reshape(shape)
    elif dt == "U8":
        t = torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).copy()).reshape(shape)
    elif dt == "F8_E4M3":
        arr = np.frombuffer(raw, dtype=np.uint8).copy()
        t = torch.from_numpy(arr).reshape(shape)
        if hasattr(torch, "float8_e4m3fn"):
            t = t.view(torch.float8_e4m3fn)
    elif dt == "I32":
        t = torch.from_numpy(np.frombuffer(raw, dtype=np.int32).copy()).reshape(shape)
    elif dt == "I64":
        t = torch.from_numpy(np.frombuffer(raw, dtype=np.int64).copy()).reshape(shape)
    else:
        raise ValueError(f"Unsupported safetensors dtype '{dt}' for '{name}' in {path}")
    return t.to(device=device)


class TfpnNonMoeShard:
    """Read-only memory-mapped reader for optional `nonmoe.tffpw` (`TFPN` v3) dense shards."""

    def __init__(
        self,
        shard_path: str | os.PathLike[str],
        tables: dict[str, Any] | str | os.PathLike[str],
        *,
        verify_checksum: bool = True,
    ) -> None:
        self.path = os.fspath(shard_path)
        self._tables = load_and_validate_tables(tables) if isinstance(tables, (str, os.PathLike)) else tables
        self._file = open(self.path, "rb")
        self._mm: mmap.mmap | None = None
        self._scales: np.ndarray | None = None
        self._words: np.ndarray | None = None
        try:
            file_size = os.fstat(self._file.fileno()).st_size
            if file_size < TFPN_HEADER_BYTES:
                raise ValueError(f"Truncated TFPN header in {self.path}")
            self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
            mv = memoryview(self._mm)
            try:
                (
                    magic, ver, nt, pad0,
                    off_meta, off_scales, off_weights,
                    tot_sc, tot_w, pad1,
                ) = struct.unpack_from("<IIIIQQQQQQ", mv, 0)
                if magic != tfq3_ref.MAGIC_TFPN or ver != tfq3_ref.FORMAT_VER:
                    raise ValueError(f"Bad TFPN header in {self.path}: magic=0x{magic:08x} ver={ver}")
                if off_scales % ALIGN_16K != 0 or off_weights % ALIGN_16K != 0:
                    raise ValueError(
                        f"Misaligned TFPN sections in {self.path}: off_scales={off_scales}, off_weights={off_weights}"
                    )
                meta_bytes = nt * TFPN_TENSOR_META_BYTES
                sc_bytes = tot_sc * 2
                w_bytes = tot_w * 4
                if (
                    off_meta < TFPN_HEADER_BYTES
                    or off_meta + meta_bytes > off_scales
                    or off_scales + sc_bytes > off_weights
                    or off_weights + w_bytes > file_size
                ):
                    raise ValueError(f"Invalid or overlapping TFPN sections in {self.path}")

                self.checksum_verified = False
                if verify_checksum and (pad0 & tfq3_ref.FLAG_XXH64_PRESENT) != 0:
                    got = tfq3_ref.xxh64_sections((
                        mv[off_meta : off_meta + meta_bytes],
                        mv[off_scales : off_scales + sc_bytes],
                        mv[off_weights : off_weights + w_bytes],
                    ))
                    if got != pad1:
                        raise ValueError(
                            f"TFPN XXH64 checksum mismatch in {self.path}: got 0x{got:016x}, expected 0x{pad1:016x}"
                        )
                    self.checksum_verified = True

                self.records: dict[str, TfpnRecord] = {}
                for i in range(nt):
                    rec_off = off_meta + i * TFPN_TENSOR_META_BYTES
                    raw_name = bytes(mv[rec_off : rec_off + 80])
                    name = raw_name.split(b"\x00", 1)[0].decode("utf-8")
                    if not name:
                        raise ValueError(f"Empty TFPN tensor name at index {i} in {self.path}")
                    R, C, bits, sc_off, w_off, w_cnt, snr_db, orig_R, orig_C, rflags = struct.unpack_from(
                        "<IIIIQQfIII", mv, rec_off + 80
                    )
                    if bits not in (3, 4, 5):
                        raise ValueError(f"Invalid TFPN bit width {bits} for '{name}' in {self.path}")
                    if R == 0 or C == 0 or R % 128 != 0 or C % 128 != 0:
                        raise ValueError(f"Invalid TFPN shape ({R}, {C}) for '{name}' in {self.path}")
                    exp_w = _mat_words(R, C, bits)
                    exp_sc = _mat_scales(R, C)
                    if w_cnt != exp_w:
                        raise ValueError(f"Word count mismatch for '{name}' in {self.path}: {w_cnt} != {exp_w}")
                    if bits == 4 and (w_off % 4) != 0:
                        raise ValueError(f"Misaligned 4-bit word_off={w_off} for '{name}' in {self.path}")
                    if w_off + w_cnt > tot_w or sc_off + exp_sc > tot_sc:
                        raise ValueError(f"Out-of-bounds offsets for '{name}' in {self.path}")
                    if name in self.records:
                        raise ValueError(f"Duplicate TFPN tensor '{name}' in {self.path}")
                    self.records[name] = TfpnRecord(
                        name=name,
                        R=R,
                        C=C,
                        bits=bits,
                        sc_off=sc_off,
                        word_off=w_off,
                        word_count=w_cnt,
                        snr_db=snr_db,
                        orig_R=orig_R or R,
                        orig_C=orig_C or C,
                        flags=rflags,
                    )
            finally:
                mv.release()

            self.file_size = file_size
            self._scales = np.frombuffer(self._mm, dtype=np.float16, count=tot_sc, offset=off_scales)
            self._words = np.frombuffer(self._mm, dtype=np.uint32, count=tot_w, offset=off_weights)
        except Exception:
            self.close()
            raise

    def tensor_names(self) -> list[str]:
        return list(self.records.keys())

    def decode_tensor(
        self,
        name: str,
        *,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        if self._mm is None or self._scales is None or self._words is None:
            raise RuntimeError("TfpnNonMoeShard is closed")
        rec = self.records[name]
        n_sc = _mat_scales(rec.R, rec.C)
        sc = self._scales[rec.sc_off : rec.sc_off + n_sc].reshape(rec.R // 128, rec.C // 128)
        wd = self._words[rec.word_off : rec.word_off + rec.word_count]
        w_np = tfq3_ref.decode_tfpn_fast_np(wd, sc, rec.R, rec.C, rec.bits, self._tables)[: rec.orig_R, : rec.orig_C]
        return torch.from_numpy(np.ascontiguousarray(w_np)).to(device=device, dtype=dtype)

    def close(self) -> None:
        self._scales = None
        self._words = None
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> TfpnNonMoeShard:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class TfpwLayer:
    """
    Lazy memory-mapped reader and PyTorch executor for a single `TFPW` v3 MoE layer (`layer_XX.tffpw`).

    Supports both pure TFQ3 (`3/4/5-bit`) and hybrid TFQ3+TFQ4 (`2/225/3/4/5-bit`) MoE layers:
      - `gate`: `[moe_inter, hidden_size]`
      - `up`:   `[moe_inter, hidden_size]`
      - `down`: `[hidden_size, moe_inter]`

    On CUDA with `tfq3_cuda_ext` installed, `forward_routed()` dispatches `{3, 4, 5}`-bit routed
    experts to the fused `tfq3_moe` kernel (including Inkling-2 tanh softcap) and `{2, 225}`-bit
    experts via `tfq4_linear` / exact decode fallback.
    """

    def __init__(
        self,
        shard_path: str | os.PathLike[str],
        tables: dict[str, Any] | str | os.PathLike[str] | None = None,
        *,
        verify_checksum: bool = True,
        expected_layer_idx: int | None = None,
        cache_experts: bool = False,
        tables_v4: Any = None,
    ) -> None:
        self.path = os.fspath(shard_path)
        self._tables: dict[str, Any] | None = (
            load_and_validate_tables(tables) if isinstance(tables, (str, os.PathLike)) else tables
        )
        self._tables_bytes: bytes | None = (
            _pack_pwv3_tables_bytes(self._tables) if self._tables is not None else None
        )
        v4_norm = _normalize_tables_v4(tables_v4)
        self._tfq4_lut: np.ndarray | None = v4_norm[0] if v4_norm is not None else None
        self._tfq4_table_kind: int | None = v4_norm[1] if v4_norm is not None else None
        self._tfq4_raw_bytes: bytes | None = v4_norm[2] if v4_norm is not None else None

        self.cache_experts = bool(cache_experts)
        self._expert_cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._cuda_moe_cache: dict[torch.device, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._tfq4_lut_dev_cache: dict[tuple[torch.device, int], torch.Tensor] = {}
        self._meta_bytes: bytes = b""
        self._file = open(self.path, "rb")
        self._mm: mmap.mmap | None = None
        self._scales: np.ndarray | None = None
        self._words: np.ndarray | None = None

        try:
            file_size = os.fstat(self._file.fileno()).st_size
            if file_size < TFPW_HEADER_BYTES:
                raise ValueError(f"Truncated TFPW header ({file_size} bytes < {TFPW_HEADER_BYTES}) in {self.path}")
            self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
            mv = memoryview(self._mm)
            try:
                magic, ver, layer_idx, n_exp, n_ten, hidden, inter, reserved0 = struct.unpack_from("<8I", mv, 0)
                if magic != tfq3_ref.MAGIC_TFPW or ver != tfq3_ref.FORMAT_VER:
                    raise ValueError(f"Bad TFPW header in {self.path}: magic=0x{magic:08x} ver={ver}")
                if expected_layer_idx is not None and layer_idx != expected_layer_idx:
                    raise ValueError(
                        f"TFPW layer index mismatch in {self.path}: header has {layer_idx}, expected {expected_layer_idx}"
                    )
                if n_exp == 0 or n_ten != 3 * n_exp:
                    raise ValueError(f"Invalid TFPW expert/tensor counts ({n_exp}, {n_ten}) in {self.path}")
                if hidden == 0 or inter == 0 or (hidden % 128) != 0 or (inter % 128) != 0:
                    raise ValueError(f"Invalid TFPW geometry hidden={hidden}, moe_inter={inter} in {self.path}")

                raw_fp8, stream_words, scale_halfs, off_meta, off_scales, off_weights = struct.unpack_from("<6Q", mv, 32)
                c_meta, c_scales, c_weights, c_res = struct.unpack_from("<4Q", mv, 80)
                meta_tag = bytes(mv[112:144]).split(b"\x00", 1)[0].decode("utf-8", errors="replace")

                if (off_scales % ALIGN_16K) != 0 or (off_weights % ALIGN_16K) != 0:
                    raise ValueError(
                        f"Misaligned TFPW sections in {self.path}: off_scales={off_scales}, off_weights={off_weights}"
                    )
                meta_bytes = n_exp * TFPW_EXPERT_META_BYTES
                scales_bytes = scale_halfs * 2
                weights_bytes = stream_words * 4
                if (
                    off_meta < TFPW_HEADER_BYTES
                    or off_meta + meta_bytes > off_scales
                    or off_scales + scales_bytes > off_weights
                    or off_weights + weights_bytes > file_size
                ):
                    raise ValueError(f"Invalid or overlapping TFPW sections in {self.path}")

                self.header = TfpwHeader(
                    magic=magic,
                    version=ver,
                    layer_idx=layer_idx,
                    num_experts=n_exp,
                    num_tensors=n_ten,
                    hidden_size=hidden,
                    moe_inter=inter,
                    flags=reserved0,
                    raw_fp8_bytes=raw_fp8,
                    stream_words=stream_words,
                    scale_halfs=scale_halfs,
                    off_meta=off_meta,
                    off_scales=off_scales,
                    off_weights=off_weights,
                    checksum_meta=c_meta,
                    checksum_scales=c_scales,
                    checksum_weights=c_weights,
                    checksum_reserved=c_res,
                    meta_tag=meta_tag,
                )

                self.checksum_verified = False
                if verify_checksum and self.header.has_xxh64:
                    for sec_name, off, length, expected in (
                        ("meta", off_meta, meta_bytes, c_meta),
                        ("scales", off_scales, scales_bytes, c_scales),
                        ("weights", off_weights, weights_bytes, c_weights),
                    ):
                        got = tfq3_ref.xxh64(mv[off : off + length], 0)
                        if got != expected:
                            raise ValueError(
                                f"TFPW {sec_name} XXH64 checksum mismatch in {self.path}: "
                                f"got 0x{got:016x}, expected 0x{expected:016x}"
                            )
                    self.checksum_verified = True

                self._meta_bytes = bytes(mv[off_meta : off_meta + meta_bytes])

                n_sc_gu = _mat_scales(inter, hidden)
                n_sc_dn = _mat_scales(hidden, inter)
                experts: list[TfpwExpertMeta] = []
                need_v3 = False
                need_v4 = False
                need_v4_1024 = False
                for e in range(n_exp):
                    e_off = off_meta + e * TFPW_EXPERT_META_BYTES
                    b_bits, hrank, gw, uw, dw, gs, us, ds, gsnr, usnr, dsnr, _, _ = struct.unpack_from(
                        "<IIQQQIIIfffII", mv, e_off
                    )
                    if b_bits not in (2, 225, 3, 4, 5):
                        raise ValueError(f"Invalid expert bit width {b_bits} for L{layer_idx} E{e} in {self.path}")
                    if b_bits in (3, 4, 5):
                        need_v3 = True
                    else:
                        need_v4 = True
                        if b_bits == 225:
                            need_v4_1024 = True
                    if b_bits == 4 and ((gw % 4) != 0 or (uw % 4) != 0 or (dw % 4) != 0):
                        raise ValueError(f"Misaligned 4-bit expert word offset for L{layer_idx} E{e} in {self.path}")
                    gu_words = _mat_words(inter, hidden, b_bits)
                    dn_words = _mat_words(hidden, inter, b_bits)
                    if (
                        gw + gu_words > stream_words
                        or uw + gu_words > stream_words
                        or dw + dn_words > stream_words
                        or gs + n_sc_gu > scale_halfs
                        or us + n_sc_gu > scale_halfs
                        or ds + n_sc_dn > scale_halfs
                    ):
                        raise ValueError(f"Out-of-bounds expert offsets for L{layer_idx} E{e} in {self.path}")
                    experts.append(
                        TfpwExpertMeta(
                            expert_idx=e,
                            b_bits=b_bits,
                            heat_rank=hrank,
                            gate_word_off=gw,
                            up_word_off=uw,
                            down_word_off=dw,
                            gate_sc_off=gs,
                            up_sc_off=us,
                            down_sc_off=ds,
                            gate_snr_db=gsnr,
                            up_snr_db=usnr,
                            down_snr_db=dsnr,
                        )
                    )

                # Auto-load sibling table binaries when required by expert bit widths
                shard_dir = os.path.dirname(os.path.abspath(self.path))
                if need_v3 and self._tables is None:
                    cand_v3 = os.path.join(shard_dir, "pw_v3_tables.bin")
                    if os.path.isfile(cand_v3):
                        self._tables = load_and_validate_tables(cand_v3)
                        self._tables_bytes = _pack_pwv3_tables_bytes(self._tables)
                    else:
                        raise ValueError(
                            f"TFPW layer {self.path} contains 3/4/5-bit experts but no pw_v3_tables.bin was provided"
                        )
                if need_v4 and self._tfq4_lut is None:
                    cand_v4 = os.path.join(shard_dir, "pw_v4_e8p_tables.bin")
                    if os.path.isfile(cand_v4):
                        self._tfq4_lut, self._tfq4_table_kind, self._tfq4_raw_bytes = load_and_validate_pwv4_tables(cand_v4)
                    else:
                        raise ValueError(
                            f"TFPW layer {self.path} contains TFQ4 (2/225-bit) experts but no pw_v4_e8p_tables.bin was provided"
                        )
                if need_v4_1024 and self._tfq4_table_kind != _tfq_torch.PWV4_TABLE_TRUE_E8P_1024:
                    raise ValueError(
                        f"TFPW layer {self.path} contains 2.25 bpw (b_bits=225) experts but pw_v4_e8p_tables.bin "
                        f"has table_kind={self._tfq4_table_kind} (expected 3)"
                    )

                self.experts = experts
                self.is_hybrid = need_v3 and need_v4
                self.has_tfq4_experts = need_v4
                self.has_tfq3_experts = need_v3
            finally:
                mv.release()

            self.file_size = file_size
            self.layer_idx = layer_idx
            self.num_experts = n_exp
            self.hidden_size = hidden
            self.moe_inter = inter
            self._scales = np.frombuffer(self._mm, dtype=np.float16, count=scale_halfs, offset=off_scales)
            self._words = np.frombuffer(self._mm, dtype=np.uint32, count=stream_words, offset=off_weights)
        except Exception:
            self.close()
            raise

    def bit_histogram(self) -> dict[int, int]:
        hist: dict[int, int] = {}
        for em in self.experts:
            hist[em.b_bits] = hist.get(em.b_bits, 0) + 1
        return dict(sorted(hist.items()))

    def avg_bpw(self) -> float:
        return sum(_bits_to_bpw(em.b_bits) for em in self.experts) / float(self.num_experts)

    def _effective_tfq4_lut_kind(self, b_bits: int) -> tuple[np.ndarray, int]:
        if self._tfq4_lut is None or self._tfq4_table_kind is None:
            raise RuntimeError("TFQ4 tables not loaded for this TfpwLayer")
        if b_bits == 225:
            return self._tfq4_lut, _tfq_torch.PWV4_TABLE_TRUE_E8P_1024
        eff_kind = (
            _tfq_torch.PWV4_TABLE_TRUE_E8P
            if self._tfq4_table_kind == _tfq_torch.PWV4_TABLE_TRUE_E8P_1024
            else self._tfq4_table_kind
        )
        return self._tfq4_lut[:256], eff_kind

    def _get_cuda_moe_buffers(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cached = self._cuda_moe_cache.get(device)
        if cached is not None:
            return cached
        if self._words is None or self._scales is None or self._tables_bytes is None:
            raise RuntimeError("TfpwLayer buffers or PWV3 tables unavailable for CUDA MoE")
        w_t = torch.from_numpy(self._words.view(np.int32).copy()).to(device=device)
        s_t = torch.from_numpy(self._scales.copy()).to(device=device)
        m_t = torch.from_numpy(np.frombuffer(self._meta_bytes, dtype=np.uint8).copy()).to(device=device)
        t_t = torch.from_numpy(np.frombuffer(self._tables_bytes, dtype=np.uint8).copy()).to(device=device)
        cached = (w_t, s_t, m_t, t_t)
        self._cuda_moe_cache[device] = cached
        return cached

    def _get_tfq4_lut_tensor(self, device: torch.device, b_bits: int) -> tuple[torch.Tensor, int]:
        lut_np, eff_kind = self._effective_tfq4_lut_kind(b_bits)
        key = (device, eff_kind)
        t = self._tfq4_lut_dev_cache.get(key)
        if t is None:
            t = torch.from_numpy(lut_np.astype(np.float16).copy()).to(device=device)
            self._tfq4_lut_dev_cache[key] = t
        return t, eff_kind

    @staticmethod
    def _apply_glu(
        g: torch.Tensor,
        u: torch.Tensor,
        *,
        swiglu_limit: float | None = None,
        gate_softcap: float | None = None,
        up_softcap: float | None = None,
    ) -> torch.Tensor:
        if gate_softcap is not None or up_softcap is not None:
            g_cap = (gate_softcap * torch.tanh(g / gate_softcap)) if gate_softcap is not None else g
            u_cap = (up_softcap * torch.tanh(u / up_softcap)) if up_softcap is not None else u
            return g_cap * torch.sigmoid(g) * u_cap
        if swiglu_limit is not None and swiglu_limit > 0.0:
            return F.silu(g.clamp(max=swiglu_limit)) * u.clamp(min=-swiglu_limit, max=swiglu_limit)
        return F.silu(g) * u

    def decode_expert(
        self,
        expert_idx: int,
        *,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Lazily decode routed expert `expert_idx` directly from the memory-mapped `TFPW` shard.

        Supports both TFQ3 (`3/4/5-bit`) and TFQ4-E8P (`2`/`225`-bit) experts.
        Returns `(W_gate, W_up, W_down)` with shapes:
          - `W_gate`: `[moe_inter, hidden_size]`
          - `W_up`:   `[moe_inter, hidden_size]`
          - `W_down`: `[hidden_size, moe_inter]`
        """
        if self._mm is None or self._scales is None or self._words is None:
            raise RuntimeError("TfpwLayer is closed")
        if expert_idx < 0 or expert_idx >= self.num_experts:
            raise IndexError(f"expert_idx={expert_idx} out of range [0, {self.num_experts})")

        cached = self._expert_cache.get(expert_idx)
        if cached is not None:
            wg, wu, wd = cached
            return (
                wg.to(device=device, dtype=dtype),
                wu.to(device=device, dtype=dtype),
                wd.to(device=device, dtype=dtype),
            )

        em = self.experts[expert_idx]
        H, I, bits = self.hidden_size, self.moe_inter, em.b_bits
        gu_words = _mat_words(I, H, bits)
        dn_words = _mat_words(H, I, bits)
        n_sc_gu = _mat_scales(I, H)
        n_sc_dn = _mat_scales(H, I)

        if bits in (3, 4, 5):
            assert self._tables is not None
            wg_np = tfq3_ref.decode_tfpn_fast_np(
                self._words[em.gate_word_off : em.gate_word_off + gu_words],
                self._scales[em.gate_sc_off : em.gate_sc_off + n_sc_gu].reshape(I // 128, H // 128),
                I, H, bits, self._tables,
            )
            wu_np = tfq3_ref.decode_tfpn_fast_np(
                self._words[em.up_word_off : em.up_word_off + gu_words],
                self._scales[em.up_sc_off : em.up_sc_off + n_sc_gu].reshape(I // 128, H // 128),
                I, H, bits, self._tables,
            )
            wd_np = tfq3_ref.decode_tfpn_fast_np(
                self._words[em.down_word_off : em.down_word_off + dn_words],
                self._scales[em.down_sc_off : em.down_sc_off + n_sc_dn].reshape(H // 128, I // 128),
                H, I, bits, self._tables,
            )
        else:
            lut_np, eff_kind = self._effective_tfq4_lut_kind(bits)
            wg_np = _tfq_torch.decode_tfq4_words_np(
                self._words[em.gate_word_off : em.gate_word_off + gu_words],
                self._scales[em.gate_sc_off : em.gate_sc_off + n_sc_gu].reshape(I // 128, H // 128),
                I, H, lut_np, eff_kind,
            )
            wu_np = _tfq_torch.decode_tfq4_words_np(
                self._words[em.up_word_off : em.up_word_off + gu_words],
                self._scales[em.up_sc_off : em.up_sc_off + n_sc_gu].reshape(I // 128, H // 128),
                I, H, lut_np, eff_kind,
            )
            wd_np = _tfq_torch.decode_tfq4_words_np(
                self._words[em.down_word_off : em.down_word_off + dn_words],
                self._scales[em.down_sc_off : em.down_sc_off + n_sc_dn].reshape(H // 128, I // 128),
                H, I, lut_np, eff_kind,
            )

        wg_cpu = torch.from_numpy(np.ascontiguousarray(wg_np))
        wu_cpu = torch.from_numpy(np.ascontiguousarray(wu_np))
        wd_cpu = torch.from_numpy(np.ascontiguousarray(wd_np))
        if self.cache_experts:
            self._expert_cache[expert_idx] = (wg_cpu, wu_cpu, wd_cpu)

        return (
            wg_cpu.to(device=device, dtype=dtype),
            wu_cpu.to(device=device, dtype=dtype),
            wd_cpu.to(device=device, dtype=dtype),
        )

    def _forward_tfq4_expert_fused(
        self,
        x: torch.Tensor,
        em: TfpwExpertMeta,
        *,
        swiglu_limit: float | None = None,
        gate_softcap: float | None = None,
        up_softcap: float | None = None,
    ) -> torch.Tensor | None:
        """Zero-expansion CUDA or MPS fused execution for a single TFQ4 (2 / 225 bpw) expert."""
        assert self._words is not None and self._scales is not None
        H, I, bits = self.hidden_size, self.moe_inter, em.b_bits
        gu_words = _mat_words(I, H, bits)
        dn_words = _mat_words(H, I, bits)
        n_sc_gu = _mat_scales(I, H)
        n_sc_dn = _mat_scales(H, I)

        if x.is_cuda:
            ext = _get_cuda_ext()
            if ext is None or not hasattr(ext, "tfq4_linear"):
                return None
            lut_t, eff_kind = self._get_tfq4_lut_tensor(x.device, bits)
            if eff_kind > int(getattr(ext, "TFQ4_MAX_TABLE_KIND", 2)):
                return None
            wg = torch.from_numpy(self._words[em.gate_word_off : em.gate_word_off + gu_words].view(np.int32).copy()).to(x.device)
            wu = torch.from_numpy(self._words[em.up_word_off : em.up_word_off + gu_words].view(np.int32).copy()).to(x.device)
            wd = torch.from_numpy(self._words[em.down_word_off : em.down_word_off + dn_words].view(np.int32).copy()).to(x.device)
            sg = torch.from_numpy(self._scales[em.gate_sc_off : em.gate_sc_off + n_sc_gu].copy()).to(x.device)
            su = torch.from_numpy(self._scales[em.up_sc_off : em.up_sc_off + n_sc_gu].copy()).to(x.device)
            sd = torch.from_numpy(self._scales[em.down_sc_off : em.down_sc_off + n_sc_dn].copy()).to(x.device)
            x_bf = x.to(torch.bfloat16)
            g = ext.tfq4_linear(x_bf, wg, sg, lut_t, eff_kind, I, H, I).to(torch.float32)
            u = ext.tfq4_linear(x_bf, wu, su, lut_t, eff_kind, I, H, I).to(torch.float32)
            act = self._apply_glu(g, u, swiglu_limit=swiglu_limit, gate_softcap=gate_softcap, up_softcap=up_softcap).to(torch.bfloat16)
            y = ext.tfq4_linear(act, wd, sd, lut_t, eff_kind, H, I, H)
            return y.to(dtype=x.dtype)

        if x.device.type == "mps":
            mps = _get_mps_mod()
            if mps is None:
                return None
            lut_t, eff_kind = self._get_tfq4_lut_tensor(x.device, bits)
            wg = torch.from_numpy(self._words[em.gate_word_off : em.gate_word_off + gu_words].view(np.int32).copy()).to(x.device)
            wu = torch.from_numpy(self._words[em.up_word_off : em.up_word_off + gu_words].view(np.int32).copy()).to(x.device)
            wd = torch.from_numpy(self._words[em.down_word_off : em.down_word_off + dn_words].view(np.int32).copy()).to(x.device)
            sg = torch.from_numpy(self._scales[em.gate_sc_off : em.gate_sc_off + n_sc_gu].copy()).to(x.device)
            su = torch.from_numpy(self._scales[em.up_sc_off : em.up_sc_off + n_sc_gu].copy()).to(x.device)
            sd = torch.from_numpy(self._scales[em.down_sc_off : em.down_sc_off + n_sc_dn].copy()).to(x.device)
            g = mps.tfq4_linear_mps(x.to(torch.float32), wg, sg, lut_t, eff_kind, I, H, I)
            u = mps.tfq4_linear_mps(x.to(torch.float32), wu, su, lut_t, eff_kind, I, H, I)
            act = self._apply_glu(g, u, swiglu_limit=swiglu_limit, gate_softcap=gate_softcap, up_softcap=up_softcap)
            y = mps.tfq4_linear_mps(act, wd, sd, lut_t, eff_kind, H, I, H)
            return y.to(dtype=x.dtype)

        return None

    def forward_expert(
        self,
        x: torch.Tensor,
        expert_idx: int,
        *,
        swiglu_limit: float | None = None,
        gate_softcap: float | None = None,
        up_softcap: float | None = None,
    ) -> torch.Tensor:
        """
        Run exact SwiGLU forward for a single routed expert `expert_idx` on input `x [..., hidden_size]`:
          - Default: `W_down @ (silu(W_gate @ x) * (W_up @ x))`
          - GLM clamped (`swiglu_limit > 0`): `gate <= limit`, `-limit <= up <= limit` before SwiGLU
          - Inkling-2 softcapped (`gate_softcap`, `up_softcap`):
            `softcap(g, gate_softcap) * sigmoid(g) * softcap(u, up_softcap)` where `softcap(v, c) = c * tanh(v / c)`.
        """
        if x.shape[-1] != self.hidden_size:
            raise ValueError(f"Expected input last dim {self.hidden_size}, got {x.shape[-1]}")
        em = self.experts[expert_idx]

        # Fused CUDA / MPS dispatch when not using decoded CPU cache
        if not self.cache_experts:
            if em.b_bits in (3, 4, 5) and x.is_cuda:
                ext = _get_cuda_ext()
                if ext is not None and hasattr(ext, "tfq3_moe"):
                    orig_shape = list(x.shape)
                    x2d = x.reshape(-1, self.hidden_size).to(torch.bfloat16)
                    w_t, s_t, m_t, t_t = self._get_cuda_moe_buffers(x.device)
                    picks = torch.full((x2d.shape[0], 1), int(expert_idx), dtype=torch.int32, device=x.device)
                    ey = ext.tfq3_moe(
                        x2d,
                        w_t,
                        s_t,
                        m_t,
                        t_t,
                        picks,
                        self.hidden_size,
                        self.moe_inter,
                        self.num_experts,
                        float(swiglu_limit or 0.0),
                        0,
                        float(gate_softcap or 0.0),
                        float(up_softcap or 0.0),
                    )
                    return ey.squeeze(1).reshape(orig_shape).to(dtype=x.dtype)
            elif em.b_bits in (2, 225) and (x.is_cuda or x.device.type == "mps"):
                y_fused = self._forward_tfq4_expert_fused(
                    x, em, swiglu_limit=swiglu_limit, gate_softcap=gate_softcap, up_softcap=up_softcap
                )
                if y_fused is not None:
                    return y_fused

        wg, wu, wd = self.decode_expert(expert_idx, dtype=torch.float32, device=x.device)
        x32 = x.to(torch.float32)
        g = F.linear(x32, wg)
        u = F.linear(x32, wu)
        act = self._apply_glu(g, u, swiglu_limit=swiglu_limit, gate_softcap=gate_softcap, up_softcap=up_softcap)
        return F.linear(act, wd).to(dtype=x.dtype)

    def forward_routed(
        self,
        x: torch.Tensor,
        expert_ids: torch.Tensor,
        expert_weights: torch.Tensor,
        *,
        swiglu_limit: float | None = None,
        gate_softcap: float | None = None,
        up_softcap: float | None = None,
    ) -> torch.Tensor:
        """
        Execute top-k routed MoE over `x [T, H]`, `expert_ids [T, K]`, `expert_weights [T, K]`.

        On CUDA with `tfq3_cuda_ext` installed, dispatches `{3, 4, 5}`-bit experts via the fused
        `tfq3_moe` CUDA kernel (including Inkling-2 tanh softcap) and seamlessly executes any
        `{2, 225}`-bit TFQ4 cold experts via per-expert TFQ4 dispatch, combining all weighted
        outputs in ascending expert ID order.
        """
        if x.ndim != 2 or expert_ids.ndim != 2 or expert_weights.ndim != 2:
            raise ValueError("forward_routed expects 2D x [T, H], expert_ids [T, K], and expert_weights [T, K]")
        T, H = x.shape
        if H != self.hidden_size or expert_ids.shape != expert_weights.shape or expert_ids.shape[0] != T:
            raise ValueError("Mismatched shapes in forward_routed")
        if T == 0 or expert_ids.shape[1] == 0:
            return torch.zeros((T, H), dtype=x.dtype, device=x.device)

        active = sorted(int(e) for e in torch.unique(expert_ids).tolist() if 0 <= int(e) < self.num_experts)

        # Fused CUDA tfq3_moe fast path for {3, 4, 5}-bit experts (+ seamless TFQ4 hybrid fallback for {2, 225})
        if x.is_cuda and self.has_tfq3_experts:
            ext = _get_cuda_ext()
            if ext is not None and hasattr(ext, "tfq3_moe"):
                tfq3_active = [e for e in active if self.experts[e].b_bits in (3, 4, 5)]
                tfq4_active = [e for e in active if self.experts[e].b_bits in (2, 225)]
                if tfq3_active:
                    w_t, s_t, m_t, t_t = self._get_cuda_moe_buffers(x.device)
                    picks_i32 = expert_ids.to(device=x.device, dtype=torch.int32).contiguous()
                    if tfq4_active:
                        picks_tfq3 = picks_i32.clone()
                        for eid in tfq4_active:
                            picks_tfq3.masked_fill_(picks_tfq3 == eid, -1)
                    else:
                        picks_tfq3 = picks_i32
                    ey = ext.tfq3_moe(
                        x.to(torch.bfloat16),
                        w_t,
                        s_t,
                        m_t,
                        t_t,
                        picks_tfq3,
                        self.hidden_size,
                        self.moe_inter,
                        self.num_experts,
                        float(swiglu_limit or 0.0),
                        0,
                        float(gate_softcap or 0.0),
                        float(up_softcap or 0.0),
                    )  # [T, K, H] bf16

                    out = torch.zeros((T, H), dtype=torch.float32, device=x.device)
                    tfq4_set = set(tfq4_active)
                    for eid in active:
                        tok_idx, slot_idx = torch.where(expert_ids == eid)
                        if eid in tfq4_set:
                            y_e = self.forward_expert(
                                x[tok_idx],
                                eid,
                                swiglu_limit=swiglu_limit,
                                gate_softcap=gate_softcap,
                                up_softcap=up_softcap,
                            ).to(torch.float32)
                        else:
                            y_e = ey[tok_idx, slot_idx].to(torch.float32)
                        w_e = expert_weights[tok_idx, slot_idx].to(torch.float32).unsqueeze(-1)
                        out.index_add_(0, tok_idx, y_e * w_e)
                    return out.to(dtype=x.dtype)

        out = torch.zeros((T, H), dtype=torch.float32, device=x.device)
        for eid in active:
            tok_idx, slot_idx = torch.where(expert_ids == eid)
            y_e = self.forward_expert(
                x[tok_idx],
                eid,
                swiglu_limit=swiglu_limit,
                gate_softcap=gate_softcap,
                up_softcap=up_softcap,
            ).to(torch.float32)
            w_e = expert_weights[tok_idx, slot_idx].to(torch.float32).unsqueeze(-1)
            out.index_add_(0, tok_idx, y_e * w_e)
        return out.to(dtype=x.dtype)

    def clear_cache(self) -> None:
        self._expert_cache.clear()
        self._cuda_moe_cache.clear()
        self._tfq4_lut_dev_cache.clear()

    def close(self) -> None:
        self._expert_cache.clear()
        self._cuda_moe_cache.clear()
        self._tfq4_lut_dev_cache.clear()
        self._scales = None
        self._words = None
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> TfpwLayer:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        return (
            f"TfpwLayer(layer={self.layer_idx}, experts={self.num_experts}, "
            f"hidden={self.hidden_size}, moe_inter={self.moe_inter}, "
            f"bits={self.bit_histogram()}, xxh64={self.checksum_verified})"
        )


class TfpwCheckpoint:
    """
    Unified standalone loader and inspector for `TFPW` v3 MoE checkpoints.

    Automatically discovers:
      - `pw_v3_tables.bin`
      - `layer_XX.tffpw` routed MoE layers (Inkling-2 `00..23` / `00..47`, GLM `03..44` / `03..77`, etc.)
      - optional `mtp_00.tffpw`
      - optional `nonmoe.tffpw` (`TFPN` v3); absent in GLM `--keep-nonmoe-bf16` and default Inkling-2
      - BF16/FP8 residual safetensors (`residual-*.safetensors`, `model*.safetensors`, `model.safetensors.index.json`)
      - optional standalone `mtp.safetensors` (Inkling-2 DSpark / MTP drafter)
    """

    def __init__(
        self,
        ckpt_dir: str | os.PathLike[str],
        *,
        verify_checksums: bool = True,
        source_dir: str | os.PathLike[str] | None = None,
        cache_experts: bool = False,
    ) -> None:
        self.ckpt_dir = os.fspath(ckpt_dir)
        self.source_dir = os.fspath(source_dir) if source_dir is not None else None
        if not os.path.isdir(self.ckpt_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {self.ckpt_dir}")

        self.config: dict[str, Any] = {}
        cfg_path = os.path.join(self.ckpt_dir, "config.json")
        if not os.path.isfile(cfg_path) and self.source_dir:
            cand_cfg = os.path.join(self.source_dir, "config.json")
            if os.path.isfile(cand_cfg):
                cfg_path = cand_cfg
        if os.path.isfile(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)

        self.tables_path: str | None = os.path.join(self.ckpt_dir, "pw_v3_tables.bin")
        self.tables_v4_path: str | None = os.path.join(self.ckpt_dir, "pw_v4_e8p_tables.bin")
        self.tables: dict[str, Any] | None = None
        self.tables_v4: tuple[np.ndarray, int, bytes] | None = None

        if os.path.isfile(self.tables_path):
            self.tables = load_and_validate_tables(self.tables_path)
        else:
            self.tables_path = None
        if os.path.isfile(self.tables_v4_path):
            self.tables_v4 = load_and_validate_pwv4_tables(self.tables_v4_path)
        else:
            self.tables_v4_path = None

        if self.tables is None and self.tables_v4 is None:
            raise FileNotFoundError(
                f"Missing pw_v3_tables.bin (and pw_v4_e8p_tables.bin) in {self.ckpt_dir}"
            )

        self.layers: dict[int, TfpwLayer] = {}
        self.layer_paths: dict[int, str] = {}
        self.mtp_tfpw_layer: TfpwLayer | None = None
        self.nonmoe: TfpnNonMoeShard | None = None
        self.nonmoe_path: str | None = None

        try:
            # 1. Discover & map layer_XX.tffpw shards
            layer_files: list[tuple[int, str]] = []
            for entry in sorted(os.listdir(self.ckpt_dir)):
                m = _LAYER_FILE_RE.match(entry)
                if m:
                    layer_files.append((int(m.group(1)), os.path.join(self.ckpt_dir, entry)))
            layer_files.sort(key=lambda pair: pair[0])
            if not layer_files:
                raise FileNotFoundError(f"No layer_XX.tffpw TFPW shards found in {self.ckpt_dir}")

            for idx, path in layer_files:
                lyr = TfpwLayer(
                    path,
                    self.tables,
                    verify_checksum=verify_checksums,
                    expected_layer_idx=idx,
                    cache_experts=cache_experts,
                    tables_v4=self.tables_v4,
                )
                self.layers[idx] = lyr
                self.layer_paths[idx] = path

            mtp_tfpw = os.path.join(self.ckpt_dir, "mtp_00.tffpw")
            if os.path.isfile(mtp_tfpw):
                self.mtp_tfpw_layer = TfpwLayer(
                    mtp_tfpw,
                    self.tables,
                    verify_checksum=verify_checksums,
                    cache_experts=cache_experts,
                    tables_v4=self.tables_v4,
                )

            # 2. Optional nonmoe.tffpw (absent in Inkling-2 default and GLM --keep-nonmoe-bf16)
            nm_path = os.path.join(self.ckpt_dir, "nonmoe.tffpw")
            if os.path.isfile(nm_path):
                if self.tables is None:
                    raise FileNotFoundError(f"Found nonmoe.tffpw but missing pw_v3_tables.bin in {self.ckpt_dir}")
                self.nonmoe_path = nm_path
                self.nonmoe = TfpnNonMoeShard(nm_path, self.tables, verify_checksum=verify_checksums)

            # 3. Residual safetensors discovery (in ckpt_dir, or fallback to source_dir for --keep-nonmoe-bf16)
            self.residual_dir: str | None = None
            self.residual_index_path: str | None = None
            self.residual_shards: list[str] = []
            self.residual_weight_map: dict[str, str] = {}
            self._discover_residuals()

            # 4. Optional standalone mtp.safetensors (Inkling-2 DSpark / MTP drafter)
            self.mtp_safetensors_path: str | None = None
            self._mtp_keys: list[str] = []
            self._discover_mtp_safetensors()

            # 5. Family & non-MoE mode classification
            self.family = self._detect_family()
            self.has_nonmoe_tfq3 = self.nonmoe is not None
            self.nonmoe_mode = "tfq3_tfpn" if self.has_nonmoe_tfq3 else "bf16_residual"
        except Exception:
            self.close()
            raise

    def _discover_residuals(self) -> None:
        search_dirs = [self.ckpt_dir]
        if self.source_dir and os.path.isdir(self.source_dir):
            search_dirs.append(self.source_dir)

        for root in search_dirs:
            idx_path = os.path.join(root, "model.safetensors.index.json")
            if os.path.isfile(idx_path):
                with open(idx_path, "r", encoding="utf-8") as f:
                    idx_obj = json.load(f)
                wm = idx_obj.get("weight_map", {})
                if isinstance(wm, dict) and wm:
                    root_real = os.path.realpath(root)
                    shards_seen: list[str] = []
                    shard_set: set[str] = set()
                    for k, rel in wm.items():
                        full = os.path.realpath(os.path.join(root, str(rel)))
                        if os.path.commonpath([root_real, full]) != root_real:
                            raise ValueError(f"Residual shard path escapes checkpoint dir: {rel}")
                        if not os.path.isfile(full):
                            raise FileNotFoundError(f"Missing residual safetensors shard: {full}")
                        self.residual_weight_map[str(k)] = full
                        if full not in shard_set:
                            shard_set.add(full)
                            shards_seen.append(full)
                    self.residual_dir = root
                    self.residual_index_path = idx_path
                    self.residual_shards = sorted(shards_seen)
                    return

            # Fallback: scan residual-*.safetensors or *.safetensors (excluding mtp.safetensors)
            cands = [
                p for p in sorted(glob.glob(os.path.join(root, "*.safetensors")))
                if os.path.basename(p) != "mtp.safetensors"
            ]
            if cands:
                self.residual_dir = root
                self.residual_shards = cands
                for sp in cands:
                    _, hdr = _read_safetensors_header(sp)
                    for k in hdr:
                        if k != "__metadata__":
                            self.residual_weight_map[k] = sp
                return

    def _discover_mtp_safetensors(self) -> None:
        search_dirs = [self.ckpt_dir]
        if self.source_dir and os.path.isdir(self.source_dir):
            search_dirs.append(self.source_dir)
        for root in search_dirs:
            cand = os.path.join(root, "mtp.safetensors")
            if os.path.isfile(cand):
                _, hdr = _read_safetensors_header(cand)
                self.mtp_safetensors_path = cand
                self._mtp_keys = sorted(k for k in hdr if k != "__metadata__")
                return

    def _detect_family(self) -> str:
        mt = str(self.config.get("model_type") or "")
        if not mt and isinstance(self.config.get("text_config"), dict):
            mt = str(self.config["text_config"].get("model_type") or "")
        if mt in ("inkling2", "inkling2_loop"):
            return "inkling2"
        if mt in ("glm5_next", "glm_moe_dsa"):
            return "glm"
        if mt == "qwen4_exp":
            return "flashnext"
        if any(k.startswith("model.llm.") for k in self.residual_weight_map):
            return "inkling2"
        if any(k.startswith("model.language_model.") for k in self.residual_weight_map):
            return "glm"
        min_layer = min(self.layers.keys())
        if min_layer == 3:
            return "glm"
        return "generic_moe"

    @property
    def layer_indices(self) -> list[int]:
        return sorted(self.layers.keys())

    @property
    def has_mtp_safetensors(self) -> bool:
        return self.mtp_safetensors_path is not None

    def get_layer(self, layer_idx: int) -> TfpwLayer:
        return self.layers[layer_idx]

    def __getitem__(self, layer_idx: int) -> TfpwLayer:
        return self.layers[layer_idx]

    def decode_expert(
        self,
        layer_idx: int,
        expert_idx: int,
        *,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.layers[layer_idx].decode_expert(expert_idx, dtype=dtype, device=device)

    def forward_expert(
        self,
        layer_idx: int,
        x: torch.Tensor,
        expert_idx: int,
        *,
        swiglu_limit: float | None = None,
        gate_softcap: float | None = None,
        up_softcap: float | None = None,
    ) -> torch.Tensor:
        return self.layers[layer_idx].forward_expert(
            x,
            expert_idx,
            swiglu_limit=swiglu_limit,
            gate_softcap=gate_softcap,
            up_softcap=up_softcap,
        )

    def residual_tensor_names(self) -> list[str]:
        return sorted(self.residual_weight_map.keys())

    def load_residual_tensor(self, name: str, *, device: str | torch.device = "cpu") -> torch.Tensor:
        if name not in self.residual_weight_map:
            raise KeyError(f"Residual tensor '{name}' not found in {self.ckpt_dir}")
        return _read_safetensors_tensor(self.residual_weight_map[name], name, device=device)

    def nonmoe_tensor_names(self) -> list[str]:
        return self.nonmoe.tensor_names() if self.nonmoe is not None else []

    def decode_nonmoe_tensor(
        self,
        name: str,
        *,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        if self.nonmoe is None:
            raise FileNotFoundError(
                f"No nonmoe.tffpw in {self.ckpt_dir} (nonmoe_mode={self.nonmoe_mode}; load from residual safetensors instead)"
            )
        return self.nonmoe.decode_tensor(name, dtype=dtype, device=device)

    def mtp_tensor_names(self) -> list[str]:
        return list(self._mtp_keys)

    def load_mtp_tensor(self, name: str, *, device: str | torch.device = "cpu") -> torch.Tensor:
        if self.mtp_safetensors_path is None:
            raise FileNotFoundError(f"No mtp.safetensors found in {self.ckpt_dir}")
        return _read_safetensors_tensor(self.mtp_safetensors_path, name, device=device)

    def summary(self) -> dict[str, Any]:
        bit_hist: dict[int, int] = {}
        total_tfpw_bytes = 0
        all_verified = True
        any_verified = False
        for lyr in self.layers.values():
            total_tfpw_bytes += lyr.file_size
            all_verified = all_verified and lyr.checksum_verified
            any_verified = any_verified or lyr.checksum_verified
            for b, cnt in lyr.bit_histogram().items():
                bit_hist[b] = bit_hist.get(b, 0) + cnt
        first = self.layers[self.layer_indices[0]]
        tot_experts = sum(bit_hist.values())
        avg_bpw = sum(_bits_to_bpw(b) * c for b, c in bit_hist.items()) / float(max(tot_experts, 1))
        is_hybrid = any(b in (2, 225) for b in bit_hist) and any(b in (3, 4, 5) for b in bit_hist)
        return {
            "ckpt_dir": self.ckpt_dir,
            "family": self.family,
            "model_type": self.config.get("model_type"),
            "num_moe_layers": len(self.layers),
            "layer_indices": self.layer_indices,
            "experts_per_layer": first.num_experts,
            "hidden_size": first.hidden_size,
            "moe_inter": first.moe_inter,
            "bit_histogram": dict(sorted(bit_hist.items())),
            "avg_bpw": avg_bpw,
            "is_hybrid": is_hybrid,
            "has_tfq4_tables": self.tables_v4 is not None,
            "tfq4_table_kind": self.tables_v4[1] if self.tables_v4 is not None else None,
            "total_tfpw_bytes": total_tfpw_bytes,
            "has_nonmoe_tfq3": self.has_nonmoe_tfq3,
            "nonmoe_mode": self.nonmoe_mode,
            "nonmoe_tensor_count": len(self.nonmoe.records) if self.nonmoe is not None else 0,
            "residual_shard_count": len(self.residual_shards),
            "residual_tensor_count": len(self.residual_weight_map),
            "has_mtp_safetensors": self.has_mtp_safetensors,
            "mtp_tensor_count": len(self._mtp_keys),
            "has_mtp_tfpw": self.mtp_tfpw_layer is not None,
            "all_checksums_verified": all_verified,
            "any_checksum_verified": any_verified,
        }

    def format_summary(self) -> str:
        s = self.summary()
        lo, hi = s["layer_indices"][0], s["layer_indices"][-1]
        lines = [
            f"TFPW Checkpoint: {s['ckpt_dir']}",
            f"  family:     {s['family']} (model_type={s['model_type']})",
            f"  MoE layers: {s['num_moe_layers']} (layer_{lo:02d}..layer_{hi:02d}.tffpw, "
            f"E={s['experts_per_layer']}, H={s['hidden_size']}, I={s['moe_inter']})",
            f"  precision:  {s['avg_bpw']:.2f} avg bpw, expert bits={s['bit_histogram']}, "
            f"size={s['total_tfpw_bytes'] / (1024 ** 3):.3f} GiB",
            f"  non-MoE:    {s['nonmoe_mode']} "
            f"(nonmoe.tffpw={s['nonmoe_tensor_count']} tensors, "
            f"residuals={s['residual_tensor_count']} tensors across {s['residual_shard_count']} shard(s))",
            f"  MTP draft:  mtp.safetensors={s['has_mtp_safetensors']} ({s['mtp_tensor_count']} tensors), "
            f"mtp_00.tffpw={s['has_mtp_tfpw']}",
            f"  XXH64:      all_verified={s['all_checksums_verified']}",
        ]
        return "\n".join(lines)

    def close(self) -> None:
        for lyr in self.layers.values():
            lyr.close()
        self.layers.clear()
        if self.mtp_tfpw_layer is not None:
            self.mtp_tfpw_layer.close()
            self.mtp_tfpw_layer = None
        if self.nonmoe is not None:
            self.nonmoe.close()
            self.nonmoe = None

    def __enter__(self) -> TfpwCheckpoint:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
