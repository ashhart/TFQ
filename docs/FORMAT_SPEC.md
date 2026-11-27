# TFQ3 Binary Format Specification (v3)

TFQ3 is a **single cross-platform binary format**: the exact same `pw_v3_tables.bin` and `.tffpw` files run unmodified on both **Apple Silicon (Metal)** and **NVIDIA GPUs (CUDA `sm_80+`)** with zero re-quantization or repacking.

All integers and floats are **little-endian**. Every GPU-mapped payload section (`scales` and `weights`) is aligned to **16,384 bytes (16 KiB)** so Apple Silicon runtimes can `mmap` files read-only and wrap sections directly with `MTLDevice.newBufferWithBytesNoCopy` (zero copies, zero peak RAM spike), while CUDA runtimes `mmap` and DMA-upload contiguous aligned buffers straight into device memory.

---

## 1. Universal Codebook File: `pw_v3_tables.bin` (`PWV3`)

Fixed size: **16,384 bytes** (active payload: **10,816 bytes**, zero-padded to 16 KiB).

| Offset | Size (bytes) | Type | Field | Description |
|---:|---:|---|---|---|
| `0` | 4 | `u32` | `magic` | `0x33565750` (`"PWV3"`) |
| `4` | 4 | `u32` | `version` | `3` |
| `8` | 24 | `f32[6]` | `a3, b3, a4, b4, a5, b5` | Affine coset scale/shift hyperparameters (`3b/4b/5b`) |
| `32` | 32 | `u32[8]` | `pad0` | Reserved (`0`) |
| `64` | 1,024 | `u32[256]` | `lut3` | 256 packed `half2` 2D centroids (`4` codebooks × `64` entries) |
| `1088` | 4,096 | `u32[1024]` | `lut4` | 1,024 packed `half2` 2D centroids (`4` codebooks × `256` entries) |
| `5184` | 4,096 | `u32[1024]` | `lut5` | 1,024 packed `half2` 2D centroids (`1` codebook × `1,024` entries) |
| `9280` | 512 | `half4[64]` | `c64_3` | 64 affine coset tuples `(sx, sy, tx, ty)` for 3-bit |
| `9792` | 512 | `half4[64]` | `c64_4` | 64 affine coset tuples `(sx, sy, tx, ty)` for 4-bit |
| `10304` | 512 | `half4[64]` | `c64_5` | 64 affine coset tuples `(sx, sy, tx, ty)` for 5-bit |
| `10816` | 5,568 | `u8[5568]` | `pad1` | Zero padding to `16,384` bytes |

- **Canonical table SHA-256 (`reference/tables/pw_v3_tables.bin`):**
  `956c7abe5d400f302eaf8a2e9d0da50ee18f90d302de786cf6b2ec4905358bb7`
  All Metal and CUDA converters embed and emit this exact canonical codebook by default unless `--tables <path>` is passed explicitly.

**Validation requirements:**
- File size `>= 10,816` bytes (canonically `16,384` bytes).
- `magic == 0x33565750` and `version == 3`.
- All 6 `f32` scalars and all `half` entries in `lut3`, `lut4`, `lut5`, `c64_3`, `c64_4`, `c64_5` must be finite (no NaN/Inf).

---

## 2. Dense / Non-MoE Shard Format: `*.tffpw` (`TFPN`)

Used for dense models (`qwen27_tfq3.tffpw` or sharded `qwen27_tfq3-00001-of-00004.tffpw` … `00004-of-00004.tffpw`) and GLM non-MoE weights (`nonmoe.tffpw`).

### File Layout

```
[0 .. 64)                      TfpnHeader (64 bytes)
[off_meta .. off_meta + N*128) TfpnTensorMeta[num_tensors] (128 bytes each)
[... zero padding ...]
[off_scales .. off_weights)    fp16 block scales (16 KiB aligned start)
[... zero padding ...]
[off_weights .. file_end)      u32 packed weight words (16 KiB aligned start & end)
```

### `TfpnHeader` (64 bytes)

```c
struct TfpnHeader {
    uint32_t magic;              // 0x4E504654 ("TFPN")
    uint32_t version;            // 3
    uint32_t num_tensors;        // number of TensorMeta entries
    uint32_t pad0;               // flags: bit 0 (0x1) = XXH64 checksum present in pad1 (0 = legacy)
    uint64_t off_meta;           // byte offset of TensorMeta[] (>= 64)
    uint64_t off_scales;         // byte offset of half[] scales (16 KiB aligned)
    uint64_t off_weights;        // byte offset of u32[] weights (16 KiB aligned)
    uint64_t total_scale_halfs;  // total fp16 scale elements
    uint64_t total_weight_words; // total u32 weight words
    uint64_t pad1;               // XXH64(seed=0) streamed over exact [meta, scales, weights] slices when pad0 & 0x1
};
```

### `TfpnTensorMeta` (128 bytes)

```c
struct TfpnTensorMeta {
    char     name[80];    // NUL-terminated canonical tensor name (e.g. "model.layers.0.mlp.gate_proj", "lm_head")
    uint32_t R;           // output rows (positive multiple of 128)
    uint32_t C;           // input columns (positive multiple of 128)
    uint32_t b_bits;      // 3, 4, or 5
    uint32_t sc_off;      // start offset in fp16 elements inside `scales`
    uint64_t word_off;    // start offset in u32 words inside `weights`
    uint64_t word_count;  // exact word count: (R/16) * (C/128) * (b_bits * 2) * 32
    float    snr_db;      // encoder-measured quantization SNR in dB
    uint32_t pad[3];      // [0] = orig_R (0 -> R), [1] = orig_C (0 -> C), [2] = flags bitmask
};
```

### Backward-Compatible Metadata Extension (`pad[3]`)
Model-agnostic conversions (arbitrary matrix dimensions zero-padded to multiples of 128, plus optional vision / drafter / embedding quantization) record metadata inside the existing `pad[3]` words with zero format break:
- `pad[0]`: unpadded original row count `orig_R` (`0` on legacy shards → treat as `R`).
- `pad[1]`: unpadded original column count `orig_C` (`0` on legacy shards → treat as `C`).
- `pad[2]`: flags bitmask (`0` on legacy shards):
  - `bit 0` (`0x1`): tensor was zero-padded to `[R, C]` from `[orig_R, orig_C]`.
  - `bit 1` (`0x2`): vision encoder / multimodal projector tensor (`--quantize-vision`).
  - `bit 2` (`0x4`): speculative drafter tensor (`--quantize-drafter`).
  - `bit 3` (`0x8`): token embedding table (`--quantize-embed`).

### Zero-Repack Tensor Parallelism (`Col-TP` & `Row-TP`)
Because every `128 × 128` tile is independently rotated and every column block `bc` carries its exact 32-bit state hash inside the final `u32` word of block `bc - 1`, any `TFPN` or `TFPW` weight matrix can be sharded across `TP` ranks at runtime with **zero weight repacking**:
- **Column-TP (`n_start, n_end` multiples of 128):** each rank computes output rows `[n_start, n_end)` independently.
- **Row-TP (`k_start, k_end` multiples of 128):** each rank rotates its input slice `X[:, k_start:k_end]` using sign offset `k_sign_off = k_start`, resumes the per-lane state hash at `bc0 = k_start / 128` directly from `bc0 - 1`'s last packed word (zero overhead), multiplies `[k_start, k_end)`, and sums partial outputs via `AllReduce` before applying the final inverse row-FWHT-128.

### Tensor Scale & Weight Indexing

For a tensor of shape `[R, C]` (`n_rb = R / 128`, `n_bc = C / 128`, `wpb = b_bits * 2`):
- **Scales:** `scales[sc_off + rb * n_bc + bc]` is the `fp16` RMS scale for $128 \times 128$ block `(rb, bc)`. Total scale elements = `n_rb * n_bc`.
- **Weights:** Partitioned into `R / 16` horizontal strips (`strip = rb * 8 + s`, $s \in [0..7]$). Strip `strip`, column block `bc` begins at:
  `word_idx = word_off + (strip * n_bc + bc) * wpb * 32`
  - **4-bit (`wpb = 8`):** Stored as `2` consecutive `uint4[32]` arrays (`h ∈ {0, 1}`, lane-major AoS: `((uint4*)blk)[h * 32 + lane]` holds the 4 `u32` subtile words for columns `[h*64 + 0..64)`).
  - **3-bit (`wpb = 6`):** Stored as `2` consecutive `uint32[3][32]` SoA planes (`h ∈ {0, 1}`, `blk[(h * 3 + p) * 32 + lane]` for $p \in \{0, 1, 2\}$ packing four 24-bit subtile codes across 96 bits).
  - **5-bit (`wpb = 10`):** Stored as `2` consecutive `uint32[5][32]` SoA planes (`h ∈ {0, 1}`, `blk[(h * 5 + p) * 32 + lane]` for $p \in \{0..4\}$ packing four 40-bit subtile codes across 160 bits).

### Contiguous Pair Fusion Rule (`canFusePair`)
Two tensors `A` (`[R0, C]`) and `B` (`[R1, C]`) in the **same shard** can execute in a single fused Metal or CUDA launch (with `sr_split = R0`) if and only if:
1. `A.C == B.C` and `A.b_bits == B.b_bits`
2. `B.sc_off == A.sc_off + (A.R / 128) * (A.C / 128)`
3. `B.word_off == A.word_off + A.word_count`

---

## 3. MoE Layer Shard Format: `layer_XX.tffpw` / `mtp_00.tffpw` (`TFPW`)

Used for routed MoE layers in `GLM-5.3-Flash` (`288` experts) and `Qwen3.8-Flash-Next` (`512` experts).

### `TfpwHeader` (144 bytes)

```c
struct TfpwHeader {
    uint32_t magic;          // 0x57504654 ("TFPW")
    uint32_t version;        // 3
    uint32_t layer_idx;      // layer index
    uint32_t num_experts;    // expert count (e.g. 288 or 512)
    uint32_t num_tensors;    // 3 * num_experts (gate, up, down)
    uint32_t hidden_size;    // model hidden_size (multiple of 128)
    uint32_t moe_inter;      // moe_intermediate_size (multiple of 128)
    uint32_t reserved0;      // flags: bit 0 (0x1) = per-section XXH64 checksums present (0 = legacy)
    uint64_t raw_fp8_bytes;  // uncompressed reference byte count
    uint64_t stream_words;   // total u32 weight words
    uint64_t scale_halfs;    // total fp16 scale elements
    uint64_t off_meta;       // byte offset of TfpwExpertMeta[num_experts]
    uint64_t off_scales;     // 16 KiB aligned fp16 scales section
    uint64_t off_weights;    // 16 KiB aligned u32 weights section
    uint64_t reserved1[4];   // when reserved0 & 0x1: [0]=XXH64(meta), [1]=XXH64(scales), [2]=XXH64(weights), [3]=0 (seed=0)
    char     meta_tag[32];   // NUL-terminated profile/tag label
};
```

### `TfpwExpertMeta` (64 bytes)

```c
struct TfpwExpertMeta {
    uint32_t b_bits;         // bit width for this expert: 3, 4, 5 (TFQ3), or 2 (2.00b E8P-256), 225 (2.25b E8P-1024)
    uint32_t heat_rank;      // HeatAlloc routing rank (0 = hottest)
    uint64_t gate_word_off;  // u32 word offset for gate_proj [moe_inter, hidden_size]
    uint64_t up_word_off;    // u32 word offset for up_proj   [moe_inter, hidden_size]
    uint64_t down_word_off;  // u32 word offset for down_proj [hidden_size, moe_inter]
    uint32_t gate_sc_off;    // fp16 scale offset for gate_proj
    uint32_t up_sc_off;      // fp16 scale offset for up_proj
    uint32_t down_sc_off;    // fp16 scale offset for down_proj
    float    gate_snr_db;    // encoder-measured SNR for gate_proj
    float    up_snr_db;      // encoder-measured SNR for up_proj
    float    down_snr_db;    // encoder-measured SNR for down_proj
    uint32_t reserved[2];    // 0
};
```

#### Hybrid Hot/Cold MoE (`TFQ3` + `TFQ4-E8P` in `TFPW`)
A single `TFPW` layer shard may mix `TFQ3` (`b_bits ∈ {3, 4, 5}`) hot/warm experts with `TFQ4` (`b_bits ∈ {2, 225}`) cold experts while sharing the exact same `2D FWHT-128` input/output rotation and `128 × 128` `fp16` scale grid (`(R / 128) * (C / 128)` scales per projection):
- **Per-projection word count `wpm(R, C, b_bits)`:**
  - `b_bits ∈ {3, 4, 5}`: `(R / 16) * (C / 128) * (2 * b_bits) * 32` (`TFQ3` trellis layout, §6.2–6.3)
  - `b_bits == 2`: `R * (C / 16)` (`8` words per 128 cols, `TFQ4` True QuIP# E8P-256, §6.4)
  - `b_bits == 225`: `R * (C / 128) * 9` (`9` words per 128 cols, `TFQ4` True QuIP# E8P-1024, §6.4)
- **Companion table files:** Whenever any expert uses `b_bits == 2` or `225`, converters emit `pw_v4_e8p_tables.bin` alongside `pw_v3_tables.bin` (`table_kind = 2` if only `2b` experts are present; `table_kind = 3` if any `225` expert is present — never downgrading an existing 32 KiB `table_kind = 3` file since its first 256 rows decode `2b` identically).
- **Standard hybrid schedules (`--inkling2-mode` / `--glm-mode`):**
  - `heat432`: top 15% experts `4b`, next 25% `3b`, bottom 60% `2.00b` (`b_bits = 2`)
  - `heat42`: top 20% experts `4b`, bottom 80% `2.00b` (`b_bits = 2`)
  - `heat43225`: top 15% experts `4b`, next 25% `3b`, bottom 60% `2.25b` (`b_bits = 225`)
  - `uniform2`: all experts `2.00b` (`b_bits = 2`); `uniform225`: all experts `2.25b` (`b_bits = 225`)

---

## 4. Mandatory Loader Validation Checks

Any compliant reader (and `build/tfq3-inspect`) enforces:
1. `magic` and `version == 3` match.
2. `off_scales % 16384 == 0` and `off_weights % 16384 == 0`.
3. Strict non-overlapping monotonic sections: `sizeof(Header) <= off_meta <= meta_end <= off_scales <= scales_end <= off_weights <= weights_end <= file_size`.
4. All arithmetic checked against 64-bit integer overflow.
5. Every `TFPN` tensor has `bits ∈ {3, 4, 5}`, every `TFPW` expert has `bits ∈ {2, 225, 3, 4, 5}`, and every `TFP4` tensor has `bits ∈ {2, 225}`, with `R > 0`, `C > 0`, `R % 128 == 0`, `C % 128 == 0`, exact expected `word_count`, and offsets strictly inside `[0, total_scale_halfs]` and `[0, total_weight_words]`.
6. No duplicate tensor names across multi-shard `TFPN` directories.
7. **Integrity checksums (`XXH64`, seed `0`, backwards-compatible):**
   - `TFPN` and `TFP4`: when `pad0 & 0x1` is set, `pad1` must equal `XXH64(seed=0)` streamed over the exact payload slices `[off_meta .. +num_tensors*128)`, `[off_scales .. +total_scale_halfs*2)`, `[off_weights .. +total_weight_words*4)` (excluding inter-section zero padding).
   - `TFPW`: when `reserved0 & 0x1` is set, `reserved1[0..2]` must equal the independent `XXH64(seed=0)` digests of `meta`, `scales`, and `weights`.
   - Legacy shards with flag bit `0` cleared (`0`) remain valid. All converters set the flag bit and checksums unconditionally; `tfq3-inspect` and reference readers verify them whenever present.

---

## 5. Sub-3-Bit Extension: `PWV4` v4 & `TFP4` v4 (`2.001` & `2.251 bpw` 8D `E8P`)

`TFQ4` shares `TFQ3`'s exact `2D FWHT-128` rotation (`SEED_COL = 0x12345678`, `SEED_ROW = 0x9E3779B9`) and `128 × 128` `fp16` tile scale grid (`0.001 bpw` overhead), allowing hybrid `TFQ3` (`3/4/5b`) + `TFQ4` (`2.00b` / `2.25b`) checkpoints to share one input rotation and one inverse row-FWHT epilogue:

- **`pw_v4_e8p_tables.bin` (`PWV4`, fixed `16,384` bytes for `table_kind ∈ {0, 1, 2}`, or `32,768` bytes for `table_kind = 3`):**
  - Header (`64` bytes):
    - `magic = 0x34565750` (`"PWV4"`) at offset `0` (`u32`)
    - `version = 4` at offset `4` (`u32`)
    - `k_codes` at offset `8` (`u32`): `256` for `table_kind ∈ {0, 1, 2}`, or `1024` for `table_kind = 3`
    - `dim = 8` at offset `12` (`u32`)
    - `table_kind` (`pad0[0]`) at offset `16` (`u32`):
      - `0` — legacy/default trained `256 × 8` magnitude codebook with 8 independent sign bits (`65,536` codewords, `16,384` B file).
      - `1` — explicit alias for the trained `256 × 8` codebook (identical decode rule to `0`, `16,384` B file).
      - `2` — **true QuIP# E8P-256 (`E8 + 1/4`)** lattice codebook (`2.001 bpw`, `16,384` B file): `256 × 8` half-integer `|D8 + 1/2|` magnitude table (exact in `fp16`), 7 free sign bits + 1 derived sign-parity bit + 1 coset bit (`±1/4` shift), giving `65,536` unique points strictly inside `E8 + 1/4`.
      - `3` — **true QuIP# E8P-1024 (`E8 + 1/4`)** lattice codebook (`2.251 bpw`, `32,768` B file, `k_codes = 1024`): deterministic algebraic enumeration of all `m_d ∈ {0, 1, 2, 3}` (`a_d = m_d + 0.5`, `4^8 = 65,536` tuples) with `sum((2*m_d + 1)^2) <= 68`, sorted by `(n2_x4, max_m, lex m)`, taking the first `1024` entries (`1024 × 8 × 2 = 16,384` bytes at offsets `64..16447`, zero-padded `16448..32767`). Its first `256` entries are **100% bit-identical** to `table_kind = 2`, so a single 32 KiB `table_kind = 3` table decodes BOTH `2.001 bpw` (`0..255`) and `2.251 bpw` (`0..1023`). Optimal base scale: `0.84f` (`11.756 dB` on $\mathcal{N}(0, 1)^8$).
      - `> 3` (or size/`k_codes` mismatch with `table_kind`) — **unknown/invalid; compliant readers MUST reject.**
    - `pad0[1..11]` at offsets `20..63` (`u32[11]`, zero).
  - Offset `64 .. 64 + k_codes * 16`: `e8p_lut[k_codes][8]` (`fp16`: `4,096` bytes for `k_codes = 256`, `16,384` bytes for `k_codes = 1024`).
  - **Canonical table SHA-256 (`reference/tables/pw_v4_e8p_tables.bin`, `table_kind = 0`):**
    `ddf28171bd5ea33e2cddc560b5543f6890f2a179bb6dce7082a66243c04b7405`
  - **Canonical true-E8P-256 table SHA-256 (`--table-kind e8p`, `table_kind = 2`, `16,384` B):**
    `f1698d7e01bb0261f6b637c1a02b6965c7b21e6226c9790a5201283f44bf27e2`
  - **Canonical true-E8P-1024 table SHA-256 (`--table-kind e8p1024` / `--bits 2.25`, `table_kind = 3`, `32,768` B):**
    `3b9c3f290bb15c5a8328cb4b19f589e2ac90c43bac0bd40943f6666a9884431f`
- **`*.tffp4` (`TFP4`, 16 KiB-aligned):**
  - Identical 64-byte header (`magic = 0x34504654`, `version = 4`, with the same `pad0`/`pad1` XXH64 checksum contract as `TFPN`) and 128-byte tensor metadata (`pad = [orig_R, orig_C, flags]`) as `TFPN`.
  - **`b_bits = 2` (`2.001 bpw`, `8` words / 128 cols, `word_count = R * (C / 16)`):** row-major packed `uint32_t[R, C / 16]`; each `uint32_t` holds two 16-bit codes (`c0 = w & 0xFFFF`, `c1 = w >> 16`), decoded per `PWV4.table_kind` (§6.4).
  - **`b_bits = 225` (`2.251 bpw`, `9` words / 128 cols, `word_count = R * (C / 128) * 9`):** word-aligned row-major packed `uint32_t[R, (C / 128) * 9]`. For each 128-column row-block (16 eight-dimensional vectors `v_0..v_15`):
    - Words `w[0..7]`: for `sub_w ∈ 0..7`, `w[sub_w] = low16(v_{2*sub_w}) | (low16(v_{2*sub_w+1}) << 16)`, where `low16 = (abs_idx & 0xFF) | (s7 << 8) | (coset << 15)` (identical bit layout to `b_bits = 2`).
    - Word `w[8]`: packs the 2 high codebook index bits (`hi2 = (abs_idx >> 8) & 3`) across all 16 vectors: `bits [2*j+1 : 2*j] = (abs_idx_j >> 8) & 3` for `j = 0..15`.

---

## 6. Decoding Specification (Normative)

This section defines the exact mapping from packed binary words and scales to spatial `fp32` weights `W[R, C]`. The Python implementations in [`reference/tfq3_reference.py`](../reference/tfq3_reference.py) and [`reference/tfq4_lattice.py`](../reference/tfq4_lattice.py) are normative; golden test vectors generated from the canonical tables live in `reference/vectors/` and are verified against the reference, Metal, and CUDA decoders by `tests/vectors_test.py`.

All unsigned integer arithmetic below is modulo $2^{32}$ (`u32`, masked with `& 0xFFFFFFFF`).

### 6.1 Common Rotation Primitives (TFQ3 & TFQ4)

Both TFQ3 and TFQ4 rotate every `128 × 128` weight tile `(rb, bc)` using deterministic Rademacher sign flips and an orthonormal 128-point Fast Walsh-Hadamard Transform (`FWHT-128`):

```python
SEED_COL = 0x12345678
SEED_ROW = 0x9E3779B9

def pw_sgn(seed: u32, idx: u32) -> f32:
    h = ((idx ^ seed) * 2654435761) & 0xFFFFFFFF
    return -1.0 if ((h >> 31) & 1) else 1.0

def fwht_128(x: f32[128]) -> f32[128]:
    # In-place radix-2 butterfly for h = 1, 2, 4, 8, 16, 32, 64:
    #   for block_start in range(0, 128, 2*h):
    #     for j in range(h):
    #       u, v = x[block_start + j], x[block_start + j + h]
    #       x[block_start + j]     = u + v
    #       x[block_start + j + h] = u - v
    # Then normalize by 1 / sqrt(128) = 0.08838834764831845.
```

Given a decoded, scaled rotated tile `W_rot[128, 128]` at **absolute** tensor coordinates `(r_off = rb * 128, c_off = bc * 128)`:

1. Column-wise `FWHT-128` followed by absolute row signs:
   `tmp[r, c] = fwht_128(W_rot[:, c])[r] * pw_sgn(SEED_ROW, r_off + r)`
2. Row-wise `FWHT-128` followed by absolute column signs:
   `W[r, c]   = fwht_128(tmp[r, :])[c] * pw_sgn(SEED_COL, c_off + c)`

> **Important (Row-Chunked & Column-TP Decoding):** Row signs always index by the **absolute row** `r_off + r` of the tensor (`r_off = 0` for standalone tensors; `r_off = n_start` when decoding a Column-TP slice or fused second projection). Equivalently, for a GEMM `Y = X @ W.T`, a kernel rotates `X` along `K` with `pw_sgn(SEED_COL, k_off + c)` + `FWHT-128`, accumulates `Y_rot = X_rot @ W_rot.T`, and unrotates `Y_rot` along `N` with `FWHT-128` + `pw_sgn(SEED_ROW, row_off + r)`.

### 6.2 TFQ3 Trellis State, Lane Mapping, and Block Resume

A TFQ3 tensor `[R, C]` has `n_rb = R / 128` row blocks, `n_bc = C / 128` column blocks, `R / 16` horizontal strips (`strip = rb * 8 + s`, `s ∈ [0..7]`), and `wpb = b_bits * 2` words per lane per column block. Each strip carries `32` independent trellis streams (`lane ∈ [0..31]`):

```python
HASH_MULT    = 0x9E3779B1
INIT_ST_BASE = 0x13579BDF

def pw_frag_home(lane: u32) -> (col: u32, row: u32):
    col = (lane & 1) * 4 + ((lane >> 3) & 1) * 8   # {0, 4, 8, 12}
    row = ((lane >> 1) & 3) + ((lane >> 4) & 1) * 4 # {0..7}
    return (col, row)
```

For strip `strip` (`strip_base = strip * n_bc * wpb * 32`), column block `bc ∈ [0..n_bc)`, and `lane ∈ [0..31]`:

- **Initial state (`bc == 0`):**
  `st = (INIT_ST_BASE ^ ((lane * HASH_MULT) & 0xFFFFFFFF)) & 0xFFFFFFFF`
- **Random-access resume (`bc > 0`, e.g., split-K or Row-TP):**
  Every bit width's 32-bit state at the start of block `bc` is stored verbatim in the last packed `u32` word of block `bc - 1` for that `(strip, lane)`:
  - `b_bits == 4`: `st = words[strip_base + (bc * 2 - 1) * 128 + lane * 4 + 3]` (`uint4` element `.w` of half `h = 1` of block `bc - 1`)
  - `b_bits == 3` or `5`: `st = words[strip_base + (bc * wpb - 1) * 32 + lane]` (the final SoA plane of half `h = 1` of block `bc - 1`)

Within each column block `bc`, the `128` columns are split into `2` halves `h ∈ {0, 1}` of `64` columns each, and each half contains `4` subtiles `k ∈ {0, 1, 2, 3}` of `16` columns each (`kt = h * 4 + k ∈ [0..7]`). Each subtile decodes `8` rotated unit-variance weights `q[0..7]` and writes them scaled by `sc = f32(scales[rb * n_bc + bc])`:

```python
hc, hr = pw_frag_home(lane)
c0 = h * 64 + k * 16 + hc
W_rot[s * 16 + hr,     c0 : c0 + 4] = q[0:4] * sc
W_rot[s * 16 + hr + 8, c0 : c0 + 4] = q[4:8] * sc
```

### 6.3 TFQ3 Subtile Decode Pseudo-Code (`3b`, `4b`, `5b`)

Let `lut[idx] = (lx, ly)` be the `fp16` pair unpacked from `lut{3,4,5}_u32[idx]` and `c64[ci] = (sx, sy, tx, ty)` be the `half4` affine coset tuple from `c64_{3,4,5}[ci]`. For every subtile, pairs `(0, 1)` share coset `c0 = c64[(h0 >> 16) & 63]` and pairs `(2, 3)` share coset `c2 = c64[(h2 >> 16) & 63]`:

```python
def affine_pair(p: (f32, f32), c: (f32, f32, f32, f32)) -> (f32, f32):
    # GPU kernels compute fma(half2(p), half2(c.x, c.y), half2(c.z, c.w)) in fp16;
    # the normative Python reference evaluates the same formula in fp32.
    return (p[0] * c[0] + c[2], p[1] * c[1] + c[3])
```

#### 4-bit (`b_bits = 4`, `wpb = 8`, AoS `uint4`)

Half `h ∈ {0, 1}` loads four `u32` words for `lane` at `idx = strip_base + (bc * 2 + h) * 128 + lane * 4`; subtile `k ∈ {0..3}` uses `wc = words[idx + k]`:

```python
def decode_8_4b(st: u32, wc: u32, lut4, c64_4) -> (new_st: u32, q: f32[8]):
    s0 = st
    h0 = (s0 * 0x9E3779B1) & 0xFFFFFFFF
    h1 = ((((s0 <<  8) | (wc >> 24)) & 0xFFFFFFFF) * 0x9E3779B1) & 0xFFFFFFFF
    h2 = ((((s0 << 16) | (wc >> 16)) & 0xFFFFFFFF) * 0x9E3779B1) & 0xFFFFFFFF
    h3 = ((((s0 << 24) | (wc >>  8)) & 0xFFFFFFFF) * 0x9E3779B1) & 0xFFFFFFFF
    c0 = c64_4[(h0 >> 16) & 63]
    c2 = c64_4[(h2 >> 16) & 63]
    p0 = lut4[((( wc >> 24)        << 2) ^ (h0 >> 22)) & 1023]
    p1 = lut4[((((wc >> 16) & 255) << 2) ^ (h1 >> 22)) & 1023]
    p2 = lut4[((((wc >>  8) & 255) << 2) ^ (h2 >> 22)) & 1023]
    p3 = lut4[((( wc        & 255) << 2) ^ (h3 >> 22)) & 1023]
    q  = [*affine_pair(p0, c0), *affine_pair(p1, c0),
          *affine_pair(p2, c2), *affine_pair(p3, c2)]
    return (wc, q)
```

#### 3-bit (`b_bits = 3`, `wpb = 6`, 3 SoA planes per half)

Half `h ∈ {0, 1}` loads three `u32` plane words at `idx = strip_base + (bc * 6 + h * 3) * 32 + lane`:
`a0 = words[idx]`, `a1 = words[idx + 32]`, `a2 = words[idx + 64]`.
These 96 bits unpack big-endian into four 24-bit codes `m24[0..3]`:

```python
m24 = [
    a0 >> 8,
    ((a0 << 16) | (a1 >> 16)) & 0xFFFFFF,
    ((a1 <<  8) | (a2 >> 24)) & 0xFFFFFF,
    a2 & 0xFFFFFF,
]

def decode_8_3b(st: u32, m24: u32, lut3, c64_3) -> (new_st: u32, q: f32[8]):
    s0 = st
    h0 = (s0 * 0x9E3779B1) & 0xFFFFFFFF
    h1 = ((((s0 <<  6) | (m24 >> 18)) & 0xFFFFFFFF) * 0x9E3779B1) & 0xFFFFFFFF
    h2 = ((((s0 << 12) | (m24 >> 12)) & 0xFFFFFFFF) * 0x9E3779B1) & 0xFFFFFFFF
    h3 = ((((s0 << 18) | (m24 >>  6)) & 0xFFFFFFFF) * 0x9E3779B1) & 0xFFFFFFFF
    new_st = ((s0 << 24) | m24) & 0xFFFFFFFF
    c0 = c64_3[(h0 >> 16) & 63]
    c2 = c64_3[(h2 >> 16) & 63]
    p0 = lut3[(((m24 >> 16) & 0xFC) ^ (h0 >> 24)) & 255]
    p1 = lut3[(((m24 >> 10) & 0xFC) ^ (h1 >> 24)) & 255]
    p2 = lut3[(((m24 >>  4) & 0xFC) ^ (h2 >> 24)) & 255]
    p3 = lut3[(((m24 <<  2) & 0xFC) ^ (h3 >> 24)) & 255]
    q  = [*affine_pair(p0, c0), *affine_pair(p1, c0),
          *affine_pair(p2, c2), *affine_pair(p3, c2)]
    return (new_st, q)
```

Notice why `bc > 0` resume works for 3-bit: `new_st` shifts in 24 bits per subtile (`4 × 24 = 96` bits per half), so after subtile `k = 3` of half `h = 1`, the low 32 bits of the bit stream equal `a2` verbatim (`words[strip_base + (bc * 6 - 1) * 32 + lane]`).

#### 5-bit (`b_bits = 5`, `wpb = 10`, 5 SoA planes per half)

Half `h ∈ {0, 1}` loads five `u32` plane words at `idx = strip_base + (bc * 10 + h * 5) * 32 + lane`:
`w0..w4 = words[idx + j * 32]` for `j ∈ {0..4}`.
These 160 bits unpack big-endian into four 40-bit codes `(hi8, lo32)`:

```python
codes_40b = [
    (w0 >> 24,         ((w0 <<  8) | (w1 >> 24)) & 0xFFFFFFFF),
    ((w1 >> 16) & 255, ((w1 << 16) | (w2 >> 16)) & 0xFFFFFFFF),
    ((w2 >>  8) & 255, ((w2 << 24) | (w3 >>  8)) & 0xFFFFFFFF),
    (w3 & 255,         w4),
]

def decode_8_5b(st: u32, hi8: u32, lo32: u32, lut5, c64_5) -> (new_st: u32, q: f32[8]):
    s0 = st
    b0 = ((hi8 << 2) | (lo32 >> 30)) & 1023
    b1 = (lo32 >> 20) & 1023
    b2 = (lo32 >> 10) & 1023
    b3 = lo32 & 1023
    h0 = (s0 * 0x9E3779B1) & 0xFFFFFFFF
    h1 = ((((s0 << 10) | b0) & 0xFFFFFFFF) * 0x9E3779B1) & 0xFFFFFFFF
    h2 = ((((s0 << 20) | (hi8 << 12) | (lo32 >> 20)) & 0xFFFFFFFF) * 0x9E3779B1) & 0xFFFFFFFF
    h3 = ((((s0 << 30) | (hi8 << 22) | (lo32 >> 10)) & 0xFFFFFFFF) * 0x9E3779B1) & 0xFFFFFFFF
    c0 = c64_5[(h0 >> 16) & 63]
    c2 = c64_5[(h2 >> 16) & 63]
    p0 = lut5[(b0 ^ (h0 >> 22)) & 1023]
    p1 = lut5[(b1 ^ (h1 >> 22)) & 1023]
    p2 = lut5[(b2 ^ (h2 >> 22)) & 1023]
    p3 = lut5[(b3 ^ (h3 >> 22)) & 1023]
    q  = [*affine_pair(p0, c0), *affine_pair(p1, c0),
          *affine_pair(p2, c2), *affine_pair(p3, c2)]
    return (lo32, q)
```

Because `new_st = lo32` and subtile `k = 3` of half `h = 1` has `lo32 = w4`, the state entering block `bc > 0` equals `words[strip_base + (bc * 10 - 1) * 32 + lane]` verbatim.

### 6.4 TFQ4 (`2.001 bpw` & `2.251 bpw`) Decode Pseudo-Code

TFQ4 stores row-major packed words and block scales `scales[R / 128, C / 128]` (`fp16`) using `e8p_lut[k_codes, 8]` (`fp16`, non-negative) and `table_kind` from `pw_v4_e8p_tables.bin`:

```python
def decode_8_true_e8p(abs_idx: u32, s7: u32, coset: u32, e8p_lut) -> f32[8]:
    # Shared by table_kind=2 (abs_idx in 0..255) and table_kind=3 (abs_idx in 0..1023)
    a      = e8p_lut[abs_idx]
    p_abs  = sum(int(math.floor(x)) for x in a) & 1
    s_full = s7 | (((bin(s7).count("1") ^ p_abs) & 1) << 7)
    out = [0.0] * 8
    for d in range(8):
        sgn = -1.0 if ((s_full >> d) & 1) else 1.0
        v = a[d] * sgn - 0.25
        out[d] = -v if coset else v
    return out

def decode_16_tfq4_200(w: u32, e8p_lut, table_kind: u32 = 0) -> f32[16]:
    # 2.001 bpw (b_bits = 2): 8 uint32 words per 128-col row-block
    out = [0.0] * 16
    if table_kind in (0, 1):
        for half_idx, code in enumerate((w & 0xFFFF, w >> 16)):
            abs_idx, sign_mask = code & 0xFF, code >> 8
            for d in range(8):
                out[half_idx * 8 + d] = e8p_lut[abs_idx][d] * (-1.0 if ((sign_mask >> d) & 1) else 1.0)
    elif table_kind in (2, 3):
        # table_kind=3's first 256 rows are bit-identical to table_kind=2
        for half_idx, code in enumerate((w & 0xFFFF, w >> 16)):
            out[half_idx * 8 : half_idx * 8 + 8] = decode_8_true_e8p(
                code & 0xFF, (code >> 8) & 0x7F, (code >> 15) & 1, e8p_lut
            )
    else:
        raise ValueError(f"Unsupported PWV4 table_kind={table_kind}")
    return out

def decode_128_tfq4_225(blk_words: u32[9], e8p_lut) -> f32[128]:
    # 2.251 bpw (b_bits = 225, requires table_kind = 3): 9 uint32 words per 128-col row-block
    hi_word = blk_words[8]
    out = [0.0] * 128
    for sub_w in range(8):
        w = blk_words[sub_w]
        for half_idx, low16 in enumerate((w & 0xFFFF, w >> 16)):
            j = sub_w * 2 + half_idx
            hi2 = (hi_word >> (2 * j)) & 3
            abs_idx = (hi2 << 8) | (low16 & 0xFF)
            out[j * 8 : j * 8 + 8] = decode_8_true_e8p(
                abs_idx, (low16 >> 8) & 0x7F, (low16 >> 15) & 1, e8p_lut
            )
    return out
```

For `b_bits = 2`, write `W_rot[r, wc*16 : wc*16 + 16] = decode_16_tfq4_200(words[r, wc], e8p_lut, table_kind) * f32(scales[r // 128, wc // 8])`. For `b_bits = 225`, write `W_rot[r, bc*128 : bc*128 + 128] = decode_128_tfq4_225(words[r, bc*9 : bc*9 + 9], e8p_lut) * f32(scales[r // 128, bc])`. In both cases, apply the exact same `128 × 128` tile inverse `2D FWHT-128` (§6.1) with `r_off = rb * 128, c_off = bc * 128`.
