# TFQ4 Sub-3-Bit Quantization (`2.00` & `2.25 bpw` `E8P` + Hybrid Hot/Cold MoE)

TFQ3 delivers 3-, 4-, and 5-bit data-free vector quantization using a 2D affine-coset codebook (`PWV3`) after `2D FWHT-128`. For large MoE checkpoints (128–512 experts), 60–80% of experts are routed infrequently ("cold experts"). **TFQ4** extends the exact same `2D FWHT-128` + universal synthetic codebook pipeline into **`2.001 bpw`** and **`2.251 bpw`** via 8D ($E_8$) sphere-packing lattices so 200B–300B+ MoEs fit comfortably inside 64–128 GB unified memory while sharing a single input/output rotation with TFQ3 hot/warm experts.

## 1. Design: 8D `E8P` Product-Sign Lattices (`4 KiB` & `16 KiB` Shared-Memory LUTs)

After TFQ3's exact `2D FWHT-128` tile rotation and single `fp16` tile scale (`128 × 128` -> `0.001 bpw` overhead), rotated weights follow $\mathcal{N}(0, 1)^8$:

- **`2.001 bpw` (`b_bits = 2`, `8` `uint32` words per 128 cols):** every 8 weights share a **16-bit code** (`2` codes per `uint32_t` = `16` weights per `uint32_t`), selected by `PWV4.table_kind` (see [`docs/FORMAT_SPEC.md`](FORMAT_SPEC.md)):
  - **`table_kind = 0` / `1` (`--table-kind trained`, default, `16 KiB` file):** low 8 bits index a trained `256 × 8` `fp16` positive-orthant Voronoi codebook (**`4,096` bytes**); high 8 bits hold 8 independent sign bits applied via branchless `half2` XOR (`256 × 256 = 65,536` codewords).
  - **`table_kind = 2` (`--table-kind e8p`, True QuIP# E8P-256 `E8 + 1/4`, `16 KiB` file):** low 8 bits index the 256 smallest-norm half-integer vectors in `|D8 + 1/2|` (exact in `fp16`, `4,096` bytes); bits `8..14` hold 7 free sign bits, sign bit 7 is derived from parity (`popcount(s7) ^ (sum(floor(a)) & 1)`), and bit `15` selects the `±1/4` coset — spanning `65,536` unique points strictly inside $E_8 + \frac{1}{4}\mathbf{1}$ (`base_scale = 0.97f`).
- **`2.251 bpw` (`b_bits = 225`, `9` `uint32` words per 128 cols, `table_kind = 3`, `--bits 2.25` / `--table-kind e8p1024`, `32 KiB` file):**
  - **Algebraic deterministic construction:** enumerate all $m_d \in \{0, 1, 2, 3\}$ ($a_d = m_d + 0.5$, $4^8 = 65{,}536$ tuples), keep $\sum (2m_d + 1)^2 \le 68$, sort by $(n^2 \times 4, \max m, \text{lex } m)$, take the first **`1024`** vectors (`1024 × 8 × 2 = 16,384` bytes in `fp16`, `262,144` unique points strictly in $E_8 + \frac{1}{4}\mathbf{1}$, `base_scale = 0.84f`).
  - **Unified dual-decode table:** the first `256` entries of `table_kind = 3` are **100% bit-identical** to `table_kind = 2`, so a single 32 KiB `pw_v4_e8p_tables.bin` (`table_kind = 3`) decodes **both** `2.001 bpw` (`0..255`) and `2.251 bpw` (`0..1023`).
  - **Word-aligned `9`-word block packing (per 128-col row-block):** `w[0..7]` packs `low16(v_{2j}) | (low16(v_{2j+1}) << 16)` identically to `2.001 bpw`; `w[8]` packs the 16 two-bit high indices `hi2_j = (abs_idx_j >> 8) & 3` at bits `[2j+1 : 2j]`.
  - **Shared-memory budget:** `16 KiB` fp16 `sm_lut[1024]` + `1 KiB` parity + `4 KiB` `sm_x` + `8 KiB` `sm_out` = `26.25–28 KiB <= 32 KiB` threadgroup/shared memory on both Apple Silicon Metal and NVIDIA CUDA.
- **100% data-free & cross-platform bit-identical:** `table_kind = 0`, `2`, and `3` require zero calibration data and produce **100% SHA-256 byte-identical** `PWV4`, `TFP4`, and hybrid `TFPW` shards on Apple Silicon Metal and NVIDIA CUDA.

## 2. Measured Quality on Real BF16 Transformer Weights `[MEASURED]`

Reproduce single-layer weight SNR via [`reference/tfq4_lattice.py`](../reference/tfq4_lattice.py), and full-model WikiText-2 perplexity + exact fp64 full-vocab KL via [`tools/bench_quantizers.py`](../tools/bench_quantizers.py) (see [`docs/BENCHMARKS.md`](BENCHMARKS.md)):

> **Dense sub-3-bit caveat (`[MEASURED]`):** On a small dense 1.5B model (`Qwen/Qwen2.5-1.5B`, all 196 linear layers quantized), `table_kind = 3` (`2.251 bpw`) cuts data-free PPL by **73.2%** vs `table_kind = 2` (`176.10 → 47.25`) and cuts Hessian-LDLQ PPL by **48.5%** (`31.05 → 16.00`, `66.62%` Top-1, `16.92 dB` Out-SNR — where 2-bit GPTQ-LDLQ at identical `2.250 bpw` collapses to `772,872` PPL). Even so, uniform `2.0–2.25 bpw` on *all* layers of a small 1.5B dense model still trails 3–4 bit; TFQ4's primary production target is **cold MoE experts** paired with 4b/3b hot experts (§4).

### 2.1 Matched Sub-3-Bit Comparison on `Qwen/Qwen2.5-1.5B`: `table_kind = 0`, `2`, and `3` (`2.001` & `2.251 bpw`) `[MEASURED]`

Evaluated on `Qwen/Qwen2.5-1.5B` (all 196 transformer linears quantized, BF16 embeddings/norms/lm_head, full `145 × 2048 = 296,815`-token WikiText-2 test split, exact fp64 full-vocab KL across all 296,815 tokens, 95% sequence-level bootstrap CIs; Hessian-LDLQ calibrated on disjoint `64 × 512` WikiText-2 train):

| Variant (`Qwen2.5-1.5B`, 196 linears) | Eff. Rate | LUT | Weight SNR | Output SNR | WikiText-2 PPL | Paired ΔNLL vs BF16 (95% CI) | Mean KL | KL p99 | Top-1 |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| BF16 Gold | 16.000 bpw | — | — | — | **9.22** | `0.0000` | `0.0000` | `0.0000` | `100.00%` |
| `PWV4 kind=0` Trained `256×8` (Data-Free) | 2.001 bpw | 4 KiB | 10.09 dB | 10.86 dB | 344.19 | `+3.6198 [+3.5689, +3.6749]` | 3.6557 | 11.0220 | 22.46% |
| `PWV4 kind=2` True E8P-256 (Data-Free, `s=0.97`) | 2.001 bpw | 4 KiB | 10.36 dB | 11.12 dB | 176.10 | `+2.9497 [+2.9118, +2.9868]` | 2.9820 | 9.4430 | 28.13% |
| `PWV4 kind=2` True E8P-256 (Data-Free, LS-scale) | 2.001 bpw | 4 KiB | 10.37 dB | 11.12 dB | 177.20 | `+2.9559 [+2.9181, +2.9931]` | 2.9883 | 9.4522 | 28.09% |
| **`PWV4 kind=3` True E8P-1024 (Data-Free, `s=0.84`)** | **2.251 bpw** | **16 KiB** | **11.73 dB** | **12.51 dB** | **47.25** | **`+1.6342 [+1.6056, +1.6622]`** | **1.6655** | **7.0625** | **46.64%** |
| `PWV4 kind=0` Trained `256×8` + Hessian-LDLQ | 2.001 bpw | 4 KiB | 8.24 dB | 15.18 dB | 33.62 | `+1.2937 [+1.1832, +1.4100]` | 1.3490 | 7.4729 | 52.38% |
| `PWV4 kind=2` True E8P-256 + Hessian-LDLQ | 2.001 bpw | 4 KiB | 8.50 dB | 15.45 dB | 31.05 | `+1.2143 [+1.1134, +1.3268]` | 1.2679 | 7.0537 | 53.77% |
| **`PWV4 kind=3` True E8P-1024 + Hessian-LDLQ** | **2.251 bpw** | **16 KiB** | **9.82 dB** | **16.92 dB** | **16.00** | **`+0.5514 [+0.5028, +0.6077]`** | **0.6026** | **4.3775** | **66.62%** |

- Adding just **`+0.25 bpw`** (`kind=2 → kind=3`, `2.001 → 2.251 bpw`) yields **`+1.37 dB`** weight SNR (`10.36 → 11.73 dB`), **`+1.39 dB`** data-free output SNR (`11.12 → 12.51 dB`), a **73.2% reduction in data-free PPL** (`176.10 → 47.25`, `-44.1%` mean KL, `+18.5 pp` Top-1), and with Hessian-LDLQ a **48.5% reduction in PPL** (`31.05 → 16.00`, **`-52.5%` mean KL**, `16.92 dB` held-out output SNR).
- CLI usage:
  - `tfq3 MODEL --bits 2 --table-kind e8p` → `table_kind = 2` (`2.001 bpw`, `16 KiB` `PWV4`)
  - `tfq3 MODEL --bits 2.25` (or `--table-kind e8p1024`) → `table_kind = 3` (`2.251 bpw`, `32 KiB` `PWV4`, `b_bits = 225`)

### 2.2 Single-Layer Synthetic & Real Weight SNR vs Baselines `[MEASURED]`

| Quantizer | Effective Rate | Synthetic Gaussian SNR | Qwen2.5-1.5B `L0.mlp.down` SNR | Llama-3.1-8B `L15.mlp.down` SNR | Output Cosine (`Y-cos`) |
|---|---:|---:|---:|---:|---:|
| Uniform Affine 2-bit (g64) | 2.50 bpw | 6.93 dB | 6.75 dB | 6.78 dB | 0.911 |
| TFQ4-D4 (256 codes, 2 KiB LUT) | 2.00 bpw | 9.37 dB | 9.37 dB | 9.36 dB | 0.943 |
| TFQ4-E8P `kind=0` (Trained `256×8`, 4 KiB LUT) | 2.001 bpw | 10.19 dB | 10.21 dB (+3.46 dB) | 10.20 dB (+3.42 dB) | 0.951 |
| **TFQ4-E8P `kind=2` (True E8P-256, 4 KiB LUT)** | **2.001 bpw** | **10.38 dB** | **10.36 dB** | — | — |
| **TFQ4-E8P `kind=3` (True E8P-1024, 16 KiB LUT)** | **2.251 bpw** | **11.76 dB** | **11.73 dB (+4.98 dB)** | — | — |
| Uniform Affine 3-bit (g64) | 3.50 bpw | 14.31 dB | 14.14 dB | 14.17 dB | 0.981 |

## 3. Measured GPU Decode Throughput `[MEASURED]` (Apple M5 Max Metal & NVIDIA GB10 CUDA)

- **Apple Silicon Metal** ([`src/kernels/metal/tfq4_metal.metal`](../src/kernels/metal/tfq4_metal.metal), `make test-tfq4-metal`, `fp64` parity **`cos = 0.99999998`**, **bit-exact `rows=1 == rows=16 == rows=17`**):

| Kernel / Table Kind | Projection Shape | Bit Rate | Footprint | `rows=1` Latency | `rows=1` BW | `rows=16` Latency |
|---|---|---:|---:|---:|---:|---:|
| `true-E8P` (`kind=2`, 4 KiB LUT) | `MoE-8E gate+up [32768 × 4096]` | 2.001 bpw | 32.0 MiB | **0.095 ms** | **353.6 GB/s** | 0.936 ms |
| `true-E8P` (`kind=2`, 4 KiB LUT) | `Qwen27 gate+up [34816 × 5120]` | 2.001 bpw | 42.5 MiB | **0.151 ms** | **295.4 GB/s** | 1.302 ms |
| `true-E8P` (`kind=2`, 4 KiB LUT) | `Qwen27 down [5120 × 17408]` | 2.001 bpw | 21.3 MiB | **0.125 ms** | **178.4 GB/s** | 1.602 ms |
| **`true-E8P-1024` (`kind=3`, 16 KiB LUT)** | `MoE-8E gate+up [32768 × 4096]` | **2.251 bpw** | **36.0 MiB** | **0.168 ms** | **225.4 GB/s** | 0.992 ms |
| **`true-E8P-1024` (`kind=3`, 16 KiB LUT)** | `Qwen27 gate+up [34816 × 5120]` | **2.251 bpw** | **47.8 MiB** | **0.159 ms** | **314.8 GB/s** | 1.410 ms |
| **`true-E8P-1024` (`kind=3`, 16 KiB LUT)** | `Qwen27 down [5120 × 17408]` | **2.251 bpw** | **23.9 MiB** | **0.146 ms** | **171.2 GB/s** | 1.726 ms |

- **NVIDIA GB10 (`sm_121`) CUDA** ([`src/kernels/cuda/tfq4_cuda.cuh`](../src/kernels/cuda/tfq4_cuda.cuh), `make test-tfq4-cuda CUDA_ARCH=sm_121`, `fp64` parity **`cos = 0.99999998`**, **bit-exact `rows=1 == rows=16 == rows=17`**, `fp16` + `bf16` verified):

| Kernel / Table Kind | Projection Shape | Bit Rate | Footprint | `rows=1` Latency | `rows=1` BW | `rows=16` Latency |
|---|---|---:|---:|---:|---:|---:|
| `trained` (`kind=0`, 4 KiB LUT) | `MoE-8E gate+up [32768 × 4096]` | 2.001 bpw | 32.0 MiB | **0.144 ms** | **232.9 GB/s** | 0.916 ms |
| `true-E8P` (`kind=2`, 4 KiB LUT) | `MoE-8E gate+up [32768 × 4096]` | 2.001 bpw | 32.0 MiB | **0.142 ms** | **236.9 GB/s** | 0.904 ms |
| `true-E8P` (`kind=2`, 4 KiB LUT) | `Qwen27 gate+up [34816 × 5120]` | 2.001 bpw | 42.5 MiB | **0.247 ms** | **180.4 GB/s** | 1.122 ms |
| `true-E8P` (`kind=2`, 4 KiB LUT) | `Qwen27 down [5120 × 17408]` | 2.001 bpw | 21.3 MiB | **0.150 ms** | **148.7 GB/s** | 0.994 ms |
| **`true-E8P-1024` (`kind=3`, 16 KiB LUT)** | `MoE-8E gate+up [32768 × 4096]` | **2.251 bpw** | **36.0 MiB** | **0.248 ms** | **152.4 GB/s** | 0.959 ms |
| **`true-E8P-1024` (`kind=3`, 16 KiB LUT)** | `Qwen27 gate+up [34816 × 5120]` | **2.251 bpw** | **47.8 MiB** | **0.294 ms** | **170.7 GB/s** | 1.270 ms |
| **`true-E8P-1024` (`kind=3`, 16 KiB LUT)** | `Qwen27 down [5120 × 17408]` | **2.251 bpw** | **23.9 MiB** | **0.166 ms** | **151.0 GB/s** | 1.209 ms |

## 3.5 Live Cross-Platform `TFP4` v4 Conversion `[MEASURED]` (`tfq4-convert` / `tfq4-convert-cuda`)

Deterministic 16 KiB-aligned `pw_v4_e8p_tables.bin` (`PWV4 v4`) + `*.tffp4` (`TFP4 v4`) shards validated via `tfq3-inspect` with **100% SHA-256 bit-identity** verified between Apple M5 Max Metal and NVIDIA GB10 CUDA (`sm_121`):

| Model | Flags | Backend | Tensors | TFQ4 Size | Rate | Mean SNR | Time | Cross-Platform Parity |
|---|---|---|---:|---:|---:|---:|---:|---|
| **Qwen2.5-1.5B** (BF16) | `--bits 2 --quantize-embed` (`kind=0`) | Metal (M5 Max) | 197 | 0.360 GiB | 2.001 bpw | 10.09 dB | 2.4 s | — |
| **Qwen2.5-1.5B** (BF16) | `--bits 2.25 --table-kind e8p1024` (`kind=3`) | **Metal (M5 Max)** | 196 | **0.343 GiB** | **2.251 bpw** | **11.73 dB** | **3.6 s** | **`SHA-256 = 53e790ff...`** |
| **Qwen2.5-1.5B** (BF16) | `--bits 2.25 --table-kind e8p1024` (`kind=3`) | **CUDA (GB10)** | 196 | **0.343 GiB** | **2.251 bpw** | **11.73 dB** | **3.4 s** | **100% byte-identical** |
| **Qwen3.8-27B-DFlash2** | `--bits 2 --quantize-drafter` (`kind=0`) | Metal / CUDA | 47 | 0.419 GiB | 2.001 bpw | 10.07 dB | 2.8 s / 6.2 s | 100% byte-identical |

## 4. Measured Hybrid Hot/Cold MoE (`TFQ3` 4b/3b + `TFQ4-E8P` 2.00b / 2.25b) `[MEASURED]`

Both `tfq3-inkling2-convert[-cuda]` and `tfq3-glm-convert[-cuda]` (and `bin/tfq3 --inkling2-mode` / `--glm-mode`) support hybrid MoE schedules (`heat432`, `heat42`, `heat43225`, `uniform2`, `uniform225`) inside standard `TFPW` v3 shards alongside `pw_v3_tables.bin` and `pw_v4_e8p_tables.bin`.

### 4.1 Real 24-Layer `Inkling-2-Mini` (`E=128`, `H=2304`, `I=1152`, 24.46B MoE Params: L0–L19 NVFP4 + L20–L23 MXFP8) `[MEASURED]`

Converted end-to-end across all 24 MoE layers on NVIDIA GB10 (`sm_121`), validated with `tfq3-inspect`, with **100% Metal == CUDA SHA-256 bit-identity** verified on both real L00 (NVFP4) and real L20 (MXFP8) shards:

| Mode | Expert Schedule (`E = 128`) | Avg Rate | 24-Layer MoE Size | Savings vs `4b` | Per-Tier Weight SNR (`L00–L23`) | Convert Time (GB10) | Metal == CUDA Parity |
|---|---|---:|---:|---:|---|---:|---|
| **`uniform4`** | `128 × 4b` | 4.001 bpw | 11.39 GiB (`12.23 GB`) | Baseline | `4b = 23.31 dB` | 379.4 s | Verified |
| **`heat43225`** | `19 × 4b / 32 × 3b / 77 × 2.25b` (`kind=3`) | **2.698 bpw** | **7.68 GiB (`8.25 GB`)** | **`-32.6%` (`-3.71 GiB`)** | `4b = 23.31`, `3b = 17.14`, `2.25b = 11.76 dB` | **189.1 s** (`2.0×`) | **100% SHA-256 identical** (L00 NVFP4 & L20 MXFP8) |
| **`heat432`** | `19 × 4b / 32 × 3b / 77 × 2.00b` (`kind=2`) | **2.548 bpw** | **7.26 GiB (`7.79 GB`)** | **`-36.3%` (`-4.14 GiB`)** | `4b = 23.31`, `3b = 17.14`, `2.00b = 10.38 dB` | **160.0 s** (`2.4×`) | **100% SHA-256 identical** (L00 NVFP4 & L20 MXFP8) |

### 4.2 Real Top-8 Router-Gated SwiGLU MoE Output SNR on `Inkling-2-Mini` (`T = 256` tokens, real `gate.weight` + `gate.bias`) `[MEASURED]`

Using data-free router-norm+bias heat ranking (zero calibration tokens), the top 40% hot/warm experts (`15% 4b + 25% 3b`) capture **73.0%** of real top-8 router selections on L00 (NVFP4) and **86.7%** on L20 (MXFP8):

| Layer (Source) | Mode | Avg bpw | MoE SwiGLU Output SNR | Output Cosine | Active-Expert Weight SNR | Router Traffic by Tier (`4b / 3b / cold`) |
|---|---|---:|---:|---:|---:|---|
| **L00 (NVFP4)** | `uniform4` | 4.001 | 19.30 dB | 0.99411 | 23.31 dB | `100.0% / 0% / 0%` |
| **L00 (NVFP4)** | **`heat43225`** | **2.698** | **11.89 dB (`+0.89 dB`)** | **0.96712** | **13.56 dB** | **`37.4% / 35.6% / 27.0%`** |
| **L00 (NVFP4)** | `heat432` | 2.548 | 11.00 dB | 0.95950 | 12.35 dB | `37.4% / 35.6% / 27.0%` |
| **L20 (MXFP8)** | `uniform4` | 4.001 | 18.64 dB | 0.99314 | 23.31 dB | `100.0% / 0% / 0%` |
| **L20 (MXFP8)** | **`heat43225`** | **2.698** | **12.67 dB (`+0.62 dB`)** | **0.97263** | **14.39 dB** | **`45.2% / 41.5% / 13.3%`** |
| **L20 (MXFP8)** | `heat432` | 2.548 | 12.05 dB | 0.96836 | 13.28 dB | `45.2% / 41.5% / 13.3%` |

## 5. Projected Large-MoE Memory Footprints `[PROJECTED / NOT VERIFIED]`

> All rows in §5 are arithmetic footprint projections only; full end-to-end serving perplexity for hybrid 2b/3b/4b large MoE checkpoints has not been measured yet.

| Model | Schedule | Avg Expert Rate | Projected Total Size | Target Hardware Class |
|---|---|---:|---:|---|
| **GLM-5.3-Flash** (288E MoE) | TFQ3 Current (`5/4/3b`) | 3.36 bpw | 130.3 GiB | 192 GB Mac / H200 / 2× GPU |
| **GLM-5.3-Flash** (288E MoE) | TFQ4 Hybrid `heat43225` (`15% 4b / 25% 3b / 60% 2.25b`) | 2.70 bpw | **107.0 GiB** | **128 GB Mac / GB10** |
| **GLM-5.3-Flash** (288E MoE) | TFQ4 Hybrid `heat432` (`15% 4b / 25% 3b / 60% 2.00b`) | 2.55 bpw | **101.6 GiB** | **128 GB Mac / GB10** |
| **GLM-5.3-Flash** (288E MoE) | TFQ4 Ultra `heat42` (`20% 4b / 80% 2.00b`) | 2.40 bpw | **96.3 GiB** | **128 GB Mac / GB10** |
| **Qwen3.8-Flash-Next** (512E MoE) | TFQ3 Current (`5/4/3b`) | 3.41 bpw | 78.0 GiB *(79.0 GiB w/ overhead)* | 96–128 GB Mac / GB10 |
| **Qwen3.8-Flash-Next** (512E MoE) | TFQ4 Hybrid `heat42` (`20% 4b / 80% 2.00b`) | 2.40 bpw | **64.1 GiB** | **96 GB Mac / 2× 48 GB** |
