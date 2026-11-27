# TFQ4 Sub-3-Bit Research Prototype (`2.0–2.25 bpw` `D4` & `E8P` Cold-Expert Lattices)

TFQ3 delivers 3-, 4-, and 5-bit data-free vector quantization using a 2D affine-coset codebook (`PWV3`) after `2D FWHT-128`. For large MoE checkpoints (288–512 experts), 60–80% of experts are routed infrequently ("cold experts"). **TFQ4** extends the same `2D FWHT-128` + universal synthetic Gaussian codebook pipeline into **2.0–2.25 bpw** via 4D ($D_4$) and 8D ($E_8$) sphere-packing lattices so 200B–300B+ MoEs fit comfortably inside 64–96 GB unified memory.

## 1. Design: 8D `E8P` Product-Sign Lattice (`4 KiB` Shared-Memory LUT)

After TFQ3's exact `2D FWHT-128` tile rotation and single `fp16` tile scale (`128 × 128` -> `0.001 bpw` overhead), rotated weights follow $\mathcal{N}(0, 1)^8$:

- **Packing (`2.000 bpw`):** every 8 weights share a **16-bit code** (`2` codes per `uint32_t` = `16` weights per `uint32_t`), selected by `PWV4.table_kind` (see [`docs/FORMAT_SPEC.md`](FORMAT_SPEC.md)):
  - **`table_kind = 0` / `1` (`--table-kind trained`, default):** low 8 bits index a trained `256 × 8` `fp16` positive-orthant Voronoi codebook (**`4,096` bytes**); high 8 bits hold 8 independent sign bits applied via branchless `half2` XOR (`256 × 256 = 65,536` codewords).
  - **`table_kind = 2` (`--table-kind e8p`, True QuIP# E8P `E8 + 1/4`):** low 8 bits index the 256 smallest-norm half-integer vectors in `|D8 + 1/2|` (exact in `fp16`, `4,096` bytes); bits `8..14` hold 7 free sign bits, sign bit 7 is derived from parity (`popcount(s7) ^ (sum(floor(a)) & 1)`), and bit `15` selects the `±1/4` coset — spanning `65,536` unique points strictly inside the shifted lattice $E_8 + \frac{1}{4}\mathbf{1}$ with an exact $O(256)$ nearest-point encoder on both Apple Silicon Metal and NVIDIA CUDA.
- **2.25 bpw tier (`16 KiB` LUT):** `1024 × 8` `fp16` magnitude codebook × `256` signs (`262,144` effective 8D codewords).
- **100% data-free:** both `kind=0` (trained from synthetic $|\mathcal{N}(0, 1)^8|$) and `kind=2` (exact $E_8 + \frac{1}{4}$ algebraic lattice) require zero calibration data.

## 2. Measured Quality on Real BF16 Transformer Weights `[MEASURED]`

Reproduce single-layer weight SNR via [`reference/tfq4_lattice.py`](../reference/tfq4_lattice.py) (measured on `1024 × 1024` tiles of real BF16 checkpoints vs. uniform affine baselines), and full-model WikiText-2 perplexity + exact fp64 full-vocab KL via [`tools/bench_quantizers.py`](../tools/bench_quantizers.py) (see [`docs/BENCHMARKS.md`](BENCHMARKS.md)):

> **Dense 2-bit caveat (`[MEASURED]`):** On a small dense 1.5B model (`Qwen/Qwen2.5-1.5B`, all 196 linear layers quantized to `2.001 bpw`), TFQ4 stays coherent where 2-bit RTN and 2-bit GPTQ collapse, and `table_kind = 2` (true QuIP# E8P) nearly halves data-free PPL vs `table_kind = 0` — but `2.0 bpw` is still severe for a dense 1.5B model and is primarily designed for **cold MoE experts**, not tiny dense models.

### 2.1 Matched `2.001 bpw` Comparison: `PWV4 table_kind = 0` (Trained) vs `table_kind = 2` (True QuIP# E8P `E8 + 1/4`) `[MEASURED]`

Evaluated on `Qwen/Qwen2.5-1.5B` (all 196 transformer linears at `2.001 bpw`, full `145 × 2048 = 296,815`-token WikiText-2 test split, exact fp64 full-vocab KL across all 296,815 tokens, 95% sequence-level bootstrap CIs; Hessian-LDLQ calibrated on `64 × 512` WikiText-2 train):

| Variant (`2.001 bpw`, `4 KiB` LUT) | Weight SNR | Output SNR | WikiText-2 PPL | Paired ΔNLL vs BF16 (95% CI) | Mean KL | KL p99 | Top-1 |
|---|---:|---:|---:|---|---:|---:|---:|
| BF16 Gold | — | — | **9.22** | `0.0000` | `0.0000` | `0.0000` | `100.00%` |
| `PWV4 kind=0` Trained `256×8` (Data-Free) | 10.09 dB | 10.86 dB | 344.19 | `+3.6198 [+3.5689, +3.6749]` | 3.6557 | 11.0220 | 22.46% |
| **`PWV4 kind=2` True QuIP# E8P (Data-Free, `s=0.97`)** | **10.36 dB** | **11.12 dB** | **176.10** | **`+2.9497 [+2.9118, +2.9868]`** | **2.9820** | **9.4430** | **28.13%** |
| `PWV4 kind=2` True QuIP# E8P (Data-Free, LS-scale) | 10.37 dB | 11.12 dB | 177.20 | `+2.9559 [+2.9181, +2.9931]` | 2.9883 | 9.4522 | 28.09% |
| `PWV4 kind=0` Trained `256×8` + Hessian-LDLQ | 8.24 dB | 15.18 dB | 33.62 | `+1.2937 [+1.1832, +1.4100]` | 1.3490 | 7.4729 | 52.38% |
| **`PWV4 kind=2` True QuIP# E8P + Hessian-LDLQ** | **8.50 dB** | **15.45 dB** | **31.05** | **`+1.2143 [+1.1134, +1.3268]`** | **1.2679** | **7.0537** | **53.77%** |

- On synthetic $\mathcal{N}(0, 1)^8$, `kind=2` achieves **`10.38 dB`** vs `10.11 dB` (`+0.27 dB`), and on real `Qwen2.5-1.5B` cuts data-free PPL by **48.8%** (`344.19 → 176.10`, `-0.670` nats paired ΔNLL) and Hessian-LDLQ PPL by **7.6%** (`33.62 → 31.05`, `-6.0%` mean KL).
- Use `tfq3 MODEL --bits 2 --table-kind e8p` (or `tfq4-convert` / `tfq4-convert-cuda --table-kind e8p`) to emit `table_kind = 2` checkpoints (100% SHA-256 bit-identical between Metal and CUDA). The default remains `--table-kind trained` (`table_kind = 0`) to preserve backwards compatibility with existing `PWV4` runtime decoders.

### 2.2 Single-Layer Synthetic & Real Weight SNR vs Baselines `[MEASURED]`

| Quantizer | Effective Rate | Synthetic Gaussian SNR | Qwen2.5-1.5B `L0.mlp.down` SNR | Llama-3.1-8B `L15.mlp.down` SNR | Output Cosine (`Y-cos`) |
|---|---:|---:|---:|---:|---:|
| Uniform Affine 2-bit (g64) | 2.50 bpw | 6.93 dB | 6.75 dB | 6.78 dB | 0.911 |
| **TFQ4-D4 (256 codes, 2 KiB LUT)** | **2.00 bpw** | 9.37 dB | **9.37 dB** | **9.36 dB** | 0.943 |
| **TFQ4-E8P (256×256, 4 KiB LUT)** | **2.00 bpw** | 10.20 dB | **10.21 dB (+3.46 dB)** | **10.20 dB (+3.42 dB)** | **0.951** |
| **TFQ4-D4 (512 codes, 4 KiB LUT)** | **2.25 bpw** | 10.34 dB | 10.34 dB | 10.34 dB | 0.956 |
| **TFQ4-E8P (1024×256, 16 KiB LUT)** | **2.25 bpw** | 11.64 dB | **11.65 dB (+4.90 dB)** | **11.64 dB (+4.86 dB)** | **0.965** |
| Uniform Affine 3-bit (g64) | 3.50 bpw | 14.31 dB | 14.14 dB | 14.17 dB | 0.981 |

Notice that `2D FWHT-128` produces virtually identical SNR on synthetic Gaussian (`10.20 dB`), real Qwen2.5 (`10.21 dB`), and real Llama-3.1 (`10.20 dB`), and **TFQ4-E8P at `2.00 bpw` gains `+3.46 dB` SNR over 2-bit affine while being 20% smaller**.

## 3. Measured GPU Decode Throughput `[MEASURED]` (Apple M5 Max Metal & NVIDIA GB10 CUDA)

- **Apple Silicon Metal** ([`src/kernels/metal/tfq4_metal.metal`](../src/kernels/metal/tfq4_metal.metal), `make test-tfq4-metal`, `fp64` parity **`cos = 0.99999998`**, **bit-exact `rows=1 == rows=16`**):

| Projection Shape | Bit Rate | Footprint | `rows=1` Latency | `rows=1` BW | `rows=16` Latency |
|---|---:|---:|---:|---:|---:|
| `Qwen27 gate+up [34816 × 5120]` | 2.001 bpw | 42.5 MiB | **0.124 ms** | **359.9 GB/s** | 1.068 ms |
| `Qwen27 down [5120 × 17408]` | 2.001 bpw | 21.3 MiB | **0.112 ms** | **198.5 GB/s** | 1.392 ms |
| `MoE-8E gate+up [32768 × 4096]` | 2.001 bpw | 32.0 MiB | **0.086 ms** | **390.2 GB/s** | 0.757 ms |

- **NVIDIA GB10 (`sm_121`) CUDA** ([`src/tools/tfq4_e8_bench.cu`](../src/tools/tfq4_e8_bench.cu), `fp64` parity **`cos = 1.00000000`**):

| Projection Shape | Bit Rate | Footprint | Latency (`rows=1`) | Achieved Bandwidth |
|---|---:|---:|---:|---:|
| `MoE-8E gate+up [32768 × 4096]` | 2.00 bpw | 32.0 MiB | **0.119 ms** | **281.9 GB/s** |
| `Qwen27 gate+up [34816 × 5120]` | 2.00 bpw | 42.5 MiB | **0.173 ms** | **258.0 GB/s** |
| `Qwen27 down [5120 × 17408]` | 2.00 bpw | 21.3 MiB | **0.070 ms** | **318.9 GB/s** |

## 3.5 Live Cross-Platform `TFP4` v4 Conversion `[MEASURED]` (`tfq4-convert` / `tfq4-convert-cuda` / `tfq3 MODEL --bits 2`)

Deterministic 16 KiB-aligned `pw_v4_e8p_tables.bin` (`PWV4 v4`) + `*.tffp4` (`TFP4 v4`) shards validated via `tfq3-inspect`:

| Model | Flags | Backend | Tensors | TFQ4 Size | Rate | Mean SNR | Time |
|---|---|---|---:|---:|---:|---:|---:|
| **Qwen2.5-1.5B** (BF16) | `--bits 2 --quantize-embed` | Metal (M5 Max) | 197 | **0.360 GiB** | 2.001 bpw | **10.09 dB** | **2.4 s** |
| **Qwen3.8-27B-DFlash2** | `--bits 2 --quantize-drafter` | Metal (M5 Max) | 47 | **0.419 GiB** | 2.001 bpw | **10.07 dB** | **2.8 s** |
| **Qwen3.8-27B-DFlash2** | `--bits 2 --quantize-drafter` | CUDA (GB10) | 47 | **0.419 GiB** | 2.001 bpw | **10.07 dB** *(exact parity)* | **6.2 s** |

## 4. Projected MoE Memory Footprints `[PROJECTED / NOT VERIFIED]` (Hot TFQ3 4b/3b + Cold TFQ4-E8P 2.0b)

> All rows in §4 are arithmetic footprint projections only; hybrid 2b/3b/4b MoE perplexity has not been measured yet.

| Model | Schedule | Avg Expert Rate | Projected Total Size | Target Hardware Class |
|---|---|---:|---:|---|
| **GLM-5.3-Flash** (288E MoE) | TFQ3 Current (`5/4/3b`) | 3.36 bpw | 130.3 GiB | 192 GB Mac / H200 / 2× GPU |
| **GLM-5.3-Flash** (288E MoE) | TFQ4 Hybrid (`15% 4b / 25% 3b / 60% 2b`) | 2.55 bpw | **101.6 GiB** | **128 GB Mac / GB10** |
| **GLM-5.3-Flash** (288E MoE) | TFQ4 Ultra (`10% 4b / 90% 2b`) | 2.20 bpw | **89.2 GiB** | **96–128 GB Mac / GB10** |
| **Qwen3.8-Flash-Next** (512E MoE) | TFQ3 Current (`5/4/3b`) | 3.41 bpw | 78.0 GiB *(79.0 GiB w/ overhead)* | 96–128 GB Mac / GB10 |
| **Qwen3.8-Flash-Next** (512E MoE) | TFQ4 Hybrid (`20% 4b / 80% 2b`) | 2.40 bpw | **64.1 GiB** | **96 GB Mac / 2× 48 GB** |
| **Qwen3.8-Flash-Next** (512E MoE) | TFQ4 Compact (`20% 3b / 80% 2b`) | 2.20 bpw | **61.3 GiB** | **64–96 GB Unified** |
