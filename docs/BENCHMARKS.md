# TFQ Verified Benchmarks & Provenance

Every section and claim in this document is explicitly tagged with its verification level:
- **`[MEASURED]`** — Measured directly on real hardware with reproducible scripts. Where pinned JSON receipts exist, SHA-256 hashes of the script, model snapshot revision, and tokenized evaluation corpus are committed under [`benchmarks/`](../benchmarks/).
- **`[SOURCE-INSPECTED]`** — Code/ABI/interface checked and compiled or unit-tested against stubs, without full end-to-end serving hardware runs.
- **`[PROJECTED / NOT VERIFIED]`** — Analytical footprint projection or unverified target; never presented as measured quality.

---

## 1. Controlled Head-to-Head Quantizer Comparison (`Qwen/Qwen2.5-1.5B`) `[MEASURED]`

**Provenance receipt:** [`benchmarks/provenance_qwen25_1p5b.json`](../benchmarks/provenance_qwen25_1p5b.json) (`schema: tfq-provenance-v2`, `tools/bench_quantizers.py --seq-len 2048 --eval-seqs 0 --kl-seqs 0`)
- **Clean commit & binary provenance:** Recorded from a clean working tree (`repo_commit: 5079156`, `git_dirty: false`) with SHA-256 hashes of `tools/bench_quantizers.py`, `tools/tfq3_hessian.py`, the compiled `tfq3-generic-convert` / `tfq4-convert` binaries, every generated `.tffpw`/`.tffp4` shard, table file, and manifest, plus raw per-sequence `seq_nlls`.
- **Model:** `Qwen/Qwen2.5-1.5B` (HF snapshot `8faed761d45a263340a0528343f099c05c9a4323`), all **196** transformer linear layers quantized; `embed_tokens`, `lm_head`, RMSNorms, and biases kept **identical BF16** across every method.
- **Disjoint splits (`Salesforce/wikitext`, `wikitext-2-raw-v1`):**
  - Calibration (used **only** by calibrated methods): `32,768` tokens (`64 × 512`) from `train` (`sha256: 43bb22dff75d332b...`).
  - Evaluation: full WikiText-2 test split at `seq_len = 2048` — **`296,815` scored tokens** (`145 × 2048`, `sha256: eb2f38d8154654e9...`) with exact `fp64` NLL, **paired per-sequence `ΔNLL` vs BF16 with 2,000-sample paired bootstrap 95% CI**, and exact `fp64`-accumulated full-vocabulary (`151,936` logits) forward KL $D_{\text{KL}}(P_{\text{BF16}} \| P_q)$ over **all `296,815` tokens**.
  - **Real-Activation Output SNR:** Mean across all 196 layers of $10 \log_{10}(\|W X_{\text{test}}^T\|_F^2 / \|(W - \hat{W}) X_{\text{test}}^T\|_F^2)$ using **held-out test activations**.
- **Baseline scope:** Non-TFQ rows below are **in-house reference implementations** in `tools/bench_quantizers.py`: `GPTQ-LDLQ (in-house, no act-order, g128)` omits activation ordering and fixes min/max group scales before error feedback; `Scalar Lloyd-Max + LDLQ (in-house ref)` uses a 1D optimal Gaussian grid after 2D-FWHT (not EXL3's trellis codebook). External maintained baselines (GPTQModel, ExLlamaV3) are outside this script and can be compared against the committed token hashes and `seq_nlls` in `benchmarks/provenance_qwen25_1p5b.json` (and `bench_quantizers.py --dataset c4` supports a second corpus).
- **Real binaries tested:** `TFQ3-4b Data-Free`, `TFQ3-3b Data-Free`, and `TFQ4-E8P 2b Data-Free` were quantized by the compiled Metal binaries (`tfq3-generic-convert` / `tfq4-convert`), validated by `tfq3-inspect`, and decoded directly from `.tffpw`/`.tffp4` shards.

### 4-Bit Head-to-Head (`Gold BF16 PPL = 9.2198`, 95% CI `[8.8225, 9.6318]`, `NLL = 2.22136`)

| Quantizer | Eff. bpw | Calib? | Weight SNR | Held-Out Out-SNR | WikiText-2 PPL (296,815 tok) | Paired `ΔNLL` vs BF16 (95% CI) | Mean KL (all tok) | p99 KL | Same Top-1 | Conv Time |
|---|---:|---|---:|---:|---:|---|---:|---:|---:|---:|
| BF16 Reference (Gold) | 16.000 | No | — | — | **9.2198** | `0.00000` | — | — | — | — |
| RTN / Affine 4b (in-house, `g64`) | 4.500 | No | 20.57 dB | 21.61 dB | 9.9375 | `+0.07495 [+0.07266, +0.07735]` | 0.09606 | 0.6645 | 85.41% | 0.5 s |
| RTN / Affine 4b (in-house, `g128`) | 4.250 | No | 19.52 dB | 20.56 dB | 10.2249 | `+0.10347 [+0.10046, +0.10652]` | 0.12230 | 0.8601 | 83.47% | 0.4 s |
| AWQ-style Salient-Scale 4b (in-house, `g128`) | 4.250 | Yes | 17.64 dB | 22.42 dB | 9.8361 | `+0.06470 [+0.06281, +0.06658]` | 0.07645 | 0.6184 | 86.65% | 10.9 s |
| GPTQ-LDLQ 4b (in-house, no act-order, `g128`) | 4.250 | Yes | 17.25 dB | 24.21 dB | 9.7188 | `+0.05270 [+0.05039, +0.05500]` | 0.05999 | 0.5435 | 88.42% | 62.2 s |
| **TFQ3-4b Data-Free (real `.tffpw`)** | **4.001** | **No** | **23.37 dB** | **24.41 dB** | **9.6436** | **`+0.04494 [+0.04313, +0.04679]`** | **0.05507** | **0.3848** | **88.64%** | **29.9 s** |
| **TFQ3-4b + Hessian-LDLQ (same `TFPN`)** | **4.001** | Yes | 19.77 dB | **25.99 dB** | **9.5979** | **`+0.04019 [+0.03790, +0.04249]`** | 0.05984 | 0.5213 | 88.39% | 241.4 s |
| Scalar Lloyd-Max + LDLQ 4b (in-house ref) | 4.001 | Yes | 18.03 dB | 25.94 dB | **9.5577** | **`+0.03599 [+0.03400, +0.03786]`** | **0.04669** | 0.4300 | **89.66%** | 94.6 s |

### 3-Bit Head-to-Head (`Gold BF16 PPL = 9.2198`)

| Quantizer | Eff. bpw | Calib? | Weight SNR | Held-Out Out-SNR | WikiText-2 PPL (296,815 tok) | Paired `ΔNLL` vs BF16 (95% CI) | Mean KL (all tok) | p99 KL | Same Top-1 | Conv Time |
|---|---:|---|---:|---:|---:|---|---:|---:|---:|---:|
| RTN / Affine 3b (in-house, `g64`) | 3.500 | No | 13.95 dB | 14.97 dB | 14.5073 | `+0.45329 [+0.44546, +0.46163]` | 0.50406 | 3.0593 | 67.52% | 0.4 s |
| AWQ-style Salient-Scale 3b (in-house, `g128`) | 3.250 | Yes | 11.01 dB | 15.80 dB | 13.6294 | `+0.39088 [+0.38220, +0.39925]` | 0.42469 | 3.0958 | 70.98% | 8.4 s |
| GPTQ-LDLQ 3b (in-house, no act-order, `g128`) | 3.250 | Yes | 10.53 dB | 17.47 dB | 12.6126 | `+0.31334 [+0.30352, +0.32380]` | 0.34027 | 2.8809 | 74.27% | 62.7 s |
| **TFQ3-3b Data-Free (real `.tffpw`)** | **3.001** | **No** | **17.31 dB** | **18.31 dB** | **11.2579** | **`+0.19971 [+0.19514, +0.20413]`** | **0.22068** | **1.5705** | **77.94%** | **22.8 s** |
| TFQ3-3b + Hessian-LDLQ (`beam=8`, `TFPN`) | 3.001 | Yes | 13.93 dB | **19.83 dB** | 11.9019 | `+0.25534 [+0.24800, +0.26289]` | 0.29709 | 2.3129 | 75.25% | 226.5 s |
| Scalar Lloyd-Max + LDLQ 3b (in-house ref) | 3.001 | Yes | 12.61 dB | **20.26 dB** | **10.9609** | **`+0.17298 [+0.16512, +0.18215]`** | **0.20569** | 1.7974 | **79.38%** | 99.8 s |

### Sub-3-Bit Stress Test (`All 196 Layers Uniform 2.00–2.25 bpw on 1.5B Dense`)

| Quantizer | Eff. bpw | Calib? | Weight SNR | Held-Out Out-SNR | WikiText-2 PPL (296,815 tok) | Paired `ΔNLL` vs BF16 (95% CI) | Mean KL (all tok) | p99 KL | Same Top-1 | Conv Time |
|---|---:|---|---:|---:|---:|---|---:|---:|---:|---:|
| RTN / Affine 2b (in-house, `g64`) | 2.500 | No | 6.54 dB | 7.34 dB | 44,228.2 | `+8.47576 [+8.40813, +8.54594]` | 8.49874 | 18.5430 | 1.91% | 0.3 s |
| GPTQ-LDLQ 2b (in-house, no act-order, `g128`) | 2.250 | Yes | 2.40 dB | 9.10 dB | 772,872.3 | `+11.33651 [+11.28333, +11.39024]` | 11.37398 | 21.5028 | 0.33% | 66.4 s |
| TFQ4-E8P `kind=0` 2b Data-Free (real `.tffp4`) | 2.001 | No | 10.09 dB | 10.86 dB | 344.19 | `+3.61985 [+3.56894, +3.67487]` | 3.65567 | 11.0220 | 22.46% | 4.2 s |
| TFQ4 True E8P-256 `kind=2` Data-Free (real `.tffp4`) | 2.001 | No | 10.36 dB | 11.12 dB | 176.10 | `+2.9497 [+2.9118, +2.9868]` | 2.9820 | 9.4430 | 28.13% | 3.5 s |
| **TFQ4 True E8P-1024 `kind=3` Data-Free (real `.tffp4`)** | **2.251** | **No** | **11.73 dB** | **12.51 dB** | **47.25** | **`+1.6342 [+1.6056, +1.6622]`** | **1.6655** | **7.0625** | **46.64%** | **3.6 s** |
| TFQ4-E8P `kind=0` 2b + Hessian-LDLQ (same `TFP4`) | 2.001 | Yes | 8.24 dB | 15.18 dB | 33.62 | `+1.29374 [+1.18321, +1.40998]` | 1.34896 | 7.4729 | 52.38% | 210.5 s |
| TFQ4 True E8P-256 `kind=2` + Hessian-LDLQ (same `TFP4`) | 2.001 | Yes | 8.50 dB | 15.45 dB | 31.05 | `+1.2143 [+1.1134, +1.3268]` | 1.2679 | 7.0537 | 53.77% | — |
| **TFQ4 True E8P-1024 `kind=3` + Hessian-LDLQ (`TFP4`)** | **2.251** | **Yes** | **9.82 dB** | **16.92 dB** | **16.00** | **`+0.5514 [+0.5028, +0.6077]`** | **0.6026** | **4.3775** | **66.62%** | — |

### Honest Takeaways from the Controlled Comparison
- **Where data-free TFQ3 wins:** At `4.001 bpw` and `3.001 bpw` with **zero calibration data** in ~23–30 seconds, data-free TFQ3 beats our in-house RTN/Affine (`4.50`/`3.50 bpw`), AWQ-style (`4.25`/`3.25 bpw`), and no-act-order GPTQ-LDLQ (`4.25`/`3.25 bpw`) baselines with non-overlapping paired 95% ΔNLL CIs at both bit widths (`+0.04494` vs `+0.05270` nats at 4-bit, `-29%` p99 KL; `+0.19971` vs `+0.31334` nats at 3-bit, `-1.35 PPL`, `-35%` mean KL, `-45%` p99 KL).
- **Where full per-column Hessian LDLQ wins:** Per-column Hessian error propagation after 2D-FWHT (`Scalar Lloyd-Max + LDLQ` in-house ref) achieves lower PPL (`9.56` vs `9.64` at 4b; `10.96` vs `11.26` at 3b) and lower mean KL (`0.0467` vs `0.0551` at 4b; `0.2057` vs `0.2207` at 3b), while data-free TFQ3 still has lower p99 tail KL (`0.3848` vs `0.4300` at 4b; `1.5705` vs `1.7974` at 3b) and zero calibration dependency. In our Python `beam=8` block-128 `TFQ3+Hessian-LDLQ` mode, single-layer held-out Output SNR improves across every bit width (`+1.58 dB` at 4b, `+1.52 dB` at 3b, `+4.32 dB` at 2b) and lowers 4b PPL (`9.60` vs `9.64`), while at 3b the smaller Python `beam=8` gives up coding gain relative to the compiled `beam=256` converter (`PPL 11.90` vs `11.26`).
- **2-bit scope note:** At 2.001 bpw across all 196 layers of a 1.5B dense model, conditional Hessian-LDLQ (`TFQ4-E8P + Hessian-LDLQ`) cuts PPL by >10× vs data-free TFQ4 (`33.62` vs `344.19`, `+4.32 dB` Out-SNR, `52.4%` Top-1 agreement vs `22.5%`), though uniform 2-bit across *all* layers of a small 1.5B dense model still degrades perplexity substantially vs `9.22` BF16. `TFQ4-E8P` is intended for **cold MoE experts** in large 100B–300B MoE models paired with 4-bit hot experts and full-precision non-MoE layers — not uniform 2-bit dense compression.

---

## 2. Qwen3.8-27B (Dense 64-Layer Hybrid GDN + Attention) `[MEASURED — TensorFold runner, no per-sequence JSON receipt]`

> *Provenance note: Sections 2–4 report hardware measurements taken on the TensorFold runtime (`ashhart/TensorFold` PRs `#4953`, `#4964`, `#4974`, `#4986`). Unlike Section 1 (`benchmarks/provenance_qwen25_1p5b.json`), they do not currently have committed per-sequence JSON receipts or script/token hashes in `benchmarks/`.*

Measured directly on **Apple M5 Ultra** (TensorFold Metal) and **NVIDIA GB10** (`sm_121`, TensorFold CUDA) using the **exact same cross-platform TFQ3 checkpoint files**.
- **Calibration data used:** **None (100% data-free)**.
- **Weight SNR (`tfq3-qwen27-convert`):** `3b = 17.17 dB`, `4b = 23.37 dB` (`lm_head = 22.70 dB`), conversion time ~3.2 min on Apple M5 Ultra; `23.35 dB` (~5.5 s/layer) on NVIDIA GB10.
- **Perplexity evaluation:** Exact `fp64` log-softmax over **4,790 tokens** (`1,717` multi-domain + `3,071` unseen README/Zig/CHANGELOG).
- **KL divergence evaluation:** Exact `fp64` full-vocab $D_{\text{KL}}(P_{\text{8-bit}} \| P_q)$ across **all 248,320 logits** over **1,022 positions**.

| Variant | Eff. Linear bpw | Disk Size | 4,790-tok PPL | Unseen 3,071-tok PPL | Mean KL | Median KL | p95 KL | p99 KL | Max KL | Same Top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MLX 8-bit/64 (Gold) | 8.500 | 27.50 GiB | 5.2468 | 6.4485 | — | — | — | — | — | — |
| MLX 4-bit/64 | 4.500 | 14.98 GiB | 5.3694 | 6.6021 | 0.03352 | 0.01183 | 0.11810 | 0.27518 | 1.0228 | 91.19% |
| **TFQ3-4b (Pure BF16 residuals)** | **4.001** | **14.37 GiB** | **5.3680** | **6.5829** | **0.02316** | **0.00927** | **0.07941** | **0.15037** | **0.7051** | **91.68%** |
| **TFQ3-4b (8-bit residuals)** | **4.001** | **13.23 GiB** | **5.3681** | **6.5855** | **0.02308** | **0.00980** | **0.07763** | **0.14865** | **0.6968** | 91.29% |
| TFQ3-3.5b (8-bit residuals) | 3.501 | 11.59 GiB | 5.4619 | 6.6478 | 0.03779 | 0.01362 | 0.15305 | 0.25983 | 0.9311 | 91.39% |

### Throughput & Startup Load (Apple M5 Ultra) `[MEASURED]`

| Variant | Cold Load (`mmap`) | `p2048` Prefill | `B=1` Decode | `B=4` Decode | 32k×8 Startup Budget |
|---|---:|---:|---:|---:|---:|
| MLX 8-bit/64 | 3.27 s | 955.8 tok/s | 27.8 tok/s | 104.4 tok/s | — |
| MLX 4-bit/64 | 3.72 s | 1,869.8 tok/s | 48.4 tok/s | 179.4 tok/s | 43.90 GiB |
| **TFQ3-4b (Pure BF16 residuals)** | **0.53 s** | 1,184.4 tok/s | **45.4 tok/s** | 140.4 tok/s | **43.29 GiB** |
| **TFQ3-4b (8-bit residuals)** | **0.20 s** | 1,178.4 tok/s | 44.4 tok/s | **142.2 tok/s** | **42.15 GiB** |
| TFQ3-3.5b (8-bit residuals) | **0.19 s** | 1,222.9 tok/s | 45.3 tok/s | **142.4 tok/s** | — |

---

## 3. Qwen3.8-Flash-Next (48L + MTP, 512-Expert MoE) `[MEASURED — TensorFold runner, 2,004 tok, no per-sequence receipt]`

> *Sample-size note: On 2,004 scored tokens, the `0.011` PPL difference between `5.996` and `6.007` (8-bit gold) is well within sampling noise and indicates parity with 8-bit gold rather than statistical superiority.*

- **Footprint breakdown:**
  - Routed MoE (49 layers, mixed 5b/4b/3b averaging **`3.411 bpw`**): `48.98 GiB` (converted in `17.5 min`; `3b = 17.13 dB`, `4b = 23.33 dB`, `5b = 29.31 dB`).
  - Non-MoE + vision (kept at MLX 8-bit/32): `6.17 GiB`.
  - PLE N-gram embeddings: `23.84 GiB` (3-bit/32) or `29.80 GiB` (4-bit/32).
- **HeatAlloc mode note:** With `--trace <file>`, expert bit tiers use calibration routing counts (**trace-guided**). Without `--trace`, `tfq3-flashnext-convert` uses **100% data-free** router L2 norm + expert weight-scale sensitivity (`src/converters/moe_heat_rank.hpp`).

| Checkpoint | Disk Size | Main NLL (1,670 tok) | Held-Out NLL (334 tok) | Combined PPL (2,004 tok) | `p2048` Prefill | `B=1` Decode | `MTP d3` Decode | `d0 == d3` |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| MLX 8-bit/32 (Gold) | 189.04 GiB | 1.7315 | 2.1003 | 6.007 | 3,422 tok/s | 79.2 tok/s | 153.9 tok/s | Bit-exact |
| MLX 4-bit/32 | 105.50 GiB | 1.8324 | 2.2006 | 6.644 (`+0.101`) | 3,914 tok/s | 106.2 tok/s | 212.1 tok/s | Bit-exact |
| **TFQ3 (3.41 bpw MoE) + PLE 3-bit** | **78.99 GiB** | **1.7298** | **2.0978** | **5.996 (`-0.002`)** | **3,855 tok/s** | **82.9 tok/s** | **154.5 tok/s** | **Bit-exact** |
| TFQ3 (3.41 bpw MoE) + PLE 4-bit | 84.95 GiB | 1.7336 | 2.1210 | 6.038 (`+0.005`) | 3,856 tok/s | 82.9 tok/s | 160.1 tok/s | Bit-exact |

---

## 4. GLM-5.3-Flash (45L, 288-Expert MoE) `[MEASURED — TensorFold runner, 4,094 tok, no per-sequence receipt]`

### Explicit Footprint & Bit-Rate Breakdown (Apples-to-Apples)

The compact TFQ3 GLM pack (`139.6 GB` / `130.0 GiB`) is smaller than a uniform 4.0 bpw + full-precision non-MoE pack (`167.5 GB`–`175.6 GB`) **primarily because it uses fewer bits on deep MoE experts (`3.333 bpw`) and quantizes non-MoE projections (`nonmoe.tffpw`), NOT because of higher coding density at equal bits**:

| Profile / Command | Routed MoE Precision | Non-MoE Precision (Attn / Shared / Head) | Routed MoE Size | Non-MoE + Residuals | Total Size | Status |
|---|---|---|---:|---:|---:|---|
| Raw FP8 Source (`GLM-5.3-Flash-FP8E-MLX8-MTP4H`) | 8-bit FP8 | Native 8-bit / BF16 untouched | 304.5 GB | 15.3 GB | 319.8 GB (297.8 GiB) | Reference |
| **TFQ3 Compact** (`heat543` L3-6 + `heat43` L7-44 + `nonmoe --fast4b`) | **3.39 bpw avg** (`heat43` = `3.333 bpw`: 96×4b + 192×3b) | 4b/5b `nonmoe.tffpw` | 128.4 GB | 11.2 GB | **139.6 GB (130.0 GiB)** | `[MEASURED]` |
| **TFQ3 Matched-4b** (`--glm-mode uniform4 --keep-nonmoe-bf16`) | **4.001 bpw uniform** (all 288E @ 4b, mean **23.361 dB** across 36,288 matrices) | Native 8-bit / BF16 untouched (63 shards, 5,332 tensors) | **152.24 GB (141.79 GiB)** | **15.26 GB (14.21 GiB)** | **167.50 GB (156.00 GiB)** | `[MEASURED]` |

### Measured 4,094-Token WikiText-2 Quality (`TFQ3 Matched-4b` vs. Raw FP8 Source) `[MEASURED]`

Evaluated on Apple M5 Ultra Metal (`tf-glm-run`, `GLM_PROMPT_NLL=128`, exact same two 2,048-token `wikitext-2-raw-v1` test chunks = `4,094` scored tokens, 100% data-free conversion in ~42 min with zero calibration data):

| Checkpoint | Buffers in RAM | Chunk 0 NLL (2,047 tok) | Chunk 1 NLL (2,047 tok) | Mean NLL (4,094 tok) | ΔNLL vs FP8 | PPL (4,094 tok) | `d0 == d3` (MTP) |
|---|---:|---:|---:|---:|---:|---:|---|
| **FP8 Gold Source** | 318.60 GB | 0.661032 | 0.508224 | 0.584628 | — | 1.7943 | Reference |
| **TFQ3 Matched-4b** (`uniform4 --keep-nonmoe-bf16`, 100% data-free) | **166.36 GB** | **0.683424** | **0.528005** | **0.605715** | **+0.0211 nats** | **1.8326 (+2.1%)** | **Bit-exact** |

### Measured Runtime Throughput (Apple M5 Ultra, Compact `130.0 GiB` Pack) `[MEASURED]`

| Metric | MLX 4-bit (181 GB) | **TFQ3 Compact (139.6 GB / 130.0 GiB)** |
|---|---:|---:|
| `p2048` Prefill | 1,400 tok/s | **1,567 tok/s (+11.9%)** |
| Decode `B=1` | 75.5 tok/s | 69.6 tok/s |
| Live Speculative (`MTP + copies`, bit-exact `d3 == d0`) | — | **87.0–104.3 tok/s** (mean `95.2 tok/s`) |
| Live Speculative (`DFlash2` 4-bit, bit-exact) | — | **85.3 tok/s** |

> **Scope note:** An earlier draft listed a 4-prompt (`235`-token) PPL comparison (`22.61` vs `24.16` FP8); that noisy figure was removed and replaced by the 4,094-token WikiText-2 evaluation above. Full coding-task (SWE-bench / Docker-graded) and `panel72` KL comparison against Hessian-calibrated packs (e.g. EXL3/EAQ) remains pending (`[NOT VERIFIED]`).

---

## 5. NVIDIA CUDA & Apple Silicon Metal Kernel Benchmarks `[MEASURED]`

### Standalone NVIDIA CUDA Kernels (`src/kernels/cuda/tfq3_cuda.cuh`, NVIDIA GB10 `sm_121`, CUDA 13.0) `[MEASURED]`

Verified via `make test-cuda CUDA_ARCH=sm_121` and `python3 tests/vectors_test.py`:
- **Single-weight-read multi-row decode (`rows = 1..64`):** `tfq3_mma_sk_kernel<BITS, MTILES>` and `tfq3_mma_fused_mt_kernel<BITS, MTILES>` (`MTILES ∈ {1,2,3,4}`, zero register spills) decode each weight tile **once per CTA** and run up to 4 `m16` row tiles in an inner loop. Split-K is fixed purely by `(N, K)` (`ks = 1` with zero partial buffer for `N >= 32768`; `ks = 4` for `qkv`; `ks = 8` for `down`/`o_proj`), cutting peak FP32 partials at `rows = 64` from **`71.3 MiB` → `10.0 MiB` (7.1× smaller)** and guaranteeing **bit-exact `rows = 1 == rows = k` across `k = 1..64`** on all 4 Qwen3.8-27B shapes (`qkv`, `o_proj`, `gate+up`, `down`) plus `N = 40960 > 34816`.
- **Golden vectors & TP:** `PASS` vs `fp64`, `reference/vectors/tfq3_vectors.json` (including `r_off = 128` and `bc > 0` trellis state resume), zero-repack `Col-TP` / `Row-TP` (`tp = 2, 4`, including compact 1-block-resume `launch_dense_tp_sliced`), fp16 activation saturation, and mixed-bit MoE.

Earlier idle-GPU `rows=1`/`rows=16` GB10 measurements (before multi-row single-read optimization) plus updated single-weight-read multi-row `rows = 1..64` cold-L2 (64 MiB L2 flush) measurements on GB10 *(note: updated multi-row run was recorded while a background service used ~89% GPU, so idle GB10 latency is lower; peak partial traffic dropped from `71.3 MiB` to `0.0–10.0 MiB`)*:

| Qwen3.8-27B Projection | Shape `[N × K]` | `ks` (`rows ≤ 64`) | `yr` @ `rows=64` | Cold-L2 `rows=1` | Cold-L2 `rows=32` (1× weight read) | Cold-L2 `rows=64` (1× weight read) |
|---|---|---:|---:|---:|---:|---:|
| `gate+up` | `[34816 × 5120]` | `1` (fused) | **0.0 MiB** | 0.486 ms (`183.5 GB/s`) | **0.488 ms (`23.36 TF/s`)** | **0.919 ms (`24.82 TF/s`)** |
| `down` | `[5120 × 17408]` | `8` (`sk`) | **10.0 MiB** | 0.236 ms (`188.7 GB/s`) | **0.300 ms (`18.99 TF/s`)** | **0.399 ms (`28.57 TF/s`)** |
| `qkv` | `[10240 × 5120]` | `4` (`sk`) | **10.0 MiB** | 0.163 ms (`161.1 GB/s`) | **0.213 ms (`15.74 TF/s`)** | **0.231 ms (`29.05 TF/s`)** |
| `o_proj` | `[5120 × 6144]` | `8` (`sk`) | **10.0 MiB** | 0.140 ms (`112.7 GB/s`) | **0.148 ms (`13.63 TF/s`)** | **0.177 ms (`22.77 TF/s`)** |
| `lm_head` | `[248320 × 5120]` | `1` (fused) | **0.0 MiB** | 5.468 ms (`116.3 GB/s`) | **5.720 ms (`14.23 TF/s`)** | 12.914 ms (`12.60 TF/s`) |

### Live TensorFold CUDA Serving (`Qwen3.8-27B-TFQ3`, NVIDIA GB10 `sm_121`) `[MEASURED]`

| Kernel / Drafter | Cold `p3989` Prefill | Decode (`128 tok`) | Greedy Parity (`temp=0`) |
|---|---:|---:|---|
| `BM=16` baseline (`--no-drafts`) | 341.8 tok/s (`11.67 s`) | 14.9 tok/s | `sha=d4942ce0f4b2` |
| **`BM=64` Tensor Core (`--no-drafts`)** | **590.8 tok/s (`6.75 s`, 1.73×)** | **15.0 tok/s** | **100% bit-exact** (`sha=d4942ce0f4b2`) |
| **`BM=64` + `DFlash2`** | — | **52.5 tok/s (`3.50×`)** | **100% bit-exact** (`sha=d4942ce0f4b2`) |
| **`BM=64` + `DSpine`** | — | **59.6 tok/s (`3.97×`)** | **100% bit-exact** (`sha=d4942ce0f4b2`) |

### Standalone Apple Silicon Metal Kernels (`src/kernels/metal/tfq3_metal.metal` & `tfq4_metal.metal`, Apple M5 Max) `[MEASURED]`

Verified via `make test-metal`, `make test-tfq4-metal`, and `python3 tests/vectors_test.py`: `PASS` vs `fp64`, bit-exact `rows=1 == rows=16 == rows=32` on TFQ3 (`rows=1 == rows=16` on TFQ4), fp16 activation saturation, golden vectors, and `Col-TP=4` / `Row-TP=4`:

| Kernel / Projection | Shape `[N × K]` | `rows=1` (`sk8` / TFQ4) | `rows=16` Verify (bit-exact) | `rows=512` (`BM=32` prefill) |
|---|---|---:|---:|---:|
| TFQ3-4b `gate+up` | `[34816 × 5120]` | 0.526 ms (`169.6 GB/s`) | **0.518 ms (`172.1 GB/s`, 11.01 TFLOP/s)** | **13.77 TFLOP/s** (13.25 ms) |
| TFQ3-4b `down` | `[5120 × 17408]` | **0.181 ms (`246.9 GB/s`)** | **0.273 ms (`163.2 GB/s`, 10.44 TFLOP/s)** | **13.18 TFLOP/s** (6.93 ms) |
| TFQ3-4b `qkv` | `[10240 × 5120]` | **0.110 ms (`239.0 GB/s`)** | **0.162 ms (`161.4 GB/s`, 10.33 TFLOP/s)** | **13.67 TFLOP/s** (3.93 ms) |
| **TFQ4-E8P 2b `MoE-8E-gu`** | `[32768 × 4096]` | **0.086 ms (`390.2 GB/s`)** | 0.761 ms (`44.1 GB/s`, `r1==r16`) | — |
| **TFQ4-E8P 2b `Qwen27-gu`** | `[34816 × 5120]` | **0.122 ms (`366.1 GB/s`)** | 1.065 ms (`41.9 GB/s`, `r1==r16`) | — |
| **TFQ4-E8P-1024 2.25b `Qwen27-gu`** (`kind=3`) | `[34816 × 5120]` | **0.159 ms (`314.8 GB/s`)** | 1.410 ms (`35.6 GB/s`, `r1==r16`) | — |
| **TFQ4-E8P-1024 2.25b `MoE-8E-gu`** (`kind=3`) | `[32768 × 4096]` | **0.168 ms (`225.4 GB/s`)** | 0.992 ms (`38.1 GB/s`, `r1==r16`) | — |

---

## 6. Model-Agnostic Converter & Cross-Platform Parity `[MEASURED]`

| Model | Mode / Flags | Backend | Tensors | Bit Allocation | Output Size | Mean SNR | Time |
|---|---|---|---:|---|---:|---:|---:|
| **Qwen2.5-1.5B** (BF16) | `--bits 4 --quantize-embed` | Metal (M5 Max) | 197 | all 4b (`4.001 bpw`) | 0.719 GiB | **23.36 dB** | 23.2 s |
| **Qwen2.5-1.5B** (BF16) | `--target-gib 0.65 --quantize-embed` | Metal (M5 Max) | 197 | `3b:44 / 4b:146 / 5b:7` (`3.614 bpw`) | **0.650 GiB** | `17.33 / 23.35 / 29.18 dB` | 23.2 s |
| **Qwen2.5-1.5B** (BF16) | `--bits 2 --quantize-embed` (TFQ4-E8P) | Metal (M5 Max) | 197 | all 2b-E8P (`2.001 bpw`) | **0.360 GiB** | **10.09 dB** | **1.3 s** |
| **Qwen2.5-1.5B** (BF16) | `--bits 2.25 --table-kind e8p1024` (`kind=3`) | **Metal / CUDA (GB10)** | 196 | all 2.25b-E8P-1024 (`2.251 bpw`) | **0.343 GiB** | **11.73 / 11.73 dB** *(SHA-256 identical)* | **3.6 s / 3.4 s** |
| **Qwen3.8-27B-DFlash2** | `--quantize-drafter --bits 4` | Metal / CUDA (GB10) | 47 | all 4b (`4.001 bpw`) | 0.837 GiB | **23.32 / 23.32 dB** | 27.7 s / 26.2 s |
| **Qwen3.8-27B-DFlash2** | `--quantize-drafter --bits 2` (TFQ4-E8P) | Metal / CUDA (GB10) | 47 | all 2b-E8P (`2.001 bpw`) | **0.419 GiB** | **10.07 / 10.07 dB** | 2.8 s / **2.4 s** |
| **Inkling-2-Mini** (24L, 128E MoE) | `--inkling2-mode heat43225` (`15/25/60%`) | CUDA (GB10) *(L00/L20 Metal==CUDA)* | 9,216 | `4b:19 / 3b:32 / 2.25b:77` (`2.698 bpw`) | **7.68 GiB** (`-32.6%` vs 4b) | `23.31 / 17.14 / 11.76 dB` | **189.1 s** |
| **Inkling-2-Mini** (24L, 128E MoE) | `--inkling2-mode heat432` (`15/25/60%`) | CUDA (GB10) *(L00/L20 Metal==CUDA)* | 9,216 | `4b:19 / 3b:32 / 2.00b:77` (`2.548 bpw`) | **7.26 GiB** (`-36.3%` vs 4b) | `23.31 / 17.14 / 10.38 dB` | **160.0 s** |

---

## 7. Ecosystem Integrations & Experimental Backends

| Component | Status | Notes |
|---|---|---|
| **PyTorch + Hugging Face `transformers`** (`integrations/pytorch/tfq_torch.py`) | **`[MEASURED]`** | `Tfq3Linear` & `Tfq4Linear` verified end-to-end loading real `.tffpw`/`.tffp4` `Qwen2.5-1.5B` checkpoints into `AutoModelForCausalLM.generate()` with zero TensorFold dependency (`tests/integrations_test.py`). |
| **PyTorch CUDA Extension** (`integrations/pytorch/tfq3_cuda_ext.cu`) | **`[SOURCE-INSPECTED]`** | Directly binds `tfq3::launch_dense` and `tfq3::launch_dense_tp_sliced` (`row_off`, `k_sign_off`, `bc0_buf`, `n_bc_buf`) from `src/kernels/cuda/tfq3_cuda.cuh`; requires PyTorch+CUDA on target host to build (`pip install -e integrations/pytorch`). |
| **vLLM Quantization Plugin** (`integrations/vllm/tfq3_quant.py`) | **`[UNIT-TESTED / LIVE SERVING PENDING]`** | Automatic `--quantization tfq3` registration, real `.tffpw` + `pw_v3_tables.bin` weight loader, per-tensor manifest bits (3b/4b/5b mixed), merged `qkv`/`gate+up` per-partition padding & row signs, and Col/Row TP (`tp=2,4`) verified numerically against `tfq3_reference` in `tests/vllm_plugin_test.py`; live multi-GPU vLLM serving pending a `vllm`+CUDA host. |
| **AMD ROCm / HIP** (`src/kernels/hip/tfq3_hip.h`) | **`[SOURCE-ONLY / NOT HARDWARE-VERIFIED]`** | Portable wave32/wave64 kernels written and syntax-checked only; **no AMD GPU hardware compilation or runtime verification has been performed yet.** |
