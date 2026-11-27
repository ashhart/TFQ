<p align="center">
  <img src="assets/tfq3-header.png" alt="TensorFold TFQ3 — 3 · 4 · 5-Bit Vector Quantization for Metal and CUDA" width="100%">
</p>

# TFQ — TensorFold Native Quantization (`TFQ3` & `TFQ4`)

**Cross-Platform: Apple Silicon (Metal) & NVIDIA GPUs (CUDA `sm_80+`) · Standalone PyTorch / Hugging Face / vLLM & TensorFold**

**TFQ3** (*TensorFold Quantization v3 — supports **3-, 4-, and 5-bit**, including mixed-bit per layer or per expert; `3` is the format version, not 3-bit only*) and **TFQ4-E8P** (`2.001 bpw` 8D E8-initialized Voronoi codebook) provide a fast, **data-free by default** (with **optional Hessian-LDLQ calibration** and **optional trace-guided MoE HeatAlloc**) weight quantization format, GPU converter suite, and standalone inference kernels.

> **One Checkpoint Format, Byte-Identical Conversion Across Platforms:**
> The same `.tffpw`/`.tffp4` checkpoint file runs unmodified on both Apple Silicon (Metal) and NVIDIA GPUs (CUDA `sm_80+`), and both Metal and CUDA converters emit **byte-for-byte identical files** (matching SHA-256) from the same source weights via embedded canonical tables (`reference/tables/`) and deterministic IEEE-754 encoder paths. *(Inference kernels on Metal vs CUDA use each architecture's native MMA and reduction schedules, so cross-vendor inference outputs match within normal fp16/bf16 rounding tolerance.)*

### How TFQ Works (and When to Use Data-Free vs. Calibrated Mode)
1. **Default mode (`100% data-free`, minutes to convert):** Applies a deterministic **randomized 2D Fast Walsh–Hadamard Transform (`2D FWHT-128`)** to every `128×128` tile, sets the `fp16` block scale to `RMS * shrink`, and encodes 2D weight pairs via a **256-wide beam search** over 32-bit trellis states against one fixed **16 KiB canonical Gaussian codebook** (`pw_v3_tables.bin`). For MoE models without `--trace`, expert sensitivity is scored **100% data-free** from router row L2 norms + expert weight-scale energy ([`src/converters/moe_heat_rank.hpp`](src/converters/moe_heat_rank.hpp)) — never index order.
2. **Optional calibration-aware modes (same binary format, zero runtime cost):**
   - **Hessian-LDLQ (`tools/tfq3_hessian.py`):** Optimizes real activation output error $\|(W - \hat{W})X^T\|_F^2$ in TFQ3's rotated domain via block-Cholesky LDLQ + closed-form Hessian-optimal fp16 tile scales, exporting standard `TFPN`/`TFP4` shards.
   - **Trace-guided HeatAlloc (`--trace <counts.txt>`):** Uses calibration routing counts (explicitly **uses calibration data**) to assign higher bit widths to hotter MoE experts.
3. **16 KiB–aligned zero-copy shards + bit-exact speculative decoding:** Fixed reduction order per `(N, K)` guarantees **100% bit-exact** `DFlash2`, `DSpine`, and `MTP` speculative verification (`--no-drafts == drafted` at `temp=0`) across the tested decode ranges: **`rows = 1..32` on Metal** and **`rows = 1..64` on CUDA** (single-weight-read multi-row decode across all Qwen3.8-27B shapes and `N > 34816`).

---

## Get Started

### 1. Install `tfq3` (one command — auto-detects Mac Metal or NVIDIA CUDA)

```bash
curl -fsSL https://raw.githubusercontent.com/ashhart/TFQ/main/install.sh | bash
```

*(Prefer building from a local clone? Run `git clone https://github.com/ashhart/TFQ.git && cd TFQ && make install`.)*

### 2. Convert any Safetensors model (or fit to your exact RAM/VRAM budget)

```bash
# Convert any model (defaults to 4-bit TFQ3 into ./<ModelName>-TFQ3)
tfq3 Qwen/Qwen3.8-27B
tfq3 meta-llama/Llama-3.1-8B-Instruct --target-gib 5.0

# Auto-fit to this machine's available GPU memory (data-free mixed 3b/4b/5b sensitivity planner)
tfq3 /path/to/any-safetensors-model --fit auto

# Quantize vision encoders, speculative drafters, embeddings, or 2.001 bpw TFQ4-E8P
tfq3 /path/to/Qwen3.8-27B-DFlash2 --quantize-drafter --bits 4
tfq3 /path/to/Qwen2.5-1.5B --bits 2 --quantize-embed   # 2.001 bpw TFQ4-E8P -> 0.360 GiB in 1.3 s

# GLM-5.3-Flash: compact (3.39 bpw avg + 4b non-MoE) OR matched uniform-4b + full-precision non-MoE
tfq3 /path/to/GLM-5.3-Flash-FP8 --glm-mode uniform4 --keep-nonmoe-bf16
# ...or the routed experts from the BF16 checkpoint (Metal & CUDA; non-MoE stays in the source)
tfq3 zai-org/GLM-5.3-Flash-BF16 --glm-mode uniform4 --keep-nonmoe-bf16

# ERNIE-5.5-Flash (CUDA): uniform 3b/4b/5b routed experts from the BF16 checkpoint; other tensors stay in the source
tfq3 baidu/ERNIE-5.5-Flash-Preview-BF16 --profile 4b

# Validate any TFQ3/TFQ4 checkpoint & check local toolchain
tfq3 inspect -v ./Qwen3.8-27B-TFQ3
tfq3 doctor
```

### 3. Run Anywhere — PyTorch / Hugging Face, vLLM, or TensorFold

TFQ checkpoints are **not locked to TensorFold**:

```bash
# Install the Python package (bundles canonical tables, Metal shaders for zero-build MPS fused ops, MoE loader & vLLM plugin):
pip install .
```

```python
# Standard PyTorch + Hugging Face Transformers (zero TensorFold dependency):
# - Apple Silicon MPS: zero-build fused Metal kernels via torch.mps.compile_shader (3/4/5b TFQ3 & 2b TFQ4 kind=0/2)
# - NVIDIA CUDA: fused Tensor Core / E8P kernels via tfq3_cuda_ext (or CPU/MPS reference fallback)
from tfq3 import load_tfq_into_hf, TfpwCheckpoint
model = load_tfq_into_hf("./Qwen2.5-1.5B-TFQ3", device="mps")
```

```bash
# Optional native PyTorch CUDA extension (`tfq3_cuda.cuh` + `tfq4_cuda.cuh`, zero BF16 weight expansion):
pip install --no-build-isolation -e integrations/pytorch

# Or serve natively on Mac (Metal) or NVIDIA GPU (CUDA) with TensorFold:
tensorfold serve ./Qwen3.8-27B-TFQ3 --host 0.0.0.0 --port 18650 --context 32768
tensorfold serve ./Qwen3.8-27B-TFQ3 --drafter /path/to/Qwen3.8-27B-DFlash2
```

See [`integrations/README.md`](integrations/README.md) for PyTorch, Hugging Face, and vLLM plugin details.

---

## Controlled Head-to-Head Quality Benchmarks `[MEASURED]`

Controlled comparison on **`Qwen/Qwen2.5-1.5B`** (all **196** transformer linear layers quantized; `embed_tokens` & `lm_head` kept BF16; full WikiText-2 test split: **`145 × 2048 = 296,815` scored tokens**; full-vocabulary exact `fp64` KL across **all `296,815` tokens**; paired per-sequence `ΔNLL` vs BF16 with 2,000-sample paired bootstrap 95% CI). Full SHA-256 provenance receipt (`git_dirty: false`, pinning clean commit `5079156`, converter binary hashes, output shard hashes, and raw `seq_nlls`): [`benchmarks/provenance_qwen25_1p5b.json`](benchmarks/provenance_qwen25_1p5b.json) (reproduce with `tools/bench_quantizers.py --seq-len 2048 --eval-seqs 0 --kl-seqs 0`).

> *Baseline note: Non-TFQ baselines below are in-house reference implementations in `tools/bench_quantizers.py` (our `GPTQ-LDLQ` baseline omits act-order and fixes group min/max scales before feedback; `Scalar Lloyd-Max + LDLQ` uses a 1D optimal Gaussian grid after 2D-FWHT, not EXL3's trellis codebook).*

| Method (`Gold BF16 PPL = 9.2198`) | Eff. bpw | Calib? | Held-Out Out-SNR | WikiText-2 PPL | Paired `ΔNLL` vs BF16 (95% CI) | Mean KL (all tok) | p99 KL | Conv Time |
|---|---:|---|---:|---:|---|---:|---:|---:|
| RTN / Affine 4b (in-house, `g64`) | 4.500 | No | 21.61 dB | 9.9375 | `+0.07495 [+0.07266, +0.07735]` | 0.09606 | 0.6645 | 0.5 s |
| AWQ-style Salient-Scale 4b (in-house, `g128`) | 4.250 | Yes | 22.42 dB | 9.8361 | `+0.06470 [+0.06281, +0.06658]` | 0.07645 | 0.6184 | 10.9 s |
| GPTQ-LDLQ 4b (in-house, no act-order, `g128`) | 4.250 | Yes | 24.21 dB | 9.7188 | `+0.05270 [+0.05039, +0.05500]` | 0.05999 | 0.5435 | 62.2 s |
| **TFQ3-4b Data-Free (real `.tffpw`)** | **4.001** | **No** | **24.41 dB** | **9.6436** | **`+0.04494 [+0.04313, +0.04679]`** | **0.05507** | **0.3848** | **29.9 s** |
| **TFQ3-4b + Hessian-LDLQ (same `TFPN`)** | **4.001** | Yes | **25.99 dB** | **9.5979** | **`+0.04019 [+0.03790, +0.04249]`** | 0.05984 | 0.5213 | 241.4 s |
| Scalar Lloyd-Max + LDLQ 4b (in-house ref) | 4.001 | Yes | 25.94 dB | **9.5577** | **`+0.03599 [+0.03400, +0.03786]`** | **0.04669** | 0.4300 | 94.6 s |
| RTN / Affine 3b (in-house, `g64`) | 3.500 | No | 14.97 dB | 14.5073 | `+0.45329 [+0.44546, +0.46163]` | 0.50406 | 3.0593 | 0.4 s |
| AWQ-style Salient-Scale 3b (in-house, `g128`) | 3.250 | Yes | 15.80 dB | 13.6294 | `+0.39088 [+0.38220, +0.39925]` | 0.42469 | 3.0958 | 8.4 s |
| GPTQ-LDLQ 3b (in-house, no act-order, `g128`) | 3.250 | Yes | 17.47 dB | 12.6126 | `+0.31334 [+0.30352, +0.32380]` | 0.34027 | 2.8809 | 62.7 s |
| **TFQ3-3b Data-Free (real `.tffpw`)** | **3.001** | **No** | **18.31 dB** | **11.2579** | **`+0.19971 [+0.19514, +0.20413]`** | **0.22068** | **1.5705** | **22.8 s** |
| Scalar Lloyd-Max + LDLQ 3b (in-house ref) | 3.001 | Yes | **20.26 dB** | **10.9609** | **`+0.17298 [+0.16512, +0.18215]`** | **0.20569** | 1.7974 | 99.8 s |

**Trade-off summary:** With zero calibration data in ~23–30 seconds, data-free TFQ3 beats our in-house RTN/Affine, AWQ-style, and no-act-order GPTQ-LDLQ baselines at both 4-bit (`ΔNLL +0.0449` vs `+0.0527`, non-overlapping 95% CIs, `-29%` p99 KL at `0.25` fewer bpw) and 3-bit (`PPL 11.26` vs `12.61`, `-45%` p99 KL at `0.25` fewer bpw). Full per-column Hessian LDLQ after 2D-FWHT (`Scalar Lloyd-Max + LDLQ` ref, and external calibrated quantizers like EXL3/QTIP) still achieves lower PPL and mean KL when calibration data and slower conversion are acceptable.

### Flagship TensorFold Runtime Families `[MEASURED — TensorFold runner, no per-sequence JSON receipt in benchmarks/]`

> *Provenance note: The table below summarizes earlier TensorFold hardware measurements; unlike the Qwen2.5-1.5B receipt above, these runs do not yet have committed per-sequence JSON receipts under `benchmarks/`. On `Qwen3.8-Flash-Next` (`2,004` scored tokens), the `0.011` PPL difference between `5.996` and `6.007` is within sampling noise (i.e. parity with 8-bit gold).*

| Model | Recipe & Exact bpw | **TFQ3 Size** | Quality Status | Metal & CUDA Runtime / Drafters |
|---|---|---:|---|---|
| **Qwen3.8-27B** (Dense 64L) | 4.001 bpw linear + BF16 or Q8 residuals (data-free) | **14.37 GiB** (BF16) / **13.23 GiB** (Q8) | **`[MEASURED]`** 4,790-tok PPL `5.368`, −31% mean KL / −46% p99 KL vs MLX 4-bit (`248,320` vocab, 1,022 pos) | Metal [`#4974`](https://github.com/ashhart/TensorFold/pull/4974) (`DFlash2` **+75%**, bit-exact) · CUDA [`#4986`](https://github.com/ashhart/TensorFold/pull/4986) (`BM=64` **1.73×** prefill, `DFlash2` **3.50×**, `DSpine` **3.97×**, bit-exact) |
| **Qwen3.8-Flash-Next** (512E MoE) | 3.411 bpw MoE + 8b non-MoE + 3b PLE | **78.99 GiB** (fits 128 GB Macs) | **`[MEASURED]`** 2,004-tok PPL `5.996` vs `6.007` 8-bit gold (within noise; `MLX 4b = 6.644`) | Metal [`#4964`](https://github.com/ashhart/TensorFold/pull/4964) (`MTP d3` **+86%**, `154.5 tok/s`, bit-exact) · CUDA [`#4986`](https://github.com/ashhart/TensorFold/pull/4986) |
| **GLM-5.3-Flash** (288E MoE) | Compact: `3.39 bpw` MoE (`heat43`=3.33 bpw) + 4b/5b `nonmoe.tffpw` (or `--glm-mode uniform4 --keep-nonmoe-bf16` for 4.00 bpw + BF16 non-MoE) | **139.6 GB** (compact) / **167.5 GB** (uniform 4b + BF16 non-MoE) | **`[MEASURED — 4,094 tok]`** Matched-4b PPL `1.8326` vs `1.7943` FP8 (`+0.0211` nats); full task eval pending | Metal [`#4953`](https://github.com/ashhart/TensorFold/pull/4953) (`p2048 +11.9%`, `MTP` `95.2 tok/s`, `DFlash2` `85.3 tok/s`, bit-exact) · CUDA [`#4986`](https://github.com/ashhart/TensorFold/pull/4986) |

*See [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) for complete tables, provenance tags, and Metal/CUDA microbenchmarks.*

---

## Requirements, Build & Verification

- **Apple Silicon (macOS, M1–M5 — Metal):** Xcode Command Line Tools (`clang++`, `xcrun metal`, `xcrun metallib`). `[MEASURED]`
- **NVIDIA GPUs (Linux, `sm_80+` — CUDA):** CUDA Toolkit (`nvcc` 12+/13+) and `g++`. `[MEASURED]`
- **AMD ROCm / HIP (`src/kernels/hip/tfq3_hip.h`):** `[SOURCE-ONLY / NOT HARDWARE-VERIFIED]` — portable wave32/wave64 kernels written and syntax-checked only; no AMD GPU hardware verification has been run yet.

```bash
git clone https://github.com/ashhart/TFQ.git
cd TFQ

# macOS (Apple Silicon Metal):
make install           # builds Metal converters (TFQ3 + TFQ4) + metallibs and installs `tfq3` into ~/.local/bin
make test              # runs Python references, golden vectors, data-free MoE ranker, strict validator controls, and GPU E2E round-trips
make test-metal        # fp64 Metal parity (3b/4b/5b bit-exact sk8 r1==r16==r32 + BM=32 prefill + Col-TP/Row-TP=4) + bench
make test-tfq4-metal   # fp64 Metal parity (2.001 bpw 8D E8-initialized Voronoi, bit-exact r1==r16) + bench

# Linux (NVIDIA CUDA — pass CUDA_ARCH=sm_80 / sm_86 / sm_89 / sm_90 / sm_100 / sm_121):
make install CUDA_ARCH=sm_121
make test CUDA_ARCH=sm_121
make test-cuda CUDA_ARCH=sm_121
make test-tfq4-cuda CUDA_ARCH=sm_121
```

---

## Documentation & Tools

- **[`docs/HOW_IT_WORKS.md`](docs/HOW_IT_WORKS.md)** — Mathematical formulation of `2D FWHT-128`, `2D-AffineCoset/half4`, state-hash split-K resume, and Metal/CUDA execution.
- **[`docs/FORMAT_SPEC.md`](docs/FORMAT_SPEC.md)** — Byte-level binary specification for `PWV3`/`PWV4` tables and `TFPN`/`TFPW`/`TFP4` shards.
- **[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md)** — Full provenance-tagged benchmarks (`[MEASURED]` / `[SOURCE-INSPECTED]` / `[NOT VERIFIED]`), plus undrafted GB10 dense receipt [`benchmarks/qwen38_27b_gb10_undrafted.json`](benchmarks/qwen38_27b_gb10_undrafted.json).
- **[`docs/TFQ4_SUB3BIT.md`](docs/TFQ4_SUB3BIT.md)** — Sub-3-bit (`2.001 bpw`) `TFQ4-E8P` cold-expert lattice format (both trained `table_kind=0/1` and True QuIP# E8P `table_kind=2`) and Metal/CUDA kernels.
- **[`docs/TENSORFOLD_LOADER_SPEC.md`](docs/TENSORFOLD_LOADER_SPEC.md)** — Runtime integration spec for GLM `--keep-nonmoe-bf16` and Inkling-2 `TFPW` v3 checkpoints.
- **[`tools/tfq3_hessian.py`](tools/tfq3_hessian.py)** — Optional calibration-aware Hessian-weighted Beam-Viterbi + LDLQ encoder (`TFPN`/`TFP4`) and Safetensors Hessian exporter (`--export-hessians` / `--load-hessians`).
- **[`tools/bench_quantizers.py`](tools/bench_quantizers.py)** — Reproducible WikiText-2 PPL, full-vocab KL, and real-activation Output SNR benchmark + SHA-256 provenance generator.
- **[`integrations/`](integrations/README.md)** — Standalone PyTorch, Hugging Face `transformers`, and vLLM integrations.
- **[`reference/tfq3_reference.py`](reference/tfq3_reference.py)** & **[`reference/tfq4_lattice.py`](reference/tfq4_lattice.py)** — Pure-Python NumPy reference decoders.

---

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).
