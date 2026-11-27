# Standalone PyTorch, Hugging Face Transformers, vLLM & Eval Integrations

TFQ3 and TFQ4 checkpoints are **not locked to TensorFold**. This directory provides standalone loaders, zero-build Apple Silicon MPS Metal kernels, NVIDIA CUDA bindings, a vLLM quantization plugin, and a Hugging Face evaluation/Model Card CLI (`tfq3-eval`) to run any `.tffpw` (`TFPN` / `TFPW` v3) or `.tffp4` (`TFP4` v4, `2.00` & `2.25 bpw`) checkpoint directly in standard PyTorch, Hugging Face `transformers`, or vLLM.

## 0. Install the `tfq3` Python Package

```bash
pip install .
# Or editable from a checkout:
pip install -e .
```

The package bundles the canonical tables, Metal shader sources (`tfq3_metal.metal`, `tfq4_metal.metal`), PyTorch/MPS/MoE/vLLM integrations, and installs the `tfq3-eval` CLI entry point.

## 1. Hugging Face `transformers` + PyTorch

```python
import tfq3
from transformers import AutoTokenizer

model = tfq3.load_tfq_into_hf("./Qwen2.5-1.5B-TFQ3", device="mps")
tok = AutoTokenizer.from_pretrained("./Qwen2.5-1.5B-TFQ3")
ids = tok("The randomized Hadamard transform", return_tensors="pt").to(model.device)
print(tok.decode(model.generate(**ids, max_new_tokens=32, do_sample=False)[0]))
```

Works identically for **TFQ3 (3/4/5-bit)** (`Tfq3Linear`), **TFQ4-E8P (2.001 bpw `table_kind=0/1/2` and 2.251 bpw E8P-1024 `table_kind=3`)** (`Tfq4Linear`), and hybrid TFQ3+TFQ4 dense checkpoints:

- **Apple Silicon MPS (zero build step):** On PyTorch builds with `torch.mps.compile_shader`, `Tfq3Linear` and `Tfq4Linear` automatically compile and dispatch [`src/kernels/metal/tfq3_metal.metal`](../src/kernels/metal/tfq3_metal.metal) and [`src/kernels/metal/tfq4_metal.metal`](../src/kernels/metal/tfq4_metal.metal) (including `2.251 bpw` E8P-1024 `table_kind=3` in [`integrations/pytorch/tfq_mps.py`](pytorch/tfq_mps.py)) directly on MPS buffers with zero Xcode/C++ build and zero BF16 weight expansion.
- **Standalone & Hybrid MoE (`TFPW` v3) checkpoints:** [`integrations/pytorch/tfq_moe.py`](pytorch/tfq_moe.py) (`tfq3.TfpwCheckpoint`, `tfq3.TfpwLayer`) reads and validates `XXH64`-checksummed `layer_XX.tffpw` shards + BF16 residuals (`Inkling-2` and `--keep-nonmoe-bf16` GLM checkpoints), supporting:
  - Pure TFQ3 (`3/4/5b`) and **hybrid TFQ3+TFQ4 MoE layers** (`b_bits ∈ {2, 225, 3, 4, 5}`, auto-loading `pw_v4_e8p_tables.bin` alongside `pw_v3_tables.bin`).
  - **Fused CUDA `tfq3_moe` dispatch** (standard SwiGLU, GLM `swiglu_limit`, and Inkling-2 `gate_softcap` / `up_softcap`) when `tfq3_cuda_ext` is installed, with seamless per-expert TFQ4 dispatch for `{2, 225}`-bit cold experts.

## 2. Native PyTorch CUDA Kernel Extension (`sm_80+`, zero BF16 weight expansion)

Build the extension binding [`src/kernels/cuda/tfq3_cuda.cuh`](../src/kernels/cuda/tfq3_cuda.cuh) and [`src/kernels/cuda/tfq4_cuda.cuh`](../src/kernels/cuda/tfq4_cuda.cuh) directly to `torch::Tensor`:

```bash
pip install --no-build-isolation -e integrations/pytorch
```

Exposes:
- `tfq3_cuda_ext.tfq3_linear`: 3/4/5-bit fused 2D-FWHT + Tensor Core linear (split-K bit-exact decode for `rows <= 16`, zero-partial `BM=64, BN=128` fused prefill for `rows >= 32`, Col-TP `row_off`, Row-TP `k_sign_off` + `bc0_buf`).
- `tfq3_cuda_ext.tfq3_moe`: 3/4/5-bit fused routed MoE SwiGLU + Inkling-2 tanh-softcapped SwiGLU (`gate_cap`, `up_cap`) returning `[rows, top_k, dims]` BF16 directly.
- `tfq3_cuda_ext.tfq4_linear`: 2.00/2.25 bpw E8P fused 2D-FWHT linear (`table_kind` 0/1 trained, 2 true E8P-256, 3 true E8P-1024 when compiled with `TFQ4_HAS_E8P1024`, accepting raw fp16 LUTs or full `PWV4` buffers, plus Col-TP & Row-TP).

## 3. vLLM Quantization Plugin (`--quantization tfq3` / `tfq4`)

[`integrations/vllm/tfq3_quant.py`](vllm/tfq3_quant.py) implements vLLM's `QuantizationConfig` (`Tfq3Config`), `LinearMethodBase` (`Tfq3LinearMethod`, `Tfq4LinearMethod`), and `FusedMoEMethodBase` (`Tfq3MoeMethod`), automatically registering `--quantization tfq3` (and `tfq4` alias) when vLLM is installed:
- **Dense TFQ3 & TFQ4 (`2.00` & `2.25 bpw`):** `Tfq3CheckpointReader`, `Tfq4CheckpointReader`, and `Tfq3Config.from_checkpoint_dir()` auto-detect `.tffpw`, `.tffp4`, `pw_v3_tables.bin`, `pw_v4_e8p_tables.bin` (`table_kind=0..3`), and per-tensor bit widths.
- **Merged projections & Tensor Parallelism (`TP > 1`):** `qkv_proj`, `gate_up_proj`, Column-Parallel (`row_off`), and Row-Parallel (`k_sign_off`) on CUDA, MPS, and portable CPU fallback.
- **Routed & Hybrid MoE (`TFPW` v3):** `Tfq3MoeMethod` and `load_tfq3_moe_weights()` wrap `tfq_moe.py` and dispatch to `tfq3_cuda_ext.tfq3_moe` on CUDA, fused MPS kernels on Apple Silicon, or portable CPU decode.

## 4. Checkpoint Evaluation & Hugging Face Model Card CLI (`tfq3-eval`)

[`tools/tfq3_eval.py`](../tools/tfq3_eval.py) provides a standalone CLI (installed as `tfq3-eval`):

```bash
# Offline layout, effective bpw, bit histogram & XXH64 verification + Model Card:
tfq3-eval --checkpoint ./my-model-tfq4 --inspect-only --model-card MODEL_CARD_SECTION.md

# WikiText-2 PPL + paired BF16 fp64 KL & Top-1 agreement + throughput benchmark:
tfq3-eval --checkpoint ./my-model-tfq3 --ref-model ./my-model-bf16 \
  --dataset wikitext2 --seq-len 2048 --eval-seqs 32 \
  --bench-throughput --model-card MODEL_CARD_SECTION.md --json eval_receipt.json
```
