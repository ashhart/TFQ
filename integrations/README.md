# Standalone PyTorch, Hugging Face Transformers & vLLM Integrations

TFQ3 and TFQ4 checkpoints are **not locked to TensorFold**. This directory provides standalone loaders, zero-build Apple Silicon MPS Metal kernels, and NVIDIA CUDA bindings to run any `.tffpw` (`TFPN` / `TFPW` v3) or `.tffp4` (`TFP4` v4) checkpoint directly in standard PyTorch, Hugging Face `transformers`, or vLLM.

## 0. Install the `tfq3` Python Package

```bash
pip install .
# Or editable from a checkout:
pip install -e .
```

The package bundles the canonical tables and Metal shader sources (`tfq3_metal.metal`, `tfq4_metal.metal`) so `import tfq3` works anywhere.

## 1. Hugging Face `transformers` + PyTorch (`[MEASURED]` end-to-end)

```python
import tfq3
from transformers import AutoTokenizer

model = tfq3.load_tfq_into_hf("./Qwen2.5-1.5B-TFQ3", device="mps")
tok = AutoTokenizer.from_pretrained("./Qwen2.5-1.5B-TFQ3")
ids = tok("The randomized Hadamard transform", return_tensors="pt").to(model.device)
print(tok.decode(model.generate(**ids, max_new_tokens=32, do_sample=False)[0]))
```

Works identically for both **TFQ3 (3/4/5-bit)** (`Tfq3Linear`) and **TFQ4-E8P (2.001 bpw, `table_kind=0/1` trained and `table_kind=2` true QuIP# E8P)** (`Tfq4Linear`).

- **Apple Silicon MPS (zero build step):** On PyTorch builds with `torch.mps.compile_shader`, `Tfq3Linear` and `Tfq4Linear` automatically compile and dispatch [`src/kernels/metal/tfq3_metal.metal`](../src/kernels/metal/tfq3_metal.metal) and [`src/kernels/metal/tfq4_metal.metal`](../src/kernels/metal/tfq4_metal.metal) directly on MPS buffers (`integrations/pytorch/tfq_mps.py`) with zero Xcode/C++ build and zero BF16 weight expansion.
- **Standalone MoE (`TFPW` v3) checkpoints:** [`integrations/pytorch/tfq_moe.py`](pytorch/tfq_moe.py) (`tfq3.TfpwCheckpoint`, `tfq3.TfpwLayer`) reads and validates `XXH64`-checksummed `layer_XX.tffpw` shards + BF16 residuals (`Inkling-2` and `--keep-nonmoe-bf16` GLM checkpoints) with lazy per-expert decode and SwiGLU forward.

## 2. Native PyTorch CUDA Kernel Extension (`sm_80+`, zero BF16 weight expansion)

Build the extension binding [`src/kernels/cuda/tfq3_cuda.cuh`](../src/kernels/cuda/tfq3_cuda.cuh) and [`src/kernels/cuda/tfq4_cuda.cuh`](../src/kernels/cuda/tfq4_cuda.cuh) directly to `torch::Tensor`:

```bash
pip install --no-build-isolation -e integrations/pytorch
```

Once installed, `Tfq3Linear.forward()` and `Tfq4Linear.forward()` on CUDA automatically dispatch to `tfq3_cuda_ext.tfq3_linear` (3/4/5-bit) and `tfq3_cuda_ext.tfq4_linear` (2-bit `table_kind=0/1` and `table_kind=2`, supporting `row_off` and `k_sign_off` TP offsets).

## 3. vLLM Quantization Plugin (`[UNIT-TESTED / LIVE SERVING PENDING]`)

[`integrations/vllm/tfq3_quant.py`](vllm/tfq3_quant.py) implements vLLM's `QuantizationConfig` (`Tfq3Config`) and `LinearMethodBase` (`Tfq3LinearMethod`), automatically registering `--quantization tfq3` via `register_quantization_config("tfq3")` when vLLM is installed:
- **Per-tensor bit widths:** `Tfq3Config.from_config()` reads `tensors` from `tfq3_manifest.json` (supporting uniform 3b/4b/5b and mixed-bit checkpoints, skipping unquantized residual layers like `lm_head`).
- **Real `.tffpw` + `pw_v3_tables.bin` loader:** `Tfq3CheckpointReader` and `load_tfq3_checkpoint_weights()` load packed shards directly into layer parameters (with per-parameter `weight_loader` hooks).
- **Merged projections:** `qkv_proj` (3-way), `gate_up_proj` (2-way), and arbitrary N-way fused layers track independent per-partition 128-row padding, bit widths, and absolute row signs (`row_off`).
- **Tensor Parallelism (`TP > 1`):** Column-Parallel (`row_off`) and Row-Parallel (`k_sign_off` + 1-block trellis state resume `bc0_buf`) dispatch directly to `tfq3_cuda_ext.tfq3_linear` on CUDA and `tfq_mps.tfq3_linear_mps` on Apple Silicon MPS, with an exact portable fallback on CPU.

Verified end-to-end against `tfq3_reference` in [`tests/vllm_plugin_test.py`](../tests/vllm_plugin_test.py). *(Note: live multi-GPU vLLM server benchmarks require installing `vllm` + `tfq3_cuda_ext` on a CUDA host.)*
