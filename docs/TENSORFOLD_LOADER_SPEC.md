# TensorFold Zig Loader Specification: GLM BF16-Residual & Inkling-2 `TFPW` v3

This specification defines the concrete Zig runtime changes required in TensorFold to load and execute two `TFQ3` (`PWV3` + `TFPW` v3) MoE checkpoint layouts produced by `TFQ`:

1. **Part 1 — GLM `--keep-nonmoe-bf16` on CUDA**: loading `layer_03..44.tffpw` (or `layer_03..77.tffpw` for `glm_moe_dsa`) + `pw_v3_tables.bin` **without** `nonmoe.tffpw`, sourcing non-MoE projections from BF16, FP8, or MLX-Q8 safetensors (bringing CUDA to full parity with the Metal loader).
2. **Part 2 — Inkling-2 (`inkling2` & `inkling2_loop`) `TFPW` v3 on CUDA and Metal**: loading `layer_00..23.tffpw` (Mini) or `layer_00..47.tffpw` (Small), deinterleaved `gate`/`up`/`down` expert weights, `residual-*.safetensors` + `model.safetensors.index.json`, and standalone `mtp.safetensors`.

All binary structures below conform to [`docs/FORMAT_SPEC.md`](FORMAT_SPEC.md) and are cross-checked against the standalone PyTorch reference loader in [`integrations/pytorch/tfq_moe.py`](../integrations/pytorch/tfq_moe.py).

---

## 0. Shared Binary Format & Integrity Contract

Both parts consume the exact same cross-platform `TFQ3` v3 artifacts (little-endian, 16 KiB-aligned GPU payload sections):

| File | Magic | Header | Record Size | Payload Alignment | Integrity (`XXH64`, seed `0`) |
|---|---|---:|---:|---:|---|
| `pw_v3_tables.bin` | `0x33565750` (`"PWV3"`) | 64 B (`10,816` B active, `16,384` B total) | — | 16 KiB | Finite check on `a3..b5`, `lut3/4/5`, `c64_3/4/5` |
| `layer_XX.tffpw` | `0x57504654` (`"TFPW"`) | 144 B (`TfpwHeader`) | 64 B (`TfpwExpertMeta`) | `off_scales`, `off_weights` mod `16384 == 0` | When `reserved0 & 0x1 != 0`: `reserved1[0..2]` = independent `XXH64` of `meta`, `scales`, `weights`; `reserved0 == 0` accepted (legacy) |
| `nonmoe.tffpw` *(optional)* | `0x4E504654` (`"TFPN"`) | 64 B (`TfpnHeader`) | 128 B (`TfpnTensorMeta`) | `off_scales`, `off_weights` mod `16384 == 0` | When `pad0 & 0x1 != 0`: `pad1` = streamed `XXH64` over `[meta, scales, weights]`; `pad0 == 0` accepted (legacy) |

Mandatory structural checks in `zig/src/cuda/tfq3_format.zig` and `zig/src/families/glm/tfq3_check.zig` before forming any device pointer or `newBufferWithBytesNoCopy` view:
- `magic` and `version == 3` match; `num_experts > 0`, `num_tensors == 3 * num_experts`.
- `hidden_size > 0`, `moe_inter > 0`, `hidden_size % 128 == 0`, `moe_inter % 128 == 0`.
- Strict monotonic non-overlapping bounds: `sizeof(Header) <= off_meta <= meta_end <= off_scales <= scales_end <= off_weights <= weights_end <= file_size` (overflow-checked).
- Every expert has `b_bits in {3, 4, 5}`, `word_off % 4 == 0` when `b_bits == 4` (`uint4` vector loads), and all three projections (`gate [I, H]`, `up [I, H]`, `down [H, I]`) strictly inside `[0, stream_words]` and `[0, scale_halfs]`.
- `XXH64` verification whenever flag bit `0` (`0x1`) is set; refuse load on mismatch with a named error (`error.Tfq3ChecksumMismatch`).

---

## Part 1: GLM `--keep-nonmoe-bf16` Checkpoints on CUDA

### 1.1 Current Gap (Metal vs. CUDA)

Running `bin/tfq3 <glm-dir> --keep-nonmoe-bf16` quantizes routed MoE layers into `layer_03..44.tffpw` (or `layer_03..77.tffpw` for `glm_moe_dsa`) and writes `pw_v3_tables.bin`, while intentionally skipping `nonmoe.tffpw` so attention, dense MLP (`L0..L2`), shared experts, router, embeddings, and `lm_head` stay at full source precision.

- **Metal (`zig/src/families/glm/tfq3.zig`, `weights.zig`, `load_plan.zig`)** already handles this mode: `hasRecords(dir, layers)` returns `true` when any `layer_NN.tffpw` exists even without `nonmoe.tffpw`, `loadNonMoe(...)` returns `null`, and every non-MoE matrix falls back to the checkpoint's safetensors.
- **CUDA (`zig/src/families/glm/cuda_tfq3.zig`)** currently blocks `--keep-nonmoe-bf16` at three hard assumptions:
  1. `cuda_tfq3.isTfq3(gpa, io, dir)` requires `hasRecords(..., "nonmoe.tffpw", magic_tfpn)`.
  2. `cuda_tfq3.Store` owns `nonmoe: cuda.tfq3.TfpnStore` unconditionally, and `loadMoeLayer` reads `s.nonmoe.tables.ptr`.
  3. `loadKda`, `loadDsa`, dense MLP (`L0..L2`), shared experts (`L3+`), `lm_head`, and `Mtp.draft` call `st.nonmoe.findChecked(...)` unconditionally.

### 1.2 Checkpoint Detection (`zig/src/cuda/tfq3_format.zig`, `zig/src/families/glm/cuda_tfq3.zig`)

1. Add `hasAnyTfpw` in `zig/src/cuda/tfq3_format.zig`:
   ```zig
   /// Returns true if `dir` contains at least one valid `layer_*.tffpw` shard with `num_experts > 0`.
   pub fn hasAnyTfpw(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) bool {
       const paths = listShards(gpa, io, dir, "layer_") catch return false;
       defer freePaths(gpa, paths);
       for (paths) |path| {
           if (hasRecords(gpa, io, dir, std.fs.path.basename(path), magic_tfpw)) return true;
       }
       return false;
   }
   ```
2. Update `isTfq3` in `zig/src/families/glm/cuda_tfq3.zig`:
   ```zig
   /// Returns true if `dir` has `pw_v3_tables.bin` plus either `nonmoe.tffpw` or any `layer_*.tffpw`.
   pub fn isTfq3(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) bool {
       if (!cuda.tfq3_format.hasTables(gpa, io, dir)) return false;
       return cuda.tfq3_format.hasRecords(gpa, io, dir, "nonmoe.tffpw", cuda.tfq3_format.magic_tfpn) or
           cuda.tfq3_format.hasAnyTfpw(gpa, io, dir);
   }
   ```
   A stray `pw_v3_tables.bin` with neither `nonmoe.tffpw` nor any `layer_*.tffpw` continues to return `false`.
3. In `zig/src/families/glm/cuda_engine.zig`, gate `c.head_unit = 128` on `has_nonmoe_tfq3` (or whenever `lm_head` is sourced from `nonmoe.tffpw`) so BF16/FP8 `lm_head` alignment rules remain unchanged.

### 1.3 Decoupled `Store` Ownership (`zig/src/families/glm/cuda_tfq3.zig`, `zig/src/cuda/tfq3.zig`)

Decouple `pw_v3_tables.bin` ownership from `TfpnStore`:

```zig
pub const Store = struct {
    tables: cuda.DeviceBuffer,
    nonmoe: ?cuda.tfq3.TfpnStore = null,
    layers: std.ArrayList(cuda.tfq3.OwnedMoeLayer) = .empty,

    pub fn open(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8) !*Store {
        const s = try gpa.create(Store);
        errdefer gpa.destroy(s);
        var tables = try cuda.tfq3.uploadTables(gpa, io, d, dir);
        errdefer tables.free();
        var nonmoe: ?cuda.tfq3.TfpnStore = null;
        if (cuda.tfq3_format.hasRecords(gpa, io, dir, "nonmoe.tffpw", cuda.tfq3_format.magic_tfpn)) {
            nonmoe = try cuda.tfq3.TfpnStore.loadFileWithTables(gpa, io, d, dir, "nonmoe.tffpw", tables.ptr);
        }
        s.* = .{ .tables = tables, .nonmoe = nonmoe };
        return s;
    }

    pub fn findChecked(s: *const Store, name: []const u8, n: usize, k: usize) !?cuda.tfq3.DenseWeight {
        const nm = if (s.nonmoe) |*store| store else return null;
        const w = nm.find(name) orelse return null;
        if (w.n != n or w.k != k) return error.Tfq3Shape;
        return w;
    }

    pub fn loadMoeLayer(s: *Store, gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8, i: u32, experts: u32, hidden: u32, inter: u32) !cuda.tfq3.MoeLayer {
        var name_buf: [32]u8 = undefined;
        const fname = try std.fmt.bufPrint(&name_buf, "layer_{d:0>2}.tffpw", .{i});
        const path = try std.fs.path.join(gpa, &.{ dir, fname });
        defer gpa.free(path);
        var owned = try cuda.tfq3.uploadTfpwLayer(io, d, path, s.tables.ptr, experts, hidden, inter, i);
        errdefer owned.deinit();
        try s.layers.append(gpa, owned);
        return owned.layer;
    }

    pub fn deinit(s: *Store, gpa: std.mem.Allocator) void {
        for (s.layers.items) |*l| l.deinit();
        s.layers.deinit(gpa);
        if (s.nonmoe) |*nm| nm.deinit();
        s.tables.free();
        gpa.destroy(s);
    }
};
```

In `zig/src/cuda/tfq3.zig`, add `TfpnStore.loadFileWithTables(..., tables_ptr: u64)` (setting an internal `owns_tables: bool = false` flag so `TfpnStore.deinit` frees `tables` only when owned).

### 1.4 Non-MoE Fallback Routing (`zig/src/families/glm/cuda_tfq3.zig`)

Every downstream block in `zig/src/families/glm/` (`cuda_kda_block.zig`, `cuda_dsa_block.zig`, `cuda_mlp.zig`, `cuda_moe.zig`, `cuda_proj.zig`, `cuda_weights.zig`) **already defines** `.bf16` (and `.fp8g`/`.mxfp8`) union arms. Only `cuda_tfq3.zig` and `DraftHead` need to populate them when `st.nonmoe == null` (or when an individual tensor is absent from `nonmoe.tffpw`):

| Component | Target Zig Type | With `nonmoe.tffpw` | Without `nonmoe.tffpw` (`--keep-nonmoe-bf16`) |
|---|---|---|---|
| KDA `in_proj` | `cuda_kda_block.In` | `.tfq3_split = .{ q, k, v, rest }` | `.bf16 = stackDequant(&.{ q, k, v, f_a, g_a, b }, ...)` *(aligned to `cuda_kda_block.matWidth`)* |
| KDA `o_proj` | `projection.Proj` | `.tfq3` | `.bf16` (or `.fp8g` if source is block-FP8 and `projection.fits` passes) |
| MLA/DSA `q_b_proj`, `o_proj`, `indexer.wq_b` | `projection.Proj` | `.tfq3` (`use_tfq3 = true`) | `.bf16` via existing `loadDsa(..., use_tfq3 = false)` path (or `.fp8g`) |
| Dense MLP (`L0..L2`) | `cuda_mlp.Weights` | `.tfq3 = .{ gate, up, down }` | `.bf16 = .{ .gu = stackDequant(&.{ gate, up }), .down = stackDequant(&.{ down }) }` |
| MoE Shared Experts (`L3..L{N-1}`) | `moe.Layer.{shared_gu, shared_down}` | `.tfq3_pair` / `.tfq3` | `.bf16 = stackDequant(&.{ gate, up })` / `.bf16 = stackDequant(&.{ down })` |
| MoE Routed Experts (`L3..L{N-1}`) | `moe.Layer.tfq3` | `st.loadMoeLayer(...)` (`TFPW` v3) | `st.loadMoeLayer(...)` (`TFPW` v3 — unchanged) |
| LM Head | `weights.Weights.head` (`projection.Proj`) | `.tfq3 = lm_head` | `.bf16 = stackDequant(&.{"lm_head"}, &.{c.vocab}, c.hidden)` |
| MTP Draft Head | `cuda_tfq3.DraftHead` | `.tfq3 = lm_head` | **Add `.bf16: u64` arm** dispatching through `cuda_b16.matmul` (zero extra VRAM: aliases `w.head.bf16`) |

#### Source Precision Handling (BF16, MLX-Q8, and Block-FP8)

`cuda_tfq3.dequantToBf16` currently reads `.bf16` and MLX affine `.u32` (4/8-bit with `.scales`/`.biases`). To support FP8 source GLM checkpoints (`F8_E4M3` `.weight` + `F32` `.weight_scale_inv` with `128 × 128` block scales) under `--keep-nonmoe-bf16`:

1. **Direct FP8 fast path (recommended when `tools.lane` supports it)**: for `projection.Proj` targets (`o_proj`, `q_b_proj`, `indexer.wq_b`, `shared_gu`, `shared_down`, `lm_head`), if `ck.get(mod ++ ".weight").dtype == .f8_e4m3` and `ck.has(mod ++ ".weight_scale_inv")`, upload as `cuda.fp8.weight(...)` (`projection.Proj.fp8g`) after checking `projection.fits(tools.lane, q)` — cutting non-MoE VRAM in half with zero dequant overhead.
2. **Universal BF16 fallback in `dequantToBf16`**: when `tw.dtype == .f8_e4m3` (for KDA stacked inputs, pre-`sm_90` GPUs, or shapes where `projection.fits` returns `error.SlicePartialsNotHeld`), load `<mod>.weight_scale_inv` (`F32`, `[ceil(rows/128), ceil(cols/128)]`) and dequantize `bf16Of(e4m3_to_f32(w[r, c]) * scale_inv[r / 128, c / 128])` directly into the contiguous BF16 buffer.
3. **Layer prefix auto-detection (`glm5_next` vs. `glm_moe_dsa`)**: probe `model.layers.0.input_layernorm.weight` vs. `model.language_model.layers.0.input_layernorm.weight` (matching `src/converters/glm/convert.cu`), so both GLM-5.3-Flash (`L3..L44`) and GLM-5.3 753B (`L3..L77`) work seamlessly.
4. **Source safetensors placement**:
   - **Colocated (default)**: `layer_03..NN.tffpw` and `pw_v3_tables.bin` reside in the same directory as the source safetensors + `model.safetensors.index.json` + `config.json` (or residual symlinks). `core.Checkpoint.openModelPrefix(gpa, io, dir, prefix)` opens them with zero extra flags.
   - **Explicit fallback (optional)**: if `dir` contains only `layer_*.tffpw` + `pw_v3_tables.bin` + `config.json` with no safetensors, accept an optional `nonmoe_dir` / `--source-dir` path for `core.Checkpoint.openModelPrefix`.

### 1.5 Scratch & Engine Adjustments (`zig/src/families/glm/cuda_engine.zig`, `cuda_state.zig`)

- In `cuda_engine.zig`: load `cuda.tfq3.Kernels` whenever `w.tfq3 != null`. If `w.tfq3.?.nonmoe == null`, dense TFQ3 split-K scratch (`max_yr_elems`) can be `0` — only `max_xrot_elems` (`max_rows * hidden`) and `max_act_rot_elems` (`max_rows * top_k * moe_inter`) need reservation for routed MoE!
- In `cuda_state.zig`: `split` detection already checks `l.mixer.kda.proj == .split or l.mixer.kda.proj == .tfq3_split`; when `nonmoe == null`, `proj == .bf16`, which routes straight to the existing monolithic BF16 KDA projection buffer.
- In `cuda_tile.zig` and `cuda_l2pf.zig`: `.bf16` and `.fp8g` variants are already handled; no changes needed.

---

## Part 2: Wiring Inkling-2 (`inkling2` & `inkling2_loop`) `TFPW` v3 MoE

### 2.1 Checkpoint Layout & Architecture Invariants

`bin/tfq3 <inkling2-dir> --profile 3.5b` (or `tfq3-inkling2-convert --copy-residuals`) produces a self-contained checkpoint directory for both Inkling-2 variants:

| Variant | `model_type` | Physical Layers | Loop Visits | Experts (`E`) | `top_k` | `hidden_size` (`H`) | `moe_inter` (`I`) | `H/128`, `I/128` | Softcaps (`gate`, `up`) |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| Inkling-2 Mini | `inkling2_loop` | `24` (`layer_00..23.tffpw`) | `48` (core `6..18` × 3 passes) | `128` | `8` | `2304` | `1152` | `18`, `9` | `4.0`, `25.0` |
| Inkling-2 Small | `inkling2` | `48` (`layer_00..47.tffpw`) | `48` (non-loop) | `384` | `8` | `4608` | `2304` | `36`, `18` | `4.0`, `25.0` |

**Directory contents:**
- `pw_v3_tables.bin`: canonical 16 KiB `PWV3` table.
- `layer_00.tffpw` .. `layer_{L-1:02d}.tffpw`: `TFPW` v3 routed MoE shards (`reserved0 & 0x1` XXH64 checksums set).
- `residual-00001-of-0000N.safetensors` + `model.safetensors.index.json`: byte-exact copy of **every** non-expert tensor (`model.llm.embed.weight`, `model.llm.unembed.weight`, norms, attention `wq_du`/`wk_dv`/`wv_dv`/`wr_du`/`wo_ud`/`gate_attn`/`attn_sink_proj`, short convolutions, `attn_res.*`, loop `controllers.*`, routers `mlp.gate.{weight,bias}`, and `model.visual.*`). Zero `.mlp.experts.*` tensors appear in `model.safetensors.index.json`.
- `mtp.safetensors` *(optional)*: copied verbatim from the source checkpoint when present (standalone DSpark / MTP drafter weights; intentionally **not** listed in `model.safetensors.index.json`).
- `nonmoe.tffpw` *(optional)*: `TFPN` v3 quantized BF16 attention projections (`wq_du`, `wo_ud`, `gate_attn`, `wk_dv`, `wv_dv`, `wr_du`), present only when `--mode nonmoe` was run explicitly.
- `config.json`, tokenizer files (`tokenizer.json`, `tiktoken/tokenizer.model`, etc.).

#### Critical Invariant 1 — Pre-Deinterleaved `gate` / `up` / `down`
In source ModelOpt checkpoints, `model.llm.layers.{L}.mlp.experts.w13_weight` packs `gate` on even rows (`2j`) and `up` on odd rows (`2j+1`). `tfq3-inkling2-convert` deinterleaves `w13` during conversion so each `TfpwExpertMeta` entry in `layer_XX.tffpw` stores three separate, contiguous matrices:
- `gate`: `[I, H]` at `(gate_word_off, gate_sc_off)`
- `up`: `[I, H]` at `(up_word_off, up_sc_off)`
- `down`: `[H, I]` at `(down_word_off, down_sc_off)`

Runtime kernels therefore read `gate`, `up`, and `down` directly with stride `1` — **no** `row0=0/1, row_step=2` deinterleaving is performed at inference time.

#### Critical Invariant 2 — Softcapped SwiGLU & Ascending-ID Combine
Inkling-2 does **not** use GLM's hard SwiGLU clamp (`lim = 7.0`). Per `zig/src/families/inkling2/loop_model.zig` (`expertActivation`, `combineExperts`):
1. **Activation**: for pre-activation `gate` value `g` and `up` value `u` (in `f32`, after inverse row-FWHT-128 and `SEED_ROW` sign multiply):
   $$\text{softcap}(x, c) = c \cdot \tanh\!\left(\frac{x}{c}\right), \qquad \text{act} = \text{bf16}\!\Big(\text{softcap}(g, c_{\text{gate}}) \cdot \sigma(g) \cdot \text{softcap}(u, c_{\text{up}})\Big)$$
   with $c_{\text{gate}} = \text{config.text.gate\_cap} = 4.0$ and $c_{\text{up}} = \text{config.text.up\_cap} = 25.0$, evaluated left-to-right in `f32`, rounded once to `bf16`, then forward-rotated with `SEED_COL` + `FWHT-128` into `act_rot` for `down`.
2. **Combine (A12)**: weighted sum over the `top_k = 8` expert `down` outputs in `f32` in **ascending expert ID order** (not router rank order), rounding once to `bf16`.

Because both Mini (`H=2304, I=1152`) and Small (`H=4608, I=2304`) are exact multiples of `128`, every `128 × 128` tile aligns cleanly with zero padding.

---

### 2.2 Shared Inkling-2 Frontend Changes (`zig/src/families/inkling2/`)

1. **`plan.zig`**: extend `Precision`:
   ```zig
   pub const Precision = enum { nvfp4, mxfp8, tfq3 };
   ```
2. **`names.zig`**:
   - Add directory-aware precision detection `precisionsWithDir(gpa, io, entries, layers, dir, out)`:
     - For each layer `i in 0..layers`:
       - If `model.llm.layers.{i}.mlp.experts.w13_weight` is present in `entries`: keep existing `.u8 => .nvfp4`, `.f8_e4m3 => .mxfp8` logic.
       - Else if `cuda.tfq3_format.hasRecords(gpa, io, dir, "layer_{i:0>2}.tffpw", magic_tfpw)` (or shared POSIX check on Metal): set `out[i] = .tfq3`.
       - Else return `error.NoExperts`.
     - If any layer is `.tfq3`, require `pw_v3_tables.bin` in `dir`.
   - In `names.experts(c, i, prec, sink)`: always emit `mlp.gate.weight` (`BF16 [E, H]`) and `mlp.gate.bias` (`F32 [E]`); skip `w13_*` and `w2_*` safetensors entries when `prec == .tfq3`.

---

### 2.3 CUDA Runtime Wiring (`zig/src/families/inkling2/` + `zig/kernels/cuda/`)

#### A. Activation Kernel Extension (`zig/kernels/cuda/tfq3_common.cuh` & `tfq3_moe.cu`)
Extend `tfq3_glu` to support Inkling-2 softcapped SwiGLU alongside GLM's clamped SwiGLU (`act_mode == 0`) and unclamped SwiGLU (`act_mode == 1`):
- Add `float gate_cap, float up_cap` parameters (or pack `act_mode == 2u` with `gate_cap = lim`, `up_cap = aux_cap`):
  ```cpp
  __device__ __forceinline__ float tfq3_glu_ink2(float g, float u, float gate_cap, float up_cap) {
      float gv = __bfloat162float(__float2bfloat16(g));
      float uv = __bfloat162float(__float2bfloat16(u));
      float gc = gate_cap * tanhf(gv / gate_cap);
      float uc = up_cap * tanhf(uv / up_cap);
      float sg = 1.0f / (1.0f + __expf(-gv));
      return __bfloat162float(__float2bfloat16((gc * sg) * uc));
  }
  ```
- In `tf_tfq3_moe_gu_swiglu`, when `act_mode == 2u`, call `tfq3_glu_ink2(...)` before multiplying by `tfq3_sgn(0x12345678u, r0 + ...)` and running `tfq3_fwht128_simd4`.
- In `zig/src/cuda/tfq3.zig`, expose `Kernels.moeSoftcap(..., gate_cap: f32, up_cap: f32, out_f32: bool)` (or add `up_cap: f32 = 0.0` to `Kernels.moe`).

#### B. Admission & Weight Loading (`cuda_native.zig`, `loop_weights.zig`)
1. **`cuda_native.zig`**:
   - Advertise `"tfq3"` in `pub const formats: []const []const u8 = &.{ "modelopt-mixed", "tfq3" };`.
   - Call `names.precisionsWithDir(...)` so `layer_00..23.tffpw` are detected cleanly when `residual-*.safetensors` replaces ModelOpt expert shards.
   - Pass `dir` (and `io`) into `weights_.loadWith(...)`.
   - Preserve the strict `if (ck.unused() != 0) return error.UnusedCheckpointTensor;` check — because `tfq3-inkling2-convert --copy-residuals` strips all `.mlp.experts.*` tensors and keeps every other `model.llm.*` tensor, every tensor in `residual-*.safetensors` is consumed cleanly (after `ck.skip("model.visual.")`).
   - Keep `attn_fp8`, `paged`, `lazy_head`, `graphs`, and `--drafter` (DFlash2) completely orthogonal and enabled as normal; only bypass `prefill_tc` NVFP4/MXFP8 expert packing on `.tfq3` layers.
2. **`loop_weights.zig`**:
   - Add to `Weights`:
     ```zig
     tfq3_tables: ?cuda.DeviceBuffer = null,
     tfq3_layers: []cuda.tfq3.OwnedMoeLayer = &.{},
     ```
     and free them in `Weights.deinit()`.
   - Add to `Layer`:
     ```zig
     tfq3_experts: ?cuda.tfq3.MoeLayer = null,
     ```
   - `arenaBytes(ck, c, opts)` needs no special-casing: `residual-*.safetensors` already omits `.mlp.experts.*`, so the bump arena allocates exact space for embeddings, norms, attention, controllers, routers, and optional lazy head.
   - In `loadWith`: if any layer in `p.layers[0..c.text.layers]` has `.precision == .tfq3`, upload `dir/pw_v3_tables.bin` once via `cuda.tfq3.uploadTables(gpa, io, d, dir)`. Inside the per-layer loop, branch on `p.layers[k].precision`:
     - `.tfq3`: call `cuda.tfq3.uploadTfpwLayer(io, d, layer_path, w.tfq3_tables.?.ptr, exp_n, h, inter, i)` (validating `layer_idx == i`, `num_experts == exp_n`, `hidden_size == h`, `moe_inter == inter`, and per-section `XXH64`), store in `w.tfq3_layers[k]` and `layer.tfq3_experts = w.tfq3_layers[k].layer`, and leave `layer.experts_packed = false`.
     - `.nvfp4` / `.mxfp8`: unchanged existing path.

#### C. Execution & Scratch (`loop_engine.zig`)
1. Add `tfq3: ?cuda.tfq3.Kernels = null` to `engine.Kernels`. Load it when any layer has `tfq3_experts != null`.
2. In `Model.initMode`, reserve TFQ3 rotation scratch **before** CUDA graph capture:
   - `xrot_elems = @min(max_rows, 256) * h` (for Mini: `256 × 2304 × 2 B = 1.13 MiB`; `cuda.tfq3.Kernels.moe` automatically chunks `rows > cap_rows` in a graph-safe deterministic loop).
   - `act_rot_elems = @min(max_rows, 256) * c.topk * c.expert_inter` (for Mini: `256 × 8 × 1152 × 2 B = 4.50 MiB`).
   - `yr_elems = 0` unless `nonmoe.tffpw` is also loaded.
3. In `runMoe` (`loop_engine.zig`): check `if (lw.tfq3_experts) |tl|` **before** `lw.experts_packed`:
   - Compute `moe_down` (`[rows, topk, H]` BF16) using `k.tfq3.?.moeSoftcap(s, tl, x, h, m.s.route_pick, c.topk, down_buf, c.topk, rows, c.topk, c.gate_cap, c.up_cap, false)` (reusing the existing `modelopt.layout` `down` slice inside `m.s.moe_scratch`!).
   - Invoke existing `k.moe.combine` (`tf_ink2moe_combine`) on `(m.s.route_pick, m.s.route_weights, down_buf, m.s.moe_out)` to perform the exact A12 ascending-expert-ID FP32 combine and single BF16 round.
   - Because all 48 loop visits share physical `w.layers[v.layer].tfq3_experts`, zero extra VRAM is needed for looped core passes `1..3`.
4. **Inkling-2 Small (`inkling2`, 48L, `E=384, H=4608, I=2304`)**: uses the identical `OwnedMoeLayer` + `cuda.tfq3.Kernels.moeSoftcap` + ascending combine path in place of `experts.zig` (EXL3) per layer; `xrot`/`act_rot` chunk cap keeps scratch under `18 MiB` even at `rows = 2048`.

---

### 2.4 Metal Runtime Wiring (`zig/src/families/inkling2/metal/`)

#### A. Zero-Copy `mmap` Loader (`metal/model.zig`, `metal/tfq3.zig`)
Create `zig/src/families/inkling2/metal/tfq3.zig` following the exact zero-copy `mmap` + `device.bufferNoCopy` pattern of `zig/src/families/glm/tfq3.zig`:

1. **`Tfq3Layer` struct**: holds `layer_idx: u32`, `num_experts: u32`, `hidden: u32`, `inter: u32`, `tables: mtl.Buffer`, `meta: mtl.Buffer`, `scales: mtl.Buffer`, `weights: mtl.Buffer`, `has_5b: bool`, `bytes: usize`.
2. **Generalized validation (no hardcoded GLM 4096/2048)**: validate `TfpwHeader` against the model's `config.text` (`hidden`, `expert_inter`, `experts`, `layer`), check `16 KiB` alignment, non-overlapping spans, `4-bit word_off % 4 == 0`, and per-section `XXH64` when `reserved0 & 0x1 != 0`.
3. **Zero-copy mapping**: `mmap` `pw_v3_tables.bin` and each `layer_00..{L-1:02d}.tffpw` read-only (`MAP_PRIVATE`), wrap 16 KiB-aligned `scales` and `weights` sections with `device.bufferNoCopy` (zero host copies, zero RAM spike on Apple Silicon unified memory), and copy `meta` (only `128 × 64 = 8 KiB` per Mini layer, `24 KiB` per Small layer) into a small shared `mtl.Buffer` when `off_meta % 16384 != 0`.
4. **`metal/model.zig` changes**:
   - Represent per-layer experts as a tagged union:
     ```zig
     pub const ExpertsWeights = union(enum) {
         modelopt: struct { gu: ExpertMatrix, down: ExpertMatrix },
         tfq3: ink2_tfq3.Layer,
     };
     ```
   - In `Model.load`:
     - Detect TFQ3 when `dir/pw_v3_tables.bin` and `dir/layer_00.tffpw` exist; set `precisions[i] = .tfq3` for every layer that has `layer_{i:0>2}.tffpw` (otherwise keep `.nvfp4`/`.mxfp8` detection via `names.precisions`).
     - `shards.shardFiles(gpa, io, dir)` reads `model.safetensors.index.json` -> opens `residual-00001-of-0000N.safetensors` (automatically skipping `mtp.safetensors` because it is not in the index) and maps all `model.llm.*` non-expert tensors zero-copy.
     - In `loadWeights`: skip `matrix(c, layer, "w13"/"w2", ...)` when `p.precision == .tfq3` and attach the mapped `ink2_tfq3.Layer` instead.
     - Compile `ink2_tfq3.Pipes` only when at least one layer is `.tfq3`.

#### B. Parameterized Metal Kernels & Dispatch (`metal/forward.zig`, `zig/kernels/metal/inkling2/`)
1. Reuse the `TFQ3` decode primitives (`pw_v3_q8_3b/4b/5b`, `pw_fwht128_simd4`, `pw_sgn`, `pw_frag_home`) from `zig/kernels/metal/glm_tfq3/common.metal` and `moe_mv.metal`, parameterized by `N_BC_H = hidden / 128` (`18` for Mini, `36` for Small) and `N_BC_I = inter / 128` (`9` for Mini, `18` for Small) instead of GLM's hardcoded `32 / 16`.
2. Replace GLM's `pw_swiglu_clamp` inside `fwht_glu_fwht` with Inkling-2's softcapped SwiGLU (`gate_cap = 4.0f`, `up_cap = 25.0f`):
   ```metal
   inline float ink2_softcap(float x, float cap) { return cap * tanh(x / cap); }
   inline float ink2_glu(float g, float u, float gate_cap, float up_cap) {
     float gv = float(bfloat(g));
     float uv = float(bfloat(u));
     float act = ink2_softcap(gv, gate_cap) * (1.0f / (1.0f + exp(-gv))) * ink2_softcap(uv, up_cap);
     return float(bfloat(act));
   }
   ```
3. In `metal/forward.zig:routedExperts`:
   - Switch on `w.experts`:
     - `.modelopt`: unchanged existing `experts.gu` + `experts.{nvfp4,mxfp8}Down` + `experts.combine`.
     - `.tfq3`: dispatch `fwht_x_fwd` (`s.mlp_normed` -> `xrot`), `tfq3_moe_gu_softcap` (`xrot` -> `act_rot` `[rows * topk, I]`), `tfq3_moe_down` (`act_rot` -> `s.down` `[rows * topk, H]` BF16), and reuse the exact existing `experts.combine(&m.experts, e, x(at(s.down, 0)), x(at(s.route_ids, 0)), x(at(s.route_weights, 0)), x(at(s.mlp_out, 0)), down_call)`!

#### C. `mtp.safetensors` (DSpark) & Optional `nonmoe.tffpw`
- **DSpark (`metal/dspark_weights.zig`)**: loads `{dir}/mtp.safetensors` directly and folds `target.weights.unembed` (`BF16 [201024, 2304]`) with `markov_head.hidden_out`/`hidden_bias`. Because `tfq3-inkling2-convert --copy-residuals` copies `mtp.safetensors` verbatim and retains `unembed` in `residual-*.safetensors`, DSpark works with **zero changes** to `dspark_weights.zig` or `dspark.zig`!
- **Optional `nonmoe.tffpw` (4/5-bit attention)**: if `dir/nonmoe.tffpw` exists, both Metal (`metal/model.zig`) and CUDA (`loop_weights.zig`) can optionally resolve `model.llm.layers.{L}.attn.{wq_du,wk_dv,wv_dv,wr_du,wo_ud,gate_attn}` from `TFPN` while keeping `attn_sink_proj` (`24 × H`, non-multiple of 128) and all norms/convs/controllers/embeddings in BF16 residuals.

---

## 3. Verification Plan

| Test | Target | Pass Criteria |
|---|---|---|
| Format & XXH64 unit tests | `zig/src/cuda/tfq3_format.zig`, `zig/src/families/glm/tfq3_check_test.zig` | Accepts valid `TFPW`/`TFPN` with `flag=1` and legacy `flag=0`; rejects 1-bit flip in `meta`, `scales`, or `weights`; rejects `b_bits not in {3,4,5}` and misaligned 4-bit `word_off`. |
| GLM `--keep-nonmoe-bf16` detection & load | `zig/src/families/glm/cuda_tfq3.zig` | `isTfq3` returns `true` for `pw_v3_tables.bin` + `layer_03.tffpw` (no `nonmoe.tffpw`), `false` for `pw_v3_tables.bin` alone; `Store.open` succeeds with `nonmoe == null`; `DraftHead.bf16` matches `lm_head` logits. |
| Inkling-2 MoE numerical parity | CUDA (`loop_engine.zig`) & Metal (`metal/forward.zig`) vs [`integrations/pytorch/tfq_moe.py`](../integrations/pytorch/tfq_moe.py) | Single-layer routed MoE (`rows ∈ {1, 8, 64}`, mixed `3b/4b/5b` experts, `gate_cap=4.0, up_cap=25.0`, ascending-ID combine) matches `TfpwLayer.forward_routed` within BF16 rounding (`cosine > 0.9999`, `max_abs < 2e-2`). |
| Residual + `mtp.safetensors` integrity | `cuda_native.zig`, `metal/model.zig`, `metal/dspark_weights.zig` | `ck.unused() == 0` on `residual-*.safetensors`; DSpark loads `mtp.safetensors` cleanly; greedy decode bit-consistent across visits `0..47` and CUDA graphs. |
| Cross-platform bit parity | Metal vs CUDA on identical `Inkling-2` / `GLM` TFQ3 checkpoints | Identical checkpoint directory runs unmodified on both backends with matching greedy tokens and expected speculative acceptance rates. |
