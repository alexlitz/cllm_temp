# ONNX Export Status — 2026-05-11

## TL;DR

`torch.onnx.export(model, (token_ids,), ...)` with `opset_version=17`,
`dynamo=False` **succeeds end-to-end** on the production model produced by
`compile_full_vm()`. The exported file (`/tmp/c4_neural_vm.onnx`, ~354 MB,
33,830 nodes) loads in `onnxruntime` and numerically matches the eager
forward (max abs diff ~4.8e-7, all argmax positions agree at the
traced seq_len).

However, the exporter constant-folded **8 distinct blockers** spread
across 9 source locations (441 `TracerWarning`s total). The traced graph
is therefore specialized to the trace-time shape — most importantly,
**dynamic `seq_len > 256` will fail at runtime** because of a
TorchScript-frozen capacity buffer in the embedding (see Blocker 5).

Use the new probe script to reproduce:

```bash
python -m c4_release.scripts.export_onnx --output /tmp/c4_neural_vm.onnx
```

## Setup

Probe script: `c4_release/scripts/export_onnx.py` (new).

Inputs:
- `token_ids`: `[1, 200]` `int64`, content `arange(200) % 256`.
- `dynamo=False` (legacy TorchScript-based exporter — `dynamo=True` has
  different semantics and is out of scope for this probe).
- `dynamic_axes`: `seq` is declared dynamic for both input and output.

Outputs:
- `logits`: `[1, seq, 276]` `float32`.

Validation:
- `onnx.load` → IR version 8, opset 17, 33,830 nodes.
- `onnxruntime.InferenceSession` → numerical match to eager at the
  traced seq_len (200).

## What works (traces clean)

| Component | Why it traces cleanly |
| --- | --- |
| `AutoregressiveVM.forward` skeleton | `embed → blocks → head`; no tensor-data-dependent branches *in the wrapper itself*. The `if cached_prefix_len > 0` branch is a Python int compare so it's constant-folded harmlessly when called with the default `cached_prefix_len=0`. |
| `NeuralVMEmbedding.embed` (the nn.Embedding lookup) | Pure tensor op. |
| `NeuralVMEmbedding._add_code_addr_keys` | Vectorized: `torch.where` + `argmax` + masked add. No `.item()`. The trace specializes `S` to 200 (acceptable). |
| `TransformerBlock.forward` | `attn → ffn → post_ops`. No `.item()`. |
| `AutoregressiveAttention.forward` (default path, `kv_cache=None`, `x_is_new_only=False`) | Pure tensor ops including ALiBi+RoPE+softmax1. The only Python branches (`if kv_cache is not None`, `if S_kv > rope_capacity`, `if H*HD == D`) are constant-folded at trace time and irrelevant when `kv_cache=None`. |
| `SoftMoEFFN` | `torch.onnx.is_in_onnx_export()` correctly routes to `_soft_forward` (all-experts parallel, no `.item()`). 17 such modules in the model — all traced clean. |
| All ALU post_ops (`BinaryOpByteZeroingPostOp`, `CarryPropagationPostOp`, `BitwiseBytePropagationPostOp`, `ComparisonCombine`) | Pure `PureFFN` subclasses; tensor-only. |
| ALU MUL/DIVMOD multi-stage `nn.Sequential` pipelines (post-early-out) | Stages themselves are FFN/linear; tracing collapses them. |

## Blockers (Python control flow folded into trace constants)

Each entry below documents:
- **Location** (file:line)
- **Blocker type** (which Python op the tracer flagged)
- **Effect on the traced graph**
- **Concrete fix**

### Blocker 1 — `FlattenedDivMod.forward` opcode-gated early-out

- **Location**: `c4_release/neural_vm/efficient_alu_divmod_split.py:449`
- **Code**:
  ```python
  if float(op_div_max.item()) < 0.1 and float(op_mod_max.item()) < 0.1:
      return x_bd
  ```
- **Warnings**: 2 (one per `.item()`).
- **Trace effect**: At trace time the sample input has no DIV/MOD opcode
  active, so the constant-folded branch returns `x_bd` unchanged. The
  exported graph **omits the entire long-division pipeline** — DIV/MOD
  would produce wrong results in the ONNX runtime.
- **Fix** (≤30 lines): replace with a tensor-only soft gate identical
  to the strategy used by `SoftMoEFFN._soft_forward`. Compute
  `pipeline(x_bd)` unconditionally; let the existing stage-3 mask
  (`(OP_DIV>0.1 OR OP_MOD>0.1) AND MARK_AX>0.5`) zero out the writeback.
  This is what the perf comment already says is the strict-subset
  invariant. The early-out is a perf optimization (~370 ms saved per
  forward call), so it should stay for the eager path. Pattern:
  ```python
  if torch.onnx.is_in_onnx_export():
      return pipeline(x_bd)
  if float(op_div_max.item()) < 0.1 and float(op_mod_max.item()) < 0.1:
      return x_bd
  return pipeline(x_bd)
  ```

### Blocker 2 — `FlattenedALUMul.forward` opcode-gated early-out

- **Location**: `c4_release/neural_vm/efficient_alu_neural.py:1133`
- **Code**:
  ```python
  if x_bd[..., self.BD.OP_MUL].max().item() < 0.1:
      return x_bd
  ```
- **Warnings**: 2.
- **Trace effect**: Same as Blocker 1 — at trace time OP_MUL is
  inactive, so the export omits the schoolbook/carry-pass MUL pipeline.
  Multiplication via ONNX runtime would give wrong results.
- **Fix** (≤10 lines): same `is_in_onnx_export()` guard as Blocker 1.

### Blocker 3 — `NeuralVMEmbedding._inject_mem_metadata` per-token Python scan

- **Location**: `c4_release/neural_vm/neural_embedding.py:476-477` (the
  `while i < S: tok = token_ids[b, i].item(); ...` loop)
- **Warnings**: 401 (the dominant warning class; 201 bool + 200 number
  warnings).
- **Trace effect**: The exporter unrolls the loop with the trace-time
  token IDs. Each `i += 9` step (MEM block) vs `i += 1` step
  (non-MEM) is hard-coded into the graph based on the sample tokens.
  At runtime the graph will write ADDR_KEY/MEM_VAL_B*/MEM_STORE bands
  **only at positions where MEM markers happened to be at trace time**.
  Real bytecode with different MEM placements will be silently wrong.
- **Fix** (≥80 lines, the hardest): vectorize. The structural invariant
  is that every MEM marker is followed by exactly 8 bytes
  (`addr0..3, val0..3`). Steps:
  1. `mem_mask = (token_ids == Token.MEM)`.
  2. Use a 9-position sliding window via `F.unfold` or
     `torch.roll`-based gather to broadcast each MEM marker's address
     bytes to the 4 val positions.
  3. Compute `addr = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)` over
     gathered tensors (`int64` arithmetic with masked add).
  4. Compute nibbles `lo = addr & 0xF`, etc., and one-hot scatter via
     `torch.scatter_add_` into the ADDR_KEY band on val-byte positions.
  5. Write `MEM_STORE = 2.0` and `MEM_VAL_B{0..3} = 1.0` via mask
     broadcast.
  All ops above are pure tensor ops — no `.item()`, no Python loop.

### Blocker 4 — `NeuralVMEmbedding._inject_mem_store` per-token scan

- **Location**: `c4_release/neural_vm/neural_embedding.py:430` (and the
  surrounding `for b in range(B): for i in range(...):` loop)
- **Trace effect**: This path triggers only when `_mem_history_end > 0`.
  At trace time `_mem_history_end = 0` (default) so the early `return`
  on `if end == 0` is taken and **nothing is traced**. Any non-zero
  `_mem_history_end` set after export would be invisible to the
  graph. Net effect: KV-cache eviction with retained MEM history will
  silently degrade attention quality if the exported model is used
  for multi-step generation.
- **Note on `set_mem_history_end`**: This is plain Python state on the
  `NeuralVMEmbedding` module (`self._mem_history_end = int`). It is set
  by `AutoregressiveVMRunner` *outside* the model.forward, not inside.
  So calling `model.forward` directly (as the export script does) is
  fine — the bug only manifests if a consumer of the ONNX graph tries
  to express "this is the history end" via the graph itself.
- **Fix** (≥40 lines): either
  (a) Drop `set_mem_history_end` entirely and infer the history boundary
  from a sentinel token (cleaner, requires tokenizer changes), or
  (b) Plumb `_mem_history_end` as an integer input tensor through the
  model.forward signature. Same vectorization recipe as Blocker 3
  applies to the body of the loop.

### Blocker 5 — `_ensure_addr_key_pos_encoding` capacity check

- **Location**: `c4_release/neural_vm/neural_embedding.py:287`
- **Code**:
  ```python
  if (self._addr_key_pos_encoding is not None
          and self._addr_key_pos_encoding_size >= S
          and ...):
      return self._addr_key_pos_encoding
  ```
- **Warnings**: 1.
- **Trace effect**: **CRITICAL DYNAMIC-AXIS BUG.** Because this method
  was called at trace time with `S=200`, the if-branch took the cached
  return path and **the entire `_compute_addr_key_table` Python loop
  was skipped**. The exported graph hard-codes a 256-row capacity for
  the ADDR_KEY table (the buffer was built at runner construction
  time). Confirmed empirically:
  ```
  seq_len=256: OK shape=(1, 256, 276)
  seq_len=257: RuntimeException: Mul broadcast — 256 by 257
  ```
- **Fix** (≤20 lines): make the table eager — call
  `_ensure_addr_key_pos_encoding(max_seq_len, device, dtype)` once in
  `__init__` after `max_seq_len` is known, so the buffer is always
  present and large enough. The trace then sees the buffer directly
  (a Parameter/register-buffer, not a Python `None`) and the if-branch
  disappears. The current "lazy grow" logic is a CPU-loop bake that
  should never run at inference anyway.

### Blocker 6 — `_compute_prefix_len` bool check + tolist

- **Location**: `c4_release/neural_vm/neural_embedding.py:134` (bool)
  and `:142` (list).
- **Warnings**: 1 + 1.
- **Trace effect**: The prefix-cache hit predicate
  (`token_ids.dim() != 2`) is constant-folded — harmless.
  The `row.tolist()` at line 142 reads Python ints from a tensor;
  the resulting prefix_len becomes a trace-time constant. The traced
  graph therefore always takes the **same prefix-cache slow/fast
  path** as the first call, regardless of runtime input.
- **Note**: Prefix cache is a **runner-managed** perf optimization
  (it's stateful on the module). For ONNX deployment we should
  bypass it entirely — call the un-cached embed path.
- **Fix** (≤15 lines): add a guard at the top of
  `NeuralVMEmbedding.forward`:
  ```python
  if torch.onnx.is_in_onnx_export():
      x = self.embed(token_ids)
      self._add_code_addr_keys(token_ids, x)  # unconditional
      self._inject_mem_metadata(token_ids, x, start_pos=0)
      return x
  # ... existing prefix-cache fast path ...
  ```
  (After fixing Blocker 3, the `_inject_mem_metadata` call becomes
  purely tensor.)

### Blocker 7 — `AutoregressiveAttention.forward` `H * HD == D` branch

- **Location**: `c4_release/neural_vm/vm_step.py:416`
- **Warnings**: 30 (one per block where `H*HD != D` — this fires for
  the L15 attention with 12 heads × 64 dims = 768 ≠ d_model 728).
- **Trace effect**: Each attention block freezes its specific
  `H*HD==D` answer at trace time. Since the model architecture is
  static (each block has fixed `num_heads`/`head_dim`/`d_model`), the
  decision *is* a constant — this warning is a false alarm. The trace
  is correct.
- **Fix** (≤5 lines, optional): replace `if H * HD == D:` with
  `if self._h_times_hd_equals_d:` (a Python bool set in `__init__`).
  Cosmetic only.

### Blocker 8 — `AutoregressiveVM.forward` `cached_prefix_len > 0` branch

- **Location**: `c4_release/neural_vm/vm_step.py:1343`
- **Warnings**: 1.
- **Trace effect**: At trace time `cached_prefix_len=0` (default), so
  the if-branch is constant-folded. The exported graph **always**
  runs the full-sequence path. Real KV-cache incremental generation
  would need a separate export with `cached_prefix_len > 0`.
- **Fix** (out of scope for ONNX-without-kv-cache): if/when KV cache
  needs to be in the exported graph, factor `forward` into two
  variants (`_forward_full`, `_forward_incremental`) and export both.

## Summary of remaining work

| Priority | Blocker | Approx. effort | Risk |
| --- | --- | --- | --- |
| **P0** | #5 (ADDR_KEY 256 cap) | 1 hr | None — eager call in `__init__` |
| **P0** | #1 (DivMod early-out) | 30 min | None — mirror MoE pattern |
| **P0** | #2 (Mul early-out) | 30 min | None — mirror MoE pattern |
| **P0** | #3 (MEM metadata vectorize) | 4-8 hr | Medium — touches L5/L8/L15 attention input; needs property tests |
| **P1** | #6 (prefix-cache ONNX bypass) | 1 hr | Low — pure perf bypass |
| **P2** | #4 (mem_history_end via tensor arg) | 4 hr | Medium — touches runner API |
| **P3** | #7/#8 (cosmetic) | 30 min | None |

Total realistic effort for a clean, fully-dynamic-shape export with
correct DIV/MOD/MUL + correct MEM injection: **~10-12 hours** of focused
work plus a parity test pass. Today's trace already covers
**static-shape inference up to seq_len ≤ 256 for non-DIV/MOD/MUL
opcodes** numerically correctly — useful for read-only programs that
don't exercise arithmetic, but a strict subset of the production
spec.

## Reproduce

```bash
PICK_GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)
CUDA_VISIBLE_DEVICES=$PICK_GPU python -m c4_release.scripts.export_onnx \
    --output /tmp/c4_neural_vm.onnx \
    --opset 17 \
    --seq-len 200

# Validate the produced file:
python -c "import onnx; m=onnx.load('/tmp/c4_neural_vm.onnx'); print(len(m.graph.node), 'nodes')"

# Run inference:
python -c "
import onnxruntime as ort, numpy as np
sess = ort.InferenceSession('/tmp/c4_neural_vm.onnx',
                             providers=['CPUExecutionProvider'])
out = sess.run(None, {'token_ids': np.arange(200, dtype=np.int64).reshape(1,200) % 256})
print(out[0].shape, out[0][0, -1].argmax())
"
```
