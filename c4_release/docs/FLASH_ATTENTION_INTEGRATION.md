# Flash Attention / SDPA Integration

## Overview

`AutoregressiveAttention.forward` (in `c4_release/neural_vm/vm_step.py`) can run its
softmax1+ALiBi+causal attention through PyTorch's
`F.scaled_dot_product_attention` (SDPA). SDPA auto-selects the Flash
Attention 2 / memory-efficient / math backend depending on shape, dtype, and
the `attn_mask` properties, which buys ~1.5-3x on the attention compute and
substantial memory savings versus a manual `Q @ K.T → softmax → @V` loop.

The model has 30+ AutoregressiveAttention blocks, so this matters even for
short sequences.

## Toggle

The behavior is gated by `use_flash_attention: bool = True` on both
`AutoregressiveAttention` and `AutoregressiveVM` (it propagates to every
block). Default is ON. Disable for byte-identity / numeric debugging:

```python
vm = AutoregressiveVM(..., use_flash_attention=False)
# or per-layer:
attn = AutoregressiveAttention(dim, num_heads=H, layer_idx=i,
                               use_flash_attention=False)
```

The toggle is automatically force-disabled when:

1. `torch.onnx.is_in_onnx_export()` returns True — the ONNX tracer does not
   support the Flash kernels, so we route through the manual path during
   export.
2. `self.W_q.is_sparse` — sparse weights mean the runner is using
   `sparse_linear` (CPU/debug path); SDPA does not accept sparse Q, K, V.

## How softmax1 is preserved

The model uses softmax1 (ZFOD: zero-fill-on-demand) instead of vanilla
softmax. softmax1 is defined as

    softmax1(s)_i = exp(s_i) / (1 + Σ_j exp(s_j))

i.e. an extra "+1" in the denominator. The key property is that when all
scores are negative, the output approaches 0 — used by the VM to represent
"address not present in cache".

SDPA gives vanilla softmax. To recover softmax1 we **append a sink K/V
column** with `K_sink = 0` and `V_sink = 0`:

- Score of any Q against `K_sink = 0` is exactly 0 (the dot product is 0,
  and we set the bias for the sink column to 0 too).
- After softmax with the appended column, the i-th non-sink probability is
  `exp(s_i) / (exp(0) + Σ_j exp(s_j))` = softmax1 exactly.
- The sink column contributes `0` to the output because `V_sink = 0`, so
  the remaining outputs match the manual softmax1 path bitwise (up to
  reduction order differences inside SDPA itself).

This is the well-known "attention sink" / "off-by-one softmax" trick.

## How ALiBi / causal / per-head mask are passed

SDPA accepts a single additive `attn_mask` of shape `[B, H, S_q, S_kv]`.
The forward builds this by stacking:

1. **ALiBi**: `-slope[1, H, 1, 1] * |q_pos - k_pos|[1, 1, S_q, S_kv]`
   computed from `cached_pos_ids` (or arange fallback). Used for L0-L2 in
   hybrid mode, all layers in pure-ALiBi mode.
2. **Causal**: `triu(-inf, diagonal=S_kv - S_q + 1)` broadcast over
   `[1, 1, S_q, S_kv]`. Encoded into the mask rather than via SDPA's
   `is_causal=True` shortcut because (a) `is_causal=True` requires square
   Q/K (we have rectangular with KV cache), and (b) it lets us combine
   trivially with ALiBi / per-head masking.
3. **Per-head soft-eviction** (`per_head_keep_mask`): when set, masks
   `[B, H, 1, S_kv]` of `-inf` for soft-evicted positions. None ⇒ skipped.

The bias is appended with a `0` column for the sink position before being
passed to SDPA.

## Numerical equivalence

Tested at fp64 against the manual path:

- No-cache + ALiBi + causal: max abs diff ~1e-15 (machine epsilon).
- With KV cache (full + incremental): max abs diff ~1e-15.
- With per-head keep mask: max abs diff ~1e-15.

At fp32 the two paths agree to ~1e-5 (the level the byte-identical tests
implicitly demand, mediated by the periodic cache flush — see
`_KV_INCREMENTAL_FLUSH` in `run_vm.py`).

## What stays in the manual path

- ONNX export (`torch.onnx.is_in_onnx_export()` ⇒ manual).
- Sparse Q/K/V/O weights (sparse_linear path is CPU debug only).
- Anyone who explicitly passes `use_flash_attention=False`.

## File pointers

- `c4_release/neural_vm/vm_step.py::AutoregressiveAttention.forward` — the
  branching logic.
- `c4_release/neural_vm/vm_step.py::AutoregressiveAttention.__init__` —
  toggle storage.
- `c4_release/neural_vm/vm_step.py::AutoregressiveVM.__init__` — model-level
  `use_flash_attention` flag that propagates to every block.
- `c4_release/neural_vm/kv_cache.py::TransformerKVCache` — cache layout
  (`cached_pos_ids`, `per_head_keep_mask`) consumed by the bias build.
