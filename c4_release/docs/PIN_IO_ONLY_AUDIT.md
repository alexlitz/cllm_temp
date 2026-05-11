# pin_io_only=True Audit: Why It Can't Be Default Yet

**Status (2026-05-11)**: `compile_full_vm(pin_io_only=True)` **builds** but the
resulting model **fails smoke tests** (`test_imm_exit` returns `1` instead of
`42`). Root cause: the bake pipeline still has direct `_SetDim`-position reads
that diverge from the compiler-allocated `dim_positions` when IO-only pinning
is on. Default has been **kept at `False`** until those are migrated.

## Stated Goal

User directive: "I don't want to have anything pinned to specific layers or
dimensions beyond what is needed to do IO."

`pin_io_only=True` realises this in the declaration layer of
`unified_compiler/migrated_ops.py::declare_setdim_compat_dims` — IO-required
dims pin compactly at positions `[0, IO_total)`, every other dim is
bump-pointer allocated above.

## What Works

- `compile_full_vm(pin_io_only=True)` builds successfully.
  - `d_model = 720`, `n_layers = 16` (vs. baseline `d_model = 512`).
  - All contract validation warnings are identical to baseline (4 expected
    READ-BEFORE-WRITE warnings on OPCODE_FLAGS / AX_CARRY_LO / AX_CARRY_HI /
    ADDR_KEY — same as `pin_io_only=False`).
- FFN right-sizing and HybridALU expansion proceed normally.
- `NeuralVMEmbedding` accepts `dim_positions` and routes through `_dim(name)`
  correctly when given a `dim_positions` dict.

## What Breaks (Smoke Test)

`tests/test_smoke.py::TestSmokeBasic::test_imm_exit` runs an `IMM 42; EXIT`
program. Expected exit code: `42`. Observed: `1`.

Test failure (after default flip):
```
FAILED c4_release/tests/test_smoke.py::TestSmokeBasic::test_imm_exit -
       assert 1 == 42
```

`test_add_basic` was running but did not complete inside the 180s timeout
(the model goes off the rails before completing 20 steps).

## Root Cause

The bake/runtime pipeline has multiple sites that hardcode `_SetDim`
positions, not `dim_positions`. When `pin_io_only=True` shifts the layout,
these sites write/read wrong residual-stream lanes.

### Site 1: `set_vm_weights` (vm_step.py:1771-2320) — primary

```python
def set_vm_weights(model, ...):
    ...
    BD = _SetDim          # line 1817
    ...
    self.W_up.data[unit, BD.IS_BYTE] = S
    self.W_up.data[unit, BD.MARK_AX] = -S * 1000
    ...
```

This monolithic function ("legacy bake") contains **2579 `BD.X`** references,
every one resolving to a `_SetDim` constant — i.e. the pre-pinning layout.
With `pin_io_only=True`, the compiler allocates dims to different positions,
but `set_vm_weights` keeps writing to legacy `_SetDim` positions.

The legacy bake is wrapped as `make_legacy_bake_op` (phase=999), so it runs
last, after all migrated ops; the migrated ops correctly use
`_as_setdim_proxy(dim_positions)`, but legacy_bake overwrites their work in
the wrong lanes.

### Site 2: `AutoregressiveVM` embedding construction (vm_step.py:1094-1095)

```python
self.embed = NeuralVMEmbedding(vocab_size, d_model)
```

No `dim_positions` is passed. `NeuralVMEmbedding._dim(name)` then falls back
to `_SetDim` (see `neural_embedding.py:67-68`). So token-embedding injection
(MARK_*, EMBED_LO/HI, ADDR_KEY, MEM_*) writes to legacy positions.

### Site 3: `compile_full_vm` drops `_layout` before model construction
(full_vm_compiler.py:179-185)

```python
model = AutoregressiveVM(
    d_model=layout.d_model,
    n_layers=layout.n_layers,
    n_heads=n_heads,
    ffn_hidden=ffn_hidden,
    max_seq_len=max_seq_len,
)
```

`layout.dim_positions` is never threaded into `AutoregressiveVM` (and hence
never into `NeuralVMEmbedding`). The compiler has the right mapping; it just
isn't plumbed through.

### Site 4: `set_active_opcode` MoE routing (vm_step.py:1198)

```python
dim = _SetDim.opcode_dim(opcode_value)
```

This drives the FFN MoE weight swap. With `pin_io_only=True`, opcode dims
have moved, so the swap fires on a wrong column.

### Site 5: `_dispatch_migrated_block_ops` synthesises `dim_positions` from
`_SetDim` (vm_step.py:2334-2337)

```python
dim_positions = {
    name: getattr(_SetDim, name) for name in dir(_SetDim)
    if not name.startswith('_') and isinstance(getattr(_SetDim, name), int)
}
```

Even though migrated block ops accept `dim_positions`, the dispatcher rebuilds
them from `_SetDim` rather than from the layout. Same problem.

## Fix List (Ordered, Smallest First)

1. **Plumb `dim_positions` through `compile_full_vm` → `AutoregressiveVM` →
   `NeuralVMEmbedding`** so token-embedding injection uses the compiler
   layout. (Easy; the embedding already supports it.)

2. **Replace `BD = _SetDim` in `set_vm_weights`** with a `dim_positions`
   proxy. Either:
   - Pass `dim_positions` to `set_vm_weights` (thread through
     `make_legacy_bake_op._bake`), OR
   - Continue migrating per-layer bakes out of `set_vm_weights` until the
     residual `BD.X` references die.

3. **Fix `_dispatch_migrated_block_ops`** to receive `dim_positions` from the
   compiler dispatch (already available — just thread it).

4. **Fix `set_active_opcode`** to read opcode dim from the model's
   `dim_positions` (store layout reference on the model).

5. **(Test code)** `neural_vm/tests/test_opcodes.py` reads `_SetDim` directly
   (17 references). Acceptable — those tests assert the legacy `_SetDim`
   layout. They should be skipped or updated when `pin_io_only=True` becomes
   default.

## Decision

Leaving `pin_io_only` default at `False` until the fix list above is worked.
The infrastructure is there (compiler computes correct layout, ops with
migrated=True use proxy correctly) — what remains is to retire the last
`_SetDim` direct reads in the bake authority and the embedding plumbing.

## Reproduction

```bash
PICK_GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)
# Build only — succeeds:
CUDA_VISIBLE_DEVICES=$PICK_GPU python -c "
from c4_release.neural_vm.unified_compiler.full_vm_compiler import compile_full_vm
m, l = compile_full_vm(pin_io_only=True)
print(l.d_model, l.n_layers)"
# 720 16

# Run smoke (after flipping default) — fails:
# c4_release/tests/test_smoke.py::TestSmokeBasic::test_imm_exit
# assert 1 == 42
```
