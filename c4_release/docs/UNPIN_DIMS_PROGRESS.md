# Un-Pinning Migration Progress (2026-05-11)

Reduces direct `_SetDim` reads in the compiler/runtime pipeline so the
compiler can drive dim allocation. Goal: make `pin_io_only=True` work as the
`compile_full_vm` default, satisfying the user directive "I don't want to
have anything pinned to specific layers or dimensions beyond what is needed
to do IO."

## Status: PARTIAL — Default NOT flipped

`pin_io_only=False` (default): baseline smoke `test_imm_exit` / `test_add_basic`
PASS (2/2 in 112 s).

`pin_io_only=True`: builds at `d_model=728` (was `720` before declaring the
missing convo-IO dims), but `test_imm_exit` still returns `286330881` instead
of `42` (improved from previously returning `1`). `test_add_basic` times out.

Default has been **left at `False`**.

## Fixes Landed

1. **`compile_full_vm` plumbs `dim_positions` into the model.** Before, the
   compiler computed `layout.dim_positions` but never handed it to
   `AutoregressiveVM`. Fix: pass `dim_positions=layout.dim_positions`.

2. **`AutoregressiveVM` accepts and stores `dim_positions`.** New kwarg
   (default `None` -> `_SetDim`, backward-compat). Stored as
   `self.dim_positions` and forwarded to `NeuralVMEmbedding(..., dim_positions=...)`.

3. **`set_active_opcode` resolves opcode-dim from `model.dim_positions`.**
   New helper `_opcode_dim_from_positions` handles both `_SetDim` (legacy)
   and dict (compiler) sources via an int->name reverse map.

4. **`_inject_mem_exec` and `_inject_mem_exec_autoregressive` (embedding-time
   injectors) use `self._dim()`** instead of bare `_SetDim` for
   `MEM_EXEC`, `MEM_STORE`, `ADDR_KEY`, `MEM_VAL_B0..B3`.

5. **`_make_hybrid_alu_wrap_op` (lookup-mode L8-L13 ALU wrappers) wires the
   structural ALU with `_as_setdim_proxy(dim_positions)`** instead of bare
   `_SetDim`. Previously the proxy was used for the lookup-FFN bakes but the
   ALU sat on top of legacy `_SetDim` positions.

6. **11 convo-I/O state dims added to `declare_setdim_compat_dims`** so they
   no longer fall back to `_SetDim` positions through the proxy and collide
   with compiler-allocated dims under `pin_io_only=True`:
   `LAST_WAS_THINKING_{START,END}`, `LAST_WAS_IO_STATE_EMIT_{BYTE,THINKING}`,
   `IO_IS_{PRTF,READ,TOOL_CALL}`, `IO_STATE`, `IO_OUTPUT_COUNT`,
   `NEXT_IO_STATE_EMIT_{BYTE,THINKING}`. Before this fix, e.g.
   `LAST_WAS_IO_STATE_EMIT_THINKING` (`_SetDim`=463) collided with
   `AX_FULL_HI[0]`, `IO_IS_PRTF` (464) collided with `AX_FULL_HI[1]`,
   corrupting the AX register projection under `pin_io_only=True`.

After these fixes, `pin_io_only=True` shifted `test_imm_exit` from
`result=1` (audit baseline) to `result=286330881`. The result `0x11111101`
suggests EMBED nibbles are getting "1" written into each lane, but the
upstream IMM->AX path is still broken.

## What Still Blocks pin_io_only=True

Diagnostic via debug script confirms only 2 fallback names remain
(`NUM_MARKERS=7`, `NUM_OPCODES=34`), and both are constants used as
iteration counts, not dim positions. So further `_SetDim`-fallback
collisions are unlikely.

Candidate remaining issues (not yet confirmed/fixed):

- **L4/L5 immediate fetch path.** L4 FFN routes immediate bytes to the AX
  marker using OUTPUT_LO/HI; L5 head 1 fetches the opcode at PC using
  ADDR_KEY. Both go through compiler-allocated dims via the proxy. But the
  positions of OUTPUT_LO/HI dropped from 174/190 (legacy) to 69/85 (pin_io)
  and ALU_LO/HI from 360/376 to 319/335. Some bake function might still
  hold stale assumptions about the relative ordering of these blocks (e.g.
  iterating over a range expecting OUTPUT_LO < ALU_LO < AX_FULL_LO).
- **Attention head budget calibration.** The migrated_ops bakes use
  hard-coded `L`/score constants tuned for the legacy 512-dim residual.
  With 728 dims those scores may need rescaling against the higher
  competition from extra non-IO lanes.
- **`compact` / `compact_moe` defaults.** `compact_moe(opcode_range=None)`
  defaults to `range(262, 296)` -- the legacy `_SetDim` opcode range. Not
  called from the compiler smoke path today, but if it ever is, it would
  shed all opcodes under `pin_io_only=True` (where they sit at 197-220).

## Sites Touched

- `c4_release/neural_vm/vm_step.py` — `AutoregressiveVM.__init__` accepts
  `dim_positions`; `set_active_opcode` uses `model.dim_positions`;
  new `_opcode_dim_from_positions` helper.
- `c4_release/neural_vm/neural_embedding.py` — `_inject_mem_exec*` migrated
  off `_SetDim` to `self._dim()`.
- `c4_release/neural_vm/unified_compiler/full_vm_compiler.py` — passes
  `dim_positions=layout.dim_positions` to `AutoregressiveVM`.
- `c4_release/neural_vm/unified_compiler/migrated_ops.py` — hybrid ALU
  wrap uses proxy; 11 convo-IO state dims declared.

## Sites Intentionally Left Alone

- `set_vm_weights` (the 2579-`BD.X`-reference legacy bake) — not called
  from `compile_full_vm` anymore. Preserved for backward-compat callers
  (direct `set_vm_weights(model)` usage in tools/tests). Per task
  constraint.
- `_dispatch_migrated_block_ops` (vm_step.py:2334) — only called from
  inside `set_vm_weights`, not reached by the compiler path.
