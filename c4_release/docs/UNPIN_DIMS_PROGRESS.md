# Un-Pinning Migration Progress (2026-05-11)

Reduces direct `_SetDim` reads in the compiler/runtime pipeline so the
compiler can drive dim allocation. Goal: make `pin_io_only=True` work as the
`compile_full_vm` default, satisfying the user directive "I don't want to
have anything pinned to specific layers or dimensions beyond what is needed
to do IO."

## Status: DONE — Default flipped to `pin_io_only=True`

`pin_io_only=True` (NEW default): `d_model=728`. Smoke `test_imm_exit` and
`test_add_basic` both PASS. Runtime audit gates (3/3) PASS, determinism
tests (4/4) PASS, L3 carry-forward primitive equivalence test PASS.

`pin_io_only=False`: still works at `d_model=512` for backward compat.

## Fixes Landed (cumulative)

### From prior agent (2026-05-11 initial pass)

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

5. **`_make_alu_postop_attach_op` (lookup-mode L8-L13 ALU wrappers) wires the
   structural ALU with `_as_setdim_proxy(dim_positions)`** instead of bare
   `_SetDim`. Previously the proxy was used for the lookup-FFN bakes but the
   ALU sat on top of legacy `_SetDim` positions.

6. **11 convo-I/O state dims added to `declare_setdim_compat_dims`** so they
   no longer fall back to `_SetDim` positions through the proxy and collide
   with compiler-allocated dims under `pin_io_only=True`:
   `LAST_WAS_THINKING_{START,END}`, `LAST_WAS_IO_STATE_EMIT_{BYTE,THINKING}`,
   `IO_IS_{PRTF,READ,TOOL_CALL}`, `IO_STATE`, `IO_OUTPUT_COUNT`,
   `NEXT_IO_STATE_EMIT_{BYTE,THINKING}`.

### Push-2 (2026-05-11 follow-up)

7. **Proxy `opcode_dim()` override (the critical fix).** `_as_setdim_proxy`'s
   `__getattr__` fallback was returning `_SetDim.opcode_dim`, a classmethod
   bound to `_SetDim` itself. Calls like `proxy.opcode_dim(Opcode.IMM)`
   therefore returned the LEGACY position (262 for OP_IMM) instead of the
   compiler-allocated one (187 under `pin_io_only=True`). The
   `_set_opcode_decode_ffn` bake was writing OP_IMM/OP_EXIT/... at the
   legacy residual lanes; the L6 routing FFN read OP_IMM at the compiler
   lane → always 0 → IMM routing dead, OUTPUT_LO got contributions only
   from ambient leakage (which decoded to byte 38 instead of 42). Fix: add
   a `_Proxy.opcode_dim` method that resolves via `dim_positions["OP_<NAME>"]`.

8. **HD-independent threshold attention scores.** `_set_threshold_attn`
   (vm_step.py), `_set_cs_threshold_attn` (setup_helpers.py), and
   `Primitives.threshold_attention` (primitives.py) all hardcoded
   `q_val = 8.0 * slope` assuming `sqrt(HD)=8` (HD=64). Under earlier
   attempts at `pin_io_only=True` with d_model=728/n_heads=8 → HD=91 →
   sqrt(HD)=9.54, the score budget got distorted and threshold heads
   misfired at d=3 instead of d=4 (shifting step structure by one byte).
   Replaced with `q_val = sqrt(HD) * slope` so scores stay
   `slope * (threshold - distance)` regardless of HD.

9. **`Primitives.carry_forward_attention` accepts a `bd=` parameter.**
   The body still referenced module-level `BD = _SetDim` for `L1H0`,
   `L1H1`, `CONST`, defaulting `src_lo`/`src_hi` to `BD.EMBED_LO/HI`.
   Under `pin_io_only=True` those positions differ, so the carry-forward
   attention was writing K-side matches at legacy positions while the
   residual stream had values at compiler positions. Fix: thread the proxy
   in via the new `bd` arg from `make_layer3_carry_forward_attn_op`.

10. **`_set_stack0_carry_attn` accepts a `BD=` parameter.** Same root cause
    as (9). The L3 op now passes `BD=proxy`.

11. **`_set_threshold_attn` accepts a `BD=` parameter.** The body had
    `BD = _SetDim` despite already taking `HD` as a parameter. The L0 op
    now passes `BD=proxy`.

## Sites Touched

- `c4_release/neural_vm/vm_step.py` — `AutoregressiveVM.__init__` accepts
  `dim_positions`; `set_active_opcode` uses `model.dim_positions`;
  `_set_threshold_attn` accepts `BD=` and uses `sqrt(HD)*slope`.
- `c4_release/neural_vm/neural_embedding.py` — `_inject_mem_exec*` migrated
  off `_SetDim` to `self._dim()`.
- `c4_release/neural_vm/setup_helpers.py` — `_set_stack0_carry_attn` accepts
  `BD=`; `_set_cs_threshold_attn` uses `sqrt(HD)*slope`.
- `c4_release/neural_vm/unified_compiler/full_vm_compiler.py` — passes
  `dim_positions=layout.dim_positions` to `AutoregressiveVM`; default
  `pin_io_only=True`.
- `c4_release/neural_vm/unified_compiler/migrated_ops.py` (now split across
  `ops/*.py`) — hybrid ALU wrap uses proxy; 11 convo-IO state dims declared.
- `c4_release/neural_vm/unified_compiler/ops/shared.py` —
  `_as_setdim_proxy.opcode_dim()` override.
- `c4_release/neural_vm/unified_compiler/ops/l0_ops.py` — `BD=proxy`.
- `c4_release/neural_vm/unified_compiler/ops/l3_ops.py` — `bd=proxy` on
  primitives; `BD=proxy` on stack0 carry.
- `c4_release/neural_vm/unified_compiler/primitives.py` — `bd=` on
  `carry_forward_attention`; `sqrt(HD)*slope` on `threshold_attention`.

## Sites Intentionally Left Alone

- `set_vm_weights` (the 2579-`BD.X`-reference legacy bake) — not called
  from `compile_full_vm` anymore. Preserved for backward-compat callers
  (direct `set_vm_weights(model)` usage in tools/tests). Per task
  constraint.
- `_dispatch_migrated_block_ops` (vm_step.py:2334) — only called from
  inside `set_vm_weights`, not reached by the compiler path.

## Test Results

`pin_io_only=True` (new default):
- `tests/test_smoke.py::TestSmokeBasic::test_imm_exit` — PASS
- `tests/test_smoke.py::TestSmokeBasic::test_add_basic` — PASS
- `tests/test_runtime_vanilla.py` (3/3 modes) — PASS
- `tests/test_compile_determinism.py` (4/4 modes) — PASS
- `tests/test_primitives_l3_carry_equivalence.py` — PASS (legacy fallback
  preserved by `bd=None` default).
