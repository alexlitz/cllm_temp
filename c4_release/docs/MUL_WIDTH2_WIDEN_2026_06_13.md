# width=2 MUL: default-on (smoke 49/2 -> 50/1) — 2026-06-13

Status: **LANDED. `mul_overflow` (100*5=500=0x01F4) passes; smoke 50/1
(only `test_simple_function` JSR arch-block remains). bnz green.** The
width=2 (8-bit x 8-bit -> 16-bit) MUL is now the production default; opt
out with `C4_MUL_WIDTH2=0`.

This completes the scaffolding from `b97a991f` (width=2 rules behind
`C4_MUL_WIDTH2`) + `c3e73ed1` (head-dim-preserving auto-widen). Two items
were finished here.

## 1. Route `MUL_RESULT_HI` through `extra_residual_dims` (bnz-safe widen)

The `MUL_RESULT_HI_LO/HI` byte-1 result band (16+16 dims) was declared in
`declare_setdim_compat_dims` (ops/shared.py). That runs in
`_bake_from_scheduled_ops` (full_vm_compiler_dynamic.py) BEFORE
`base_head_dim` is captured, so the band grew d_model and the head_dim was
re-derived from the WIDENED width -> every attention head repartitioned ->
`test_bnz_branch` regressed.

Fix: inject `{MUL_RESULT_HI_LO:16, MUL_RESULT_HI_HI:16}` into
`extra_residual_dims` at the top of `compile_full_vm_dynamic` (gated by
`mul_width2_enabled()`), and in `_bake_from_scheduled_ops`:

* declare the `extra_residual_dims` bands BEFORE the `add_op` loop so ops
  that reference them (the L13 `layer13_mul_result_hi_relay`) pass
  `add_op`'s undeclared-dim validation;
* compute `base_head_dim` from the d_model EXCLUDING those bands (and any
  SSA alias of them), so the head-dim-preserving `_widen_pad` rounds the
  widened width up to a multiple of the BASE head_dim and ADDS a head
  (872 -> 981, n_heads 8 -> 9) instead of repartitioning existing heads.

This also threads the bands through the disk/in-proc cache key (which
hashes `extra_residual_dims` but NOT `C4_MUL_WIDTH2`), so flag-on and
flag-off builds never share a serialised model.

## 2. Operand-magnitude rescale + cell-0 artifact blocker (the compute fix)

The width=2 5-way AND read clean 1.0 one-hots in the DSL unit test, but
the real MARK_AX MUL row delivers (probed spec_k=0 via
`tools/probe_mul_operand_vectors.py`):

* ALU_LO / ALU_HI (operand A nibbles): ~5.84 true one-hot PLUS a
  value-proportional ~5.5 index-0 magnitude artifact on cell 0 (nearly
  equal to a true nibble — the documented `project_operand_gather_
  hybrid_encoding` Wall-1). When A < 16 the cell-0 artifact and the true
  zero-nibble STACK to ~11.4.
* AX_CARRY_LO / AX_CARRY_HI (operand B nibbles): clean ~1.0-1.3 one-hot,
  ~0.29 cell-0 floor.

A naive rescale (or the old 30/30/40/thr150) lets spurious rules fire:
the doubled cell-0 artifact alone (plus marker + a heavy AX_CARRY) clears
threshold without the operand-A nibble being required, so `a_hi` ranges
freely and spurious LARGE products pollute `MUL_RESULT_HI`. Because the
L13 byte-1 relay copies the RAW `MUL_RESULT_HI` band into `AX_FULL` (a
softmax V@O copy, not an argmax), a noisy band corrupts the byte-1 emit —
this regressed `mul_basic` to a 3-byte garbage emit (4849920).

Fix = the proven `_layer10_alu_ordering_engine` technique (ops/l10_ops.py):
a BALANCED 5-way AND (every operand term load-bearing) with ALU weighted
LIGHTLY (dirty band) + AX_CARRY moderately + a strong NEGATIVE blocker on
the OTHER non-zero ALU cells (`operand_a_artifact_blocker_weight` on
`wide_mul_rules`). The blocker suppresses an `a_nib=0` rule whenever A's
true nibble is a different non-zero cell. Tuned offline against the real
probed operand vectors (`tools/tune_mul_width2.py`, full 16^4 SwiGLU
forward sim):

```
operand_a_cond_weight (ALU)               = 0.6
operand_b_cond_weight (AX_CARRY)          = 6.0
marker_cond_weight                        = 4.0
operand_a_artifact_blocker_weight         = 3.0
threshold                                 = 19.5
```

The objective was CLEAN one-hot result bands (not just correct argmax):
6x7 -> OUTPUT_LO[10], OUTPUT_HI[2], MUL_RESULT_HI_LO/HI[0] (byte0=0x2A,
byte1=0x00); 100x5 -> [4],[15],[1],[0] (byte0=0xF4, byte1=0x01). 11/12
probed cases decode the correct 16-bit product; the lone miss is 9x9,
whose ALU_HI is corrupted by an UPSTREAM gather defect reading nibble 3
(not the cell-0 artifact the blocker targets) — out of scope.

`wide_mul_rules`'s new `operand_a_artifact_blocker_weight` defaults to
0.0, so the DSL byte-identity unit test (clean one-hot path) is unchanged
(`tests/test_wide_alu_dsl.py` 82/82).

## Verification

* default (no flag): `pytest tests/test_smoke.py` -> **50 passed / 1
  failed** (`test_simple_function` JSR arch-block; bnz + mul_basic +
  mul_overflow + 32-bit ALU + bitwise + shift + cmp all green).
* `C4_MUL_WIDTH2=0`: 49/2 (pre-width2 build; mul_overflow decodes 20).
* `pytest tests/test_wide_alu_dsl.py` -> 82/82 byte-identity.

## Artifacts

* `tools/probe_mul_operand_vectors.py` — dump the real MARK_AX MUL-row
  operand vectors (JSON) for the tuner.
* `tools/tune_mul_width2.py` — offline full-16^4 SwiGLU forward weight
  search against those vectors.
* `tools/probe_mul_output_band.py` / `tools/probe_mul_width2_operands.py`
  — read the L11 input/output residual bands at the MUL row.
