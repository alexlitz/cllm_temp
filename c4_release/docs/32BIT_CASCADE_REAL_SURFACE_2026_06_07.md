# 32-bit cascade override — real surface attribution (2026-06-07)

Attribution doc for the `ebb3f09a` non-collapsed binary-ALU synth
override in `c4_release/neural_vm/batched_pure_neural.py:2239-2254`.
Companion to `COLLAPSED_STEP_REAL_SURFACE_2026_06_07.md` and
`IMM_OVERRIDE_REAL_SURFACE_2026_06_07.md`.

## TL;DR

The `ebb3f09a` override is **not separable from the collapsed-step
override (`f3342968`)**: per `REMOVAL_3_DEEP_DIVE_2026_06_05.md` it
extended `_NON_COLLAPSED_RECOVERY_OPS` to ADD/SUB/OR/XOR/AND but did
not introduce a new runner-side computation. The "32-bit cascade"
naming is misleading — there is no `OUTPUT_HI byte-1..3` lane to
repair. The real surface is **the autoregressive AX byte-emit chain
for the binary-pop step when that step is emitted at all** (Case A in
the deep-dive trace below) vs. step-fusion to EXIT (Case B, which the
collapsed-step override handles). Both cases share root cause with
`f3342968`'s L27→L28→L34 cascade. Override stays in place pending the
same multi-rule structural fix.

## Override surface (what the code does today)

`c4_release/neural_vm/batched_pure_neural.py:2239-2254`:

- Trigger: `exec_op in _NON_COLLAPSED_RECOVERY_OPS = (ADD, SUB, OR,
  XOR, AND, EQ, NE)`.
- Synthesizes the AX via `_compute_alu_legacy(exec_op,
  last_pushed_value, prev_ax)` and overwrites the just-emitted step's
  REG_AX bytes via `_override_register_in_last_step`.
- Operands: `ax_rhs = prev_ax` (the post-IMM AX, RHS per C4 semantics);
  `stack_val = last_pushed_value` (the runner shadow).

The override fires when the model DOES emit the binop as its own step
(Case A) but writes wrong AX bytes — the dual of `f3342968` which
handles Case B (model fuses binop into IMM step). Both cases observed
across the 5 tests in scope.

## Tests covered (current weights, 2026-06-07)

Per `OVERRIDE_REMOVAL_STATUS_2026_06_06.md`, removing only this
override regresses 5 tests:

| Test | Opcode | Case |
|---|---|---|
| `test_eq_false` | EQ (Shape B) | A (non-collapsed) |
| `test_ne_true` | NE | A |
| `test_sub_16bit` | SUB | A |
| `test_or_16bit` | OR | A |
| `test_xor_16bit` | XOR | A |

ADD-specific (`test_add_16bit`, `test_add_carry_cascade`) fails under
this override alone — they need the IMM-AX override (`b5cf7099`) plus
the collapsed-step override (`f3342968`) too because Case A's
`_override_register_in_last_step` silently no-ops when no REG_AX token
appears in the scan-back window. ADD specifically suppresses the
REG_AX emit so the runner has nothing to write to.

## Why "32-bit cascade" is the wrong framing

Per `REMOVAL_3_DEEP_DIVE_2026_06_05.md`:

1. **All `wide_*_rules` callers use `width_bytes=1`**
   (`alu_ops.py:774, 1530, 1540, 1553, 1563`). The multi-byte ADD/SUB
   cascade is not baked through the declarative DSL `wide_add_rules`
   path at all.
2. The actual multi-byte ALU lives in the imperative
   `AddSub5StageBlock` (`efficient_alu_addsub_split.py`) via
   `build_add_layers(NIBBLE, ...)`. NIBBLE = 4 bits × 8 positions =
   full 32-bit prefix carry-lookahead. The cascade IS handled
   correctly at the ALU layer — byte-identical to a reference 32-bit
   add.
3. `OUTPUT_LO` / `OUTPUT_HI` are **16 dims each** — one-hot lo/hi
   nibble of **one byte**, not byte-1..3 lanes per
   `dim_registry.py:606-609`. The 32-bit AX value is emitted
   byte-by-byte across 4 autoregressive token positions, not as a
   single residual position.

The bug is not at the ALU. The bug is at the **autoregressive AX
byte-emit chain**: at each of the 4 AX byte token positions (byte 0,
1, 2, 3) within the binop step, the model must emit the correct byte
of the ALU result. For ADD/SUB/OR/XOR/AND values like 200+100=300
(=0x012C) the high byte (0x01) and mid byte (0x2C) are wrong, while
0+0=0 trivially works (the default OUTPUT path).

## Case attribution per failing test

| Test | Inputs | Expected | Observed (override off) | Case |
|---|---|---|---|---|
| `test_eq_false` | EQ(17, 18) | 0 | 1 (Shape B both-nibbles-nonzero) | A |
| `test_ne_true` | NE(17, 18) | 1 | 0 (same) | A |
| `test_sub_16bit` | 200 - 100 | 100 | 100 (default; happens to match) but byte-1 corrupted upstream | A |
| `test_or_16bit` | 0xFF00 | OR with 0x00FF | 0xFFFF observed wrong | A |
| `test_xor_16bit` | 0xFF00 XOR 0x00FF | 0xFFFF | corrupted high bytes | A |

The CMP cluster (EQ/NE) hits the **L6 byte-1 divergence** documented
in memory note `project_eq_byte1_l6_divergence.md`: candidate ops
`layer6_attn` / `layer6_routing_ffn` / `layer6_relay_heads` corrupt
byte-1 emit for `(17, 17)` style "both nibbles nonzero" inputs.

The bitwise/SUB cluster hits the same **L27 → L28 → L34 cascade** as
the collapsed-step case, but at the AX byte-1..3 emit positions
instead of STACK0_byte0. The L34 `tail_bit32_result_correction`
suppressor over-fires on amplified OUTPUT_LO/HI residuals from L27
(`layer15_nibble_copy`) self-amplified by L28
(`lev_stack0_byte0_preserve_*` and sibling `tail_stack0_*` families).

## Why "carry-drop layer L9 ADD cascade / L10 OUTPUT_HI" is wrong

The brief framing of "L9/L10 chain drops carry" presupposes a
multi-byte writeback at a single residual position. **There isn't
one.** L9's `AddSub5StageBlock` writes 8 nibble outputs simultaneously
(byte 0 lo/hi, byte 1 lo/hi, ..., byte 3 lo/hi) into the
`AX_CARRY_LO/HI` 16-dim bands at the step's AX marker position. The
correctness probe in `efficient_alu_addsub_split.py` confirms
byte-identity to reference 32-bit add for arbitrary operands.

The drop happens **downstream**, at the **AX byte-1/2/3 marker
position emit stage** (L10-L16), where the model must read the
8-nibble `AX_CARRY_LO/HI` band and select the matching nibble per
position to write into `OUTPUT_LO/HI`. The suppressor cascade at
L27→L28→L34 amplifies stale `OUTPUT_LO/HI` from prior step (the
STACK0 byte the prev PSH stored) and crushes the correct write.

## What was tried (and the relevant reverts)

| Date | Commit | Change | Result |
|---|---|---|---|
| 2026-06-07 04:15 | `1ed6b5b6` | L16 `lev_stack0_byte0_preserve_*` OP_LEV gate strengthen + remove `f3342968` AND `ebb3f09a` | Reverted same day |
| 2026-06-07 13:37 | `4f627491` | Revert of `1ed6b5b6` | Restored baseline 45/51 |

No 32-bit-specific fix attempt has landed — Removal-3 attempts have
all concluded "tied to Removal-2". The original `ebb3f09a` commit body
itself flagged that ADD specifically remained broken because the
model doesn't emit a REG_AX marker in the ADD step; that observation
remains valid 2 days later.

## Recommended next step (not attempted this session)

Per the deep-dive recommendation: **fix Removal 2 first**. The L27→
L28→L34 cascade fix (see `COLLAPSED_STEP_REAL_SURFACE_2026_06_07.md`
"Recommended next step") would simultaneously fix the AX byte-1..3
emit chain for all 5 32-bit cascade tests because they share root
cause with the collapsed-step STACK0_byte0 suppression. A separate
32-bit-cascade fix is not warranted.

For the EQ_FALSE / NE_TRUE subset, the L6 byte-1 divergence per
memory note `project_eq_byte1_l6_divergence.md` is an independent
fix surface; candidate ops are `layer6_attn`, `layer6_routing_ffn`,
`layer6_relay_heads`. Per the memory note, block-by-block diff has
already localized the divergence to L6 (block 6).

## Attribution conclusion

- **Real surface**: shared with `f3342968`'s L27→L28→L34 cascade at
  AX byte-1..3 marker positions (not STACK0_byte0). Plus L6 byte-1
  divergence for Shape-B CMPs (EQ_FALSE / NE_TRUE).
- **Brief's framing refuted**: There is no L9 ADD cascade carry drop.
  `wide_add_rules` not used at width > 1; `AddSub5StageBlock` is
  byte-identity correct. `OUTPUT_LO/HI` are 16-dim one-hot per byte,
  not byte-1..3 lanes. The drop is at autoregressive byte emit
  (L10-L16), not at the ALU.
- **Fix scope**: tied to Removal-2 collapsed-step fix; no independent
  32-bit-cascade fix is correct. Plus L6 byte-1 fix for the Shape-B
  CMP subset (independent, candidate ops identified).
- **Override status**: load-bearing for 5 tests; stays in place
  pending the joint Removal-2/3 multi-rule fix.

## Cross-references

- `c4_release/neural_vm/batched_pure_neural.py:2239-2254` — 32-bit
  cascade override.
- `c4_release/neural_vm/efficient_alu_addsub_split.py` —
  `AddSub5StageBlock`, the byte-identity-correct 32-bit ALU.
- `c4_release/neural_vm/alu/ops/add.py`, `sub.py` —
  `build_add_layers(NIBBLE)` / `build_sub_layers(NIBBLE)` (8 positions
  × 4 bits = full 32-bit prefix carry-lookahead).
- `c4_release/neural_vm/dim_registry.py:606-609` — `OUTPUT_LO`,
  `OUTPUT_HI` are 16 dims one-hot per byte.
- `c4_release/neural_vm/unified_compiler/ops/alu_ops.py:774,1530,1540,1553,1563`
  — all `wide_*_rules` callers use `width_bytes=1`.
- `c4_release/docs/REMOVAL_3_DEEP_DIVE_2026_06_05.md` — parent
  deep-dive (this doc's source).
- `c4_release/docs/REMOVAL_3_FINDINGS_2026_06_05.md` — original
  Removal-3 findings.
- `c4_release/docs/COLLAPSED_STEP_REAL_SURFACE_2026_06_07.md` —
  companion attribution for `f3342968`. Cascade L27→L28→L34 detail.
- `c4_release/docs/OVERRIDE_REMOVAL_STATUS_2026_06_06.md` — current
  smoke-impact table.
- Memory note `project_eq_byte1_l6_divergence.md` — L6 byte-1 fix
  surface for Shape-B EQ/NE.
- Reverted joint attempt: `1ed6b5b6` (reverted by `4f627491`).

## Confidence

- **High** that the `wide_*_rules` path is NOT the failing layer
  (source-verified all callers `width_bytes=1`).
- **High** that `OUTPUT_LO/HI` are single-byte 16-dim one-hot bands,
  not byte-1..3 lanes (`dim_registry.py` source).
- **High** that ADD specifically fails because the binop step omits
  REG_AX emit (`ebb3f09a` commit body + override no-op logic).
- **High** that Removal-3 is tied to Removal-2 (cascade L27→L28→L34
  is shared root cause for STACK0_byte0 AND AX byte-1..3 emit).
- **Medium** that the Shape-B EQ_FALSE / NE_TRUE subset is an
  independent L6 fix (per memory note).
- **Low** that any 1-compile fix can land independently for Removal 3.
  The override is the right pragmatic stay-in-place decision until
  Removal-2's model fix is identified.
