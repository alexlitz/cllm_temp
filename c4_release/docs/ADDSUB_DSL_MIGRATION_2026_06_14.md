# Declarative ADD/SUB migration (efficient mode) — 2026-06-14

Moves the production efficient-mode L8 byte-0 ADD/SUB compute OFF the
imperative `nn.Module` forwards (`AddSub5StageBlock` = `BDToGEConverter` +
`build_add/sub_layers` GE forwards + `GEToBDConverter`) and INTO the
declarative IR (`wide_alu_dsl.wide_add_rules` / `wide_sub_rules` lowered via
`Primitives.lower_ffn_rules`). Folds in the pending +5 fix declaratively.

## What landed

### DSL extensions (`wide_alu_dsl.py`, backward-compatible)

`wide_add_rules` / `wide_sub_rules` gained 5 optional kwargs (all default to
the historical behavior — the existing
`tests/test_wide_alu_dsl.py::test_wide_{add,sub}_rules_byte_identity_*` pass
unchanged):

- `operand_a_cond_weight` / `operand_b_cond_weight` / `marker_cond_weight` /
  `threshold` — parametrize the AND condition weights (was hardcoded
  40/30/30 + thr 80) so the lookup can be magnitude-matched to dirty operands.
- `operand_a_artifact_blocker_weight` — adds a negative condition on every
  OTHER non-zero operand-A nibble cell in the lane, suppressing the spurious
  `a_nib=0` rule the gather's value-proportional index-0 magnitude artifact
  would otherwise fire (the same technique `wide_mul_rules` width=2 and
  `_layer10_alu_ordering_engine_rules` use).
- `result_write_amplitude` — write the result band at a dominant amplitude
  (vs the 2.0/S lookup default). This is the declarative **+5 fix**.
- `final_carry_dim` / `final_borrow_dim` — redirect the LAST byte's
  carry/borrow-out to a specific dim (SUB borrow -> CARRY+2, distinct from
  ADD overflow -> CARRY+1).
- `carry_signal_weight` — absolute (operand-weight-decoupled) inter-byte
  carry discrimination, so the cascade flips cleanly when the operand weights
  are light (the dirty-operand regime).

The carry cascade dim is now written at a CONTROLLED amplitude (~2.0/S)
rather than the dominant result amplitude, so the next byte's carry-read
math sees a known signal.

### Declarative composite (`efficient_alu_addsub_split.py`)

`DeclarativeAddSubBlock` — a thin structural composite holding two
rule-derived `PureFFN` passes, run sequentially in `forward`
(`hi_ffn(lo_ffn(x))`). NO imperative compute. A single FFN forward cannot
self-cascade the inter-nibble carry (`W_up` only reads the input residual),
so the lo pass writes CARRY+0 and the hi pass reads it in the SAME block's
forward — exactly how the imperative `AddSub5StageBlock` ran its 5-stage
`nn.Sequential` inside one block. **One block** => the physical block count
is identical to the imperative (so lea / absolute-position ops are
unaffected).

### Wrap install (`ops/alu_ops.py`)

`make_efficient_l8_addsub_wrap_op` now installs `DeclarativeAddSubBlock`
(declarative) or `AddSub5StageBlock` (imperative) into
`model.blocks[8].post_ops`, gated by `addsub_declarative_enabled()`. The
byte-0 rule set is `_build_addsub_wrap_rules` (lo lane + hi lane partition of
`wide_add_rules`/`wide_sub_rules` width_bytes=2). Carry/borrow map:
- inter-nibble carry/borrow -> CARRY+0 (transient, the L9-lookup scratch dim)
- byte-0 ADD overflow -> CARRY+1, byte-0 SUB borrow -> CARRY+2
  (the dims the downstream L10 `CarryPropagationPostOp` reads).

### Tests (`tests/test_addsub_decl_wrap.py`)

- `compare_symbolic_to_lowered_ffn` lowering-contract check (dims resolve).
- Byte-0 ADD/SUB value identity on CLEAN one-hot operands over the full
  `(a, b)` value grid (a 0..255 step 17, b step 23): the two-pass decode ==
  Python `(a +/- b) & 0xFF`, with CARRY+1 (ADD overflow) / CARRY+2 (SUB
  borrow) set correctly. 24/24 pass.

## Verification

- `tests/test_addsub_decl_wrap.py`: 24/24 pass (CPU).
- `tests/test_wide_alu_dsl.py`: unchanged, all pass (DSL extension is
  backward-compatible).
- Smoke (efficient, spec_k=0), DEFAULT (imperative): 51/0.
- The declarative byte-0 add/sub is byte-identical to the imperative on clean
  operands AND in isolation on the real model (the lo/hi passes run standalone
  produce the correct OUTPUT byte + CARRY flags for e.g. 5-6=0xFF).

## Why it is DEFAULT OFF (the remaining wave)

The live MARK_AX operand bands are DIRTY (the documented operand-gather
hybrid encoding, `project_operand_gather_hybrid_encoding`): operand A in
ALU_LO/HI arrives ~6.4 at the true nibble PLUS a value-proportional ~5.4
index-0 magnitude artifact (+ ~0.45 cell-8/15); operand B in AX_CARRY_LO/HI
arrives ~0.96 with a ~0.31 index-0 floor. Probed spec_k=0 via
`tools/probe_addsub_operand_vectors.py` (input at the AddSub block).

The imperative `BDToGEConverter._clean_onehot` THRESHOLDS the operand bands
to 0/1 BEFORE the GE add — that is what makes it dirty-operand robust.

### Progress: operand-cleanup pre-pass fixes BYTE 0

`DeclarativeAddSubBlock` now runs an OPERAND-CLEANUP `PureFFN` pre-pass
(`_build_addsub_cleanup_rules`, the same div-style `step_function_rule`
artifact-subtraction the bitwise wrap uses) before the lo/hi lookups. With it,
on the REAL model (efficient, spec_k=0, `C4_ADDSUB_DECLARATIVE=1`):

- `add_basic` (10+32=0x2A): byte 0 **0x2A correct** ✓ (full result correct).
- `add_16bit` (200+100=0x12C): byte 0 **0x2C correct** ✓ (was flooded).
- `sub_16bit` (300-100=0xC8): byte 0 **0xC8 correct** ✓ (was flooded).

So the cleanup eliminates the CARRY+0 over-accumulation / OUTPUT_HI flooding
that broke byte 0. Byte-0 add/sub is now byte-identical to the imperative on
the dirty operands.

### Remaining gap: high-byte (byte 1..3) relay amplitude

The multi-byte HIGH bytes are still wrong (`add_16bit` -> `0x808002c`,
`sub_16bit` -> `0xf0f000c8`, `sub_basic` 5-6 -> `0x0` instead of
`0xffffffff`). Byte 0 is right; bytes 1-3 come out `0x80`/`0xF0` garbage.
This is the DOWNSTREAM multi-byte assembly (L10 `CarryPropagationPostOp` +
the L13/L15 byte-1 relay), not byte 0:

- The +5 fix writes OUTPUT byte 0 at a DOMINANT ~20.0 amplitude (residual
  ~60). The byte-1 relay (L13/L15 `*_high_byte_relay`) is a softmax V@O COPY
  of OUTPUT/AX_FULL — a 10x-larger source saturates the relayed high byte
  (hence the `0x80`/`0xF0` saturation pattern). The imperative wrote OUTPUT
  at 2.0, so its relay stayed in range.
- The CARRY+1/+2 flags are also written at the controlled ~2.0/S carry
  amplitude (residual ~6) vs the imperative's exact 2.0; the downstream
  CarryPropagation thresholds may need that reconciled.

### The fix for the next wave

1. Write OUTPUT byte 0 at the dominant amplitude ONLY for the L9-leak
   out-vote, but feed the byte-1 RELAY from a normal-amplitude copy (or
   clamp the relay source) so the high-byte relay does not saturate. Simplest:
   reduce the dominant amplitude to the minimum that still out-votes the L9
   leak (the leak floor is ~83 uniform with a +12.6 cell-0 spike; the
   imperative's 2.0 lost by ~2, so ~16-20 is needed — but check whether a
   smaller value keeps the relay in range), or block the L9 leak directly
   (the brief's alternative: "block the block-N ALU_LO->OUTPUT_LO leak").
2. Reconcile the CARRY+1/+2 amplitude with the downstream
   CarryPropagation thresholds (match the imperative's 2.0).
3. Re-gate: smoke 51/0 + add/sub full-trace, then flip
   `addsub_declarative_enabled` default to ON.

## Tools added

- `tools/probe_addsub_grid.py` — add/sub value-grid pass count (ids 0-49
  ADD, 50-99 SUB from the 1096 corpus) + operand-magnitude dump (`mag`).
- `tools/probe_addsub_operand_vectors.py` — dumps the AddSub-input MARK_AX
  operand vectors to JSON for offline tuning.
- `tools/tune_addsub_wrap.py` — offline (numpy) SwiGLU-AND simulator that
  tunes the lo/hi weights against the probed operand vectors + the
  downstream L9 leak floor.
- `tools/probe_addsub_carry_diff.py` — block-level CARRY/OUTPUT diff between
  the declarative and imperative wraps.
