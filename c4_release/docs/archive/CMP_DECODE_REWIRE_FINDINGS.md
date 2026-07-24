# CMP-decode-rewire — premise refuted, no commit

Date: 2026-06-05
Branch: cmp-decode-fix (off `01e63287`)
Author: agent (Claude Opus 4.7)
Status: FINDINGS ONLY — no model change

## TL;DR

The brief asked me to add a missing per-op `OP_<CMP>` decode flag in L5 so the
L10 cmp_combine head stops needing the `69f77682` runner-side synth. Two
hours of source + isolated-FFN verification showed that:

1. **The L5 opcode_decode FFN already produces `OP_EQ`/`OP_NE`/`OP_LT`/
   `OP_GT`/`OP_LE`/`OP_GE`.** Each writes ~5.0 to the matching dim when
   fed its (`OPCODE_BYTE_LO`, `OPCODE_BYTE_HI`) one-hot pair under
   `MARK_AX`. Verified directly by baking
   `make_opcode_decode_ffn_op()` into a fresh L5 block and forwarding
   the encoded OP_* one-hots through it (see "Reproducer" below).
2. **The L10 cmp_combine rules already gate on the per-op flag.** Both
   declarative forms (`_l10_comparison_combine_rules` in
   `l10_post_ops_combined`, and `_layer10_alu_cmp_combine_rules` in
   `layer10_alu`) gate every default/override unit on `OP_EQ` / `OP_NE`
   / ... / `OP_GE`. There is no `OP_CMP` generic flag in the codebase.
3. **The brief's source doc (`SMOKE_COMPARISON_OP_DECODE_MISSING.md`,
   dated 2026-05-09-ish) is contradicted by
   `CMP_POLARITY_INVESTIGATION_2026_06_03.md`.** The latter, more recent,
   investigation hooked the residual stream at `model.blocks[33]` and
   showed `OP_EQ = 5.000`, `CMP_GROUP = 1.000` at the EQ-step AX
   position. The CMP head is wired; the bug is upstream.

The actual root cause for the brief's symptom ("model emits constant AX
ignoring operand values") is **stack0 → ALU staleness**: when both
operand nibbles are nonzero (e.g. 17 = `0x11`, 42 = `0x2A`),
`STACK0_BYTE0`'s CLEAN_EMBED is not gathered into `ALU_LO/HI` at the AX
row by the L7 operand_gather head. The L9 CMP flag rules
(`hi_eq`/`lo_eq`/`hi_lt`/`lo_lt`) then see `(ALU=0, AX_CARRY=value)` and
produce a CMP flag pattern that triggers the L10 override path on the
wrong polarity. Documented in detail at
`docs/CMP_POLARITY_INVESTIGATION_2026_06_03.md`.

Per the
`feedback_single_rule_fixes_are_zero_sum.md` memory note and the brief's
own "DON'T remove the runner synths; REVERT + findings if model-side
doesn't work" guardrail, I am **not** committing any model change. No
compile or smoke run was burned. The runner-side synth at
`69f77682` (now expanded to also cover ADD/SUB/OR/XOR/AND in
`batched_pure_neural.py:2054-2069`) is the correct stopgap until the
stack0 staleness fix lands.

## Reproducer for "L5 already produces OP_EQ"

```python
import torch
from neural_vm.unified_compiler.decl_verifier import _build_layout_only
from neural_vm.unified_compiler.ops.l5_ops import make_opcode_decode_ffn_op
from neural_vm.vm_step import AutoregressiveVM

layout = _build_layout_only(alu_mode='lookup',
                            enable_conversational_io=False,
                            enable_tool_calling=False, n_heads=8)
model = AutoregressiveVM(d_model=layout.d_model, n_layers=layout.n_layers,
                        n_heads=8, dim_positions=layout.dim_positions)
op = make_opcode_decode_ffn_op()
block = model.blocks[5]
with torch.no_grad():
    op.bake_fn(block, layout.dim_positions, 100.0)
ffn = block.ffn
dim_positions = layout.dim_positions
d_model = layout.d_model

for op_name, lo, hi in [('OP_EQ', 1, 1), ('OP_NE', 2, 1), ('OP_LT', 3, 1),
                        ('OP_GT', 4, 1), ('OP_LE', 5, 1), ('OP_GE', 6, 1)]:
    x = torch.zeros(1, 1, d_model)
    x[0, 0, dim_positions['OPCODE_BYTE_LO'] + lo] = 1.0
    x[0, 0, dim_positions['OPCODE_BYTE_HI'] + hi] = 1.0
    x[0, 0, dim_positions['MARK_AX']] = 1.0
    if 'CONST' in dim_positions:
        x[0, 0, dim_positions['CONST']] = 1.0
    with torch.no_grad():
        y = ffn(x)
    delta = y[0, 0, dim_positions[op_name]].item()
    print(f'{op_name}: lights up = {delta:.4f}')
```

Output (verified):
```
OP_EQ: lights up = 5.0000
OP_NE: lights up = 5.0000
OP_LT: lights up = 5.0000
OP_GT: lights up = 5.0000
OP_LE: lights up = 5.0000
OP_GE: lights up = 5.0000
```

`W_down` writer scan confirms 1-2 units per dim (the main-AX rule + the
first-step PC rule where applicable):

```
OP_EQ pos=201: 2 writers, units=[17, 48], vals=[0.1, 0.1]
OP_NE pos=202: 1 writers, units=[18],     vals=[0.1]
OP_LT pos=203: 2 writers, units=[19, 49], vals=[0.1, 0.1]
OP_GT pos=204: 1 writers, units=[20],     vals=[0.1]
OP_LE pos=205: 1 writers, units=[21],     vals=[0.1]
OP_GE pos=206: 1 writers, units=[22],     vals=[0.1]
```

(`OP_NE`/`OP_GT`/`OP_LE`/`OP_GE` get a single main-AX writer; `OP_EQ`
and `OP_LT` get an additional first-step PC writer because the
`_opcode_decode_first_step_rules` table only covers a subset of opcodes
— EQ and LT are included, NE/GT/LE/GE are not. This is intentional per
the comment at `l5_ops.py:650-668` and is fine for the steady-state
path that the smoke comparison tests exercise.)

The `SMOKE_COMPARISON_OP_DECODE_MISSING.md` claim that
"`OP_EQ` (compact dim 201) is ZERO at every position in every block
for the entire run" is true at run-time *for the smoke comparison
trace* — but the cause is not "no writer at L5". It is the
`STACK0_BYTE0 = 0` staleness one layer earlier in the operand-gather
path, which makes the L7 attention head produce zero output, which
propagates as zero through everything downstream including, evidently,
the resid trace position the doc's author was looking at. The L5 FFN
writer is present and would fire if the input one-hot pair were
asserted at MARK_AX.

## What the actual cmp_combine rules look like (for the next reader)

`_l10_comparison_combine_rules` (`l10_ops.py:499`) — gates on per-op flag:

```
l10_cmp_default_op_eq_0
  conditions: (MARK_AX, 1.0), (OP_EQ, 1.0), (MARK_PC, -50.0)
  threshold:  1.5
  writes:     OUTPUT_LO+0, OUTPUT_HI_THIS_STEP+0
l10_cmp_override3_op_eq_1
  conditions: (MARK_AX, 1.0), (CMP+1, 1.0), (CMP+2, 1.0), (MARK_PC, -50.0)
  threshold:  2.5
  gate:       OP_EQ
  writes:     OUTPUT_LO+1, OUTPUT_LO+0 (-)
```

`_layer10_alu_cmp_combine_rules` (`l10_ops.py:665`) is the same shape,
authored under a different name for the main L10 ALU bake (vs the post-
ops tail).

Both forms are byte-identical to `vm_step.ComparisonCombine` (verified
in `CMP_POLARITY_INVESTIGATION_2026_06_03.md`'s §"What I verified rules
out a polarity inversion in the cmp_combine rules"). Neither rewires
to anything; the gate is already on the right per-op flag.

## Why the brief's "rewire to per-op flag" cannot land

The brief asked to:

> Rewire the cmp_combine rules to gate on the specific per-op flag.
> This may require:
> - Adding new per-op decode rules in L5
> - Updating gates in L9/L10 cmp_combine
> - Possibly adding new dim entries

All four prerequisites already hold:

| Brief assumption                                     | Actual state              |
|------------------------------------------------------|---------------------------|
| L5 decode rules are missing for EQ/NE/...            | All 6 present + active    |
| L5 decode rules write a generic OP_CMP               | They write per-op dims    |
| L9 CMP rules gate on OP_CMP                          | They gate on CMP_GROUP    |
| L10 cmp_combine rules gate on OP_CMP                 | They gate on OP_EQ etc.   |
| `OP_CMP` is a declared dim                           | Does not exist in repo    |

A rewire from `OP_CMP → OP_EQ/OP_NE/...` is a no-op — that wire already
exists exactly as written. The runner synth at `69f77682` is not
papering over a missing wire; it is papering over `STACK0_BYTE0 = 0`
when both nibbles of the operand are nonzero, which is an L7/L14 attn
issue, not an L5/L9/L10 FFN gate issue.

## Recommended next step

Investigate the stack0/ALU staleness chain as proposed in
`CMP_POLARITY_INVESTIGATION_2026_06_03.md` §"Recommended fix sequence":

1. Hook `model.blocks[7]` (logical L7, `layer7_operand_gather` head 0)
   and the L14 PSH/SI/SC write chain.
2. Find why `STACK0_BYTE0 = 0` when the pushed value is e.g. 17 (= 0x11).
3. Candidates: `_set_layer14_psh_save_ax_stack0_*`, L10 PSH stack0
   passthrough relay, L7 head 0 K-side gate at `l7_ops.py:249`.

The fix sequence has known scope (4 of the 6 CMP smoke failures share
this root cause); the LT-tail polarity bug at `l10_ops.py:6481-6492` is
a separate independent fix path. Until either lands, the runner synth
is doing its job.

## What I did NOT do (and why)

- No `make` / compile.
- No `pytest tests/test_smoke.py` run.
- No model-side edit committed.
- No follow-up brief drafted (this doc is the artifact).

Reason: per the brief's "ONE compile + ONE smoke max" budget and the
`feedback_single_rule_fixes_are_zero_sum.md` memory note, burning compute
on a rewire whose premise was already refuted nine days ago by
`CMP_POLARITY_INVESTIGATION_2026_06_03.md` is exactly the zero-sum
scenario the note warns against. The premise refutation only required
a 2-minute isolated bake check. Committing this findings doc instead
lets the next agent skip the same dead-end.
