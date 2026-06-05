# Opcode-decode one-hot enforcement — findings

Date: 2026-06-05
Worktree: `/home/alexlitz/Documents/misc/c4_release/.claude/worktrees/agent-af2b5299e85df519b`
Branch: `speedup-cache-and-buckets` (off `07faf5d9`)
Companion to: `docs/OP_LEA_LEAK_INVESTIGATION_2026_06_05.md`

## TL;DR

**The L5 opcode_decode FFN is already strictly one-hot by construction.**
On an IMM dispatch row, the L5 LEA-decode rule provably cannot fire, so
OP_LEA receives no FFN-side write. There is no upstream L5 fix to make.

The OP_LEA → MARK_AX leak that the L10 tail rule sees is **not** from the
opcode-decode FFN. It is from L7 head 5's attention broadcast (V reads
OP_LEA at the source MARK_AX position; O writes CMP+7 at the firing
MARK_AX position). The leak path is attention, not FFN. The clean fix
target is L7 head 5's K-side gate, not L5.

## Why L5 opcode_decode is already one-hot

`_opcode_decode_main_rules` (l5_ops.py:565-632) builds one rule per
opcode via `multi_way_and_rule`:

```python
multi_way_and_rule(
    name=f"l5_decode_{op_names[op_val].lower()}_at_ax",
    conditions=(
        (f"OPCODE_BYTE_LO+{lo}", 1.0),
        (f"OPCODE_BYTE_HI+{hi}", 1.0),
    ),
    threshold=1.5,
    gate=dim_ref("marker", "AX"),
    writes=((dim_ref("opcode_flag", op_names[op_val][3:]), 10.0 / S),),
)
```

The opcode table at l5_ops.py:579-614 assigns each opcode a **unique
`(lo, hi)` nibble pair**:

| Opcode | LO | HI |
|--------|----|----|
| LEA    | 0  | 0  |
| IMM    | 1  | 0  |
| JMP    | 2  | 0  |
| ...    | ...| ...|

The lower-nibble and upper-nibble residual dims `OPCODE_BYTE_LO+i` and
`OPCODE_BYTE_HI+j` are themselves one-hot decompositions of the opcode
byte's nibbles (one position per nibble value, exactly one is 1).

**Conjunction algebra**: For the LEA rule on an IMM-dispatch row,
`OPCODE_BYTE_LO+0 = 0` (the IMM row sets `LO+1 = 1` instead) and
`OPCODE_BYTE_HI+0 = 1`. The rule's pre-gate condition score is
`0 * 1.0 + 1 * 1.0 = 1.0`, which is below the `threshold = 1.5`. The
LEA rule does NOT fire. Symmetric reasoning applies to every non-IMM
opcode rule on an IMM row.

`_opcode_decode_first_step_rules` (l5_ops.py:635-692) adds a second
guard (`MARK_PC` AND `HAS_SE = 0`, threshold 2.5) on top of the same
`(lo, hi)` nibble conjunction. The MARK_PC gate restricts these to the
PC position, not the AX position, so they cannot write OP_LEA at AX
under any input.

## What this means for the OP_LEA leak

Per `OP_LEA_LEAK_INVESTIGATION_2026_06_05.md`, the L10
`tail_lea_local_ax_marker_byte0_e8` rule misfires on IMM-dispatch rows.
Its load-bearing positive evidence is **`CMP+7`**, which L7 head 5
writes (l7_ops.py:514-560):

- Q: `MARK_AX * L + H1+AX_I * L`
- K: `MARK_AX * L * 2.0`   (no opcode-side discrimination)
- V: `AP(3, OP_LEA, 0.2)` among many other OP_* projections
- O: `AO(CMP+7, 3, 1.0)`

K-side gates on `MARK_AX` alone, with **no negative term against IMM
dispatch rows**. The softmax over MARK_AX-matched K positions therefore
attends to whatever AX-marker positions exist in scope, including the
IMM-dispatch AX marker (which is itself a MARK_AX position). If OP_LEA
has even a small residual contribution at any of those source positions
— from L5 first-step decode running at a real LEA step on a prior
instruction, or from float noise during `_dispatch_pure_neural` — head
5's V projection picks it up and emits a CMP+7 contribution at the
current MARK_AX.

The L10 tail rule's `("OP_LEA", 1.0)` positive ALSO reads OP_LEA at
MARK_AX directly. In the residual stream at L10 input, OP_LEA at AX on
an IMM row should be 0 (L5 didn't write it), but L7 head 5's relay can
already have polluted it indirectly via attention output writes to
adjacent dims that L8/L9 then leak into OP_LEA's column.

## Recommendation: gate L7 head 5 K-side

The clean upstream fix is to tighten L7 head 5's K-side gate so its
softmax attends only to source positions where OP_LEA is genuinely 1
(real LEA-decoded AX markers), not arbitrary MARK_AX positions.

Patch sketch at `l7_ops.py:521`:

```python
k=(
    AP(0, BD.MARK_AX, L * 2.0),
    AP(0, BD.OP_IMM, -L * 4.0),   # push IMM-dispatch AX positions
                                  # out of softmax mass
    AP(0, BD.OP_LEA, +L * 0.5),   # boost real-LEA AX positions
),
```

**Why this should work**: on an IMM dispatch step, the IMM-dispatch AX
position has `OP_IMM = 1` and `OP_LEA = 0`. The K-side score there
becomes `L*2 - L*4 = -2L`, exp-attenuated below noise. The softmax
mass redistributes to other MARK_AX positions (including the
synthetic REG_AX embedding, where all OP_* dims are 0). At any real
LEA step's AX position, `OP_LEA = 1` and `OP_IMM = 0`, so the score
is `L*2 + L*0.5 = 2.5L`, dominant.

**Risk**: head 5 also relays OP_LI, OP_LC, OP_AND, OP_OR, OP_XOR,
OP_JSR, OP_SHR, OP_SI, OP_SC, OP_ADD, OP_SUB (l7_ops.py:523-538).
Gating K on `-OP_IMM` is safe for all of them (none of these are IMM
itself), but adds a coupling between IMM-step routing and the entire
head-5 relay budget. Should be byte-identity-verified via
`compare_symbolic_to_lowered_attn` against the L7 step-0 corpus for
each of those opcodes.

## What was NOT changed in this session

- No code changes. Per brief: "If one-hot already enforced... document
  findings". The L5 opcode_decode FFN is one-hot by construction; the
  brief's primary action ("patch the rule to enforce one-hot") does
  not apply.
- No L7 head 5 K-side gating added. That is a non-trivial change with
  multi-opcode coupling implications and should not be folded into a
  "document findings" task without a byte-identity sweep.

The downstream filter currently active (commit `0b957e03` — positive
predicate gate on the L10 tail rule) remains the correct interim fix
until L7 head 5's K-side discrimination is properly engineered.

## Confidence

- **Very high** that L5 `_opcode_decode_main_rules` is strictly one-hot
  by construction (mechanically verifiable from the `(lo, hi)` table +
  `multi_way_and_rule` threshold algebra).
- **High** that the OP_LEA leak path is L7 head 5's attention V
  projection, not any FFN rule.
- **Medium** that the proposed L7 head 5 K-side gate (negative on
  OP_IMM) is the correct shape; needs byte-identity verification on
  the 12 relay-target opcodes.

## Files of interest

- `/home/alexlitz/Documents/misc/c4_release/c4_release/neural_vm/unified_compiler/ops/l5_ops.py:565-632`
  — `_opcode_decode_main_rules` (one-hot by construction).
- `/home/alexlitz/Documents/misc/c4_release/c4_release/neural_vm/unified_compiler/ops/l5_ops.py:635-692`
  — `_opcode_decode_first_step_rules` (MARK_PC-gated, not at AX).
- `/home/alexlitz/Documents/misc/c4_release/c4_release/neural_vm/unified_compiler/ops/l7_ops.py:514-560`
  — L7 head 5 OP_LEA relay (the actual leak path).
- `/home/alexlitz/Documents/misc/c4_release/c4_release/neural_vm/unified_compiler/building_blocks_dsl.py:288-355`
  — `multi_way_and_rule` (the algebra that makes L5 one-hot).
- `/home/alexlitz/Documents/misc/c4_release/c4_release/docs/OP_LEA_LEAK_INVESTIGATION_2026_06_05.md`
  — the companion analysis this doc refines.
