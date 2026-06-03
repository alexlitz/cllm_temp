# Cross-step SSA Rename Anti-pattern Audit

Date: 2026-06-02
Scope: `c4_release/neural_vm/unified_compiler/ops/*.py`
Trigger: Runtime-trace agent identified `l5_ops.py:527` (`OPCODE_BYTE_LO.*.-1`)
as an AX-write regression root cause — on step 1 the cross-step read returns
0 because there is no prior step. The same op writes `OPCODE_BYTE_LO`
in the same scheduled layer (via `layer5_fetch`), and on step 2+ the read
"lucks into" the previous step's write.

This audit enumerates that pattern across the ops corpus.

## Headline numbers

| Metric | Count |
|---|---|
| Total `<DIM>.*.-1` reads across ops/ | **83** |
| Suspect: same-file (≈ same-layer) writer of base dim exists | **42** |
| Safe: no same-file writer (cross-layer or cross-step legitimate) | 41 |

Suspect = the cross-step alias is being used as an SCC-breaking schedule
fiction. The base dim's value is being produced *in the same step* by
another op in the same file, and the consumer is reading a phantom
"previous step" slot that:
- on step 1 is 0 (regression);
- on step 2+ aliases to the prior-step write of the same dim and
  *appears* correct.

## Suspect breakdown by introducing commit

The "RENAME commit" is the commit that introduced the `.*.-1` literal at the
current location, via `git log -L <line>,<line>:<path>`.

| Count | Commit | Subject |
|---|---|---|
| 14 | c5598039 | ops/l6: rename PREV_STEP reads to SSA cross-step form (Phase 9.B) |
| 6  | 4a89eb21 | ops/l8: rename PREV_STEP reads to SSA cross-step form (Phase 9.B) |
| 5  | e3981d3e | ops/l3: EMBED_LO SSA rename + dep_anchor read-only (Phase 9.B SCC #2 retire) |
| 3  | f80ca1d9 | ops/l3,l4: rename PREV_STEP reads to SSA cross-step form (Phase 9.B) |
| 2  | 2a684f8a | ops/l10: drop phase= from layer10_stack0_byte_relay (Phase 9.B SSA dissolved its SCC) |
| 2  | 073e5af1 | ops/layer3_ffn: EMBED_LO -> EMBED_LO.*.-1 SSA (Phase 9.B follow-up) |
| 2  | 1f209895 | ops/l5: rename PREV_STEP reads to SSA cross-step form (Phase 9.B) |
| 2  | cc13c079 | ops/l6: add _layer6_attn_dep_anchor for V4 holdouts |
| 1  | 0fa605e8 | ops/l10: layer10_carry_relay CARRY -> CARRY.*.-1 SSA (SCC sub-cycle C) |
| 1  | 5452dd49 | ops/l10: drop phase= from layer10_sp_byte_passthrough (Phase 9.B) |
| 1  | 8ef9c159 | ops/l10: rename PREV_STEP reads to SSA cross-step form (Phase 9.B) |
| 1  | 96691d98 | ops/l10_alu: rename ALU_HI read to ALU_HI.*.-1 (Phase 9.B SCC retire) |
| 1  | 001ad82d | ops/layer8_alu: ALU_LO -> ALU_LO.*.-1 SSA (Phase 9.B SCC reduction) |
| 1  | ac4e7b9a | phase9: demo op layer8_head6_ax_carry_refresh on SSA dim names |

All 42 suspect renames are concentrated in **Phase 9.B SSA dissolution** work
(plus a handful of Phase 8.A predecessors via `_PREV_STEP` aliases that were
later mass-renamed to `.*.-1` in Phase 9.B). The earliest-cross-step
breakdown (i.e. when each consumer first became cross-step) shows the same
ops repeatedly: ADDR_B*, AX_CARRY_*, EMBED_*, OPCODE_BYTE_LO, ALU_*, CARRY,
TEMP, CMP, OUTPUT_*.

## Top 10 highest-risk (op, dim) pairs

Ranked by number of same-file writers of the base dim (more writers ⇒ more
opportunity for an in-step Y→X collision; the original L5 bug had exactly
one writer (`layer5_fetch`)).

| # | File | Op | Dim | Same-file writers | Rename commit |
|---|---|---|---|---|---|
| 1 | l10_ops.py | layer10_psh_stack0_passthrough_bake | OUTPUT_LO | 11 | 8ef9c159 |
| 2 | l10_ops.py | l10_post_ops_combined | OUTPUT_LO | 11 | 2a684f8a |
| 3 | l6_ops.py | _layer6_attn_dep_anchor | AX_CARRY_LO/HI | 4 each | cc13c079 |
| 4 | l6_ops.py | putchar_think_protocol | AX_CARRY_LO/HI | 4 each | c5598039 |
| 5 | l6_ops.py | layer6_attn | AX_CARRY_LO/HI | 3 each | c5598039 |
| 6 | l6_ops.py | layer6_routing_ffn | AX_CARRY_LO/HI | 3 each | c5598039 |
| 7 | l6_ops.py | _layer6_ffn_dep_anchor | AX_CARRY_LO/HI | 3 each | c5598039 |
| 8 | l6_ops.py | layer6_relay_heads | AX_CARRY_LO/HI | 3 each | c5598039 |
| 9 | l5_ops.py | opcode_decode_ffn | OPCODE_BYTE_LO | 1 (`layer5_fetch`) | 1f209895 |
|10 | l5_ops.py | _opcode_decode_ffn_dep_anchor | OPCODE_BYTE_LO | 1 (`layer5_fetch`) | 1f209895 |

Note: #9 is the known regression. #1 and #2 are the highest-fanout candidates
and the most likely next regressions if they're being hit on step 1 anywhere
in the corpus.

Other concentrated clusters:
- `l3_ops.py`: `layer3_ffn`, `_layer3_ffn_dep_anchor`, `layer3_carry_forward_attn`
  all read `EMBED_LO/EMBED_HI/TEMP.*.-1` and write the same dim within the
  L3 batch (introduced e3981d3e / 073e5af1 / f80ca1d9).
- `l8_ops.py`: `layer8_mem_to_alu` reads `ADDR_B{0,1,2}_{LO,HI}.*.-1` with
  same-file writer `layer8_sp_gather_bake` (introduced 4a89eb21).
- `l10_ops.py`: `layer10_carry_relay*`, `layer10_alu`, `l10_post_ops_combined`
  cluster around CARRY / ALU_HI / OUTPUT_LO/HI (introduced
  0fa605e8 / 96691d98 / 2a684f8a / 8ef9c159).

## Safe records (41) — why they're not suspect

The "safe" set still has cross-step reads, but no same-file writer of the
base dim. The base dim is produced in an *earlier* scheduled layer, so the
cross-step alias is reading a genuinely-prior-step residual. These do not
exhibit the step-1=0 anti-pattern because the producer ran in the prior
step's later layer, before the consumer's layer in the next step.
Examples: `l7_ops.py:layer7_operand_gather` reads `OUTPUT_LO/HI.*.-1` —
no L7 op writes OUTPUT_*. They are written in L8/L10/L13/L14.

(There is a residual risk on the *very first* step even for safe records,
but the original mitigation — initializing prev-step residuals to a sentinel
or zero — has historically been correct here because the consumer is in a
non-bootstrap layer.)

## Recommended migration plan

**This is largely a templatable mass migration, with a small per-case tail.**
Rationale: 36 of the 42 suspects were introduced by 4 mechanical mass-rename
commits (c5598039 / 4a89eb21 / e3981d3e / f80ca1d9 / 073e5af1 / 1f209895 /
8ef9c159) that uniformly converted `<DIM>_PREV_STEP` literals to
`<DIM>.*.-1`. They share one root cause: an SCC analysis that flagged the
same-layer write→read edge as a cycle and was "broken" by pretending the
read was cross-step.

### Tier A — Templatable revert (≈ 30 records, 5 commits)

For these, the right fix is to revert the cross-step rename to a same-step
read AND restore the explicit `phase=` ordering (or `requires=`) that the
Phase 9.B commits dropped. Each of the rename commits has a paired
"drop phase=" commit (e.g. `1f5814a7`, `fc9ad8ad`, `b635160f`, etc.) — a
clean revert of *both* halves restores the legacy intra-layer ordering.
Group by file:
- `l6_ops.py` (14): revert `c5598039` + `1f5814a7` (`layer6_attn`),
  `e863dfd4` (`layer6_relay_heads`), companion phase-drop commits for
  `layer6_routing_ffn` / dep_anchors.
- `l8_ops.py` (6): revert `4a89eb21` + `001ad82d` + `8814bc7b`.
- `l3_ops.py` (8): revert `e3981d3e` + `f80ca1d9` + `073e5af1` + `c744e58c`
  + `18a90a4d`.
- `l5_ops.py` (2): revert `1f209895` + restore `fc9ad8ad` (already half-done
  via `d174ac85`).
- `l10_ops.py` (~6): revert `2a684f8a` / `5452dd49` / `8ef9c159` /
  `96691d98` / `0fa605e8` + companion phase-drops.

Risk: re-introduces manageable SCC edges that the scheduler used to handle
via explicit `phase=` ordering. Phase 8.A added those edges deliberately
to support the same-step Y→X dependency; the Phase 9.B SSA fiction was an
optimization that wrongly assumed the consumer never runs on step 1 with
no prior write.

### Tier B — Per-case (≈ 12 records)

These need a different fix, NOT a clean revert:

- **`l10_ops.py:l10_post_ops_combined` / `layer10_psh_stack0_passthrough_bake`
  OUTPUT_LO (writers=11)**: 11 same-file writers means a clean revert would
  re-introduce a multi-way intra-layer cycle. Instead, introduce an explicit
  step-1 prologue that initializes OUTPUT_LO at the start of the L10 batch
  (analogous to `layer14_temp_clear`'s role for TEMP), then keep `.*.-1`.
- **`l6_ops.py` AX_CARRY_LO/HI fan-out (12 records, all 5 L6 ops)**: all
  five L6 ops cross-step-read AX_CARRY and at least one of them writes it
  in the same step. A clean revert would create a 5-way SCC. Recommended:
  designate one canonical L6 writer of AX_CARRY (likely `layer6_relay_heads`),
  give it `phase=6` and make the others read same-step + `requires=`.
- **`l6_ops.py:putchar_think_protocol` AX_CARRY_LO/HI (2)**: this op is
  flag-gated and may not run every step; cross-step read may be intentional.
  Verify against runtime trace before reverting.
- **`l8_ops.py:layer8_head6_ax_carry_refresh OUTPUT_LO`** (introduced
  `ac4e7b9a` — explicitly a Phase 9 demo): revert candidate, but the
  commit message says "demo on SSA dim names", so confirm intent.

### Tier C — No action

The 41 safe records (no same-file writer) are correct as written.

## Methodology

- `audit.py`: regex parses each `CompilerIR(name="…", reads={…}, writes={…})`
  block per ops file; records every `"<DIM>.*.-1"` literal with its exact
  source line; classifies as suspect if any other op in the same file
  writes the base dim.
- `deepblame.py`: runs `git log -L <line>,<line>:<file>` per literal to walk
  back through the file-split commit and find both the rename commit
  (the one that introduced `.*.-1`) and the earliest cross-step
  introduction (`_PREV_STEP` or `.*.-1`).
- Raw data: `.agent-logs/cross_step_audit_data.json` (83 records with
  per-line blame, same-file writers, global writers, rename + introduce
  commits).

## Caveats

- "Same-file" is a proxy for "same scheduled layer". Most ops files
  correspond 1:1 to a layer; exceptions: `alu_ops.py`, `model_ops.py`,
  `flag_gated_ops.py`, `control_flow_heads.py`, `all_core_ops.py` —
  records in these files are filtered into the safe bucket here, but a
  human should sanity-check each one against the scheduler's actual
  layer batches. Only `control_flow_heads.py` (1 op, 3 records) appears
  in the data; the others are absent.
- The "lucky on step 2+" claim is not verified at runtime for each record
  in this audit; it is the structural hypothesis. A follow-up should run
  the SSA scheduler on each suspect and confirm that the cross-step read
  is satisfied by the same-file writer's prior-step output.
