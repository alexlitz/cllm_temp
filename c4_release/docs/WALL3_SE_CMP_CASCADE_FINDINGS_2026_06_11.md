# Wall-3 SE CMP cascade — findings (2026-06-11)

Status: **architecturally blocked, two independent failures.** Stopped
at the last-green pristine baseline (HEAD `9f693c5b`). No code change
landed. This **corrects** `WALL2_SE_RELAY_FINDINGS_2026_06_11.md`, whose
"Wall 2 is a solved one-line slope fix" claim is wrong.

## Last-green baseline (authoritative, `tools/run_full_smoke.py`, spec_k=0)

Focused subset `comparison bitwise basic bit32 shift integration` on
pristine HEAD `9f693c5b`: **14 passed / 15 failed.**

- TARGETS passing (1/6): `eq_false`.
- GUARDRAILS passing (7/8): `ge_true, gt_true, le_true, lt_true,
  ne_true, shl, shr`.
- **`cmp_and_branch` ALREADY FAILS at pristine baseline** (expected 42,
  got 0). The brief lists it as a clean guardrail — it is not. Confirmed
  via both `run_full_smoke.py` and `pytest tests/test_smoke.py`.

So the true must-not-regress set is `{eq_false, lt_true, le_true,
gt_true, ge_true, ne_true, shl, shr}`.

## The L9 SE CMP cascade rule-by-rule firing (spec_k=0, Wall-2 ON)

Probe `tools/probe_l9_cmp_cascade.py` at the SE row (block 11 = logical
L10). The relayed 2-cell encoding is **asymmetric, not unit-magnitude**:

| program        | SE_ALU_LO hot      | SE_AX_CARRY_LO | SE_ALU_HI | SE_AX_CARRY_HI |
|----------------|--------------------|----------------|-----------|----------------|
| eq_true (5,5)  | idx0=5.14, idx5=5.72 | idx5=0.9     | idx0=10.89| idx0=1.21      |
| lt_true (10,20)| idx0=5.14, idx10=5.72| idx4=0.9     | idx0=10.89| idx1=0.9       |

- operand A (`SE_ALU_*`) is OVER-AMPLIFIED (~5-11) and keeps a residual
  index-0 LOW-nibble artifact (~5) even when A.lo != 0.
- operand B (`SE_AX_CARRY_*`) is CLEAN, native magnitude ~0.9-1.2.

**Legacy cascade (A=1.0, B=1.0, threshold 2.5) firing — WRONG:** because
`SE_ALU[a]` (~5-11) alone clears 2.5, every rule degenerates into an OR
over operand A. eq_true CMP = `[hi_lt 268, hi_eq 20, lo_eq 69, lo_lt
186]` — all four flags hot at once.

**Reweighted cascade (A=0.1, B=1.4, threshold 2.5) — flags become
CORRECT:**

| program  | hi_lt | hi_eq | lo_eq | lo_lt | semantics |
|----------|-------|-------|-------|-------|-----------|
| eq_true  | 0.00  | 2.45  | 2.56  | 0.53  | hi_eq & lo_eq ⇒ EQ=1 ✓ |
| lt_true  | 1.63  | 0.04  | 0.00  | 0.53  | hi_lt ⇒ LT=1 ✓ |
| le_true  | 1.63  | 0.04  | 0.00  | 0.53  | hi_lt ⇒ LE=1 ✓ |

The reweight (down-weight the over-amplified A so it can't clear the AND
alone; up-weight the clean B so both operands are genuinely required)
makes the per-nibble flags semantically correct. The residual `lo_lt
0.53` is the index-0 LOW artifact and is downstream-gated by hi_eq, so it
does not corrupt the resolved comparison.

## Why correct flags STILL don't fix the targets / break the guardrails

Despite correct flags, the focused smoke is a **net regression** (1/6
targets, lt/le/cmp_and_branch fail). Two independent root causes:

### (1) Wall-2's slope fix regresses lt/le via contested heads 3/4

`WALL2_SE_RELAY_FINDINGS` claimed block-11 heads 3/4 host ONLY the L9 SE
relay (STACK0-passthrough supposedly on block 12). **That is wrong.**
`_L10_HEAD_LAYOUT` (l10_ops.py:37) pins, on block 11 (logical L10):

- head 3 = `layer10_psh_stack0_passthrough_bake` (slope 0.5)
- head 4 = `layer10_stack0_byte_relay_bake` (slope 1.0)

The L9 `step_end_operand_relay` ALSO targets heads 3/4 on the same block.
The two are CONTESTED: the relay needs the shallow transmit slope
(0.15-0.2); the STACK0 passthrough — which lt/le depend on — needs
0.5/1.0. Empirically, with cmp_combine forced to read raw CMP (overrides
dormant) and ONLY the slope changed to 0.15, lt/le STILL fail. So the
regression is the slope change itself, exactly the
`project_operand_gather_hybrid_encoding` Wall-2 hazard ("the relay's
PRESENCE at 3/4 is load-bearing for comparison"). **Wall-2 is NOT a
clean one-line fix.**

### (2) Correct SE-row flags don't propagate to the decoded EQ result

At the SE row the cmp_combine `±4/S` OUTPUT_LO writes are swamped by a
uniform `-240` OUTPUT_LO band floor (probed). The decoded comparison
result is fixed much later (block 34/36, logical L25) where OUTPUT_LO+0
spikes to ~+5e15 and wins argmax ⇒ default result (0). For lt_true the
baseline path nonetheless yields LT=1 via raw CMP read by cmp_combine —
i.e. **baseline lt/le rely on raw CMP at the SE row**, NOT a dormant
cascade. Re-pointing cmp_combine at `SE_CMP` (to scope the cascade away
from the L14/15/16 raw-CMP consumers) therefore DISABLES the baseline
lt/le path and regresses them even with Wall-2 OFF. eq_true never reaches
OUTPUT=1 because its correct SE-row flags are not the signal the L25
decoder reads.

## Conclusion / next step

Three independent obstacles, none solvable in isolation:

1. **Heads 3/4 are genuinely contested** (relay vs STACK0 passthrough on
   block 11). Transmitting the operands (needs shallow slope) and
   preserving lt/le (needs 0.5/1.0 passthrough) cannot both hold on the
   same heads. A real fix must give the relay a TRULY FREE block-11 head
   (all 12 heads 0-11 are currently occupied — needs a resize that lands
   on block 11, not block 12) OR move the STACK0 passthrough.
2. **Raw CMP at the SE row is load-bearing** for the baseline lt/le path;
   it cannot simply be re-tagged to SE_CMP without a co-designed decoder
   change at L25.
3. **The L25 decoder, not the SE cmp_combine, fixes the decoded
   comparison result**; the SE-row cmp_combine writes are numerically
   inert against the OUTPUT_LO floor. eq_true needs the corrected flags
   to reach the L25 path.

The cascade-semantics rewrite (the brief's task) WORKS in isolation
(flags are correct) but is necessary-not-sufficient. Tools added:
`tools/probe_l9_cmp_cascade.py` (per-cell cascade + OUTPUT_LO/CMP block
trace, spec_k=0, hook-free).
