# Wall-4 — SE CMP result decode row is the architectural wall (2026-06-11)

Status: **architecturally blocked at a SHARPER, fully-evidenced layer.**
Stopped at the last-green pristine baseline (HEAD `481612c3`). No
behavioural code change landed. This **supersedes** Piece C of the
3-part coordinated-fix brief and Wall-3's "L25 floor" framing: the SE
cmp_combine result is inert not (only) because of an OUTPUT_LO floor,
but because **the exit code is decoded from the binop step's MARK_AX-row
OUTPUT_LO, while the entire migrated SE cmp machinery writes at the
MARK_SE_ONLY row.** The two rows are different sequence positions; the
SE result physically cannot reach the decode relay.

## What this session PROVED (all spec_k=0, hook-free, batched smoke path)

Pieces A and B from the brief were composed and **work exactly as
designed** — they are NOT the blocker. Piece C is.

### Pristine baseline (focused subset `comparison bitwise basic bit32 shift integration`)

**14 passed / 15 failed.** Matches Wall-3. `eq_false` is the only TARGET
passing; `cmp_and_branch` ALREADY FAILS at baseline (the brief lists it
as a guardrail — it is not). True must-not-regress set: `{eq_false,
lt_true, le_true, gt_true, ge_true, ne_true, shl, shr}`.

### Piece A — relay re-home to FREE block-11 heads 5/6 (VERIFIED MECHANICALLY CORRECT)

Re-homed `layer9_step_end_operand_relay` from L9 head-pool indices 3/4
(→ runtime block-11 heads 3/4, slope-clobbered to 0.5/1.0 by the
phase-999 `layer10_residual_alibi_slopes` op) to **head 5** (head B = 6),
the genuinely-free block-11 slots (absent from the slope-writer table in
`ALIBI_SLOPE_COLLISION_MAP_2026_06_11.md`). Implementation: pin in
`_L9_HEAD_LAYOUT` (`l9_ops.py`).

Transmission is RESTORED (probe_se_relay.py, L9_BLOCK=11, on the A-only build):

| dim            | baseline | after re-home |
|----------------|---------:|--------------:|
| SE_ALU_LO      | ~1.41    | **4.99–5.38** |
| SE_ALU_HI      | ~0       | **4.98–9.83** |
| SE_AX_CARRY_LO | ~0.001   | **0.82–0.87** |
| SE_OP_EQ       | ~0.01    | **4.30**      |
| SE_CMP_GROUP   | ~0.002   | **0.86**      |

**But Piece A alone REGRESSES the focused subset 14 → 12** (lt_true and
le_true PASS → FAIL). Restoring transmission activates the L9 SE CMP
cascade, which under the live downstream path corrupts the lt/le result.
Net-neutral was the brief's claim for the 5/6 re-home; it is NOT
net-neutral. (This reproduces Wall-2's finding for BOTH the 3/4-slope-fix
and the 5/6-move.)

### Piece B — reweight the SE cascade A=0.1 / B=1.4 (VERIFIED: FLAGS BECOME CORRECT)

Applied to all four loops of `_layer9_cmp_rules` (hi_eq/lo_eq/hi_lt/lo_lt):
operand-A (`SE_ALU_*`) condition weight 1.0 → 0.1, operand-B
(`SE_AX_CARRY_*`) 1.0 → 1.4, threshold unchanged (2.5). The SE cascade
firing table (probe_l9_cmp_cascade.py, spec_k=0, on the A+B build) is now
**semantically correct**:

| program           | CMP+0 hi_lt | CMP+1 hi_eq | CMP+2 lo_eq | CMP+3 lo_lt | flags ⇒ |
|-------------------|------------:|------------:|------------:|------------:|---------|
| eq_true (5==5)    | 0.00        | **1.80**    | **1.11**    | 0.19        | EQ ✓    |
| lt_true (10<20)   | **1.10**    | 0.00        | 0.00        | 0.19        | LT ✓    |
| le_true (10<20)   | **1.10**    | 0.00        | 0.00        | 0.19        | LE ✓    |

The flags are exactly what the comparison logic needs. **Yet the focused
subset stays 12 / 29** — lt_true and le_true STILL FAIL (got 0), eq_true
STILL FAILS (got 0). Correct flags do NOT repair lt/le and do NOT fix eq.

## The wall, located exactly (Piece C is unreachable)

`probe_l9_cmp_cascade.py` with `TRACE_OUTPUT` traces OUTPUT_LO at the
binop **SE row** across blocks. On the A+B build (correct flags):

```
eq_true: CMP=[hi_eq 1.80, lo_eq 1.11]   (correct EQ flags)
lt_true: CMP=[hi_lt 1.10]               (correct LT flags)
  blk11/12/13 SE-row OUTPUT_LO: -240.0 at EVERY index 0..15 (uniform floor)
  blk34/35     SE-row OUTPUT_LO: index 0 wins argmax (+5.4e15), all others
               -4.5e16  -> decodes 0, IDENTICAL for eq_true and lt_true.
```

Two independent facts, each fatal to Piece C as specified:

1. **The SE-row OUTPUT_LO is swamped by a uniform −240 floor** at blocks
   11–13 (where the SE cmp_combine ±4/S writes land). The result writes
   are numerically invisible. This is Wall-3's "−240 floor" — confirmed.

2. **The SE row is NOT the exit-decode site.** The exit code is
   `BatchedPureNeuralRunner._decode_exit_code`: it reads the AX register
   bytes emitted at EXIT. Those bytes come from `AX_FULL`, populated by
   **L3 head 5** (`layer3_carry_forward_attn`, `l3_ops.py:1177`), which
   reads **prev-step OUTPUT_LO at the MARK_AX-marker row**
   (`OUTPUT_LO.*.-1` back-edge). So the decode-relevant value is the
   binop step's **AX-row** OUTPUT_LO — a DIFFERENT sequence position from
   the SE row. PROOF: at the SE row, eq_true (FAIL, exit 0) and lt_true
   (PASS at baseline, exit 1) have **byte-identical OUTPUT_LO across all
   blocks 11/34/35/36** (both decode 0 at the SE row). Their different
   exit codes therefore cannot originate at the SE row.

**Why lt/le pass at baseline but break when the cascade activates:** at
baseline the relay is dead, the SE cascade is dormant (CMP+0..3 = 0 at
the SE row), and lt/le get their correct answer from the AX-row default
output path. Activating the cascade (Piece A) drives raw `CMP+i` which
is relayed cross-step back to MARK_AX (per the L9 cascade's documented
"BZ/BNZ predicate gate reads the raw cascade at MARK_AX cross-step via
prev-step OUTPUT relays", `l9_ops.py:550`), corrupting the AX-row value
lt/le depended on — independent of whether the SE flags are correct.

**Piece C as specified ("amplify the SE cmp_combine OUTPUT writes to
override the −240 floor") cannot work:** even an infinite-magnitude write
at the SE row lands on a position the L3-head-5 decode relay never reads.
The only Piece C that could close the loop is a *new* SE→AX OUTPUT relay
that copies the SE-row result back to the binop AX-row OUTPUT_LO before
L3 head 5's cross-step read — but that op collides head-on with the live
AX-row default path that lt/le currently use, and re-architecting the
AX-row decode is a multi-op co-design, not a "surgical floor override".

## Conclusion / next step (the real frontier spec)

Pieces A + B are the *solved* parts: relay re-home to free block-11 heads
5/6 (transmission restored) + cascade reweight A=0.1/B=1.4 (flags
correct). They are necessary and they work. The unsolved frontier is a
**decode-row mismatch**, not a floor:

> The migrated SE cmp_combine writes the comparison result at MARK_SE_ONLY,
> but the exit-code decode relay (L3 head 5 → AX_FULL) reads the binop
> step's MARK_AX-row OUTPUT_LO. A real fix must make the SE-row result
> land on the AX-row OUTPUT_LO of the same step (or re-point L3 head 5 at
> the SE row) WITHOUT disturbing the AX-row default path the baseline
> lt/le/gt/ge rely on. This is a co-designed decode-row migration, the
> genuine multi-session frontier.

Tools added: `tools/probe_ax_row_output.py` (traces OUTPUT_LO/CMP at the
binop MARK_AX row — the real decode site — across blocks 8..36, spec_k=0,
hook-free). The A/B edits were validated then reverted; the worktree is
held at the green pristine baseline (14/29 focused, full smoke
unchanged).
