# Wave 1 A3.6 — dim 602 zeroing attribution (2026-06-09)

Follow-up to A3.5 (commit 6ff4a9c0) which proposed:

> Recommended next-step probe: dump every ATTN/FFN write to dim 602
> at S0@114 BI_1 row across all 32 physical blocks to identify the
> zeroing writer.

**This pass ran that probe. There is no zeroing writer.** The
broadcast simply never lands at S0@114, 148, 183, or 218; the dim
stays at 0.00 through every block, every sub-block (pre → ATTN →
post-ATTN → FFN → post-FFN). The A3.5 "aliased dim contention"
hypothesis (Issue #3) is **disproved**.

## Probe results (S0@80 BI_1 vs. S0@114 BI_1)

`tools/probe_a3_6_dim_602_zeroing.py` dumps the STACK0_BYTE_VAL_1_LO
(dim 602) and STACK0_BYTE_VAL_1_HI (dim 618) bands at every block
boundary for each S0 BI_1 row. Per-block columns: argmax index,
argmax value, abs-sum (L1) for the band; signed ATTN delta and
signed FFN delta to band L1.

```
S0@80 BI_1 row @ p=82  (working frame)
  L11: pre=0,  attn=0,  ffn=0,  out=0
  L12: attn LO_d=+3.00, HI_d=+3.00  out=3.0 LO[2], 3.0 HI[0]  <-- broadcast lands
  L13..L31: ATTN +3, FFN -3 (cancel)  out stays 3.0  <-- carried forward
```

```
S0@114 BI_1 row @ p=116  (broken frame)
  L0..L31: pre=0, attn_d=0, ffn_d=0, out=0  <-- nothing ever writes
```

Same nothing-pattern at S0@148, 183, 218.

## Why the broadcast doesn't fire at later frames

`tools/probe_a3_6_q_anchor_check.py` dumps the pre-L12 residual at
each AX BI_1 row (candidate K rows) and each STACK0 BI_1 row
(candidate Q rows). Two findings:

1. **OP_PSH is set only at MARK_AX d=0 (p=100), not at any AX BI_h
   byte row.** Specifically OP_PSH at p=102 (AX BI_1 of the PSH
   step) = 0.00. The current spec's K-side gate
   `AP(33, OP_PSH, M)` at slot 33 therefore contributes 0 at every
   K candidate row. The K-side OP_PSH gate is dead. Both K@67 (IMM
   step AX BI_1) and K@102 (PSH step AX BI_1) have identical slot
   33 alignment ≈ 2M (MARK_AX + BYTE_INDEX_1 contributions only).

2. **CLEAN_EMBED at the two K rows differs by step semantics.** AX
   BI_1 at p=67 (IMM 0x200): LO[2]=1, HI[0]=1 (= byte 0x02, the AX
   register's byte-1 of 0x200). AX BI_1 at p=102 (PSH): LO[0]=1,
   HI[0]=1 (= byte 0x00, the PSH instruction's byte-1 — PSH is a
   1-byte opcode, byte 1 is meta scratch, NOT the AX register
   state). K@102 carries the wrong value.

So at Q@p=82 (S0@80, before any PSH happens), only K@67 is causal;
mass → K@67 → V copies CLEAN_EMBED[byte=2] → broadcast lands as
3.0 at slot 2. At Q@p=116 (S0@114, after PSH), both K@67 and K@102
are causal with identical alignment scores. Softmax1 with two K
candidates of equal magnitude and an unbounded +1 baseline puts
~50% mass on each K. K@102 contributes wrong data
(CLEAN_EMBED[byte=0]) and K@67 contributes correct data
(CLEAN_EMBED[byte=2]). Even if the average were taken, the result
at dim 604 should be 0.5 * 3.0 = 1.5. But the probe reads 0.00.

**Mechanism for the 0.00 read:** softmax1 baseline 1.0 + 2*exp(Q·K)
denominator with very large Q·K terms dominates → both K weights
≈ 0.5. V_LO at the two rows is the one-hot CLEAN_EMBED_LO band; at
K@67 it's slot 2 with value 1, at K@102 it's slot 0 with value 1.
The O matrix writes `slot_k → STACK0_BYTE_VAL_1_LO + k`. With 50%
mass on each K, slot 2 gets 0.5 contribution and slot 0 gets 0.5
contribution, scaled by O=3.0 → 1.5 each. But the probe reports
both dim 602 and dim 604 as 0.00. This implies softmax1 mass is
dominated by the +1 baseline rather than the K candidates — the Q·K
scores are not actually large enough relative to the softmax1
baseline at non-broadcast frames. (At S0@80 with one K candidate
the baseline is "diluted" away by causal masking; with two K
candidates the baseline is half-as-effective at suppression too.)

Hand-verification of the slot-33 dot product:

* Q slot 33 at Q row = `1*STACK0_BYTE_1 + 1*BYTE_INDEX_1 - 1*CONST
  = 1 + 1 - 1 = 1`.
* K slot 33 at K row (AX BI_1) = `M*MARK_AX + M*BYTE_INDEX_1 +
  M*OP_PSH - M*OP_SI/SC/JSR/ENT = M + M + 0 - 0 = 2M`.
* Q*K slot 33 = 1 * 2M = 2M = 1e6 (with M=500*S, S=100, L=100).

Slot 0 main: Q*K = 4L^2 + (suppression contributions) ≈ 4e4. So
slot-33 dominates: total Q·K ≈ 2M = 1e6 at BOTH K@67 and K@102.

The softmax1 baseline 1.0 vs. exp(1e6) — `exp(1e6)` overflows to
+inf at fp32. So in practice both K rows get *finite* weight after
clamping at fp16/bf16 boundaries — depending on whether the
implementation uses softmax1's "subtract max" trick, the mass may
go entirely to one K row (numerical tiebreak) or split arbitrarily.
With identical Q·K at K@67 and K@102, the implementation has no
preference signal.

**The root cause is that the K-side OP_PSH gate is dead** (OP_PSH
is on the wrong row for the K rule to read it), so the head cannot
distinguish the IMM-step AX BI_h row from the PSH-step AX BI_h row.
At every post-PSH frame, multiple K candidates compete with no
gating signal.

## What does NOT work as a fix

- Move OP_PSH gate to Q side — same problem: OP_PSH is not set at
  the STACK0 Q row either (A3 diagnostic confirmed this in commit
  fd109b86).
- Restrict K to MARK_AX d=0 row — V would not be CLEAN_EMBED (which
  only fires at byte-index rows); the head can't read AX register
  state.
- Single-rule whack-a-mole on the K gate (e.g., subtract OP_IMM
  from K slot 33) — OP_IMM is also at d=0 only, same problem.
- An L11/L12 FFN broadcast that propagates OP_PSH from d=0 to BI_h
  rows of the same step — this works but is a new scaffolding op
  with non-trivial dim claims and gate boundary work (~A3 itself
  in effort).

## What WOULD work

Per A3.5 "Combined fix path", the recommended structural fix is:

> A separate persistence head (mirror
> `layer10_stack0_byte_relay` head 6) that carries
> STACK0_BYTE_VAL_h across STACK0 frames.

That is, accept that the broadcast only fires at S0@80, then
*relay* the value to subsequent STACK0 frames via a copy head whose
Q reads at STACK0+BI_h of frame N and K reads at STACK0+BI_h of
frame N-1 (or directly at the seed frame via a "most recent"
mechanism). This avoids the OP_PSH gate problem entirely.

Alternative: introduce a new dim `STEP_OP_PSH_AT_BI_h` that an L9
or earlier FFN populates at the BI_h byte row of the active PSH
step (reading OP_PSH from the d=0 row via attention into a
band-position-keyed dim). Then K-side `AP(33, STEP_OP_PSH_AT_BI_h,
M)` at K row would be functional.

Both are multi-op structural changes — outside this pass's scope.

## Memory smoke

Baseline: 1/6 passing, 5/6 failing (45/51 full smoke). Confirmed
pre-pass and post-pass: no change (this pass is diagnostic; no
weight changes).

## Files

* `c4_release/tools/probe_a3_6_dim_602_zeroing.py` — per-block,
  per-sub-block ATTN/FFN delta dump for dim 602 + 618 bands at all
  5 STACK0 BI_1 rows. **Disproved the A3.5 zeroing hypothesis.**
* `c4_release/tools/probe_a3_6_q_anchor_check.py` — pre-L12 dim
  inspection of OP_PSH and CLEAN_EMBED at AX BI_1 K candidate rows.
  Identified the dead K-side OP_PSH gate.
* `c4_release/neural_vm/unified_compiler/ops/l10_ops.py:1659-1776`
  — `_layer10_psh_ax_broadcast_head_spec` (broadcast head with dead
  K-side OP_PSH gate at slot 33).
* `c4_release/docs/A3_BROADCAST_DIAGNOSTIC_2026_06_07.md` — A3
  predecessor (original Q-side OP_PSH gate diagnosis).
* `c4_release/docs/A3_5_L14_CONSUMER_DIAGNOSTIC_2026_06_09.md` —
  A3.5 predecessor (zeroing-writer hypothesis, now disproved).
