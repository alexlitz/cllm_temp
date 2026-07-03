# Wave 1 A3.5 L14 mem_generation consumer diagnostic (2026-06-09)

Follow-up to A3.4 (commit b02be67c) which fixed the L10 broadcast Q
anchor (use ``STACK0_BYTE_h`` as the row marker; move OP_PSH gating to
K-side). The A3.4 commit's verification probe showed
``STACK0_BYTE_VAL_1_LO=2/3.00`` at S0@80 BI_1 row — confirming the
broadcast fires once. The A3.5 brief asserts the broadcast now
"correctly populates STACK0_BYTE_VAL_h_LO/HI from PSH" and pins the
remaining failure on the L14 mem_generation consumer.

**This pass found that the consumer-only framing is incomplete: the
broadcast only writes at the FIRST STACK0 frame in context, not at
subsequent frames that exist when SI fires. The L14 K-side gating
is ALSO mismatched relative to the broadcast row. Both layers
contribute and a single-layer fix doesn't move smoke.**

## Memory smoke baseline

- 1/6 passing (test_si_li_zero, accidental — byte = 0 is the OUTPUT
  band default).
- 5/6 failing: result is always 512 = 0x200 = the address itself,
  not the stored value (42, 99, etc).
- Full smoke: 45/51 passing — unchanged across A3.4, A3.5.

## What works

The L10 broadcast head (heads 8/9/10 at physical block L12) does
write to ``STACK0_BYTE_VAL_h_LO/HI`` (verified via direct W_o
inspection at dim 602/618 = ``STACK0_BYTE_VAL_1_LO/HI``):

```
W_o[602, base+0..15] = 3.0   # STACK0_BYTE_VAL_1_LO slots
W_o[618, base+16..31] = 3.0  # STACK0_BYTE_VAL_1_HI slots
```

And the L14 mem_generation V reads them (heads 1/2/3 at physical
block L27, slots 32-47/48-63 → OUTPUT_LO/HI).

## What's broken — three intertwined issues

### Issue #1 — broadcast fires only at the FIRST STACK0 frame

For the test program ``IMM 0x200, PSH, IMM 42, SI, IMM 0x200, LI,
EXIT``, autoregressive context has 5 STACK0 frames at p=80, 114, 148,
183, 218. After block L12:

```
S0@ 80 BI1 @ p=82:  VAL1_LO=2/3.00  (correct — byte 1 of 0x200 = 0x02)
S0@114 BI1 @ p=116: VAL1_LO=0/0.00
S0@148 BI1 @ p=150: VAL1_LO=0/0.00
S0@183 BI1 @ p=185: VAL1_LO=0/0.00
S0@218 BI1 @ p=221: VAL1_LO=0/0.00
```

The post-PSH STACK0 frames have **no broadcast value**. Root cause:

- The broadcast Q gate fires at **every** STACK0+BI_h row (slot 0
  has STACK0_BYTE_h + BYTE_INDEX_h + IS_BYTE positives = +3L; slot 33
  has STACK0_BYTE_h + BYTE_INDEX_h = ~+1).
- The K-side slot 33 has ``AP(33, OP_PSH, M)`` but OP_PSH is set at
  the **MARK_AX d=0 row** (p=100), NOT at the AX BI_h byte rows
  (p=101..104). At the K rows the broadcast attends to (MARK_AX BI_h
  byte rows), OP_PSH=0.
- So K[33] = M*BI_h(=1) + M*MARK_AX(=0) + M*OP_PSH(=0) = M for
  **both** the IMM step's AX BI_h row (K@67) and the PSH step's AX
  BI_h row (K@102).
- Q@p=82 (S0@80 BI_1) has only **one** causal K candidate (K@67) —
  full mass → V*3.0 broadcast lands.
- Q@p=116 (S0@114 BI_1) has **two** causal K candidates (K@67, K@102)
  — mass splits 50/50. Both have CL_LO[2]=1 (AX byte 1 of 0x200 =
  0x02 at both IMM and PSH). The broadcast SHOULD land at ~3.0 too…
  but probe reads 0/0.0. Mechanism still unclear (possibly the
  softmax1 +1 baseline cancels in the multi-K regime; possibly L11
  FFN re-zeros the dim — see Issue #3).

### Issue #2 — L14 head_h K-side selector mismatched with broadcast row

L14 mem_generation head_h (h=1, 2, 3) at ``ops/l14_ops.py:480``:

```python
byte_idx_dim = [None, BD.BYTE_INDEX_0, BD.BYTE_INDEX_1, BD.BYTE_INDEX_2][h]
k.append(AP(0, byte_idx_dim, L))
```

For h=1 the K selects BYTE_INDEX_0 rows. The L10 broadcast for
VAL_1 lives at the BYTE_INDEX_1 row of each STACK0 frame. So even
if the broadcast were broadly populated, head_1's K rows wouldn't
intersect with the broadcast.

Naive fixes attempted:

1. **Add ``AP(0, STACK0_BYTE_h, L)``** as an additional K positive.
   No smoke improvement — the recent K rows (alibi prefers them)
   still don't have VAL_h.

2. **Add ``AP(0, BYTE_INDEX_h, L)``** as an additional K positive
   (so K selects both BI_(h-1) for PSH and BI_h for SI). No smoke
   improvement.

Alibi 5.0 on L14 heads 0-7 strongly prefers recent K rows. From
Q@p=190 (SI MEM addr_b1 row) to K@p=82 (S0@80 BI1 — the only row
with the broadcast) is distance 108, alibi penalty -540. Any K row
within ~30 of Q gets +100 advantage. The broadcast at p=82 is
effectively unreachable.

### Issue #3 — dim aliasing (Phase 7.E semantic refs)

```
dim 602: ['FORMAT_PTR_LO', 'STACK0_BYTE_VAL_1_LO']
dim 618: ['FORMAT_PTR_HI', 'STACK0_BYTE_VAL_1_HI']
dim 729: ['PC_VIA_LEV_DETECTOR_HI', 'STACK0_BYTE_VAL_2_LO']
dim 745: ['BP_VIA_LEV_DETECTOR', 'STACK0_BYTE_VAL_2_HI']
dim 761: ['SP_VIA_LEV_DETECTOR', 'STACK0_BYTE_VAL_3_LO']
```

5 of 6 STACK0_BYTE_VAL_h dims are aliased onto pre-existing dims.
Any other op that zeroes (or overwrites) ``FORMAT_PTR_LO``,
``PC_VIA_LEV_DETECTOR_HI``, ``BP_VIA_LEV_DETECTOR``, or
``SP_VIA_LEV_DETECTOR`` at STACK0 byte h rows will silently clobber
the broadcast. This may be why Issue #1's S0@114 reads as zero
despite the broadcast head's softmax math saying ~3.0 should land
there.

Suggested next-step probe: dump every ATTN/FFN write to dim 602 at
S0@114 BI_1 row across all 32 physical blocks. Identify the zeroing
writer.

## Combined fix path (recommended)

The single-layer L14 consumer fix the A3.4 commit anticipated does
NOT exist. Three-step remediation:

1. **Fix broadcast distribution** — change the L10 broadcast Q gate
   to FIRE ONLY at the S0 frame matching the current PSH step, OR add
   a separate persistence head (mirror ``layer10_stack0_byte_relay``
   head 6) that carries STACK0_BYTE_VAL_h across STACK0 frames.

2. **Audit dim 602/618/729/745/761 aliasing** — confirm no other
   op zeros these positions at STACK0 byte rows. If contention
   exists, allocate a new non-aliased dim family for the broadcast.

3. **Fix L14 K-side** — change ``byte_idx_dim`` index from
   ``[None, BI_0, BI_1, BI_2][h]`` to ``[None, BI_1, BI_2, BI_3][h]``
   AND verify post-step ``_clear_l14_mem_generation_overbroad_sp_suppression``
   still produces correct PSH behavior. The "off by one" K may be
   compensated by a downstream layer for the PSH path — needs probing.

None of these is a single-rule whack-a-mole fix; see memory note
``feedback_single_rule_fixes_are_zero_sum.md``.

## Files

* ``c4_release/tools/probe_a3_5_l14_consumer.py`` — section-by-section
  probe (broadcast value at each STACK0 BI_h row, MEM_ADDR_SRC at MEM
  rows, all BI_1 K candidates with broadcast residual, OUTPUT at MEM
  rows after L14). Run with ``python c4_release/tools/probe_a3_5_l14_consumer.py``.
* ``c4_release/tools/probe_a3_5_broadcast_persistence.py`` — cross-
  layer dump of STACK0_BYTE_VAL_1 at each STACK0 BI_1 row. Confirms
  the broadcast only fires at S0@80.
* ``c4_release/neural_vm/unified_compiler/ops/l10_ops.py:1659-1776``
  — broadcast spec (Q anchor + K-side OP_PSH gate).
* ``c4_release/neural_vm/unified_compiler/ops/l14_ops.py:417-581``
  — mem_generation consumer (heads 0-3 addr; heads 4-7 val).
* ``c4_release/docs/A3_BROADCAST_DIAGNOSTIC_2026_06_07.md`` — A3.4
  predecessor doc (broadcast Q anchor fix).
