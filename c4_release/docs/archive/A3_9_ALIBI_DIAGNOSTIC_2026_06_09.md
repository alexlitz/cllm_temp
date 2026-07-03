# Wave 1 A3.9 — ALiBi pin doesn't help; slot-33 K-side dominates (2026-06-09)

Follow-up to A3.7 (commit 6bfc009a) which pinned ALiBi slope=1.0 on the
L10 broadcast heads 8/9/10 per the BLOG_SPEC "KV Cache Pruning" §
"latest-write-wins" mechanism, and A3.8 (commit dfe729f9) which declared
the L14 consumer reads. The A3.7 pin landed verifiably at
``model.blocks[12].attn.alibi_slopes`` (slots 8/9/10 = 1.0) but the
broadcast persistence probe (``probe_a3_5_broadcast_persistence.py``)
still reported VAL1_LO=0/0.00 at S0@114 BI_1 row and at every subsequent
STACK0 frame. This pass investigates why the ALiBi pin alone is
insufficient.

## Hypothesis testing

Per the diagnostic plan:

1. **Slope=1.0 isn't steep enough at distance ~30.** REJECTED. Distance
   K@67→Q@116 = 49, K@102→Q@116 = 14. ALiBi differential = 35 nats,
   which in softmax space gives e^35 ≈ 1.5e15 — overwhelming if content
   scores are equal.

2. **Pin is on the wrong block.** REJECTED. Direct dump of
   ``model.blocks[X].attn.alibi_slopes`` for X = 10, 11, 12, 13:

   ```
   L10 (num_heads=8):  [0.5, 0.25, 0.125, 0.0625, 0.0313, 0.0156, 0.0078, 0.0039]
   L11 (num_heads=8):  [5.0, 1.0, 1.0, 0.5, 1.0, 0.0156, 0.0078, 0.0039]
   L12 (num_heads=12): [0.63, 1.0, 1.0, 0.16, 0.099, 0.0625, 1.0, 0.025,
                        1.0, 1.0, 1.0, 0.0039]
   L13 (num_heads=8):  default decay
   ```

   L12 has 12 heads (post-resize) and slopes[8/9/10] = 1.0, matching
   the A3.7 pin. Pin lives where intended.

3. **softmax1 interaction.** REJECTED — see hypothesis 4. The +1
   baseline contributes negligible weight when content scores are
   ~1e4.

4. **Content-addressed K-side dwarfs ALiBi.** CONFIRMED, with refinement.

   Direct dump of Q*K for head 8 at Q@p=116 (S0@114 BI_1):

   ```
    kp        dot    alibi      total  dist
   100   46747.79   -16.00   46731.79    16  <-- MARK_AX d=0 row WINS
   102   32093.47   -14.00   32079.47    14
    67   32093.47   -49.00   32044.47    49
    65   23240.92   -51.00   23189.92    51
   116   22823.71    -0.00   22823.71     0
   ```

   The head attends to K@100 — the MARK_AX d=0 row of the PSH step —
   NOT to K@102 (AX byte_1 row). Why: at K@100, slot 33 has
   ``MARK_AX*M + OP_PSH*M = 2M = 1e4`` while at K@102 only ``BI_1*M = M
   = 5e3``. Slot 33 swings 5000 nats — far more than ALiBi can
   compensate over 16 vs 14 positions.

   Worse, K@100 has CLEAN_EMBED = 0 (the AX marker row carries the
   register marker, not the AX value), so the broadcast V writes
   zero. The probe shows VAL1_LO = 0 across all 32 blocks.

## Fix

`c4_release/neural_vm/unified_compiler/ops/l10_ops.py:1686-1700` —
``_layer10_psh_ax_broadcast_head_spec``: set ``M = 0.0`` (was
``50.0 * S = 5000``).

Rationale: with M=0, slot-33 K-side contributions vanish. The
remaining K-selection signal is:

* slot 0 K-side: ``L*MARK_AX + L*BI_h + L*IS_BYTE + L*(H1+AX_IDX)``.
  At AX byte_h row (e.g. K@102): MARK_AX=0, BI_h=1, IS_BYTE=1,
  H1+AX_IDX=1 → 3L = 300.
* At MARK_AX d=0 row (e.g. K@100): MARK_AX=1, BI_h=0, IS_BYTE=0,
  H1+AX_IDX=1 → 2L = 200.
* AX byte_h rows now beat MARK_AX rows by ~10000 nats on slot 0
  (Q[0]*K[0] differential).
* Among AX byte_h candidates, ALiBi slope=1.0 picks the most recent.

## Post-fix probe

`probe_a3_5_broadcast_persistence.py`:

```
S0@ 80 BI1: VAL1_LO = slot 2 / 3.00   <-- byte 0x02 of 0x200, correct
S0@114 BI1: VAL1_LO = slot 0 / 3.00   <-- broadcast LANDS but wrong content
S0@148 BI1: VAL1_LO = slot 0 / 3.00   <-- broadcast LANDS but wrong content
S0@183 BI1: VAL1_LO = slot 2 / 3.00   <-- byte 0x02, correct
S0@218 BI1: VAL1_LO = slot 2 / 0.00   <-- broadcast doesn't fire (Q gate?)
```

The broadcast now fires at every STACK0 frame (was firing only at
S0@80 pre-fix). At S0@114, S0@148 the value is wrong because K@102,
K@135 (most-recent AX BI_1 rows) carry the WRONG CLEAN_EMBED:

```
K@67  (AX BI_1 inside IMM 0x200 step): CLEAN_EMBED_LO = slot 2  CORRECT
K@102 (AX BI_1 inside PSH step):       CLEAN_EMBED_LO = slot 0  WRONG
K@135 (AX BI_1 inside IMM 42 step):    CLEAN_EMBED_LO = slot 0  PSH'd already
K@170 (AX BI_1 inside IMM 0x200 step): CLEAN_EMBED_LO = slot 2  CORRECT
```

C4's instruction layout puts the immediate-value bytes at AX BI_h
rows only for OP_IMM. At OP_PSH and OP_SI the AX BI_h rows hold
zero/meta bytes, not the live AX register value. The BLOG_SPEC's
"latest-write-wins" idea presumes each step REWRITES the full register
file, but the current weight layout doesn't do that — only OP_IMM
writes AX.

## Smoke

- Pre-fix: 1/6 memory tests pass (test_si_li_zero — trivially passes
  because OUTPUT default is 0). 45/51 full smoke.
- Post-fix: same numbers. No regression, no improvement.

The fix is **necessary** (broadcast now lands at all 5 STACK0 frames vs
1 before) but **not sufficient** because the K candidates ALiBi selects
between carry the wrong content at non-IMM steps.

## What's blocked next

Per A3.6's "What WOULD work": a separate persistence head (mirror of
``layer10_stack0_byte_relay`` head 6) that broadcasts CLEAN_EMBED from
the most-recent OP_IMM step's AX byte_h row, not from the most-recent
MARK_AX row of any step. Alternative: bake an FFN rule that copies
the AX register value onto every AX byte_h row of subsequent
non-IMM steps. Both are multi-op structural changes.

## Files

* ``c4_release/neural_vm/unified_compiler/ops/l10_ops.py:1686-1700`` —
  the M=0 change.
* ``c4_release/tools/probe_a3_9_alibi_attn_weights.py`` — dumps ALiBi
  slopes for blocks 10-13 and computes per-K-row attention weights
  for head 8 at Q@p=116.
* ``c4_release/docs/A3_7_*`` (commit 6bfc009a) — predecessor A3.7
  ALiBi pin landing.
* ``c4_release/docs/A3_6_DIM_602_ZEROING_ATTRIBUTION_2026_06_09.md``
  — A3.6 predecessor (disproved zeroing-writer hypothesis).
