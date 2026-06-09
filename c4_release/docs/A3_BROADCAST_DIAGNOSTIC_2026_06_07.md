# Wave 1 A3 broadcast head diagnostic (2026-06-09)

Follow-up to commits `fd109b86` (L10 broadcast heads) + `6c5be295`
(L14 mem_generation read migration). The A3.2 commit's caveat:

> the deeper SI failure mode remains... likely follow-ups:
> verify L10 broadcast head actually populates STACK0_BYTE_VAL_h.

Memory smoke is still 1/6 (`test_si_li_zero` passes by accident — it
just needs the byte = 0 default that's already in the OUTPUT band).
This doc closes the verification step.

## TL;DR

The L10 broadcast heads' weights are baked correctly (W_o magnitude
48.0 at the slot-8/9/10 STACK0_BYTE_VAL_h_LO/HI output columns), and
the L14 mem_generation V-read weights for those dims are baked
correctly (W_v magnitude 16.0 at the consumer block). **The Q gate
never fires.** The new dim family stays at zero across every layer,
every position, every step.

**Bug:** the broadcast head's Q gate includes `AP(0, BD.OP_PSH, L)`
and a slot-33 complement gated on `OP_PSH`. At the Q row
(MARK_STACK0 + BYTE_INDEX_h), **OP_PSH is zero at every layer**.
OP_PSH only fires at the AX/instruction row, not at the STACK0 byte
rows of the autoregressively-emitted STACK0 frame. The Q dot product
is dominated by the negative constants, the softmax collapses to a
uniform mass over the K-key set, and the head's V contribution to
STACK0_BYTE_VAL_h sums to ~0 (V is also zero at non-MARK_AX rows).

**Diagnosis:** **broadcast side (producer) is broken** — but not at
the spec/weight level. The spec is wrong about WHERE the gating
signal lives.

**Recommendation:** keep the K-side `OP_PSH` complement at slot 33
(softmax1-aware), but drop the Q-side `AP(0, BD.OP_PSH, L)` and
the Q-side slot-33 `AP(33, BD.OP_PSH, 10000.0)` term. Rely on the
K-side to gate the head OP_PSH-exclusively: when OP_SI / OP_SC /
OP_JSR / OP_ENT is the active op, the K-side complement at slot 33
goes negative (per the existing `AP(33, BD.OP_SI/SC/JSR/ENT,
-10000.0)` terms on the Q side — those work because they suppress
at the K row, where the active-op marker dim is set, NOT at the
STACK0 Q row). Move the OP-exclusivity onto the K side or onto a
context-broadcast dim available at the STACK0 Q row (e.g. an
`L9_CURRENT_OP_PSH` band the model already populates).

## Probe values

`tools/probe_a3_broadcast_diagnostic.py` runs `test_si_li_roundtrip`
(`IMM 0x200, PSH, IMM 42, SI, IMM 0x200, LI, EXIT`) with forward
hooks on physical blocks 0-31 and reports STACK0_BYTE_VAL_h_LO/HI
residuals at the STACK0 byte rows.

Layout note: `_expand_wrapper_blocks` (phase 1300) splits HybridALU
+ post_op composites, so the logical L10 broadcast bake target lands
at physical block 12 in the final model; the logical L14
mem_generation consumer lands at physical block 27. The probe maps
`after_L10` -> physical block 12, `after_L14` -> physical block 27.

### Weight-level baseline (CPU, CSR off)

```
L12 (nh=12) WRITES STACK0_BYTE_VAL_1_LO mag=48.00   # producer baked
L27 (nh=8)  READS  STACK0_BYTE_VAL_1_LO mag=16.00   # consumer baked
```

Both ends are wired. The weights are not the problem.

### Residual at STACK0 byte rows (after L10 broadcast block, byte 1 row)

For PSH of IMM 0x200, the AX byte 1 = 0x02. Expected at
`STACK0+BYTE_INDEX_1`: `STACK0_BYTE_VAL_1_LO[2] > 0`,
`STACK0_BYTE_VAL_1_HI[0] > 0`. Observed (probe, after physical
block 12):

```
STACK0 marker  OP_PSH_at_marker  V1_LO       V1_HI       CL_LO   CL_HI
@  80          0.00              0/0.00     0/0.00      0/1.0   0/1.0
@ 114          0.00              0/0.00     0/0.00      0/1.0   0/1.0
@ 148          0.00              0/0.00     0/0.00      0/1.0   0/1.0
@ 183          0.00              0/0.00     0/0.00      0/1.0   0/1.0
@ 218          0.00              0/0.00     0/0.00      0/1.0   0/1.0
```

**Every STACK0 byte row, every step: V1/V2/V3 LO/HI are exactly 0.**
The broadcast does not land.

### Cross-layer dump at STACK0 + BYTE_INDEX_1 row (verifies persistence)

```
STACK0@114, BYTE_INDEX_1 row @ p=116:
  L9  (phys 11): V1_LO=0/0.00  V1_HI=0/0.00
  L10 (phys 12): V1_LO=0/0.00  V1_HI=0/0.00   <- producer block, EXPECTED >0
  L11 (phys 13): V1_LO=0/0.00  V1_HI=0/0.00
  L13 (phys 25): V1_LO=0/0.00  V1_HI=0/0.00
  L14 (phys 27): V1_LO=0/0.00  V1_HI=0/0.00   <- consumer block reads 0
```

### Root-cause probe: OP_PSH residual at STACK0 byte rows

```
STACK0@114, d=0..9 (covers the marker + 9 byte/scratch rows)
across physical blocks L0..L12 (covers everything up to and
including the broadcast):

OP_PSH residual is 0.00 at every (layer, row) combination.
```

OP_PSH is set on the AX-source row (the instruction-fetch row),
not on the STACK0 byte rows. The Q gate's
`AP(0, BD.OP_PSH, L)` term contributes 0 at the Q row, leaving
the gate dominated by `AP(0, BD.CONST, -L*2.0)` = -200 baseline.
The softmax dispatch from each Q row is ~uniform across all K
positions (no row is "preferred"), so the head's V contribution
is the unweighted average of V across the sequence — which is
~zero (V is zero at non-MARK_AX rows by design).

## Diagnosis

**Producer-side bug** in the Q gate of
`_layer10_psh_ax_broadcast_head_spec`. The gate needs to fire at
`MARK_STACK0 + BYTE_INDEX_h` rows during the OP_PSH step, but the
gate predicates "during OP_PSH step" by reading the OP_PSH dim at
the Q row — which is the wrong row to look at. OP_PSH lives on the
instruction-fetch row (where MARK_AX fires, which is the K side
here), not on the STACK0 byte rows.

The K side already has the correct OP_PSH semantics implicitly: K
only fires at the AX byte source row when MARK_AX is set, and that
row IS the OP_PSH-active row. The slot-33 K complement
(`AP(33, MARK_AX, M)`, `AP(33, byte_index_dim, M)`,
`AP(33, CONST, 100.0)`) is sound and lights up only on the intended
K row. The slot-33 Q baseline at -30000 + multi-positive
(MARK_STACK0=10000, byte_index_dim=10000, OP_PSH=10000) sums to
+90 IF OP_PSH=1.0 at the Q row, but in practice OP_PSH=0 there, so
the slot-33 Q is at -10000. The slot-33 K positive only sums to ~100,
which can't overcome -10000.

Net: the head can never produce a positive (Q,K) dot product at the
intended (STACK0+byte_h row, AX byte_h row) pair. Softmax goes
~uniform → V averages over the sequence → ~zero V contribution.

## Recommended fix

Two-line change to `_layer10_psh_ax_broadcast_head_spec`:

1. Drop the Q-side `AP(0, BD.OP_PSH, L)` term and adjust the
   negative CONST baseline accordingly (currently `-L * 2.0` =
   -200 with two positives MARK_STACK0+byte_index_dim summing to
   +200; drop one positive → CONST = -L * 1.0).
2. Drop the Q-side slot-33 `AP(33, BD.OP_PSH, 10000.0)` term and
   re-balance the slot-33 baseline (currently -30000 + 3 positives
   summing to +30000 needs to become -20000 + 2 positives summing
   to +20000).

The OP-exclusivity terms `AP(33, BD.OP_SI, -10000.0)` etc. should
move from the Q side to the K side (they currently suppress at the
Q row, which is meaningless since OP_SI is also 0 there). Better:
add a K-side `AP(33, BD.OP_PSH, M)` so the slot-33 K-side requires
OP_PSH at the MARK_AX K row, which IS the correct gating semantics.

Alternative (lower risk): introduce a new dim
`CURRENT_OP_PSH_STACK0_BROADCAST` populated by an L9 FFN at every
STACK0 byte row of the current PSH step. Q can then read this dim
at the STACK0 Q row. This avoids changing the gating algebra but
adds a new scaffolding op (similar effort to A3 itself).

The minimal path is option 1: remove the Q-side OP_PSH terms and
rely on K-side gating. **Not implemented in this diagnostic pass**
(per the brief — probe + doc only, no model changes).

## Smoke baseline

`pytest c4_release/tests/test_smoke.py`: 45 passed, 6 failed (no
regression vs. post-A3.2 commit `6c5be295`). Failing tests:
`TestSmokeAddress::test_lea_basic`, `TestSmokeMemory::*` (5/6 mem
tests still fail; `test_si_li_zero` accidentally passes).

## Files

* `c4_release/tools/probe_a3_broadcast_diagnostic.py` — the probe.
* `c4_release/neural_vm/unified_compiler/ops/l10_ops.py:1644-1770`
  — `_layer10_psh_ax_broadcast_head_spec` (the buggy Q gate).
* `c4_release/neural_vm/unified_compiler/ops/l14_ops.py:540-572`
  — `_layer14_mem_generation_head_specs` (the reader; works
  correctly given a non-zero broadcast dim).
* `c4_release/docs/L8_SP_GATHER_STACK0_AUDIT_2026_06_07.md` —
  pre-A3 audit that motivated the broadcast head design.
