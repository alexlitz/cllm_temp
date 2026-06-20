# The positional-invariant mechanism — auto-shift anchors on STEP_TOKENS

2026-06-20. Phase-1 prototype. Mechanism: `neural_vm/unified_compiler/positional_invariant.py`.
Golden `4958b35b18108745` UNCHANGED (no-op at STEP_TOKENS=35).

Pairs with the audit (`docs/POSITIONAL_INVARIANT_AUDIT_2026_06_20.md`,
`tools/lint_positional_invariants.py`) which *finds* the shift-risk surface;
this doc is the *fix* — make the frame assumption declared and compiler-computed
so the STACK0 drop (`STEP_TOKENS 35 -> 30`) is a no-op for every anchor instead
of a per-cluster re-anchor (div/mod, operand-CAM, ROOT A/B — whack-a-mole).

## The make-or-break finding: there are exactly TWO anchor classes

Settled empirically by `tools/probe_posinv_frame.py` (reads the
`token_layout.py` parametrized positions in both frames) and by reading the
producers in `l0_ops` / `l1_ops` / `l2_ops`. A "positional" anchor depends on
the frame in one of two opposite ways:

### Class 1 — MARKER-RELATIVE distance-bank anchors (ALREADY frame-invariant)

`H<k>+MEM_I`, `L2H0+MEM_I`, and the L2-produced `MEM_VAL_B*` / `BYTE_INDEX_*`
flags that derive from them. The `+i` offset is a **marker-TYPE slot index**
into the fixed-width 7-slot threshold-head bank
(`PC=0 AX=1 SP=2 BP=3 MEM=4 SE=5`), NOT a token distance. The L0/L1/L2 threshold
*attention* computes "is a marker of type `i` within distance `t`" by ALiBi
**distance** attention — distance from the nearest marker, recomputed at
runtime. When the frame shrinks and the MEM marker moves `25 -> 20`, the
distance attention TRACKS it, so the flag still fires on the byte immediately
after the MEM marker in BOTH frames.

> **`MEM_VAL_B0` and friends need NO shift.** They are frame-invariant by
> construction. This is why the operand-CAM (`make_layer8_mem_to_alu_op`), once
> it was re-expressed onto `L2H0+MEM_I` / `MEM_VAL_B2`, is already correct in
> the 30-token frame — there is nothing left to auto-shift there.

The mechanism's only job for Class 1 is to **resolve the marker-slot index from
the frame** (`marker_bank_index("MEM") == 4` in both frames) so ops stop
hand-coding `MEM_I = 4` (scattered across `l0_ops`, `l1_ops`, `l2_ops`,
`l8_ops`, `compiler.py`) and the audit can recognise the ref as *declared
marker-relative* rather than an UNGUARDED bare offset.

### Class 2 — ABSOLUTE-SLOT byte flags (the genuinely-broken class)

`STACK0_BYTE0..3` ("STACK0 byte N position"), and any `d=k-from-<marker>`
anchor whose `k` rows between the marker and the target **include the dropped
STACK0 block**. The canonical example is the L1 `STACK0_BYTE0` flag: a
`d=6-from-BP` positional flag (`L1H4[BP]` d<=6.5 AND `IS_BYTE` AND NOT `H1[BP]`
d>4.5). In the 35-token frame d=6-from-BP is STACK0 byte 0. In the 30-token
frame the STACK0 block is gone, so **MEM addr byte 0 lands at d=6-from-BP** —
the flag MISFIRES onto an unrelated row and corrupts every `STACK0_BYTE0`
consumer (L3/L10/L11/L14/L16).

This is the class that needs the auto-shift. The prior fix was a hand-coded
per-op env branch:

```python
_stack0_byte0_threshold = 1.0e9 if no_stack0_emit_enabled() else 1.5
```

— a make-unreachable threshold so the unit stays allocated (the 5-unit
byte-count guard) but never fires. Repeated by hand for div/mod, operand-CAM,
ROOT A/B.

## The mechanism (`positional_invariant.py`)

Three pure helpers, each computed from `Token.STEP_TOKENS` (the single
env-flag authority) — **no per-op `no_stack0_emit_enabled()` branch**:

| Helper | Class | Returns |
|--------|-------|---------|
| `marker_bank_index(name)` | 1 | the threshold-bank slot index for a marker; frame-invariant (proves `MEM_I=4` in both frames), replaces the literal `MEM_I = 4`. |
| `frame_byte_is_emitted(marker, k)` | 2 | `True` iff the `d=k-from-<marker>` byte is EMITTED in the active frame; `False` iff it lives in the dropped STACK0 block. |
| `invariant_threshold(live, suppressed, marker, k)` | 2 | `live` when the byte is emitted, else `suppressed`. The auto-shift primitive for an absolute-slot threshold rule. |

The annotation form the brief proposed (`positional_invariant="STEP_TOKENS=35"`
on `FFNRule` / `DeclarativeAttentionHeadSpec`) is realised here as a **call-site
declaration**: a rule that owns an absolute-slot anchor calls
`invariant_threshold(live, suppressed, marker, k)` instead of writing a bare
threshold. This is strictly cleaner than a struct field for this codebase
because the broken anchors are authored as **imperative weight writes / rule
thresholds inside `bake` closures and rule generators** (e.g. `attn.W_k[base+1,
BD.L2H0 + MEM_I]`, `threshold=...`), not as a declarative field the lowering
post-processes — so the auto-shift has to happen *where the integer is
produced*, which is exactly what a helper call does. (A struct field would still
require every imperative writer to consult it, i.e. the same call.) The compiler
"auto-shift" IS the helper resolving the integer for the active `STEP_TOKENS`.

### Why this is a no-op at STEP_TOKENS=35

`marker_bank_index` returns the same integers the literals encoded;
`invariant_threshold` returns exactly `live`. So the lowered weights are
byte-identical → golden `4958b35b` unchanged (verified, see below).

## The prototype (one anchor, both gates GREEN)

Re-expressed the L1 `STACK0_BYTE0` anchor (`ops/l1_ops.py
_threshold_ffn_rules`) — the canonical Class-2 anchor — through the mechanism:

```python
BP_I = marker_bank_index("BP")                       # Class-1: resolves to 3 in both frames
...
_stack0_byte0_threshold = invariant_threshold(       # Class-2: auto-shift
    live=1.5, suppressed=1.0e9, marker="BP", k=6,
)
```

deleting the `1.0e9 if no_stack0_emit_enabled() else 1.5` hand-branch and the
`no_stack0_emit_enabled` import. The diff is the import, `BP_I`, and the
threshold — nothing else.

### Gate 1 — byte-identical golden (35-tok): PASS

`CUDA_VISIBLE_DEVICES="" python tools/_isa_golden_hash.py`
→ `state_dict_sha256 = 4958b35b18108745…` **UNCHANGED**. The mechanism is a
no-op in the golden frame.

### Gate 2 — 30-tok auto-fix with ZERO hand-tuning: PASS

`tools/probe_posinv_l1_anchor.py` (builds the REAL `_threshold_ffn_rules` list
in both frames):

```
STACK0_BYTE0 rule threshold  35-tok = 1.5          # fires (byte-identical literal)
STACK0_BYTE0 rule threshold  30-tok = 1000000000.0 # AUTO-suppressed, no env branch
mechanism == old hand-fix in BOTH frames           # behavioural drop-in
```

The campaign model (`C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1`) compiles
cleanly through the prototype; because the threshold and `BP_I` resolve to the
identical values the deleted hand-branch produced (`1e9` / `3`), the 30-token
weights are byte-identical to the pre-mechanism hand-coded path — the mechanism
*reproduces the manual re-anchor automatically*.

Isolation proof of the mechanism itself: `tools/probe_posinv_mechanism.py`
(no model build) — all checks PASS.
Two-class frame proof: `tools/probe_posinv_frame.py`.

## Sweep plan (Phase 2) — applying the mechanism across the audit catalog

The audit ranks **2146 refs | 1993 UNGUARDED (813 with a distance offset)**.
The two-class finding makes most of these **already-correct or mechanical**:

### Step 0 — partition the catalog by class (mechanical, no GPU)

Extend `lint_positional_invariants.py` to tag each ref Class-1 vs Class-2:

* **Class-1 (marker-relative)** — any `H<k>+<idx>` / `L1H*+<idx>` / `L2H0+<idx>`
  distance-bank ref, and reads of `MEM_VAL_B*` / `BYTE_INDEX_*`. These are
  ALREADY frame-invariant. The sweep is a **pure refactor**: replace the
  literal `MEM_I = 4` / `PC_I, AX_I … = 0,1,2,3,4,5` with `marker_bank_index()`
  so the invariance is *declared* (and the audit reclassifies them out of
  UNGUARDED). **No behavioural change, byte-identical both frames.** This is the
  bulk of the 813 distance-offset refs. ~mechanical; one byte-identity gate per
  edited op.

* **Class-2 (absolute-slot)** — reads of `STACK0_BYTE0..3` and any
  `d=k-from-BP` anchor reaching into the dropped block (the
  `efficient_alu_neural` cummax, the L10 `stack0_byte_relay` /
  `stack0_persistence` / `bp_byte_passthrough` heads, the L14 borrow-cascade
  minuend rows). These need a real decision: **re-point** to the marker-relative
  equivalent (preferred — e.g. operand byte-1 from `STACK0_BYTE1` →
  `MEM_VAL_B2`, the route head-7 already took) **or auto-neutralize** via
  `invariant_threshold` when the consumer is genuinely dead under the campaign
  (the L1 `STACK0_BYTE0` prototype). This needs judgment per op.

### Step 1 — ranked migration order (from the audit's per-op table)

Do the Class-2 ops in descending UNGUARDED+DISTANCE_OFFSET order, each as
probe→re-express→gate:

1. `efficient_alu_neural` `STACK0_BYTE1` cummax (div/mod — already hand-fixed;
   convert to `marker_bank_index`/re-point to make it the second worked example).
2. `_layer14_mem_generation_head_specs` (+`_with_overrides`) — 82 refs, the
   biggest single surface.
3. `_layer15_memory_lookup_heads_0_3_specs_with_overrides` — 26.
4. `_layer7_memory_head_specs` / `_layer8_sp_gather_head_specs` — operand gather.
5. The L10 passthrough family (`bp_byte_passthrough`, `stack0_byte_relay`,
   `stack0_persistence`) — Class-2 STACK0 reads, re-point or neutralize.

### Per-anchor gate (every step)

1. `lint_positional_invariants.py --dim <dim>` — enumerate the op's refs.
2. Re-express: `marker_bank_index` (Class-1) or `invariant_threshold` /
   re-point (Class-2).
3. **Gate A**: `tools/_isa_golden_hash.py == 4958b35b` (35-tok no-op).
4. **Gate B**: `tools/probe_posinv_<op>.py` — anchor resolves correctly in the
   30-tok frame (threshold / row / marker-slot), same pattern as
   `probe_posinv_l1_anchor.py`.
5. **Gate C** (shared FFN/attn only): `tools/lint_cross_op_ffn.py` /
   `lint_cross_op_attention.py`.
6. **Gate D** (campaign verdict): `tools/flag_regression_gate.py --flag …` on the
   affected cluster (the only GPU-equivalent step; CPU `cpu_full_trace`).
7. Re-run the audit — the op moves UNGUARDED → declared-invariant.

### Estimated effort

* **Class-1 refactor**: ~80% of the 813 distance-offset refs; mechanical;
  batchable per-file with one byte-identity gate each. Low risk.
* **Class-2 re-express**: the ~6 op families above (~40–60 refs); each needs
  judgment (re-point vs neutralize) + a campaign verdict gate. This is the real
  Phase-2 work, but it is now a *bounded, gated* migration instead of an
  open-ended hunt — and the two roots the audit confirmed (div/mod,
  operand-CAM) are already done, so they become the regression fixtures.

## Greenlight

Both make-or-break gates are GREEN: byte-identical golden at 35-tok AND the
30-tok anchor auto-fixes with zero hand-tuning. The positional machinery is
**not** too tangled for a clean auto-shift — the critical finding is the
opposite: most of the catalog is the already-invariant Class 1 (a declare-only
refactor), and the genuinely-broken Class 2 is a small, bounded set the
`invariant_threshold` / re-point primitive handles. Phase 2 (the sweep) is
greenlit.
