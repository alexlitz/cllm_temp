# Removal 3 findings — `test_add_16bit` residual override dependence

Date: 2026-06-05
Worktree: `.claude/worktrees/agent-a0f6dd6c937a7562e`
Branch: `speedup-cache-and-buckets` (off `1bdb721d`)
Parent plan: `RUNNER_OVERRIDE_REMOVAL_PLAN_2026_06_05.md`
Sister deep-dive: `REMOVAL_3_DEEP_DIVE_2026_06_05.md`
Brief: refute the "32-bit cascade" framing for test_add_16bit and
localize the actual residual root cause.

## TL;DR

`test_add_16bit` (`IMM 200, PSH, IMM 100, ADD, EXIT` → expects 300)
still needs the `b5cf7099` IMM AX runner override **not because of a
32-bit cascade bug**, but because the same L10
`tail_lea_local_ax_marker_byte0_e8` leak that Removal 1's
`MEM_ADDR_SRC` predicate fix targeted (`0b957e03`) **also fires on
`IMM 200`** — and the predicate fix only silences it for the subset
of IMM bytes where Removal 1 was probed (`IMM 0xFF` family). The leak
discriminator is **`FETCH_LO+8`** (low nibble == 8), which is high
for `0xC8 = 200` but zero for `0xFF`.

Removing the override drops the model's emitted AX for `IMM 200` to
`0xFFE8` (= 65512), which propagates as the pushed operand and
poisons the collapsed-step ADD recovery. **The 32-bit ADD path
itself is not broken** — the collapsed-step recovery would synthesize
300 correctly from the clean bytecode IMM values. The blocker is
upstream, at the IMM-step AX emit.

**No clean model-side fix is bounded this session.** The leak
class is the same one Removal 1 partially fixed; tightening the L10
predicate further or adding a `FETCH_LO+8` blocker risks regressing
legitimate LEA-byte evaluation. Override remains in place.
Documentation only.

## Token-emit trace (override ON, working case)

Captured via dispatch wrapping on the batched runner, CPU build.
Bytecode: `[IMM, 200, PSH, IMM, 100, ADD, EXIT]` packed.

```
[pre]  idx=0 op=1  (IMM 200)  neural_ax=65512   prev_ax=0     last_push=None
[post] idx=0 op=1  (IMM 200)  last_ax=200       last_pc=10    last_push=None
[pre]  idx=1 op=13 (PSH)      neural_ax=200     prev_ax=200   last_push=None
[post] idx=1 op=13 (PSH)      last_ax=200       last_pc=18    last_push=200
[pre]  idx=2 op=1  (IMM 100)  neural_ax=100     prev_ax=200   last_push=200
[post] idx=2 op=1  (IMM 100)  last_ax=300       last_pc=34    last_push=200
result: [('', 300)]
```

Key observations:

1. **`IMM 200` (0xC8) emits `neural_ax = 65512 = 0xFFE8`**. Byte 0 =
   `0xE8` (LEA-tail signature), byte 1 = `0xFF`. The runner override
   immediately rewrites this to 200.
2. **`IMM 100` (0x64) emits `neural_ax = 100`**, correctly. Same step
   pattern, just a different IMM byte → no leak.
3. The model collapses `IMM 100, ADD` into one step (`last_pc` jumps
   to idx 4 = EXIT). The existing collapsed-step recovery
   (`batched_pure_neural.py:2133-2156`) synthesizes
   `_compute_alu_legacy(ADD, last_pushed_value=200, bytecode_imm=100)
   = 300` and overrides AX. Test passes.

Without the IMM AX override, step 0's wrong `neural_ax = 0xFFE8`
poisons `prev_ax` going into step 1 (PSH), which sets
`last_pushed_value = 0xFFE8`. The collapsed-step recovery then
computes `0xFFE8 + 100 = 0xFFFE4C = 16770124`, test fails.

## Why Removal 1 (`MEM_ADDR_SRC` predicate) fixed `IMM 0xFF` but
   not `IMM 200`

The L10 `tail_lea_local_ax_marker_byte0_e8` rule
(`l10_ops.py:6561-6591`) currently has:

```python
conditions=(
    ("MARK_AX",       1.0),
    ("HAS_SE",        1.0),
    ("OP_LEA",        1.0),
    ("CMP+7",         1.0),    # L7 head 5 OP_LEA relay
    ("FETCH_LO+8",    2.0),    # low nibble == 8
    ("FETCH_HI+15",   0.2),    # high nibble == F
    ("MEM_ADDR_SRC",  5.0),    # Removal 1 fix (commit 0b957e03)
    ("IS_BYTE",      -10.0),
    ...
),
threshold=14.0,
```

Static positive sum upper bound on IMM dispatch rows (where MARK_AX,
HAS_SE, OP_LEA-leak, CMP+7-leak are all 1.0):

- `IMM 0xFF` (low nibble F, hi nibble F): `MARK_AX(1) + HAS_SE(1) +
  OP_LEA(1) + CMP+7(1) + FETCH_LO+8(0) + FETCH_HI+15(0.2) = 4.2`.
  **Below threshold 14 even without MEM_ADDR_SRC.** Rule cannot fire.
- `IMM 200 = 0xC8` (low nibble 8, hi nibble C): `MARK_AX(1) +
  HAS_SE(1) + OP_LEA(1) + CMP+7(1) + FETCH_LO+8(2.0) +
  FETCH_HI+15(0) = 6.0`. Still below 14 by structural math.

But empirically `IMM 200` emits 0xE8 byte 0 — so an unmodeled
positive contribution (~8) must be present. Per
`OP_LEA_LEAK_INVESTIGATION_2026_06_05.md`, the candidates are:

1. **L7 head 5 attention broadcast amplifying `OP_LEA` / `CMP+7`
   above 1.0** at the AX marker via softmax mass distribution. The
   c9e9dc28 attempt at `OP_IMM: -1e9` could not suppress this because
   the leak is positive evidence, not negative-evidence shortfall.
2. **`MEM_ADDR_SRC` itself leaking ≥ 1.0** on IMM dispatch rows via
   some same-step cross-position broadcast. If MEM_ADDR_SRC is
   actually live at MARK_AX on the IMM 0xC8 row, the predicate
   contributes 5.0 — INVERTING what the Removal 1 fix attempts. The
   predicate then ADDS to the firing score rather than gating it
   off, raising the IMM 200 score to 11.0 (still below 14, but
   closer).

Combined with attention-broadcast amplification, score crosses 14 on
IMM 200 but not IMM 0xFF. This is consistent with the observation
that **the predicate strengthens the rule for some IMM rows it was
supposed to gate**.

## Refuting the brief's hypotheses

Brief asked to identify which model layer breaks the high-byte
cascade:

- **L5 byte-decode failing AX byte 1**: Not the issue. The wrong
  byte is byte 0 (0xE8 instead of 0xC8). Byte 1 (0xFF) is residual
  noise from the same step, not from a missing byte-1 emit.
- **L7 operand_gather missing STACK0_BYTE1 vs STACK0_BYTE0**: Not
  the issue. PSH emits `neural_ax=200` correctly (the override is
  sticky in `s.last_ax`). The trace shows the collapsed-step
  recovery handles ADD via `last_pushed_value`, not STACK0_BYTE*.
- **Multibyte ADD wide_add_rules not propagating carry**: Not the
  issue. Per `REMOVAL_3_DEEP_DIVE_2026_06_05.md`, the wide_add_rules
  callers all use `width_bytes=1`; the actual multi-byte ADD goes
  through the imperative `AddSub5StageBlock`. And the trace shows the
  ADD step is COLLAPSED into the IMM 100 step — no ADD step is even
  emitted by the model. So no ADD wiring is on the failing path.

The root cause is **the L10 tail_lea leak on `IMM 200`**, the same
class Removal 1 partially addressed. The Removal 3 "32-bit cascade"
framing is structurally misleading; the test passes purely via
collapsed-step recovery once the IMM AX values are clean.

## Why test_xor_basic and test_add_carry_cascade DID recover under
   Removal 1

Per Removal 1 (`0b957e03`) commit message, removing the IMM override
after the predicate fix recovers `test_xor_basic` and
`test_add_carry_cascade`. Both use `IMM 0xFF` (low nibble F, hi
nibble F) — the L10 rule's static positive sum is 4.2, below
threshold 14 by 9.8 points. Even if attention broadcast adds ~5 from
OP_LEA/CMP+7, the rule still stays below threshold for these IMM
bytes. `test_add_16bit` uses `IMM 200 = 0xC8` (low nibble 8 → +1.8
extra from FETCH_LO+8 vs the 0xFF case) — that 1.8-point
differential is enough for the same attention-broadcast leak to push
the rule over threshold.

## Bounded model-fix candidates

None bounded this session. The leak class spans three coupled
surfaces:

### Option A — `FETCH_LO+8` blocker on the L10 tail_lea rule

Add `("FETCH_LO+8", -50.0)` to the conditions. Pro: directly
silences the IMM-0xC8 misfire. Con: real LEA steps may have
non-trivial FETCH_LO+8 at MARK_AX too (LEA's imm encodes the BP
offset; offsets with byte0 low nibble == 8 like LEA -8 are common
and load-bearing). Would need a per-byte sweep to verify the
blocker doesn't suppress every legitimate LEA byte 0 = 0xE8 case.

### Option B — Tighten L7 head 5 K-side gate

Per OP_LEA_LEAK_INVESTIGATION § Option B: add an
`OP_IMM*-L*100` term to L7 head 5's K projection so IMM-dispatch
AX positions score negatively in the K softmax, attenuating the
OP_LEA → CMP+7 broadcast at MARK_AX on IMM rows. This is a load-
bearing relay; needs verification all real LEA AX markers still get
CMP+7 = 1.

### Option C — Make L5 LEA decode mutually exclusive

Add explicit `-10/S` LEA-suppression writes on the other 33 OP_*
decoders. Largest delta (~1100 new writes), highest blast radius,
but cleanest architectural fix. Out of scope for a 1-compile session.

**Recommendation**: leave the runner override in place. The Removal
3 brief's hypothesis (32-bit cascade in ADD) is refuted by the
trace; the actual residual is a continuation of Removal 1's leak
class, not a new failure mode. A follow-up should sweep all 256 IMM
byte values to confirm which subset triggers the leak under the
current predicate fix.

## Files referenced

- `c4_release/neural_vm/batched_pure_neural.py:1985-2011` — the
  `b5cf7099` override under investigation.
- `c4_release/neural_vm/batched_pure_neural.py:2133-2156` —
  collapsed-step recovery that test_add_16bit relies on once IMM
  values are clean.
- `c4_release/neural_vm/unified_compiler/ops/l10_ops.py:6561-6591`
  — `tail_lea_local_ax_marker_byte0_e8`, the leak rule with the
  Removal 1 `MEM_ADDR_SRC` predicate.
- `c4_release/neural_vm/dim_registry_dynamic.py:327-332` —
  `FETCH_LO` / `FETCH_HI` 16-dim one-hot layouts (low/high nibble
  of immediate byte 0).
- `c4_release/tests/test_smoke.py:500-522` — test_add_16bit /
  test_add_carry_cascade bytecode.
- `c4_release/docs/OP_LEA_LEAK_INVESTIGATION_2026_06_05.md` —
  upstream analysis of the L7 head 5 / CMP+7 broadcast.
- `c4_release/docs/REMOVAL_3_DEEP_DIVE_2026_06_05.md` — sister
  doc; this finding refines its "model never emits REG_AX during ADD
  step" hypothesis (which describes Case A; the actual trace
  exercises Case B / collapsed step, so the bug is upstream at IMM
  rather than at ADD).
- `c4_release/docs/REMOVAL_1_IMM_OVERRIDE_2026_06_05.md` —
  Removal 1 doc; documents the partial predicate fix.

## Confidence

- **High**: the IMM 200 step emits `neural_ax = 65512 = 0xFFE8` with
  override ON (captured trace).
- **High**: the test passes via collapsed-step recovery, not via the
  ADD step's neural AX emit (last_pc jumps from idx 2 to idx 4 = EXIT
  after the IMM 100 step).
- **High**: removing the IMM override propagates the wrong AX through
  `last_pushed_value` and breaks collapsed-step recovery.
- **High**: the L10 `tail_lea_local_ax_marker_byte0_e8` rule's
  FETCH_LO+8 positive (+2.0) differentiates IMM 0xC8 from IMM 0xFF,
  matching the empirical Removal-1 recover-vs-regress split.
- **Medium-High**: the attention-broadcast leak documented in
  OP_LEA_LEAK_INVESTIGATION is the unmodeled ~8-point positive that
  lets the rule cross threshold 14 on IMM 0xC8 even with the
  predicate fix.
- **Low**: that any 1-compile fix lands this session — Option A
  risks LEA regressions; Options B/C are out of scope.

## Commit

`docs(removal-3): test_add_16bit residual investigation`. Doc only;
no code change. Override at `batched_pure_neural.py:1985-2011` and
predicate fix at `l10_ops.py:6561-6591` remain as-is.
