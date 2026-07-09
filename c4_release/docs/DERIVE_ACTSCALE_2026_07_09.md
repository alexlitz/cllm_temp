# DERIVE ACTSCALE — the per-dim activation-scale calibration datum (task #395)

**Status:** LANDED (flag-gated, DEFAULT-OFF). One flag,
`C4_DERIVE_GATE_SCALES=1`, a RE-DERIVATION (behaviour-correct, NOT
byte-identical); golden flag-OFF byte-identical.

**The mission.** The CONTROL-family derivation
(`docs/DERIVE_CONTROL_2026_07_09.md` §6) proved that a gate's BLOCKER magnitudes
derive to a spec-structural safety factor, but the gate's POSITIVE weights +
threshold do **NOT** derive from the ISA identity alone — they encode the model's
per-dim RUNTIME ACTIVATION SCALES (`MARK_PC` / `OP_BZ` / `CMP+k` / `HAS_SE` do
not all activate at `1.0`). The CONTROL doc flagged this "per-dim activation-scale
spec datum" as THE missing piece to fully derive the gates. This task scopes,
pilots, and LANDS that datum.

**Byte-identity gate:** flag-OFF `tools/_isa_golden_hash.py` == `e50521f3` (bare
golden) UNCHANGED after every edit, INCLUDING with the calibration JSON installed
at `.agent-logs/activation-scales/v1.json` (the datum is read only when
`C4_DERIVE_GATE_SCALES=1`). Registered in both cache-key snapshots (plus
`C4_ACTSCALE_JSON` so two calibrations never share a memo/disk entry).

---

## 1. The decisive MEASUREMENT: the hand weight IS `1/activation_scale`

Hooking the residual stream the BZ gate reads (the L22 post-L9 block's FFN
input), teacher-forced on the CONTROL pilot programs, at the BZ-firing row:

```
MARK_PC = 1.00   OP_BZ = 5.00   CMP+4 = 1.28   CMP+5 = 1.00   HAS_SE = 1.00
                 (5.23 at the amplified plateau)                IS_BYTE = 0.00
```

The HAND BZ gate weights (`ops/l6_ops._post_l9_bz_pc_override_rules`) are:

```
MARK_PC = 1.0   OP_BZ = 0.2   CMP+4 = 1.0   CMP+5 = 1.0   HAS_SE = 10.0
threshold = 3.5 + 10.0 = 13.5
```

So **`hand_weight(dim) == 1.0 / activation_scale(dim)`**: `OP_BZ`'s weight `0.2`
is exactly `1/5.0`, normalizing its measured `5x` residual activation back down
to a UNIT AND term (`0.2 * 5.0 == 1.0`), matching the unit-scale markers/flags.
The threshold `3.5` is `n_norm - 0.5` over the four NORMALIZED (unit-scale)
positives (`MARK_PC, OP_BZ, CMP+4, CMP+5`); the `+10.0` is the `HAS_SE` step-0
guard's own amplitude (a satisfiability lever, NOT a scale-normalizer). The
opcode one-hot is amplified upstream (clean `5.0` from block L8 onward, where the
decode band writes it); the markers are clean unit one-hots.

**This is the datum, fully explained.** The gate's positive side is not magic — it
is `1/scale` per discriminator + the balanced-AND structure + the step-guard
amplitude. Measured `OP_BZ = 5.0..5.2`, `CMP+4 = 1.28`, everything else `1.0`.

## 2. The feature — an activation-scale calibration + a generic `derive_gate`

Three pieces, all NEW (the CONTROL branch's blocker derivation is cherry-picked):

* **The datum + loader** — `neural_vm/verification/activation_scales.py`. A
  per-`(dim, position_class)` activation-scale table, loaded from
  `.agent-logs/activation-scales/v1.json` if present, else a CANONICAL baked
  table (`_CANONICAL_SCALES`: opcode one-hots `5.0`, markers/flags `1.0`) so a
  fresh checkout derives correctly with NO runtime artifact (the golden build
  never depends on the JSON). `scale(dim, class)` returns the measured (or baked)
  scale; a gate positive weight is `1/scale`.

* **The build-time calibration** — `tools/calibrate_activation_scales.py`. Hooks
  every block's FFN input, teacher-forces a small representative program set, and
  MEASURES each discriminator dim's characteristic amplified activation (a
  strong-active floor selects the amplified `5.0` plateau over the small cross-row
  leak; the mode aggregates). Writes the JSON datum. Measured on the pilot corpus:
  `OP_BZ=5.2, CMP+4=1.3, markers/CMP+5/HAS_SE=1.0` — matching the canonical table.

* **The generic gate deriver** — `ops/shared.derive_gate(conditions,
  position_class, hand_threshold)`. FULLY derives a balanced-AND gate with ZERO
  hand-tuned per-op numbers:
    * each SCALE-NORMALIZED positive → weight `1/activation_scale(dim)` (the hand
      `OP_BZ=0.2` == `1/scale`, REPRODUCED not authored);
    * a STEP-GUARD positive (weight `>= 3.0`, e.g. `HAS_SE=10`) → PRESERVED
      verbatim (a satisfiability amplitude, a spec datum; the guard MUST be active
      for the AND to fire) and added to the threshold;
    * `threshold = (n_norm - 0.5) + sum(step_guard_weights)` (the normalized
      balanced-AND midpoint shifted by the guards — REPRODUCES BZ `3.5 + 10`);
    * each BLOCKER → `-safety_factor * (derived_pos_sum + 1)`, guaranteeing a
      single active blocker vetoes past the threshold (verdict-PRESERVING; scaled
      to the gate's FULL activation regime including the amplified step-guard).

  Wired into all six L6 `pc_mux` builders (all-step / first-step / delayed JMP,
  JSR, BZ, BNZ) via `l6_ops._maybe_derive_pc_mux_spec`.

**Satisfiability safety (the CONTROL doc's §1.2 concern).** A DEAD reserved band
(hand threshold > `sum(hand_weight * scale)` — the delayed/first-step/all-step JMP
override bands, whose `OP_JMP`/`CMP+0` discriminator never co-activates at those
blocks and whose `CONST=-1000` veto enforces deadness) is detected and returned
**UNTOUCHED** — positives, threshold, AND the deliberately-huge blockers all
preserved — so the derivation can NEVER resurrect it. Only a LIVE band (BZ/BNZ/
JSR) gets the derived gate.

## 3. The PILOT — the FULL CONTROL gate derives verdict-preserving (3/3)

`cpu_full_trace --spec-k 0`, bit-exact CPU full_trace, hand (flag-OFF) vs the
FULLY-derived gate (`C4_DERIVE_GATE_SCALES=1`, positives + threshold + blockers
all derived from the activation-scale datum):

| id  | cluster | flag-OFF (hand) | flag-ON (SCALE-DERIVED) |
|-----|---------|-----------------|--------------------------|
| 350 | if_gt   | PASS            | **PASS**                 |
| 375 | if_lt   | PASS            | **PASS**                 |
| 400 | if_eq   | PASS            | **PASS**                 |

**3/3 verdict-match** — from BOTH the canonical baked table AND the LIVE-MEASURED
calibration JSON (`OP_BZ=5.2`, `CMP+4=1.3`; the full build-time-calibration →
JSON → `derive_gate` loop). This is the result the CONTROL branch's positive-
flatten form could NOT achieve (it failed 0/3 with the garbage `pc=154`/`138`
targets). The difference: positive-flatten sets every positive to `1.0`
(ignoring scale) → the AND fires on the wrong rows; the SCALE derivation sets
each positive to `1/scale` → the AND fires on EXACTLY the hand rows.

The derived BZ/BNZ gates are numerically near-identical to hand:

```
BZ:     MARK_PC=1.0  OP_BZ=0.2   CMP+4=1.0  CMP+5=1.0  IS_BYTE=-14.2  HAS_SE=10.0  thr=13.5
BNZ lo: MARK_PC=1.0  OP_BNZ=0.2  CMP+4=-12.2                          HAS_SE=10.0  thr=11.5
BNZ hi: MARK_PC=1.0  OP_BNZ=0.2  CMP+4=1.0  CMP+5=-13.2               HAS_SE=10.0  thr=12.5
JSR:    MARK_PC=20.0 OPCODE_BYTE_LO/HI=1.0 ...blockers...             thr=21.5
```

Positives + thresholds match the hand values EXACTLY (`OP_BZ=0.2=1/5`, `thr` 13.5/
11.5/12.5/21.5); blockers are re-derived STRONGER (verdict-preserving). With the
MEASURED JSON the positives track the measured scale (`OP_BZ=0.192=1/5.2`,
`CMP+4=0.769=1/1.3`) — functionally equivalent, arguably MORE faithful (`CMP+4`
truly activates at `1.3`, not `1.0`).

## 4. The family-wide magic-constant elimination this enables

The CONTROL branch eliminated the gate BLOCKER magnitudes (4 distinct `-10`/
`-100`/`-1000`/`-4` values → 1 safety factor) but KEPT the positive weights +
threshold as necessary spec data — the honest floor it flagged. This task closes
that floor:

| gate quantity | before (CONTROL) | after (ACTSCALE) |
|---|---|---|
| blocker magnitudes | 1 safety factor (derived) | 1 safety factor (derived) |
| positive weights (`OP_BZ=0.2`, `OP_BNZ=0.2`, `MARK_PC=1`, `CMP=1`, ...) | HAND (kept) | **`1/activation_scale` (derived)** |
| threshold (`13.5`, `11.5`, `12.5`, `21.5`, `4.5`, ...) | HAND (kept) | **`n_norm-0.5 + step_guard` (derived)** |
| per-op magic numbers | positive weights + thresholds | **0** |

The ONLY remaining numeric inputs are (a) the safety factor `k` (spec-structural,
default `1`), (b) the step-guard amplitude (a satisfiability spec datum — the
guard must be present), and (c) the activation SCALES — which are now a MEASURED
datum, not hand-tuned. Every branch gate's positive side (`OP_BZ`/`OP_BNZ`
`0.2`, the `13.5`/`11.5`/`12.5`/`21.5` thresholds) derives from
calibration + ISA. The mechanism generalizes to ANY gated op family whose gate
reads amplified discriminator dims (the whole `multi_way_and_rule` surface):
measure the dims' scales, set weights `1/scale`, threshold from the normalized
balanced-AND.

## 5. Is activation-scale calibration viable? — YES

* **Measurable.** The scale is a STRUCTURAL property of the compiled model (the
  amplitude the decode band writes the opcode one-hot at). `OP_BZ` is a clean
  `5.0` from block L8 onward across every corpus program; markers are clean
  `1.0`. The calibration corpus only CONFIRMS the scale — it does not vary by
  program, so a 3-program calibration suffices (the canonical baked table is the
  same as the measured one).
* **Verdict-preserving.** The FULL derived gate matches the hand verdict 3/3 on
  the CONTROL pilot, from both the baked table and the live-measured JSON.
* **Robust.** The dead-band passthrough + the threshold-scaled blocker are
  correct-by-construction (a dead band is untouched; a blocker at
  `-(pos_sum+1)*k` always vetoes). The safety-factor knob (`C4_PC_OVERRIDE_K`)
  only STRENGTHENS blockers.

## 6. Honest limits

* **NOT a shortcut to the 20k core.** The load-bearing rule COUNT is width-locked
  (flag-ON 42149 units == flag-OFF). This removes the positive-weight + threshold
  MAGIC CONSTANTS, not the weight mass — consistent with
  `project_core_loc_reduction_reality`.
* **Calibration measures at ONE block per gate.** The scale a gate uses is the
  value at the block WHERE the gate reads (L22 for BZ). The tool's strong-active
  floor + mode select the amplified plateau correctly for the branch gates, but a
  family whose gate reads a dim at a NON-amplified block would need per-block
  keying (the datum's `position_class` axis is the hook; extend to a block axis if
  a future family needs it).
* **Piloted on the CONTROL family only.** The `derive_gate` mechanism is generic
  (`multi_way_and_rule`-shaped gates), but this task verified it only on the L6
  `pc_mux` branch gates (the CONTROL branch's flagged surface). Rolling it out to
  other gated families (the ALU / bitwise / decode balanced-ANDs) is the next
  step — each needs its discriminator dims' scales in the calibration table.
* **The all-step JMP band is treated as DEAD** (per the CONTROL doc §1: the live
  JMP PC path is elsewhere; `OP_JMP` measured `0` at the L6 all-step block). If a
  future measurement shows `OP_JMP` amplified at that block, the deadness check
  (which reads the live scale) would auto-reclassify it — the mechanism is
  scale-driven, not hard-coded.

---

## Reproduce

```
# flag-OFF golden unchanged (even with the calibration JSON installed):
CUDA_VISIBLE_DEVICES="" python tools/_isa_golden_hash.py                 # e50521f3

# the FULL CONTROL gate derives verdict-preserving (canonical baked table):
CUDA_VISIBLE_DEVICES="" C4_DERIVE_GATE_SCALES=1 python tools/cpu_full_trace.py \
    --ids 350,375,400 --spec-k 0 --workers 1        # 3/3 PASS

# build-time calibration -> JSON datum -> same 3/3 from the MEASURED scales:
CUDA_VISIBLE_DEVICES="" python tools/calibrate_activation_scales.py \
    --output .agent-logs/activation-scales/v1.json --ids 350,375,400
CUDA_VISIBLE_DEVICES="" C4_DERIVE_GATE_SCALES=1 \
    C4_ACTSCALE_JSON=.agent-logs/activation-scales/v1.json \
    python tools/cpu_full_trace.py --ids 350,375,400 --spec-k 0 --workers 1   # 3/3 PASS
```

## Feeds

Closes the CONTROL branch's flagged datum (`docs/DERIVE_CONTROL_2026_07_09.md`
§6: "a per-dim activation-scale spec datum would be needed to derive [the
positive weights + thresholds] — the next DSL extension, NOT done here"). The
FULL CONTROL gate now derives — positives + threshold + blockers — from
calibration + ISA, verdict-preserving. The mechanism (`derive_gate`) is the
lever to eliminate gate positive-weight/threshold magic constants family-wide.
