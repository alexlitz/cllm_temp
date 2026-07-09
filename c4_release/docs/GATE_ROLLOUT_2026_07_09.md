# GATE ROLLOUT — `derive_gate` family-wide (task #452)

**Status:** LANDED (flag-gated, DEFAULT-OFF). Extends the proven L6 CONTROL
`derive_gate` mechanism (`docs/DERIVE_ACTSCALE_2026_07_09.md`) to two more gated
op families. One flag, `C4_DERIVE_GATE_SCALES=1` (the SAME flag as the CONTROL
pilot — it now also derives the rolled-out families). A RE-DERIVATION
(behaviour-correct, verdict-preserving; NOT byte-identical). Golden flag-OFF
byte-identical.

**Byte-identity gate:** flag-OFF `tools/_isa_golden_hash.py` == `e50521f3` (bare
golden) UNCHANGED after every edit — the calibration-table extension + the
`derive_and_gate_maybe` wiring are read ONLY when `C4_DERIVE_GATE_SCALES=1`.
Flag-ON the model builds with the SAME 42149 FFN units (no dead units, width
locked) and a re-derived state_dict (`84fb792b`).

---

## 1. What rolled out

The CONTROL pilot wired `derive_gate` into the six L6 `pc_mux` branch gates. This
task extends it to the AND-shaped **AX-/SP-marker corrector gates** via a new
family-wide opt-in helper and a per-`position_class` calibration axis.

| piece | file | what |
|---|---|---|
| per-class scale axis | `verification/activation_scales.py` | `_CANONICAL_CLASS_SCALES`: opcode one-hots read `5.2` at `mark==AX` / `mark==SP` but `1.0` under `"*"`. `scale()` / `canonical()` consult it. |
| family-wide opt-in | `ops/shared.derive_and_gate_maybe` | the generalization of `l6_ops._maybe_derive_pc_mux_spec` to ANY `multi_way_and_rule` gate: returns the derived `(conditions, threshold)` when the flag is ON, the hand values OFF. |
| family #1 (16 gates) | `ops/l16_ops` `l16_jmp_ax_preserve_lo_{k}` | JMP-preserves-AX corrector, fires at `mark==AX`. |
| family #2 (2 gates) | `ops/l6_ops` `l6_jsr_sp_fixup_{lo,hi}` | JSR SP-byte fixup, fires at `mark==SP`. |
| calibration | `tools/calibrate_activation_scales.py` | measures the full opcode set at `mark==AX` / `mark==SP` (confirms the `5.2` datum). |
| unit tests | `tests/test_derive_gate_scales.py` | +4 tests (12 total), GPU-free. |

## 2. The decisive datum: the per-`position_class` opcode scale

The rolled-out gates hand-tune their opcode discriminator to weight **`0.2`**:

```
l16 jmp_ax_preserve:  (OP_JMP, 0.2) (MARK_AX, 1.0) ...blockers...  threshold=1.5
l6  jsr_sp_fixup:      (OP_JSR, 0.2) (MARK_SP, 1.0) (HAS_SE, -1.0) ...  threshold=1.5
```

`0.2 == 1/5.2 == 1/activation_scale(OP_*, "mark==AX")` — the SAME reciprocal the
BZ gate proved, now at the AX/SP marker rows. **MEASURED** (probe over the func /
var / loop corpus, mode of the strong-active magnitudes at the AX-marker rows):

```
OP_LEA = 5.23   OP_ENT = 5.23   (mode over 100+ AX rows)   FETCH_LO = 1.0
```

So `derive_gate` at `position_class="mark==AX"` sets `OP_JMP -> 1/5.2 = 0.192`
(== the hand `0.2`), `MARK_AX -> 1.0`, `threshold -> (n_norm - 0.5) = 1.5` (the
hand value). Positives + threshold REPRODUCED from the datum. The **blockers are
PRESERVED verbatim** (`preserve_blockers=True`) — see §3.1: they are
broadcast-defeat guards, not safety-factor vetoes.

### Why a per-`position_class` axis was REQUIRED

The same opcode one-hot has a DIFFERENT scale at different blocks — the
per-block-keying `docs/DERIVE_ACTSCALE_2026_07_09.md §6` flagged as the honest
limit. Decisively:

* `OP_JMP` reads **~0** at the L6 all-step JMP override block (a DEAD reserved
  band — the live JMP PC path is elsewhere). So under the `"*"` wildcard
  `OP_JMP` stays `1.0`, and the L6 pc_mux deadness guard keeps that band dead.
* `OP_JMP` reads **5.2** at the `mark==AX` marker row where `jmp_ax_preserve`
  fires (the opcode flag is broadcast in-step to every marker row).

A flat table could not represent both. `_CANONICAL_CLASS_SCALES` keys the
amplified opcode scale by class (`mark==AX` / `mark==SP`), so the L6 `"*"`
deadness and the L16 `mark==AX` liveness coexist on ONE datum. The L6 pc_mux
path queries `"*"` (unchanged); the rolled-out families pass their firing class.
(Unit tests `test_class_keyed_opcode_scale_ax_vs_star`,
`test_l6_all_step_jmp_stays_dead_under_star_class`.)

## 3. Verdict-preserving proof

* **Golden flag-OFF == `e50521f3`** after every edit (the datum + wiring are
  invisible when the flag is off).
* **Flag-ON build:** SAME 42149 FFN units, no dead units (width-locked, per
  `project_core_loc_reduction_reality`); state_dict `84fb792b` (re-derived).
* **`jmp_ax_preserve` (mark==AX):** derived `OP_JMP=0.192≈0.2`, `MARK_AX=1.0`,
  `thr=1.5` (hand-exact); every blocker vetoes past threshold even when all
  positives fire. (`test_l16_jmp_ax_preserve_reproduces_hand_at_ax_class`.)
* **`jsr_sp_fixup` (mark==SP)** — the two firing regimes match hand exactly:
  * bootstrap JSR (`HAS_SE=0`): hand `0.2*5.2 + 1.0 = 2.0 ≥ 1.5` FIRE; derived
    FIRE.
  * later JSR (`HAS_SE=1`): hand `2.0 - 1.0 = 1.0 < 1.5` VETO; derived
    `2.0 - 2.19 < 1.5` VETO. (`test_l6_jsr_sp_fixup_fire_and_veto_regimes_preserved`.)
* **CPU full_trace** (`cpu_full_trace --spec-k 0`), the authoritative verdict:
  `id 550` (func_identity, exercises JSR/ENT) — flag-OFF **PASS**, flag-ON
  **PASS** (identity(70)==70). Verdict preserved.
* **Unit suite:** 12/12 pass (GPU-free).

### 3.1 The blocker regime — PRESERVED, not re-derived (a landed regression fix)

`derive_gate`'s default re-derives blockers to the safety-factor veto
`-(pos_sum+1)` — correct for the L6 CONTROL gates, whose blockers ARE
safety-factor vetoes sized to the firing-row positive sum. But the l16/l6 JSR/JMP
corrector blockers are a DIFFERENT class: **broadcast-defeat guards**. The opcode
flag is broadcast in-step to EVERY marker row, so `OP_JSR`/`OP_JMP` reads ~5.2 at
non-firing rows too; the `-1e6` / `-10` blockers exist to veto the gate at those
broadcast rows (a much larger perturbation than the firing-row positive sum). The
first flag-ON run PROVED this: re-deriving the blockers to `-(pos_sum+1)≈-2.2`
was too weak at the broadcast rows and regressed `id550` (`div_step=6`,
`pc=138` vs oracle `pc=42`). The fix: `derive_and_gate_maybe(...,
preserve_blockers=True)` — derive the positives + threshold from the scale datum,
KEEP the hand blocker magnitudes. This is the correct scoping (the blockers are a
structural guard, not a magic constant the safety factor can replace); the two
families use it.

## 4. Magic constants eliminated

Per rolled-out gate the hand form carries: the opcode weight `0.2`, `MARK_*` unit
weights, the threshold, and the blocker magnitudes. `derive_gate` derives ALL of
these from (a) the calibrated per-`(dim, class)` scale + (b) the balanced-AND
structure + (c) the single safety factor `k`:

| gate quantity | hand | derived |
|---|---|---|
| opcode weight (`OP_JMP=0.2`, `OP_JSR=0.2`) | HAND | **`1/scale(OP, class)`** |
| marker weight (`MARK_AX=1`, `MARK_SP=1`) | HAND | **`1/scale=1`** |
| threshold (`1.5`) | HAND | **`n_norm - 0.5`** |
| blocker magnitudes (`-10`, `-2`, `-1`, `-1e6`) | HAND | HAND (broadcast-defeat guard — §3.1) |

Across the 2 families (18 gates: 16 `jmp_ax_preserve` + 2 `jsr_sp_fixup`) the
per-gate opcode-reciprocal `0.2` weights and the `1.5` threshold derive to `0`
hand-tuned per-op constants. The blockers are PRESERVED (§3.1: a structural
broadcast-defeat guard, not a magic constant — re-deriving them regressed
verdicts). The remaining numeric inputs are the MEASURED activation scales (a
datum, not a hand constant) and the preserved blocker guards. (For the L6 CONTROL
family, whose blockers ARE safety-factor vetoes, `derive_gate` also derives the
blockers — that path is unchanged.)

## 5. Honest limits — the `1/scale` regime is NARROW

The inventory (grep of all `multi_way_and_rule` gates + a `derive_gate`
reproduction check per gate) surfaced a decisive structural finding: **most gated
op families are NOT `1/scale` balanced-ANDs**, so `derive_gate` does NOT fit them.
Two distinct non-fit regimes:

1. **INVERSE-scale ALU gates (l8 LEA/ADJ/ENT, ~2000 units).** MEASURED at the AX
   rows: `ALU_LO` (operand-A) reads at scale **45** and is DOWN-weighted to
   `1.0`; `FETCH_LO` reads at scale `1.0` and is UP-weighted to `20.0` (the
   dominant discriminator). This is the OPPOSITE of `1/scale` — a tie-break
   magnitude design (weight ∝ scale⁻¹ would set `ALU_LO=0.022`, breaking the
   gate). The l8 ALU family is NOT a `derive_gate` target. (Its threshold
   arithmetic also folds a per-block operand amplification the flat deadness
   check can't see, so `derive_gate` mis-classifies the ADJ/ENT `thr=85` bands as
   dead — a second reason to leave them hand-authored.)

2. **MAGNITUDE-fire opcode gates (l10 `tail_cmp_*`, l16 `jsr_mem_addr0_f8`,
   l10 `tail_lea_*_amplify`).** These weight the opcode at `1.0` (not `0.2`) and
   let its amplified `5.2` residual dominate a high threshold (the comment on
   `l16_jsr_mem_addr0_f8` literally reads "OP_JSR is the in-step opcode broadcast
   ~= 11.4 ... single-handedly cleared the 500 threshold"). Here
   `hand_weight != 1/scale`, so `derive_gate` would REWEIGHT the opcode to `0.19`
   and BREAK the gate. NOT targets.

An initial static-text fit scan (before the opcode scale was corrected to `5.2`)
over-counted "49 clean fits"; re-running with the corrected `mark==AX` opcode
scale drops it to 34 gates, of which only ~13 carry an opcode discriminator, and
of THOSE only the ones with a LITERAL `0.2` opcode weight are true fits — the
`jmp_ax_preserve` + `jsr_sp_fixup` families landed here, plus a handful of
single scattered gates each with idiosyncratic soft-blockers (e.g. `HAS_SE=-1`
that must stay a soft penalty, not a hard veto) that need per-gate judgement.

**So the rollout is deliberately scoped to the two clean opcode-reciprocal
families.** The mechanism generalizes to any `1/scale` balanced-AND, but the
`1/scale` regime is a MINORITY of the gate surface — the ALU inverse-scale gates
and the magnitude-fire gates are structurally different and stay hand-authored.
This is the honest counterpart to the CONTROL doc's "generalizes to ANY gated op
family whose gate reads amplified discriminator dims": it does — but only the
ones that NORMALIZE (weight `1/scale`), not the ones that AMPLIFY (weight ∝
scale) or INVERT (down-weight the amplified operand).

### Other limits

* **NOT a core-LOC reduction.** The load-bearing unit COUNT is width-locked
  (42149 flag-ON == flag-OFF). This removes the positive-weight + threshold +
  blocker MAGIC CONSTANTS, not weight mass (`project_core_loc_reduction_reality`).
* **Per-class scale is baked, corpus-confirmed.** The `5.2` opcode scale at
  `mark==AX` is a canonical baked datum (MEASURED, mode over the corpus); the
  calibration JSON is an optional refresh. A future family firing at a marker
  class not yet in `_CANONICAL_CLASS_SCALES` (e.g. `mark==MEM`, `mark==STACK0`)
  needs that class measured + added before wiring.

---

## Reproduce

```
# flag-OFF golden unchanged:
CUDA_VISIBLE_DEVICES="" python tools/_isa_golden_hash.py                 # e50521f3

# flag-ON builds width-locked (no dead units), re-derived state_dict:
CUDA_VISIBLE_DEVICES="" C4_DERIVE_GATE_SCALES=1 python tools/_isa_golden_hash.py  # 84fb792b, 42149 units

# verdict preserved on a JSR/ENT program (both PASS):
CUDA_VISIBLE_DEVICES="" python tools/cpu_full_trace.py --ids 550 --spec-k 0 --workers 1
CUDA_VISIBLE_DEVICES="" C4_DERIVE_GATE_SCALES=1 python tools/cpu_full_trace.py --ids 550 --spec-k 0 --workers 1

# unit tests (GPU-free):
CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_derive_gate_scales.py -q     # 12 passed
```

## Feeds

Extends `docs/DERIVE_ACTSCALE_2026_07_09.md` (the CONTROL pilot) to two more
families and CLOSES its per-block-keying honest limit for the AX/SP marker gates
(the per-`position_class` scale axis). Records the decisive negative result:
`derive_gate`'s `1/scale` rule is a NARROW fit — the ALU inverse-scale gates and
the magnitude-fire opcode gates are structurally different regimes and stay
hand-authored.
