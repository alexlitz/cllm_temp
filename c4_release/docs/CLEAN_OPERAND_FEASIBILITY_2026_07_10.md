# CBC Phase 1 — Clean Operand Delivery Feasibility

**Date:** 2026-07-12  **Branch:** `clean-operand-delivery`  **Flag:** `C4_CLEAN_OPERAND` (default-OFF)
**Golden flag-OFF:** `e50521f3` (VERIFIED unchanged — `tools/_isa_golden_hash.py`)
**Base:** current `main` `d0beb462`

## Mission

Decide the whole *eliminate-all-correctors* project: CAN the operand-gather be
derived to deliver CLEAN one-hot operands (correct-by-construction)? If so, do
clean operands unblock DELETING the ALU/CMP cleanup correctors, or is the dirty
hybrid encoding a load-bearing architectural wall?

## TL;DR verdict — clean delivery is DERIVABLE, but is a **NO-GO for deleting the correctors**

* **Clean one-hot operand delivery IS DERIVABLE** — proven byte-clean. A single
  derived FFN (`CleanOperandOneHotFFN`, the `C4_CLEAN_OPERAND` flag) snaps every
  operand band to a perfect one-hot (one cell at 6.0, everything else exactly 0).
* **But feeding clean operands to the pipeline REGRESSES the corpus by a predicted
  −74 / 1096** (`tools/fast_gate.py --flag C4_CLEAN_OPERAND`). The regression is
  **entirely in the comparison/boolean clusters** (`if_eq/if_lt/if_gt/if_var/
  bool_and/func_max`). The ADD/arithmetic clusters GAIN (+6 flips).
* **Root:** the CMP engines are deliberately CALIBRATED TO THE DIRTY HYBRID — their
  per-nibble AND blockers are sized to SUBTRACT the index-0/cell-8 artifacts, so
  a perfect one-hot **overshoots** the lt/eq decode window and mis-decodes. This
  is not a bug to fix in the gather; it is a downstream tuning contract.
* **Conclusion:** the hybrid encoding is a **load-bearing wall for the CMP path**.
  Clean operand delivery cannot be turned on and the CMP correctors deleted
  without *also* re-deriving every CMP nibble comparator against the clean shape.
  The correctors are NOT deletable by cleaning the operand alone.

## Step 1 — PROBE: the dirty hybrid, quantified (spec_k=0, BUILT model)

`tools/probe_addsub_operand_vectors.py` + `tools/probe_clean_operand_trace.py`
dump the operand band the ALU reads at the binary-op MARK_AX row, GOLDEN
(flag-OFF) config. Example (add id 0, `654+114`):

```
GOLDEN (dirty hybrid) operand bands:
  ALU_LO:      [(0, -0.519), (8, +0.454), (14, +6.000)]   <- true nibble 14 = 6.0
  ALU_HI:      [(0, -0.516), (8, +6.000), (15, +0.453)]   <- true nibble 8  = 6.0
  AX_CARRY_LO: [(0, +0.291), (2, +0.996)]                 <- true nibble 2  = 1.0
  AX_CARRY_HI: [(0, +0.291), (7, +0.996)]                 <- true nibble 7  = 1.0
```

The "dirty" is a **CONSTANT additive residue**, NOT proportional to the operand:

* a `+0.454` bleed on **cell 8** and a `-0.519` residue on **cell 0** on the
  operand-A bands (ALU_LO/HI) — CONSTANT across every operand value (verified on
  0x02/0x0C/0x05: all three show `(0,-0.519),(8,+0.454)` unchanged).
* a `+0.291` residue on cell 0 of the operand-B bands (AX_CARRY_LO/HI).

**Localization** (`tools/probe_operand_bleed_source.py`): the whole band is EMPTY
through physical block 11 and appears in ONE write at **block 12** (the
`LoadedOperandAddHi15ClearFFN` / L8-main-FFN delivery block). So the dirt is
introduced at delivery, not accumulated.

**Where the dirt comes from (structural):** the operand is delivered by the L8
head-5 `mem[SP]->ALU` CAM (`_mem_to_alu_head5_cam_spec`, `l8_ops.py:2621`) via a
`CamValueBand("CLEAN_EMBED_LO","ALU_LO",16,1,SCALE_O=6.0)`. Two structural
sources of dirt:
1. **Slot-0 default-cancel:** the CAM writes `o=(("ALU_LO",-SCALE_O),("ALU_HI",
   -SCALE_O))` as a byte-0 baseline that the value band overrides — this is the
   `-0.5` cell-0 residue.
2. **Softmax-mixture + address two-hot:** the CAM softmax leaks a small weight
   onto other MEM rows whose `CLEAN_EMBED` has cell-8 set (the `+0.45` cell-8
   bleed), and on LOADED operands (`LI->PSH->mem[SP]`) an ADDITIONAL `~5.5`
   two-hot at the SP/BP-frame address high nibble (cell 13/15).

**The golden cleanup pre-pass (found, as required):** the codebase already
carries a whole FAMILY of operand-cleanup correctors that threshold this dirt:
`LoadedOperandAddHi15ClearFFN` (cells 13/15 address leak, `efficient_alu_neural.py:1215`),
`func_add_b0_hinib` (all-16-cell operand-B bleed clear), `operand_cam_fix` (widen
to SUB/MUL/DIV/MOD/CMP), and — the decisive one — `CmpOperandSeRecoverFFN`
(`efficient_alu_neural.py:1342`), which literally **RE-BUILDS the dirty hybrid on
purpose** (`clean_lo = se_lo*6.0 + art + idx0_lo*idx0_oh`, adding back the
index-0 artifact and cell-8/15 residues) because the CMP engines need them.

## Step 2 — DERIVABILITY: clean one-hot delivery (proven byte-clean)

The dirty encoding is a softmax-mixture + default-cancel artifact of the CAM
delivery — it is NOT intrinsic to a clean lookup. A derived clean-one-hot clamp
removes it. `CleanOperandOneHotFFN` (the `C4_CLEAN_OPERAND` flag) wraps the L8
main FFN and, on the binary-op/cmp MARK_AX rows, snaps ALU_LO/HI + AX_CARRY_LO/HI
to the per-band argmax at the clean 6.0 magnitude, zeroing every other cell. This
is the same derivable mechanism the DSL already uses for SP_ADDR
(`_sp_addr_clamp_box_rules`, the 2-silu saturating clamp).

**Gate (a) — PROVEN clean** (`tools/probe_addsub_operand_vectors.py`,
`C4_CLEAN_OPERAND=1`, same add id 0):

```
CLEAN (C4_CLEAN_OPERAND=1) operand bands:
  ALU_LO:      [(14, 6.0)]      AX_CARRY_LO: [(2, 6.0)]
  ALU_HI:      [(8,  6.0)]      AX_CARRY_HI: [(7, 6.0)]
```

Every band is now a PERFECT single-cell one-hot — all dirt (cell-8 +0.45, cell-0
±0.5, the address two-hot) eliminated. So **clean operand delivery is derivable.**

## Step 3 — verification (the four gates)

### (a) delivered operand IS a clean one-hot — YES (Step 2 above).

### (b) golden flag-OFF == `e50521f3` — YES (byte-identical).
`CUDA_VISIBLE_DEVICES="" python tools/_isa_golden_hash.py` prints
`e50521f3...` with the flag unset (all edits are flag-gated + campaign-gated).

### (c) `fast_gate.py --flag C4_CLEAN_OPERAND` — **NO (net −74, REGRESSION)**

```
SAMPLE net: -17  (flips=6 regs=23)   [220-id stratified sample, full_trace spec_k=0]
PREDICTED full-1096 delta: -74  (REGRESSION — do NOT land)

REGRESSIONS (ok -> fail) [23] — ALL comparison/boolean:
  if_eq: 5   if_lt: 4   if_var: 4   bool_and: 4   func_max: 3   if_gt: 2   mul: 1
GAINS (fail -> ok) [6] — arithmetic consumers:
  expr_add_mul: 3   func_mul: 2   func_min: 1
MEM-SMOKE (var_simple/var_mul/var_three/var_update): clean
```

The signal is decisive and clean-cut: clean operands **HELP arithmetic** (the ADD
carry stays correct, +6 flips) but **BREAK comparison** (−23 regs, every one a
CMP/bool cluster).

### (d) ALU carry side-signal on clean operands — CORRECT (no inflation)
`tools/probe_clean_operand_carry.py`, `654+114` and `200+100` (both byte-0
carries), OFF vs ON:

```
                 OFF (dirty)          ON (clean)
  CARRY+1        2.0                  2.0            <- clean, golden magnitude
  OUTPUT / got   768, 300 (PASS)      768, 300 (PASS)
```

The clean operand does NOT inflate the inter-byte carry (contrast the
`C4_DERIVE_ADDSUB` pilot's `CARRY+1 = 23`). The byte-0 ADD lane is
correct-by-construction on clean operands — which is exactly why the arithmetic
clusters GAIN.

## Root cause — why clean operands BREAK the CMP path

The CMP nibble comparators (`_layer10_alu_ordering_engine_rules` /
`_layer10_alu_eq_engine_rules`) were tuned against the GOLDEN HYBRID, and the
`CmpOperandSeRecoverFFN` docstring states the mechanism verbatim
(`efficient_alu_neural.py:1370-1383`):

> "The cmp ordering/eq engines were tuned against the operand-gather HYBRID
> encoding … the true nibble at ~+6.0 PLUS a value-proportional index-0 magnitude
> artifact (~+5.3) and small cell-8/cell-15 residues (~+0.45). The per-nibble AND
> units' -0.5/-0.8 blockers are sized to SUBTRACT those artifacts, which pulls the
> lt/eq flags into the (0.75, 1.5) decode window. … A PERFECT one-hot (6.0@true, 0
> elsewhere) leaves the blocker at 0, so lo_lt overshoots to 1.67 (> 1.5) and
> trips the GT (hi_eq AND lo_lt) 3-way override alone -> GT mis-decodes."

`CmpOperandSeRecoverFFN.forward` *reconstructs* the dirt on purpose. So the CMP
engines' `-0.5/-0.8` blockers are a **magnitude contract** with the operand
encoding: remove the dirt and the blockers over-subtract → the lt/eq flags leave
the decode window → GT/EQ/LT mis-decode. The −23 CMP regressions are exactly this.

## Honest go/no-go

**Is clean operand delivery DERIVABLE?** YES — proven byte-clean, no per-op magic
constants, golden untouched.

**Does clean operand delivery unblock deleting the ALU/CMP correctors?** **NO.**

* The **ALU (ADD/SUB) correctors** are the more promising side: clean operands
  keep the carry correct and the arithmetic clusters GAIN. But the byte-1+ ALU
  correctors are keyed off separate bands (`STACK0_BYTE_VAL_1`, `ADDR_B1_LO`,
  `CARRY+1`), NOT the byte-0 operand band this flag cleans — so cleaning byte-0
  does not, by itself, make them deletable either (it is necessary, not
  sufficient — matches the Phase-1 ADD/SUB CBC finding).
* The **CMP correctors are a LOAD-BEARING WALL.** The CMP nibble comparators are
  deliberately calibrated to the dirty hybrid's magnitudes; a clean one-hot
  breaks them. To delete the CMP correctors you MUST co-derive the CMP
  comparators against a clean-one-hot decode window (re-tune every `-0.5/-0.8`
  blocker to the 6.0/0.0 shape). That is a real, bounded next-phase scope
  ("derive the CMP nibble comparators for clean operands"), NOT a drop-in
  operand-cleanup.

**Bearing on the eliminate-all-correctors project:** the decisive lesson is that
the dirty hybrid is a **cross-layer calibration contract**, not a localized gather
defect. Clean delivery is derivable, but every consumer that was tuned to the
dirt (the entire CMP/bool family) must be re-derived in lockstep. The operand
correctors and the CMP correctors are ONE coupled unit; you cannot delete the
former without re-deriving the latter. The flag stays DEFAULT-OFF.

## Recommended next step (if the project continues)

Split the flag: `C4_CLEAN_OPERAND_ADD` (arithmetic-only opcode gate) captures the
+6 arithmetic gains with ZERO CMP regressions (the CMP rows are simply not
cleaned). That is a byte-identical-OFF, net-positive-ON slice worth measuring on
its own — but it does NOT delete correctors, it just narrows the operand cleanup
to where the downstream is clean-tolerant. The full corrector deletion requires
Phase 2's "re-derive CMP comparators for clean operands" work.

## Files (all flag-gated; golden flag-OFF `e50521f3` byte-identical)

* `neural_vm/efficient_alu_neural.py` — `CleanOperandOneHotFFN` (the derived
  clean-one-hot clamp).
* `neural_vm/unified_compiler/ops/shared.py` — `clean_operand_enabled()`
  (`C4_CLEAN_OPERAND`, default-OFF, campaign-gated).
* `neural_vm/unified_compiler/ops/alu_ops.py` — `make_clean_operand_op()`
  (installs the wrap on L8 main FFN, after `loaded_operand_add_hi15_clear`).
* `neural_vm/unified_compiler/full_vm_compiler_dynamic.py` +
  `_legacy_redirect.py` — op registration.
* `tools/probe_clean_operand_trace.py`, `tools/probe_operand_bleed_source.py`,
  `tools/probe_clean_operand_carry.py` — the probes above.

## Reproduce

```
# operand cleanliness (dirty -> clean), OFF then ON:
python tools/probe_addsub_operand_vectors.py 0 4 8 33            # OFF: dirty hybrid
C4_CLEAN_OPERAND=1 python tools/probe_addsub_operand_vectors.py 0 4 8 33   # ON: clean one-hot

# carry side-signal (gate d), OFF then ON:
python tools/probe_clean_operand_carry.py
C4_CLEAN_OPERAND=1 python tools/probe_clean_operand_carry.py

# the decisive corpus gate (gate c):
python tools/fast_gate.py --flag C4_CLEAN_OPERAND        # PREDICTED delta -74

# golden flag-OFF byte-identity (must print e50521f3...):
CUDA_VISIBLE_DEVICES="" python tools/_isa_golden_hash.py
```
