# CORRECTOR REMOVAL ROADMAP — bespoke-fix inventory by LOC, deletability, free deletes + derive-root missions

**Date:** 2026-07-13  **Branch:** `corrector-removal-roadmap` (off `main` `38a97600`)
**Golden flag-OFF:** `e50521f3` (`tools/_isa_golden_hash.py`)
**Full-1096 baseline:** 593/1096 (`--criterion full_trace --spec-k 0 --max-steps-cap 40`)
**Scope:** the ops-core = `neural_vm/unified_compiler/ops/*.py` (66,298 LOC) +
the campaign FFN wrappers in `neural_vm/efficient_alu_neural.py` (3,143 LOC).

## MISSION FRAMING

The LOC-reduction priority is **REMOVING bespoke correctors**, not collapsing
enumerations (which are done / LOC-neutral, per
`project_core_loc_reduction_reality`). A *bespoke corrector* is a hand-authored
fix that patches ONE root/case downstream (tail output-byte guarantee banks,
operand-cleanup / SE-recover wrapper FFNs, AX-dump/emission patches, the
individually-named fix ops). It is NOT a derived family (bitwise / cmp-engine /
shift / memory-CAM / decode / wide-ALU), whose rules are load-bearing computed
lookups and whose LOC is authoring-surface not corrector-debt.

**The decisive lesson from the CLEAN_OPERAND CBC phase**
(`docs/CLEAN_OPERAND_FEASIBILITY_2026_07_10.md`): the dirty-hybrid operand
encoding is a **cross-layer calibration contract**, not a localised gather
defect. Clean operand delivery is derivable, but every consumer tuned to the
dirt (the entire CMP/bool family) must be re-derived *in lockstep*. So the
operand-cleanup correctors split cleanly into (A) the ones a landed derivation
already subsumes, (B) the ones whose deletion needs a specific root derived
first, and (C) the ones guarding a deep multi-layer wall.

---

## STEP 1 — RANKED BESPOKE-CORRECTOR INVENTORY (by LOC)

LOC measured by `def`→next-top-level-`def` span at golden `e50521f3`.

| # | Corrector (family) | File | LOC | Class | Root / case it corrects |
|--:|--------------------|------|----:|:-----:|-------------------------|
| 1 | **`_tail_bit32_result_correction_rules`** (+ `make_..._op` 162) | `l10_ops.py` | **4,599** (+162) | **C** | L25 tail: ~40 sub-families of "guarantee OUTPUT byte = value observed in a (lo,hi) source lane" — MEM-store addr bytes, PC-row bytes, ALU-result bytes, SP/BP frame bytes. Width-locked, load-bearing (appending ANY rule breaks byte-identity, `project_l10_tail_bank_width_sensitive`). |
| 2 | **L6 PC-override / SP-fixup family** (`_layer6_*_pc_override_rules` ×6, `_jsr_sp_fixup`, `_ent_after_jsr_sp_byte0_fixup` 281, `_psh_stack0_marker_override`, `_post_l9_bz/bnz_override`) | `l6_ops.py` | **827** (rules) +~130 (IR lowerers) | **B/C** | Per-opcode PC-mux + branch-target byte writes (JMP/JSR/BZ/BNZ) + ENT/JSR SP-byte fixups. DERIVATION_ROADMAP §5 maps these to one `pc_override(gate,target)` / `cancel_residual_rule` shape (CONTROL derive-root G9). |
| 3 | **L14 clear/guard family** (`_layer14_clear_output_corruption` 100, `_clear_l14_mem_generation_overbroad_sp_suppression` 323, `_temp_clear` 81, `_clear_addr_key_pollution` 48, `_clear_mem_marker_output` 52, `_guard_l14_output_units_on_step_boundary` 95, +IR) | `l14_ops.py` | **723** | **C** | L14 mem-generation over-broad-write clears: TEMP-band pollution, addr-key pollution, mem-marker OUTPUT corruption, step-boundary output guard. Guards the memory-CAM generation path. |
| 4 | **`MulOperandSeRecoverFFN`** | `efficient_alu_neural.py` | **218** | **B** | Rebuilds the dirty-hybrid operand-A on the MUL MARK_AX row from `SE_ALU` so the L11 mul-partial lookup decodes at the golden magnitude. Deleting needs the mul lookup re-derived for clean one-hot. |
| 5 | **`BitwiseOperandSeRecoverFFN`** | `efficient_alu_neural.py` | **158** | **B** | Recovers operand-A from `SE_ALU` on OR/XOR/AND MARK_AX rows after L14 ALU-clear crushes it, so the bitwise nibble lookup decodes. Deleting needs the bitwise engine re-derived for clean operand (or the L14 clear scoped off bitwise rows). |
| 6 | **`CmpOperandSeRecoverFFN`** | `efficient_alu_neural.py` | **141** | **B** | **The load-bearing wall.** Re-materialises the dirty hybrid (true nibble +6.0 PLUS index-0 art +5.3, cell-8/15 residues) on CMP MARK_AX rows so the ordering/eq engines' `-0.5/-0.8` per-nibble blockers land the lt/eq flags in the (0.75,1.5) decode window. A clean one-hot overshoots → GT/EQ/LT mis-decode. |
| 7 | **`LoadedOperandAddHi15ClearFFN`** (+ install glue `make_loaded_operand_add_hi15_clear_op` 97) | `efficient_alu_neural.py` | **127** (+97) | **A** | Clears the SP/BP-frame address-nibble leak (cell 13/15, +5.5) + operand-B hi-nibble bleed (+1.0) on the loaded-operand **ADD** MARK_AX row. **Now runs as the `inner` of `CleanOperandOneHotFFN` (CLEAN_OPERAND_ADD, default-ON) — the outer clean-snap already zeros every non-argmax ALU_HI cell on the ADD row before this runs → provably inert.** |
| 8 | **`CmpLoadedOperandCleanFFN`** | `efficient_alu_neural.py` | **121** | **B** | L9 loaded-operand clean for the CMP path (companion to #6). Same clean-operand-CMP derive-root. |
| 9 | **`CleanOperandOneHotFFN`** | `efficient_alu_neural.py` | 99 | — | NOT a corrector — the DERIVED clean one-hot (CLEAN_OPERAND_ADD, +13). Listed because it SUBSUMES #7 (and, with the CMP derive-root, #4/#5/#6/#8). |
| 10 | **`make_no_stack0_pc_highbyte_clear_op`** | `l0_ops.py` | 151 | C | Clears leaked PC high-byte in the no-stack0 campaign emission frame. |
| 11 | **`make_ax_hibyte_clear_allstep_op`** | `l11_ops.py` | 124 | C | AX high-byte all-step clear (AX byte-1/2/3 0xFF-leak family, `project_ax_byte1_dump_is_h1_onehot_wall`). |
| 12 | **`make_l10_add_high_byte_adder_op`** | `l10_ops.py` | 95 | C | Multi-byte ADD high-byte completer; the false-fire target of the SIGNEXT_LEA guard (#14). |
| 13 | **`_l10_jsr_bp_byte3_clear_rules`** (`C4_JSR_BP_BYTE3_CLEAR`) | `l10_ops.py` | 79 | C | JSR BP byte-3 clear. |
| 14 | **`ShiftOutputClearFFN`** (`C4_SHIFT_OUTPUT_B0_CLEAR`) | `efficient_alu_neural.py` | 77 | B | Clears a shift byte-0 OUTPUT residue; tied to the shift-derive (DERIVE_SHIFT, already default-ON) inter-byte bit-spill. |
| 15 | **`_ax_byte1_signext_lea_blockers`** (`C4_AX_BYTE1_SIGNEXT_LEA`, 20) + `_strengthen_l10_*_wrong_byte_blockers` (38) + `_arith_guard_addsub_blockers` (24) + `_tail_lea_e8_ent_byte0/1_blockers` (30) + `_lea_e8_nested_ent_axdump_blockers` (15) | `l10_ops.py` | **~127** | **C** | Small per-case AX byte-1 / LEA-E8 / arith-guard blockers. Each guards a specific cross-step opcode-residue false-fire (byte-1, NOT byte-0 → NOT subsumed by CLEAN_OPERAND_ADD). |
| 16 | **AX-dump / emission patch family** (`make_ax_byte1_dump_repopulate_op`, `make_ax_byte1_carry_overflow_flag_op`, `make_b1_to_output_op`, `make_sili_b1_restore_op`, `make_bp_save_dump_repopulate_op`, `make_jsr_pc_byte1_emit_op`) | `l11_ops.py` / `l14_ops.py` / `model_ops.py` | ~400 (est.) | C | Register-byte emission repopulation for the cross-step carry bands (AX byte-1, BP-save, SILI byte-1, JSR PC byte-1). DERIVATION_ROADMAP maps these to the generic `cross_step_carry` + R-BYTE emitter (EMIT G5). |
| 17 | Small flag-gated loop/case fixes (`_l10_loop_si_byterow_marker_clear` 31, `loop_lea_b0_e0/e8`, `absdiff_*`, `nonfirst_psh_sp`, etc.) | `l10_ops.py` | ~200 (est.) | C | Per-cluster loop/absdiff/psh case fixes, each behind a default-ON kill-switch flag. |

**Bespoke-corrector total (items 1–17, ex. the derived #9):** ≈ **8,900 LOC**
of the 66.3k ops core + 3.1k efficient_alu wrapper file (≈ 13% of the combined
surface). Item #1 alone is 4,761 LOC (69% of the corrector debt), item #2
+957, item #3 +723.

### What is EXCLUDED (derived families, not correctors)

- L10 bitwise (OR/XOR/AND 574 each) / cmp ordering+eq engines / wide-add/sub —
  `wide_alu_dsl` computed lookups, load-bearing, DERIVE_* default-ON.
- L15/L14 memory-CAM heads, L5 decode FFN (DecodeSpec), L13 shifts — derived.
- `_byte_value_writeback_rules` / `_computed_byte_writeback_route_rules` — the
  #389 computed-copy route (the collapse *target* for #1, not a corrector).

---

## STEP 2 — CLASSIFICATION (A / B / C)

### CLASS A — NOW-REDUNDANT (a landed derivation makes it deletable)

**A1. `LoadedOperandAddHi15ClearFFN` + its cell-widening companions
(`C4_LOADED_OPERAND_ADD_HI15_CLEAR`, `C4_FUNCADD_ALU_HI13_CLEAR`,
`C4_FUNC_ADD_B0_HINIB`) — inventory item #7, 127 + 97 glue = 224 LOC.**

*Why redundant:* CLEAN_OPERAND_ADD (`CleanOperandOneHotFFN`, landed default-ON
this cycle, +13) wraps `LoadedOperandAddHi15ClearFFN` as its `inner` and runs
the clean-snap FIRST (`forward`: snap ALU_LO/HI + AX_CARRY_LO/HI to the per-band
argmax one-hot at 6.0, zeroing EVERY other cell, on the 5 arithmetic dims incl
`OP_ADD` — `efficient_alu_neural.py:1412`). The Hi15Clear family is **ADD-only**
and clears ALU_HI cells 13/15 (or all-16 under `FUNC_ADD_B0_HINIB`) in the
window `(0.5, 5.85)`. After the clean-snap, those cells are already exactly 0 on
the ADD row → the inner's contaminant window finds nothing → **provably inert**.
The `CleanOperandOneHotFFN` docstring states this verbatim: "It GENERALISES the
address-leak / index-0-artifact correctors (`LoadedOperandAddHi15ClearFFN`, the
func-add hi-nibble clear …)".

*Fast-gate evidence (§RESULTS):* `tools/fast_gate.py --flag
C4_LOADED_OPERAND_ADD_HI15_CLEAR` (OFF = corrector removed; ON = present),
CLEAN_OPERAND_ADD default-ON in both — **byte-identical corpus verdicts** (same
`got_ax`, same `divergence_step`) on every diverging id across
add/sub/mul/func_add/func_mul/var_update/var_mul/expr_add_mul/absdiff/if_var,
net = **0**. Corroborated by a **numerical inertness proof** (`max_abs_diff =
0.0` between the clean-wrap-over-Hi15Clear and clean-wrap-over-bare-PureFFN
forward on a synthetic dirty ADD MARK_AX row). ⟹ **FREE −224 LOC delete
(EXECUTED, §STEP 3).**

### CLASS B — DERIVABLE-ROOT (deletion needs a specific root derived first)

Ranked by LOC payoff. Each is a "recover the dirty hybrid so the calibrated
engine decodes" wrapper; the root is "re-derive that engine against a clean
one-hot" (the Phase-2 the CLEAN_OPERAND feasibility doc scoped).

| Corrector | LOC | Derive-root required | Payoff |
|-----------|----:|----------------------|-------:|
| **`CmpOperandSeRecoverFFN` + `CmpLoadedOperandCleanFFN`** (#6+#8) | **262** | **Re-derive the L10 CMP nibble comparators for a clean one-hot** — re-tune every `_layer10_alu_ordering_engine` / `_alu_eq_engine` `-0.5/-0.8` blocker to the 6.0/0.0 decode window (`DERIVE_CMP`, `C4_FUNC_CMP_OPERAND_CLEAN` derive-root). Unblocks flipping `CLEAN_OPERAND` (the 11-consumer flag) — captures the −74→+ CMP arithmetic gain too. | −262 + the CMP arithmetic pass-gain |
| **`MulOperandSeRecoverFFN`** (#4) | **218** | Re-derive the L11 mul-partial lookup for clean operand (extend CLEAN_OPERAND_ADD's `OP_MUL` snap to the L11 read, verify the width-2 product survives). Partly in flight (`C4_MUL_*_SE_RECOVER`, `C4_WIDE_MUL_BYTE1_COMPUTED`). | −218 |
| **`BitwiseOperandSeRecoverFFN`** (#5) | **158** | Scope the L14 ALU-clear OFF the bitwise operand rows (it crushes operand-A → the recover exists only to undo that), OR re-derive the bitwise nibble lookup for the crushed shape. `DERIVE_BITWISE` default-ON is the engine side. | −158 |
| **`ShiftOutputClearFFN`** (#14) | 77 | Fold the shift byte-0 OUTPUT residue into the `wide_shift_rules` `carry_rule="bit_spill"` emit (DERIVE_SHIFT already default-ON) so no post-clear is needed. | −77 |
| **L6 PC-override / branch-byte family** (#2) | 827+ | CONTROL derive-root G9: emit exclusive `pc_override(gate,target)` gates at authoring via `cancel_residual_rule`; retire `make_branch_override_patch_op`. | −827 (authoring; unit-neutral) |

**Class-B total payoff:** ≈ **1,542 LOC** behind 5 bounded derive-root missions.

### CLASS C — ARCHITECTURAL (guards a deep multi-layer root; hard/multi-session)

| Corrector | LOC | Why it's architectural |
|-----------|----:|------------------------|
| `_tail_bit32_result_correction_rules` (#1) | 4,761 | ~40 output-byte-guarantee sub-families; width-locked & load-bearing (`project_l10_tail_bank_width_sensitive`). The `byte_copy`/#389-route collapse (EMIT G4) is a re-EXPRESSION (unit-neutral), not a removal — the guarantees are the correctness spine of the emission frame. Only the deep clean R-FRAME+R-BYTE emitter (EMIT G5) makes them removable, and only in lockstep. |
| L14 clear/guard family (#3) | 723 | Guards the memory-CAM generation over-broad writes; removal needs the CAM store-direction + provenance-anchor spec (MEMORY G2) so the generation is exclusive by construction. |
| AX-dump / emission patch family (#16) | ~400 | Cross-step register-byte carry (AX byte-1, BP-save, SILI, JSR PC byte-1). Removal = generic `cross_step_carry` + R-BYTE emitter (EMIT G5, the deepest lever). `C4_AX_BYTE1_*` is an H1-one-hot architectural wall (`project_ax_byte1_dump_is_h1_onehot_wall`). |
| `make_no_stack0_pc_highbyte_clear` (#10), `make_ax_hibyte_clear_allstep` (#11), `make_l10_add_high_byte_adder` (#12), the small byte-1/LEA blockers (#15), loop/case fixes (#17) | ~600 | Each guards a specific cross-step opcode-residue false-fire in the campaign emission frame (byte-1, not byte-0 → NOT subsumed by any byte-0 clean-operand derivation). Collapse only under the clean generic emitter. |

**Class-C total:** ≈ **6,500 LOC** — the deep spine; removable only via the
EMIT/MEMORY derive-roots (multi-session), NOT free.

---

## STEP 3 — FREE DELETES EXECUTED

**A1 — `LoadedOperandAddHi15ClearFFN` removed (−224 LOC). EXECUTED (commit
`5ebb8da2`).**

Deleted:
- `LoadedOperandAddHi15ClearFFN` class (`efficient_alu_neural.py`, **−127**).
- `make_loaded_operand_add_hi15_clear_op` (`alu_ops.py`, **−97**) + its
  registration in `full_vm_compiler_dynamic.py` + its export in
  `_legacy_redirect.py`.
- `make_clean_operand_op` now `requires={"after": ("efficient_l8_addsub_wrap",)}`
  and wraps the raw L8 operand-delivery `PureFFN` directly (block 12 chain is now
  `CleanOperandOneHotFFN → PureFFN`, was `→ LoadedOperandAddHi15ClearFFN →`).
- `faithful_interpreter.COMPOSITE_ALU_FFN`: swapped the removed
  `LoadedOperandAddHi15ClearFFN` name for `CleanOperandOneHotFFN` (the outer
  wrap now recognised for zero-opaque IR coverage).

Left in place (inert, byte-harmless, not on any live build path anymore): the 4
now-dead flag defs (`loaded_operand_add_hi15_clear_enabled`,
`funcadd_alu_hi13_clear_enabled`, `func_add_b0_hinib_enabled`,
`operand_cam_fix_enabled`) + their cache-key entries — removing them changes no
weights; a follow-up dead-flag sweep can drop them (~110 LOC) once a fleet pass
re-runs the full memo/serialised-cache-key audit.

**Re-verification (all gates PASS):**
- **golden flag-OFF byte-identity:** `tools/_isa_golden_hash.py` == `e50521f3`
  (unchanged — the lookup/golden build never installed the wrap).
- **campaign build:** block-12 ffn = `CleanOperandOneHotFFN → PureFFN` (Hi15Clear
  gone); all other SeRecover wraps intact.
- **fast-gate `--base main` (main corrector-present vs HEAD corrector-removed),
  affected clusters (74 ids × 10 clusters):** base 53p/21f, HEAD 53p/21f, **NET =
  0** (0 gains, 0 regressions, 0 per-id verdict differences — byte-identical
  campaign build).
- **smoke:** `pytest tests/test_smoke.py` **51/51 PASS** (spec_k=0, GPU, isolated
  cache).

**Running −LOC total (free deletes this session): −224.**

---

## STEP 4 — RANKED DERIVE-ROOT MISSIONS (Class B, by LOC payoff)

For the LOC-reduction fleet, in leverage order:

1. **Re-derive the L10 CMP comparators for a clean one-hot** → delete
   `CmpOperandSeRecoverFFN` + `CmpLoadedOperandCleanFFN` (**−262**) AND unblock
   the full `C4_CLEAN_OPERAND` flag (the CMP arithmetic pass-gain the −74 masked).
   The single highest-leverage corrector-removal mission. Root spec:
   `DERIVE_CMP`, `C4_FUNC_CMP_OPERAND_CLEAN`.
2. **Re-derive the L11 mul-partial for clean operand** → delete
   `MulOperandSeRecoverFFN` (**−218**). Partly in flight
   (`C4_WIDE_MUL_BYTE1_COMPUTED`, `C4_MUL_*_SE_RECOVER`).
3. **Scope the L14 ALU-clear off bitwise operand rows** → delete
   `BitwiseOperandSeRecoverFFN` (**−158**).
4. **CONTROL G9 `pc_override` derive** → collapse the L6 PC-override / branch
   family (**−827** authoring, unit-neutral).
5. **Fold shift byte-0 residue into `wide_shift_rules` bit-spill emit** → delete
   `ShiftOutputClearFFN` (**−77**).

**Deep (Class C, multi-session, NOT corrector-removal missions per se):** EMIT
G5 clean R-FRAME+R-BYTE emitter (unlocks #1 tail-bank collapse to `byte_copy`,
the AX-dump family, the byte-1 blockers — the ~7k Class-C spine); MEMORY G2
provenance-anchor (unlocks the L14 clear family).

---

## VERIFICATION GATES (this doc + any delete)

- Golden flag-OFF byte-identity: `CUDA_VISIBLE_DEVICES="" tools/_isa_golden_hash.py`
  == `e50521f3` (this doc reads ops + writes markdown only → unchanged).
- Corrector deletability: `tools/fast_gate.py --flag <kill-switch>` net ≥ 0
  (OFF = corrector removed).
- Full: `run_1096_canonical --criterion full_trace --spec-k 0 --max-steps-cap 40`
  ≥ 593; `pytest tests/test_smoke.py` 51/51.

## RESULTS (evidence for the A1 free delete)

### 1. Numerical inertness proof (the primary evidence)

On the built campaign model, block 12's ffn chain was
`CleanOperandOneHotFFN → LoadedOperandAddHi15ClearFFN → PureFFN`. Feeding a
synthetic dirty ADD MARK_AX row (true nibble @3 = 6.0, cell-15 leak = 5.5,
operand-B bleed @4 = 1.0) and comparing the forward of `clean(x)` (wrap over
Hi15Clear over PureFFN) vs `clean_noinner(x)` (wrap over bare PureFFN):

```
max |y_with_inner - y_without_inner| on ADD MARK_AX rows = 0.0
=> LoadedOperandAddHi15ClearFFN is INERT (byte-identical) after the clean-snap
```

The clean-snap (which runs FIRST, `forward` at `efficient_alu_neural.py:1412`)
keeps only the per-band argmax at 6.0 and zeros every other ALU_HI cell on the
ADD row, so cells 13/15/4 are already 0 before the inner's contaminant-window
clear runs → the inner has nothing to clear.

### 2. fast-gate `C4_LOADED_OPERAND_ADD_HI15_CLEAR` (OFF=removed, ON=present)

Byte-level per-id compare on the 8 diverging ids logged across the affected
clusters — `got_ax` and `divergence_step` **IDENTICAL** OFF vs ON on every row
(idx 275/279/283/287 expr_add_mul, 1046/1052/1058/1064 absdiff family). The
other 66 ids pass in both states. Net = 0.

### 3. fast-gate `--base main` (main present vs HEAD removed), 74 ids × 10 clusters

```
OFF (base=main, corrector present): pass=53 fail=21
ON  (HEAD, corrector removed):      pass=53 fail=21
GAINS 0   REGRESSIONS 0   NET = 0   (0 per-id verdict differences)
```

### 4. golden flag-OFF byte-identity

```
CUDA_VISIBLE_DEVICES="" tools/_isa_golden_hash.py
  state_dict_sha256 = e50521f32b0ed952d5730f79b63adb8c4c78f4d4f0466d3bcbaa354bb3c90e86  (== e50521f3)
```

### 5. smoke

```
pytest tests/test_smoke.py  ->  51 passed, 1 deselected  (spec_k=0, GPU)
```

**Conclusion:** the A1 delete (`LoadedOperandAddHi15ClearFFN`, −224 LOC) is a
verified FREE delete — byte-identical golden, byte-identical campaign build, net
0 corpus, smoke 51/51. Committed `5ebb8da2`.

---

## APPENDIX — corrector-debt LOC by file (measured at `e50521f3`, pre-delete)

| File | total LOC | corrector LOC (of that) | biggest corrector |
|------|----------:|------------------------:|-------------------|
| `l10_ops.py` | 13,737 | ~5,300 | `_tail_bit32_result_correction_rules` 4,599 |
| `l6_ops.py` | 4,851 | ~960 | PC-override / SP-fixup family |
| `l14_ops.py` | 5,912 | ~1,100 | clear/guard family 723 + AX-emission |
| `efficient_alu_neural.py` | 3,143 | ~815 | SeRecover family (Cmp/Mul/Bitwise) 517 |
| `l11_ops.py` | 3,200 | ~400 | AX-dump emission family |
| `l0_ops.py` | 1,037 | 151 | no-stack0 PC hi-byte clear |

The corrector debt is **heavily concentrated in the L25 tail bank (Class C,
item #1, 4,761 LOC = 53% of all corrector debt)** — deletable only via the deep
EMIT G5 clean-emitter derive-root, not free. The FREE / near-free wins are the
operand-cleanup wrappers (Class A done −224; Class B −715 behind 3 bounded
engine-derive missions).
</content>
