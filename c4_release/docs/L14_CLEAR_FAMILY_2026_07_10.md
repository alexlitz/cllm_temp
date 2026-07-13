# L14 CLEAR / GUARD CORRECTOR FAMILY — inventory, free deletes, derive-root missions

**Date:** 2026-07-13  **Branch:** `l14-clear-family` (off `main`, HEAD was
`cdb6c23f` = 593/1096)
**Golden flag-OFF:** `e50521f3` (`tools/_isa_golden_hash.py`) — **UNCHANGED**
after this session's delete.
**Scope:** `neural_vm/unified_compiler/ops/l14_ops.py` — the L14
clear/guard/suppress/zero corrector bank (CORRECTOR_REMOVAL_ROADMAP item #3,
Class C).

This is the L14 companion to `docs/CORRECTOR_REMOVAL_ROADMAP_2026_07_10.md`
(item #7 / Class A landed the first free delete, `LoadedOperandAddHi15ClearFFN`,
−224). Here we SCOPE the L14 clear/guard family, execute the FREE deletes it
contains, and classify the rest by the derive-root that would subsume them.

---

## SUMMARY

- **1 FREE delete EXECUTED (−314 LOC):** `_clear_l14_mem_generation_overbroad_sp_suppression`
  — a dead 322-LOC imperative post-bake helper with **NO caller anywhere in the
  repo** (its logic was folded into the LIVE declarative spec
  `_layer14_mem_generation_head_specs_with_overrides`, Wave 2E). Byte-identical
  golden `e50521f3`, smoke 51/51.
- **NO other free deletes exist in the L14 clear family.** A numerical inertness
  probe (`tools/_probe_l14_clear_inertness.py`) proves all 4 installed clear ops
  ACTIVELY fire on the memory-generation corpus (max |W_down·hidden| = 1700 /
  2.0 / 91.7 / 91.7), and the 6 suppress/zero ops guard live cross-layer roots.
- **The rest is genuinely Class C** — it guards the memory-CAM generation
  over-broad writes and the AX-byte-1-3 cross-step leak. Removal needs the
  MEMORY G2 provenance-anchor derive-root (generation exclusive by
  construction) and/or the EMIT G5 clean R-BYTE emitter (AX-byte carry), both
  multi-session. Classified in §4.

**LOC accounting:** the roadmap scoped item #3 at "723 LOC" (the 6 core
clear/guard function bodies). The FULL installed L14 clear/guard/suppress/zero
family (incl. all `make_*` glue + the AX-byte-zero cluster + the two
default-suppress ops) is **≈1,812 LOC**, plus the now-deleted 314-LOC dead
helper. After this session `l14_ops.py` is **5,598 LOC** (was 5,912).

---

## STEP 1 — FAMILY INVENTORY (measured at `cdb6c23f`)

`kind` legend: **G**=guard helper (post-bake weight tweak), **C**=clear op
(installed FFN units), **S**=suppress op (installed), **Z**=AX-byte-zero op
(installed), **DEAD**=no caller.

| Fn / op | LOC | kind | installed? | Root it guards |
|---------|----:|:----:|:----------:|----------------|
| `_clear_l14_mem_generation_overbroad_sp_suppression` | **314** | DEAD | **no** | (was) L14 MEM-gen H1[SP] over-suppression — folded into live spec |
| `_guard_l14_output_units_on_step_boundary` | 94 | G | yes (called by all 4 clear ops) | Stops L14 OUTPUT cleanup units from firing on inter-row (non-VM-structure) positions |
| `_block_l14_jsr_ax_zero_on_stack0_bytes` | 32 | G | yes | Keeps JSR AX-byte-zeroing off the JSR return-address STACK0 rows |
| `_disable_l14_stack0_jsr_hi0_default` | 44 | G | yes (called by output_corruption bake) | Wipes the JSR-STACK0-HI0 marker unit (net W_down 0 — allocated for chain-offset parity) |
| `_boost_l14_psh_mem_marker_high_nibbles` | 38 | G | yes (called by output_corruption bake) | PSH MEM-marker high-nibble boost extension |
| `layer14_temp_clear` (rules 80 + op 79) | 159 | C | yes (units 0-3) | TEMP[0] clear at PC/OP_LEV; TEMP[8/9] negative-residue clamp; ADD byte-1 high-zero cleanup |
| `layer14_clear_addr_key_pollution` (47 + 81) | 128 | C | yes (units 4-51) | Cancels ADDR_KEY residue L9 attn leaves on non-MEM/non-marker rows (−4.0/S per cell) |
| `layer14_clear_output_corruption` (99 + 78) | 177 | C | yes (units 52-69) | Boosts OUTPUT[0] at STACK0 byte rows to out-vote L14 attention bleed |
| `layer14_clear_mem_marker_output` (51 + 79) | 130 | C | yes (units 70-133) | Cancels −115 L14-attn OUTPUT corruption at the MEM marker for OP_JSR/OP_ENT |
| `layer14_mem_addr_src_default_suppress` (61 + 94) | 155 | S | yes | Cancels the L3 `MEM DEFAULT +0.940` baseline at SI/SC store rows (MEM_ADDR_SRC=1) |
| `layer14_jsr_mem_default_suppress` (66 + 101) | 167 | S | yes | Same L3 +0.940 cancel on the PSH/JSR/ENT store path (MEM_STORE=1, MEM_ADDR_SRC=0) |
| `layer14_jsr_ax_bytes_zero` (74 + 97) | 171 | Z | yes | Restores AX bytes 1-3 = 0x00 across JSR (8-bit-AX convention) |
| `layer14_alu_nocarry_ax_bytes_zero` (55 + 118) | 173 | Z | yes | AX bytes 1-3 = 0 for AND/OR/XOR/SHR (byte-sized result), gated on TEMP[7] |
| `layer14_ent_ax_bytes_zero` (59 + 89) | 148 | Z | yes | AX bytes 1-3 = 0 at ENT step 0 (fixes `test_lea_basic` SP-byte→AX leak) |
| `layer14_lc_ax_bytes_zero` (52 + 144) | 196 | Z | yes | AX bytes 1-3 = 0 for LC (1-byte char load) |

(LOC for installed ops = `_rules` body + `make_*_op` factory; the 5-line `_ir`
wrappers are omitted from the per-row LOC but included in the file total.)

### Shared root map

The whole family exists because L14 is where the memory-generation attention
(`layer14_mem_generation`) writes MEM addr/value tokens AND the AX register
output cascades. Three roots dominate:

1. **Memory-CAM over-broad writes** (`temp_clear`, `clear_output_corruption`,
   `clear_mem_marker_output`, `clear_addr_key_pollution`, both
   `*_default_suppress`, and the whole `_guard_*`/`_block_*`/`_disable_*` guard
   scaffolding). The L14 mem-gen heads and the L9 ADDR_KEY leak and the L3
   +0.940 MEM-default all write onto rows the generation also owns → each clear
   cancels one over-broad contribution so the generation lands exactly. The
   guard helper (`_guard_l14_output_units_on_step_boundary`) is the shared
   boundary-bias that keeps every clear unit from firing on inter-row junk.
2. **AX bytes 1-3 cross-step leak** (the 4 `*_ax_bytes_zero` Z-ops). C4's 8-bit
   AX convention needs AX[1..3]=0 whenever a step doesn't compute a wide result;
   the register-dump band leaks the SP/opcode residue there, so each op re-zeros
   the byte-1-3 nibbles per opcode class (JSR / nocarry-ALU / ENT / LC). This is
   the L14 face of the `project_ax_byte1_dump_is_h1_onehot_wall` H1-one-hot
   architectural wall (roadmap item #16, EMIT G5).

---

## STEP 2 — REDUNDANCY TEST (numerical inertness)

Does any landed derivation already subsume an L14 clear op? The candidate is
**`CleanOperandOneHotFFN` (CLEAN_OPERAND_ADD, default-ON)** — the derivation
that subsumed the Class-A `LoadedOperandAddHi15ClearFFN`.

**Answer: NO.** `CleanOperandOneHotFFN` wraps the **L8 main block (physical
block 11/12)** and cleans the **operand-A/B bands (ALU_LO/HI, AX_CARRY_LO/HI)**
on the binary-op MARK_AX row (`efficient_alu_neural.py:1215`). The L14 clear
family lives on the **L14 memory-generation FFN (physical block 34** in the
campaign build**)** and writes the **OUTPUT / TEMP / ADDR_KEY / MEM-marker**
bands. Different layer, different bands, different rows — **zero overlap**. No
landed derivation touches the L14 memory-generation OUTPUT/TEMP/ADDR_KEY path.

### Inertness probe (`tools/_probe_l14_clear_inertness.py`)

Method (the same numerical-inertness proof used for Hi15Clear): locate the
clear-op FFN block by its ADDR_KEY-pollution diagonal signature
(`W_down[ADDR_KEY+k, unit 4+k] ≠ 0`, k=0..47 → physical block 34), hook its
forward, run 8 representative memory-gen programs (ids spread across the
corpus), and measure the max per-unit W_down contribution
`|hidden[u]| · ‖W_down[:,u]‖∞` over every row of every step.

```
selected clear-op FFN block: physical 34
clear op                  max |W_down contrib|   verdict
temp_clear                             1700.23   ACTIVE   units[0:4]
addr_key_pollution                        2.00   ACTIVE   units[4:52]
output_corruption                        91.72   ACTIVE   units[52:70]
mem_marker_output                        91.72   ACTIVE   units[70:134]
fully-inert units (0 contrib on sample): 2 / 134
```

**Verdict: all 4 installed L14 clear ops are ACTIVE (not inert).** They fire and
write real corrections on the memory-generation corpus. Only 2 of 134 units
show 0 contribution on the sample (chain-offset parity units) — not a
deletable op. **There is no free delete among the LIVE L14 clear ops.**

Consistency with the build: all 134 clear units are counted in the golden build
("TOTAL FFN units 42149 … no dead units") — i.e. they write nonzero weights
into the golden state_dict, so removing any would CHANGE `e50521f3`. A delete is
therefore byte-identity-BREAKING and must go through a derive-root, not a free
delete.

---

## STEP 3 — FREE DELETES EXECUTED

### F1 — `_clear_l14_mem_generation_overbroad_sp_suppression` removed (−314 LOC). EXECUTED (commit `8714edc8`).

*Why it's free:* the helper is a legacy imperative post-bake override for the
L14 MEM-generation heads. A repo-wide grep for the symbol finds ONLY docstring
`:func:` cross-references (l14_ops.py lines 694/966/1565) — **no caller on any
build path, or in any test, anywhere in the repo.** Its logic (the slot-44 ENT
old-BP SP-suppression, the `slot44_marker_block_s = 1_000_000.0` blockers, the
value-head CLEAN_EMBED×2 gain) was folded into the LIVE declarative spec
`_layer14_mem_generation_head_specs_with_overrides` (Wave 2E), which is what
`make_layer14_mem_generation_op`'s bake actually lowers. The in-file comment at
the old line 2128 states it verbatim: *"this imperative … helper is no longer
invoked (folded into the spec)."*

Because it never executes, deleting it changes NO weights → the golden
state_dict is bit-identical.

Deleted: the 322-line function body (replaced by a 6-line provenance comment).
Updated: the 3 stale `:func:` docstring cross-references (they no longer point
at a deleted symbol).

*Verification (all gates PASS):*
- **golden flag-OFF byte-identity:** `tools/_isa_golden_hash.py` (CPU,
  `CUDA_VISIBLE_DEVICES=""`) = `e50521f32b0ed952d5730f79b63adb8c4c78f4d4f0466d3bcbaa354bb3c90e86`
  = **e50521f3** (unchanged pre/post delete).
- **smoke:** `pytest tests/test_smoke.py` → **51 passed** (spec_k=0, GPU 0,
  isolated cache), 206.91s.
- **import + inertness probe:** `l14_ops` imports clean;
  `hasattr(..., '_clear_l14_mem_generation_overbroad_sp_suppression') == False`.
- **`test_l14_per_op.py::mem_generation`** 8 failures are **PRE-EXISTING** at
  `cdb6c23f` (byte-identical `AttributeError: _SetDim has no attribute
  STACK0_BYTE_VAL_1_LO` — a stale test-stub dim, not our change; confirmed by
  checking out the pre-delete `l14_ops.py` and reproducing the same fail).

**Running −LOC total (L14 clear family, this session): −314.**
`l14_ops.py`: 5,912 → 5,598.

---

## STEP 4 — CLASS-C CLASSIFICATION + DERIVE-ROOT MISSIONS (the non-free rest)

All remaining L14 clear/guard/suppress/zero ops are **Class C** (guard a deep
multi-layer root; byte-identity-BREAKING to remove). Grouped by the derive-root
that would make them removable.

### C1 — Memory-CAM generation clears → **MEMORY G2 (provenance-anchor)**

`layer14_temp_clear`, `layer14_clear_output_corruption`,
`layer14_clear_mem_marker_output`, `layer14_clear_addr_key_pollution`,
`layer14_mem_addr_src_default_suppress`, `layer14_jsr_mem_default_suppress`,
plus the guard scaffolding (`_guard_l14_output_units_on_step_boundary`,
`_disable_l14_stack0_jsr_hi0_default`, `_boost_l14_psh_mem_marker_high_nibbles`).
**≈1,090 LOC.**

*Root:* the L14 mem-generation heads write MEM addr/value onto rows that ALSO
carry (a) the L9 ADDR_KEY attention leak, (b) the L3 `MEM DEFAULT +0.940`
baseline, and (c) the −115 L14-attn OUTPUT corruption at the MEM marker. Each
clear cancels one over-broad contribution. They are load-bearing (inertness
probe: 1700 / 91.7 / 2.0 contributions).

*Derive-root:* **MEMORY G2** — give the memory-CAM store a provenance-anchor +
store-direction spec so the generation is **exclusive by construction** (writes
ONLY the MEM addr/value cells at the MEM rows, reads a clean store-value marker
rather than the polluted STACK0/OUTPUT residue). Then no L3-baseline cancel and
no attn-corruption clear are needed → the whole C1 cluster collapses. Related
in-flight probes: `tools/_probe_l14_store_cam_derive.py`,
`tools/_probe_l14_store_vo.py`. This is multi-session (the same MEMORY G2 the
roadmap names for item #3).

### C2 — AX bytes 1-3 zero restore → **EMIT G5 (clean R-BYTE emitter)**

`layer14_jsr_ax_bytes_zero`, `layer14_alu_nocarry_ax_bytes_zero`,
`layer14_ent_ax_bytes_zero`, `layer14_lc_ax_bytes_zero`, plus the
`_block_l14_jsr_ax_zero_on_stack0_bytes` guard. **≈720 LOC.**

*Root:* C4's 8-bit-AX convention requires AX[1..3]=0 on any step that doesn't
compute a wide result, but the register-dump band leaks SP/opcode residue into
those nibbles. Each op re-zeros byte 1-3 per opcode class (JSR / nocarry-ALU /
ENT / LC). This is the L14 face of the H1-one-hot architectural wall
(`project_ax_byte1_dump_is_h1_onehot_wall`) — the SAME cross-step register-byte
carry root as roadmap item #16 (the AX-dump / emission patch family).

*Derive-root:* **EMIT G5** — the generic `cross_step_carry` + R-BYTE emitter
that decides each register byte's value by construction (carry band +
re-pointed dump), so byte 1-3 are computed 0 rather than leaked-then-cleared.
This is the deepest lever (the same one that unlocks the L25 tail bank and the
AX-dump family); **do NOT throw a single agent at it** (two agents bounced on
the byte-1 root per MEMORY). Multi-session.

---

## VERIFICATION GATES (this doc + the F1 delete)

- Golden flag-OFF byte-identity: `CUDA_VISIBLE_DEVICES="" tools/_isa_golden_hash.py`
  == `e50521f3`. **PASS** (unchanged).
- Smoke: `pytest tests/test_smoke.py` (spec_k=0). **PASS 51/51.**
- Inertness (per-op): `tools/_probe_l14_clear_inertness.py` — all 4 live clear
  ops ACTIVE (not deletable); dead helper had 0 callers (deletable).
- Full-1096 (defer to main, GPU-batched): expected UNCHANGED at 593 (the F1
  delete is dead-code removal → byte-identical build).

## FILES

- `neural_vm/unified_compiler/ops/l14_ops.py` — F1 delete + 3 docstring-ref
  updates (5,912 → 5,598 LOC).
- `tools/_probe_l14_clear_inertness.py` — the L14 clear-op inertness probe
  (new; block-34 auto-locate by ADDR_KEY diagonal, per-unit W_down·hidden max).
- `docs/L14_CLEAR_FAMILY_2026_07_10.md` — this doc.
