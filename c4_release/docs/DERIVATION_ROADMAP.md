# DERIVATION ROADMAP — the correct-by-construction path to <5K core LOC + 100% passing

**Status:** EXECUTION ROADMAP (consolidates the 5 family maps
`semantic_spec_{DECODE,MEMORY,ALU,CONTROL,EMIT_FRAMING}.md` + the merged
`DERIVE_DECODE_PILOT_2026_07_03.md`). READ-only synthesis: no weights touched.
**Golden gate:** `tools/_isa_golden_hash.py` ==
`81557d21422f3eada0a87c677b00dced41cc26c3ee3bfb094c5eeb71c9b4d3cb` (`81557d21`),
verified UNCHANGED building this doc (this file reads ops + writes markdown only).

---

## HEADLINE VERDICT

**Is <5K core + 100% passing reachable? — SPLIT verdict, honestly:**

1. **The *spec* is tiny and 100% real.** Every one of the 38 opcodes reduces to
   **one data row** over ~6 generic sub-machines (decode-lookup, nibble-adder,
   CAM, ALU-compute, PC-mux, cross-step-carry, frame state-machine). The
   per-opcode DATA is **~38 rows ≈ 400–600 LOC**. This is proven, not aspirational:
   the DECODE pilot (#391) derives the *entire* L5 opcode-decode FFN
   **byte-for-byte** from a `DecodeSpec` table with **zero per-opcode rules**
   (golden hash held flag-ON). Decode is a lookup; the thesis is real.

2. **The *generic engine* is largely built and can be compact.** The
   `isa_semantics_dsl` + `building_blocks_dsl` + `wide_alu_dsl` generators
   (~5.7k LOC today) already express decode, CAM, cross-step-carry, all bitwise
   / shift / 8-bit-add-sub / byte-0-mul, the nibble adder, and the byte-copy
   route. With the 3 in-flight gap-primitives (`marker_broadcast`,
   `multi_pass_rules`, cross-lane `byte_copy`) it covers the whole ISA. A
   compacted engine target is **~3–4k LOC**.

3. **The *irreducible infra floor* is the wall.** The weight-lowering that MUST
   remain (`ir.py` FFNRule→W_*, attention-spec→W_qkvo, `full_vm_compiler_dynamic`,
   `layer_compiler`, `primitives.py`, `dim_registry` + resolver, the 3
   allocators) measures **~13.6k LOC today**, compactable to a **hard floor of
   ~6–8k**. This — not the spec — is what determines <5K feasibility.

**Bottom line:** **<5K for the SPEC + a lean engine's *authoring surface* IS
reachable** (spec ~0.5k + compacted DSL ~3–4k). **<5K for the WHOLE core
including the weight-lowering infra is NOT honestly reachable** — the honest
floor is **~8–12k** (spec+engine ~4–5k + irreducible lowering infra ~6–8k),
dominated by the irreducible FFRule→W_* / attn→W_qkvo lowering + the
compiler/layout/allocator substrate that no amount of spec-derivation removes.
The current **63.8k ops core → ~4–5k spec+engine + ~6–8k infra ≈ 10–13k core**
is the load-bearing endpoint; the "<5K" number is reachable only for the
*declarative authoring surface* (spec + generators), and should be stated that
way. **100% passing IS reachable** and is gated by closing a small, enumerated
set of derivation gaps (§5), NOT by the LOC reduction.

The two goals are **coupled but separable**: LOC reduction is
*re-expression* (byte-identity-preserving, verdict-neutral); 100%-passing is
*gap-closure* (byte-identity-BREAKING, verdict-improving). The rollout below
does re-expression per family; the gap→cluster map (§5) does gap-closure.

---

## 1. PRIMITIVE INVENTORY

The generic substrate lives in three DSL files
(`neural_vm/unified_compiler/`): `isa_semantics_dsl.py` (2464 LOC, the
ISA-level generators), `building_blocks_dsl.py` (1555 LOC, the FFN/attn
factories), `wide_alu_dsl.py` (1658 LOC, the arithmetic generators). For each
primitive: **what it derives** + **what it re-expresses byte-identically TODAY**.

### 1.a ISA-level generators (`isa_semantics_dsl.py`)

| Primitive | Derives | Byte-identical TODAY | Family |
|-----------|---------|----------------------|--------|
| **`decode_band(DecodeSpec)`** (:2395) | opcode-byte → `OP_<NAME>` marker: two-nibble one-hot AND per opcode, per marker-context band (main-at-AX / first-step-PC / all-step-PC / all-step-JSR) + TEMP-clear hygiene | **The ENTIRE L5 decode FFN — all 90 rules IR-identical, golden hash held flag-ON** (`C4_DERIVE_DECODE`, pilot #391). Zero per-opcode code. | DECODE |
| **`cam_lookup(CamLookupSpec)`** (:2082) | addressed load/store: `CamKeyMatch` (query marker → key signature) + `CamValueBand[]` relays + opcode blockers + CONST confirm | **L7 operand-gather head 0** (Q@MARK_AX, K@STACK0_BYTE0, V/O CLEAN_EMBED→ALU). Reference re-expression, golden held. | MEMORY/ALU/CONTROL |
| **`cross_step_carry(CrossStepCarrySpec)`** (:682) | carry a register byte across a step that doesn't re-derive it: `_PREV` band + unconditional carry head (alibi≈0.5) + gated L25 dump | **`BP_SAVE_PREV` (ENT saved-BP) bit-for-bit, flag-on AND flag-off.** Also live: AX byte-1 (`H1_PREV_STEP`), STACK0 byte-0. | EMIT_FRAMING |
| `full_width_byte_emission(FullWidthByteEmissionSpec)` (:1146) | full-width (byte 0..3) register emission into OUTPUT nibbles | AX / register byte emission path | EMIT_FRAMING |
| `consumer_lookahead_gate(ConsumerLookaheadGateSpec)` (:1592) | classify the *next* instruction (reuses the two-nibble decode AND) to gate a STACK0 dump | `#221` consumer-lookahead ops (l5_ops tail), the nibble-rotation adder (`_nibble_rotation_chain_rules`) reuse | EMIT_FRAMING |
| `make_mixed_dim_map_ffn_op` / `_mixed_dim_map` (:560) | generic gated per-cell dim→dim copy block | the DumpBlock / value-route shape (IMM stage-4 value route, 32 rules) | shared |

### 1.b FFN / attention factories (`building_blocks_dsl.py`)

| Primitive | Derives | Byte-identical TODAY |
|-----------|---------|----------------------|
| `multi_way_and_rule` (:288) / `multi_way_or_rules` (:358) | balanced-AND / OR one-hot gate — the atom under every decode + ALU lookup rule | every layer's FFN bakes |
| `step_function_rule` / `one_hot_indicator_rule` / `band_range_check_rules` (:121/178/229) | thresholded indicators, band membership | V1–V7 migration corpus-wide |
| `lookup_table_rules` (:457) | computed `for a: for b: f(a,b)` → one AND-rule per combo | the shared shape under bitwise/shift/add |
| `byte_clear_rules` (:950) | zero a byte band under a gate | AX high-byte no-carry clears (L10/L14) |
| `byte_route_rules` (:1037) | gated per-nibble `src_band → dst_band` copy | IMM value-route, multibyte routing FFN |
| `carry_relay_rules` (:1111) | relay a carry/borrow lane to the next byte's rule | ADD/SUB inter-byte cascade relays |
| `memory_load_attention` / `fetch_byte_attention` / `binary_address_lookup_attention` (:654/773/1251) | the attention side of CAM: content-match address, copy value; 24-bit binary addr-MATCH | L5 fetch heads, L15 LI/LC load heads (24-bit match slots 4–27) |
| `cancel_residual_rule` (:407) | subtract a prior writer's OUTPUT (cancel-then-write) | the PC-override cancel band (JMP/BZ/BNZ/JSR) |
| `opcode_expert_rules` (:1177) | per-opcode gated rule bundle | opcode-gated banks |

### 1.c Arithmetic generators (`wide_alu_dsl.py`)

| Primitive | Derives | Byte-identical TODAY | Ceiling |
|-----------|---------|----------------------|---------|
| `bitwise_rules(and/or/xor)` (:58) | `result = a op b` per nibble pair, 512 rules/op | L10 OR/XOR/AND (574 each) | none — fully derivable |
| `wide_add_rules` (:166) | per-byte `(a+b+cin)%16` + carry cascade via `carry_base+b` | L8 add_lo, L10 carry-prop | none |
| `wide_sub_rules` (:400) | per-byte `(a-b-bin)%16` + borrow cascade | L8 sub, L14 borrow | none |
| `wide_shift_rules` (:624) | per-byte `(n<<k)&0xFF` / `n>>k` | L13 shifts (4096) | inter-byte bit-spill = `carry_rule="bit_spill"` (G5, additive) |
| `wide_mul_rules` (:737) | flat `a*b` lookup | L11 mul-partial, **width ∈ {1,2} ONLY** | **width>2 intractable — `16^(2·width)`** → needs `multi_pass_rules` |
| `wide_div_rules` (:1010) / `wide_div_rules_ge_format` (:1170) | per-nibble / flat 256×256 `a//b`,`a%b` | POC / 8-bit | **width>1 mathematically wrong per-nibble** → `multi_pass_rules` |
| `wide_ge_add_rules` / `wide_ge_sub_rules` (:1371/1520) | GE-workspace add/sub | DIV/MOD long-division inner steps | — |

### 1.d The #389 computed-copy route (byte-copy without enumeration)

`_byte_value_writeback_rules` (`l10_ops.py:5997`) + its **COMPUTED counterpart
`_computed_byte_writeback_route_rules`** (`l10_ops.py:6081`) express the L25
tail's "guarantee OUTPUT byte = v observed in a (lo,hi) source lane" as a
**computed copy** (LO route unit + HI route unit fire together → the observed
byte), not a 256-way per-value lookup. Probe: `tools/_probe_m8_computed_writeback.py`
(identical firing region). This is the pilot for the M8 collapse (§2 EMIT
family) — the ~40 tail families × up-to-256 rules become one route op each.

### 1.e The 3 gap-primitives being built NOW (not yet in the tree)

| Primitive | Closes | Shape | Amortized cost |
|-----------|--------|-------|----------------|
| **`marker_broadcast`** | IMM/multibyte relay (DECODE-pilot G-IMM-RELAY): copy `OP_<NAME>` from the AX-marker to the op's own byte positions | attention head: Q@(IS_BYTE, H1[byte_i]) K@MARK_AX V/O copy a flag band to itself. NOT a CAM (no address key). | ~40 LOC generator, reused by EVERY multi-byte opcode |
| **`multi_pass_rules`** | wide MUL (width>2) + wide DIV/MOD (width>1) — the ONE hard ALU floor | emit a SEQUENCE of FFN passes with explicit inter-pass residual propagation; ~9 passes (MUL schoolbook) / ~24 (DIV long-division). Path 1 in `DSL_W5_MULDIV_LIMIT.md`. | several sessions; unblocks `efficient_alu_*.py` deletion (W6) |
| **cross-lane `byte_copy`** | M8 tail-bank collapse (EMIT G4) + any "copy this 8-bit lane to OUTPUT" | a `byte_copy(src_lane → OUTPUT)` lowering primitive: 1 op replaces each 256-rule enumerated family | additive; verdict-neutral; the #389 route is the byte-identity proof |

**Coverage after the 3 land:** decode (100%), MEMORY (LEA/ADJ/POP + LI/SI/SC/LC
via CAM), CONTROL (PC-mux + CMP + CAM), ALU (bitwise/shift/add/sub/byte-0-mul
today; wide MUL/DIV via `multi_pass_rules`), EMIT (R-FRAME + R-BYTE + byte_copy).
**Every opcode's frame is then a data row over these primitives with zero new
mechanisms.**

---

## 2. PER-FAMILY ROLLOUT PLAN

For each family: the primitives that derive it, the hand-authored rules/correctors
that get DELETED when it flips to sole-derived, and the LOC/FFN-unit count. The
rollout is **ordered by (deletable-LOC × readiness)**; decode is first (proven
byte-identical). "Units" = load-bearing FFN hidden units (fixed count — collapsing
DOES NOT reduce units, per `project_core_loc_reduction_reality`; the LOC win is in
the *authoring source*, the count-neutral collapse of per-band builders into
data + one loop). **Two kinds of win are tracked separately:**
- **AUTHORING-LOC** (source lines of hand builders collapsed to data+loop) —
  verdict-neutral, byte-identity-preserving.
- **DEAD-ROW** (provably-unreachable enumerated rows dropped, e.g. SP-byte2
  256→2) — verdict-neutral, unit-count-REDUCING, flag-gated.

### Rollout order (highest leverage first)

| # | Family | Primitives | Readiness | Deletable AUTHORING-LOC | Deletable DEAD-ROWS / notes |
|---|--------|-----------|-----------|-------------------------|-----------------------------|
| **1** | **DECODE** (all 38 opcodes' identity stage) | `decode_band` | **PROVEN byte-identical** (pilot #391, golden held flag-ON) | ~200 (6 hand band-builders → 1 loop + `DecodeSpec` data ~90) | 0 (90 rules load-bearing) — **flip `C4_DERIVE_DECODE` default-ON** |
| **2** | **EMIT/FRAMING** (shared by ALL opcodes) | R-FRAME (`phase_a_ffn` edge list), R-BYTE (`_head_bake_rules`), `cross_step_carry`, `byte_copy` (#389 route) | R-FRAME+R-BYTE generic today; byte_copy in flight | ~40 tail families collapsed via `_byte_value_writeback_rules` factory | **~40 × up-to-256 rows** = the single biggest DEAD-ROW lever; SP-byte2 256→2 (`C4_SP_BYTE2_CARRY`, 0 violations / 657600 samples) is the proven template |
| **3** | **ALU — derivable regime** (ADD/SUB/OR/XOR/AND/SHL/SHR/byte-0 MUL + carry cascade) | `wide_add/sub_rules`, `bitwise_rules`, `wide_shift_rules`, `lookup_table_rules` | generators byte-identity unit-tested TODAY | ~16k units authored via generators already; source collapse of L8/L10/L13 loops | 0 (units load-bearing); **the majority of the ALU is already computed-lookup** |
| **4** | **MEMORY — ALU half** (LEA, ADJ, POP + all SP±8) | nibble-rotation adder (`_nibble_rotation_chain_rules`), `SpDeltaSpec` | derivable TODAY (`opcode_mapper` = `FFN_COMPOSITE=ADD`) | LEA/ADJ/SP-update bands → `(base_reg, delta_source, gate)` data | G4 SP-marker firmware (`e0→e8`, 0xF8) survives until marker convention spec'd |
| **5** | **CONTROL — PC-mux + CMP + frame** (JMP/JSR/ADJ/EXIT + EQ..GE) | `cancel_residual_rule` (PC-override), `imm*8+2` encoder (2 constants), CMP comparator + 6-row truth table, adder | JMP/JSR/CMP derivable TODAY; branch-target encoder is a 2-constant table | JMP 160 + BZ 81 + BNZ 160 + JSR 80 overrides → 1 `pc_override(gate, target)` shape; CMP 290 → comparator + truth table | `make_branch_override_patch_op` whole-model post-pass DELETED (G9: emit exclusive gates at authoring) |
| **6** | **MEMORY — CAM half** (LI, LC, SI, SC, PSH) | `cam_lookup` (bidirectional), `binary_address_lookup_attention`, `marker_broadcast` | read side re-expressed (G1); needs store/emit direction + 24-bit `CamKeyMatch` variant + provenance anchor (G2) | L15 LI/LC heads, L14 mem-gen, store-provenance CAM → `MemOp` data rows | provenance-anchor SPEC datum (G2) is load-bearing; append-with-recency (G5) |
| **7** | **CONTROL — LEV + branch byte-1** | `cam_lookup` (pop-CAM), full-width PC adder + branch encoder | LEV pops are generic CAM; PC-byte-1 BUILT behind flags | LEV routing 865 units → `frame_delta` ordered list + pops | branch byte-1 default-ON (`C4_JSR_PC_BYTE1`/`C4_PC_BYTE1_CARRY`) blocked by SEPARATE root (task #350), not PC math |
| **8** | **ALU — wide MUL/DIV/MOD** | `multi_pass_rules` (NEW) | **BLOCKED** on the one missing lowering primitive | replaces `FlattenedALUMul` (9-stage) + `FlattenedDivMod` (8×3 loop) once `multi_pass_rules` exists | unblocks `efficient_alu_*.py` deletion (W6) — several sessions |

**Rollout mechanics (per step):** author the family through the generator/spec →
`compare_symbolic_to_lowered_ffn` / `_isa_golden_hash.py` byte-identity gate
(must hold `81557d21` for a verdict-neutral collapse) → for DEAD-ROW cuts, flag
the collapse (default-OFF preserves golden count) and prove 0 violations over the
corpus (the SP-byte2 template) → then flip default-ON as a separate geometry cut.
Cross-op safety: `lint_cross_op_ffn` / `lint_cross_op_attention` for shared bands,
`flag_regression_gate` for campaign-config cross-cluster.

---

## 3. IRREDUCIBLE INFRA FLOOR

The weight-lowering that REMAINS after everything derives — measured (`wc -l`):

| Module | LOC | Role | Reducible? |
|--------|----:|------|-----------|
| `ir.py` | 3119 | FFNRule→W_* lowering, attn-spec→W_qkvo, `compare_symbolic_*` gates | partial — the byte-identity comparators (~430 in `verification.py`) are tooling-adjacent; core lowering ~2k irreducible |
| `full_vm_compiler_dynamic.py` | 3589 | `compile_full_vm_dynamic` entry, auto-widen, band collection, cache keys, alu_mode branch | partial — the `efficient` alu_mode branch (~750 LOC from :2839) DELETES with W6; ~2.5k irreducible |
| `layer_compiler.py` | 2881 | `_dispatch_operation_ir`, per-op lowering dispatch, phase ordering | partial — dispatch is ~1.5k irreducible; the rest is per-kind handlers that thin as ops unify |
| `primitives.py` | 2135 | `DeclarativeAttentionHeadSpec`, `Primitives.byte_value_writes` etc. — the weight-write vocabulary | mostly irreducible (the W_* write primitives) |
| `dim_registry.py` | 2197 | the 78 category-registered slots + `dim_ref` authoring | partial — much is data tables; ~1k irreducible |
| `dim_registry_dynamic.py` | 776 | dynamic dim positions over the built layout | irreducible |
| `dim_resolver.py` | 253 | `DimResolver` over BUILT `layout.dim_positions` | irreducible |
| `dim_allocator.py` | 723 | residual dim first-fit allocator | irreducible |
| `ffn_unit_allocator.py` | 389 | per-layer FFN hidden-unit allocator | irreducible |
| `attention_head_allocator.py` | 543 | per-layer head allocator | irreducible |
| **SUBTOTAL (brief's floor set)** | **~13.6k** | | |
| `predicates.py` | 1631 | scope / dominates_at predicate language | partial |
| `ir_types.py` | 527 | IR dataclasses | irreducible |
| `band_guarantees.py` / `slot_registry.py` / `ssa_dim.py` / misc | ~1.6k | verification + SSA + slot infra | partial |

**Minimal LOC floor:** the truly irreducible weight-lowering (FFNRule→W_*,
attn→W_qkvo, dispatch, the 3 allocators, dim resolution, the write-primitive
vocabulary) is **~6–8k LOC**. The compaction targets that get it there:

1. **`isa_semantics_dsl` vs `wide_alu_dsl` overlap** — both express the "computed
   one-hot lookup rule per operand combination" shape (decode AND == bitwise AND
   == add-lookup AND). Unify onto ONE `lookup_table_rules` core (already in
   `building_blocks_dsl`) → the wide_alu generators become thin `f`-parameterized
   callers. Est. **−1 to −1.5k**.
2. **Dead lowering paths** — the `alu_mode=="efficient"` branch
   (`full_vm_compiler_dynamic.py:2839+`, ~750 LOC) + `efficient_alu_*.py`
   composites DELETE the moment `multi_pass_rules` derives wide MUL/DIV (W6).
   Est. **−1k+**.
3. **`_legacy_redirect.py` (1473 LOC)** — the `legacy_bake` migration bridge. The
   census says the model is 100% live-declarative with ZERO live `_set_layer*`
   writers; the bridge is a shrinking migration scaffold. As Phase 7 completes it
   deletes. Est. **−1k+**.
4. **`primitives.py` / `dim_registry.py` data tables** — much is registered DATA
   (the 78 slots, the write vocab) that moves to a compact registry file, not
   code. Est. **−0.5 to −1k**.

Compacted floor ≈ **6–8k** irreducible.

---

## 4. <5K FEASIBILITY VERDICT

**The arithmetic:** `spec (38 opcode rows, ~data) + generic engine + irreducible
infra`:

| Component | Compacted target | Notes |
|-----------|-----------------:|-------|
| Spec (38 opcode rows over 6 sub-machines) | **~0.4–0.6k** | pure DATA — proven by `DecodeSpec` (90 decode rules from ~90 lines of table) |
| Generic engine (3 DSL files, unified + 3 gap-primitives) | **~3–4k** | 5.7k today → unify decode/alu lookup cores + add the 3 gap generators |
| Irreducible infra (lowering + allocators + dim resolution) | **~6–8k** | the hard wall (§3) |
| **TOTAL core** | **~10–12.5k** | |

**Is <5K literally reachable?**

- **For the DECLARATIVE AUTHORING SURFACE (spec + engine): YES, ~4–5k.** The
  thing a contributor reads and edits to define/change the ISA is the spec
  (~0.5k) + the generators (~3–4k). That surface is under 5K and is the honest
  interpretation of "the correct-by-construction ISA-semantic-spec derivation."
- **For the WHOLE core including weight-lowering infra: NO — the honest floor is
  ~8–12k.** The FFRule→W_* / attn→W_qkvo lowering, the compiler/layout pipeline,
  the 3 allocators, and dim resolution are irreducible substrate that spec-
  derivation does not remove. Claiming <5K for the whole core would require
  deleting the weight-materialization machinery itself, which is not possible —
  the model IS its weights.

**The SPECIFIC compactions to approach the floor** (in leverage order):
1. Unify the decode-AND / bitwise-AND / add-lookup-AND onto one
   `lookup_table_rules` core (−1 to −1.5k engine).
2. Land `multi_pass_rules` → delete `efficient_alu_*.py` + the `alu_mode`
   branch (−1.75k+ infra, W6).
3. Complete Phase 7 → delete `_legacy_redirect.py` (−1.5k infra).
4. Move `primitives.py` / `dim_registry.py` DATA tables to a compact registry
   (−0.5 to −1k).
5. Collapse the ~40 M8 tail families to `byte_copy` route ops (DEAD-ROW cut;
   trims the largest enumerated FFN banks — verdict-neutral, unit-count win).

**Honest headline:** state the target as **"<5K declarative authoring surface
(spec + engine) + ~6–8k irreducible lowering infra ≈ 10–13k core"** — a
**~5×** reduction from 63.8k, where the ISA is 100% a data-derived spec and the
residual is unavoidable weight-materialization substrate. The "<5K" claim is
TRUE and load-bearing for the authoring surface; it is NOT true for the whole
core, and the roadmap should not over-claim it.

---

## 5. 100%-PASSING GATE — the gap→cluster map

**The gate:** `tools/run_1096_canonical.py --criterion full_trace` in the
campaign config (golden 30-token, `spec_k=0`), self-checked on CPU by
`tools/cpu_full_trace.py` (bit-exact vs GPU full_trace, the ONLY tool that sees
the fixed-30-token-slice cumulative desync — `interp_oracle_gate.py`
over-claims). 1096/1096 full_trace pass.

**Current picture (from the failure-map memory notes):** full_trace baseline
~114/1096; ~437 fails concentrate at ONE divergence step per cluster across 6
step-bands; 233 deep-loops unmapped (NO position ceiling — they share per-step
roots). The claim "close the gaps ⟹ 100%" made concrete, cluster by cluster:

| Failure cluster | Divergence | Root (per the family maps) | Which derivation gap FIXES it |
|-----------------|-----------|----------------------------|-------------------------------|
| **var_three / var_multilocal / nested LI** (~88, step-10) | wrong value loaded from a re-stored local | MEMORY **G2** provenance-anchor (store binds addr→value at the AX-marker, not the value row) + **G5** append-with-recency | **CAM store-direction + provenance-anchor SPEC datum** (§2 step 6). The load CAM keys on the marker's ADDR_B0 / copies the marker's AX_CARRY → correct value. Hook-verified generalizes 2→3+ locals. |
| **AX byte-1/2/3 high-nibble** (part of ~452 step-1 dump-truncation) | non-AX-writing steps truncate AX to byte-0; H1 one-hot caps value ≤4 | EMIT **R-BYTE** must emit the correct high byte; the carry band + re-pointed dump | **`cross_step_carry` (AX byte-1) + correct R-BYTE OUTPUT-nibble emit** (§2 step 2). The two-part carry-band + re-pointed dump build (deliberate, not single-agent). |
| **arith byte-0 + if/bool cmp** (~70, step-3) | OUTPUT-band self-reinforcement crushes byte-0; CMP margin | ALU **G1/G2** (byte-0 add/sub + carry cascade) clean-operand emit; CONTROL **G4-margin** | **ALU derivable-regime sole-derive** (§2 step 3) — clean computed-lookup emit removes the OUTPUT-leak; the CMP-margin firmware (`C4_CMP_FLAG_MARGIN_FIX`) is re-justified under the marker/scale convention |
| **wide MUL / DIV / MOD** (mul/div/mod clusters, ~24/21 partial) | per-nibble is wrong / flat table intractable past 8-bit | ALU **G3/G4** — the one HARD FLOOR | **`multi_pass_rules`** (§2 step 8) — the schoolbook / long-division cascade as sequenced FFN passes. THE single highest-leverage DSL investment. |
| **func return / LEV** (~60, step-6) + **func-args/absdiff** (~137, step-11) | LEV 4-register teardown ordering; branch-target PC ≥ 0x100 byte-1 | CONTROL **G5** (LEV frame_delta order + running-SP) + **G3** (pc_width=2 byte-1) | **LEV `frame_delta` ordered CAM pops + full-width PC adder** (§2 step 7). Branch byte-1 is BUILT+VERIFIED behind flags; default-ON blocked by SEPARATE root (task #350 step-0 JSR AX leak), not the PC construction. |
| **expr OUTPUT-crush** (~30, step-5) | rules read OUTPUT×100 in silu gate AND re-write OUTPUT → residue bootstraps | EMIT **G5** ≠STEP_TOKENS drift + the OUTPUT-band self-reinforcement megaroot | **clean R-FRAME + R-BYTE** (§2 step 2): under a clean generic emitter where `NEXT_*` dominates the marker row unconditionally, the stray-byte / 37-token miscount **cannot occur by construction** — the drift class VANISHES |
| **var LI** (~28, step-7) | LI value-load decode | MEMORY CAM load path (24-bit addr-MATCH) | **CAM load sole-derive** (§2 step 6) with the `CamKeyMatch` 24-bit variant |
| **deep loops** (233 unmapped) | > cap-length diverging loop/gcd/rec | same 0xFF / high-nibble family + per-step roots; NO position ceiling (PC byte-stable to pos 10,116) | **per-step roots above + the raised cap.** Deep clusters reduce to the SAME per-step gaps (AX byte-1, PC byte-1, provenance) shared across positions — closing them closes the deep loops too. |

**"Close the gaps ⟹ 100%" is concrete:** each cluster maps to exactly one
enumerated derivation gap from the family maps, and each gap is either (a) a
generator sole-derive that removes a correctness-breaking corrector (var LI,
arith byte-0, LEV), (b) a SPEC datum to add (provenance-anchor G2, pc_width=2
G3), or (c) the one hard lowering primitive (`multi_pass_rules`, wide MUL/DIV).
The EMIT G5 result is the deepest lever: a *clean* R-FRAME+R-BYTE lowering makes
the entire ≠STEP_TOKENS framing-drift class (the mega-root gating ~760+
AX-wrong-AND-PC-wrong fails) impossible by construction. There is **no cluster
that lacks a mapped gap** — 100% is gated by closing this finite, enumerated set,
not by any architectural wall (the deep-loop / position-ceiling "walls" were
tested-false).

---

## Reproduce / gate

```
CUDA_VISIBLE_DEVICES="" python tools/_isa_golden_hash.py            # 81557d21 (this doc: unchanged)
CUDA_VISIBLE_DEVICES="" C4_DERIVE_DECODE=1 python tools/_isa_golden_hash.py   # 81557d21 (decode derived)
python tools/run_1096_canonical.py --criterion full_trace          # the 100% gate
tools/cpu_full_trace.py --ids <list>                               # bit-exact CPU self-check
```

Sources consolidated: `docs/semantic_spec_{DECODE,MEMORY,ALU,CONTROL,EMIT_FRAMING}.md`,
`docs/DERIVE_DECODE_PILOT_2026_07_03.md`, `docs/DSL_W5_MULDIV_LIMIT.md`, and the
failure-map / core-LOC-reality memory notes. LOC measured by `wc -l` at golden
`81557d21`; primitive line numbers from the live DSL files.
