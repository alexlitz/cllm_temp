# DERIVE MEMORY — the KV binary-address CAM derivation (2026-07-09)

**Flag:** `C4_DERIVE_MEMORY` (DEFAULT OFF). **Branch:** `derive-memory`
(off main `59a9de19`). **Golden flag-OFF:** `e50521f3...` (unchanged, verified).
**Flag-ON build:** `8b01fa30...` (`C4_CAMPAIGN=1 C4_DERIVE_MEMORY=1`).

MISSION: derive the MEMORY family (LI 9, LC 10, SI 11, SC 12, PSH 13) from
BLOG_SPEC §408-412 — memory as a KV-attention binary-address CAM — replacing
hand-authored per-op memory head enumeration + magic constants with the ONE
`cam_binary_address_match` spec mechanism, and check whether the clean CAM fixes
the SI/SC store-provenance two-root wall + the LI value-load.

---

## 0. The BLOG_SPEC §408-412 mechanism (verbatim intent)

> "The program memory works by having a store instruction which attends to its
> registers to get the address and value it is storing, it represents the
> address in binary and sets the key to +scale for ones, -scale for zeros …
> retrieve this position in memory by attending with a query identical to the
> key … positional bias via ALiBi to heavily upweight more recent items, such
> that it strictly prioritized exact address match and subject to that having
> priority to the most recently written store …"

That is ONE bidirectional primitive:

- **STORE (SI/SC/PSH):** emit a MEM/marker row whose KEY encodes the address in
  binary (`+scale` per 1-bit, `-scale` per 0-bit) and carries the value.
- **LOAD (LI/LC):** query with a key identical to the store key → the softmax
  peaks on the exact-address row; ALiBi recency resolves re-stores to the latest.

---

## 1. STATE OF THE DERIVATION — it is SUBSTANTIALLY ALREADY DONE (default path)

The `cam_binary_address_match` primitive and its data types
(`CamBinaryAddressBlock` = the per-bit `±scale` comparator, `CamDiscriminatorSlot`
= the heterogeneous opcode/marker/lifecycle gates as DATA, `CamValueBand` = the
value relay, `CamStoreRoute` = the store/emit direction) live in
`neural_vm/unified_compiler/isa_semantics_dsl.py`. The MEMORY-family heads are
**already re-expressed through it byte-identically on the DEFAULT (flag-OFF)
path** — this is the completed work of tasks #396/#403/#405/#406/#408/#409/#412:

| head | file | derived via | status |
|------|------|-------------|--------|
| L7 `operand_gather` head 0 (operand-A gather) | `l7_ops.py:442` | `cam_lookup` | derived + hand-form deleted |
| L8 multibyte-fetch / mem-to-alu / IMM-relay | `l8_ops.py:1756,2578,2858` | `cam_binary_address_match` / `marker_broadcast` | derived |
| L13 `mem_addr_gather` 0-2, relay 3/6, store 4/5 | `l13_ops.py:461,547,628,791,859,1111,1282` | `cam_lookup` + `CamStoreRoute` | derived |
| L14 `mem_generation` store heads | `l14_ops.py:1483` | `cam_binary_address_match(direction=store)` | derived |
| **L15 `li_lc_stack0_h{0..3}` (the LI/LC value load)** | `l15_ops.py:1205` | `cam_binary_address_match` (`CamBinaryAddressBlock` 24-bit comparator, slots 4-27) | derived |
| L15 `si_store_addr_cam` head 16 (store-provenance) | `l15_ops.py:2977` | `cam_binary_address_match` (empty addr block, per-nibble `CamDiscriminatorSlot`) | derived |

**Proof:** `tests/test_isa_semantics_dsl.py` (21 CAM/store/mem tests pass,
incl. `test_l15_li_lc_load_derived_is_byte_identical_to_handbuilt` = 4/4 heads ==
the legacy writes) + the whole-model golden hash unchanged (`e50521f3`).

So the byte-identical *lowering* of the MEMORY family from the spec's ONE CAM
primitive is complete and needs no flag. This corrects the 2026-07-04
`semantic_spec_MEMORY.md` §3 audit (written before #396 landed), which reported
the L15 LI/LC heads as "NOT byte-identically cam_lookup-expressible": the
`CamKeyMatch.query_extra`/`key_extra` extension + the `CamBinaryAddressBlock`
24-bit comparator closed exactly that gap.

## 2. WHAT `C4_DERIVE_MEMORY` ADDS — the derived-CAM MEMORY-FIX path

A byte-identical re-expression of the *hand-authored value-row load* cannot fix
the SI/LI walls, because the hand-authored load read the wrong rows (the store
VALUE rows, provenance-blind on BOTH axes — memory note
`project_si_store_provenance_two_root_wall`). A CORRECT derivation of §408-412
implies a *different* wiring: key the LI-query on the target address and read the
value from where the clean `(address,value)` binding actually lives — the store
**AX-MARKER** row. Those two derived-CAM fix heads already exist as clean
`cam_binary_address_match` heads with ZERO per-op magic-number correctors:

1. **L15 head-16 `layer15_si_store_addr_cam`** (`si_store_addr_enabled`,
   `l15_ops.py:2977`) — the store-provenance CAM. Q keys `AX_CARRY` (LI target
   addr) → K keys the store marker's `ADDR_B0` → V copies the marker's
   `AX_CARRY` (the clean stored value) → O writes OUTPUT. Recency ALiBi slope
   0.05 resolves re-stores (var_update). Fixes the `var_mul`/multilocal
   `LI a → b's value` wall. Every gate is a `CamDiscriminatorSlot` DATA row
   (CONST softmax1 sink, MARK_AX/OP_LI firing gates, per-nibble address match,
   OP_PSH/OP_LI/self vetoes, MARK_AX-require, zero-address firing veto), NOT
   hand-tuned FFN correctors.
2. **L15 head-0 store-row veto** (`var_three_li_enabled`) — extends head-0's
   existing `non_load_suppression` (OP_JSR/ENT/LEA/IMM) to OP_SI/OP_SC so a
   store row carrying a stray `OP_LI_RELAY` (var_three `SI b`) does not fire the
   load. Additive slot-0 Q writes as DATA.

`C4_DERIVE_MEMORY` is the SINGLE umbrella entry point that OR-floors both
predicates ON (mirroring `campaign_enabled()`), so a caller enables "the derived
clean-CAM memory-fix family" with one flag. Explicit per-flag `=0` still opts
out. These heads read the 30-token campaign MEM-from-SP provenance signals, so
the flag is meaningful only with `C4_CAMPAIGN=1`; at the golden 35-token frame
flag-ON is byte-identical to flag-OFF.

### Magic-constant / LOC posture

- The MEMORY heads are authored as the compact CAM spec DATA above — the
  24-bit binary comparator is ONE `CamBinaryAddressBlock`, the per-op gates are
  `CamDiscriminatorSlot` literals, the value relay is `CamValueBand`. The
  per-cell hand-authored Q/K/V/O directive construction for the L7/L8/L13/L14/L15
  memory heads is DELETED (tasks #396-412).
- `C4_DERIVE_MEMORY` itself adds **ZERO new weights or magic constants** — it is
  a pure flag consolidation: `+1` umbrella predicate in `shared.py`, an OR-in on
  the two existing fix predicates, and one cache-key entry in each of the two
  snapshots. It surfaces the already-built derived-CAM memory-fix path under one
  BLOG_SPEC-derived name.

---

## 3. VERIFICATION

### Gates (this session, this worktree `59a9de19` + `derive-memory`)

- **Golden flag-OFF byte-identity:** `tools/_isa_golden_hash.py` (flag unset) ==
  `e50521f32b0ed952...` — UNCHANGED. The umbrella defaults OFF and does not
  perturb the golden build. ✅
- **Umbrella equivalence:** `C4_CAMPAIGN=1 C4_DERIVE_MEMORY=1` state_dict ==
  `C4_CAMPAIGN=1 C4_SI_STORE_ADDR=1 C4_VAR_THREE_LI=1` state_dict = `8b01fa30...`
  (byte-identical) — the umbrella floors exactly the two fix flags, no more, no
  less. ✅ Flag-ON build compiles cleanly (L15 16→17 heads, d_model auto-widens
  1221, 42150 FFN units, no dead units). ✅
- **Flag-predicate truth table** (`derive_memory` / `si_store_addr` /
  `var_three_li`): bare `(F,F,F)`; `DERIVE_MEMORY=1` `(T,T,T)`;
  `DERIVE_MEMORY=1 SI_STORE_ADDR=0` `(T,F,T)`; `DERIVE_MEMORY=1 VAR_THREE_LI=0`
  `(T,T,F)` — explicit-wins semantics correct. ✅
- **DSL byte-identity tests:** `test_isa_semantics_dsl.py` 21/21 pass (the LI/LC
  load derived == handbuilt + store CAM). ✅
- **Baseline flag-OFF full_trace:** var_simple id250 = **PASS** (`cpu_full_trace
  --spec-k 0`, campaign-default). ✅

### Verdict — does the clean CAM deliver the stored value on LI?

The head-16 store-provenance CAM and the head-0 store-row veto that
`C4_DERIVE_MEMORY` activates are the SAME weights that were AR-verified correct
in prior sessions (memory note `project_si_store_provenance_two_root_wall`,
agents ab4c2337 / a4d5daf9, `cpu_full_trace --spec-k 0`):

- **var_mul id275** (`a=23;b=47;return a*b`): flag-OFF diverges at step 11
  (`LI a` → returns b's 47); flag-ON the divergence MOVES to step 15 — i.e.
  `LI a` now returns **23** (steps 11-14 pass, step14 ax=47). The remaining
  step-15 divergence is the SEPARATE 16-bit MUL wall (`a*b`), out of MEMORY
  scope.
- **var_three id304** (2-local prologue reaching the LI loads): flag-OFF
  `LI a` → 19 (=c, WRONG); flag-ON advances to step 19 with `LI a` = **14**
  (CORRECT). The CAM generalizes 2→3 locals with NO head-code change (the LO+HI
  nibble-match slots discriminate all 3 markers). `var_three_li` (the head-0
  veto) fixes the distinct `SI b` stray-relay desync that gates the earlier
  var_three ids.
- **var_simple id250/251 + func_identity id550:** HOLD — head-16 fires at
  exactly the one genuine LI-load row per program and delivers the right value
  (990, 70); no regression.

So the clean derived CAM DOES deliver the stored value on LI, including the
multi-local var_mul / var_three cases — it fixes the store-provenance two-root
wall at the source (address-keyed query → marker CAM → clean marker value),
exactly as the mission's blueprint predicted.

**AUTHORITATIVE re-verification on this worktree** (`cpu_full_trace --spec-k 0
C4_CAMPAIGN=1 C4_DERIVE_MEMORY=1`): var_mul id275 — see §4 (in-flight under
machine contention; the build + flag wiring are proven above).

---

## 4. HONEST LIMITS

- **`C4_DERIVE_MEMORY` is a flag CONSOLIDATION, not a new derivation.** The
  substantive DSL derivation of the MEMORY heads from `cam_binary_address_match`
  was already landed (#396-412); the store-provenance + var_three fix heads were
  already built (#361/#397/#420, `C4_SI_STORE_ADDR`/`C4_VAR_THREE_LI`). This
  session added the single BLOG_SPEC-derived umbrella name + proved it is
  byte-identical to the explicit combination and byte-identical-to-golden OFF.
- **It does not make any currently-failing program FULLY pass.** var_mul is
  gated downstream by the 16-bit MUL arithmetic wall; var_three by the SI-b
  OUTPUT-relay crush + the ADD-accumulate root; func_max/func-arg by the LEV
  return-PC restore. Those are ALU / framing / control roots, NOT store
  provenance — the memory value delivery is fixed, the arithmetic/control that
  consumes it is a separate scope.
- **The call-site-arg LI path (func_max/min/add — args passed by PSH, no callee
  SI store) is NOT covered here.** That needs the ORDINAL CAM (`C4_PSH_ORDINAL`
  + `C4_SI_ARG_CAM`, memory note Phase 1a/1b) which does not exist on this base
  commit (`59a9de19`); it lives on a sibling worktree pending merge. When those
  land, they are the natural additional predicates for this umbrella to floor.
- **The SP marker firmware (G4) and the append-with-recency memory model (G5)**
  from `semantic_spec_MEMORY.md` remain outside the CAM derivation (SP-adder
  edge-case patches + a transformer-embedding property, respectively).

## 5. FILES

- `neural_vm/unified_compiler/ops/shared.py` — `derive_memory_enabled()` (new
  umbrella predicate) + OR-in on `si_store_addr_enabled()` /
  `var_three_li_enabled()`.
- `neural_vm/unified_compiler/full_vm_compiler_dynamic.py` — `C4_DERIVE_MEMORY`
  in BOTH cache-key snapshots.
- `docs/FLAG_REGISTRY.md` — registry row.
- Derivation primitive: `neural_vm/unified_compiler/isa_semantics_dsl.py`
  (`cam_binary_address_match`, `CamBinaryAddressBlock`, `CamDiscriminatorSlot`,
  `CamValueBand`, `CamStoreRoute`). Derived heads:
  `l7_ops.py`/`l8_ops.py`/`l13_ops.py`/`l14_ops.py`/`l15_ops.py` (§1 table).
