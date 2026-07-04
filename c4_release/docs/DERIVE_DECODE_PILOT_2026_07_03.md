# Decisive core→20k pilot — ONE opcode 100%-derivable (task #391)

**Status:** LANDED + FLIPPED TO SOLE PATH (increment 2: `decode_band` is now
the ONLY L5 decode path; the `C4_DERIVE_DECODE` flag AND the hand-authored
decode builders are DELETED — the first real ops-core LOC deletion via the
100%-derivable architecture).
**Flag:** `C4_DERIVE_DECODE` — **REMOVED** (was default OFF; derivation is now
unconditional). No env toggle; `_opcode_decode_ffn_rules` always derives.
**Golden gate:** `tools/_isa_golden_hash.py` ==
`81557d21422f3eada0a87c677b00dced41cc26c3ee3bfb094c5eeb71c9b4d3cb` (`81557d21`)
— UNCHANGED (the derivation == the deleted hand-authored weights, so the
sole-path build is byte-identical to golden). The `~250` LOC of hand-authored
builders in `l5_ops.py` (`_opcode_decode_main_rules`,
`_opcode_decode_first_step_rules`, `_opcode_decode_temp_clear_rules`,
`_opcode_decode_all_step_pc_rules`, `_opcode_decode_all_step_jsr_rules`,
`_opcode_decode_jsr_temp0_blank_rule`) are deleted; only the
`_derived_decode_spec` + `decode_band` call remain. This proves the
derive→sole-path→delete rollout template for every opcode family.

## The pilot claim + what was proven

The decisive claim of the core→20k program: an opcode's low-level weights are
100% DERIVABLE from a declarative ISA-semantic spec via a GENERIC lowering,
with ZERO hand-authored per-opcode rules — every corrector is a derivation-gap
DEFECT, not an exception.

**Proven this increment:** the entire **L5 opcode-decode FFN** (the opcode's
IDENTITY stage — mandatory first stage of every opcode's 30-token frame) is
100% derived by a new generic engine, `decode_band(DecodeSpec)`, from the ISA
`Opcode` table + a small band-context table, with **zero hand-authored
per-opcode rules**. The derived weights are **byte-for-byte identical** to the
hand-authored path (golden hash held flag-ON).

Piloted opcode: **IMM** (`AX = imm`) — the tightest end-to-end path (no CAM
address-match, no ALU carry cascade). Its full-frame map + honest gap-list is
in §4.

## 1. The spec schema (what a generic engine is TOLD)

`neural_vm/unified_compiler/isa_semantics_dsl.py`, new types:

- **`DecodeSpec`** — `opcode_table: ((byte, "OP_<NAME>"), ...)` (the SOLE
  source of per-row data) + an ordered `bands` tuple + `opcode_flag_ref`
  (the `(opcode_flag, NAME)` role resolver, injected so the DSL stays free of
  the `dim_registry` import).
- **`DecodeContext`** — one marker-context band (main-at-AX / first-step-PC /
  all-step-PC / all-step-JSR). Carries the per-context CONSTANTS
  (`threshold`, `gate`, `extra_conditions`, `write_scale`), the opcode SUBSET
  (`opcodes`, `None` => full table), the ONE per-(opcode,context)
  `write_override` cell (JSR@PC → `TEMP+0`), and an optional `enabled` flag
  predicate (Root-B all-step-JSR is `C4_NESTED_JSR_PC_FIX`-gated).
- **`ScratchClearBand`** — the TEMP[1..31]-clear hygiene band (NOT decode; a
  separate opcode-independent primitive, docs G4).
- **`BlankUnit`** — a reserved-blank hidden unit (unit-52, docs G5 layout
  constraint).

## 2. The generic engine (what was added to isa_semantics_dsl)

`decode_band(spec) -> DecodeBundle`. Its `rules_builder()`:

- `_decode_context_rules(ctx, spec)` — the ONLY place opcode rows are
  produced. Loops over the context's opcode subset; for each opcode value
  computes `lo = byte & 0xF` / `hi = (byte >> 4) & 0xF`, ANDs the two nibble
  one-hots with the context's constant `extra_conditions`, and writes the
  opcode's marker (or the `write_override`). **ZERO per-opcode branch.**
- `_scratch_clear_rules(band)` — the TEMP-clear hygiene loop.
- a no-op `FFNRule` for each `BlankUnit`.

The engine also derives the FFN Operation's `reads`/`writes` dep-graph dim
sets structurally from the bands.

## 3. The proof (STEP 4 — 100% derivation + verdict-match)

- **Rule-level:** all **90/90** rules IR-identical between the derived path
  (`_derived_opcode_decode_ffn_rules`) and the hand path
  (`_opcode_decode_ffn_rules`) — names, conditions, thresholds, gates, gate
  weights/bias, and writes all match.
- **Weight-level (the decisive gate):**
  - flag-OFF `tools/_isa_golden_hash.py` = `81557d21` (default path untouched).
  - flag-ON `C4_DERIVE_DECODE=1 tools/_isa_golden_hash.py` = `81557d21`
    (generic-engine-derived decode weights byte-identical to hand-authored).
- **Verdict-match (GATE 2):** byte-identical `state_dict` ⇒ identical model ⇒
  identical full_trace verdict on EVERY program, ZERO regressions, by
  construction. (The hash over the full `state_dict` is a strictly stronger
  equivalence than a per-program run.)

**DECODE derivation = 100%.** Every field of every decode rule except
`(byte, NAME)` is a per-context constant; `(byte, NAME)` comes straight from
`Opcode`. The generic engine reproduces the entire family — main-at-AX (34) +
first-step-PC (18) + reserved blank (1) + TEMP-clear (31) + all-step-PC (5) +
flag-gated all-step-JSR (1) — with zero opcode-specific code.

## 4. IMM full-frame map + the honest gap-list

IMM (`AX = imm`) across its 30-token frame, per stage classified
**DERIVED-NOW** / **SPEC gap** / **LOWERING gap** / **SHARED substrate**
(shared = opcode-agnostic machinery every opcode reuses; NOT an IMM defect).

| # | stage | mechanism | opcode-specific? | status |
|---|-------|-----------|------------------|--------|
| 1 | **decode → OP_IMM** | L5 main-at-AX rule + first-step-PC rule (2 nibble-AND rules) | yes (its identity) | **DERIVED-NOW** — comes out of `decode_band`, byte-identical |
| 2 | fetch immediate byte | L5 fetch heads 0/3 copy CODE[PC+1] → `FETCH_LO/HI` / `AX_CARRY` | no (address/fetch machinery) | SHARED substrate (docs DECODE G6 — a separate "instruction fetch" spec family) |
| 3 | relay OP_IMM → byte positions | L8 head 4 (`layer8_op_imm_relay`, `opcodes={"OP_IMM"}`, ALiBi 0.5) | yes | **LOWERING gap (small)** — a marker-broadcast attention head: Q@(IS_BYTE,H1[AX_I]) K@MARK_AX V/O copy OP_IMM. NOT a `cam_lookup` (no address key; it is a "copy-a-flag-to-my-own-byte-positions" relay). Needs a `marker_broadcast` head primitive (new, ~1 generator). |
| 4 | value route imm → OUTPUT | L8 multibyte routing FFN, **32 rules** gated `OP_IMM` — per-nibble `AX_CARRY_LO/HI[k] → OUTPUT_LO/HI[k]` at byte positions | yes | **DERIVED-NOW-shaped** — a clean gated per-cell copy block, exactly the shape `DumpBlock` / a `value_route` primitive already expresses (`gate={AX_CARRY_*}+k`, cond `(IS_BYTE, H1[AX_I], OP_IMM, MARK_AX:-4)`, write `OUTPUT_*+k`). Not yet wired to a generator, but no new mechanism. |
| 5 | emit byte → token | canonical OUTPUT_LO/HI → LM-head byte path | no | SHARED substrate |
| 6 | ENT-frame IMM discriminators | L8 `ent_lo`/`ent_borrow` carry an `("OP_IMM", -1000)` NOT-blocker so an ENT step (OP_IMM=0) is byte-identical | no (an ENT-frame defect, not IMM) | FIRMWARE (belongs to ENT's frame; a clean per-step OP_IMM one-hot — which decode already produces — is the justification) |

**IMM opcode-specific derivation today:**
- Stage 1 (decode): **100% DERIVED**, byte-identical, landed.
- Stage 4 (value route, 32 rules): shape is 100% expressible by existing DSL
  primitives (gated per-cell copy); a `value_route` generator lowering is
  additive, no new mechanism → **derivable, not-yet-wired**.
- Stage 3 (OP_IMM relay head): the ONE genuine LOWERING gap — a
  marker-broadcast (copy-my-flag-to-my-byte-positions) attention head has no
  generator yet. It is small and generic (every multi-byte opcode reuses the
  identical relay), not IMM-specific behavior.

### Gap classification (SPEC vs LOWERING)

- **G-DECODE (closed):** decode is a pure lookup — `decode_band` derives it
  100%, zero gaps. [was the strongest-case member; now PROVEN generic]
- **G-IMM-RELAY [LOWERING]:** the OP_IMM→byte-positions relay head needs a
  `marker_broadcast` head generator (Q on the consuming byte's marker-bank
  slot, K on the source marker, V/O copy a flag band to itself). ~40 LOC
  amortized generator; reused by every multi-byte opcode. NOT a spec gap — the
  spec datum is just "broadcast OP_<NAME> to this op's byte positions."
- **G-IMM-ROUTE [LOWERING, near-closed]:** the 32-rule value route is a gated
  per-cell copy = the existing `DumpBlock` / a thin `value_route` shape. Wiring
  it through a generator is additive; the weights are already
  primitive-expressible.
- **No SPEC-expressiveness gap for IMM.** IMM carries no per-opcode arithmetic
  or address behavior beyond "route the fetched immediate into AX/OUTPUT under
  OP_IMM" — all of it is data + the three generic sub-machines (decode lookup,
  marker-broadcast relay, gated value-route copy).

## 5. LOC comparison

| item | hand-authored | spec + amortized engine |
|------|---------------|--------------------------|
| L5 decode rule builders (`_opcode_decode_main/first_step/temp_clear/all_step_pc/all_step_jsr/blank`) | ~200 LOC of per-band builders (each re-declares the loop + name + conds) | `decode_band` engine ~120 LOC (amortized across ALL opcodes + reused by any future decode block) + `DecodeSpec` DATA ~90 LOC (pure table) |
| per-opcode decode LOC | 1 table row / opcode (34 rows × 4 contexts) | 1 table row / opcode (SAME data, but ONE loop lowers all contexts — no per-band re-declaration) |

The engine is **count-neutral on the decode rules** (90 rules either way — they
are load-bearing weights, per the core-LOC-reality memory note) but
**collapses the six hand-written band builders into one generic loop + pure
data**, and the engine amortizes across every future decode block. The payoff
is not raw −LOC; it is that decode is now **provably a derivation, not a set of
hand-authored exceptions**.

## 6. GO / NO-GO on 100%-derivable

- **DECODE stage: GO — 100% derivable, PROVEN byte-identical.** This is the
  decisive, honest result: the opcode-identity stage of every opcode is a pure
  generic lowering with zero per-opcode rules.
- **IMM full frame: GO on the two opcode-specific stages (decode DERIVED,
  value-route primitive-expressible); ONE small LOWERING gap** (the
  marker-broadcast relay head generator, G-IMM-RELAY). No SPEC-expressiveness
  gap. The remaining stages (fetch, emit) are SHARED opcode-agnostic substrate
  — correctly NOT IMM's to derive.

**Honest bottom line:** the pilot proves the generic-lowering thesis on the
decode family end-to-end (100%, byte-identical), and reduces the rest of IMM to
**two additive generator wirings with zero new mechanisms and zero
spec-expressiveness gaps**. Nothing in IMM's frame is a genuine per-opcode
exception; every corrector maps to a generic sub-machine.

## Reproduce

```
CUDA_VISIBLE_DEVICES="" python tools/_isa_golden_hash.py                 # 81557d21 (flag-OFF)
CUDA_VISIBLE_DEVICES="" C4_DERIVE_DECODE=1 python tools/_isa_golden_hash.py  # 81557d21 (derived)
```

Rule-level diff (90/90 identical):
`l5_ops._opcode_decode_ffn_rules(100.0)` vs
`l5_ops._derived_opcode_decode_ffn_rules(100.0)`.
