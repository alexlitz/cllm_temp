# Semantic Spec — the EMIT / FRAMING family

**Status:** DESIGN / READ-only inventory (100%-derivable architecture prep,
sibling to `semantic_spec_DECODE.md`, `semantic_spec_MEMORY.md`,
`semantic_spec_CONTROL.md`, `semantic_spec_ALU.md`; feeds the generic-lowering
pilot #391).
**Golden gate:** no weight change. `tools/_isa_golden_hash.py` ==
`81557d21422f3eada0a87c677b00dced41cc26c3ee3bfb094c5eeb71c9b4d3cb` (`81557d21`)
before and after this doc — this file only READS ops and WRITES documentation.

This document maps the machinery **shared by ALL opcodes**: how each execution
step emits exactly `Token.STEP_TOKENS` tokens (30 campaign / 35 golden) encoding
the post-instruction register state, and how each register byte lands on the LM
head via the canonical `OUTPUT_LO/HI` nibble one-hots. Because every opcode ends
by emitting the same N-token register frame, EMIT/FRAMING is the single biggest
generic-derivation win: fixing it once fixes the per-step scaffolding for the
entire ISA.

**TL;DR — emission is TWO generic primitives, not per-opcode logic.**

1. **The frame is a marker state-machine.** One 6-entry (30-token) transition
   table in **L0** (`phase_a_ffn`, `neural_vm/unified_compiler/ops/l0_ops.py`)
   drives `NEXT_PC → NEXT_AX → NEXT_SP → NEXT_BP → NEXT_MEM → NEXT_SE → (loop)`.
   It is opcode-AGNOSTIC: the SAME chain runs for LEA, ADD, JSR, EXIT — the
   payload bytes between markers differ, the frame does not.
2. **The byte is a nibble lookup.** The LM head reads EVERY value byte `b` from
   `head.weight[b, OUTPUT_LO+(b&0xF)] + head.weight[b, OUTPUT_HI+(b>>4)]`
   (`model_ops._head_bake_rules`). Each opcode's job is just to LAND its result
   in `OUTPUT_LO/HI` before the byte row; the decode-to-token is one shared rule
   over 256 bytes.

Everything else in the family — the byte-1/2/3 register "dump", the H-band
carriers, the cross-step SP-byte2 / AX-byte1 / PC-byte1 / BP-byte1 carry bands,
the L25 tail correction bank, the M8 enumerated byte-writeback banks — is
**LOWERING** noise around those two primitives: repair machinery for the cases
where an opcode fails to land the right nibble in `OUTPUT` at the right row. The
GAP-LIST (§3) classifies each as SPEC (a datum the generic model must carry) vs
LOWERING (an artifact a clean generic emitter would not produce).

---

## 0. Where emission lives (block / layer)

The step frame is produced across the whole forward pass, but the *framing*
machinery (as opposed to per-opcode value production) concentrates in three
places:

| Stage | Logical layer / file | Role |
|-------|---------------------|------|
| **Frame state-machine** | **L0** `phase_a_ffn` (`l0_ops.py:100`) | reads the H0..H4 marker threshold heads, emits the `NEXT_*` marker-schedule flags that sequence the N-token frame. |
| **Value → OUTPUT** | per-opcode ops (L5..L23) | each opcode lands its register result in the canonical `OUTPUT_LO/HI` nibble one-hots (and the `*_THIS_STEP` variants). |
| **Cross-step carry** | L9/L11 heads + L25 dump (`isa_semantics_dsl.cross_step_carry`) | re-emit a register byte on a step that does NOT freshly write it (the "carry register when unwritten" primitive). |
| **Tail correction / emit** | **L25 tail** (`l10_ops.py` `_tail_bit32_result_correction_rules`, 2059 units + post-ops) | last-writer OUTPUT correction bank: the M8 enumerated byte-writeback banks live here. |
| **Byte → token** | **model head bake** (`model_ops._head_bake_rules`) | LM head reads `OUTPUT_LO/HI` nibbles → argmax the byte token; `NEXT_*` flags → argmax the marker token. |

Physical-block note: L0 is physical block 0; L25-tail is the last logical layer
before the head (see `docs/PROBE_GROUNDTRUTH_2026_06_10.md` for the block↔layer
map). The 17→59 physical-block expansion does not touch the frame chain.

---

## 1. INVENTORY

### 1.a The N-token frame layout (`neural_vm/token_layout.py`)

`neural_vm/token_layout.py` is the AUTHORITATIVE positional map (parametrized on
`Token.STEP_TOKENS`; `constants.py:71` resolves the env flag). Two layouts:

**35-token (golden flag-OFF, `C4_NO_STACK0_EMIT=0`):**

| pos | field | pos | field |
|----:|-------|----:|-------|
| 0 | REG_PC marker | 20 | STACK0 marker |
| 1–4 | PC bytes b0..b3 (LE) | 21–24 | STACK0 bytes b0..b3 |
| 5 | REG_AX marker | 25 | MEM marker |
| 6–9 | AX bytes b0..b3 | 26–29 | MEM addr b0..b3 |
| 10 | REG_SP marker | 30–33 | MEM val b0..b3 |
| 11–14 | SP bytes b0..b3 | 34 | STEP_END / HALT |
| 15 | REG_BP marker | | |
| 16–19 | BP bytes b0..b3 | | |

**30-token (campaign DEFAULT, `C4_NO_STACK0_EMIT=1`):** the STACK0 block
(marker + 4 bytes) is DROPPED. Register markers stay at `{0,5,10,15}`; MEM
marker 25→20, MEM addr 26..29→21..24, MEM val 30..33→25..28, STEP_END 34→29.
`POS_STACK0_*` are exported as `None`. `POS_END_MARKER = STEP_TOKENS - 1`.

The frame is thus **4 register blocks (PC/AX/SP/BP)** + **1 MEM block**
(addr+val) + **1 STEP_END**, each register block = `[marker, b0, b1, b2, b3]`
(little-endian). This is the semantic-spec DATA of §2.a.

### 1.b Frame state-machine — L0 `phase_a_ffn` (`l0_ops.py:100`, ~60 LOC)

The frame is generated by a **marker-transition FFN**: 6 units (30-token) / 7
units (35-token), one per marker→next-marker edge. Reads the L0 threshold-head
outputs `H0..H4` (per-marker-type one-hots produced by `layer0_ff` attention on
distance-from-marker), writes the `NEXT_*` schedule flag the LM head consumes.

The transition table (30-token, `_phase_a_ffn_rules`, l0_ops.py:122):

```
(H0+SE)                 -> NEXT_PC       # STEP_END seen -> start next frame at PC
(H1+PC)  AND NOT (H0+PC)-> NEXT_AX       # 1 past PC marker -> schedule AX
(H1+AX)  AND NOT (H0+AX)-> NEXT_SP
(H1+SP)  AND NOT (H0+SP)-> NEXT_BP
(H1+BP)  AND NOT (H0+BP)-> NEXT_MEM      # 30-token: SKIP STACK0, go straight to MEM
(H3+MEM) AND NOT (H2+MEM)-> NEXT_SE      # distance-from-MEM (addr+val done) -> STEP_END
```

35-token adds one row: `(H1+BP) -> NEXT_STACK0` and `(H4+BP) -> NEXT_MEM` (the
STACK0 block is 5 tokens so MEM is fired at distance-9 from BP). The 30-token
build drops the STACK0 row → the model never emits `Token.STACK0`. This is the
ENTIRE structural difference between the two layouts (one gated by
`_no_stack0_emit()`, l0_ops.py:14).

**This is opcode-agnostic.** No `OP_<NAME>` appears in the chain — the frame runs
identically for every instruction. LOC: the whole rule builder is ~50 lines
(l0_ops.py:100-160); the op factory + anchors ~90 more.

### 1.c Register-emission machinery — the canonical OUTPUT path

**The byte→token rule (`model_ops._head_bake_rules`, model_ops.py:2104, one
loop over 256 bytes).** The LM head decodes EVERY value-byte token from the
canonical nibble one-hots:

```
for b in 0..255:
    head.weight[b, OUTPUT_LO+(b & 0xF)] = 5.0
    head.weight[b, OUTPUT_HI+(b >> 4)] = 5.0
    head.bias[b] = -5.0          # (byte 0 overridden to -4.0)
    head.weight[b, NEXT_*]      += -80.0   # a byte token can NEVER win a marker row
```

Marker tokens win their rows the dual way: `head.weight[REG_PC, NEXT_PC]=20`,
`bias=-10` (and the same for AX/SP/BP/MEM/STEP_END/HALT). So the argmax at a
given frame position is:

* **marker row** (`NEXT_<REG>` fired by L0): the `REG_<REG>` token wins
  (`+20` vs every byte's `-80`).
* **value-byte row** (no `NEXT_*` fired): the byte whose `(lo,hi)` nibbles match
  the two lit `OUTPUT_LO/HI` one-hots wins (`+10` vs `-5` bias).

So a register's value bytes are emitted purely by whatever op wrote its
`OUTPUT_LO/HI` nibbles for that row. **The value-emission rule is one shared
256-row lookup; the per-opcode work is only "put the right nibble in OUTPUT."**

**Byte-1/2/3 "dump" + H-band carriers.** Historically a byte value > the byte-0
row was emitted through per-value `H1/H2/H3_DUMP_OUT` LM-head columns (the
"H-band"). Per the 2026-07 refactor (memory:
`project_register_emission_is_shared_h_onehots_structural_root`), AX byte-1/2/3
were **consolidated onto the canonical OUTPUT path**: `make_b1_to_output_op`
(l11_ops.py:2126, 16 rules) DECODES the carried `H<k>_DUMP_OUT+off` one-hot back
into `OUTPUT_LO+(v&0xF)` / `OUTPUT_HI+(v>>4)` at the byte-1 row, and the H-band
LM-head columns were dropped. The H-band is now purely an INTERNAL cross-step
CARRIER (filled by `_ax_byte1_dump_repopulate_rules`), no longer an emission
path. `OUTPUT_HI_THIS_STEP` / `OUTPUT_HI_PREV_STEP` split lets a byte-writeback
distinguish the value this step wrote from the carried-forward one (B9 split,
`docs/B9_OUTPUT_HI_SPLIT_SPEC.md`).

### 1.d Cross-step register carry — `isa_semantics_dsl.cross_step_carry`

A register byte that a step does NOT freshly re-derive (e.g. SP byte-2 when the
opcode only touches byte-0, or the saved-BP after ENT) must still be EMITTED with
its correct value. The **cross-step carry** primitive
(`neural_vm/unified_compiler/isa_semantics_dsl.py:682`, `CrossStepCarrySpec`)
supplies the identical 4-part structure for every such carry:

1. a dedicated `_PREV` residual band (`register_residual_band`, `never_share`);
2. an UNCONDITIONAL carry HEAD (`alibi_slope≈0.5`, Q/K match the prev row's
   per-byte positional signature, V reads `<src>.*.-1` cross-step, O writes
   `_PREV`);
3. a gated DUMP FFN at the L25 tail (`multi_way_and_rule` per cell, GATED on the
   carried `_PREV+j` band, re-supplies the emit band) — **the carry-vs-fresh
   gate lives HERE**, expressed only in `dump_gate_conditions`;
4. the band-pass/kill discriminator, folded into the dump gate.

The API SHAPE enforces "head unconditional, gate in the FFN" (there is no
head-gate field). Live carries built this way (from `full_vm_compiler_dynamic.py`
band registrations + memory notes):

| carry | band | producer / consumer | status |
|-------|------|--------------------|--------|
| AX byte-1 | `H1_PREV_STEP` (+H2/H3) | L11 head 1069 → L25 dump | live (golden) |
| STACK0 byte-0 | `STACK0_B0_*` | L9 head 5 → L25 dump | live (SOLE Root-2 survivor) |
| ENT saved-BP | `BP_SAVE_PREV` | L11 head 3212 → L25 dump | live (func/nested/rec) |
| SP byte-2 | (collapsed, see §1.f) | binary-pop step | flag `C4_SP_BYTE2_CARRY` |
| PC byte-1 | `PC_BYTE1_*` | R3 band | flag `C4_PC_BYTE1_CARRY` (#393, in flight) |
| BP byte-1 | `BP_BYTE1_*` | R3 band | flag `C4_BP_BYTE1_CARRY` (#394, pending) |

`BP_SAVE_PREV` is the byte-identity PROOF that the generator reproduces a
hand-built carry bit-for-bit (flag-on AND flag-off); it is the reference the
generic engine's "carry register when unwritten" primitive must satisfy.

### 1.e L25 tail emission + byte-writeback banks (M8)

The L25 tail is the LAST-WRITER OUTPUT correction block:
`_tail_bit32_result_correction_rules` (`l10_ops.py:6043`),
`_L10_FFN_UNIT_LAYOUT_TAIL_BIT32_TOTAL = 2059` units, plus ~40 sub-generators.
About 40 of them are the SAME shape — the **evidence-keyed byte-value
writeback** (M8): under a fixed structural evidence gate, for each byte value
`v = lo | (hi<<4)` observed in a `(OUTPUT_LO+lo, OUTPUT_HI+hi)` source lane,
GUARANTEE `OUTPUT byte = v`. Each hand-authored family is a 16×16 nibble loop
(256 units) writing `Primitives.byte_value_writes(v)`.

These are now authored through ONE factory, `_byte_value_writeback_rules`
(`l10_ops.py:5960`): a family collapses from a ~40-line nested loop to a single
keyword-argument call while emitting the byte-IDENTICAL rule tuple.
`stack0_store_loaded_output_rules` (l10_ops.py:7194) is the landed pilot (#389,
J5). The factory is the "generic computed byte-copy" the M8 GAP wants — but it
still *enumerates* 256 rules per family (see §3, gap M8).

### 1.f The SP-byte2 collapse — the proof the enumeration is data, not code

`sp_pop_carry_rules` byte-2 family enumerated one FFN unit per `old_value` in
`range(256)` (`tail_sp_pop_carry_byte2_00`..`_ff`), 256 rules, each writing
`(old+1)&0xFF` on the pop step. Corpus reality (memory:
`project #390`, l10_ops.py:798): SP never leaves `[0x0FE10, 0x10000]`, so SP
byte-2 takes **exactly two values (0x00, 0x01)** and the ONLY live carry is
`0x00→0x01`. `byte2 = 0x01 iff (SP & 0xFFFF)==0 else 0x00` has ZERO violations
over 657,600 per-step SP samples. `C4_SP_BYTE2_CARRY=1` collapses the 256-rule
bank to 2 (drops 254 provably-dead siblings), byte-identical on every reachable
step — a verdict-neutral ~0.5k-LOC deletion. **This is the template for the M8
GAP**: the enumerated bank encodes a single computed fact `next = (old+1) &
0xFF`, and 254/256 rows are structurally unreachable.

---

## 2. SEMANTIC-SPEC DATA — the frame as data

### 2.a The frame model

The emitted step is a fixed sequence of **register blocks**, each block a marker
token followed by its 4 little-endian value bytes:

```
STEP_FRAME := [ BLOCK(PC), BLOCK(AX), BLOCK(SP), BLOCK(BP),
                (BLOCK(STACK0) only if 35-token),
                MEM_BLOCK, STEP_END ]

BLOCK(REG)  := [ marker=REG_<REG>,  value_byte(REG, 0..3) ]   # 5 tokens
MEM_BLOCK   := [ marker=MEM, addr_byte(0..3), val_byte(0..3) ] # 9 tokens
```

so a step emits, at these positions (30-token / campaign):

```
step emits [ PC_marker, PC_b0, PC_b1, PC_b2, PC_b3,
             AX_marker, AX_b0..b3,  SP_marker, SP_b0..b3,  BP_marker, BP_b0..b3,
             MEM_marker, MEM_addr_b0..b3, MEM_val_b0..b3,  STEP_END ]
       at   [ 0, 1,2,3,4, 5,6,7,8,9, 10,11,12,13,14, 15,16,17,18,19,
             20, 21,22,23,24, 25,26,27,28, 29 ]
```

The ONLY per-layout datum is `emit_stack0 ∈ {true(35), false(30)}`; everything
else (block order, byte order little-endian, marker set) is a constant. A generic
engine derives the whole positional map from `(register_list, emit_stack0,
mem_has_addr_and_val)`.

### 2.b The two generic emission rules

**R-FRAME (marker schedule).** For a frame of blocks `B0..Bn`, fire `NEXT_<B_{i+1}>`
one token past `B_i`'s marker after `B_i`'s value bytes are done (distance =
block width). Concretely the L0 transition table is:

```
edge(B_i -> B_{i+1}) := (H_wide[+marker_i(B_i)] AND NOT H_narrow[+marker_i(B_i)]) -> NEXT_<B_{i+1}>
```

where `H_wide/H_narrow` are the distance-from-marker threshold heads (H0..H4).
The table is a pure list of `(from_block, to_block)` edges over the block order;
dropping/adding a block (STACK0) is one row.

**R-BYTE (value → token).** For each byte value `v` and each register value
byte, the LM head emits `v` iff `OUTPUT_LO+(v&0xF)` AND `OUTPUT_HI+(v>>4)` are the
lit nibble one-hots at that row and no `NEXT_*` marker flag is set:

```
emit_token(row) := argmax_b [ 5*OUTPUT_LO[b&0xF] + 5*OUTPUT_HI[b>>4] - 5   ]  # value row
                 = REG_<REG>  when NEXT_<REG> fired                          # marker row
```

The per-opcode work reduces to **"land `OUTPUT_LO/HI` = nibble one-hots of the
result byte at each value row."** That is the interface every opcode's lowering
must hit; the emission itself is these two rules, shared.

### 2.c The cross-step carry as data (`CrossStepCarrySpec`)

A carry is fully described by the `CrossStepCarrySpec` fields (§1.d): `name`,
`band_name/width`, `carry_byte_count`, the per-byte positional match
`(match_q_band{k} ↔ match_k_band{k})`, `k_prefer/k_reject` signature, the clean
`value_src_lo/hi` bands, and `dump_gate_conditions` (the carry-vs-fresh gate).
The generic model: *"when the producing step's clean byte is not re-derived by
the consuming step, copy `<src>` across the step boundary into `_PREV`, then
re-supply it into `OUTPUT` at the consuming row gated by `dump_gate_conditions`."*

---

## 3. GAP-LIST — what resists generic derivation

Classification: **[SPEC]** = a datum the generic model must carry;
**[LOWERING]** = an artifact a clean generic emitter would NOT produce (repair
noise around the two primitives).

### G1. The frame chain is FULLY generic. [LOWERING: clean]
`phase_a_ffn`'s transition table is a pure block-order edge list; `emit_stack0`
is the only datum. A generic engine reproduces both layouts from
`(register_list, emit_stack0)` with zero opcode-specific code. **Decisive "frame
is one shared primitive" result** — the 30/35 switch is one boolean.

### G2. The byte→token decode is FULLY generic. [LOWERING: clean]
`_head_bake_rules` is one 256-row loop `head[b] = OUTPUT_LO[b&0xF] +
OUTPUT_HI[b>>4]` + a uniform `NEXT_*` suppression. No per-opcode data. A generic
engine emits the whole head from the byte range + marker set.

### G3. Cross-step carry IS a general primitive — but its GATE is per-carry SPEC. [SPEC]
`cross_step_carry` proves the 4-part structure is uniform (BP_SAVE_PREV is
byte-identity-gated). The generic "carry register when unwritten" primitive
exists. What it CANNOT infer is `dump_gate_conditions` — the discriminator of
*which* consuming rows re-supply vs let the fresh value stand (BP: `OP_ENT`;
AX byte-1: the `AX_CARRY_OVERFLOW` two-stage kill; STACK0 b0: precursor flags).
This is genuine SPEC DATA (one gate tuple per carry), but it is a SHORT list, not
per-opcode logic. **The SP-byte2 collapse (§1.f) shows the carry VALUE itself is
computed (`old+1`), never enumerated.**

### G4. The M8 enumerated byte-writeback banks are LOWERING, not SPEC. [LOWERING]
The ~40 tail families each enumerate 256 rules to express `guarantee OUTPUT byte
= v` for `v` read out of a `(lo,hi)` nibble lane. Semantically this is ONE
computed copy: `OUTPUT = copy(source_lane)`. The enumeration exists only because
the FFN substrate lacks a "copy this 8-bit lane to OUTPUT" primitive, so it is
spelled as 256 one-hot AND rules. `_byte_value_writeback_rules` already
FACTORS the authoring; the SP-byte2 collapse PROVES most rows are dead. **A
generic engine with a `byte_copy(src_lane → OUTPUT)` lowering primitive replaces
each 256-rule family with one op.** This is the biggest LOC lever in the family
(~40 families × up-to-256 rules) and it is pure lowering — no verdict change, no
new SPEC datum. Classify the whole M8 bank as LOWERING to be collapsed.

### G5. The ≠STEP_TOKENS framing drift (root #388) is a LOWERING bug, not SPEC. [LOWERING]
`Token.STEP_TOKENS` is the frame length authority, but production decodes a
FIXED-`STEP_TOKENS`-slice; when a step emits the WRONG count the slice desyncs
and every later register byte is misread. Observed: the post-ENT step emits **37
tokens** (2 leading stray `0xFF`) instead of 35 — `func_*/nested_*/rec_*/var_*`
diverge here (`docs/FUNC_LEV_IS_LI_FROM_FRAME_37TOKEN_DESYNC_2026_06_14.md`;
memory `project_if_bool_expr_is_stack0_highnibble_framing_drift`). Root: a value
byte fails to yield to the next marker (the L0 STEP_END-row stray-byte
suppression `−80*NEXT_PC` is out-voted; l0_ops.py:548-807 documents the
mechanism), so a spurious byte token is emitted BEFORE the next marker, inflating
the count. **This is the emission primitive MISFIRING** — the frame model itself
is correct (30/35), but the R-BYTE decode leaks a stray byte at a marker row.
The generic emitter, if the two primitives fire cleanly (marker always wins its
row via `NEXT_*`), produces the exact-N frame BY CONSTRUCTION and this class
cannot occur. So #388 is a repair for an under-strength suppression, not a spec
gap — it VANISHES under a clean generic lowering that makes `NEXT_*` dominate the
marker row unconditionally.

### G6. `OUTPUT_HI_THIS_STEP` vs `OUTPUT_HI_PREV_STEP` split is a SPEC datum. [SPEC]
The byte-writeback banks and carries need to distinguish the value THIS step
computed from the one carried forward (B9 split). This is a real semantic field
(which step produced the byte) the generic model must carry per write — a
one-bit `this_step`/`prev_step` tag on each OUTPUT write, NOT per-opcode logic.

### G7. L25 tail unit ORDER / count is byte-identity-load-bearing. [LOWERING]
`_L10_FFN_UNIT_LAYOUT_TAIL_BIT32_TOTAL = 2059` and the tail-bank append order
are asserted for the golden hash; the SP-byte2 / M8 collapses are gated OFF
precisely so the golden count is preserved. This is a lowering constraint (the
weights are order-sensitive though the behavior is not), identical in kind to
DECODE's G5. A generic emitter reproduces byte-identity only if it emits banks in
the declared order; the collapse lands as a verdict-neutral geometry cut behind a
flag.

### G8. Fetch/threshold heads (H0..H4) are shared addressing, not emission. [scoping]
The L0 marker chain READS the H0..H4 distance-from-marker threshold heads
(`layer0_ff`). Those heads are addressing machinery shared by ALL opcodes and
are NOT opcode-parameterized; a FRAMING engine treats `H0..H4` as given inputs
(the same scoping as DECODE's fetch-attention G6).

---

## 4. Verdict — does the whole family derive from ONE generic primitive?

**Almost — from TWO, cleanly.** Frame emission is:

* **R-FRAME** — a marker-schedule state machine over the block-order edge list
  (`phase_a_ffn`), opcode-agnostic, 30/35 selected by one boolean `emit_stack0`.
  FULLY generic (G1).
* **R-BYTE** — the LM head reading each value byte from `OUTPUT_LO/HI` nibble
  one-hots, one 256-row lookup, opcode-agnostic. FULLY generic (G2).

Every opcode reduces to "land the result nibbles in `OUTPUT` at each value row";
the frame + decode are shared. The residue is NOT per-opcode behavior:

* **cross-step carry** is a general primitive (`cross_step_carry`); its per-carry
  `dump_gate_conditions` + the `this_step/prev_step` tag are the only SPEC data
  (G3, G6) — short lists, and the carried VALUE is computed, not enumerated
  (SP-byte2 proof, §1.f).
* the **M8 enumerated byte-writeback banks** (G4) and the **tail order/count**
  (G7) are pure LOWERING to be collapsed by a `byte_copy(src→OUTPUT)` primitive
  (the biggest LOC lever, verdict-neutral).
* the **≠STEP_TOKENS drift** (root #388, G5) is an emission MISFIRE (an
  under-strength marker-row suppression), not a spec gap — it cannot occur under
  a clean R-FRAME + R-BYTE lowering where the marker always wins its row.

So EMIT/FRAMING is the decisive SHARED-across-all-opcodes win: two generic
primitives cover the entire step scaffolding, the carry is a third general
primitive with small per-carry gate DATA, and the rest is enumeration to be
computed away. Nothing in the family is a per-opcode special case.

### Verification performed (READ-only; no weights touched)
* Frame layout read from `neural_vm/token_layout.py` (authoritative,
  parametrized on `Token.STEP_TOKENS`); 30-token campaign vs 35-token golden
  positions confirmed; `constants.py:90` asserts `TOKENS_PER_STEP == (30 if
  C4_NO_STACK0_EMIT else 35)`.
* L0 transition table read from `l0_ops.py:122` (6 edges 30-token, 7 edges
  35-token; STACK0 row is the sole difference, gated by `_no_stack0_emit`).
* Byte→token rule read from `model_ops._head_bake_rules` (model_ops.py:2104):
  `head[b, OUTPUT_LO+(b&0xF)]=5, head[b, OUTPUT_HI+(b>>4)]=5`, one loop over 256.
* Cross-step carry structure read from `isa_semantics_dsl.CrossStepCarrySpec` /
  `cross_step_carry` (isa_semantics_dsl.py:248/682); band registrations from
  `full_vm_compiler_dynamic.py:1565-1593`.
* M8 factory `_byte_value_writeback_rules` (l10_ops.py:5960) + the landed pilot
  `stack0_store_loaded_output_rules` (l10_ops.py:7194); SP-byte2 collapse
  (l10_ops.py:798, `C4_SP_BYTE2_CARRY`, 256→2, ZERO violations / 657600 samples).
* `#388` drift mechanism from `docs/FUNC_LEV_IS_LI_FROM_FRAME_37TOKEN_DESYNC_2026_06_14.md`
  + the L0 STEP_END-row suppression narrative (l0_ops.py:548-807).
* Golden hash unchanged: `81557d21` (this doc reads ops + writes markdown only).

### Feeds
This spec is the FRAMING half of the generic-lowering pilot (#391) and the
sibling of `semantic_spec_{DECODE,MEMORY,CONTROL,ALU}.md`. R-FRAME + R-BYTE + the
`byte_copy(src→OUTPUT)` primitive (to collapse M8, G4) are the reference the
engine's shared step-scaffolding lowering must satisfy; the carry gate + the
`this_step/prev_step` tag are the only per-instance SPEC data it must carry.
