# Semantic Spec — the MEMORY family (LI, LC, SI, SC, LEA, PSH, POP/ADJ)

**Status:** READ + DESIGN. No weights modified. Golden flag-OFF hash
`81557d21422f3eada0a87c677b00dced41cc26c3ee3bfb094c5eeb71c9b4d3cb` (`81557d21`)
is unchanged by this doc (gate: `tools/_isa_golden_hash.py`).

**Purpose (100%-derivable-architecture prep):** map the hand-authored lowering
for the MEMORY family, re-express each op's behaviour as *semantic-spec DATA*
(address-compute + value-route + stack-pointer-update), and enumerate the
GAP-LIST — what resists generic derivation from that data. Feeds the generic
engine (task #391 "prove ONE opcode 100%-derivable"; companion to
`semantic_spec_CONTROL.md`, task #392).

Cross-refs: memory notes `project_si_store_provenance_two_root_wall`,
`project_l7_operand_gather_not_broken`, `project_l15_li_stack0_byte_attribution`,
`project_l16_psh_e0_sentinel_is_load_bearing_accident`. Source of truth for the
ISA one-liners: `neural_vm/opcode_mapper.py:28-43` and
`neural_vm/embedding.py:130-197`.

---

## 0. The MEMORY family and its ISA semantics

From `neural_vm/opcode_mapper.py` / `neural_vm/embedding.py` (opcode numbers are
the byte values in the compiled stream). C4 is a stack machine: `AX` is the
accumulator, `SP` the stack pointer (grows DOWN, 8-byte aligned:
`constants.STACK_ALIGNMENT = 8`), `BP` the frame/base pointer. The instruction
word is opcode(1) + immediate(4) + padding(3) = 8 bytes (`constants.py:16-18`).

| op  | # | one-liner (ISA)              | class in `opcode_mapper` | address source | value route | SP update |
|-----|---|------------------------------|--------------------------|----------------|-------------|-----------|
| LEA | 0 | `AX = BP + imm`              | `FFN_COMPOSITE` (ADD)    | —              | compute BP+imm → AX | — |
| LI  | 9 | `AX = *AX` (load 8 bytes)    | `ATTENTION_NEEDED`       | AX (pointer)   | mem[AX] → AX | — |
| LC  |10 | `AX = *(char*)AX` (load 1)   | `ATTENTION_NEEDED`       | AX             | mem[AX] → AX | — |
| SI  |11 | `*pop = AX` (store 8 bytes)  | `ATTENTION_NEEDED`       | popped stack top | AX → mem[addr] | SP += 8 (pop) |
| SC  |12 | `*(char*)pop = AX` (store 1) | `ATTENTION_NEEDED`       | popped stack top | AX → mem[addr] | SP += 8 (pop) |
| PSH |13 | `SP -= 8; *SP = AX`          | `ATTENTION_NEEDED`       | new SP         | AX → mem[SP]; AX → STACK0 | SP -= 8 |
| ADJ | 7 | `SP += imm`                  | `FFN_COMPOSITE` (ADD)    | —              | — | SP += imm |
| POP | — | `SP += 8` (ADJ special-case) | `FFN_COMPOSITE` (ADD)    | —              | — | SP += 8 |

The `opcode_mapper` classification IS the first-order gap classification:
**LEA/ADJ/POP are pure ALU (`AX/SP = reg + const`) — derivable from an ADD
primitive; LI/LC/SI/SC/PSH need the address-keyed memory CAM
(`ATTENTION_NEEDED`).** Everything below refines that.

C4 memory is modelled in the transformer as MEM tokens carried in the KV
sequence: a store emits a `MARK_MEM` token whose residual holds the address
nibbles (`ADDR_B0/1/2`) and value nibbles (`MEM_VAL_B0..B3`); a load is an
attention lookup that content-matches the address and copies the value. There
is no separate RAM — "memory" is attend-by-address over the emitted MEM/STACK0
token history.

---

## 1. INVENTORY — per op, the rules / attention-CAMs that implement it

All file paths are `neural_vm/unified_compiler/ops/`. "Head N" = the attention
head index in that layer's block; layer→physical-block map in
`docs/PROBE_GROUNDTRUTH_2026_06_10.md` (27 logical layers, 37 physical blocks).

### 1a. Address computation (BP-relative / SP-relative — the ALU half)

| op | mechanism | file:LOC |
|----|-----------|----------|
| LEA operand relay | L7 operand-gather **head 1**: Q@`MARK_AX` gated `OP_LEA/ADJ/ENT`, K@`MARK_BP`/`MARK_SP`, V/O copy `OUTPUT_LO/HI` → `ALU_LO/HI` (pulls the live frame BP/SP into the ALU). | `l7_ops.py:389-438` |
| LEA = base + imm | L8 ALU FFN: `_layer8_alu_lea_lo_rules` (256 units: `result=(a+b)%16`, operand A=`ALU_LO` [=BP], operand B=`FETCH_LO` [=imm one-hot], gate `OP_LEA`) + `_layer8_alu_lea_carry_rules` + `_layer8_alu_lea_axb2_rules`. | `l8_ops.py:416-453, 536-590, 1117-1180` |
| ADJ = SP + imm | L6 `_layer6_adj_ax_route_rules` + `_layer6_adj_sp_writeback_rules`. | `l6_ops.py:1273-1360` |

LEA is literally the ADD ALU gated on `OP_LEA` with operand B rebound from the
carry lane (`AX_CARRY`) to the immediate lane (`FETCH_LO`). This is the exact
"composite = ADD(reg, imm)" the `opcode_mapper` predicts.

### 1b. Stack-pointer update (the SP-math half)

| op | mechanism | file:LOC |
|----|-----------|----------|
| PSH `SP -= 8` | L6 `_layer6_psh_sp_decrement_rules` → `_layer6_sp_decrement_rules` (per-nibble `new_k=(k-8)%16` gated `PSH_AT_SP + MARK_SP`, hi-byte borrow gated on lo-byte < 8). | `l6_ops.py:970-1040` |
| JSR `SP -= 8` | same helper, gated `CMP+4 + MARK_SP`. | `l6_ops.py:981-989` |
| POP / binary-pop `SP += 8` | L6 `make_binary_pop_sp_increment_op` + `_layer6_binary_pop_sp_increment_rules` (nibble rotate +8 with carry). | `l6_ops.py:4144-4381` |
| SP byte-0 marker fixups | L10 `sp_pop_marker_increment_rules` (`e0→e8` etc. on the pop step), `_nonfirst_psh_sp_fix` (byte-0 0xF8 on the 2nd+ PSH). Tail correctors, several campaign-gated. | `l10_ops.py:735-893, 6331+` |
| SP byte-2 pop-carry | cross-step carry band (`SP_BYTE2_PREV`), replaced the 256-unit enumeration (task #390). | `l11_ops.py` (carry) / `l10_ops.py` |

SP-update is the borrow/carry nibble arithmetic of `±8` gated on the marker +
opcode flag — the SAME nibble-rotation chain the PC-increment and LEA use, just
a fixed `±8` constant. Structurally derivable from an "add-constant-to-register"
primitive; the tail correctors are firmware patches over frame-desync edge cases
(see gap-list G4).

### 1c. The load path — LI / LC (value read from mem[AX])

| stage | mechanism | file:LOC |
|-------|-----------|----------|
| addr bytes into the addr band | L13 `layer13_mem_addr_gather` heads 0-2: Q@`MEM_VAL_Bj`, K picks the MEM addr-byte-j row via `L1H*[MEM_I]` marker signature, V/O copy `CLEAN_EMBED` → `ADDR_Bj_LO/HI` (+ `ADDR_Bj_VALID` lifecycle). | `l13_ops.py:331-440` |
| the value load itself | L15 `layer15_memory_lookup` heads **0-3** (`li_lc_stack0_h{0..3}`): per byte-index head, Q gated `OP_LI_RELAY`/`OP_LC_RELAY` + `CMP+3` (POP dual-role), K anchors `MEM_STORE`, **slots 4-27 = 24-bit binary address MATCH** (`ADDR_B0/1/2` Q vs K, bit-encoded), V/O copy the matched MEM row's `CLEAN_EMBED_LO/HI` → `OUTPUT_LO/HI`. ALiBi slope 0.05 → most-recent matching store wins. | `l15_ops.py:1110-1290` |
| zero-address / committed guards | `_l15_li_zeroaddr_cam_on`, `_l15_li_addr_cam_discriminator_on`, `LI_ZEROADDR_COMMITTED` band (L14). | `l15_ops.py:352-450`, `l14_ops.py:29-42` |

LI/LC ARE the addressed-load CAM: **query = LI marker keyed on the target
address; key = each MEM row's address signature; value = that row's stored
byte.** The 24-bit binary bit-match (slots 4-27) is the content-addressable
comparator.

### 1d. The store path — SI / SC (value write to mem[popped addr]) + PSH

| stage | mechanism | file:LOC |
|-------|-----------|----------|
| emit the MEM token (addr + value from AX) | L14 `layer14_mem_generation` heads 0-7: heads 0-3 gather MEM addr bytes, heads 4-7 gather MEM val bytes from AX. | `l14_ops.py:1317+` (layout `:140-147`) |
| MEM_STORE_AT_VAL commit flag | L7 `make_layer7_mem_store_relay_op` writes `MEM_STORE_AT_VAL` (the "committed store value" marker). | `l7_ops.py:582-677` (band `:42`) |
| store-top address (post-pop SP) | L15 `layer15_store_stack0_sp_byte0_addr` **head 12**: Q@`MARK_STACK0 + HAS_SE` (store markers), K@`MARK_SP`, V/O copy `OUTPUT_LO/HI` (post-pop SP byte0) → `ADDR_B0_LO/HI`. | `l15_ops.py:3487-3546` |
| **store-provenance CAM** (multilocal) | L15 `layer15_si_store_addr_cam` **head 16** (campaign, `C4_SI_STORE_ADDR`, DEFAULT-OFF): Q@LI-emit row keyed on `AX_CARRY` (target addr), K@candidate markers keyed on `ADDR_B0` (store addr), V copies the marker's `AX_CARRY` (clean value) → `OUTPUT`. | `l15_ops.py:2885-3000` |
| PSH broadcast AX → STACK0 | L10 `layer10_psh_ax_broadcast` heads 8-10 (`AX_b1/2/3 → STACK0_BYTE_VAL_1/2/3`) + `layer10_psh_stack0_passthrough`. | `l10_ops.py:3879-4030, 4667-4717, 5109+` |
| store value → OUTPUT (loaded-back) | L10 `stack0_store_loaded_output_rules` (tail bank; task #389 collapse target). | `l10_ops.py:7194-7227` |

### 1e. The operand-A gather + the mem[SP] relay (the shared CAM substrate)

| mechanism | role | file:LOC |
|-----------|------|----------|
| L7 `layer7_operand_gather` **head 0** | operand-A gather: Q@`MARK_AX`, K@`STACK0_BYTE0`, V/O `CLEAN_EMBED → ALU`. **Now expressed via `cam_lookup(...)`.** | `l7_ops.py:300-368` |
| L4 `make_layer4_sp_to_addr_key_op` | stages live SP as nibble one-hots into `ADDR_KEY` at the AX marker (the Q-side of the mem[SP] CAM). Currently `enable=False`. | `l4_ops.py:806+` |
| L8 `make_layer8_mem_to_alu_op` **head 5** | mem[SP] → ALU: Q@AX gated binary-pop ops, K@MEM val-byte-0 gated `MEM_STORE`, 12-bit addr match, V/O `CLEAN_EMBED → ALU`. Currently `enable=False`. | `l8_ops.py:2504-2650` |

---

## 2. SEMANTIC-SPEC DATA — each op as (address-compute, value-route, SP-update)

The claim of this section: every MEMORY op reduces to a small tuple of DATA over
three sub-machines that ALREADY exist as parameterized generators. The
generators live in `neural_vm/unified_compiler/isa_semantics_dsl.py`:

- **`cam_lookup(CamLookupSpec)`** — the addressed load/store primitive (§1e L7
  head 0 is the byte-identity reference re-expression). One `CamKeyMatch`
  (query marker → key signature), a tuple of `CamValueBand` relays
  (source_band → target_band), optional opcode blockers + a CONST confirm slot.
- **the nibble-rotation chain** (`l4_ops._nibble_rotation_chain_rules`,
  reused by `consumer_lookahead_gate`) — the `reg ± const` adder used by
  PC+offset, LEA, ADJ, SP±8.
- **`cross_step_carry(CrossStepCarrySpec)`** — the cross-step value/marker carry
  (SP byte-2 pop-carry, BP_SAVE_PREV, STACK0 byte-0, AX byte-1).

### 2a. The data model

```
MemOp {
  opcode:        LEA | LI | LC | SI | SC | PSH | ADJ | POP
  address_compute: AddrSpec        # WHERE the effective address comes from
  value_route:     ValueRouteSpec  # WHAT byte moves, from where to where
  sp_update:       SpDeltaSpec | None
}

AddrSpec (one of):
  REG_PLUS_IMM(base=BP|SP, imm)     # LEA, ADJ  -> nibble-rotation-chain adder
  FROM_AX                           # LI, LC    -> AX holds the pointer
  FROM_POPPED_STACK_TOP             # SI, SC    -> popped stack slot holds addr
  FROM_NEW_SP                       # PSH       -> writes at the decremented SP
  NONE                              # POP       -> pure SP arithmetic

ValueRouteSpec (one of):
  COMPUTE_TO_AX(alu_expr)           # LEA: BP+imm -> AX (ALU FFN)
  MEM_LOAD(addr=AddrSpec) -> AX     # LI/LC: cam_lookup, key=address, val=byte
  MEM_STORE(addr=AddrSpec, src=AX)  # SI/SC/PSH: emit MEM token addr+value
  NONE                              # ADJ/POP

SpDeltaSpec:
  DELTA(sign=+|-, amount=8|imm, gate=(opcode_flag, marker))  # nibble adder ±k
```

### 2b. Per-op instantiation (the data)

```python
LEA = MemOp(
  address_compute = NONE,
  value_route     = COMPUTE_TO_AX(ADD(base=BP, imm)),   # l8 alu_lea, gate OP_LEA
  sp_update       = None,
)                                                        # 100% ALU-derivable

ADJ = MemOp(NONE, NONE, SpDeltaSpec(+, imm, (OP_ADJ, MARK_SP)))     # ALU-derivable
POP = MemOp(NONE, NONE, SpDeltaSpec(+, 8,   (binary_pop, MARK_SP))) # ALU-derivable

LI = MemOp(
  address_compute = FROM_AX,
  value_route     = MEM_LOAD(
      # == cam_lookup:
      key_match  = CamKeyMatch(query=OP_LI_RELAY_marker, key=ADDR_B0/1/2_match),
      value_band = CamValueBand(CLEAN_EMBED_LO/HI -> OUTPUT_LO/HI, width=16),
      recency    = alibi_slope 0.05),  # most-recent matching store wins
  sp_update       = None,
)
LC = LI with 1-byte value (single CLEAN_EMBED byte)

SI = MemOp(
  address_compute = FROM_POPPED_STACK_TOP,   # store-top addr = post-pop SP row
  value_route     = MEM_STORE(addr, src=AX), # l14 mem_generation emits MEM token
  sp_update       = SpDeltaSpec(+, 8, (OP_SI, MARK_SP)),  # the pop
)
SC = SI with 1-byte value

PSH = MemOp(
  address_compute = FROM_NEW_SP,
  value_route     = MEM_STORE(addr=new_SP, src=AX) + broadcast(AX -> STACK0),
  sp_update       = SpDeltaSpec(-, 8, (PSH_AT_SP, MARK_SP)),  # the decrement
)
```

### 2c. The recurring primitive — one "addressed load/store" CAM

The load (LI/LC), the operand-A gather (L7 head 0), the mem[SP] relay
(L8 head 5), the LEV saved-BP / return-addr lookups (L15 heads 4-11), and the
store-provenance CAM (L15 head 16) are ALL the same shape, differing only in:

```
(query_marker, key_signature, value_source_band, target_band, alibi_slope)
```

That IS the `CamLookupSpec` tuple. The store side (SI/SC/PSH) is the same
primitive run in reverse: instead of *reading* a MEM row by address it *emits*
one (L14 mem_generation gathers addr + value from the AX/SP frame into a MEM
token). So the family collapses to **ONE bidirectional addressed-memory
primitive** parameterized by direction (load = read-by-address,
store = emit-with-address) + the (query, key, value, target) tuple.

---

## 3. GAP-LIST — what resists generic derivation

Classified as **SPEC gap** (missing data in the semantic model → add data) vs
**LOWERING gap** (the generic engine can't yet emit this weight shape → extend
the lowering) vs **FIRMWARE gap** (a hand-tuned magnitude/edge-case patch that
is not implied by the ISA and would need a different, non-declarative
justification).

### G1. The addressed load/store CAM — IS a generic primitive. **[LOWERING, solved-in-principle]**
`cam_lookup` already exists and the L7 operand-gather head is byte-identically
re-expressed through it (golden hash held). The store-provenance CAM (§1d head
16) and the LI load head fit the SAME shape. **Verdict: the "attend-by-address,
copy-value" primitive IS general and IS in the DSL.** The remaining lowering
work is (a) generalizing the 24-bit binary address-MATCH block (L15 slots 4-27)
into a `CamKeyMatch` variant (currently the match is hand-unrolled per bit; the
`CamKeyMatch` API models a single dim-pair, not a 24-bit comparator), and (b)
adding the *store/emit* direction to the generator (today `cam_lookup` only
models the read/relay direction; the L14 mem_generation emit side is still
hand-built). Both are additive API work, not a semantic gap.

#### DERIVE→FLIP→DELETE audit (2026-07-04, golden `81557d21`)

Auditing every MEMORY-family attention head against the CURRENT `cam_lookup`
API for a *byte-identical* flip (`grep cam_lookup(` = 1 site; `marker_broadcast(`
= 1 site):

- **L7 `layer7_operand_gather` head 0 — ALREADY FLIPPED + DELETED.** It is the
  sole `cam_lookup(...)` call (`l7_ops.py:365`); its hand-authored form
  (`_set_layer7_operand_gather`) survives ONLY in the dead `vm_step.py.current`
  backup — NOT on the build path. Nothing left to flip or delete here.
- **L13 `layer13_mem_addr_gather` heads 0-2 — NOT byte-identically
  cam_lookup-expressible.** These are a per-byte MARKER-RELATIVE positional
  gather (Q@`MEM_VAL_B{0..3}`, K@`L1H{1,2}/H0[MEM_I]` addr-byte-j signature,
  NOT a value/address content key) PLUS a slot-34 `ADDR_Bj_VALID` lifecycle
  bit + a two-band V/O relay. `cam_lookup`'s single `CamKeyMatch` (one
  query/key dim-pair) + `CamValueBand` shape cannot express the positional
  marker match or the VALID-bit slot. This is the closer relative of
  `MarkerBroadcastSpec` (a positional relay), not the address CAM.
- **L15 `li_lc_stack0_h{0..3}` (the LI/LC load heads) — NOT byte-identically
  cam_lookup-expressible.** The row select is the 24-bit binary address MATCH
  hand-unrolled across slots 4-27 (3 addr bytes × 2 nibbles × 4 bits), which
  the `CamKeyMatch` single-dim-pair API cannot model (the G1(a) lowering gap);
  head 0 further carries ~40 heterogeneous discriminator slots (the
  campaign-gated `_l15_li_*` families) far beyond one key-match + blocker
  overlay.
- **L15 `layer15_si_store_addr_cam` head 16 (`C4_SI_STORE_ADDR`), L8
  `layer8_mem_to_alu` head 5 — cam_lookup-SHAPED but not a byte-identical
  flip** (head 16 keys a 2-nibble address match with 5 tight veto slots; L8
  head 5 needs the store/emit direction, G1(b)).

**Bottom line:** the ONE MEMORY head byte-identically re-expressible via the
present `cam_lookup` API (L7 operand-gather head 0) is already the sole derived
path and its hand-authored rules are already gone. Every other l7/l13/l15
memory head requires the ADDITIVE DSL work catalogued here (24-bit `CamKeyMatch`
comparator, store/emit direction, VALID lifecycle, positional marker match) —
flipping them TODAY would NOT be byte-identical and would break golden. The
"derive → prove → flip → delete" template is validated end-to-end (L7 head 0 +
the L8 IMM `marker_broadcast` `derive_imm_enabled` gate); the remaining
MEMORY-head LOC reduction is BLOCKED on the G1 API extension, not on flipping
work.

### G2. Store provenance = a two-axis address+value binding, not just a CAM. **[SPEC, partially closed]**
Per `project_si_store_provenance_two_root_wall`: the naive load CAM reads the
store *value rows*, which are provenance-blind on BOTH axes — they carry neither
the address (Root 1: `ADDR_B0=0x00`, the L13 gather finds no addr row in the
30-token re-emission) nor the clean value (Root 2: the value-row `CLEAN_EMBED` is
address-like). The clean `(address, value)` pair lives ONLY on the store
**AX-marker** row. The fix (L15 head 16) keys the CAM on the MARKER's `ADDR_B0`
and copies the MARKER's `AX_CARRY` — i.e. the semantic model must record that a
store binds address→value **at the marker**, not the value row. **Spec-data to
add:** each `MEM_STORE` carries a `provenance_anchor = AX_MARKER` field so the
load CAM's `key_signature` and `value_source` resolve to the marker row. With
that field the CAM is generic; without it the generic engine would wire the
wrong K/V rows. This is the single most load-bearing missing datum in the model.

### G3. LEA / ADJ / SP±8 = the nibble-rotation adder — generic. **[LOWERING, solved]**
`REG_PLUS_IMM` and `SpDeltaSpec` both lower to `_nibble_rotation_chain_rules`
(the same chain PC+offset and `consumer_lookahead_gate` reuse). LEA's only
specialization is rebinding operand B from the carry lane to the immediate lane
(`FETCH_LO`) — a data flag (`operand_b_source = IMM`), not a new mechanism.
**Verdict: fully derivable from an "add-register-and-constant" primitive; the
data is just `(base_reg, delta_source, gate)`.**

### G4. SP byte-0/2 marker fixups + the tail correctors. **[FIRMWARE]**
`sp_pop_marker_increment_rules` (`e0→e8`), `_nonfirst_psh_sp_fix` (0xF8 byte-0),
the byte-2 pop-carry band, and several campaign-gated dominators
(`C4_SP_POP_MARKER_CMP3_HARDGATE`, `C4_SP_POP_CARRY_BYTE0_DOMINATE`) are NOT
implied by `SP ± 8`. They are patches over the 8-byte-alignment boundary
interacting with the marker-token bookkeeping and the 35→30-token campaign
re-frame. These do NOT derive from the ISA; the generic engine would emit the
clean `SP ± 8` and these edge cases would resurface as failures. **Verdict: a
correctly-derived SP adder plus a clean marker convention should make MOST of
these unnecessary; the residue is genuine firmware and must be either
re-justified or absorbed into a cleaner marker model. Flag these as
"non-derivable until the marker convention is itself spec'd."**

### G5. The MEM-token representation is implicit (no explicit RAM). **[SPEC]**
"Memory" is the emitted MEM/STACK0 token history attended-by-address. The
semantic model above says `MEM_STORE(addr) → MEM_LOAD(addr)` but the substrate
is a KV sequence + ALiBi recency, so two writes to the SAME address resolve by
*most-recent-token*, not by overwrite. `var_update` (re-store to a local) works
only because recency picks the latest store. **Spec-data to add:** the memory
model is *append-with-recency-resolution*, and the load's `alibi_slope` is the
overwrite semantics. A generic engine needs this recorded or it will mis-handle
re-stores (and it caps addressable distance — see the KV window). This is a
SPEC-level property of the C4-on-transformer embedding, not a bug.

### G6. LC vs LI, SC vs SI = byte-width, a data field. **[SPEC, trivial]**
LC/SC are the 1-byte variants of LI/SI. The only difference is
`value_width = 1 byte` vs `8 bytes` (which bytes of `CLEAN_EMBED` / how many
addr-byte heads participate). **Verdict: a `value_width` field on
`MEM_LOAD`/`MEM_STORE`; fully derivable.**

### G7. PSH is a store + a register broadcast. **[SPEC]**
PSH is not only `mem[SP] = AX`; it also broadcasts AX into the STACK0 bands
(`layer10_psh_ax_broadcast`) so the NEXT op's operand-A gather (L7 head 0, which
keys on `STACK0_BYTE0`) can read it. So PSH's `value_route` is
`MEM_STORE(new_SP, AX) + STACK0_BROADCAST(AX)`. **Spec-data to add:** the STACK0
broadcast is a second value-route target on PSH. This is the coupling between
the store primitive and the operand-gather CAM — the two are one data-flow.

### G8. The store-top address needs a live post-pop SP. **[LOWERING]**
SI/SC's `FROM_POPPED_STACK_TOP` requires the *post-pop* SP byte-0
(`layer15_store_stack0_sp_byte0_addr` head 12). The address-compute for a store
depends on the SP-update having *already* happened this step (pop then store).
**Verdict: an ordering constraint (`sp_update before address_compute` for
stores), not a new mechanism — but the generic engine must sequence the two
sub-machines correctly.**

---

## 4. Bottom line — does the family derive from a generic primitive?

**Yes, substantially.** The MEMORY family reduces to THREE parameterized
sub-machines that already exist as generators:

1. **the nibble-rotation adder** — LEA, ADJ, POP, and all SP±8 (G3). Pure ALU;
   `opcode_mapper` already calls these `FFN_COMPOSITE = ADD(reg, const)`.
2. **the addressed load/store CAM** (`cam_lookup`) — LI, LC, SI, SC, PSH, plus
   the operand-gather / mem[SP] / LEV lookups (G1). ONE bidirectional
   attend-by-address, copy-value primitive.
3. **the cross-step carry** — the SP high-byte pop-carry and the STACK0/BP/AX
   carries (`cross_step_carry`).

The store CAM is a **clean, generic address-keyed primitive** (memory note
`project_si_store_provenance_two_root_wall` confirms it generalizes 2→3+ locals
with no head-code change once the provenance anchor is right) — it is the best
generic-derivation candidate in the family and is already re-expressed through
the DSL on the read side.

The GAPS that are genuinely NOT derivable from the ISA one-liners:
- **G2 (provenance anchor)** — a required SPEC datum: a store binds
  address→value at the AX-marker row, not the value row. This is the ONE piece
  the generic engine must be TOLD; everything else follows.
- **G5 (append-with-recency memory model)** — a SPEC property of the
  transformer embedding of C4 memory (overwrite = most-recent token).
- **G4 (SP marker firmware)** — hand-tuned edge-case patches that a clean marker
  convention should mostly obviate; the residue is not ISA-derivable and must be
  re-justified.

Everything else (G3 adder, G6 byte-width, G7 PSH broadcast, G8 store ordering)
is data fields / ordering constraints over the three existing generators, not
new mechanisms. **The family is a strong candidate for the generic engine
pilot: LEA/ADJ/POP are decisively ALU-derivable today; LI/SI derive once the
provenance-anchor + recency data (G2, G5) are added to the spec.**
