# Semantic Spec — the CONTROL family (JMP, JSR, BZ, BNZ, ENT, LEV, ADJ, EXIT + CMP EQ/NE/LT/GT/LE/GE)

**Status:** READ + DESIGN. No weights modified by this doc. Golden flag-OFF hash
`81557d21422f3eada0a87c677b00dced41cc26c3ee3bfb094c5eeb71c9b4d3cb` (`81557d21`)
is the historical flag-OFF this doc was written against; the current default
golden is `91f55411` (gate: `tools/_isa_golden_hash.py`) — this doc only READS
ops and WRITES documentation.

**LANDED — the frame-step / register-delta primitive (2026-07).** The CONTROL
per-step register updates now DERIVE from a compact spec via
`isa_semantics_dsl.register_delta(RegisterDeltaSpec)` / `frame_step(specs)` (the
CONTROL analogue of `pc_mux`), expressing §2a's `PcNextSpec` + `FrameDeltaSpec[]`
as DATA:

* **SEQUENTIAL_ADD** (`reg + const` nibble adder, §2c) — the L3 sequential
  PC-next `PC + INSTR_WIDTH` (first-step CONSTANT default band + increment
  lo/hi + carry) is DERIVED + the hand-authored `_register_default_ffn_rules`
  PC bands (52 units) DELETED. G1 (default-writer) + G10 (frame `SP_DELTA` adder
  shape) closed for the PC track.
* **BRANCH_TARGET** (`idx_to_pc` encoder, §1d/§G2) — the model_ops JSR PC
  override (80 units: cancel + `imm*8+2` encoder + reserved band) is DERIVED +
  the hand loop DELETED. Same encoder `pc_mux`'s `imm_to_byte_addr` uses.

Byte-identical: whole-model `state_dict` hash unchanged at `91f55411`; the JSR
override holds flag-OFF AND flag-ON. **Residual DSL gap (still hand-authored):**
the `C4_JSR_PC_BYTE1` byte-1 stage (G3, PC ≥ 0x100) is SPLICED over the derived
reserved band — the frame primitive covers the byte-0 ISA encoder; the byte-1
carry band (`pc_width = 2`) is the next DSL extension. The SP/BP/STACK0 frame
DELTAS (ENT push/assign, LEV pops — G5) and their `_register_default_ffn_rules`
byte-default bands are NOT yet routed through `frame_step` (a further
extension: a `RegisterDelta` `PUSH`/`ASSIGN`/`POP-CAM` kind over the same
ordered-delta lowering).

**Purpose (100%-derivable-architecture prep):** map the hand-authored lowering
for the CONTROL family, re-express each op's behaviour as *semantic-spec DATA*
(a **PC-next** function + a **branch-condition** + a **frame-delta**), and
enumerate the GAP-LIST — what resists generic derivation from that data,
classified SPEC vs LOWERING vs FIRMWARE. Third of three companion maps
(`semantic_spec_DECODE.md`, `semantic_spec_MEMORY.md`); feeds the generic
lowering engine (task #391, pilot #392).

Source of truth for the ISA one-liners: `neural_vm/opcode_mapper.py:28-90`,
`neural_vm/embedding.py:130-197`, and `neural_vm/constants.py`
(`INSTR_WIDTH = 8`, `PC_OFFSET = 2`, `STACK_ALIGNMENT = 8`; the encoded byte
address of instruction index `i` is `pc(i) = i * 8 + 2`).

Cross-refs: memory notes `project_gcd_pc_byte1_construction_verified`,
`project_if_bool_expr_is_stack0_highnibble_framing_drift`,
`project_wave_b_se_decode_is_not_the_if_bool_lever`,
`project_root_a_lev_routing_bp_byte1_crush`, `project_jsr_simple_function_arch_blocked`,
`project_l16_psh_e0_sentinel_is_load_bearing_accident`; task #344 (if_var
step-11 branch-target PC), #350 (gcd step-0 JSR AX leak), #393/#394 (PC/BP
byte-1 carry bands).

---

## 0. The CONTROL family and its ISA semantics

C4 is a stack machine (`AX` accumulator, `SP` stack pointer growing DOWN 8-byte
aligned, `BP` frame pointer, `PC` program counter). The instruction word is
opcode(1) + immediate(4) + padding(3) = 8 bytes. The CONTROL family is
everything that (a) writes `PC` to something other than the sequential next
instruction, or (b) manipulates the call frame, or (c) computes a comparison
flag that a branch later consumes.

| op  | # | one-liner (ISA) | class in `opcode_mapper` | PC-next | frame Δ |
|-----|---|-----------------|--------------------------|---------|---------|
| JMP | 2 | `PC = imm`                         | control | **branch-target** `imm*8+2` | — |
| JSR | 3 | `push PC; PC = imm`                | control | **branch-target** `imm*8+2` (return-addr `PC+8` pushed) | `SP -= 8` |
| BZ  | 4 | `if AX==0: PC = imm`               | control | **cond** `imm*8+2` if `AX==0` else `PC+8` | — |
| BNZ | 5 | `if AX!=0: PC = imm`               | control | **cond** `imm*8+2` if `AX!=0` else `PC+8` | — |
| ENT | 6 | `push BP; BP = SP; SP -= imm`      | control | seq `PC+8` | `SP -= 8` (BP save) then `BP=SP`, `SP -= imm` |
| ADJ | 7 | `SP += imm`                        | ALU (ADD) | seq `PC+8` | `SP += imm` |
| LEV | 8 | `SP = BP; BP = pop; PC = pop`      | control | **return-addr** (popped) | `SP=BP`, pop BP, pop PC (`SP += 16` net) |
| EXIT| 38| halt / return exit-code            | control | (terminal) | — |
| EQ  |17 | `AX = (pop == AX)`                 | compare | seq `PC+8` | — (pops 1) |
| NE  |18 | `AX = (pop != AX)`                 | compare | seq | — |
| LT  |19 | `AX = (pop <  AX)`                 | compare | seq | — |
| GT  |20 | `AX = (pop >  AX)`                 | compare | seq | — |
| LE  |21 | `AX = (pop <= AX)`                 | compare | seq | — |
| GE  |22 | `AX = (pop >= AX)`                 | compare | seq | — |

The one-liners collapse to **three sub-machines** that the whole family shares:

1. **PC-next** — every op picks its next PC from `{sequential PC+8, branch-target
   imm*8+2, return-addr popped-from-stack}`. Sequential is the default (L3);
   the branch ops OVERRIDE it.
2. **branch-condition** — BZ/BNZ gate the override on the CMP-derived AX-zero
   flag. The CMP compares (EQ..GE) *produce* that flag into `AX`; the branch ops
   *consume* it. The compares are otherwise ALU (they write an ALU-style result
   to AX), so they live in the ALU family for their VALUE and in CONTROL only for
   the flag they feed a downstream branch.
3. **frame-delta** — JSR/ENT/LEV/ADJ mutate `SP`/`BP` by `±8`/`±imm`/`=BP`.
   Same nibble-rotation adder the MEMORY family uses (see `semantic_spec_MEMORY.md`
   §2c), plus the LEV pop-CAM (attend-by-address to the saved-BP / return-PC
   slots).

So **CONTROL derives from a generic primitive iff** those three sub-machines are
generic: (1) a mux over PC sources gated by opcode + condition, (2) a
CMP→flag→branch-taken predicate, (3) the `reg ± const` adder + the LEV pop-CAM.
The rest of this doc shows each is *substantially* generic and localizes the
residue.

---

## 1. INVENTORY — per op / per stage, the rules + heads that implement it

All paths are `neural_vm/unified_compiler/ops/`. "Head N" = attention head index
in that layer's block; layer→physical-block map in
`docs/PROBE_GROUNDTRUTH_2026_06_10.md` (27 logical layers, 37 physical blocks).
Rule COUNTS below are live (`_fn(100.0)` measured at golden `81557d21`).

### 1a. PC update — sequential PC-next (the default every op starts from)

| stage | mechanism | file:LOC | count |
|-------|-----------|----------|-------|
| L3 first-step PC default | `_register_default_ffn_rules` units 0-3: `MARK_PC and not HAS_SE → PC = PC_OFFSET+INSTR_WIDTH` (set + HAS_SE-keyed undo). | `l3_ops.py:191-229` | 4 |
| L3 PC increment lo/hi | units 86-101 (lo `(k+8)%16`) + 102-117 (hi copy), gated `MARK_PC ∧ HAS_SE ∧ ¬OP_LEV`. | `l3_ops.py:513-540` | 32 |
| L3 PC carry correction | units 118-133: lo-nibble ≥ `16-INSTR_WIDTH` carries into hi (`EMBED_LO[8..15]` → hi+1). | `l3_ops.py:542-562` | 16 |
| L3 seq PC byte-1 carry (flag) | `_pc_byte1_output_rules` + `_jsr_pc_byte1_seq_carry_stage_rules` (`C4_PC_BYTE1_CARRY` / `C4_JSR_PC_BYTE1`, DEFAULT-OFF; task #393). | `l3_ops.py:952-1071` | 0 off |
| L4 PC relay | `make_layer4_pc_relay_op` stages the live PC into `EMBED_*` for L3 to increment. | `l4_ops.py:165` | attn |

`_register_default_ffn_rules` is the big shared L3 FFN op (434 LOC, ~134 rules
total incl. the SP/BP/STACK0 marker-defaults that belong to MEMORY, not
CONTROL). The PC-increment core is the nibble-rotation adder `new = (old + 8) %
256` with a lo→hi carry: **the exact `reg + const` adder** from
`semantic_spec_MEMORY.md` §2c (there `SP ± 8`), here `PC + 8` gated `¬OP_LEV`
(LEV supplies its own PC from the return-addr, so the increment is suppressed).

### 1b. Branch-target PC — JMP / BZ / BNZ (the override)

| op | mechanism | file:LOC | count |
|----|-----------|----------|-------|
| JMP all-step override | `_layer6_all_step_jmp_pc_override_rules` units 320-383: cancel prev OUTPUT + copy target. | `l6_ops.py:507-...` | 64 |
| JMP first-step / delayed / AX-route bands | L6 routing FFN banks `first_step_jmp_pc_override` (256-320), `delayed_jmp_pc_override` (192-256), `jmp_ax_route` (160-192). | `l6_ops.py:86-103,218-227` | 96 |
| BZ taken override | `_post_l9_bz_pc_override_rules` (runs in an FFN block AFTER L9 so CMP is same-step-fresh). | `l6_ops.py:4797-4912` | 81 |
| BNZ taken override | `_post_l9_bnz_pc_override_rules` (2 groups: lo_nonzero / hi_nonzero). | `l6_ops.py:4915-4989` | 160 |
| dispatcher op | `make_post_l9_bz_bnz_pc_override_op` (pinned `after: layer10_alu`). | `l6_ops.py:5002-5098` | — |
| imm→byte-addr convert | `_append_pc_byte0_imm_to_byte_addr_rules`: `target = imm*INSTR_WIDTH + PC_OFFSET`, gate `FETCH_LO+k`. | `l6_ops.py:459-504` | 32/site |
| high-nibble ≥16 fix | `_append_branch_pc_byte0_odd_imm_hi_correction_rules` (`C4_IFVAR_BZ_HI_NIBBLE`, DEFAULT-ON): `+8` byte-0 hi for odd `FETCH_HI`. | `l6_ops.py:4724-...` | +48 |
| defensive gate | `make_branch_override_patch_op` (model post-pass, phase 1100): any FFN unit that fires on `MARK_PC ∧ OP_{JMP/LEV/BZ/BNZ}/CMP[0]` AND writes OUTPUT gets `-S` blockers for every OTHER opcode. | `model_ops.py:1667-1785` | scan |

**The branch-target computation is a single shared idea:** the branch immediate
lands in `FETCH_LO/HI` as a raw *instruction index* `i` (not a byte address);
the override cancels the sequential OUTPUT and writes `PC = i*8+2` per nibble via
a gate-on-`FETCH_LO+k` lookup table (`target_lo = (k*8+2)&0xF`,
`target_hi = ((k*8+2)>>4)&0xF`). BZ/BNZ add the CMP-flag gate on top of the same
JMP target machinery (see 1c). The `imm*8+2` table is identical for JMP, JSR
(§1d), BZ and BNZ — it is the `idx→pc` encoder from `constants.py:40`.

### 1c. CMP flag compute — EQ/NE/LT/GT/LE/GE → the branch-taken predicate

| stage | mechanism | file:LOC | count |
|-------|-----------|----------|-------|
| L8 CMP_GROUP flag | L8 marks the comparison-opcode group so L9 can gate on it (`SE_CMP_GROUP` relay). | `l8_ops.py` (group) | — |
| L9 CMP cascade | `_layer9_cmp_rules`: 4 cross-product banks — `hi_eq`(16→CMP+1), `lo_eq`(16→CMP+2), `hi_lt`(120→CMP+0), `lo_lt`(120→CMP+3), 4-way AND at `MARK_SE_ONLY`, gate `SE_CMP_GROUP`. | `l9_ops.py:552-685` | 272 |
| L10 cmp_combine | `_layer10_alu_cmp_combine_rules`: per-op 1 default + 1-3 override units combining the 4 CMP nibbles into the 0/1 AX result (EQ=`hi_eq∧lo_eq`, LT=`hi_lt ∨ (hi_eq∧lo_lt)`, …). | `l10_ops.py:1917-2050` | 18 |
| CMP-margin fix | `C4_CMP_FLAG_MARGIN_FIX` (DEFAULT-ON): raises the equal-high-nibble GT/GE decision margin (l10 line 2802; shared.py:761). | `l10_ops.py:2802`, `shared.py:761` | — |
| branch predicate | BZ reads `CMP+4/CMP+5` (the AX-zero flag), BNZ reads `¬CMP+4` / `CMP+4∧¬CMP+5` — see 1b override conditions. | `l6_ops.py:4825-4951` | — |

The CMP compute is a **two-nibble magnitude comparator**: L9 emits the four
raw relations (`hi_eq, lo_eq, hi_lt, lo_lt`) from per-nibble one-hot AND
cross-products (the same `a<b` / `a==b` enumeration for every compare op — NOT
per-op), and L10 `cmp_combine` composes them into each op's boolean via a tiny
truth-table (18 units for all six ops). The BZ/BNZ predicate then reads the
`AX==0` flag those compares wrote (`CMP+4/CMP+5` = the byte-0/carry zero
markers). So EQ..GE are **one comparator + a 6-row truth table**; the branch is
**one predicate over the AX-zero flag**.

### 1d. JSR — push return-addr + branch-target + SP decrement

| stage | mechanism | file:LOC | count |
|-------|-----------|----------|-------|
| JSR IS_JSR decode | L5 `_opcode_decode_first_step_rules` + `_opcode_decode_all_step_jsr_rules` write **`TEMP+0`** (IS_JSR scratch, NOT `OP_JSR`) at MARK_PC (`C4_NESTED_JSR_PC_FIX`, DEFAULT-ON). | `l5_ops.py:958-988` | 1 |
| JSR PC override | `_function_call_jsr_pc_override_rules`: 16 cancel-LO + 16 cancel-HI + 16 `FETCH_LO→target_lo` + 16 FETCH_HI (reserved/byte-1 stage) + 16 `FETCH_LO→target_hi`. Gate `MARK_PC ∧ TEMP[0] ∧ ¬{other opcodes} ∧ ¬IS_BYTE`. | `model_ops.py:382-489` | 80 |
| JSR host + relays | `make_function_call_weights_op` bakes the override FFN + the L6 H7 JSR-PC-OUTPUT relay + (flag) byte-1 relay head. | `model_ops.py:914-1067` | — |
| JSR PC byte-1 (flag) | `_jsr_pc_byte1_relay_head_spec` + `make_jsr_pc_byte1_emit_op` (`C4_JSR_PC_BYTE1`, DEFAULT-OFF; delivers target ≥ 0x100 — gcd/rec). | `model_ops.py:1141-1342` | 0 off |
| JSR SP decrement | L6 `_layer6_sp_decrement_rules` gated `CMP+4 ∧ MARK_SP` (shared with PSH). | `l6_ops.py:981-989` (per MEMORY §1b) | — |
| JSR AX passthrough | `_function_call_jsr_ax_passthrough_rules` copies `AX_CARRY→OUTPUT` at MARK_AX on OP_JSR (UNCONDITIONAL). Flag `C4_JSR_AX_CLEAN` (DEFAULT-OFF, task #350) appends a matched negative-clear band that zeroes the copy on the JSR step — the entry `JSR main` (step 0) amplifies an undefined WEAK AX_CARRY into a confident garbage step-0 AX (the gcd/rec cluster wall); ABI-safe since a call replaces AX with the callee's RETURN value. | `model_ops.py` | — |

JSR = **branch-target (same `imm*8+2` encoder as JMP/BZ)** + **push the
return-PC** (the return-addr `PC+8` that L3's sequential increment already
computed is pushed as the frame's saved PC) + **`SP -= 8`** (the shared
decrement). The one JSR quirk vs JMP is that it decodes to `TEMP+0` (IS_JSR)
not `OP_JSR` at the PC marker (per `semantic_spec_DECODE.md` §G3) so the PC
override can gate on it without the L5-attention opcode leak.

### 1e. ENT / LEV — frame setup + teardown

| stage | mechanism | file:LOC | count |
|-------|-----------|----------|-------|
| ENT SP decrement (BP save) | shared `_layer6_sp_decrement_rules` (`SP -= 8` for the pushed BP), same nibble adder. | `l6_ops.py:970-1040` | — |
| ENT `BP = SP` + `SP -= imm` | ENT frame-save store writes saved-BP to the new BP slot; frame-size imm subtracts from SP. | `l16_ops.py:1307+`, `l10_ops.py:9299+,11172+` | — |
| ENT AX-carry / frame-size imm | `make_l10_ent_axcarry_op` recovers the ENT frame-size immediate from the AX byte-0 dump (multilocal ENT). | `l10_ops.py:11277-11307` | ~30 |
| ENT-after-JSR SP fixup | `make_layer6_ent_after_jsr_sp_byte0_fixup_op` (prologue link JSR→ENT). | `l6_ops.py:2340,3124` | — |
| LEV addr relay (saved BP) | L9 head 0 `make_lev_addr_relay_op`: Q@MARK_SP gated OP_LEV, copy BP byte-0 → `ADDR_B0` (the key for the saved-BP pop-CAM). | `l9_ops.py:1453-1524` | attn |
| LEV BP→PC relay (return PC) | L9 head 1 `make_lev_bp_to_pc_relay_op`: same relay at MARK_PC for the return-addr pop. | `l9_ops.py:1527-...` | attn |
| LEV routing (the teardown) | `_layer16_lev_routing_rules`: `SP=BP` (`sp_bp_plus16` band), PC-cancel + PC-from-TEMP (return-addr materialize), AX restore-from-STACK0. Monolithic 865-unit bank. | `l16_ops.py:571-810` | 865 |
| LEV detector (disabled alt) | `control_flow_heads.make_lev_detector_head_op` — "form 2" attend-to-prior-LEV head, `enable=False` (byte-identical no-op). | `control_flow_heads.py:181-299` | 0 |

ENT = **`SP -= 8` (push BP)** + **`BP = SP`** + **`SP -= imm` (allocate
locals)** — two nibble-adder deltas plus a register copy. LEV = **`SP = BP`**
(copy) + **pop BP** + **pop PC** (two attend-by-address pop-CAMs into the freed
stack slots, the L9 relays keying the saved-BP/return-PC rows and L16 routing
materializing them). LEV is the *inverse* of ENT+JSR: it restores what they
saved. This is the single most rule-heavy CONTROL op (865 units) because the
teardown touches SP, BP, PC, AND AX in one step (see gap G5).

### 1f. ADJ / EXIT

| op | mechanism | file:LOC |
|----|-----------|----------|
| ADJ = `SP += imm` | `_layer6_adj_ax_route_rules` + `_layer6_adj_sp_writeback_rules` (the ADD ALU gated OP_ADJ; classified `FFN_COMPOSITE` by `opcode_mapper`). | `l6_ops.py:1273-1360` (per MEMORY §1a) |
| ADJ SP+8 (POP) | `make_binary_pop_sp_increment_op` (nibble rotate +8 with carry). | `l6_ops.py:4144` |
| EXIT AX-carry | `make_l10_exit_axcarry_op` (recovers exit-code into AX; terminal, no PC-next). | `l10_ops.py:12248-12278` |

ADJ is **pure ALU** — `SP += imm` via the nibble adder, no PC-next change, no
branch. It appears in CONTROL only because `opcode_mapper` groups it with the
frame ops; semantically it is `semantic_spec_MEMORY.md`'s `SpDeltaSpec(+, imm)`.
EXIT is terminal (halts); its only residual work is delivering the exit-code to
AX.

### 1g. LOC / count summary (CONTROL core)

| op | primary rule fns | live rule count | notes |
|----|------------------|----------------:|-------|
| seq PC-next | `_register_default_ffn_rules` (PC bands) | 52 | shared L3 FFN (434 LOC total, PC ≈ 52 units) |
| JMP | `_layer6_all_step_jmp_pc_override_rules` + 3 L6 banks | 160 | 4 PC-source bands (all-step/first-step/delayed/ax-route) |
| BZ | `_post_l9_bz_pc_override_rules` | 81 | +48 shared hi-nibble fix |
| BNZ | `_post_l9_bnz_pc_override_rules` | 160 | 2 groups (lo/hi nonzero) |
| JSR | `_function_call_jsr_pc_override_rules` | 80 | + IS_JSR decode + relays |
| CMP (all 6) | `_layer9_cmp_rules` + `_layer10_alu_cmp_combine_rules` | 272 + 18 | one comparator + 6-row truth table |
| LEV | `_layer16_lev_routing_rules` | 865 | SP=BP + pop BP + pop PC + AX restore |
| ENT | l10/l16 ent bands | ~60 | SP-=8, BP=SP, SP-=imm |
| ADJ | l6 adj bands | ~32 | pure ALU (SP += imm) |
| EXIT | `make_l10_exit_axcarry_op` | ~30 | terminal |
| defensive | `make_branch_override_patch_op` | model scan | zeroes spurious branch units |

**CONTROL rule mass ≈ 2000 units**, dominated by LEV (865, the 4-register
teardown), CMP (290, the comparator), and the branch-target overrides (~480
across JMP/BZ/BNZ/JSR). Almost all of it is loop-generated (one unit per nibble
`k`), NOT per-op hand code.

---

## 2. SEMANTIC-SPEC DATA — CONTROL as (PC-next, branch-condition, frame-delta)

Every CONTROL op reduces to a tuple over three sub-machines that already exist
as parameterized generators (the nibble-rotation adder + the CMP comparator +
the pop-CAM = `cam_lookup`). The data model:

### 2a. The data model

```
ControlOp {
  opcode:      JMP | JSR | BZ | BNZ | ENT | LEV | ADJ | EXIT | EQ..GE
  pc_next:     PcNextSpec        # WHERE the next PC comes from
  condition:   CondSpec | None   # gate on a CMP-derived flag (branch ops)
  frame_delta: FrameDeltaSpec[]  # ordered SP/BP mutations (0..3)
}

PcNextSpec (one of):
  SEQUENTIAL                     # PC = PC + INSTR_WIDTH   (the L3 default)
  BRANCH_TARGET(imm)             # PC = imm*INSTR_WIDTH + PC_OFFSET  (idx→pc encoder)
  RETURN_ADDR(pop_from=STACK)    # PC = popped saved-PC     (LEV; cam_lookup)
  TERMINAL                       # halt                     (EXIT)

CondSpec (one of):               # what makes the BRANCH_TARGET fire vs SEQUENTIAL
  NONE                           # JMP/JSR: always take the target
  AX_ZERO                        # BZ:  take target iff AX == 0   (CMP+4∧CMP+5)
  AX_NONZERO                     # BNZ: take target iff AX != 0   (¬CMP+4 ∨ ¬CMP+5)

FrameDeltaSpec (ordered):
  SP_DELTA(sign, amount=8|imm)   # nibble adder ±k
  BP_ASSIGN(source=SP)           # register copy BP = SP        (ENT)
  BP_RESTORE(pop_from=STACK)     # BP = popped saved-BP         (LEV; cam_lookup)
  SP_ASSIGN(source=BP)           # SP = BP                      (LEV; register copy)
  PUSH(value=PC|BP)              # emit a MEM/STACK0 save slot  (JSR pushes PC, ENT pushes BP)

CmpOp {                          # EQ..GE: produce AX_ZERO's input flag
  opcode:      EQ | NE | LT | GT | LE | GE
  result:      truth_table over (hi_eq, lo_eq, hi_lt, lo_lt)   # 6-row DATA
}
```

### 2b. Per-op instantiation (the data)

```python
JMP  = ControlOp(pc_next=BRANCH_TARGET(imm), condition=NONE, frame_delta=[])
JSR  = ControlOp(pc_next=BRANCH_TARGET(imm), condition=NONE,
                 frame_delta=[PUSH(PC), SP_DELTA(-, 8)])       # return-addr = seq PC+8
BZ   = ControlOp(pc_next=BRANCH_TARGET(imm), condition=AX_ZERO,    frame_delta=[])
BNZ  = ControlOp(pc_next=BRANCH_TARGET(imm), condition=AX_NONZERO, frame_delta=[])
ENT  = ControlOp(pc_next=SEQUENTIAL, condition=NONE,
                 frame_delta=[SP_DELTA(-, 8), PUSH(BP), BP_ASSIGN(SP), SP_DELTA(-, imm)])
LEV  = ControlOp(pc_next=RETURN_ADDR(STACK), condition=NONE,
                 frame_delta=[SP_ASSIGN(BP), BP_RESTORE(STACK)])  # +pop PC into pc_next
ADJ  = ControlOp(pc_next=SEQUENTIAL, condition=NONE, frame_delta=[SP_DELTA(+, imm)])
EXIT = ControlOp(pc_next=TERMINAL, condition=NONE, frame_delta=[])

# The compares FEED the branch condition — they are ALU for their VALUE
# (semantic_spec_ALU territory) and appear here only as the flag producer:
EQ = CmpOp(result = hi_eq AND lo_eq)                 # -> AX = 0/1, sets AX_ZERO
NE = CmpOp(result = NOT (hi_eq AND lo_eq))
LT = CmpOp(result = hi_lt OR (hi_eq AND lo_lt))
GT = CmpOp(result = NOT (hi_lt OR hi_eq_lo_le))      # symmetric via truth table
LE = CmpOp(result = hi_lt OR (hi_eq AND (lo_lt OR lo_eq)))
GE = CmpOp(result = NOT (hi_lt OR (hi_eq AND lo_lt)))
```

Every per-row field except `(opcode, pc_next-kind, condition, frame_delta-list)`
is a **band constant**: the `imm*8+2` encoder, the `±8` amount, the CMP
truth-table rows, the `MARK_PC/MARK_SP` gate, the `2.0/S` write scale. The ONLY
per-op DATA is those four tuple fields — a ~10-row table for the whole family.

### 2c. The recurring primitives — three generators, reused

1. **the nibble-rotation adder** (`reg ± const`) — `PC + 8` (L3 seq),
   `SP ± 8` (JSR/ENT/ADJ/POP), `SP -= imm` (ENT locals), and the `imm*8+2`
   branch-target encoder are ALL this adder (the branch encoder is `imm*8`
   = a fixed 3-bit left-shift + `+2`, expressed as a per-`k` lookup table
   `k → (k*8+2)`). Same chain the MEMORY family calls `FFN_COMPOSITE = ADD`.
2. **the CMP comparator** (`_layer9_cmp_rules`) — one two-nibble magnitude
   comparator emitting `(hi_eq, lo_eq, hi_lt, lo_lt)`, composed by a 6-row
   truth table (`cmp_combine`). Shared by all six compares AND the BZ/BNZ
   AX-zero predicate.
3. **the pop-CAM** (`cam_lookup`, per `semantic_spec_MEMORY.md` §2c) — LEV's
   `pop BP` and `pop PC` are attend-by-address loads of the saved-BP /
   return-PC stack slots (L9 relays key them, L16 routing materializes them).
   The SAME addressed-load primitive as LI/LC.

So CONTROL adds **no new mechanism** beyond MEMORY+ALU: it is a **PC-source mux**
(SEQUENTIAL default, overridden by BRANCH_TARGET or RETURN_ADDR) + a
**CMP-flag predicate** on the override + the three shared generators for the
frame math and the pops.

---

## 3. GAP-LIST — what resists generic derivation

Classified **SPEC** (missing data in the semantic model → add data) vs
**LOWERING** (the engine can't yet emit this weight shape → extend the lowering)
vs **FIRMWARE** (a hand-tuned edge-case patch not implied by the ISA).

### G1. PC-source mux = SEQUENTIAL-default + opcode override. **[LOWERING, solved-in-principle]**
The whole PC-next machine is "L3 writes `PC+8` unconditionally; the branch ops
CANCEL that OUTPUT (16 `-OUTPUT_LO[k]` gate units) and WRITE the target." This
cancel-then-rewrite idiom is uniform across JMP (`_layer6_all_step_jmp…`), BZ,
BNZ, and JSR — same 16-cancel + 16-target shape, differing only in the gate
(opcode + condition) and the target source. **Verdict:** a generic
`pc_override(gate, target_source)` lowering emits all four. The engine needs the
cancel band as a first-class primitive ("suppress the default writer's OUTPUT
before writing mine"), which is the same override idiom the MEMORY tail
correctors use — additive lowering work, not a semantic gap.

### G2. The `imm*8+2` branch-target encoder is generic DATA. **[SPEC, trivial → LOWERING]**
The branch immediate arrives as a raw instruction *index* `i` in `FETCH_LO/HI`;
the target byte address is `i*INSTR_WIDTH + PC_OFFSET`
(`constants.py:40`, `idx_to_pc`). Every branch op (JMP/JSR/BZ/BNZ) lowers this
identically via `_append_pc_byte0_imm_to_byte_addr_rules` (gate `FETCH_LO+k`,
write `(k*8+2)&0xF` lo / `((k*8+2)>>4)&0xF` hi). **Verdict:** the encoder is a
pure function of `(INSTR_WIDTH, PC_OFFSET)` — a per-`k` lookup table the engine
generates from two constants. Fully derivable; the ONLY subtlety is that the
model reads the index from `FETCH_LO`, so the engine must know the immediate is
an *index* not an address (one `imm_kind = INSTR_INDEX` spec field).

### G3. Branch-target PC ≥ 0x100 (byte-1) is a MISSING SPEC datum + off-by-default LOWERING. **[SPEC + LOWERING]**
This is THE known CONTROL gap (memory `project_gcd_pc_byte1_construction_verified`,
tasks #344 / #393 / #350). The default branch-target encoder only writes byte-0
(`(i*8+2) & 0xFF`); it drops the byte-1 nibble `((i*8+2) >> 8)`. For a target
index ≥ 32 (byte address ≥ 0x100) the high byte is lost → gcd/rec JSR-to-high,
if_var step-11 BZ-to-130 fail (id430: got pc=2 want pc=130). The construction
EXISTS and is VERIFIED behind flags (`C4_JSR_PC_BYTE1` for JSR,
`C4_PC_BYTE1_CARRY`/`C4_IFVAR_BZ_HI_NIBBLE` for BZ/BNZ) — flag-OFF golden
byte-identical, flag-ON gcd 900 goes bit-exact PC through the whole high-PC
region. **Verdict:** the SPEC must carry `pc_width = 2 bytes` (the target can
exceed 0x100) as a first-class field, and the LOWERING must emit the byte-1
stage+carry+emit band (a cross-step carry, task #393's `C4_PC_BYTE1_CARRY`)
unconditionally. The blocker to flipping default-ON is a SEPARATE co-located
root (step-0 JSR-to-main AX index leak, task #350), NOT the PC construction. A
generic engine that emits the full-width PC adder + branch encoder closes this
by construction; today's byte-0-only default is the LOWERING gap.

### G4. CMP = one comparator + a 6-row truth table. **[SPEC, closed]**
`_layer9_cmp_rules` (272 units) is a single two-nibble magnitude comparator
emitting `(hi_eq, lo_eq, hi_lt, lo_lt)` — the SAME cross-product enumeration for
every compare op (no per-op branches; the op-specific behavior is entirely in
L10 `cmp_combine`'s 18-unit truth table). **Verdict:** EQ..GE derive from
`(comparator, 6-row boolean truth-table)`; the truth table is pure DATA
(§2b). The comparator itself is the `a<b`/`a==b` nibble enumeration a generic
engine emits from "compare two 16-bit registers." The one FIRMWARE residue:
`C4_CMP_FLAG_MARGIN_FIX` (DEFAULT-ON) nudges the equal-high-nibble GT/GE
decision margin (l10:2802, shared.py:761) because the 30-token campaign frame
compresses the FLAG_EQ margin — a hand-tuned magnitude, not an ISA fact. Flag
that cell "non-derivable until the margin/scale convention is spec'd."

### G5. LEV is a 4-register atomic teardown — the CAM sequencing is a SPEC datum. **[SPEC + LOWERING]**
LEV (`_layer16_lev_routing_rules`, 865 units — the single biggest CONTROL op)
does `SP=BP` (register copy) THEN `pop BP` (CAM) THEN `pop PC` (CAM) IN ONE
STEP, and also restores AX from the freed STACK0 slot. The three pops read
DIFFERENT freed stack slots that only become valid AFTER `SP=BP` executes — an
ordering dependency (`sp_assign before the pops`, mirror of MEMORY G8's
"pop-then-store" for SI). The BP/PC pops are `cam_lookup` (L9 relays key the
saved-BP/return-PC address; L16 materializes), so the primitive is generic, but
the SPEC must record (a) the 4-way delta ORDER (`SP=BP; popBP; popPC; restoreAX`)
and (b) that each pop's address is the *running* SP after the prior delta.
**Verdict:** the pops are generic CAM; the residue is a `frame_delta` ORDERING +
running-SP dependency the engine must sequence (SPEC), plus emitting three
CAM reads whose keys depend on the intermediate SP (LOWERING). Memory
`project_root_a_lev_routing_bp_byte1_crush` documents the load-bearing L16/L25
BP-byte1 crush that makes this fragile today — a marker-convention artifact
(same class as MEMORY G4), not an ISA fact.

### G6. JSR decodes to IS_JSR (`TEMP+0`), not `OP_JSR`, at the PC marker. **[SPEC]**
Inherited straight from `semantic_spec_DECODE.md` §G3: the JSR PC override gates
on `TEMP+0` (the IS_JSR scratch flag) not `OP_JSR`, because L5 attention leaks
raw `OP_*` flags into subsequent-step MARK_PC rows and the strong opcode
blockers (`_function_call_jsr_pc_override_conditions`, `-4` per other opcode)
need a clean per-step flag. **Verdict:** one per-(opcode,context) `write_dim`
override cell in the DECODE spec (`JSR @ PC → TEMP+0`); the CONTROL engine reads
`condition_flag = TEMP0(IS_JSR)` for JSR's PC override instead of `OP_JSR`. This
is the resolved `jsr_simple_function_arch_blocked` finding — expressible, one
cell.

### G7. Same-step CMP freshness = an ordering constraint. **[LOWERING]**
BZ/BNZ must read the CMP flag the SAME step it was produced, but L6 (where the
legacy override lived) runs BEFORE L9 (the CMP writer), so on step 1 the
cross-step CMP alias resolved to 0 and the branch never fired
(`docs/CMP_PATH_AUDIT.md`). The fix (`make_post_l9_bz_bnz_pc_override_op`, pinned
`after: layer10_alu`) relocates the override to an FFN block after L9 so `CMP`
is same-step-fresh. **Verdict:** an ordering constraint (`branch_predicate after
cmp_compute`), NOT a new mechanism — but the generic engine's scheduler must
place the branch override strictly after the CMP producer. Same shape as MEMORY
G8 (store's address-compute after the pop).

### G8. Cross-step branch bookkeeping (BZ_TARGET_FRESH) + framing drift. **[FIRMWARE / SPEC]**
The BZ override carries a cross-step `BZ_TARGET_FRESH` bit (`C5` fix,
`docs/BZ_TARGET_FRESH_CROSS_STEP_2026_06_09.md`): a BZ-taken step writes the bit
so the NEXT step's cancel band doesn't spuriously subtract the just-written
target from the OUTPUT bank. This is a patch over the cross-step OUTPUT-residual
cancel interacting with a taken branch — NOT an ISA fact. Related: the
`if/bool/expr` "PC-wrong" cluster is FRAMING DRIFT, not a branch bug
(`project_if_bool_expr_is_stack0_highnibble_framing_drift`): the comparison step
emits ≠STEP_TOKENS tokens so the fixed-slice decoder misreads the (correct) PC —
task #388's ≠STEP_TOKENS root, orthogonal to CONTROL derivation. **Verdict:** the
cross-step cancel bookkeeping is FIRMWARE that a clean "override the default
writer" primitive (G1) with a proper step-local OUTPUT band should obviate;
the framing drift is a separate decode-frame SPEC property, not a CONTROL op.

### G9. The defensive branch-override patch is a whole-model post-pass. **[LOWERING / FIRMWARE]**
`make_branch_override_patch_op` (phase 1100) scans EVERY FFN unit and, if it
fires on `MARK_PC ∧ OP_{JMP/LEV/BZ/BNZ}/CMP[0]` AND writes OUTPUT, zeros its
`W_up` for every OTHER opcode. This exists because L5 attention leaks opcode
flags into later MARK_PC rows, so branch/LEV override units would fire
spuriously. **Verdict:** this is a *defensive* topology sweep, not per-op
semantics — a generic engine that emits opcode-exclusive gates on the override
units (strong `-S` blockers per non-target opcode, which the JSR override
already does inline via `_function_call_jsr_pc_override_conditions`) makes the
whole-model sweep unnecessary. It is a LOWERING convention (emit exclusive gates
at authoring time) that replaces a FIRMWARE post-pass.

### G10. ENT is two SP deltas + a register copy; ADJ/EXIT are trivial. **[SPEC, closed]**
ENT = `SP-=8; push BP; BP=SP; SP-=imm` — two `SP_DELTA` nibble-adder calls + a
`BP_ASSIGN(SP)` register copy + a `PUSH(BP)` store (the MEMORY store primitive).
ADJ = `SP += imm` (pure ALU, MEMORY `SpDeltaSpec`). EXIT = TERMINAL + AX
exit-code delivery. **Verdict:** all three are `frame_delta` lists over the
existing adder + store/copy generators; the ENT frame-size `imm` recovery
(`make_l10_ent_axcarry_op`) is the same "read the immediate" the MEMORY LEA/ADJ
use. No new mechanism; the residue is the SP marker firmware inherited from
MEMORY G4 (the `e0→e8`/`0xF8` byte-0 fixups on the ENT-after-JSR prologue link).

**Landed (2026-07):** the ADJ `SP_DELTA(+, imm)` / `SpDeltaSpec` is now a
first-class DSL kind. `SP += imm` adds a RUNTIME operand (the immediate in
`FETCH_HI`), which the compile-time `SequentialAddDelta` (`reg + const` nibble
rotation) cannot express, so the L9 ADJ hi-nibble band was a hand-authored
512-unit amplified adder shared-in-shape with LEA/ADD. The new
`RuntimeAddDelta` / `RegisterDeltaSpec(kind="runtime_add")` — the `SpDelta`
runtime adder — names the amount as a LIVE operand band and DELEGATES to the
shared L9 amplified nibble adder (`wide_alu_dsl.amplified_nibble_adder_rules`,
the same generator LEA/ENT call). `l9_ops._adj_control_op` expresses the whole
ADJ hi-nibble band as ONE `control_op` frame descriptor with a single
`RuntimeAddDelta` row; the hand-authored 512-unit cross-product loop is DELETED.
Byte-identical (golden hash unchanged at `91f55411`). The LEA/ADD-shared portion
of the amplified adder stays intact (LEA routes through the same shared
generator directly). `op="sub"` covers the ENT/JSR SP-DECREMENT-by-runtime case.

---

## 4. Bottom line — does CONTROL derive from a generic primitive?

**Yes, substantially — CONTROL is a PC-source MUX over three generators the
other families already own.** The family reduces to:

1. **a PC-source mux** — SEQUENTIAL (L3 `PC+8` adder) is the default; JMP/BZ/BNZ
   OVERRIDE it with BRANCH_TARGET (`imm*8+2` encoder), LEV with RETURN_ADDR
   (pop-CAM), EXIT with TERMINAL. The override is a uniform cancel-then-write
   idiom (G1); the target encoder is a two-constant lookup table (G2).
2. **a CMP-flag predicate** — one two-nibble comparator + a 6-row truth table
   produce the AX-zero flag (G4); BZ/BNZ gate their override on it (a
   same-step-fresh ordering constraint, G7).
3. **the three shared generators** — the nibble-rotation adder (all PC/SP/BP
   `±const` and the branch encoder), the CMP comparator, and the pop-CAM
   (`cam_lookup`, LEV's pops + JSR/ENT's pushes). CONTROL introduces **zero new
   mechanisms** beyond MEMORY + ALU.

The GAPS that are genuinely NOT derivable from the ISA one-liners:
- **G3 (branch-target byte-1 / PC ≥ 0x100)** — a required SPEC datum (`pc_width
  = 2`) + a LOWERING that emits the byte-1 carry band unconditionally. The
  construction is BUILT and VERIFIED behind flags; the default byte-0-only path
  is the last real CONTROL correctness gap (gcd/rec/if_var high branches). Its
  default-ON flip is blocked by a SEPARATE root (task #350), not the PC math.
- **G5 (LEV 4-register atomic teardown ordering)** — a SPEC `frame_delta` order
  + running-SP dependency; the pops themselves are generic CAM.
- **G6 (JSR → IS_JSR at PC)** — one DECODE write-override cell (inherited).
- **G4-margin / G8 / G9 (CMP margin, BZ cross-step bookkeeping, defensive
  override patch)** — FIRMWARE/marker-convention patches a clean override
  primitive + spec'd marker convention should mostly obviate.

Everything else (G1 mux idiom, G2 encoder, G7/G10 ordering + frame-delta lists)
is data fields / ordering constraints over the three existing generators. **The
family is a strong candidate for the generic engine: JMP/JSR/ADJ/EXIT and the
CMP comparator are decisively derivable today; BZ/BNZ derive once the branch
byte-1 (G3) + same-step-CMP ordering (G7) are spec'd; LEV derives once the
4-delta ordering (G5) is recorded.** Control flow IS a generic primitive — a
PC-source mux gated by opcode + a CMP-derived condition, with the frame math
delegated to the shared adder/CAM.

### Verification performed (READ-only; no weights touched)
* Golden hash unchanged: `81557d21` before and after (this doc reads ops + writes
  markdown only; `git status` shows no tracked changes).
* Live rule counts confirmed at golden: L9 CMP = 272, L10 cmp_combine = 18,
  L16 lev_routing = 865, post-L9 BZ = 81, BNZ = 160, JSR PC override = 80.
* Constants grounded: `INSTR_WIDTH=8`, `PC_OFFSET=2`, `STACK_ALIGNMENT=8`;
  `idx_to_pc(i) = i*8+2` (`constants.py:40`), so `imm*8+2` is the branch encoder.
* Flag defaults confirmed: `C4_NESTED_JSR_PC_FIX` ON, `C4_IFVAR_BZ_HI_NIBBLE` ON,
  `C4_CMP_FLAG_MARGIN_FIX` ON, `C4_JSR_PC_BYTE1` / `C4_PC_BYTE1_CARRY` OFF
  (byte-0-only golden default → the G3 gap).

### Feeds
This spec is the third input to the generic lowering engine (pilot #391).
The (PcNextSpec, CondSpec, FrameDeltaSpec) tuple + the CMP truth-table is the
reference schema the engine's control-family lowering must satisfy, alongside
`semantic_spec_DECODE.md` (the opcode→marker map) and `semantic_spec_MEMORY.md`
(the adder + CAM generators CONTROL reuses).
