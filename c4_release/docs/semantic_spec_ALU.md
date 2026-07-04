# Semantic Spec — the ALU family (the hardest, decisive for core→20k)

**Status:** DESIGN / READ-only inventory (100%-derivable architecture prep,
parallel to the generic-lowering pilot #391, sibling of the DECODE map
(`docs/semantic_spec_DECODE.md`) and the CONTROL map #392).
**Golden gate:** no weight change. `tools/_isa_golden_hash.py` ==
`81557d21422f3eada0a87c677b00dced41cc26c3ee3bfb094c5eeb71c9b4d3cb`
(`81557d21`) before AND after this doc — this file only READS ops and
WRITES documentation. (Verified: hash identical pre/post.)

This document maps the CURRENT hand-authored lowering that COMPUTES each ALU
opcode (ADD/SUB/MUL/DIV/MOD/OR/XOR/AND/SHL/SHR), expresses each as
**semantic-spec DATA** (`AX = f(A, B)` arithmetic), and produces the
**derivation GAP-LIST** — the decisive question of whether a generic
"arithmetic-spec → FFN rules" engine can DERIVE the byte-wise compute
(carry / borrow / partial-product / long-division cascades) from a compact
spec, or whether each op needs bespoke weights.

**TL;DR — the ALU is the *split* case, not a uniform floor.** The ALU
divides cleanly into TWO regimes:

1. **DERIVABLE by a computed lowering (the majority):** ADD, SUB, the byte-0
   MUL schoolbook, the inter-byte ADD/SUB carry/borrow cascade, all three
   bitwise ops (OR/XOR/AND), and both shifts (SHL/SHR). Every one of these is
   **already** a Python `for a in 16: for b in 16: result = f(a,b)` loop that
   computes the result AT BUILD TIME and emits one FFN AND-rule per input
   combination. There is **zero per-value hand-tuning** — the arithmetic
   `f(a,b)` (mod-16 add, mod-16 sub, `(a*b)` schoolbook nibble, `a&b`, `a^b`,
   `a|b`, `n<<k`, `n>>k`) *is* the generator. A generic engine that can emit a
   "computed one-hot lookup rule per operand combination" reproduces this
   whole regime from a one-line arithmetic spec + a lowering shape.

2. **THE HARD FLOOR — a genuine spec-vs-lowering wall for WIDE MUL/DIV/MOD:**
   the multi-byte MUL partial-product carry-lookahead cascade
   (`FlattenedALUMul`, 9 stages) and the multi-byte DIV/MOD long-division loop
   (`FlattenedDivMod`, 8 outer × 3 inner iterations on a GE workspace) **cannot
   be a single-pass lookup**. A flat cross-product table for MUL grows as
   `16^(2·width)` (16.8M rules at width-3, 4.3B at width-4 — intractable);
   per-nibble DIV is *mathematically wrong* (`0xFF/0x0F=0x11` ≠ per-nibble
   `1` and a guard). The compute is DERIVABLE (the schoolbook / long-division
   *algorithm* is a compact spec), but the LOWERING is not single-pass — it
   needs a **multi-pass / GE-workspace IR construct that does not yet exist**
   (`multi_pass_rules`, see Path 1 in `docs/DSL_W5_MULDIV_LIMIT.md`). This is
   the one place the derivability program hits a real ceiling: the arithmetic
   is compact-spec-expressible, but the current single-forward FFN-rule
   lowering cannot express the cross-pass carry chain — a NEW lowering
   primitive is required, not a bigger table.

So the ALU verdict is **conditionally 100%-derivable**: fully derivable for
the 8-bit / byte-0 / bitwise / shift / carry-cascade core (the bulk of the
LOC), and derivable-with-a-new-multi-pass-lowering-primitive for wide MUL/DIV.
No op needs per-value bespoke weights; the wall is a LOWERING-generality gap
(multi-pass), not a SPEC-expressiveness gap.

---

## 0. Where the ALU lives (blocks / files) + the production path

The ALU compute is spread across **L8 → L14** (byte-0 nibble compute, carry
cascade, high-byte relay/clear) plus the **L25 tail correction bank** (block
41). The CANONICAL live path is `alu_mode="lookup"` — the production default
(`full_vm_compiler_dynamic.py:1791`). Under `lookup`:

* **byte-0 ADD/SUB** = the L8 `layer8_alu` FFN (computed nibble lookup).
* **inter-byte ADD/SUB carry/borrow** = the L10 declarative `CarryPropagation`
  rules (computed `lo+hi*16+1` per pair).
* **bitwise OR/XOR/AND** = the L10 `_layer10_alu_bitwise_*` rules.
* **MUL** = L11 `mul_partial` (schoolbook partial nibbles) + L12 `mul_combine`
  (+ the L14 high-byte relay + `MUL_RESULT_HI` band for width-2).
* **DIV/MOD** = the `FlattenedDivMod` composite (GE-workspace long division),
  installed as a `blocks[10].post_ops` entry.
* **SHL/SHR** = the L13 `layer13_shifts` FFN (computed per-byte shift lookup).

The `alu_mode="efficient"` path swaps in the `efficient_alu_*.py` composites
(`AddSub5StageBlock`, `FlattenedALUMul`, `ALUShiftComposite`, `ALUAndOrXor`)
and, for DIV, the byte-accurate `wide_div_rules_ge_format` lookup. **Efficient
mode is NOT the production golden** — it is the pre-DSL / alternate composite
path (see the `if alu_mode == "efficient"` branch at
`full_vm_compiler_dynamic.py:2839`). This spec documents the LOOKUP path
(golden) as authoritative and treats the composites as the multi-byte
fallback the DSL has not yet subsumed.

### The generic substrate already present (`isa_semantics_dsl.py` + `wide_alu_dsl.py`)

Per the brief, the substrate already carries parameterized generators — but
they cover the DERIVABLE regime only:

| Generator | File | What it derives | Regime |
|-----------|------|-----------------|--------|
| `bitwise_rules(op=and/or/xor)` | `wide_alu_dsl.py:58` | 512 rules/op, `result=a op b` per nibble pair | fully derivable |
| `wide_add_rules` | `wide_alu_dsl.py:166` | per-byte `(a+b+cin)%16`, carry cascade via `carry_base+b` dims | fully derivable |
| `wide_sub_rules` | `wide_alu_dsl.py:400` | per-byte `(a-b-bin)%16`, borrow cascade | fully derivable |
| `wide_shift_rules` | `wide_alu_dsl.py:624` | per-byte `(n<<k)&0xFF` / `n>>k` | fully derivable |
| `wide_mul_rules` | `wide_alu_dsl.py:737` | flat `a*b` lookup, **width ∈ {1,2} ONLY** | derivable to 8-bit; wall past |
| `wide_div_rules` | `wide_alu_dsl.py:1010` | per-nibble `a//b`,`a%b`, **width=1 ONLY** | POC; wrong multi-byte |
| `wide_div_rules_ge_format` | `wide_alu_dsl.py:1170` | flat 256×256 byte-accurate `a//b` | derivable to 8-bit; wall past |
| `_nibble_rotation_chain_rules` | `l4_ops.py:552` | `target = source + CONST` w/ inter-nibble carry (PC+K) | **constant-offset adder ONLY** — not a 2-operand ALU adder |
| `cam_lookup` | `isa_semantics_dsl.py:2082` | content-addressed operand gather (feeds the ALU its A/B) | fetch, not compute |

Note the two brief-cited pieces precisely:
`_nibble_rotation_chain_rules` is the **adder for a compile-time CONSTANT
offset** (PC+1..+4, address arithmetic) — it rotates a nibble one-hot by a
fixed `k` and cascades the resulting carry. It is NOT the ADD opcode (which
adds two *runtime* operands); the runtime two-operand adder is `wide_add_rules`
(a full `a×b` lookup) / the L8 `add_lo` loop. `cam_lookup` re-expresses L7
operand-gather (it DELIVERS A and B to the ALU bands) but computes nothing.

---

## 1. INVENTORY — per ALU op, the live-path compute + rule/unit LOC

Rule counts are LIVE (`_layerN_*_rules(100.0)` executed; see §4). "LOC" here =
FFN hidden UNITS (one per rule), the load-bearing count — the *source* lines
are far fewer because each family is a `for a: for b:` loop.

### 1.a The byte-0 / bitwise / shift compute (all COMPUTED lookups)

| Op(s) | Live producer (block) | File:sym | Units | Compute (build-time) |
|-------|----------------------|----------|------:|----------------------|
| ADD/SUB byte-0 (+ LEA/ADJ/ENT/LEV/CMP co-tenants) | L8 `layer8_alu` FFN | `l8_ops.py:1317` `_layer8_alu_rules` | **2023** | `add: (a+b)%16`, `sub: (a-b)%16`, carry when `a+b≥16` |
| OR | L10 `_layer10_alu_bitwise_or` | `l10_ops.py:2185` | 574 | `a\|b` per nibble |
| XOR | L10 `_layer10_alu_bitwise_xor` | `l10_ops.py:2193` | 574 | `a^b` per nibble |
| AND | L10 `_layer10_alu_bitwise_and` | `l10_ops.py:2201` | 574 | `a&b` per nibble |
| CMP (EQ/NE/LT/GT/LE/GE) ordering* | L10 `_layer10_alu_ordering_engine` | `l10_ops.py:2652` | 272 | comparison-flag lookup |
| SHL/SHR | L13 `layer13_shifts` FFN | `l13_ops.py:277` `_layer13_shifts_rules` | **4096** | `(n<<k)&0xFF` / `n>>k` per (byte,amount,value) |

\* CMP is its own family (feeds branches) but shares the L8/L10 ALU bands and
the operand-gather; listed for completeness. The comparison spec is a separate
"CMP family" doc target; here it is a co-tenant of the ALU band.

### 1.b The MUL compute (schoolbook, split byte-0 vs high-byte)

| Stage | Live producer (block) | File:sym | Units | Compute |
|-------|----------------------|----------|------:|---------|
| MUL partial products | L11 `mul_partial` | `l11_ops.py:388` `_mul_partial_rules` | **4096** | `partial = ((a_lo*b_lo)//16 + a_lo*b_hi) % 16` per (a_lo,b_lo,b_hi) |
| MUL combine | L12 `mul_combine` | `l12_ops.py:199` `mul_combine_rules` | (triple-loop) | sum staggered partials into result nibbles |
| MUL high-byte relay (width-2) | L14 `layer14_alu_high_byte_relay` | `l14_ops.py:1543` | (relay head) | route byte-1 → `MUL_RESULT_HI` band → AX_FULL |

The width-2 (8-bit × 8-bit → 16-bit) MUL result byte-1 uses the dedicated
`MUL_RESULT_HI_LO/HI` residual band (registered op-locally in `alu_ops.py:24`,
flag `C4_MUL_WIDTH2` default-ON) to avoid clobbering `OUTPUT_LO+32`
(= ADDR_KEY).

### 1.c The DIV/MOD compute (long division — NOT a lookup on the golden path)

| Stage | Live producer | File:sym | Form | Compute |
|-------|--------------|----------|------|---------|
| DIV/MOD composite | `blocks[10].post_ops` | `efficient_alu_divmod_split.py:318` `FlattenedDivMod` | 4 stage ops + 1 install (`alu_ops.py:2174` `make_alu_divmod_composite_ops`) | GE-workspace **8 outer × 3 inner** MSB→LSB shift-and-subtract long-division loop |

`FlattenedDivMod` operates on the GE workspace `[B,seq,8,160]` (a 9-nibble
accumulator that lives only in GE) — it is an `nn.Sequential` of
BDToGE → DIV pipeline → MOD pipeline → GEToBD stages. This is the ONE ALU op
with no computed-FFN-rule form on the golden path; the DSL's
`wide_div_rules_ge_format` (efficient mode) is a flat 256×256 lookup that only
handles single-byte dividends.

### 1.d The inter-byte carry/borrow cascade (the memory "#1 wall") — DERIVABLE

The multi-byte ADD/SUB carry/borrow — memory-note
`project_sub16_mul16_var_root_is_l14_borrow_cascade`'s crux — is on the LIVE
path as **declarative computed rules**, NOT a hand-authored imperative cascade:

| Cascade | Live producer (block) | File:sym | Units | Compute |
|---------|----------------------|----------|------:|---------|
| Inter-byte ADD/SUB (byte 0→1→2) | L10 declarative `CarryPropagation` ×3 | `l10_ops.py:1393` `_l10_carry_propagation_rules` | **512 × 3** = 1536 | ADD: `new = lo + hi*16 + 1`; SUB: `-1`; per (lo,hi) pair, per byte |
| Binary-op byte zeroing | L10 `_l10_binary_op_byte_zeroing` | `l10_ops.py:1281` | 8 | zero AX bytes 1..3 on no-carry ops |
| L14 SUB no-borrow high-byte passthrough | L14 `layer14_sub_noborrow_high_byte_passthrough` | `l14_ops.py:3563` | — | pass minuend byte-1 when no borrow |
| L14 SUB full-borrow flag | L14 `layer14_sub_full_borrow_flag` | `l14_ops.py:3736` | 1 | `minuend byte1==0 ∧ borrow-in` indicator |

The declarative `_l10_carry_propagation_rules` proves the byte-wise carry
cascade is **build-time computable** (`new_val = lo + hi*16 ± 1`, per pair) —
the inter-byte propagation is expressed via the `carry_base+b` cascade dims
that the next byte's rule reads. This is the same "carry-base dim" mechanism
`wide_add_rules` uses. The L14 `_ops` correctors around it (`l14_ops.py`,
~24 ALU-related ops) are **band-hygiene** (clear the AX high bytes on no-carry
ops, pass the SUB minuend on the no-borrow path) — corrective, not the compute.

### 1.e The L25 tail result-correction bank (block 41) — DERIVABLE (byte-value engine)

| Bank | File:sym | Rules | Shape |
|------|----------|------:|-------|
| `tail_bit32_result_correction` | `l10_ops.py:6043` `_tail_bit32_result_correction_rules` | **2063** | ~40 families, each a `for lo: for hi:` byte-value writeback (`value = lo\|(hi<<4)`) under a fixed evidence gate |

Already consolidated (task J5/K1, `l10_ops.py:5960` `_evidence_keyed_byte_value_writeback`)
into ONE parametrized engine: each of the ~40 families is *"under evidence gate
`base_conditions`, for each byte value `v` observed in a (lo,hi) source lane,
guarantee OUTPUT byte = v"*. This is a **generic byte-copy-with-evidence
primitive**, fully derivable — the per-family DATA is just `(base_conditions,
source lane, strength)`, not per-value logic.

### 1.f Total live ALU-family unit footprint

Summing the live counts above: L8 alu 2023 + L10 bitwise 3×574 + L10 CMP-order
272 + L10 carry-prop 3×512 + L10 binzero 8 + L11 mul-partial 4096 + L13 shifts
4096 + L25 tail 2063 ≈ **16.3k FFN units** for the FFN-expressible ALU core
(the biggest single chunks are L11 mul-partial and L13 shifts at 4096 each,
then the L25 tail bank and L8 alu at ~2k). DIV/MOD adds the `FlattenedDivMod`
GE pipeline (not unit-counted the same way — it is a workspace composite).
This ~16k is a large share of the ~65k ISA-ops core the memory note
(`project_core_loc_reduction_reality`) flags as load-bearing.

---

## 2. SEMANTIC-SPEC DATA — each op as `AX = f(A, B)` arithmetic

C4 ALU semantics (`AX` = accumulator, operand A = stack top, B = AX):

| Op | Arithmetic (`AX = f(A,B)`) | Byte-wise realization | Spec is compact? |
|----|-----------------------------|------------------------|------------------|
| ADD | `AX = A + B` (mod 2³²) | per-byte `(a+b+cin)%256`, `cout = sum≥256` | YES — one line |
| SUB | `AX = A - B` (mod 2³²) | per-byte `(a-b-bin)%256`, `bout = a<b+bin` | YES — one line |
| MUL | `AX = A * B` (mod 2³²) | schoolbook Σ partial products + column carry cascade | YES (algorithm), NO single-pass |
| DIV | `AX = A / B` (÷0 → convention) | MSB→LSB long division (shift-subtract-select) | YES (algorithm), NO single-pass |
| MOD | `AX = A % B` | remainder of the same long division | YES (algorithm), NO single-pass |
| OR  | `AX = A \| B` | per-nibble `a\|b` (bit-independent) | YES — one line |
| XOR | `AX = A ^ B` | per-nibble `a^b` (bit-independent) | YES — one line |
| AND | `AX = A & B` | per-nibble `a&b` (bit-independent) | YES — one line |
| SHL | `AX = A << B` | per-byte `(n<<k)&mask`, inter-byte bit spill | YES (per-byte lookup; spill = future) |
| SHR | `AX = A >> B` | per-byte `n>>k`, inter-byte bit spill | YES (per-byte lookup; spill = future) |

The spec each op needs is a **`(f, carry_rule, width)` triple**, where `f` is
the per-lane arithmetic (`add`/`sub`/`and`/`or`/`xor`/`shift`/`mul_nibble`/
`div_step`) and `carry_rule` names how a lane's overflow feeds the next
(`carry_base+b` for ADD/SUB, column-cascade for MUL, iteration-accumulator for
DIV). The operand *source* bands (`ALU_LO/HI`, `AX_CARRY_LO/HI`, `FETCH_LO`)
and result *dest* bands (`OUTPUT_LO/HI`, `MUL_RESULT_HI`) are spec fields, but
they are constant per op-family (~6 lines), not per-value. The `f` for the
derivable regime is literally the Python operator — the generator IS the spec.

### The lowering shape (the SAME for the whole derivable regime)

```
for each operand combination (a, b) in the lane cross-product:
    result = f(a, b, carry_in)                       # BUILD-TIME compute
    emit multi_way_and_rule(
        conditions = [marker, A_band+a, B_band+b, (carry_base+(b-1) if b>0)],
        threshold  = balanced-AND,
        gate       = OP_<NAME>,
        writes     = [result_band + result, (carry_base+b if carry_out)],
    )
```

Every derivable op is this loop with a different `f` and lane count. A generic
engine that emits this shape from `(f, bands, width)` reproduces ADD, SUB, OR,
XOR, AND, SHL, SHR, byte-0 MUL, and the carry cascade byte-identically.

---

## 3. GAP-LIST — what a generic engine CAN and CANNOT derive

Convention: **[spec-expressiveness]** = a field the spec must carry;
**[lowering-generality]** = whether the generic "spec → FFN rules" engine emits
it without op-specific code.

### G1. ADD/SUB/OR/XOR/AND/SHL/SHR/byte-0-MUL — FULLY DERIVABLE. [lowering: clean]
Each is a computed one-hot cross-product lookup: `result = f(a,b)` evaluated at
build time, one FFN AND-rule per combination, uniform balanced-AND gate. Zero
per-value hand-tuning. The `wide_alu_dsl.py` generators ALREADY do exactly this
(`wide_add_rules` / `bitwise_rules` / `wide_shift_rules`) and are byte-identity
unit-tested. **A generic engine reproduces the entire bitwise + shift + 8-bit
add/sub + byte-0 mul band from a one-line `f` per op.** This is the decisive
"most of the ALU is a computed lookup" result — and it is the *majority* of the
~16k ALU units.

### G2. Inter-byte ADD/SUB carry/borrow cascade — DERIVABLE via a cascade-dim. [spec: 1 field]
The carry propagation (memory's "#1 wall") is derivable: each byte's rule reads
`carry_base+(b-1)` and writes `carry_base+b`, with `new = lo+hi*16±1` computed
at build time (`_l10_carry_propagation_rules`, `wide_add_rules`). The spec
needs ONE extra field — the `carry_base` dim + the "carry_in / carry_out /
suppress-when-carry" three-way rule split — which is a *fixed pattern* the
engine emits for any op declaring `carry_rule="ripple"`. It is NOT a single-
forward self-cascade (a lane cannot read a carry its own forward writes), so it
lowers as one FFN PASS PER BYTE (the L8 lo/hi two-pass, the L10 ×3 byte
instances). That per-byte-pass expansion is a **lowering pattern**, derivable
from `(width, carry_rule)`, not per-op code. The historical "L14 borrow cascade
reads wrong minuend" bug is a *band-routing* corrector (SUB minuend from
`STACK0_BYTE_VAL_{b+1}` not OUTPUT), which the spec carries as the operand-
source field — a data fix, still derivable.

### G3. WIDE MUL (width > 2) — THE HARD FLOOR. [lowering: NEW multi-pass primitive required]
A flat operand cross-product lookup grows as `16^(2·width)`: width-2 = 65,536
rules (tractable, the DSL's ceiling), width-3 = 16.8M, width-4 = 4.3B —
intractable, and `W_up` can only express a bounded number of hyperplanes per
layer. Wide MUL needs the `FlattenedALUMul` **9-stage schoolbook pipeline**
(partial products → 3 carry passes → generate/propagate → binary carry-
lookahead → final correction → combine), whose cross-STAGE carry chain a
single-forward FFN lookup **fundamentally cannot express**. The MUL *algorithm*
is a compact spec (schoolbook is ~10 lines), but the current single-pass
FFN-rule lowering cannot emit it — a **`multi_pass_rules` IR construct**
(emit a sequence of FFN passes with explicit inter-pass residual propagation,
Path 1 in `docs/DSL_W5_MULDIV_LIMIT.md`) is required. This is a genuine
**[lowering-generality] gap**, not a spec gap: the spec is expressible, the
lowering primitive does not yet exist.

### G4. WIDE DIV/MOD — THE HARD FLOOR. [lowering: NEW multi-pass primitive required]
Per-nibble independent division is **mathematically wrong** for multi-byte
(`0xFF/0x0F=0x11`, but per-nibble gives `1` and a ÷0 guard → `0xF0`): the
quotient/remainder is a non-local function of the whole dividend/divisor. Wide
DIV needs `FlattenedDivMod`'s **8-outer × 3-inner long-division loop** on a
9-nibble GE accumulator — a bit-serial shift-subtract-select cascade whose
cross-iteration data dependency cannot flatten to a single-pass lookup. As with
MUL: the long-division *algorithm* is compact, but the lowering needs the
`multi_pass_rules` (or GE-format) primitive. The single-byte case IS derivable
(`wide_div_rules_ge_format`, a flat 256×256 byte-accurate lookup), so DIV/MOD is
derivable at 8-bit and floored past it — the SAME boundary as MUL.

### G5. SHL/SHR inter-byte bit-spill — DERIVABLE but currently per-byte-only. [spec: 1 field]
The live shift is per-byte independent (`wide_shift_rules` "no inter-byte
propagation"); a full 32-bit shift spills bits ACROSS byte boundaries, which
needs the same carry-dim cascade as G2 (a lane reads the neighbor's spilled
bits). Derivable via the cascade-dim pattern; the current build just hasn't
wired the spill (it targets the 8-bit-in-a-byte cases that dominate the corpus).
Not a floor — a `carry_rule="bit_spill"` the engine can emit, same shape as G2.

### G6. Operand delivery is a SEPARATE (non-compute) primitive. [scoping]
The ALU reads its operands from `ALU_LO/HI` (A) + `AX_CARRY_LO/HI` (B) +
`FETCH_LO` (LEA). Producing those clean one-hots is the L7 operand-gather /
`cam_lookup` job (memory: the "dirty operand" index-0 artifact,
`_ADDSUB_OPERAND_A_W`/`operand_a_artifact_blocker_weight`). The ALU-compute
engine should treat the operand bands as GIVEN INPUTS; the artifact-blocker
weights are a *delivery-cleanup* concern, not ALU arithmetic. A generic ALU
engine emits clean-operand rules; the dirty-operand blockers are a separate
"operand hygiene" spec the delivery family owns. (This is why the DSL
generators default `operand_a_artifact_blocker_weight=0.0` — the clean form is
the byte-identity target; blockers are added only at the dirty MARK_AX install.)

### G7. The L25 tail correction bank — DERIVABLE (evidence-keyed byte-copy). [lowering: clean]
The 2063-rule tail bank is ALREADY a single parametrized engine
(`_evidence_keyed_byte_value_writeback`): ~40 families each *"guarantee OUTPUT
byte = the value observed in a (lo,hi) source lane, under a fixed evidence
gate."* The per-family DATA is `(base_conditions, source_lane, strength)` — a
generic byte-copy-with-evidence primitive, fully derivable. It is corrective
band-hygiene (re-assert a byte that an upstream leak crushed), NOT ALU
arithmetic, but it lowers cleanly from a compact family table.

### G8. Unit ORDER / band routing is byte-identity-load-bearing. [lowering constraint]
As with DECODE-G5: the FFN unit order + the residual band each lane writes
(`OUTPUT_LO+32` = ADDR_KEY collision → the `MUL_RESULT_HI` reroute; SUB borrow
→ `CARRY+2` not `CARRY+1`; the L14 clear-chain layout) are byte-identity
constraints. The *behavior* is order-independent (rules gate disjointly) but the
*weights* are not. A generic engine reproduces the hash only if it emits banks
in the declared order with the declared band routing — a lowering constraint,
not a derivability gap.

---

## 4. VERDICT + reproducibility

**Is the ALU 100%-derivable from a compact arithmetic spec + a computed
lowering, or a hard floor? — BOTH, cleanly split, and this is decisive for
core→20k:**

* **DERIVABLE (the bulk, ~16k units): YES.** ADD, SUB, all bitwise, both
  shifts, byte-0 MUL, and the inter-byte carry/borrow cascade are ALREADY
  computed lookups (`result = f(a,b)` at build time, one rule per combination)
  — the arithmetic *is* the generator, zero per-value bespoke weights. The
  `wide_alu_dsl.py` generators prove this byte-identically. A generic engine
  reproduces this entire regime from a `(f, bands, width, carry_rule)` spec of
  ~6 lines per op. **This confirms the ALU is NOT a per-op hand-weight floor
  for the 8-bit / byte-wise core** — it is a computed-lowering target, exactly
  the core→20k thesis.

* **HARD FLOOR (wide MUL/DIV/MOD only): a LOWERING-generality gap, not a
  spec-expressiveness gap.** Multi-byte MUL (partial-product carry-lookahead)
  and DIV/MOD (long division) cannot be single-pass FFN lookups — the flat
  table is intractable (`16^(2·width)`) and per-nibble DIV is mathematically
  wrong. The compute is compact-spec-expressible (schoolbook / long-division
  algorithms), but the current single-forward FFN-rule lowering cannot emit the
  cross-pass carry chain. Closing it needs a **NEW `multi_pass_rules` (or
  GE-format) IR primitive** — several sessions of DSL surface work
  (`docs/DSL_W5_MULDIV_LIMIT.md`), after which wide MUL/DIV also become
  derivable. Until then `FlattenedALUMul` / `FlattenedDivMod` remain the
  authoritative multi-byte weights.

**Net for the derivability program:** the ALU is the *split-verdict* member.
Unlike DECODE (a clean lookup, fully derivable) it has a real ceiling — but the
ceiling is a **single missing lowering primitive** (multi-pass cascade), NOT a
scatter of per-op bespoke weights. Once `multi_pass_rules` exists, the ALU is
100%-derivable from compact arithmetic specs. The core→20k reduction is
therefore reachable for the ALU **conditional on building one multi-pass
lowering construct**; the 8-bit/byte-wise/bitwise/shift/carry-cascade majority
is derivable TODAY.

### Verification performed (READ-only; no weights touched)
* Live rule counts executed (S=100.0): L8 alu **2023**, L10 bitwise OR/XOR/AND
  **574 each**, L10 CMP-order **272**, L10 carry-prop **512/instance** (×3),
  L10 binzero **8**, L11 mul-partial **4096**, L13 shifts **4096**, L25 tail
  **2063**.
* Confirmed the DERIVABLE generators compute `f(a,b)` at build time:
  `_layer8_alu_add_lo_rules` (`(a+b)%16`), `_layer10_alu_bitwise_*`
  (`a op b`), `_layer13_shifts` (`(n<<k)&0xFF`), `_mul_partial_rules_for_a_lo`
  (`(carry + a_lo*b_hi)%16`), `_l10_carry_propagation_rules` (`lo+hi*16±1`).
* Confirmed the FLOOR: `wide_mul_rules` raises `NotImplementedError` for
  `width_bytes>2`; `wide_div_rules` raises for `width_bytes>1` (with the
  `0xFF/0x0F` counter-example); `FlattenedDivMod` is the GE long-division
  composite on the golden path.
* Confirmed production `alu_mode="lookup"` (default at
  `full_vm_compiler_dynamic.py:1791`); efficient composites gated on
  `alu_mode=="efficient"`.
* Golden hash unchanged: `81557d21` (this doc reads ops + writes markdown only).

### Feeds
This spec + the DIV/MUL floor is the input to the generic lowering engine
(pilot #391) and the sibling of `docs/semantic_spec_DECODE.md` (fully
derivable) / CONTROL (#392). The decisive delta from DECODE: DECODE needs only
a lookup engine; the ALU additionally needs a **multi-pass cascade lowering
primitive** (`multi_pass_rules`) before wide MUL/DIV are derivable. That
primitive is the single highest-leverage DSL investment for the core→20k ALU
reduction.
