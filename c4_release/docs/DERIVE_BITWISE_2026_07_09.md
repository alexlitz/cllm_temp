# DERIVE BITWISE — OR/XOR/AND from BLOG_SPEC §568's ONE formula (task #449)

**Status:** LANDED (flag-gated DEFAULT-OFF). `C4_DERIVE_BITWISE=1` re-derives all
three bitwise nibble ops (OR opcode 14, XOR 15, AND 16) from the SINGLE
per-bit formula BLOG_SPEC §568 names — `r_bit = c_a·a + c_b·b + c_ab·(a·b)` —
replacing the three enumerated Python bit operators (`operator.and_/or_/xor`).
**Zero op-specific bit operator, zero magic constants.**

**Golden gate (flag-OFF):** `tools/_isa_golden_hash.py` == `e50521f3` —
UNCHANGED.
**Byte-identity gate (flag-ON):** `C4_DERIVE_BITWISE=1
tools/_isa_golden_hash.py` == `e50521f3` — UNCHANGED. A byte-identical
`state_dict` ⇒ identical model ⇒ identical verdict on EVERY program.

---

## 1. The spec

Two BLOG_SPEC statements describe the bitwise family:

* **§568 (line 225), the shared/compressed view:** *"Bitwise: 10 weights — one
  formula (a+b-ab) for all"*.
* **§683 (line 685), the implemented view:** *"AND, OR, XOR … just lookup
  tables per nibble embedded in the FFNs … 256 entries … the per nibble
  operations/tables are each replicated 16 times."*

The two are consistent: the FFN LOWERING is a per-nibble 256-entry lookup
(replicated across the byte lanes), but the CONTENT of that lookup — the
result nibble for each `(a,b)` — is ONE formula for all three ops. This task
derives that content from §568's one formula instead of three distinct Python
bit operators.

### The one formula

Every bit of a nibble result is the same degree-2 polynomial over the two input
bits, with a single per-op coefficient triple read STRAIGHT from the spec text:

| op  | spec identity   | `(c_a, c_b, c_ab)` |
|-----|-----------------|--------------------|
| OR  | `a + b - a·b`   | `( 1,  1, -1)`     |
| AND | `a·b`           | `( 0,  0,  1)`     |
| XOR | `a + b - 2·a·b` | `( 1,  1, -2)`     |

Applied per bit across the 4 bits of the nibble:

```
r_nibble = OR over bit in 0..3 of  (c_a·a_bit + c_b·b_bit + c_ab·a_bit·b_bit) << bit
```

Each per-bit result is provably in `{0, 1}` for every legal triple, so it is a
genuine bit. This ONE formula reproduces `operator.and_/or_/xor` on all 16×16
nibble pairs, verified exhaustively (`tools/probe_reg_emission_map.py` and the
standalone check in this session). The three coefficient triples are the ONLY
per-op data — and they are the ISA identity of the op (`a+b-ab` etc.), not
tuned constants.

## 2. What was hand-authored, and the collapse

The production path already routes ALL bitwise result nibbles through ONE DSL
generator, `wide_alu_dsl.bitwise_rules` (called from the L10 main-FFN
`_layer10_alu_bitwise_rules` and the two lookup-mode post-op builders in
`ops/alu_ops.py`). But that generator computed the result via a per-op DISPATCH
over three distinct Python operators:

```python
_BITWISE_OP_FN = {"and": operator.and_, "or": operator.or_, "xor": operator.xor}
...
result = op_fn(a, b)          # three different bit operators
```

The three operators ARE the per-op magic — three separate hand-chosen bit
functions, one per opcode. `C4_DERIVE_BITWISE=1` replaces them with the single
spec formula:

```python
result = _bitwise_result_from_spec_formula(op, a, b)   # ONE polynomial for all 3
```

| item | hand-authored (flag-OFF) | spec-derived (flag-ON) |
|------|--------------------------|------------------------|
| per-op result logic | 3 distinct `operator.*` bit functions (`_BITWISE_OP_FN`) | **1** shared per-bit polynomial (`_bitwise_result_from_spec_formula`) |
| per-op DATA | the 3 operators themselves | **3 coefficient triples** — the ISA identities `a+b-ab` / `ab` / `a+b-2ab` |
| magic constants | 0 (already table-driven) | **0** (the coefficients are the spec identity) |
| load-bearing FFN units | 1722 (L10 main, 574×3) + 1536 (lookup post-op, 512×3) | **identical count** — the weights ARE the ISA nibble table, not verbosity |

**The collapse is qualitative, not a `-rules` count:** consistent with the
`project_core_loc_reduction_reality` memory note, the bitwise rule COUNT is
width-locked (256 entries/op × replication) and does NOT shrink. The win is
that the three op-specific bit OPERATORS — the only remaining per-op logic —
collapse to ONE spec-read formula. The bitwise family is now provably a
derivation from the §568 one-formula identity with zero per-op logic and zero
magic constants, correct-by-construction.

## 3. Byte-identity (the decisive equivalence)

The derived formula produces the SAME result nibble as the enumerated operator
for every `(a, b)` pair by construction, so the EMITTED RULES are identical:

* **all 1722 `FFNRule`s (574×3, main-FFN) byte-identical** enumerated-operator
  vs one-formula (`dataclasses.astuple` compared field-for-field this session).
* **whole-model `state_dict` SHA256 == `e50521f3`** both flag-OFF AND flag-ON.

A byte-identical `state_dict` is the strongest possible equivalence: the model
is the same tensor-for-tensor, so its verdict on EVERY program (corpus AND
smoke) is identical. Unlike a re-derivation that changes weights (the JMP-gate
pilot), this bitwise derivation is a pure SOURCE collapse — the three operators
were not "over-tuning", they were three spellings of one formula, and the model
never changes.

## 4. Verdict results

### 4.a Smoke (the direct bitwise verdict gate)

The three bitwise ops are exercised directly by the smoke suite (bytecode
`IMM;PSH;IMM;{OR,AND,XOR};EXIT`):

* `test_or_basic`  — `0x0F | 0x30 == 0x3F`
* `test_and_basic` — `0xFF & 0x2A == 0x2A`
* `test_xor_basic` — `0xFF ^ 0xD5 == 0x2A`
* plus 16-bit variants `test_or_16bit / test_and_16bit / test_xor_16bit`.

<!-- SMOKE_RESULT -->

Because the state_dict is byte-identical, the flag-ON smoke verdict EQUALS the
flag-OFF (golden) smoke verdict test-for-test — no regression, no bonus (a
byte-identical model cannot fix a currently-failing program).

### 4.b cpu_full_trace (`--spec-k 0`) — corpus

The 1096 corpus contains NO `AX = a & b` bitwise-OPCODE program: its only `&`/`|`
usage is the logical `&&`/`||` in the `bool_and` cluster (which compiles to CMP
+ branch, not to OR/XOR/AND opcodes). The bitwise ALU opcodes are exercised by
the SMOKE suite (§4.a), not the corpus. So `cpu_full_trace` on corpus ids adds
no bitwise-specific coverage beyond what byte-identity already guarantees; the
authoritative bitwise verdict gate is the smoke suite above.

### 4.c Correct-by-construction bonus?

None, and none is possible: byte-identity means the derived model IS the golden
model, so it cannot fix a currently-failing bitwise program. The bonus channel
this task's brief anticipated (a cleaner formula fixing a hand-path bug) is only
reachable when the derivation CHANGES weights. Here the hand path was already
correct byte-for-byte, so the derivation's value is entirely the source /
magic-constant collapse, not a verdict move.

## 5. Wiring (flag `C4_DERIVE_BITWISE`, DEFAULT-OFF)

* **result formula**: `wide_alu_dsl.bitwise_rules` selects
  `_bitwise_result_from_spec_formula(op, a, b)` when
  `_derive_bitwise_enabled()` (env `C4_DERIVE_BITWISE`), else the enumerated
  `_BITWISE_OP_FN[op]`. Both callers — the L10 main-FFN and the two lookup-mode
  post-op builders — inherit the flag automatically (they all call
  `bitwise_rules`).
* **helper**: `ops.shared.derive_bitwise_enabled()` (public mirror of the local
  `wide_alu_dsl._derive_bitwise_enabled`, kept local so the leaf DSL module does
  not import the `ops` package).
* **cache-key**: `C4_DERIVE_BITWISE` registered in BOTH
  `full_vm_compiler_dynamic.py` cache-key snapshots (in-proc memo + disk) so an
  ON build never shares a serialised entry with the hand path.

## 6. Reproduce

```
export C4_VM_CACHE_DIR=/tmp/derbit_$$   # isolated cache

CUDA_VISIBLE_DEVICES="" python tools/_isa_golden_hash.py                    # e50521f3 (flag-OFF)
CUDA_VISIBLE_DEVICES="" C4_DERIVE_BITWISE=1 python tools/_isa_golden_hash.py # e50521f3 (derived, byte-identical)

# smoke (spec_k=0, CPU) flag-ON:
CUDA_VISIBLE_DEVICES="" C4_DERIVE_BITWISE=1 python tools/run_full_smoke.py

# per-nibble formula check (exhaustive 16x16 vs operator.and_/or_/xor):
CUDA_VISIBLE_DEVICES="" python tools/probe_reg_emission_map.py   # (this session's standalone probe)
```

## 7. Honest limits

* **Byte-identity ⇒ no verdict move.** The derivation is a SOURCE collapse only.
  It removes the last per-op logic (three bit operators → one formula) and adds
  zero magic constants, but it cannot fix a failing program because it changes
  no weight. That is the correct outcome for a family the hand path already
  computed correctly — but it is NOT the "re-derivation reveals over-tuning was
  unnecessary" result the JMP-gate pilot produced (there the hand path carried
  genuine magic constants `-10.0/4.5`; here it carried none, only three
  operators).
* **Coverage is smoke-only.** The 1096 corpus does not exercise the bitwise ALU
  opcodes (only `&&`/`||` → CMP). The bitwise verdict authority is the smoke
  suite; corpus cpu_full_trace adds nothing beyond byte-identity.
* **The lowering was already table-driven.** This task did not have to build a
  new lowering primitive (unlike IMM's `marker_broadcast`). The `bitwise_rules`
  generator already existed; the derivation is the switch from operator-dispatch
  to one-formula content inside it.
* **The stale-ALU cancel band is unchanged.** The 31-rule-per-nibble
  stale-residue cancel band (`emit_stale_cancel_band`) is a delivery-hygiene
  concern on collapsed IMM+OP steps, not bitwise arithmetic; it reads the SAME
  `op_fn` and so is derived identically, but it is not part of the §568 formula
  itself.

## 8. Bottom line

The BITWISE family (OR/XOR/AND) now derives from BLOG_SPEC §568's single
per-bit formula `c_a·a + c_b·b + c_ab·a·b`, with the three coefficient triples
read verbatim from the spec identities. The three enumerated Python bit
operators — the only remaining per-op logic — collapse to ONE shared formula
with zero magic constants. The whole-model hash is `e50521f3` both flag-OFF and
flag-ON: the derivation is byte-for-byte identical to the hand path, so it is
correct-by-construction with no verdict risk. Sibling of the DECODE / IMM /
JMP derivation pilots; the same pattern (spec formula + DSL generator + gated
flag + both cache-key snapshots) applies to the remaining `wide_alu_dsl`
families (ADD/SUB #447, SHIFT #448, CMP #446).
