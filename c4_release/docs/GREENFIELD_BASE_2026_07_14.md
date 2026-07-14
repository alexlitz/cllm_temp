# Green-field Base Integration — 2026-07-14

Consolidates the scattered green-field minimal-ISA pieces into **one coherent
`greenfield-base`** branch, resolves the two DESIGN.md versions, wires the
per-op oracle harness to the **real** substrate compiler through a thin adapter,
and records the per-op **fan-out baseline** table. This unblocks the opcode
fan-out.

Branch: `greenfield-base` (off `greenfield-substrate`). Do **not** merge to main.

---

## 1. Branches merged

| Branch                          | Contributes                                   | Status |
|---------------------------------|-----------------------------------------------|--------|
| `greenfield-substrate` (base)   | the working compiler (`isa/layout/model/dsl/compile_ffn/compile_attn/compiler/test_slice` + DESIGN.md) | base |
| `greenfield-isa-spec`           | `c4_min/ISA_SPEC.md`                           | merged (no-ff) |
| `greenfield-decode-robustness`  | `c4_min/DECODE_ROBUSTNESS.md`                  | merged (no-ff) |
| `greenfield-oracle-harness`     | `oracle.py`, `run_oracle.py`, `test_oracle_harness.py` + a 2nd DESIGN.md + `__init__.py` | merged (no-ff), conflicts resolved |
| `greenfield-corpus`             | *(expected: CONFORMANCE_CORPUS.md, conformance_ids.json)* | **SKIPPED — pending** |
| `greenfield-baking-primer`      | *(expected: BAKING_PRIMER.md, baking_examples.py)*        | **SKIPPED — pending** |

### Lineage caveat (important for future merges)

All sibling branches were cut from the **main lineage** (`e45eb4ff`), whose
`c4_min/` is an *older, different* directory — NOT from the substrate. Each
sibling deletes that old `c4_min/` and re-adds only its own new file(s). Git's
3-way merge nonetheless resolved the doc-only branches cleanly (the substrate
`.py` files are "added-vs-ancestor" on the base side and survive); the
oracle-harness branch produced the expected add/add conflicts on `DESIGN.md` and
`__init__.py`, resolved by hand.

`greenfield-corpus` / `greenfield-baking-primer` do **not** contain their
intended files (`conformance_ids.json`, `BAKING_PRIMER.md`, `baking_examples.py`
are absent; the branch tips are the unrelated `e45eb4ff` merge). They were never
populated — **skipped and flagged pending**. Re-run their generation, then add
the produced files directly onto `greenfield-base` (a plain merge would only
delete the substrate `c4_min/`).

---

## 2. DESIGN.md resolution (one coherent file)

The **substrate DESIGN.md is authoritative** for the compiler / layout /
architecture (sections (a)–(f) + the skeleton REPORT). The oracle-harness
DESIGN.md's **"Model interface"** contract was folded in verbatim-in-spirit as a
new **section (g) — Model interface (oracle-harness contract)**, capturing:
the reference-semantics (expected) side, the harness's expected stub interface,
the real substrate interface, the adapter reconciliation, and the verdict. No
interface contract was lost. `__init__.py` was resolved to one coherent package
docstring covering the substrate + the harness.

---

## 3. Oracle wired to the REAL compiler (adapter)

The harness (`run_oracle.py`) was written against a **stub**:

```
c4_min.compile.compile_program(prog: Program) -> state_dict
c4_min.model.run(state_dict, prog, *, max_steps) -> Decoded
```

The real substrate exposes a different shape:

```
c4_min.compiler.compile_program(prog=[(name, imm), ...]) -> (model, layout, code)
c4_min.compiler.run(model, layout, code) -> list[int]   # per-step AX
```

New file **`c4_min/oracle_adapter.py`** reconciles them:

* `prog_to_ops(Program)` — decode each `bytecode` word (`op | imm<<8`) into the
  substrate's `[(op_name, imm), ...]` (opcode id → name via `isa.NAMES`).
* `RealBackend` — drives `compiler.compile_program` + `compiler.run`, wrapping
  the per-step AX list into a `Decoded` (exit-code-only: the straight-line slice
  tracks AX, not PC — a valid weaker conformer per the harness contract).
* `expected_8bit(Program)` — ground truth: takes the (full-coverage, 32-bit)
  reference-oracle exit code and **masks it to 8 bits**. This is the correct
  width bridge — the neural_vm reference is 32-bit and disagrees with the 8-bit
  substrate on wrap/underflow boundary cases (`255+1→256` vs `0`; `7-9→2^32-2`
  vs `254`). It keeps every op-class computable on the expected side (the slice
  interpreter cannot decode ENT/ADJ/JSR/LEV).

`run_oracle.py`: `_get_model_backend()` now returns `RealBackend`; `run_op_class`
/ `run_all` take an `expected_fn` (defaults to the 32-bit oracle so the mock
self-test is unchanged; the real run passes `expected_8bit`).

### SUB-underflow gap closed

The baseline surfaced a real substrate bug: SUB computed `a-b` but had **no fold
on underflow**, so `7-9 → -2` decoded to `0` instead of `254`. Fixed in
`compiler.py`: SUB now computes `(a - b + 256)` and re-uses the existing mod-256
fold gadget (the "signed SUB wrap" DESIGN.md already anticipated). Non-underflow
SUB and ADD-wrap are unchanged. Added `test_sub_underflow_wraps_mod_256`.

---

## 4. Fan-out baseline (real compiler vs 8-bit reference ISA)

`OMP_NUM_THREADS=4 PYTHONPATH=$(pwd) python c4_min/run_oracle.py`

**OP-CLASS PASS: 4/30**

| PASS (implemented slice) | FAIL (fan-out targets)                                    |
|--------------------------|-----------------------------------------------------------|
| `IMM` 3/3                | `MUL DIV MOD` (arith)                                      |
| `ADD` 4/4                | `EQ NE LT GT LE GE` (cmp)                                  |
| `SUB` 4/4                | `AND OR XOR` (bitwise)                                     |
| `PSH` 2/2                | `SHL SHR` (shift)                                          |
|                          | `LI LC SI SC` (memory)                                     |
|                          | `LEA ENT ADJ JSR LEV` (frame/call)                        |
|                          | `JMP BZ BNZ` (control flow)                               |

`HALT`/`EXIT` is not a standalone op-class in the harness; it is the terminator
in every generated program and passes in the 4 PASS classes.

**Failure kinds** (all `model_error`, i.e. the op is not built — not a
mis-computation):

* `NotImplementedError('op X not in slice compiler')` — op reaches
  `compiler._step_rules` but has no rule: `EQ NE LT GT LE GE AND OR XOR SHL SHR
  LI SI JMP BZ BNZ ENT JSR LEV`.
* `KeyError('opcode id N not in c4_min ISA')` — op id absent from the substrate
  `isa.py` opcode table entirely: `MUL(27) DIV(28) MOD(29) LC(10) SC(12)
  ADJ(7)`. These need an `isa.py` entry first, then a rule.

**Nuance:** `LEA` itself *is* wired in `_step_rules`, but its representative
program uses `ENT` (unimplemented), so the LEA class fails on the ENT dependency,
not on LEA. Implementing ENT (+ SI/LI) will let LEA pass with no LEA change.

---

## 5. Gate status

* `pytest c4_min/` → **14 passed** (8 `test_slice` incl. the new SUB-underflow +
  6 `test_oracle_harness`).
* `python c4_min/run_oracle.py --self-test` → **PASSED** (correct model 30/30
  PASS, buggy model 30/30 FAIL — the harness detects both).

---

## 6. Fan-out starting point (the FAIL list = work remaining)

Grouped by op-class (reusable gadgets amortise cost within a group):

1. **cmp** `EQ NE LT GT LE GE` — threshold/step gadget → `ALU_LO`/AX 0-or-1.
2. **bitwise** `AND OR XOR` — 8-bit bit-decompose gadget.
3. **shift** `SHL SHR` — shift gadget (+ offset fold, like SUB).
4. **arith (new isa ids)** `MUL DIV MOD (27/28/29)` — add ids to `isa.py`
   (+ interpreter) then rules.
5. **memory** `LI SI` (+ `LC/SC` ids 10/12) — addr→cell attention head; needs a
   data segment in the compiler (absent in the straight-line slice).
6. **frame/call** `ENT LEV JSR` (+ `ADJ` id 7) + `LEA` (unblocked by ENT) —
   SP/BP frame + PC push/pop; requires cross-position carry.
7. **control flow** `JMP BZ BNZ` — PC-driven fetch (branches); the biggest
   substrate extension (DESIGN.md (e), un-exercised by the straight-line slice).

The oracle harness now drives the real compiler, so each new op can be validated
by re-running `run_oracle.py --op-classes <OP>` — no other harness change needed.
