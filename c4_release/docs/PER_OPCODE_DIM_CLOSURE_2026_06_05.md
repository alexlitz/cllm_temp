# Per-Opcode Dim Closure — 2026-06-05

Closure of declared `reads={...}` / `writes={...}` sets per `(opcode, block)`
across the compiled `LayerCompiler` layout. Source of truth: the L0–L7
declaration audit (commits `618b74e7..7786912b`) plus the surrounding L8–L16
claims commits that landed alongside. Built statically from
`_build_layout_only(alu_mode="lookup", n_heads=8)` — no model bake, no
weight inspection.

## Methodology

For each `(opcode X, block B)` we walk every `Operation` in
`layout.ops_per_layer[B]`, plus every `Operation` in `layout.block_ops` /
`layout.model_ops` whose `layer_idx == B`, and accumulate `reads` and
`writes` from the ops that **fire** for opcode X:

* An op fires for X iff `op.opcodes` is empty (opcode-agnostic, fires every
  step) OR `X in op.opcodes`.
* Model ops with `layer_idx=None` are bake-time/whole-model events and are
  excluded — they do not represent a per-step per-block dispatch and would
  spuriously inflate every block's closure if broadcast.

Output cells: `(opcode, block_idx)` → `{reads, writes, active_op_count,
op_names}`. See `tools/per_opcode_closure.json` for the raw data;
`tools/per_opcode_closure.py` for the generator.

## Shape

* **Logical layers from the compiler: 18** (`layout.n_layers == 18`).
  Wrapper expansion at bake time grows this to ~30 physical blocks (see
  `compile_full_vm` output), but ops are declared at the logical layer
  granularity, which is what governs skippability.
* **Opcodes covered: 31** — the requested IMM..EXIT list. (The user-stated
  count of "30 opcodes" is one off; LEA/IMM/JMP/JSR/BZ/BNZ/ENT/ADJ/LEV +
  LI/LC/SI/SC/PSH + OR/XOR/AND + EQ/NE/LT/GT/LE/GE + SHL/SHR +
  ADD/SUB/MUL/DIV/MOD + EXIT enumerates to 31.) Excluded: OP_NOP,
  OP_GETCHAR, OP_PUTCHAR.
* **Total cells: 18 × 31 = 558.** (The user-stated target of 1050 assumes
  35 × 30 — the 35 refers to the legacy wrapper-expanded view; the
  declaration audit operates at 18 logical layers.)

## Headline result

**Skippable cells (zero reads AND zero writes): 55 / 558 (9.9%).**

All 55 skippable cells concentrate on two blocks:

| Block | Skippable opcodes | Active opcodes              | Role                             |
|-------|-------------------|------------------------------|----------------------------------|
| L13   | 25                | LI, LC, SI, SC, SHL, SHR    | Memory addressing + bit shifts   |
| L16   | 30                | LEV                          | LEV-only routing block           |

Every other block (L0–L12, L14, L15, L17) has at least one opcode-agnostic
op (e.g. PC-emission at L0/L1/L3, contract validation, post-op attach
hooks) and is therefore "active" for every opcode under the strict
zero-R/zero-W criterion.

## Per-Opcode Summary

| Opcode | Active | Skipped | Union R | Union W | Skipped blocks |
|--------|--------|---------|---------|---------|----------------|
| IMM | 16 | 2 | 106 | 102 | L13, L16 |
| LEA | 16 | 2 | 104 | 103 | L13, L16 |
| JMP | 16 | 2 | 106 | 102 | L13, L16 |
| JSR | 16 | 2 | 107 | 102 | L13, L16 |
| BZ  | 16 | 2 | 104 | 102 | L13, L16 |
| BNZ | 16 | 2 | 104 | 102 | L13, L16 |
| ENT | 16 | 2 | 105 | 102 | L13, L16 |
| ADJ | 16 | 2 | 104 | 102 | L13, L16 |
| LEV | 17 | 1 | 104 | 102 | L13 |
| LI  | 17 | 1 | 105 | 102 | L16 |
| LC  | 17 | 1 | 105 | 102 | L16 |
| SI  | 17 | 1 | 105 | 102 | L16 |
| SC  | 17 | 1 | 105 | 102 | L16 |
| PSH | 16 | 2 | 105 | 102 | L13, L16 |
| OR  | 16 | 2 | 104 | 103 | L13, L16 |
| XOR | 16 | 2 | 104 | 103 | L13, L16 |
| AND | 16 | 2 | 104 | 103 | L13, L16 |
| EQ  | 16 | 2 | 104 | 104 | L13, L16 |
| NE  | 16 | 2 | 104 | 104 | L13, L16 |
| LT  | 16 | 2 | 104 | 104 | L13, L16 |
| GT  | 16 | 2 | 104 | 104 | L13, L16 |
| LE  | 16 | 2 | 104 | 104 | L13, L16 |
| GE  | 16 | 2 | 104 | 104 | L13, L16 |
| SHL | 17 | 1 | 104 | 102 | L16 |
| SHR | 17 | 1 | 104 | 102 | L16 |
| ADD | 16 | 2 | 104 | 103 | L13, L16 |
| SUB | 16 | 2 | 104 | 103 | L13, L16 |
| MUL | 16 | 2 | 104 | 102 | L13, L16 |
| DIV | 16 | 2 | 104 | 103 | L13, L16 |
| MOD | 16 | 2 | 104 | 103 | L13, L16 |
| EXIT| 16 | 2 | 106 | 102 | L13, L16 |

Union R / Union W are the closure-of-closures: the set of distinct dim
names read / written by **any** firing op for that opcode across **all**
blocks. The numbers are remarkably uniform (most opcodes hit ~104 read
dims and ~102 write dims) because the opcode-agnostic step-emission ops
dominate the union.

## Per-Block Opcode-Specific vs Opcode-Agnostic Mix

Active op counts per block, broken down by `op.opcodes` annotation:

| Block | Active ops on OP_IMM | Of which opcode-specific | Always-on opcode-agnostic |
|-------|---------------------|--------------------------|----------------------------|
| L0  | 3  | 0  | 3  |
| L1  | 2  | 0  | 2  |
| L2  | 4  | 0  | 4  |
| L3  | 5  | 0  | 5  |
| L4  | 5  | 0  | 5  |
| L5  | 4  | 0  | 4  |
| L6  | 9  | 1 (`layer6_routing_ffn`) | 8 |
| L7  | 5  | 0  | 5  |
| L8  | 10 | 0  | 10 |
| L9  | 4  | 0  | 4  |
| L10 | 13 | 0  | 13 |
| L11 | 2  | 0  | 2  |
| L12 | 2  | 0  | 2  |
| L13 | 1  | 0  | 1  |
| L14 | 3  | 0  | 3  |
| L15 | 3  | 0  | 3  |
| L16 | 0  | 0  | 0  |
| L17 | 1  | 0  | 1  |

Across IMM, only L6 carries an opcode-specific op (the routing FFN, which
gates on `OP_IMM/OP_EXIT/OP_NOP/OP_JMP/OP_JSR`). Every other firing op is
opcode-agnostic — PC carry, marker emission, ALU post-op attach shells,
etc. The pattern repeats across most opcodes; the heavier opcode-specific
clusters land at L5/L6/L8 for control flow, L8–L13 for ALU, and L14/L15
for memory. See the JSON cell data for the per-opcode breakdown.

## Key Insight: skippable surface

Under the strict criterion (R=0 ∧ W=0), 55 cells are skippable. That's
**9.9% of the (opcode, block) matrix.** This is a lower bound; under a
weaker criterion ("no opcode-specific op fires") the skippable surface
balloons by an order of magnitude because most blocks have only
opcode-agnostic always-on work for most opcodes. The strict criterion is
the safe starting point for a layer-skip impl — anything looser needs
runtime gating around the opcode-agnostic ops to preserve their
side-effects.

* **L16 is the easiest skip win**: it does exactly one thing (LEV
  routing). 30/31 opcodes ignore it entirely.
* **L13 is the second easiest**: only LI/LC/SI/SC/SHL/SHR enter the block
  with non-zero closure; 25/31 opcodes ignore it.
* **L0–L12, L14, L15, L17 are not skippable** under R=0/W=0 because each
  has at least one opcode-agnostic step-emission op.

## Files

* `tools/per_opcode_closure.py` — generator (re-run to refresh the JSON).
* `tools/per_opcode_closure.json` — raw `(opcode, block)` → cell payload.
