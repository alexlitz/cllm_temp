# DSL W5 MUL/DIV migration — known limits

Companion to `docs/DSL_W3_ADDSUB_LIMIT.md`. Records the gap between the
current `wide_mul_rules` / `wide_div_rules` DSL helpers and full
multi-byte semantics, and lays out the path to close it.

## Where things stand (this branch)

`c4_release/neural_vm/unified_compiler/wide_alu_dsl.py`:

| Helper            | Supported widths              | Rule count           |
| ----------------- | ----------------------------- | -------------------- |
| `wide_mul_rules`  | `width_bytes ∈ {1, 2}`        | 256 / 65,536         |
| `wide_div_rules`  | `width_bytes = 1`             | 256                  |

In the DSL's per-byte stacking convention each "byte" is a single 4-bit
nibble lane, so `width_bytes=2` for MUL is an 8-bit × 8-bit multiply
producing a 16-bit (4-nibble) result.

## wide_mul_rules — width_bytes > 2 is intractable as a flat lookup

The width_bytes=2 implementation is a **5-way AND lookup** over the full
operand cross-product: one rule per `(a_lo, a_hi, b_lo, b_hi)` quad. The
rule count grows as `16 ** (2 * width_bytes)`:

| width_bytes | bits | rules         | feasibility               |
| ----------- | ---- | ------------- | ------------------------- |
| 1           | 4    | 256           | trivial (POC)             |
| 2           | 8    | 65,536        | tractable (this branch)   |
| 3           | 12   | 16,777,216    | intractable               |
| 4           | 16   | 4,294,967,296 | impossible                |

Past 8-bit operands the flat cross-product approach breaks down — both
because the unit count explodes the FFN matrix size and because
SwiGLU's `W_up` can only express a bounded number of distinct
input-pattern hyperplanes per layer.

## Why the legacy composite doesn't have this limit

`FlattenedALUMul` (see `efficient_alu_neural.py:1066+`) implements a
**9-stage schoolbook pipeline** that operates on the GE workspace
`[B, seq, 8, 160]`:

1. `BDToGE` — unpack BD operand bands into GE rows.
2. Schoolbook partial products — emit `width_bytes**2` per-nibble
   products (each is the same nibble-pair lookup the DSL already has).
3. Carry pass 1, 2, 3 — sum the staggered partial-product columns into
   a single per-column total, with explicit carry across columns.
4. Genprop — extract generate/propagate flags per column.
5. Binary-lookahead — log-depth carry chain across the column array.
6. Final-correction — apply the resolved carry to each column.
7. `MulCombine` — assemble the columns into the result byte lanes.
8. `GEToBD` — write the result back to the BD residual bands.

Each stage is its own FFN/attention pair acting on the GE workspace.
The carry propagation across columns is what a single-pass DSL lookup
fundamentally cannot express — it needs an inter-pass cascade.

## wide_div_rules — multi-byte is mathematically wrong as a per-nibble lookup

Per-nibble independent division **does not compose** into wide-operand
division. Concrete counter-example:

```
  0xFF / 0x0F = 0x11  remainder 0x00     (correct, integer)

  Per-nibble:
    nibble0: F / F = 1 rem 0     →  q_lane0 = 1, r_lane0 = 0
    nibble1: F / 0 → guard       →  q_lane1 = 0, r_lane1 = F

  Reassembled per-nibble: q = 0x01, r = 0xF0   — WRONG.
```

The relationship between dividend nibbles, divisor nibbles, quotient
nibbles, and remainder nibbles is non-local: each output nibble of the
quotient depends on the entire dividend / divisor pair, not just the
co-located lane. There is no per-nibble lookup table that captures wide
DIV semantics, regardless of how many rules are emitted.

`FlattenedDivMod` (see `efficient_alu_divmod_split.py`) implements wide
DIV as an **8-outer × 3-inner long-division loop** on the GE workspace
— a bit-serial shift-and-subtract cascade. As with MUL, the cross-stage
data dependency cannot be flattened into a single-pass lookup.

## Two paths forward (same as W3 AddSub)

### Path 1: Multi-pass DSL primitive
Add a `multi_pass_rules` IR construct that emits a sequence of FFN
layers with explicit residual-position propagation between passes.
`wide_mul_rules(width_bytes > 2)` and `wide_div_rules(width_bytes > 1)`
would then expand into ~9 passes (for MUL) or ~24 passes (for DIV)
matching the legacy composite stage count. Substantial DSL surface
work.

### Path 2: GE-format DSL extension
Add `wide_ge_mul_rules` / `wide_ge_div_rules` that emit rules operating
on the GE workspace `[seq, byte_row, nibble_col]` rather than the BD
band offsets. This mirrors how the existing composites work and shares
the same blocker as Path 2 in the W3 doc — i.e. needs GE-format helpers
that don't yet exist.

Either path is several sessions of work. For practical purposes, the
multi-byte MUL/DIV legacy composites (`FlattenedALUMul`,
`FlattenedDivMod`) remain authoritative for `width_bytes > 2` (MUL) and
`width_bytes > 1` (DIV).

## Status

Multi-byte MUL (`width_bytes > 2`) via DSL: **deferred**.
Multi-byte DIV (`width_bytes > 1`) via DSL: **deferred**.

W6 deletion of `efficient_alu_*.py` is blocked on this plus the W3
AddSub gap.
