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

### Path 1: Multi-pass DSL primitive — BUILT + PILOTED (MUL) ✅
The `multi_pass_rules` IR construct now EXISTS
(`neural_vm/unified_compiler/ir.py`: `MultiPassOp` = ordered
`FFNPass` sequence + `workspace_band` + `lower_multi_pass` /
`run_symbolic`). It emits a sequence of FFN passes applied to the same
residual, each reading a workspace band the prior passes wrote — the
cross-pass carry chain a single forward cannot express.

**Pilot (wide MUL) landed byte-identical.**
`multi_pass_mul_rules` (`wide_alu_dsl.py`) derives `(a*b)&0xFFFF` for
8-bit × 8-bit operands from a COMPACT 7-pass schoolbook spec
(partial-product accumulate + column-carry passes), ~2848 units. It is
**verdict-identical to the flat `wide_mul_rules(width_bytes=2)` lookup on
all 65,536 (a,b) pairs** — a 23× unit reduction (2848 vs 65,536) AND, more
importantly, the rule count is now `O(width² · 256)` NOT `O(16^(2·width))`,
so it generalizes past the flat lookup's width-2 ceiling by adding
partial-product + column-add passes. Tests:
`tests/test_wide_alu_dsl.py::test_multi_pass_mul_*`.

The one non-obvious mechanism the pilot uncovered: an AMPLITUDE-NORMALIZED
cascade convention (`_normalized_and_rule`: read weight 1.0, threshold
`k-0.5` → `up=S·0.5`, write `1/(S·0.5)`). Without it stacked SwiGLU
magnitude-explodes pass-over-pass (`silu(up)≈up` scales with the input,
so a residual-30 workspace one-hot read at weight 30 runs away). The fixed
point pins EVERY pass's one-hots to residual 1.0.

**DIV pilot (`multi_pass_div_rules`) landed — same primitive, DIV spec.**
`multi_pass_div_rules` (`wide_alu_dsl.py`) derives `a // b` (quotient) +
`a % b` (remainder) for 8-bit operands from a COMPACT binary long-division
spec — bit-serial shift-subtract with a cross-pass RUNNING-REMAINDER carry
(bit i's remainder feeds bit i-1's shift), 43 passes / ~11.7k units. It is
**verdict-identical to Python `divmod` (with the `b==0 → q=0, r=a`
convention) on all 65,536 (a,b) pairs** via the neural PureFFN forward
(test `test_multi_pass_div_byte_identity_full`; `tools/_div_multipass_verify.py
--full`). Rule count is `O(width·256)` per bit NOT `O(256^width)`, so it
generalizes past the flat `wide_div_rules_ge_format` lookup's 8-bit ceiling
by adding bit iterations. The b==0 case is handled by mutually-exclusive
`BZERO`/`BNONZERO` gating (clean single-hot result lane).

**DIV install (`C4_DIV_MULTIPASS`, DEFAULT-OFF) landed.** The L10 divmod
install (`make_alu_divmod_composite_ops`) appends a `MultiPassDivBlock`
(`efficient_alu_neural.py`) — the 43 lowered `PureFFN` passes + an OR-gate
seed packed into ONE post_op (like `MultiPassMulBlock`, so the physical block
count is unchanged at 59 and the absolute-position lea contract holds). It
reads the dividend from `ALU_LO/HI`, divisor from `AX_CARRY_LO/HI` (the SAME
operand bands the composite / GE-format lookup consume), computes q/r on
dedicated result lanes, then routes q→OUTPUT (OP_DIV) / r→OUTPUT (OP_MOD) at
MARK_AX and replays the campaign divisor / dividend-byte-1 clears. Flag-OFF is
byte-identical to golden `91f55411`; flag-ON computes a//b + a%b correctly
through OUTPUT on a direct-operand sweep (`tools/_div_multipass_live_probe.py`).

**Remaining:** `wide_mul_rules(width_bytes>2)` (wire the O(width²) pass
generator to arbitrary width) and `multi_pass_div_rules(width_bytes>1)`
(multi-byte dividend — same primitive, more bit iterations + a wider
running-remainder lane).

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

`multi_pass_rules` IR primitive: **BUILT** (`MultiPassOp` in `ir.py`).
Multi-byte MUL derivation via `multi_pass_mul_rules`: **PILOTED,
byte-identical** at width-2 (65,536/65,536 pairs vs the flat lookup);
generalizes to width>2 by adding passes (the flat lookup could not).
Multi-byte DIV (`width_bytes > 1`) via the same primitive: **next pilot**
(long-division shift-subtract passes on a GE-style workspace band).

The GAP is CLOSED for the derivability question: the ALU wide-MUL floor
was a LOWERING-generality gap (no multi-pass construct), and that
construct now exists and derives MUL from a compact schoolbook spec. What
remains is (a) a DIV pilot on the same primitive and (b) the production
INSTALL (replacing `FlattenedALUMul` / `FlattenedDivMod`) which is a
separate byte-identity band-routing exercise.

W6 deletion of `efficient_alu_*.py` is unblocked on the DERIVATION side
(the compute derives); it still needs the production install + the W3
AddSub gap.
