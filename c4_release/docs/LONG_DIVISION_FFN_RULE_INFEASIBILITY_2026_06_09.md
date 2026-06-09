# `LongDivisionModule` → `FFNRule` migration — structural infeasibility (2026-06-09)

Status: **infeasible at single-`FFNOp` granularity**. Companion to
[`DSL_W5_MULDIV_LIMIT.md`](DSL_W5_MULDIV_LIMIT.md) and
[`LONG_DIVISION_BUG36_2026_06_09.md`](LONG_DIVISION_BUG36_2026_06_09.md).
Records why the Bug #36 "full FFNRule migration" of
`LongDivisionModule.forward` cannot ship as additional rules under the
existing `wide_div_rules` DSL.

## Verification of prior conclusion

`acc087be0` attempted a 2-byte (8-bit × 8-bit) flat cross-product
lookup: 65,536 rules per opcode batch, 131,072 rules total. That
sidesteps long division entirely by precomputing every `(a, b) -> (q, r)`
pair at bake time.

That approach does **not** cover the actual operand width consumed by
`LongDivisionModule`: per `divmod_longdiv.py:9-22` and the NIBBLE config
(`NUM_POSITIONS = 8`), the dividend `a` and divisor `b` are each
**8-nibble = 32-bit** values. The flat lookup grows as
`16 ** (2 * width_bytes)`:

| `width_bytes` | operand bits | lookup rules | feasible? |
|---|---|---|---|
| 1 | 4 | 256 | yes (current POC) |
| 2 | 8 | 65,536 | tractable but does not match VM operand width |
| 4 (32-bit) | 16 | 4.29e9 | impossible |
| 8 (matches `NUM_POSITIONS=8`) | 32 | 1.84e19 | impossible |

The 1096 corpus includes operands up to `dividend ≈ 2549` (per
`DSL_W5_MULDIV_LIMIT.md`), which exceeds 8 bits — so even the
width_bytes=2 path drops 1096 coverage. Flat lookup is not a path to
parity with `FlattenedDivMod` for the real VM operand width.

`ac01fcc9`'s conclusion stands: full migration is infeasible at single
`FFNOp` granularity. Commit `e85dedd6` already ships the alternative
deliverable (declarative wrapper exposing the BD-format I/O surface to
`dim_contracts_audit`).

## Why long division does not flatten — exact counts

`LongDivisionModule.forward` (`divmod_longdiv.py:224-300`) does, per
forward pass:

- **8 outer iterations** (one per dividend nibble, MSB-to-LSB).
- Inside each outer iteration:
  - 1 × `_shift_left_one_nibble` on a 9-nibble `partial` state vector
    (`partial = partial * 16 + a[i]`).
  - **15 trial multiplies** (k = 1..15). Each `_trial_multiply` is a
    9-nibble carry-chain over a 9-wide `q * b` partial-product array,
    where carry-out of position `j` enters position `j+1` (a 9-step
    sequential dependency).
  - **15 `_compare_le`** invocations. Each is an MSB-to-LSB
    9-position scan with sequential `eq_so_far` propagation.
  - 1 × `_trial_multiply` for the resolved `q_i * b` (a 16th trial).
  - 1 × `_subtract` (9-nibble borrow-chain, position `j+1` depends on
    borrow-out of position `j`).

That is, per forward pass:

| sub-op | per outer | total over 8 outer |
|---|---|---|
| shift-left | 1 | 8 |
| trial-multiply | 16 (15 trials + 1 resolved) | 128 |
| compare-le | 15 | 120 |
| subtract | 1 | 8 |

Every one of these is a **sequential cascade**: each nibble-position's
output depends on the carry/borrow/eq-flag from the previous position
within that sub-op, AND the `partial` state at outer step `i+1` depends
on the subtract result from outer step `i`. That is two distinct levels
of inter-step state — within a sub-op (cross-nibble cascade) and across
outer iterations (partial-dividend state).

## Why an `FFNRule` set cannot express that

A single `FFNOp` is one FFN layer. Rules within a layer fire in
parallel on the same residual snapshot; **a rule cannot read what
another rule in the same layer wrote**. That is the structural
property: the lowering pipeline writes one row of `W_up` / `W_gate`
per rule, and the SwiGLU + `W_down` accumulate into the residual
after all rules have read it.

Consequences for long division:

1. **Cross-nibble carries (within a sub-op).** `_trial_multiply` needs
   `digit[j] = floor((q*b[j] + carry_from_j-1) / 16)`. A single FFN
   layer cannot pipe `carry_from_j-1` to the rule at position `j` —
   that carry is the output of one rule and the input of another.
2. **Cross-iteration `partial` state.** `partial` at outer step `i+1`
   = `subtract(shift_left(partial_i, a[i+1]), q_i * b)`. A single FFN
   layer has no notion of "iteration"; rules see only the residual
   snapshot at layer entry. The cascade across i = 7..0 cannot be
   expressed without separate FFN layers per iteration.
3. **Trial-and-pick monotone count.** `q_i = count of k in 1..15 with
   k*b <= partial_i` is itself derived from 15 `_compare_le` results
   that each chain MSB-to-LSB through 9 positions. No single-rule
   threshold reads the conjunction of 9 sequential equalities.

## What it *would* take (deferred)

The two paths in `DSL_W5_MULDIV_LIMIT.md §"Two paths forward"` apply
directly:

### Path 1: `multi_pass_rules` DSL primitive

An IR construct that emits a **sequence of FFN layers** with named
residual-position propagation between passes. The 8-outer × ~17-inner
structure above would lower to ≈ 24 FFN passes (or 8 if the inner
shift/trial/subtract collapses per-iteration). Each pass would carry
the `partial` state in residual slots shared between passes. New
verifier surface required (multi-pass `verify_rule_scopes`,
cross-pass dim ownership).

### Path 2: GE-format DSL

Mirror what `FlattenedDivMod` does: lift the rules onto the
`[seq, byte_row, nibble_col]` GE workspace where positional structure
is first-class. Needs GE-format helpers that do not yet exist (no
`ge_constant_write` / `ge_gated_write` in `building_blocks_dsl.py`).

Both are several sessions of DSL work. For Bug #36 the existing
`FlattenedDivMod` composite remains authoritative; the declarative
wrapper at `e85dedd6` is the IR-visible surface that
`dim_contracts_audit` and `decl_verifier.verify_rule_scopes` can
reason about.

## Cross-references

- [`DSL_W5_MULDIV_LIMIT.md`](DSL_W5_MULDIV_LIMIT.md) — the same
  structural limit at the DSL-helper level, including the
  `0xFF / 0x0F = 0x11` per-nibble counter-example.
- [`LONG_DIVISION_BUG36_2026_06_09.md`](LONG_DIVISION_BUG36_2026_06_09.md)
  — Bug #36 oracle-localization and structural-gap brief.
- `c4_release/neural_vm/alu/ops/divmod_longdiv.py:103-300` —
  `LongDivisionModule` reference implementation (the structural target).
- `c4_release/neural_vm/efficient_alu_divmod_split.py` —
  `FlattenedDivMod` 4-stage composite that wraps it.
- `c4_release/neural_vm/unified_compiler/wide_alu_dsl.py:786-882` —
  `wide_div_rules` (POC; raises `NotImplementedError` for
  `width_bytes > 1`).
- `c4_release/neural_vm/unified_compiler/ops/alu_ops.py:1389-` —
  `make_alu_divmod_composite_ops` (4-op declarative wrapper landed
  at `e85dedd6`).
