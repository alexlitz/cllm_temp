# DSL W3 AddSub migration — known limit

The W3 migration attempt at commit `69b46486` (not cherry-picked to main)
revealed a hard limit on band-offset DSL for multi-byte arithmetic.

## The gap

The `wide_add_rules` / `wide_sub_rules` helpers stack nibbles as:
```
operand_base + (byte_idx * 16 + nibble_idx)
```

But the BD residual stream's wide-ALU bands (`ALU_LO`, `ALU_HI`,
`AX_CARRY_LO/HI`, `OUTPUT_LO/HI`) are each only 16 dims wide. There's
a fortunate adjacency:
- `ALU_LO+16 == ALU_HI`
- `OUTPUT_LO+16 == OUTPUT_HI`
- `AX_CARRY_LO+16 == AX_CARRY_HI`

So `width_bytes=2` would align cleanly (byte 0 = LO nibble, byte 1 = HI
nibble). But for `width_bytes >= 3`, `ALU_LO+32 == CARRY` and
`ALU_LO+48` lands in `CLEAN_EMBED_HI` — the band-stacked rules would
corrupt unrelated bands.

## Why the legacy composite sequences carries differently

`AddSub5StageBlock` does 32-bit ADD/SUB by sequencing carries across
**byte rows in GE-format `[B, seq, 8, 160]`**, not by extending the
nibble band. The 5 stages operate on the GE workspace, not the BD
residual.

The current DSL helpers (`wide_add_rules`, `wide_sub_rules`,
`wide_div_rules`, `wide_mul_rules`) all work on the BD band-offset
model, which CANNOT express the byte-row sequencing.

## Two paths to close the gap

### Path 1: Wider BD bands
Allocate `width_bytes * 16` width for the wide-ALU bands so the
nibble stacking works. Requires re-baking the bands and updating every
consumer that reads `ALU_HI` / `OUTPUT_HI` / `CARRY` as a separate
name.

### Path 2: GE-format DSL extension
Add a parallel set of helpers (`wide_ge_add_rules` etc.) that emit
rules operating on the GE workspace `[seq, byte, nibble]` rather than
the BD band offsets. This mirrors how the existing composites work.

Either path is several sessions of work. For ADD/SUB/DIV/MOD,
single-byte (8-bit) operation already works via the existing helpers
(see W4 `2538d20f`, W5 `388b6342`). Multi-byte for these opcodes is
deferred to a follow-up wave.

## Status

Multi-byte ADD/SUB/DIV/MOD via DSL: **deferred**. Current legacy
composites (AddSub5StageBlock, FlattenedDivMod multi-byte, FlattenedALUMul)
remain authoritative for `width_bytes > 1`. W6 deletion of
`efficient_alu_*.py` is blocked on this.

## Update (W3 retry, GE-format helpers landed)

`wide_alu_dsl.wide_ge_add_rules` and `wide_alu_dsl.wide_ge_sub_rules`
take Path 2 above. They emit rules over a GE-format dim layout where
each byte row gets its own per-position slot block — no nibble stacking
inside a single 16-dim band, so no collision into `CARRY` /
`CLEAN_EMBED_HI` for `width_bytes > 2`. Per-position dim names follow
the convention `p{b}_NIB_A`, `p{b}_NIB_B`, `p{b}_RESULT`,
`p{b}_CARRY_OUT` / `p{b}_BORROW_OUT`.

Byte-identity passes for 16-bit (`width_bytes=4`) and 32-bit
(`width_bytes=8`) ADD and SUB — see
`tests/test_wide_alu_dsl.py::test_wide_ge_add_rules_byte_identity_*`
and the SUB equivalents. The tests pre-inject the per-byte
carry/borrow chain into the input residual (same single-pass strategy
as the existing W3 multi-byte BD test in
`test_wide_add_rules_byte_identity_16bit/32bit`), so they validate
per-nibble rule semantics and the cross-position dim layout.

### Path forward for full W3 retry

1. **Wire the GE-format helpers into a per-layer bake.** The helpers
   are pure FFNRule generators; the next step is to lower them through
   `Primitives.lower_ffn_rules` inside a wide-ALU layer alongside the
   existing BD↔GE projection (`BDToGEConverter` / `GEToBDConverter`,
   see `efficient_alu_neural.py:236-415`). The bake order mirrors
   `AddSub5StageBlock`: BD→GE (stage 0), per-byte ADD/SUB lookup (the
   DSL rules, replacing stages 1-3), GE→BD writeback (stage 4).
2. **Carry cascade across passes.** A single FFN forward cannot
   self-cascade carries (W_up only reads the input residual). The
   cross-byte cascade can be expressed either by stacking
   `width_bytes` FFN layers (each resolving one byte's lookup and
   writing the next byte's carry) or by replicating the binary
   carry-lookahead used in `AddCarryLookaheadFFN` /
   `SubBorrowLookaheadFFN`. Either is a follow-up step — the per-byte
   rule semantics validated by this test are the load-bearing piece.
3. **DIV / MOD / MUL.** The same dim-layout trick (per-position GE
   slot blocks) generalises to the other multi-byte arithmetic ops.
   `wide_ge_div_rules` / `wide_ge_mul_rules` can be added in follow-up
   waves with identical structure to the ADD/SUB helpers here.
4. **W6 deletion of `efficient_alu_*.py`.** Becomes feasible once
   (1) and (2) are in place for ADD/SUB and DIV/MOD. The 8-bit slices
   are already covered by the existing BD-mode helpers.
