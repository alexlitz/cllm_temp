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
