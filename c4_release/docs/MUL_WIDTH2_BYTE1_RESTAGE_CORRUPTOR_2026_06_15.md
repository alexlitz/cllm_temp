# width=2 MUL byte-1 emit: FlattenedALUMul AX_FULL re-stage corruptor — 2026-06-15

Status: **ROOT 1 LANDED (commit 22bae24f). mul full_trace 28 -> 46/50 (+18).
smoke 51/0. add/sub/div/mod guards unchanged.** Root 2 (operand-A
high-nibble gather) documented as the remaining wall.

## The symptom

After the width=2 MUL widen (`C4_MUL_WIDTH2=1`, default ON) landed
(`docs/MUL_WIDTH2_WIDEN_2026_06_13.md`), 22/50 `mul_*` programs still failed
full_trace. The product's BYTE 0 emitted correctly but BYTE 1 was garbage:

```
97*94 = 0x239E  -> emitted 0xFD9E  (byte0 0x9E OK, byte1 0x23 -> 0xFD)
21*59 = 0x04D7  -> emitted 0xF0D7  (byte1 0x04 -> 0xF0)
65*98 = 0x18E2  -> emitted 0x48E2  (byte1 0x18 -> 0x48)
```

The wrong byte-1 values (0xFD / 0xF0 / 0x48) are the 0xFF-leak sentinel
family.

## The root (spec_k=0, BUILT dims `dim_positions`, d_model=1090)

The width=2 MUL pipeline is:
1. **L11 (block 13)**: the width=2 `wide_mul` computes the product's byte 1
   into the dedicated `MUL_RESULT_HI_LO/HI` band. **This is byte-CORRECT and
   IMMORTAL** — `MUL_RESULT_HI` carries 0x23 (97*94) / 0x04 (21*59) and
   survives unchanged across *every* block 13..30.
2. **L13 (block 15)**: `layer13_mul_result_hi_relay` (attn head 6) copies
   `MUL_RESULT_HI -> AX_FULL` at the MUL `MARK_AX` row (156). Correct: AX_FULL
   = 0x23.
3. **L19 (block 30)**: `layer15_alu_high_byte_relay` (attn head 8, K-gated on
   OP_MUL/OP_SHL/OP_OR/OP_XOR) copies `AX_FULL -> OUTPUT` at the MUL row, which
   the LM head emits as byte 1.

The bug sits BETWEEN steps 2 and 3. **Physical block 26 (logical L15) has a
SECOND `FlattenedALUMul` composite** — installed by
`make_l12_alu_postop_attach_op` and expanded by `_expand_wrapper_blocks` into
its own block (the `docs/L17_TAIL_MUL_DOUBLE_FIRE.md` double-fire composite).
On the MUL `MARK_AX` row OP_MUL is still hot, so it RE-FIRES, and its
`GEToBDConverter.forward` (`efficient_alu_neural.py:380-401`) **re-stages
AX_FULL byte 1 from its OWN GE high-byte RESULT**, gated by
`wide_op = (OP_MUL|OP_SHL|OP_SHR|OP_DIV|OP_MOD) * MARK_AX`.

That composite's high-byte GE computation DISAGREES with the width=2 wide_mul
band (e.g. it produces 0xFD where wide_mul produced 0x23), so it **overwrites
the correct AX_FULL staging** (zeroes cells 3,2 → writes cells 13,15). The
byte-1 emit relay at block 30 then faithfully copies the garbage.

Trace (97*94, AX_FULL_LO/HI argmax at the MUL row 156):

```
blk13 (L11)  AX_FULL=0x00   (not yet staged)
blk15 (L13)  AX_FULL=0x23   <- L13 relay staged correctly
blk25 (L14)  AX_FULL=0x23
blk26 (L15)  AX_FULL=0xFD   <- FlattenedALUMul re-stage CLOBBERS it
blk30 (L19)  AX_FULL=0xFD   -> emit relay copies 0xFD
```

## The fix (commit 22bae24f)

In `GEToBDConverter.forward`, when `mul_width2_enabled()` is on, EXCLUDE
OP_MUL from the `wide_op` mask that drives the AX_FULL byte-1 re-stage. The
L13 relay's correct `MUL_RESULT_HI` staging then survives to the emit.
SHL/SHR/DIV/MOD keep their AX_FULL byte-1 re-stage unchanged.

* Forward-only change (no `__init__` weight writes) -> the baked param hash is
  byte-identical to HEAD at every flag setting.
* With `C4_MUL_WIDTH2=0` OP_MUL is kept in the mask -> byte-identical to the
  pre-width2 path (which has no `MUL_RESULT_HI` band).
* Byte 0 was never affected (it is computed by the same wide_mul + the
  GEToBDConverter OUTPUT path that the composite re-fire does not corrupt).

### Verification

* `mul` full_trace **28 -> 46/50 (+18)**.
* `smoke` 51/0.
* Guards unchanged: add 40/50, sub 44/50, div 47/50, mod ~48/50.
* `expr_paren` 18 -> 19.

## ROOT 2 — the remaining 4 mul fails (operand-A high-nibble gather)

`mul_20 (9*98)`, `mul_29 (89*26)`, `mul_36 (9*44)`, `mul_43 (9*5)` still fail.
These are the documented operand-gather hybrid-encoding Wall-1: operand A's
STACK0 byte-0 HIGH nibble is corrupted upstream (89=0x59 -> 0x39, 9=0x09 ->
0x39 high nibble 3), so the MUL computes the wrong product before the byte-1
chain even runs. The corrupt nibble is present at the TOKEN/embedding level on
the STACK0 value row, i.e. the PSH'd value's high nibble is mis-emitted — the
same surface as `project_operand_gather_hybrid_encoding_is_cmp_alu_root` and
`project_ax_byte1_dump_is_h1_onehot_wall`. NOT a single-rule fix; a separate
multi-session wall.

## expr-with-mul cascades (expr_add_mul/expr_mul_div/expr_mod)

Still 0/0/3. The MUL intermediate result is now CORRECT, but these chain MUL
with ADD/DIV/MOD on the 16-bit intermediate (e.g. `14*56/8`: MUL=784=0x310 OK,
but the DIV reads the dividend's byte 1 wrong -> emits 769 ≈ undivided). These
are DIV/ADD/MOD multi-byte-intermediate + framing-drift roots (tasks 212/221/
224), NOT MUL-cluster.

## Artifacts

* `tools/probe_mul_emit_tokens.py` — decode emitted REG_AX bytes vs expected.
* `tools/probe_mul_multibyte_trace.py` — L11 operands + MUL_RESULT_HI + AX_FULL
  across the chain.
* `tools/probe_mul_byte1_logits.py` — byte-1 emit logits + AX_FULL/AX_CARRY
  block trace at the generating position.
* `tools/probe_mul_byte1_blocktrace.py` — OUTPUT band per block at the emit
  token.
