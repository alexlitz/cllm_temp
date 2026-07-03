# width=2 MUL byte-1 emit: FlattenedALUMul AX_FULL re-stage corruptor — 2026-06-15

Status: **ROOT 1 LANDED (commit 22bae24f). mul full_trace 28 -> 46/50 (+18).
smoke 51/0. add/sub/div/mod guards unchanged.** **ROOT 2 RESOLVED 2026-06-18
(flag `C4_MUL_STACK0_BYTE39_GUARD`): mul 46 -> 50/50; the remaining 4 fails were
a single L10 tail rule (`byte_39_from_e8_addr`) misfiring on operand-A low
nibble 9, NOT an embedding wall — see the ROOT 2 section below.**

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

## ROOT 2 — RESOLVED 2026-06-18 (single L10 tail rule, NOT an embedding wall)

`mul_20 (9*98)`, `mul_29 (89*26)`, `mul_36 (9*44)`, `mul_43 (9*5)` — all
operand-A **low nibble 9** — were attributed above to an operand-gather /
embedding-level high-nibble corruption ("a separate multi-session wall"). That
attribution was WRONG: the corrupt STACK0 byte-0 (0x39 = 57) is produced by a
SINGLE L10 tail FFN rule, surgically fixable.

Root (spec_k=0 GPU full_trace, BUILT dims): the rule
`tail_stack0_store_top_e8_from_e0_byte_39_from_e8_addr`
(`ops/l10_ops.py::stack0_store_top_e8_from_e0_output_rules`) restores the
stored byte `0x39` at the e8/e0 local-store address transition (load-bearing
for genuine `0x39` store-pops). Its activation is driven almost entirely by the
UNBOUNDED `("OUTPUT_LO+9", 100.0)` term: at a binary-op STACK0 byte-0 emit row
the operand's low nibble 9 lands in OUTPUT_LO+9 at magnitude ~3654, so
`100 * 3654 = 365428` ALONE trips the `threshold=20000` even though every
store-pop witness (MEM_STORE / EMBED_LO+8 / EMBED_HI+14 / MEM_ADDR_SRC) is COLD.
The rule then forces the byte to 0x39 — corrupting the STACK0 byte-0 of EVERY
binary op whose operand-A low nibble is 9 (low nibble 9 by accident matches the
operand, so ALU_LO is right; high nibble 3 is the phantom that breaks the
width=2 MUL's operand-A high nibble). Verified by sweeping operand-A 1..255: the
step-3 STACK0 byte-0 is byte-correct for EVERY low nibble except 9, where it is
universally 0x39.

The genuine e8->e0 store-pop the rule exists for is a MEMORY restore that
carries `MEM_ADDR_SRC`; the binary-op false-fire never does (probed: false-fires
have `MEM_ADDR_SRC == 0` AND `ADDR_B0_HI+14 == -2`). Fix = require MEM_ADDR_SRC
as a HARD gate (a -1e6 baseline only the +2e6 MEM_ADDR_SRC witness lifts above
zero), so the arithmetic-result row can never trip the rule no matter how large
its single OUTPUT nibble is.

Flag: `C4_MUL_STACK0_BYTE39_GUARD` (DEFAULT OFF, byte-identical flag-OFF;
`shared.mul_stack0_byte39_guard_enabled` + the L10 rule + the cache-key
disambiguation in `full_vm_compiler_dynamic.py`). Verified GPU full_trace
spec_k=0: **mul 46 -> 50/50**; ids 0-299 flag-ON vs flag-OFF **0 regressions,
15 flips** (the 4 mul + 11 ADD/SUB/DIV/MOD that shared the low-nibble-9 root —
this rule was a CROSS-CLUSTER corruptor, not MUL-specific); tripwire (112-prog,
all 56 clusters) 0 regressions / 11 flips; smoke 51/0; flag-OFF model param hash
identical to base e3f80da3.

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
