# Removal 1 — real IMM 0xE0+ corruption surface (investigation findings)

Date: 2026-06-06
Base: `main` HEAD `c2d28057`
Prior brief: `REMOVAL_1_RETRY_2026_06_06.md`

## TL;DR

With the `b5cf7099` IMM AX override at
`c4_release/neural_vm/batched_pure_neural.py:2016-2024` removed (probe
only — NOT committed), per-token + per-layer FFN-input-residual capture
on the raw neural AX byte 0 emit gives a discriminator that splits at
0xE0:

| IMM (hex) | Got AX bytes        | Expected | Result |
|-----------|---------------------|----------|--------|
| 0x2A      | `[0x2A,0,0,0]`      | 0x2A     | PASS   |
| 0xDF      | `[0xDF,0,0,0]`      | 0xDF     | PASS   |
| 0xE0      | `[0x01,0,0,0]`      | 0xE0     | FAIL   |
| 0xE8      | `[0xE8,0xFF,0,0]`   | 0xE8     | FAIL   |
| 0xFF      | `[0xE8,0xFF,0,0]`   | 0xFF     | FAIL   |

The corruption surface lives at the **L8 multibyte-fetch attention head 3**,
not at L34 (the L34 residual probe stays clean) and not at L5 opcode
decode (opcode decode IS correct — IMM is recognized as opcode 1).

## Offending block

- File: `c4_release/neural_vm/unified_compiler/ops/l8_ops.py:1454-1502`
  - `_layer8_multibyte_fetch_head_spec` — declarative attention head spec.
  - Baked via `make_layer8_multibyte_fetch_bake_op`
    (`l8_ops.py:1376-1443`), `bake_fn` at lines 1404-1411.
- Head index: `head_idx=3` of L8 attn (per
  `_L8_HEAD_LAYOUT_BY_NAME["layer8_multibyte_fetch_bake.head_3"]`).
- O writes: `AX_CARRY_LO/HI` (slots 32..47 and 48..63 of the head's
  output band) at the AX-marker row.

## Per-layer residual evidence

At fwd=6 (the AX-marker position, where the LM head predicts AX byte 0),
per-layer hook capture of FFN-input residuals shows the AX_CARRY values
diverge at L8 attention output:

```
IMM 0xDF (PASS): L8 AC_LO[k=5,+7.4]  AC_HI[k=15,+342.7]  -> AX byte 0 = 0xDF
IMM 0x2A (PASS): L8 AC_LO[k=12,+343] AC_HI[k=4,+342.7]   -> AX byte 0 = 0x2A
IMM 0xE0 (FAIL): L8 AC_LO[k=2,+347]  AC_HI[k=2,+22.1]    -> AX byte 0 = 0x01
IMM 0xE8 (FAIL): L8 AC_LO[k=10,+343] AC_HI[k=2,+22.1]    -> AX byte 0 = 0xE8
IMM 0xFF (FAIL): L8 AC_LO[k=5,+7.4]  AC_HI[k=1,+342.7]   -> AX byte 0 = 0xE8
```

PASS cases all show AC_HI peaks at +342.7 — the full softmax mass landed
on one byte (one head match). FAIL cases for 0xE0 show AC_HI peak only
+22.1 (split mass — softmax is between many ADDR_KEY-collision keys);
0xE8/0xFF show a peak at the wrong slot (carrying byte 0xE8/0xFF's bit
patterns mis-routed to nibble offsets that emit 0xE8 byte 0 + 0xFF byte 1
instead of 0xFF byte 0).

L7 head 5's K-side `OP_IMM` blocker (`ff4edb61`) closes one leak, and
the L10 `tail_lea_local_ax_marker_byte0_e8` rule's L34 residual probe is
clean (sum=1.0 at MARK_AX, threshold 7). The remaining surface is at L8
head 3.

## Why IMM ≥ 0xE0 specifically

The L8 head 3 attention scores are dot products of
- Q at the AX marker (slots 0..15 = `FETCH_LO+k`, slots 16..31 =
  `FETCH_HI+k`, slots 35..50 = `ADDR_KEY+32+k`)
- K at every byte position (slots 0..47 = `ADDR_KEY+0..47`)

For IMM in [0x00, 0xDF], a single byte position dominates the softmax
and the routed CLEAN_EMBED writes to AX_CARRY recover the IMM value
exactly. For IMM in [0xE0, 0xFF] the IMM byte's CLEAN_EMBED nibble
pattern + the bytecode layout reproduce a Q/K mass split among multiple
"aliased" key positions: the result is that AX_CARRY ends up routed
with the bit pattern `byte_0=0xE8 byte_1=0xFF` for 0xE8/0xFF (a fixed
two-byte template) and `byte_0=0x01` (== the IMM opcode byte itself!)
for 0xE0. The 0x01 byte for 0xE0 strongly suggests the IMM-byte-0 query
is collapsing onto the **opcode byte at PC+0** instead of the IMM byte
at PC+1 (the address-mid-nibble ADDR_KEY for IMM ≥ 0xE0 collides with
the opcode-byte ADDR_KEY pattern in a way the head can't disambiguate).

## Why NOT L5 or L10

- L5 opcode decode (`l5_ops.py:565`) reads OPCODE_BYTE_LO/HI and writes
  OP_IMM. For all 5 probe programs (PASS and FAIL) the model correctly
  routes through IMM dispatch: the failure is downstream of opcode
  decode, in the IMM-byte VALUE fetch, not in the opcode VALUE.
- L10 `tail_lea_local_ax_marker_byte0_e8` (`l10_ops.py:6587-6629`) was
  ruled out by the L34 residual probe (sum=1.0, well under threshold 7).
  Confirmed clean post the L7 head 5 K-side fix (`ff4edb61`).

## Recommended fix (NOT applied here — investigation only)

The L8 head 3 Q-side needs a stronger discriminator that breaks the
ADDR_KEY collision for IMM bytes in [0xE0, 0xFF]. Concrete approach:

1. Add a Q-side `AP(slot, BD.MARK_AX, ...)` term plus a K-side
   `AP(slot, BD.HAS_SE, -...)` blocker so the head can't bleed to
   ADDR_KEY-aliased non-IMM byte positions.
2. Alternatively, add an L9 or L10 corrective rule that, when
   `OP_IMM AND MARK_AX`, overwrites AX_CARRY_LO/HI from
   `CLEAN_EMBED_LO/HI` of the byte at addr=PC+1 via a position-
   addressed (not content-addressed) read.

Either fix lets the override at
`batched_pure_neural.py:2016-2024` be removed without regressing
test_xor_basic, test_add_16bit, test_add_carry_cascade.

## Files referenced

- `c4_release/neural_vm/unified_compiler/ops/l8_ops.py:1454-1502` —
  `_layer8_multibyte_fetch_head_spec` (the offending head).
- `c4_release/neural_vm/unified_compiler/ops/l8_ops.py:1376-1443` —
  `make_layer8_multibyte_fetch_bake_op` (bake site).
- `c4_release/neural_vm/batched_pure_neural.py:2016-2024` — IMM AX
  override (`b5cf7099`, stays load-bearing).
- `c4_release/neural_vm/unified_compiler/ops/l5_ops.py:226-373` — L5
  fetch heads (head 0 stages FETCH_LO/HI at AX from TEMP=PC+1; correct).
- `c4_release/neural_vm/unified_compiler/ops/l10_ops.py:6587-6629` —
  L10 `tail_lea_local_ax_marker_byte0_e8` (ruled out — L34 clean).

## Acceptance

- Override removal regresses test_xor_basic (0x2A→0), test_add_16bit
  (300→0), test_add_carry_cascade (0x100→1); matches prior brief.
- IMM 0xDF / 0x2A pass with override removed; IMM 0xE0/E8/FF fail.
  Discriminator (0xDF→PASS, 0xE0→FAIL) is structural at L8 head 3, not
  L34 or L5.
- Override restored before this commit; doc is the only on-disk
  change.
