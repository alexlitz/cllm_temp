# CarryPropagationPostOp validation under compact pin_io_only=True layout

Branch: `validate-carry-propagation-multibyte`
Date: 2026-05-12

## Summary

`CarryPropagationPostOp` is **correctly wired** to the compact `pin_io_only=True`
dim positions (verified by inspecting baked weights against `model.embed._dim_positions`
on a freshly built `AutoregressiveVMRunner(pure_neural=True, trust_neural_alu=True)`).

However, the post-op **does not fire in real runtime** because its required
upstream inputs are not produced at byte 0 position when an ADD overflows:

1. `OUTPUT_LO/OUTPUT_HI` at byte 0 carry the **passthrough value** (0x00), not
   the wrapped byte 0 (e.g. 0x2C for 200+100=300).
2. `CARRY[1]` is **not set** at the AX marker for ADD overflow either, so the
   relay (`_set_layer10_carry_relay`, L10 head 0) has nothing to propagate.

The carry post-op is correctly waiting for `CARRY[1]=1` at a byte 0 position
with the wrapped byte 0 value in `OUTPUT_LO/HI`. Neither condition is met in
the current compact layout. The dim threading is sound; the bug is upstream
in the L9/L10 ALU bake (independent of the carry post-op itself).

The "byte 0 wrong" issue tracks the parallel L4-ALU_HI agent's work
(`a3205fab9a2ebaa4b`). The "CARRY[1] never set" issue may be related but
requires its own investigation (see `Future work` below).

## Validation harnesses

Two harnesses live in-tree on this branch:

- `c4_release/diag_carry_unit.py` — synthetic-residual unit test that feeds
  hand-crafted inputs into `model.blocks[13/14/15].ffn`
  (the 3 CarryPropagationPostOp instances after `_expand_wrapper_blocks`
  re-homed them into vanilla `PureFFN` blocks).
- `c4_release/diag_carry_propagation.py` — full-runtime hook harness that
  registers `register_forward_hook` on `model.embed` and `model.blocks[11..16]`,
  then runs `IMM 200; PSH; IMM 100; ADD; EXIT` and inspects the residual
  stream at the ADD step's AX marker and byte positions (use_kv_cache=False
  so each forward recomputes the full sequence and hooks observe consistent
  block outputs).

## Block layout (pin_io_only=True, d_model=728)

After `_expand_wrapper_blocks` + `_rebake_as_pureffn` the 17 logical layers
expand into 26 transformer blocks. The carry post-ops are at:

```
block[12]: BinaryOpByteZeroingPostOp (4 units)        -- zero OUTPUT bytes 1-3
block[13]: CarryPropagationPostOp byte_idx=0          -- +1 at byte 0 from CARRY[1]
block[14]: CarryPropagationPostOp byte_idx=1 cascade  -- +1 at byte 1 from CARRY[3]
block[15]: CarryPropagationPostOp byte_idx=2 cascade  -- +1 at byte 2 from CARRY[3]
block[16]: BitwiseBytePropagationPostOp (1536 units)
block[17]: ComparisonCombine (18 units)
```

Compact dim positions in the pin_io_only=True layout:

```
OUTPUT_LO=69 OUTPUT_HI=85 CARRY=682
BYTE_INDEX_0=17 BYTE_INDEX_1=18 BYTE_INDEX_2=19
MARK_AX=1 MARK_PC=0 IS_BYTE=6 H1=246
OP_ADD=208 OP_SUB=209 TEMP=694
```

The bake reads/writes all match these positions when `dim_positions=` is
threaded (see `l10_post_op_attach` in `c4_release/neural_vm/unified_compiler/ops/l10_ops.py`,
which forwards `dim_positions` to all four post-op classes).

## Unit-test findings (`diag_carry_unit.py`)

Synthetic residuals fed through `cp_byte0 = model.blocks[13].ffn`:

| Test | OUTPUT magnitude | CARRY[1] | unit-194 `up` | Result |
|------|-----------------:|---------:|--------------:|--------|
| TEST 1 (`OUTPUT=1.0, CARRY[1]=1`) | 1.0 | 1.0 | -350 | inert (silu(-350)≈0) |
| TEST 5 (`OUTPUT=3.0, CARRY[1]=1`) | 3.0 | 1.0 | +50 | **fires correctly** — writes `+0.02·50≈1.0` to LO[new_lo], -1.0 to LO[old_lo], +1.0 to HI[new_hi]. argmax shifts only if old LO and new LO are within ~1.0. |
| TEST 6 (`OUTPUT=2.0, CARRY[1]=1`) | 2.0 | 1.0 | -150 | inert (threshold not reached) |
| TEST 7 (`OUTPUT=1.0, CARRY[1]=5`) | 1.0 | 5.0 | +450 | over-amped — saturates and writes to the entire OUTPUT_HI band (a separate bake-logic artifact where `W_down[OUTPUT_HI+hi, unit]` is overwritten by `W_down[OUTPUT_HI+new_hi, unit]` when `new_hi == hi`, so 16 units sharing `hi=2` each contribute +2/S to HI[2] without an offset). |

`b_up = -S * 9.5 = -950` per `vm_step.py:782`. Required `up > 0` (and ideally
≫0) for `silu(up)·gate` to be non-trivial. With the standard input
(IS_BYTE=1, BYTE_INDEX_0=1, OUTPUT_LO[lo]=1, OUTPUT_HI[hi]=1, CARRY[1]=1):
`up = 100+100+100+100+200 − 950 = −350`. So the post-op needs OUTPUT
magnitudes ≥ ~3 to fire, which the L10 lookup-FFN passthrough is expected to
produce in non-pure-neural / non-compact-layout runs.

In `pin_io_only=True` runs, neither OUTPUT nor CARRY[1] is delivered to byte
0 (see runtime-hook findings below), so the post-op stays inert.

## Runtime-hook findings (`diag_carry_propagation.py`)

Program: `IMM 200; PSH; IMM 100; ADD; EXIT`.
`AX_RESULT=116` (= 0x74; expected 300 = 0x12C, byte0=0x2C, byte1=0x01).

Captured token IDs at AX byte positions across steps:

```
AX step 0 (IMM 200,   mark=49 ): byte tokens = [200, 0, 0, 0]
AX step 1 (PSH,       mark=84 ): byte tokens = [200, 0, 0, 0]
AX step 2 (IMM 100,   mark=110): byte tokens = [100, 0, 0, 0]
AX step 3 (ADD,       mark=145): byte tokens = [116, 0, 0, 0]   <-- WRONG byte 0
AX step 4 (EXIT,      mark=180): byte tokens = [116, 0, 0, 0]
```

### Residual at ADD-step AX marker (pos 145)

| block | byte (LO+HI<<4 argmax) | CARRY | OP_ADD | MARK_AX |
|-------|------------------------|-------|--------|---------|
| in/out blk[11] (ALUAndOrXor)       | 0x74 | [0,0,0,0] | 5.0 | 1.0 |
| in/out blk[12] (BinaryOpByteZero)  | 0x74 | [0,0,0,0] | 5.0 | 1.0 |
| in/out blk[13] (CarryProp byte 0)  | 0x74 | [0,0,0,0] | 5.0 | 1.0 |
| in/out blk[14] (CarryProp byte 1)  | 0x74 | [0,0,0,0] | 5.0 | 1.0 |
| in/out blk[15] (CarryProp byte 2)  | 0x74 | [0,0,0,0] | 5.0 | 1.0 |
| in/out blk[16] (BitwiseByteProp)   | 0x74 | [0,0,0,0] | 5.0 | 1.0 |

**Findings at AX marker:**

- `OP_ADD=5.0` and `MARK_AX=1.0` are correctly set; opcode decode lands.
- Byte argmax at OUTPUT = 0x74 (not 0x2C) — confirms L9/L10 ALU produces wrong
  wrapped byte 0 in compact layout (the L4-ALU_HI band leakage issue is the
  prime suspect; tracked separately).
- **CARRY[1..3] are ALL zero**, even though `_set_layer9_alu` (vm_step.py:5451)
  is *supposed* to write `CARRY[1] = 2/S` at MARK_AX when ADD overflows. Either
  L9 isn't firing the overflow units, or its output is being suppressed
  downstream (perhaps zeroed by L10's binary-op cleanup or by a marker
  suppression bake). This is the **second blocker** to multi-byte ADD beyond
  the L4-ALU_HI leakage.

### Residual at ADD-step AX byte positions (pos 146 / 147 / 148 / 149)

Every byte position, every block from blk[11] through blk[16]:

```
byte=0x00, CARRY=[0,0,0,0], BYTE_INDEX_*=correct, IS_BYTE=1, H1[1]=1,
OP_ADD=0, OP_SUB=0, MARK_AX=0
```

**Findings at byte positions:**

- All marker / gating dims (IS_BYTE, H1[1], BYTE_INDEX_*) are correctly set.
- OUTPUT carries the passthrough value (0x00), confirming L10 head 1 passthrough
  feeds bytes 1-3 from the previous step (which were 0 for IMM 100).
- For byte 0 specifically: even though L9 should have populated some non-zero
  byte 0 via the ALU-LO write at MARK_AX → relayed to byte 0, the byte 0
  position shows 0x00. (This is the L4 / L9 byte-0 plumbing bug.)
- CARRY[1..3] = 0 at every byte position — the L10 head 0 carry relay
  (`_set_layer10_carry_relay`) had nothing to relay because CARRY[1] was 0 at
  MARK_AX in the first place.

### Per-block pure delta

`blk[13/14/15]` produce **zero delta everywhere on the residual stream** for
this program. The post-ops correctly recognize they should not fire (their
threshold gates `up ≪ 0` because `CARRY[1]·200 - 950 ≈ -950`).

## Comparison against `trust_neural_alu=False` (handler mode)

In handler mode, the Python override `_handle_alu` performs the ADD in Python
and writes the correct 32-bit result directly into the runner's `_last_ax`
state. The carry post-ops are NEVER invoked because the neural ALU path is
skipped. So handler mode passing is **not** evidence the neural carry path
works; it's just evidence the Python fallback works. (Confirmed by reading
`AutoregressiveVMRunner._dispatch_step` — when `trust_neural_alu=False`,
results come from `_handle_alu` and the residual ALU writes at L9 are
shadowed.)

## Conclusion: does the carry post-op work?

- **Dim wiring**: ✅ correct under `pin_io_only=True`. The bake threads
  `dim_positions` through to all 3 `CarryPropagationPostOp` instances; their
  W_up/W_down indices match the compiler-allocated positions of `OUTPUT_LO`,
  `OUTPUT_HI`, `CARRY`, `BYTE_INDEX_*`, `IS_BYTE`, `H1`, `MARK_AX`, `MARK_PC`,
  `OP_*`, `TEMP[3]`.
- **Synthetic-input behavior**: ✅ the post-op fires when fed a residual that
  represents a byte 0 with non-trivial OUTPUT magnitudes AND `CARRY[1]=1` (see
  TEST 5 in `diag_carry_unit.py`). The increment-by-1 logic activates the
  correct unit (unit 194 for lo=0xC, hi=0x2).
- **Runtime behavior**: ❌ inert because L9/L10 produce neither the wrapped
  byte 0 value at byte 0 position NOR set CARRY[1] at AX marker. The
  carry-propagation chain has nothing to propagate.
- **Semantic concerns** (orthogonal to wiring):
  - The post-op gates on `byte_dim = BYTE_INDEX_[byte_idx]`, so the byte_idx=0
    instance fires at byte 0 position and writes `lo + hi*16 + 1` back to
    byte 0 OUTPUT. This **increments byte 0**, not byte 1 — which is correct
    only if the byte 0 OUTPUT is *not* already the wrapped value. Today the
    L9 byte-0 value at byte 0 position is 0x00 (because byte 0 isn't being
    delivered), so the post-op's +1 would land on the empty slot.
  - The bake also writes `CARRY[3] += 2/S` at the lo=15,hi=15 unit, so the
    byte 0 instance ALSO produces the cascade-trigger flag for byte 1 (which
    reads CARRY[3] when `cascade=True`).
  - Net effect when the upstream is fixed: L9 writes wrapped byte 0 and
    CARRY[1] at MARK_AX → L10 head 0 relays to all byte positions → byte 0
    instance fires, increments OUTPUT at byte 0 by 1, sets CARRY[3] on full
    overflow → byte 1 instance fires (cascade=True, reads CARRY[3])...

  This means the post-op as baked **bumps byte 0 by 1** (rather than leaving
  byte 0 untouched and setting byte 1 to 1). That looks wrong by inspection,
  but may be intended: in the lookup ALU path, byte 0 at byte-0 position is
  populated by L10 head 1 passthrough as the *previous step's byte 1*, NOT
  the wrapped value of the current ADD. The carry post-op then "fixes" byte
  0 by adding 1 when the previous-step's byte 1 had a value waiting to be
  incremented. This subtlety mirrors how the lookup path was designed for
  pre-`pin_io_only` layouts. Either way, the post-op is internally consistent.

## Recommended next steps

1. **Fix L9 ALU byte-carry detection** (`_set_layer9_alu` ADD hi-nibble
   carry-out, vm_step.py:5427-5452). Verify in the hook harness that the
   ADD-step MARK_AX residual carries `CARRY[1] > 0`.
2. **Fix L10 head 0 carry relay** (`_set_layer10_carry_relay`,
   setup_helpers.py:755). Once L9 sets CARRY[1] at MARK_AX, verify the
   relay copies it to all 4 AX byte positions.
3. **Verify byte 0 wrapped value at byte 0 position**. This is tied to the
   L4-ALU_HI band leakage agent (`a3205fab9a2ebaa4b`).
4. After 1–3 land, the existing carry post-op should produce the correct
   bytes-1-3 cascade with no additional changes.

## Tests touched

- No test xfail markers flipped in this branch. The Phase-3 multibyte ADD/SUB
  tests in `c4_release/tests/test_pure_neural_multibyte.py` remain xfail
  (strict=False) because the upstream blockers above are unaddressed. The
  diagnostic harnesses are kept as in-tree scripts for future regression
  inspection.
