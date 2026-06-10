# Smoke triage — 2026-06-10

Status: **30/51 PASS, 21 FAIL, 1 deselected** in this worktree at HEAD
`5592f110` (branch `main`). Smoke wall: 184.7s.

Run command:
```
pytest c4_release/tests/test_smoke.py --tb=line -v 2>&1 | tee /tmp/smoke_full.log
```

This doc is a **root-cause triage matrix** so the fix fleet has accurate
targets. Existing root-cause docs are cited per cluster; do not
re-investigate them.

## TL;DR — Failure clusters by ROOT CAUSE

| # | Cluster | Tests | Likely surface | In-flight agent? |
|---|---|---|---|---|
| 1 | **AX bytes 1-3 = 0xFF leak (32-bit)** | 7 | autoregressive AX byte-emit chain bytes 1..3 | YES — known agent |
| 2 | **AX-zero / value-1 collapse (LHS leak)** | 4 | binop AX synth dropping to `last_pushed_value=1` or 0 | NO |
| 3 | **CMP / GT-GE positional flag emit** | 4 | comparison combine writes RHS imm into AX (or fails to clear) | NO (Wave B C1 may help) |
| 4 | **JSR PC fetch / return address** | 1 | function-call return value clobbered with `0x1008` | NO |
| 5 | **SUB low-byte zero, high-byte 0xFF** | 1 | SUB AX_byte0 zero + 0xFF leak on byte 1 (mixed cluster 1+2) | partial (cluster 1) |
| 6 | **ADJ failing to PSH-pop result (basic)** | 1 | ADJ delivers 10 instead of 42 (stack pop logic) | NO |
| 7 | **Memory 16-bit roundtrip — high byte missing** | 1 | SI stores byte 0 only; LI loads byte 0 | NO |
| 8 | **MUL/DIV/MOD collapse to 1** | 3 | mul/div/mod synth returning RHS=1 collapse | NO |

Picks for highest ROI: (a) Cluster 1 AX 0xFF leak (covered) clears 7;
(b) Cluster 3 CMP positional clears 4; (c) Cluster 2 LHS=1 leak clears
4 (often the same root as 3).

## Failure matrix — 21 tests

Byte values are little-endian (`byte 0 = LSB`).

| # | Test | Expected | Got (dec) | Got (LE bytes) | Cluster |
|---|---|---|---|---|---|
|  1 | basic::test_sub_basic              | 42       | 65280       | `[0, 255, 0, 0]`   | 5 (SUB) |
|  2 | basic::test_mul_basic              | 42       | 1           | `[1, 0, 0, 0]`     | 8 (MUL/DIV/MOD) |
|  3 | basic::test_div_basic              | 42       | 1           | `[1, 0, 0, 0]`     | 8 (MUL/DIV/MOD) |
|  4 | basic::test_mod_basic              | 3        | 1           | `[1, 0, 0, 0]`     | 8 (MUL/DIV/MOD) |
|  5 | functioncall::test_simple_function | 42       | 4104        | `[8, 16, 0, 0]`    | 4 (JSR PC fetch) |
|  6 | bitwise::test_or_basic             | 0x3F     | 0x00FFFF3F  | `[63, 255, 255, 0]`| 1 (AX 0xFF leak) |
|  7 | bitwise::test_and_basic            | 42       | 1           | `[1, 0, 0, 0]`     | 2 (LHS leak / AND=1) |
|  8 | bitwise::test_xor_basic            | 42       | 0x00FFFF35  | `[53, 255, 255, 0]`| 1 (AX 0xFF leak) |
|  9 | comparison::test_eq_false          | 0        | 1           | `[1, 0, 0, 0]`     | 3 (CMP positional) |
| 10 | comparison::test_ne_true           | 1        | 0           | `[0, 0, 0, 0]`     | 3 (CMP positional) |
| 11 | comparison::test_gt_true           | 1        | 16          | `[16, 0, 0, 0]`    | 3 (CMP positional) |
| 12 | comparison::test_ge_true           | 1        | 16          | `[16, 0, 0, 0]`    | 3 (CMP positional) |
| 13 | address::test_adj_sp               | 42       | 10          | `[10, 0, 0, 0]`    | 6 (ADJ pop) |
| 14 | memory::test_si_li_16bit_value     | 0x1234   | 52          | `[52, 0, 0, 0]`    | 7 (SI/LI hi-byte) |
| 15 | bit32::test_add_16bit              | 300      | 65356       | `[76, 255, 0, 0]`  | 1 (AX 0xFF leak) |
| 16 | bit32::test_add_carry_cascade      | 256      | 233         | `[233, 0, 0, 0]`   | 2 (carry not propagated) |
| 17 | bit32::test_sub_16bit              | 255      | 4294967295  | `[255, 255, 255, 255]` | 1 (AX 0xFF leak) |
| 18 | bit32::test_or_16bit               | 4095     | 16777215    | `[255, 255, 255, 0]` | 1 (AX 0xFF leak) |
| 19 | bit32::test_and_16bit              | 255      | 1           | `[1, 0, 0, 0]`     | 2 (LHS leak / AND=1) |
| 20 | bit32::test_xor_16bit              | 4080     | 16777200    | `[240, 255, 255, 0]` | 1 (AX 0xFF leak) |
| 21 | bit32::test_mul_overflow           | 500      | 1           | `[1, 0, 0, 0]`     | 8 (MUL collapse) |

## Cluster 1 — AX bytes 1-3 = 0xFF leak (7 tests)

Tests: `or_basic`, `xor_basic`, `add_16bit`, `sub_16bit`, `or_16bit`,
`xor_16bit`, `add_16bit`.

Pattern: byte 0 of `got` matches the expected low byte, but bytes 1, 2
(and sometimes 3) are spuriously `0xFF`. Examples:

- `or_basic`: 0x3F got `0x00FFFF3F` (low byte 0x3F correct, bytes 1-2 0xFF).
- `add_16bit`: 300 (0x012C) got `0x0000FF4C` (low byte 0x4C correct, byte 1 0xFF).
- `sub_16bit`: 255 (0xFF) got `0xFFFFFFFF` (low byte correct, bytes 1-3 0xFF).
- `or_16bit`: 0x0FFF got `0x00FFFFFF` (byte 1 should be 0x0F, got 0xFF).

Suspected layer: autoregressive AX byte-emit chain for binop steps (the
"non-collapsed binary-ALU" override path described in
`docs/32BIT_CASCADE_REAL_SURFACE_2026_06_07.md`). The `wide_*_rules` are
1-byte; multi-byte cascade is handled in `AddSub5StageBlock` at the ALU.
The bug surface is the **per-position AX byte-emit for bytes 1..3**,
which here leaks an "all-ones" sentinel. Plausible upstream culprit is a
stale residual or position-marker confusion that fires `0xFF` on every
non-LSB AX byte position. Cross-reference
`docs/EDGE_POW2_OP_IMM_LEAK.md`, `docs/L29_AMPLIFIER_ATTRIBUTION_2026_06_09.md`.

In-flight: per the brief, "a separate agent is on this".

## Cluster 2 — LHS/AX collapse to 1 (4 tests)

Tests: `and_basic`, `and_16bit`, `add_carry_cascade` (variant).

Pattern: AND result is `1` regardless of inputs — neither expected
`0x2A`/`0xFF` nor any predictable byte transformation. Looks like the
runner is using `last_pushed_value=1` and writing `1` directly into AX
(skipping the AND combine entirely), or the override returns `(1 AND
anything)=1` (likely the LHS=1 of `IMM 1`-then-PSH being read as LHS at
the binop step). `add_carry_cascade` (got `233 = 0xE9`) is included here
because: 0xFF+1 should carry to 0x100, but byte 0 = 0xE9 with no carry
to byte 1 — i.e. the low-byte ADD also dropped operands.

Suspected layer: PSH/AX shadow — `last_pushed_value` or `prev_ax`
shadow is wrong on these steps. May share root with
`docs/SMOKE_SERIAL_MUL_SI_LI_2026_06_09.md` (multiplication / push
chain) or `docs/EDGE_POW2_OP_IMM_LEAK.md`.

No in-flight agent identified.

## Cluster 3 — CMP / comparison positional emit (4 tests)

Tests: `eq_false`, `ne_true`, `gt_true`, `ge_true`.

Pattern: comparison op result polarity flipped or wrong magnitude:

- `eq_false` (expected 0): got 1 — EQ returns TRUE when operands differ.
- `ne_true` (expected 1): got 0 — NE returns FALSE when operands differ.
  (NE/EQ are exact inversions of each other, both produce `EQ`-style 1.)
- `gt_true`, `ge_true` (expected 1): got `16` — the **immediate value
  16** (probably the second operand 10... no, `0x10` byte-decoded looks
  like the IMM literal `Opcode.GT (=16) `or `0x10` byte). Note GT's
  opcode in `Opcode` enum likely correlates: the AX is holding the
  opcode byte at the emit step.

Suspected layer: L10 `_layer10_alu_cmp_combine_rules` and CMP combine
default branch. The recent commit b9c8c8db migrated these to STEP_END;
this triage may indicate the migration broke the GT/GE/EQ-false path or
left an incomplete override.

Wave B Cluster 1 (docs/WAVE_B_CLUSTER_1_PLAN_2026_06_10.md) is the
plan-of-record for this exact rule family; the migration appears to be
mid-flight (commit b9c8c8db landed L10 rows 1-7) so a follow-up
cleanup is needed.

## Cluster 4 — JSR PC fetch / return value (1 test)

Test: `test_simple_function` (expected 42, got 4104 = `0x1008`).

Pattern: `[8, 16, 0, 0]` LE = `0x1008`. Byte 0 = 8 (likely the JSR's
target offset imm, `JSR 3` has packed encoding that yields byte 8 of
the next instruction position). Byte 1 = `0x10` = Opcode.LEV (`0x10`)
or LEA/PSH (Opcode enum has values in low byte). The runner returns
something that mixes return PC bytes into the EXIT AX result instead of
the function body's `IMM 42` value.

Suspected layer: L6/L10 JSR PC fetch byte attribution, or L8 ENT/LEV
frame management. Likely the return-from-LEV step writes the saved PC
into AX instead of the IMM(42).

No in-flight agent identified. (May be the same root as
`docs/L10_PSH_ADDR_ENT_BUG.md` from user-memory note
`project_l10_psh_addr_ent_bug.md`.)

## Cluster 5 — SUB low byte zero + high byte 0xFF (1 test)

Test: `test_sub_basic` (expected 42, got 65280 = `0xFF00`).

Pattern: `[0, 255, 0, 0]` — low byte completely wrong (zero, not 42 nor
0xD8 = 42-50 wrap), high byte is `0xFF` leak. Both halves are broken,
implying SUB combine fired neither low nor hi correctly: low byte
produced zero (likely missed the `(50 - 8)` subtraction) and high byte
got the same 0xFF leak as Cluster 1.

Suspected layer: dual surface — `_layer8_alu_sub_lo_rules` /
`_layer8_alu_sub_borrow_rules` failing for the small-magnitude case
(Wave B Cluster 2, mid-flight per commit b1e1f91e), and the same AX
high-byte 0xFF leak as Cluster 1.

In-flight: partial (cluster 1 covers high byte; low byte is in Wave B
C2 territory).

## Cluster 6 — ADJ pop failure (1 test)

Test: `test_adj_sp` (expected 42, got 10).

Pattern: `IMM 42 ; PSH ; ADJ 8 ; EXIT` — `got=10` is unexpected. Neither
42, nor 8 (ADJ imm), nor 0 (cleared AX). Value 10 (`0x0A`) is suspicious
— could be the byte representation of `Opcode.ADJ` (likely `0x0A`).

Suspected layer: ADJ + AX shadow: the runner emits the **opcode byte**
of ADJ into AX rather than preserving the post-PSH AX value of 42. ADJ
should not modify AX, only SP. The bug is "ADJ overwrites AX with the
opcode byte". Likely L8/L10 ADJ scope leakage.

No in-flight agent identified.

## Cluster 7 — Memory SI/LI 16-bit roundtrip (1 test)

Test: `test_si_li_16bit_value` (expected 0x1234 = 4660, got 52 = `0x34`).

Pattern: byte 0 correct (0x34), byte 1 missing (should be 0x12, got 0).
SI stored only the low byte; LI loaded only the low byte. The
multi-byte memory roundtrip is fundamentally broken for hi-byte.

Suspected layer: L13/L14/L15 SI/LI hi-byte attribution.
`docs/L14_SI_LI_CONSUMER_DIAGNOSTIC_2026_06_09.md` already targets this
area; `docs/L15_LI_STACK0_BYTE_ATTRIBUTION` per memory note covers
adjacent failure mode.

No active fix agent identified.

## Cluster 8 — MUL/DIV/MOD collapse to 1 (3 tests)

Tests: `mul_basic` (6*7→1), `div_basic` (84/2→1), `mod_basic` (43%10→1),
plus `mul_overflow` (100*5→1).

Pattern: result is always `1`. The runner is returning the RHS=1 leak,
or the AX is being overwritten with a "result-present" flag that holds
value 1. Same value-1 leak as Cluster 2's AND tests, suggesting common
root in the binop step's AX byte emission.

Suspected layer: L11/L12 MUL partial + combine (migrated to STEP_END in
commit 83cf2636 — Wave B Cluster 4). Possibly the migration left a
default-branch that emits `1` when the override does not fire.

`docs/DIV_22_FAILING_ATTRIBUTION_2026_06_09.md` already targets DIV
attribution. `docs/LONG_DIVISION_BUG36_2026_06_09.md` and
`docs/LONG_DIVISION_FFN_RULE_INFEASIBILITY_2026_06_09.md` cover the
DIV/MOD declarative-rule infeasibility — that gap predicts the present
collapse.

No active fix agent.

## Clusters NOT covered by existing in-flight agents

The 21 failures factor into **5 distinct fix surfaces**, and **4 of
them have NO active agent**:

1. **Cluster 1 (AX 0xFF leak)** — covered.
2. **Cluster 2 (LHS=1 / AND collapse)** — UNCOVERED.
3. **Cluster 3 (CMP/GT-GE positional)** — partially mid-flight via
   Wave B C1; needs follow-up override.
4. **Cluster 4 (JSR return value)** — UNCOVERED.
5. **Cluster 6 (ADJ overwrites AX)** — UNCOVERED.
6. **Cluster 7 (SI/LI hi-byte)** — UNCOVERED (diagnostics exist, no
   active fix).
7. **Cluster 8 (MUL/DIV/MOD collapse)** — UNCOVERED.

Highest-priority new agents: Cluster 8 (3 tests, well-diagnosed via
existing DIV/MOD docs); Cluster 3 (4 tests, Wave B C1 mid-flight);
Cluster 2 (3-4 tests, may share root with C8).
