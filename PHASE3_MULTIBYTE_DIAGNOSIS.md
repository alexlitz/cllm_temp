# Phase 3 Multibyte AX Tests — Diagnosis (2026-05-10)

## Test status under `--runxfail`

Ran `c4_release/tests/test_pure_neural_multibyte.py` with `--runxfail`:

| Test | Status | Notes |
|------|--------|-------|
| `test_six_imms` | **XPASS** (passes) | Phase 1 multi-IMM work landed |
| `test_ten_imms` | **XPASS** (passes) | Phase 1 multi-IMM work landed |
| `test_add_overflow_300` | FAIL | 200+100 → 124 (expect 300) |
| `test_add_overflow_510` | FAIL | 255+255 → 0xFAFAFAFE-ish |
| `test_mul_to_high_byte_300` | FAIL | 30*10 → 0 |
| `test_mul_two_byte_result_10000` | FAIL | 100*100 → 0 |
| `test_mul_max_byte_pair` | FAIL | 255*255 → 4210752255 (0xFAFAFAFF) |
| `test_sub_borrow_into_byte1` | FAIL | depends on ADD |
| `test_sub_borrow_zero_byte0` | FAIL | depends on ADD |

So 2 of the 9 already pass; **`test_six_imms` and `test_ten_imms` can have their xfail removed.**

## Root cause is NOT a Phase-3-specific bug

The Phase 3 tests fail because **byte 0 itself is wrong in pure_neural mode** — not
because of any multi-byte routing bug. Below are pure_neural results for ADDs and MULs
that do not even require multi-byte semantics:

```
pure_neural mode (no set_active_opcode injection):
  1 + 2  = 19    (expect 3)    — high nibble +1
  5 + 3  = 24    (expect 8)    — high nibble +1
  10 + 20 = 46   (expect 30)   — high nibble +1
  3 + 4  = 23    (expect 7)    — high nibble +1
  50 + 50 = 68   (expect 100)  — high nibble different
  100 + 100 = 120 (expect 200) — high nibble different
  10 + 32 = 58   (expect 42)   — same as A2 in-flight bug
  200 + 100 = 124 (expect 300) — byte 0 wrong AND no byte-1 carry
  200 + 56 = 80  (expect 256)  — byte 0 wrong AND no byte-1 carry
  255 + 1 = 0xFAFAFAFF (expect 256) — sign-extension to all bytes
  
  2 * 3 = 0      (expect 6)    — MUL returns 0
  5 * 5 = 0      (expect 25)
  255 * 255 = 0xFAFAFAFF (expect 65025) — sign-extension
```

The same programs in `trust_neural_alu=True` mode (NOT pure_neural) are mostly fine:

```
trust_neural_alu mode (with set_active_opcode):
  5 + 3 = 8        OK
  200 + 100 = 44   (byte 0 only, truncated — expected for 8-bit AX)
  10 + 32 = 42     OK
```

## Mechanism

- `pure_neural=True` skips `set_active_opcode(opcode)` calls on the model.
- `set_active_opcode` swaps each block's FFN to opcode-specific MoE sub-matrices.
- Without it, the FFN runs with `_full` matrices — every unit is active.
- Units gated on `BD.OP_ADD` (via W_gate) ARE supposed to discriminate via the gate
  value, but the residual stream has spurious activations in OP-flag dims at
  positions other than `MARK_AX`, so additional units misfire and bias the result.

The high-nibble +1 pattern on small ADDs is consistent with the carry_in=1 unit
firing in addition to the carry_in=0 unit (CARRY discrimination breaks down when
gate ≈ 0 vs ≈ 1 difference is insufficient at full-matrix scale).

The 0xFAFAFAFF pattern on 255+anything and 255×255 matches **A1 in-flight
(sign-extension when low-byte MSB set)**. Bytes 1-3 = 0xFA = 250 = roughly
255 minus the byte 0 correction.

## Why Phase 3 cannot land yet

Every Phase 3 multibyte test does:
```
  IMM a
  PSH
  IMM b
  {ADD|SUB|MUL}
  EXIT
```

If byte 0 of the ALU result is already corrupted (which it is, in pure_neural mode),
no multi-byte routing fix can produce the correct 32-bit result. Phase 3 work is
**blocked on A1 (255 sign-ext) and A2 (ADD byte-0 in pure_neural)**.

## What can land now

- Remove `@pytest.mark.xfail` from `test_six_imms` and `test_ten_imms` — both XPASS.
- The remaining 7 should stay xfail until A1+A2 land. After A1+A2, re-run this
  diagnostic; if byte 0 is correct but bytes 1-3 are wrong, the multi-byte routing
  needs its own fix (probably in L10 attn / CarryPropagationPostOp byte_idx=0/1/2
  thresholds when running with full matrices).

## Files

- `diag_compare_modes.py` — reproduces the `trust_neural_alu` vs `pure_neural`
  divergence on the same ADD programs.
- `diag_add_simple.py` — pure_neural ADDs only.
- `diag_add_v2.py` — short ADDs (no overflow) showing the +16 high-nibble offset.
- `diag_mul_simple.py` — pure_neural MULs all return 0 or sign-extended garbage.
- `diag_add_overflow.py` — single 200+100=300 trace (returns 124).
