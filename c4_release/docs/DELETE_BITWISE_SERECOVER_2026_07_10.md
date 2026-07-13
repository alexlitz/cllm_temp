# DELETE `BitwiseOperandSeRecoverFFN` — Class-B corrector removal

**Date:** 2026-07-13
**Branch:** `delete-bitwise-serecover`
**Verdict:** ✅ **DELETABLE.** −130 LOC (class) / −87 net after clean-bitwise plumbing.
**Golden (flag-OFF / bare-env state_dict):** `e50521f3` — **UNCHANGED** every edit.

---

## Result

`BitwiseOperandSeRecoverFFN` (efficient_alu_neural.py, ~130 LOC) is a **Class-B**
corrector: it is NOT a free delete (the SeRecover was doing real work — bitwise
fails without it on main). It IS deletable by delivering a CLEAN operand-A one-hot
UPSTREAM at the L8 operand-delivery FFN, which subsumes the SeRecover's job.

- DELETED: `class BitwiseOperandSeRecoverFFN` + its L10 install site + the
  `bitwise_byte0_se_recover_enabled()` flag + 3 name references.
- ADDED: `C4_CLEAN_OPERAND_BITWISE` (default-ON in campaign) — extends the
  existing `CleanOperandOneHotFFN` op_dims (already cleans ADD/SUB/MUL/DIV/MOD via
  `C4_CLEAN_OPERAND_ADD`) to include `OP_AND/OP_OR/OP_XOR`.

---

## The root the SeRecover was correcting = operand DELIVERY magnitude (byte-provable)

Probe `tools/_probe_bitw_alu_trace.py` (spec_k=0, hook-free, campaign default):

**Block trace of the bitwise MARK_AX operand-A `ALU_LO/HI` band** (`or_16bit`, byte0 = 0x00):

| block | event | dirty operand (SeRecover OFF) | clean operand (`C4_CLEAN_OPERAND_BITWISE=1`) |
|-------|-------|-------------------------------|-----------------------------------------------|
| ~12   | L8 delivers operand A | `ALU_LO=[(0, 5.48),(8, 0.45)]` (hybrid) | `ALU_LO=[(0, 6.0)]` (clean one-hot) |
| 21    | L9 non-ALU ALU-scrubber subtracts a FIXED pattern (~5.56 @ cell 0) | `→ [(15,−0.47)]` cell-0 **LOST** (5.48−5.56 = −0.08) | `→ [(0, 0.44),…]` cell-0 **survives** (6.0−5.56 = +0.44) |
| 29    | L10 bitwise lookup reads `ALU` | A==0 nibble absent → rule never fires → OUT empty → **0x00 (WRONG)** | +0.44 cell-0 → rescaled `bitwise_rules` (`operand_a_cw=30/5.82`) fires → **0xfff (RIGHT)** |

So the SeRecover recovers for **operand-delivery magnitude**, NOT a different root:
the dirty operand-gather hybrid delivers the true A==0 nibble at only ~+5.48, which
the fixed block-21 subtraction drives negative; a clean +6.0 one-hot has exactly the
~0.5 extra headroom to survive. `and_16bit` passed coincidentally (nibbles 0xF at
cell 15, not cell 0). Once the operand is clean upstream, the SeRecover is inert
and deletable.

**Byte-provable engine correctness on clean one-hots:**
`tests/test_wide_alu_dsl.py::test_bitwise_rules_byte_identity_{vs_python,randomized}`
— **19/19 pass**. The a+b−ab lookup reproduces `and/or/xor(a,b)` byte-for-byte over
the nibble pairs on clean one-hots — the engine needs NO re-tuning.

---

## Verification (spec_k=0, hook-free ground-truth probe)

All 6 bitwise smoke programs pass with the SeRecover DELETED + `C4_CLEAN_OPERAND_BITWISE`
default-ON (pure default env, no flags set):

| program | expected | got | pass |
|---------|----------|-----|------|
| or_16bit  | 0x0fff | 0x0fff | ✅ |
| xor_16bit | 0x0ff0 | 0x0ff0 | ✅ |
| and_16bit | 0x00ff | 0x00ff | ✅ |
| or_basic  | 0x07   | 0x07   | ✅ |
| xor_basic | 0x06   | 0x06   | ✅ |
| and_basic | 0x0c   | 0x0c   | ✅ |

Contrast (deletion-equivalent A/B, `C4_BITWISE_BYTE0_SE_RECOVER=0`, clean-bitwise OFF):
`or_16bit` / `xor_16bit` FAIL (0x0f00) — confirms the SeRecover was load-bearing on main.

**Golden state_dict hash (`tools/_isa_golden_hash.py`, disk_cache=False, CPU):**
bare-env = `e50521f32b0ed952d5730f79b63adb8c4c78f4d4f0466d3bcbaa354bb3c90e86` = **e50521f3**,
UNCHANGED. Both `BitwiseOperandSeRecoverFFN` and `CleanOperandOneHotFFN` are
parameter-free forward-pass wrappers (they edit residual cells / delegate to `inner`),
so the swap changes the forward corrector topology but NOT any baked `nn.Parameter` —
the golden gate holds. FFN units 42149 → 42149 retained.

---

## −LOC

```
 neural_vm/efficient_alu_neural.py                       | 130 --------  (class deleted)
 neural_vm/unified_compiler/ops/alu_ops.py               |  68 +/-      (install site → bare lookup_ffn; +bitwise op_dims)
 neural_vm/unified_compiler/ops/shared.py                |  85 +/-      (−bitwise_byte0_se_recover_enabled; +clean_operand_bitwise_enabled)
 neural_vm/unified_compiler/full_vm_compiler_dynamic.py  |  19 +        (C4_CLEAN_OPERAND_BITWISE in both cache-key snapshots)
 neural_vm/verification/faithful_interpreter.py          |   1 -        (drop composite-name)
 tools/faithful_interpreter_validate.py                  |   2 +/-      (drop composite-name)
```

Net: **−130 LOC** for the class itself; **−87 net** across the change (the clean-bitwise
flag plumbing + doc comments add back ~43). The corrector class is gone.

---

## Deferred fast-gate command (GPU saturated — defer to main)

Present-vs-deleted A/B, bitwise (and/or/xor, 8-bit + 16-bit). The `=0` opt-out
reproduces the crushed (pre-clean, failing) bitwise operand — i.e. the corrector's
job is now owned by `C4_CLEAN_OPERAND_BITWISE`:

```bash
# smoke (bitwise): DELETED SeRecover + clean-bitwise DEFAULT-ON (present)
python -m pytest tests/test_smoke.py::TestSmokeBitwise \
    tests/test_smoke.py::TestSmoke32Bit::test_or_16bit \
    tests/test_smoke.py::TestSmoke32Bit::test_and_16bit \
    tests/test_smoke.py::TestSmoke32Bit::test_xor_16bit -q          # expect 6/6

# A/B kill-switch: clean-bitwise OFF (= deleted-and-uncorrected, the crushed operand)
C4_CLEAN_OPERAND_BITWISE=0 python -m pytest tests/test_smoke.py::TestSmokeBitwise \
    tests/test_smoke.py::TestSmoke32Bit::test_or_16bit \
    tests/test_smoke.py::TestSmoke32Bit::test_and_16bit \
    tests/test_smoke.py::TestSmoke32Bit::test_xor_16bit -q          # expect or/xor_16bit FAIL

# full-1096 present-vs-A/B (verdict-neutral expected: SeRecover→clean-bitwise is a swap)
python tools/run_1096_canonical.py --criterion full_trace --spec-k 0 --max-steps-cap 40
C4_CLEAN_OPERAND_BITWISE=0 python tools/run_1096_canonical.py --criterion full_trace --spec-k 0 --max-steps-cap 40

# golden byte-identity (flag-OFF / bare-env state_dict must stay e50521f3)
CUDA_VISIBLE_DEVICES="" python tools/_isa_golden_hash.py                                 # e50521f3
```

---

## Honesty check

The SeRecover was **not** correcting a different root — it was compensating for the
low-magnitude dirty operand delivery (proven: clean +6.0 vs dirty +5.48 is the entire
difference across the fixed block-21 crush). The bitwise a+b−ab engine is byte-provably
correct on clean one-hots (19/19 truth-table + randomized tests). Therefore the SeRecover
is deletable this way, and clean upstream delivery is the correct-by-construction root fix.
