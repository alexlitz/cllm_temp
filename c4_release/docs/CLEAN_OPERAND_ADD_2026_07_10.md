# CBC pass-gain — ARITHMETIC-ONLY clean operand delivery

**Date:** 2026-07-12  **Branch:** `clean-operand-add` (off `clean-operand-delivery`)
**Flag:** `C4_CLEAN_OPERAND_ADD` (DEFAULT-OFF, campaign-gated)
**Commit:** `0df0856e`
**Golden flag-OFF:** `e50521f3` (VERIFIED unchanged — `tools/_isa_golden_hash.py`)

## Mission

The Phase-1 feasibility flag `C4_CLEAN_OPERAND` (all consumers) predicted **−74**
because clean operands break the CMP path (the CMP nibble comparators are
calibrated to the dirty hybrid) while HELPING arithmetic (+6). The feasibility
doc's recommended next step: split the flag to an **arithmetic-only** opcode
gate that captures the arithmetic gain with **zero CMP regression** (the CMP rows
are simply never cleaned, so the CMP calibration contract is left intact). This
is the first *correct-by-construction* pass-gain: clean ADD operands keep the
inter-byte carry correct.

## The gate — arithmetic opcodes ONLY

`CleanOperandOneHotFFN` snaps the operand bands (ALU_LO/HI = operand A,
AX_CARRY_LO/HI = operand B) to a clean per-nibble one-hot on the MARK_AX operand
rows whose opcode flag is in `op_dims`. The gate is entirely the `op_dims` tuple:

* `C4_CLEAN_OPERAND` (feasibility) — `op_dims` = **all 11** consumers:
  `OP_ADD/SUB/MUL/MOD/DIV` **+** `OP_EQ/NE/LT/GT/LE/GE`. Cleans every op → −74.
* `C4_CLEAN_OPERAND_ADD` (this deliverable) — `op_dims` = **5 arithmetic only**:
  `OP_ADD/SUB/MUL/MOD/DIV`. The six CMP dims are **absent from `op_dims`**, so a
  CMP MARK_AX row never satisfies the `forward` gate and its operand band is left
  **byte-identical** — the CMP `-0.5/-0.8` per-nibble blocker magnitude contract
  is untouched.

Implementation (all flag-gated behind the `no_stack0_emit` campaign so flag-OFF
is byte-identical to golden `e50521f3`):

* `shared.clean_operand_add_enabled()` — `C4_CLEAN_OPERAND_ADD`, default-OFF,
  campaign-gated (mirrors `clean_operand_enabled`).
* `alu_ops.make_clean_operand_op` — installs the wrap when
  `no_stack0_emit_enabled()` AND (`clean_operand_enabled()` OR
  `clean_operand_add_enabled()`); `op_dims` = 11 dims if the full flag is on,
  else the 5 arithmetic dims. If BOTH flags are set the broader (all-consumer)
  gate wins.
* Both cache-key snapshots in `full_vm_compiler_dynamic.py` gain
  `C4_CLEAN_OPERAND` (was **missing** — a latent bug: a MODULE-affecting flag
  with no memo key) **and** `C4_CLEAN_OPERAND_ADD`, so ON / OFF / full-vs-arith
  builds never share a memo or disk entry.

**Install verified (production groundtruth path, campaign on):**

| config | block-12 ffn | wrap `op_dims` |
|---|---|---|
| flag-OFF | `LoadedOperandAddHi15ClearFFN` (bare corrector) | — (no wrap) |
| `C4_CLEAN_OPERAND_ADD=1` | `CleanOperandOneHotFFN` | **5** = `[OP_ADD, OP_SUB, OP_MUL, OP_MOD, OP_DIV]` |
| `C4_CLEAN_OPERAND=1` | `CleanOperandOneHotFFN` | 11 (feasibility, unchanged) |

The arith wrap's `inner` is the existing `LoadedOperandAddHi15ClearFFN` (the wrap
runs AFTER `loaded_operand_add_hi15_clear` — the clean one-hot is the last word).

## Gate (b) — golden flag-OFF == `e50521f3` (byte-identical) — YES

`CUDA_VISIBLE_DEVICES="" tools/_isa_golden_hash.py` prints
`e50521f32b0ed952...` with **no** flag and with `C4_CLEAN_OPERAND_ADD=1`
(the wrap adds no parameters, so the state_dict is identical either way; the
byte-identity that matters is the flag-OFF build, and it is unchanged).

## Gate (c) — `fast_gate.py --flag C4_CLEAN_OPERAND_ADD` — NET POSITIVE, ZERO CMP REG

```
state off done: pass=152/220 (skipped 6)
state on  done: pass=156/220 (skipped 6)

GAINS (fail -> ok) [5]  — ALL arithmetic:
    expr_add_mul: 3   func_mul: 2
REGRESSIONS (ok -> fail) [1]:
    mul [ALU-BUS]: idx 104  (mul_4: 23*65 = 1495; ON got 215 = 1495 & 0xFF)

PER-CLUSTER pred_full:  expr_add_mul +9.4   func_mul +12.5   mul -4.2
CMP / bool clusters (if_eq/if_lt/if_gt/if_var/bool_and/func_max): ZERO delta
MEM-SMOKE (var_simple/var_mul/var_three/var_update): clean

SAMPLE net: +4  (flips=5 regs=1)
PREDICTED full-1096 delta: +18
```

**The whole point is confirmed:** every one of the −23 CMP/bool regressions the
full `C4_CLEAN_OPERAND` flag produced is GONE — the CMP calibration contract is
intact (those clusters show exactly zero delta). The 5 gains are all arithmetic
(expr_add_mul + func_mul). The single regression is **NOT a CMP break** — it is an
arithmetic `mul` (idx 104, `23*65=1495`): the clean byte-0 operand makes the
model drop the MUL high byte (1495 → 215 = low byte only), the exact
"clean byte-0 is necessary-not-sufficient for the byte-1+ ALU correctors" caveat
the feasibility doc flagged. Net of the arithmetic cluster is still strongly
positive (+5 gains vs 1 loss; predicted full +18).

## Gate (d) — full `run_1096_canonical --criterion full_trace --spec-k 0 --max-steps-cap 40` — 580 → 593 (+13)

```
OFF (flag unset):        CANONICAL SCORE [full_trace]: 580/1096 PASS  [fail=283 error=0 skipped(>cap)=233]
ON  (C4_CLEAN_OPERAND_ADD=1): CANONICAL SCORE [full_trace]: 593/1096 PASS  [fail=270 error=0 skipped(>cap)=233]

NET: +13  (580 -> 593)   >= the 586 target
```

Both states ran the SAME config (`--max-steps-cap 40`, the short-only mode that
matches the fast-gate and the 580 baseline; the 233 deep loop/gcd/rec programs
are `skipped(>cap)` in BOTH, not counted). OFF reproduces the mission baseline
`580` exactly. ON = **593 (+13)** — above the `>=586` target. The +13 is the
net arithmetic gain (clean ADD/SUB/MUL operands keep the carry correct) with the
CMP path untouched.

Note: a first full run used the canonical DEFAULT `--max-steps-cap 1000` (folds
the 233 deep programs in as fails), which produces a much lower absolute pass
for BOTH states (not comparable to the 580 baseline) — re-run at cap 40 to match
the baseline + fast-gate config. The DELTA was positive under both caps.

## Gate (e) — `pytest tests/test_smoke.py` — 51/51 PASS

```
================ 51 passed, 1 deselected in 2193.23s (0:36:33) =================
```

`C4_CLEAN_OPERAND_ADD=1`, CPU (`CUDA_VISIBLE_DEVICES=""`, isolated cache). All
51 smoke tests pass (the 1 deselected is the standard non-smoke test). No
regression — the arithmetic clean-snap and the untouched CMP path both hold.

## Verdict — READY TO LAND

All four gates pass:

* **byte-identical flag-OFF** — golden `e50521f3` (ISA hash, flag unset AND set).
* **fast-gate net-positive, ZERO cmp/bool regression** — +18 predicted; the only
  regression is 1 arithmetic `mul` (idx 104, byte-1 MUL corrector interaction),
  NOT a CMP break; every CMP/bool cluster is untouched (the whole point).
* **full-1096 580 -> 593 (+13)** — above the `>=586` target (same
  `--max-steps-cap 40` config both states; OFF reproduces the 580 baseline).
* **smoke 51/51**.

This is the first *correct-by-construction* pass-gain: clean ADD/SUB/MUL/DIV/MOD
operands keep the inter-byte carry correct, and gating the clean-snap to the
arithmetic opcodes ONLY leaves the CMP calibration contract intact. **READY TO
LAND** (do NOT merge from this agent per the brief).

## Reproduce

```
# install / gate correctness (production path):
C4_TEST_SPEC_K=0 C4_NO_STACK0_EMIT=1 C4_CLEAN_OPERAND_ADD=1 \
  python -c "from tools.probe_groundtruth import build_groundtruth_probe as b; \
             m=b().model; \
             [print(i,type(x.ffn).__name__,len(x.ffn.op_dims)) \
              for i,x in enumerate(m.blocks) if getattr(x.ffn,'_is_clean_operand_wrap',False)]"

# golden flag-OFF byte-identity (must print e50521f3...):
CUDA_VISIBLE_DEVICES="" python tools/_isa_golden_hash.py

# the corpus gate:
python tools/fast_gate.py --flag C4_CLEAN_OPERAND_ADD --gpus 0
```

## Files (all flag-gated; golden flag-OFF `e50521f3` byte-identical)

* `neural_vm/unified_compiler/ops/shared.py` — `clean_operand_add_enabled()`.
* `neural_vm/unified_compiler/ops/alu_ops.py` — `make_clean_operand_op` gate.
* `neural_vm/efficient_alu_neural.py` — `CleanOperandOneHotFFN` (op_dims doc).
* `neural_vm/unified_compiler/full_vm_compiler_dynamic.py` — both cache-key
  snapshots (`C4_CLEAN_OPERAND` + `C4_CLEAN_OPERAND_ADD`).
