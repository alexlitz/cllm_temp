# LAND — `C4_CLEAN_OPERAND_ADD` DEFAULT-ON (first correct-by-construction pass-gain)

**Date:** 2026-07-13
**Flag flipped:** `C4_CLEAN_OPERAND_ADD` DEFAULT-OFF → **DEFAULT-ON** (campaign-gated; opt out `=0`)
**Also registered:** `C4_CLEAN_OPERAND` (feasibility, DEFAULT-OFF) in both cache-key snapshots
**Net verdict:** full-1096 `full_trace` **580 → 593 (+13)**, byte-identical-OFF, smoke 51/51, opaque 0
**Status:** LANDED to main (verified net-positive land-to-main)

---

## What this is

The first *correct-by-construction* (CBC) pass-gain. `CleanOperandOneHotFFN`
wraps the L8 main FFN (physical block 12 — the operand-delivery block) and snaps
the ALU_LO/HI (operand A) + AX_CARRY_LO/HI (operand B) bands to a clean
per-nibble one-hot on the MARK_AX operand rows whose opcode flag is in
`op_dims`. The `C4_CLEAN_OPERAND_ADD` slice gates the clean-snap to the FIVE
ARITHMETIC opcodes ONLY (`OP_ADD/SUB/MUL/DIV/MOD`); the six CMP dims
(`OP_EQ/NE/LT/GT/LE/GE`) are ABSENT from `op_dims`, so a CMP MARK_AX row never
satisfies the `forward` gate and its operand band is left byte-identical — the
CMP nibble-comparator `-0.5/-0.8` per-nibble blocker magnitude contract (which
is calibrated to the dirty hybrid) is untouched → **zero CMP regression**.

Clean ADD/SUB/MUL/DIV/MOD operands keep the inter-byte carry correct, which is
where the +13 comes from (the arithmetic gain the full `C4_CLEAN_OPERAND`
feasibility flag also produced, but WITHOUT the −74 CMP break the full flag
incurs).

## The flip (three points, all consistent)

`C4_CLEAN_OPERAND_ADD` was flipped DEFAULT-OFF → DEFAULT-ON at all three env-read
sites (env-unset default `"0"` → `"1"`):

1. `neural_vm/unified_compiler/ops/shared.py` — `clean_operand_add_enabled()`.
2. `neural_vm/unified_compiler/full_vm_compiler_dynamic.py` — memo cache-key snapshot (#1).
3. `neural_vm/unified_compiler/full_vm_compiler_dynamic.py` — serialised cache-key snapshot (#2).

`C4_CLEAN_OPERAND` (the full-consumer feasibility flag) stays DEFAULT-OFF and is
registered in BOTH snapshots (it was MISSING before — a latent cache-collision
bug for a MODULE-affecting flag with no memo key; the branch added it, this land
keeps it). The four cache-key tuples are distinct:

| config | `C4_CLEAN_OPERAND` key | `C4_CLEAN_OPERAND_ADD` key | wrap installed |
|---|---|---|---|
| DEFAULT (both unset) | False | **True** | `CleanOperandOneHotFFN`, **5** arith dims |
| `C4_CLEAN_OPERAND_ADD=0` (escape hatch) | False | False | none (bare `LoadedOperandAddHi15ClearFFN`) |
| `C4_CLEAN_OPERAND=1` | True | True | `CleanOperandOneHotFFN`, 11 dims |
| `C4_CLEAN_OPERAND=1 C4_CLEAN_OPERAND_ADD=0` | True | False | `CleanOperandOneHotFFN`, 11 dims |

Install verified on the production groundtruth model path (campaign default):
block 12 ffn = `CleanOperandOneHotFFN` with `op_dims` len **5** =
`[OP_ADD, OP_SUB, OP_MUL, OP_MOD, OP_DIV]`; escape hatch reverts block 12 to the
bare `LoadedOperandAddHi15ClearFFN` corrector.

## Verification (all isolated caches)

### Golden hash — H_new = `e50521f3` (byte-identical, state_dict-NEUTRAL)

```
DEFAULT (flag now ON):                state_dict_sha256 = e50521f32b0ed952d5730f79b63adb8c4c78f4d4f0466d3bcbaa354bb3c90e86
Escape hatch (C4_CLEAN_OPERAND_ADD=0): state_dict_sha256 = e50521f32b0ed952d5730f79b63adb8c4c78f4d4f0466d3bcbaa354bb3c90e86
```

**H_new == `e50521f3` == the prior golden.** The flip is **state_dict-NEUTRAL**:
`CleanOperandOneHotFFN` adds NO parameters (it wraps `inner` and does the
one-hot snap in `forward()` using only existing dims), AND `tools/_isa_golden_hash.py`
builds in `alu_mode="lookup"` where the wrap is never even scheduled (it lives
in the `alu_mode="efficient"` campaign build only). So the byte-identity gate is
unchanged either way, and the escape hatch trivially reproduces `e50521f3`. The
behavioral change is a forward-time wrap in the efficient-ALU campaign path,
which is exactly what the 1096/smoke runs exercise.

### pytest tests/test_smoke.py — 51/51 PASS

```
================= 51 passed, 1 deselected in 155.48s (0:02:35) =================
```

DEFAULT build (flag-ON), `C4_TEST_SPEC_K=0 C4_SMOKE_SPEC_K=0`, GPU, isolated
cache. (Note: an earlier CPU run under 2-GPU contention was starved — the memory
note "smoke false-fails under GPU contention" applies; the authoritative result
is the uncontended GPU run above.)

### full-1096 `run_1096_canonical --criterion full_trace --spec-k 0 --max-steps-cap 40` — 580 → 593 (+13)

Same config both states, apples-to-apples, isolated caches, separate GPUs:

```
OFF (C4_CLEAN_OPERAND_ADD=0):  CANONICAL SCORE [full_trace]: 580/1096 PASS (52.92%)  [fail=283 error=0 skipped(>cap)=233]  wall=5219s
ON  (DEFAULT, flag-ON):        CANONICAL SCORE [full_trace]: 593/1096 PASS (54.11%)  [fail=270 error=0 skipped(>cap)=233]  wall=5074s

NET: +13  (580 -> 593)
```

OFF reproduces the mission baseline `580` exactly; ON = **593 (+13)**. The 233
deep loop/gcd/rec programs are `skipped(>cap)` in BOTH (not counted), so the
delta is a clean +13 on the runnable corpus. Independently reproduced (this land
re-ran both states from scratch; matches the branch's build-agent result).

### faithful-interpreter opaque — still 0

```
BLOCK FFN COVERAGE
  n_ir_executable = 59
  opaque_skipped  = []
  -> ZERO opaque_skipped: 59/59 blocks IR-executable.
```

The wrap is not installed off the campaign (bare-env / lookup build), so the
coverage validator is unaffected (still 0 opaque).

## Merged to main

* Work branch `land-clean-operand-add` = main + `clean-operand-add` (clean
  fast-forwardable merge, no conflicts) + the DEFAULT-ON flip.
* Merged to `main`. Final main default golden = **`e50521f3`** (unchanged,
  state_dict-neutral); escape hatch `C4_CLEAN_OPERAND_ADD=0` = `e50521f3`;
  smoke 51/51; full-1096 593; opaque 0.
* `docs/FLAG_REGISTRY.md` + `CLAUDE.md` golden line updated.

## Reproduce

```
# install / gate (production groundtruth path):
C4_TEST_SPEC_K=0 C4_NO_STACK0_EMIT=1 python -c \
 "from tools.probe_groundtruth import build_groundtruth_probe as b; m=b().model; \
  [print(i, type(x.ffn).__name__, len(x.ffn.op_dims)) for i,x in enumerate(m.blocks) \
   if getattr(x.ffn,'_is_clean_operand_wrap',False)]"   # -> 12 CleanOperandOneHotFFN 5

# golden byte-identity (must print e50521f3...):
CUDA_VISIBLE_DEVICES="" python tools/_isa_golden_hash.py                       # DEFAULT
CUDA_VISIBLE_DEVICES="" C4_CLEAN_OPERAND_ADD=0 python tools/_isa_golden_hash.py # escape hatch

# smoke:
C4_TEST_SPEC_K=0 C4_SMOKE_SPEC_K=0 python -m pytest tests/test_smoke.py -q

# the corpus gate (both states):
python tools/run_1096_canonical.py --criterion full_trace --spec-k 0 --max-steps-cap 40                        # 593
C4_CLEAN_OPERAND_ADD=0 python tools/run_1096_canonical.py --criterion full_trace --spec-k 0 --max-steps-cap 40 # 580
```

## Files

* `neural_vm/efficient_alu_neural.py` — `CleanOperandOneHotFFN` (param-free wrap).
* `neural_vm/unified_compiler/ops/shared.py` — `clean_operand_add_enabled()` (DEFAULT-ON), `clean_operand_enabled()` (DEFAULT-OFF).
* `neural_vm/unified_compiler/ops/alu_ops.py` — `make_clean_operand_op` (install gate, `op_dims` = 5 arith or 11 full).
* `neural_vm/unified_compiler/full_vm_compiler_dynamic.py` — both cache-key snapshots (both flags, `C4_CLEAN_OPERAND_ADD` default `"1"`, `C4_CLEAN_OPERAND` default `"0"`).
* `neural_vm/unified_compiler/_legacy_redirect.py` — `make_clean_operand_op` export.
