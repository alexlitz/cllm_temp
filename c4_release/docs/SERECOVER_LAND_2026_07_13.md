# SERECOVER_LAND — reconstruct the Cmp+Mul SeRecover delete (−359) + func_min LT-fix onto current main

**Branch** `serecover-land` (off current `main` @ `5e322ae5`, campaign default) ·
**Flag** `C4_ALU_OPERAND_SURVIVE` (**DEFAULT-ON**, opt out `=0`) +
`C4_ALU_OPERAND_SURVIVE_CMP_RAW` (**DEFAULT-OFF**, the func_min LT-fix) ·
**Reconstructs** `serecover-delete` HEAD `80573338` (based on `aead086e`,
12 commits behind main) onto current main ·
**Deleted** `CmpOperandSeRecoverFFN` (−141) + `MulOperandSeRecoverFFN` (−218) =
**−359 LOC** from `neural_vm/efficient_alu_neural.py` ·
**Golden flag-OFF** `9dda2af1` — UNCHANGED (byte-identical to current main
default) · **Default-ON golden** `1c04c3fd`.

## TL;DR

This is a pure **reconstruction / rebase** of the completed, fast-gated
(+19 net / 0-regression on the operator's private gate) SeRecover-delete +
func_min LT-fix work onto current main. No new design work — the four parts
(block-15 L9-clear operand spare + func_min LT-fix, block-17 head-4 CMP Q-veto,
the −359 class deletion + 3 attach-site collapses, and DEFAULT-ON flag
registration) are exactly the branch `serecover-delete` deliverable described in
`docs/SERECOVER_DELETE_2026_07_13.md` + `docs/FUNCMIN_FIX_2026_07_13.md`,
re-applied onto `5e322ae5`.

## 1. What was reconstructed (the 8 branch commits)

Cherry-picked in order onto `main` @ `5e322ae5`:

| commit | subject |
|---|---|
| `16773986` | Delete Cmp+Mul OperandSeRecoverFFN via derived root (−359 LOC) |
| `cb4c6873` | interp gate treats CleanOperandOneHotFFN as composite |
| `1f982765` | Scope L9 raw-band operand spare to arith/bitwise only |
| `665cf107` | FUNCMIN_FIX deliverable + probe spec_k=0 |
| `480e2e01` | register C4_ALU_OPERAND_SURVIVE_CMP_RAW in FLAG_REGISTRY |
| `08183c9b` | fix(func_min): exclude ONLY OP_LT from raw-band spare |
| `2e18196e` | correct FUNCMIN_FIX to LT-only exclusion |
| `80573338` | FUNCMIN_FIX §4 — LT-only fix CONFIRMED 5/5 PASS |

Plus one reconciliation commit `79ae6254` (below).

## 2. The three reconciliations against current main

Current main advanced 12 commits past the branch's fork-point `aead086e`. Three
of those changes overlapped the SeRecover diff and were reconciled by hand (NOT
blind-copied):

1. **`efficient_alu_neural.py` — BitwiseSeRecover + ShiftOutputClear already
   deleted by main.** The cherry-pick of `16773986` auto-merged cleanly here:
   main had already removed `BitwiseOperandSeRecoverFFN` (commit `785baf8f`) and
   `ShiftOutputClearFFN`, and the incoming commit removes `CmpOperandSeRecoverFFN`
   + `MulOperandSeRecoverFFN`. Net: all four gone, no conflict in the class body.

2. **`faithful_interpreter.py` composite-name allowlist — CONFLICT resolved.**
   Main's `COMPOSITE_ALU_FFN` tuple had already dropped `ShiftOutputClearFFN`
   (kept `Cmp`/`Mul`SeRecover); the incoming commit drops `Cmp`/`Mul` (didn't
   know about the Shift drop). Resolution: keep main's ShiftOutputClear removal
   AND apply the Cmp/Mul removal — all three gone, retaining main's explanatory
   comment for the Shift removal. (Same one-line resolution in
   `tools/faithful_interpreter_validate.py`, which auto-merged.)

3. **`full_vm_compiler_dynamic.py` — dual→single cache-key snapshot.** The
   lowering-cut (`c425c182`) merged main's former DUAL inline snapshots
   (`compile_full_vm_dynamic` ~2744 + `_bake_from_scheduled_ops` ~3778) into ONE
   `_build_cache_key_snapshot`. The branch commits registered `C4_ALU_OPERAND_SURVIVE`
   (`16773986`) + `C4_ALU_OPERAND_SURVIVE_CMP_RAW` (`1f982765`) TWICE (once per
   old snapshot). Resolution: the incoming compiler hunks (which target the
   deleted dual structure) were dropped during cherry-pick; both flags are
   registered ONCE in the merged `_build_cache_key_snapshot` (commit `79ae6254`),
   next to `C4_MUL_B1_DELIVERY`, with the correct default values
   (`C4_ALU_OPERAND_SURVIVE` default `"1"`; `C4_ALU_OPERAND_SURVIVE_CMP_RAW`
   default `"0"`). Compiler diff vs main = `+22` (was `+43` in the old dual form).

The l9_ops / model_ops / all_core_ops / alu_ops / l11_ops changes (the actual
fix + deletion + attach-site collapses) applied with no reconciliation.

## 3. The −359 delete confirmed

* `neural_vm/efficient_alu_neural.py` — `class CmpOperandSeRecoverFFN` (−141) and
  `class MulOperandSeRecoverFFN` (−218) **DELETED** (net `-359` in the diff, one
  line shift): `git grep -c "class CmpOperandSeRecoverFFN|class MulOperandSeRecoverFFN"
  = 0`.
* `alu_ops.py` ×2 (`make_efficient_l10_andorxor_wrap_op` CMP-recover attach →
  bare `cleanup_ffn`; `make_efficient_l11_alumul_wrap_op` MUL-recover attach →
  bare `new_ffn`) + `l11_ops.py` ×1 (multipass-MUL attach → bare `mp_block`):
  the three SeRecover attach sites collapsed to the inner FFN. No live
  instantiation of either class remains (only docstring/comment references).
* Composite-name allowlists (`faithful_interpreter.py`,
  `tools/faithful_interpreter_validate.py`): the two class-name strings removed.

## 4. Golden proofs (this branch, current main)

* **Flag-OFF (`C4_ALU_OPERAND_SURVIVE=0`), wrappers deleted** →
  `state_dict_sha256 = 9dda2af1f81e7c96...` = **byte-identical to current main
  default `9dda2af1`** (`tools/_isa_golden_hash.py`, `disk_cache=False`,
  CPU/CUDA-hidden, campaign default). The zero-param wrapper deletes + the
  fix-OFF path change nothing. ✅ PROOF that the delete is byte-neutral.
* **Default-ON (`C4_ALU_OPERAND_SURVIVE=1`, the shipped default)** →
  `state_dict_sha256 = 1c04c3fd58f4c814...` — a NEW golden, distinct from
  flag-OFF, confirming the flag is genuinely weight-affecting (the block-15
  operand-spare lands in the lookup build as extra `W_up` NOT-blocker columns).

> Note: the branch's original docs quote flag-OFF golden `e50521f3` and
> default-ON `ee6d0867`. Those were the values at the **fork-point** `aead086e`,
> BEFORE main flipped `C4_R_FRAME_TAIL` + `C4_MUL_B1_DELIVERY` (which moved the
> default golden to `9dda2af1`). On current main the flag-OFF golden is
> `9dda2af1` (= main default; the delete is byte-neutral) and default-ON is
> `1c04c3fd`. The func_min LT-fix + the −359 delete are orthogonal to the two
> main-ahead flips, so the reconstruction was conflict-light exactly as the
> FUNCMIN_FIX §5 rebase note predicted.

## 5. Behaviour (unchanged from the gated branch)

`_CMP_OPCODES_SPARED = (OP_EQ, OP_NE, OP_GT, OP_LE, OP_GE)` — **OP_LT EXCLUDED**
by default (the func_min id675 fix, `l9_ops._spare_opcodes()`); the opt-in
`C4_ALU_OPERAND_SURVIVE_CMP_RAW=1` restores all-CMP-in (LT back → func_min
regresses). Import-verified:
`l9._spare_opcodes()` (default) =
`('OP_EQ','OP_NE','OP_GT','OP_LE','OP_GE','OP_MUL','OP_DIV','OP_MOD','OP_OR','OP_XOR','OP_AND')`.

Expected fast-gate outcome (per the operator's prior +19/0-reg run on the
`80573338` branch, now reconstructed identically): func_min id675 PASS + the 4
gains (`bool_and` id1088, `func_mul` id612, `if_eq` id408, `if_var` id433) hold,
0 regressions (flag-OFF byte-identical `9dda2af1` → no passing program can move
under OFF).

## 6. Gate — DEFERRED to the operator (GPU)

flag-OFF byte-identity (`9dda2af1`) + default-ON golden (`1c04c3fd`) captured on
CPU here. The authoritative A/B fast gate (expect **+19 / 0-regression**) is
deferred to a free GPU / main:

```
cd c4_release
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=$(pwd) \
C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
python tools/fast_gate.py --flag C4_ALU_OPERAND_SURVIVE
```

CPU re-verify of the 5 ids (spec_k=0 full_trace, the fast-gate criterion):

```
C4_ALU_OPERAND_SURVIVE=1 python tools/_probe_funcmin_verdict.py \
  --ids 675,612,408,433,1088
```

## 7. Files (this branch vs main)

* `neural_vm/efficient_alu_neural.py` — the two SeRecover classes DELETED (−359).
* `neural_vm/unified_compiler/ops/l9_ops.py` — `_SPARE_OPCODES_ARITH`,
  `_CMP_OPCODES_SPARED` (LT excluded), `_spare_cmp_raw_enabled`,
  `_spare_opcodes`, `_alu_operand_survive_enabled`, block-15 clear spare.
* `neural_vm/unified_compiler/ops/model_ops.py` — `make_cmp_h4_qveto_op` (phase 1450).
* `neural_vm/unified_compiler/ops/all_core_ops.py` — register the veto op.
* `neural_vm/unified_compiler/ops/alu_ops.py`, `.../ops/l11_ops.py` — 3 attach
  sites collapsed to the bare inner FFN.
* `neural_vm/unified_compiler/full_vm_compiler_dynamic.py` — both flags in the
  SINGLE merged `_build_cache_key_snapshot` (reconciliation).
* `neural_vm/verification/faithful_interpreter.py`,
  `tools/faithful_interpreter_validate.py` — composite-name allowlists trimmed.
* `docs/FLAG_REGISTRY.md` — the two survival-flag entries + the SE_RECOVER
  helper flags marked inert.
* Probes (tooling-only): `tools/_probe_serecover_inert.py`,
  `tools/_probe_funcmin_verdict.py`, `tools/_probe_funcmin_leak.py`,
  `tools/_probe_h4qveto_flag.py`, `tools/_probe_builtclear.py`,
  `tools/_probe_spare_mul_operand.py`, `tools/_probe_spare_deletability.py`.
