# LAND — first corrector removal: `LoadedOperandAddHi15ClearFFN` delete

**Date:** 2026-07-13
**Landed on:** `main` (merge commit `a3fad336`, merging branch
`corrector-removal-roadmap`, off `main` `38a97600`)
**Golden flag-OFF:** `e50521f3` — **UNCHANGED** (byte-identical, verified below)
**Full-1096 baseline:** 593/1096 (unchanged — the delete is numerically inert)

## What landed

The **first −LOC corrector removal** off the CORRECTOR_REMOVAL_ROADMAP
(`docs/CORRECTOR_REMOVAL_ROADMAP_2026_07_10.md`, item #7 / Class A "free
delete"). Deleted the bespoke corrector `LoadedOperandAddHi15ClearFFN` and
all of its live install/registration/export/interpreter glue:

- **`neural_vm/efficient_alu_neural.py`** — the `LoadedOperandAddHi15ClearFFN`
  `nn.Module` class (−127 LOC).
- **`neural_vm/unified_compiler/ops/alu_ops.py`** — the
  `make_loaded_operand_add_hi15_clear_op` factory + its registration/export
  (−92 net; −101 raw, replaced by a short provenance comment).
- **`neural_vm/unified_compiler/full_vm_compiler_dynamic.py`** — the install
  call site (net +1 comment).
- **`neural_vm/unified_compiler/_legacy_redirect.py`** — dropped export (−1).
- **`neural_vm/verification/faithful_interpreter.py`** — dropped the
  corrector's interpreter mirror reference (net +3 comment).

Kept additively: `docs/CORRECTOR_REMOVAL_ROADMAP_2026_07_10.md` (the inventory
+ A/B/C classification driving this and the follow-on missions) and the roadmap
probes.

## Why it's a free / byte-identical delete

The corrector cleared the SP/BP-frame address-nibble leak (ALU_HI cell 13/15,
+5.5) and the operand-B hi-nibble bleed (+1.0) on the loaded-operand **ADD**
MARK_AX row. Since the landed **`C4_CLEAN_OPERAND_ADD`** flip (default-ON), the
L8 main FFN (physical block 12) is wrapped by `CleanOperandOneHotFFN`, whose
outer clean-snap **already zeros every non-argmax ALU_HI cell on the ADD row
before this inner corrector would run** → the corrector is provably inert
(forward `max_abs_diff = 0.0` on the ADD rows). Removing it changes no weights
and no forward output.

`C4_CLEAN_OPERAND_ADD` is itself a param-free forward-wrap that only fires in
the efficient-ALU campaign build, so the bare-env `state_dict` golden was
already `e50521f3` with the corrector present; deleting the corrector leaves it
`e50521f3`.

## Verification (this land)

- **Golden byte-identity** (`tools/_isa_golden_hash.py`, `disk_cache=False`,
  CPU / `CUDA_VISIBLE_DEVICES=""`):
  `state_dict_sha256 = e50521f32b0ed952d5730f79b63adb8c4c78f4d4f0466d3bcbaa354bb3c90e86`
  = **e50521f3**. MATCHES the golden. The delete is byte-identical.
- **Build/import**: `compile_full_vm_dynamic(disk_cache=False)` builds clean
  on CPU (exit 0, TOTAL FFN units 42149 → 42149, 100% retained). Import +
  bare compile confirmed. (GPU smoke NOT re-run — GPUs saturated; smoke is
  implied 51/51 since the weights are byte-identical to the 51/51 golden.)
- **Pre-land evidence** (branch): fast-gate `--base main` net = 0 (74 ids,
  53p/21f both states, 0 verdict diffs); smoke 51/51.

## LOC delta

`git diff --shortstat 38a97600 a3fad336`:

- Code only (excl. docs): **5 files, +26 / −242** (net **−216 LOC**).
- The two corrector-carrying files (`efficient_alu_neural.py` +
  `alu_ops.py`): +9 / −228 = net **−219 LOC**; the corrector proper is
  **−224** (class −127 + op glue −97, per the roadmap), the rest being small
  net-additive provenance comments in the compiler + interpreter.
- Full merge incl. docs: 6 files, +301 / −242.

## ops-core LOC now

`neural_vm/unified_compiler/ops/*.py` = **66,206 LOC** (post-land).

This is the **first −LOC corrector removal landed**. Next on the roadmap: the
Class-B derive-root missions (CMP −262, mul −218, bitwise −158, shift −77)
that each unlock a further Class-A/B corrector delete.
