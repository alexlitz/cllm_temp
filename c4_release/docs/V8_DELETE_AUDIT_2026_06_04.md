# V8 efficient_alu_* delete audit (2026-06-04)

Branch: `dsl-v8-cleanup` (worktree `/tmp/c4-dsl-v8-cleanup`).
Author: read-only audit + partial-deletion agent.
Goal: delete the 6 `efficient_alu_*.py` files per Wave V8 brief.

## TL;DR

Full deletion is **not safe in this wave**. Partial deletion of 3 dead
files (1,421 LoC) + 1 dead-imports cleanup in `vm_step.py` is safe. The
remaining 3 files (2,360 LoC) are load-bearing for the production
lookup-mode runtime and need a follow-up wave.

| File | LoC | Status | Reason |
|---|---:|---|---|
| `efficient_alu_8bit.py` | 516 | **DELETE** | Zero callers in `neural_vm/` or `tests/`. Only `test_archive/` and `docs/archive/` mention it. |
| `efficient_alu_byte.py` | 248 | **DELETE** | Zero callers. Only `docs/archive/EFFICIENT_BYTE_ALU.md`. |
| `efficient_alu_integrated.py` | 657 | **DELETE** | Zero callers. Only `test_archive/test_{all_efficient_ops,comprehensive_efficient_alu,shift_only}.py` and `docs/archive/OPTION_D_*.md`. |
| `efficient_alu_neural.py` | 1554 | **KEEP** | Load-bearing: `_make_alu_postop_attach_op` (shared.py:206-214) instantiates `eau.FlattenedALUMul.build_fully_baked`, `eau.ALUShiftComposite`, `eau.ALUAndOrXor`, etc. for production lookup-mode L8-L13. `_ensure_l11_mul_module` instantiates `FlattenedALUMul` for the 9 L11/L12 MUL flatten ops. `_ALUShiftCompositeBuilder` instantiates `ALUShiftComposite` for L13. `alu_ops.py` imports `ShiftBDToGEStage`/`ShiftPrecomputeStage`/`ShiftSelectStage`/`ShiftGEToBDStage` for the L13 composite. |
| `efficient_alu_addsub_split.py` | 288 | **KEEP** | Load-bearing: `_make_alu_postop_attach_op` and `efficient_l8_addsub_wrap_op` install `AddSub5StageBlock` into L8/L9 `post_ops` in production lookup mode. Self-references `BDToGEConverter`/`GEToBDConverter` from `efficient_alu_neural.py`. |
| `efficient_alu_divmod_split.py` | 518 | **KEEP** | Load-bearing: `_FlattenedDivModBuilder.ensure` instantiates `FlattenedDivMod` in `shared.py` for the production L10 DIV/MOD install op (lookup mode). Self-references `BDToGEConverter`/`GEToBDConverter`. |

**Deletable total: 3 files / 1,421 LoC.**

## Why the 3 keep-files cannot be deleted in V8

The `wide_alu_dsl` rule-derived path is `alu_mode='efficient'` only.
Production smoke runs `alu_mode='lookup'` (the default at every
`make_lN_alu_postop_attach_op(alu_mode='lookup')` factory in
`alu_ops.py`). In lookup mode:

- L8/L9 install `AddSub5StageBlock(S, BD)` into `post_ops[0]`.
- L10 installs `eau.ALUAndOrXor(S, proxy)` into `post_ops[0]` AND
  `FlattenedDivMod(S, BD)` into `post_ops` via the divmod install op.
- L11/L12 install `FlattenedALUMul.build_fully_baked(S, proxy)` into
  `post_ops[0]`. (L11's postop is dead per the L17 tail MUL fix; L12
  remains the canonical attach.)
- L13 installs `eau.ALUShiftComposite(S, proxy)` into `post_ops[0]` AND
  builds the 5-stage `ALUShiftComposite` via `_ALUShiftCompositeBuilder`
  for `block.ffn`.

Each of these composite classes wraps multiple sub-FFN stages
(`BDToGEConverter`/`_BDToGEStage` → schoolbook/longdiv/precompute →
`GEToBDConverter`/`_GEToBDStage`). The forward math is non-trivial
and not yet re-expressed as `FFNRule` lists for the `alu_mode='lookup'`
path. Re-expressing all of them as rules and then deleting the
imperative composites is the proper V8 follow-up.

## What this PR delivers

1. **docs(v8): V8 audit doc** — this file.
2. **dsl-v8: drop unused efficient_alu_* imports in vm_step.py** —
   `ALUAndOrXor`/`ALUMul`/`ALUDivMod`/`AddSub5StageBlock` are only
   referenced in comments in `vm_step.py`. Drop the imports.
3. **dsl-v8: delete `efficient_alu_8bit.py`** — zero callers.
4. **dsl-v8: delete `efficient_alu_byte.py`** — zero callers.
5. **dsl-v8: delete `efficient_alu_integrated.py`** — zero callers.

## Smoke baseline

Per `docs/STATUS_1096_2026_06_04.md`: **29 / 52 pass** at HEAD
`cef9d046`. V8 target is `>= 29 / 52` retained.

## Follow-up wave (V9?)

To complete the V8 deletion of the remaining 3 files, the lookup-mode
ALU path needs to be rule-lowered. Specific work items:

- L8/L9: rewrite `AddSub5StageBlock` as `wide_add_rules` / `wide_sub_rules`
  for the lookup-mode default (currently only the `alu_mode='efficient'`
  wrap uses these rules).
- L10 bitwise: `ALUAndOrXor` → `bitwise_rules` (already done for
  efficient mode in `efficient_l10_andorxor_wrap_op`; promote to lookup).
- L10 divmod: `FlattenedDivMod` → `wide_div_rules` for cross-nibble cases
  (current W4 POC is per-nibble only, NOT byte-identical for multi-byte
  inputs — see `docs/DSL_W5_MULDIV_LIMIT.md`).
- L11/L12 mul: `FlattenedALUMul` → `wide_mul_rules` for multi-byte
  (current W5 POC is `width_bytes=1` only).
- L13 shift: `ALUShiftComposite` → `wide_shift_rules` (W2 implemented;
  needs the lookup-mode install rewired through it).

Each is a separate small wave with its own byte-identity validation
against the current composite.
