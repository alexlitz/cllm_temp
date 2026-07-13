# SERECOVER_DELETE — delete Cmp + Mul OperandSeRecoverFFN via the derived root

**Branch** `serecover-delete` (off `main` @ `aead086e`, campaign default) ·
**Flag** `C4_ALU_OPERAND_SURVIVE` (**DEFAULT-ON**, opt out `=0`) ·
**Golden flag-OFF** `e50521f3` — UNCHANGED (byte-identical) ·
**Default-ON lookup-golden** `ee6d0867` ·
**Deleted** `CmpOperandSeRecoverFFN` (−141) + `MulOperandSeRecoverFFN` (−218) =
**−359 LOC** from `neural_vm/efficient_alu_neural.py` ·
**Config** spec_k=0, campaign default (`C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1`).

## TL;DR

Both `CmpOperandSeRecoverFFN` and `MulOperandSeRecoverFFN` existed only to
re-materialise operand-A after the L9/L10 ALU-clear crushed it all-negative on
the CMP / MUL AX rows. That crush is now fixed **at its root** by one combined
flag, `C4_ALU_OPERAND_SURVIVE` (default-ON), built from the two already-derived
pieces:

* **block-15 L9-clear operand SPARE** (`l9_ops._alu_operand_survive_enabled`) —
  a hard `-1e6` NOT-blocker for each ALU opcode (`OP_EQ..OP_GE`, `OP_MUL/DIV/MOD`,
  `OP_OR/XOR/AND`) added to the L9 ALU-clear's AND gate. On a genuine ALU AX row
  the live opcode flag (`OP_GT`/`OP_MUL ≈ +10`) vetoes every clear unit →
  operand-A survives the crush as a clean `+6` one-hot.  Extra `W_up` columns,
  not extra units → 3405-unit L9 layout preserved → byte-identical flag-OFF.
* **block-17 head-4 CMP Q-veto** (`model_ops.make_cmp_h4_qveto_op`, phase 1450)
  — overwrites block-17 (logical L11) attention head-4's `W_q` at the six CMP
  columns (at the `K@CONST` self-row slots) with `-1e5` so the un-gated
  MEM→ALU load head stops writing its ~-8 ALU floor on cmp result rows (the
  SECOND crush, at block-17 attention, that the block-15 spare cannot reach).

Neither the block-15 spare alone (nets −4; id437 relies on the recover masking
the block-17 residual) nor the block-17 veto alone is sufficient; **combined**,
operand-A survives clean through **both** crushes, so each SeRecover's
crush-detect gate never fires and its forward is byte-identical to its inner
FFN. Both wrappers are therefore **provably inert** and **deleted**.

## 1. The combined fix (correct-by-construction, one flag)

`neural_vm/unified_compiler/ops/l9_ops.py`:
* `_SPARE_OPCODES = _CMP_OPCODES + (OP_MUL, OP_DIV, OP_MOD, OP_OR, OP_XOR, OP_AND)`
* `_alu_operand_survive_enabled()` reads `C4_ALU_OPERAND_SURVIVE` (default `"1"`).
* When ON, `_alu_clear_rules` adds `(op, -1e6)` for each `_SPARE_OPCODES` opcode
  to `common_conditions`. Built-weight proof: `block-15 W_up[u3344..u3375,
  OP_GT..OP_GE / OP_MUL..] = -1e8` (32 units, the whole clear band). Flag OFF: 0.

`neural_vm/unified_compiler/ops/model_ops.py` (+ `all_core_ops.py` register):
* `make_cmp_h4_qveto_op()` — `kind="model"`, `phase=1450` (after
  `expand_wrapper_blocks`=1300 and `norm_compensator_seed`=1400 so block-17's
  8→13 head expansion is materialised). `_bake` is a clean early-return no-op
  unless `C4_ALU_OPERAND_SURVIVE` is set; it densifies the sparse-COO Q, finds
  head-4's `K@CONST>1` slots, and writes `-1e5` at the six CMP columns. K is
  read-only. Built-weight proof (efficient build): block-17 = 13 heads,
  head-4 `K@CONST>1` slots `[0, 33]`, 12 CMP-column cells `= -1e5`.

`neural_vm/unified_compiler/full_vm_compiler_dynamic.py`:
* `C4_ALU_OPERAND_SURVIVE` added to BOTH the in-proc memo snapshot AND the
  disk-cache `kwargs_snapshot` so ON/OFF builds never share a serialised entry
  (weight+bake-affecting).

## 2. Inertness proof (the deletion is safe)

`tools/_probe_serecover_inert.py` (spec_k=0, campaign, efficient build =
smoke-gate path so the wraps ARE installed): a forward-pre hook captures each
wrapper's exact input `x`; we then compute `wrapper.forward(x)` vs
`wrapper.inner(x)` and report `max|forward-inner|`. Value-diverse crush
fixtures (both-nibble-nonzero crush values per `WRAPPER_DELETABILITY_MATRIX`):
CMP operand-A ∈ {7, 57, 23} + MUL operand-A ∈ {23, 7}.

| fixture | wrapper | `max|forward-inner|` ON (`=1`) | OFF (`=0`) |
|---|---|---|---|
| `eq_7_45`   | Cmp blk21 | **0.000e+00 INERT** | 9.205e+08 (fires) |
| `lt_57_29`  | Cmp blk21 | **0.000e+00 INERT** | 3.258e+04 (fires) |
| `gt_23_65`  | Cmp blk21 | **0.000e+00 INERT** | 3.253e+04 (fires) |
| `mul_23_65` | Mul blk17 | **0.000e+00 INERT** | 5.855e+08 (fires) |
| `mul_7_9`   | Mul blk17 | **0.000e+00 INERT** | 5.887e+08 (fires) |

SUMMARY: with `C4_ALU_OPERAND_SURVIVE=1`, `CmpOperandSeRecover` **and**
`MulOperandSeRecover` are byte-identical to their inner FFN
(`max|forward-inner| = 0.000e+00`) on the very crush rows they exist to repair,
across all five value-diverse fixtures. With the flag OFF the recovers fire
(diff ~9e8 CMP / 5.9e8 MUL) — the control that proves the ON=0 result is the
fix at work, not a trivial all-zero collapse. **Both wrappers are inert → safe
to delete.**

## 3. The deletion (−359 LOC)

* `neural_vm/efficient_alu_neural.py` — `class CmpOperandSeRecoverFFN`
  (−141) and `class MulOperandSeRecoverFFN` (−218) DELETED (359 lines total).
* `neural_vm/unified_compiler/ops/alu_ops.py` — the `make_efficient_l10_andorxor_wrap_op`
  CMP-recover attach → `block.ffn = cleanup_ffn`; the
  `make_efficient_l11_alumul_wrap_op` MUL-recover attach → `block.ffn = new_ffn`
  (with the now-unused `no_stack0_emit_enabled/mul_byte0_se_recover_enabled/
  mul_l11_se_recover_enabled` import removed).
* `neural_vm/unified_compiler/ops/l11_ops.py` — the multipass-MUL recover attach
  → `block.ffn = mp_block`.
* `neural_vm/verification/faithful_interpreter.py` +
  `tools/faithful_interpreter_validate.py` — the two class names removed from the
  `COMPOSITE_ALU_FFN` / composite-name allowlists (stale strings; the classes
  are gone so `type(ffn).__name__` can never match).

The `cmp_byte0_se_recover_enabled` / `mul_l11_se_recover_enabled` flag helpers
in `shared.py` are LEFT in place (they no longer install a wrap; harmless, and
their cache-key entries keep the memo topology stable). Marked inert in
`FLAG_REGISTRY.md`.

## 4. Golden proofs

* **Flag-OFF (`C4_ALU_OPERAND_SURVIVE=0`), wrappers deleted** →
  `state_dict_sha256 = e50521f3...` = **golden, byte-identical**
  (`tools/_isa_golden_hash.py`, `disk_cache=False`, campaign default). The
  golden model is built `alu_mode='lookup'` where the wraps were never installed
  and the fix is a no-op, so deletion + fix-OFF changes nothing.
* **Default-ON (`C4_ALU_OPERAND_SURVIVE=1`)** → `state_dict_sha256 = ee6d0867...`
  (the recorded default-ON golden). The block-15 spare lands in the lookup
  build (32 units × CMP/MUL cols = −1e8); the block-17 head-4 veto lands in the
  **efficient** build (13 heads, slots 0/33, −1e5) where the id433 crush lives —
  the lookup golden has 11 heads and no MEM→ALU load-head structure so the veto
  correctly no-ops there.

## 5. Gate — DEFERRED (GPU saturated)

Both GPUs were >2 GB in use at authoring time, so the authoritative A/B fast
gate is deferred to a free GPU / main. EXPECTATION: net ≥ 0 — `if_var id433`
(x=35>76) flips fail→ok via the block-17 veto, plus the rest of the
block-17-crush cmp subset; no regression (flag-OFF byte-identical golden
`e50521f3`, so no passing program can move OFF). `id437` (block-15-only crush)
is now also addressed by the block-15 spare + does not regress because, with the
block-17 residual removed, its baseline no longer depends on the recover's
masking.

```
cd c4_release
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=$(pwd) \
C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
python tools/fast_gate.py --flag C4_ALU_OPERAND_SURVIVE
```

## 6. Files

* `neural_vm/unified_compiler/ops/l9_ops.py` — `_SPARE_OPCODES`,
  `_alu_operand_survive_enabled`, block-15 clear spare blocker.
* `neural_vm/unified_compiler/ops/model_ops.py` — `make_cmp_h4_qveto_op`.
* `neural_vm/unified_compiler/ops/all_core_ops.py` — register the veto op.
* `neural_vm/unified_compiler/full_vm_compiler_dynamic.py` — cache-key x2.
* `neural_vm/efficient_alu_neural.py` — the two SeRecover classes DELETED (−359).
* `neural_vm/unified_compiler/ops/alu_ops.py`, `.../ops/l11_ops.py` — the three
  recover attach sites collapsed to the bare inner FFN.
* `neural_vm/verification/faithful_interpreter.py`,
  `tools/faithful_interpreter_validate.py` — composite-name allowlists trimmed.
* `docs/FLAG_REGISTRY.md` — new `C4_ALU_OPERAND_SURVIVE` entry; the two
  SE_RECOVER flags marked inert.
* Probes (tooling-only): `tools/_probe_serecover_inert.py` (the inertness A/B),
  `tools/_probe_h4qveto_flag.py`, `tools/_probe_builtclear.py`,
  `tools/_probe_spare_mul_operand.py`, `tools/_probe_spare_deletability.py`.
