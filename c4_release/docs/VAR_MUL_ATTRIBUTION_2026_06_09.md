# var_mul Cluster Attribution — 2026-06-09

## Scope

`tests/test_suite_1000.py` generator (lines 98-106) emits 25 programs
of the form:

```c
int main() { int a; int b; a = N1; b = N2; return a * b; }
```

All 25 fail in `test_suite_1096_pure_neural_pytest.py` under
`-k var_mul --runxfail`. Every failure manifests identically:
`neural_exit == 1` against the expected product (declarative correct,
17 steps).

## First-mismatch shape (verifier)

`tools/attribute_1096_failure.py --test_id var_mul_5` (a=17, b=3,
expected=51) — and spot-checks against `var_mul_3`, `var_mul_10`,
`var_mul_21` — all yield the same first divergence:

| Field              | Value                                |
|--------------------|--------------------------------------|
| step               | 0                                    |
| slot               | `MEM_value1`                         |
| expected_token     | 0x00                                 |
| neural_token       | 0xff                                 |
| suspect_dims       | `OUTPUT_LO`, `OUTPUT_HI`             |
| first_mismatch_op  | `layer3_carry_forward_attn`          |

The flagged L3 op is a **known dynamic-verifier false positive at
step 0** (no prev step → residual is correctly zero, verifier reads
it as "claimed but silent"). Documented in
`docs/VAR_L3_SP_BYTE2_2026_06_07.md` §"First-mismatch op (verifier)".

## Bytecode confirms shared root with var_*/if_var_*

`src/compiler.py` lowering of the var_mul source emits, as word 0:

```
JSR imm=3   (raw word 771 = (3<<8)|3)
```

i.e. step 0 is the **function-prologue JSR** that pushes return-PC
`0x0a` to `SP=0xFFFC`. Symbolic MEM_value bytes for that push are
`(0x0a, 0x00, 0x00, 0x00)`; neural emits 0xff at byte 1. This is
**bit-identical** to the divergence shape for `var_simple_0` and
`if_var_0` recorded by `tools/attribute_1096_failure.py` on 2026-06-07
(see `docs/VAR_L3_SP_BYTE2_2026_06_07.md`).

## True writer — same head 5 of `layer14_mem_generation`

Per the 06-07 walk of `neural_vm/unified_compiler/ops/l14_ops.py:584-628`
(MEM val byte heads 4-7, `h=1` = MEM_value byte-1 generator at
head_idx 5):

- Slot 2 (STACK0-source K-positions) at `l14_ops.py:616-618`:
  ```python
  elif h == 1:
      k.append(AP(2, BD.H2   + BP_I,  L))
      k.append(AP(2, BD.L1H4 + BP_I, -L))
  ```
  These are **BP-frame byte-1 selectors** (positional flags emitted
  by ENT). For JSR step 0 there is no prior BP frame, so the
  STACK0-byte-1 K-target collapses and softmax degenerates to a
  default position whose CLEAN_EMBED carries 0xff token bits.
- Head 5 V-then-O byte copy (lines 561-573) writes the softmax-
  weighted CLEAN_EMBED nibbles into `OUTPUT_LO[15]/OUTPUT_HI[15]` →
  argmax-decoded token = 0xff at MEM_value1.

## Attribution verdict

**Shared root with the `var_*` / `if_var_*` L14 JSR step-0
MEM_value1 attribution cluster.** Not a collapsed-step downstream
issue — the *very first* JSR token decode is wrong, all subsequent
17 steps inherit garbage. The pattern is identical across:

- `var_simple_*` (single ENT main, single JSR)
- `if_var_*` (single JSR + branch)
- `var_three_*` (single JSR + 3 locals)
- **`var_mul_*` (single JSR + 2 locals + MUL)** ← this attribution

The MUL opcode in the body is **not** the failure point; control
never reaches it cleanly because the return-PC push at step 0 is
corrupted.

## Why mul_* (non-var) does NOT collapse the same way

Spot check `mul_5` (`14 * 81`) — also fails, but at a **different**
divergence:

| Test         | step | slot      | expected | neural   |
|--------------|------|-----------|----------|----------|
| var_mul_5    |    0 | MEM_value1 |     0x00 |     0xff |
| mul_5        |    1 | SP_byte3   |     0x00 |  REG_BP  |

Plain `mul_` (literal-only `return A*B`) goes via the same JSR step-0
but **declaratively only emits 5 steps** (no locals → no ENT-frame
needs), so the residual decode pattern differs and the MEM_value1
0xff is masked by a competing winner on step 0. The `mul_` failure
is a **different** cluster (SP-byte3 step 1, OUTPUT/SP_CARRY suspect),
not the var_mul root.

## Fix recommendation

Same as `docs/VAR_L3_SP_BYTE2_2026_06_07.md` §Recommendation. The
single-rule whack-a-mole has netted 0/5 historical attempts
(memory note `feedback_single_rule_fixes_are_zero_sum`). The right
fix is upstream in `layer14_mem_generation` head 5:

1. Audit head-5 slot-2 K-positions on the JSR-only path —
   `STACK0_BYTE1` (PC-byte-1) vs `H2+BP_I` (BP-frame byte-1).
   Cross-check the legacy `_set_layer14_mem_generation` in
   `neural_vm/vm_step.py` for whether the migration collapsed the
   JSR/ENT distinction at this head.
2. If head-5 K is unified, audit upstream `STACK0_BYTE1` population
   at L7/L9 for JSR step 0 (no prior frame).

No DSL change shipped — attribution only. Monotone non-decreasing
maintained vacuously.

## Refs

- `docs/VAR_L3_SP_BYTE2_2026_06_07.md` — `var_simple_0` / `if_var_0`
  identical-shape attribution (2026-06-07).
- `docs/VAR_CLUSTER_JSR_PATH_FINDINGS_2026_06_06.md` — earlier
  MEM_addr1 → MEM_addr2 shift.
- `docs/VAR_REAL_ATTRIBUTION_2026_06_05.md` — earlier var cluster.
- `tools/attribute_1096_failure.py` — automated brief generator.
- `neural_vm/unified_compiler/ops/l14_ops.py:584-628` — MEM val
  byte heads 4-7 (head 5 = MEM_value byte-1 prime suspect).
- `tests/test_suite_1000.py:98-106` — var_mul test generator.
- `src/compiler.py:27-60` — opcode enum (JSR=3) for bytecode decode.
- Memory note `feedback_single_rule_fixes_are_zero_sum` — 0/5 net
  positive on single-rule patches in this cluster.
