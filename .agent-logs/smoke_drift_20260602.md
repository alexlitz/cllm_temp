# Smoke drift triage 2026-06-02

Follow-up to `.agent-logs/regression_scan_20260602_0733.md` §6 (the
`runner_strict_compile_blocks_smoke` blocker) and commit
`9d215dbd run_vm: pass strict=False to compile_full_vm_dynamic`. The
strict-mode fix unblocked smoke setup. This doc characterises the **39
test failures** that surface once setup proceeds.

## Observed counts

- `pytest c4_release/tests/test_smoke.py -q` at `speedup-cache-and-buckets`
  HEAD (`50b9e91a`): **12 passed, 39 failed, 1 deselected, 102.8s**.
- All 39 failures share the same shape: `assert 0 == <expected>`.
  The model returns `exit_code = 0` instead of the program's intended
  AX value.

## Failure groups (all 39)

Every failure is the same assertion family: `_eq(expected)` raises
`assert 0 == <expected>`. Groups by test class:

| Class | failed | tests |
|---|---:|---|
| `TestSmokeBasic` | 6/7 | `test_imm_exit`, `test_add_basic`, `test_sub_basic`, `test_mul_basic`, `test_div_basic`, `test_mod_basic` |
| `TestSmokeControlFlow` | 3/? | `test_jmp_forward`, `test_bz_branch`, `test_bnz_branch` |
| `TestSmokeFunctionCall` | 1/? | `test_simple_function` |
| `TestSmokeBitwise` | 3/3 | `test_or_basic`, `test_and_basic`, `test_xor_basic` |
| `TestSmokeComparison` | 6/? | `test_eq_true`, `test_lt_true`, `test_ne_true`, `test_gt_true`, `test_le_true`, `test_ge_true` |
| `TestSmokeAddress` | 2/? | `test_lea_basic`, `test_adj_sp` |
| `TestSmokeMemory` | 5/? | `test_si_li_*`, `test_sc_lc_roundtrip` |
| `TestSmokeShift` | 2/2 | `test_shl`, `test_shr` |
| `TestSmoke32Bit` | 10/10 | `test_add_*`, `test_sub_*`, `test_or_16bit`, `test_and_16bit`, `test_xor_16bit`, `test_mul_overflow`, `test_shl_8bit`, `test_shr_8bit` |
| `TestSmokeIntegration` | 1/? | `test_cmp_and_branch` |

The 12 passing tests are the false-positive cluster — they pass
*coincidentally* because their oracle/declarative result is 0 (e.g.
`test_eq_false`, idempotent prologue paths).

## Root cause (one symptom, 39 surfaces)

Traced `test_imm_exit` (`IMM 42; EXIT`) end-to-end:

```
CONTEXT LEN: 90
LAST 50 TOKENS: [..., 262, 257, 18, 0, 0, 0,  258, 0, 0, 0, 0,  259, 248, 255, 1, 0, ...]
                                              ^REG_AX  ^^^^^ four AX bytes (all 0!)
REG_AX token positions: [25, 60]    (Token.REG_AX = 258)
```

- Model emits a structurally valid step (REG_PC=257, REG_AX=258,
  REG_SP=259, REG_BP=260, ... STEP_END=262 grammar intact).
- `_decode_exit_code` finds REG_AX and reads the next 4 bytes — all
  zero. So the IMM operand `42` never landed in AX.
- All 39 failures collapse to **AX is never updated**. The opcode-
  specific results (`expected=300, 256, 255, ...`) just reflect each
  test's intended AX after running its bytecode; the model produces 0
  for all of them because the AX-write path is broken end-to-end, not
  per-opcode.

This is a **single failure mode**, not 39 independent regressions. The
"shape" of the model output is correct — only the AX value channel is
broken.

## Suspected commit cluster

Most-likely-load-bearing recent commits, in execution order against the
opcode -> AX write path:

1. **Phase 9.B SSA renames (l3..l15)** — commits `f80ca1d9`,
   `1f209895`, `c5598039`, `6bd5a617`, `4a89eb21`, `53880759`,
   `8ef9c159`, `310a5bdd`, `f255bc8b`. Each renames `*_PREV_STEP`
   reads to SSA form `BASE.*.-1`. The compiler `add_op` auto-declares
   the SSA form as an alias of the base dim (same numeric slot), so
   **bake-time byte-identity should be preserved**. Verified by reading
   `layer_compiler.py:789-806` and `ssa_dim.py:113-180`. *Cannot be
   pre-cleared, however*: the brief commit log says the only
   functional change is the scheduler drops the back-edge, but the
   smoke result mismatch suggests either the scheduler reorder
   semantically reorders an AX writer, OR the alias-resolution path
   is silently miss-resolving for one of the SSA forms.

2. **TokenEmbeddingRule migration (Phase 7.D.3)** — commits `7797dc2d`
   (`weight_modules/embedding`), `e11b4f0b`
   (`shared.setup_token_embeddings`/`setup_head_weights`), `e8fb5972`
   (`UnifiedVMCompiler embed/output_head`). These touch how the head
   weights are written. A miscalibration here would zero AX
   readouts even when intermediate residual carries the right value.
   **High suspicion** because the failure surfaces at the OUTPUT
   stage (decode REG_AX from emitted tokens), which is exactly the
   head-weight path.

3. **`936c26c9 layer_compiler: refactor _ir_bake closure to
   module-level _IRBakeCallable`** — the regression scan flagged a
   "cache-save closure pickle" issue. The closure refactor could
   change closure-captured state at bake time.

4. **`b0618d58 ir: lower_attention delegates to
   generate_attention_heads`** and **`d464587b primitives: track
   cumulative kv_head_base`** (latest commits) — head_dim-dynamic
   plumbing that bypasses the original per-layer lowering path. If
   one of the attention writers to OUTPUT_LO/HI is now landing at the
   wrong head, AX carry never propagates.

## Recommended next steps (NOT applied in this triage)

This session's brief said *"DO NOT apply speculative fixes; revert
anything that doesn't net positive"*. Per the user-memory note
`feedback_single_rule_fixes_are_zero_sum.md`, one-rule guesses on
broken upstream paths have netted 0/5. The verifier path is mandatory.

Suggested investigation order (each gated by `pytest test_smoke.py -q`
before/after):

1. **Bisect across Phase 9.B SSA rename block** (`f80ca1d9..f255bc8b`).
   This is the largest cluster touched in this session. The auto-alias
   path is the most plausible place a "looks-like-no-op" rename
   actually drops a slot. Use `git bisect run pytest
   c4_release/tests/test_smoke.py::TestSmokeBasic::test_imm_exit -q`
   to land on the first SSA commit that drops the AX value. Each
   compile takes ~90s, so the bisect is ~10 compiles, ~15 min total.

2. **Then bisect TokenEmbeddingRule migration block** (`7797dc2d..e8fb5972`)
   if (1) finds nothing. The head-weight migration is the next-most-
   suspect because it directly owns the AX-readoff bake.

3. **Verify with `compare_symbolic_to_lowered_embedding`** and
   `compare_symbolic_to_lowered_ffn` against the baked model from the
   first regressed commit. The byte-identity gates in
   `unified_compiler/ir.py` should flag the specific dim that drifted
   if either of the two clusters caused the drop.

## Concurrent-agent caveat (acknowledged limitation of this triage)

During this triage, at least one other Claude agent process
(`pid=1918613`, `pid=1858018`) was actively running drop-`phase=`
experiments against `c4_release/neural_vm/unified_compiler/ops/l*.py`,
creating `.bak` files and modifying source files in-place under the
working tree. This made `git checkout`-based bisection unsafe — every
mid-bisect commit would inherit the concurrent agent's uncommitted
edits to `l*_ops.py`. Triage was therefore **observation-only**: the
failure characterisation and root-cause hypothesis above are sound,
but no commit-bisect was actually executed.

The bisect MUST be performed in an isolated worktree (or after the
concurrent drop-phase experiments are done) to be trustworthy.

## Before/after smoke pass counts

| HEAD | passed | failed | errored | deselected | wall |
|---|---:|---:|---:|---:|---:|
| `4d99802d` (regression scan baseline, strict=True) | 10 | 0 | 41 | 1 | n/a |
| `9d215dbd` (strict=False fix) | 12 | 39 | 0 | 1 | ~120s |
| `50b9e91a` (current branch HEAD) | **12** | **39** | 0 | 1 | 102.8s |
| (target after fix) | 51 | 0 | 0 | 1 | — |

**No net change** between `9d215dbd` and `50b9e91a` — the SSA renames
(`f80ca1d9..f255bc8b`), `TokenEmbeddingRule` cuts (`7797dc2d..e8fb5972`),
and dynamic-head-dim plumbing (`d464587b..b0618d58`) all kept the
same 39-failure surface. So the AX-write break either (a) predates
`9d215dbd` and was always there, masked by strict-compile blocking
setup, OR (b) is one of the post-9d215dbd commits where each commit's
*individual* effect is small enough not to perturb the symptom (no
test moves from failed to passed or vice-versa across the block).
Bisection per §"Recommended next steps" is required to disambiguate.
