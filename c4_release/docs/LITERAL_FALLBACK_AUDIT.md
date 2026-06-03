# Literal Fallback Audit (sibling-of-L10-blocker)

**Date:** 2026-06-03
**Scope:** `c4_release/neural_vm/**/*.py`
**Trigger:** L10 `d_model=512` literal fallback (`l10_ops.py:6764`) exploded under dim-alloc shift; need to find sibling latent literals.

**Patterns searched:** `d_model=512`, `d_model = 512`, `or 512`, `n_heads=8`, `head_dim=64`, `ffn_hidden=4096`.

## Classification rubric

- **HIGH** — executes inside an active bake/lowering path; silently substitutes the literal when the expected derivation fails (the L10 pattern).
- **MED** — constructor default with a real override at every documented call site.
- **LOW** — test fixture, docstring/comment, or explicit assertion.

## Top 5 HIGH-risk

1. **`unified_compiler/ops/l10_ops.py:6622`** — `d_model = block.ffn.W_up.shape[1] if hasattr(block.ffn, "W_up") else 512`. Tail-FFN bake; any tail block missing `W_up` (e.g., declarations-only IR or rebuilt non-PureFFN) silently lowers against 512 even when the model is wider, mis-pinning OUTPUT/CARRY/CMP. Exact sibling of the blocker just fixed at line 6764.
2. **`unified_compiler/layer_compiler.py:638`** — `head_dim = 64` when `attn is None or not hasattr(attn, "W_q")`. Routes every declarations-only op factory through `head_dim=64`; a 12-head L15 (`d_model//8 == 64` only by coincidence) or any future non-MHA shape will lower attention IRs against the wrong slot stride.
3. **`unified_compiler/compiler.py:3488`** — `head_dim = d_model // 8  # 64 (based on default 8 heads)` inside `_resize_l15_attention`. Hard-divides by `8` regardless of `attn.num_heads`; if a model is rebuilt with `n_heads != 8` (allowed by `full_vm_compiler_dynamic.py`), L15's resize writes `head_dim` into `attn.head_dim` (line 3493) that disagrees with the rest of the stack.
4. **`unified_compiler/full_vm_compiler_dynamic.py:1171`** — `ffn_hidden = max(widths) if widths else 4096`. The `else 4096` branch fires when `compiled_model.blocks` is empty (e.g., synthetic / partial rebuild fixtures), seeding the rebuild with a stale literal that downstream `_BakeDim`/post-ops will trust.
5. **`vm_step.py:1612-1618`** — `FullModel.__init__(... d_model=512, n_heads=8, ffn_hidden=4096 ...)`. Multiple runners (`fast_runner.py:30`, `batch_runner.py:126`, `batch_runner_v2.py:41`, `transformer_first_runner.py:35`, `run_vm.py:197`, `single_op_executor.py:33`, `moe_vm.py:33`, `batched_pure_neural.py:247`) re-state the same literal triplet instead of importing from one source. A single arch-spec migration that updates `vm_step.FullModel`'s default but misses one runner reintroduces the L10-style drift.

## MED — constructor defaults (override expected)

- `vm_step.py` 665, 765, 831, 1006, 1141, 1275 — post-op `__init__(d_model=512, ...)`; production callers pass explicit `d_model`, but the L10 incident shows the override path can be skipped.
- `dim_allocator.py:57` `DEFAULT_POOL_WIDTH = 512` — only used as `Allocator(d_model=DEFAULT_POOL_WIDTH)` default; documented, but identical failure mode if a wider model forgets the override.

## LOW — fixtures / docs / asserts

- `tests/test_dim_registry.py:254,261` — asserts d_model=512 in registry report.
- `setup_helpers_l4.py:134`, `setup_helpers_l6.py:321`, `unified_compiler/model_shape_constraint.py:36,38`, `unified_compiler/ir.py:1164`, `vm_step.py:2214,2218,2518`, `dim_registry.py:462` — all comments/docstrings.
- `nibble_bytecode_executor.py:402`, `fully_neural_vm.py:374`, `autoregressive_nibble_vm.py:498` — `AutoregressiveVM(d_model=1280, ...)` demo entry points (literal but in a `__main__` driver, not a bake path).

## Sweep gaps

- `or 512`, `or 4096`, `or 64`, `or 8` searches returned no new fallback patterns (only string matches in comments).
- `getattr(... "d_model" ... 512)` style fallbacks: none in active bake code.

## Recommendation

Promote items 1-3 to fix candidates immediately; they share the exact L10 failure shape (silent literal substitution under a derivation failure). Item 4 is dormant today but will fire the first time the dynamic rebuilder is invoked on a partial fixture. Item 5 is a defense-in-depth refactor: route runners through one arch-spec constant.
