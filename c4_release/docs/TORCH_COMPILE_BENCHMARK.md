# torch.compile Benchmark (2026-05-12)

Measurement of the opt-in `compile_mode="reduce-overhead"` path landed in
commit `ce2a704` ("Add torch.compile support + grouped-GEMM SoftMoE +
.item() blockers").

## Environment

- Hardware: CUDA 12.8, single GPU (auto-detected via `torch.cuda.is_available()`)
- Python: 3.12.6
- PyTorch: bundled with project (inductor + dynamo)
- Branch: `enable-compile-mode-default`
- Worktree: `/home/alexlitz/Documents/misc/c4_release/.claude/worktrees/agent-af1a23973a8f5a92d`

Test harness: `pytest c4_release/tests/test_smoke.py::TestSmokeBasic::test_imm_exit -v --tb=line --timeout=...`

The test runs `[(IMM, 42), EXIT]` through `quick_runner` (pure_neural, trust_neural_alu)
end-to-end — model build/bake + at least two autoregressive forward passes
(one for IMM, one for EXIT).

## Results

| Configuration                              | Wall time      | Notes                                            |
| ------------------------------------------ | -------------- | ------------------------------------------------ |
| `compile_mode="none"` (baseline / default) | **78.85 s**    | Single forward path, no torch.compile.           |
| `compile_mode="reduce-overhead"` (1st run) | **>500 s** (timed out) | Compile warmup did not finish in 500s wall.     |
| `compile_mode="reduce-overhead"` (post-warmup) | not measured | Run aborted before warmup completed.            |
| Speedup ratio                              | n/a            | Cannot compute; warmup never converged.          |

### Compile warmup observations

During the timed-out run the logs showed:

- `_inductor/utils.py: Not enough SMs to use max_autotune_gemm mode` (informational).
- Two CUDAGraph dynamic-shape warnings: *"We have observed 9 distinct sizes."*
  Suggests the compiled forward is being re-recorded for each new token-count
  shape we feed it, defeating the CUDA-graph fast path under
  `reduce-overhead`.
- The trace was still inside `torch/fx/node.py` when the 500s pytest-timeout
  fired.

The model has ~30 blocks with distinct FFN shapes after `_right_size_ffns`
(`AutoregressiveVMRunner.__init__`, run_vm.py:240-272 design comment).
Dynamo retraces per block; `recompile_limit=128` and
`force_parameter_static_shapes=False` are already set, but first-run
compile is still dominated by per-block tracing + Inductor codegen +
CUDA-graph recording.

## Decision

**Stay with opt-in default (`compile_mode=None`).**

Per the task instructions: *"If first-run compile time is too slow
(>5min), document the issue and stay with opt-in default."* Observed
first-run was >8min wall (timed out at 500s = 8m20s) without completing,
so the threshold is clearly exceeded.

`tests/conftest.py` now reads `C4_COMPILE_MODE` (default `"none"`) and
threads it through `quick_runner` and `neural_runner`. CI / users can
opt in with:

```bash
C4_COMPILE_MODE=reduce-overhead python -m pytest c4_release/tests/test_smoke.py::TestSmokeBasic::test_imm_exit
```

## Gate-suite verification (default off)

With `C4_COMPILE_MODE` unset (default `"none"`):

```
timeout 800 python -m pytest \
  c4_release/tests/test_smoke.py::TestSmokeBasic::test_imm_exit \
  c4_release/tests/test_runtime_vanilla.py \
  c4_release/tests/test_layer_idx_consistency.py \
  c4_release/tests/test_compile_determinism.py \
  -v --tb=line --timeout=600
```

Result: **11 passed in 751.04 s** (one wall is dominated by ~3x `test_compile_determinism` and the smoke-test session build).

## Follow-ups required before flipping the default on

1. **Stabilize input-shape contract.** The CUDAGraph "9 distinct sizes"
   warning means the forward is being called at 9 different `seq_len`
   values during a 2-step program. Either pad inputs to a small fixed
   set (e.g. round seq_len up to powers of 2), or set
   `torch._inductor.config.triton.cudagraph_skip_dynamic_graphs=True`
   to skip CUDA-graph capture for dynamic shapes and accept a smaller
   speedup.
2. **Investigate per-block re-tracing.** Even with
   `force_parameter_static_shapes=False` the trace runs for >8 min;
   either the guard surface is still too large or Inductor codegen is
   the dominant cost. A trace-time profile (`TORCH_LOGS=recompiles,inductor`)
   would narrow it down.
3. **Add a post-warmup benchmark.** Once first-run finishes in a
   reasonable window, run `test_imm_then_nop_then_exit` twice back-to-back
   inside the same session (e.g. with `pytest --count=2` or a custom
   fixture) and record the second wall time vs the first to measure
   the steady-state speedup.

## Multi-step program (`test_imm_then_nop_then_exit`)

Not measured this round — the multi-step test lives in
`c4_release/tests/test_pure_neural_pc.py::TestPureNeuralPCBasic::test_imm_then_nop_then_exit`
and uses the `pure_neural_runner` fixture (separate from `quick_runner`).
Once the compile warmup is brought into a workable range it should be
the canonical post-warmup benchmark since it exercises the autoregressive
forward across three opcodes inside one session, amortizing the warmup
cost over multiple steps.
