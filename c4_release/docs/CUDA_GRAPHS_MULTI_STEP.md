# CUDA Graphs / Multi-Step Capture Exploration

**Status:** investigated 2026-05-12. Recommendation: **do not pursue**.

Branch: `cuda-graphs-multi-step-capture`.

## Motivation

`torch.compile(mode="reduce-overhead")` captures CUDA graphs per compiled
submodule and replays them across calls, but the dispatcher still runs
per-call Python (Dynamo guards, output-buffer mark, the actual
``__call__``). A more aggressive idea: capture a full N-token generation
loop as **one** CUDA graph and replay it per VM step — in particular,
combined with speculative decoding, capture "model + spec verification +
cache append for K speculative tokens" as a single graph. Per VM step,
replay 1 graph instead of K model launches.

## Prototype

`c4_release/neural_vm/cuda_graph_bench.py` — a session-scoped benchmark
script that times three modes on a representative fixed-shape forward:

1. **baseline** — eager ``model.forward(token_ids)``.
2. **compile** — ``torch.compile(model, mode="reduce-overhead")``.
3. **cuda_graph** — manual ``torch.cuda.graph()`` capture-and-replay of a
   single forward pass, with pre-allocated input buffers.

Run:

```
timeout 360 python -m c4_release.neural_vm.cuda_graph_bench
# Default: modes=['baseline','cuda_graph']
# Override via CUDA_GRAPH_BENCH_MODES, CUDA_GRAPH_BENCH_SEQLENS.
```

## Findings

### 1. Per-step forward is **not** launch-overhead-bound

Benchmark, seq_len=128, 30 iters after 5-iter warmup, GPU-event timing:

| Mode                  | GPU mean | Host mean | Capture cost |
|-----------------------|----------|-----------|--------------|
| baseline (eager)      | ~70 ms   | ~72 ms    | —            |
| manual CUDA graph     | ~235 ms  | ~257 ms   | ~950 ms      |

The manual graph replay was **~3.3× slower** than eager. The model has
~30 transformer blocks with distinct FFN shapes (after right-sizing +
post-op expansion). Per-kernel work is large enough that the launch
overhead is negligible compared to compute; capturing them into a graph
forces all kernels through a single stream and apparently fails to
overlap as well as the eager path's natural pipelining.

(The headline number above included some GPU contention from concurrent
test runs on the same device, so absolute values are noisy. The
direction — graph replay slower than eager — was consistent across
runs.)

### 2. Capture is non-trivial: model has multiple data-dependent branches

`torch.cuda.graph()` requires the captured stream contain only
GPU-only kernel launches; any CPU/GPU sync (``.item()``,
``.to(cpu)``, ``torch.equal``, ``if t.any():``) raises
``cudaErrorStreamCaptureUnsupported`` at ``capture_end()``.

Sync points discovered in the model's forward path:

* `neural_embedding.py::_prefix_cache_matches` — `.to(cpu)` and
  `torch.equal` for prefix-cache hit detection. Already gated behind
  ``torch.compiler.is_compiling()``; can be patched by hijacking that
  signal.
* `efficient_alu_divmod_split.py::forward` — ``.item()`` for an early-
  exit when DIV/MOD aren't active. Already gated behind
  ``torch.compiler.is_compiling()``.
* `alu/ops/divmod_longdiv.py::LongDivisionModule.forward` — ``if
  b_is_zero.any():`` for divide-by-zero fixup. **Not** gated. Must be
  manually rewritten to apply the mask unconditionally (algebraically
  equivalent when ``b!=0``, which all tests assume).

The bench script patches all three to make capture succeed. Production
integration would need either:

  a) sweep the codebase and gate every `.item()`/`.any()`/`.equal`
     under ``torch.compiler.is_compiling()`` (or a custom flag) so the
     same fallback runs under both ``torch.compile`` and
     ``torch.cuda.graph()``;
  b) maintain a dedicated "capture-safe" forward path on the model,
     diverging from the eager path.

Either is a non-trivial refactor that contradicts the project's
preference for keeping forward pure-tensor.

### 3. Multi-step (K-token) capture would face the same wall

The exploratory goal was capturing K speculative steps as one graph.
For the c4 model that means K serial forwards inside the capture stream,
each of which would need (a) all the sync-point patches above and (b) a
mechanism to write the *predicted* token from step ``i`` into the input
buffer of step ``i+1`` without CPU mediation. CUDA graphs support
in-place buffer updates via captured kernel writes, so step ``i+1`` could
read from ``input[:, i+1:i+2]`` after step ``i`` writes its argmax there.
But:

  * The model's `embed.set_mem_history_end(...)` setter mutates Python
    state on the embedding; this is called between forwards in the
    speculative runner. Capturing across that boundary requires lifting
    the setter into a tensor flag, OR doing K captures that share a
    static input but differ in shape.
  * The model already runs at ~70 ms per forward at seq_len=128; K=4
    speculative steps would be ~280 ms per *graph*, and replay was
    already 3× slower than eager. The realistic win would be eliminating
    the per-call Python overhead between forwards in the spec runner
    (`drafts = []; for i in active_idx: ...; logits = model.forward(...)`),
    which on these workloads measures in single-digit milliseconds, not
    the multi-hundred-ms regime where CUDA graphs help.

## Tradeoffs (for the record)

| Aspect                     | Manual CUDA graph                          |
|----------------------------|--------------------------------------------|
| Per-call kernel-launch eliminated | Yes (replay = one launch).         |
| Per-call Python eliminated | Only between captured forwards. Surrounding spec/dispatch is still Python. |
| Memory footprint           | One copy of all intermediate activations stays alive (the captured-graph workspace). For this 30-block model: O(seq_len × d_model × n_blocks) extra. |
| First-capture cost         | ~950 ms (warmup + capture) per shape.       |
| Fixed shape                | Yes. A different seq_len needs a new capture. The decode loop sees ≥9 distinct seq_lens (`torch.compile` already warns about this). |
| Compatibility w/ KV cache  | Incompatible. KV cache mutates layer Python state in-place; can't be captured cleanly. Would require disabling the incremental fast path (same as ``compile_mode != "none"`` already does). |
| Compatibility w/ pos_ids   | OK — pos_ids are tensor-side.               |
| Compatibility w/ spec dec  | In principle yes, but multi-step capture needs the data-dependent ALU branches rewritten (see §2). |

## Recommendation: **drop**

For this model:

* Per-call forward time is dominated by compute, not launch overhead.
  Replacing eager-mode kernel launches with a CUDA-graph replay yields
  no speedup (and was measurably slower).
* The capture path requires invasive refactors to remove `.any()` /
  `.item()` from the forward — work that competes with the
  existing `torch.compile` integration which already handles those
  cases via graph breaks.
* The "K-step capture" idea only pays off if per-call Python overhead
  between forwards is significant relative to GPU time. The
  speculative runner's verification loop is already pure-tensor for
  the model forward; only the per-element argmax-CPU-transfer +
  ``_step_one`` dispatch is Python, and those are bounded by token
  emit rate, not forward latency.

If a future model variant has many tiny kernels (e.g. heavily quantized
weights, smaller FFNs, sparse compaction), the conclusion may change.
Worth re-measuring at that point with the same bench script.

## Knob exposed: `AutoregressiveVMRunner(enable_cuda_graphs=False)`

The kwarg is plumbed through the runner but currently raises
``NotImplementedError`` when set to True. This keeps the API surface
stable for future experiments — if/when a follow-up effort hardens the
capture path, the gate is in place. Default ``False``.

See `c4_release/neural_vm/run_vm.py::AutoregressiveVMRunner.__init__`.
