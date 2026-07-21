# CUDA-graph capture of the lean 7-14 layer forward

`qwen_lean_cuda_graph.py` wraps the lean forward (`qwen_lean_forward.LeanQwenVM.forward`
— the #695 foundation: pure-torch RoPE + RMSNorm + softmax + SwiGLU over 7-14 layers,
6 CAM heads, no `transformers` machinery) in a **CUDA graph** for fixed-shape replay.
It kills the per-step Python block-loop dispatch + the 7-10 sequential per-layer
kernel launches, replaying the whole fused kernel schedule with one `graph.replay()`.

## Why (and the honest expectation)

On these ~120-200 MB compacted models the per-layer GEMM / softmax / RMSNorm kernels
are tiny, so the **launch + Python dispatch** overhead is a real slice of the naive
per-step wall-clock. A CUDA graph records the exact kernel sequence once and replays
it with zero Python and zero per-launch CPU cost.

The win is **modest-to-large depending on the driver**, exactly as expected:

- **naive driver** (one forward per VM step, small window): launch/dispatch overhead
  dominates → the **biggest** win (~2x).
- **speculative driver** (K steps batched into one `[B,Smax,H]` forward): the big
  batched forward already amortises launch well (window-scaling is flat), so the
  graph win is **modest** (~1.1-1.4x). This is the honest finding — speculation
  already does most of the launch-amortisation the graph would.

## The bucketing scheme

A CUDA graph needs **static input shapes AND static memory addresses**. The lean
forward's shape varies two ways, so we **bucket by the exact `(B, S)` window shape** —
one lazily-captured graph per shape, cached in a dict, replayed thereafter:

| driver | forward shape | shapes seen per run |
|---|---|---|
| naive | `[1, S, H]`, `S` = BOS + n_store + 5 reg frames + STEP_END | a couple (`S` grows with the store log: e.g. `(1,7)`, `(1,8)`) |
| speculative | `[B, Smax, H]`, `B` = drafted steps in the block (≤ `block_steps`) | one per `(block_B, Smax)` (full blocks + the tail block's smaller B) |

In practice a whole program touches only a handful of shapes, so a couple of captures
cover an entire run (the capture cost is paid once and amortised over every replay of
that shape). `GraphedLeanForward.shapes` reports what was captured.

**Optional `pad_window` mode**: pad every naive `S` up to a single `max_window` so
ONE graph covers all naive steps (the pad rows sit at a far causal position →
causally invisible → the real query row decodes byte-identically; the wrapper strips
the pad rows before returning). A coarser bucket — trades a little wasted compute on
short windows for a single capture. Byte-identity is asserted in the test suite.

The graph wraps `LeanQwenVM.forward` with `past=None` (both production drivers rebuild
the whole window each step / block and call `past=None`), so the fixed-shape
full-window forward is the correct capture target. The incremental KV-cache path
(verified Linf~6e-8) is a separate replay target for the async-KV agent; a per-bucket
graph over a fixed cache length drops in the same way against this scaffolding.

## Byte-identity

Replay runs the **same kernels over the same weights** as the eager lean forward, so
the graphed forward is **byte-for-byte identical** to `LeanQwenVM.forward` (Linf = 0.0,
not ~1e-8 — it is literally the same recorded ops), and thus to the HF `Qwen2Model`
and to `isa.interpret` where the model is correct. The graph path is a pure evaluator
swap: ZERO change to the bake, the weights, or the decode.

`test_qwen_lean_cuda_graph.py` (11 tests, GPU-guarded) pins:
- single-forward replay `torch.equal` the eager forward, and the 2nd same-shape call
  replays (does not recapture);
- shape bucketing (a new `(B,S)` captures, a seen one replays);
- the **naive graphed driver** == the eager naive driver on the full arith / cmp /
  branch / loop / memory / var battery;
- the **speculative graphed driver** == the eager speculative driver (loops + memory),
  with real forwards saved;
- the `pad_window` bucket is byte-identical (one graph covers all windows);
- CPU fallback (no CUDA → transparent eager, same trace).

## Entry points

```python
from c4_min import qwen_full_vm as Q, qwen_lean_forward as LF
from c4_min import qwen_lean_cuda_graph as CG

vm   = Q.build(code_size=24, subset=Q.SUBSET_MEM_CMP)
vm.embed = vm.embed.to("cuda:1")
lean = LF.LeanQwenVM.from_full_vm(vm, device="cuda:1")

# drop-in graphed drivers (byte-identical to LF.run_program_lean / speculative_run_lean)
r  = CG.run_program_lean_graphed(lean, code, max_steps=64)
rs = CG.speculative_run_lean_graphed(lean, code, block_steps=32)

# or wrap the forward directly (reuse the warm graph pool across calls)
g = CG.GraphedLeanForward(lean)                 # pad_window=16 for the single-graph bucket
hidden = g(x, q_positions)                       # == lean.forward(x, None, q_positions)[0]
g.stats()                                        # {shapes, n_capture, n_replay, ...}
```

- **`GraphedLeanForward(lean, pad_window=None, warmup_iters=3)`** — the replay wrapper.
  `__call__(x, q_positions) -> hidden [B,S,H]`. On CPU / no-CUDA it transparently
  runs the eager forward (`enabled=False`), so the same driver code runs everywhere.
- **`run_program_lean_graphed(...)`** — naive driver, each forward a graph replay.
- **`speculative_run_lean_graphed(...)`** — perfect-draft speculation, each block's
  batched forward a graph replay (the ideal fixed-shape target).

## Measured numbers (honest, this branch)

Device cuda:1, fp32, `code_size=24`. ms/VM-step = amortised wall-clock per VM step
(graph capture amortised out via `warmup=2`). `python -m c4_min.bench_lean_forward
--device cuda:1 --only graph`.

### base subset (7 layers) — GPU warmed

| program | steps | naive eager | naive graph | naive x | spec eager | spec graph | spec x |
|---|---|---|---|---|---|---|---|
| arith_add | 5  | 8.13 | 5.03 | **1.62x** | 3.83 | 2.93 | 1.31x |
| arith_sub | 5  | 7.98 | 3.96 | **2.01x** | 3.26 | 2.98 | 1.09x |
| if_bz     | 4  | 7.86 | 4.06 | **1.94x** | 3.49 | 2.53 | 1.38x |
| loop_cd5  | 22 | 8.16 | 4.08 | **2.00x** | 2.58 | 2.33 | 1.10x |
| loop_cd20 | 82 | 8.33 | 4.30 | **1.94x** | 2.45 | 2.37 | 1.04x |

**AGGREGATE (base, 7L):** naive eager 8.26 → graphed 4.27 ms/step (**1.94x**,
−4.0 ms/step); spec eager 2.60 → graphed 2.42 ms/step (**1.08x**, −0.19 ms/step).

### mem+cmp subset (10 layers)

| program | steps | naive eager | naive graph | naive x | spec eager | spec graph | spec x |
|---|---|---|---|---|---|---|---|
| cmp_eq    | 5  | 9.83  | 4.55 | **2.16x** | 3.65 | 2.65 | 1.38x |
| cmp_gt    | 5  | 9.62  | 4.52 | **2.13x** | 3.88 | 2.82 | 1.38x |
| mem_si_li | 7  | 10.67 | 5.37 | **1.99x** | 3.88 | 3.21 | 1.21x |
| var_add   | 10 | 12.05 | 5.85 | **2.06x** | 4.03 | 3.66 | 1.10x |
| loop_cd20 | 82 | 10.18 | 4.58 | **2.22x** | 2.51 | 2.37 | 1.06x |

**AGGREGATE (mem+cmp, 10L):** naive eager 10.34 → graphed 4.75 ms/step (**2.18x**,
−5.6 ms/step); spec eager 2.85 → graphed 2.58 ms/step (**1.11x**, −0.27 ms/step).

### Reading the numbers honestly

- **Naive driver: a solid ~2x.** Halving the per-step latency (~8-12 → ~4-6 ms/step)
  by removing the 7-10 sequential per-layer launches + Python block-loop per step —
  the graph replays the whole stack as one CPU-cheap `graph.replay()`. This is the
  headline win and it is real and repeatable.
- **Speculative driver: a modest ~1.1-1.4x.** The batched `[B,Smax,H]` forward already
  amortises the launch (the window-scaling is flat — one big forward per block), so the
  graph only shaves the residual Python + the single big-forward launch. As the brief
  anticipated: *speculation already amortises launch well, so the graph win is MODEST
  on top of it.* The two optimisations are complementary — speculation cuts the NUMBER
  of forwards (K steps per forward), the graph cuts the cost of EACH forward.
- **Timing variance note:** the ABSOLUTE ms/step is GPU-clock sensitive on this shared
  GPU — a cold-clock timing window can transiently inflate a row ~4x on BOTH eager and
  graph (the RATIO stays ~1-2x). The base table above is with the GPU pre-warmed (a
  few throwaway forwards first); the bare fixed-shape forward is a stable 4.5 ms (7L) /
  6.5 ms (10L). The per-driver ms/step and the eager/graph ratio are the robust signal —
  warm the GPU for the cleanest absolute numbers.

## For the follow-on agents

- **async KV-eviction**: the same `(B, S=cache_len + new)` bucketing captures a graph
  over `LeanQwenVM.forward(x, past, q_positions)` with a FIXED cache length — pad the
  KV cache to a bucketed max length and replay. The scaffolding (per-shape capture
  dict, device-pinned capture, pad-and-strip) drops in directly.
- **iterative muldiv**: the graph wraps whatever block stack the bake produces, so an
  iterative MUL/DIV gadget added to `qwen_full_vm._block_specs` is captured for free —
  no change to `qwen_lean_cuda_graph.py`.
