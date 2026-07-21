# The LEAN compacted forward — performance foundation

`qwen_lean_forward.py` is a **lean native forward** of the COMPACTED (7-14 layer,
6-head) fused C4 VM. It is the performance foundation the three follow-on
optimization agents (CUDA-graph/fusion, async KV-eviction, iterative muldiv) build
on. This doc is the entry-point map + the honest measured numbers.

## What it is (and what it is NOT)

The fused VM has three forward implementations, in decreasing weight:

| forward | module | shape | fp32 params | ms/VM-step (GPU) |
|---|---|---|---|---|
| **deep pure-forward** | `nibble_pure_forward_complete` | 308 blocks, dim 2392, 23 heads | **~128 GB** | ~65 ms/step (sparse/streamed runner) |
| **HF-Qwen2Model (compacted)** | `qwen_full_vm` | 7-14 layers, 14 q-heads (6 CAM), dim ~960-1150 | ~120-200 MB | ~19-23 ms/step |
| **lean-native (compacted)** | `qwen_lean_forward` | SAME weights as HF, hand-written forward | ~120-200 MB | **~9-12 ms/step** |

> **Measured deep-model footprint (this branch, honest):** the 308-block reference
> is **~127.9 GB of fp32 parameters** — it does NOT fit on a 24 GB GPU (dense `.to`
> OOMs) and thrashes a 125 GB host (only its sparse/streamed runners — the source of
> the ~65 ms/step figure — make it tractable). The compacted models are ~3 orders of
> magnitude smaller (~120-200 MB), which is the whole point of the compaction. So the
> honest three-way is **lean 9-12 ms/step vs HF 19-23 ms/step vs the deep reference's
> ~65 ms/step** — the compacted lean forward is a ~5-7x step-time reduction AND a
> ~1000x memory reduction over the deep model.

The compaction (5 register CAM heads + 1 memory head, each register gathered on ONE
frame token by a single content-match + recency) is `qwen_full_vm`'s. The lean
forward **reuses `qwen_full_vm.build`'s bake verbatim** — it copies the baked
weights straight out of the `transformers.Qwen2Model` — and runs them through a
minimal decoder-only forward (RoPE + RMSNorm + softmax + SwiGLU, GQA `repeat_kv`),
dropping the HF `Cache` object / `position_ids` / `cache_position` plumbing /
attention-mask construction / `ALL_ATTENTION_FUNCTIONS` dispatch / `dynamic_rope_update`.

The math is identical to `transformers.models.qwen2.modeling_qwen2`, so the lean
forward decodes **byte-for-byte the same registers as the HF model** — even where
the model is wrong (e.g. the known signed-compare `LT`/`GE` failure on this branch,
which the lean forward reproduces exactly). Only the *evaluator* is lean.

## Entry points (the reusable hooks)

```python
from c4_min import qwen_full_vm as Q
from c4_min import qwen_lean_forward as LF

vm   = Q.build(code_size=24, subset=Q.SUBSET_MEM_CMP)   # ONE bake (HF Qwen2Model)
lean = LF.LeanQwenVM.from_full_vm(vm, device="cuda:1")  # extract weights, lean eval
```

- **`LeanQwenVM.from_full_vm(vm, device, dtype)`** — extract the baked weights of a
  built `QwenFullVM` into flat tensors on a device. ONE bake, two evaluators (HF +
  lean) share byte-identical weights.
- **`LeanQwenVM.forward(x, past=None, q_positions=None) -> (hidden, new_past)`** —
  the lean block-stack forward. `x` = `[B,S,H]` embedded/overlaid residual; `past` =
  per-layer `[(K,V,pos), ...]` KV cache (or None); `q_positions` = `[S]` or `[B,S]`
  absolute positions. **This is the single hook the CUDA-graph / fusion agent wraps**
  (it is a pure `torch` forward — no HF, no Python control flow inside).
- **`run_program_lean(lean, code, max_steps)`** — naive one-forward-per-VM-step
  driver (mirrors `qwen_full_vm.run_program` exactly). Byte-exact vs `isa.interpret`.
- **`speculative_run_lean(lean, code, block_steps=32, ...)`** — perfect-draft
  speculation: the deterministic VM drafts the whole register trace for free
  (`draft_program_lean`), then the lean forward verifies `block_steps` steps per
  forward by **batching the independent per-step windows** into one `[B,S,H]`
  forward (`_build_spec_batch`). Byte-identical to the naive driver, with real
  forwards saved. **This is the hook the async-KV / iterative-muldiv agents extend.**

The batched-window verify is where async KV-eviction plugs in: `LeanQwenVM.forward`
already accepts a `past` KV cache and per-row `q_positions`, so a windowed KV cache
with bounded eviction (the `pf_speculative` / `batched_speculative` policy) drops in
without touching the bake.

## Measured numbers (honest, this branch)

Device: GPU (cuda), fp32, `code_size=24`. `run_program` = naive one-forward-per-step;
ms/step is the amortised wall-clock per VM step.

### Three-way head-to-head (ms per VM step)

| subset | layers | lean-native | HF-Qwen2 | HF/lean overhead | deep 308-block |
|---|---|---|---|---|---|
| base   | 7  | **9.29** | 19.21 | 2.07x (+9.9 ms) | (see below) |
| mem+cmp | 10 | **11.68** | 22.85 | 1.96x (+11.2 ms) | (see below) |

- **lean == HF byte-for-byte on ALL programs** (arith / cmp / branch / loop /
  memory / var / zfod / latest-write-wins).
- The HF `transformers` wrapper roughly **doubles** the per-step cost — that
  ~10-11 ms/step is pure machinery overhead (Cache, mask, dispatch, rope-update).
- vs the ~65 ms/step deep 308-block reference, the lean compacted forward is a
  **~5-7x** step-time reduction (compaction from 308 blocks / 23 heads / dim 2392 to
  7-10 layers / 6 CAM heads / dim ~1000 is the bulk; the lean evaluator removes the
  HF overhead on top) — plus a ~1000x memory reduction (128 GB → ~200 MB).

CPU (8 threads) same-device three-way corroborates the ratio: lean 15-21 ms/step,
HF 23-35 ms/step (1.5-1.6x overhead). The deep 308-block reference could not be
timed densely on this host (128 GB fp32 → swap-thrash), consistent with its
sparse/streamed-only ~65 ms/step figure. Run:
`python -m c4_min.bench_lean_forward --device cuda:1`.

### Window-size scaling (lean forward, mem+cmp)

The lean forward is essentially **flat in window length** — the per-forward cost is
dominated by the 10-layer stack launch, not the sequence:

| stream_len | fwd ms | ms/token |
|---|---|---|
| 7   | 6.61 | 0.944 |
| 15  | 6.96 | 0.464 |
| 39  | 6.90 | 0.177 |
| 71  | 6.92 | 0.098 |

→ big windows (batched speculation) amortise the launch nearly perfectly. This is
why the speculation batches K steps into one forward.

### Speculation (perfect draft, block_steps=32)

| program | steps | naive fwd | spec fwd | forwards saved | wall speedup | exact |
|---|---|---|---|---|---|---|
| loop_cd5  | 22  | 22  | 1 | 22.0x | 4.3x | ✔ |
| loop_cd20 | 82  | 82  | 3 | 27.3x | 4.3x | ✔ |
| loop_cd60 | 242 | 242 | 8 | 30.2x | 4.4x | ✔ |
| var_add   | 10  | 10  | 1 | 10.0x | 3.0x | ✔ |

The deterministic VM is a *perfect* draft, so **every** drafted step is accepted
(the verify decodes the same registers the naive per-step forward would) — the
forwards-saved is `steps / ceil(steps/block_steps)`. Wall speedup (3-4.4x) is below
forwards-saved because a `block_steps`-wide window is bigger per forward, but the
flat window-scaling above keeps it a solid net win. The speculation trace is
byte-identical to the naive lean driver (asserted in `test_qwen_lean_forward.py`).

## Byte-exactness

- `run_program_lean` == `qwen_full_vm.run_program` (HF) byte-for-byte on every
  benchmark program (the bench asserts `lean == HF`).
- `run_program_lean` == `isa.interpret` on every program where the model is correct.
- `speculative_run_lean` == `run_program_lean` byte-for-byte.
- 18 tests in `test_qwen_lean_forward.py` pin all of the above.

## For the follow-on agents

- **CUDA-graph / fusion**: wrap `LeanQwenVM.forward`. It is a fixed-shape-friendly
  pure-torch stack (RoPE/RMSNorm/softmax/SwiGLU); the register decode reads
  `hidden[:, -1]`. The naive driver rebuilds the window each step (fixed max shape
  for a given code), so a captured graph over `forward` is straightforward.
- **async KV-eviction**: `forward` already takes a `past` per-layer KV cache and
  per-row `q_positions`. Port the `pf_speculative` bounded-eviction (softmax-value
  zero-head skip) policy over `new_past`; the compacted store log is already
  latest-write-wins compacted by the driver.
- **iterative muldiv**: MUL/DIV/MOD need the pruned FFN table today (`SUBSET_MULDIV`,
  `mdm_keys`). The lean forward runs the same baked table; an iterative gadget slots
  into the block stack the same way (extend `qwen_full_vm._block_specs`, re-extract).
