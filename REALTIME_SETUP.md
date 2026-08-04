# Real-time doom-on-transformer — setup, honest ladder, and run recipe (2026-08-04)

## Honest status: NOT real-time, and not yet measured sub-1s/frame.
- Best **measured**: composed `verify_blocks` = **838 µs/step ≈ 96 min/frame** (block-0-dominated; a7933b6b `_agent_composed_floor.py`).
- Best **estimate** (block-0 fixed): **~91 µs/step** → with the render step-count cut (358K steps) **~12–33 s/frame** (a6f982aa — analytical, unverified, build was memory-blocked).
- FLOP floor: **0.0071 µs/step** → 142M steps/s. Real-time (35 fps × 358K steps/frame = 12.5M steps/s) is **11× UNDER** that ceiling → **FLOP-feasible on one A5000**, but only if the overhead ladder below is fully paid.

## The consolidated stack (all on consolidate-0.5b @ b5fb5afd, all DEFAULT-OFF)
`C4_FUSED_MEGABLOCK` (on-chip dead-FFN kernel) · `C4_GPU_VERIFY` (on-GPU decode) · GPU-vectorized eviction · `C4_GRAPH_BLOCK0` (block-0 S-chunk graph) · `C4_BLOCK0_SQ_CHUNK`/O(K) flash band · `C4_FUSED_DELTA_FFN` · per-row block-skip. Golden `069cc32f` byte-identical flags-off. Rust draft (665M/s, 16-byte GPU-friendly capture) on branch `rust-draft-speculator-c4draft`. Render step-count cut (6.89M→358K, 19.2×, byte-exact on the *draft*) on `render-superinstruction-step-collapse`.

## The ladder to real-time (each step's × and status)
1. **838 → ~91 µs/step** — block-0 graph (`C4_GRAPH_BLOCK0`) collapses its ~635 µs host dispatch → device floor. **IMPLEMENTED (9e4ea62f), byte-exact 33/33, speedup UNMEASURED (memory).**
2. **~91 → ~µs/step** — whole-step graphing across ALL remaining blocks + giant-K amortization (drive the residual per-step dispatch, ~12,800× the FLOP floor, toward compute-bound). **NOT BUILT.** This is the dominant remaining wall (a6f982aa).
3. **block-0 40 µs compute** — fold block-0's `W_o`+ingest-FFN into the fused-delta COO path (block-0 keeps dense SparseFFN today). **NOT BUILT** (clean follow-up; a1215209 flagged it).
4. **Step-count on the TRANSFORMER** — the render superinstruction (358K steps) is proven on the *draft* but needs a model-side "render-macro" op to verify it, OR host-offloaded render. **NOT BUILT (golden-moving).** Note the emit floor: the transformer must still output ~128K framebuffer chars/frame regardless.

## RUN RECIPE (produces the honest composed number; needs ≥26 GB MemAvailable — currently blocked by unrelated 65GB+33GB jobs)
```
# 1. composed full-step floor at giant K, OFF-vs-ON, byte-exact + golden:
CUDA_VISIBLE_DEVICES=0 C4_PF_CFM=1 python -m c4_min._agent_block0_graph_report --device cuda:0
# 2. no-attention ceiling (what everything-except-block-0 can hit):
CUDA_VISIBLE_DEVICES=0 C4_PF_CFM=1 C4_BLOCK0_BYPASS=1 python -m c4_min._agent_block0_bypass_floor
# 3. project frame time = (measured µs/step) × 358,058 (render-reduced) and × 6,889,264 (raw)
```

## Verdict
Real-time is FLOP-feasible on one A5000, but requires ladder steps 2–4 (whole-step graph + giant-K + block-0 FFN-fold + a model-side render-macro) — none built, none measured. The infrastructure (composed stack + harnesses + draft) is in place; the blockers are (a) machine memory to measure step 1, and (b) the unbuilt levers 2–4. Second GPU is the trivial backstop (K-batch splits linearly).
