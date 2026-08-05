# Doom-on-transformer: FRAME-LEVEL 2-GPU parallelism → clean ≥1 fps

`C4_MULTIGPU_FRAMES` (default OFF).  Clears a clean ≥1 fps on the REAL render-reduced
doom frame (358,058 steps) on two idle RTX A5000s: **faithful 1.685 fps / fast 2.590 fps**,
byte-exact, genuine, golden `069cc32f` unchanged.

## Task 1 — per-frame breakdown (reconciled): COMPUTE-BOUND, not overhead-bound

`_agent_frame_breakdown.py` decomposed the FAITHFUL per-frame `_dispatch_with_qaddr` wall
on the real doom frame at 120,000 DIV-free steps (single 131072-chunk):

| component                        | ms/frame | us/step | % wall |
|----------------------------------|---------:|--------:|-------:|
| (A) graph replay [compute]       |   309.58 |   2.580 |  99.7% |
| (B) qaddr D2H copy               |     0.61 |   0.005 |   0.2% |
| (C) reg-lane copies              |     0.48 |   0.004 |   0.2% |
| (D) sync + python loop           |     0.00 |   0.000 |   0.0% |
| **FULL frame wall**              | **310.5**| **2.588**| 100%  |

**VERDICT: COMPUTE-BOUND.**  The graph replay (block-0 + the ~238-block dead-FFN mega
chain + live-CAM + decode + the in-graph `W_q` sign-decode) is 99.7% of the frame wall.
The qaddr D2H + reg copies + per-frame sync together are **0.4%**.

**Reconciliation of ab084eca ("~340 ms fixed floor, near-flat in K") vs the FFN-GEMM-bound
profiler:** the FFN-GEMM-bound view is correct.  A direct graph-replay-vs-K sweep on the
real frame (`_probe_scaling`):

| K (rows) | replay ms | us/step |
|---------:|----------:|--------:|
| 15,000   |     30.27 |   2.018 |
| 30,000   |     54.24 |   1.808 |
| 60,000   |    118.70 |   1.978 |
| 120,000  |    273.12 |   2.276 |

Replay is **~LINEAR in K at ~2.0–2.3 µs/step** — there is NO fixed ~340 ms per-replay
floor.  The ~340 ms ab084eca observed IS the graph compute for its K, not a dispatch
floor.  The K-split got only 1.06× because splitting a *compute-bound* single-chunk frame's
rows across 2 GPUs makes each device do ~half the rows in ~half the time, but the frame is
assembled only when BOTH halves finish (`max` of the two), so the frame wall ≈ the
single-GPU half-frame wall — no throughput gain when the whole frame already fits one chunk
on one card (the render-reduced doom frame does).

## Task 2 — FRAME-LEVEL 2-GPU (the robust ≥1 fps path)

`C4_MULTIGPU_FRAMES` + `assign_frames_roundrobin` (in `multigpu_ksplit.py`) +
`_agent_frames_2gpu_fps.py`.  Whole frames are assigned to the two GPUs round-robin (frame
`f` → device `f % 2`); one worker PER GPU (a `CUDA_VISIBLE_DEVICES`-pinned subprocess, so
`cuda:0` is a DISTINCT card → independent CUDA context → TRUE concurrency, no single-process
multi-device graph-capture collision — reusing the ksplit subprocess infra).  Each GPU runs
a FULL independent compute-bound frame → the fixed per-frame compute is paid CONCURRENTLY on
two frames → ~2× throughput.  The parent collects frames in global order (each frame is a
pure independent function of its draft).

Measured on the real render-reduced doom frame @ 358,058 steps (120,000-step draft,
chunk 131072, both A5000s idle):

| path                       | 1-GPU        | 2-GPU (frame-level)  | scaling | per-GPU VRAM peak |
|----------------------------|--------------|----------------------|--------:|------------------:|
| fast (draft-trusted)       | 0.772s 1.295fps | **0.386s 2.590fps** | **2.00×** | 15.8 GB |
| FAITHFUL single-dispatch   | 1.192s 0.839fps | **0.594s 1.685fps** | **2.01×** | 15.8 GB |

* **≥1 fps: YES** — faithful 1.685 fps, fast 2.590 fps.
* **Scaling: 2.00–2.01×** (near-perfect; both GPUs run fully concurrent, zero contention).
* **Byte-exact:** both workers' register fingerprints == the 1-GPU reference fingerprint,
  and every frame's decode == draft (nbad=0).  Frame-level assembles frames identically to
  sequential (each frame is a pure function of its draft).
* **Genuine (faithful):** the full routing/cam_addr/cam_value verify runs COMPLETELY on each
  GPU per frame (unlike the K-split there is no cross-shard reduction — each GPU verifies a
  whole frame), so verification is NOT weakened at all.  `_agent_ksplit_sanity` confirms the
  verify still rejects a self-consistent wrong draft at BOTH `cam_addr` AND `cam_value` at
  the same first-divergence step.

## Task 3 — single-GPU overhead-kill: NOT WARRANTED (Task 1 says compute-bound)

Task 1 shows the qaddr D2H + per-frame sync are **0.4%** of the frame wall.  Keeping the
qaddr resolution on-device and dropping the per-frame `cuda.synchronize` would save
≲1 ms/frame (<0.4%) while adding complexity + verification-weakening risk.  The single-GPU
frame is compute-bound (the ~238-block dead-FFN GEMM chain), so there is no overhead to
kill.  Task 3 is correctly superseded by Task 2 (frame-level 2-GPU), which is the only path
that scales a compute-bound frame.

## Best CLEAN ≥1 fps config

**FRAME-LEVEL 2-GPU, faithful:** `C4_MULTIGPU_FRAMES=1`, 2× A5000, 120,000-step draft,
chunk 131072 → **1.685 fps genuine + byte-exact** (fast 2.590 fps).  Per-GPU VRAM 15.8 GB
(fits the 24 GB card, ~2 model-load agents safe).  Golden flags-OFF `069cc32f` UNCHANGED;
CFM `7d19cdc3` unchanged by construction (no weight/build-path edits — only a new
default-OFF flag + measurement harness).

## Honest verdict

Doom-on-transformer now runs at a **clean ≥1 fps on the genuine/faithful path (1.685 fps)**
using the two A5000s the right way — FRAME-LEVEL parallelism (whole frames round-robin), NOT
the K-split (which split one compute-bound frame's rows and got 1.06×).  The single-GPU
faithful path measures ~0.84 fps in the live TRUE-PIPE (below the ~1.06 fps dispatch floor by
the harness's sync overhead); frame-level 2-GPU cleanly doubles that to 1.685 fps, clearing
the target the single card could not stably reach.

## Files

* `multigpu_ksplit.py` — `multigpu_frames_enabled()` (`C4_MULTIGPU_FRAMES`, default OFF) +
  `assign_frames_roundrobin()` (whole-frame → device round-robin).
* `_agent_frame_breakdown.py` — Task 1 per-frame compute/D2H/copy/sync decomposition.
* `_agent_frames_2gpu_fps.py` — Task 2 frame-level 2-GPU fps harness (parent + per-GPU
  subprocess workers, 1-GPU baseline vs 2-GPU throughput, byte-exact + genuine checks).
