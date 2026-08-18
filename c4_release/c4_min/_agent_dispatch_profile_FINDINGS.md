# Doom-on-transformer single-dispatch profile — FINDINGS (measurement-first)

Base: merged `f7b2df5e` (fast-build/single-dispatch baseline: `precomputed_schedule.py`,
`direct_local_cam.py`, `_agent_wholeframe_giantk.py`). Model: 242 blocks, d_model=1416,
recurrent-divmod, dense_kernel. Frame: `nested_120_255` = **462,979 DIV-free steps**.
GPU: A5000 (24 GB, ~768 GB/s HBM, 6 MB L2). Harness: `_agent_dispatch_profile.py`,
`_agent_roofline_probe.py`, `_agent_bf16_probe.py`, `_agent_dense_vs_sparse_probe.py`.

Steady-state dispatch measured at chunk=65536 (chunk-invariant plateau, see P1):
**2.49 µs/step** (the "~3.09" baseline is the same number modulo chunk size + GPU
contention; the mega chain plateaus 1.72–2.05 µs/step across chunks 16k–131k).

## TASK 1 — dispatch breakdown (2.49 µs/step, CUDA-event sub-graphs)

| phase                        | µs/step | %     |
|------------------------------|---------|-------|
| mega dead-FFN chain (238 blk)| 1.724   | 69.1  |
| live-CAM (3 blk scatter+FFN) | 0.690   | 27.7  |
| block-0 FFN                  | 0.082   | 3.3   |
| decode lanes (snap+ax)       | 0.026   | 1.0   |
| launch / graph-replay overhead| ~0     | ~0    |

Single CUDA graph → launch overhead is already ~0 (negative residual = noise). The FFN
GEMM chain (mega + live + block-0) is ~100% of the time. Dense W_o GEMM already
eliminated (a4949ddf / on-chip residual). **Gathers cost ~0 at dispatch** (precomputed
deltas). Residual traffic is NOT the wall (bf16 probe below).

## TASK 1 — build breakdown (5.77 µs/step)

| phase                | µs/step | %     |
|----------------------|---------|-------|
| gather_cam_sparse    | 2.82    | 48.8  | ← resolve KV/code rows (the gather-resolution)
| ingest table         | 1.42    | 24.7  | ← block-0 frame ingest resolve
| routing (mega build) | 0.58    | 10.0  |
| wo_delta (W_o(cam))  | 0.50    | 8.7   |
| decode_targets       | 0.32    | 5.6   |
| embed_h0             | 0.13    | 2.3   |

**Gather-resolution (cam_sparse + ingest) = ~73% of the build.**

## TASK 2 — memory-BW roofline (THE ceiling)

Bytes/step (chunk=65536, fp32):
- hidden scratch [Dff,C] write+read: 22,098 MB/replay → **92% of bytes** (DOMINANT)
- residual [D,C] load + active writes: 1,843 MB/replay
- gather (deltas + h0_folded): 374 MB/replay
- weight stream (CSR val+col): 1.3 MB (fits L2 — reused across all K rows)
- **total ≈ 371 KB/step**

- **HBM-BW floor (if kernels hit 768 GB/s peak) = 483 ns/step**
- **MEASURED = 2,493 ns/step = 149 GB/s achieved = only 19% of peak**

### PLAIN ANSWER: is 80 ns/step achievable on one A5000?
**NO.** Even at ideal HBM saturation the floor is **~483 ns/step (≈6× over 80 ns)**, and
the measured dispatch is 2,493 ns/step (~31× over 80 ns). The frame is memory/kernel-BW
bound far above 80 ns. 80 ns/step ⇒ 358,058 steps × 80 ns = 28.6 ms = 35 fps would need
the whole 371 KB/step to move in 80 ns = **4.6 TB/s**, 6× the A5000's HBM. Not on one card.

The dominant term is the **hidden-activation scratch [Dff,C]** streamed per dead block
(up/gate write it, down reads it), NOT the residual. Proven by the bf16 residual probe:
halving the resident residual dtype → **1.004× (no gain)**. And achieved BW is only 19%
of peak → the sparse-CSR per-nnz kernels are **kernel-efficiency-bound** (poor coalescing/
occupancy on 238 small sequential blocks), below the HBM roofline itself.

## TASK 3 — per-lever ceilings (grounded)

1. **Easier gathers (draft emits resolved row-index).** Build: gather-resolution
   (cam_sparse 2.82 + ingest 1.42) = **~4.24 µs/step of the 5.77 build (73%)** — a draft
   that emits resolved indices collapses this to a copy (~0.2 µs). **Build ceiling ≈ −4 µs/step.**
   Dispatch: gather is already ~0 (precomputed deltas) → **dispatch ceiling ≈ 0.** Big
   build win, zero dispatch win.

2. **In-place scatter into resident residual.** The full-D residual read/write is NOT the
   dispatch wall (residual = 8% of bytes; bf16 no-op). The `forward_static_delta` `h.clone()`
   (full [1,C,D] copy) micro-benched at 0.0185 vs 0.0019 µs/step in-place ×3 live blocks =
   **≈ −0.05 µs/step**. Real but tiny (~2% of dispatch). Already touches only n_out dims.

3. **Opcode-segmented dense GEMMs vs sparse megablock.** MEASURED: dense-GEMM equivalent
   of the same chain = **38.4 µs/step vs 1.84 µs/step sparse → dense is 21× SLOWER.** The
   per-row compaction is already tight; dense does full work on ~99.7%-zero rows. **This
   lever REGRESSES.** Ceiling = negative. Do not pursue.

4. **Kill launch overhead.** Single CUDA graph → launch overhead already ~0 at dispatch
   (measured residual ≈ −0.03 µs, i.e. noise). **Ceiling ≈ 0** — already done. (Build side
   is host-Python, not launch; covered by lever 1.)

5. **Compact/dedup weights (~800 unique vs 163,062 nnz).** Weight stream is already only
   1.3 MB (fits L2, reused across all K) — **weight HBM traffic is 0.006% of bytes/step**,
   so deduping weights saves ~0 (the 149-GB/s bottleneck is the [Dff,C] hidden buffer, not
   the weights). Deduped index_select micro-benched ≈ replicated (both fit L2). **Dispatch
   ceiling ≈ 0.**
   *Low-precision (bf16/int8):* would halve the DOMINANT hidden-scratch traffic → up to
   ~2× on the mega chain IF the sparse kernels were BW-bound — but they run at 19% of peak,
   so realistic gain is smaller until kernel efficiency is fixed. **BYTE-EXACTNESS RISK:
   HIGH** — bf16/int8 activations break the exact-fp nibble-decode contract the whole
   verify relies on (residual bf16 is chain-internal-only and already gated non-byte-exact;
   int8 activations would need re-proving the decode margin). Flag, don't assume free.

## TASK 4 — ranked recommendation

| rank | lever | µs/step saved | side | composes? |
|------|-------|---------------|------|-----------|
| 1 | (1) draft-emitted gather indices | **≈ −4.0** | BUILD | independent |
| 2 | improve sparse-kernel efficiency* | up to ≈ −1.0 to −1.4 | DISPATCH | with all |
| 3 | (2) in-place residual scatter | ≈ −0.05 | DISPATCH | overlaps kernel-eff |
| — | (4) launch overhead | ~0 (done) | — | — |
| — | (5) dedup weights | ~0 (already L2) | — | — |
| — | (3) dense GEMMs | **+36 (REGRESSES)** | — | do not do |

\* NOT one of the five, but it's the real dispatch lever the measurements expose: the mega
chain runs at 19% of HBM peak → the ceiling if the sparse [Dff,C] kernels were BW-optimal
is ~483 ns/step (from 2493), i.e. the dispatch could drop ~5× to the roofline. The five
named levers do NOT touch this (they're gather/residual/launch/weight, all already ~0 at
dispatch, or regress). The user's five are essentially exhausted on the DISPATCH side.

### Projected fps if top levers land (358,058 steps/frame)
Dispatch is the per-step continuous cost (build/capture amortize once per frame):
- **Today**: 2.49 µs/step × 358,058 = **0.892 s/frame ≈ 1.12 fps** (dispatch only).
- **Lever 1 (build)**: does not change per-frame *continuous* dispatch (build is amortized),
  but cuts the one-time build 5.77→~1.7 µs/step. On a per-frame basis (build+dispatch),
  6.14→3.4 µs/step-equivalent — build was ~half the per-frame cost, so ~1.8× on cold frames.
- **Lever 2**: 2.49→2.44 µs/step → 0.874 s/frame ≈ 1.14 fps (negligible).
- **Ceiling (fix sparse-kernel efficiency to roofline, ~5×)**: 0.48 µs/step × 358,058 =
  **0.173 s/frame ≈ 5.8 fps**. Still ~6× short of 35 fps and ~2,160× short of the 80 ns/step
  raw-real-time target on ONE A5000.

## Already done (do not re-solve)
Single CUDA-graph dispatch (launch ~0), per-row compaction (dense is 21× worse), on-chip
residual (dense W_o GEMM eliminated; residual is 8% of bytes and bf16-insensitive),
precomputed gathers (dispatch-side gather ~0), fast vectorized build.
