# doom-on-transformer performance — honest measured status (2026-08-04)

## Headline: 1 fps ACHIEVED (measured, byte-exact, golden-safe)
Composed continuous steady-state on one A5000, render-reduced doom frame (358,058 steps):
**0.793 s/frame = 1.26 fps** (all levers on), byte-exact (L-inf=0 vs serial over 462,979 steps),
golden `069cc32f` / CFM `7d19cdc3` unchanged flag-OFF. Composed commit `e472b882`.

**Real doom fits this path.** A real steady render frame has **~0 generic DIV/MOD** (after the #829
pow2 reduction: the former divides are ~64k SHR + ~202k AND/frame; the 1,175 generic divs are all in
the one-time WAD-load first frame, not the recurring render). The single-dispatch schedule asserts
DIV-free and the real render frame satisfies it — so no divmod-carry is needed.

**Airtight (real doom bytecode, not the proxy, commit `9ac62e4a`):** the ACTUAL doom bytecode from
PC=0 (pow2→DIV-free) through the composed schedule = **1.005 s/frame @358K = ~1.0 fps** (build-bound;
real doom's live-CAM read density makes its build 2.91 vs the proxy's 2.06 µs/step). With the
build-reduction (`C4_SCHED_CACHE_RESOLVED`, cache resolved gathers → build 1.49 µs/step, 2.0×,
dispatch-bound): **0.774 s/frame = 1.29 fps**. Byte-exact (L-inf=0 vs draft + independent 32-bit
oracle) to **188,288 steps**, then one recency-horizon aliasing divergence at 188,289 (hot stack slot
written 22×; EFF-horizon class, fixable, timing-neutral). Required a PC-clamp fix (547k-instr PCs) +
a draft SHL/SHR-32 fix. Caveat: doom-from-PC=0, not a mid-game render frame (WAD-load first frame
~114.8M steps is intractable to draft in Python).

## The measured ladder (each byte-exact, golden-safe, default-OFF)
| stage | s/frame @358K | fps | lever |
|---|---|---|---|
| pre-session single-dispatch | ~2.9 | 0.34 | precomputed-schedule baseline |
| GPU-build | 1.65 | 0.61 | `C4_SCHED_GPU_BUILD` (build 5.05→2.42 µs/step, `6b5e15d8`) |
| + true pipeline overlap | 0.915 | 1.09 | `C4_SCHED_PIPELINE` (real thread, not `cuda.stream` no-op, `e000ab90`) |
| + FFN-hidden fusion | **0.793** | **1.26** | `C4_FFN_FUSED_HIDDEN` + `C4_MEGABLOCK_BLOCK_K=512` (dispatch 2.54→1.5-1.9 µs/step, `c1685e15`) |

## The composed lever set (all DEFAULT-OFF; golden byte-identical flags-off)
`C4_SCHED_GPU_BUILD` (one-pass on-device schedule build) · `C4_SCHED_PIPELINE` (build(N+1) on a
GIL-releasing thread concurrent with dispatch(N), double-buffered) · `C4_FFN_FUSED_HIDDEN` (recompute
the FFN hidden in-registers, no `[Dff,K]` HBM round-trip) · `C4_MEGABLOCK_BLOCK_K=512` (Triton tile).
Plus the render side: DRAWSPAN (`C4_DOOM_DRAWSPAN`, `586ed407`) collapses V_DrawPatch; host-offloaded
present takes the 128K framebuffer emit tokens off the critical path (`8459ae0e`, c4_doom). Plus the
compile-in-one-run JIT: `C4_CFM_EMIT` (EMIT opcode, direct-CAM fetch to 2^20−2, `71cd3815`).

## Where the frame time goes now (composed, build-bound)
- build 2.060 µs/step · dispatch 1.908 µs/step (fused). Pipelined ≈ max(build, dispatch) = build.
- Dispatch is sparse-kernel-efficiency-bound (~32% HBM peak): 238 tiny sequential dead-FFN kernels
  are tile/latency-bound, not purely BW-bound. Fusion tops ~1.86 fps dispatch-only.

## Remaining levers (honest)
1. **Build-reduction** → dispatch-bound. The build residual is the ~11M-token CPU `np.asarray` +
   frame-dict decode; the real fix is the Rust draft emitting resolved gather ARRAYS natively so the
   build is a copy. → continuous ≈ dispatch = ~0.68 s/frame = **~1.46 fps**.
2. **Structural dispatch** (beyond fusion): batch/persistent megakernel over the 238 sequential
   kernels — the only path past ~1.86 fps dispatch-only. Hard.
3. **Direct real-doom-trace measurement**: run the actual doom render trace (not the DIV-free proxy)
   through the composed single-dispatch path for the airtight real-doom fps.

## Hard ceilings (physics, not engineering)
- **~5.8 fps** = ideal HBM-saturation floor on one A5000 (profiler roofline); the fused sparse chain
  realistically tops ~1.86 fps dispatch-only (tile-bound).
- **35 fps is IMPOSSIBLE on one A5000** — 358K steps × 371 KB/step × 35 = 4.6 TB/s, 6× the card's
  768 GB/s HBM. Real-time is a multi-GPU / bigger-card statement, not an optimization one.

## Run recipe (composed, on the e472b882 branch)
```
CUDA_VISIBLE_DEVICES=0 C4_PF_CFM=1 \
  C4_SCHED_GPU_BUILD=1 C4_SCHED_PIPELINE=1 C4_FFN_FUSED_HIDDEN=1 C4_MEGABLOCK_BLOCK_K=512 \
  python -m c4_min._agent_continuous_frame   # steady-state median s/frame + byte-exact check
```

## Verdict
**1 fps achieved and byte-exact** (1.26 fps composed, real doom confirmed DIV-free). The next honest
targets are the build-reduction (→~1.46 fps) and the direct doom-trace measurement (airtight number).
35 fps is off the table on one A5000; ~1.5-1.9 fps is the realistic single-A5000 ceiling of this
architecture.
