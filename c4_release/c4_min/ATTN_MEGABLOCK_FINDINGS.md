# Attention-block fused megakernel (C4_ATTN_MEGABLOCK) — MEASURED

Base `consolidate-0.5b-2026-07-22` @ `2f4bae6b` (post FFN wave-batch).  GPU: A5000
(24 GB, ~768 GB/s HBM, 6 MB L2).  Model: 242 blocks, `dim=1416`, `C4_PF_CFM=1`,
recurrent-divmod, dense_kernel.  Composed on-chip precomputed-schedule fast path (all
landed levers on: dead-block-fusion + direct-CAM + on-chip residual + precomputed
schedule + GPU-build + fused-hidden + FFN wave-batch + `C4_MEGABLOCK_BLOCK_K`).

## Live-CAM attention cost attribution (the wall this task targets)

3 live-CAM blocks (doom CFM config), each H=24, HD=59:

| block | name           | cam heads              | FFN nnz (up/gate/down) | W_q+W_o nnz |
|------:|----------------|------------------------|------------------------|-------------|
|  2    | code-select    | h23/code(12b)          | 0 / 0 / 0  (=0)        | 29 + 6      |
|  7    | mem-cam        | h20/mem(32b)           | 33 / 33 / 33 (=99)     | 68 + 16     |
| 11    | stack-pop-cam  | h21/pop, h22/lev (32b) | 18 / 18 / 18 (=54)     | 136 + 32    |

**The live-CAM wall is NOT GEMM/nnz work.**  Live-CAM FFN nnz = **153** total vs the
dead-FFN chain's **163,062** (0.0009×).  `W_o(cam_out)` folds on-chip to a 2-10-dim
scatter-add (near-free).  The wall is that the 3 live blocks each ran `SparseFFN.forward`
as a **DENSE `F.linear`** (a full `[K, Dff≈4096]` intermediate) + **full-width `silu`/bias/
multiply** over it + an **`h.clone()`** of the whole `[1,K,D]` residual + a `[1,K,D]<->[D,K]`
**transpose at every mega<->live boundary** — on a ~99.996%-zero FFN.

Kernel attribution of the whole-step graph body (on-chip path, chunk=96 K, bk256), OFF:

| bucket             |    %  | what |
|--------------------|------:|------|
| elementwise        | 37.5% | live-FFN bias-adds + `x + W_down(...)` on full `[K,Dff]`/`[K,D]` |
| silu               | 19.9% | live-FFN + block-0 full-width `silu(up)*gate` |
| FFN-mega (triton)  | 19.8% | the dead-FFN chain (`_fused_upgate_silu` + `_down_delta`) |
| copy/clone         | 10.7% | the 3 per-live `h.clone()` |
| memcpy DtoD        |  5.0% | mega<->live transposes |
| GEMM (W_o/attn)    |  6.2% | block-0 W_o + small residual GEMMs |

So ~**73% of the step body is the live-CAM dense-FFN + clone + transpose**, only ~20% is the
(already wave-batch-fused) dead-FFN chain.  This is the "live-CAM ~92% / ~11× the FFN chain"
wall the task describes (on the dense-W_o `verify_blocks` profile it shows as GEMM; on the
on-chip path it shows as elementwise+silu+clone).  It is **tile/launch/BW-bound overhead**,
not big compute.

## C4_ATTN_MEGABLOCK — the fused megablock analog for the attention blocks

Route each live block's FFN through the SAME sparse fused-hidden `[D,K]` megakernel
(`_MegaFFN.run_fused`) the dead-FFN chain uses (computes ONLY active output rows, snapshots
only read residual dims — no full-width `silu`, no `[Dff,K]` HBM hidden, no clone), on the
region's SINGLE persistent `[D,K]` residual, with the `W_o(cam_out)` inject as an in-place
scatter-add.  The whole region (mega segments + live-CAM FFN + live-CAM delta + decode) runs
on ONE resident `[D,K]`: the per-boundary transposes collapse to ONE in-transpose (the decode
reads the `[D,K]` rows directly — no out-transpose), the 3 clones vanish, and the dense
full-width FFN becomes the sparse active-only megakernel.

### Byte-exact (L-inf=0, strict no-op)

- Per-live-block: routing block 7 / 11 FFN through `_MegaFFN.run_fused` vs the dense
  `forward_static_delta` = **L-inf = 0.000e+00** on real doom residuals (block 2 has empty FFN).
- End-to-end whole-step graph (OFF vs ON), decoded PC/SP/BP/AX over the 96 K / 131 K doom
  chunk = **L-inf = 0** (0 mismatches).
- Op corpus (loop_countdown, nested, malloc, malloc_free, matmul — arith/branch/load/store/
  memory/mul): **ALL L-inf = 0**.

### Dispatch µs/step + fps (byte-exact)

| config              | chunk  | bk  | dispatch µs/step OFF -> ON | speedup |
|---------------------|--------|-----|----------------------------|---------|
| whole-step replay   | 96 K   | 256 | 1.989 -> 1.113             | **1.79×** |
| whole-step replay   | 131 K  | 256 | 2.007 -> 1.243             | 1.61× (pre out-transpose kill: 1.61×; with it 1.80×) |

Per-frame (96,469-step DIV-free frame, cache-resolved build ~70 ms, TRUE-PIPE
≈ max(build, dispatch)), one chunk:

| bk  | 1-GPU OFF -> ON            | 2-GPU OFF -> ON            |
|-----|---------------------------|----------------------------|
| 256 | 3.80 -> **6.85 fps** (1.80×) | 7.61 -> **13.69 fps** (1.80×) |
| 512 | 3.90 -> **7.17 fps** (1.84×) | 7.79 -> **14.34 fps** (1.84×) |

The measured OFF baseline (3.80-3.90 fps 1-GPU / 7.6-7.8 fps 2-GPU @96 K one chunk) matches
the task's stated 4.08 / 8.12 baseline; the **1.79-1.84× dispatch win** projects the task's
baseline to **~7.3 fps 1-GPU / ~14.6 fps 2-GPU**.

### Kernel attribution ON (the new wall)

| bucket             |    %  |
|--------------------|------:|
| `_fused_upgate_silu` (dead chain) | 35.7% |
| `_down_delta` (dead chain)        | 35.3% |
| copy/clone (the ONE input transpose) | 19.0% |
| GEMM (block-0 W_o) | 3.4% |
| `_fused_hidden_delta_snap` (live-CAM FFN, blocks 7/11, now fused) | **0.05%** |

**The live-CAM FFN dropped from ~57% to 0.05% of the step** — fully absorbed into the fused
megakernel path.  The remaining ~24% elementwise+silu live-FFN wall is GONE.

## Honest verdict

Fusing the attention blocks buys **1.79-1.84× on the whole doom-active step / dispatch**,
byte-exact (L-inf=0).  The new wall is the (already wave-batch-fused) **dead-FFN mega chain
itself** (71% of the step: `_fused_upgate_silu` + `_down_delta`), plus the single irreducible
`[1,C,D]->[D,C]` input transpose (19%, the bridge from block-0's dense output to the `[D,K]`
chain).  The live-CAM attention is no longer the wall.  Further gains must come from the
dead-FFN kernel efficiency (it runs at ~19-30% HBM peak — occupancy-bound on the small
sequential blocks, per the dispatch-profile roofline) or from folding block-0 into the `[D,K]`
region to kill the last transpose — NOT from the attention blocks.

## Golden / VRAM

- Flag-OFF golden **UNCHANGED `069cc32f`** (`CUDA_VISIBLE_DEVICES="" python -m
  c4_min._fingerprint_build`).  `C4_ATTN_MEGABLOCK` default-OFF, pure gated tooling — no
  build-path writes.
- VRAM peak **UNCHANGED**: 13.35 GB @96 K, 14.91 GB @131 K one chunk (OFF == ON).

## Files

- `precomputed_schedule.py` — `attn_megablock_enabled()` (`C4_ATTN_MEGABLOCK`, default OFF) +
  `PrecomputedStepGraph._run_region_fused_dk` / `_decode_from_dk` (the fused `[D,K]` region +
  row-space decode).
- `fused_megablock.py` — `MegaBlockChain.run_chain_inplace_ext` (run a mega segment in place
  on an externally-owned `[D,K]` buffer).
- `_agent_attn_mega_attrib.py` — live-CAM cost attribution (CPU, lean).
- `_agent_attnmega_byteexact.py` — per-live-block `_MegaFFN` vs dense byte-exact gate.
- `_agent_attnmega_e2e_verify.py` — whole-step OFF-vs-ON byte-exact + timing.
- `_agent_attnmega_corpus.py` — op-corpus byte-exact gate.
- `_agent_attnmega_frame_fps.py` — per-frame fps OFF vs ON.
- `_agent_graph_kernel_prof.py` — whole-step body kernel attribution.
