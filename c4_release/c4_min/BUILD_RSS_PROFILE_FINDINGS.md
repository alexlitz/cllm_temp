# c4_min model BUILD RSS profile — where the ~108 GB goes (2026-08-03)

Profiled the pure-forward doom-model build path with per-stage `/proc/self/status`
VmRSS sampling (`c4_min/_agent_build_rss_profile.py`). golden `069cc32f` confirmed
unchanged (streaming build, code_size=32) throughout.

## Two build paths, two very different footprints

### CFM streaming (`C4_PF_CFM=1`) — the PERF FLEET path (#809/#814) — ALREADY LEAN
`build_lib_model_streaming(code_size=3978, recurrent_divmod=True, addr32=True)`
with `C4_PF_CFM=1` (what `_agent_doom_directcam.py` / `_agent_doom_cfm_continuous.py`
/ `_agent_banded_doom_profile.py` / `_agent_doom_megakernel.py` all set):

    PEAK RSS = 1.48 GB   build wall = 9.7 s   dim = 1392 (FIXED, code_size-independent)
    finished sparse model nnz = 76,938   dense-equiv FFN = 0.3 GB

CFM makes the residual `dim` INDEPENDENT of code_size (`pf_code_size=1`), so the
program lives in the KV code frames, not a baked O(code_size) PC one-hot table.
This path does NOT balloon.

### NON-CFM streaming (`C4_PF_CFM=0`) — the ~108 GB HAZARD
Same `build_lib_model_streaming`/`build_compact_sparse_streaming` but WITHOUT CFM,
so `pf_code_size = code_size` and `dim` grows LINEARLY with code_size:

    code_size=32   -> dim=2461    peak ~1.2 GB
    code_size=200  -> dim=4140    peak ~1.7 GB
    code_size=800  -> dim=7774    peak 10.6 GB   (full build, 130 s)
    code_size=2048 -> dim~22000   peak ~60 GB    (extrapolated)
    code_size=3978 -> dim~44000   peak ~100+ GB  (OOM/swap — reproduced hitting 118 GB used)

The culprit script is `_agent_build_scale.py` (sweeps code_size 64..2048 WITHOUT
setting `C4_PF_CFM=1`) and any non-CFM doom-scale build.

### DENSE builder (`build_pure_forward_complete_model`) — 21 GB even at code_size=32
`build_pure_forward_complete_model(code_size=32)` (used by many tests/tools and by
`build_lib_model`, the non-streaming wrapper) peaks **21 GB at CS=32** in
`Transformer.__init__` alone: it pads EVERY one of the 242 stored blocks to the
global `hidden_max=4320 × dim`, ~all zeros. Scales to ~375 GB at doom scale.
(conftest.py already documents this dense path at 54-108 GB and routes tests to
the streaming build; `build_lib_model`'s own docstring says ~62 GB.)

## Ranked RSS contributors for the non-CFM streaming build (CS=800, dim=10143 pre-pack)

1. **Baked dense attention `Attn(dim,n_heads,max_seq)` — O(dim²), the runaway term.**
   3 baked CAM blocks (frame-ingest + mem-cam + stack-pop), each `W_q/W_k/W_v/W_o`
   is `dim×dim`: `3 × dim² × 4 weights × 4 bytes = 4.94 GB` at CS=800; grows
   QUADRATICALLY. `_ZeroAttn` already spares the 301 un-baked blocks, but the
   baked ones are dense.
2. **`_remap_attn` transients — O(dim²), in the final block-streaming loop.** For
   each baked block it allocates several `torch.zeros(new_dim, new_dim)` (one per
   W_q/W_k/W_v/W_o) to re-lay head partitions — another O(new_dim²) spike (this is
   where the CS=800 full build peaks at 10.6 GB).
3. **FFN block_specs (dense ragged) — O(dim).** 3.33 GB at CS=800; held for the
   whole build (shared by nn.Parameter with the intermediate, so NOT doubled).
4. **Transient `_FFN(dim,hid)` zeros — O(dim), small.** `_build_stream_intermediate`
   and the final loop each allocate a fresh zero `_FFN` then immediately replace its
   params via `_swap_ffn` — 0.53 GB throwaway for the largest block.

## Fixes landed (byte-identical, load-path only) — `compact_alloc.py`

1. **`_remap_attn` writes directly into the pre-zeroed `dst`** — no longer allocates
   a second `torch.zeros(new_dim, new_dim)` per W_q/W_k/W_v/W_o. Only the tiny
   `local×new_dim` per-head slice is built. Removes an O(new_dim²) transient per
   baked block in the final streaming loop. Byte-identical.
2. **`_new_ffn_from` builds ragged FFN params directly** — drops the throwaway
   `FFN.__init__` `torch.zeros(hidden, dim)×3` that `_swap_ffn` immediately
   discarded (both in `_build_stream_intermediate` and the final loop). Byte-identical.
3. **Build-RSS GUARD** (`C4_BUILD_RSS_GUARD`, default ON; ceiling `C4_BUILD_RSS_CEIL_GB`,
   default 24 GB): the one term the streaming build can NOT make cheap is the DENSE
   `[dim,dim]` baked CAM attention, which is O(dim²) and therefore O(code_size²)
   WITHOUT CFM. The guard estimates `#baked × 4 × dim² × 4 bytes` up front and
   FAILS FAST with actionable guidance (→ set `C4_PF_CFM=1`) instead of silently
   allocating ~84 GB and swap-killing the box. Byte-identity untouched (only aborts
   a build that would swap; a build that passes is bit-for-bit as before).

## Results (2026-08-03, after fixes)

| build                         | dim   | peak RSS | wall  | nnz    | outcome |
|-------------------------------|-------|----------|-------|--------|---------|
| golden (streaming, CS=32)     | 2461  | 2.45 GB  | 27 s  | —      | `069cc32f` byte-identical |
| **CFM doom (perf fleet)**     | 1392  | **1.59 GB** | **9.6 s** | 76,938 | lean, unchanged |
| non-CFM doom (CS=3978)        | 41929 | (was ~108 GB / OOM) | <1 s | — | **guard fires: "~84 GB dense baked attn", points to CFM** |
| non-CFM CS≤1024               | ≤12397| ≤~11 GB  | —     | —      | builds normally (guard silent) |

**The perf-fleet doom build (`C4_PF_CFM=1`) already builds at 1.59 GB in 9.6 s.**
The ~108 GB was the NON-CFM doom "scale wall" build (`_agent_doom_continuous.py`
#777, superseded ~12 h later by the CFM path `_agent_doom_cfm_continuous.py` #779)
and `_agent_build_scale.py`'s non-CFM sweep. Non-CFM doom is architecturally
O(code_size²) in the dense baked attention and cannot fit ≤12 GB; CFM (dim fixed,
code_size-independent) IS the sparse-resident fix and it is what every current
perf-fleet doom script sets.

## Concurrency
At 1.59 GB/build (CFM) the box (125 GB RAM) comfortably runs MANY model agents
concurrently — a safe cap of ~10-15 model builds at once (vs 1-at-a-time under the
old 108 GB dense/non-CFM path). The guard guarantees no single agent can silently
swap-kill the box.
