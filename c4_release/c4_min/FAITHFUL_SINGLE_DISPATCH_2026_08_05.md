# FAITHFUL SINGLE-DISPATCH (C4_FAITHFUL_SINGLE_DISPATCH) — the genuinely-computing fast path

2026-08-05.  Gets the genuinely-computing ("full vanilla") doom-on-transformer path onto
the FAST single-dispatch substrate and measures how close to 1 fps it gets on real doom.

## The problem it resolves

The genuine path today (`C4_FAITHFUL_ATTN_EVICT`) is ~3000× too slow ONLY because it
forces the single-dispatch OFF (it's a draft-trust lever) and falls back to
`verify_blocks` (~6–36 ms/step vs the fast path's ~1.9 µs/step device).  The fix is to
make the single-dispatch FAITHFUL — verify the draft-derived schedule instead of
trusting it — not abandon it.

The fast single-dispatch (`precomputed_schedule`) runs the whole DIV-free doom K-batch as
ONE fused per-row map (block-0 ingest → mega dead-FFN chain → live CAM blocks → decode),
but every CAM read's ADDRESS + VALUE + the code-fetch ROUTING op is INJECTED from the
draft and TRUSTED: a self-consistent wrong draft (audit scenario E) is ACCEPTED.

## What was built (gate default-OFF)

`C4_FAITHFUL_SINGLE_DISPATCH` keeps the fast fused per-row map (byte-exact decode, fused
FFN, A-class) and adds an INDEPENDENT verification of the three draft-derived quantities,
each derived from the MODEL's OWN genuine computation, with a first-divergence terminal
FAIL (mirroring `C4_DIRECT_CAM_VERIFY_ADDR`):

1. **ROUTING (opcode)** — the model's own code-fetch query address (sign-decode of
   `W_q(x)` on `CODE_QRY_BIN`) is confirmed == the PC the draft's schedule routed on.
2. **ADDRESS** (`aa85a9d2` composed) — the model's own queried address (sign-decode of
   `W_q(x)` on `QRY_BIN`/`SP_QRY_BIN`/`LEV_QRY_BIN`) is compared to the draft's resolved
   read address, O(n_bits) per read.
3. **VALUE (the genuine softmax read, `9d9b8ec0` mechanism)** — replaces the draft-value
   INJECTION with latest-write-wins over the COMMITTED (evicted / bounded) stores at the
   address the MODEL queried — the `C4_FAITHFUL_ATTN_EVICT` value read wired into the
   single dispatch (NOT the verify_blocks fallback).  Evicted rows contribute ~0 by
   latest-write-wins/ALiBi+ZFOD, so over-evicted == over-full-log, byte-exact.

The model's genuine query addresses are decoded IN-GRAPH (one extra `W_q(x)` + sign-decode
per live CAM block, the query that block would otherwise discard) — so the verify rides on
the ONE fast dispatch pass, no second forward.  The value re-resolution is one vectorized
latest-write-wins over the model's decoded addresses (numpy, GIL-releasing).

Files: `faithful_single_dispatch.py` (new: plan + genuine value re-resolve + 3-way
verify), `precomputed_schedule.py` (in-graph qaddr decode in `PrecomputedStepGraph` +
`run_faithful_verify` single-pass), `pf_speculative.py` (`verify_blocks` routes to the
faithful path when the flag is on, does NOT force single-dispatch off).

## GENUINE-COMPUTATION PROOF (rejects the wrong draft direct-CAM accepts)

* **Address layer** (audit scenario E: LI read pointed at a planted wrong store @201=777
  while the true mem[200]=66):  fast-SD ACCEPTS (rubber-stamp, accepts step 6);
  faithful-SD REJECTS at step 5 (`cam_addr`: model_addr=200 ≠ draft_addr=201).
* **Value layer** (address correct, value stale: store 66 then 999 to addr 200, draft
  injects the STALE 66 while committed latest-write = 999):  address check passes
  (200==200) but the genuine value re-resolution (999) ≠ the draft's injected 66 →
  CAUGHT by `cam_value`.  This is the `C4_FAITHFUL_ATTN_EVICT` value read a pure address
  check cannot catch.

## BYTE-EXACT on correct execution (no false positive)

DIV-free battery (add/mul/mem/func_lev/bigimm/jmp/bnz) + a 6315-step deep loop
(sum 1..300 = 45150) + a 120K-step real-doom slice: faithful-SD == fast-SD == draft
targets, L-inf=0, and the verify ACCEPTS every step (e.g. the deep loop: 2404 address +
2404 value + 6315 routing checks all pass; real doom: 42509 addr + 42509 value + 120000
routing checks pass, 0 decode mismatches).

## THE MEASURED fps ON REAL DOOM (render-reduced 358,058 steps, clean A5000, 120K draft)

| path | SERIAL | TRUE-PIPE (double-buffer) |
|---|---|---|
| fast (draft-trusted) | 1.736 s → 0.576 fps | 1.020 s → **0.980 fps** |
| **FAITHFUL single-dispatch** | 3.032 s → 0.330 fps | 1.778 s → **0.562 fps** |

**Verdict: the faithful single-dispatch does NOT hit 1 fps — it reaches ~0.56 fps** (vs
the fast path's ~0.98 fps here; the earlier record's 1.26 fps was a slightly different
config).  But it is only **1.74× the fast path**, a ~1700× improvement over the ~3000×
of the verify_blocks fallback the genuine path was stuck at.

### Cost breakdown (µs/step, from the fps run)
* fast dispatch: 1.88 (fused block-0 + mega + live-CAM + decode)
* faithful dispatch (incl. in-graph W_q sign-decode addr verify): 2.61  (**+0.73**)
* value-verify (CPU latest-write-wins re-resolution + compare): **+1.92**
* build (schedule tables): 3.9 (shared; overlaps dispatch in TRUE-PIPE)

The dominant added cost is the CPU value re-resolution (+1.92 µs/step), then the in-graph
address decode (+0.73 µs/step).  In this TRUE-PIPE measurement the value-verify runs on
the main thread AFTER the dispatch (not overlapped), so the pipe ratio (1.74×) ≈ the
serial ratio.

### What would close the gap to 1 fps
* **Overlap the value-verify onto the pipeline thread** (it is GIL-releasing numpy) so a
  frame's value-verify hides under the NEXT frame's build+dispatch → removes ~1.9 µs/step
  from the critical path.
* **Vectorize the value-verify per-read Python compare loops** (the address/value
  per-step `for` loops in `verify_faithful` are the residual CPU cost; a fully-vectorized
  numpy gather/compare would cut it further).
* The fast path itself is ~0.98 fps here (not 1.26); the residual to 1 fps is shared with
  the fast path (build+dispatch), not unique to the faithful add.

VRAM peak (faithful graph only): **14.4 GB** at chunk 131072 (fits the 24 GB card).

## THE FIX — value-verify PIPELINED + VECTORIZED (2026-08-05, gate still default-OFF)

The genuine value re-resolution (latest-write-wins over the committed store-log) is a **pure
function of the store-log — knowable BEFORE the dispatch**.  So it is split into:
1. **A DRAFT-ONLY PRECOMPUTE** (`build_faithful_precompute`, `faithful_single_dispatch.py:355`)
   — the heavy vectorized `_genuine_value_at` latest-write-wins at every read address, done on
   the GIL-releasing background build thread CONCURRENT with the previous frame's replay
   (`PipelinedScheduleBuilder(..., faithful=True)`, `precomputed_schedule.py:1407`,`1433`,`1469`,
   `1510` — its `_run` now also runs `build_faithful_precompute`, `wait()` swaps it in).
2. **A CHEAP VECTORIZED CRITICAL-PATH COMPARE** (`verify_faithful_fast`,
   `faithful_single_dispatch.py:418`) — no Python per-read loop; one `maddr[step_arr]` gather +
   `armed & (m!=da)` / `armed & (pg!=dv)` array compares + `argmax` first-divergence.

Byte-identical first-divergence verdict to `verify_faithful` (equivalence proof — module-4
note in `faithful_single_dispatch.py`): the value check only fires at ARMED steps, and where
the address MATCHES `genuine(model_addr)==genuine(draft_addr)==pre_genuine`; where it
MISMATCHES the address `_note` dominates the same step.  Verified byte-for-byte on the whole
DIV-free battery + the 6315-step loop + the 40 000-step doom slice (`vd == vd_ref` assertion in
`_agent_faithful_sd_doom_fps.py`, and the pure-numpy `_agent_faithful_pipe_equiv.py` across 6
scenarios).

### MEASURED — the value-verify came off the critical path (CPU-only A/B, contention-immune)

`_agent_faithful_valueverify_ab.py` (40 000-step real-doom, identical verdict):

| | old critical path | new critical path | pipelined onto build thread |
|---|---|---|---|
| value-verify | 1.393 µs/step (`verify_faithful`) | **0.0047 µs/step** (`verify_faithful_fast`) | 1.110 µs/step (`build_faithful_precompute`, hidden) |

→ **298× less** on the critical path (1.393 → 0.0047 µs/step); the heavy re-resolution is
hidden under the next frame's build+dispatch on the pipeline thread.  In the full fps harness
the critical-path value-verify measures **0.006 µs/step** (was 1.92).

### fps — projection at the doc's UNCONTENDED fast dispatch (1.88 µs/step) + in-graph addr (+0.73)

| faithful critical path | µs/step | fps @ 358,058 |
|---|---|---|
| OLD (value-verify on critical path) | 1.88 + 0.73 + 1.92 = 4.53 | 0.617 (≈ the 0.562 above) |
| **NEW (value-verify pipelined+vectorized)** | 1.88 + 0.73 + 0.005 = **2.615** | **1.068 fps** |

**Verdict: with the value-verify off the critical path the genuine faithful single-dispatch
crosses ~1 fps (1.07 fps projected at the uncontended fast baseline)** — matching the fast
path's ~0.98 fps.  The remaining faithful-vs-fast delta is the in-graph address decode
(+0.73 µs/step), which was already deemed acceptable.

CAVEAT (honesty): the live end-to-end fps re-measurement on THIS machine could not reproduce
the doc's uncontended 1.88 µs/step fast dispatch — a persistent background GPU job inflated the
fast dispatch to ~5.6 µs/step, so the absolute live fps (fast 0.42 fps, faithful 0.22 fps
TRUE-PIPE) is contention-bound and NOT comparable to the doc's clean-A5000 numbers.  The
value-verify removal (298×) and the byte-exact/genuine verdicts ARE contention-immune (pure
CPU) and are the load-bearing results; the 1.07-fps figure is the projection at the doc's own
uncontended dispatch baseline.  VRAM peak (faithful graph) unchanged at 13.9 GB @ chunk 131072.

## THE BUILD-REDUCTION — the faithful path driven DISPATCH-BOUND, MEASURED live @ ~1 fps (2026-08-05, gate still default-OFF)

The pipelined-value-verify projection above assumed the fast build.  A clean-A5000 live
re-measure (120K draft, real render-reduced doom, `_agent_faithful_build_reduce_fps.py`) showed
the value-verify was ALREADY off the critical path (0.005 µs/step), but the real bottleneck was
the **SCHEDULE BUILD** — the faithful `PipelinedScheduleBuilder` re-resolved the CAM-sparse
latest-write-wins + the value-precompute EVERY frame (it never got the `C4_SCHED_CACHE_RESOLVED`
benefit).  Measured baseline: faithful **build 6.783 µs/step** ≫ dispatch 2.623 → BUILD-bound →
2.641 s/frame → **0.379 fps**.

The fix drives the faithful BUILD below the dispatch with three composed build-reductions
(all DEFAULT OFF, runtime-only):

1. **`C4_SCHED_CACHE_RESOLVED` on the faithful schedule build** — the faithful builder already
   routes through `_build_schedule_tables` (via `build_schedule_tables_only`), so it honors the
   resolved-gather cache directly once the flag is set.  Cut the per-frame CAM-sparse re-resolve
   (the ~73% of the build) to a one-time cost + per-frame copy.
2. **`C4_FAITHFUL_PRECOMPUTE_CACHE`** (`faithful_single_dispatch.py`, `build_faithful_precompute`)
   — the value-verify analog of `C4_SCHED_CACHE_RESOLVED`: the genuine `_genuine_value_at`
   latest-write-wins is a PURE function of the immutable store-log, so cache the resolved
   `FaithfulPrecompute` on the draft (keyed by store-log identity + mask + n) and reuse it per
   frame.  Byte-identical (immutable arrays, read-only in `verify_faithful_fast`); a changed
   store-log (new committed store) MISSES and recomputes — never stale.  Proven element-identical
   + genuine-preserving in `_agent_faithful_precompute_cache_equiv.py`.
3. **`C4_FAITHFUL_PRECOMPUTE_THREAD`** (`precomputed_schedule.py`, `PipelinedScheduleBuilder._run`)
   — for the UNCACHED (genuinely re-drafting) case, run the value-precompute on a SECOND inner
   thread CONCURRENT with the schedule build (both GIL-releasing numpy), so the build thread's
   steady state is `max(precompute, schedule-build)` not their sum.

### MEASURED — clean A5000, 120K draft, real render-reduced doom (358,058 steps), TRUE-PIPE

Two clean-card runs (`_agent_faithful_build_reduce_fps.py`, GPU idle @ measure):

| path | build µs/step | dispatch µs/step | value-verify | s/frame | fps | bound |
|---|---|---|---|---|---|---|
| FAITHFUL baseline | 6.78 / 4.77 | 2.62 / 2.61 | 0.005 | 2.64 / 2.19 | **0.38 / 0.46** | BUILD |
| **FAITHFUL + build-reduction** | **1.16 / 0.77** | 2.64 / 2.62 | 0.004 | **1.08 / 1.20** | **0.93 / 0.83** | **DISPATCH** |
| fast + `C4_SCHED_CACHE_RESOLVED` (ref) | 1.15 / 0.76 | 1.89 / 1.88 | — | 0.77 / 0.76 | 1.30 / 1.31 | — |

**Verdict: the build-reduction crosses the goal it targeted — the faithful build drops
6.78 → 1.16 (run 1) / 4.77 → 0.77 (run 2) µs/step, far below the ~2.62 dispatch, making the
genuine path DISPATCH-BOUND** (was BUILD-bound).  The faithful **dispatch FLOOR** (2.62 µs/step ×
358,058 = 0.94 s = **~1.06 fps**) is at ~1 fps.  The measured TRUE-PIPE is **0.83–0.93 fps**
(1.08–1.20 s/frame), a **~2× improvement** over the 0.38–0.46-fps baseline, sitting ~10–14% below
the dispatch floor.  That residual gap is the harness's double-buffer / per-frame
`cuda.synchronize` overhead (DISPATCH-side), and the faithful-vs-fast delta is the **in-graph
address decode (+0.73 µs/step dispatch)** — the inherent cost of the genuine per-read address
re-derivation (already deemed acceptable).  Byte-exact (mismatches=0, verify accepts all 42,502
addr + 42,502 value + 120,000 routing); VRAM peak 15.8 GB (fits the 24 GB card).  Goldens
`069cc32f` / `7d19cdc3` UNCHANGED (all three flags default OFF, runtime-only, weight-neutral).

**HONEST: the genuine path does NOT quite reach a stable ≥1 fps in the live TRUE-PIPE
measurement (0.83–0.93 fps) — but the BUILD bottleneck the task targeted is fully resolved
(build ≪ dispatch, dispatch-bound); the remaining sub-1-fps residual is purely dispatch-side
(the +0.73 µs/step genuine in-graph address decode + the harness's synchronize overhead), NOT
build cost.  The dispatch floor itself is ~1.06 fps.**

### Honest residual to a solid ≥1 fps

The genuine path is dispatch-bound at the ~1.06-fps dispatch floor but measures 0.927 fps due to
the ~14% TRUE-PIPE overhead (the harness's per-frame `torch.cuda.synchronize` + double-buffer
event sync above the raw dispatch).  Closing that overhead (or shaving the +0.75 µs/step in-graph
address decode) is the last step to a stable ≥1 fps genuine path; both are dispatch-side, not
build-side — the build bottleneck this fix targeted is fully resolved.

## Composition — what composed cleanly vs needed new work

* **ADDRESS verify (`aa85a9d2`)**: the sign-decode (`_decode_model_query_addr`) and the
  `DivergenceSink` first-divergence-stop pattern composed cleanly — reused directly.
* **VALUE read (`9d9b8ec0`)**: the latest-write-wins-over-committed-stores identity
  composed cleanly (same algebra as `_resolve_reads_vec`), but keyed on the MODEL's
  decoded address (new): the genuine value is `latest_write_wins(model_addr, committed
  stores)`, independent of the draft's claimed address/value.
* **NEW WORK — the routing verify + in-graph residual extraction**: the single-dispatch is
  a fully-fused graph with NO KV cache and NO per-block residual exposure, so the genuine
  query addresses had to be decoded IN-GRAPH (`decode_model_query_addrs` /
  `_decode_qaddr_at` — a new `W_q(x)` sign-decode at each live CAM block, folded into the
  captured graph body).  This is the piece that makes "genuine value reads on the single
  dispatch" tractable without a second forward.

## Goldens (flag default-OFF, runtime-only, no weight authoring)

* `069cc32f…` (flags-OFF) — UNCHANGED
* `7d19cdc3…` (CFM) — UNCHANGED
* `069cc32f…` with `C4_FAITHFUL_SINGLE_DISPATCH=1` — UNCHANGED (runtime forward wrapper)
* `069cc32f…` with `C4_FAITHFUL_SINGLE_DISPATCH=1 C4_SCHED_CACHE_RESOLVED=1
  C4_FAITHFUL_PRECOMPUTE_CACHE=1 C4_FAITHFUL_PRECOMPUTE_THREAD=1` — UNCHANGED (all build-
  reduction levers are runtime caches / thread scheduling, weight-neutral)

## Honest gaps

* The genuine path is now DISPATCH-BOUND (build 1.159 < dispatch 2.637 µs/step) and measures
  **0.927 fps** TRUE-PIPE (up 2.4× from 0.379); the dispatch FLOOR is ~1.06 fps.  The residual
  ~14% between measured pipe and the dispatch floor is double-buffer / `cuda.synchronize`
  overhead in the harness, and the faithful-vs-fast delta is the +0.75 µs/step in-graph address
  decode — both DISPATCH-side, not build-side.  The build bottleneck this fix targeted is
  resolved.
* Like `C4_DIRECT_CAM_VERIFY_ADDR`, the address check conservatively skips `model_addr==0`
  reads (an unasserted query in the batched residual view is indistinguishable from a
  genuine address-0 read) — never a false positive, documented limit.  A wrong read the
  model would have resolved to address 0 is caught by the VALUE layer instead (genuine
  value at address 0 vs injected).
