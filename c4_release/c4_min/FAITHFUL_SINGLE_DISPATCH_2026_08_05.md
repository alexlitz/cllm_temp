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

## Honest gaps

* Does NOT reach 1 fps (0.56 fps); the residual is the value-verify CPU cost + the shared
  fast-path build/dispatch (which is itself ~0.98 fps here).
* The value-verify is not overlapped in the current TRUE-PIPE measurement (main-thread
  post-dispatch) — pipelining it onto the build thread is the clearest next lever.
* Like `C4_DIRECT_CAM_VERIFY_ADDR`, the address check conservatively skips `model_addr==0`
  reads (an unasserted query in the batched residual view is indistinguishable from a
  genuine address-0 read) — never a false positive, documented limit.  A wrong read the
  model would have resolved to address 0 is caught by the VALUE layer instead (genuine
  value at address 0 vs injected).
