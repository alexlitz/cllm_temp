# The GENUINELY-COMPUTING path: real softmax attention over the EVICTED cache

`C4_FAITHFUL_ATTN_EVICT` (DEFAULT OFF) — 2026-08-05

Resolves the faithfulness/speed tension the fast-path audit
(`DOOM_FASTPATH_FAITHFULNESS_AUDIT_2026_08_05.md`) found: the direct-CAM fast path
is DRAFT-TRUSTED for the memory-read resolution (a self-consistent wrong draft is
ACCEPTED — scenario E), while the fully-faithful softmax path (which independently
recomputes every read) OOMs at doom's KV scale (O(S) score over 150k–262k rows).

## The idea

Compose the REAL softmax1+ALiBi attention verify WITH eviction: run the genuine
O(cache) softmax over the EVICTED / bounded KV cache (`exact_evict` liveness
schedule, ~1–10K live) instead of the full O(S) store log. The model's OWN query
does every retrieval — no `resolve_load_rows(draft)` injection, no direct-CAM, no
direct-local, no frozen-row-skip (all of those are the B-class draft-trust levers
the faithful path replaces). Every A-class faithful lever (dead-block fusion,
fused-delta FFN / megablock, bounded + evicted KV, banded/flash local attention)
still composes.

Evicted entries contribute ~0 to softmax1 by the latest-write-wins / ALiBi + ZFOD
identity, so real-attention-over-evicted == real-attention-over-full-log,
byte-exact — proven empirically below.

## Mechanism (files changed)

* `pf_speculative.py` — `faithful_attn_evict_enabled()` (new gate). When ON:
  `exact_evict` is implied (the eviction policy is the schedule drop),
  `_frozen_skip_enabled()` returns False (the genuine CAM heads SCORE the frozen
  rows' KV — frozen-row-skip is unsafe without direct-CAM), and the
  `C4_PRECOMPUTED_SCHEDULE` fast path is bypassed (it resolves the gathers up front
  off the draft).
* `direct_cam_batched.py` — `direct_cam_batched_enabled()` returns False under the
  faithful flag (the CAM heads revert to real O(cache) scoring).
* `direct_local_cam.py` — `direct_local_cam_enabled()` returns False under the
  faithful flag (block-0 ingest heads genuinely score the emitted-token frame).

Runtime-only: NO weight authoring. Golden `069cc32f` (and CFM `7d19cdc3`)
byte-identical flag-OFF **and** flag-ON.

## VERIFY

### 1. HEADLINE — genuine computation CATCHES the wrong draft direct-CAM ACCEPTS

Self-consistent wrong draft (audit scenario E): the true store `mem[200]=66` stays
in the KV; the draft's LI read is corrupted to resolve to a WRONG planted row
(addr=201, val=777) and its declared frame ax is set to 777. The LI token frame
still carries address 200 (the model's genuine query).

| path | LI-read (step 5) | verdict |
|---|---|---|
| direct-CAM (draft-trusted) | **ACCEPTED** (accepts step 5, the wrong read; injects 777 == the wrong frame) | RUBBER-STAMP |
| faithful-attn-evict | **CAUGHT/REJECTED** at step 5 (`got_ax=66` recomputed by the model's real query, `want_ax=777`) | GENUINE |

The model's real query for address 200 retrieves the true value 66, which ≠ the
draft's claimed 777 — so the faithful path rejects the read the moment it happens.
Direct-CAM injects the draft's 777 and accepts it. **Genuine computation proven.**

### 2. BYTE-EXACT (eviction did not change the genuine-attention result)

`faithful (evict) == faithful (full-log, evict=False) == K=1 reference == draft`,
L-inf=0, on: the whole DIV-free battery (add/sub/mul/div/mod/and/shl/eq/lt/lea/
bz/bnz/jmp/**jsr_lev**/**si_li**), `loop_countdown60` (915 steps), and the deep
**nested_14x77 loop (16651 steps)** — every case `ref == full_log == evict`, at
K=512 and K=2048. (mul is the FixedMul multiply core; si_li is the memory-read
path; jsr_lev the function-call path.)

### 3. SCALE — does it FIT (bounded working set, no OOM)?

nested_10x24 (3949 steps, 1067 logical stores, `matched=True`, correct answer 10):

| K | max live cache | total evicted | peak VRAM |
|---|---|---|---|
| 2048 | 113 542 | 688 605 | 7.19 GB |
| 8192 | 113 528 | 688 521 | 20.28 GB |

The live working set is BOUNDED (~113K regardless of K; eviction dropped ~689K
rows to hold it there — 6× more evicted than live). It FITS on a 24 GB card at
every K. VRAM scales with K (the span S ~ K·30 the FFN runs over), NOT with the
store-log length — that is exactly what eviction buys: the O(S) store-log softmax
the audit found OOMing is replaced by O(bounded-cache).

### 4. COST — the honest price of genuine computation

Same program + K, faithful-attn-evict vs draft-trusted direct-CAM (both
`matched=True`):

| K | faithful µs/step | direct-CAM µs/step | ratio |
|---|---|---|---|
| 2048 | 7 168 | 5 953 | **1.2×** |
| 8192 | 36 143 | 9 911 | 3.6× |

Genuine computation costs 1.2–3.6× the direct-CAM cost (K-dependent; the penalty
grows with K because the faithful path forgoes frozen-row-skip and runs the FFN
over the whole K·30-row span). Both are tractable — no OOM. The direct-CAM ~1.29
fps real-time figure is the fully-composed doom-scale number with all graph /
real-time levers; the same-program controlled ratio here is the apples-to-apples
cost.

### 5. GOLDEN — flags-OFF unchanged

Golden `069cc32f` (flag OFF), CFM `7d19cdc3` (flag OFF), and golden `069cc32f`
with `C4_FAITHFUL_ATTN_EVICT=1` (flag ON) — all unchanged (the flag is
runtime-only, no weight authoring).

## Honest gaps

* The faithful path forgoes `C4_FROZEN_ROW_SKIP` (which requires direct-CAM), so it
  runs the FFN over ALL span rows (S ~ K·30), not just the K query rows — higher
  VRAM + slower than the composed direct-CAM path. This is the honest price of
  genuine computation.
* The doom `FixedMul` *function-call* image (`_build_call_program`) drafts to a
  ref answer of 0 under the bare `draft_pf_program` (the JSR/ADJ call-frame ABI is
  not set up by the plain drafter) — the FixedMul multiply itself is covered
  byte-exact by the battery's `mul` op. The genuine-doom function-call slice needs
  the composed-doom driver setup (unbuilt here).
