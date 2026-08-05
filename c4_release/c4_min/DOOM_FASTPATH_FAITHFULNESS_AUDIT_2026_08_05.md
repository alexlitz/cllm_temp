# Doom-on-transformer FAST PATH — faithfulness audit (2026-08-05)

Auditor: verifies whether the fast path is a *real autoregressive transformer with
speculative decoding* or has degenerated into a *rubber-stamp for the draft*.
Read-only + light in-process experiments (all `_agent_*` scratch removed). Golden
`069cc32f` untouched — every lever audited is DEFAULT-OFF.

Repo: `c4_release/c4_min/`. Core files read: `pf_speculative.py`,
`direct_cam_batched.py`, `direct_local_cam.py`, `nibble_evict_schedule.py`,
`local_attention.py`, `sparse_forward.py`, `fused_sparse_ffn.py`,
`fused_megablock.py`, `precomputed_schedule.py`, `block_moe_divmod.py`,
`selfemu_direct_cam.py`, `tight_attn_compose.py`, `REALTIME_SETUP.md`.

---

## Q1 — Optimization inventory, classified by faithfulness

**(A) = faithful pure-forward** (runs the real transformer math, byte-identical
because it *is* the same computation on the nonzeros). **(B) = draft-assisted
shortcut** (replaces part of the forward with a value the *draft* already resolved;
byte-identical in RESULT only because the draft is a perfect trace — NOT
independently recomputed by the model).

| Optimization (flag) | Class | What it replaces | Byte-exactness guarantee |
|---|---|---|---|
| Conditional / tensor sparsity (`sparse_forward.py` CSR/`materialize_dense`) | **A** | dense `A@x` over 99.998%-zero weights | L∞=0: sparse `A@x` == dense on nonzeros; zeros contribute 0 either way (`sparse_forward.py:16-18`) |
| Flash attention / softmax1 BOS-sink (`C4_FLASH_ATTN`, `flash_softmax1`) | **A** | the `[H,Sq,Sk]` masked-softmax1 score matrix | tiled online-softmax1 over a sink column == plain softmax1; fp32 ~1e-6 < nibble margin (`local_attention.py:225-244`) |
| Banded / sliding-window local attn (`C4_BANDED_LOCAL_ATTN`, window=30) | **A** | full `Q@Kᵀ` for the ingest heads | out-of-band weight is provably 0 (ALiBi recency + huge match logit → ZFOD); never scored (`local_attention.py:19-24, 211-224`) |
| Dead-block fusion + per-row block-skip (`C4_DEAD_BLOCK_FUSION`) | **A** | attention sublayer of the ~238 zero-value blocks | dead block's `W_v==W_o==0` ⇒ attn output ≡ `x` regardless of keys (`pf_speculative.py:876-889`) |
| Fused-delta / fused-hidden FFN (`C4_FUSED_DELTA_FFN`; = the "`FFN_FUSED_HIDDEN`" up+gate+silu-once) | **A** | 3 SpMM launches + silu*mul glue per block | same nonzeros/accum order; hidden written once; only fp-accum residue (`fused_sparse_ffn.py:435-456`) |
| Fused megablock / CUDA-graph dead-FFN chain (`C4_FUSED_MEGABLOCK`, `C4_GRAPH_BLOCK0`, whole-step graph) | **A** | per-block launches + HBM residual round-trips | same nonzeros, same per-output column-sorted accum order; attn identity on dead blocks; residual resident in L2 (`fused_megablock.py:31-38`) |
| KV eviction / pruning (`C4_EXACT_EVICT`, `C4_EVICT_SCHEDULE`, content prune) | **A** | O(S²) content-comparison prune | drops only rows no read ever resolves to (latest-write-wins liveness); `n_read_after_free==0` audited (`pf_speculative.py:2305-2315`) |
| CUDA-graph / single-dispatch (`C4_PRECOMPUTED_SCHEDULE`) | **A (compute) + B (routing/gathers)** | per-op CPython dispatch | replays the SAME kernel stream; **but the routing set + every gather it feeds are draft-resolved** (see below) (`precomputed_schedule.py:61-74`) |
| **Direct global CAM (`C4_DIRECT_CAM_BATCHED` / `C4_DIRECT_CAM_VEC`)** | **B** | the LI/LC/pop/LEV/code-fetch global heads' O(S) softmax1 score | head output reconstructed from `resolve_load_rows(draft)` — the draft's addr→value; the model's `W_q` query for those heads is computed then **discarded** (`direct_cam_batched.py:24-43, 372, 484`) |
| **Direct local CAM (`C4_DIRECT_LOCAL_CAM`)** | **B** | block-0's ~20 register-ingest heads' windowed softmax | head output = nibbles decoded straight from `draft.tokens` (`build_resolved_frames`); no scoring (`direct_local_cam.py:16-31, 249-250`) |
| Precomputed-schedule ROUTING (`build_live_index` / `divfree_carry_blocks`) | **B** (benign superset) | per-step model-decoded expert/block routing | uses `draft.frames[s]["op"]` to pick the DIV-free carry; asserts DIV-free and falls back otherwise; the carry is a *static superset* of live blocks so it is a valid route, but the routing DECISION is the draft's opcode, not the model's decode (`pf_speculative.py:1544-1548`, `precomputed_schedule.py:1232-1240`) |
| block-MoE divmod-skip in `verify_blocks` (`block_moe=True`) | **B** | running the ~262-block divmod span every step | skip decided by `draft.frames[s]["op"] in {DIV,MOD}` (`pf_speculative.py:1544-1548`) |
| DivModRouter (`block_moe_divmod.DivModRouter`) — the *standalone* MoE forward | **A** | — | reads the model's OWN `OP_IS[DIV]/[MOD]` guard lanes off the residual, thresholds+ORs (`block_moe_divmod.py:130-138`). This is the model-decoded route; **it is NOT the one `verify_blocks` uses.** |

---

## Q2 — THE DECIDING EXPERIMENT: detect divergence, or rubber-stamp?

Method: built the lean streaming model once (GPU/CPU, ~2-4 GB, memory-safe), a tiny
fully-verifying program `IMM 200; PSH; IMM 66; SI (mem[200]=66); IMM 200; LI; HALT`
(clean draft → **7/7 accepted, final_ax=66**), then corrupted one quantity at a time
and re-ran `verify_blocks`.

| Experiment | Path | Result | Verdict |
|---|---|---|---|
| **Wrong register TOKEN** (flip an AX byte token in a frame) | softmax | mismatch at the ingest step: `got_ax=196693 want_ax=196608` (also reproduced on the store/load prog: `got_ax=99 want_ax=66`) | **INDEPENDENTLY VERIFIED — caught** |
| **Wrong store VALUE in KV**, frames clean | softmax | mismatch at LI step: `got_ax=99 want_ax=66` (model gathered the corrupted KV) | **INDEPENDENTLY VERIFIED — caught** |
| **Wrong store ADDRESS** (200→201), value clean, frames clean | softmax | mismatch at LI step: `got_ax=0 want_ax=66` (model's query for 200 finds no store → softmax1 sink → ZFOD 0) | **INDEPENDENTLY VERIFIED — caught (the model's OWN query resolves the address)** |
| **Wrong store VALUE injected by direct-CAM**, frames clean | direct-CAM | mismatch at LI step: `got_ax=99 want_ax=66` (injected value still compared vs the clean frame) | caught *only because the FRAME target was clean* |
| **Wrong value + self-consistent draft** (store_log=99, frames=99, tokens=99) | direct-CAM | **7/7 accepted, all_matched=True, final_ax=99** — a WRONG answer ACCEPTED | **DRAFT-TRUSTED — rubber-stamped** |

### One-line verdict per quantity
- **Token: INDEPENDENTLY VERIFIED.** The block-0 register-ingest reads the emitted
  token frame and the decode compares it — a corrupted register token is always caught
  (softmax path). Under `C4_DIRECT_LOCAL_CAM` the ingest is *also* draft-injected, so a
  self-consistent token corruption would rubber-stamp identically to the value case.
- **Address/value: DRAFT-TRUSTED on the direct-CAM fast path.** The read's
  address→row→value is resolved entirely by `resolve_load_rows(draft)` and injected;
  the model's `W_q` query for the CAM heads is computed then **overwritten**
  (`direct_cam_batched.py:372` builds Q, `:484`/`:500` overwrite the CAM heads' output).
  The ACCEPT compares the model's (injected) output against the *draft's own* frame —
  both derived from the same draft — so a self-consistent bad read is accepted (scenario
  E). On the **softmax path (direct-CAM OFF) the address IS independently recomputed**
  by the model's query (scenarios "wrong-KV-value" and "wrong-address" both caught).
- **Routing: DRAFT-TRUSTED on the fast path** (`draft.frames[s]["op"]` selects the
  DIV-free carry / divmod-skip), but benign: the carry is a static *superset* of live
  blocks so it can only over-run, and DIV/MOD steps assert-fallback to the full stack. A
  genuinely model-decoded router (`DivModRouter`) exists and is byte-exact but is **not
  wired into `verify_blocks`**.

### The crux
The fast path verifies **the model's decoded register file (PC/AX/SP/BP) against the
draft's declared register file, first-divergence stop**. It does **NOT** verify that
the draft's *memory-read resolution* (which store row an address selects) is what the
model's own query would select — because with direct-CAM ON the model's query is never
used for those heads. So the fast path is:

> **Real verification of the REGISTER TRANSITION and TOKEN INGEST; draft-TRUSTED for the
> global-CAM MEMORY-READ RESOLUTION (address→value) and for block-level ROUTING.**

It is *not* a blanket rubber-stamp (register/token corruption is caught), but it is
*not* a fully independent recompute either: the byte-exactness of the memory reads rests
on `resolve_load_rows` being a proven-exact stand-in for the softmax1+ALiBi winner
(latest-write-wins == the CAM winner), which is a *mathematical identity argument*, not a
runtime check. If that identity ever failed to hold (e.g. an address the CAM would alias
differently than latest-write-wins), the fast path would silently accept the draft's
resolution. The softmax path (all draft-assist OFF) has no such gap.

---

## Q3 — Autoregressive + legitimately speculative?

- **Autoregressive / causal-prefix accept: YES.** Acceptance is decided in
  `verify_blocks` (scalar loop `pf_speculative.py:2117-2190`; GPU-verify
  `:1957-2075`; precomputed-schedule `precomputed_schedule.py:1281-1289`). All three
  decode the model's registers per query row and **stop at the first divergence**
  (`first_bad = argmax(bad)`, `n_ok = first_bad if any_bad else K`). The parallel
  single-dispatch verifies K positions at once — this is standard spec-decoding *parallel
  verification*; the ACCEPT is still a causal prefix (accepted count = first-divergence
  index), NOT assume-all-accept. Confirmed experimentally: even the CLEAN draft on the
  malloc program stopped at step 16 (16/60) rather than blanket-accepting.
- **Legit speculation: YES, real compare.** The compare is an integer argmax requant
  decode of the model's hidden state vs the draft targets
  (`got_pc != want_pc | (got_ax&mask) != want_ax | got_sp!=.. | got_bp!=..`), file rows
  a no-op, halt rows AX-only. On mismatch it returns a `first_mismatch` FAIL — the driver
  reports "model argmax != draft" (`pf_speculative.py:2441-2445`). There is NO
  "assume-accept-because-draft-is-byte-exact" branch. (Note: it does not *re-draft* the
  diverged step from the true forward — a mismatch is a terminal FAIL for that program,
  not a per-step fallback. That is fine for the perfect-draft use case but means it is a
  *verifier*, not a *repairing* speculative decoder.)
- **Faithful path exists + is byte-exact: YES.** `run_pure_forward_cached` is the
  token-by-token autoregressive KV-cached driver; `spotcheck_vs_cached`
  (`pf_speculative.py:2463-2513`) proves the speculative verify is byte-identical to it,
  AND the fast (`fast=True`) overlay is byte-identical to the slow overlay
  (`fast=False`). Experimentally confirmed both softmax and direct-CAM paths
  `all_matched=True` on clean drafts (7/7 and 250/250).

### Faithful vs fast tradeoff (measured)
- **Fast composed path: 838 µs/step** measured (`REALTIME_SETUP.md`,
  `_agent_composed_floor.py`, block-0-dominated) at doom scale. FLOP floor 0.0071
  µs/step; estimated ~91 µs/step once block-0 is graphed.
- **Faithful un-composed path (all draft-assist OFF, full masked-softmax1 on all heads,
  full-H growing KV, dense-weight materialize per block):** its per-step cost is the
  O(S) global score on ~250 blocks × full-H + per-block `.to_dense()` — it OOMs on a 24
  GB card at even a few hundred doom-scale steps (reproduced: dense `W_q` materialize
  OOM). On a tiny working-set loop (byte-safe, ~7 rows) BOTH paths measured identical
  (~57 ms/step here) because at small S there is no O(S) score to remove and both are
  host-dispatch-bound — i.e. **the draft-assist speedup is entirely a large-KV
  phenomenon** (doom's 150k–262k persistent-heap rows). The honest statement: the fast
  path buys tractability at doom scale (the faithful path does not fit / does not finish);
  at small scale it is neutral, and both are byte-exact.

---

## Q4 — Self-emulation applicability

The self-emulation = the transformer emulating its own FP-matmul forward as a **c4
program** run by the C ONNX runtime on the native c4 VM. Cost is dominated by MAC count
(~503,776 dense MACs → ~8,600 sparse, ~58× MAC reduction) at a flat ~94–101 draft-VM
steps/MAC; it emulates FP matmuls, ~262k persistent-heap KV/step working set; launch-
bound (49 ms/step → 7.2 ms via block-fusion); composed self-forward ≈ **18 min**.

| Fast-path lever | Transfers to self-emu? | Why / expected benefit |
|---|---|---|
| **Direct-CAM O(1) memory read** | **YES — already ported (#831)** | `selfemu_direct_cam.py` wires `resolve_load_rows` into `pos_sparse_bounded.BoundedBlock` / `pf_kbatch.KBatchBoundedBlock`; the emulated VM writes memory heavily so `n_store` grows O(S) — direct-gather kills that growing softmax (up to ~9× on the doom self-emu measurement; same B-class draft-trust caveat as above). |
| **Precomputed schedule / single-dispatch** | **PARTIAL** | The self-emu's block structure is the emulated-matmul node graph, which is FP and per-node data-dependent (not a fixed DIV-free carry). The *routing* superset trick doesn't map (there's no divmod-span to statically skip); the *host-dispatch collapse* (one graph-per-chunk replay) DOES map to the emulated node loop and is the natural next lever, but is unbuilt. |
| **`FFN_FUSED_HIDDEN` / fused-delta FFN** | **YES in spirit** | The emulated matmul nodes are the analog of the FFN GEMMs; fusing up/gate/silu-once and collapsing SpMM launches transfers directly to the sparse-COO emulated forward — this is exactly the block-fusion that already cut 49 ms → 7.2 ms/step (the launch-bound → compute-bound move). Faithful (A-class), no draft-trust. |
| **Pipeline / double-buffer overlap** | **YES** | Device/host overlap is orthogonal to what's being emulated; applies to the emulated node stream the same way. Modest (hides host dispatch behind device). |
| **Flash / banded / dead-block / sparse tensors** | **YES (A-class)** | These are generic pure-forward levers on ANY sparse transformer forward; they apply to the self-emulated forward's own attention/FFN nodes wherever it materializes them. |

**Honest expected benefit for self-emu:** the composed self-forward is ~18 min today.
The direct-CAM port (#831) already removed the growing-store softmax. The next real win
is the **launch-collapse / whole-step-graph over the emulated node loop** (the same
host-dispatch wall that dominates doom) — this is the block-fusion 49→7.2 ms class of
speedup, plausibly another several-× on the launch-bound portion, but it is **unbuilt**
and the working-set (~262k KV/step, FP matmul emulation) means the self-emu will remain
orders of magnitude slower than the native doom fast path — it emulates the *arithmetic*,
so it can never beat the FLOP floor of the thing it emulates. Estimate: single-digit-×
achievable via the transferable A-class levers + the already-landed direct-CAM; **not**
real-time, and the B-class draft-trust caveat carries over (a self-emu direct-CAM read is
draft-trusted the same way).

---

## Bottom line
- The fast path is a **genuine autoregressive, first-divergence-prefix, real-compare
  speculative VERIFIER** — not a blanket rubber-stamp. Register-transition and
  token-ingest corruption are caught (proven).
- BUT with the direct-CAM levers ON, the **memory-read address resolution and the
  block routing are DRAFT-TRUSTED, not independently recomputed** — a *self-consistent*
  wrong memory read is accepted (proven: scenario E, 7/7 accept of a wrong value). The
  byte-exactness there is an *identity argument* (latest-write-wins == CAM winner), not a
  runtime check.
- The **fully faithful path exists and is byte-exact** (softmax, model-decoded address
  via its own query; `run_pure_forward_cached` reference; `spotcheck_vs_cached` gate),
  and on that path a wrong address/value IS caught. It is the slower, memory-bound
  baseline the whole composed stack optimizes away (fast composed = 838 µs/step measured;
  faithful un-composed OOMs at doom scale).
- Thesis assessment: "it's a vanilla autoregressive transformer" is **true of the
  faithful path and of the register/token verification on the fast path**; it is
  **overstated for the fast path's memory-reads and routing**, which trust the draft's
  resolution under a proven-but-unchecked identity. An honest statement is: *"a vanilla
  autoregressive transformer, verified in parallel by speculative decoding, with the
  memory-CAM reads and divmod routing resolved by the perfect draft under a
  latest-write-wins identity (default-OFF; the fully-scored softmax path is byte-exact
  and independently resolves them)."*
