# Phase 8 — Closing the Fully-Dynamic Gap

_Drafted 2026-06-02. Author: Phase 7 closing-audit follow-up.
Baseline: branch `speedup-cache-and-buckets`, HEAD `069f62c`
(after merge of strict-flip default-on, Phase 7.A.5 final)._

This plan extends `docs/PHASE_7_FULLY_DYNAMIC_PLAN.md` based on the
findings in `.agent-logs/phase_7_closing_audit.md`. Phase 7 took the
declarative-compilation work from "demo-able" to "majority of cells".
Phase 8 is the closing wave: take the remaining structural pins off,
delete the static-phase scheduler, and hit the headline KV target.

---

## Section 1 — Why Phase 8

### What Phase 7 achieved (closing-audit numbers)

* **Declarative share**: 68 / 83 authored ops (**82%**); 241,080 /
  306,932 weight cells (**78.5%**). From V1 → V2 census:
  imperative_heavy 36 → 12 (−24), imperative_medium 12 → 0 (−12),
  imperative_trivial 7 → 2 (−5).
* **IR coverage**: 72 / 116 ops carry IR (62%); of the 44 no-IR
  ops, 33 are `no_op` stubs.
* **Verifier drift**: `scan_static_claims.py` reports 50 / 50 ok,
  0 problems, 0 inert; FFN + attention sweeps report **0 real_bug**.
* **Allocator pin removal**: L0/L1/L2 (7.B.1), L3/L4/L5 (7.B.2),
  L12/L13/L14 (7.B.5), L10/L15/L16 (7.B.6), L9/L11 (7.B.4),
  L6/L7/L8 (7.B.3 partial) FFN pins dropped, all byte-identical.
* **Scheduler**: `compile_full_vm_dynamic` strict-mode admission with
  cycle-aware fallback landed (7.A.5) and is now the **default**
  (`b038fed`). `phase_inconsistent_with_deps` = 0;
  `phase_required_but_undeclared` = 1 (`layer14_demo_phase6_wave7`).
* **Cycle SCC**: 88 ops in one SCC of size 68 (down from worse, but
  far above the ≤10 target). Top back-edge dims: OUTPUT_LO (54),
  OUTPUT_HI_THIS_STEP (36), TEMP (26), ADDR_KEY (17), OUTPUT_HI (14).
* **Semantic dim refs**: `(category, role)` index landed (7.E.1);
  L8 ALU pilot migrated 6 substages (7.E.2). Corpus-wide
  `dim_ref(` adoption: **10 call sites / ≈5%**.
* **Model-level bakes**: `TokenEmbeddingRule` exists; `head_bake`,
  `embedding_bake`, `initial_pc_bake` migrated (7.D.2).
* **KV eviction**: `kv_liveness_analyzer` + `KVEvictionState` +
  `--kv-eviction-policy` flag + byte-identity gate + measurement
  harness all landed. Analyzer coverage **1.4062%**; realised
  drop **0.00%**; tensor-shape drop **0.00%**. Gap to target: full 30%.

### What's left

The closing audit's score table makes the gap explicit. Phase 7 closed
the easy 80% — Phase 8 has to close the structural 20%:

* OUTPUT_HI back-edges (36) + OUTPUT_LO residual back-edges (54)
  blocking SCC collapse — 7.A.3 alias attempt was reverted.
* Attention-head pins (~34 non-None pins) — `AttentionHeadAllocator`
  still couples slope/W_o mapping to `head_idx`.
* 12 remaining imperative_heavy ops (`l10_post_ops_combined`
  39,284 cells dominates).
* `dim_ref(category, role)` at 5% adoption corpus-wide.
* `compile_full_vm_dynamic` is now strict-default, but the
  `compile_full_vm` static-phase path still ships and is the
  fallback. Phase pruning still exists.
* KV eviction analyzer at 0% realised drop (blocked on cache
  compartmentalisation + cycle classifier).
* 1096 corpus root-cause work (SP_BYTE0 fix in flight; div outlier
  band + `var_*` category waiting).

---

## Section 2 — Goal: "fully dynamic" acceptance criteria

Phase 8 is **done** when all of the following hold simultaneously,
in one head-of-branch commit:

| Metric | Phase 7 close | Phase 8 target |
|--------|---------------|----------------|
| Declarative cells (`lower_*` paths) | 78.5% | **≥ 95%** |
| `imperative_heavy` ops (census v2) | 12 | **0** |
| `dep_graph_cycle_member` ops in SCC | 88 (1 SCC of 68) | **≤ 10** |
| KV peak memory reduction (`measure_kv_eviction`) | 0% | **≥ 30%** |
| Attention head pins (non-None `pin=`) | ~34 | **0** (alibi + W_o decoupled) |
| `dim_ref(category, role)` adoption | 5% (10 sites) | **≥ 50%** |
| Static `compile_full_vm` path | shipped + fallback | **deleted** |
| Phase-pruning scheduler branch | shipped | **deleted** |
| `phase_required_but_undeclared` | 1 | **0** |
| Verifier drift (`scan_static_claims`) | 0 / 50 | **0** (no regression) |
| FFN sweep `real_bug` | 0 / 116 | **0** (no regression) |
| Attention sweep `real_bug` | 0 / 86 | **0** (no regression) |

Plus a **closing demo** (sub-wave 8.H): one new corrective op authored
entirely in declarative IR (`FFNRule` + `DeclarativeAttentionHeadSpec`)
with **zero `pin=` arguments**, **zero literal `+N` dim offsets**, and
**zero `phase=` ordinal** — the scheduler must place it from
`requires=` alone. This is the executable proof of Goal 2
("fix-at-compiler-level").

---

## Section 3 — Wave structure

Twelve sub-waves grouped into four streams. Where the audit lists
work as "in progress", we capture the in-flight commit + remaining
acceptance criteria; otherwise we scope from scratch.

### Stream 1 — Scheduler / cycle decomposition

#### 8.A — Finish OUTPUT_LO + OUTPUT_HI decomposition (SCC ≤ 10)

_Status: in progress (7.A.3 OUTPUT_LO split landed in `cce4e9e`;
OUTPUT_HI split landed + reverted, see commits `e8eba9c` / `80c7288` /
`c13dae4`)._

* **8.A.1** Build `tools/analyze_output_hi_readers.py` per the audit's
  recommendation (mirror of the OUTPUT_LO followup tool). Catalogue
  every reader of `OUTPUT_HI_THIS_STEP` and classify each as
  step-local, prev-step, or genuinely cross-step.
* **8.A.2** Reintroduce `OUTPUT_HI_PREV_STEP` alias with per-reader
  audit-driven routing. The reverted attempt failed because alias
  rename broke downstream ops that didn't separate THIS_STEP from
  PREV_STEP; the new attempt must touch every reader explicitly.
* **8.A.3** Decompose TEMP back-edges (26): TEMP is multi-purpose
  scratch — split into `TEMP_THIS_STEP` vs `TEMP_CROSS_STEP` by
  reader.
* **8.A.4** Decompose ADDR_KEY (17), AX_CARRY_LO (12), ALU_LO (9),
  ADDR_B0_HI (5), AX_CARRY_HI (3) using the same per-reader pattern.
* **8.A.5** Backfill `requires=` for `layer14_demo_phase6_wave7`
  (drops `phase_required_but_undeclared` to 0).
* **Acceptance**: `analyze_scheduler.py` reports
  `dep_graph_cycle_member ≤ 10`; `phase_required_but_undeclared = 0`;
  byte-identical 1096 corpus and smoke tests.

### Stream 2 — Allocator / pin removal

#### 8.B — AttentionHeadAllocator decoupling + corpus-wide attn pin drop

_Status: **DONE (8.B.1, 8.B.2, 8.B.3)** as of overnight cycles. L9 attn
pin drop landed (`8cf75955`); L10/L13/L14/L15 remaining._

* **8.B.1** ✅ Add `dynamic_first_fit` mode to `AttentionHeadAllocator`
  (`c4efc27`).
* **8.B.2** ✅ Decouple `alibi_slope` + threshold `out_base` from
  `head_idx` (`e791fef`).
* **8.B.3** ✅ Drop `pin=` from L0/L1/L2 attention heads (`01088f1`).
* **8.B.4** Continue corpus-wide attn pin drops: L9 (done `8cf75955`);
  remaining L10/L13/L14/L15. Each layer: switch to `dynamic_first_fit`,
  verify byte-identity via W_q/W_k/W_v/W_o + alibi_slopes diff.
* **8.B.5** Sweep `grep -c "pin=" ops/` after all per-layer drops;
  document any residual exceptions.
* **Acceptance**: 0 non-None `pin=` in `ops/` head-allocator calls;
  byte-identical sweep on all 86 heads (still 0 real_bug).

#### 8.C — Migrate remaining 12 imperative_heavy ops

_Status: not started; targets enumerated in audit Section 2._

* **8.C.1** `l10_post_ops_combined` (39,284 cells) — the L10 carry
  propagation + comparison + binary-op-byte-zeroing migrations
  already exist individually; collapse the umbrella op into IR.
  Single largest unlock (60% of remaining imperative cells).
* **8.C.2** Census heuristic audit. `layer6_routing_ffn`,
  `embedding_bake`, `head_bake`, `initial_pc_bake` are flagged
  imperative because their `bake_fn` writes cells outside `lower_*`
  even though IR exists (audit §2 + §3). Either (a) update the
  census taxonomy or (b) remove the residual imperative branch.
  Decide per-op.
* **8.C.3** `l15_attention_resize` (1,973 cells) and
  `function_call_weights` (2,910 cells) — runtime-shape decisions.
  Extend `RuntimeAttentionFragment` to cover them or document as
  IR-exception with `kind="runtime_shape"`.
* **8.C.4** L10 passthrough_bakes (211 + 150 + 348 + 111 cells) and
  `conversational_io_output_routing` (128 cells) — small mop-up
  migrations using the L14 cleanup-chain pattern.
* **Acceptance**: census v2 reports `imperative_heavy = 0`;
  `declarative + via_helper ≥ 95%` of cells.

### Stream 3 — Semantic refs / static-path deletion

#### 8.D — Scale (category, role) migration to corpus-wide (>50%)

_Status: in progress; pilot L8 ALU done (7.E.2 merge `c3680e0`).
Audit Tier 4 enumerates the next set._

* **8.D.1** L9 ALU, L10 ALU, L11 mul_partial, L12 mul_combine,
  L13 shifts — apply the L8 ALU pattern. Per audit, these five ops
  account for ~140k cells; migrating them drops literal-offset count
  by ~70% in one go.
* **8.D.2** L3, L4, L5 FFN ops — second-tier migration.
* **8.D.3** Decide which `+N` offsets are genuinely structural
  (look-up table indices, e.g. opcode-decode column offsets) and
  retain them with an explicit `# structural offset` comment.
* **8.D.4** Add `decl_verifier.verify_dim_refs_resolved` check so a
  rule that references `dim_ref("ALU", "lo")` resolves to the same
  literal as the pre-migration `+12` and the verifier gates the
  rename.
* **Acceptance**: `grep -c "dim_ref(" ops/` ≥ 50% of total dim-ref
  call sites; verifier passes; byte-identical sweep.

#### 8.G — Delete `compile_full_vm` static path + phase pruning

_Status: in progress; strict-default flip landed (`b038fed`);
reroute landed (`e6da5f4` — `use_static_path=False` default);
KV eviction kwarg port done (in same merge)._

* **8.G.1** ✅ Port `kv_eviction_policy` + `kv_eviction_n_steps` kwargs
  into `compile_full_vm_dynamic` (was static-only blocker).
* **8.G.2** ✅ Reroute `compile_full_vm` to dynamic by default
  (`e6da5f4`). Static path preserved behind `use_static_path=True`
  flag with deletion TODO.
* **8.G.3** **Actually delete** the static-phase code path (~550 LOC
  in `full_vm_compiler.py` per `static_path_deletion_prep.md`). After
  8.A lands SCC ≤ 10 + 8.D corpus-wide dim_ref done. Audit confirms
  no production caller blocks deletion.
* **8.G.4** **Actually delete** phase-pruning code (~250-350 LOC in
  `_topological_sort`, `_assign_layers`, `_phase_key` of
  `layer_compiler.py`). Mirror in `compile_full_vm_dynamic`.
* **8.G.5** Delete `phase=` ordinal from op factories where the
  scheduler can derive it from `requires=`. Audit count: 282
  `phase=` references; target **0** (only for documented anchor ops
  where requires-only doesn't suffice). Per-layer batches in flight.
* **8.G.6** Delete `layer_idx=` from `ops/` calls where it's
  redundant. Audit count: 189; target **0** (allocator chooses).
  Per-layer batches in flight.
* **Acceptance**: `compile_full_vm` removed from public API; phase
  pruning code deleted; 0 `phase=` / 0 `layer_idx=` literals in ops;
  static test suite + 1096 corpus byte-identical under dynamic-only path.

### Stream 4 — KV eviction / corpus fixes / demo

#### 8.E — KV eviction completeness

_Status: in progress; 7.F.1/2/4 landed. **Blocked on 8.A** (cycle
decomp feeds the analyzer)._

**Goal (revised):** every KV entry that is provably dead at step S
(per the declarative IR's liveness analysis) is evicted no later
than step S. Not a memory-% target. Memory drop is the downstream
consequence.

* **8.E.1** Refine analyzer's cycle classifier: `_PREV_STEP` reads
  should not promote the base dim to cycle-conservative. (Today,
  any base-dim reader inside the SCC marks the dim as cycle-conservative,
  causing ~80 dim names to never be evictable.)
* **8.E.2** Per-(layer, head, dim-group) cache compartmentalisation.
  Today's `cached_k[B, H, S, HD]` packs every dim into one row, so
  the AND across dims keeps every row live even when most are dead.
  Split the cache into dim-groups so eviction applies per-group.
* **8.E.3** Loosen "semantic-overwrite" category from "every later
  step" to "very next step" — catches register carries like `REG_AX`
  automatically (when step N+1 overwrites it, step N's entry is dead).
* **8.E.4** Build the **oracle analyzer**: walk the declarative IR
  with no cycle conservatism; for each `(step, position, dim)` compute
  the actual last-reader step. The runtime analyzer's job is to match
  the oracle.
* **8.E.5** Build the **completeness gate**: after each step, run both
  runtime analyzer and oracle. Assert (a) every entry the oracle says
  is dead at step S has been evicted by step S (no late-evictions);
  (b) every entry the runtime evicted is also dead per the oracle (no
  false-positives that would break correctness).
* **8.E.6** Determinism gate: spec-decode and main-decode at the same
  step S evict identical entry sets.
* **8.E.7** Wire `test_kv_eviction.py` byte-identity gate + completeness
  gate into CI.
* **Acceptance**: completeness gate green on 1096 corpus sample
  (200 inputs): **0 late-evictions, 0 false-positives, determinism
  preserved**. Memory drop reported as a side effect, not gated.

#### 8.F — 1096 corpus targeted fixes (parallelizable)

_Status: in progress; SP_BYTE0 fix in flight per user-memory note
`project_l3_sp_byte0_dormant.md`. Subsequent priorities per the
root-cause diagnostic._

* **8.F.1** L3 / L7 SP_BYTE0 (real logic lives at `vm_step.py:4604+`,
  not the dormant `_rewrite_initial_sp_marker_to_f8`). See memory
  note.
* **8.F.2** div/mod outlier band — captured but not triaged at Phase 7
  close.
* **8.F.3** `var_*` category — `if_var` root cause flagged at
  `l16_bp_frame_byte1_ff` (memory note
  `project_l16_bp_frame_byte1_ff_dual.md`). Fix the negative-write
  side, not the L15 nibble_copy aggressor side.
* **8.F.4** L10 PSH `addr0_e0 missing OP_ENT guard` (open bug at
  `l10_ops.py:3888-3927`; blocks func_identity_*; memory note
  `project_l10_psh_addr_ent_bug.md`).
* **Acceptance**: 1096 corpus pass-rate strictly improves; no new
  sentinel regressions under both declarations-only + full-flag
  modes.

#### 8.I — Final 100% verification pass

_Status: not started; closing audit re-run._

* **8.I.1** Run all sweeps (FFN, attn, claims, census v2, scheduler,
  KV eviction quantification) and confirm 100% of `Section 2` table.
* **8.I.2** Run full 1096 corpus + spec-decode + smoke; net delta
  vs Phase 7 baseline must be ≥ 0 with no real_bug regressions.
* **8.I.3** Write `c4_release/.agent-logs/phase_8_closing_audit.md`
  mirroring the Phase 7 closing audit format.
* **8.I.4** Map each metric to the user's original five vision components
  (see Section 7 below) and confirm 100% adoption — this is the
  acceptance forcing-function.
* **Acceptance**: all 12 metrics in Section 2 hit 100% or "deleted";
  closing audit is committed.

#### 8.H — Phase 9 demo (zero-pin declarative op)

_Status: not started; closing demo._

* **8.H.1** Pick a corrective op need that surfaced from 8.F
  (e.g. a small `var_*` fix). Implement it end-to-end:
  * 100% declarative IR (`FFNRule` + `DeclarativeAttentionHeadSpec`).
  * `pin=None` on every allocator call.
  * `dim_ref(category, role)` for every dim ref (no `+N` offsets).
  * No `phase=` ordinal; `requires=` only.
  * Byte-identity gate via `compare_symbolic_to_lowered_*`.
* **8.H.2** Document the work in
  `docs/HOW_TO_ADD_A_CORRECTIVE_OP.md` v2 — replace the 8-step
  recipe with the pin-free version.
* **Acceptance**: the new op compiles under
  `compile_full_vm_dynamic`-only, with zero pins / zero literal
  offsets / zero phase ordinal; verifier green; 1096 corpus
  byte-identical baseline + measurable improvement on the targeted
  sentinel.

---

## Section 4 — Dependency graph

```
                    7.F.5 (analyzer per-dim split, audit Tier 1.1)
                    7.F.6 (cache compartmentalisation, audit Tier 1.2)
                              \
                               v
8.A (SCC ≤ 10)  ---------->  8.E (KV ≥ 30%)
   |                            |
   |                            v
   |                          7.F.3 CI gate
   v
8.D (dim_ref ≥ 50%) <-- requires 8.A for unambiguous category mapping
   |
   v
8.G.2 (delete static path) <-- requires 8.A (SCC ≤ 10) + 8.D + 8.G.1

8.B (attn allocator decouple) ---> all-attn-pin-removal (8.B.3, 8.B.4)

8.C (imperative_heavy → 0) -- mostly independent; 8.C.1 (L10) can
                              run in parallel with 8.A.

8.F (1096 fixes) -- parallel to everything; pulls in 8.D, 8.H downstream.

8.H (demo) -- gated on 8.B (zero pins), 8.D (dim_ref), 8.G.3 (no phase=).
              Final wave; integrates the rest.
```

Critical path: **8.A → 8.E → 8.G.2** drives the schedule. 8.B, 8.C,
8.D, 8.F can run in parallel agent streams.

---

## Section 5 — Risk register

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|------------|--------|------------|
| R1 | OUTPUT_HI re-decomposition (8.A.2) re-breaks downstream readers like the reverted Phase 7.A.3 attempts | HIGH | HIGH | Land 8.A.1 (per-reader analyzer) **first**; require zero rename-failures in the analyzer output before touching code. Single-rule whack-a-mole is documented zero-sum (memory note). |
| R2 | AttentionHeadAllocator slope/W_o decoupling (8.B.1) is load-bearing for byte-identity — primitives mismatch silently | HIGH | HIGH | Gate every change with `compare_symbolic_to_lowered_attn`. `e791fef` already cut the first half; finish under the same byte-identity gate. |
| R3 | 8.E.2 cache compartmentalisation changes runtime layout — kernel/CUDA-graph regression | MED | HIGH | Keep the compartment count fixed at compile time; benchmark against `CUDA_GRAPHS_MULTI_STEP.md` baseline before merging. |
| R4 | KV target 30% is analyzer-limited and not achievable even with full SCC collapse | MED | HIGH | Quantify the analyzer ceiling **before** 8.E.2: build a "perfect oracle" version that assumes zero cycle conservatism; if its dividend is < 30%, raise the cache-compartment count or scope back the target. |
| R5 | Census heuristic in 8.C.2 mis-classifies migrated ops as imperative — wasted effort chasing 0 imperative_heavy when 12 already moved | MED | MED | Land the census taxonomy update **before** counting cells in 8.C acceptance. |
| R6 | Deleting `compile_full_vm` (8.G.2) breaks an external caller (tests, benchmark harnesses) | MED | MED | Run the full test suite + 1096 corpus before deletion; keep one release with a deprecation shim. |
| R7 | 1096 fixes (8.F) regress under one flag mode but improve under another (sentinel baseline depends on env flags, memory note) | HIGH | MED | Quote both `declarations-only` and full-flag sentinel counts in every 8.F brief; never quote historical numbers (per memory note). |
| R8 | 8.D corpus-wide `dim_ref(` migration breaks structural look-up offsets (genuine `+N` indices into tables) | LOW | MED | 8.D.3 explicit decision step; 8.D.4 verifier check forces every rename to round-trip. |
| R9 | Attention-head pin removal (8.B.3, 8.B.4) interacts with KV eviction (8.E.2) — both touch the head-dim mapping | MED | LOW | Sequence: land 8.B fully before 8.E.2 cache compartmentalisation; reorder if needed. |
| R10 | Final demo (8.H) reveals a structural pin or `phase=` that no wave caught | LOW | LOW | Use the demo as the acceptance forcing-function; pad estimate by 1 wave for late discovery. |

---

## Section 6 — Total estimate

* **Sub-waves**: 8 (8.A, 8.B, 8.C, 8.D, 8.E, 8.F, 8.G, 8.H), each with
  3–5 numbered tasks → **30 tasks** total.
* **Agent estimate**: ~22 single-task agents + ~6 audit / verifier
  agents + 1 closing-demo agent = **~30 agent runs**, in 5–6
  parallelizable batches.
* **Wall clock** (assuming 4 parallel agent streams matching the
  Phase 7 cadence of 65 commits / ~3 weeks):
  * 8.A: 5–7 days (longest single-stream; per-reader audit + 4
    decomposition steps).
  * 8.B: 3–4 days.
  * 8.C: 4–5 days (8.C.1 dominates).
  * 8.D: 5–7 days (corpus-wide rename).
  * 8.E: 3–4 days **after** 8.A unblocks.
  * 8.F: ongoing in parallel.
  * 8.G: 2 days **after** 8.A + 8.D.
  * 8.H: 2 days **after** 8.B + 8.D + 8.G.3.
  * **Total wall clock**: ~3 weeks (longest critical path
    8.A → 8.E → 8.G.2 → 8.H is ~14 days; rest parallelises behind it).

This matches the audit's expectation that Phase 8 is the "closing
20%": Phase 7 took ~3 weeks for 65 commits across 5 sub-waves;
Phase 8 should land ~80 commits across 8 sub-waves in a similar
window.

---

## Section 7 — Explicit mapping to the user's original vision

The user's 5-component vision (from the original conversation):

> "Once we have things fully dynamic there will be no more
> imperative setting and then all fixes will be fixes on the level
> of the compiler and no non-io dims will be hardcoded and no layer
> indices will be hardcoded and the weights will be just outputed
> via compilation of the declarative spec."

This decomposes to **5 explicit goals**, each mapped to Phase 8 work:

| # | Vision goal | Concrete target | Owned by sub-wave(s) | 100% condition |
|---|---|---|---|---|
| V1 | "no more imperative setting" | 100% declarative cells; 0 imperative_heavy ops | 8.C | census v2 reports `imperative_heavy=0` AND `declarative+via_helper=100% of cells` |
| V2 | "all fixes will be fixes on the level of the compiler" | Every new corrective op authored in IR; no bake_fn surgery | 8.H (forcing function); 8.D (dim refs) | 8.H demo op exists with zero pins / zero literal offsets / zero phase ordinal |
| V3 | "no non-io dims will be hardcoded" | 100% `dim_ref(category, role)` adoption where applicable; structural lookups documented | 8.D | `grep -c "dim_ref(" ops/` ≥ all role-meaningful sites; structural exceptions explicitly tagged |
| V4 | "no layer indices will be hardcoded" | 0 `layer_idx=` literals; 0 `phase=` literals (compiler derives both) | 8.G.5, 8.G.6 + 8.A.5 backfill | `grep -c "layer_idx=" ops/` = 0 AND `grep -c "phase=" ops/` = 0 |
| V5 | "weights just output via compilation of the declarative spec" | Single compile path; static path + phase pruning deleted | 8.G.3, 8.G.4 | `compile_full_vm` symbol removed; `_topological_sort` phase-pruning branch removed |

### Additional non-vision goals (Phase 8 also closes these)

| Goal | Sub-wave | 100% condition |
|---|---|---|
| KV eviction completeness | 8.E | **Every KV entry that is provably dead (per the declarative IR's liveness analysis) is evicted no later than the step its last reader runs.** Not a memory-% target — a correctness target: 0 late-evictions, 0 false-positives (false-lives never evicted), determinism between spec-decode and main-decode. The memory drop is the downstream consequence, not the metric. |
| Cycle graph collapse | 8.A | `dep_graph_cycle_member ≤ 10` |
| 1096 corpus pass rate | 8.F | strict improvement vs Phase 7 baseline; no real_bug regressions |
| Closing audit | 8.I | `phase_8_closing_audit.md` written; all metrics confirmed |

**Phase 8 is "done" when all 5 vision goals + 4 non-vision goals
land in one head-of-branch commit, attested by 8.I.**

---

_End of plan._
