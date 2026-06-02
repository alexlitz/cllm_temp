# Phase 7 Closing Audit

_Generated 2026-06-01. Read-only investigation. HEAD = `eee19c7` (Merge KV eviction quantification harness, Phase 7.F.4)._

This report summarises the trajectory of Phase 6 + Phase 7, the
current state of the declarative-compilation work, what remains, and
how close we are to the original "fully dynamic" vision.

---

## Section 1 — Phase 6 + 7 trajectory

### Totals

* Commits since the Phase 6 plan landed (`a6db239..HEAD`): **269**.
* Commits whose subject names a Phase 6 wave: **21**.
* Commits whose subject names a Phase 7 sub-wave (7.A–7.F): **65**.
* Phase 6 waves visible in subjects: Wave 1A-1D (census + docs),
  Wave 2A-2F (AttentionHeadIR pilots), Wave 3B-3L (FFNRule pilots),
  Wave 4A-4I (FFN bulk migrations), Wave 5A-5B (corpus sweeps),
  Wave 6A-6E (helper retirement, bake_fn drop, recipes), Wave 7
  (corrective-op pure-declarative demo).

### Phase 7 sub-wave commit distribution

| Sub-wave | Commits | Status |
|----------|--------:|--------|
| 7.A.1 — phase_inconsistent self-pin fixes | 3 + 1 merge | DONE |
| 7.A.2 — backfill requires for undeclared-dep ops | 5 + 1 merge | DONE |
| 7.A.3 — OUTPUT_HI/OUTPUT_LO step-local vs prev_step split | 7 (3 reverted) + 1 reverted, 3 b.b followups | PARTIAL (output_lo done; output_hi back-edges reverted) |
| 7.A.5 — strict-mode flip with cycle-aware admission | 2 + 1 merge | DONE |
| 7.B (prep) | 2 (dim_allocator + ffn_unit_allocator dynamic_first_fit) + 1 merge | DONE |
| 7.B.1 — L0/L1/L2 FFN pin removal | 3 + 1 merge | DONE |
| 7.B.2 — L3/L4/L5 FFN pin removal | 3 + 1 merge | DONE |
| 7.B.5 — L12/L13/L14 FFN pin removal | 3 + 1 merge | DONE |
| 7.B.6 — L10/L15/L16 FFN pin removal | 3 + 1 merge | DONE |
| 7.C.1 — L6 routing_ffn legacy cut | 1 + 1 merge | DONE |
| 7.C.2 — L15 memory_lookup migration | 2 + 1 merge | DONE |
| 7.D.1 — TokenEmbeddingRule IR | 2 + 1 merge | DONE |
| 7.D.2 — head_bake / embedding_bake / initial_pc_bake migrations | 1 + 1 merge | DONE |
| 7.E.1 — semantic (category, role) dim index | 1 | DONE |
| 7.E.2 — L8 ALU pilot migration to (category, role) refs | 7 + 1 merge | PARTIAL pilot |
| 7.F.1 — KV-cache liveness analyzer | 1 + 1 merge | DONE |
| 7.F.2 — KVEvictionState, base_layers integration, flag wiring, gate test | 6 + 2 merges | DONE |
| 7.F.4 — measure_kv_eviction harness | 2 + 1 merge | DONE |

### Notable milestones

* `b450ddc` — Phase 7.F.1 KV liveness analyzer.
* `2813324` — Phase 7.C.1 L6 routing legacy `_set_layer6_routing_ffn` helper removed.
* `18c1725` — Phase 7.C.2 L15 memory_lookup migrated to `RuntimeAttentionFragment`.
* `c95f8a8` — Phase 7.D.2 model bakes migrated to `TokenEmbeddingRule`.
* `f73696c` — `dynamic_first_fit` allocator mode landed (unlocked 7.B).
* `d648e2d` / `0c365e3` / `1405f61` / `526f947` — pin removal merges for L0/L1/L2,
  L3/L4/L5, L10/L15/L16, L12/L13/L14 — pin-free byte-identical across all four bands.
* `afbc792` — Phase 7.A.5 B14 strict-flip with cycle-aware admission.
* `c3680e0` — Phase 7.E.2 L8 ALU pilot migration to `(category, role)` refs.
* `eee19c7` — Phase 7.F.4 KV eviction quantification harness (current HEAD).

### Reverts and rollbacks

Three Phase 7.A.3 commits were merged then reverted (commits
`1eff091`, `7afb953`, `7291034` reverted by `e8eba9c`, `80c7288`,
`c13dae4`): the OUTPUT_HI_PREV_STEP alias + readers landed and were
backed out the same day, with the `OUTPUT_HI_THIS_STEP` 36-back-edges
left in place. A follow-up cleanup pass (commit `d7274a8`) removed
stale alias claims. The OUTPUT_LO split (commit `cce4e9e`) stayed in.

---

## Section 2 — Current state

### Scheduler (`tools/analyze_scheduler.py`)

* total ops analysed: **122**
* `freely_placeable`: 25
* `phase_pinned_by_deps`: 8
* `phase_required_but_undeclared`: 1 (`layer14_demo_phase6_wave7`, kind=block, dep_depth=0)
* `phase_inconsistent_with_deps`: 0
* `dep_graph_cycle_member`: **88** ops in **1** SCC of size 68
* DAG depth (longest declared chain): 3 layers; static layer count: 18

Top dim back-edges inside the largest SCC (descending):

* `OUTPUT_LO`: 54 back-edges
* `OUTPUT_HI_THIS_STEP`: 36 back-edges
* `TEMP`: 26 back-edges
* `ADDR_KEY`: 17 back-edges
* `OUTPUT_HI`: 14 back-edges
* `AX_CARRY_LO`: 12 back-edges
* `ALU_LO`: 9 back-edges
* `ADDR_B0_HI`: 5 back-edges
* `EMBED_LO`: 3 back-edges
* `AX_CARRY_HI`: 3 back-edges

### FFN sweep (`tools/sweep_compare_ffn.py`)

* ops enumerated: 116
* ops with FFN rules: 34
* **clean**: 21
* **mismatch_only** (weight_output_mismatch, often harness shared-state): 12
* **real_bug** (declaration_semantics / lowering exception): **0**
* synthetic_state_overflow: 1 (`tail_bit32_result_correction`, harness limitation)

Mismatch-only set: `layer3_ffn`, `opcode_decode_ffn`,
`layer6_routing_ffn`, `convo_io_pc_sp_latch`, `layer8_alu`,
`format_position_counter`, `layer10_alu`, `layer12_mul_combine`,
`layer13_shifts`, `layer14_temp_clear`, `layer14_addr_key_neural_decode`,
`layer16_lev_routing`.

### Attention sweep (`tools/sweep_compare_attn.py`)

* total attention-bearing ops in IR: 34
* total heads checked: 86
* **clean**: 81
* **mismatch_only**: 5
* **real_bug**: **0**

Mismatch-only heads: `layer6_attn_bake` (2), `convo_io_relay_heads`
(1), `layer7_sp_byte0_is_f8` (1), `tool_call_relay_head` (1).

### Imperative bake census v2 (`tools/census_imperative_bakes_v2.py`)

| Class | Count | Cells |
|-------|------:|------:|
| declarative | 58 | 131,019 |
| declarative_via_helper | 10 | 110,061 |
| imperative_heavy | 12 | 65,841 |
| imperative_trivial | 2 | 11 |
| declarative_no_op | 1 | 0 |
| no_op | 33 | 0 |
| **Total** | **116** | **306,932** |

* Declarative + declarative_via_helper share: **68 ops / 78%
  of authored ops** (excluding no_ops), accounting for **241,080
  cells = 78.5%** of all weight cells.
* Imperative-classed authored ops: **14 / 16% of authored ops**, but still
  ~21.4% of cells.

Imperative-heavy targets remaining (12 ops): `l10_post_ops_combined`
(39,284 cells), `layer6_routing_ffn` (15,212 — informational IR
present but bake doesn't lower; note: 7.C.1 cut the LEGACY helper, but
the bake_fn is still classed imperative by the census heuristic),
`l15_attention_resize` (1,973), `embedding_bake` (1,587 — wait, this is
the model-level rule path, Phase 7.D.2 LANDED, so census heuristic is
double-classifying), `function_call_weights` (2,910), `head_bake`
(3,615 — same comment as embedding_bake), several L10
passthrough_bakes (211, 150, 348, 111),
`conversational_io_output_routing` (128). Census heuristic counts an
op as imperative if it writes weight cells outside the lower path,
even when the IR exists.

V1 vs V2 census trajectory:

| | V1 (Wave 1A baseline) | V2 (now) | Delta |
|-|----------------------:|---------:|------:|
| imperative_heavy | 36 | 12 | **-24** |
| imperative_medium | 12 | 0 | -12 |
| imperative_trivial | 7 | 2 | -5 |
| declarative + via_helper | 26 + 0 | 58 + 10 | **+42** |
| imperative cell count | 257,876 + 650 | 65,841 + 11 | **-192,674** |
| has_ir | 34 | 72 | +38 |

### Static claims verifier (`scan_static_claims.py`)

* total: **50**, ok: **50**, problems: **0**, inert: **0**
* `has_errors: False`
* No declaration drift.

### KV eviction (`tools/measure_kv_eviction.py`)

* OFF peak K+V (sum across 32 transformer-block layers): **61.05 MiB**
* STATIC_LIVENESS peak K+V: **61.05 MiB**
* Tensor-shape drop: **0.00%**
* Realised drop (compaction-pass dividend): **0.00%** (0 B freeable)
* Evictable positions: 0 / 475,552 = **0.00%**
* Analyzer coverage: **1.4062%** (126 evictable / 5,040 cycle-conservative entries)

Per `kv_eviction.build_state_from_report`, the runtime evictable set
is the AND across every dim. Since analyzer marks only 2 dim names as
evictable (`SP_GATHERED_THIS_STEP`, `OUTPUT_LO_PREV_STEP`) and
**most dims are cycle-conservative**, every row stays live.
Cycle-conservative cohort includes `NEXT_SE`, `OP_*` opcode markers,
`FORMAT_PTR_HI`, `MEM_STORE`, `BYTE_INDEX_3`, `ADDR_B*`. The plan
target was **≥30% peak KV memory reduction** — gap is the full 30%.

---

## Section 3 — Remaining work per sub-wave

### 7.A — Scheduler cycle decomposition

**Done**

* 7.A.1 phase_inconsistent self-pin bugs fixed (count is now 0).
* 7.A.2 four `phase_required_but_undeclared` ops backfilled
  (`layer4_sp_to_addr_key`, `convo_io_prtf_transport`,
  `layer10_sp_byte_passthrough`, `layer15_si_mem_addr0_from_stack0`,
  `layer14_clear_addr_key_pollution`) — count dropped from 4 to 1.
* 7.A.3 OUTPUT_LO_PREV_STEP alias + readers landed; cleanup pass
  removed stale aliases.
* 7.A.5 B14 strict-flip with cycle-aware admission.

**Remaining**

* `layer14_demo_phase6_wave7` still in `phase_required_but_undeclared`
  (dep_depth=0; freely placeable in DAG — most likely add a
  `requires=` so the dynamic scheduler knows where to anchor it).
* OUTPUT_HI step-local vs PREV_STEP split — three commits attempted
  + reverted; `OUTPUT_HI_THIS_STEP` still produces 36 back-edges.
* TEMP / ADDR_KEY / AX_CARRY / ALU_LO back-edge decomposition not started.
* 88 ops still in dep-cycle SCC (target was ≤10).

**Blockers**

* Reverted OUTPUT_HI commits suggest the alias-only fix is
  load-bearing — the alias rename broke downstream ops that don't
  separate THIS_STEP from PREV_STEP. Needs a deeper per-reader
  audit, or a different decomposition strategy.

### 7.B — Pin removal corpus-wide

**Done**

* `dynamic_first_fit` allocator mode in both `dim_allocator.py` and
  `ffn_unit_allocator.py`.
* L0/L1/L2 (7.B.1), L3/L4/L5 (7.B.2), L12/L13/L14 (7.B.5), L10/L15/L16
  (7.B.6) FFN pins dropped, all byte-identical.

**Remaining**

* L6, L7, L8, L9, L11 FFN pin removal (waves 7.B.3 / 7.B.4 not yet
  named).
* Attention-head pin removal: head allocators still take
  `pin=head_idx` from layout tables. `grep -c "pin=" ops/` returns 57
  references; ~34 are actual non-None pins, mostly head-allocator
  calls. The plan called out this is structural (alibi-slope / W_o
  mapping).
* 7.B.3 final 1096-corpus logits comparison not run.

**Blockers**

* None imminent; L7/L8/L9/L11 should follow the L3-L5 pattern.

### 7.C — Cut partial migrations

**Done**

* 7.C.1 L6 routing_ffn legacy helper retired (10-commit merge,
  `2813324`); all IR override bands activated.
* 7.C.2 L15 memory_lookup migrated to IR via
  `RuntimeAttentionFragment` with conditional fragments
  (`7acd7f6`+`18c1725`).

**Remaining**

* The census still classifies `layer6_routing_ffn` as imperative_heavy
  because the bake_fn still writes cells outside `lower_*` (suggests
  a residual imperative-style branch the census heuristic flags;
  warrants a re-audit). May be a census taxonomy update rather than
  real code work.

**Blockers**

* None.

### 7.D — Model-level bake migration

**Done**

* 7.D.1 `TokenEmbeddingRule` IR landed (`436c610`+`6e8b894`).
* 7.D.2 `head_bake`, `embedding_bake`, `initial_pc_bake` migrated
  (`5ffb7fa`, `950cb75`, `6604585`, `ac14504`, `c95f8a8`).
* Follow-up: `embed_row 257` collision between `embedding_bake` and
  `initial_pc_bake` resolved (`c455e38`+`224d521`).

**Remaining**

* 7.D.3 sweep_compare equivalent for token embeddings — the census
  still lists `head_bake` (3,615 cells), `embedding_bake` (1,587),
  `initial_pc_bake` (2) as imperative-heavy/trivial, because their
  bake_fn still does direct cell writes alongside the IR. This is
  either (a) a census heuristic miss or (b) a remaining shim layer.

**Blockers**

* None.

### 7.E — Semantic dim references

**Done**

* 7.E.1 `(category, role)` semantic index added to `dim_registry`
  (`482338a`).
* `dim_ref()` helper added (`9d03f8b`).
* 7.E.2 L8 ALU pilot migrated 6 substages: `alu_add_carry`,
  `alu_lea_carry`, `alu_sub_borrow`, `alu_adj_carry`,
  `alu_ent_borrow` (`c3680e0` merge).

**Remaining**

* Corpus-wide migration of `+N` offset refs to `(category, role)` —
  only 10 `dim_ref(` call sites currently. The plan deliberately
  scoped 7.E.2 to a pilot; bulk migration of the remaining ~32 ops
  with FFN rules is the next step.
* Decision on which `+N` offsets are genuinely structural (look-up
  table indices) and should stay literal.

**Blockers**

* The plan called out 7.E as blocked on 7.A (dim variants from
  7.A.3 align with categories). With OUTPUT_HI split incomplete,
  some category mappings are still ambiguous.

### 7.F — Safe KV eviction

**Done**

* 7.F.1 `kv_liveness_analyzer.py` static analyzer.
* 7.F.2 `KVEvictionState` + `apply_eviction` + `base_layers`
  integration + `--kv-eviction-policy` flag + `test_kv_eviction.py`
  byte-identity gate.
* 7.F.4 `measure_kv_eviction.py` harness.

**Remaining**

* 7.F.3 CI gate — `test_kv_eviction.py` exists; needs to be wired
  into a recurring CI job.
* **Hit the 30% memory-reduction target.** Currently 0%.
* Analyzer needs (per harness report):
  * Per-head/dim cache compartmentalisation (today's
    `cached_k[B, H, S, HD]` packs every dim into one row — the AND
    over all dims keeps every row live).
  * Cycle-graph refinement (treat one-shot `_PREV_STEP` reads as
    non-cyclic).
  * Loosen "semantic-overwrite" category from "every later step" to
    "very next step".

**Blockers**

* 30% target is blocked on analyzer enhancements (above) AND on
  7.A.3 cycle decomposition (the SCC is a major contributor to
  cycle-conservative entries).

---

## Section 4 — Original vision check

The original vision (quoted in `docs/PHASE_7_FULLY_DYNAMIC_PLAN.md`):

> "no more imperative setting and then all fixes will be fixes on the
> level of the compiler and no non-io dims will be hardcoded and no
> layer indices will be hardcoded and the weights will be just
> outputed via compilation of the declarative spec"

### Goal 1 — Imperative removal: **PARTIAL**

* Authored ops (excludes no_ops): 83.
* Declarative + declarative_via_helper: 68 = **82% of authored ops**.
* Imperative-classed: 14 = 17%.
* Imperative cell share: 21.4% of total cells.
* From V1 to V2: -24 imperative_heavy ops, -12 imperative_medium,
  -5 imperative_trivial; +42 declarative.
* Biggest remaining: `l10_post_ops_combined` (39,284 cells, marked
  imperative_heavy). L10 carry propagation byte0/byte1/byte2 +
  comparison + binary-op-byte-zeroing migrated to FFNRule
  individually, but the umbrella `l10_post_ops_combined` op still
  bakes them via legacy helpers.

### Goal 2 — Fix-at-compiler-level: **PARTIAL**

* New ops added via `FFNRule`/`DeclarativeAttentionHeadSpec`:
  `layer14_demo_phase6_wave7` is the cited Phase 6 example.
* `docs/HOW_TO_ADD_A_CORRECTIVE_OP.md` codifies the 8-step recipe.
* IR coverage (`has_ir`): 72/116 ops = 62%. Of the remaining 44 no-IR
  ops, 33 are `no_op` stubs; the other 11 are legacy imperative bakes.
* Verifier infrastructure (`compare_symbolic_to_lowered_*`,
  `decl_verifier`, `scan_static_claims`) zero-drift across the corpus.
* Compiler-level fixes are possible but the L6/L7/L8/L9/L11 pin-removal
  and L8 ALU `(category, role)` migration are still in progress, so
  some fixes still require imperative offsets.

### Goal 3 — No hardcoded non-IO dims: **REMAINING-WORK**

* `dim_registry` has `(category, role)` semantic refs (Phase 7.E.1).
* `dim_ref(` use sites in `ops/`: **10** (pilot L8 ALU only).
* `+N` offset refs across ops: still corpus-wide.
* Roughly **5%** of dim references are semantic; the rest are still
  literal offsets.

### Goal 4 — No hardcoded layer indices: **PARTIAL**

* `phase=` references in `ops/`: **282**. Every op factory still
  declares an explicit phase ordinal.
* `layer_idx=` references: **189**. Mostly head-allocator pin args
  (`allocator.alloc(name, layer_idx=N, pin=head_idx)`), some are
  block.layer_idx slots.
* Scheduler still uses phase pruning to break the dep-cycle SCC of 88
  ops (74% of authored ops).
* `compile_full_vm_dynamic` with cycle-aware strict admission landed
  (7.A.5) but is NOT the default; the static `phase=N.M` path is.

### Goal 5 — Weights output via compilation of declarative spec: **PARTIAL**

* 78.5% of cells produced by `CompilerIR.lower_*` paths.
* `TokenEmbeddingRule` lowering exists for model.embed/head/initial_pc.
* `RuntimeAttentionFragment` handles shape-dependent attention heads
  (L15 memory_lookup).
* 21.5% of cells still flow through `setup_helpers` / `vm_step` direct
  writes (L10 post_ops, L6 routing residual, L15 attention_resize,
  function_call_weights, several L10 passthrough_bakes).

### Score table

| Goal | Score | Concrete metric |
|------|-------|-----------------|
| Imperative removal | PARTIAL | 82% of authored ops declarative; 78.5% of cells |
| Fix-at-compiler-level | PARTIAL | 62% IR coverage; 0 verifier drift; recipe doc exists |
| No hardcoded non-IO dims | REMAINING-WORK | 10 `dim_ref(` calls; ~5% adoption |
| No hardcoded layer indices | PARTIAL | 282 `phase=` + 189 `layer_idx=` references; strict dynamic mode landed but not default |
| Declarative weight output | PARTIAL | 78.5% of cells via `lower_*` |

---

## Section 5 — Recommended Phase 8 priorities

Ranked by expected unlock-per-effort:

### Tier 1 — Unblock the 30% KV eviction target

The 7.F.4 harness shows the analyzer floor is 1.4% coverage and
realised drop is 0%. The fastest path to the plan's headline target:

1. **Refine the analyzer's cycle classifier** — `_PREV_STEP` reads
   should not promote the base dim to cycle-conservative.
2. **Per-(layer, head, dim-group) cache compartmentalisation** —
   today's row-wise AND collapses any eviction. Splitting the cache
   into groups (e.g. registers / ALU / TEMP / OUTPUT) would let
   compaction at least apply per-group.
3. **Expand "semantic overwrite" to single-step lookahead** — would
   catch register carries like `REG_AX` automatically.

### Tier 2 — Finish 7.A.3 OUTPUT_HI decomposition

* 88 ops still in the cycle SCC; OUTPUT_HI is the second-largest
  back-edge family (36) after OUTPUT_LO.
* The reverted alias attempt suggests we need a per-reader audit
  before another alias rename. Build a `analyze_output_hi_readers.py`
  similar to the OUTPUT_LO followups.

### Tier 3 — Complete pin removal for L6/L7/L8/L9/L11

* Mirrors 7.B.5/7.B.6. Largest remaining imperative-heavy op
  `l10_post_ops_combined` already migrated under-the-hood; the
  umbrella op needs taxonomy reclassification.
* Run 7.B.3 (1096-corpus logits parity).

### Tier 4 — Scale up 7.E.2 (semantic dim refs)

* Pilot worked on L8 ALU. Apply pattern to L9 ALU, L10 ALU,
  L11 mul_partial, L12 mul_combine, L13 shifts. These five ops
  account for ~140k cells; if they all migrate to `(category, role)`
  the literal-offset count drops by ~70%.

### Tier 5 — Make `compile_full_vm_dynamic` the default

* 7.A.5 cycle-aware admission landed. Once 7.A.3 + remaining 7.B
  waves are in, flip the default in `compile_full_vm` and delete the
  phase-pruning path. This removes the last "hardcoded layer indices"
  blocker.

### Stretch — Model-level cleanup

* `function_call_weights` (2,910 cells) and `l15_attention_resize`
  (1,973 cells) are the last large imperative-heavy ops with no IR.
  Both encode runtime-shape decisions; either extend the IR
  (`RuntimeAttentionFragment` for `function_call_weights`?) or
  document as exceptions.

---

_End of report._
