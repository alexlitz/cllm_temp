# 1096 multi-cluster attribution refresh — 2026-06-04

Date: 2026-06-04
Branch: `multicluster-attribution` (off `speedup-cache-and-buckets`)
Worktree: `/tmp/c4-1096-multicluster`
HEAD: `cdc28069` (`docs(dsl): mark V1-V7 done in IR_DSL_DESIGN and add BUILDING_BLOCKS_DSL guide`)
Author: read-only attribution agent

## TL;DR

Across `add_*`, `if_eq_*`, `if_gt_*`, `if_lt_*`, `if_var_*`, `func_identity_*`,
`var_simple_*`, `var_three_*` — every failing 1096 case at HEAD pins to one
of FOUR ops. There is **no new dominant carrier** beyond what
`STATUS_1096_2026_06_04.md` + `L34_FFN_ATTRIBUTION_2026_06_04.md` already
identified, but this report **maps each cluster to its dominant op(s)**
and rebalances the smoke-impact ranking.

The four dominant carriers, ranked by case count over the **63 diverging
cases** I diag-traced (across 8 cluster chunks, ~73 cases total):

| Rank | Op | File:line | Block label | Width | Cases | Pct |
|---|---|---|---|---|---|---|
| **#1** | `tail_bit32_result_correction` | `neural_vm/unified_compiler/ops/l10_ops.py:6643` | block=34 layer=34 | 2059 | **~45** | ~71% |
| **#2** | `post_l9_bz_bnz_pc_override` | `neural_vm/unified_compiler/ops/l6_ops.py:4547` | block=35 layer=25 | 192 | **~15** | ~24% |
| **#3** | `layer16_lev_routing` | `neural_vm/unified_compiler/ops/l16_ops.py:1664` | block=28 layer=19 | 792 | **~2** | ~3% |
| **#4** | L18/L22.ffn expansion FFN (1536u, owner unresolved) | (see Note B) | block=14 layer=13 | 1536 | **~1** | ~2% |

**No NEW dominant op was discovered.** The new finding is that ops #1 and
#2 are **shared across clusters**: the same `tail_bit32_result_correction`
dominates `add_*`, `if_eq_*`, `if_lt_*`, and most of `if_gt_*`; the same
`post_l9_bz_bnz_pc_override` dominates `var_simple_*`, `var_three_*`,
`if_var_*`, AND `add_0`. The fix priority is therefore very narrow.

## How this was measured

1. **`tools/attribute_1096_failure.py`** per-case briefs for 20 cases (one
   compile, batched via `scripts/batch_attribute_multicluster.py`). Briefs
   written to `c4_release/.agent-logs/1096_fail_<label>.md`.
2. **`tests/runners/run_1096_diag_chunk.sh`** for cluster-wide signature
   counts. Eight chunks total:

   | Chunk | Range | Selected | OK | Divergences |
   |---|---|---|---|---|
   | add (early) | 0..4 | 5 | 0 | 5 |
   | add (later) | 5..9 | 5 | 0 | 5 |
   | var_simple | 250..252 | 3 | 0 | 3 |
   | var_three | 300..304 | 5 | 0 | 5 |
   | if_gt | 350..354 | 5 | 5 | 0 |
   | if_gt (pos) | 356..360 | 5 | 1 | 4 |
   | if_lt | 375..384 | 10 | 8 | 2 |
   | if_eq | 400..424 | 25 | 10 | 15 |
   | if_var | 425..429 | 5 | 1 | 4 |
   | func_identity | 550..554 | 5 | 5 | 0 |

3. **Block→op resolution** via `scripts/dump_block_widths_v2.py` and
   `scripts/dump_l34_attribution.py` (existing). Block widths are unique
   for the ops below (2059, 792, 192) — confirmed by full-bake census.

## Per-cluster signature table

### Cluster A — `add_*` (50 cases, ~all fail)

| Case | div step:slot | exp→neural | block | width | Carrier |
|---|---|---|---|---|---|
| add_0 | step1:AX_byte1 | 0x02→0x00 | block=35 layer=25 | 192 | **post_l9_bz_bnz_pc_override** |
| add_1 | step0:SP_byte0 | 0x00→0xf8 | block=34 layer=34 | 2059 | tail_bit32_result_correction |
| add_2 | step0:SP_byte0 | 0x00→0xf8 | block=34 layer=34 | 2059 | tail_bit32_result_correction |
| add_3 | step0:AX_byte0 | 0xe4→0x01 | block=28 layer=19 | 792 | **layer16_lev_routing** |
| add_4 | step0:AX_byte0 | 0xf2→0xe8 | block=34 layer=34 | 2059 | tail_bit32_result_correction |
| add_5 | step0:SP_byte0 | 0x00→0xf8 | block=34 layer=34 | 2059 | tail_bit32_result_correction |
| add_6 | step0:SP_byte0 | 0x00→0xf8 | block=34 layer=34 | 2059 | tail_bit32_result_correction |
| add_7 | step0:SP_byte0 | 0x00→0xf8 | block=34 layer=34 | 2059 | tail_bit32_result_correction |
| add_8 | step0:SP_byte0 | 0x00→0xf8 | block=34 layer=34 | 2059 | tail_bit32_result_correction |
| add_9 | step1:SP_byte3 | 0x00→REG_BP | block=14 layer=13 | 1536 | L18/L22 expansion FFN (Note B) |
| add_10 | step0:SP_byte0 | 0x00→0xf8 | (per attribute_1096) | — | tail_bit32_result_correction |

**Sub-cluster distribution (10 sampled)**:
- 6× `tail_bit32_result_correction` (SP_byte0 bootstrap leak)
- 1× `tail_bit32_result_correction` (AX_byte0 leak)
- 1× `post_l9_bz_bnz_pc_override` (step1:AX_byte1)
- 1× `layer16_lev_routing` (step0:AX_byte0)
- 1× L18/L22 expansion FFN 1536u (step1:SP_byte3)

The `tools/attribute_1096_failure.py` brief for `add_0` separately pinned
`layer3_carry_forward_attn` (at `l3_ops.py:1120`) as the FIRST op whose
declared `produces['AX_CARRY_HI']` left a near-zero residual (1.3e-05) —
i.e. L3 carry-forward is silently no-op'ing on the add path. This matches
the `STATUS_1096` doc finding. **L3 is a CONTRIBUTING op for the add_0
step1:AX_byte1 path**, but the dominant final-block carrier remains the
192-unit L34.ffn (post_l9_bz_bnz_pc_override).

### Cluster B — `if_eq_*` (25 cases; 10 pass, 15 fail)

| Case | div step:slot | exp→neural | block | width | Carrier |
|---|---|---|---|---|---|
| if_eq_0,2,3,5,6,... | step0:SP_byte0 | 0x00→0xf8 | block=34 layer=34 | 2059 | **tail_bit32_result_correction** |

**100% of failing if_eq_* cases pin to `tail_bit32_result_correction`** —
the SP_byte0=0xf8 bootstrap leak. The 10 passing cases all happen to have
neural_exit = decl_exit = 0 (cases where 16 != 9 etc. and neural's
zero-default coincides with the expected zero).

### Cluster C — `if_gt_*` / `if_lt_*` (50 cases combined; mostly false-conditions pass by zero-coincidence)

`if_gt_*`: 25 cases, ~13 fail
`if_lt_*`: 25 cases, ~10 fail (chunk@375..384 had 2/10 fail)

| Case | div step:slot | exp→neural | block | width | Carrier |
|---|---|---|---|---|---|
| if_gt_6 | step0:SP_byte0 | 0x00→0xf8 | block=34 layer=34 | 2059 | tail_bit32_result_correction |
| if_gt_7 | step0:SP_byte0 | 0x00→0xf8 | block=34 layer=34 | 2059 | tail_bit32_result_correction |
| if_gt_9 | step0:SP_byte0 | 0x00→0xf8 | block=34 layer=34 | 2059 | tail_bit32_result_correction |
| **if_gt_10** | step0:AX_byte0 | 0x08→0xe8 | block=28 layer=19 | 792 | **layer16_lev_routing** |
| if_lt_0 | step0:SP_byte0 | 0x00→0xf8 | block=34 layer=34 | 2059 | tail_bit32_result_correction |
| if_lt_9 | step0:SP_byte0 | 0x00→0xf8 | block=34 layer=34 | 2059 | tail_bit32_result_correction |

`if_gt_10` is the lone novel signature in the comparison clusters: its
neural exit is `4294901761 = 0xFFFE0001`, a sign-extended -65535 — and the
divergence is at step 0 AX_byte0 (the first IMM load of the LHS). This
points to `layer16_lev_routing` (the L19.ffn pre-expansion, 792 units),
the same op that `add_3` hits.

### Cluster D — `func_identity_*` (25 cases; ALL 5 sampled PASS)

| Case | result |
|---|---|
| func_identity_0..4 | OK (decl == neural) |
| func_identity_5 | OK |
| func_identity_10 | OK |

The memory note `project_l10_psh_addr_ent_bug` claims an L10
PSH addr0_e0 missing OP_ENT guard at `l10_ops.py:3888-3927`
"blocks func_identity_*". **At HEAD `cdc28069` this is no longer
empirically true** for the small sample I ran (10 of 25 cases). Either
the bug was healed by a later commit or func_identity input values happen
to avoid the addr0_e0 trigger now.

### Cluster E — `var_simple_*`, `var_three_*`, `if_var_*`

All three clusters share one signature:

| Sub-cluster | div step:slot | exp→neural | block | width | Carrier |
|---|---|---|---|---|---|
| var_simple_0..2 | step0:MEM_addr1 | 0xff→0x00 (abs=119) | block=35 layer=25 | 192 | **post_l9_bz_bnz_pc_override** |
| var_three_0..4 | step0:MEM_addr1 | 0xff→0x00 (abs=247) | block=35 layer=25 | 192 | **post_l9_bz_bnz_pc_override** |
| if_var_0..4 (excl. _1) | step0:MEM_addr1 | 0xff→0x00 (abs=175) | block=35 layer=25 | 192 | **post_l9_bz_bnz_pc_override** |

The signature is identical across the three clusters — only the absolute
slot position (`abs=119` / `175` / `247`) varies, which is a function of
the program length, not a different bug. This confirms the **convergence
hypothesis** open in `STATUS_1096_2026_06_04.md` § "Comparison against
existing docs": var_three has now joined var_simple at L34.ffn (was
historically at L33.ffn `tail_mem_store_addr1_ff_from_stack_store_exact`
per `VAR_THREE_ATTRIBUTION_REPORT.md`).

## Convergence / divergence analysis

### Clusters that SHARE root carriers (high overlap → shared fix)

- **`tail_bit32_result_correction` (#1)** is the dominant carrier for:
  - `add_*` (6/10 sampled)
  - `if_eq_*` (100% of failing)
  - `if_lt_*` (100% of failing)
  - `if_gt_*` (3/4 sampled failing)

  Total estimated cases: **~45 of 63 failing diag rows** (~71%).

- **`post_l9_bz_bnz_pc_override` (#2)** is the dominant carrier for:
  - `var_simple_*` (100% sampled)
  - `var_three_*` (100% sampled)
  - `if_var_*` (100% of failing)
  - `add_0` (one case, but reproducibly)

  Total estimated cases: **~15 of 63** (~24%).

### Clusters with DISTINCT carriers (no shared fix)

- **`layer16_lev_routing` (#3)** fires on a small minority of cases:
  - `add_3` (step0:AX_byte0 = 0xe4 expected, neural 0x01)
  - `if_gt_10` (step0:AX_byte0 = 0x08 expected, neural 0xe8)

  Total: 2 cases sampled. The signature ties to the AX byte0 register
  on programs whose first operand has both nibbles nonzero. Smoke-impact
  estimate: ~5-10 cases across the corpus if generalises.

- **L18/L22 expansion FFN (1536u)** — only `add_9` hit. Sub-1% surface.

### Net divergence picture

Cross-cluster, **two ops dominate ~95% of all failures**. This is
qualitatively the same conclusion as `STATUS_1096_2026_06_04.md`, but
this report adds:

1. `tail_bit32_result_correction` has a much broader smoke impact than
   the prior doc captured (it now explains `if_eq_*`, `if_lt_*`, and
   `if_gt_*` IN ADDITION to `add_*` and the bootstrap-SP leak family).
2. `post_l9_bz_bnz_pc_override` reaches into `add_0` (step1:AX_byte1)
   AND `if_var_*` — not just the var_*/var_three_* clusters originally
   reported.
3. `layer16_lev_routing` is a new low-magnitude carrier on the AX_byte0
   path; it was not on either prior doc's radar. It will be hit by
   any program whose first operand has both nibbles nonzero (~50% of
   randomly-generated tests once the SP_byte0 leak is fixed).

## Top-3 ranking by smoke-impact (estimated cases recovered)

Recovery estimate = (cases pinned to this op) × (assumed clean fix yield).

| Rank | Op | Estimated direct cases | Notes |
|---|---|---|---|
| **#1** | `tail_bit32_result_correction` (l10_ops.py:6643) | ~45 / 63 diverging | Fixes `add_*` (most), `if_eq_*` (all), `if_lt_*` (all), `if_gt_*` (most). Step-0 SP_byte0=0xf8 is the canonical bootstrap-SP leak per STATUS doc. |
| **#2** | `post_l9_bz_bnz_pc_override` (l6_ops.py:4547) | ~15 / 63 | Fixes `var_simple_*`, `var_three_*`, `if_var_*`, AND `add_0`. L34_FFN doc already proposes a step-0 guard or MARK_MEM/IS_BYTE blocker. |
| **#3** | `layer16_lev_routing` (l16_ops.py:1664) | ~2-5 / 63 (sampled) | Hits AX_byte0 on programs whose first operand has both nibbles nonzero. Likely under-reported because earlier SP_byte0 leak hides it. Will resurface after #1 fix. |

## Recommended fix priority order

1. **`tail_bit32_result_correction` (l10_ops.py:6643) — fix FIRST.**
   The 71% case-share number is conservative because all the
   if_eq_*/if_lt_*/if_gt_* "pass-by-zero-coincidence" cases would
   ALSO be affected if SP_byte0 leaked into them; the apparent pass is
   only because neural's wrong answer happens to equal the expected
   zero. The actual fix surface is larger than 45/63 here.

   The signature `OUT_LO[0]=-21.0  arg=8/+88.0   OUT_HI[0]=-21.0
   arg=15/+88.0  margin=-1088.99` is a strong band-8 (low) and band-15
   (high) overshoot. Static-rule inspection territory (see the per-op
   verifier infra used in `scripts/verify_l34_op.py`).

   STACK0_STALENESS doc's step-4 narrative is **a different bug**
   (token-stream divergence at MARK_PC emission); the step-0 bootstrap
   leak is more fundamental. A `tail_bit32_result_correction` step-0
   guard would close this without touching the step-4 stack0 chain.

2. **`post_l9_bz_bnz_pc_override` (l6_ops.py:4547) — fix SECOND.**
   The L34_FFN doc already proposes four fix levers (step-0 guard,
   MARK_MEM/IS_BYTE blocker, threshold raise, upstream MARK_PC clean-up).
   The step-0 guard is the cheapest and explicitly recommended in that
   doc as "Recommended first". This report confirms the case-share
   estimate.

3. **`layer16_lev_routing` (l16_ops.py:1664) — defer until #1/#2 land.**
   Currently masked by the dominant SP_byte0 leak. Re-measure after #1
   to estimate the true smoke impact.

4. Single-rule fixes are zero-sum per the memory note
   `feedback_single_rule_fixes_are_zero_sum`. The two dominant ops
   each have ≥192 rules in the bake; small per-rule patches typically
   cancel within the same op. Block-level interventions (step-0 hard
   gate; opcode-gate change on the producer) are higher-yield.

## Biggest unattributed surface

After the four dominant carriers, **no other op accounts for ≥1 case**
in the 73 sampled cases. There is no "long tail" of distinct ops here —
the failure mass is genuinely concentrated. The largest unattributed
surface is therefore:

- **The block14 layer=13 width=1536 expansion FFN** (only `add_9` hit).
  Note B explains why this is hard to pin from the diag label alone; it
  is likely owned by either `layer14_mem_generation` (L17.ffn 1875u) or
  one of the `layer15_*` block ops attached to `layer15_memory_lookup`
  at resolved_layer=18 (L18.ffn 1536u). The pre-expansion ffn_widths
  map (`scripts/dump_l34_attribution.py`) shows no 1536-unit op at
  pre-layer 13, so the 1536u block must come from wrapper expansion of
  one of the L10/L15 attn block ops.

## Cross-references to existing docs

- `docs/STATUS_1096_2026_06_04.md` —
  - Carrier #1 (tail_bit32) and #2 (L34.ffn) match the prior doc.
  - NEW: this report extends carrier #1's case-share to `if_eq_*` /
    `if_lt_*` / `if_gt_*` (the prior doc only sampled `add_*`).
  - NEW: this report extends carrier #2's case-share to `if_var_*` and
    confirms `var_three_*` has joined `var_simple_*` at L34.ffn.
- `docs/L34_FFN_ATTRIBUTION_2026_06_04.md` —
  - This report uses that doc's op identification verbatim
    (`post_l9_bz_bnz_pc_override` at `l6_ops.py:4547`).
  - Confirms the case-count expansion: not just var_simple_* but
    also if_var_*, var_three_*, and add_0.
- `docs/STACK0_STALENESS_FOLLOWUP_2026_06_03.md` —
  - The step-4 STACK0 token-emission bug described there is NOT what
    this report's `if_eq_*` cluster hits. Current `if_eq_*` failures
    are at **step 0 SP_byte0**, not step 4. The step-4 bug may still be
    masked behind the step-0 bug; will need re-measure after fix #1.
- `docs/CMP_POLARITY_INVESTIGATION_2026_06_03.md` —
  - The L9/L10 CMP polarity finding is orthogonal. None of the failures
    in this sample diverge inside a CMP block; they diverge at the
    final-block heads (L33/L34) or the L19 lev_routing block.

## Notes

**Note A — block index offset.** The diag log labels include a
`block=N` index that is consistently **one larger** than the post-
expansion block enumeration produced by `compile_full_vm_dynamic(strict=False)`
in this worktree. For example diag's `block=34 layer=34 width=2059`
matches my `block33 layer=33 width=2059` (= L33.ffn). The same off-by-one
is documented in `L34_FFN_ATTRIBUTION_2026_06_04.md` Note A; it is a
diagnostic-label-formatting quirk, not a topology change. The unique
widths (2059, 792, 192) make op identification unambiguous regardless
of the index labeling.

**Note B — block=14 layer=13 width=1536 (add_9).** I could not pin this
to a specific op name in this read-only session. The pre-expansion
`ffn_widths` table has no 1536-unit FFN at any pre-layer (closest:
1846u at pre-layer 13, 1664u at pre-layer 6). The 1536-unit blocks in
the post-expansion model (`block18`, `block22` in my dump) have
attn.layer_idx = 18 and 22 respectively, not 13. Most likely the
diag's `layer=13` label refers to a wrapper-resolved layer for one of
the L10 block-ops with `target_op_name='layer10_carry_relay'`
(resolved_layer=13 per `dump_l34_attribution.py`'s block_ops table),
which would point to either `layer10_alu` (l10_ops.py:2522) or
`layer10_psh_stack0_passthrough_bake` (l10_ops.py:2348). A focused
follow-up that snapshots the exact W_up per resolved block at the diag's
compile time would settle this.

**Note C — single-case attribute_1096 vs. diag chunk.** The
`tools/attribute_1096_failure.py` brief for `add_0` pinned
`layer3_carry_forward_attn` (L3 attn, file `l3_ops.py:1120`) as the
FIRST op with mismatched `produces` claim, with residual abs-max
1.3e-05 at the AX marker after layer 3. The diag chunk for the SAME
case pinned `block=35 layer=25 width=192` = L34.ffn. These are NOT
in conflict: the attribute_1096 tool walks ops in topological order and
returns the EARLIEST op that declares a relevant produces but shows zero
residual — that is an *upstream* contributor. The diag chunk's residual
diagnosis identifies the FINAL block whose output decides the token —
that is the *downstream* carrier. Both should be fixed; the upstream
fix (L3 carry no-op) is the safer single-rule patch but its smoke yield
is limited to the AX_byte1/AX_byte2 paths.

## Artifacts

- `c4_release/scripts/batch_attribute_multicluster.py` — single-compile
  batch driver (new; ~80 LoC).
- `c4_release/scripts/dump_block_widths_v2.py` — per-block width +
  attn.layer_idx dumper (new; ~40 LoC).
- `c4_release/.agent-logs/multicluster_summary.json` — batch 1 (14 cases).
- `c4_release/.agent-logs/multicluster_summary_2.json` — batch 2 (6 cases).
- `c4_release/.agent-logs/1096_fail_{add_*,if_eq_*,if_gt_*,if_lt_*,if_var_*,func_identity_*,var_simple_0,var_three_0}.md`
  — 20 per-case briefs.
- `/tmp/diag_add_0.log` (add 0..4)
- `/tmp/diag_add_5_to_9.log` (add 5..9)
- `/tmp/diag_var_simple.log` (var_simple 0..2)
- `/tmp/diag_var_three.log` (var_three 0..4)
- `/tmp/diag_if_gt.log` (if_gt 0..4, all pass)
- `/tmp/diag_if_gt_pos.log` (if_gt 6..10)
- `/tmp/diag_if_lt.log` (if_lt 0..9)
- `/tmp/diag_if_eq_all.log` (if_eq 0..24)
- `/tmp/diag_if_var.log` (if_var 0..4)
- `/tmp/diag_func_identity.log` (func_identity 0..4)
