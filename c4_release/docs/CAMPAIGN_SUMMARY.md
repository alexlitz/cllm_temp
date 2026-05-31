# Neural-VM 1096-Test Campaign Summary (2026-05-29 / 2026-05-30)

Base branch: `speedup-cache-and-buckets @ 4d069f7`
Final head (C9): `e00709f` (B7-4 ADDR_B0_VALID landed)
Owner: multi-agent campaign, 7 batches + follow-ups (C-units)

---

## 1. Overview

**Goal.** Raise the 1096-test neural-VM pass rate; restore spec-decoding +
batched-KV functionality on the production pure-neural mode; and migrate the
L10 tail-correction family from strength-escalating proxies to declarative
upstream evidence.

**Duration.** ~30 hours of agent work over 2026-05-29 -> 2026-05-30.

**Effort.** ~60 sub-agents spawned across 7 numbered batches (B1..B7) plus
follow-up C-units (C1 retest, C2/C3 per-op harnesses, C9 this doc). The
campaign produced ~70 commits on `speedup-cache-and-buckets` plus dozens of
branches on `investigation/*`, `proposal/*`, `audit/*`, and `integration/*`.

**Final pass rate.** 304/1096 = 27.7% on the declarations-only pure-neural
path (per the C1 retest, which superseded mixed-handler numbers). The
B7-6/B7-7 architectural cleanup is staged but not yet landed; v4 retest
will measure that delta once B7 lands.

---

## 2. Pass-rate progression

| Phase | Pass | Total | Rate | Notes |
|-------|-----:|------:|-----:|-------|
| Pre-batch (mixed handlers + decl) | 387 | 1096 | 35.3% | misleading — counted handler-resident paths |
| Pre-batch (pure-neural only) | ~298 | 1096 | 27.2% | true baseline for the campaign |
| After batch 5 (v2 integration) | 304 | 1096 | 27.7% | +6 vs. pure-neural baseline |
| After batch 7 (integration/final, B7-1..5 landed) | 304 | 1096 | 27.7% | structural dims allocated; consumers not yet rewired |
| After v4 (B7-6/B7-7 land) | TBD | 1096 | TBD | pending L10 family rewrite |

The headline metric did not move dramatically because the bulk of B6/B7 work
was **structural plumbing** (new dims, new producers, new lifecycle bits).
The pass-rate payoff arrives when B7-6 (L10 `tail_sp_marker_*` refactor) and
B7-7 (L10 `tail_mem_store_addr0_*` refactor) consume those dims and drop the
strength-escalating siblings.

What did move: spec-decoding correctness (restored via `e7c4d63`),
KV-eviction validation (`4d069f7` revalidated by B5-G), and the regression
floor — B3-eta's ENT-main misfire and B2-A's step-5 misfire are no longer
present in v2/v3.

---

## 3. Bug catalog

Bugs surfaced and tracked during the campaign. Status: F = fixed and
landed; D = diagnosed with PLAN/REPORT; U = unresolved.

| # | Bug | File:line | Status | Tests unblocked |
|---|-----|-----------|--------|-----------------|
| 1 | `tail_mem_store_addr0_e0_from_psh_sp_no_addr_src_authority` overpowers ENT-main (5e9 strength) | `l10_ops.py:3889` | F | ENT step set |
| 2 | `tail_mem_store_addr0_00_from_global_exact` false-fires at step 5 (OUTPUT proxy ambiguity) | `l10_ops.py:3696` | F (via 1ef127c, B5-D partial reroute) | global-store ids |
| 3 | `tail_sp_marker_byte0_f8_from_initial_stack_exact` fires on residual MARK_SP >1.0 | `l10_ops.py:3465` | D (B7-6 plan) | initial-SP ids |
| 4 | `MEM_ADDR_SRC=1` injection wrongly fired on PSH/POP (not SI/SC) | `l7_ops.py` MEM head 7 | F (`0316600`) | PSH/POP store/load ids |
| 5 | Current-step MEM marker missing in batched runner (causes false STACK0 attention) | batched runner | F (`e2f1e8e` Recovery B2-H) | batched-only ids |
| 6 | L11 -> L12 wide-MUL amplitude mismatch | `alu_ops.py` MUL pipeline | F (`cc55474`) | MUL ids |
| 7 | L10 SP-marker rule circular OUTPUT self-amp (B5-J's HAS_SE -1e9 was a hack) | `l10_ops.py:3465` | D (B7-6 plan) | initial-SP ids; pending |
| 8 | L4 `sp_to_addr_key` mis-flagged as enabled when STACK0 mem-attention disabled | L4 op | F-as-absent (`f461e57` -- flagged disabled) | none directly; unblocks diagnosis |
| 9 | spec_k cache corruption on draft VM divergence | spec engine | F (`e7c4d63` fallback + draft MEM marker) | spec-decoding regression suite |
| 10 | KV eviction max_tokens corruption on bounded batches | runner | F (`4d069f7` validated by B5-G) | KV-eviction suite |
| 11 | L17 hidden FFNs (`l10_post_ops_combined` + `tail_bit32_result_correction`) mis-attributed to L17 | layout `vm_step.py:2627` | D (B6-L inventory) | none; investigation primer |
| 12 | ADDR_B0 staging stale at MARK_MEM rows (residual leak from MARK_STACK0) | `l8_ops.py` SP gather | F partial (`a9a45b6` fires SP gather at MARK_SP); full fix B7-3/B7-5 | tail-addr family |
| 13 | L10 OUTPUT_LO+8 / OUTPUT_HI+15 proxy for SP byte 0 = 0xF8 (no structural signal) | `l10_ops.py:3465` | D (B7-2 dim allocated `e00709f`); consumer pending | SP marker family |
| 14 | No `IN_STEP_FRESH` lifecycle bit -> tail rules can't distinguish fresh vs. residual | L1 attn | F (`2da184b` slot 96 allocated); consumer pending | tail family |
| 15 | No `ADDR_B0_VALID` lifecycle bit -> L10 family can't gate on freshness | L13 gather | F (`e00709f` slot 97 allocated); consumer pending | tail-addr family |
| 16 | No `SP_GATHERED_THIS_STEP` lifecycle bit | L8 gather | F (`3fe4ffa` allocated); consumer pending | SP-marker family |
| 17 | L0 H5/H6/H7 attention heads dead (write, no reader, 21 dims wasted) | `l0_ops.py:141` | D (B6-K usage map) | reclaim 21 dims |
| 18 | L16 `e0`/`f0` STACK0 marker fails to preserve post-PSH stack top across IMM | `l16_ops.py` | F (`6f8920b`) | post-PSH IMM ids |
| 19 | L10 OP_IMM AX byte 1 not zeroed | `l10_ops.py` (`tail_ax_imm_byte1_hi_zero`) | F (`cb110ce`) | IMM AX-byte-1 ids |
| 20 | L11/L12 MUL declaration alignment regression | `alu_ops.py` MUL | F (`b29d3ed`) | MUL regression suite |
| 21 | Block27/layer27 = L17 -- B4-B misattribution | `l10_ops.py:1289`, `l10_ops.py:4529` | D (B6-L inventory) | investigation primer |
| 22 | L7 head-5 ENT-SP fetch-gate limitation | `l7_ops.py` head 5 | D (`846d8fe` documented) | ENT-SP ids; unresolved |
| 23 | B3-gamma MARK_SP/OUTPUT inversion (rejected; can't be solved at L10) | `l10_ops.py:3465-3502` | D (`5ef58b0` rejection) | initial-SP ids; structural fix B7-6 |
| 24 | ONNX runtime path: cummax + dynamo guard broken | exporter | U (`7d4a5e3` audit) | export pipeline |
| 25 | C runtime path: build broken, no baseline | C runtime | U (`48c13a6` audit) | C-runtime baseline |
| 26 | `absdiff` (0/25) and `nested_quad` (0/16) categories entirely dead | `tests/` | D (`a8924be`) | 41 ids; unresolved |

Bugs 13-16 are the **structural plumbing** that enables the v4 retest delta.

---

## 4. Per-batch contributions

- **B1 (active-opcode purity, MEM_EXEC, spec-decoding existing)**: established
  the spec-decoding regression suite and pure-neural-mode gating; framed the
  blocker stack (ALU-HI leakage, PC progression, heap LI/LC, JSR/LEV).
- **B2 (recovery + spec / batched-KV)**: Recovery B2-H landed the current-step
  MEM marker fix in the batched runner (`e2f1e8e`); B2-A surfaced the global
  addr0 step-5 misfire; B2-B inverted MARK_SP/OUTPUT (later rejected by B3-gamma).
- **B3 (alpha/eta/gamma/epsilon)**: documented the step0->step1 0xE0 PSH-vs-PC
  rule (`0cde3d3`), tamed the 0xE0 PSH strength + ENT block (`87a75e4`),
  rejected the MARK_SP inversion as L10-unsolvable (`5ef58b0`), and added
  B2-H staticmethod cleanup + helper unit tests (`812a38f`).
- **B4 (architectural plans + per-op harnesses)**: B4-A added OP_ENT/LEV/IMM
  blockers to L10+L16 0xE0 rules (`527f540`); B4-G staged 20 validated
  branches for PR open; B4-H wrote the L10 `tail_mem_store_addr0_*`
  architectural refactor PLAN (`e672177`); landed per-op audit harnesses
  for L0/L2/L4 (`7fbf4a7`), L5 (`ae2c51f`), L8 (`14e17fe`), L9 (`2bd4264`),
  L11/L12 (`d3f22fe`), L13 (`1865bfe`), L14 (`acd3ce9`), L15 (`068dcbc`),
  L16 (`4d3a7dc`), L6 (`c8af933`).
- **B5 (revalidation + reroute)**: B5-D rerouted L10 tail rules through L13
  ADDR_B0 (Path 2 partial, `66d9e12`); B5-F spec_k revalidation (`1ada801`);
  B5-G KV eviction revalidation (`2aa07f6`); B5-H production-config
  validation (`0abd829`); B5-J investigated HAS_SE lifecycle gate on the SP
  marker F8 rule (`d0065e8`, found -5 regression, rejected hack).
- **B6 (investigations + audits)**: B6-G L7-L9 structural-signal audit
  (`8d7d86a`); B6-K BD dim usage map identifying 21 reclaimable dims at
  H5/H6/H7 (`8d3ee75`); B6-L L17 post-op inventory (`974471e`) showing
  block27/layer27 = L10 tail; B6-B augmented L10 tail addr-family with
  soft ADDR_B0/B1/B2 evidence (`d4b2a90`); B6-D L16 e0 STACK0 fix (`6f8920b`).
- **B7 (structural plumbing)**: B7-1 IN_STEP_FRESH at L1 head 5
  (`2da184b`, slot 96); B7-2 SP_BYTE0_IS_F8 dim + L7 head-6 producer
  (`35e0d6b`, slot 95); B7-4 ADDR_B0_VALID lifecycle bit (`e00709f`, slot
  97); B7-5 SP_GATHERED_THIS_STEP (`3fe4ffa`, slot 98); supporting fix to
  fire L8 SP gather at MARK_SP (`1c11364` / `a9a45b6`); B7-X L16 helper
  refactor (`fdaf8e0`). B7-6 / B7-7 consumer rewrites are staged but not
  yet merged.

---

## 5. Key architectural findings

1. **The L10 tail-correction family is the dominant bug surface.** The 18
   strength-escalating rules in `l10_ops.py:3450-4100` are responsible for
   most of the OUTPUT-byte regressions surfaced in B2-A, B2-B, B3-eta,
   B5-J. The rules each pick a "novel" combination of `ALU_LO` / `CMP` /
   `PSH_AT_SP` / `OP_JSR` / `MEM_ADDR_SRC` proxies, then crank `strength`
   until they out-vote their predecessors. Strength has reached 5e9 on one
   rule.

2. **21 reclaimable dims at slots 95-115.** L0's H5/H6/H7 attention heads
   (thresholds 14.5/19.5/24.5) write but no downstream op reads. B6-K's
   usage map made the dim budget look saturated only because every dim has
   *some* writer; counting *consumers* exposed the headroom. B7 used 4 of
   these slots (95/96/97/98) for `SP_BYTE0_IS_F8`, `IN_STEP_FRESH`,
   `ADDR_B0_VALID`, `SP_GATHERED_THIS_STEP`.

3. **Four foundation lifecycle dims unlock declarative rewrites.** The
   wishlist from B6-G + B4-H crystallized into four upstream signals:
   - `IN_STEP_FRESH` (L1 head 5, slot 96): 1.0 from PC marker through SE
   - `SP_BYTE0_IS_F8` (L7 head 6 extension, slot 95): the in-step structural
     fact that B5-J's `HAS_SE -1e9` hammer was trying to approximate
   - `ADDR_B0_VALID` (L13 gather extension, slot 97): the L13 op KNOWS when
     its gather fires; emitting it as a bit lets L10 gate on freshness
   - `SP_GATHERED_THIS_STEP` (L8 gather extension at MARK_SP, slot 98):
     resolves residual leakage from MARK_STACK0 -> MARK_MEM
   With these four bits, the L10 tail-correction family shrinks from ~18
   strength-escalating rules to ~5 bounded-strength declarative dispatchers.

4. **L10 -> L13 refactor via ADDR_B0 (B5-D + B6-B partial; B7-5/B7-6
   pending).** Rule Q (`tail_mem_store_addr0_e8_from_local_frame_addr_exact`,
   `l10_ops.py:4028`) is the design exemplar: it reads `ADDR_B0_LO+8 /
   ADDR_B0_HI+14` directly with bounded strength. B4-H proved the pattern
   generalises to all five address values. The B5-D and B6-B commits added
   the helper and partial reroute; the full sweep is the B7-7 task.

5. **L17 owns L10's tail.** B6-L confirmed the layout's L17 carries
   `block.ffn = l10_post_ops_combined` + `block.post_ops[0] =
   tail_bit32_result_correction`. Future investigations into "layer 27 / 30
   / 31" should consult `vm_step.py:2215-2470` and `_expand_wrapper_blocks`
   to resolve the source factory in `l10_ops.py`.

---

## 6. What remains broken

Per the v4 retest analysis target (~792 still-failing ids):

- **Initial-SP F8 marker family** (~estimated 20-40 ids): blocked on B7-6.
  Rule A still uses `OUTPUT_LO+8 / OUTPUT_HI+15` proxy + `HAS_SE -1e9` hack.
- **MEM-store addr0 family** (~estimated 80-120 ids): blocked on B7-7.
  Rules C/E/F/G/H/K/L/M/N/O/P still strength-escalate.
- **Initial step0 -> step1 0xE0 transition** (`0cde3d3` doc): B3-alpha
  noted PSH-vs-PC ambiguity at step 0; unresolved.
- **L7/L3 attention defects** (per-op harnesses now exist via C2/C3): the
  L7 head-5 ENT-SP fetch-gate limitation (`846d8fe`) is documented but
  unresolved.
- **L4 `sp_to_addr_key` disabled-plan flag** (`f461e57`, B3-θ): flag is
  set to "absent" pending STACK0 mem-attention enablement.
- **`absdiff` (0/25) + `nested_quad` (0/16)**: dead categories per B3
  investigation (`a8924be`); unresolved root cause.
- **MUL / DIV / SHL / SHR wide-byte families**: partial fixes via
  `cc55474` (amplitude) and `b29d3ed` (alignment); regression surface
  remains.
- **PRTF / conv-I/O modes**: post-V9 design landed (`feddfc1`); pure-neural
  pass rate not yet measured on these ids.
- **ONNX export** (`7d4a5e3`): broken (cummax + dynamo guard).
- **C runtime** (`48c13a6`): build broken, no baseline.

Rough category breakdown of the ~792 failing ids:
- L10 tail-family-blocked: ~150-200 (resolved when B7-6/B7-7 land)
- Wide-byte ALU (MUL/DIV/SHL/SHR): ~120-150
- Stack/JSR/LEV protocol edges: ~80-100
- conv-I/O / PRTF: ~150-200 (not in scope this campaign)
- Dead categories (absdiff/nested_quad/etc.): ~50-80
- Long-tail (single-id regressions): ~100-150

---

## 7. Recommendations for next campaign

1. **Land B7-6 + B7-7 first.** The structural dims (B7-1..5) are allocated
   and produced; the consumer rewrites (B7-6 for `tail_sp_marker_*`, B7-7
   for `tail_mem_store_addr0_*`) deliver the architectural cleanup that
   pays off the dim-allocation cost. Expected delta: bounded-strength
   replacements drop rule count from 18 -> ~5 per family and unblock
   ~150-200 ids.
2. **Use the new per-op harnesses for L3/L7 investigation.** C2/C3 added
   harnesses for L0, L2, L4, L5, L6, L8, L9, L11, L12, L13, L14, L15, L16.
   The L7 head-5 ENT-SP fetch-gate limitation and L3 carry-forward
   attention (head 2) are now isolated-testable.
3. **Address the L4 `sp_to_addr_key` flag.** B3-θ marked it absent
   pending STACK0 mem-attention enablement. Enabling the attention path
   (currently `enable=False` gates in `l8_ops.py`) opens an alternate
   address gather pipeline that bypasses several L10 tail rules.
4. **Reclaim the 17 remaining dims at slots 99-115.** B6-K identified
   them; B7 used 4. The remaining 17 can host the full
   `SP_BYTE0_VALUE` 32-dim one-hot (per B7-4 task spec) and
   `ADDR_B0_HI_VALID` / `ADDR_B1_VALID` / `ADDR_B2_VALID` lifecycle
   bits, completing the structural-signal foundation.
5. **Investigate the dead categories.** `absdiff` (0/25) and
   `nested_quad` (0/16) imply a missing compiler path, not a tail bug.
   `a8924be` documented the surface; root cause is unidentified.
6. **Fix ONNX + C-runtime export paths.** Both are currently broken
   (`7d4a5e3`, `48c13a6`); needed for deployment validation outside the
   PyTorch reference.
7. **Re-run the 1096 diagnostic after B7-6/B7-7** to produce v4 numbers;
   that's the campaign payoff measurement.

---

## Cross-references

- B6-G L7-L9 audit: `git show origin/investigation/l7-l9-structural-audit:.agent-logs/l7-l9-structural-audit/REPORT.md`
- B6-K BD dim usage map: `git show origin/investigation/bd-dim-usage-map:.agent-logs/bd-dim-usage-map/REPORT.md`
- B6-L L17 post-op inventory: `git show origin/investigation/l17-post-op-inventory:.agent-logs/l17-post-op-inventory/REPORT.md`
- B4-H L10 refactor PLAN: `git show origin/proposal/l10-tail-correction-family:.agent-logs/l10-tail-family-refactor/PLAN.md`
- Per-op harness root: `.agent-logs/` per-batch subdirectories
- Sub-agent brief defaults: `~/.claude/projects/-home-alexlitz-Documents-misc-c4-release/memory/feedback_agent_briefs.md`
