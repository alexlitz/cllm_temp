# Post-Phase-7 regression scan (2026-06-02 07:33)

Sweep run against `speedup-cache-and-buckets` HEAD `4d99802d`
("layer10_carry_relay_bake: inline lower call (Phase 8.C
declarative_via_helper)"). Note: the branch supplied by the task
brief listed commits (`2a1fe510`, `ca6b93eb`, ...) that are NOT
present on `speedup-cache-and-buckets`. The reset and reported HEAD
above are what was actually scanned. All commands run with
`CUDA_VISIBLE_DEVICES=1`.

## 1. `analyze_scheduler.py` (Phase A scheduler diagnostic)

Tool: `c4_release/tools/analyze_scheduler.py`. Output written to
`c4_release/.agent-logs/scheduler_phase_a_2026_06_02.{md,csv}` (this
location, not under `c4_release/c4_release/.agent-logs`).

- total ops analysed: **129**
- declared block-layer count: **18**
- dependency-DAG depth: **3 layers**
- `freely_placeable`: 25
- `phase_pinned_by_deps`: 9
- `phase_required_but_undeclared`: 1
- `phase_inconsistent_with_deps`: 0
- `dep_graph_cycle_member`: **94** (4 SCCs)
- largest SCC size: **26** (`_layer11_ffn_dep_anchor`,
  `_layer12_ffn_dep_anchor`, `_layer13_attn_dep_anchor`,
  `_layer14_attn_dep_anchor`, `l10_post_ops_combined`, `layer10_alu`,
  `layer10_carry_relay`, `layer10_carry_relay_bake`,
  `layer10_stack0_byte_relay`, `layer10_stack0_byte_relay_bake`,
  ...+16 more).

### Top back-edge dims inside the largest SCC

Only three back-edge dims are reported (note: the script's "top 10"
header is misleading — the cycle dependency only surfaces these three
load-bearing dims for the current corpus):

| dim | back-edges | example |
|-----|-----------:|---------|
| `ALU_HI` | 2 | `layer10_stack0_byte_relay` → `layer9_alu`; `layer10_stack0_byte_relay_bake` → `layer9_alu` |
| `ALU_LO` | 1 | `layer16_lev_routing` → `_layer11_ffn_dep_anchor` |
| `DIV_STAGING` | 1 | `layer10_alu` → `layer6_routing_ffn` |

## 2. `census_imperative_bakes_v2.py` (per-class counts)

Tool: `c4_release/tools/census_imperative_bakes_v2.py`. Required
`PYTHONPATH=c4_release` because the tool imports `tests._per_op_audit`
(it's not on `sys.path` by default — the script's `sys.path.insert`
points one level too high). Took ~120s and ~1 GB+ RAM; an earlier
attempt was OOM-killed when run without the explicit PYTHONPATH +
unbuffered output. Reports `123` total ops:

| classification | count | post-7A (Jun 2, 05:43) | delta |
|---|---:|---:|---:|
| `declarative` | **76** | 72 | **+4** |
| `no_op` | 40 | 40 | 0 |
| `declarative_via_helper` | **1** | 5 | **-4** |
| `declarative_no_op` | 2 | 2 | 0 |
| `declarative_with_residual` | 2 | 2 | 0 |
| `imperative_trivial` | 1 | 1 | 0 |
| `imperative_heavy` | 1 | 1 | 0 |

per_kind: attn=18, block=68, ffn=12, model=25 (unchanged from
baseline). `informational_ir_count: 0`.

**Movement**: 4 ops migrated from `declarative_via_helper` to plain
`declarative` since 05:43. This matches the inlining commits visible
in the recent log (`layer10_carry_relay_bake`,
`opcode_decode_ffn`, `layer4_ffn`, ...). **Not a regression — net
declarative-coverage gain.**

## 3. `sweep_compare_ffn.py`

Tool: `c4_release/tools/sweep_compare_ffn.py`. Required
`PYTHONPATH=c4_release` for `tests._per_op_audit` import. Wrote
`c4_release/c4_release/.agent-logs/sweep_compare_ffn_phase6_wave5a.{md,json}`.
Runtime ~97 s.

```
clean=21  mismatch_only=13  real_bug=0  synthetic_state_overflow=1  other_failure=0  with_ffn_rules=35
```

Compared to `sweep_compare_ffn_post_7A.md` (clean=20,
mismatch_only=12, real_bug=0, total with_ffn_rules=32):

- +1 clean, +1 mismatch_only, +1 synthetic_state_overflow, +3
  with_ffn_rules. Net: 3 newly IR-carrying ops, 1 added clean, 1 added
  mismatch-only, 1 new overflow bucket member.
- `real_bug` stays at **0**.

Not flagging as a regression: the +1 mismatch_only and the +1
synthetic_state_overflow are both new ops entering the corpus
(not previously-clean ops degrading). Net IR coverage went up.

## 4. `sweep_compare_attn.py`

Tool: `c4_release/tools/sweep_compare_attn.py`. Wrote
`c4_release/c4_release/.agent-logs/sweep_compare_attn_phase6_wave5b.{md,json}`.
Runtime ~19 s.

```
total=89  clean=83  mismatch_only=6  real_bugs=0
```

Compared to `sweep_compare_attn_post_7A.md` (total=86, clean=81,
mismatch_only=5, real_bugs=0):

- +3 total heads, +2 clean, +1 mismatch_only.
- `real_bugs` stays at **0**.

Not a regression — additional heads entered the corpus (35 ops vs 34
post-7A); +1 mismatch_only matches +1 head expansion. **Clean rate
83/89 = 93.3%** vs prior 81/86 = 94.2%. Slight clean-rate dip
(-0.9 pp) but absolute clean count went up.

## 5. `compile_full_vm_dynamic(strict=False)` compile status

Invocation supplied by the brief returns a TUPLE `(model, layout)`,
NOT a model. Running the literal command yields:

```
AttributeError: 'tuple' object has no attribute 'parameters'
```

This is a **brief-command bug** (not a code regression). Confirmed by
unwrapping the tuple: `compile_full_vm_dynamic(strict=False)` returns
`(model, layout)` and total params = **167,655,130**.

Sub-totals from compile stdout:
- TOTAL FFN units: 57205 -> 44547 (77.9% retained)
- PHASE 0 EXPANSIONS: 14 post_ops (6 re-baked into vanilla PureFFN)
- Total blocks: 17 -> 32
- L14.ffn: 1875 -> 1874 units (-1 dead)

Cache save warning observed (cosmetic, model still returned):

```
compile_full_vm: failed to save cache .../0cac3a685c...pt
  (Can't get local object '_resolve_default_bake_fn.<locals>._ir_bake');
  returning in-memory model anyway
```

**Potential regression note**: the cache-save failure is real. If
the disk-cache pickle is required for `AutoregressiveVMRunner` startup
in production, the closure capture of `_ir_bake` will keep blocking
cache writes. Logged here for triage; pickle failure does NOT block
in-process compile (model is returned).

## 6. `pytest tests/test_smoke.py`

The command supplied (`pytest c4_release/tests/test_smoke.py`)
requires `PYTHONPATH=/home/alexlitz/Documents/misc/c4_release` because
`conftest.py` and `run_vm.py` import absolute paths
(`neural_vm.run_vm`, etc.). Even after fixing the path,
**`AutoregressiveVMRunner` is broken on this branch**:

```
====== ERROR at setup of TestSmokeBasic.test_imm_exit =========
c4_release/neural_vm/run_vm.py:329:
    self.model, _layout = compile_full_vm_dynamic(...)
.../full_vm_compiler_dynamic.py:766:
    raise StrictModeUnschedulableError(
E   StrictModeUnschedulableError: compile_full_vm_dynamic(strict=True)
    refused to compile: 107 ops cannot be placed by declared
    dependencies alone.
E     dep_graph_cycle_member (106): _layer11_ffn_dep_anchor, ... (+96 more)
E     phase_required_but_undeclared (1): putchar_think_protocol
```

Without `-x`, smoke yields: **10 passed, 1 deselected, 41 errors**.

**REGRESSION — confirmed.** Tag: `runner_strict_compile_blocks_smoke`.
The runner's `compile_full_vm_dynamic(strict=True)` call inside
`run_vm.py:329` is incompatible with the current declarative-IR
state. Either the runner should pass `strict=False` /
`allow_sealed_cycles=True`, or the cycle members listed above need
explicit `requires={"after": ...}` declarations (per
`DYNAMIC_SCHEDULER_MIGRATION_PLAN.md` units B9/B12). 41 smoke tests
currently fail to set up.

Note that the analyze_scheduler diagnostic shows **94** cycle
members under the analyzer's pruning rules, while strict compile
counts **107** ops as unplaceable (106 cycle + 1
phase_required_but_undeclared). The two tools disagree on cycle-vs-
DAG boundary detection — flagged for follow-up.

## 7. V2 contract tests

The brief listed four test files. Only two exist on this branch:

- `tests/test_phase_8h_if_eq_demo.py` — **MISSING** (lives on commit
  `30ac3f07` which is NOT on `speedup-cache-and-buckets`).
- `tests/test_lev_detector_head.py` — **MISSING**.
- `tests/test_dynamic_head_dim.py` — present.
- `tests/test_dynamic_attention_heads.py` — present.

Running only the two that exist (with
`PYTHONPATH=/home/alexlitz/Documents/misc/c4_release` since both
import `from c4_release.neural_vm...`):

```
collected 36 items
c4_release/tests/test_dynamic_head_dim.py .................. [ 50%]
c4_release/tests/test_dynamic_attention_heads.py .........  [100%]
============================== 36 passed in 0.38s ==============================
```

**36/36 pass** on the two available files. Cannot run the other two
(file not found in repo). Tag: `v2_contract_files_missing_on_branch`
— informational, not a code regression (these tests don't exist on
this branch yet).

## Regression summary

| # | check | result | regression? |
|---|---|---|---|
| 1 | analyze_scheduler | 94 cycle members, 4 SCCs, depth 3 | no — matches `scheduler_phase_a_2026_06_02.md` |
| 2 | census v2 per_class | declarative 72→76, dvh 5→1 | **no — net improvement** |
| 3 | sweep_ffn | clean 21 / mismatch 13 / real_bug 0 | no — real_bug stays 0; corpus grew |
| 4 | sweep_attn | total 89, clean 83, real_bugs 0 | no — real_bugs stays 0; corpus grew |
| 5 | dynamic compile | 167,655,130 params; cache-save failure | minor — pickle closure regression |
| 6 | smoke pytest | **41 errors** (runner strict-compile fails) | **YES — runner_strict_compile_blocks_smoke** |
| 7 | V2 contract tests | 36/36 of available 2 files pass | n/a — 2 files don't exist on branch |

### New regressions tagged

1. **`runner_strict_compile_blocks_smoke`** — `run_vm.py:329` invokes
   `compile_full_vm_dynamic(strict=True)` which raises
   `StrictModeUnschedulableError` against the current IR. 41 smoke
   tests fail to set up. Fix: either pass `strict=False` /
   `allow_sealed_cycles=True` in the runner, OR finish the cycle
   decomposition work (Phase B9/B12) so strict mode succeeds.

2. **`compile_cache_pickle_closure`** — `_resolve_default_bake_fn.<locals>._ir_bake`
   is captured as a local closure inside the cache-save path, breaking
   pickle. Compile still returns a usable in-memory model, but the
   on-disk compile cache cannot be persisted.

### Brief-command bugs (not regressions, but caller should fix)

- Step 5's literal `m = compile_full_vm_dynamic(...)` is wrong;
  the function returns `(model, layout)`. Use
  `m, _ = compile_full_vm_dynamic(strict=False)`.
- Steps 6 & 7 need `PYTHONPATH=/home/alexlitz/Documents/misc/c4_release`
  for the `c4_release.*` absolute imports in `tests/test_dynamic_*`.
- Steps 2-4 need either `cd c4_release` or
  `PYTHONPATH=/home/alexlitz/Documents/misc/c4_release/c4_release` for
  the `tests._per_op_audit` imports. The tools' `sys.path.insert`
  uses `_HERE.parents[2]` which lands ONE LEVEL TOO HIGH.
