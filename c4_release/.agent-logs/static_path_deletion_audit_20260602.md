# Phase 8.G — Static Path Deletion Audit (2026-06-02)

_Baseline: `speedup-cache-and-buckets` @ `9a83de0c`._
_Predecessor doc: `static_path_deletion_prep.md` (2026-06-02, `a241b2d1`)._

Read-only audit; no code changes. Focused on the **current state** after
Phase 8.G.3 already deleted the static body. The remaining question for
V5 is: *what still keeps the `compile_full_vm` entry-point name alive*,
and which call sites can be flipped to `compile_full_vm_dynamic`
directly so we can eventually retire the redirect?

---

## 1. Current state of `full_vm_compiler.py`

`compile_full_vm` is **already a thin redirect** as of Phase 8.G.3
(commit `a241b2d1` "docs: static path deletion prep"). The static
phase-pruning body and `compare_compile_paths` are gone. See:

* `c4_release/neural_vm/unified_compiler/full_vm_compiler.py:629-789`
  — function signature plus an unconditional
  `compile_full_vm_dynamic(...)` delegation. No SCC / phase logic
  remains in this file.
* `c4_release/neural_vm/unified_compiler/full_vm_compiler.py:760-766`
  — explicit comment confirming the redirect lives on only for
  backward-compat with the external API; the static body was deleted
  in 8.G.3.
* `c4_release/neural_vm/unified_compiler/full_vm_compiler_dynamic.py:835`
  — same confirmation from the dynamic side: "the historical
  byte-identity gate (`compare_compile_paths`) was removed alongside
  the static body".
* `c4_release/neural_vm/unified_compiler/full_vm_compiler_dynamic.py:1124,1173,1240`
  — three call sites that attach KV-eviction state inside the dynamic
  bake driver (`_static._attach_kv_eviction_state`). This is the
  kwarg-port the prep doc flagged as a precondition for static
  deletion; it has landed.

What still lives in `full_vm_compiler.py`:

| Region | Lines | Role |
|--------|------:|------|
| Module docstring | 1-31 | Project-level intro. |
| Env-flag constants + `_TRUTHY_ENV_VALUES` | 60-63 | Reused by dynamic via `_static._env_flag_enabled`. |
| `DeclarativeBakeRequirementError`, `DeclarativeBakeAuthorityReport`, `_looks_like_wrapper_model_op`, `inspect_declarative_bake_authority`, `enforce_declarative_bake_authority` | 66-186 | Authoritative declarative-bake gate. Imported by `_bake_from_scheduled_ops` in the dynamic path. |
| `detect_staleness_violations`, `build_staleness_registry` re-exports | 194-210 | Thin wrappers around `LayerCompiler`. |
| `migrated_ops` re-export block | 211-233 | Re-exports `all_core_ops`, `declare_setdim_compat_dims`, `make_l10_*`, `make_l11_*`, `make_l12_*`, etc. Several tests import these via `full_vm_compiler`. |
| `derive_layout` | 236-251 | Layout-only helper. |
| `_CACHE_FORMAT_VERSION`, `_UNPICKLABLE_OP_FIELDS`, `_strip_*`, `_cache_dir`, `_hash_source_bytes`, `_cache_key`, `_try_load_cached`, `_try_save_cached` | 254-463 | Disk-cache machinery. Used by `_bake_from_scheduled_ops` (dynamic path) via `_static._cache_key`, `_static._try_load_cached`, `_static._try_save_cached`. |
| `_attach_kv_eviction_state`, `_iter_attention_heads_in_op` | 466-626 | KV-eviction state attach. Three call sites in dynamic path. |
| `compile_full_vm` redirect | 629-789 | The only callable that still uses the historical name. |

`full_vm_compiler.py` is now **"shared infra + entry-point alias"**.
The prep doc's suggested rename to `compiler_infra.py` is the natural
next step once the redirect is gone.

---

## 2. Call-site catalog (current, post-`a241b2d1`)

`grep -rn "compile_full_vm(" c4_release/ --include="*.py" | grep -v
"_dynamic"` reports **105 hits across 49 unique files** (vs. the prep
doc's 99 / 47 — three test files were added since: `test_kv_eviction_gates.py`,
`test_kv_eviction_long_context.py`, `test_l10_post_op_attach.py`,
`test_l10_post_ops_combined_per_op.py`; one new tool added; some hits
are doc-string mentions).

`grep -rn "from .*full_vm_compiler import"` (excluding `_dynamic`)
reports **48 unique importers**.

`grep -rn "compile_full_vm_dynamic("` reports importers in only
**4 production-style files** so far: `tests/test_compile_dynamic_strict_mode.py`,
`tests/test_compile_dynamic_byte_identical.py`,
`tests/test_compile_with_allocators.py`, plus the redirect itself.

### Classification

Every call site below routes through the redirect and so already runs
on the dynamic compiler. The question is whether it can be **flipped
to `compile_full_vm_dynamic`** (a) trivially, (b) after one shared
helper is re-homed, or (c) only when something heavier lands.

#### Trivial flip (no kwargs / only shared kwargs)

These can change `compile_full_vm` → `compile_full_vm_dynamic` with a
mechanical edit. **31 files / ~70 call sites.** Examples:

| File:line | Kwargs |
|-----------|--------|
| `c4_release/neural_vm/run_vm.py:329` | `enable_conversational_io`, `enable_neural_io_think_protocol`, `alu_mode`, `n_heads`, `ffn_hidden`, `max_seq_len`, `enable_moe_routing` |
| `c4_release/neural_vm/fast_runner.py:43` | shared kwargs |
| `c4_release/neural_vm/transformer_first_runner.py:53` | shared kwargs |
| `c4_release/neural_vm/batch_runner.py:153` | shared kwargs |
| `c4_release/neural_vm/batch_runner_v2.py:72,379` | shared kwargs + `_cached_model` fast path at :379 |
| `c4_release/neural_vm/cuda_graph_bench.py:32` | positional `()` |
| `c4_release/neural_vm/dim_registry.py:1944` | positional `()` |
| `c4_release/neural_vm/contracts.py:239` | positional `()` |
| `c4_release/scripts/export_onnx.py:56` | positional `()` |
| `c4_release/tools/export_autoregressive.py:304` | positional `()` |
| `c4_release/tools/simple_count.py:13` | shared kwargs |
| `c4_release/tools/count_total_params.py:12,18` | `alu_mode=...` |
| `c4_release/tools/full_param_analysis.py:9,37` | `alu_mode=...` |
| `c4_release/tools/demo_debugging_tools.py:84` | positional `()` |
| `c4_release/tools/rebuild_and_test.py:17` | positional `()` |
| `c4_release/neural_vm/unified_compiler/decl_verifier.py:1956,2498,2864,3069,3206` | `S=`, `alu_mode=`, `disk_cache=False` |
| `c4_release/neural_vm/tests/test_dimension_dataflow.py:114,130,142` | positional `()` |
| `c4_release/neural_vm/tests/test_dim_registry.py:298` | positional `()` (in-body) |
| `c4_release/neural_vm/tests/test_memory_stress.py:37` | positional `()` |
| `c4_release/neural_vm/tests/test_opcodes.py:142` | positional `()` |
| `c4_release/neural_vm/tests/test_opcodes_fast.py:55` | positional `()` |
| `c4_release/neural_vm/tests/test_strict_neural_predictions.py:40` | positional `()` |
| `c4_release/tests/test_addr_key_neural_decode.py:49,70,194` | positional `()` |
| `c4_release/tests/test_alibi_mem_attn.py:47,65` | positional `()` |
| `c4_release/tests/test_compile_determinism.py:40,41,54,55,65,66,76` | `alu_mode=`, `enable_conversational_io=`, `enable_tool_calling=` |
| `c4_release/tests/test_declarative_bake_gate.py:148,158,277` | `disk_cache=False`, `declarations_only=True` |
| `c4_release/tests/test_declarative_verification.py:494` | `S=`, `alu_mode=`, `enable_conversational_io=`, `n_heads=` |
| `c4_release/tests/test_dim_ownership.py:438` | positional `()` |
| `c4_release/tests/test_ent_lev_neural.py:76,243` | `n_heads=`, `ffn_hidden=` |
| `c4_release/tests/test_enable_moe_routing.py` (2 hits) | env-flag-driven `enable_moe_routing` |
| `c4_release/tests/test_full_model_add_trace.py:360` | shared kwargs |
| `c4_release/tests/test_l1_in_step_fresh.py:194` | positional `()` |
| `c4_release/tests/test_l7_sp_byte0_is_f8.py` (1 hit) | positional `()` |
| `c4_release/tests/test_l8_sp_gathered_sentinel.py:171,243` | `disk_cache=False`, `S=100.0` |
| `c4_release/tests/test_layer6_head_allocation.py:51,147` | positional `()` |
| `c4_release/tests/test_layer_idx_consistency.py:115,163` | positional `()` |
| `c4_release/tests/test_network_purity.py:31,258` | `n_heads=`, `ffn_hidden=` |
| `c4_release/tests/test_onnx_export.py:67,101` | `n_heads=`, `ffn_hidden=` |
| `c4_release/tests/test_runtime_vanilla.py:143,148,153` | `alu_mode=`, `enable_conversational_io=` |
| `c4_release/tests/test_softmax_sharpness.py:622,723` | positional `()` |
| `c4_release/tests/test_staleness_invariants.py:316` | positional `()` |
| `c4_release/tests/debug_l6_attention.py:17` | positional `()` |
| `c4_release/debug_archive/diag_divmod_synthetic.py:22` | positional `()` |

#### Blocked-on-strict (would surface admission failures)

`compile_full_vm_dynamic` defaults to `strict=True, allow_sealed_cycles=True`.
The redirect at `full_vm_compiler.py:767` does NOT pass `strict=` so
callers inherit the dynamic default. **So today every redirect call
already runs the strict admission gate.** That means flipping the call
sites is byte-identical to what already happens behind the redirect,
which is the load-bearing observation: the `phase=` tiebreaker / SCC
blocker called out in the brief is **NOT a redirect-flipping blocker**
— it is a `phase=`-field-deletion blocker (the next sub-wave 8.G.3).

What WOULD break if a caller flipped to `compile_full_vm_dynamic`
WITHOUT changing kwargs:

* `compile_full_vm_dynamic` accepts a `strict=` kwarg the static name
  doesn't. Callers that want strict-off must pass `strict=False`
  explicitly. Today only `tests/test_compile_dynamic_strict_mode.py`
  exercises this kwarg, so flipping any other caller silently
  enables strict (which is already enabled at the redirect, so no
  behavioural change).

#### KV-eviction call sites — formerly blocked, now unblocked

Per the prep doc, the KV-eviction kwargs were the gating concern. They
have been ported to dynamic (`full_vm_compiler_dynamic.py:803-804,
1107-1108, 1124-1128, 1173-1177, 1240-1244`), so:

| File:line | Notes |
|-----------|-------|
| `c4_release/tools/measure_kv_eviction.py:403,413` | `kv_eviction_policy=KVEvictionPolicy.OFF` / `STATIC_LIVENESS`. Flip is now safe. |
| `c4_release/tests/test_kv_eviction.py:81,89,99,117` | `kv_eviction_policy=`, `kv_eviction_n_steps=` exercised. Flip is now safe. |
| `c4_release/tests/test_kv_eviction_gates.py:119,129,140,492` | Same. |
| `c4_release/tests/test_kv_eviction_long_context.py:246,265` | Same. |

#### Sites that import shared helpers (NOT `compile_full_vm`)

These do not block deletion of the redirect; they import non-redirect
symbols. They'd still need `full_vm_compiler.py` (renamed
`compiler_infra.py`) on disk after the redirect goes. **5 files:**

| File:line | Imported symbols |
|-----------|------------------|
| `c4_release/tests/test_l10_post_op_attach.py:5-7` | `declare_setdim_compat_dims` (re-export) |
| `c4_release/tests/test_l10_post_ops_combined_per_op.py:52-54` | `declare_setdim_compat_dims` (re-export) |
| `c4_release/tests/test_staleness_invariants.py:310-312` | `compile_full_vm` (would flip with others) |
| `c4_release/tests/test_dim_ownership.py:431-433` | `compile_full_vm` (would flip with others) |
| `c4_release/tests/test_declarative_verification.py:491-493` | `compile_full_vm` (would flip with others) |

`declare_setdim_compat_dims` re-exported from `migrated_ops.py:213-233`
in `full_vm_compiler.py` is the load-bearing shared symbol — its real
definition is at `ops/shared.py:630`. Test imports could be retargeted
to `migrated_ops` or `ops.shared` to drop their dependence on
`full_vm_compiler` entirely.

#### Doc-string-only / shim mentions

* `c4_release/neural_vm/weight_setter.py:114, 120` — text references
  in the `set_vm_weights` deprecation shim. Drop strings when the
  redirect goes.
* `c4_release/tests/test_addr_key_neural_decode.py:7`, `test_alibi_mem_attn.py:7`,
  `test_l8_sp_gathered_sentinel.py:15,232`, `test_layer_idx_consistency.py:22,25`,
  `test_softmax_sharpness.py:426`, `test_runtime_vanilla.py:142,147,152`,
  `test_kv_eviction.py:3,79,131`, `run_vm.py:292,342`, `scripts/export_onnx.py:79`,
  `decl_verifier.py:42,1677,1804,1964,2466,2506`,
  `unified_compiler/full_vm_compiler_dynamic.py:810,812,832,833` —
  all docstring / comment references. Rewrite during the redirect-removal PR.

---

## 3. Low-hanging deletions (independent of any SCC / phase blocker)

### 3.1 Archived / unused

* `c4_release/debug_archive/diag_divmod_synthetic.py:22` — single
  positional `compile_full_vm()` call in `debug_archive/`. The folder
  is unreferenced by tests, runners, or tools (`grep -rn
  "debug_archive" c4_release/ --include="*.py"` returns no
  non-self hits in production code). **Safe to delete the file** or
  flip the import as part of a corpus-wide sed. No SCC dependency.

### 3.2 Debug / one-off tools

The following all live under `c4_release/tools/` and are
positional-`compile_full_vm()` calls with no shared state:

* `c4_release/tools/simple_count.py:13`
* `c4_release/tools/count_total_params.py:12,18`
* `c4_release/tools/full_param_analysis.py:9,37`
* `c4_release/tools/rebuild_and_test.py:17`
* `c4_release/tools/demo_debugging_tools.py:84`

These can flip to `compile_full_vm_dynamic` in a single mechanical
PR. **Tool-only churn, no test coverage to revalidate beyond running
the tools.** ~6 files, ~10 LOC.

### 3.3 Doc-string-only references (no callsite)

* `c4_release/neural_vm/weight_setter.py:114, 120` — drop string
  references in the `set_vm_weights` deprecation message. **2 LOC.**
* `c4_release/scripts/export_onnx.py:79` — print statement
  references `compile_full_vm()` by name. **1 LOC.**

### 3.4 The redirect itself

`c4_release/neural_vm/unified_compiler/full_vm_compiler.py:629-789`
(161 LOC including the docstring). Independent of any SCC blocker —
deletion is gated only on the corpus-wide caller flip.

---

## 4. Blockers (per caller class)

| Class | Files | Blocker | Estimated work |
|-------|------:|---------|----------------|
| Trivial flip (positional / shared kwargs) | 43 files / ~100 callsites | None | 1 mechanical PR, ~100 LOC of edits. |
| KV-eviction callers | 4 files / 10 callsites | None (port already landed) | Same PR. |
| Doc-string-only references | ~14 files / ~30 mentions | None | Same PR. |
| Imports of re-exported helpers | 2 files | Pick: leave `full_vm_compiler.py` as `compiler_infra.py`, or retarget imports to `ops/shared.py` / `migrated_ops.py` | 5-10 LOC. |
| Redirect deletion | 1 file | All callers flipped | ~160 LOC delete. |

**Nothing in the audit blocks on SCC > 0 or on the `phase=` tiebreaker.**

The brief flagged "phase= tiebreaker blocked by SCC>0" as a blocker for
8.G.1 prep. After the audit: the SCC/phase blocker is for **8.G.3
phase= field deletion** (the prep doc's section 3, lines 188-269), not
for the redirect removal that 8.G.1 actually targets. The redirect is
already byte-identical to the dynamic path because the static body is
already gone; flipping the named entry point doesn't change scheduler
behaviour.

---

## 5. Recommended order

1. **PR A (corpus-wide sed, no behaviour change)** — flip all 43
   trivial / KV-eviction callers from `compile_full_vm` to
   `compile_full_vm_dynamic`. Update docstrings. Leave the redirect in
   place for one release as deprecation cover.
   * Touch: ~50 production / test / tool files, ~150 LOC.
   * Validation: full test suite (declarative bake gate, runtime
     vanilla, KV eviction, compile determinism, dim ownership,
     staleness invariants).

2. **PR B (re-export retargeting)** —
   `test_l10_post_op_attach.py`, `test_l10_post_ops_combined_per_op.py`
   retargeted to import `declare_setdim_compat_dims` from
   `ops/shared.py` (or keep `migrated_ops`) directly. Drops the last
   non-redirect import of `full_vm_compiler`.
   * Touch: 2 test files, ~4 LOC.

3. **PR C (redirect deletion + module rename)** — delete
   `compile_full_vm` (lines 629-789), rename `full_vm_compiler.py` to
   `compiler_infra.py`, update imports. Independent of the SCC blocker.
   * Touch: 1 module rename, ~160 LOC delete.

4. **PR D and later (8.G.3 phase= deletion)** — gated on SCC ≤ 10 per
   `PHASE_8_PLAN.md`. Distinct work from the redirect removal.

---

## 6. Findings summary

* `compile_full_vm` already has no static logic — it's purely a
  named-entry-point alias for `compile_full_vm_dynamic`.
* `compare_compile_paths` is already gone.
* KV-eviction kwargs already ported to dynamic, with three
  `_attach_kv_eviction_state` call sites mirroring the historical
  static layout.
* **Zero call sites are blocked on phase= / SCC.** The brief's
  premise that 8.G.1 prep is blocked is no longer accurate as of
  `a241b2d1`.
* ~43 caller files can be mechanically flipped today; 2 files import
  re-exported helpers and need a small retarget; the rest are
  doc-string mentions.
* Independent low-hanging deletions: `debug_archive/diag_divmod_synthetic.py`,
  6 debug/perf tools, doc-string references in `weight_setter.py` /
  `scripts/export_onnx.py`.

_End of audit._
