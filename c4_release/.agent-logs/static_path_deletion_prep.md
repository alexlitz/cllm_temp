# Phase 8.G — Static Path Deletion Prep (Read-only Investigation)

_Drafted 2026-06-02. Baseline: `speedup-cache-and-buckets` @ `01088f1`._

This log enumerates the work needed to delete `compile_full_vm`
(static-phase entry point) and the phase-pruning scheduler branch per
Phase 8.G acceptance criteria
(`docs/PHASE_8_PLAN.md` §3, Stream 3, sub-wave 8.G).

Scope: investigation only. No code changes.

---

## 1. Caller inventory

`grep -rn "compile_full_vm(" c4_release/` reports **99 non-doc hits**
across **47 unique files**. Per-file counts and classification:

| File | Hits | Class | Notes |
|------|-----:|-------|-------|
| `neural_vm/run_vm.py` | 3 | **Production** | `AutoregressiveVMRunner` — primary runtime entrypoint for `test_smoke`, `test_1096`, neural-decode runner. |
| `neural_vm/fast_runner.py` | 1 | **Production** | `FastVMRunner` — used by perf benches. |
| `neural_vm/transformer_first_runner.py` | 1 | **Production** | Alt runner. |
| `neural_vm/batch_runner.py` | 1 | **Production** | Batched runner (used by tools). |
| `neural_vm/batch_runner_v2.py` | 2 | **Production** | Batched runner V2 (incl. one `_cached_model` fast-path). |
| `neural_vm/cuda_graph_bench.py` | 1 | **Production** | CUDA graph perf bench. |
| `neural_vm/contracts.py` | 1 | **Production** | Contract validation entry. |
| `neural_vm/dim_registry.py` | 1 | **Production** | Dim-registry inspection helper. |
| `neural_vm/weight_setter.py` | 2 | **Production** | Deprecation shim/docstring references; legacy `set_vm_weights` redirect. |
| `neural_vm/unified_compiler/full_vm_compiler.py` | 1 | **Production** | The `def compile_full_vm` itself. |
| `neural_vm/unified_compiler/full_vm_compiler_dynamic.py` | 1 | **Production** | `compare_compile_paths(...)` calls static for byte-identity comparison. |
| `neural_vm/unified_compiler/decl_verifier.py` | 5 | **Production** | Declarative verifier mode A/B/C entrypoints. |
| `scripts/export_onnx.py` | 2 | **Production** | ONNX export script. |
| `tools/export_autoregressive.py` | 1 | **Production** | TorchScript / standalone-export tool. |
| `tools/measure_kv_eviction.py` | 2 | **Production** | KV eviction quantification harness (uses `kv_eviction_policy=`). |
| `tools/count_total_params.py` | 2 | Debug | Param accounting. |
| `tools/full_param_analysis.py` | 2 | Debug | Param breakdown. |
| `tools/simple_count.py` | 1 | Debug | Quick param count. |
| `tools/rebuild_and_test.py` | 1 | Debug | Dev rebuild helper. |
| `tools/demo_debugging_tools.py` | 1 | Debug | Demo entrypoint. |
| `debug_archive/diag_divmod_synthetic.py` | 1 | Debug | Archived diagnostic. |
| `neural_vm/tests/test_dimension_dataflow.py` | 3 | Test | Layer-compiler test. |
| `neural_vm/tests/test_dim_registry.py` | 1 | Test | Dim-registry test. |
| `neural_vm/tests/test_memory_stress.py` | 1 | Test | Memory stress. |
| `neural_vm/tests/test_opcodes.py` | 1 | Test | Opcode coverage. |
| `neural_vm/tests/test_opcodes_fast.py` | 1 | Test | Opcode coverage fast. |
| `neural_vm/tests/test_strict_neural_predictions.py` | 1 | Test | Strict-neural pred test. |
| `tests/test_compile_determinism.py` | 8 | Test | Determinism gates. |
| `tests/test_runtime_vanilla.py` | 6 | Test | Runtime invariants. |
| `tests/test_kv_eviction.py` | 6 | Test | Uses `kv_eviction_policy=`. |
| `tests/test_layer_idx_consistency.py` | 4 | Test | Layer-idx invariants. |
| `tests/test_l8_sp_gathered_sentinel.py` | 4 | Test | L8 SP sentinel test. |
| `tests/test_addr_key_neural_decode.py` | 4 | Test | Addr-key neural decode. |
| `tests/test_softmax_sharpness.py` | 3 | Test | Softmax probe. |
| `tests/test_declarative_bake_gate.py` | 3 | Test | Declarative bake gate. |
| `tests/test_alibi_mem_attn.py` | 3 | Test | Alibi memory attn. |
| `tests/test_onnx_export.py` | 2 | Test | ONNX export contract. |
| `tests/test_network_purity.py` | 2 | Test | Network-purity. |
| `tests/test_layer6_head_allocation.py` | 2 | Test | L6 head alloc. |
| `tests/test_ent_lev_neural.py` | 2 | Test | ENT/LEV neural. |
| `tests/test_enable_moe_routing.py` | 2 | Test | MoE routing. |
| `tests/test_staleness_invariants.py` | 1 | Test | Staleness scan. |
| `tests/test_l7_sp_byte0_is_f8.py` | 1 | Test | L7 SP byte0. |
| `tests/test_l1_in_step_fresh.py` | 1 | Test | L1 in-step fresh. |
| `tests/test_full_model_add_trace.py` | 1 | Test | ADD trace. |
| `tests/test_dim_ownership.py` | 1 | Test | Dim ownership. |
| `tests/test_declarative_verification.py` | 1 | Test | Declarative verification. |
| `tests/debug_l6_attention.py` | 1 | Debug | L6 attn debug. |

**Class totals:**

* **Production**: 13 files / 24 callsites.
* **Test**: 26 files / 65 callsites.
* **Debug / archived**: 8 files / 10 callsites.

Note: `neural_vm/unified_compiler/full_vm_compiler_dynamic.py:1237`
calls the static path *internally* via `_static.compile_full_vm` in
`compare_compile_paths` for byte-identity comparison; once the static
path is deleted, this comparator either (a) gets deleted with it or
(b) is rewritten to diff `compile_full_vm_dynamic` against a baked
golden artifact.

---

## 2. Kwarg gap analysis

### Static `compile_full_vm` kwargs (full_vm_compiler.py:535)

```
S, *,
enable_conversational_io,
enable_tool_calling,
enable_neural_io_think_protocol,
alu_mode,
n_heads,
ffn_hidden,
max_seq_len,
pin_io_only,
disk_cache,
use_dynamic_ffn,
enable_moe_routing,
positional_encoding,
attention_normalization,
rope_base,
use_rms_norm,
rms_norm_eps,
require_declarative_bake,
declarations_only,
kv_eviction_policy,    # UNIQUE TO STATIC
kv_eviction_n_steps,   # UNIQUE TO STATIC
```

### Dynamic `compile_full_vm_dynamic` kwargs (full_vm_compiler_dynamic.py:778)

```
S, *,
enable_conversational_io,
enable_tool_calling,
enable_neural_io_think_protocol,
alu_mode,
n_heads,
ffn_hidden,
max_seq_len,
pin_io_only,
disk_cache,
use_dynamic_ffn,
enable_moe_routing,
positional_encoding,
attention_normalization,
rope_base,
use_rms_norm,
rms_norm_eps,
require_declarative_bake,
declarations_only,
strict,                # UNIQUE TO DYNAMIC (default True; B14)
allow_sealed_cycles,   # UNIQUE TO DYNAMIC (default True)
```

### Kwargs unique to the static path

| Kwarg | Dynamic supports? | Callers using it | Migration |
|-------|-------------------|------------------|-----------|
| `kv_eviction_policy` | **No** (no plumbing yet) | `tools/measure_kv_eviction.py` (2 callsites, `KVEvictionPolicy.OFF` and `STATIC_LIVENESS`), `tests/test_kv_eviction.py` (2 callsites at lines 90 and 101). The plumbing also lives in static-compile internals: `full_vm_compiler.py:556`, `:702-703`, `:720-721`, `:859-860`, `:949-950`. | **Port required.** The dynamic path's final `_bake_from_scheduled_ops` body (`full_vm_compiler_dynamic.py:1024+`) must accept `kv_eviction_policy` + `kv_eviction_n_steps` and run the same `_attach_kv_eviction_state(...)` hook that lives in `full_vm_compiler.py:466`. The helper itself is already shared (lives in static module but the implementation is plain function — can either re-home or keep imported). Cache-key entries must mirror static (`KVEvictionPolicy(...).value`, `int(n_steps)`). Two test files + one tool to retarget. |
| `kv_eviction_n_steps` | **No** | Same as above; only `tools/measure_kv_eviction.py:356` and `tests/test_kv_eviction.py:102`. | Bundled with `kv_eviction_policy` migration. |

### Static-path-only behaviour buried in the body

These are NOT kwargs but call-site differences worth flagging for the
deletion PR:

* `_attach_kv_eviction_state(model, layout, ...)` runs at three sites
  in static (cache-hit @ 717, declarations-only cache-hit @ 856,
  post-bake @ 946). Dynamic path's `_bake_from_scheduled_ops` mirrors
  most of static but has **no** `_attach_kv_eviction_state` call.
  Confirmed via `grep -n "_attach_kv_eviction_state"
  full_vm_compiler_dynamic.py` (no hits).
* `C4_VALIDATE_ON_COMPILE` hook (static @ 960-979) is **not** mirrored
  in dynamic. The verifier-on-compile pathway must be kept (used by
  CI flags) — port to dynamic or factor into a shared helper.

### Kwargs unique to the dynamic path

| Kwarg | Static supports? | Callers using it | Migration |
|-------|------------------|------------------|-----------|
| `strict` | No (static is implicitly "phase-pruned, never strict") | Only `tests/test_compile_dynamic_strict_mode.py` (B14 admission tests). Defaults to `True` since `b038fed`. | Keep on dynamic; not a static-deletion blocker. |
| `allow_sealed_cycles` | No | Same callers as `strict`. Defaults to `True`. | Keep on dynamic; not a static-deletion blocker. |

### Caller migration table (production only)

| Caller | Kwargs used | Migratable? |
|--------|-------------|-------------|
| `run_vm.py:329` | `enable_conversational_io`, `enable_neural_io_think_protocol`, `alu_mode`, `n_heads`, `ffn_hidden`, `max_seq_len`, `enable_moe_routing` | **Yes** (all shared) |
| `fast_runner.py:43` | — (sweeps file to confirm) | Probably yes |
| `transformer_first_runner.py:53` | Same | Probably yes |
| `batch_runner.py:153` / `batch_runner_v2.py:72,379` | Same | Probably yes |
| `cuda_graph_bench.py:32` | none (positional `()`) | **Yes** |
| `dim_registry.py:1900` | none | **Yes** |
| `contracts.py:239` | none | **Yes** |
| `weight_setter.py` | (only docstring/legacy redirect references) | N/A — drop strings |
| `scripts/export_onnx.py:56` | none | **Yes** |
| `tools/export_autoregressive.py:304` | none | **Yes** |
| `tools/measure_kv_eviction.py:343,353` | `kv_eviction_policy`, `kv_eviction_n_steps`, `disk_cache` | **Blocked** on `kv_eviction_*` port to dynamic. |
| `decl_verifier.py` (5 callsites) | `S=`, `alu_mode=` typical | **Yes** (S + alu_mode are both shared) |
| `full_vm_compiler_dynamic.py:1237` (`compare_compile_paths`) | many — internal byte-identity comparator | Delete with static path or rewrite against a frozen golden artifact. |

---

## 3. Phase-pruning code in `layer_compiler.py`

`layer_compiler.py` has **87 `phase` references** across the file.
The load-bearing phase-pruning sites and what each does:

### Operation field

* **`layer_compiler.py:197`** — `phase: Optional[float] = None` field
  declaration on `Operation`. Doc rationale at lines 149-154.

### `Operation.__post_init__` deprecation hook (B15 prep)

* **`layer_compiler.py:349-379`** — `_resolve_default_bake_fn` +
  deprecation-warn hook for `phase=N.M`. Gated on
  `C4_PHASE_STRICT_MODE=1` AND `C4_PHASE_DEPRECATION_WARN=1`. Silent
  by default. Per the inline comment this block "drops entirely in
  B15" — i.e., **this is the prep-only deprecation scaffold**; 8.G can
  remove the deprecation hook (lines 355-379) when the static path
  goes.

### `_topological_sort` phase pruning (the headline)

* **`layer_compiler.py:1255-1349`** — `_topological_sort`. Phase-based
  cycle breaker at lines **1288-1296** ("u→v dropped iff
  `u.phase > v.phase`, plus same-phase attn-before-ffn"). This is the
  exact rule mirrored in `compute_dynamic_schedule._phase_prunes`
  (`full_vm_compiler_dynamic.py:226-234`). **Once the static path is
  gone**, this branch can stay in the dynamic helper or move out — but
  the inline pruning here in `_topological_sort` is shared between
  static `compile_full_vm` AND `compile_full_vm_dynamic` (the dynamic
  path still feeds ops through `LayerCompiler`), so it cannot be
  deleted standalone; it has to be replaced when the dynamic
  scheduler stops needing phase pruning (B12/B13 / 8.A.x).

* **`layer_compiler.py:1300-1330`** — `requires_after_ops` /
  `requires_same_layer_as_ops` resolution. The B9 cross-step
  exception at **1308-1323** also keys off `phase` (`ref_op.phase >
  v.phase`). Removing phase ordinal requires replacing this exception
  with explicit cross-step markers.

### `_assign_layers` phase usage

* **`layer_compiler.py:1351-1484`** — pinning + same-layer phase
  ordering. Phase consults at:
  * **1373** — `layer_phase_kinds[(layer, kind)]` tracking same-phase
    slot-sharing.
  * **1389-1402** — B9 cross-step exception for `requires["after"]`
    (mirror of `_topological_sort`).
  * **1428-1438** — same-layer-write/read allowance for pinned ops
    (writer.phase < reader.phase, or attn-then-ffn at same phase).
  * **1450-1458** — slot bumping when phases differ.
  * **1478-1483** — `writers_at_layer` updates with op's phase for
    later checks.

### Model-op phase sort

* **`layer_compiler.py:1011-1014`** — `model_ops.sort(key=lambda o:
  (o.phase if o.phase is not None else 0))`. Model ops are emitted in
  phase order after the per-layer ops. Replace with `requires=` chain
  before deletion.

### Staleness analyzer phase usage

* **`layer_compiler.py:1167-1244`** — same-step staleness check.
  Lines 1167-1179 spell out the same-step / same-phase / earlier-phase
  rule; 1190-1244 implements it. This is purely a diagnostic; can
  remain after static deletion but should switch to dep-derived
  step-locality if 8.A.5 lands and `phase` is removed corpus-wide.

### Phase 7.A.5 strict mode adaptation (B14)

* **`layer_compiler.py:964-967`** — comment confirming the staleness
  pass requires phase. Replace with dep-derived signal at 8.G.3.

### Outside `layer_compiler.py` (referenced for awareness)

* **`full_vm_compiler_dynamic.py:165-172`** — `_phase_key` (sort key
  for the dynamic schedule's tiebreaker).
* **`full_vm_compiler_dynamic.py:175-260`** — `_build_phase_pruned_graph`.
  Mirrors `_topological_sort`'s phase rule exactly. **Once `phase` is
  gone from op factories (8.G.3)**, this helper collapses to the
  unpruned dep graph and the phase-key tiebreaker disappears.

---

## 4. Scope of deletion

### LOC to delete in the static path

| Region | File:lines | LOC | Notes |
|--------|------------|----:|-------|
| `def compile_full_vm` body | `full_vm_compiler.py:535-980` | **446** | Whole function. Includes signature, docstring, kwargs handling, cache wiring, dispatch, MoE compaction, KV eviction attach (3 sites), `C4_VALIDATE_ON_COMPILE` hook. |
| Shared helpers (kept) | `full_vm_compiler.py:66-534` | 469 | NOT deleted — `_static.all_core_ops`, `_static._cache_key`, `_static._try_load_cached`, `_static._try_save_cached`, `_static._attach_kv_eviction_state`, `_static.enforce_declarative_bake_authority`, all factory `make_*` re-exports, env-flag helpers are all dependencies of the dynamic path (`full_vm_compiler_dynamic.py` imports `_static` 30+ times). |
| `compare_compile_paths` | `full_vm_compiler_dynamic.py:1237-1340` (approx) | ~100 | Calls static `compile_full_vm` for the byte-identity comparator. Either delete (loses the comparator) or rewrite against a frozen golden state-dict. |

**Net deletion if we cut `compile_full_vm` only**: ~**446 LOC** plus
roughly ~100 LOC of `compare_compile_paths` cleanup = **~550 LOC**.
The 469 LOC of shared helpers in `full_vm_compiler.py` stays (the
file shrinks to a "shared infra" module; consider renaming to
`compiler_infra.py` post-deletion).

### LOC to delete in phase pruning (`layer_compiler.py`)

| Region | Lines | LOC | Notes |
|--------|-------|----:|-------|
| `Operation.phase` field + deprecation hook | `layer_compiler.py:197`, `:349-379` | ~32 | Field declaration is 1 line; hook is ~30 lines (the deprecation scaffold is explicitly "B15 prep" — drops with this wave). |
| `_topological_sort` phase pruning | `layer_compiler.py:1288-1296` | 9 | The `if u.phase > v.phase: continue` block + same-phase attn-before-ffn check. |
| `_topological_sort` B9 cross-step exception | `:1308-1323` | 16 | Cross-step `requires["after"]` skip. Replace with explicit cross-step semantics. |
| `_assign_layers` phase logic | `:1373`, `:1389-1402`, `:1428-1458`, `:1478-1483` | ~50 | Multiple sites; same-layer phase-ordering rule. Substantial rework — pinning logic interleaves with phase, so this is **not** a clean snip. |
| Model-op phase sort | `:1011-1014` | 4 | Replace with dep order or insertion order. |
| Staleness analyzer phase usage | `:1167-1244` | ~78 | Purely diagnostic; can be rewritten in terms of dep ordering. Probably stays mostly intact but with `phase` references replaced by dep-derived step locality. |
| `LayerCompiler.compile` phase comments | `:964-967` | 4 | Stale comment after rewrite. |

**Phase-pruning total**: ~**150-200 LOC** real code change in
`layer_compiler.py` (deletion + rewrite), depending on how
`_assign_layers` is restructured.

Plus the parallel implementations in `full_vm_compiler_dynamic.py`:

| Region | Lines | LOC | Notes |
|--------|-------|----:|-------|
| `_phase_key` | `:165-172` | 8 | Delete when `_PHASE_TIEBREAK_FAR` is no longer needed. |
| `_build_phase_pruned_graph._phase_prunes` | `:226-234` | 9 | Drop the inner pruning check. |
| `_build_phase_pruned_graph` whole prune branch | `:175-260` | ~85 | Collapses to "build dep graph" when `phase` is gone — substantial simplification (delete pruning, keep edge construction). |
| `compute_dynamic_schedule` phase fallback | `:269-364` | ~95 | The cycle-fallback-to-phase branch (line 360-362 `source[name] = "phase"`) becomes dead code when SCC ≤ 10 (per 8.A acceptance) AND `phase` is gone. |

**Dynamic-side phase code**: ~**100-150 LOC** simplification.

### Total scope estimate

* **Static-path body delete**: ~446 LOC + ~100 LOC comparator = **~550 LOC**.
* **Phase-pruning rewrite**: ~150-200 LOC in `layer_compiler.py` + 100-150 LOC in `full_vm_compiler_dynamic.py` = **~250-350 LOC**.
* **`phase=` ordinals in ops factories**: **284 references** in
  `c4_release/neural_vm/unified_compiler/ops/` (333 in the broader
  `neural_vm/` tree). Per `PHASE_8_PLAN.md` 8.G.3, target is **≤ 30**.
  Net deletion: **~250 lines** across ~16 layer-op files (plus any
  `requires=` additions needed to replace each `phase=`).

**Grand total**: ~**1050-1150 LOC** of churn (delete + rewrite) for
the full 8.G wave (8.G.1 → 8.G.4).

### Callers needing migration (production)

13 production files, 24 callsites. Migration table:

| Migration class | Files | Notes |
|-----------------|------:|-------|
| Trivial (no kwargs / shared kwargs only) | 11 | Mechanical sed of `compile_full_vm(` → `compile_full_vm_dynamic(`. Includes `run_vm.py`, `fast_runner.py`, `*_runner.py`, `cuda_graph_bench.py`, `contracts.py`, `dim_registry.py`, `scripts/export_onnx.py`, `tools/export_autoregressive.py`, plus the 5 callsites in `decl_verifier.py`. |
| Blocked on `kv_eviction_*` port to dynamic | 1 (+1 test, +1 prod-tool) | `tools/measure_kv_eviction.py` and `tests/test_kv_eviction.py`. Requires (a) porting `kv_eviction_policy` / `kv_eviction_n_steps` kwargs onto `compile_full_vm_dynamic` and (b) mirroring the three `_attach_kv_eviction_state` call sites into the dynamic body. |
| Docstring / shim only | 1 | `neural_vm/weight_setter.py` — text references only. |
| Internal comparator | 1 | `full_vm_compiler_dynamic.py:1237` (`compare_compile_paths`). Delete or rewrite. |

Test callers (26 files, 65 callsites) and debug callers (8 files, 10
callsites) follow the production migration class breakdown:
~30 trivial, ~5 require `kv_eviction_*` port, the rest are
positional `compile_full_vm()` no-arg uses.

### Order of operations (per Phase 8 plan)

1. **8.G.1** — deprecation banner on `compile_full_vm` + route all
   trivial callers (the ~95% that pass only shared kwargs) to
   `compile_full_vm_dynamic`. Single PR, no semantic change.
2. **Port `kv_eviction_*` kwargs to dynamic** (precondition for the
   blocked callers above). Migrate
   `tools/measure_kv_eviction.py` + `tests/test_kv_eviction.py`.
3. **Port `C4_VALIDATE_ON_COMPILE` hook** to dynamic body (or factor
   into a shared helper). One small PR.
4. **8.G.2** — gated on 8.A landing (SCC ≤ 10). Delete `def
   compile_full_vm`, delete `compare_compile_paths`, rename
   `full_vm_compiler.py` → `compiler_infra.py` (optional).
5. **8.G.3** — strip `phase=` from op factories (≤ 30 target);
   delete phase pruning in `_topological_sort`, `_assign_layers`,
   `_build_phase_pruned_graph`, `compute_dynamic_schedule`.
6. **8.G.4** — strip redundant `layer_idx=` from op factories
   (≤ 50 target).

---

## 5. Risks (delta from `PHASE_8_PLAN.md` §5)

* **R6** (`PHASE_8_PLAN.md`) — "Deleting `compile_full_vm` breaks an
  external caller (tests, benchmark harnesses)". **65 test callsites
  + 10 debug callsites** confirm this risk is real but tractable;
  ~95% are trivial migrations. Mitigation: keep a one-release
  deprecation shim at `full_vm_compiler.compile_full_vm = compile_full_vm_dynamic`
  during 8.G.1.
* **New risk**: `compare_compile_paths` is the byte-identity gate
  between static and dynamic. Deleting the static path means the
  byte-identity comparator loses its left operand. Mitigation:
  freeze a state-dict snapshot at the last green run before 8.G.2 and
  diff future dynamic builds against that golden artifact.
* **New risk**: `_attach_kv_eviction_state` lives in
  `full_vm_compiler.py` (the to-be-shrunk module) and is called from
  three sites inside `compile_full_vm`. Moving it to dynamic must
  preserve the **post-bake / pre-cache-save** ordering (line 946 in
  static is after MoE compaction at 931-937 and before cache save at
  953-954). Tag for `kv_eviction_*` port PR.

---

_End of log._
