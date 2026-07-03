# TEST PRUNE MAP (read-only analysis)

Generated 2026-07-03, golden `b4d2ab27`. **Analysis only — no test was
deleted.** This map informs a later, careful test-suite reduction. Every
bucket below must still be re-verified (run the file, confirm it is
truly dead / redundant) before ANY deletion; several "prunable" files are
load-bearing byte-identity fixtures whose removal is COUPLED to first
deleting the dead helper they pin.

## Scope

- **`tests/` top-level:** 308 `test_*.py` files, **102,663 LOC**.
- **`tests/runners/`:** 4 files, 475 LOC (harness support — KEEP).
- **`tests/archive/`:** 1 file (`test_speculator.py`, 347 LOC — already
  archived).
- Total ≈ **108k LOC / ~313 files** (matches the ~108k / 328 estimate;
  the remainder is `conftest.py`, `__init__.py`, and fixture data).

## Bucket summary (prunable buckets ordered by LOC)

| # | Bucket | Files | LOC | Prune posture |
|---|--------|------:|----:|---------------|
| (a) | Legacy `_set_layer*` byte-identity fixtures (pinned to now-DEAD helpers) | 23 | ~11,022 | Prunable, but **COUPLED** to deleting the helpers |
| (c1) | Foreign-arch adapter tests (Qwen / Mixtral) — conditionally skipped | 13 | ~4,598 | Prune or gate behind an extra `transformers` marker |
| (c2) | ONNX / bundler / C-runtime export tests | 8 | ~3,746 | Prune down to one export smoke |
| (c3) | Conversational-IO / tool-use / think-protocol tests | 12 | ~2,903 | Consolidate to 1–2 IO suites |
| (c4) | Scratch debug / trace / inspect single-purpose tests | 18 | ~1,929 | Prune (superseded by `cpu_full_trace` + faithful interp) |
| (c5) | Demo-program tests (sudoku / quine / complex_programs) | 6 | ~1,906 | Keep one; the rest are demos |
| (c6) | `prtf` / `printf` ad-hoc print-formatting probes | 15 | ~1,251 | Consolidate to 1 prtf suite |
| **prunable subtotal** | | **~95** | **~27,355** | ~27% of `tests/` LOC |
| (d) | Core / keep (smoke, 1096, allocators, per-op declarative, verifiers, DSL, KV, dim-registry) | ~213 | ~75,300 | KEEP |

(Buckets overlap slightly in principle but the file sets above are
disjoint by filename. `_set_layer` bucket (a) and the per-op declarative
bakes in (d) are the ONLY genuinely-coupled pair — see note in (a).)

---

## (a) Legacy `_set_layer*` byte-identity fixtures — ~11,022 LOC / 23 files

These import and CALL the dead `_set_layer*` helpers in `vm_step.py` /
`setup_helpers_l*.py` to build an "expected" reference weight matrix, then
assert the declarative bake is byte-identical to it. They exist to PROVE the
Phase 6/7 imperative→declarative migration was byte-clean. Now that the model
is 100% live-declarative and `set_vm_weights()` is removed, the helpers are
orphaned dead fixtures — but these tests are the *only remaining consumers*
of those helpers.

**Prune posture:** each file is deletable EXACTLY when its pinned helper is
deleted (bucket J2, task #372). Deleting the test first loses the last
byte-identity guard on the helper; deleting the helper first breaks the test
import. They must be removed as a PAIR (helper + fixture), per layer, with the
`_isa_golden_hash.py` gate confirming `b4d2ab27` unchanged after each pair.

Top files by LOC (import/call counts from grep):

| LOC | imports | calls | file |
|----:|--------:|------:|------|
| 2095 | 0 | 1 | `test_declarative_ffn_bakes_l16.py` |
| 1269 | 10 | 10 | `test_declarative_ffn_bakes_l6.py` |
| 836 | 2 | 1 | `test_declarative_ffn_bakes_l15.py` |
| 734 | 22 | 24 | `test_l14_output_cleanup.py` |
| 665 | 1 | 1 | `test_l13_mem_addr_gather.py` |
| 594 | 2 | 1 | `test_l15_memory_lookup_isolated.py` |
| 542 | 1 | 1 | `test_l15_per_op.py` |
| 461 | 2 | 2 | `test_l12_mul_combine.py` |
| 446 | 18 | 9 | `test_declarative_attention_specs.py` |
| 434 | 3 | 4 | `test_declarative_ffn_bakes_l9.py` |
| 376 | 6 | 3 | `test_declarative_ffn_bakes_l1_l5.py` |
| 352 | 4 | 2 | `test_l8_per_op.py` |
| 351 | 8 | 8 | `test_declarative_ffn_bakes_l10_alu.py` |
| 304 | 0 | 1 | `test_l11_mul_partials.py` |
| 288 | 4 | 4 | `test_l3_pc_byte1_carry.py` |
| 234 | 0 | 2 | `test_layer6_head_allocation.py` |
| 211 | 4 | 4 | `test_declarative_ffn_bakes_l13.py` |
| 188 | 12 | 6 | `test_declarative_l10_passthrough_specs.py` |
| 173 | 1 | 3 | `test_declarative_ffn_bakes_l3.py` |
| 137 | 0 | 3 | `test_arithmetic_no_handlers.py` |
| 123 | 4 | 2 | `test_l8_l9_pc_marker_blockers.py` |
| 121 | 4 | 3 | `test_declarative_ffn_bakes_l8.py` |
| 88 | 2 | 2 | `test_mul_freshness_metadata.py` |

> NOTE: `test_smoke_pure_neural.py` references `_set_layer*` only in
> docstrings / `skip`-reasons (no import/call) — it is NOT in this bucket;
> it is a core smoke variant (KEEP, see (d)).

## (b) `@pytest.mark.skip` / `xfail` — status marker inventory (NOT its own bucket)

There is no `@pytest.mark.legacy` marker in the suite. Skip/xfail are
scattered across many otherwise-live files (a skipped case ≠ a prunable
file). Notable heavy-xfail / heavy-skip files (candidates for a per-case
audit, not blanket deletion):

- `test_smoke_pure_neural.py` — 28 xfail (documented Phase-7 memory gaps).
- `test_step_end_compute_isolated.py` — 16 xfail.
- `test_qwen_r8_e2e.py` — 9 xfail (arch-blocked; see (c1)).
- `test_onnx_runtime_1096.py` — 8 skip; `test_pure_autoregressive.py` — 9 skip.
- `test_quine.py` / `test_onnx_export.py` — 6 skip each.

Action: run `pytest --collect-only -q` + `pytest -rsx` to get the live
skip/xfail census, then decide per-case (some xfails are the intended
regression tripwires and MUST stay).

## (c) Redundant / superseded / conditionally-inert

### (c1) Foreign-arch adapters (Qwen / Mixtral) — ~4,598 LOC / 13 files
Gated on `pytest.importorskip("transformers")` / `_have_transformers_mixtral()`;
inert in the default env. Qwen R8 is documented arch-blocked (ALiBi↔RoPE,
memory `project_qwen_r8_alibi_rope_arch_blocked.md`). Files: `test_qwen_*`
(9), `test_mixtral_*` (4). Prune posture: move to an optional `foreign_arch`
suite or delete if the Qwen/Mixtral export path is abandoned.

### (c2) ONNX / bundler / C-runtime export — ~3,746 LOC / 8 files
`test_onnx_export*.py`, `test_onnx_runtime_1096.py`, `test_bundler*.py`,
`test_c4_bundler.py`, `test_c_runtime*.py`. Heavy skip counts. Keep ONE
export smoke; the rest are packaging-path variants.

### (c3) Conversational-IO / tool-use — ~2,903 LOC / 12 files
`test_conversational_io*.py` (7 variants: `_comprehensive`, `_final`,
`_proper`, `_quick`, `_manual_bytecode`, plain, `_without_`), `test_io_*`,
`test_tool_use_io.py`, `test_v18_convo_io_*`. Clear duplication — several are
iterations of the same IO harness (see the CONVERSATIONAL_IO_*.md doc churn).
Consolidate to 1–2.

### (c4) Scratch debug / trace / inspect — ~1,929 LOC / 18 files
Single-purpose probe tests: `test_*debug*`, `test_*_trace*`,
`test_what_tokens.py`, `test_when_next_thinking_end.py`, `test_token_at_113.py`,
`test_first_token_debug.py`, `test_baseline_debug.py`, `test_marker_*`,
`test_sanity_check.py`, `test_observe_smoke.py`. Superseded by
`tools/cpu_full_trace.py` + the faithful interpreter for verdict/attribution.

### (c5) Demo-program tests — ~1,906 LOC / 6 files
`test_sudoku.py`, `test_sudoku_fast.py`, `test_quine.py`,
`test_complex_programs.py` (+ program fixtures). Nice demos; keep one, the
rest are showcase.

### (c6) `prtf` / `printf` print-formatting probes — ~1,251 LOC / 15 files
`test_prtf_*` (13), `test_printf_simple.py`, `test_full_printf.py`. Many are
tiny single-assert scratch files (`test_prtf_simple_run.py` 38 LOC,
`test_prtf_with_debug.py` 48 LOC). Consolidate to one `test_prtf.py`.

## (d) Core / KEEP — ~75,300 LOC

Do NOT prune. Load-bearing gates + contracts:

- **Smoke:** `test_smoke.py` (1375), `test_smoke_pure_neural.py` (823),
  `test_pure_neural_smoke_ratchet*.py`, `test_observe_smoke.py`.
- **1096 corpus:** `test_1096_neural_declarative_diagnostic.py` (1396),
  `test_1096_teacher_forced_lowering_audit.py` (981),
  `test_suite_1096_pure_neural_pytest.py`, `test_suite_1096_pytest.py`,
  `test_suite_1000.py`, `test_c_runtime_1096.py`, `test_bundler_1096.py`.
- **Allocator contracts:** `test_dim_allocator.py`,
  `test_ffn_unit_allocator.py`, `test_attention_head_allocator.py`,
  `test_compile_with_allocators.py` (allocator contract tests named in
  CLAUDE.md).
- **Per-op declarative bakes** (the live-migration regression tests):
  `test_l{0..16}_per_op.py`, `test_l10_tail_correction.py` (7666 — the single
  largest test file; the tail-corrector byte-identity oracle),
  `test_l10_post_op*.py`, `test_alu_wide_composites_per_op.py`,
  `test_comparison_combine_per_op.py`, `test_add_sub_byte_postops.py`.
  (These OVERLAP conceptually with (a) but do NOT import `_set_layer`; they
  gate the declarative bake directly — KEEP.)
- **DSL / IR / verifier:** `test_isa_semantics_dsl.py`,
  `test_wide_alu_dsl.py`, `test_building_blocks_dsl.py`, `test_compiler_ir.py`,
  `test_ir_types.py`, `test_declarative_verification.py`,
  `test_decl_*`/`test_declarative_*` verifier tests, `test_contribution_algebra.py`,
  `test_predicates_*`, `test_lint_*` (the ratchet lints incl.
  `test_lint_cross_op_attention.py`, `test_lint_raw_ffn_rule.py`,
  `test_dim_resolver.py`).
- **Dim registry / semantic refs (Phase 7.E):** `test_dim_registry_categories.py`,
  `test_dim_semantics_*.py`, `test_phase_7_e_2_dim_ref_migration.py`,
  `test_dim_oracle.py`, `test_dim_alias_verifier.py`, `test_residual_band_registry.py`.
- **KV cache / eviction:** `test_kv_*` (~10 files) — the KV-eviction gate suite.
- **Purity / structural:** `test_network_purity.py`, `test_structural_guarantees.py`,
  `test_staleness_invariants.py`, `test_compile_dynamic_byte_identical.py`,
  `test_compile_flag_parity.py`, `test_architecture_toggles.py`.

## Recommended prune sequence (later work)

1. **(c4) + (c6) scratch/prtf** — safest, self-contained, superseded by
   `cpu_full_trace`. ~3.2k LOC.
2. **(c3) conversational-IO** consolidation — ~2.9k LOC, clear duplication.
3. **(c1) + (c2) foreign-arch + export** — gate behind optional markers or
   delete if those paths are abandoned. ~8.3k LOC.
4. **(a) `_set_layer` fixtures** — LAST, and only PAIRED with helper deletion
   (task #372, per layer, `_isa_golden_hash.py` gate each pair). ~11k LOC
   test + the ~9k LOC of dead helpers together.

Each step re-runs `pytest tests/test_smoke.py` + `tools/_isa_golden_hash.py`
(must stay `b4d2ab27`) before the next.
