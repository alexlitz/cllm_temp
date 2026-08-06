# Documentation Index

**Purpose**: Central index of the documentation in the C4 Neural VM project.

**Golden ledger**: default build `7d4afe61` (`C4_BP_RESTORE_HIBYTE` DEFAULT-ON — the
general-correctness fix). Rollback `069cc32f` (`C4_BP_RESTORE_HIBYTE=0`). The pure-neural
`neural_vm/` 35-token golden is `e50521f3` (all campaign flags off). Sanity-check the
default with `python -m c4_min._fingerprint_build` → `7d4afe61`. See
[FLAG_REGISTRY.md](FLAG_REGISTRY.md) / [DOOM_FLAG_REGISTRY.md](DOOM_FLAG_REGISTRY.md).

---

## Finalization (2026-08-06 capstone)

### Flagship + reproduction
- [DOOM_ON_TRANSFORMER_CAPSTONE_2026_08_06.md](DOOM_ON_TRANSFORMER_CAPSTONE_2026_08_06.md) — **★ THE capstone.** Doom (and a general C toolchain) byte-exact on a vanilla-shaped transformer; honest record of what works, the measured numbers, and the projections measurement corrected.
- [RUN_RECIPES.md](RUN_RECIPES.md) — copy-pasteable quickstart to re-run every headline result (CPU recipes verified by running).

### C-toolchain status
- [CAPSTONE_TOOLCHAIN_STATUS.md](CAPSTONE_TOOLCHAIN_STATUS.md) — which C-toolchain claims hold via the byte-exact CPU / native c4 execution (compiler + transpiler), audited against a native `./c4` oracle.
- [SELFHOST_BASELINE.md](SELFHOST_BASELINE.md) — the real self-hosting C compiler the #848/#853 transformer route targets, and whether a CPU-VM form of it exists.

### Performance
- [PERF_LADDER_FINAL.md](PERF_LADDER_FINAL.md) — definitive honest perf ladder (#841 + #849): every number measured (on stated GPU/frame) or explicitly labelled a projection.
- [`../c4_min/COMPOSED_FULL_STEP_841.md`](../c4_min/COMPOSED_FULL_STEP_841.md) — #841 composed full VM step, measured end-to-end and byte-exact vs the unfused reference, with attention/FFN/decode breakdown.

### Flags & goldens
- [DOOM_FLAG_REGISTRY.md](DOOM_FLAG_REGISTRY.md) — every `c4_min/` doom perf-fleet + pure-forward `C4_*` flag (default golden `7d4afe61` / rollback `069cc32f`).
- [FLAG_REGISTRY.md](FLAG_REGISTRY.md) — every `neural_vm/` + `tools/` `C4_*` flag (campaign config; neural_vm golden `e50521f3`).

### Doom vs general boundary
- [DOOM_VS_GENERAL_BOUNDARY.md](DOOM_VS_GENERAL_BOUNDARY.md) — where DOOM-specific work ends and the general c4-VM core begins, at the flag and module level.
- [HF_MODEL_FIT.md](HF_MODEL_FIT.md) — how `qwen_full_vm.build(...)` scales `(n_layers, hidden, intermediate)` to the op-set, and which stock HF (Qwen2) configs host which subset.
- [ARCHIVE_INDEX.md](ARCHIVE_INDEX.md) — read-only audit index of unmerged branch work (negative results, prior-session perf levers, findings/measurement branches).

---

## Architecture & spec

- [README.md](README.md) — main architecture overview
- [OPCODE_TABLE.md](OPCODE_TABLE.md) — complete opcode reference
- [BLOG_SPEC.md](BLOG_SPEC.md) — the authoritative build spec (follow exactly)
- [AGENT_CONTEXT.md](AGENT_CONTEXT.md) — contributor/agent orientation
- [PROBE_GROUNDTRUTH_2026_06_10.md](PROBE_GROUNDTRUTH_2026_06_10.md) — block↔layer map + spec_k=0 ground-truth probe
- [CAMPAIGN_SUMMARY.md](CAMPAIGN_SUMMARY.md) — the 1096-pass campaign narrative
- [CONSOLIDATION_REPORT.md](CONSOLIDATION_REPORT.md) — consolidation state of the codebase

## Declarative weight-authoring (mandatory reading before writing an op)

- [FFN_RULE_MIGRATION_PATTERN.md](FFN_RULE_MIGRATION_PATTERN.md) — FFN DSL: `FFNRule.constant_write` / `gated_write`, byte-identity gate
- [ATTENTION_HEAD_IR_MIGRATION_PATTERN.md](ATTENTION_HEAD_IR_MIGRATION_PATTERN.md) — attention DSL: `DeclarativeAttentionHeadSpec`, `AP`/`AO`
- [HOW_TO_ADD_A_CORRECTIVE_OP.md](HOW_TO_ADD_A_CORRECTIVE_OP.md) — 8-step recipe for a new corrective op
- [BUILDING_BLOCKS_DSL.md](BUILDING_BLOCKS_DSL.md) — BLOG_SPEC §504-568 → constructor mapping
- [RESIDUAL_BAND_REGISTRY_2026_06_13.md](RESIDUAL_BAND_REGISTRY_2026_06_13.md) — op-local over-width residual bands
- [DIM_OWNERSHIP_REGISTRY.md](DIM_OWNERSHIP_REGISTRY.md) — `(layer, scope, identifier, column)` claim registry
- [PHASE_7_FULLY_DYNAMIC_PLAN.md](PHASE_7_FULLY_DYNAMIC_PLAN.md) — current wave acceptance criteria

## Weight-compiler design

- [WEIGHT_SETTING_APPROACHES.md](WEIGHT_SETTING_APPROACHES.md) — hand-set vs compiled weights
- [WEIGHT_COMPILER_DESIGN.md](WEIGHT_COMPILER_DESIGN.md) — compiler design philosophy
- [WEIGHT_COMPILER_PRIMITIVES.md](WEIGHT_COMPILER_PRIMITIVES.md) — primitive operations
- [GRAPH_WEIGHT_COMPILER.md](GRAPH_WEIGHT_COMPILER.md) — graph-based compilation

## Testing & verification

- [TESTING_CHECKLIST.md](TESTING_CHECKLIST.md) — testing checklist
- [MEMORY_TEST_COVERAGE.md](MEMORY_TEST_COVERAGE.md) — memory test analysis and gaps
- [CPU_FULL_TRACE_TRUTHFUL_2026_06_17.md](CPU_FULL_TRACE_TRUTHFUL_2026_06_17.md) — the CPU self-check that reproduces GPU framing verdicts
- [CROSS_OP_ATTENTION_LINT_2026_06_17.md](CROSS_OP_ATTENTION_LINT_2026_06_17.md) — shared-head softmax regression lint
- [`../neural_vm/tests/README_MEMORY_TESTS.md`](../neural_vm/tests/README_MEMORY_TESTS.md) — memory stress test guide
- [TEST_PRUNE_MAP.md](TEST_PRUNE_MAP.md) — dead-fixture / legacy-test prune map

## Project management

- [DOCUMENT_FIXES.md](DOCUMENT_FIXES.md) — documentation fixes log

---

## Notes on this index

- This index was rebuilt on 2026-08-06 during docs finalization. It lists only docs
  whose target files exist. A prior version linked ~27 architecture files
  (`C4_NATIVE_VM.md`, `NEURAL_COMPILER.md`, `IO_ATTENTION_MECHANISM.md`,
  `KV_CACHE_EVICTION.md`, `ONNX_EXPORT.md`, `BUNDLER_GUIDE.md`, `MANDELBROT_EXAMPLE.md`,
  `SELF_HOSTING.md`, `QUINE.md`, `SPECULATIVE_DECODING.md`, …) that no longer exist in
  `docs/`; those dead links were dropped rather than left dangling. Historical
  attribution / status docs live in [`archive/`](archive/) and
  [ARCHIVE_INDEX.md](ARCHIVE_INDEX.md).
- There are ~330 dated design/attribution docs in `docs/` beyond the curated list above.
  This index curates the load-bearing entrypoints; use the finalization capstone and the
  authoring-pattern guides as the front door, and browse `docs/` (and `docs/archive/`)
  directly for the historical debugging record.

## Documentation standards

- **File naming**: `UPPERCASE_WITH_UNDERSCORES.md` for docs; `README.md` for
  directory guides.
- **Structure**: start with `# Title`; include a date and status line; use `##` for
  major sections; cross-reference related docs (and verify the link resolves).
- **Updates**: log fixes in [DOCUMENT_FIXES.md](DOCUMENT_FIXES.md); add an entry here
  when creating a new load-bearing doc; keep status markers (✅ ⚠️ ❌) consistent.
- **Golden discipline**: any docs/analysis change must leave the default golden
  `7d4afe61` unchanged (`python -m c4_min._fingerprint_build`). Any weight-affecting
  change must intend the hash it produces.

## External resources

- [C4 Compiler Original](https://github.com/rswier/c4) — original C4 implementation
- [Anthropic Research](https://www.anthropic.com/research) — transformer research
- [ONNX Documentation](https://onnx.ai/onnx/) — ONNX format reference

---

**Last Updated**: 2026-08-06 (docs finalization: index rebuilt, dead links pruned,
finalization capstone docs integrated)
**Maintainer**: See git history
