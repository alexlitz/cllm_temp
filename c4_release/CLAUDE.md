# CLAUDE.md — c4_release project notes

Project-specific guidance for Claude agents working in `c4_release/`.
Pairs with the global `~/.claude/CLAUDE.md` and the user's auto-memory.
The README in `docs/README.md` is the public-facing project overview;
this file is the contributor-facing one.

## Architecture in one paragraph

The neural VM is a transformer (**27 logical layers**, `d_model=872`,
`vocab=276`; head count varies per block) whose weights compute the full
C4 instruction set deterministically. Logical layers expand to **37
physical blocks** post-`_expand_wrapper_blocks` (each logical layer's
`post_ops` become passthrough blocks): L8 +1, L14 +8, L25 +1. For the
authoritative block↔layer map and the spec_k=0 ground-truth probe, see
[`docs/PROBE_GROUNDTRUTH_2026_06_10.md`](docs/PROBE_GROUNDTRUTH_2026_06_10.md)
and [`tools/probe_groundtruth.py`](tools/probe_groundtruth.py). Every
opcode (LEA, IMM, JMP, JSR, BZ, BNZ, ENT, ADJ, LEV, LI, SI, PSH,
OR/XOR/AND, EQ/NE/LT/GT/LE/GE, SHL/SHR, ADD/SUB/MUL/DIV/MOD, EXIT) is
implemented by hand-authored weights, lowered from declarative IR
(`FFNRule` + `DeclarativeAttentionHeadSpec`). The "fix-a-bug" workflow
is: identify the residual drift → write the smallest declarative spec
that pulls it back → byte-identity gate it → commit.

## Declarative IR is the primary weight-authoring path

All ops ARE declarative. The model is now **100% live-declarative**:
every production weight is written by the IR lowering pipeline
(`compile_full_vm_dynamic`), and a `settrace` census confirms **zero**
live `_set_layer*` imperative writers on the build path. The legacy
`vm_step.set_vm_weights()` entry point has been REMOVED (see
`weight_setter.py:111`); the `_set_layer*` functions still present in
`vm_step.py` / `setup_helpers_*.py` are now orphaned dead fixtures kept
only for their docstring cross-references and legacy byte-identity tests
(prune target — see [`docs/TEST_PRUNE_MAP.md`](docs/TEST_PRUNE_MAP.md)).
Imperative `block.ffn.W_*[...] = X` or `attn.W_q[...] = Y` writes are
legacy and MUST NOT be added. The pattern guides are mandatory reading
before writing any op:

- [`docs/FFN_RULE_MIGRATION_PATTERN.md`](docs/FFN_RULE_MIGRATION_PATTERN.md)
  — FFN DSL: `FFNRule.constant_write` / `FFNRule.gated_write`,
  `conditions` / `threshold` / `gate*` / `writes`, `scope` and
  `dominates_at` predicate language, byte-identity gate via
  `compare_symbolic_to_lowered_ffn`. Worked example:
  `_layer14_temp_clear_rules`.
- [`docs/ATTENTION_HEAD_IR_MIGRATION_PATTERN.md`](docs/ATTENTION_HEAD_IR_MIGRATION_PATTERN.md)
  — Attention DSL: `DeclarativeAttentionHeadSpec` with `AP` /
  `AO` constructors, per-layer `_LN_HEAD_LAYOUT` tables, allocator-pinned
  `head_idx`, alibi-slope handling. Worked example:
  `_layer7_memory_head_specs`.
- [`docs/HOW_TO_ADD_A_CORRECTIVE_OP.md`](docs/HOW_TO_ADD_A_CORRECTIVE_OP.md)
  — short opinionated recipe (8 steps) for a new corrective op without
  reproducing the full DSL.
- Building-blocks DSL: `step_function_rule`, `one_hot_indicator_rule`,
  `multi_way_and_rule`, `band_range_check_rules`, `cancel_residual_rule`,
  `lookup_table_rules`, `multi_way_or_rules` from
  `neural_vm/unified_compiler/building_blocks_dsl.py`. Used by every
  layer's FFN bakes (V1-V7 migration). Tests:
  `tests/test_building_blocks_dsl.py`. New op code MUST go through one
  of these helpers; raw `FFNRule.constant_write` / `gated_write` calls
  outside the DSL modules are ratcheted by
  [`tools/lint_raw_ffn_rule.py`](tools/lint_raw_ffn_rule.py).
  See [`docs/BUILDING_BLOCKS_DSL.md`](docs/BUILDING_BLOCKS_DSL.md) for
  the BLOG_SPEC §504-568 → constructor mapping.

Key IR types (all in `neural_vm/unified_compiler/ir.py`):

| Type                                | Role                                                              |
|-------------------------------------|-------------------------------------------------------------------|
| `FFNRule`                           | One declared FFN hidden unit (`constant_write` or `gated_write`). |
| `FFNOp`                             | One layer's collection of `FFNRule`s.                             |
| `DeclarativeAttentionHeadSpec`      | One attention head's Q/K/V/O writes (in `primitives.py`).         |
| `AttentionHeadIR` / `AttentionOp`   | Per-layer head bundle inside the IR.                              |
| `RuntimeAttentionFragment`          | Imperative attention fragment with a runtime `(attn) -> bool` predicate. Used for shape-dependent heads (e.g. L15 `memory_lookup` branches on `attn.num_heads`). |
| `TokenEmbeddingRule`                | Per-token write into `model.embed.embed.weight`, `model.head.weight`, or `model.head.bias`. Used by `head_bake` / `embedding_bake` / model-level bakes (Phase 7.D). |
| `CompilerIR` / `LayerSpec`          | Top-level container. Carries layers + embedding rules.            |

The lowering pipeline (`CompilerIR.lower_ffn`, `CompilerIR.lower_attention`,
`CompilerIR.lower_token_embeddings`) is the only path that should write
weights. The dispatcher in
`neural_vm/unified_compiler/layer_compiler.py:_dispatch_operation_ir`
picks lowering targets from `op.kind` (`"ffn"`, `"attn"`, `"block"`,
`"model"`).

## Allocators

Three first-fit allocators replace hand-picked layer/unit/head/slot
indices. Each accepts `pin=<idx>` for byte-identity-preserving
migrations, or `pin=None` once the layer's Phase 7.B wave is in.

- [`neural_vm/dim_allocator.py`](neural_vm/dim_allocator.py) — residual
  dim slots. Paired with `dim_registry_dynamic.py`. Wider models pass
  `d_model=`.
- [`neural_vm/ffn_unit_allocator.py`](neural_vm/ffn_unit_allocator.py)
  — per-layer FFN hidden units. Default budget `DEFAULT_LAYER_MAX_UNITS
  = 4096`. Op chains share a layout table (see `_L14_CLEANUP_CHAIN_LAYOUT`
  in `l14_ops.py` for the pattern); the per-op `_l14_chain_alloc`
  helper routes the static pin through a fresh allocator so the
  structure is auditable.
- [`neural_vm/attention_head_allocator.py`](neural_vm/attention_head_allocator.py)
  — per-layer head indices. `DEFAULT_LAYER_MAX_HEADS = 8`. Never
  aliases — two ops writing the same `head_idx` in the same layer is a
  hard error.

## Op-local residual bands (over-width families)

An op that needs a FRESH residual band past the natural d_model (an
"over-width" family like the AX byte-1 carry, the Root 2 STACK0 byte-0
carry, or the width=2 MUL result) declares it **op-locally** — never by
hand-editing a shared dict. At the top of the op's `lN_ops.py` module:

```python
from .residual_band_registry import register_residual_band
register_residual_band("MY_BAND", 7, owner="make_my_op",
                       flag=None, never_share=True)
```

[`ops/residual_band_registry.py`](neural_vm/unified_compiler/ops/residual_band_registry.py)
populates a module-level registry at import time;
`compile_full_vm_dynamic` AUTO-COLLECTS every active band
(`collect_registered_residual_bands`) and feeds the union into the SAME
head-dim-preserving auto-widen + cache keys + `_LIVENESS_NEVER_SHARE`. A
`flag=<zero-arg predicate>` band is collected only when the flag is on
(flag-off → byte-identical smaller d_model); a `never_share=True` band
keeps a private dim-liveness slot. **A new band-adding op never touches
`full_vm_compiler_dynamic.py` or `layer_compiler.py`** — this is what
eliminates the cross-lane merge conflict. Full API + the legacy
`_PRODUCTION_EXTRA_RESIDUAL_DIMS` migration:
[`docs/RESIDUAL_BAND_REGISTRY_2026_06_13.md`](docs/RESIDUAL_BAND_REGISTRY_2026_06_13.md).

## Verification + sweep tools

Run these BEFORE committing any new op or rule change:

- **`tools/_isa_golden_hash.py` — the authoritative flag-OFF byte-identity
  gate.** The current golden (default build) is `state_dict_sha256 =
  81557d21422f3eada0a87c677b00dced41cc26c3ee3bfb094c5eeb71c9b4d3cb`
  (short `81557d21`) after the STACK0-b0 dead-dump-machinery deletion
  (2026-07; verdict-neutral geometry cut, was `b4d2ab27`). Any
  docs/analysis change must leave this unchanged; any weight-affecting
  change must intend the hash it produces. Use `81557d21` for all future
  flag-OFF byte-identity checks (historical `b4d2ab27` builds predate the
  STACK0-b0 dump deletion).
- **`tools/lint_dim_resolution.py` — MANDATORY ratchet for any tool/op that
  resolves a residual-dim POSITION.** The blessed resolution path is
  `neural_vm.unified_compiler.dim_resolver.DimResolver` over the BUILT
  `layout.dim_positions` (NOT the static registry, which the widen-repack
  moves ~93% of dims off of — the documented false-"dead-signal" trap). The
  lint flags static-registry `.start` / `.resolve_dim()` / `.slots[...]`
  positional reads; opt out per-line with `# dim-resolution-lint: allow`.
- **`tools/lint_semantic_dim_ref.py` — Phase 7.E authoring ratchet (companion
  to `lint_dim_resolution.py`).** Flags a RAW role-meaningful base-slot NAME
  in ops `FFNRule` authoring (a `(NAME, weight)` condition / gate / write
  tuple whose NAME is one of the 78 category-registered slots — markers,
  `byte_index`, `output_lo/hi`, `alu_*`, `ax_carry_*`, `cmp_flag`, memory
  families, `opcode_flag`, tagged STACK0 bands) instead of authoring it via
  `dim_registry.dim_ref(category, role, offset)`. `reads=`/`writes=` metadata
  SETs, `dp["NAME"]` attention resolvers, and unbound/non-family flags
  (`PSH_AT_SP`, `IS_BYTE`, `H1+i`, `EMBED_LO+k`, `OUTPUT_HI_THIS_STEP+k`) are
  NOT flagged. WARNs + prints the current-offender count by default (baseline
  **1883** across 26 ops files at golden `b4d2ab27`; l0-l16's FFN authoring
  is ported → 0). `--max N` / `--strict` FAIL when the count grows (CI
  ratchet); per-line opt-out `# sem-dim-ref-lint: allow`; `--demo` proves it
  discriminates a raw ref from a `dim_ref` one.
- `compare_symbolic_to_lowered_ffn(ir, dim_positions, S=100.0)` /
  `compare_symbolic_to_lowered_attn(...)` /
  `compare_symbolic_to_lowered_embedding(...)` — byte-identity gate.
  Lives in `neural_vm/unified_compiler/ir.py`.
- `neural_vm/unified_compiler/decl_verifier.py` — `verify_rule_scopes`,
  `verify_rule_strength`, `_collect_ffn_rules_from_op`. Cross-checks
  `scope` / `dominates_at` claims against the actual contribution
  algebra (sign-aware, per S-2 follow-up).
- `tools/sweep_compare_ffn.py` / `tools/sweep_compare_attn.py` —
  corpus-wide bucket of `declaration_semantics` / `lowering` /
  `weight_output_mismatch` failures. Read-only.
- `verify_claims_static` — `(layer, scope, identifier, column)` claims
  must match every observed weight write. See
  [`docs/DIM_OWNERSHIP_REGISTRY.md`](docs/DIM_OWNERSHIP_REGISTRY.md).
- **`tools/cpu_full_trace.py --ids <list>` — MANDATORY CPU self-check for
  any framing / full_trace fix.** It runs the byte-identical CPU
  *autoregressive* decode (`FaithfulAutoregressiveRunner`) and reports the
  SAME per-program pass/fail verdict as `tools/run_1096_canonical.py
  --criterion full_trace`, on CPU, in ~85s/program, no GPU. **DO NOT** use
  `interp_oracle_gate.py` / a re-anchored single forward to self-check a
  framing fix: it decodes each step independently and CANNOT see the
  production fixed-35-token-slice cumulative desync (the 34/37-token
  miscount), so it OVER-CLAIMS PASS for the `var_*` / `func_identity` /
  `nested_*` / `if_var` / `expr_*` framing-drift clusters the GPU fails (the
  canonical `func_identity` "passes step 9" on CPU → 0/150 GPU false
  positive). `cpu_full_trace` reproduces every framing FAIL byte-for-byte
  (validated 6/6 vs GPU on current main, golden `b4d2ab27` flag-OFF). The
  only CPU-vs-GPU disagreements are the documented SATURATED-TIE handful
  (`add` high-byte / `expr_paren` / `expr_mul_div`) where the MODEL itself is
  fp-accumulation-order divergent (~1e22 logits, gap=0) — and there CPU is
  conservatively STRICTER (FAILs a GPU-pass), never the dangerous direction
  (it never PASSes a GPU-fail). `interp_oracle_gate.py` stays the tool for
  RULE ATTRIBUTION (which FFN rule owns a wrong byte), NOT framing verdicts.
  Tooling only (not on any build path → model byte-identical). Validation
  table:
  [`docs/CPU_FULL_TRACE_TRUTHFUL_2026_06_17.md`](docs/CPU_FULL_TRACE_TRUTHFUL_2026_06_17.md).
- **`tools/lint_cross_op_attention.py` — MANDATORY for any op that
  modifies a SHARED / pre-existing attention head.** `compare_symbolic_to_lowered_attn`
  only checks one op in isolation (single query, hardmax), so it CANNOT
  see that adding a Q/K/V/O slot to a shared head changes the GLOBAL
  softmax output for OTHER ops/contexts served by that head — the exact
  blind spot that let the byte-0 fix (`C4_OPERAND_GATHER_PSH_ROWSELECT`)
  look like +8 when the full run was −39 (it broke `var_simple` +
  `expr_mod` via the shared L7 operand-gather head's softmax1
  normalization). This lint builds the model flag-OFF/flag-ON (CPU,
  `disk_cache=False`), auto-detects modified shared heads, and asserts the
  post-softmax head OUTPUT is unchanged at a battery of OTHER-op /
  OTHER-context probe rows. It is the authoring-time counterpart to the
  GPU tripwire (`tools/gpu_tripwire.py`). Run
  `python tools/lint_cross_op_attention.py --flag C4_MY_FIX --expect <rows>`;
  the `--demo` flag proves it discriminates byte-0 (flagged) from a clean
  band/LM-head fix (passed). Design:
  [`docs/CROSS_OP_ATTENTION_LINT_2026_06_17.md`](docs/CROSS_OP_ATTENTION_LINT_2026_06_17.md).
- **`tools/lint_cross_op_ffn.py` — MANDATORY for any op that modifies a
  SHARED FFN region / residual band** (the l14 ALU block, where
  ADD/SUB/MUL/DIV/MOD/SHL all write+read overlapping `OUTPUT_LO/HI` +
  `ALU_LO/HI` dims; the l16 materializers; any shared `OUTPUT`/`ALU` band).
  `compare_symbolic_to_lowered_ffn` verifies ONE op in ISOLATION (single
  query, hardmax), so it CANNOT see that two ops sharing a residual band /
  FFN region interact — the exact blind spot that let a "MUL-only" l14 fix's
  smooth-`silu` firing silently shift what add/sub/div READ in the shared l14
  ALU band one block later → a −60 regression that BOTH golden byte-identity
  AND the isolated-op check missed (silu is non-zero for OTHER inputs too, and
  an attention `W_o` write is re-weighted by a GLOBAL softmax). This lint is
  the FFN counterpart to `lint_cross_op_attention.py`: it builds the model
  flag-OFF/flag-ON (CPU, `disk_cache=False`, own cache dir), auto-detects the
  blocks whose FFN / attn-`W_o` changed and the SHARED OUTPUT/ALU band they
  write (which OTHER ops read downstream), and asserts the post-FFN residual
  at those shared dims is UNCHANGED at a battery of OTHER-op / OTHER-context
  probe rows (ADD/SUB/DIV/MOD/SHL/SHR operand frames with several competing
  STACK0/ALU candidate rows, so both the softmax normalization and the silu
  firing are exercised). Run
  `python tools/lint_cross_op_ffn.py --flag C4_MY_FIX --expect OP_MUL`; the
  `--demo` flag proves it discriminates the mul-l14 entanglement (flagged:
  perturbs add/sub/div's OUTPUT read band) from a clean band-separable fix
  (passed: writes a private TEMP dim only). CPU, ~the time of two builds
  (~40s for the full `--demo`), runs BEFORE a GPU ever sees the fix. The two
  `--demo` fixtures (`C4_FFN_LINT_MULL14_DEMO` / `C4_FFN_LINT_CLEAN_DEMO` in
  `ops/l14_ops.py`) are flag-gated tooling, both DEFAULT OFF → production build
  byte-identical to golden.
- **`tools/flag_regression_gate.py` — MANDATORY flag-ON cross-cluster gate
  for any campaign-config fix.** The byte-identity gates
  (`compare_symbolic_to_lowered_ffn`, `tools/_isa_golden_hash.py`) only
  verify the **flag-OFF** golden (35-token) model. A change can be
  byte-identical OFF yet silently **regress a whole cluster FLAG-ON** in the
  30-token *campaign* config (`C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1`)
  — the exact blind spot that let a mul `l14` fix pass golden byte-identity
  while crushing add/sub/div ~−60 in the campaign config. This gate runs a
  per-cluster REPRESENTATIVE sample (`tools/flag_regression_sample.json`: 72
  ids across ALL 56 clusters, 4 each for add/sub/mul/div/mod, the four
  MEM-SMOKE `var_*` clusters always included) in BOTH states (fix ON vs OFF)
  **within the campaign config**, scores each with the validated bit-exact
  `cpu_full_trace` verdict (`--spec-k 0`, `--workers 2`), and reports any
  cluster that goes `ok -> fail` (a REGRESSION — non-zero exit, BLOCK) or
  `fail -> ok` (a flip — gain). The MEM-SMOKE clusters get an explicit tag so
  a campaign fix can never silently break the SI/LI store-load path. Run
  `python tools/flag_regression_gate.py --flag C4_MY_FIX` (the fix's
  kill-switch; OFF leaves it unset, ON sets it, both inside the campaign env)
  or `--base <commit>` (HEAD vs base via a throwaway `git worktree`, no
  stash). It is the CPU automated form of the by-hand "cross-cluster verify"
  and runs in a few minutes (`--clusters add,sub,div,mul` to narrow). PROVEN:
  it CATCHES the re-applied mul-`l14` regression (commit `ba06deaa`, behind
  `C4_MUL_BLK33_CLAWBACK`) flagging add/sub/div `ok->fail`, and passes CLEAN
  on a no-op flag. Memory discipline: `--workers 2` max, dedicated
  `C4_VM_CACHE_DIR=/tmp/c4cache_reggate`, ~1 bake per state. Tooling only
  (golden `b4d2ab27` unchanged).

## Tests

- `tests/test_1096_*` — the 1096 corpus regression suite. Sentinel
  baseline depends on env flags (see memory note
  `project_1096_sentinel_baseline.md`); do not quote historical numbers
  in fix briefs.
- `tests/test_suite_1096_pure_neural_pytest.py` — pure-neural decode
  vs teacher-forced.
- `tests/test_smoke_*` — narrow opcode/path coverage.
- `tests/test_l14_output_cleanup.py`, `tests/test_l10_tail_correction.py`
  — templates for per-op declarative tests.
- `tests/test_dim_allocator.py`, `tests/test_ffn_unit_allocator.py`,
  `tests/test_attention_head_allocator.py` — allocator contracts.

## Workflow constraints

- **Do not stash.** Worktrees must hold their own state; `git stash`
  pollutes the shared stash list.
- **`gh` CLI is sandbox-blocked** — use the verifier scripts
  (`decl_verifier.py`, `scan_l10_solo.py`, etc.) for debugging. See
  memory note `feedback_agent_briefs.md`.
- **Single-rule whack-a-mole is zero-sum.** A new corrective op stacked
  on a broken upstream rule nets zero in 0/5 historical agent
  attempts. Use verifier output, not manual diagnosis. See memory note
  `feedback_single_rule_fixes_are_zero_sum.md`.
- **Open-bug notes live in user memory.** Before working on L10
  PSH/`MEM_addr0`, L3 SP-byte0, L16 BP-frame, etc., re-read the
  relevant memory entries — they encode real-time blocker context.

## Where the work lives

- `neural_vm/unified_compiler/` — THE declarative compiler (sole live
  weight-authoring path; entry `compile_full_vm_dynamic` in
  `full_vm_compiler_dynamic.py`). Subdir `ops/` contains the per-layer
  `lN_ops.py` files; each exports `make_*` op factories. The old
  standalone `unified_compiler/compiler.py` (`UnifiedVMCompiler`, ~3.8k
  LOC) is DELETED (commit `0bb9c5af`); a couple of docstrings still
  cross-reference it historically.
- `neural_vm/vm_step.py` / `neural_vm/setup_helpers_l*.py` — **dead**
  legacy `_set_layerN_*` weight writers. `set_vm_weights()` (the only
  entry that called them) has been removed, so NOTHING on the build path
  invokes them. Kept for legacy byte-identity tests + docstring
  cross-refs; prune target (see [`docs/TEST_PRUNE_MAP.md`](docs/TEST_PRUNE_MAP.md)).
- `tools/` — sweep + audit scripts. Self-contained Python; no model
  bake side effects.
- `docs/` — design docs, migration guides, phase plans. New designs
  always land here first. Flag inventory:
  [`docs/FLAG_REGISTRY.md`](docs/FLAG_REGISTRY.md).

## Phase status (as of July 2026, golden `b4d2ab27`)

- **Model is 100% live-declarative.** The imperative→declarative cut is
  COMPLETE: `settrace` confirms ZERO live `_set_layer*` writers on the
  build path, and `set_vm_weights()` is removed. The last live imperative
  writer (L8 `_set_layer8_alu`) was migrated. The old `UnifiedVMCompiler`
  (`compiler.py`, ~3.8k LOC) is DELETED.
- **Phase 6 (declarative authoring)**: COMPLETE for FFN ops + attention
  heads. L6 `routing_ffn` and L15 `memory_lookup` cut via Phase 7.C.
- **AX byte-emission consolidation**: DONE. AX byte-1/2/3 now emit from
  the canonical `OUTPUT_LO/HI` nibbles (the L25 `b1_to_output` decode);
  the per-value H-band (`H*_DUMP_OUT`) LM-head emission patchwork is
  removed and the `C4_B1_TO_OUTPUT` flag is DELETED (OUTPUT-canonical is
  the sole unconditional byte-1 path, commit `1866d62f`). The dead
  `ax_li_byte23_zero` / `ax_byte23_dump_zero` correctors were removed.
- **Phase 7.A (scheduler cycle decomposition)**: in progress;
  `OUTPUT_HI` split into `OUTPUT_HI_PREV_STEP` for cross-step reads.
- **Phase 7.B (pin removal)**: pending; `pin=` is still corpus-wide.
- **Phase 7.C (partial-migration cuts)**: L6 routing, L15 memory_lookup
  landed.
- **Phase 7.D (model-level bakes)**: `TokenEmbeddingRule` exists and is
  in production use (`head_bake` / `embedding_bake` model-level bakes);
  further migration continues.
- **Phase 7.E (semantic dim refs)**: PARTIALLY LANDED — `dim_ref("category",
  "role")` authoring (from `dim_registry.py`) is in production across many
  layers (l1/l2/l3/l4/l5/l6/l8/l9/l11/l12/l13/l14 + model/alu ops); the
  remaining `+N`-offset refs are being ported byte-identically (`l0/l7/l16`
  in flight). A canonical `DimResolver`
  (`unified_compiler/dim_resolver.py`) + the `tools/lint_dim_resolution.py`
  ratchet now enforce BUILT-layout dim resolution.
- **Phase 7.F (KV eviction)**: blocked on 7.A + 7.D.

**Config entry point:** `C4_CAMPAIGN=1` is the single flag that flips the
30-token campaign config (the fix fleet's DEFAULT); the golden flag-OFF
(35-token) build is byte-identical. Every runtime `C4_*` flag is
inventoried in [`docs/FLAG_REGISTRY.md`](docs/FLAG_REGISTRY.md).

When in doubt, check
[`docs/PHASE_7_FULLY_DYNAMIC_PLAN.md`](docs/PHASE_7_FULLY_DYNAMIC_PLAN.md)
for the current wave acceptance criteria.
