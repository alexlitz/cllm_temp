# How to add a corrective op

A short, opinionated recipe for adding a new corrective op — a small bake that
patches one residual drift (an FFN cleanup unit, or a single attention head)
once a failure case is reproducible. Optimized for the common case: "smoke
test X is red; one residual dim Y is wrong on layer Z; write the minimum
declarative spec that pulls Y back into range without touching anything
else".

This guide is **opinionated and short on purpose**. For the full DSL see:

- [`FFN_RULE_MIGRATION_PATTERN.md`](FFN_RULE_MIGRATION_PATTERN.md) — every
  `FFNRule` field, the conditions / writes / gate / scope / `dominates_at`
  semantics, the lowering contract, and the verifier expectations.
- [`ATTENTION_HEAD_IR_MIGRATION_PATTERN.md`](ATTENTION_HEAD_IR_MIGRATION_PATTERN.md)
  — `DeclarativeAttentionHeadSpec`, `AP` / `AO` constructors, the layout-table
  pattern, alibi-slope overrides, and shared-head co-ownership.
- [`PHASE_6_DECLARATIVE_WEIGHT_AUTHORING_PLAN.md`](PHASE_6_DECLARATIVE_WEIGHT_AUTHORING_PLAN.md)
  / [`PHASE_7_FULLY_DYNAMIC_PLAN.md`](PHASE_7_FULLY_DYNAMIC_PLAN.md) — why
  the bake is declarative and where the pin-removal work lives.

Do not reproduce the DSL here — link to those guides and keep the recipe
tight.

## 1. Identify the failing case

Reproduce the bug on a concrete input from one of these sources, in order of
preference:

1. **1096 corpus** — the canonical regression set. Find a sample whose
   pure-neural decode disagrees with teacher-forced or fast-VM output.
   `tests/test_1096_neural_declarative_diagnostic.py` and
   `tests/test_suite_1096_pure_neural_pytest.py` are the usual entry points;
   pick a failing sample by id and pin it.
2. **Smoke test** — `tests/test_smoke_*.py` covers narrow scenarios
   (function calls, branches, multibyte ADD). If a smoke test is red, the
   failure is already minimized.
3. **Hand-built C bytecode** — last resort. Compile the smallest C snippet
   that hits the buggy opcode/path and reuse it as the test input.

Capture the failing sample's `(opcode, position, expected_residual_value,
observed_residual_value)` tuple before doing anything else. Without that
4-tuple you cannot decide whether the fix is too narrow, too broad, or
correct.

## 2. Locate where the bug surfaces

Walk the residual stream forward until the discrepancy appears. Two tools
do most of the work:

- `decl_verifier.py` (use it as a module via the helpers
  `verify_rule_strength`, `verify_rule_scopes`,
  `_collect_ffn_rules_from_op`) — flags rules whose scope or dominance does
  not match the actual cell pattern. Memory: this is the verifier-based
  debug path, not manual diagnosis (per `feedback_single_rule_fixes_are_zero_sum.md`).
- `tools/sweep_compare_ffn.py` / `tools/sweep_compare_attn.py` — bucket
  every op's IR into `declaration_semantics` / `lowering` /
  `weight_output_mismatch` failures. Run before any fix; the bucket your
  candidate op will join tells you whether your problem is upstream.

Write down: **(layer L, residual dim D, position predicate P)** where the
drift first appears. A "TEMP[0] is +2.0 at the PC marker on OP_LEV rows"
statement is enough to write the rule.

## 3. FFN rule or attention head?

| Symptom                                                                                 | Pick                                                                |
|-----------------------------------------------------------------------------------------|---------------------------------------------------------------------|
| One residual dim is off at a fixed position class (PC marker, IS_BYTE, MARK_AX, …)      | **FFN rule** — one new `gated_write`                                |
| Several output cells are wrong with the same firing condition (cleanup family)          | **FFN rules** — a small rule family in one IR                       |
| A value needs to be gathered from another token (prev step, sibling byte slot)          | **Attention head** — `DeclarativeAttentionHeadSpec`                 |
| The same head writes need to fire only on a runtime shape (e.g. when `num_heads ≥ 10`)  | **`RuntimeAttentionFragment`** — see L15 memory_lookup              |
| Per-token write into `model.embed`, `model.head.weight`, `model.head.bias`              | **`TokenEmbeddingRule`** — model-level bake (Phase 7.D / Wave 6D)   |

If the bug is "downstream layer copies a value that's already wrong",
fix the upstream rule. A corrective op stacked on top of a broken
upstream rule is single-rule whack-a-mole and will net zero (see memory:
`feedback_single_rule_fixes_are_zero_sum.md`).

## 4. Write the rule (or spec)

Open the layer's op file (`neural_vm/unified_compiler/ops/lN_ops.py`),
add a `_<op>_rules(S)` (FFN) or `_<op>_specs(BD)` (attention) helper next
to the existing ones, and pattern-match on a known-good neighbour.
Recommended references:

- **Single-rule FFN cleanup**: `_layer14_temp_clear_rules` in
  `c4_release/neural_vm/unified_compiler/ops/l14_ops.py:1107` — 4 units,
  4 `FFNRule.gated_write` calls, scope + `dominates_at` threaded. The
  worked example in the FFN guide is this one.
- **Multi-rule FFN family**: `_layer14_clear_addr_key_pollution_rules`
  (same file, ~48 rules) — one rule per output cell with a shared
  blocker pattern.
- **Attention head spec**: `_layer7_memory_head_specs` in
  `l7_ops.py:365` — six heads, layout-table-resolved `head_idx`, mix of
  Q/K/V/O writes.
- **Runtime-conditional attention**: L15 `memory_lookup` uses
  `RuntimeAttentionFragment` for the `attn.num_heads`-dependent branches
  (see `l15_ops.py` and `ir.py:424` for the dataclass).

Keep the rule as small as the bug demands. Every extra term widens the
firing pattern and risks colliding with another op's `dominates_at`.

## 5. Attach to an `Operation`

Build a one-layer `CompilerIR` and attach it as `compiler_ir=` (static
spec) or `compiler_ir_factory=` (spec depends on `dim_positions` /
`head_dim`):

```python
def _foo_ir(S: float = 100.0) -> CompilerIR:
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(_foo_rules(S))
    return ir

def make_foo_op() -> Operation:
    return Operation(
        name="layerN_foo_correction",
        phase=N + 0.5,
        reads={...}, writes={...},
        kind="block",
        bake_fn=bake, declarative_bake_fn=bake,
        compiler_ir=_foo_ir(),
        layer_idx=N,
        migrated=True,
        claims={(N, "ffn_W_down", "0", "<OUT_DIM>+0")},
        smoke_tests={"<the test id you fixed>"},
        spec_section="<docs section if any>",
    )
```

The dispatcher (`layer_compiler.py:_dispatch_operation_ir`, line ~553)
calls `ir.lower_ffn(...)` for `kind="ffn"` and both
`ir.lower_attention(...)` + `ir.lower_ffn(...)` for `kind="block"`. If
the op only declares IR (no residual imperative tail), the `bake_fn`
body collapses to that one dispatch call. For an attention-only op use
`_lower_*_ir` analogues in the layer's helpers.

Register the op in the layer's `make_*` factory list (the layer file
exports them via the registry).

## 6. Allocate units / heads / dims

Three allocators, all in `neural_vm/`:

- `dim_allocator.py` — residual-stream slots. New dims go in
  `dim_registry`/`dim_registry_dynamic`; only touch if your op needs a
  fresh dim category.
- `ffn_unit_allocator.py` — per-layer FFN hidden units. Use `pin=` to
  match an existing chain layout (see `_L14_CLEANUP_CHAIN_LAYOUT` in
  `l14_ops.py:80` for the pattern). Drop `pin=` once Phase 7.B's
  dynamic first-fit is on for that layer.
- `attention_head_allocator.py` — per-layer head indices. Use a
  `_LN_HEAD_LAYOUT` table next to the spec and resolve `head_idx` via
  `_LN_HEAD_LAYOUT_BY_NAME["<op>.head_<n>"]` so reshuffling is one
  edit (see `_L7_HEAD_LAYOUT` in `l7_ops.py:30`).

Pinning rule of thumb: **pin everything by default** until the layer's
allocator wave finishes; the byte-identity gate is your safety net. Once
Phase 7.B has landed for a layer, switch to `pin=None` for the new op
(other ops in the layer will already have lost their pins).

## 7. Validate

Each commit must clear, in this order:

1. **Byte-identity gate** —
   `compare_symbolic_to_lowered_ffn(ir, dim_positions, S=100.0).ok` for
   FFN ops, `compare_symbolic_to_lowered_attn(...)` for attention. Both
   live in `neural_vm/unified_compiler/ir.py`. Failure here means the
   lowering and the symbolic execution of your spec disagree — abort
   and re-read step 4.
2. **Claims** — `verify_claims_static` on the corpus. Every `W_down`
   (or `W_v`/`W_o`) write your rule emits must appear in `claims=`
   with the correct unit/slot index.
3. **Verifier** — `decl_verifier.verify_rule_strength(op, registry)`
   and `verify_rule_scopes(op, ...)` on the new op. Use the helpers in
   `scan_l10_solo.py` / `scan_static_claims.py` as a template harness.
4. **Sweep tools** — `tools/sweep_compare_ffn.py` /
   `tools/sweep_compare_attn.py` corpus-wide. The new op should not
   move any other op's bucket; if it does, your `dominates_at` is too
   wide.
5. **Smoke + 1096** — re-run the failing smoke test (now green) and
   the 1096 sample you captured in step 1 (delta is one cell or the
   sample is now decoded correctly). 1096 sentinel baseline depends on
   env flags — see memory note `project_1096_sentinel_baseline.md`
   before quoting numbers in commit messages.

## 8. Add a regression test

Drop a focused test in `c4_release/tests/test_<op>.py` next to the
existing corrective-op tests:

- [`test_l14_output_cleanup.py`](../tests/test_l14_output_cleanup.py)
  — pure-spec assertions on the head/FFN weights (no full model bake;
  uses `_StubFFN` / `_StubAttn`).
- [`test_l10_tail_correction.py`](../tests/test_l10_tail_correction.py)
  — single-rule `CompilerIR` round-trip via `compare_symbolic_to_lowered_ffn`.

The test must reference the failing case from step 1 in its docstring
or via the `smoke_tests=` field, so a future regression bisect can find
it. Keep it independent of the 1096 corpus where possible — corpus
runs are slow and flaky as a per-op regression signal.

## 9. Commit + ship

Conventional commit body:

```
phaseN.X: add layerN_<op>_correction (<scope>)

Fixes the L<N> drift on <position predicate>: <observed> -> <expected>.
One <kind="ffn"|"block"> op, <K> rules / <H> heads, claims pinned to
unit/head <idx>. compare_symbolic_to_lowered_<ffn|attn>.ok, smoke
<test_id> now green, 1096 sample <id> now decoded correctly.
```

One op per commit. If the fix needs more than one op, split — keeps
bisect surface small if a later wave introduces a regression.

## See also

- [`FFN_RULE_MIGRATION_PATTERN.md`](FFN_RULE_MIGRATION_PATTERN.md) — full
  FFN DSL.
- [`ATTENTION_HEAD_IR_MIGRATION_PATTERN.md`](ATTENTION_HEAD_IR_MIGRATION_PATTERN.md)
  — full attention DSL.
- [`DECLARATIVE_VERIFICATION.md`](DECLARATIVE_VERIFICATION.md) — what the
  verifier checks on `scope` / `dominates_at` / `claims`.
- [`DIM_OWNERSHIP_REGISTRY.md`](DIM_OWNERSHIP_REGISTRY.md) — claims-as-witnesses
  semantics.
- [`PHASE_6_DECLARATIVE_WEIGHT_AUTHORING_PLAN.md`](PHASE_6_DECLARATIVE_WEIGHT_AUTHORING_PLAN.md)
  — why every new op is declarative.
- [`PHASE_7_FULLY_DYNAMIC_PLAN.md`](PHASE_7_FULLY_DYNAMIC_PLAN.md) — the
  pin-removal / model-level / scheduler waves that change a few details
  in steps 5-6 as they land.
