# IR DSL Design — FFN + Attention as the Source of Every Weight

Goal: **`FFNRule`, `DeclarativeAttentionHeadSpec`, and `TokenEmbeddingRule`
are the entire VM DSL.** Every weight in every block derives from them.
No imperative `block.ffn.W_up.data[...] = X` writes anywhere.

The 3,781 lines of imperative composites in `efficient_alu_*.py` get
re-expressed as rule lists. The compiler is free to optimize the
lowering — including fusing many rules into a fast composite at
compile time — but the **source of truth** is rules.

Status: **Waves V1-V7 DONE (2026-06-04). Lint ratchet (V9) in place.**

---

## 0. Wave status summary

The migration is delivered in numbered waves. V1-V7 ship the
building-blocks DSL plus per-layer FFN-bake rewrites. V8-V9 close
out the regression machinery.

| Wave | Scope | Status | Tests |
|---|---|---|---|
| V1  | Building-blocks DSL — `step_function_rule`, `one_hot_indicator_rule`, `multi_way_and_rule`, `band_range_check_rules`, `cancel_residual_rule`, `lookup_table_rules`, `multi_way_or_rules` | **DONE** | `tests/test_building_blocks_dsl.py` |
| V2  | Bit/floor extensions — `magic_floor_rules`, `bit_range_extract_rules`, `opcode_expert_rules`, attention helpers (`efficient_exp_attention`, `memory_load_attention`, `fetch_byte_attention`) | **DONE** | `tests/test_building_blocks_dsl.py` |
| V3  | Per-layer FFN bake migration (L0-L4) onto DSL helpers | **DONE** | per-layer smoke + byte-identity |
| V4  | Per-layer FFN bake migration (L5-L9) | **DONE** | per-layer smoke + byte-identity |
| V5  | Per-layer FFN bake migration (L10-L13) | **DONE** | per-layer smoke + byte-identity |
| V6  | Per-layer FFN bake migration (L14-L17) | **DONE** | per-layer smoke + byte-identity |
| V7a | Module-replacement op annotation (claims sentinels) | **DONE** | `verify_claims_static` |
| V7b | L16 final remnants (nonstore_mem + psh_no_borrow + lea_local + stack0_cancel_lev_sp) through `multi_way_and_rule` | **DONE** | smoke |
| V8  | Wide-ALU DSL (`wide_alu_dsl.py`) — wide add/sub/mul/div/shift/bitwise helpers | in-progress | per-stage byte-identity |
| V9  | Raw-FFNRule lint ratchet + docs (this wave) | **DONE** | `tests/test_lint_raw_ffn_rule.py` |

Raw `FFNRule.constant_write` / `gated_write` counts at V9 baseline:
**76 calls across 9 files**, tracked in
`tools/lint_raw_ffn_rule.py::_BASELINE`. Lint exits 1 when any
baselined file grows or any new file appears.

---

## 1. Current state (audit)

| Component | Status |
|---|---|
| Per-layer FFN ops (118 of 166) | Already rule-driven via `FFNRule.constant_write` / `gated_write` |
| Attention heads (95%) | Already declarative via `DeclarativeAttentionHeadSpec` (`AP`/`AO`) |
| Token embeddings | Already declarative via `TokenEmbeddingRule` (Phase 7.D done) |
| **Efficient ALU composites (~3,781 lines)** | **Imperative; bypass the DSL entirely** |
| RuntimeAttentionFragment escape hatch | 4 sites for shape-dependent attention |

The remaining gap is the efficient ALU composites and ~4 attention
shape-dependent sites.

---

## 2. The composite landscape (what needs migration)

| File | Lines | Composites | Stages |
|---|--:|---|---|
| `efficient_alu_neural.py` | 1554 | FlattenedALUMul, ALUAndOrXor, GEToBDConverter, BDToGEConverter, PureNeuralALU base | 9-stage MUL pipeline + base classes |
| `efficient_alu_addsub_split.py` | ~500 | AddSub5StageBlock | 5 stages |
| `efficient_alu_divmod_split.py` | 518 | FlattenedDivMod | Long-division pipeline |
| `efficient_alu_byte.py` | 248 | Byte-level helpers | — |
| `efficient_alu_integrated.py` | 657 | Integration utilities | — |

Plus per-opcode wrapper ops in `alu_ops.py` (37 entries).

---

## 3. The DSL primitives (what we use as building blocks)

### 3.1 Already exists

- **`FFNRule.constant_write(name, writes, conditions, threshold)`** — one FFN unit, always fires when conditions sum > threshold.
- **`FFNRule.gated_write(name, writes, conditions, threshold, gate, gate_weight)`** — one FFN unit with multiplicative gate.
- **`DeclarativeAttentionHeadSpec(head_idx, q, k, v, o)`** — one attention head with `AP` (Q/K weights) and `AO` (O writes).
- **`TokenEmbeddingRule(token, table, target_dim, value)`** — one row write into embed/head/head_bias.

### 3.2 New helpers needed (DSL extensions)

To express multi-stage composites cleanly, the DSL needs a few
higher-level constructors that LOWER to FFNRule + AttnHeadSpec lists.
Each is a function from typed inputs to a rule list, NOT a new IR type:

```python
# Per-byte wide arithmetic — generates N FFNRule per byte.
def wide_add_rules(
    *,
    operand_a_dim_base: str,   # e.g. "AX_LO"
    operand_b_dim_base: str,   # e.g. "ALU_LO"
    result_dim_base: str,      # e.g. "OUTPUT_LO"
    carry_dim_base: str,       # e.g. "CARRY"
    width_bytes: int,
    opcode_gate: str,          # e.g. "OP_ADD"
    marker_gate: str,          # e.g. "MARK_AX"
    S: float,
) -> tuple[FFNRule, ...]: ...

def wide_sub_rules(...) -> tuple[FFNRule, ...]: ...
def wide_mul_rules(...) -> tuple[FFNRule, ...]: ...  # the 9-stage pipeline expressed as rules
def wide_div_rules(...) -> tuple[FFNRule, ...]: ...
def wide_shift_rules(*, direction: Literal["left", "right"], ...) -> tuple[FFNRule, ...]: ...
def bitwise_rules(*, op: Literal["and", "or", "xor"], ...) -> tuple[FFNRule, ...]: ...
```

These live in a new module: `c4_release/neural_vm/unified_compiler/wide_alu_dsl.py`.
They're **plain Python functions**, no new IR types. They emit the
same `FFNRule` lists that already drive the per-layer FFN ops.

### 3.3 New attention helpers

```python
def memory_load_attention(
    *,
    addr_dim: str,         # where to read the address from
    value_target_dim: str, # where the loaded byte lands
    byte_width: int,
) -> DeclarativeAttentionHeadSpec: ...

def fetch_byte_attention(
    *,
    pc_offset: int,        # PC+k offset to fetch from
    target_dim: str,
) -> DeclarativeAttentionHeadSpec: ...
```

Same shape — Python functions returning typed spec lists.

### 3.4 What replaces RuntimeAttentionFragment

For shape-dependent heads (L15 `memory_lookup` branching on `num_heads`),
parameterize `DeclarativeAttentionHeadSpec` over a shape variable
provided at compile time:

```python
@parameterized_attn_spec
def l15_memory_lookup(num_heads: int) -> DeclarativeAttentionHeadSpec:
    if num_heads == 8:
        return DeclarativeAttentionHeadSpec(...)
    else:  # 4
        return DeclarativeAttentionHeadSpec(...)
```

Still declarative; the branching is at spec-generation time, not at
forward() time.

---

## 4. Lowering pipeline

The current `Primitives.lower_ffn_rules(ffn, rules, dim_positions, start_unit, S)`
already lowers FFNRule lists into per-unit weights. Each rule maps to
one or two hidden units (constant_write → 1 unit; gated_write with
gate=condition_dim → ~2 units depending on threshold structure).

For composites with hundreds-of-thousands of rules (e.g. wide_mul_rules
across 4 bytes with carry might emit ~600 rules), the lowering remains
linear time. The bottleneck is bake time, not rule count.

### 4.1 Compile-time fusion (optional optimization)

After rules → weights, the compiler may detect repeated patterns and
fuse adjacent units into a single fast PyTorch module (the way
FlattenedALUMul does today). This is a lowering optimization — the
fused module is byte-identical to the rule-lowered weights.

The verification gate: **fused composite forward == lowered rules forward**
on randomized input. Same byte-identity contract as today's
`compare_symbolic_to_lowered_ffn`.

### 4.2 The mode-mismatch problem disappears

Today:
- Lookup mode bakes declarative rules into a vanilla PureFFN.
- Efficient mode bakes a handwritten composite (FlattenedALUMul etc).

After DSL migration:
- Both modes start from the same rule list.
- Lookup mode lowers rules into PureFFN (slow inference, easy debug).
- Efficient mode lowers rules into a fused composite (fast inference, same weights).
- Byte-identity is automatic — same source, same lowering output, optional speed-up wrapper.

---

## 5. Migration order

Easiest → hardest (so we ship value fast):

| Wave | Composite | Lines | Sessions | Why this order |
|---|---|--:|--:|---|
| W1 | Bitwise (ALUAndOrXor) | ~200 | 1-2 | Simplest — per-byte op, no carry |
| W2 | Shift (ALUShiftComposite) | ~300 | 2-3 | Per-byte select; precompute + select stages |
| W3 | AddSub (AddSub5StageBlock) | ~500 | 3-4 | 5 stages; carry propagation. First wide-arithmetic migration. |
| W4 | DivMod (FlattenedDivMod) | ~518 | 4-6 | Long-division; most stages; uses wide_sub as helper |
| W5 | MUL (FlattenedALUMul) | ~700 | 5-8 | 9-stage pipeline; the boss fight |
| W6 | Delete `efficient_alu_*.py` | — | 1 | Cleanup after all 5 migrated |
| W7 | Attention shape-parameterization | ~50 | 1-2 | Remove RuntimeAttentionFragment escape hatch |
| **Total** | | **3.8k → 0** | **17-26 sessions** | |

Each wave is independently shippable and gated by byte-identity per
composite. Stop after any wave and the codebase is in a working state.

---

## 6. Validation framework

Per wave:

1. **Byte-identity per stage** — for each composite's forward(), compare
   `composite.forward(x)` vs `rule_lowered_ffn.forward(x)` on randomized
   input. Must match bit-for-bit.
2. **Smoke + 1096 pass count** — must not regress.
3. **Compile-time fusion test** — if a fused composite is shipped,
   verify it matches rule-lowered weights bit-for-bit.

Plus the existing checks (compiler-step0 safety, claims static,
produces_consumes_dynamic) extend to the migrated rules automatically.

---

## 7. What this unlocks

After all 7 waves land:

- **0 imperative `*.W_*.data[...]` writes corpus-wide.** The DSL is THE
  source.
- **Byte-identity is automatic.** Lookup vs efficient mode lowerings
  diverge in fused-composite step only; the rule source is shared.
- **Strict mode is exhaustive.** The 48 imperative-only ops disappear;
  the compiler can verify everything.
- **Mode-mismatch class of bug eliminated.** Fix once, lands in both modes.
- **The 3.8k lines of imperative composite collapse to ~200-300 lines of rule-generating helpers.**
- **New opcodes are easy** — write a behavioral spec; the rule helpers
  generate the rules; the compiler does the rest.

---

## 8. Risks + mitigations

| Risk | Mitigation |
|---|---|
| Bake time regresses (more rules to lower) | Compile-time fusion keeps runtime composites |
| Numerical precision drift in lowered rules vs handwritten composite | Per-stage byte-identity gate catches drift |
| Wave W5 (MUL) is genuinely large | Subdivide by stage; ship per-stage |
| Lowered weights are bigger (more units) | Fused composite step keeps deployed size |
| Some composite uses tricks the rule DSL can't express | Extend DSL helpers in that wave; not a blocker |

---

## 9. Open design questions

1. **wide_*_rules helpers** — should they emit FFNRule lists directly, or a new `FFNStage` typed wrapper that groups related rules?
2. **Compile-time fusion** — automatic (compiler detects patterns) or annotation-driven (`@fuse_to_composite`)?
3. **Stage isolation** — should each composite stage become its own `Operation` with its own dep edges? Or stay grouped under the wrapper op?
4. **Shape parameterization** for attention — Python functions returning specs, or a typed `ShapeVar` system?
5. **Test fixture for byte-identity** — extend `compare_symbolic_to_lowered_*` or build a per-composite fixture?

---

## 10. Effort summary

| Phase | Sessions |
|---|--:|
| Design + DSL helper scaffolding | 1-2 |
| Wave W1 Bitwise | 1-2 |
| Wave W2 Shift | 2-3 |
| Wave W3 AddSub | 3-4 |
| Wave W4 DivMod | 4-6 |
| Wave W5 MUL (sub-divide by stage) | 5-8 |
| Wave W6 Cleanup | 1 |
| Wave W7 Attention shape-parameterization | 1-2 |
| **Total** | **18-28 sessions** |

Each wave independently valuable. The compounding effect after waves
W1-W4: lookup mode and efficient mode no longer diverge for those
opcodes; mode-mismatch bugs in their family eliminated.

---

**Status**: DRAFT pending user review. After approval, becomes
authoritative DSL design. Tracked as parent task + 8 sub-tasks.
