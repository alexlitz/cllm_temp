# Building-Blocks DSL — `building_blocks_dsl.py`

The building-blocks DSL is the canonical, BLOG_SPEC-aligned vocabulary
for authoring FFN and attention weights. Every layer's FFN bake should
compose rules from these constructors; raw
`FFNRule.constant_write(...)` / `FFNRule.gated_write(...)` calls
outside the DSL modules are tracked and ratcheted by
[`tools/lint_raw_ffn_rule.py`](../tools/lint_raw_ffn_rule.py).

This document maps each constructor back to the section of
[`BLOG_SPEC.md`](BLOG_SPEC.md) §504-568 it realizes, gives a usage
example, and pins the verification convention used in
[`tests/test_building_blocks_dsl.py`](../tests/test_building_blocks_dsl.py).

Source: [`neural_vm/unified_compiler/building_blocks_dsl.py`](../neural_vm/unified_compiler/building_blocks_dsl.py).
Wide-ALU follow-on: [`neural_vm/unified_compiler/wide_alu_dsl.py`](../neural_vm/unified_compiler/wide_alu_dsl.py)
(Wave V8).

---

## 1. Section → constructor map

| BLOG_SPEC §        | Concept                                  | Constructor                  | Lowering              |
|--------------------|------------------------------------------|------------------------------|-----------------------|
| §504-506           | Step function `H_ε(x)`                   | `step_function_rule`         | 1 `FFNRule`           |
| §510-518           | Point indicator (one-hot realization)    | `one_hot_indicator_rule`     | 1 `FFNRule`           |
| §520               | Lookup tables (one-hot keyed)            | `lookup_table_rules`         | N `FFNRule`s          |
| §522               | Cancelling residuals                     | `cancel_residual_rule`       | 1 `FFNRule`           |
| §524-540           | Range check `R_{a,b}` over one-hot band  | `band_range_check_rules`     | `hi-lo+1` `FFNRule`s  |
| §547-555           | Efficient floor via fp32 MAGIC trick     | `magic_floor_rules`          | multi-`FFNRule` band  |
| §558               | Bit range extraction                     | `bit_range_extract_rules`    | multi-`FFNRule` band  |
| §561-564           | Efficient exp via softmax1 + ALiBi-0     | `efficient_exp_attention`    | 1 `DeclarativeAttentionHeadSpec` |
| §561-564 (variant) | Address-keyed memory load                | `memory_load_attention`      | 1 `DeclarativeAttentionHeadSpec` |
| §561-564 (variant) | PC-offset bytecode fetch                 | `fetch_byte_attention`       | 1 `DeclarativeAttentionHeadSpec` |
| §566               | Mixture-of-Experts opcode routing        | `opcode_expert_rules`        | N `FFNRule`s (gated)  |
| §568 → §590        | N-way conjunction / disjunction          | `multi_way_and_rule` / `multi_way_or_rules` | 1 or N `FFNRule`s |

Each helper returns plain `FFNRule` (or `DeclarativeAttentionHeadSpec`)
instances. Callers append the result to a per-layer rule list — the
compiler does the rest.

---

## 2. Code examples

### 2.1 `step_function_rule` — §504-506

Hard-thresholded Heaviside over a scalar dim.

```python
from c4_release.neural_vm.unified_compiler.building_blocks_dsl import step_function_rule

# Fire IO_SET when ALU_HI >= 3
rule = step_function_rule(
    input_dim="ALU_HI",
    threshold=2.5,            # target - 0.5 for integer-valued inputs
    write_dim="IO_SET",
    write_value=2.0,           # canonical magnitude
    S=100.0,
    name="alu_hi_ge_3",
)
```

### 2.2 `one_hot_indicator_rule` — §510-518

Point indicator on a one-hot band cell.

```python
from c4_release.neural_vm.unified_compiler.building_blocks_dsl import one_hot_indicator_rule

# Fire OUTPUT_HI when ALU_LO band == 7
rule = one_hot_indicator_rule(
    band="ALU_LO",
    value=7,
    write_dim="OUTPUT_HI",
    S=100.0,
    name="alu_lo_eq_7",
)
```

### 2.3 `band_range_check_rules` — §524-540

Inclusive integer range check over a one-hot band.

```python
from c4_release.neural_vm.unified_compiler.building_blocks_dsl import band_range_check_rules

# Fire OP_GROUP when ALU_LO in [4, 9]
rules = band_range_check_rules(
    band="ALU_LO",
    lo=4,
    hi=9,
    write_dim="OP_GROUP",
    S=100.0,
    name="alu_lo_in_range",
)
# rules is a tuple of 6 FFNRules, one per in-range cell.
```

### 2.4 `multi_way_and_rule` — §568 / §590

Balanced N-way conjunction. Derives the threshold from condition
weights when not given explicitly.

```python
from c4_release.neural_vm.unified_compiler.building_blocks_dsl import multi_way_and_rule

# Fire WRITE_NIBBLE only when MARK_OP_LI AND PC_TOP_INSTR AND VAR_X are all set.
rule = multi_way_and_rule(
    conditions=(
        ("MARK_OP_LI", 1.0),
        ("PC_TOP_INSTR", 1.0),
        ("VAR_X", 1.0),
    ),
    writes=(("WRITE_NIBBLE", 0.02),),     # write_weight = 2/S at S=100
    name="li_x_write",
)
```

### 2.5 `multi_way_or_rules` — §568 / §590

Per-condition OR. Emits one rule per condition; writes accumulate when
conditions overlap.

```python
from c4_release.neural_vm.unified_compiler.building_blocks_dsl import multi_way_or_rules

rules = multi_way_or_rules(
    conditions=("MARK_OP_JMP", "MARK_OP_JSR", "MARK_OP_BZ", "MARK_OP_BNZ"),
    writes=(("BRANCHY", 0.02),),
    S=100.0,
    name="branchy_or",
)
```

### 2.6 `cancel_residual_rule` — §522

Self-cancel a dim when it is set (or write the negation into a
different dim).

```python
from c4_release.neural_vm.unified_compiler.building_blocks_dsl import cancel_residual_rule

# Clear OUTPUT_LO+5 when the cell is active.
rule = cancel_residual_rule(
    input_dim="OUTPUT_LO+5",
    S=100.0,
    name="output_lo_5_cancel",
)
```

### 2.7 `lookup_table_rules` — §520

N-entry lookup keyed on a one-hot band.

```python
from c4_release.neural_vm.unified_compiler.building_blocks_dsl import lookup_table_rules

# When OPCODE_BAND active cell ∈ {0,1,2,3}, write a different stack-depth delta.
rules = lookup_table_rules(
    key_band="OPCODE_BAND",
    key_to_writes={
        0: (("SP_DELTA", +1.0),),
        1: (("SP_DELTA", -1.0),),
        2: (("SP_DELTA", +2.0),),
        3: (("SP_DELTA",  0.0),),
    },
    S=100.0,
    name_prefix="opcode_sp_delta",
)
```

### 2.8 Attention helpers — §561-564

Each returns a single `DeclarativeAttentionHeadSpec` that drops into a
layer's head list.

```python
from c4_release.neural_vm.unified_compiler.building_blocks_dsl import (
    efficient_exp_attention,
    memory_load_attention,
    fetch_byte_attention,
)

# Exp via softmax1+ALiBi-0
exp_spec = efficient_exp_attention(
    head_idx=3, input_dim="ALU_SCALAR", output_dim="EXP_OUT",
    bias=2.0, S=100.0, name="exp_alu",
)

# Address-keyed load (key = addr, value = byte at addr)
load_spec = memory_load_attention(
    head_idx=4, addr_dim="MEM_addr0", value_target_dim="MEM_loaded",
    byte_width=1, S=100.0, name="mem_load_byte0",
)

# Fetch byte at PC+offset
fetch_spec = fetch_byte_attention(
    head_idx=5, pc_offset=2, target_dim="INSTR_BYTE2", S=100.0,
)
```

---

## 3. Verification convention

Every constructor ships with two verification gates in
[`tests/test_building_blocks_dsl.py`](../tests/test_building_blocks_dsl.py):

1. **Symbolic-vs-lowered at S=1, calibrated state.** Single-rule
   primitives (`step_function_rule`, `one_hot_indicator_rule`,
   `cancel_residual_rule`, `multi_way_and_rule`) are gated by
   `compare_symbolic_to_lowered_ffn(ir, dim_positions, S=1.0)` with
   inputs set to `0.5 + _SILU_ONE_INPUT` so `silu(S·input) = 1.0`
   exactly. The lowered weights must produce the same byte-level
   contribution as the symbolic rule.

2. **Neural byte-identity at S=100, one-hot.** Multi-rule primitives
   (`band_range_check_rules`, `lookup_table_rules`, `multi_way_or_rules`)
   are run through the lowered FFN at S=100 with one-hot binary inputs
   and compared bit-for-bit against the ground truth (a Python
   reference that mirrors the §524 / §520 / §568 semantics directly).

The two gates together establish "symbolic IR ≡ lowered weights ≡
blog-spec semantics" for every primitive.

| Constructor               | Gate at S=1 (symbolic ≡ lowered) | Gate at S=100 (lowered ≡ ground truth) |
|---------------------------|----------------------------------|-----------------------------------------|
| `step_function_rule`      | yes                              | (covered by S=1 gate)                   |
| `one_hot_indicator_rule`  | yes                              | yes (per-cell sweep)                    |
| `cancel_residual_rule`    | yes                              | (covered by S=1 gate)                   |
| `multi_way_and_rule`      | yes                              | (covered by S=1 gate)                   |
| `band_range_check_rules`  | (multi-rule)                     | yes (full band sweep)                   |
| `lookup_table_rules`      | (multi-rule)                     | yes (per-key sweep)                     |
| `multi_way_or_rules`      | (multi-rule)                     | yes (per-condition sweep)               |
| `magic_floor_rules`       | (multi-rule)                     | yes (range sweep)                       |
| `bit_range_extract_rules` | (multi-rule)                     | yes (range sweep)                       |
| `efficient_exp_attention` | analytic check vs `e^N`          | softmax1+ALiBi-0 byte-identity          |
| `memory_load_attention`   | per-addr byte-identity           | byte-identity                            |
| `fetch_byte_attention`    | per-offset byte-identity         | byte-identity                            |
| `opcode_expert_rules`     | (multi-rule)                     | per-opcode gating sweep                  |

---

## 4. Canonical usage reference

The canonical usage and verification recipes live in
[`tests/test_building_blocks_dsl.py`](../tests/test_building_blocks_dsl.py).
When in doubt about parameter conventions (`write_value`, `S`,
`threshold`), read the test for the primitive in question — every
constructor has at least one minimal symbolic test and one
multi-state neural test that documents the intended call shape.

For the layer-level migration patterns that consume these helpers,
see also:

- [`FFN_RULE_MIGRATION_PATTERN.md`](FFN_RULE_MIGRATION_PATTERN.md) —
  the per-layer rewrite recipe.
- [`ATTENTION_HEAD_IR_MIGRATION_PATTERN.md`](ATTENTION_HEAD_IR_MIGRATION_PATTERN.md)
  — attention DSL.
- [`IR_DSL_DESIGN.md`](IR_DSL_DESIGN.md) — the wave-by-wave migration
  status (V1-V7 DONE; V8 wide-ALU DSL in flight; V9 lint ratchet
  shipped).

---

## 5. Lint ratchet

`tools/lint_raw_ffn_rule.py` walks `neural_vm/**/*.py` and flags raw
`FFNRule.constant_write` / `FFNRule.gated_write` calls outside the
DSL modules. The current tree has **76 raw calls across 9 files**
(captured as the V9 baseline). The lint exits 1 when:

- a baselined file grows beyond its baseline count, or
- a previously unbaselined file gains a raw call.

Each migration through a building-blocks helper should decrement the
file's baseline entry (or remove it) in the same commit so the ratchet
only moves downward.

Run locally:

```bash
python c4_release/tools/lint_raw_ffn_rule.py            # exit 0 → no regression
python c4_release/tools/lint_raw_ffn_rule.py --list     # full hit listing
python c4_release/tools/lint_raw_ffn_rule.py --json     # machine-readable
```

Tests: `c4_release/tests/test_lint_raw_ffn_rule.py`.
