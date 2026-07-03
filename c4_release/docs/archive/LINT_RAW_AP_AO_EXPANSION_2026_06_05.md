# Lint expansion: raw `AP(...)` / `AO(...)` ratchet (2026-06-05)

Extends `c4_release/tools/lint_raw_ffn_rule.py` to also forbid raw
`AP(slot, dim, weight)` and `AO(out_dim, slot, weight)` calls outside
the declarative-IR / building-blocks DSL modules.

## Motivation

Wave V9 ratcheted `FFNRule.constant_write` / `FFNRule.gated_write`
calls out of every non-DSL module. The same anti-pattern persists at
the attention layer: ops still compose `DeclarativeAttentionHeadSpec`s
by hand-rolling `AP(...)` / `AO(...)` lists, bypassing the
building-block attention helpers and the `attention_head_allocator`
plumbing. New ops should compose through those helpers so the
BLOG_SPEC §504-568 building blocks are the source of truth for
attention specs too.

## Allow-listed modules (no warning)

| File                                                                     | Why exempt                                                |
|--------------------------------------------------------------------------|-----------------------------------------------------------|
| `c4_release/neural_vm/unified_compiler/primitives.py`                    | `AP` / `AO` are defined here.                              |
| `c4_release/neural_vm/unified_compiler/building_blocks_dsl.py`           | Attention building-block helpers lower to `AP`/`AO` here. |
| `c4_release/neural_vm/unified_compiler/wide_alu_dsl.py`                  | Wide-ALU helpers (Waves W1-W7).                            |
| `c4_release/neural_vm/unified_compiler/ir.py`                            | IR core types.                                             |
| `c4_release/neural_vm/attention_head_allocator.py`                       | Allocator legitimately composes raw attention writes.      |

The FFN allow-list is unchanged (`building_blocks_dsl.py`,
`wide_alu_dsl.py`, `ir.py`).

## Behaviour

- FFN ratchet remains the blocking CI gate (exit 1 on regression).
- AP/AO ratchet is **advisory by default** (warning + per-file
  baseline summary; exit code 0). Pass `--strict-ap-ao` to flip AP/AO
  regressions to fatal too.
- Both ratchets walk downward only: decrement the baseline entry in
  the same commit that migrates a file onto building-block helpers.

JSON shape extended: the original `total_raw_calls` / `regressions` /
`new_files` top-level keys are preserved (existing test asserts these);
AP/AO data lives under a nested `"ap_ao"` key with the same triple.

## Violation count (baseline at 2026-06-05)

Total: **1142** raw `AP(...)` / `AO(...)` calls across **16** files.

| Hits | File                                                                 |
|-----:|----------------------------------------------------------------------|
|  311 | `c4_release/neural_vm/unified_compiler/ops/l10_ops.py`               |
|  161 | `c4_release/neural_vm/unified_compiler/ops/l14_ops.py`               |
|  147 | `c4_release/neural_vm/unified_compiler/ops/l6_ops.py`                |
|  108 | `c4_release/neural_vm/unified_compiler/ops/l7_ops.py`                |
|   61 | `c4_release/neural_vm/unified_compiler/ops/l8_ops.py`                |
|   57 | `c4_release/neural_vm/unified_compiler/ops/model_ops.py`             |
|   56 | `c4_release/neural_vm/unified_compiler/ops/l3_ops.py`                |
|   49 | `c4_release/neural_vm/unified_compiler/ops/l15_ops.py`               |
|   47 | `c4_release/neural_vm/unified_compiler/ops/l5_ops.py`                |
|   33 | `c4_release/neural_vm/unified_compiler/ops/l9_ops.py`                |
|   25 | `c4_release/neural_vm/unified_compiler/ops/l4_ops.py`                |
|   23 | `c4_release/neural_vm/unified_compiler/ops/flag_gated_ops.py`        |
|   21 | `c4_release/neural_vm/unified_compiler/ops/l13_ops.py`               |
|   20 | `c4_release/neural_vm/unified_compiler/ops/l1_ops.py`                |
|   15 | `c4_release/neural_vm/unified_compiler/ops/control_flow_heads.py`    |
|    8 | `c4_release/neural_vm/unified_compiler/ops/l2_ops.py`                |
| **1142** | **TOTAL** |

## Migration priority

Top three files (`l10_ops.py`, `l14_ops.py`, `l6_ops.py`) hold 54% of
the violations (619/1142). Each is a candidate for a dedicated
building-block attention helper:

- **`l10_ops.py` (311)** — almost certainly the largest concentration
  is the PSH/LI/SI memory-bank heads and the L10 tail-correction op
  family; many heads share a `Q match opcode + K match prev step + V
  copy band` template that warrants a single helper.
- **`l14_ops.py` (161)** — output-cleanup chain heads. The
  `_L14_CLEANUP_CHAIN_LAYOUT` already centralises slot allocation;
  a `cleanup_chain_head_spec` helper would fold the per-head
  AP/AO lists too.
- **`l6_ops.py` (147)** — routing FFN attention scaffolding. After the
  Phase 7.C `routing_ffn` cut, these heads are the next obvious target.

## Verification

- Existing unit tests (`tests/test_lint_raw_ffn_rule.py`) pass
  unchanged (6/6).
- `python c4_release/tools/lint_raw_ffn_rule.py` on the current tree
  exits 0 with both blocks reporting "OK ... within baseline".
- Simulated regression (lowered baseline for `l10_ops.py`) under
  `--strict-ap-ao` exits 1 and prints the per-file delta as expected.

## How to migrate a file

1. Identify the per-head pattern: each head's Q/K/V/O writes that
   share structural shape.
2. Add a constructor to `building_blocks_dsl.py` that takes the
   high-level parameters (slot bands, scaling constants, ALiBi-slope
   hint) and returns the assembled `DeclarativeAttentionHeadSpec`.
3. Replace the per-head AP/AO list at the op site with a single call
   to the new helper.
4. Decrement the file's entry in `_AP_AO_BASELINE` (delete it when
   the count drops to 0) in the SAME commit.
5. `pytest c4_release/tests/test_lint_raw_ffn_rule.py` to confirm the
   ratchet still holds.
