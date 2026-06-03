# Smoke fixes don't land: efficient mode replaces ALU FFNs

## The gap

Smoke tests (`tests/test_smoke.py`) run via `conftest.py:399` with
`trust_neural_alu=True` → `alu_mode='efficient'` (see `run_vm.py:317`).
This mode REPLACES the declarative L8/L10/L11/L13 FFN bakes with
hand-baked composite modules.

Recently-applied declarative IR rule fixes targeting these layers are
**INERT** for smoke. They land only in lookup mode (which the 1096
corpus runs in).

## Audit by layer

| Layer | Efficient wrapper | Replaces block.ffn? | Status |
|-------|-------------------|---------------------|--------|
| L8 | `make_efficient_l8_addsub_wrap_op` (full_vm_compiler_dynamic.py:1537) | Yes — `block.ffn = ALUAddSub` | Lookup-only fix |
| L10 | `make_efficient_l10_andorxor_wrap_op` (alu_ops.py:366-406) | Yes — `block.ffn = ALUAndOrXor` | Lookup-only fix |
| L11 | `make_efficient_l11_alumul_wrap_op` (full_vm_compiler_dynamic.py:1539) | Yes — `block.ffn = FlattenedALUMul` | Lookup-only fix |
| L13 | `make_alu_shift_composite_ops` (all_core_ops.py:266) | Replaces L13 FFN | Lookup-only fix |
| L16 | (no efficient wrapper) | No | **Fix applies** |

## What landed this session

| Fix | File:line | Smoke (efficient) | 1096 (lookup) |
|----|----|----|----|
| OP_IMM -1M → -1e9 | `l16_ops.py:758` | **Applies** | Applies |
| _cmp_default + MARK_PC | `l10_ops.py:680` | **Inert** | Applies |
| L11 MUL attach removal | `all_core_ops.py:497-518` | **Inert** (efficient wrapper unchanged) | Applies |

## Efficient-mode equivalents (where the real fix needs to go)

| Cluster | Lookup-mode rule | Efficient-mode equivalent |
|---|---|---|
| if_eq | `_cmp_default` (l10_ops.py:665) | `ComparisonCombine` (vm_step.py:649) — **already has MARK_PC blocker** since 2026-05-09. Bug is elsewhere; re-diagnose. |
| MUL double-fire | L11/L12 post-op attach | `FlattenedALUMul` instances baked by efficient wrappers. May or may not double-fire — needs separate trace. |
| edge_pow2 | L16 OP_IMM gate | Same (L16 not replaced). |
| absdiff | run_vm.py:2094 BZ override hoist | Same hoist needed; not ALU-mode-specific. |
| add_* | L10/L9 ALU declarative | `ALUAddSub` / `PureNeuralALU` |

## Strategy

For each cluster diagnosis, dual-track:

1. **Lookup-mode declarative fix** — fixes 1096 corpus (which uses lookup).
2. **Efficient-mode equivalent** — fixes smoke (which uses efficient).

The MARK_PC blocker example shows the patterns CAN already match.
The rule: **any FFN-rule edit at L8/L10/L11/L13 needs a matching
`vm_step.py` PostOp / `efficient_alu_*` edit**.

## Open follow-ups

- Smoke if_eq: re-trace through `ComparisonCombine` (the MARK_PC blocker
  is already there; bug must be elsewhere — maybe `BDToGEConverter` or
  the L9 ALU eq result encoding).
- Smoke MUL double-fire: check whether `_expand_wrapper_blocks` also
  double-fires the efficient L11+L12 wrappers.
- Smoke absdiff: BZ override hoist needs to apply for both pure_neural
  paths (handler-mode AND pure-neural; not ALU-mode-specific).

## Status

Documentation only. No code changes.
