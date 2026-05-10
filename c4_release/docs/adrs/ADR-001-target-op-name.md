# ADR-001: Op-reference binding for block ops

Status: Accepted (2026-05)

## Context

The `LayerCompiler` migration replaces inline `_set_layerN_*` calls in `set_vm_weights` with declarative `Operation` factories in `unified_compiler/migrated_ops.py`. Each `Operation` has a `bake_fn` that writes weights to a target block.

During migration, two layer-index spaces coexist:
- The **compiler-assigned** layer: derived from the layout planner and may shift as ops are re-ordered or batched.
- The **legacy** index: the `model.blocks[N]` slot the original `_set_layerN_*` function wrote to.

When these diverge — typically because the planner has packed earlier blocks more tightly — a migrated op silently bakes into the wrong block. There is no runtime exception; the model just produces wrong outputs at the affected layer, and the failure surfaces only as a downstream test regression.

## Decision

Migrated ops that still have a legacy `set_vm_weights` counterpart MUST be declared with `kind="block"` and an explicit `layer_idx=N` matching the legacy block index. The compiler honors `layer_idx` and ignores the planner suggestion for pinned ops.

The pin can be lifted once every legacy reference for that block has been removed from `set_vm_weights` and the planner is the sole source of layer assignment.

## Consequences

- **Pro:** No silent block-mismatch during the transition. Tests catch breakage at the migrated op, not three layers downstream.
- **Pro:** Migration can proceed op-by-op without a flag-day cutover.
- **Con:** The planner's freedom is constrained on pinned blocks until cleanup completes. This is acceptable because the constraint is temporary.
- **Maintenance:** Each migrated `Operation` factory carries a comment noting whether its `layer_idx` is a temporary pin (remove when legacy refs go) or a permanent placement.
