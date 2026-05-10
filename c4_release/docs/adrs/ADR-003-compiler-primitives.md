# ADR-003: Declarative weight-setting primitives

Status: Accepted (2026-05)

## Context

Pre-migration, weight-setting code was a sea of raw indexed assignments: `ffn.W_up[unit, dim] = value`, scattered across `_set_layerN_*` functions. Each function knew the dim layout, the unit-allocation scheme, and the magic activation thresholds. This worked but had three problems:

1. **Brittle.** Changing the dim layout required hand-editing dozens of call sites.
2. **Opaque.** The intent (e.g. "this unit decrements register R when opcode O is active") was buried in arithmetic.
3. **Resists re-baking.** The compiler couldn't reason about a raw assignment, so layout planning had to treat every legacy function as a black box.

## Decision

Introduce a `Primitives` class (`c4_release/neural_vm/unified_compiler/primitives.py`) exposing named, intent-revealing helpers:

- `register_decrement_unit`
- `marker_write_unit`
- `opcode_gated_pc_override`
- `byte_passthrough_chain`
- `threshold_attention_head`
- `carry_forward_attention_head`
- `nibble_rotation_chain`
- `memory_addr_head`

Each method takes high-level arguments (which register, which opcode, which threshold) and translates them to weight writes via the allocator and dim-bridge layers. New ops MUST call into `Primitives` rather than mutate `W_up` / `W_down` / `attn` matrices directly.

## Consequences

- **Pro:** Intent is named in code; review becomes possible without simulator runs.
- **Pro:** The compiler can introspect primitive calls and re-plan layout without touching op bodies.
- **Pro:** Dim-layout changes happen inside `Primitives` only.
- **Con:** Adding a genuinely new pattern requires extending `Primitives` first, which is friction. This is the right friction — it forces shared vocabulary.
- **Migration:** Legacy raw-patch sites are converted opportunistically, op by op. No flag day.
