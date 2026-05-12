# Residual-dim staleness invariants — bake author guide

**Status**: Phase 3 / Agent G of
[`ARCH_LEAKAGE_FIX_PLAN.md`](ARCH_LEAKAGE_FIX_PLAN.md) — landed
2026-05-12. Opt-in framework instrumentation; warnings, no hard fails yet.

## What it is

Every `Operation` may declare two optional dim->register maps:

```python
@dataclass
class Operation:
    ...
    produces: Dict[str, str] = field(default_factory=dict)
    consumes_fresh: Dict[str, str] = field(default_factory=dict)
```

The `LayerCompiler` aggregates these across all ops at `compile()` time
and emits a `STALENESS VIOLATION` warning whenever an op declares
`consumes_fresh={dim: register}` but no earlier-phase op in the same
step declares `produces={dim: register}` for the matching pair.

This catches a class of latent bug the existing `reads`/`writes` dep
graph cannot see: **a consumer reading a residual dim that nominally
carries the right *name* but in fact holds a stale (cross-step or
leftover) *value*** because no in-step op refreshed it.

## Canonical example — the L8 AX_CARRY stale-value bug (commit `3d1b700`)

Before commit `3d1b700`, the L8 lookup ALU consumed `AX_CARRY_LO` at the
AX marker as operand 1 for ADD / SUB. The only producer of `AX_CARRY`
was L3 head 1, which reads `EMBED_LO/HI` from the **previous step's**
AX byte 0 token. In sequences like `IMM 10 / PSH / IMM 32 / ADD`, the
ADD step's `AX_CARRY_LO` landed on `10` (two steps stale) instead of
`32`, producing `ALU + AX_CARRY = 10 + 10 = 20` and a bogus output.

Commit `3d1b700` added a fresh L8 attn head 6 that refreshes
`AX_CARRY_LO/HI` from the prev step's AX marker `OUTPUT_LO/HI` in the
current step, *before* the L8 ALU fires. With staleness invariants
declared, that fix becomes a compile-time contract:

```python
# l8_ops.py

def make_layer8_head6_ax_carry_refresh_op() -> Operation:
    return Operation(
        name="layer8_head6_ax_carry_refresh",
        phase=8.05,
        ...
        produces={
            "AX_CARRY_LO": "AX_byte0",
            "AX_CARRY_HI": "AX_byte0",
        },
    )


def make_layer8_alu_op() -> Operation:
    return Operation(
        name="layer8_alu",
        phase=8.2,
        ...
        consumes_fresh={
            "AX_CARRY_LO": "AX_byte0",
        },
    )
```

Removing the head-6 op now warns at compile time:

```
STALENESS VIOLATION: op 'layer8_alu' consumes_fresh dim='AX_CARRY_LO'
register='AX_byte0' but no earlier-phase op in the same step produces it
```

The regression test
(`tests/test_staleness_invariants.py::test_removing_l8_head6_surfaces_ax_carry_staleness`)
codifies this invariant.

## Declaring producers / consumers

### `produces`

Declare `produces` whenever your op writes the canonical in-step fresh
value of a residual dim for a particular register slot. The map key is
the residual dim name (as declared via `compiler.declare_dim(...)`);
the value is a string register identifier you make up to encode the
*semantics* of the slot:

```python
# An op that writes the prev-step AX byte 0 value into AX_CARRY at the
# current step's AX marker:
produces={
    "AX_CARRY_LO": "AX_byte0",
    "AX_CARRY_HI": "AX_byte0",
}

# An op that routes AX_CARRY -> OUTPUT at AX byte positions for IMM:
produces={
    "OUTPUT_LO": "AX_byte0",
    "OUTPUT_HI": "AX_byte0",
}
```

The register string is opaque to the analyzer; it just needs to match
the consumer's declaration. By convention pick a name that describes
the position/semantics ("AX_byte0", "PC_marker", "STACK0_b0") rather
than a layer index — the same logical register may move between
layers.

### `consumes_fresh`

Declare `consumes_fresh` whenever your op relies on a residual dim
being refreshed *in the current step*. Cross-step-only reads (e.g.,
ALiBi-based prev-step relays where the operation explicitly attends
to a past position) should NOT declare `consumes_fresh` — they're
fine reading stale-from-the-name-perspective values because the
attention pattern selects the correct past position.

```python
# L8 ALU consumes the current step's AX byte 0 value to use as
# operand 1 for ADD/SUB/LEA:
consumes_fresh={"AX_CARRY_LO": "AX_byte0"}
```

### When NOT to declare `consumes_fresh`

`L3 head 1` (`make_layer3_carry_forward_attn_op`) reads `MARK_AX` at
*past* positions to relay prev-step `EMBED_LO/HI` into the current
step's `AX_CARRY`. It writes `AX_CARRY_LO/HI` but does NOT consume
a fresh in-step `AX_CARRY` — it produces one from cross-step data.
Leaving its `consumes_fresh` empty is the correct annotation.

## Analyzer algorithm

The analyzer (`LayerCompiler._detect_staleness_violations`) scans every
op with `consumes_fresh` declarations. For each `(dim, register)` pair
declared by a consumer, it walks the producer registry for the same
key. A producer counts as "in-step" if:

  1. `producer.name == consumer.name` (an op that self-satisfies), OR
  2. `consumer.phase is None and producer.phase is None` (both
     unordered), OR
  3. `consumer.phase` is set and `producer.phase <= consumer.phase`.

If no producer matches, the analyzer emits a `STALENESS VIOLATION`
warning via `warnings.warn`. Warnings are deterministic in order
(sorted by `(dim, register)` then consumer name).

## Inspection helpers

For tests / debugging tools that want to inspect the staleness graph
without re-running `compile`:

```python
from c4_release.neural_vm.unified_compiler.full_vm_compiler import (
    build_staleness_registry,
    detect_staleness_violations,
)

# producers: (dim, register) -> [(op_name, phase), ...]
# consumers: (dim, register) -> [(op_name, phase), ...]
producers, consumers = build_staleness_registry(compiler)

# Returns the list of warning message strings (without re-warning).
messages = detect_staleness_violations(compiler)
```

`LayerCompiler.build_staleness_registry()` exposes the same API for
ops that prefer the direct method.

## Relation to other Phase 3 instrumentation

- **Agent F (`op-position-aware-writes`)**: refactors
  `Operation.writes` -> `Set[Tuple[PositionTag, str]]`. Catches an
  orthogonal class of bug (an op writing a dim at a position it
  shouldn't). Both agents extend `Operation` with distinct field names
  so they can merge cleanly.

- **Agent B (`op-dim-ownership-registry`)**: catches `(layer, scope,
  identifier)` collisions on attn / ffn / embed weight slots. Lives at
  the weight-cell granularity, while staleness lives at the residual-
  dim granularity. Both layers compose naturally — a slot-level
  collision and a residual-dim staleness violation are different
  failure modes.

## Current proof-of-concept coverage

| Op | Phase | produces / consumes_fresh |
|---|---|---|
| `layer8_head6_ax_carry_refresh` | 8.05 | produces `AX_CARRY_LO/HI` -> `AX_byte0` |
| `layer8_multibyte_routing`     | 8.3  | produces `OUTPUT_LO/HI` -> `AX_byte0` |
| `layer8_alu`                   | 8.2  | consumes_fresh `AX_CARRY_LO` -> `AX_byte0` |

The L3 head 1 op (`make_layer3_carry_forward_attn_op`) intentionally
omits a `consumes_fresh` declaration: it relays prev-step EMBED into
the current step's AX_CARRY but does not consume a fresh current-step
AX_CARRY value.

Extending coverage is opt-in. As bake authors add `produces` /
`consumes_fresh` declarations, the analyzer's coverage grows
automatically.
