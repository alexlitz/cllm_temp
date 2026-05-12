# Dim-ownership claim registry — bake author guide

**Status**: Phase 1 / Agent B of
[`ARCH_LEAKAGE_FIX_PLAN.md`](ARCH_LEAKAGE_FIX_PLAN.md) — landed
2026-05-12. Opt-in framework instrumentation; warnings, no hard fails yet.

## What it is

Each `Operation` accepts an optional `claims: Set[Tuple[int, str, str]]`
field. Each tuple is `(layer_idx, scope, identifier)` and declares "this
op writes a specific weight slot". The `LayerCompiler` aggregates claims
across all registered ops at `compile()` time and emits a
`DIM-OWNERSHIP COLLISION` warning whenever two ops claim the same triple.

The registry catches **(head, slot)** and **(unit, dim)** collisions that
the existing `reads`/`writes` (dim-name granularity) cannot. Today's L5
head 6 collision fix (commit `c1a5398`) is the motivating example: two
`_set_*` functions both wrote to rows `6*HD+1..6*HD+16` of `attn5.W_v`
and silently aliased opcode-flag values into ENT/JSR relay values. With
claims declared, the compiler would have emitted::

    DIM-OWNERSHIP COLLISION: layer=5 scope='attn_W_v' identifier='6_5'
        claimed by 2 ops: ['function_call_weights', 'layer5_fetch']

## Allowed scopes

`ALLOWED_CLAIM_SCOPES` (in `layer_compiler.py`):

| Scope         | Identifier convention      | Slot meaning                  |
|---------------|----------------------------|-------------------------------|
| `attn_W_v`    | `"<head>_<slot>"`          | row of `attn.W_v`             |
| `attn_W_k`    | `"<head>_<slot>"`          | row of `attn.W_k`             |
| `attn_W_q`    | `"<head>_<slot>"`          | row of `attn.W_q`             |
| `attn_W_o`    | `"<head>_<slot>"`          | column of `attn.W_o`          |
| `ffn_W_up`    | `"<unit_idx>"`             | row of `ffn.W_up` (hidden u.) |
| `ffn_W_down`  | `"<unit_idx>"`             | column of `ffn.W_down`        |
| `ffn_W_gate`  | `"<unit_idx>"`             | row of `ffn.W_gate`           |
| `embed_row`   | `"<token_id>"`             | row of token-embedding table  |

For attention scopes the row index = `head_idx * HD + slot`. The
identifier carries `<head>_<slot>` (not the raw row index) so the same
claim string is portable across different `HD` values — useful when
`d_model` changes due to `pin_io_only` layout shifts.

## Declaring claims

```python
def make_my_attn_op() -> Operation:
    _claims = set()
    # Head 7 V slots 1..32 (PC→AX_CARRY JSR relay on L6 attn).
    for slot in range(1, 33):
        _claims.add((6, "attn_W_v", f"7_{slot}"))
    return Operation(
        name="my_op",
        reads={...},
        writes={...},
        kind="attn",
        bake_fn=bake,
        claims=_claims,
    )
```

Claims are validated at `add_op` time:

- Bad scope strings raise `ValueError`.
- Non-3-tuples raise `ValueError`.
- Non-`int` layer indices raise `ValueError`.
- Non-`str` identifiers raise `ValueError`.

## Inspecting the registry

```python
compiler = LayerCompiler()
# ... declare dims, add ops ...
registry = compiler.build_claim_registry()
# registry: Dict[(layer, scope, ident), List[op_name]]

# Manually scan for collisions without compiling:
for key, owners in registry.items():
    if len(owners) >= 2:
        print(f"COLLISION at {key}: {owners}")
```

`compiler.compile()` runs `_detect_claim_collisions()` automatically and
warns via `warnings.warn(...)` on every collision found.

## Currently annotated ops

| Op | Branch where claimed | Slots claimed |
|---|---|---|
| `layer5_fetch` | `op-dim-ownership-registry` | L5 attn5 heads 0..5, V slots 32..63 |
| `function_call_weights` | `op-dim-ownership-registry` | L5 attn5 heads 5/6 V slots 1..32; L6 attn6 head 7 V slots 1..32 |

More ops will be annotated as bake authors opt in. Until full coverage,
the registry only catches collisions between annotated ops.

## Known-benign collisions (production)

Surfacing the registry today flags one production collision:

- **`(5, "attn_W_v", "5_32")`** — claimed by both `layer5_fetch`
  (CLEAN_EMBED_LO[0] → OPCODE_BYTE_LO[0] path) and
  `function_call_weights` (EMBED_HI[15] → TEMP[31] ENT BP→TEMP path).
  Both write to row `5*HD + 32` but at distinct columns (input dims).
  Because the row-granularity registry can't see column disjointness,
  this surfaces as a warning even though the two writes don't actually
  alias: the attention patterns of the two bakes are gated by mutually
  exclusive markers (PC vs STACK0) so V is only delivered through W_o
  at non-overlapping positions.

The production smoke test (`test_dim_ownership.py`) keeps an explicit
`KNOWN_BENIGN_COLLISIONS` allowlist for this entry. Any *new* collision
the registry surfaces is a real bug to investigate before being added
to the list.

The row-granularity limitation will be addressed by Phase 3 / Agent F
(position-aware writes): augment the claim identifier with the input
column it touches, which would distinguish "row 32, col EMBED_HI+15"
from "row 32, col CLEAN_EMBED_LO+0" and avoid the false positive.

## Backwards compatibility

`Operation.claims` defaults to an empty set, so every existing op without
claims continues to work. The collision scan only warns; it never raises.
That keeps the legacy bake path safe while the framework gains structural
defense against the recurring "two writers, same slot" pattern.

## Future work (Phase 3, plan doc)

- **Make collisions a hard error** once enough ops are annotated.
- **Position-aware `writes`** (Phase 3 / Agent F): augment `writes` with
  a `PositionTag` so the framework can catch "L4 incidentally writes
  OUTPUT_LO at AX marker" collisions that aren't slot collisions but
  semantic residual-stream collisions.

## See also

- [`ARCH_LEAKAGE_FIX_PLAN.md`](ARCH_LEAKAGE_FIX_PLAN.md) — the umbrella
  plan that scopes this registry as Phase 1 / Agent B.
- `c4_release/neural_vm/unified_compiler/layer_compiler.py` —
  `ALLOWED_CLAIM_SCOPES`, `Operation.claims`,
  `LayerCompiler.build_claim_registry`,
  `LayerCompiler._detect_claim_collisions`.
- `c4_release/tests/test_dim_ownership.py` — synthetic + production
  smoke coverage.
