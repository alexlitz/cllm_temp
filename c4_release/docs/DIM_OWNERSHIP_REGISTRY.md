# Dim-ownership claim registry — bake author guide

**Status**: Phase 1 / Agent B of
[`ARCH_LEAKAGE_FIX_PLAN.md`](ARCH_LEAKAGE_FIX_PLAN.md) — landed
2026-05-12. Extended with column granularity (Phase 3 / Agent F)
2026-05-12. Opt-in framework instrumentation; warnings, no hard fails
yet.

## What it is

Each `Operation` accepts an optional
`claims: Set[Tuple[int, str, str, Optional[str]]]` field. Each tuple is
`(layer_idx, scope, identifier, column)` and declares "this op writes a
specific weight slot at a specific input column". The `LayerCompiler`
aggregates claims across all registered ops at `compile()` time and emits
a `DIM-OWNERSHIP COLLISION` warning whenever two ops claim the same
4-tuple.

Legacy 3-tuples `(layer_idx, scope, identifier)` are accepted and
auto-promoted to `(layer_idx, scope, identifier, None)` at `add_op` time
for back-compat. Mixing 3-tuples and 4-tuples in the same registry is
safe: `column=None` is treated as a distinct literal value, not a
wildcard — so a partially-migrated op (column-aware) does not produce
spurious warnings against a peer still using row-only claims.

The registry catches **(head, slot, column)** and **(unit, dim)**
collisions that the existing `reads`/`writes` (dim-name granularity)
cannot. Today's L5 head 6 collision fix (commit `c1a5398`) is the
motivating example: two `_set_*` functions both wrote to rows
`6*HD+1..6*HD+16` of `attn5.W_v` **and the same input columns
`EMBED_LO+k`** — true value aliasing. With claims declared, the
compiler would have emitted::

    DIM-OWNERSHIP COLLISION: layer=5 scope='attn_W_v' identifier='6_5'
        column='EMBED_LO+4'
        claimed by 2 ops: ['function_call_weights', 'layer5_fetch']

Column granularity also resolves a class of *false positives* the
row-only registry produced: two ops writing the same row at distinct
input columns no longer warn. The production model's L5 head 5 row 32
case (``layer5_fetch`` writes ``CLEAN_EMBED_LO+0``,
``function_call_weights`` writes ``EMBED_HI+15``) is now silent without
needing an explicit allowlist.

## Allowed scopes

`ALLOWED_CLAIM_SCOPES` (in `layer_compiler.py`):

| Scope         | Identifier convention      | Column convention                | Slot meaning                  |
|---------------|----------------------------|----------------------------------|-------------------------------|
| `attn_W_v`    | `"<head>_<slot>"`          | `"<INPUT_DIM>+<offset>"` or None | row of `attn.W_v`             |
| `attn_W_k`    | `"<head>_<slot>"`          | `"<INPUT_DIM>+<offset>"` or None | row of `attn.W_k`             |
| `attn_W_q`    | `"<head>_<slot>"`          | `"<INPUT_DIM>+<offset>"` or None | row of `attn.W_q`             |
| `attn_W_o`    | `"<head>_<slot>"`          | `"<INPUT_DIM>+<offset>"` or None | column of `attn.W_o`          |
| `ffn_W_up`    | `"<unit_idx>"`             | `"<INPUT_DIM>+<offset>"` or None | row of `ffn.W_up` (hidden u.) |
| `ffn_W_down`  | `"<unit_idx>"`             | `"<OUTPUT_DIM>+<offset>"` or None| column of `ffn.W_down`        |
| `ffn_W_gate`  | `"<unit_idx>"`             | `"<INPUT_DIM>+<offset>"` or None | row of `ffn.W_gate`           |
| `embed_row`   | `"<token_id>"`             | must be None                     | row of token-embedding table  |

For attention scopes the row index = `head_idx * HD + slot`. The
identifier carries `<head>_<slot>` (not the raw row index) so the same
claim string is portable across different `HD` values — useful when
`d_model` changes due to `pin_io_only` layout shifts.

The `column` element of the 4-tuple is the input-dim name plus its
position offset within that dim, e.g. `"CLEAN_EMBED_LO+0"` for the
zero-th nibble of the clean low-byte embedding. `column=None` is
accepted for ops that haven't migrated to column granularity yet; the
collision detector treats `None` as a literal value (not a wildcard),
so partial migration is safe. `embed_row` is intrinsically row-granular
and therefore MUST have `column=None` (validated at `add_op` time).

## Declaring claims

```python
def make_my_attn_op() -> Operation:
    _claims = set()
    # Head 7 V slots 1..32 (PC→AX_CARRY JSR relay on L6 attn).
    #   W_v[7*HD + 1 + k, OUTPUT_LO + k] = 1.0    for k=0..15 (slot 1..16)
    #   W_v[7*HD + 17 + k, OUTPUT_HI + k] = 1.0   for k=0..15 (slot 17..32)
    for k in range(16):
        _claims.add((6, "attn_W_v", f"7_{1 + k}",  f"OUTPUT_LO+{k}"))
        _claims.add((6, "attn_W_v", f"7_{17 + k}", f"OUTPUT_HI+{k}"))
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
- Tuples that aren't 3- or 4-arity raise `ValueError`.
- Non-`int` layer indices raise `ValueError`.
- Non-`str` identifiers raise `ValueError`.
- Non-`str` (and non-`None`) columns raise `ValueError`.
- `embed_row` scope with a non-`None` column raises `ValueError`.

3-tuple claims are accepted (auto-promoted to 4-tuple with
`column=None`) so existing call sites continue to work.

## Inspecting the registry

```python
compiler = LayerCompiler()
# ... declare dims, add ops ...
registry = compiler.build_claim_registry()
# registry: Dict[(layer, scope, ident, column), List[op_name]]

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

With column-granular claims, the production model emits **zero**
dim-ownership collision warnings. The previous row-only false positive
at ``(5, "attn_W_v", "5_32")`` — where ``layer5_fetch`` writes
``CLEAN_EMBED_LO+0`` and ``function_call_weights`` writes
``EMBED_HI+15`` — is no longer surfaced: the column element of the
claim distinguishes the two cells.

The corresponding ``KNOWN_BENIGN_COLLISIONS`` allowlist has been
retired. Any collision warning in ``test_dim_ownership.py`` is now a
real bug to investigate.

## Backwards compatibility

`Operation.claims` defaults to an empty set, so every existing op without
claims continues to work. The collision scan only warns; it never raises.
That keeps the legacy bake path safe while the framework gains structural
defense against the recurring "two writers, same slot" pattern.

## Future work

- **Make collisions a hard error** once enough ops are annotated.
- **Position-aware `writes`**: augment `writes` with a `PositionTag` so
  the framework can catch "L4 incidentally writes OUTPUT_LO at AX
  marker" collisions that aren't slot collisions but semantic
  residual-stream collisions. Column granularity (this iteration)
  addresses input-side disjointness; output-position granularity is the
  next step.

## See also

- [`ARCH_LEAKAGE_FIX_PLAN.md`](ARCH_LEAKAGE_FIX_PLAN.md) — the umbrella
  plan that scopes this registry as Phase 1 / Agent B.
- `c4_release/neural_vm/unified_compiler/layer_compiler.py` —
  `ALLOWED_CLAIM_SCOPES`, `Operation.claims`,
  `LayerCompiler.build_claim_registry`,
  `LayerCompiler._detect_claim_collisions`.
- `c4_release/tests/test_dim_ownership.py` — synthetic + production
  smoke coverage.
