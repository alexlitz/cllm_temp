# AttentionHeadIR migration pattern (Phase 6 Wave 1B)

Canonical recipe for migrating one imperative attention bake into the
declarative `AttentionHeadIR` form so the compiler synthesizes Q/K/V/O
weights from data. Companion to
[`PHASE_6_DECLARATIVE_WEIGHT_AUTHORING_PLAN.md`](PHASE_6_DECLARATIVE_WEIGHT_AUTHORING_PLAN.md)
and to `FFN_RULE_MIGRATION_PATTERN.md` (Wave 1C).

## 1. What this pattern does

Legacy attention bakes in `vm_step._set_layerN_*` write directly into
`attn.W_q`, `attn.W_k`, `attn.W_v`, `attn.W_o`:

```python
attn.W_q.data[h*HD + 0, BD.MARK_AX]      = 15.0
attn.W_k.data[h*HD + 0, BD.MARK_AX]      = 30.0
attn.W_v.data[h*HD + 1, BD.OP_LI]        =  0.2
attn.W_o.data[BD.OP_LI_RELAY, h*HD + 1]  =  1.0
```

Each line is one `(matrix, row, col, weight)` quadruple — a flat
imperative description of the head. The compiler has no way to reason
about it, dedup it, verify it, or relayout it.

The Phase 6 form moves every such quadruple into a
[`DeclarativeAttentionHeadSpec`](../neural_vm/unified_compiler/primitives.py)
inside an
[`AttentionHeadIR`](../neural_vm/unified_compiler/ir.py)
inside the op's `compiler_ir` (or `compiler_ir_factory`). The op no
longer touches `attn.W_*` directly; instead, the compiler lowers the
spec via `CompilerIR.lower_attention` →
`Primitives.generate_attention_head`. Same bytes hit the matrices, but
authorship is data, not code.

Key types involved (all in `neural_vm/unified_compiler/`):

| Type | Defined in | Role |
|---|---|---|
| `AttentionProjectionWrite` (`AP`) | `primitives.py:31` | One `(slot, dim, weight)` Q/K/V write. |
| `AttentionOutputWrite` (`AO`) | `primitives.py:46` | One `(out_dim, slot, weight)` W_o write. |
| `DeclarativeAttentionHeadSpec` | `primitives.py:56` | One head: `head_idx`, tuples of `q`, `k`, `v`, `o`. |
| `AttentionHeadIR` | `ir.py:271` | CompilerIR adapter wrapping a spec with optional `name`/`metadata`. |
| `AttentionOp` | `ir.py:224` | One layer's collection of heads (`rules: list[AttentionHeadIR]`). |
| `CompilerIR.layer(i).attention` | `ir.py:701` | Where heads get attached for layer `i`. |
| `AttentionHeadAllocator` | `attention_head_allocator.py` | Names + pins `head_idx` per layer. |

The lowering machinery is already wired: an op exposes
`compiler_ir=CompilerIR(...)` (or builds one lazily via
`compiler_ir_factory(dim_positions, head_dim)`), and the layer compiler
calls `CompilerIR.lower_attention` to emit the same writes the
imperative helper used to do.

## 2. Worked example — `layer7_memory_heads`

This op is the cleanest reference because it owns six L7 heads (2-7) and
has already been migrated to the declarative form. The head_idx
allocator landed in the same commit, so it also demonstrates the
allocator-pinning idiom.

### Pre-migration (imperative, conceptual)

The legacy `vm_step._set_layer7_memory_heads(attn, BD, HD)` wrote
hundreds of `attn.W_*[...] = ...` quadruples — one per Q/K/V/O cell for
each of heads 2 through 7. The op's `bake_fn` simply called that
helper. Adding or auditing a head meant reading imperative weight code.

### Post-migration (declarative, current)

See [`l7_ops.py:262-520`](../neural_vm/unified_compiler/ops/l7_ops.py).
Three layers:

**(a) Layout table is the single source of truth for head_idx**
([`l7_ops.py:30-65`](../neural_vm/unified_compiler/ops/l7_ops.py)):

```python
_L7_HEAD_LAYOUT = (
    ("layer7_memory_heads.head_2",       2),  # gather prev AX byte 0
    ("layer7_memory_heads.head_3",       3),  # gather prev AX byte 1
    ...
    ("layer7_memory_heads.head_7",       7),  # MEM flag broadcast
)
_L7_HEAD_LAYOUT_BY_NAME = {n: h for n, h in _L7_HEAD_LAYOUT}

def _allocate_layer7_heads() -> AttentionHeadAllocator:
    allocator = AttentionHeadAllocator(layer_max_heads=8)
    for name, head_idx in _L7_HEAD_LAYOUT:
        allocator.alloc(name, 7, pin=head_idx)
    return allocator
```

`pin=head_idx` preserves byte-identity with the legacy bake (which used
literal `head_idx=N` everywhere) while declaring the L7 head axis to
the allocator. Once Wave 6D drops `pin=`, the allocator will auto-fit
new heads.

**(b) Head specs are pure data**
([`l7_ops.py:365-520`](../neural_vm/unified_compiler/ops/l7_ops.py)):

```python
def _layer7_memory_head_specs(BD):
    L = 15.0; MEM_I = 4; AX_I = 1; SP_I = 2; BP_I = 3
    specs = [
        DeclarativeAttentionHeadSpec(
            head_idx=_L7_HEAD_LAYOUT_BY_NAME["layer7_memory_heads.head_7"],
            q=(
                AP(0, BD.MARK_MEM, L),
                AP(0, BD.H3 + MEM_I, L),
                AP(0, BD.H1 + AX_I, -L),
                ...
            ),
            k=(AP(0, BD.MARK_MEM, L),),
            v=(
                AP(1, BD.MEM_STORE,    1.0),
                AP(2, BD.MEM_ADDR_SRC, 1.0),
                AP(3, BD.OP_JSR,       1.0),
                AP(4, BD.OP_ENT,       1.0),
            ),
            o=(
                AO(BD.MEM_STORE,    1, 1.0),
                AO(BD.MEM_ADDR_SRC, 2, 1.0),
                AO(BD.OP_JSR,       3, 1.0),
                AO(BD.OP_ENT,       4, 1.0),
            ),
        )
    ]
    # heads 2-4 (band gather), 5 (flag relay), 6 (PSH/CMP relay) appended below
    ...
    return tuple(specs)
```

Notes on the annotations:

- `head_idx` is resolved through the layout table, not literal — so
  reshuffling head assignments is a one-line edit.
- `AP(slot, dim, weight)` and `AO(out_dim, slot, weight)` are the
  compact constructors from `primitives.py`; `slot` is head-local
  (final row = `head_idx * HD + slot`).
- `BD` is `_as_setdim_proxy(dim_positions)`, so dim names resolve
  through whatever layout the compiler picked (legacy or auto-fit).
- Tuples (not lists) make the spec hashable and trivially comparable.

**(c) The op exposes the IR; bake does only the residual work**
([`l7_ops.py:267-347`](../neural_vm/unified_compiler/ops/l7_ops.py)):

```python
def bake(block, dim_positions, S):
    attn = block.attn
    BD = _as_setdim_proxy(dim_positions)
    allocator      = _allocate_layer7_ffn_units(); block.ffn._l7_unit_allocator   = allocator
    head_allocator = _allocate_layer7_heads();     block.attn._l7_head_allocator  = head_allocator
    if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
        attn.alibi_slopes[1] = 5.0
        attn.alibi_slopes[5] = 5.0
        attn.alibi_slopes[6] = 5.0
    HD = attn.W_q.shape[0] // attn.num_heads
    Primitives.generate_attention_heads(attn, _layer7_memory_head_specs(BD), HD)

return Operation(
    name="layer7_memory_heads", phase=7, kind="block",
    bake_fn=bake, declarative_bake_fn=bake,
    compiler_ir_factory=_layer7_memory_heads_ir,  # <-- IR exposed here
    layer_idx=7, migrated=True, claims=_claims,
    ...
)

def _layer7_memory_heads_ir(dim_positions, HD) -> CompilerIR:
    BD = _as_setdim_proxy(dim_positions)
    ir = CompilerIR()
    ir.layer(0).attention.extend(_layer7_memory_head_specs(BD))
    return ir
```

The bake currently still calls `Primitives.generate_attention_heads`
directly for byte-identity validation, but Wave 6B will shrink the body
to the standard `_lower_via_compiler_ir(block, ir, dim_positions, S)`
once `compare_symbolic_to_lowered_attn` (Wave 1D) has been green for an
entire release cycle. The IR factory is the load-bearing piece; the
bake body becomes mechanical.

Alibi slope overrides and allocator-pinning remain in the bake because
they are residual-side bookkeeping, not weight writes. They will move
into the IR's `metadata` block in Wave 6.

## 3. The 7-step recipe

1. **Identify the head's Q/K/V/O writes in the legacy bake.**
   Open `vm_step._set_layerN_*` (or the inline bake). For each head
   `h`, list every `attn.W_q[h*HD+s, d] = w`, `attn.W_k[..]`,
   `attn.W_v[..]`, `attn.W_o[d, h*HD+s] = w`. Group by head so each
   spec is self-contained. Treat `attn.alibi_slopes[h] = s` and any
   row-multiply post-step (e.g. the K-doubling on L7 head 5) as
   carry-along metadata you will fold into the spec weights.

2. **Express each head as a `DeclarativeAttentionHeadSpec`.**
   ```python
   DeclarativeAttentionHeadSpec(
       head_idx=...,
       q=(AP(slot, BD.<DIM>, weight), ...),
       k=(AP(...), ...),
       v=(AP(...), ...),
       o=(AO(BD.<OUT_DIM>, slot, weight), ...),
   )
   ```
   Use `_as_setdim_proxy(dim_positions)` as `BD`. Use tuples, not
   lists. Fold any post-step row multiply into the affected
   projection's `weight`s.

3. **Wrap in `AttentionHeadIR` with allocator-resolved `head_idx`.**
   Add the head to your layer's `_LN_HEAD_LAYOUT` table; resolve
   `head_idx` via `_LN_HEAD_LAYOUT_BY_NAME["<op>.head_<n>"]`. Pin to
   the legacy `head_idx` until byte-identity is locked:
   ```python
   allocator.alloc("layerN_<op>.head_<n>", layer_idx=N, pin=existing_idx)
   ```
   Drop the `pin=` only after Wave 5 validation passes (this is the
   Wave 6D auto-fit demonstration).

4. **Expose via `compiler_ir=AttentionOp(heads=[...])` (or factory).**
   For ops whose head set is layout-independent, set `compiler_ir`
   directly on the `Operation(...)`. For ops where dim positions
   matter, pass `compiler_ir_factory=_<op>_ir` returning a
   `CompilerIR` whose `layer(0).attention` is populated. The
   layer compiler calls
   [`_operation_compiler_ir`](../neural_vm/unified_compiler/ir.py)
   (`ir.py:1149`) and lowers it automatically.

5. **Shrink the bake_fn body.** Replace the imperative writes with
   `_lower_via_compiler_ir(block, ir, dim_positions, S)` (or the
   equivalent direct `Primitives.generate_attention_heads(attn, specs,
   HD)` call for now). Keep only the residual side-effects (alibi
   slopes, allocator hookups, claim-set) outside the lowering.

6. **Byte-identity gate via `compare_symbolic_to_lowered_attn`** (Wave
   1D will land the tool — until then use
   `verify_attention_head(ops_for_competition=...)` from
   [`attention_verifier.py:462`](../neural_vm/unified_compiler/attention_verifier.py),
   plus a diff of `attn.W_*` against the pre-migration baseline). Run
   per query position. Any non-zero diff aborts the migration: revert
   and identify whether the imperative bake had a hidden side-effect
   the spec doesn't capture (almost always: row multiply, alibi slope,
   shared slot with another op).

7. **Commit per head/op with the byte-identity gate green.** One head
   (or one tightly-coupled head family like L7 heads 2-4) per commit.
   Title: `phase6: migrate <op> to AttentionHeadIR`. Body includes
   the verifier output and the diff against the imperative baseline.
   Per-head commits keep the bisect surface small if a later wave
   introduces a regression.

## 4. Gotchas

- **Alibi slopes.** `attn.alibi_slopes[h] = s` is not part of the
  spec yet. Until Wave 6 adds it to `AttentionHeadIR.metadata`, set it
  in the bake_fn alongside the lowering call (see L7 example above).
  Forgetting it produces a head that softmaxes correctly per query but
  attends to the wrong position.

- **Heads sharing a layer.** Two ops can co-own one head_idx (e.g.,
  L7 head 7 is primary-owned by `layer7_memory_heads` and reused by
  `format_pointer_extraction` when conversational-IO is gated on).
  Both ops must resolve their `head_idx` through the same layout
  table; the allocator forbids re-pinning the slot, so the extension
  op reads `_LN_HEAD_LAYOUT_BY_NAME` rather than calling
  `alloc(..., pin=...)` a second time. Document the shared ownership
  in the layout-table comment.

- **Cross-step heads needing PREV_STEP refs after B9.** Heads that
  read the previous step's residual (e.g. step-end relay heads, ALU
  carry forward) currently express dim refs against the current
  step's layout. After the B9 output_hi split (see
  `B9_OUTPUT_HI_SPLIT_SPEC.md`) some dims move; specs that hardcode
  these must either go through `BD` (recommended — `_as_setdim_proxy`
  resolves through the active layout) or be parameterized over a
  `prev_step_positions` map. Avoid raw integer dim literals in any
  cross-step spec.

- **Heads with no explicit Q (wildcard scope).** Some legacy heads
  set `W_q` to a constant that softmaxes uniformly over every token —
  e.g., `W_q[h*HD, BD.CONST] = q_val`. Express this with `AP(0,
  BD.CONST, q_val)` and a single K threshold; do NOT omit the Q
  tuple, or the symbolic verifier will warn on a zero-norm query.

- **Row multiplies after the helper.** Some legacy bakes do
  `attn.W_k.data[h*HD] *= 2.0` after the main helper to sharpen the
  softmax (the L7 head-5 example). Fold the multiplier into each
  `AP(...)` weight in the spec; do NOT preserve the post-step
  multiply in the bake — that would double-apply.

- **Shared slots inside one head.** Multiple `AP` entries with the
  same `slot` accumulate (per `Primitives.generate_attention_head`,
  the write is `W_*[base+slot, dim] = weight`, but spec authors are
  expected to keep each `(slot, dim)` pair unique). The L7 head-5
  spec deliberately uses `AP(4, BD.OP_AND, 0.2)` plus `AP(4,
  BD.OP_OR, 0.2)` plus `AP(4, BD.OP_XOR, 0.2)` for OR-style relay —
  three writes into the same slot, distinct dims. Confirm the
  imperative bake intended this; if it intended one-of-N, refactor
  with distinct slots.

## 5. Migration checklist

For each op being migrated, run through these before opening the PR:

- [ ] All `attn.W_q`, `W_k`, `W_v`, `W_o` writes for this op are
      moved into a `DeclarativeAttentionHeadSpec` per head.
- [ ] Each head's `head_idx` is sourced from
      `_LN_HEAD_LAYOUT_BY_NAME`, not a literal.
- [ ] The op's layer has an `_LN_HEAD_LAYOUT` entry + allocator
      factory; `pin=` is preserved.
- [ ] `Operation(...)` carries `compiler_ir=` or
      `compiler_ir_factory=`. `migrated=True` is set.
- [ ] `declarative_bake_fn=bake` mirrors `bake_fn=bake`.
- [ ] `bake_fn` body is reduced to allocator setup + alibi slope
      override + a single `Primitives.generate_attention_heads(...)`
      call (or `_lower_via_compiler_ir(...)` once Wave 1D lands).
- [ ] `claims=` covers every `(layer, "attn_W_v"|"attn_W_o", slot,
      dim)` written by the spec — fed by a loop over the spec, not
      hand-maintained.
- [ ] `compare_symbolic_to_lowered_attn` (or `verify_attention_head`
      until Wave 1D) green on this op.
- [ ] `attn.W_*.data` diff against pre-migration baseline is
      bit-exact zero.
- [ ] Commit message: `phase6: migrate <op> to AttentionHeadIR`.
      One head/family per commit.
- [ ] `verify_claims_static` still green corpus-wide.

When every op in `all_core_ops()` clears the checklist, Phase 6 Wave 2
is complete and Wave 4 (BIG helpers) inherits the same pattern at scale.
