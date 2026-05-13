# Declarative bake: making Operation specs the implementation

Status: vision doc — not yet a roadmap.

## The question

Today's `Operation` dataclass carries 11+ annotation fields (`claims`,
`produces`, `consumes_fresh`, `stale_after`, `opcodes`, `requires`,
`reset_after_step`, `alibi_slopes`, `postcondition`, `step_idx`,
`smoke_tests`, `spec_section`, `compaction_safe`, `ffn_units_used`, plus
sentinel keys `__module_replacement` / `__structural`). The
`decl_verifier` checks each one against the actual bake behavior at
test time.

**But declarations are descriptive, not generative.** The real
implementation lives in `vm_step.py::_set_*` helper functions — hundreds
of imperative Python lines that write `attn.W_q[i, j] = val`,
`ffn.W_up[k, m] = scale` cell-by-cell.

**Why aren't the declarations the implementation directly?**

## What "declarative bake" would look like

Instead of:

```python
def _set_layer7_operand_gather(attn, S, BD, HD):
    L = 15.0
    base = 0 * HD
    attn.W_q[base, BD.MARK_AX] = L
    attn.W_q[base, BD.OP_LEA] = -L
    attn.W_k[base, BD.STACK0_BYTE0] = L
    for k in range(16):
        attn.W_v[base + 1 + k, BD.CLEAN_EMBED_LO + k] = 1.0
        attn.W_v[base + 17 + k, BD.CLEAN_EMBED_HI + k] = 1.0
    for k in range(16):
        attn.W_o[BD.ALU_LO + k, base + 1 + k] = 6.0   # amplified
        attn.W_o[BD.ALU_HI + k, base + 17 + k] = 6.0
```

…we'd write:

```python
Operation(
    name="layer7_operand_gather",
    kind="attn",
    head=0,
    q_gates={"MARK_AX": +L, "OP_LEA": -L},
    k_attends={"STACK0_BYTE0": +L},
    v_copy_from={"CLEAN_EMBED_LO": 1.0, "CLEAN_EMBED_HI": 1.0},
    output_to={"ALU_LO": 6.0, "ALU_HI": 6.0},
    requires={"STACK0_BYTE0": "within_one_step"},
    opcodes={"OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV", "OP_MOD",
             "OP_AND", "OP_OR", "OP_XOR", "OP_SHL", "OP_SHR",
             "OP_LEA", "OP_ADJ", "OP_ENT"},
    produces={"ALU_LO": "AX_byte0", "ALU_HI": "AX_byte0"},
)
```

…and a single compiler pass `generate_weights(spec, target, layout)`
turns the spec into the W_q/W_k/W_v/W_o writes. The `_set_*` helper
disappears.

## Benefits

1. **Single source of truth.** No drift possible — the spec IS the
   implementation. The verifier becomes redundant for these ops.
2. **More verifiable.** Specs are structural; humans + tools can
   reason about them. Imperative code requires line-by-line inspection.
3. **Composable primitives.** "Attention head with Q/K/V/O projections"
   becomes a first-class primitive, not an unspoken convention scattered
   across 50 helper functions.
4. **Globally optimizable.** The compiler can see all bakes at once and
   make consistency decisions (e.g., normalize all ALiBi slopes,
   auto-pad FFN units to match `ffn_units_used`).
5. **Annotation-as-source.** Today's `opcodes={'OP_ADD'}` field would
   automatically generate `W_q[base, BD.OP_ADD] = +L` — one less hand-
   written line that can drift from the declaration.

## Why we're not there yet

1. **Historical.** Code grew imperatively over months. Declarations
   landed as a verification overlay weeks ago, not as the original
   interface.
2. **Expressiveness gap.** Simple attention heads (gather X, copy Y,
   write Z) fit the declarative template cleanly. The hard cases:
   - Carry-propagation post-ops (write into MLP intermediate state)
   - Lookup-table FFNs (DIV/MOD, AND/OR/XOR — table lookup per opcode)
   - MUL multi-stage pipelines (8 sub-FFNs with carry propagation)
   - ALiBi slope mutations (kind="model" ops that touch attn config)
   - Wrapper installs (`block.ffn = HybridALUBlock(...)`)

   These are handled today via `kind="block"` wrapper installs +
   special-case `bake_fn` bodies. The declarative DSL would need
   first-class support for each pattern (lookup tables as `tables=`,
   pipelines as `stages=[...]`, module replacement as `replace_module=`).

3. **Already partway there.** `c4_release/neural_vm/unified_compiler/primitives.py`
   has `Primitives.carry_forward_attention()` — exactly this pattern at
   the primitive level. ~30% of attention bakes already call into it
   instead of writing cells directly. The MUL pipeline ops in
   `ops/l11_ops.py` / `ops/l12_ops.py` are also semi-declarative —
   each stage is a self-contained sub-op.

4. **Bootstrapping cost.** A complete declarative DSL upfront would
   require anticipating every weight-write pattern. Imperative is easier
   to iterate. The `Primitives` module grows organically — every time a
   new bake discovers a reusable pattern, it gets pulled up.

## Migration path

Each annotation we add is a step toward generation:

| Effort | Impact |
|---|---|
| Extend `Primitives` to cover all common attention patterns (relay, gather, broadcast, marker-gated) | Replaces ~40% of bake_fns with `Primitives.*` calls |
| Add `Primitives.lookup_table_ffn(input_dims, output_dim, table)` | Replaces L10 bitwise + L13 shift bakes |
| Convert `opcodes` annotation from descriptive to generative — compiler emits `W_q[base, OP_X] = +L` for every entry in `opcodes` | Removes hand-written opcode-gate lines |
| Convert `produces` annotation to generative — compiler emits the `W_o` row from `produces={dim: marker}` + scale | Removes hand-written output-write loops |
| Single `generate_weights(spec, target, layout)` routine that consumes all annotation fields | Replaces `bake_fn` for ~70% of ops |
| `vm_step.py::_set_*` helpers gradually deleted as their callers move to specs | `vm_step.py` shrinks dramatically |

## End state

`Operation` becomes the only source of truth. `c4_release/neural_vm/vm_step.py`
shrinks to just the runtime `forward` paths. The verifier is redundant
for declaratively-baked ops (there's nothing to verify — the spec IS
the implementation; what remains to verify is whether the spec matches
the BLOG_SPEC intent, which is what `spec_section` annotations already
point at).

## What it would take

- 4-6 weeks of focused effort
- Deep familiarity with what each bake_fn does (we have this — every
  bake has a docstring + comments)
- A growing `Primitives` module + a `generate_weights()` routine + a
  `BakeSpec` extension to `Operation`
- Per-op migration is independent — can land op-by-op without big-bang

All of today's annotation work is laying the groundwork. Every
`produces` / `opcodes` / `alibi_slopes` we add is one less thing the
imperative bake_fn needs to express.

## Next concrete step

Pick one bake with a small, well-defined surface area and convert it
end-to-end as proof-of-concept. Candidate: `_set_layer7_memory_heads`
(~50 lines, only does V-relays — perfectly fits the simple-attention
template). Success criteria: replacing the helper with a declarative
spec produces byte-identical weights.

If that lands cleanly, repeat for the ~20 other "simple attention"
ops. By the time those land, ~40% of bake code is gone and the
patterns for the harder cases (lookup tables, wrappers) are clearer.
