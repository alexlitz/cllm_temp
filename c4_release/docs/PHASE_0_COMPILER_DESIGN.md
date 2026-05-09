# Phase 0: Compiler Design for Auto-Allocated Architecture

**Status:** Design doc. The infrastructure partly exists (`unified_compiler/auto_allocator.py`) but is not wired to production. Full work is multi-week.

This doc captures (a) the user-visible architectural target, (b) what's already built, and (c) the concrete migration plan to get there.

---

## Target architecture

The compiler decides everything about the model layout from the operation specification:

1. **`d_model` auto-allocated.** The compiler computes the total residual-stream dimension from the operations being compiled. Today's hardcoded `d_model=512` becomes a derived quantity.

2. **Per-opcode dim allocation.** Different opcodes write to different dim ranges, so they don't fight for the same `OUTPUT_LO`/`OUTPUT_HI` space.
    - `ADD_RESULT` for ADD, `OR_RESULT` for OR, etc. — each opcode has its own staging area.
    - A final routing layer copies the active opcode's result to the canonical AX register dims.
    - Eliminates whole class of "OP_X leaks at MARK_PC" bugs because each op only writes to its own slot.

3. **Opcode indicator** is the single source of truth for "which opcode is this step?"
    - One dim per opcode (already exists: `OP_LEA`, `OP_IMM`, ..., `OP_NOP`).
    - Set by the embedding from the bytecode (already done).
    - Every routing FFN gates on this rather than threshold-checking residual flags.

4. **Layer allocation compiler.** Takes a list of operations and produces a model layout:
    - Operation list: `[CarryForwardPC, IncrementPC, FetchOpcode, FetchImmediate, ALUAddSub, ALUOrXorAnd, ..., RoutePCByOpcode, RouteAXByOpcode]`.
    - Each op declares `reads`, `writes`, `computes`.
    - Compiler runs dependency analysis, picks block layer for each op, sizes each layer's FFN to its actual unit count.

End result: `set_vm_weights` becomes a thin glue function that calls the compiler. The 2700-line hand-crafted current implementation is replaced by per-operation primitives that compose.

---

## What's already built

Located in `c4_release/neural_vm/unified_compiler/`:

| Module | What it does |
|---|---|
| `auto_allocator.py` | `AutoAllocator` class. Takes `DimSpec` declarations (name, size, written_by, read_by, pinned, aliasable) and assigns positions. Supports liveness-based reuse. **Works, not wired.** |
| `primitives.py` | 2472 lines of declarative primitives (PC carry, ALU stages, MEM ops, ...). Operation-named, not layer-named. **Works, partial coverage.** |
| `compiler.py` | 3966 lines: `UnifiedVMCompiler`. **Works for some opcodes, not full coverage.** |
| `ir.py` | `CompilerIR`, `AttentionOp`, `FFNOp`, `LayerSpec` data classes. |
| `verification.py` | Compares manual vs compiled models for equivalence. |

Today's production path goes through `set_vm_weights` (hand-set, layer-named, hardcoded). The compiler infrastructure is parallel, partly built, and untested in production.

---

## Concrete migration plan

This is the actual work required. Each phase ends with a green test gate.

### M1 — Per-opcode dim allocation in `AutoAllocator` (1 week)

Currently `AutoAllocator` has shared dims like `OUTPUT_LO`. Add a sibling primitive for per-opcode allocation:

```python
allocator.declare_per_opcode(
    name="RESULT", size=16,
    opcodes=[OP_ADD, OP_SUB, OP_OR, OP_XOR, OP_AND, OP_SHL, OP_SHR, OP_MUL, OP_DIV, OP_MOD, OP_EQ, ...],
    written_by=ALU_LAYER,
    read_by=[ROUTE_AX_LAYER],
)
# expands to 11 separate dim ranges: ADD_RESULT, SUB_RESULT, OR_RESULT, etc.
```

**Test gate:** unit tests on the allocator producing the expected ranges.

### M2 — Layer allocation compiler skeleton (1 week)

New module `unified_compiler/layer_compiler.py`:

```python
class LayerCompiler:
    def __init__(self):
        self.ops: List[Operation] = []
    
    def add(self, op: Operation): ...
    
    def compile(self) -> ModelLayout:
        # 1. Build dependency DAG
        # 2. Topological sort
        # 3. Assign each op to earliest legal layer
        # 4. Compute d_model from peak live-dim usage
        # 5. Compute n_layers from longest dependency chain
        return ModelLayout(d_model=..., n_layers=..., ops_per_layer=[...])
```

Each `Operation` has:
- `name: str`
- `reads: List[str]` — dim names it depends on
- `writes: List[str]` — dim names it produces
- `kind: AttentionOp | FFNOp` — what kind of weights it bakes
- `gate: Optional[OpcodeFlag]` — only fires on specific opcode

**Test gate:** compile a small spec (just IMM + EXIT) and produce a working model.

### M3 — Migrate ALU primitives to operation-based (2 weeks)

Move each `_set_layerN_*` setup function to a self-contained `Operation` primitive:

```python
# vm_step.py: _set_layer8_alu(ffn8, S, BD)  ← removed
# new: unified_compiler/ops/alu.py
class ALUAddSubOp(Operation):
    reads = ["ALU_LO", "AX_CARRY_LO", "OP_ADD", "OP_SUB", "MARK_AX"]
    writes = ["ADD_RESULT_LO", "SUB_RESULT_LO", "ADD_CARRY", "SUB_BORROW"]
    
    def bake(self, ffn, dims, S):
        # Same weight-setting as today's _set_layer8_alu, but layer-agnostic
        ...
```

Migrate ALU first since they're the simplest cross-cutting ops.

**Test gate:** Phase 1 + Phase 2 gate tests still pass with compiler-generated weights for ALU. Set `NEURAL_VM_WEIGHT_MODE=compiled` in CI.

### M4 — Migrate routing primitives (2 weeks)

Migrate the per-opcode L6 routing logic. Each opcode gets its own routing primitive:

```python
class RouteIMMOp(Operation):
    reads = ["FETCH_LO", "FETCH_HI", "OP_IMM", "MARK_AX"]
    writes = ["AX_BYTE_0_LO", "AX_BYTE_0_HI"]  # or whatever the per-opcode result name is
    ...

class RouteAXByOpcode(Operation):
    """Final routing: select active opcode's result and write to canonical AX dims."""
    reads = ["IMM_RESULT_LO", "ADD_RESULT_LO", "OR_RESULT_LO", ..., "OP_IMM", "OP_ADD", "OP_OR", ...]
    writes = ["OUTPUT_LO", "OUTPUT_HI"]
```

This is where per-opcode dim allocation pays off: each ALU op writes to its own staging dim, routing reads from the matching one based on opcode flag, only one writer per dim per layer.

**Test gate:** all gate tests pass; spurious-firing patches (`OPCODE_BLOCK_MAP`, `MARK_PC` blockers) become unnecessary because the architectural separation prevents the underlying bug.

### M5 — Eliminate `set_vm_weights` and the hand layout (1 week)

`weight_setter.py` only has `_set_compiled_weights`. The 2700-line `set_vm_weights` deletes. Block setup is entirely compiler-driven.

**Test gate:** all production tests still pass. `NEURAL_VM_WEIGHT_MODE=hand_set` is removed.

### M6 — Auto `d_model` and `n_layers` (1 week)

Stop hardcoding `d_model=512, n_layers=17` in `run_vm.py`. The compiler outputs these as derived quantities:

```python
layout = compiler.compile()
model = AutoregressiveVM(d_model=layout.d_model, n_layers=layout.n_layers)
compiled_setter(model, layout)
```

**Test gate:** all tests pass; `d_model` and `n_layers` are functions of the compiled spec.

---

## Estimate

**M1–M6 total: 8 weeks** of focused architectural work, with a full test pass gate at each milestone.

This replaces "Phase 0" in the gap analysis. After M6, every block in the model has exactly one PureFFN and every layer assignment is the compiler's call.

---

## What we should NOT do

- **Half-rewrite the production path** while the compiler is still incomplete. Today's hand-set weights pass 1096+ tests; mid-migration would break that.
- **Try to satisfy the policy via cosmetic class renames** (e.g., make wrappers inherit from `PureFFN` while the forward still does non-FFN ops). Type compliance without semantic compliance is a lie.
- **Mix hand-set and compiled paths in the same run.** Either-or, gated by `WeightMode`.

---

## Recommended next step

If we want progress on Phase 0 in the next session, the highest-leverage start is **M1** (per-opcode dim allocation) since it's a pure-Python infrastructure change with no model-runtime impact, fully testable in isolation. Once M1 lands, M2-M6 can be done sequentially with stable test gates between them.
