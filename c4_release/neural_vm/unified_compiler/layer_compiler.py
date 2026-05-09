"""Layer-allocation compiler MVP.

Takes a list of declarative operations (each with reads/writes/kind/bake_fn) and
produces a ModelLayout that specifies:

    - d_model (auto-computed from peak live-dim usage)
    - n_layers (auto-computed from longest dependency chain)
    - which operation lives at which layer
    - which dim each name maps to

This is a thin layer on top of the existing AutoAllocator (which handles dim
positions). The new piece here is layer assignment by dependency analysis.

NOTHING about this MVP is wired to production yet. It's standalone and tested
in isolation. Production weight setting still goes through vm_step.set_vm_weights.

Example:
    compiler = LayerCompiler()
    compiler.declare_dim("MARK_PC", size=1)
    compiler.declare_dim("EMBED_LO", size=16)
    compiler.declare_dim("OUTPUT_LO", size=16)

    compiler.add_op(Operation(
        name="pc_carry_forward",
        reads={"MARK_PC"},
        writes={"EMBED_LO"},
        kind="attn",
        bake_fn=lambda module, dims, S: ...,
    ))
    compiler.add_op(Operation(
        name="pc_increment",
        reads={"EMBED_LO"},
        writes={"OUTPUT_LO"},
        kind="ffn",
        bake_fn=lambda module, dims, S: ...,
    ))

    layout = compiler.compile()
    # layout.d_model = 33  (computed from dim sizes + alignment)
    # layout.n_layers = 2  (carry_forward at layer 0, increment at layer 1)
    # layout.ops_at(0) = [pc_carry_forward]
    # layout.ops_at(1) = [pc_increment]
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set


@dataclass
class Operation:
    """A single declarative operation that the compiler can place at any layer.

    `reads` and `writes` are dim names (declared via `compiler.declare_dim(...)`).
    `kind` is "attn" or "ffn" — determines whether the op programs the block's
    attention or feed-forward.
    `bake_fn(module, dim_positions, S)` is invoked at compile time to actually
    write the weights into the assigned attention or FFN module.

    `phase` is an optional ordering hint. When the dep graph has ambiguity
    (e.g., two ops both read+write the same dim at different positions, which
    the dim-name-only dep model can't distinguish), the compiler uses phase to
    break ties: an edge u→v in the dep graph is dropped if u.phase > v.phase.
    Smaller phase = earlier. Use the original `_set_layerN_*` layer number as
    the phase for migrated ops to preserve hand-set order.
    """

    name: str
    reads: Set[str]
    writes: Set[str]
    kind: str  # "attn" or "ffn"
    bake_fn: Callable
    phase: Optional[int] = None

    def __hash__(self):
        return hash(self.name)


@dataclass
class ModelLayout:
    """Output of LayerCompiler.compile().

    Attributes:
        d_model: total residual-stream dimension, derived from peak live-dim usage
        n_layers: total number of transformer blocks needed
        ops_per_layer: ops_per_layer[i] is the list of ops the compiler placed at layer i
        dim_positions: map of dim_name -> start position in residual stream
        dim_sizes: map of dim_name -> size (so position range is [pos, pos+size))
    """

    d_model: int
    n_layers: int
    ops_per_layer: List[List[Operation]]
    dim_positions: Dict[str, int]
    dim_sizes: Dict[str, int]

    def ops_at(self, layer: int) -> List[Operation]:
        return self.ops_per_layer[layer]

    def dim_range(self, name: str) -> range:
        start = self.dim_positions[name]
        return range(start, start + self.dim_sizes[name])


class LayerCompiler:
    """Layer-allocation compiler MVP.

    Algorithm:
      1. Build dependency graph: edge u -> v if v reads any dim that u writes.
      2. Topological sort.
      3. For each op in topo order, assign to earliest layer where all deps satisfied.
         Multiple ops can share a layer if they don't depend on each other AND
         they target different module kinds (attn vs ffn) — a single transformer
         block has one attention and one FFN per layer.
      4. Allocate dims: simple bump-pointer (no liveness reuse for MVP).
    """

    def __init__(self):
        self.ops: List[Operation] = []
        self.dims: Dict[str, int] = {}  # name -> size
        self._op_by_name: Dict[str, Operation] = {}

    def declare_dim(self, name: str, size: int):
        if name in self.dims and self.dims[name] != size:
            raise ValueError(
                f"Dim {name!r} already declared with size {self.dims[name]}; "
                f"got {size}"
            )
        self.dims[name] = size

    def add_op(self, op: Operation):
        if op.kind not in ("attn", "ffn"):
            raise ValueError(f"op.kind must be 'attn' or 'ffn'; got {op.kind!r}")
        if op.name in self._op_by_name:
            raise ValueError(f"Operation name {op.name!r} already added")
        # Validate that every read/write is declared.
        for d in op.reads | op.writes:
            if d not in self.dims:
                raise ValueError(
                    f"Op {op.name!r} references undeclared dim {d!r}"
                )
        self.ops.append(op)
        self._op_by_name[op.name] = op

    # --------------------------------------------------------------------
    # Compilation
    # --------------------------------------------------------------------

    def compile(self) -> ModelLayout:
        """Produce a ModelLayout from the declared ops and dims."""
        topo = self._topological_sort()
        layer_assignment = self._assign_layers(topo)
        dim_positions = self._allocate_dims()

        n_layers = (max(layer_assignment.values()) + 1) if layer_assignment else 0
        d_model = sum(self.dims.values())

        ops_per_layer: List[List[Operation]] = [[] for _ in range(n_layers)]
        for op in self.ops:
            ops_per_layer[layer_assignment[op.name]].append(op)

        return ModelLayout(
            d_model=d_model,
            n_layers=n_layers,
            ops_per_layer=ops_per_layer,
            dim_positions=dim_positions,
            dim_sizes=dict(self.dims),
        )

    # --------------------------------------------------------------------
    # Internals
    # --------------------------------------------------------------------

    def _topological_sort(self) -> List[Operation]:
        """Return ops sorted so each op comes after every op that writes a dim it reads.

        Cycles: writes-then-reads on the same dim across ops is fine (downstream op
        sees upstream's write). A *cycle* would be op A reads dim X written by B,
        and B reads dim Y written by A. We detect cycles and raise.
        """
        # writers[d] = list of ops that write dim d
        writers: Dict[str, List[Operation]] = {d: [] for d in self.dims}
        for op in self.ops:
            for d in op.writes:
                writers[d].append(op)

        # Build edges: u -> v iff v.reads ∩ u.writes
        # Drop edges where both ops have phase set AND u.phase > v.phase, i.e.,
        # the writer comes later in the hand-set order than the reader. These
        # are spurious deps caused by the dim-name-only dep model (multiple ops
        # writing the "same" dim at different positions).
        in_edges: Dict[str, Set[str]] = {op.name: set() for op in self.ops}
        out_edges: Dict[str, Set[str]] = {op.name: set() for op in self.ops}
        for v in self.ops:
            for d in v.reads:
                for u in writers.get(d, ()):
                    if u.name == v.name:
                        continue
                    # Phase-based pruning of spurious deps
                    if (u.phase is not None and v.phase is not None
                            and u.phase > v.phase):
                        continue
                    in_edges[v.name].add(u.name)
                    out_edges[u.name].add(v.name)

        # Kahn's algorithm
        ready = [op for op in self.ops if not in_edges[op.name]]
        # Stable order: by insertion order, so determinism
        ready.sort(key=lambda o: self.ops.index(o))
        result: List[Operation] = []
        while ready:
            op = ready.pop(0)
            result.append(op)
            for v_name in sorted(out_edges[op.name]):
                in_edges[v_name].discard(op.name)
                if not in_edges[v_name]:
                    ready.append(self._op_by_name[v_name])
            ready.sort(key=lambda o: self.ops.index(o))

        if len(result) != len(self.ops):
            stuck = [name for name, ins in in_edges.items() if ins]
            raise ValueError(f"Dependency cycle detected; stuck ops: {stuck}")
        return result

    def _assign_layers(self, topo: List[Operation]) -> Dict[str, int]:
        """Earliest-layer-first assignment respecting deps and one-attn-one-ffn-per-layer."""
        # writes[d] = layer where d is written (highest among multiple writers)
        writes_layer: Dict[str, int] = {}
        # layer_kinds[layer] = set of kinds already used at that layer
        layer_kinds: Dict[int, Set[str]] = {}
        assignment: Dict[str, int] = {}

        for op in topo:
            # earliest layer >= max(writes_layer of every read) + 1
            earliest = 0
            for d in op.reads:
                if d in writes_layer:
                    earliest = max(earliest, writes_layer[d] + 1)
            # advance until kind slot is free
            layer = earliest
            while True:
                kinds_used = layer_kinds.setdefault(layer, set())
                if op.kind not in kinds_used:
                    kinds_used.add(op.kind)
                    break
                layer += 1
            assignment[op.name] = layer
            for d in op.writes:
                writes_layer[d] = max(writes_layer.get(d, -1), layer)
        return assignment

    def _allocate_dims(self) -> Dict[str, int]:
        """Simple bump-pointer dim allocation.

        Order: by declaration order. Future work: liveness-based reuse via the
        existing AutoAllocator in unified_compiler/auto_allocator.py.
        """
        positions: Dict[str, int] = {}
        cursor = 0
        for name, size in self.dims.items():
            positions[name] = cursor
            cursor += size
        return positions


def build_model_from_layout(layout: ModelLayout, S: float = 100.0):
    """Construct an AutoregressiveVM from a compiled layout and bake all ops.

    This is the bridge from "layout produced by compiler" to "working model".

    For each layer:
      - The model's block at that layer index has its attn programmed by all
        ops with kind="attn" assigned to that layer.
      - The block's ffn is programmed by all ops with kind="ffn".

    Each op's bake_fn is called with (target_module, dim_positions, S) where
    target_module is the attn or ffn at the assigned block.

    Note: this assumes a single op per (layer, kind). The layer compiler
    enforces this via _assign_layers; multiple ops of the same kind would
    need to be combined upstream into a single op, or split across layers.

    Args:
        layout: ModelLayout from LayerCompiler.compile()
        S: SwiGLU activation scale (passed to each bake_fn)

    Returns:
        AutoregressiveVM instance with weights baked.
    """
    # Lazy import to avoid circular dep when unified_compiler is loaded standalone
    from ..vm_step import AutoregressiveVM

    if layout.n_layers == 0:
        # Trivial empty layout — return a zero-layer model
        # (AutoregressiveVM doesn't support 0 layers; we use 1 with no ops baked)
        n_layers = 1
    else:
        n_layers = layout.n_layers

    # Auto d_model is derived from layout, not hardcoded.
    model = AutoregressiveVM(d_model=layout.d_model, n_layers=n_layers)

    # Wrap the entire bake pass in no_grad so bake_fn implementations can do
    # in-place writes to leaf Parameters (the pattern used throughout
    # vm_step.py's hand-set weights).
    import torch as _torch
    with _torch.no_grad():
        for layer_idx, ops_at_layer in enumerate(layout.ops_per_layer):
            block = model.blocks[layer_idx]
            for op in ops_at_layer:
                if op.kind == "attn":
                    target = block.attn
                elif op.kind == "ffn":
                    target = block.ffn
                else:
                    raise ValueError(f"Unknown op kind {op.kind!r}")
                op.bake_fn(target, layout.dim_positions, S)

    return model
