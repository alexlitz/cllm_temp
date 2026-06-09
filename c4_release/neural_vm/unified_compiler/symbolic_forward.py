"""Symbolic forward executor — multi-step program runner over the DSL IR.

The existing :class:`DSLInterpreter` (``dsl_interpreter.py``) drives a
single residual-state dict through a flat list of Operations and is the
substrate for byte-identity FFN/attention unit tests. This module wraps
it into a *program runner*: given a fixed input program (a list of
encoded C4 instructions) and a compile-time layout, advance step-by-step
and record every residual dim's symbolic value at every block boundary.

Why this exists
---------------

The deployed transformer is autoregressive: each token step runs all
~18 blocks (attn + ffn) over the residual stream. The compile pipeline
bakes the weights, but for debugging we want to *interpret* the same
declarative IR forward without spending the ~1-2 minute cold-compile
cost. The output is a per-(step, block, dim) symbolic trace that can be
diffed against a teacher-forced expected residual to localise drift.

Public surface
--------------

* :class:`SymbolicForwardRunner` — main entry point.

  * ``.step()``: advance by 1 token. Runs every block's attn + ffn ops
    in scheduled order, recording the post-block residual state.
  * ``.get_residual(block_idx, position, dim_name)``: read one
    symbolic value at a recorded snapshot.
  * ``.get_dim_trace(dim_name)``: every recorded value of ``dim_name``
    across all (step, position, block) triples.
  * ``.diff_against_expected(expected_trace)``: find the first triple
    where this runner disagrees with a reference trace.

* Minimal embedding mapper (:func:`default_embedding_for_instruction`)
  for the demo case: maps an encoded ``(opcode, imm)`` to the indicator
  dims (``OP_IMM`` / ``OP_PSH`` / ``OP_EXIT`` / ``IMM_AX_*``) that the
  embedding bake writes. Callers with a real ``TokenEmbeddingRule``
  table can override via the ``embedding_fn`` kwarg.

This is *not* a replacement for the compile pipeline. Numerics agree
with the lowered FFN at byte identity for binary one-hot inputs at
``S >> 0`` (the existing
``compare_symbolic_to_lowered_ffn`` gate); soft transitions near
threshold can diverge. The runner is for compile-time diagnosis, not
deployment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .dsl_interpreter import DSLInterpreter, InterpreterStep


# ---------------------------------------------------------------------------
# Opcode constants — local copy of the canonical C4 ISA encoding.
# Mirrors ``tests/test_unified_memory.py`` and the embedding bake.
# ---------------------------------------------------------------------------

OP_LEA = 0
OP_IMM = 1
OP_JMP = 2
OP_JSR = 3
OP_BZ = 4
OP_BNZ = 5
OP_ENT = 6
OP_ADJ = 7
OP_LEV = 8
OP_LI = 9
OP_SI = 11
OP_PSH = 13
OP_ADD = 25
OP_SUB = 26
OP_MUL = 27
OP_EXIT = 38


# Inverse table — opcode int -> short name. Used to set the matching
# ``OP_<NAME>`` indicator dim in the default embedding mapper.
_OPCODE_NAMES: Dict[int, str] = {
    OP_LEA: "LEA", OP_IMM: "IMM", OP_JMP: "JMP", OP_JSR: "JSR",
    OP_BZ: "BZ", OP_BNZ: "BNZ", OP_ENT: "ENT", OP_ADJ: "ADJ",
    OP_LEV: "LEV", OP_LI: "LI", OP_SI: "SI", OP_PSH: "PSH",
    OP_ADD: "ADD", OP_SUB: "SUB", OP_MUL: "MUL", OP_EXIT: "EXIT",
}


def encode_instr(op: int, imm: int = 0) -> int:
    """Pack ``(op, imm)`` into the 32-bit instruction word used by the VM.

    Mirrors ``tests/test_unified_memory.py::pack_instr`` so demo bytecode
    constructed here byte-identically matches the test harness.
    """
    return (op & 0xFF) | ((imm & 0xFFFFFF) << 8)


def decode_instr(word: int) -> Tuple[int, int]:
    """Inverse of :func:`encode_instr`."""
    return (word & 0xFF, (word >> 8) & 0xFFFFFF)


# ---------------------------------------------------------------------------
# Default embedding mapper
# ---------------------------------------------------------------------------


def default_embedding_for_instruction(
    op: int,
    imm: int,
    *,
    pc: int = 0,
    activation_value: float = 1.0,
) -> Dict[str, float]:
    """Map a single ``(op, imm)`` instruction to a one-hot residual state.

    Sets:

    * ``OP_<NAME>+0 = activation_value`` for the matching opcode (e.g.
      ``OP_IMM`` / ``OP_PSH`` / ``OP_EXIT``).
    * ``CONST+0 = 1.0`` — the ubiquitous bias dim every layer relies on.
    * ``IMM_AX_LO+<n> = 1.0`` for the low-nibble of each AX byte
      (``n`` in 0..15) and ``IMM_AX_HI+<n>`` for the high nibble. This is
      the same one-hot decomposition the L0/L1 embedding bakes use for
      the IMM payload.

    Caller can replace this entirely by passing
    ``embedding_fn=lambda op, imm, pc: { ... }`` to the runner — the
    default is the minimum sufficient state to drive the IMM/PSH/EXIT
    demo path.
    """
    state: Dict[str, float] = {"CONST+0": 1.0}
    name = _OPCODE_NAMES.get(op)
    if name is not None:
        state[f"OP_{name}+0"] = activation_value
    # Decompose imm into 3 bytes (24-bit immediate); each byte into
    # low/high nibbles. The neural VM keeps separate LO/HI dims per byte.
    for byte_idx in range(3):
        byte_val = (imm >> (byte_idx * 8)) & 0xFF
        lo = byte_val & 0x0F
        hi = (byte_val >> 4) & 0x0F
        # Byte 0 lives on the IMM_AX_*_0 family; byte 1 on _1; byte 2 on _2.
        suffix = "" if byte_idx == 0 else f"_{byte_idx}"
        state[f"IMM_AX_LO{suffix}+{lo}"] = 1.0
        state[f"IMM_AX_HI{suffix}+{hi}"] = 1.0
    # CLEAN_EMBED_LO/HI — the L0/L1 cleanup bake's output. The bake is a
    # passthrough from the raw IMM nibbles to CLEAN_EMBED_*, so the
    # symbolic forward seed matches it directly. Without this, downstream
    # attention V-side reads of CLEAN_EMBED (e.g. the L10 PSH AX
    # broadcast head) see zero V activation and never propagate — the
    # same class of bug ``apply_attention_specs`` had before the
    # int↔string key bridge fix, exposed at the L0/L1 layer instead.
    # The oracle's per-token projection sets CLEAN_EMBED_LO/HI from the
    # register byte values at every register slot; the bag-of-dims form
    # collapses every set nibble onto position=step_idx. Mirror that
    # here by setting CLEAN_EMBED_LO/HI from the IMM byte values so the
    # V→O propagation test path can drive end-to-end without a manual
    # ``initial_state`` injection.
    for byte_idx in range(3):
        byte_val = (imm >> (byte_idx * 8)) & 0xFF
        lo = byte_val & 0x0F
        hi = (byte_val >> 4) & 0x0F
        state[f"CLEAN_EMBED_LO+{lo}"] = 1.0
        state[f"CLEAN_EMBED_HI+{hi}"] = 1.0
        state[f"EMBED_LO+{lo}"] = 1.0
        state[f"EMBED_HI+{hi}"] = 1.0
    # PC position — record so callers can disambiguate steps. Kept as a
    # scalar dim rather than a one-hot since downstream consumers vary.
    state["PC+0"] = float(pc)
    return state


# ---------------------------------------------------------------------------
# Trace types
# ---------------------------------------------------------------------------


@dataclass
class BlockSnapshot:
    """Snapshot of the residual state immediately after one block fires.

    * ``step_idx`` — which token step (0-indexed; ``step()`` increments).
    * ``position`` — the sequence-position the token landed on (same as
      ``step_idx`` for a strict autoregressive trace).
    * ``block_idx`` — which transformer block (0 .. n_blocks-1).
    * ``state`` — a *copy* of the residual-state dict after the block's
      attn + ffn ops fired.
    * ``ops_fired`` — names of the ops the runner dispatched, in order.
    """

    step_idx: int
    position: int
    block_idx: int
    state: Dict[str, float] = field(default_factory=dict)
    ops_fired: List[str] = field(default_factory=list)


@dataclass
class TraceEntry:
    """One (step, position, block, value) tuple in a dim trace."""

    step_idx: int
    position: int
    block_idx: int
    value: float


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class SymbolicForwardRunner:
    """Multi-step symbolic forward executor over a compiled IR layout.

    Parameters
    ----------
    compiler
        A compiler-like object exposing:

        * ``ops_per_layer``: ``List[List[Operation]]`` — the per-block
          op schedule. Typically ``ModelLayout.ops_per_layer`` from
          :class:`LayerCompiler.compile`. The runner also accepts a
          ``LayerCompiler`` instance directly, in which case it expects
          ``compiler.layout`` or a ``compiler.compile()`` already done.
        * ``dim_positions`` (optional): map ``dim_name -> int``, passed
          to the underlying :class:`DSLInterpreter` for ops that expose
          a ``compiler_ir_factory`` instead of a pre-built ``compiler_ir``.
        * ``block_ops`` / ``model_ops`` (optional): block-level and
          model-level ops. Block-level are dispatched at their bound
          ``layer_idx`` after that layer's attn + ffn pass; model-level
          are dispatched once per step after all blocks.

    program_bytecode
        Sequence of 32-bit instruction words (each ``encode_instr(op,
        imm)``). The runner advances one instruction per ``step()``.

    embedding_fn
        Optional callable ``(op, imm, pc) -> Dict[str, float]`` mapping
        a decoded instruction to the residual state at embedding time.
        Defaults to :func:`default_embedding_for_instruction`.

    initial_state
        Optional extra dims to inject at the *start of every step*
        (before the per-token embedding map). Useful for sticking in
        loop-invariant state (e.g. MARK_PC_PIN+0 = 1.0).
    """

    def __init__(
        self,
        compiler: Any,
        program_bytecode: Sequence[int],
        *,
        embedding_fn: Optional[Callable[[int, int, int], Mapping[str, float]]] = None,
        initial_state: Optional[Mapping[str, float]] = None,
    ):
        self.compiler = compiler
        self.program_bytecode = list(program_bytecode)
        self.embedding_fn = embedding_fn or (
            lambda op, imm, pc: default_embedding_for_instruction(op, imm, pc=pc)
        )
        self._initial_state_seed: Dict[str, float] = (
            dict(initial_state) if initial_state else {}
        )

        # Pull the schedule. We accept either a layout-like (``ops_per_layer``
        # attribute) or a raw list-of-lists.
        self.ops_per_layer: List[List[Any]] = self._extract_schedule(compiler)
        self.n_blocks: int = len(self.ops_per_layer)
        self.dim_positions: Optional[Dict[str, int]] = self._extract_dim_positions(
            compiler,
        )
        self.block_ops: List[Any] = list(getattr(compiler, "block_ops", []) or [])
        self.model_ops: List[Any] = list(getattr(compiler, "model_ops", []) or [])

        # Step counter + recorded snapshots.
        self.step_idx: int = -1  # -1 = no steps taken yet
        self.snapshots: List[BlockSnapshot] = []
        # Current residual state — mutated in place by the underlying
        # DSLInterpreter as we walk the blocks for the current step.
        self._current_state: Dict[str, float] = {}

    # ----- schedule extraction --------------------------------------------

    def _extract_schedule(self, compiler: Any) -> List[List[Any]]:
        # Layout-style object
        ops_per_layer = getattr(compiler, "ops_per_layer", None)
        if ops_per_layer is not None:
            return [list(layer_ops) for layer_ops in ops_per_layer]
        # LayerCompiler with a cached layout
        layout = getattr(compiler, "layout", None)
        if layout is not None:
            return [list(layer_ops) for layer_ops in layout.ops_per_layer]
        raise TypeError(
            "SymbolicForwardRunner: compiler must expose ops_per_layer "
            "(or a .layout with one); got "
            f"{type(compiler).__name__} with neither"
        )

    def _extract_dim_positions(self, compiler: Any) -> Optional[Dict[str, int]]:
        dp = getattr(compiler, "dim_positions", None)
        if dp is not None:
            return dict(dp)
        layout = getattr(compiler, "layout", None)
        if layout is not None and getattr(layout, "dim_positions", None) is not None:
            return dict(layout.dim_positions)
        return None

    # ----- single-block dispatch ------------------------------------------

    def _resolve_block_layer(self, op: Any) -> Optional[int]:
        """Return the block index a kind="block" op binds to (or None)."""
        # Prefer ``target_op_name`` (op-reference binding) if present.
        target_name = getattr(op, "target_op_name", None)
        if target_name is not None:
            for block_idx, layer_ops in enumerate(self.ops_per_layer):
                for placed in layer_ops:
                    if getattr(placed, "name", None) == target_name:
                        return block_idx
        return getattr(op, "layer_idx", None)

    def _run_block(
        self, block_idx: int, interp: DSLInterpreter,
    ) -> Tuple[List[InterpreterStep], List[str]]:
        """Run one block: attn ops first, then ffn ops, then any block
        ops pinned to this layer index.

        Mirrors the layer-compiler convention that a transformer block
        has one attention pass followed by one FFN pass per layer.
        """
        steps: List[InterpreterStep] = []
        ops_fired: List[str] = []
        layer_ops = self.ops_per_layer[block_idx]
        # Split by kind so attn fires before ffn within the block.
        attn_ops = [op for op in layer_ops if getattr(op, "kind", "ffn") == "attn"]
        ffn_ops = [op for op in layer_ops if getattr(op, "kind", "ffn") == "ffn"]
        other_ops = [
            op for op in layer_ops
            if getattr(op, "kind", "ffn") not in ("attn", "ffn")
        ]
        for op in attn_ops + ffn_ops + other_ops:
            steps.append(interp.apply_operation(op))
            ops_fired.append(getattr(op, "name", "<anon>"))
        # Block-level ops bound to this layer.
        for op in self.block_ops:
            if self._resolve_block_layer(op) == block_idx:
                steps.append(interp.apply_operation(op))
                ops_fired.append(getattr(op, "name", "<anon>"))
        return steps, ops_fired

    # ----- public API: step ------------------------------------------------

    def step(self) -> List[BlockSnapshot]:
        """Advance by one token step.

        Returns the list of :class:`BlockSnapshot` recorded for this
        step (one per block, in block order). Also appended to
        ``self.snapshots``.
        """
        self.step_idx += 1
        if self.step_idx >= len(self.program_bytecode):
            raise IndexError(
                f"SymbolicForwardRunner.step: program has "
                f"{len(self.program_bytecode)} instructions; step "
                f"{self.step_idx} is past the end"
            )
        word = self.program_bytecode[self.step_idx]
        op, imm = decode_instr(word)
        # Embedding-time state: seed + embedding map.
        embed_state: Dict[str, float] = dict(self._initial_state_seed)
        embed_state.update(self.embedding_fn(op, imm, self.step_idx))
        # Drive through every block with the existing DSLInterpreter.
        interp = DSLInterpreter(
            initial_state=embed_state, dim_positions=self.dim_positions,
        )
        new_snapshots: List[BlockSnapshot] = []
        for block_idx in range(self.n_blocks):
            _steps, ops_fired = self._run_block(block_idx, interp)
            snap = BlockSnapshot(
                step_idx=self.step_idx,
                position=self.step_idx,
                block_idx=block_idx,
                state=dict(interp.state),
                ops_fired=ops_fired,
            )
            new_snapshots.append(snap)
            self.snapshots.append(snap)
        # Model-level ops (head, embedding writes, post-pass) — record on
        # the last block snapshot rather than creating a virtual block.
        for op in self.model_ops:
            interp.apply_operation(op)
        if self.model_ops and new_snapshots:
            new_snapshots[-1].state = dict(interp.state)
            self.snapshots[-1] = new_snapshots[-1]
        self._current_state = dict(interp.state)
        return new_snapshots

    def run_all(self) -> List[BlockSnapshot]:
        """Convenience: call ``step()`` until the program ends.

        Returns the cumulative ``self.snapshots`` list.
        """
        while self.step_idx + 1 < len(self.program_bytecode):
            self.step()
        return self.snapshots

    # ----- query API -------------------------------------------------------

    @staticmethod
    def _state_key(dim_name: str) -> str:
        """Normalise ``DIM`` to ``DIM+0`` to match DSLInterpreter keying."""
        if "+" in dim_name:
            return dim_name
        return f"{dim_name}+0"

    def get_residual(
        self, block_idx: int, position: int, dim_name: str,
    ) -> float:
        """Read one symbolic value at a recorded ``(block_idx, position)``.

        Returns 0.0 if the snapshot exists but the dim was never written,
        or raises ``LookupError`` if the snapshot wasn't recorded yet.
        """
        for snap in self.snapshots:
            if snap.block_idx == block_idx and snap.position == position:
                key = self._state_key(dim_name)
                if key in snap.state:
                    return snap.state[key]
                # Fall back to bare-name lookup for callers using a
                # state-key directly (e.g. "OUT+5").
                return snap.state.get(dim_name, 0.0)
        raise LookupError(
            f"SymbolicForwardRunner: no snapshot at block_idx={block_idx} "
            f"position={position} (step the runner first)"
        )

    def get_dim_trace(self, dim_name: str) -> List[TraceEntry]:
        """Every recorded value of ``dim_name`` across all snapshots."""
        key = self._state_key(dim_name)
        out: List[TraceEntry] = []
        for snap in self.snapshots:
            if key in snap.state:
                value = snap.state[key]
            elif dim_name in snap.state:
                value = snap.state[dim_name]
            else:
                continue
            out.append(
                TraceEntry(
                    step_idx=snap.step_idx,
                    position=snap.position,
                    block_idx=snap.block_idx,
                    value=value,
                )
            )
        return out

    # ----- diff API --------------------------------------------------------

    def diff_against_expected(
        self,
        expected_trace: Mapping[Tuple[int, int, int, str], float],
        *,
        atol: float = 1e-6,
    ) -> Optional[Tuple[Tuple[int, int, int, str], float, float]]:
        """Find the first ``(step, position, block, dim)`` where this
        runner disagrees with ``expected_trace``.

        ``expected_trace`` is a dict keyed by
        ``(step_idx, position, block_idx, dim_name)`` -> expected value.
        Returns ``None`` if every expected entry matches (within ``atol``)
        or a ``((key), actual, expected)`` triple at the first
        divergence. Keys missing from this runner's snapshots are
        treated as 0.0.
        """
        # Sort keys so "first divergence" is deterministic — by
        # (step, position, block, dim_name).
        for key in sorted(expected_trace.keys()):
            step_idx, position, block_idx, dim_name = key
            expected = expected_trace[key]
            try:
                actual = self.get_residual(block_idx, position, dim_name)
            except LookupError:
                actual = 0.0
            if abs(actual - expected) > atol:
                return (key, actual, expected)
        return None


__all__ = [
    "BlockSnapshot",
    "TraceEntry",
    "SymbolicForwardRunner",
    "default_embedding_for_instruction",
    "encode_instr",
    "decode_instr",
    "OP_LEA", "OP_IMM", "OP_JMP", "OP_JSR", "OP_BZ", "OP_BNZ",
    "OP_ENT", "OP_ADJ", "OP_LEV", "OP_LI", "OP_SI", "OP_PSH",
    "OP_ADD", "OP_SUB", "OP_MUL", "OP_EXIT",
]
