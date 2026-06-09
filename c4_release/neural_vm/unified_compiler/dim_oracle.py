"""Reference oracle for residual dim values across an autoregressive C4 run.

Pairs with :mod:`symbolic_forward` (the *actual* per-block residual trace
from running the declarative IR through the DSL interpreter): given a
C4 bytecode program, this module computes the *expected* residual value
at every ``(step, position, block, dim)`` tuple by walking a pure-Python
reference C4 VM and projecting its register state into residual-dim
space.

Why a separate oracle?
----------------------

Single-rule whack-a-mole fixes have a 0/5 historical track record (see
the user-memory note ``feedback_single_rule_fixes_are_zero_sum.md``).
The recurring blocker has been that we know what ``symbolic_forward``
produces, but we have no machine-checkable statement of what the
residual *should* be. The oracle closes that gap: it's a deterministic
"this dim must equal indicator(register_byte_h_lo == k)" statement that
the block-by-block diff (see :mod:`dim_diff`) can compare against.

Public surface
--------------

* :class:`RegisterState` — snapshot of one VM step (AX, BP, SP, PC, MEM,
  STACK).
* :class:`ReferenceOracle(program_bytecode)` — wraps the reference VM
  and exposes ``state_at_step(n) -> RegisterState``.
* :func:`project_state_to_residual` — maps a register state to the
  expected ``(position, dim_name) -> float`` table for the *simple* dim
  families (CLEAN_EMBED_LO/HI, MARK_AX, MARK_STACK0, BYTE_INDEX_h,
  STACK0_BYTE_VAL_h_LO/HI).
* :func:`expected_trace` — every ``(step, position, block,
  expected_value)`` tuple for a single dim across the whole program.

Scope: this module covers the *embedding-time* and *broadcast-time*
dims — those whose expected value at a given position is a pure function
of the VM state for that step. The cross-step OUTPUT_LO/HI family is
listed as ``DEFERRED`` (see :data:`DEFERRED_DIM_FAMILIES`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .symbolic_forward import (
    OP_ADD,
    OP_ADJ,
    OP_BNZ,
    OP_BZ,
    OP_ENT,
    OP_EXIT,
    OP_IMM,
    OP_JMP,
    OP_JSR,
    OP_LEA,
    OP_LEV,
    OP_LI,
    OP_MUL,
    OP_PSH,
    OP_SI,
    OP_SUB,
    decode_instr,
)


# Token layout — every step emits 35 tokens. See neural_vm/token_layout.py.
TOKENS_PER_STEP = 35

# Position of each marker / byte slot within a single step's token window.
POS_PC_MARKER = 0
POS_PC_BYTE0 = 1   # byte 0 (LSB) ... byte 3 at POS_PC_BYTE0 + 3

POS_AX_MARKER = 5
POS_AX_BYTE0 = 6

POS_SP_MARKER = 10
POS_SP_BYTE0 = 11

POS_BP_MARKER = 15
POS_BP_BYTE0 = 16

POS_STACK0_MARKER = 20
POS_STACK0_BYTE0 = 21

POS_MEM_MARKER = 25
POS_MEM_ADDR_BYTE0 = 26
POS_MEM_VAL_BYTE0 = 30

POS_STEP_END = 34


# Map register name -> (marker_pos, byte0_pos). Used for projection.
_REGISTER_SLOTS: Dict[str, Tuple[int, int]] = {
    "PC":     (POS_PC_MARKER, POS_PC_BYTE0),
    "AX":     (POS_AX_MARKER, POS_AX_BYTE0),
    "SP":     (POS_SP_MARKER, POS_SP_BYTE0),
    "BP":     (POS_BP_MARKER, POS_BP_BYTE0),
    "STACK0": (POS_STACK0_MARKER, POS_STACK0_BYTE0),
}


# Dim families the oracle currently models. The right-hand-side names a
# projection rule; the actual closure lives in :func:`project_state_to_residual`.
SUPPORTED_DIM_FAMILIES: Tuple[str, ...] = (
    "CLEAN_EMBED_LO",
    "CLEAN_EMBED_HI",
    "EMBED_LO",
    "EMBED_HI",
    "MARK_PC",
    "MARK_AX",
    "MARK_SP",
    "MARK_BP",
    "MARK_STACK0",
    "MARK_MEM",
    "MARK_SE",
    "BYTE_INDEX_0",
    "BYTE_INDEX_1",
    "BYTE_INDEX_2",
    "BYTE_INDEX_3",
    "IS_BYTE",
    "IS_MARK",
    "CONST",
    "STACK0_BYTE_VAL_1_LO",
    "STACK0_BYTE_VAL_1_HI",
    "STACK0_BYTE_VAL_2_LO",
    "STACK0_BYTE_VAL_2_HI",
    "STACK0_BYTE_VAL_3_LO",
    "STACK0_BYTE_VAL_3_HI",
)


# Dim families whose expected value crosses step boundaries or depends
# on multi-pass autoregressive teacher-forcing. Diff'ing these against
# the oracle would be misleading until the projection is extended.
DEFERRED_DIM_FAMILIES: Tuple[str, ...] = (
    "OUTPUT_LO",       # autoregressive next-step byte emission
    "OUTPUT_HI",
    "AX_CARRY_LO",     # mid-block ALU carry propagation
    "AX_CARRY_HI",
    "ADDR_KEY",        # memory-lookup address gather across steps
    "MEM_ADDR_SRC",
)


__all__ = [
    "RegisterState",
    "ReferenceOracle",
    "TOKENS_PER_STEP",
    "SUPPORTED_DIM_FAMILIES",
    "DEFERRED_DIM_FAMILIES",
    "project_state_to_residual",
    "expected_trace",
    "is_supported_dim",
]


# ---------------------------------------------------------------------------
# State container
# ---------------------------------------------------------------------------


@dataclass
class RegisterState:
    """Snapshot of the reference C4 VM after one instruction.

    Attributes
    ----------
    step_idx
        The 0-indexed step this state describes. ``step_idx=0`` is the
        state AFTER the first instruction has executed.
    pc, ax, sp, bp
        32-bit register values (low 32 bits used; values are masked into
        ``0..0xFFFFFFFF`` by the reference VM).
    stack0
        Top-of-stack value at the time the step token bundle was emitted.
        For PSH this is the just-pushed AX; for POPs it's the value left
        on top after the pop.
    memory
        Address-keyed dict of byte values in the simulated memory. Only
        addresses that have been written are present.
    halted
        ``True`` once the program executed an EXIT or returned from main
        via LEV. Subsequent ``state_at_step`` calls return the final
        snapshot.
    """

    step_idx: int
    pc: int = 0
    ax: int = 0
    sp: int = 0
    bp: int = 0
    stack0: int = 0
    memory: Dict[int, int] = field(default_factory=dict)
    halted: bool = False


# ---------------------------------------------------------------------------
# Reference VM
# ---------------------------------------------------------------------------


class ReferenceOracle:
    """Pure-Python reference C4 VM, snapshotted per step.

    The VM mirrors :mod:`neural_vm.nibble_bytecode_executor` semantics for
    the integer-register subset (IMM, PSH, POP via LEV/ADJ, ALU, control
    flow). Memory writes use byte-addressable storage so the resulting
    ``memory`` dict matches the per-byte projection used by the
    embedding bake (one entry per address byte).

    Parameters
    ----------
    program_bytecode
        Sequence of 32-bit instruction words (``encode_instr(op, imm)``).
    code_base
        Base address of the code segment. PC values are stored relative
        to this; the default ``0`` makes step-N's PC equal to ``N * 8``
        before fetch.
    initial_sp
        Stack pointer at program start. Default ``0x100000`` matches the
        nibble executor and the integration tests.
    """

    INSTR_WIDTH = 8  # bytes per encoded instruction

    def __init__(
        self,
        program_bytecode: Sequence[int],
        *,
        code_base: int = 0,
        initial_sp: int = 0x100000,
        max_cycles: int = 10000,
    ):
        self.program = list(program_bytecode)
        self.code_base = code_base
        self.initial_sp = initial_sp
        self.max_cycles = max_cycles
        self._snapshots: List[RegisterState] = []
        self._run()

    # ----- public API -----------------------------------------------------

    def state_at_step(self, step_idx: int) -> RegisterState:
        """Return the register state AFTER step ``step_idx`` executed.

        Negative indices and indices past the program length are clamped
        to the closest valid step. The reference VM always produces at
        least one snapshot for non-empty programs.
        """

        if not self._snapshots:
            return RegisterState(step_idx=0)
        if step_idx < 0:
            return self._snapshots[0]
        if step_idx >= len(self._snapshots):
            return self._snapshots[-1]
        return self._snapshots[step_idx]

    @property
    def num_steps(self) -> int:
        return len(self._snapshots)

    def all_states(self) -> List[RegisterState]:
        return list(self._snapshots)

    # ----- execution loop -------------------------------------------------

    def _run(self) -> None:
        pc = self.code_base
        ax = 0
        sp = self.initial_sp
        bp = sp
        stack: Dict[int, int] = {}  # word-level (8-byte) stack slots
        memory_bytes: Dict[int, int] = {}  # byte-level data memory

        stack0 = 0
        halted = False
        cycle = 0

        while cycle < self.max_cycles:
            cycle += 1
            instr_offset = (pc - self.code_base) // self.INSTR_WIDTH
            if instr_offset < 0 or instr_offset >= len(self.program):
                halted = True
                break

            opcode, imm = decode_instr(self.program[instr_offset])
            pc_next = pc + self.INSTR_WIDTH

            if opcode == OP_IMM:
                ax = imm & 0xFFFFFFFF
            elif opcode == OP_LEA:
                ax = (bp + imm) & 0xFFFFFFFF
            elif opcode == OP_JMP:
                pc_next = imm
            elif opcode == OP_JSR:
                sp -= 8
                stack[sp] = pc_next
                pc_next = imm
            elif opcode == OP_BZ:
                if ax == 0:
                    pc_next = imm
            elif opcode == OP_BNZ:
                if ax != 0:
                    pc_next = imm
            elif opcode == OP_ENT:
                sp -= 8
                stack[sp] = bp
                bp = sp
                sp = (sp - imm) & 0xFFFFFFFF
            elif opcode == OP_ADJ:
                sp = (sp + imm) & 0xFFFFFFFF
            elif opcode == OP_LEV:
                sp = bp
                if sp in stack:
                    bp = stack[sp] & 0xFFFFFFFF
                    sp += 8
                if sp in stack:
                    pc_next = stack[sp] & 0xFFFFFFFF
                    sp += 8
                else:
                    halted = True
            elif opcode == OP_LI:
                addr = ax
                val = 0
                if addr in stack:
                    val = stack[addr]
                else:
                    for i in range(8):
                        val |= memory_bytes.get(addr + i, 0) << (i * 8)
                ax = val & 0xFFFFFFFF
            elif opcode == OP_SI:
                if sp in stack:
                    addr = stack[sp]
                    sp += 8
                    for i in range(8):
                        memory_bytes[addr + i] = (ax >> (i * 8)) & 0xFF
            elif opcode == OP_PSH:
                sp -= 8
                stack[sp] = ax
            elif opcode == OP_ADD:
                if sp in stack:
                    top = stack[sp]; sp += 8
                    ax = (top + ax) & 0xFFFFFFFF
            elif opcode == OP_SUB:
                if sp in stack:
                    top = stack[sp]; sp += 8
                    ax = (top - ax) & 0xFFFFFFFF
            elif opcode == OP_MUL:
                if sp in stack:
                    top = stack[sp]; sp += 8
                    ax = (top * ax) & 0xFFFFFFFF
            elif opcode == OP_EXIT:
                halted = True
            else:
                # Unknown opcode — treat as a no-op rather than crash;
                # the oracle is best-effort for the C4 ISA subset.
                pass

            # STACK0 = the current top-of-stack value AFTER the step.
            stack0 = stack.get(sp, 0)

            self._snapshots.append(
                RegisterState(
                    step_idx=len(self._snapshots),
                    pc=pc & 0xFFFFFFFF,
                    ax=ax & 0xFFFFFFFF,
                    sp=sp & 0xFFFFFFFF,
                    bp=bp & 0xFFFFFFFF,
                    stack0=stack0 & 0xFFFFFFFF,
                    memory=dict(memory_bytes),
                    halted=halted,
                )
            )

            pc = pc_next
            if halted:
                break


# ---------------------------------------------------------------------------
# State -> residual projection
# ---------------------------------------------------------------------------


def _byte_lo(value: int, byte_idx: int) -> int:
    return ((value >> (byte_idx * 8)) & 0xFF) & 0x0F


def _byte_hi(value: int, byte_idx: int) -> int:
    return ((value >> (byte_idx * 8)) & 0xFF) >> 4


def is_supported_dim(dim_name: str) -> bool:
    """Return True if the oracle has a projection rule for ``dim_name``."""
    base = dim_name.split("+", 1)[0]
    return base in SUPPORTED_DIM_FAMILIES


def project_state_to_residual(
    state: RegisterState,
    *,
    step_position_base: Optional[int] = None,
    per_token: bool = False,
) -> Dict[Tuple[int, str], float]:
    """Project a :class:`RegisterState` to expected ``(position, dim_key)``
    -> value entries for the *current* step.

    Two modes
    ---------

    Default (``per_token=False``) — emit one entry per dim at the step's
    canonical position (``step_idx`` itself, matching the
    ``SymbolicForwardRunner`` snapshot position). The residual state in
    ``symbolic_forward`` is one bag-of-dims per (step, block), so this is
    the form the diff consumes.

    Per-token (``per_token=True``) — emit one entry per (token-position,
    dim) within the step's 35-token window. Useful for callers that want
    to inspect "what should MARK_AX be at the AX-marker token row?"; the
    base position is ``state.step_idx * TOKENS_PER_STEP`` (or
    ``step_position_base`` if supplied).

    Returns
    -------
    Dict[(position, dim_key), float]
        ``dim_key`` is the canonical ``"NAME+offset"`` form used by the
        :class:`~unified_compiler.dsl_interpreter.DSLInterpreter` state
        dict.
    """

    if per_token:
        base = (
            step_position_base
            if step_position_base is not None
            else state.step_idx * TOKENS_PER_STEP
        )
        return _project_per_token(state, base)
    # Bag-of-dims form: every supported dim collapsed onto position=step_idx.
    return _project_bag_of_dims(state)


def _emit_register_nibbles(out, base_pos: int, byte0_pos: int, value: int):
    for h in range(4):
        byte_val = (value >> (h * 8)) & 0xFF
        lo = byte_val & 0x0F
        hi = (byte_val >> 4) & 0x0F
        pos = base_pos + byte0_pos + h
        out[(pos, f"CLEAN_EMBED_LO+{lo}")] = 1.0
        out[(pos, f"CLEAN_EMBED_HI+{hi}")] = 1.0
        out[(pos, f"EMBED_LO+{lo}")] = 1.0
        out[(pos, f"EMBED_HI+{hi}")] = 1.0


def _project_per_token(state, base: int) -> Dict[Tuple[int, str], float]:
    out: Dict[Tuple[int, str], float] = {}

    # CONST is set at every token by the embedding.
    for tok in range(TOKENS_PER_STEP):
        out[(base + tok, "CONST+0")] = 1.0

    marker_positions = {
        POS_PC_MARKER, POS_AX_MARKER, POS_SP_MARKER, POS_BP_MARKER,
        POS_STACK0_MARKER, POS_MEM_MARKER, POS_STEP_END,
    }
    for tok in range(TOKENS_PER_STEP):
        if tok in marker_positions:
            out[(base + tok, "IS_MARK+0")] = 1.0
        else:
            out[(base + tok, "IS_BYTE+0")] = 1.0

    out[(base + POS_PC_MARKER,     "MARK_PC+0")] = 1.0
    out[(base + POS_AX_MARKER,     "MARK_AX+0")] = 1.0
    out[(base + POS_SP_MARKER,     "MARK_SP+0")] = 1.0
    out[(base + POS_BP_MARKER,     "MARK_BP+0")] = 1.0
    out[(base + POS_STACK0_MARKER, "MARK_STACK0+0")] = 1.0
    out[(base + POS_MEM_MARKER,    "MARK_MEM+0")] = 1.0
    out[(base + POS_STEP_END,      "MARK_SE+0")] = 1.0

    for _reg_name, (_marker_pos, byte0_pos) in _REGISTER_SLOTS.items():
        for h in range(4):
            out[(base + byte0_pos + h, f"BYTE_INDEX_{h}+0")] = 1.0
    for h in range(4):
        out[(base + POS_MEM_ADDR_BYTE0 + h, f"BYTE_INDEX_{h}+0")] = 1.0
        out[(base + POS_MEM_VAL_BYTE0 + h, f"BYTE_INDEX_{h}+0")] = 1.0

    reg_values = {
        "PC":     state.pc,
        "AX":     state.ax,
        "SP":     state.sp,
        "BP":     state.bp,
        "STACK0": state.stack0,
    }
    for reg_name, value in reg_values.items():
        _, byte0_pos = _REGISTER_SLOTS[reg_name]
        _emit_register_nibbles(out, base, byte0_pos, value)

    for h in range(1, 4):
        byte_val = (state.stack0 >> (h * 8)) & 0xFF
        lo = byte_val & 0x0F
        hi = (byte_val >> 4) & 0x0F
        pos = base + POS_STACK0_BYTE0 + h
        out[(pos, f"STACK0_BYTE_VAL_{h}_LO+{lo}")] = 1.0
        out[(pos, f"STACK0_BYTE_VAL_{h}_HI+{hi}")] = 1.0

    return out


def _project_bag_of_dims(state) -> Dict[Tuple[int, str], float]:
    """Single-position bag: collapse the step's expected dims onto the
    ``symbolic_forward`` snapshot's per-step canonical position
    (``step_idx``).

    Includes the indicator dims (any marker / byte_index that fires
    *anywhere* in the step's window) and the one-hot nibble cells for
    PC / AX / SP / BP / STACK0 byte values.
    """

    pos = state.step_idx
    out: Dict[Tuple[int, str], float] = {}

    # CONST + marker / byte_index indicators — set in the step's bag
    # because at least one token in the window carries the flag. The
    # bag-of-dims form expects 1.0 (set) regardless of which token.
    out[(pos, "CONST+0")] = 1.0
    out[(pos, "IS_BYTE+0")] = 1.0
    out[(pos, "IS_MARK+0")] = 1.0
    for name in ("MARK_PC", "MARK_AX", "MARK_SP", "MARK_BP",
                 "MARK_STACK0", "MARK_MEM", "MARK_SE"):
        out[(pos, f"{name}+0")] = 1.0
    for h in range(4):
        out[(pos, f"BYTE_INDEX_{h}+0")] = 1.0

    # Nibble cells — one-hot per register byte. Multiple registers can
    # set the same cell if their bytes share a nibble value; that's
    # consistent with the residual stream's "writes accumulate" rule.
    def _set_one_hot(family: str, cell: int):
        out[(pos, f"{family}+{cell}")] = 1.0

    for value in (state.pc, state.ax, state.sp, state.bp, state.stack0):
        for h in range(4):
            byte_val = (value >> (h * 8)) & 0xFF
            _set_one_hot("CLEAN_EMBED_LO", byte_val & 0x0F)
            _set_one_hot("CLEAN_EMBED_HI", (byte_val >> 4) & 0x0F)
            _set_one_hot("EMBED_LO", byte_val & 0x0F)
            _set_one_hot("EMBED_HI", (byte_val >> 4) & 0x0F)

    # STACK0_BYTE_VAL_h_LO / HI — one-hot of the STACK0 byte h nibbles.
    for h in range(1, 4):
        byte_val = (state.stack0 >> (h * 8)) & 0xFF
        out[(pos, f"STACK0_BYTE_VAL_{h}_LO+{byte_val & 0x0F}")] = 1.0
        out[(pos, f"STACK0_BYTE_VAL_{h}_HI+{(byte_val >> 4) & 0x0F}")] = 1.0

    return out


# ---------------------------------------------------------------------------
# expected_trace
# ---------------------------------------------------------------------------


def expected_trace(
    oracle: ReferenceOracle,
    dim_name: str,
    *,
    n_blocks: int,
) -> Dict[Tuple[int, int, int, str], float]:
    """Build the ``(step, position, block, dim_key) -> expected_value`` table
    for a single dim family across every step of ``oracle``.

    The expected value is *block-invariant*: the oracle treats the dim
    as a target that any qualifying block ought to maintain. ``n_blocks``
    is the number of blocks in the layout (so the per-block diff has
    something to compare against at every snapshot).

    ``dim_name`` is the family name (e.g. ``"STACK0_BYTE_VAL_1_LO"``);
    the returned keys carry the per-offset ``"NAME+k"`` form.
    """

    if not is_supported_dim(dim_name):
        raise ValueError(
            f"expected_trace: dim {dim_name!r} is not in "
            f"SUPPORTED_DIM_FAMILIES (see DEFERRED_DIM_FAMILIES for the "
            f"families the oracle does not yet model)"
        )

    out: Dict[Tuple[int, int, int, str], float] = {}
    for state in oracle.all_states():
        per_pos = project_state_to_residual(state)
        for (pos, dim_key), value in per_pos.items():
            base = dim_key.split("+", 1)[0]
            if base != dim_name:
                continue
            for block_idx in range(n_blocks):
                out[(state.step_idx, pos, block_idx, dim_key)] = value
    return out
