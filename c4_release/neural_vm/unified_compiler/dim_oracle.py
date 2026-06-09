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
  STACK0_BYTE_VAL_h_LO/HI) and the same-step projections of
  OUTPUT_LO/HI, AX_CARRY_LO/HI, ADDR_KEY, and MEM_ADDR_SRC.
* :func:`expected_trace` — every ``(step, position, block,
  expected_value)`` tuple for a single dim across the whole program.

Scope: this module covers the *embedding-time* and *broadcast-time*
dims — those whose expected value at a given position is a pure function
of the VM state for that step.

OUTPUT_LO/HI and AX_CARRY_LO/HI are projected as same-step residuals
(the value the op tree ought to have produced by the end of the step).
Cross-step PREV_STEP semantics (``.*.-1`` aliases) are still owned by
the actual op-tree; the oracle compares against the same-step
materialisation since that is the value any same-step writer should
emit. See :data:`DEFERRED_DIM_FAMILIES` for the (now empty) list of
families whose projection rule is still missing.
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
    # Same-step ALU / address-gather projections. The op tree writes
    # these dims in the AX-marker / MEM-marker / PC-marker rows during
    # the *current* step; the oracle pins the expected one-hot off the
    # reference VM's AX / MEM-value / PC registers. Cross-step PREV_STEP
    # reads (the ``.*.-1`` SSA aliases) are intentionally *not* projected
    # — those are routing back-edges through the KV cache, not residual
    # values produced this step.
    "OUTPUT_LO",
    "OUTPUT_HI",
    "AX_CARRY_LO",
    "AX_CARRY_HI",
    "ADDR_KEY",
    "MEM_ADDR_SRC",
)


# Dim families whose expected value crosses step boundaries or depends
# on multi-pass autoregressive teacher-forcing. Diff'ing these against
# the oracle would be misleading until the projection is extended.
#
# The OUTPUT_LO/HI, AX_CARRY_LO/HI, ADDR_KEY, and MEM_ADDR_SRC families
# previously listed here graduated to ``SUPPORTED_DIM_FAMILIES`` once
# the oracle learned to project same-step ALU / address-gather
# materialisations from the reference VM register state. No deferred
# families remain at this time; the tuple is kept (empty) so callers
# that test ``DEFERRED_DIM_FAMILIES`` membership do not need to branch
# on attribute existence.
DEFERRED_DIM_FAMILIES: Tuple[str, ...] = ()


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
    opcode
        Opcode dispatched on this step (one of the ``OP_*`` constants
        from :mod:`symbolic_forward`). Used by the OUTPUT_LO/HI and
        ADDR_KEY projections to decide whether the MEM marker row is
        carrying a store value or is idle. ``None`` for the synthetic
        pre-program snapshot.
    mem_addr
        Address that the memory marker row's bus is gated on for this
        step. ``None`` when the step is not a memory-bus op (everything
        outside PSH / SI / SC / LI / LC). PSH uses the just-decremented
        SP; SI/SC use the just-popped top-of-stack; LI/LC use AX.
    mem_value
        Value the memory bus carries this step (32-bit). For stores this
        is the AX value being written; for loads it is the value just
        read into AX. ``None`` outside memory-bus ops.
    is_store
        ``True`` for SI / SC / PSH steps (MEM_ADDR_SRC fires). False for
        loads or non-memory ops.
    """

    step_idx: int
    pc: int = 0
    ax: int = 0
    sp: int = 0
    bp: int = 0
    stack0: int = 0
    memory: Dict[int, int] = field(default_factory=dict)
    halted: bool = False
    opcode: Optional[int] = None
    mem_addr: Optional[int] = None
    mem_value: Optional[int] = None
    is_store: bool = False


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

            # Per-step memory-bus snapshot. Populated only for the
            # opcodes that drive the MEM marker row; everything else
            # leaves the bus idle (``None``).
            mem_addr: Optional[int] = None
            mem_value: Optional[int] = None
            is_store = False

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
                mem_addr = addr & 0xFFFFFFFF
                mem_value = ax
                # LI is a load — MEM_ADDR_SRC stays zero.
                is_store = False
            elif opcode == OP_SI:
                if sp in stack:
                    addr = stack[sp]
                    sp += 8
                    for i in range(8):
                        memory_bytes[addr + i] = (ax >> (i * 8)) & 0xFF
                    mem_addr = addr & 0xFFFFFFFF
                    mem_value = ax & 0xFFFFFFFF
                    is_store = True
            elif opcode == OP_PSH:
                sp -= 8
                stack[sp] = ax
                # PSH drives the MEM bus with addr=SP (post-dec) and
                # value=AX. MEM_ADDR_SRC = 0 (address is SP, not STACK0).
                mem_addr = sp & 0xFFFFFFFF
                mem_value = ax & 0xFFFFFFFF
                is_store = True
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
                    opcode=opcode,
                    mem_addr=mem_addr,
                    mem_value=mem_value,
                    is_store=is_store,
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


def _emit_output_at_marker(out, marker_pos: int, value: int):
    """OUTPUT_LO/HI one-hots for the byte-0 nibbles of ``value`` at a
    marker row. The op tree's L8/L16 ALU rules emit ``OUTPUT_LO+lo`` and
    ``OUTPUT_HI+hi`` for the AX byte 0 nibble at the AX marker (and the
    matching MEM byte 0 for the MEM marker, PC byte 0 for the PC marker).
    """
    byte0 = value & 0xFF
    lo = byte0 & 0x0F
    hi = (byte0 >> 4) & 0x0F
    out[(marker_pos, f"OUTPUT_LO+{lo}")] = 1.0
    out[(marker_pos, f"OUTPUT_HI+{hi}")] = 1.0


def _emit_ax_carry_at_marker(out, marker_pos: int, value: int):
    """AX_CARRY_LO/HI one-hots for the byte-0 nibbles of ``value`` at a
    marker row. AX_CARRY mirrors the ALU staging path: by the end of the
    step it carries the same byte-0 nibble values OUTPUT carries (L3
    head 1 broadcasts EMBED_LO/HI -> AX_CARRY_LO/HI at the AX marker).
    """
    byte0 = value & 0xFF
    lo = byte0 & 0x0F
    hi = (byte0 >> 4) & 0x0F
    out[(marker_pos, f"AX_CARRY_LO+{lo}")] = 1.0
    out[(marker_pos, f"AX_CARRY_HI+{hi}")] = 1.0


def _emit_addr_key_at_mem(out, mem_marker_pos: int, addr: int):
    """ADDR_KEY one-hots for the three address bytes at the MEM marker.

    The L4/L5 SP-gather + L15 store-stack0-sp-byte0-addr ops project
    the address being accessed this step into ADDR_KEY (48 cells, 3 nibble
    one-hots). The aliases ADDR_B0_HI/ADDR_B1_HI/ADDR_B2_HI pin the
    semantics: cell ``k`` of ``ADDR_KEY[0..15]`` is byte-0 hi nibble,
    ``ADDR_KEY[16..31]`` is byte-1 hi nibble, ``ADDR_KEY[32..47]`` is
    byte-2 hi nibble.
    """
    b0 = addr & 0xFF
    b1 = (addr >> 8) & 0xFF
    b2 = (addr >> 16) & 0xFF
    out[(mem_marker_pos, f"ADDR_KEY+{(b0 >> 4) & 0x0F}")] = 1.0
    out[(mem_marker_pos, f"ADDR_KEY+{16 + ((b1 >> 4) & 0x0F)}")] = 1.0
    out[(mem_marker_pos, f"ADDR_KEY+{32 + ((b2 >> 4) & 0x0F)}")] = 1.0


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

    # Same-step ALU / MEM-bus projections.
    #
    # OUTPUT_LO/HI: marker-row one-hot of the byte-0 nibbles of the AX
    # / MEM-value / PC registers. The op tree's L8/L16 ALU rules write
    # OUTPUT_LO[ax & 0xF] at the AX-marker row; the PSH/SI MEM bus
    # writes OUTPUT_LO[mem_value & 0xF] at the MEM-marker row; L16
    # post-ENT carry writes OUTPUT_LO[pc & 0xF] at the PC-marker row.
    _emit_output_at_marker(out, base + POS_AX_MARKER, state.ax)
    _emit_output_at_marker(out, base + POS_PC_MARKER, state.pc)
    if state.mem_value is not None:
        _emit_output_at_marker(out, base + POS_MEM_MARKER, state.mem_value)

    # AX_CARRY_LO/HI: AX-marker only (L3 carry-forward head 1).
    _emit_ax_carry_at_marker(out, base + POS_AX_MARKER, state.ax)

    # ADDR_KEY: MEM-marker only, 3-nibble address one-hot.
    if state.mem_addr is not None:
        _emit_addr_key_at_mem(out, base + POS_MEM_MARKER, state.mem_addr)

    # MEM_ADDR_SRC: single-cell flag at the MEM-marker row. Fires for
    # SI / SC stores (address comes from STACK0); does not fire for
    # PSH (address is SP) or loads.
    if state.opcode is not None and state.is_store and state.opcode != OP_PSH:
        out[(base + POS_MEM_MARKER, "MEM_ADDR_SRC+0")] = 1.0

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

    # Same-step ALU / MEM-bus dims (collapsed onto step position; the
    # underlying per-token semantics live in the per-token projection
    # above — see ``_emit_output_at_marker`` for the marker-row rules).
    #
    # OUTPUT_LO/HI: the AX-marker row writes the AX byte-0 nibbles. The
    # PC-marker row writes the PC byte-0 nibbles. The MEM-marker row
    # writes the mem-bus byte-0 nibbles for store/load steps. Multiple
    # cells can fire if AX / PC / MEM nibbles disagree; that is the
    # documented "writes accumulate" behaviour of the residual stream.
    for value in (state.ax, state.pc):
        byte0 = value & 0xFF
        out[(pos, f"OUTPUT_LO+{byte0 & 0x0F}")] = 1.0
        out[(pos, f"OUTPUT_HI+{(byte0 >> 4) & 0x0F}")] = 1.0
    if state.mem_value is not None:
        byte0 = state.mem_value & 0xFF
        out[(pos, f"OUTPUT_LO+{byte0 & 0x0F}")] = 1.0
        out[(pos, f"OUTPUT_HI+{(byte0 >> 4) & 0x0F}")] = 1.0

    # AX_CARRY_LO/HI: one-hot of AX byte 0 nibbles (AX marker only).
    ax_byte0 = state.ax & 0xFF
    out[(pos, f"AX_CARRY_LO+{ax_byte0 & 0x0F}")] = 1.0
    out[(pos, f"AX_CARRY_HI+{(ax_byte0 >> 4) & 0x0F}")] = 1.0

    # ADDR_KEY: 3-nibble one-hot of the mem-bus address. Only fires on
    # memory-bus ops; non-memory steps leave the bus idle.
    if state.mem_addr is not None:
        addr = state.mem_addr
        b0 = addr & 0xFF
        b1 = (addr >> 8) & 0xFF
        b2 = (addr >> 16) & 0xFF
        out[(pos, f"ADDR_KEY+{(b0 >> 4) & 0x0F}")] = 1.0
        out[(pos, f"ADDR_KEY+{16 + ((b1 >> 4) & 0x0F)}")] = 1.0
        out[(pos, f"ADDR_KEY+{32 + ((b2 >> 4) & 0x0F)}")] = 1.0

    # MEM_ADDR_SRC: flag, set on SI/SC stores only.
    if state.opcode is not None and state.is_store and state.opcode != OP_PSH:
        out[(pos, "MEM_ADDR_SRC+0")] = 1.0

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
