"""Pre-step symbolic residual builder for collapsed-step bug fixes.

Background
----------

When an agent investigates a *collapsed-step* L9 bug (the dispatcher
fuses an ``IMM`` + ``CMP`` pair into a single transformer step rather
than two), the verifier-driven workflow needs to test the candidate
L9 fix against a synthetic residual that represents the state *just
before* the collapsed step's compute layers fire. Historically that
required spinning up the full ~18-layer transformer plus the embedding
bake — about 60 seconds per probe, which dominates the iteration loop.

This module sidesteps the compile/bake by walking the existing
:class:`~.dim_oracle.ReferenceOracle` forward and projecting each
register state through :func:`~.symbolic_forward.default_embedding_for_instruction`.
The result is a *bag-of-dims* dict keyed by ``"NAME+offset"`` that
mimics the residual at the AX-marker position at the start of the
requested step. It composes byte-identically with the
:func:`~.ir.compare_symbolic_to_lowered_ffn` byte-identity gate (the
gate accepts the same dict shape via its ``state`` parameter).

API at a glance
---------------

>>> from c4_release.neural_vm.unified_compiler.symbolic_state_builder import (
...     Instruction, IMM, PSH, EQ, EXIT, state_after_program, to_tensor,
... )
>>> state = state_after_program([IMM(5), PSH, IMM(5), EQ, EXIT], step=3)
>>> state["REG_AX_BYTE0_LO+5"]   # AX byte 0 lo nibble = 5
1.0
>>> state["STACK0_BYTE0_LO+5"]   # STACK0 byte 0 lo nibble = 5
1.0

Convert the dict into a flat residual tensor for unit tests / the
byte-identity gate:

>>> import torch
>>> dim_positions = {"REG_AX_BYTE0_LO": 0, "STACK0_BYTE0_LO": 16}
>>> tensor = to_tensor(state, d_model=64, dim_positions=dim_positions)
>>> tensor.shape
torch.Size([64])

The module intentionally exposes a *narrow* surface: programs in,
dict (or tensor) out. The heavy lifting (VM semantics, embedding
projection, byte/nibble decomposition) is all delegated to the
existing oracle and embedding helpers — this is a thin orchestration
layer, not a re-implementation.

Scope
-----

The state dict covers the dims an L9-style collapsed-step fix needs:

* ``OP_<NAME>+0`` — opcode indicator for the *requested* step's
  instruction (the one whose compute layers have not yet fired).
* ``REG_AX_BYTEh_LO/HI+k`` — per-byte AX nibble one-hots (h in 0..3).
* ``STACK0_BYTEh_LO/HI+k`` — per-byte STACK0 nibble one-hots.
* ``REG_PC_BYTEh_LO/HI+k``, ``REG_SP_BYTEh_LO/HI+k``,
  ``REG_BP_BYTEh_LO/HI+k`` — per-byte register one-hots.
* ``CLEAN_EMBED_LO/HI+k``, ``EMBED_LO/HI+k`` — collapsed-onto-step
  embedding one-hots (matches the oracle's bag-of-dims projection so
  this seed is interchangeable with an oracle-projected residual).
* ``IMM_AX_LO[_byte]+k``, ``IMM_AX_HI[_byte]+k`` — immediate-payload
  one-hots for the *requested* step (so IMM-decode rules can fire).
* ``CONST+0`` — the ubiquitous bias dim every FFN rule depends on.
* ``PC+0`` — scalar PC value (for any rule keying on absolute PC).

Cross-step alias families (``.*.-1`` PREV_STEP routes) are *not*
projected — those are KV-cache back-edges, not residuals. The
:class:`~.dim_oracle.ReferenceOracle` is the source of truth for the
register trajectory; if a caller needs more dim families (OUTPUT_LO,
ADDR_KEY, etc.), pass the resulting state through
:func:`~.dim_oracle.project_state_to_residual` and merge — both
return dicts in the same shape.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Union

from .dim_oracle import ReferenceOracle
from .symbolic_forward import (
    OP_ADD,
    OP_ADJ,
    OP_AND,
    OP_BNZ,
    OP_BZ,
    OP_DIV,
    OP_ENT,
    OP_EQ,
    OP_EXIT,
    OP_GE,
    OP_GT,
    OP_IMM,
    OP_JMP,
    OP_JSR,
    OP_LE,
    OP_LEA,
    OP_LEV,
    OP_LI,
    OP_LT,
    OP_MOD,
    OP_MUL,
    OP_NE,
    OP_OR,
    OP_PSH,
    OP_SHL,
    OP_SHR,
    OP_SI,
    OP_SUB,
    OP_XOR,
    decode_instr,
    default_embedding_for_instruction,
    encode_instr,
)

__all__ = [
    "Instruction",
    "state_after_program",
    "to_tensor",
    "describe_state",
    "IMM",
    "PSH",
    "EXIT",
    "ADD",
    "SUB",
    "MUL",
    "DIV",
    "MOD",
    "EQ",
    "NE",
    "LT",
    "GT",
    "LE",
    "GE",
    "OR",
    "XOR",
    "AND",
    "SHL",
    "SHR",
    "LEA",
    "JMP",
    "JSR",
    "BZ",
    "BNZ",
    "ENT",
    "ADJ",
    "LEV",
    "LI",
    "SI",
]


# ---------------------------------------------------------------------------
# Instruction type — the input the brief asked for
# ---------------------------------------------------------------------------


class Instruction(NamedTuple):
    """A single C4 instruction = ``(opcode_int, immediate)``.

    Callers usually build these via the factory helpers (``IMM(5)``,
    ``PSH``, ``EQ``, ``EXIT`` ...) so test programs read like the
    bytecode in the bug brief:

    >>> program = [IMM(42), EXIT]
    >>> program = [IMM(5), PSH, IMM(5), EQ, EXIT]

    A raw ``(op, imm)`` tuple is also accepted (it is a ``NamedTuple``
    after all), and an encoded ``int`` instruction word is upgraded
    transparently — both forms are tolerated by
    :func:`state_after_program`.
    """

    op: int
    imm: int = 0


# Factories for zero-immediate ops (the common case).
PSH = Instruction(OP_PSH, 0)
EXIT = Instruction(OP_EXIT, 0)
ADD = Instruction(OP_ADD, 0)
SUB = Instruction(OP_SUB, 0)
MUL = Instruction(OP_MUL, 0)
DIV = Instruction(OP_DIV, 0)
MOD = Instruction(OP_MOD, 0)
EQ = Instruction(OP_EQ, 0)
NE = Instruction(OP_NE, 0)
LT = Instruction(OP_LT, 0)
GT = Instruction(OP_GT, 0)
LE = Instruction(OP_LE, 0)
GE = Instruction(OP_GE, 0)
OR = Instruction(OP_OR, 0)
XOR = Instruction(OP_XOR, 0)
AND = Instruction(OP_AND, 0)
SHL = Instruction(OP_SHL, 0)
SHR = Instruction(OP_SHR, 0)
LI = Instruction(OP_LI, 0)
SI = Instruction(OP_SI, 0)
LEV = Instruction(OP_LEV, 0)


# Factories for ops that *do* carry an immediate.
def IMM(value: int) -> Instruction:
    return Instruction(OP_IMM, value & 0xFFFFFFFF)


def LEA(value: int) -> Instruction:
    return Instruction(OP_LEA, value & 0xFFFFFFFF)


def JMP(value: int) -> Instruction:
    return Instruction(OP_JMP, value & 0xFFFFFFFF)


def JSR(value: int) -> Instruction:
    return Instruction(OP_JSR, value & 0xFFFFFFFF)


def BZ(value: int) -> Instruction:
    return Instruction(OP_BZ, value & 0xFFFFFFFF)


def BNZ(value: int) -> Instruction:
    return Instruction(OP_BNZ, value & 0xFFFFFFFF)


def ENT(value: int) -> Instruction:
    return Instruction(OP_ENT, value & 0xFFFFFFFF)


def ADJ(value: int) -> Instruction:
    return Instruction(OP_ADJ, value & 0xFFFFFFFF)


# ---------------------------------------------------------------------------
# Program normalisation
# ---------------------------------------------------------------------------


_InstrLike = Union[Instruction, tuple, int]


def _normalize_program(program: Sequence[_InstrLike]) -> List[int]:
    """Convert a mixed list of ``Instruction`` / ``(op, imm)`` tuples /
    encoded ints into the encoded-word list the
    :class:`ReferenceOracle` consumes."""
    out: List[int] = []
    for item in program:
        if isinstance(item, Instruction):
            out.append(encode_instr(item.op, item.imm))
        elif isinstance(item, tuple) and len(item) == 2:
            op, imm = item
            out.append(encode_instr(int(op), int(imm)))
        elif isinstance(item, int):
            # Already-encoded instruction word.
            out.append(item)
        else:
            raise TypeError(
                f"state_after_program: unsupported instruction {item!r}; "
                f"expected Instruction, (op, imm) tuple, or encoded int"
            )
    return out


# ---------------------------------------------------------------------------
# Per-register nibble emission
# ---------------------------------------------------------------------------


def _emit_register_nibbles(
    out: Dict[str, float], reg_name: str, value: int,
) -> None:
    """Emit ``REG_<NAME>_BYTEh_LO+k`` / ``REG_<NAME>_BYTEh_HI+k`` one-hots
    for every byte of ``value``.

    Mirrors the layout the embedding bake uses when it writes a
    register's bytes into the residual at marker rows. The bag-of-dims
    form collapses every byte onto a single position.
    """
    for h in range(4):
        byte_val = (value >> (h * 8)) & 0xFF
        lo = byte_val & 0x0F
        hi = (byte_val >> 4) & 0x0F
        out[f"REG_{reg_name}_BYTE{h}_LO+{lo}"] = 1.0
        out[f"REG_{reg_name}_BYTE{h}_HI+{hi}"] = 1.0


def _emit_stack0_nibbles(out: Dict[str, float], value: int) -> None:
    """Emit ``STACK0_BYTEh_LO+k`` / ``STACK0_BYTEh_HI+k`` one-hots for
    every byte of the top-of-stack value.

    Distinct from ``REG_AX_*`` etc. because STACK0 is what an opcode
    like ``EQ`` / ``ADD`` consumes alongside AX during its compute
    step. Tests verifying "STACK0 has the pushed value" use these dims
    directly.
    """
    for h in range(4):
        byte_val = (value >> (h * 8)) & 0xFF
        lo = byte_val & 0x0F
        hi = (byte_val >> 4) & 0x0F
        out[f"STACK0_BYTE{h}_LO+{lo}"] = 1.0
        out[f"STACK0_BYTE{h}_HI+{hi}"] = 1.0


# ---------------------------------------------------------------------------
# The state builder — main entry point
# ---------------------------------------------------------------------------


def state_after_program(
    program: Sequence[_InstrLike],
    step: int,
    *,
    code_base: int = 0,
    initial_sp: int = 0x100000,
    extra_state: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    """Return a symbolic residual dict representing the state AT a given
    step (before the step's compute layers fire).

    The returned dict mimics the residual at the AX-marker position at
    the *start* of step ``step``. Concretely:

    * AX / SP / BP / STACK0 carry the register values *after* step
      ``step - 1`` executed (or the initial values if ``step == 0``).
    * PC, opcode indicator, and IMM_AX one-hots reflect the *current*
      instruction at ``program[step]`` — the one whose compute layers
      have not yet fired.

    This is the exact shape an L9 dispatcher fix wants to test against:
    "given the residual after IMM 5, PSH, IMM 5 has settled, does my
    L9 EQ-dispatch rule fire when step 3 (EQ) is up next?".

    Parameters
    ----------
    program
        Sequence of instructions. Each element may be an
        :class:`Instruction`, a raw ``(op, imm)`` tuple, or an encoded
        32-bit instruction word.
    step
        Step index (0-based) whose pre-compute state to return. Must
        satisfy ``0 <= step < len(program)``.
    code_base, initial_sp
        Forwarded to :class:`ReferenceOracle`. Defaults match the test
        harness defaults.
    extra_state
        Optional dict of additional ``"NAME+offset" -> float`` entries
        merged on top of the projected state. Useful for sticking in
        flag dims (e.g. ``"MARK_PC_PIN+0": 1.0``) that the program
        itself does not set.

    Returns
    -------
    Dict[str, float]
        Bag-of-dims dict in the canonical ``"NAME+offset"`` form. Pass
        directly to :class:`~.dsl_interpreter.DSLInterpreter` as
        ``initial_state`` or to
        :func:`~.ir.compare_symbolic_to_lowered_ffn` as ``state=``.
    """
    if step < 0:
        raise ValueError(f"state_after_program: step must be >= 0, got {step}")
    if step >= len(program):
        raise ValueError(
            f"state_after_program: step {step} is past program end "
            f"(program has {len(program)} instructions)"
        )

    encoded = _normalize_program(program)

    # ------------------------------------------------------------------
    # Step 1: figure out the register state at the *start* of `step`.
    #
    # `ReferenceOracle.state_at_step(i)` returns the state AFTER step i
    # executed. So the state at the START of `step` is the state AFTER
    # `step - 1`, or the initial state when step == 0.
    # ------------------------------------------------------------------
    oracle = ReferenceOracle(
        encoded, code_base=code_base, initial_sp=initial_sp,
    )

    if step == 0:
        # Synthetic pre-program state — all registers at their initial
        # values; PC at code_base; AX/BP/STACK0 all zero.
        ax = 0
        sp = initial_sp
        bp = initial_sp
        stack0 = 0
        pc = code_base
    else:
        prev = oracle.state_at_step(step - 1)
        ax = prev.ax
        sp = prev.sp
        bp = prev.bp
        stack0 = prev.stack0
        # PC at start of step `step` is whatever PC the previous step
        # left in flight. The oracle stores the *post-step* PC, which
        # is already the next-instruction PC under straight-line code
        # and the branch target after a taken jump. Match that.
        pc = prev.pc

    # ------------------------------------------------------------------
    # Step 2: project the current instruction's embedding (so the
    # opcode indicator + IMM payload dims reflect the upcoming step,
    # not the previous one).
    # ------------------------------------------------------------------
    op, imm = decode_instr(encoded[step])
    state: Dict[str, float] = dict(
        default_embedding_for_instruction(op, imm, pc=pc)
    )

    # ------------------------------------------------------------------
    # Step 3: add per-register byte nibble one-hots from the register
    # state at the start of the step. These mimic the residual values
    # downstream layers expect to read for AX / SP / BP / PC / STACK0.
    # ------------------------------------------------------------------
    _emit_register_nibbles(state, "AX", ax)
    _emit_register_nibbles(state, "SP", sp)
    _emit_register_nibbles(state, "BP", bp)
    _emit_register_nibbles(state, "PC", pc)
    _emit_stack0_nibbles(state, stack0)

    # The PC scalar dim — overwrite whatever the embedding wrote (the
    # embedding seeds PC+0 = float(step_idx); here we want the actual
    # VM PC, which is what downstream PC-keyed rules use).
    state["PC+0"] = float(pc)

    # ------------------------------------------------------------------
    # Step 4: merge any caller-supplied extras (last so they win on
    # collisions — that is the caller's prerogative).
    # ------------------------------------------------------------------
    if extra_state:
        for k, v in extra_state.items():
            state[k] = float(v)

    return state


# ---------------------------------------------------------------------------
# Tensor conversion helper
# ---------------------------------------------------------------------------


def to_tensor(
    state: Mapping[str, float],
    d_model: int,
    *,
    dim_positions: Optional[Mapping[str, int]] = None,
    dtype: Any = None,
) -> Any:
    """Convert a state dict into a 1-D residual tensor for unit tests.

    Parameters
    ----------
    state
        Dict in the canonical ``"NAME+offset"`` form (as returned by
        :func:`state_after_program`).
    d_model
        Residual width. Entries whose resolved index falls outside
        ``[0, d_model)`` are silently dropped (matches the
        contribution-algebra "writes outside the layout are dead" rule).
    dim_positions
        Optional ``"DIM_FAMILY" -> base_index`` map. When supplied, a
        key ``"FOO+k"`` lands at ``dim_positions["FOO"] + k``. When
        omitted, each key in ``state`` is parsed as ``"FOO+k"`` and the
        helper invents a stable layout by enumerating distinct family
        names in insertion order with a 64-cell stride — enough for
        every one-hot family the builder emits.
    dtype
        Optional torch dtype. Defaults to ``torch.float32``.

    Returns
    -------
    torch.Tensor
        Shape ``(d_model,)``. Use ``residual.unsqueeze(0)`` to feed a
        single-position attention forward, or
        ``residual.unsqueeze(0).unsqueeze(0)`` for the (batch, seq, d)
        convention.

    Examples
    --------
    >>> state = state_after_program([IMM(42), EXIT], step=0)
    >>> # When the test only needs a couple of dims, a tiny layout works:
    >>> layout = {"REG_AX_BYTE0_LO": 0, "OP_IMM": 16, "CONST": 17}
    >>> tensor = to_tensor(state, d_model=64, dim_positions=layout)
    """
    # Local import — module shouldn't drag torch in for non-tensor callers.
    import torch

    if dtype is None:
        dtype = torch.float32

    tensor = torch.zeros(d_model, dtype=dtype)

    # Build (or accept) a family -> base_index layout.
    if dim_positions is None:
        # Auto-layout: 64-cell stride per family in insertion order.
        # 64 is the largest nibble family this module emits (4 bytes *
        # 16 cells = 64), so each family gets its own contiguous block.
        family_to_base: Dict[str, int] = {}
        stride = 64
        next_base = 0
        for key in state:
            family = key.split("+", 1)[0]
            if family not in family_to_base:
                family_to_base[family] = next_base
                next_base += stride
        dim_positions = family_to_base

    for key, value in state.items():
        if "+" in key:
            family, offset_s = key.split("+", 1)
            try:
                offset = int(offset_s)
            except ValueError:
                continue
        else:
            family, offset = key, 0
        base = dim_positions.get(family)
        if base is None:
            continue
        idx = base + offset
        if 0 <= idx < d_model:
            tensor[idx] = float(value)

    return tensor


# ---------------------------------------------------------------------------
# Diagnostic helpers — convenient when inspecting a state dict at the REPL
# ---------------------------------------------------------------------------


def describe_state(state: Mapping[str, float]) -> str:
    """Return a short human-readable summary of a state dict.

    Useful in test failures (``assert ... , describe_state(state)``)
    and at the REPL. Not part of the byte-identity contract — purely a
    debugging convenience.
    """
    set_keys = sorted(k for k, v in state.items() if v != 0.0)
    families: Dict[str, List[str]] = {}
    for k in set_keys:
        family = k.split("+", 1)[0]
        families.setdefault(family, []).append(k.split("+", 1)[1])
    lines = [f"state_dict[{len(set_keys)} set]:"]
    for family in sorted(families):
        cells = families[family]
        lines.append(f"  {family}: {', '.join(cells)}")
    return "\n".join(lines)
