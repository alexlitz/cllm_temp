"""Adapter wiring the per-op ORACLE HARNESS to the REAL c4_min compiler+model.

The harness (``run_oracle.py``) was written against a *stub* interface:

    c4_min.compile.compile_program(prog: Program) -> state_dict
    c4_min.model.run(state_dict, prog, *, max_steps) -> Decoded

The REAL substrate (``compiler.py``) exposes a different shape:

    compiler.compile_program(prog=[(name, imm), ...]) -> (model, layout, code)
    compiler.run(model, layout, code) -> list[int]   # per-step AX

This module reconciles the two (see DESIGN.md section (g)):

  * ``prog_to_ops`` decodes ``Program.bytecode`` (the c4 encoding
    ``op | imm<<8``) into the substrate's ``[(op_name, imm), ...]`` form.
  * ``RealBackend`` compiles+runs the real substrate and assembles a
    ``Decoded`` (exit-code-only: the straight-line slice tracks AX, not PC).
  * ``expected_8bit`` computes the ground truth via the substrate's OWN
    clean-room 8-bit reference interpreter (``isa.interpret``) so the expected
    semantics match the model's 8-bit width (the neural_vm reference is 32-bit
    and disagrees on wrap/underflow boundary cases — see DESIGN.md (g)).

Ops outside the implemented slice (IMM/LEA/PSH/ADD/SUB/HALT) raise
``NotImplementedError`` inside the compiler; the harness records that as
``model_error`` — the FAN-OUT baseline (op not built yet).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from . import isa
from . import compiler as _compiler
from .oracle import Decoded, Expected, Program


# The reference-VM EXIT opcode id (oracle uses OP_EXIT=38); the substrate's
# clean-room HALT shares that value (isa.HALT == 38).
_EXIT = 38


def prog_to_ops(prog: Program) -> List[Tuple[str, int]]:
    """Decode a harness ``Program`` into the substrate ``[(op_name, imm), ...]``.

    Each encoded word is ``op | (imm << 8)``. The reference programs terminate
    on ``EXIT`` (id 38); the substrate calls that op ``HALT`` (same id), so we
    map it by name. Raises ``KeyError`` if an opcode id is unknown to the
    substrate ISA (a genuinely out-of-subset instruction).
    """
    ops: List[Tuple[str, int]] = []
    for word in prog.bytecode:
        op = int(word) & 0xFF
        imm = (int(word) >> 8) & 0xFF
        name = isa.NAMES.get(op)
        if name is None:
            raise KeyError(f"opcode id {op} not in c4_min ISA (prog {prog.label})")
        ops.append((name, imm))
    return ops


def expected_8bit(prog: Program, *, max_steps: int = 256) -> Expected:
    """Ground truth via the substrate's clean-room 8-bit reference interpreter.

    Matches the model-under-test's 8-bit width exactly (unlike the 32-bit
    neural_vm oracle). ``exit_code`` is the AX on the HALT step, zero-extended
    into the 32-bit field the harness compares against.
    """
    try:
        ops = prog_to_ops(prog)
    except KeyError as exc:
        return Expected(exit_code=None, steps=None, halted=False,
                        error=f"{prog.label}: {exc}")
    code = [isa.Instr(isa.BY_NAME[name], imm) for name, imm in ops]
    emitted = isa.interpret(code, max_steps=max_steps)
    if not emitted:
        return Expected(exit_code=None, steps=None, halted=False,
                        error=f"{prog.label}: produced no output")
    halted = any(ins.op == isa.HALT for ins in code)
    exit_code = int(emitted[-1]) & 0xFFFFFFFF
    return Expected(exit_code=exit_code, steps=len(emitted), halted=halted)


class RealBackend:
    """Drives the REAL c4_min compiler+model; returns a harness ``Decoded``.

    ``compile_program`` compiles the substrate model and stashes ``(model,
    layout, code)``; ``run`` executes it and wraps the per-step AX trace into a
    ``Decoded``. Exit-code-only (no per-step PC trace in the slice), which the
    harness accepts as a valid weaker conformer.
    """

    def compile_program(self, prog: Program) -> dict:
        ops = prog_to_ops(prog)
        model, layout, code = _compiler.compile_program(ops)
        return {"model": model, "layout": layout, "code": code}

    def run(self, state_dict: dict, prog: Program, *, max_steps: int = 64) -> Decoded:
        model, layout, code = (state_dict["model"], state_dict["layout"],
                               state_dict["code"])
        emitted: List[int] = _compiler.run(model, layout, code)
        exit_code: Optional[int]
        exit_code = (int(emitted[-1]) & 0xFFFFFFFF) if emitted else None
        return Decoded(exit_code=exit_code, steps=len(emitted), halted=True)
