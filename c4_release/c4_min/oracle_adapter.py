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
  * ``expected_8bit`` reconciles the ground truth to the substrate's 8-bit
    width by masking the (full-coverage, 32-bit) reference-oracle exit code to
    8 bits — the neural_vm reference is 32-bit and disagrees on wrap/underflow
    boundary cases (see DESIGN.md (g)).

Ops outside the implemented slice (IMM/LEA/PSH/ADD/SUB/HALT) raise
``NotImplementedError`` inside the compiler; the harness records that as
``model_error`` — the FAN-OUT baseline (op not built yet).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from . import isa
from . import compiler as _compiler
from .oracle import Decoded, Expected, Program
from .oracle import expected_for_program as _expected_ref


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


def expected_8bit(prog: Program, *, max_steps: int = 64) -> Expected:
    """Ground truth reconciled to the substrate's 8-bit width.

    The reference oracle (``oracle.expected_for_program`` via
    ``neural_vm.verification.symbolic_program``) is a full 32-bit VM and covers
    every op-class — but it disagrees with the 8-bit substrate on wrap/underflow
    boundary cases (e.g. ``255+1 -> 256`` vs ``0``, ``7-9 -> 2**32-2`` vs
    ``254``). Since the substrate is defined 8-bit (DESIGN.md (a)–(c)), we take
    the 32-bit oracle's exit code and **mask it to 8 bits** — the correct
    semantic bridge (``256 & 0xFF == 0``, ``(2**32-2) & 0xFF == 254``). This
    keeps full op-class coverage on the expected side (the neural_vm interpreter
    knows ENT/ADJ/JSR/LEV etc., which the substrate's slice interpreter does
    not) while matching the model's width for the implemented ops.
    """
    exp = _expected_ref(prog, max_steps=max_steps)
    if exp.exit_code is None:
        return exp
    return Expected(
        exit_code=int(exp.exit_code) & 0xFF,
        steps=exp.steps,
        halted=exp.halted,
        trace=exp.trace,
        error=exp.error,
    )


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
