"""Symbolic declarative byte-signature helpers for neural divergence checks.

These helpers run a program through ``SymbolicDeclarativeProgramRunner`` and
return, per VM step, the expected byte values that the neural model is
required to emit at each slot of the 35-token step layout
(``PC(5) + AX(5) + SP(5) + BP(5) + STACK0(5) + MEM(9) + SE(1)``).

The signatures are intentionally not derived from the neural model output —
they're the lowering target. Tests use them to:

* Pin down the *symbolic* byte the neural model is supposed to produce at a
  particular (step, slot), without substituting Python results into the
  neural pipeline.
* Build small per-cell residual probes for declarative bake rules without
  hand-rolling the expected nibble values.
* Locate the (step, slot) that diverged when comparing neural vs declarative
  token streams.
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


_STEP_SLOT_NAMES: Tuple[str, ...] = (
    "REG_PC",
    "PC_byte0",
    "PC_byte1",
    "PC_byte2",
    "PC_byte3",
    "REG_AX",
    "AX_byte0",
    "AX_byte1",
    "AX_byte2",
    "AX_byte3",
    "REG_SP",
    "SP_byte0",
    "SP_byte1",
    "SP_byte2",
    "SP_byte3",
    "REG_BP",
    "BP_byte0",
    "BP_byte1",
    "BP_byte2",
    "BP_byte3",
    "STACK0",
    "STACK0_byte0",
    "STACK0_byte1",
    "STACK0_byte2",
    "STACK0_byte3",
    "MEM",
    "MEM_addr0",
    "MEM_addr1",
    "MEM_addr2",
    "MEM_addr3",
    "MEM_value0",
    "MEM_value1",
    "MEM_value2",
    "MEM_value3",
    "STEP_END",
)

# Subset of ``_STEP_SLOT_NAMES`` that holds byte-valued tokens (0..255). The
# marker slots (REG_*, MEM, STACK0, STEP_END) hold special token IDs ≥ 256.
_BYTE_SLOT_NAMES: Tuple[str, ...] = tuple(
    name for name in _STEP_SLOT_NAMES
    if "byte" in name or "addr" in name or "value" in name
)


@dataclass(frozen=True)
class SymbolicStepByteSignature:
    """Expected byte values per slot for one symbolic VM step.

    ``slot_bytes`` covers the byte-valued slots only. ``tokens`` is the full
    35-token sequence (marker tokens + bytes) the neural model should emit
    for this step, in order — handy for diffing against an autoregressive
    neural context dump.
    """

    step_index: int
    opcode: int
    opcode_name: str
    imm: int
    slot_bytes: Dict[str, int]
    tokens: Tuple[int, ...]
    halted: bool

    def byte(self, slot: str) -> int:
        if slot not in _BYTE_SLOT_NAMES:
            raise KeyError(f"{slot!r} is not a byte slot (it's a marker)")
        return self.slot_bytes[slot]

    def nibbles(self, slot: str) -> Tuple[int, int]:
        value = self.byte(slot)
        return value & 0xF, (value >> 4) & 0xF


def _u32(value: int) -> int:
    return value & 0xFFFFFFFF


def _u32_bytes(value: int) -> Tuple[int, int, int, int]:
    value = _u32(value)
    return (
        value & 0xFF,
        (value >> 8) & 0xFF,
        (value >> 16) & 0xFF,
        (value >> 24) & 0xFF,
    )


def symbolic_byte_signatures(
    bytecode: Sequence[int],
    data: Sequence[int] | bytes = b"",
    *,
    max_steps: Optional[int] = None,
) -> List[SymbolicStepByteSignature]:
    """Return the expected per-step byte signature for ``bytecode``.

    The result list has one entry per executed VM step, in order. Each entry
    is the declarative truth for the 35-token step the neural model is
    expected to emit.
    """

    from neural_vm.verification.symbolic_program import (
        SymbolicDeclarativeProgramRunner,
    )
    from neural_vm.vm_step import Token

    runner = SymbolicDeclarativeProgramRunner()
    state = runner.init_state(bytecode, data)

    signatures: List[SymbolicStepByteSignature] = []
    while True:
        if max_steps is not None and state.steps >= max_steps:
            break
        if not runner.step(state):
            break
        trace = state.trace[-1]

        pc_bytes = _u32_bytes(state.pc)
        ax_bytes = _u32_bytes(state.ax)
        sp_bytes = _u32_bytes(state.sp)
        bp_bytes = _u32_bytes(state.bp)
        stack0_value = state.mem_read(state.sp)
        stack0_bytes = _u32_bytes(stack0_value)
        mem_addr_bytes = _u32_bytes(trace.mem_addr)
        mem_value_bytes = _u32_bytes(trace.mem_value)

        slot_bytes: Dict[str, int] = {
            "PC_byte0": pc_bytes[0],
            "PC_byte1": pc_bytes[1],
            "PC_byte2": pc_bytes[2],
            "PC_byte3": pc_bytes[3],
            "AX_byte0": ax_bytes[0],
            "AX_byte1": ax_bytes[1],
            "AX_byte2": ax_bytes[2],
            "AX_byte3": ax_bytes[3],
            "SP_byte0": sp_bytes[0],
            "SP_byte1": sp_bytes[1],
            "SP_byte2": sp_bytes[2],
            "SP_byte3": sp_bytes[3],
            "BP_byte0": bp_bytes[0],
            "BP_byte1": bp_bytes[1],
            "BP_byte2": bp_bytes[2],
            "BP_byte3": bp_bytes[3],
            "STACK0_byte0": stack0_bytes[0],
            "STACK0_byte1": stack0_bytes[1],
            "STACK0_byte2": stack0_bytes[2],
            "STACK0_byte3": stack0_bytes[3],
            "MEM_addr0": mem_addr_bytes[0],
            "MEM_addr1": mem_addr_bytes[1],
            "MEM_addr2": mem_addr_bytes[2],
            "MEM_addr3": mem_addr_bytes[3],
            "MEM_value0": mem_value_bytes[0],
            "MEM_value1": mem_value_bytes[1],
            "MEM_value2": mem_value_bytes[2],
            "MEM_value3": mem_value_bytes[3],
        }

        end_token = Token.HALT if state.halted else Token.STEP_END
        tokens: Tuple[int, ...] = (
            Token.REG_PC, *pc_bytes,
            Token.REG_AX, *ax_bytes,
            Token.REG_SP, *sp_bytes,
            Token.REG_BP, *bp_bytes,
            Token.STACK0, *stack0_bytes,
            Token.MEM, *mem_addr_bytes, *mem_value_bytes,
            end_token,
        )

        signatures.append(
            SymbolicStepByteSignature(
                step_index=len(signatures),
                opcode=trace.opcode,
                opcode_name=trace.name,
                imm=trace.imm,
                slot_bytes=slot_bytes,
                tokens=tokens,
                halted=state.halted,
            )
        )

        if state.halted:
            break

    return signatures


def find_steps_by_opcode(
    signatures: Sequence[SymbolicStepByteSignature],
    opcode_name: str,
) -> List[SymbolicStepByteSignature]:
    return [s for s in signatures if s.opcode_name == opcode_name]


def step_token_stream(
    signatures: Sequence[SymbolicStepByteSignature],
) -> List[int]:
    """Flatten the per-step token tuples into one stream.

    Useful for cross-checking against ``_append_symbolic_step_tokens`` in
    ``test_1096_neural_declarative_diagnostic.py``.
    """

    stream: List[int] = []
    for sig in signatures:
        stream.extend(sig.tokens)
    return stream


__all__ = [
    "SymbolicStepByteSignature",
    "find_steps_by_opcode",
    "step_token_stream",
    "symbolic_byte_signatures",
    "_BYTE_SLOT_NAMES",
    "_STEP_SLOT_NAMES",
]
