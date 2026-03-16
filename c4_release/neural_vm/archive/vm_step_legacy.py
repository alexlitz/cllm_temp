"""
VM Step (Legacy) - Non-autoregressive VM pipeline.

Chains: Register Read -> PureALU -> Register Write -> PC Update
into a single forward pass with NO Python control flow.

All stages are PureFFN subclasses wrapped in SoftMoEFFN for opcode routing.
The runner only handles I/O boundaries and memory interface.

Architecture (45 layers):
  [ClearComputeSlots] -> [LoadRegisters] -> [PureALU(34)] ->
  [WriteAX+BP+SP] -> [PC Increment (unconditional, 7)] -> [WritePC (branches)]

Register layout (within DIM=160):
  PC:  slots 104-111 (E.HEAP_BASE, 8 nibbles across positions)
  SP:  slots 112-119 (E.HEAP_PTR)
  BP:  slots 120-127 (E.HEAP_END)
  AX:  slots 128-135 (E.AX_BASE)

ARCHIVED: This module contains the non-autoregressive VM pipeline.
The active VM uses the autoregressive architecture in vm_step.py.
"""

import torch
import torch.nn as nn
from ..embedding import E, Opcode
from ..base_layers import PureFFN, PureAttention, FlattenedPureFFN, bake_weights
from .pure_moe import MoE
from .pure_alu import build_pure_alu


# =============================================================================
# Register slot definitions
# =============================================================================

PC_BASE = E.HEAP_BASE    # 104
SP_BASE = E.HEAP_PTR     # 112
BP_BASE = E.HEAP_END     # 120
AX_BASE = E.AX_BASE      # 128


# =============================================================================
# Opcode groups
# =============================================================================

BINARY_OPS = [
    Opcode.ADD, Opcode.SUB, Opcode.MUL, Opcode.DIV, Opcode.MOD,
    Opcode.AND, Opcode.OR, Opcode.XOR, Opcode.SHL, Opcode.SHR,
    Opcode.EQ, Opcode.NE, Opcode.LT, Opcode.GT, Opcode.LE, Opcode.GE,
]

RESULT_TO_AX_OPS = BINARY_OPS + [
    Opcode.IMM, Opcode.LEA, Opcode.LI, Opcode.LC, Opcode.GETCHAR,
]

RESULT_TO_PC_OPS = [
    Opcode.JMP, Opcode.BZ, Opcode.BNZ, Opcode.JSR,
    Opcode.BEQ, Opcode.BNE,
]

ENT_OPS = [Opcode.ENT]
SP_FROM_CARRY_OPS = [Opcode.ENT, Opcode.ADJ]
AX_TO_NIBB_OPS = BINARY_OPS
AX_TO_NIBA_OPS = [Opcode.PSH, Opcode.LI, Opcode.LC, Opcode.PUTCHAR, Opcode.EXIT]
BP_TO_TEMP_OPS = [Opcode.LEA, Opcode.ENT]
SP_TO_CARRYIN_OPS = [Opcode.ENT, Opcode.ADJ]

NON_BRANCH_OPS = [
    Opcode.LEA, Opcode.IMM, Opcode.ENT, Opcode.ADJ, Opcode.LEV,
    Opcode.LI, Opcode.LC, Opcode.SI, Opcode.SC, Opcode.PSH,
    Opcode.OR, Opcode.XOR, Opcode.AND,
    Opcode.EQ, Opcode.NE, Opcode.LT, Opcode.GT, Opcode.LE, Opcode.GE,
    Opcode.SHL, Opcode.SHR,
    Opcode.ADD, Opcode.SUB, Opcode.MUL, Opcode.DIV, Opcode.MOD,
    Opcode.GETCHAR, Opcode.PUTCHAR, Opcode.EXIT,
    Opcode.NOP, Opcode.POP,
    Opcode.MALC, Opcode.FREE, Opcode.MSET, Opcode.MCMP,
    Opcode.OPEN, Opcode.READ, Opcode.CLOS, Opcode.PRTF,
]

ALL_OPS = list(set(
    BINARY_OPS + RESULT_TO_AX_OPS + RESULT_TO_PC_OPS + ENT_OPS +
    SP_FROM_CARRY_OPS + AX_TO_NIBA_OPS + BP_TO_TEMP_OPS +
    NON_BRANCH_OPS + RESULT_TO_PC_OPS
))


# =============================================================================
# Stage 1: Register Read (Pre-ALU Operand Loading)
# =============================================================================

class ClearComputeSlots(PureFFN):
    SLOTS_TO_CLEAR = [E.RAW_SUM, E.CARRY_IN, E.CARRY_OUT, E.RESULT, E.TEMP,
                      E.SHIFT_EXTRACT_A, E.SHIFT_EXTRACT_B,
                      E.TEMP_A2, E.TEMP_B2]

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=len(self.SLOTS_TO_CLEAR) * 2)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE
        for i, slot in enumerate(self.SLOTS_TO_CLEAR):
            self.b_up[i * 2] = S
            self.W_gate[i * 2, slot] = -1.0
            self.W_down[slot, i * 2] = 1.0 / S
            self.b_up[i * 2 + 1] = -S
            self.W_gate[i * 2 + 1, slot] = 1.0
            self.W_down[slot, i * 2 + 1] = 1.0 / S


class _RegisterCopyFFN(PureFFN):
    def __init__(self, src_slot: int, dst_slot: int):
        self._src_slot = src_slot
        self._dst_slot = dst_slot
        super().__init__(E.DIM, hidden_dim=2)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE
        self.b_up[0] = S
        self.W_gate[0, self._src_slot] = 1.0
        self.W_down[self._dst_slot, 0] = 1.0 / S
        self.b_up[1] = -S
        self.W_gate[1, self._src_slot] = -1.0
        self.W_down[self._dst_slot, 1] = 1.0 / S


class _RegisterClearFFN(PureFFN):
    def __init__(self, reg_slot: int):
        self._reg_slot = reg_slot
        super().__init__(E.DIM, hidden_dim=2)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE
        self.b_up[0] = S
        self.W_gate[0, self._reg_slot] = -1.0
        self.W_down[self._reg_slot, 0] = 1.0 / S
        self.b_up[1] = -S
        self.W_gate[1, self._reg_slot] = 1.0
        self.W_down[self._reg_slot, 1] = 1.0 / S


class LoadAXtoNIBB(_RegisterCopyFFN):
    def __init__(self): super().__init__(AX_BASE, E.NIB_B)

class LoadAXtoNIBA(_RegisterCopyFFN):
    def __init__(self): super().__init__(AX_BASE, E.NIB_A)

class LoadBPtoTEMP(_RegisterCopyFFN):
    def __init__(self): super().__init__(BP_BASE, E.TEMP)

class LoadSPtoCARRYIN(_RegisterCopyFFN):
    def __init__(self): super().__init__(SP_BASE, E.CARRY_IN)

class LoadAXtoRESULT(_RegisterCopyFFN):
    def __init__(self): super().__init__(AX_BASE, E.RESULT)


class _OverwriteFromResultFFN(PureFFN):
    def __init__(self, reg_slot: int):
        self._reg_slot = reg_slot
        super().__init__(E.DIM, hidden_dim=4)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE
        self.b_up[0] = S
        self.W_gate[0, E.RESULT] = 1.0
        self.W_down[self._reg_slot, 0] = 1.0 / S
        self.b_up[1] = -S
        self.W_gate[1, E.RESULT] = -1.0
        self.W_down[self._reg_slot, 1] = 1.0 / S
        self.b_up[2] = S
        self.W_gate[2, self._reg_slot] = -1.0
        self.W_down[self._reg_slot, 2] = 1.0 / S
        self.b_up[3] = -S
        self.W_gate[3, self._reg_slot] = 1.0
        self.W_down[self._reg_slot, 3] = 1.0 / S


class OverwriteAXFromResult(_OverwriteFromResultFFN):
    def __init__(self): super().__init__(AX_BASE)

class OverwritePCFromResult(_OverwriteFromResultFFN):
    def __init__(self): super().__init__(PC_BASE)

class WriteTEMPtoBP(_RegisterCopyFFN):
    def __init__(self): super().__init__(E.TEMP, BP_BASE)

class WriteCARRYINtoSP(_RegisterCopyFFN):
    def __init__(self): super().__init__(E.CARRY_IN, SP_BASE)


# =============================================================================
# Stage 4: PC Increment (unconditional)
# =============================================================================

class PCIncrementFFN(PureFFN):
    def __init__(self):
        super().__init__(E.DIM, hidden_dim=4)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE
        self.b_up[0] = S
        self.W_up[0, E.POS] = -S * 100
        self.b_gate[0] = 8.0
        self.W_down[PC_BASE, 0] = 1.0 / S
        self.b_up[1] = -S
        self.W_up[1, E.POS] = -S * 100
        self.b_gate[1] = -8.0
        self.W_down[PC_BASE, 1] = 1.0 / S


class PCOverflowFFN(PureFFN):
    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE
        self.W_up[0, PC_BASE] = S
        self.b_up[0] = -S * 15.0
        self.b_gate[0] = 1.0
        self.W_down[PC_BASE, 0] = -16.0 / S
        self.W_down[E.CARRY_OUT, 0] = 1.0 / S
        self.W_up[1, PC_BASE] = S
        self.b_up[1] = -S * 16.0
        self.b_gate[1] = 1.0
        self.W_down[PC_BASE, 1] = 16.0 / S
        self.W_down[E.CARRY_OUT, 1] = -1.0 / S


class PCOverflowClearFFN(PureFFN):
    def __init__(self):
        super().__init__(E.DIM, hidden_dim=4)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE
        self.b_up[0] = S
        self.W_gate[0, E.CARRY_OUT] = -1.0
        self.W_down[E.CARRY_OUT, 0] = 1.0 / S
        self.b_up[1] = -S
        self.W_gate[1, E.CARRY_OUT] = 1.0
        self.W_down[E.CARRY_OUT, 1] = 1.0 / S
        self.W_up[2, PC_BASE] = S
        self.b_up[2] = -S * 15.0
        self.b_gate[2] = 1.0
        self.W_down[PC_BASE, 2] = -16.0 / S
        self.W_down[E.CARRY_OUT, 2] = 1.0 / S
        self.W_up[3, PC_BASE] = S
        self.b_up[3] = -S * 16.0
        self.b_gate[3] = 1.0
        self.W_down[PC_BASE, 3] = 16.0 / S
        self.W_down[E.CARRY_OUT, 3] = -1.0 / S


class PCCarryPropagateAttention(PureAttention):
    def __init__(self):
        super().__init__(E.DIM, num_heads=1, causal=False)

    @bake_weights
    def _bake_weights(self):
        mask = torch.full((E.NUM_POSITIONS, E.NUM_POSITIONS), -1e9)
        for i in range(1, E.NUM_POSITIONS):
            mask[i, i - 1] = 0.0
        mask[0, 0] = 0.0
        self.mask.copy_(mask)
        self.W_v[E.CARRY_IN, E.CARRY_OUT] = 1.0
        self.W_o[E.CARRY_IN, E.CARRY_IN] = 1.0


class PCCarryPropagateFlatFFN(FlattenedPureFFN):
    def __init__(self):
        super().__init__(hidden_dim=14)

    def _bake_weights(self):
        S = E.SCALE
        fi = self._flat_idx
        with torch.no_grad():
            for i in range(1, E.NUM_POSITIONS):
                u = (i - 1) * 2
                self.b_up.data[u] = S
                self.W_gate.data[u, fi(i - 1, E.CARRY_OUT)] = 1.0
                self.W_down.data[fi(i, E.CARRY_IN), u] = 1.0 / S
                self.b_up.data[u + 1] = -S
                self.W_gate.data[u + 1, fi(i - 1, E.CARRY_OUT)] = -1.0
                self.W_down.data[fi(i, E.CARRY_IN), u + 1] = 1.0 / S


class PCCarryIterFFN(PureFFN):
    def __init__(self):
        super().__init__(E.DIM, hidden_dim=6)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE
        self.b_up[0] = S
        self.W_gate[0, E.CARRY_IN] = 1.0
        self.W_down[PC_BASE, 0] = 1.0 / S
        self.b_up[1] = -S
        self.W_gate[1, E.CARRY_IN] = -1.0
        self.W_down[PC_BASE, 1] = 1.0 / S
        self.b_up[2] = S
        self.W_gate[2, E.CARRY_IN] = -1.0
        self.W_down[E.CARRY_IN, 2] = 1.0 / S
        self.b_up[3] = -S
        self.W_gate[3, E.CARRY_IN] = 1.0
        self.W_down[E.CARRY_IN, 3] = 1.0 / S
        self.b_up[4] = S
        self.W_gate[4, E.CARRY_OUT] = -1.0
        self.W_down[E.CARRY_OUT, 4] = 1.0 / S
        self.b_up[5] = -S
        self.W_gate[5, E.CARRY_OUT] = 1.0
        self.W_down[E.CARRY_OUT, 5] = 1.0 / S


# =============================================================================
# Build Function
# =============================================================================

def build_vm_step(num_carry_iters: int = 7) -> nn.Sequential:
    """Build complete VM step as nn.Sequential."""
    layers = []

    layers.append(ClearComputeSlots())

    BZ_BNZ_OPS = [Opcode.BZ, Opcode.BNZ]
    layers.append(MoE(
        [LoadAXtoNIBB() for _ in AX_TO_NIBB_OPS] +
        [LoadAXtoNIBA() for _ in AX_TO_NIBA_OPS] +
        [LoadBPtoTEMP() for _ in BP_TO_TEMP_OPS] +
        [LoadSPtoCARRYIN() for _ in SP_TO_CARRYIN_OPS] +
        [LoadAXtoRESULT() for _ in BZ_BNZ_OPS],
        AX_TO_NIBB_OPS + AX_TO_NIBA_OPS + BP_TO_TEMP_OPS +
        SP_TO_CARRYIN_OPS + BZ_BNZ_OPS,
    ))

    alu = build_pure_alu(num_carry_iters=num_carry_iters)
    for layer in alu:
        layers.append(layer)

    layers.append(MoE(
        [OverwriteAXFromResult() for _ in RESULT_TO_AX_OPS] +
        [WriteTEMPtoBP() for _ in ENT_OPS] +
        [WriteCARRYINtoSP() for _ in SP_FROM_CARRY_OPS],
        RESULT_TO_AX_OPS + ENT_OPS + SP_FROM_CARRY_OPS,
    ))

    layers.append(PCIncrementFFN())
    layers.append(PCOverflowClearFFN())
    layers.append(PCCarryPropagateFlatFFN())
    layers.append(PCCarryIterFFN())
    layers.append(PCOverflowFFN())
    layers.append(PCCarryPropagateFlatFFN())
    layers.append(PCCarryIterFFN())

    layers.append(MoE(
        [OverwritePCFromResult() for _ in RESULT_TO_PC_OPS],
        RESULT_TO_PC_OPS,
    ))

    return nn.Sequential(*layers)


# =============================================================================
# Helpers for embedding manipulation
# =============================================================================

def set_register(embedding: torch.Tensor, reg_base: int, value: int):
    for i in range(E.NUM_POSITIONS):
        nibble = (value >> (i * 4)) & 0xF
        embedding[0, i, reg_base] = float(nibble)


def get_register(embedding: torch.Tensor, reg_base: int) -> int:
    value = 0
    for i in range(E.NUM_POSITIONS):
        nib = int(round(embedding[0, i, reg_base].item()))
        nib = max(0, min(15, nib))
        value |= (nib << (i * 4))
    return value


def set_opcode(embedding: torch.Tensor, opcode: int):
    embedding[0, :, E.OP_START:E.OP_START + E.NUM_OPS] = 0.0
    embedding[0, :, E.OP_START + opcode] = 1.0


def set_immediate(embedding: torch.Tensor, value: int):
    for i in range(E.NUM_POSITIONS):
        nibble = (value >> (i * 4)) & 0xF
        embedding[0, i, E.NIB_B] = float(nibble)


def set_nib_a(embedding: torch.Tensor, value: int):
    for i in range(E.NUM_POSITIONS):
        nibble = (value >> (i * 4)) & 0xF
        embedding[0, i, E.NIB_A] = float(nibble)


def set_position_encoding(embedding: torch.Tensor):
    for i in range(E.NUM_POSITIONS):
        embedding[0, i, E.POS] = float(i)


def get_result(embedding: torch.Tensor) -> int:
    value = 0
    for i in range(E.NUM_POSITIONS):
        nib = int(round(embedding[0, i, E.RESULT].item()))
        nib = max(0, min(15, nib))
        value |= (nib << (i * 4))
    return value


def test_vm_step():
    """Quick smoke test of build_vm_step."""
    print("Building VM step...")
    step = build_vm_step(num_carry_iters=7)

    num_layers = len(step)
    print(f"VM step has {num_layers} layers")

    assert isinstance(step, nn.Sequential), "vm_step should be nn.Sequential"

    print("\nTest: IMM 42")
    emb = torch.zeros(1, E.NUM_POSITIONS, E.DIM)
    set_position_encoding(emb)
    set_opcode(emb, Opcode.IMM)
    set_immediate(emb, 42)
    set_nib_a(emb, 42)

    with torch.no_grad():
        out = step(emb)

    ax = get_register(out, AX_BASE)
    print(f"  AX = {ax} (expected 42)")

    result = get_result(out)
    print(f"  RESULT = {result}")

    print("\nVM step smoke test complete!")
    return step


if __name__ == "__main__":
    test_vm_step()
