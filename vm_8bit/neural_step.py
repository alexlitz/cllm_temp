"""
Neural state transition module for the 8-bit VM.

Takes a flat state vector (registers, opcode flags, ALU result) and computes
the next VM state entirely through baked SwiGLU FFN layers. No Python
branching in forward().

Layer ordering respects data dependencies (each layer reads previous layer's output):
  L0: CondDetect   — AX_IS_ZERO_NEG = step(AX_RAW >= 1)
  L1: PrimaryFlags — AX_IS_ZERO, IS_ALU_OP, OP_DEFAULT_*
  L2: CondFlags    — BZ_TAKEN, BNZ_TAKEN (depend on AX_IS_ZERO from L1)
  L3: PcFlags      — PC_GO_IMM, PC_GO_NEXT (depend on L2 + L1)
  L4: AxTransition — new_AX = OP_IMM*IMM + IS_ALU_OP*ALU_RESULT + default*AX
  L5: SpTransition — new_SP = OP_PSH*SP-1 + IS_ALU_OP*SP+1 + default*SP
  L6: PcTransition — new_PC = PC_GO_IMM*IMM + PC_GO_NEXT*NEXT_PC (residual)
  L7: OutputRoute  — OUTPUT_CHAR = OP_PUTCHAR*AX, HALT = OP_EXIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .neural_alu import NeuralALU, E8, N, S


class SEmb:
    PC = 0; AX = 1; AX_RAW = 2; SP = 3; BP = 4
    IMM = 5; STACK_TOP = 6; NEXT_PC = 7; SP_DEC = 8; SP_INC = 9
    CONST = 10; ALU_RESULT = 11
    NEW_PC = 12; NEW_AX = 13; NEW_SP = 14; NEW_BP = 15
    OUTPUT_CHAR = 16; HALT = 17; DO_PUSH = 18
    OP_IMM = 19; OP_PSH = 20
    OP_ADD = 21; OP_SUB = 22; OP_MUL = 23
    OP_AND = 24; OP_OR = 25; OP_XOR = 26
    OP_DIV = 27; OP_MOD = 28; OP_SHL = 29; OP_SHR = 30
    OP_PUTCHAR = 31; OP_EXIT = 32
    OP_JMP = 33; OP_BZ = 34; OP_BNZ = 35
    OP_JSR = 36; OP_LEV = 37; OP_ENT = 38; OP_ADJ = 39
    OP_LI = 40; OP_SI = 41; OP_EQ = 42; OP_NE = 43; OP_LT = 44; OP_GT = 45
    AX_IS_ZERO_NEG = 46; AX_IS_ZERO = 47; IS_ALU_OP = 48
    BZ_TAKEN = 49; BNZ_TAKEN = 50
    OP_DEFAULT_AX = 51; OP_DEFAULT_SP = 52; OP_DEFAULT_PC = 53
    PC_GO_IMM = 54; PC_GO_NEXT = 55
    DIM = 56


_ALU_OPS = [SEmb.OP_ADD, SEmb.OP_SUB, SEmb.OP_MUL,
            SEmb.OP_AND, SEmb.OP_OR, SEmb.OP_XOR,
            SEmb.OP_DIV, SEmb.OP_MOD, SEmb.OP_SHL, SEmb.OP_SHR]
_AX_MOD_OPS = [SEmb.OP_IMM] + _ALU_OPS
_SP_MOD_OPS = [SEmb.OP_PSH] + _ALU_OPS
_PC_SPECIAL_OPS = [SEmb.OP_JMP, SEmb.OP_BZ, SEmb.OP_BNZ,
                   SEmb.OP_EXIT, SEmb.OP_JSR, SEmb.OP_LEV]


def _cc(W, G, D, h, up_slot, gate_slot, out_slot, weight):
    W[h, up_slot] = S; G[h, gate_slot] = 1.0; D[out_slot, h] = weight / S
    W[h+1, up_slot] = -S; G[h+1, gate_slot] = -1.0; D[out_slot, h+1] = weight / S


class FFN(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.W_up = nn.Parameter(torch.zeros(hidden_dim, dim))
        self.b_up = nn.Parameter(torch.zeros(hidden_dim))
        self.W_gate = nn.Parameter(torch.zeros(hidden_dim, dim))
        self.b_gate = nn.Parameter(torch.zeros(hidden_dim))
        self.W_down = nn.Parameter(torch.zeros(dim, hidden_dim))
        self.b_down = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        up = F.linear(x, self.W_up, self.b_up)
        gate = F.linear(x, self.W_gate, self.b_gate)
        return x + F.linear(F.silu(up) * gate, self.W_down, self.b_down)


class NeuralStep(nn.Module):
    def __init__(self):
        super().__init__()
        self.alu = NeuralALU()
        D = SEmb.DIM
        self.l0_cond = FFN(D, 2)
        self.l1_primary = FFN(D, 4 + len(_ALU_OPS)*2 + 2 + 2*len(_AX_MOD_OPS) + 2 + 2*len(_SP_MOD_OPS) + 2 + len(_PC_SPECIAL_OPS)*2)
        self.l2_cond = FFN(D, 2 + 4)
        self.l3_pc_flags = FFN(D, 3*2 + 5*2)
        self.l4_ax = FFN(D, len(_AX_MOD_OPS)*4)
        self.l5_sp = FFN(D, len(_SP_MOD_OPS)*4)
        self.l6_pc = FFN(D, 2*4)
        self.l7_out = FFN(D, 4)
        self._bake_all()

    def forward(self, state, alu_input=None):
        if alu_input is not None:
            with torch.no_grad():
                alu_out = self.alu(alu_input)
            lo = alu_out[:, 0, E8.RESULT]
            hi = alu_out[:, 1, E8.RESULT]
            state = state.clone()
            state[:, SEmb.ALU_RESULT] = (lo + hi * 16.0) / 255.0

        state = self.l0_cond(state)
        state = self.l1_primary(state)
        state = self.l2_cond(state)
        state = self.l3_pc_flags(state)
        state = self.l4_ax(state)
        state = self.l5_sp(state)
        state = self.l6_pc(state)
        state = self.l7_out(state)
        return state

    def _bake_all(self):
        with torch.no_grad():
            self._bake_l0()
            self._bake_l1()
            self._bake_l2()
            self._bake_l3()
            self._bake_l4()
            self._bake_l5()
            self._bake_l6()
            self._bake_l7()

    def _bake_l0(self):
        W, b, G, D = self.l0_cond.W_up, self.l0_cond.b_up, self.l0_cond.W_gate, self.l0_cond.W_down
        h = 0
        W[h, SEmb.AX_RAW] = S; b[h] = 0; G[h, SEmb.CONST] = 1.0; D[SEmb.AX_IS_ZERO_NEG, h] = 1.0/S; h += 1
        W[h, SEmb.AX_RAW] = S; b[h] = -S; G[h, SEmb.CONST] = 1.0; D[SEmb.AX_IS_ZERO_NEG, h] = -1.0/S; h += 1

    def _bake_l1(self):
        W, G, D = self.l1_primary.W_up, self.l1_primary.W_gate, self.l1_primary.W_down
        h = 0
        # AX_IS_ZERO = CONST - AX_IS_ZERO_NEG
        _cc(W, G, D, h, SEmb.CONST, SEmb.CONST, SEmb.AX_IS_ZERO, 1.0); h += 2
        _cc(W, G, D, h, SEmb.AX_IS_ZERO_NEG, SEmb.AX_IS_ZERO_NEG, SEmb.AX_IS_ZERO, -1.0); h += 2
        # IS_ALU_OP
        for op in _ALU_OPS:
            _cc(W, G, D, h, op, op, SEmb.IS_ALU_OP, 1.0); h += 2
        # OP_DEFAULT_AX
        _cc(W, G, D, h, SEmb.CONST, SEmb.CONST, SEmb.OP_DEFAULT_AX, 1.0); h += 2
        for op in _AX_MOD_OPS:
            _cc(W, G, D, h, op, op, SEmb.OP_DEFAULT_AX, -1.0); h += 2
        # OP_DEFAULT_SP
        _cc(W, G, D, h, SEmb.CONST, SEmb.CONST, SEmb.OP_DEFAULT_SP, 1.0); h += 2
        for op in _SP_MOD_OPS:
            _cc(W, G, D, h, op, op, SEmb.OP_DEFAULT_SP, -1.0); h += 2
        # OP_DEFAULT_PC
        _cc(W, G, D, h, SEmb.CONST, SEmb.CONST, SEmb.OP_DEFAULT_PC, 1.0); h += 2
        for op in _PC_SPECIAL_OPS:
            _cc(W, G, D, h, op, op, SEmb.OP_DEFAULT_PC, -1.0); h += 2

    def _bake_l2(self):
        W, G, D = self.l2_cond.W_up, self.l2_cond.W_gate, self.l2_cond.W_down
        h = 0
        # BZ_TAKEN = OP_BZ * AX_IS_ZERO
        _cc(W, G, D, h, SEmb.OP_BZ, SEmb.AX_IS_ZERO, SEmb.BZ_TAKEN, 1.0); h += 2
        # BNZ_TAKEN = OP_BNZ - OP_BNZ*AX_IS_ZERO
        _cc(W, G, D, h, SEmb.OP_BNZ, SEmb.OP_BNZ, SEmb.BNZ_TAKEN, 1.0); h += 2
        _cc(W, G, D, h, SEmb.OP_BNZ, SEmb.AX_IS_ZERO, SEmb.BNZ_TAKEN, -1.0); h += 2

    def _bake_l3(self):
        W, G, D = self.l3_pc_flags.W_up, self.l3_pc_flags.W_gate, self.l3_pc_flags.W_down
        h = 0
        # PC_GO_IMM = OP_JMP + BZ_TAKEN + BNZ_TAKEN
        _cc(W, G, D, h, SEmb.OP_JMP, SEmb.OP_JMP, SEmb.PC_GO_IMM, 1.0); h += 2
        _cc(W, G, D, h, SEmb.BZ_TAKEN, SEmb.BZ_TAKEN, SEmb.PC_GO_IMM, 1.0); h += 2
        _cc(W, G, D, h, SEmb.BNZ_TAKEN, SEmb.BNZ_TAKEN, SEmb.PC_GO_IMM, 1.0); h += 2
        # PC_GO_NEXT = (OP_BZ - BZ_TAKEN) + (OP_BNZ - BNZ_TAKEN) + OP_DEFAULT_PC
        _cc(W, G, D, h, SEmb.OP_BZ, SEmb.OP_BZ, SEmb.PC_GO_NEXT, 1.0); h += 2
        _cc(W, G, D, h, SEmb.BZ_TAKEN, SEmb.BZ_TAKEN, SEmb.PC_GO_NEXT, -1.0); h += 2
        _cc(W, G, D, h, SEmb.OP_BNZ, SEmb.OP_BNZ, SEmb.PC_GO_NEXT, 1.0); h += 2
        _cc(W, G, D, h, SEmb.BNZ_TAKEN, SEmb.BNZ_TAKEN, SEmb.PC_GO_NEXT, -1.0); h += 2
        _cc(W, G, D, h, SEmb.OP_DEFAULT_PC, SEmb.OP_DEFAULT_PC, SEmb.PC_GO_NEXT, 1.0); h += 2

    def _bake_l4(self):
        W, G, D = self.l4_ax.W_up, self.l4_ax.W_gate, self.l4_ax.W_down
        h = 0
        for op, val in [(SEmb.OP_IMM, SEmb.IMM)] + [(o, SEmb.ALU_RESULT) for o in _ALU_OPS]:
            _cc(W, G, D, h, op, val, SEmb.AX, 1.0); h += 2
            _cc(W, G, D, h, op, SEmb.AX, SEmb.AX, -1.0); h += 2

    def _bake_l5(self):
        W, G, D = self.l5_sp.W_up, self.l5_sp.W_gate, self.l5_sp.W_down
        h = 0
        for op, val in [(SEmb.OP_PSH, SEmb.SP_DEC)] + [(o, SEmb.SP_INC) for o in _ALU_OPS]:
            _cc(W, G, D, h, op, val, SEmb.SP, 1.0); h += 2
            _cc(W, G, D, h, op, SEmb.SP, SEmb.SP, -1.0); h += 2

    def _bake_l6(self):
        W, G, D = self.l6_pc.W_up, self.l6_pc.W_gate, self.l6_pc.W_down
        h = 0
        _cc(W, G, D, h, SEmb.PC_GO_IMM, SEmb.IMM, SEmb.PC, 1.0); h += 2
        _cc(W, G, D, h, SEmb.PC_GO_IMM, SEmb.PC, SEmb.PC, -1.0); h += 2
        _cc(W, G, D, h, SEmb.PC_GO_NEXT, SEmb.NEXT_PC, SEmb.PC, 1.0); h += 2
        _cc(W, G, D, h, SEmb.PC_GO_NEXT, SEmb.PC, SEmb.PC, -1.0); h += 2

    def _bake_l7(self):
        W, G, D = self.l7_out.W_up, self.l7_out.W_gate, self.l7_out.W_down
        h = 0
        _cc(W, G, D, h, SEmb.OP_PUTCHAR, SEmb.AX, SEmb.OUTPUT_CHAR, 1.0); h += 2
        _cc(W, G, D, h, SEmb.OP_EXIT, SEmb.CONST, SEmb.HALT, 1.0); h += 2
