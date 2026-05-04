"""
Fully autoregressive 8-bit Neural VM (Von Neumann architecture).

forward() is a pure sequence of tensor operations and SwiGLU FFN layers.
No Python for-loops, no Python if/else dispatch, no Python arithmetic on
extracted values. The model operates on a flat state vector containing
unified memory (256 bytes, also holds bytecode), registers, and flags.

Architecture (9 FFN layers + ALU + tensor ops):
  [tensor: derive NEXT_PC, SP_DEC, SP_INC from registers]
  L0: Clear + Fetch (MEM[PC]->OPCODE, MEM[PC+1]->IMM)
  L1: MemRead (MEM[SP], MEM[AX], MEM[BP], MEM[BP+1])
  L2a: Decode (opcode indicators + AX_IS_ZERO_NEG)
  L2b: Flags (AX_IS_ZERO, IS_ALU_OP, IS_CMP_OP, OP_DEFAULT_*)
  L2c: Cond (BZ_TAKEN, BNZ_TAKEN)
  L2d: PCFlags (PC_GO_IMM, PC_GO_NEXT)
  [tensor: nibble extract, opcode route, ALU forward, recombinE, flags]
  L3: Transition (AX, SP, PC, BP, output, mem write)
  L4: MemWrite
  [tensor: snap PC, SP, AX, BP to clean integers]

Supports all 29 opcodes from config.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .neural_alu import NeuralALU, E8, N, S
from .config import Op, INSTR_WIDTH, STACK_INIT


class FE:
    MEM = 0
    PC = 256; AX = 257; SP = 258; BP = 259
    OPCODE = 260; IMM = 261; STACK_TOP = 262
    CONST = 263
    ALU_RESULT = 264
    OP_IMM = 265; OP_PSH = 266
    OP_ADD = 267; OP_SUB = 268; OP_MUL = 269
    OP_AND = 270; OP_OR = 271; OP_XOR = 272
    OP_DIV = 273; OP_MOD = 274; OP_SHL = 275; OP_SHR = 276
    OP_PUTCHAR = 277; OP_EXIT = 278
    OP_JMP = 279; OP_BZ = 280; OP_BNZ = 281
    AX_IS_ZERO_NEG = 282; AX_IS_ZERO = 283; IS_ALU_OP = 284
    BZ_TAKEN = 285; BNZ_TAKEN = 286
    OP_DEFAULT_AX = 287; OP_DEFAULT_SP = 288; OP_DEFAULT_PC = 289
    PC_GO_IMM = 290; PC_GO_NEXT = 291
    NEXT_PC = 292; SP_DEC = 293; SP_INC = 294
    OUTPUT_CHAR = 295; HALT = 296
    MEM_WR_ADDR = 297; MEM_WR_VAL = 298
    OP_JSR = 299; OP_ENT = 300; OP_LEV = 301; OP_ADJ = 302
    OP_LI = 303; OP_SI = 304
    LI_VAL = 305
    LEV_NEW_BP = 306; LEV_NEW_PC = 307
    OP_EQ = 308; OP_NE = 309; OP_LT = 310; OP_GT = 311; OP_LE = 312; OP_GE = 313
    IS_CMP_OP = 314
    ALU_IS_ZERO = 315; CMP_LT = 316
    OP_GETCHAR = 317; INPUT_VAL = 318
    DIM = 319


_ALU_OPS = [FE.OP_ADD, FE.OP_SUB, FE.OP_MUL, FE.OP_AND, FE.OP_OR, FE.OP_XOR,
            FE.OP_DIV, FE.OP_MOD, FE.OP_SHL, FE.OP_SHR]
_CMP_OPS = [FE.OP_EQ, FE.OP_NE, FE.OP_LT, FE.OP_GT, FE.OP_LE, FE.OP_GE]
_AX_MOD_OPS = [FE.OP_IMM] + _ALU_OPS + [FE.OP_LI, FE.OP_GETCHAR] + _CMP_OPS
_SP_MOD_OPS = [FE.OP_PSH, FE.OP_JSR] + _ALU_OPS + _CMP_OPS + [FE.OP_ADJ, FE.OP_ENT, FE.OP_LEV, FE.OP_SI]
_PC_SPECIAL_OPS = [FE.OP_JMP, FE.OP_BZ, FE.OP_BNZ, FE.OP_EXIT, FE.OP_JSR, FE.OP_LEV]

_OPCODE_INDICATORS = [
    (Op.IMM, FE.OP_IMM), (Op.JMP, FE.OP_JMP), (Op.JSR, FE.OP_JSR),
    (Op.BZ, FE.OP_BZ), (Op.BNZ, FE.OP_BNZ),
    (Op.ENT, FE.OP_ENT), (Op.ADJ, FE.OP_ADJ), (Op.LEV, FE.OP_LEV),
    (Op.LI, FE.OP_LI), (Op.LC, FE.OP_LI), (Op.SI, FE.OP_SI), (Op.SC, FE.OP_SI),
    (Op.PSH, FE.OP_PSH),
    (Op.ADD, FE.OP_ADD), (Op.SUB, FE.OP_SUB), (Op.MUL, FE.OP_MUL),
    (Op.AND, FE.OP_AND), (Op.OR, FE.OP_OR), (Op.XOR, FE.OP_XOR),
    (Op.DIV, FE.OP_DIV), (Op.MOD, FE.OP_MOD), (Op.SHL, FE.OP_SHL), (Op.SHR, FE.OP_SHR),
    (Op.EQ, FE.OP_EQ), (Op.NE, FE.OP_NE), (Op.LT, FE.OP_LT),
    (Op.GT, FE.OP_GT), (Op.LE, FE.OP_LE), (Op.GE, FE.OP_GE),
    (Op.PUTCHAR, FE.OP_PUTCHAR), (Op.EXIT, FE.OP_EXIT),
    (Op.GETCHAR, FE.OP_GETCHAR),
]

_CLEAR_SLOTS = [
    FE.OPCODE, FE.IMM, FE.STACK_TOP, FE.ALU_RESULT,
    FE.AX_IS_ZERO_NEG, FE.AX_IS_ZERO, FE.IS_ALU_OP, FE.IS_CMP_OP,
    FE.BZ_TAKEN, FE.BNZ_TAKEN,
    FE.OP_DEFAULT_AX, FE.OP_DEFAULT_SP, FE.OP_DEFAULT_PC,
    FE.PC_GO_IMM, FE.PC_GO_NEXT,
    FE.OUTPUT_CHAR, FE.HALT, FE.MEM_WR_ADDR, FE.MEM_WR_VAL,
    FE.LI_VAL, FE.LEV_NEW_BP, FE.LEV_NEW_PC,
    FE.ALU_IS_ZERO, FE.CMP_LT,
]
_CLEAR_SLOTS += _ALU_OPS + _CMP_OPS + [
    FE.OP_IMM, FE.OP_PSH, FE.OP_JMP, FE.OP_EXIT,
    FE.OP_PUTCHAR, FE.OP_BZ, FE.OP_BNZ,
    FE.OP_JSR, FE.OP_ENT, FE.OP_LEV, FE.OP_ADJ,
    FE.OP_LI, FE.OP_SI, FE.OP_GETCHAR]


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


def _ind3(W_up, b_up, W_gate, W_down, h, sel_slot, target, gate_slot, dest, weight=1.0):
    W_up[h, sel_slot] = S; b_up[h] = -S * (target - 1)
    W_gate[h, gate_slot] = 1.0
    W_down[dest, h] = weight / S
    W_up[h+1, sel_slot] = S; b_up[h+1] = -S * target
    W_gate[h+1, gate_slot] = 1.0
    W_down[dest, h+1] = -2.0 * weight / S
    W_up[h+2, sel_slot] = S; b_up[h+2] = -S * (target + 1)
    W_gate[h+2, gate_slot] = 1.0
    W_down[dest, h+2] = weight / S
    return h + 3


def _clr(W_up, W_gate, W_down, h, slot):
    W_up[h, FE.CONST] = S; W_gate[h, slot] = -1.0; W_down[slot, h] = 1.0 / S
    W_up[h+1, FE.CONST] = -S; W_gate[h+1, slot] = 1.0; W_down[slot, h+1] = 1.0 / S
    return h + 2


def _cc(W_up, W_gate, W_down, h, up_slot, gate_slot, dest, weight):
    W_up[h, up_slot] = S; W_gate[h, gate_slot] = 1.0; W_down[dest, h] = weight / S
    W_up[h+1, up_slot] = -S; W_gate[h+1, gate_slot] = -1.0; W_down[dest, h+1] = weight / S
    return h + 2


class AutoregressiveVM(nn.Module):
    _ALU_OP_SLOTS = [FE.OP_ADD, FE.OP_SUB, FE.OP_MUL, FE.OP_AND, FE.OP_OR,
                     FE.OP_XOR, FE.OP_DIV, FE.OP_MOD, FE.OP_SHL, FE.OP_SHR]
    _ALU_E8_DIMS = [E8.OP_ADD, E8.OP_SUB, E8.OP_MUL, E8.OP_AND, E8.OP_OR,
                    E8.OP_XOR, E8.OP_DIV, E8.OP_MOD, E8.OP_SHL, E8.OP_SHR]
    _CMP_OP_SLOTS = [FE.OP_EQ, FE.OP_NE, FE.OP_LT, FE.OP_GT, FE.OP_LE, FE.OP_GE]
    _REG_SLOTS = [FE.PC, FE.SP, FE.AX, FE.BP]

    def __init__(self, compile_weights=True, snap_between_layers=False):
        super().__init__()
        self._snap = snap_between_layers
        self.alu = NeuralALU()
        self.register_buffer('_alu_op_idx', torch.tensor(self._ALU_OP_SLOTS))
        self.register_buffer('_alu_e8_idx', torch.tensor(self._ALU_E8_DIMS))
        self.register_buffer('_cmp_op_idx', torch.tensor(self._CMP_OP_SLOTS))
        self.register_buffer('_reg_idx', torch.tensor(self._REG_SLOTS))
        self.register_buffer('_snap_dims', torch.tensor(list(range(FE.PC)) + [FE.PC, FE.AX, FE.SP, FE.BP]))
        n_clear = len(_CLEAR_SLOTS) * 2
        n_fetch = 256 * 3 * 2
        self.l0_clear_fetch = FFN(FE.DIM, n_clear + n_fetch)
        self.l1_memread = FFN(FE.DIM, 256 * 3 * 4)
        self.l2a_decode = FFN(FE.DIM, len(_OPCODE_INDICATORS) * 3 + 2)
        n_flags = 4 + (len(_ALU_OPS) + len(_CMP_OPS)
                   + 2 * len(_AX_MOD_OPS) + 2 * len(_SP_MOD_OPS)
                   + 2 + len(_PC_SPECIAL_OPS)) * 2
        self.l2b_flags = FFN(FE.DIM, n_flags)
        self.l2c_cond = FFN(FE.DIM, 6)
        self.l2d_pcflags = FFN(FE.DIM, 18)
        n_l3 = 254
        self.l3_transition = FFN(FE.DIM, n_l3)
        self.l4_memwrite = FFN(FE.DIM, 256 * 3)
        if compile_weights:
            self._bake_all()

    def forward(self, state):
        state = state.clone()
        state[:, FE.NEXT_PC] = torch.remainder(state[:, FE.PC] + 2, 256)
        state[:, FE.SP_DEC] = torch.remainder(state[:, FE.SP] - 1, 256)
        state[:, FE.SP_INC] = torch.remainder(state[:, FE.SP] + 1, 256)

        state = self.l0_clear_fetch(state)
        if self._snap:
            state[:, self._snap_dims] = state[:, self._snap_dims].round().clamp(0, 255)
        state = self.l1_memread(state)
        if self._snap:
            state[:, self._snap_dims] = state[:, self._snap_dims].round().clamp(0, 255)
        state = self.l2a_decode(state)
        state = self.l2b_flags(state)
        state = self.l2c_cond(state)
        state = self.l2d_pcflags(state)
        state = self._run_alu(state)
        state = self.l3_transition(state)
        if self._snap:
            state[:, self._snap_dims] = state[:, self._snap_dims].round().clamp(0, 255)
        state = self.l4_memwrite(state)

        state[:, self._reg_idx] = state[:, self._reg_idx].round().clamp(0, 255)
        return state

    def _run_alu(self, state):
        b_val = state[:, FE.STACK_TOP].round().clamp(0, 255)
        ax_val = state[:, FE.AX].round().clamp(0, 255)

        alu_in = torch.zeros(state.shape[0], N, E8.DIM, device=state.device)
        alu_in[:, 0, E8.NIB_A] = torch.remainder(b_val, 16)
        alu_in[:, 1, E8.NIB_A] = b_val.div(16, rounding_mode='floor').clamp(max=15)
        alu_in[:, 0, E8.NIB_B] = torch.remainder(ax_val, 16)
        alu_in[:, 1, E8.NIB_B] = ax_val.div(16, rounding_mode='floor').clamp(max=15)

        flags = state[:, self._alu_op_idx].clamp(0, 1)
        alu_in[:, 0, self._alu_e8_idx] = flags
        alu_in[:, 1, self._alu_e8_idx] = flags

        cmp_flag = state[:, self._cmp_op_idx].clamp(0, 1).sum(dim=1).clamp(max=1)
        alu_in[:, 0, E8.OP_SUB] = torch.max(alu_in[:, 0, E8.OP_SUB], cmp_flag)
        alu_in[:, 1, E8.OP_SUB] = torch.max(alu_in[:, 1, E8.OP_SUB], cmp_flag)

        alu_out = self.alu(alu_in)
        alu_result = (alu_out[:, 0, E8.RESULT] + alu_out[:, 1, E8.RESULT] * 16).round().clamp(0, 255)

        state[:, FE.ALU_RESULT] = alu_result
        state[:, FE.ALU_IS_ZERO] = (alu_result == 0).float()
        state[:, FE.CMP_LT] = (b_val < ax_val).float()
        return state

    def _bake_all(self):
        with torch.no_grad():
            self._bake_l0()
            self._bake_l1()
            self._bake_l2a()
            self._bake_l2b()
            self._bake_l2c()
            self._bake_l2d()
            self._bake_l3()
            self._bake_l4()

    def _bake_l0(self):
        ffn = self.l0_clear_fetch
        W, b, G, D = ffn.W_up, ffn.b_up, ffn.W_gate, ffn.W_down
        h = 0
        for slot in _CLEAR_SLOTS:
            h = _clr(W, G, D, h, slot)
        for p in range(256):
            h = _ind3(W, b, G, D, h, FE.PC, p, FE.MEM + p, FE.OPCODE)
        for p in range(256):
            src = FE.MEM + ((p + 1) & 0xFF)
            h = _ind3(W, b, G, D, h, FE.PC, p, src, FE.IMM)

    def _bake_l1(self):
        ffn = self.l1_memread
        W, b, G, D = ffn.W_up, ffn.b_up, ffn.W_gate, ffn.W_down
        h = 0
        for a in range(256):
            h = _ind3(W, b, G, D, h, FE.SP, a, FE.MEM + a, FE.STACK_TOP)
        for a in range(256):
            h = _ind3(W, b, G, D, h, FE.AX, a, FE.MEM + a, FE.LI_VAL)
        for a in range(256):
            h = _ind3(W, b, G, D, h, FE.BP, a, FE.MEM + a, FE.LEV_NEW_BP)
        for a in range(256):
            h = _ind3(W, b, G, D, h, FE.BP, a, FE.MEM + ((a + 1) & 0xFF), FE.LEV_NEW_PC)

    def _bake_l2a(self):
        ffn = self.l2a_decode
        W, G, D = ffn.W_up, ffn.W_gate, ffn.W_down
        b = ffn.b_up
        h = 0
        for op_val, fe_slot in _OPCODE_INDICATORS:
            W[h, FE.OPCODE] = S; b[h] = -S * (op_val - 1); G[h, FE.CONST] = 1.0; D[fe_slot, h] = 1.0 / S; h += 1
            W[h, FE.OPCODE] = S; b[h] = -S * op_val; G[h, FE.CONST] = 1.0; D[fe_slot, h] = -2.0 / S; h += 1
            W[h, FE.OPCODE] = S; b[h] = -S * (op_val + 1); G[h, FE.CONST] = 1.0; D[fe_slot, h] = 1.0 / S; h += 1
        W[h, FE.AX] = S; G[h, FE.CONST] = 1.0; D[FE.AX_IS_ZERO_NEG, h] = 1.0 / S; h += 1
        W[h, FE.AX] = S; b[h] = -S; G[h, FE.CONST] = 1.0; D[FE.AX_IS_ZERO_NEG, h] = -1.0 / S; h += 1

    def _bake_l2b(self):
        ffn = self.l2b_flags
        W, G, D = ffn.W_up, ffn.W_gate, ffn.W_down
        h = 0
        h = _cc(W, G, D, h, FE.CONST, FE.CONST, FE.AX_IS_ZERO, 1.0)
        h = _cc(W, G, D, h, FE.AX_IS_ZERO_NEG, FE.AX_IS_ZERO_NEG, FE.AX_IS_ZERO, -1.0)
        for op in _ALU_OPS:
            h = _cc(W, G, D, h, op, op, FE.IS_ALU_OP, 1.0)
        for op in _CMP_OPS:
            h = _cc(W, G, D, h, op, op, FE.IS_CMP_OP, 1.0)
        h = _cc(W, G, D, h, FE.CONST, FE.CONST, FE.OP_DEFAULT_AX, 1.0)
        for op in _AX_MOD_OPS:
            h = _cc(W, G, D, h, op, op, FE.OP_DEFAULT_AX, -1.0)
        h = _cc(W, G, D, h, FE.CONST, FE.CONST, FE.OP_DEFAULT_SP, 1.0)
        for op in _SP_MOD_OPS:
            h = _cc(W, G, D, h, op, op, FE.OP_DEFAULT_SP, -1.0)
        h = _cc(W, G, D, h, FE.CONST, FE.CONST, FE.OP_DEFAULT_PC, 1.0)
        for op in _PC_SPECIAL_OPS:
            h = _cc(W, G, D, h, op, op, FE.OP_DEFAULT_PC, -1.0)

    def _bake_l2c(self):
        ffn = self.l2c_cond
        W, G, D = ffn.W_up, ffn.W_gate, ffn.W_down
        h = 0
        h = _cc(W, G, D, h, FE.OP_BZ, FE.AX_IS_ZERO, FE.BZ_TAKEN, 1.0)
        h = _cc(W, G, D, h, FE.OP_BNZ, FE.OP_BNZ, FE.BNZ_TAKEN, 1.0)
        h = _cc(W, G, D, h, FE.OP_BNZ, FE.AX_IS_ZERO, FE.BNZ_TAKEN, -1.0)

    def _bake_l2d(self):
        ffn = self.l2d_pcflags
        W, G, D = ffn.W_up, ffn.W_gate, ffn.W_down
        h = 0
        h = _cc(W, G, D, h, FE.OP_JMP, FE.OP_JMP, FE.PC_GO_IMM, 1.0)
        h = _cc(W, G, D, h, FE.OP_JSR, FE.OP_JSR, FE.PC_GO_IMM, 1.0)
        h = _cc(W, G, D, h, FE.BZ_TAKEN, FE.BZ_TAKEN, FE.PC_GO_IMM, 1.0)
        h = _cc(W, G, D, h, FE.BNZ_TAKEN, FE.BNZ_TAKEN, FE.PC_GO_IMM, 1.0)
        h = _cc(W, G, D, h, FE.OP_BZ, FE.OP_BZ, FE.PC_GO_NEXT, 1.0)
        h = _cc(W, G, D, h, FE.BZ_TAKEN, FE.BZ_TAKEN, FE.PC_GO_NEXT, -1.0)
        h = _cc(W, G, D, h, FE.OP_BNZ, FE.OP_BNZ, FE.PC_GO_NEXT, 1.0)
        h = _cc(W, G, D, h, FE.BNZ_TAKEN, FE.BNZ_TAKEN, FE.PC_GO_NEXT, -1.0)
        h = _cc(W, G, D, h, FE.OP_DEFAULT_PC, FE.OP_DEFAULT_PC, FE.PC_GO_NEXT, 1.0)

    def _bake_l3(self):
        ffn = self.l3_transition
        W, G, D = ffn.W_up, ffn.W_gate, ffn.W_down
        h = 0

        for op, val in ([(FE.OP_IMM, FE.IMM)]
                        + [(o, FE.ALU_RESULT) for o in _ALU_OPS]
                        + [(FE.OP_LI, FE.LI_VAL)]
                        + [(FE.OP_GETCHAR, FE.INPUT_VAL)]):
            h = _cc(W, G, D, h, op, val, FE.AX, 1.0)
            h = _cc(W, G, D, h, op, FE.AX, FE.AX, -1.0)

        h = _cc(W, G, D, h, FE.OP_EQ, FE.ALU_IS_ZERO, FE.AX, 1.0)
        h = _cc(W, G, D, h, FE.OP_EQ, FE.AX, FE.AX, -1.0)

        h = _cc(W, G, D, h, FE.OP_NE, FE.CONST, FE.AX, 1.0)
        h = _cc(W, G, D, h, FE.OP_NE, FE.ALU_IS_ZERO, FE.AX, -1.0)
        h = _cc(W, G, D, h, FE.OP_NE, FE.AX, FE.AX, -1.0)

        h = _cc(W, G, D, h, FE.OP_LT, FE.CMP_LT, FE.AX, 1.0)
        h = _cc(W, G, D, h, FE.OP_LT, FE.AX, FE.AX, -1.0)

        h = _cc(W, G, D, h, FE.OP_GT, FE.CONST, FE.AX, 1.0)
        h = _cc(W, G, D, h, FE.OP_GT, FE.CMP_LT, FE.AX, -1.0)
        h = _cc(W, G, D, h, FE.OP_GT, FE.ALU_IS_ZERO, FE.AX, -1.0)
        h = _cc(W, G, D, h, FE.OP_GT, FE.AX, FE.AX, -1.0)

        h = _cc(W, G, D, h, FE.OP_LE, FE.CMP_LT, FE.AX, 1.0)
        h = _cc(W, G, D, h, FE.OP_LE, FE.ALU_IS_ZERO, FE.AX, 1.0)
        h = _cc(W, G, D, h, FE.OP_LE, FE.AX, FE.AX, -1.0)

        h = _cc(W, G, D, h, FE.OP_GE, FE.CONST, FE.AX, 1.0)
        h = _cc(W, G, D, h, FE.OP_GE, FE.CMP_LT, FE.AX, -1.0)
        h = _cc(W, G, D, h, FE.OP_GE, FE.AX, FE.AX, -1.0)

        for op, val in [(FE.OP_PSH, FE.SP_DEC), (FE.OP_JSR, FE.SP_DEC)] + [(o, FE.SP_INC) for o in _ALU_OPS + _CMP_OPS]:
            h = _cc(W, G, D, h, op, val, FE.SP, 1.0)
            h = _cc(W, G, D, h, op, FE.SP, FE.SP, -1.0)
        h = _cc(W, G, D, h, FE.OP_ADJ, FE.IMM, FE.SP, 1.0)
        h = _cc(W, G, D, h, FE.OP_ENT, FE.SP_DEC, FE.SP, 1.0)
        h = _cc(W, G, D, h, FE.OP_ENT, FE.SP, FE.SP, -1.0)
        h = _cc(W, G, D, h, FE.OP_ENT, FE.IMM, FE.SP, -1.0)
        h = _cc(W, G, D, h, FE.OP_LEV, FE.BP, FE.SP, 1.0)
        h = _cc(W, G, D, h, FE.OP_LEV, FE.SP, FE.SP, -1.0)
        h = _cc(W, G, D, h, FE.OP_LEV, FE.CONST, FE.SP, 2.0)
        h = _cc(W, G, D, h, FE.OP_SI, FE.SP_INC, FE.SP, 1.0)
        h = _cc(W, G, D, h, FE.OP_SI, FE.SP, FE.SP, -1.0)

        h = _cc(W, G, D, h, FE.OP_ENT, FE.SP_DEC, FE.BP, 1.0)
        h = _cc(W, G, D, h, FE.OP_ENT, FE.BP, FE.BP, -1.0)
        h = _cc(W, G, D, h, FE.OP_LEV, FE.LEV_NEW_BP, FE.BP, 1.0)
        h = _cc(W, G, D, h, FE.OP_LEV, FE.BP, FE.BP, -1.0)

        h = _cc(W, G, D, h, FE.PC_GO_IMM, FE.IMM, FE.PC, 1.0)
        h = _cc(W, G, D, h, FE.PC_GO_IMM, FE.PC, FE.PC, -1.0)
        h = _cc(W, G, D, h, FE.PC_GO_NEXT, FE.NEXT_PC, FE.PC, 1.0)
        h = _cc(W, G, D, h, FE.PC_GO_NEXT, FE.PC, FE.PC, -1.0)
        h = _cc(W, G, D, h, FE.OP_LEV, FE.LEV_NEW_PC, FE.PC, 1.0)
        h = _cc(W, G, D, h, FE.OP_LEV, FE.PC, FE.PC, -1.0)

        h = _cc(W, G, D, h, FE.OP_PUTCHAR, FE.AX, FE.OUTPUT_CHAR, 1.0)
        h = _cc(W, G, D, h, FE.OP_EXIT, FE.CONST, FE.HALT, 1.0)

        h = _cc(W, G, D, h, FE.CONST, FE.CONST, FE.MEM_WR_ADDR, 256.0)
        h = _cc(W, G, D, h, FE.OP_PSH, FE.SP_DEC, FE.MEM_WR_ADDR, 1.0)
        h = _cc(W, G, D, h, FE.OP_PSH, FE.OP_PSH, FE.MEM_WR_ADDR, -256.0)
        h = _cc(W, G, D, h, FE.OP_JSR, FE.SP_DEC, FE.MEM_WR_ADDR, 1.0)
        h = _cc(W, G, D, h, FE.OP_JSR, FE.OP_JSR, FE.MEM_WR_ADDR, -256.0)
        h = _cc(W, G, D, h, FE.OP_ENT, FE.SP_DEC, FE.MEM_WR_ADDR, 1.0)
        h = _cc(W, G, D, h, FE.OP_ENT, FE.OP_ENT, FE.MEM_WR_ADDR, -256.0)
        h = _cc(W, G, D, h, FE.OP_SI, FE.STACK_TOP, FE.MEM_WR_ADDR, 1.0)
        h = _cc(W, G, D, h, FE.OP_SI, FE.OP_SI, FE.MEM_WR_ADDR, -256.0)

        h = _cc(W, G, D, h, FE.OP_PSH, FE.AX, FE.MEM_WR_VAL, 1.0)
        h = _cc(W, G, D, h, FE.OP_JSR, FE.NEXT_PC, FE.MEM_WR_VAL, 1.0)
        h = _cc(W, G, D, h, FE.OP_ENT, FE.BP, FE.MEM_WR_VAL, 1.0)
        h = _cc(W, G, D, h, FE.OP_SI, FE.AX, FE.MEM_WR_VAL, 1.0)

    def _bake_l4(self):
        ffn = self.l4_memwrite
        W, b, G, D = ffn.W_up, ffn.b_up, ffn.W_gate, ffn.W_down
        h = 0
        for a in range(256):
            W[h, FE.MEM_WR_ADDR] = S; b[h] = -S * (a - 1)
            G[h, FE.MEM_WR_VAL] = 1.0; G[h, FE.MEM + a] = -1.0
            D[FE.MEM + a, h] = 1.0 / S; h += 1
            W[h, FE.MEM_WR_ADDR] = S; b[h] = -S * a
            G[h, FE.MEM_WR_VAL] = 1.0; G[h, FE.MEM + a] = -1.0
            D[FE.MEM + a, h] = -2.0 / S; h += 1
            W[h, FE.MEM_WR_ADDR] = S; b[h] = -S * (a + 1)
            G[h, FE.MEM_WR_VAL] = 1.0; G[h, FE.MEM + a] = -1.0
            D[FE.MEM + a, h] = 1.0 / S; h += 1


class AutoregressiveRunner:
    def __init__(self):
        self.model = AutoregressiveVM()
        self.model.eval()
        self._input_buf = []

    def load(self, bytecode):
        self.bytecode = bytecode

    def set_input(self, chars):
        self._input_buf = list(chars)

    def run(self, max_steps=1000):
        state = torch.zeros(1, FE.DIM)
        for i, b in enumerate(self.bytecode):
            state[0, FE.MEM + i] = float(b & 0xFF)
        state[0, FE.SP] = float(STACK_INIT)
        state[0, FE.BP] = float(STACK_INIT)
        state[0, FE.CONST] = 1.0
        output = []

        for _ in range(max_steps):
            state[0, FE.INPUT_VAL] = float(ord(self._input_buf[0]) & 0xFF) if self._input_buf else 255.0

            with torch.no_grad():
                state = self.model(state)

            if state[0, FE.OP_PUTCHAR].clamp(0, 1).round().item() > 0.5:
                output.append(chr(int(round(state[0, FE.OUTPUT_CHAR].item())) & 0xFF))

            if state[0, FE.OP_GETCHAR].clamp(0, 1).round().item() > 0.5 and self._input_buf:
                self._input_buf.pop(0)

            if state[0, FE.HALT].item() > 0.5:
                break

        return "".join(output), 0
