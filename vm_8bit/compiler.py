"""
Compiler for the 8-bit Neural VM.

Separates weight-programming from the model definition. Usage:

    from vm_8bit.compiler import VMCompiler
    from vm_8bit.autoregressive_vm import AutoregressiveVM

    model = AutoregressiveVM()
    VMCompiler.compile(model)

The compiler sets all FFN weights so that forward() implements the full
8-bit ISA (29 opcodes) through pure SwiGLU tensor operations.
"""

import torch
import torch.nn as nn

from .neural_alu import NeuralALU, E8, N, S
from .autoregressive_vm import (
    FE, FFN, _CLEAR_SLOTS, _OPCODE_INDICATORS,
    _ALU_OPS, _CMP_OPS, _AX_MOD_OPS, _SP_MOD_OPS, _PC_SPECIAL_OPS,
    _ind3, _clr, _cc,
)


class VMCompiler:
    """Compiles the 8-bit ISA into AutoregressiveVM weights."""

    @staticmethod
    def compile(model: nn.Module) -> None:
        with torch.no_grad():
            VMCompiler._compile_l0(model)
            VMCompiler._compile_l1(model)
            VMCompiler._compile_l2a(model)
            VMCompiler._compile_l2b(model)
            VMCompiler._compile_l2c(model)
            VMCompiler._compile_l2d(model)
            VMCompiler._compile_l3(model)
            VMCompiler._compile_l4(model)

    @staticmethod
    def verify(model_inline: nn.Module, model_compiled: nn.Module) -> bool:
        for (n1, p1), (n2, p2) in zip(
                model_inline.named_parameters(), model_compiled.named_parameters()):
            if n1 != n2:
                print(f"Name mismatch: {n1} vs {n2}")
                return False
            if not torch.equal(p1, p2):
                diff = (p1 - p2).abs()
                print(f"Weight mismatch in {n1}: max_diff={diff.max().item():.6e}")
                return False
        for (n1, b1), (n2, b2) in zip(
                model_inline.named_buffers(), model_compiled.named_buffers()):
            if n1 != n2:
                print(f"Buffer name mismatch: {n1} vs {n2}")
                return False
            if not torch.equal(b1, b2):
                print(f"Buffer mismatch in {n1}")
                return False
        return True

    @staticmethod
    def _compile_l0(model):
        ffn = model.l0_clear_fetch
        W, b, G, D = ffn.W_up, ffn.b_up, ffn.W_gate, ffn.W_down
        h = 0
        for slot in _CLEAR_SLOTS:
            h = _clr(W, G, D, h, slot)
        for p in range(256):
            h = _ind3(W, b, G, D, h, FE.PC, p, FE.MEM + p, FE.OPCODE)
        for p in range(256):
            src = FE.MEM + ((p + 1) & 0xFF)
            h = _ind3(W, b, G, D, h, FE.PC, p, src, FE.IMM)

    @staticmethod
    def _compile_l1(model):
        ffn = model.l1_memread
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

    @staticmethod
    def _compile_l2a(model):
        ffn = model.l2a_decode
        W, G, D = ffn.W_up, ffn.W_gate, ffn.W_down
        b = ffn.b_up
        h = 0
        for op_val, fe_slot in _OPCODE_INDICATORS:
            W[h, FE.OPCODE] = S; b[h] = -S * (op_val - 1); G[h, FE.CONST] = 1.0; D[fe_slot, h] = 1.0 / S; h += 1
            W[h, FE.OPCODE] = S; b[h] = -S * op_val; G[h, FE.CONST] = 1.0; D[fe_slot, h] = -2.0 / S; h += 1
            W[h, FE.OPCODE] = S; b[h] = -S * (op_val + 1); G[h, FE.CONST] = 1.0; D[fe_slot, h] = 1.0 / S; h += 1
        W[h, FE.AX] = S; G[h, FE.CONST] = 1.0; D[FE.AX_IS_ZERO_NEG, h] = 1.0 / S; h += 1
        W[h, FE.AX] = S; b[h] = -S; G[h, FE.CONST] = 1.0; D[FE.AX_IS_ZERO_NEG, h] = -1.0 / S; h += 1

    @staticmethod
    def _compile_l2b(model):
        ffn = model.l2b_flags
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

    @staticmethod
    def _compile_l2c(model):
        ffn = model.l2c_cond
        W, G, D = ffn.W_up, ffn.W_gate, ffn.W_down
        h = 0
        h = _cc(W, G, D, h, FE.OP_BZ, FE.AX_IS_ZERO, FE.BZ_TAKEN, 1.0)
        h = _cc(W, G, D, h, FE.OP_BNZ, FE.OP_BNZ, FE.BNZ_TAKEN, 1.0)
        h = _cc(W, G, D, h, FE.OP_BNZ, FE.AX_IS_ZERO, FE.BNZ_TAKEN, -1.0)

    @staticmethod
    def _compile_l2d(model):
        ffn = model.l2d_pcflags
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

    @staticmethod
    def _compile_l3(model):
        ffn = model.l3_transition
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

    @staticmethod
    def _compile_l4(model):
        ffn = model.l4_memwrite
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
