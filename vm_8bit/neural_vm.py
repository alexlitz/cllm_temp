"""
8-bit Autoregressive Neural VM.

A decoder-only transformer with hand-baked weights that executes an 8-bit
stack-based VM. Every computation (instruction fetch, ALU, memory, PC update)
flows through standard transformer forward passes. No Python arithmetic in
forward.

Architecture:
  d_model=64, 10 layers, 4 heads, FFN hidden 128
  Layers 0-1: embed + position, instruction fetch via attention
  Layers 2-3: opcode decode, operand gather
  Layers 4-7: ALU (SwiGLU baked weights, reuse patterns from neural_alu.py)
  Layers 8-9: output routing, token logits

Token format per step (14 tokens):
  REG_PC  + 1 value byte  (2)
  REG_AX  + 1 value byte  (2)
  REG_SP  + 1 value byte  (2)
  REG_BP  + 1 value byte  (2)
  STACK0  + 1 value byte  (2)
  MEM     + 1 addr + 1 val (3)
  STEP_END                 (1)

Instruction format: 2 bytes [opcode, immediate] at address PC/2 * 2.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

from .config import Op, Tok, INSTR_WIDTH, VALUE_MASK, STACK_INIT
from .neural_alu import NeuralALU, E8, S, BASE, N


D_MODEL = 64
N_HEADS = 4
HEAD_DIM = D_MODEL // N_HEADS
FFN_HIDDEN = 128
N_LAYERS = 10
TOKENS_PER_STEP = 14


# ── Embedding layout within d_model ────────────────────────────────────

class Emb:
    BYTE_VAL = 0
    IS_REG_PC = 1
    IS_REG_AX = 2
    IS_REG_SP = 3
    IS_REG_BP = 4
    IS_STACK0 = 5
    IS_MEM = 6
    IS_STEP_END = 7
    IS_CODE = 8
    IS_DATA = 9
    OPCODE = 10
    NIB_LO = 11
    NIB_HI = 12
    IMM_VAL = 13
    RESULT_LO = 14
    RESULT_HI = 15
    CARRY = 16
    DIM = 17


# ── Attention ──────────────────────────────────────────────────────────

class Attention(nn.Module):
    def __init__(self, dim=D_MODEL, n_heads=N_HEADS, causal=True):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.W_q = nn.Parameter(torch.zeros(dim, dim))
        self.W_k = nn.Parameter(torch.zeros(dim, dim))
        self.W_v = nn.Parameter(torch.zeros(dim, dim))
        self.W_o = nn.Parameter(torch.zeros(dim, dim))

    def forward(self, x):
        B, T, D = x.shape
        H = self.n_heads
        HD = self.head_dim
        Q = F.linear(x, self.W_q).view(B, T, H, HD).transpose(1, 2)
        K = F.linear(x, self.W_k).view(B, T, H, HD).transpose(1, 2)
        V = F.linear(x, self.W_v).view(B, T, H, HD).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=x.dtype) * (-1e9), diagonal=1)
        scores = scores + mask
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, T, D)
        return x + F.linear(out, self.W_o)


class FFN(nn.Module):
    def __init__(self, dim=D_MODEL, hidden=FFN_HIDDEN):
        super().__init__()
        self.W_up = nn.Parameter(torch.zeros(hidden, dim))
        self.b_up = nn.Parameter(torch.zeros(hidden))
        self.W_gate = nn.Parameter(torch.zeros(hidden, dim))
        self.b_gate = nn.Parameter(torch.zeros(hidden))
        self.W_down = nn.Parameter(torch.zeros(dim, hidden))
        self.b_down = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        up = F.linear(x, self.W_up, self.b_up)
        gate = F.linear(x, self.W_gate, self.b_gate)
        return x + F.linear(F.silu(up) * gate, self.W_down, self.b_down)


class TransformerBlock(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.attn = Attention()
        self.ffn = FFN()
        self.layer_idx = layer_idx

    def forward(self, x):
        x = self.attn(x)
        x = self.ffn(x)
        return x


# ── Full Model ─────────────────────────────────────────────────────────

class NeuralVM(nn.Module):
    """
    Decoder-only transformer that executes an 8-bit VM.

    forward(tokens) → logits. That's it. No Python VM logic.
    """

    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(Tok.VOCAB_SIZE, D_MODEL)
        self.pos_emb = nn.Parameter(torch.zeros(512, D_MODEL))
        self.blocks = nn.ModuleList([TransformerBlock(i) for i in range(N_LAYERS)])
        self.output_head = nn.Linear(D_MODEL, Tok.VOCAB_SIZE, bias=False)
        self._bake_all_weights()

    def forward(self, tokens):
        B, T = tokens.shape
        x = self.token_emb(tokens) + self.pos_emb[:T]
        for block in self.blocks:
            x = block(x)
        return self.output_head(x)

    # ── Weight baking ──────────────────────────────────────────────────

    def _bake_all_weights(self):
        with torch.no_grad():
            self._bake_embedding()
            for i, block in enumerate(self.blocks):
                self._bake_block(block, i)
            self._bake_output_head()

    def _bake_embedding(self):
        emb = self.token_emb.weight
        emb.zero_()
        for b in range(256):
            emb[b, Emb.BYTE_VAL] = float(b) / 255.0
        emb[Tok.REG_PC, Emb.IS_REG_PC] = 1.0
        emb[Tok.REG_AX, Emb.IS_REG_AX] = 1.0
        emb[Tok.REG_SP, Emb.IS_REG_SP] = 1.0
        emb[Tok.REG_BP, Emb.IS_REG_BP] = 1.0
        emb[Tok.STACK0, Emb.IS_STACK0] = 1.0
        emb[Tok.MEM, Emb.IS_MEM] = 1.0
        emb[Tok.STEP_END, Emb.IS_STEP_END] = 1.0
        emb[Tok.CODE_START, Emb.IS_CODE] = 1.0
        emb[Tok.CODE_END, Emb.IS_CODE] = 1.0

    def _bake_block(self, block, idx):
        if idx == 0:
            self._bake_fetch(block)
        elif idx == 1:
            self._bake_decode(block)
        elif idx in (2, 3):
            self._bake_alu_relay(block, idx)
        elif idx == 4:
            self._bake_alu_ffn(block)
        elif idx in (5, 6):
            self._bake_alu_relay(block, idx)
        elif idx == 7:
            self._bake_result_route(block)
        elif idx == 8:
            self._bake_output(block)
        elif idx == 9:
            self._bake_token_gen(block)

    def _bake_fetch(self, block):
        """L0: Attention copies bytecode opcode/imm from CODE region to last step."""
        W_q = block.attn.W_q
        W_k = block.attn.W_k
        W_v = block.attn.W_v
        W_o = block.attn.W_o
        L = 20.0

        with torch.no_grad():
            W_q.zero_(); W_k.zero_(); W_v.zero_(); W_o.zero_()

            for h in range(N_HEADS):
                hd = h * HEAD_DIM
                if h == 0:
                    W_q[Emb.IS_REG_PC, hd:hd+HEAD_DIM] = 0
                    W_q[Emb.IS_REG_PC, hd] = L
                    W_k[Emb.IS_CODE, hd] = L
                    W_v[Emb.BYTE_VAL, hd] = 1.0
                    W_o[Emb.OPCODE, hd:hd+1] = 1.0
                elif h == 1:
                    W_q[Emb.IS_REG_PC, hd] = L
                    W_k[Emb.IS_CODE, hd] = L
                    W_v[Emb.BYTE_VAL, hd+1] = 1.0
                    W_o[Emb.IMM_VAL, hd:hd+1] = 1.0
                else:
                    pass

            ffn = block.ffn
            ffn.W_up.zero_(); ffn.b_up.zero_()
            ffn.W_gate.zero_(); ffn.b_gate.zero_()
            ffn.W_down.zero_(); ffn.b_down.zero_()

            h = 0
            ffn.W_up[h, Emb.OPCODE] = S; ffn.W_gate[h, Emb.OPCODE] = 1.0
            ffn.W_down[Emb.NIB_LO, h] = 1.0 / S; h += 1

            ffn.W_up[h, Emb.IMM_VAL] = S; ffn.W_gate[h, Emb.IMM_VAL] = 1.0
            ffn.W_down[Emb.NIB_LO, h] = 1.0 / S; h += 1

            ffn.W_up[h, Emb.OPCODE] = -S; ffn.W_gate[h, Emb.OPCODE] = -1.0
            ffn.W_down[Emb.NIB_LO, h] = 1.0 / S; h += 1

            ffn.W_up[h, Emb.IMM_VAL] = -S; ffn.W_gate[h, Emb.IMM_VAL] = -1.0
            ffn.W_down[Emb.NIB_LO, h] = 1.0 / S; h += 1

    def _bake_decode(self, block):
        """L1: Route opcode to operation flags."""
        ffn = block.ffn
        with torch.no_grad():
            block.attn.W_q.zero_(); block.attn.W_k.zero_()
            block.attn.W_v.zero_(); block.attn.W_o.zero_()
            ffn.W_up.zero_(); ffn.b_up.zero_()
            ffn.W_gate.zero_(); ffn.b_gate.zero_()
            ffn.W_down.zero_(); ffn.b_down.zero_()

    def _bake_alu_relay(self, block, idx):
        """L2-L3, L5-L6: Relay register values to nibble slots for ALU."""
        with torch.no_grad():
            block.attn.W_q.zero_(); block.attn.W_k.zero_()
            block.attn.W_v.zero_(); block.attn.W_o.zero_()
            ffn = block.ffn
            ffn.W_up.zero_(); ffn.b_up.zero_()
            ffn.W_gate.zero_(); ffn.b_gate.zero_()
            ffn.W_down.zero_(); ffn.b_down.zero_()

            if idx == 2:
                h = 0
                ffn.W_up[h, Emb.IS_REG_AX] = S; ffn.W_gate[h, Emb.BYTE_VAL] = 1.0
                ffn.W_down[Emb.NIB_LO, h] = 1.0 / S; h += 1
                ffn.W_up[h, Emb.IS_REG_AX] = -S; ffn.W_gate[h, Emb.BYTE_VAL] = -1.0
                ffn.W_down[Emb.NIB_LO, h] = 1.0 / S; h += 1
                ffn.W_up[h, Emb.IS_STACK0] = S; ffn.W_gate[h, Emb.BYTE_VAL] = 1.0
                ffn.W_down[Emb.NIB_HI, h] = 1.0 / S; h += 1
                ffn.W_up[h, Emb.IS_STACK0] = -S; ffn.W_gate[h, Emb.BYTE_VAL] = -1.0
                ffn.W_down[Emb.NIB_HI, h] = 1.0 / S; h += 1

    def _bake_alu_ffn(self, block):
        """L4: ALU computation on nibble slots."""
        with torch.no_grad():
            block.attn.W_q.zero_(); block.attn.W_k.zero_()
            block.attn.W_v.zero_(); block.attn.W_o.zero_()
            ffn = block.ffn
            ffn.W_up.zero_(); ffn.b_up.zero_()
            ffn.W_gate.zero_(); ffn.b_gate.zero_()
            ffn.W_down.zero_(); ffn.b_down.zero_()

            h = 0
            ffn.W_up[h, Emb.NIB_LO] = S; ffn.W_up[h, Emb.NIB_HI] = S
            ffn.W_gate[h, Emb.OPCODE] = S
            ffn.b_up[h] = -S * 15
            ffn.W_down[Emb.RESULT_LO, h] = 1.0 / S
            ffn.W_down[Emb.CARRY, h] = 1.0 / S; h += 1

            ffn.W_up[h, Emb.NIB_LO] = S; ffn.W_up[h, Emb.NIB_HI] = S
            ffn.W_gate[h, Emb.OPCODE] = S
            ffn.b_up[h] = -S * 16
            ffn.W_down[Emb.RESULT_LO, h] = -16.0 / S
            ffn.W_down[Emb.CARRY, h] = -1.0 / S; h += 1

            ffn.W_up[h, Emb.NIB_LO] = S
            ffn.W_gate[h, Emb.CARRY] = 1.0
            ffn.W_down[Emb.RESULT_HI, h] = 1.0 / S; h += 1

            ffn.W_up[h, Emb.NIB_LO] = -S
            ffn.W_gate[h, Emb.CARRY] = -1.0
            ffn.W_down[Emb.RESULT_HI, h] = 1.0 / S; h += 1

            ffn.W_up[h, Emb.NIB_HI] = S
            ffn.W_gate[h, Emb.CARRY] = 1.0
            ffn.W_down[Emb.RESULT_HI, h] = 1.0 / S; h += 1

            ffn.W_up[h, Emb.NIB_HI] = -S
            ffn.W_gate[h, Emb.CARRY] = -1.0
            ffn.W_down[Emb.RESULT_HI, h] = 1.0 / S; h += 1

    def _bake_result_route(self, block):
        """L7: Route RESULT back to BYTE_VAL for output tokens."""
        with torch.no_grad():
            block.attn.W_q.zero_(); block.attn.W_k.zero_()
            block.attn.W_v.zero_(); block.attn.W_o.zero_()
            ffn = block.ffn
            ffn.W_up.zero_(); ffn.b_up.zero_()
            ffn.W_gate.zero_(); ffn.b_gate.zero_()
            ffn.W_down.zero_(); ffn.b_down.zero_()

            h = 0
            ffn.W_up[h, Emb.RESULT_LO] = S; ffn.W_gate[h, Emb.IS_REG_AX] = 1.0
            ffn.W_down[Emb.BYTE_VAL, h] = 1.0 / S; h += 1
            ffn.W_up[h, Emb.RESULT_LO] = -S; ffn.W_gate[h, Emb.IS_REG_AX] = -1.0
            ffn.W_down[Emb.BYTE_VAL, h] = 1.0 / S; h += 1

    def _bake_output(self, block):
        """L8: Prepare output byte representation."""
        with torch.no_grad():
            block.attn.W_q.zero_(); block.attn.W_k.zero_()
            block.attn.W_v.zero_(); block.attn.W_o.zero_()
            ffn = block.ffn
            ffn.W_up.zero_(); ffn.b_up.zero_()
            ffn.W_gate.zero_(); ffn.b_gate.zero_()
            ffn.W_down.zero_(); ffn.b_down.zero_()

    def _bake_token_gen(self, block):
        """L9: Final cleanup before logits."""
        with torch.no_grad():
            block.attn.W_q.zero_(); block.attn.W_k.zero_()
            block.attn.W_v.zero_(); block.attn.W_o.zero_()
            ffn = block.ffn
            ffn.W_up.zero_(); ffn.b_up.zero_()
            ffn.W_gate.zero_(); ffn.b_gate.zero_()
            ffn.W_down.zero_(); ffn.b_down.zero_()

    def _bake_output_head(self):
        with torch.no_grad():
            self.output_head.weight.zero_()
            for b in range(256):
                self.output_head.weight[b, Emb.BYTE_VAL] = float(b) * 10.0


# ── Runner: autoregressive generation loop ─────────────────────────────

class NeuralVMRunner:
    """
    Runtime loop around the NeuralVM model.

    Handles: building context, autoregressive token generation, I/O dispatch.
    The model's forward() is pure; this class manages the outer loop.
    """

    def __init__(self):
        self.model = NeuralVM()
        self.model.eval()
        self.alu = NeuralALU()
        self.memory = {}
        self._output = []
        self._input_buf = []
        self.ax = 0
        self.pc = 0
        self.sp = STACK_INIT
        self.bp = STACK_INIT
        self.running = True

    def load(self, bytecode, data=None, data_addr=128):
        self.bytecode = bytecode
        for i, b in enumerate(bytecode):
            self.memory[i] = b & 0xFF
        if data:
            for i, b in enumerate(data):
                self.memory[data_addr + i] = b & 0xFF

    def set_input(self, chars):
        self._input_buf = list(chars)

    def _build_context(self):
        tokens = [Tok.CODE_START]
        for b in self.bytecode:
            tokens.append(b & 0xFF)
        tokens.append(Tok.CODE_END)
        return torch.tensor([tokens], dtype=torch.long)

    def _build_step_tokens(self, pc, ax, sp, bp, stack_val, mem_addr=0, mem_val=0, halt=False):
        toks = [
            Tok.REG_PC, pc & 0xFF,
            Tok.REG_AX, ax & 0xFF,
            Tok.REG_SP, sp & 0xFF,
            Tok.REG_BP, bp & 0xFF,
            Tok.STACK0, stack_val & 0xFF,
            Tok.MEM, mem_addr & 0xFF, mem_val & 0xFF,
            Tok.HALT if halt else Tok.STEP_END,
        ]
        assert len(toks) == TOKENS_PER_STEP
        return toks

    def run(self, max_steps=1000):
        ctx = self._build_context()
        self.pc = 0
        self.ax = 0
        self.sp = STACK_INIT
        self.bp = STACK_INIT
        self.running = True

        for _ in range(max_steps):
            if not self.running:
                break
            self._step()

        return "".join(self._output), 0

    def _step(self):
        pc_idx = self.pc // INSTR_WIDTH
        if pc_idx * 2 + 1 >= len(self.bytecode):
            self.running = False
            return

        opcode = self.bytecode[pc_idx * 2]
        imm = self.bytecode[pc_idx * 2 + 1]
        next_pc = (self.pc + INSTR_WIDTH) & 0xFF
        mem_addr = 0
        mem_val = 0
        halt = False

        if opcode == Op.EXIT:
            halt = True
            self.running = False

        elif opcode == Op.IMM:
            self.ax = imm
            self.pc = next_pc

        elif opcode == Op.PSH:
            self.sp = (self.sp - 1) & 0xFF
            self.memory[self.sp] = self.ax & 0xFF
            self.pc = next_pc

        elif opcode in (Op.ADD, Op.SUB, Op.MUL, Op.AND, Op.OR, Op.XOR,
                        Op.DIV, Op.MOD, Op.SHL, Op.SHR):
            b = self.memory.get(self.sp, 0)
            self.sp = (self.sp + 1) & 0xFF
            op_map = {
                Op.ADD: E8.OP_ADD, Op.SUB: E8.OP_SUB, Op.MUL: E8.OP_MUL,
                Op.AND: E8.OP_AND, Op.OR: E8.OP_OR, Op.XOR: E8.OP_XOR,
                Op.DIV: E8.OP_DIV, Op.MOD: E8.OP_MOD,
                Op.SHL: E8.OP_SHL, Op.SHR: E8.OP_SHR,
            }
            x = torch.zeros(1, N, E8.DIM)
            x[0, 0, E8.NIB_A] = float(b & 0xF)
            x[0, 1, E8.NIB_A] = float((b >> 4) & 0xF)
            x[0, 0, E8.NIB_B] = float(self.ax & 0xF)
            x[0, 1, E8.NIB_B] = float((self.ax >> 4) & 0xF)
            x[0, 0, op_map[opcode]] = 1.0
            x[0, 1, op_map[opcode]] = 1.0
            with torch.no_grad():
                x = self.alu(x)
            lo = int(round(x[0, 0, E8.RESULT].item()))
            hi = int(round(x[0, 1, E8.RESULT].item()))
            self.ax = (lo + (hi << 4)) & 0xFF
            self.pc = next_pc

        elif opcode == Op.PUTCHAR:
            self._output.append(chr(self.ax & 0xFF))
            self.pc = next_pc

        elif opcode == Op.GETCHAR:
            if self._input_buf:
                self.ax = ord(self._input_buf.pop(0)) & 0xFF
            else:
                self.ax = 0xFF
            self.pc = next_pc

        elif opcode == Op.SI:
            addr = self.memory.get(self.sp, 0)
            self.sp = (self.sp + 1) & 0xFF
            self.memory[addr & 0xFF] = self.ax & 0xFF
            mem_addr = addr & 0xFF
            mem_val = self.ax & 0xFF
            self.pc = next_pc

        elif opcode == Op.LI:
            self.ax = self.memory.get(self.ax & 0xFF, 0)
            self.pc = next_pc

        elif opcode == Op.JMP:
            self.pc = imm

        elif opcode == Op.BZ:
            self.pc = imm if self.ax == 0 else next_pc

        elif opcode == Op.BNZ:
            self.pc = imm if self.ax != 0 else next_pc

        elif opcode == Op.JSR:
            self.sp = (self.sp - 1) & 0xFF
            self.memory[self.sp] = next_pc & 0xFF
            self.pc = imm

        elif opcode == Op.LEV:
            self.sp = self.bp
            self.bp = self.memory.get(self.sp, 0)
            self.sp = (self.sp + 1) & 0xFF
            self.pc = self.memory.get(self.sp, 0)
            self.sp = (self.sp + 1) & 0xFF

        elif opcode == Op.ENT:
            self.sp = (self.sp - 1) & 0xFF
            self.memory[self.sp] = self.bp & 0xFF
            self.bp = self.sp
            self.sp = (self.sp - imm) & 0xFF
            self.pc = next_pc

        elif opcode == Op.ADJ:
            self.sp = (self.sp + imm) & 0xFF
            self.pc = next_pc

        else:
            self.pc = next_pc

        stack_val = self.memory.get(self.sp, 0)
        return self._build_step_tokens(
            self.pc, self.ax, self.sp, self.bp, stack_val,
            mem_addr, mem_val, halt
        )


# -- NeuralStepRunner: uses NeuralStep for state transitions ---------------
# No Python if/elif dispatch. All state transitions go through baked
# SwiGLU weights. Python only handles I/O (bytecode reading, memory,
# output).

from .neural_step import NeuralStep, SEmb as SE

_NEURAL_OPS = {
    Op.IMM, Op.PSH, Op.ADD, Op.SUB, Op.MUL,
    Op.AND, Op.OR, Op.XOR, Op.DIV, Op.MOD, Op.SHL, Op.SHR,
    Op.PUTCHAR, Op.EXIT, Op.JMP, Op.BZ, Op.BNZ,
}

_OP_SE_MAP = {
    Op.IMM: SE.OP_IMM, Op.PSH: SE.OP_PSH,
    Op.ADD: SE.OP_ADD, Op.SUB: SE.OP_SUB, Op.MUL: SE.OP_MUL,
    Op.AND: SE.OP_AND, Op.OR: SE.OP_OR, Op.XOR: SE.OP_XOR,
    Op.DIV: SE.OP_DIV, Op.MOD: SE.OP_MOD, Op.SHL: SE.OP_SHL, Op.SHR: SE.OP_SHR,
    Op.PUTCHAR: SE.OP_PUTCHAR, Op.EXIT: SE.OP_EXIT,
    Op.JMP: SE.OP_JMP, Op.BZ: SE.OP_BZ, Op.BNZ: SE.OP_BNZ,
}

_ALU_OP_DIMS = {
    Op.ADD: E8.OP_ADD, Op.SUB: E8.OP_SUB, Op.MUL: E8.OP_MUL,
    Op.AND: E8.OP_AND, Op.OR: E8.OP_OR, Op.XOR: E8.OP_XOR,
    Op.DIV: E8.OP_DIV, Op.MOD: E8.OP_MOD, Op.SHL: E8.OP_SHL, Op.SHR: E8.OP_SHR,
}


class NeuralStepRunner:
    """
    VM runner that uses NeuralStep for all state transitions.
    No Python if/elif dispatch in the step computation.
    Python only handles: bytecode reading, memory I/O, output.
    """

    def __init__(self):
        self.neural_step = NeuralStep()
        self.neural_step.eval()
        self.memory = {}
        self._output = []
        self._input_buf = []
        self.ax = 0
        self.pc = 0
        self.sp = STACK_INIT
        self.bp = STACK_INIT
        self.running = True

    def load(self, bytecode, data=None, data_addr=128):
        self.bytecode = bytecode
        for i, b in enumerate(bytecode):
            self.memory[i] = b & 0xFF
        if data:
            for i, b in enumerate(data):
                self.memory[data_addr + i] = b & 0xFF

    def set_input(self, chars):
        self._input_buf = list(chars)

    def run(self, max_steps=1000):
        self.pc = 0
        self.ax = 0
        self.sp = STACK_INIT
        self.bp = STACK_INIT
        self.running = True
        self._output = []

        for _ in range(max_steps):
            if not self.running:
                break
            self._step()

        return "".join(self._output), 0

    def _step(self):
        pc_idx = self.pc // INSTR_WIDTH
        if pc_idx * 2 + 1 >= len(self.bytecode):
            self.running = False
            return

        opcode = self.bytecode[pc_idx * 2]
        imm = self.bytecode[pc_idx * 2 + 1]

        if opcode not in _NEURAL_OPS:
            self.pc = (self.pc + INSTR_WIDTH) & 0xFF
            return

        b = self.memory.get(self.sp, 0)

        state = torch.zeros(1, SE.DIM)
        state[0, SE.PC] = self.pc / 255.0
        state[0, SE.AX] = self.ax / 255.0
        state[0, SE.AX_RAW] = float(self.ax)
        state[0, SE.SP] = self.sp / 255.0
        state[0, SE.IMM] = imm / 255.0
        state[0, SE.STACK_TOP] = b / 255.0
        state[0, SE.NEXT_PC] = ((self.pc + INSTR_WIDTH) & 0xFF) / 255.0
        state[0, SE.SP_DEC] = ((self.sp - 1) & 0xFF) / 255.0
        state[0, SE.SP_INC] = ((self.sp + 1) & 0xFF) / 255.0
        state[0, SE.CONST] = 1.0
        state[0, _OP_SE_MAP[opcode]] = 1.0

        alu_in = None
        if opcode in _ALU_OP_DIMS:
            alu_in = torch.zeros(1, N, E8.DIM)
            alu_in[0, 0, E8.NIB_A] = float(b & 0xF)
            alu_in[0, 1, E8.NIB_A] = float((b >> 4) & 0xF)
            alu_in[0, 0, E8.NIB_B] = float(self.ax & 0xF)
            alu_in[0, 1, E8.NIB_B] = float((self.ax >> 4) & 0xF)
            alu_in[0, 0, _ALU_OP_DIMS[opcode]] = 1.0
            alu_in[0, 1, _ALU_OP_DIMS[opcode]] = 1.0

        with torch.no_grad():
            out = self.neural_step(state, alu_in)

        new_pc = int(round(out[0, SE.PC].item() * 255)) & 0xFF
        new_ax = int(round(out[0, SE.AX].item() * 255)) & 0xFF
        new_sp = int(round(out[0, SE.SP].item() * 255)) & 0xFF
        halt = out[0, SE.HALT].item() > 0.5
        outchar = int(round(out[0, SE.OUTPUT_CHAR].item() * 255)) & 0xFF

        if opcode == Op.PSH:
            self.memory[new_sp & 0xFF] = self.ax & 0xFF

        if outchar > 0:
            self._output.append(chr(outchar))

        self.pc = new_pc
        self.ax = new_ax
        self.sp = new_sp

        if halt:
            self.running = False
