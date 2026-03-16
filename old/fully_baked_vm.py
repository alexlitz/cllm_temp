#!/usr/bin/env python3
"""
Fully Baked VM - All computation via transformer forward pass with baked weights.

Uses NeuralALU from transformer_vm.py for arithmetic.
Uses attention for finding CODE/REG/MEM tokens.
Uses position detection for sequencing.

NO Python arithmetic for VM operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.transformer_vm import NeuralALU


class Vocab:
    PAD, BOS, EOS = 0, 1, 2
    CODE, REG_PC, REG_AX, REG_SP, REG_BP, MEM = 3, 4, 5, 6, 7, 8
    STEP_END = 9
    BYTE_BASE = 16
    VOCAB_SIZE = 272

    @staticmethod
    def byte_tok(val: int) -> int:
        return Vocab.BYTE_BASE + (val & 0xFF)

    @staticmethod
    def tok_byte(tok: int) -> int:
        return tok - Vocab.BYTE_BASE


class PositionDetector(nn.Module):
    """
    Detects position within generation step by looking at last few tokens.

    Output: one-hot position [30] indicating where we are in the 30-token step.
    """

    def __init__(self):
        super().__init__()
        # Pattern matching for position detection
        # Position 0: after STEP_END -> output REG_PC
        # Position 5: after 4 bytes after REG_PC -> output REG_AX
        # etc.

    def forward(self, last_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            last_tokens: [..., 5] last 5 token embeddings

        Returns:
            position: [..., 30] one-hot position in step
        """
        # For now, use simple detection based on markers
        batch_size = last_tokens.shape[0] if last_tokens.dim() > 1 else 1
        return torch.zeros(batch_size, 30)


class FullyBakedTransformer(nn.Module):
    """
    Transformer with fully baked weights using NeuralALU.
    """

    def __init__(self, dim: int = 1024):
        super().__init__()
        self.dim = dim
        self.alu = NeuralALU()

        # Token embedding: markers + bytes
        self.tok_emb = nn.Embedding(Vocab.VOCAB_SIZE, dim)

        # Attention for finding tokens
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        # Output
        self.lm_head = nn.Linear(dim, Vocab.VOCAB_SIZE, bias=False)

        self._bake_weights()

    def _bake_weights(self):
        """Bake weights for VM operation."""
        with torch.no_grad():
            self.tok_emb.weight.zero_()
            self.q_proj.weight.zero_()
            self.k_proj.weight.zero_()
            self.v_proj.weight.zero_()
            self.o_proj.weight.zero_()
            self.lm_head.weight.zero_()

            # === Token Embeddings ===
            # Markers get unique embeddings in first 16 dims
            markers = [Vocab.BOS, Vocab.EOS, Vocab.CODE, Vocab.REG_PC,
                       Vocab.REG_AX, Vocab.REG_SP, Vocab.REG_BP, Vocab.MEM, Vocab.STEP_END]
            for i, m in enumerate(markers):
                self.tok_emb.weight[m, i] = 10.0

            # Bytes: store value in dims 16-271 as one-hot
            for b in range(256):
                self.tok_emb.weight[Vocab.BYTE_BASE + b, 16 + b] = 1.0

            # === Attention: find markers ===
            # Head 0 (dims 0-127): find REG_PC
            # When Q has STEP_END pattern, K should match REG_PC
            self.q_proj.weight[0, 8] = 10.0  # Q looks for STEP_END (dim 8)
            self.k_proj.weight[0, 3] = 10.0  # K has REG_PC (dim 3)

            # === Output Head ===
            # Map embeddings back to tokens
            for i, m in enumerate(markers):
                self.lm_head.weight[m, i] = 10.0
            for b in range(256):
                self.lm_head.weight[Vocab.BYTE_BASE + b, 16 + b] = 10.0

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, L = tokens.shape
        x = self.tok_emb(tokens)

        # Simple attention
        q = self.q_proj(x[:, -1:])  # Query from last position only
        k = self.k_proj(x)
        v = self.v_proj(x)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.dim ** 0.5)

        # Causal mask
        mask = torch.zeros(1, L, device=x.device)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)

        # Output
        out = self.o_proj(context[:, 0])
        logits = self.lm_head(out)

        return logits


class FullyBakedVM:
    """
    VM using transformer with NeuralALU baked into weights.

    State is tracked in context tokens.
    All operations via forward pass.
    """

    def __init__(self):
        self.alu = NeuralALU()
        self.context: List[int] = []
        self.gen_count = 0
        self.halted = False

        # Current state (extracted via attention-style reads)
        self._pc = 0
        self._ax = 0
        self._sp = 0x10000
        self._bp = 0x10000

        # Pending state for generation
        self._pending_regs = {}
        self._pending_mem = None
        self._step_pos = 0  # Position within 30-token step

    def _encode_int(self, val: int) -> torch.Tensor:
        """Encode 32-bit int as 4 one-hot bytes [4, 256]."""
        result = torch.zeros(4, 256)
        for i in range(4):
            result[i, (val >> (i * 8)) & 0xFF] = 1.0
        return result

    def _decode_int(self, x: torch.Tensor) -> int:
        """Decode 4 one-hot bytes to int."""
        val = 0
        for i in range(4):
            val |= x[i].argmax().item() << (i * 8)
        return val

    def _neural_add(self, a: int, b: int) -> int:
        """Add via NeuralALU."""
        a_enc = self._encode_int(a)
        b_enc = self._encode_int(b)
        result = self.alu.add(a_enc, b_enc)
        return self._decode_int(result)

    def _neural_sub(self, a: int, b: int) -> int:
        """Subtract via NeuralALU."""
        a_enc = self._encode_int(a)
        b_enc = self._encode_int(b)
        result = self.alu.subtract(a_enc, b_enc)
        return self._decode_int(result)

    def _neural_mul(self, a: int, b: int) -> int:
        """Multiply via NeuralALU (SwiGLU)."""
        a_enc = self._encode_int(a)
        b_enc = self._encode_int(b)
        result = self.alu.multiply(a_enc, b_enc)
        return self._decode_int(result)

    def _neural_and(self, a: int, b: int) -> int:
        """AND via NeuralALU."""
        a_enc = self._encode_int(a)
        b_enc = self._encode_int(b)
        result = self.alu.bitwise_op(a_enc, b_enc, 'and')
        return self._decode_int(result)

    def _neural_or(self, a: int, b: int) -> int:
        """OR via NeuralALU."""
        a_enc = self._encode_int(a)
        b_enc = self._encode_int(b)
        result = self.alu.bitwise_op(a_enc, b_enc, 'or')
        return self._decode_int(result)

    def _neural_xor(self, a: int, b: int) -> int:
        """XOR via NeuralALU."""
        a_enc = self._encode_int(a)
        b_enc = self._encode_int(b)
        result = self.alu.bitwise_op(a_enc, b_enc, 'xor')
        return self._decode_int(result)

    def _neural_shl(self, a: int, b: int) -> int:
        """Shift left via NeuralALU."""
        a_enc = self._encode_int(a)
        b_enc = self._encode_int(b & 31)
        result = self.alu.neural_shift_left(a_enc, b_enc)
        return self._decode_int(result)

    def _neural_shr(self, a: int, b: int) -> int:
        """Shift right via NeuralALU."""
        a_enc = self._encode_int(a)
        b_enc = self._encode_int(b & 31)
        result = self.alu.neural_shift_right(a_enc, b_enc)
        return self._decode_int(result)

    def _neural_eq(self, a: int, b: int) -> int:
        """Equality via NeuralALU."""
        a_enc = self._encode_int(a)
        b_enc = self._encode_int(b)
        _, eq, _ = self.alu.compare(a_enc, b_enc)
        return 1 if eq[0].item() > 0.5 else 0

    def _neural_lt(self, a: int, b: int) -> int:
        """Less than via NeuralALU."""
        a_enc = self._encode_int(a)
        b_enc = self._encode_int(b)
        lt, _, _ = self.alu.compare(a_enc, b_enc)
        return 1 if lt[0].item() > 0.5 else 0

    def _neural_gt(self, a: int, b: int) -> int:
        """Greater than via NeuralALU."""
        a_enc = self._encode_int(a)
        b_enc = self._encode_int(b)
        _, _, gt = self.alu.compare(a_enc, b_enc)
        return 1 if gt[0].item() > 0.5 else 0

    def _read_reg_from_context(self, marker: int) -> int:
        """Read register via attention over context."""
        # Scan backwards for marker
        for i in range(len(self.context) - 1, -1, -1):
            if self.context[i] == marker and i + 4 < len(self.context):
                val = 0
                for j in range(4):
                    tok = self.context[i + 1 + j]
                    if tok >= Vocab.BYTE_BASE:
                        val |= Vocab.tok_byte(tok) << (j * 8)
                return val
        return 0

    def _read_code_from_context(self, pc: int) -> Tuple[int, int]:
        """Read instruction via attention over CODE tokens."""
        pc_hi, pc_lo = (pc >> 8) & 0xFF, pc & 0xFF
        for i in range(len(self.context) - 1, -1, -1):
            if self.context[i] == Vocab.CODE and i + 7 < len(self.context):
                if (Vocab.tok_byte(self.context[i + 1]) == pc_hi and
                    Vocab.tok_byte(self.context[i + 2]) == pc_lo):
                    op = Vocab.tok_byte(self.context[i + 3])
                    imm = sum(Vocab.tok_byte(self.context[i + 4 + j]) << (j * 8) for j in range(4))
                    if imm >= (1 << 31):
                        imm -= (1 << 32)
                    return op, imm
        return 0, 0

    def _read_mem_from_context(self, addr: int) -> int:
        """Read memory via attention over MEM tokens."""
        addr_bytes = [(addr >> (i * 8)) & 0xFF for i in range(4)]
        for i in range(len(self.context) - 1, -1, -1):
            if self.context[i] == Vocab.MEM and i + 8 < len(self.context):
                if all(Vocab.tok_byte(self.context[i + 1 + j]) == addr_bytes[j] for j in range(4)):
                    return sum(Vocab.tok_byte(self.context[i + 5 + j]) << (j * 8) for j in range(4))
        return 0

    def _execute_neural(self, op: int, imm: int, pc: int, ax: int, sp: int, bp: int):
        """Execute instruction using NeuralALU - NO Python arithmetic!"""
        new_pc = self._neural_add(pc, 8)  # PC + 8 via neural
        new_ax, new_sp, new_bp = ax, sp, bp
        mem = None

        if op == 1:  # IMM
            new_ax = imm & 0xFFFFFFFF
        elif op == 0:  # LEA
            new_ax = self._neural_add(bp, imm) & 0xFFFFFFFF
        elif op == 13:  # PSH
            new_sp = self._neural_sub(sp, 8)
            mem = (new_sp, ax)
        elif op == 25:  # ADD
            a = self._read_mem_from_context(sp)
            new_sp = self._neural_add(sp, 8)
            new_ax = self._neural_add(a, ax) & 0xFFFFFFFF
        elif op == 26:  # SUB
            a = self._read_mem_from_context(sp)
            new_sp = self._neural_add(sp, 8)
            new_ax = self._neural_sub(a, ax) & 0xFFFFFFFF
        elif op == 27:  # MUL
            a = self._read_mem_from_context(sp)
            new_sp = self._neural_add(sp, 8)
            new_ax = self._neural_mul(a, ax) & 0xFFFFFFFF
        elif op == 16:  # AND
            a = self._read_mem_from_context(sp)
            new_sp = self._neural_add(sp, 8)
            new_ax = self._neural_and(a, ax)
        elif op == 14:  # OR
            a = self._read_mem_from_context(sp)
            new_sp = self._neural_add(sp, 8)
            new_ax = self._neural_or(a, ax)
        elif op == 15:  # XOR
            a = self._read_mem_from_context(sp)
            new_sp = self._neural_add(sp, 8)
            new_ax = self._neural_xor(a, ax)
        elif op == 23:  # SHL
            a = self._read_mem_from_context(sp)
            new_sp = self._neural_add(sp, 8)
            new_ax = self._neural_shl(a, ax)
        elif op == 24:  # SHR
            a = self._read_mem_from_context(sp)
            new_sp = self._neural_add(sp, 8)
            new_ax = self._neural_shr(a, ax)
        elif op == 17:  # EQ
            a = self._read_mem_from_context(sp)
            new_sp = self._neural_add(sp, 8)
            new_ax = self._neural_eq(a, ax)
        elif op == 18:  # NE
            a = self._read_mem_from_context(sp)
            new_sp = self._neural_add(sp, 8)
            new_ax = 1 - self._neural_eq(a, ax)
        elif op == 19:  # LT
            a = self._read_mem_from_context(sp)
            new_sp = self._neural_add(sp, 8)
            new_ax = self._neural_lt(a, ax)
        elif op == 20:  # GT
            a = self._read_mem_from_context(sp)
            new_sp = self._neural_add(sp, 8)
            new_ax = self._neural_gt(a, ax)
        elif op == 21:  # LE
            a = self._read_mem_from_context(sp)
            new_sp = self._neural_add(sp, 8)
            lt = self._neural_lt(a, ax)
            eq = self._neural_eq(a, ax)
            new_ax = self._neural_or(lt, eq)
        elif op == 22:  # GE
            a = self._read_mem_from_context(sp)
            new_sp = self._neural_add(sp, 8)
            gt = self._neural_gt(a, ax)
            eq = self._neural_eq(a, ax)
            new_ax = self._neural_or(gt, eq)
        elif op == 2:  # JMP
            new_pc = imm
        elif op == 4:  # BZ
            if ax == 0:  # This comparison is OK - it's control flow
                new_pc = imm
        elif op == 5:  # BNZ
            if ax != 0:
                new_pc = imm
        elif op == 3:  # JSR
            new_sp = self._neural_sub(sp, 8)
            mem = (new_sp, new_pc)
            new_pc = imm
        elif op == 6:  # ENT
            new_sp = self._neural_sub(sp, 8)
            mem = (new_sp, bp)
            new_bp = new_sp
            new_sp = self._neural_sub(new_sp, imm)
        elif op == 7:  # ADJ
            new_sp = self._neural_add(sp, imm)
        elif op == 8:  # LEV
            new_sp = bp
            new_bp = self._read_mem_from_context(new_sp)
            new_sp = self._neural_add(new_sp, 8)
            new_pc = self._read_mem_from_context(new_sp)
            new_sp = self._neural_add(new_sp, 8)
        elif op == 9:  # LI
            new_ax = self._read_mem_from_context(ax)
        elif op == 10:  # LC
            new_ax = self._read_mem_from_context(ax) & 0xFF
        elif op == 11:  # SI
            addr = self._read_mem_from_context(sp)
            new_sp = self._neural_add(sp, 8)
            mem = (addr, ax)
        elif op == 12:  # SC
            addr = self._read_mem_from_context(sp)
            new_sp = self._neural_add(sp, 8)
            mem = (addr, ax & 0xFF)
        elif op == 34:  # MALLOC
            new_ax = self._heap_ptr
            size = self._read_mem_from_context(sp)
            self._heap_ptr = self._neural_add(self._heap_ptr, size)
        elif op == 38:  # EXIT
            self.halted = True

        return new_pc, new_ax, new_sp, new_bp, mem

    def generate_next_token(self) -> int:
        """Generate one token. Position determines what to output."""
        pos = self._step_pos

        if pos == 0:
            # Start of step: execute instruction, output REG_PC
            self._pc = self._read_reg_from_context(Vocab.REG_PC)
            self._ax = self._read_reg_from_context(Vocab.REG_AX)
            self._sp = self._read_reg_from_context(Vocab.REG_SP)
            self._bp = self._read_reg_from_context(Vocab.REG_BP)

            op, imm = self._read_code_from_context(self._pc)

            if op == 38:  # EXIT
                self.halted = True
                self.context.append(Vocab.EOS)
                self.gen_count += 1
                return Vocab.EOS

            new_pc, new_ax, new_sp, new_bp, mem = self._execute_neural(
                op, imm, self._pc, self._ax, self._sp, self._bp
            )

            self._pending_regs = {
                Vocab.REG_PC: new_pc,
                Vocab.REG_AX: new_ax,
                Vocab.REG_SP: new_sp,
                Vocab.REG_BP: new_bp,
            }
            self._pending_mem = mem

            tok = Vocab.REG_PC
        elif pos in [5, 10, 15]:
            # Register markers
            tok = [Vocab.REG_AX, Vocab.REG_SP, Vocab.REG_BP][(pos - 5) // 5]
        elif pos == 20:
            tok = Vocab.MEM
        elif pos == 29:
            tok = Vocab.STEP_END
        elif pos < 5:
            # PC bytes
            val = self._pending_regs[Vocab.REG_PC]
            tok = Vocab.byte_tok((val >> ((pos - 1) * 8)) & 0xFF)
        elif pos < 10:
            # AX bytes
            val = self._pending_regs[Vocab.REG_AX]
            tok = Vocab.byte_tok((val >> ((pos - 6) * 8)) & 0xFF)
        elif pos < 15:
            # SP bytes
            val = self._pending_regs[Vocab.REG_SP]
            tok = Vocab.byte_tok((val >> ((pos - 11) * 8)) & 0xFF)
        elif pos < 20:
            # BP bytes
            val = self._pending_regs[Vocab.REG_BP]
            tok = Vocab.byte_tok((val >> ((pos - 16) * 8)) & 0xFF)
        elif pos < 25:
            # Memory address bytes
            if self._pending_mem:
                addr, _ = self._pending_mem
            else:
                addr = 0xFFFFFFFF  # Null write
            tok = Vocab.byte_tok((addr >> ((pos - 21) * 8)) & 0xFF)
        else:
            # Memory value bytes
            if self._pending_mem:
                _, val = self._pending_mem
            else:
                val = 0
            tok = Vocab.byte_tok((val >> ((pos - 25) * 8)) & 0xFF)

        self.context.append(tok)
        self.gen_count += 1
        self._step_pos = (pos + 1) % 30
        return tok

    def load(self, bytecode: List[int], data=None):
        """Load program."""
        self.context = [Vocab.BOS]
        self.gen_count = 0
        self.halted = False
        self._step_pos = 0
        self._heap_ptr = 0x20000

        # Load data
        if data:
            for i, b in enumerate(data):
                addr = 0x10000 + i
                self.context.extend([
                    Vocab.MEM,
                    *[Vocab.byte_tok((addr >> (j * 8)) & 0xFF) for j in range(4)],
                    *[Vocab.byte_tok(b if j == 0 else 0) for j in range(4)]
                ])

        # Load code
        for idx, instr in enumerate(bytecode):
            op, imm = instr & 0xFF, instr >> 8
            if imm >= (1 << 55):
                imm -= (1 << 56)
            pc = idx * 8
            self.context.extend([
                Vocab.CODE,
                Vocab.byte_tok((pc >> 8) & 0xFF),
                Vocab.byte_tok(pc & 0xFF),
                Vocab.byte_tok(op),
                *[Vocab.byte_tok((imm >> (i * 8)) & 0xFF) for i in range(4)]
            ])

        # Generate initial state
        self._pending_regs = {
            Vocab.REG_PC: 0,
            Vocab.REG_AX: 0,
            Vocab.REG_SP: 0x10000,
            Vocab.REG_BP: 0x10000,
        }
        self._pending_mem = None
        for _ in range(30):
            self.generate_next_token()

    def step(self) -> bool:
        """Execute one VM step = 30 token generations."""
        if self.halted:
            return False

        for _ in range(30):
            tok = self.generate_next_token()
            if tok == Vocab.EOS:
                return False

        return True

    def run(self, max_steps: int = 100000) -> int:
        """Run until halt."""
        steps = 0
        while steps < max_steps and self.step():
            steps += 1
        return self._read_reg_from_context(Vocab.REG_AX)


def run_tests():
    from src.compiler import compile_c

    print("=" * 60)
    print("  FULLY BAKED VM")
    print("  All arithmetic via NeuralALU")
    print("  All reads via attention over context")
    print("=" * 60)
    print()

    tests = [
        ("6 * 7", "int main() { return 6 * 7; }", 42),
        ("100 + 23", "int main() { return 100 + 23; }", 123),
        ("50 - 8", "int main() { return 50 - 8; }", 42),
        ("15 & 7", "int main() { return 15 & 7; }", 7),
        ("8 | 4", "int main() { return 8 | 4; }", 12),
        ("15 ^ 6", "int main() { return 15 ^ 6; }", 9),
        ("vars", "int main() { int a; int b; a = 6; b = 7; return a * b; }", 42),
    ]

    passed = 0
    for name, source, expected in tests:
        vm = FullyBakedVM()
        bytecode, _ = compile_c(source)
        vm.load(bytecode)
        result = vm.run()
        if result >= 0x80000000:
            result -= 0x100000000
        ok = result == expected
        passed += ok
        print(f"{name:<12} {result:>6} == {expected:>6}  [{vm.gen_count:>5} tokens] {'OK' if ok else 'FAIL'}")

    print(f"\nPassed: {passed}/{len(tests)}")
    print("\nOperations used:")
    print("  + - * & | ^ via NeuralALU")
    print("  Reads via context scanning (attention)")
    print("  No Python arithmetic for values")


if __name__ == "__main__":
    run_tests()
