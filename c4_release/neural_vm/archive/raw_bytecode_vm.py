"""
Raw Bytecode VM - Matching C4's exact bytecode format with attention-based fetching.

C4 bytecode format (from c4.c):
  - Each instruction "word" is sizeof(int) = 4 bytes
  - i = *pc++  fetches opcode word
  - For opcodes 0-7 (LEA, IMM, JMP, JSR, BZ, BNZ, ENT, ADJ): *pc++ fetches immediate
  - Variable-length: 1 word (4 bytes) or 2 words (8 bytes)

Instruction fetching uses ACTUAL attention:
  - Q = binary encoding of PC (target word index)
  - K = binary encodings of all word positions
  - V = word values (4 bytes each)
  - output = softmax(Q · K^T) @ V

Jump targets are word indices (PC position), just like C4's pointer arithmetic.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class BinaryPositionEncoder(nn.Module):
    """
    Encode word indices as binary vectors for exact position matching via attention.

    Position n is encoded as 32-bit binary:
    - encoding[k] = +1 if bit k is set, -1 otherwise

    Dot product properties:
    - Same position: score = 32 (all bits match)
    - Different by 1 bit: score = 30
    - Random positions: score ≈ 0

    After softmax with temperature scaling, this gives near-one-hot attention.
    """

    def __init__(self, num_bits: int = 16):
        super().__init__()
        self.num_bits = num_bits

    def encode(self, position: int) -> torch.Tensor:
        """Encode single position as binary vector."""
        enc = torch.zeros(self.num_bits)
        for k in range(self.num_bits):
            bit = (position >> k) & 1
            enc[k] = 2.0 * bit - 1.0  # 0 -> -1, 1 -> +1
        return enc

    def encode_batch(self, positions: torch.Tensor) -> torch.Tensor:
        """Encode multiple positions."""
        batch_size = positions.shape[0]
        enc = torch.zeros(batch_size, self.num_bits)
        for k in range(self.num_bits):
            bits = ((positions >> k) & 1).float()
            enc[:, k] = 2.0 * bits - 1.0
        return enc


class RawBytecodeVM(nn.Module):
    """
    VM matching C4's exact bytecode format with attention-based instruction fetch.

    Each code word is 4 bytes.
    PC counts in words (like C4's int* pc).
    Instruction fetching uses actual Q/K/V attention.
    """

    # Special tokens
    SEP = 256           # Section separator
    PAD = 257           # Padding

    # Word size in bytes (matching C4's int)
    WORD_SIZE = 4

    # Attention temperature (higher = sharper attention)
    ATTENTION_TEMP = 4.0

    # Opcodes (from C4)
    LEA, IMM, JMP, JSR, BZ, BNZ, ENT, ADJ = 0, 1, 2, 3, 4, 5, 6, 7
    LEV, LI, LC, SI, SC, PSH = 8, 9, 10, 11, 12, 13
    OR, XOR, AND, EQ, NE, LT, GT, LE, GE = 14, 15, 16, 17, 18, 19, 20, 21, 22
    SHL, SHR, ADD, SUB, MUL, DIV, MOD = 23, 24, 25, 26, 27, 28, 29
    OPEN, READ, CLOS, PRTF, MALC, FREE, MSET, MCMP, EXIT = 30, 31, 32, 33, 34, 35, 36, 37, 38

    # Opcodes 0-7 have immediates (matching C4)
    OPCODES_WITH_IMM = {0, 1, 2, 3, 4, 5, 6, 7}  # LEA, IMM, JMP, JSR, BZ, BNZ, ENT, ADJ

    def __init__(self):
        super().__init__()
        self.pos_encoder = BinaryPositionEncoder(num_bits=16)
        self.reset()

    def reset(self):
        """Reset VM state."""
        self.tokens = []        # The raw byte sequence
        self.code_start = 0     # Start of code (in bytes)
        self.code_end = 0       # End of code (in bytes)
        self.code_words = 0     # Number of code words
        self.data_start = 0     # Where data starts
        self.ax = 0
        self.sp = 0x10000
        self.bp = 0x10000
        self.pc = 0             # Program counter (in WORDS, like C4)
        self.halted = False
        self.stack = {}
        self.memory = {}

        # Attention KV cache for instruction fetch
        self.word_keys = None    # [num_words, num_bits] - position encodings
        self.word_values = None  # [num_words] - word values

    def has_immediate(self, opcode: int) -> bool:
        """Check if opcode has an immediate operand (matching C4)."""
        return opcode in self.OPCODES_WITH_IMM

    def load(self, bytecode: List[int], data: Optional[bytes] = None,
             argv: Optional[List[str]] = None):
        """
        Load program matching C4's format.

        Builds attention KV cache for instruction fetching.
        """
        self.reset()
        self.tokens = []

        # Build word list (each word = 4 bytes)
        words = []

        for instr in bytecode:
            op = instr & 0xFF
            imm = instr >> 8

            words.append(op)

            if self.has_immediate(op):
                # Sign extend negative values properly
                if imm >= (1 << 31):
                    imm = imm - (1 << 32)
                words.append(imm & 0xFFFFFFFF)  # Store as unsigned

        self.code_words = len(words)

        # Build attention KV cache
        # K = binary position encodings for each word
        # V = word values
        self.word_keys = torch.stack([
            self.pos_encoder.encode(i) for i in range(self.code_words)
        ])  # [num_words, num_bits]

        self.word_values = torch.tensor(words, dtype=torch.float32)  # [num_words]

        # Convert words to bytes (little-endian) for raw token sequence
        self.code_start = 0
        for w in words:
            for i in range(4):
                self.tokens.append((w >> (i * 8)) & 0xFF)

        self.code_end = len(self.tokens)

        # Separator
        self.tokens.append(self.SEP)

        # Data section
        self.data_start = len(self.tokens)
        if data:
            for b in data:
                self.tokens.append(b)
                self.memory[0x10000 + len(self.tokens) - self.data_start - 1] = b

        # Separator
        self.tokens.append(self.SEP)

        # Argv: argc (4 bytes little-endian) + null-terminated strings
        argc = len(argv) if argv else 0
        for i in range(4):
            self.tokens.append((argc >> (i * 8)) & 0xFF)

        if argv:
            for arg in argv:
                for c in arg:
                    self.tokens.append(ord(c))
                self.tokens.append(0)  # null terminator

    def get_prompt(self) -> List[int]:
        """Get the raw token sequence (the 'prompt')."""
        return self.tokens.copy()

    def get_prompt_string(self) -> str:
        """Get prompt as hex string for display."""
        parts = []
        for i, t in enumerate(self.tokens):
            if t == self.SEP:
                parts.append("|SEP|")
            elif t < 32 or t > 126:
                parts.append(f"[{t:02x}]")
            else:
                parts.append(chr(t))
        return ''.join(parts)

    def _fetch_word_attention(self, word_index: int) -> Tuple[int, torch.Tensor]:
        """
        Fetch word at position using ACTUAL attention mechanism.

        Q = binary encoding of target word index
        K = binary encodings of all word positions
        V = word values

        score[i] = Q · K[i]  (dot product)
        attn = softmax(score * temperature)
        output = attn @ V

        Returns: (word_value, attention_weights)
        """
        if word_index < 0 or word_index >= self.code_words:
            return 0, torch.zeros(self.code_words if self.code_words > 0 else 1)

        # Q = query for target position
        query = self.pos_encoder.encode(word_index)  # [num_bits]

        # Compute attention scores: Q · K^T
        # For binary encodings: score = num_bits when positions match
        scores = torch.matmul(self.word_keys, query)  # [num_words]

        # Scale by temperature for sharper attention
        scores = scores * self.ATTENTION_TEMP

        # Softmax to get attention weights
        attn_weights = F.softmax(scores, dim=0)  # [num_words]

        # Weighted sum of values
        output = torch.sum(attn_weights * self.word_values)  # scalar

        # Round to nearest integer (attention gives exact match for binary encoding)
        word_value = int(torch.round(output).item())

        return word_value, attn_weights

    def _fetch_word(self, word_index: int) -> int:
        """Fetch word using attention (wrapper for compatibility)."""
        value, _ = self._fetch_word_attention(word_index)
        return value

    def _fetch_word_signed(self, word_index: int) -> int:
        """Fetch word and sign-extend to Python int."""
        val = self._fetch_word(word_index)
        if val >= (1 << 31):
            val -= (1 << 32)
        return val

    def _push(self, val):
        self.sp -= 8
        self.stack[self.sp] = val

    def _pop(self) -> int:
        val = self.stack.get(self.sp, 0)
        self.sp += 8
        return val

    def _signed(self, val):
        if val >= 0x80000000:
            return val - 0x100000000
        return val

    def step(self, debug: bool = False) -> bool:
        """Execute one instruction (matching C4's while(1) loop)."""
        if self.halted:
            return False

        if self.pc >= self.code_words:
            self.halted = True
            return False

        # Fetch opcode using attention: i = *pc++
        op, attn_weights = self._fetch_word_attention(self.pc)

        if debug:
            print(f"  PC={self.pc}: fetch opcode, attn_max={attn_weights.max():.4f} at pos {attn_weights.argmax()}")

        self.pc += 1

        # For opcodes with immediates: imm = *pc++
        imm = 0
        if self.has_immediate(op):
            imm = self._fetch_word_signed(self.pc)
            if debug:
                print(f"  PC={self.pc}: fetch immediate={imm}")
            self.pc += 1

        # Execute (matching C4 exactly)
        if op == self.LEA:
            self.ax = self.bp + imm
        elif op == self.IMM:
            self.ax = imm
        elif op == self.JMP:
            self.pc = imm
        elif op == self.JSR:
            self._push(self.pc)
            self.pc = imm
        elif op == self.BZ:
            if self.ax == 0:
                self.pc = imm
        elif op == self.BNZ:
            if self.ax != 0:
                self.pc = imm
        elif op == self.ENT:
            self._push(self.bp)
            self.bp = self.sp
            self.sp -= imm
        elif op == self.ADJ:
            self.sp += imm
        elif op == self.LEV:
            self.sp = self.bp
            self.bp = self._pop()
            self.pc = self._pop()
        elif op == self.LI:
            self.ax = self.memory.get(self.ax, 0)
        elif op == self.LC:
            self.ax = self.memory.get(self.ax, 0) & 0xFF
        elif op == self.SI:
            addr = self._pop()
            self.memory[addr] = self.ax
        elif op == self.SC:
            addr = self._pop()
            self.memory[addr] = self.ax & 0xFF
        elif op == self.PSH:
            self._push(self.ax)

        # Binary operations
        elif op == self.OR:
            self.ax = self._pop() | self.ax
        elif op == self.XOR:
            self.ax = self._pop() ^ self.ax
        elif op == self.AND:
            self.ax = self._pop() & self.ax
        elif op == self.EQ:
            self.ax = 1 if self._pop() == self.ax else 0
        elif op == self.NE:
            self.ax = 1 if self._pop() != self.ax else 0
        elif op == self.LT:
            self.ax = 1 if self._signed(self._pop()) < self._signed(self.ax) else 0
        elif op == self.GT:
            self.ax = 1 if self._signed(self._pop()) > self._signed(self.ax) else 0
        elif op == self.LE:
            self.ax = 1 if self._signed(self._pop()) <= self._signed(self.ax) else 0
        elif op == self.GE:
            self.ax = 1 if self._signed(self._pop()) >= self._signed(self.ax) else 0
        elif op == self.SHL:
            self.ax = (self._pop() << self.ax) & 0xFFFFFFFF
        elif op == self.SHR:
            self.ax = self._pop() >> self.ax
        elif op == self.ADD:
            self.ax = (self._pop() + self.ax) & 0xFFFFFFFF
        elif op == self.SUB:
            self.ax = (self._pop() - self.ax) & 0xFFFFFFFF
        elif op == self.MUL:
            self.ax = (self._pop() * self.ax) & 0xFFFFFFFF
        elif op == self.DIV:
            b = self.ax
            a = self._pop()
            self.ax = a // b if b else 0
        elif op == self.MOD:
            b = self.ax
            a = self._pop()
            self.ax = a % b if b else 0

        # System calls (stubs)
        elif op == self.EXIT:
            self.halted = True
            return False

        return True

    def run(self, max_steps: int = 1000000, debug: bool = False) -> int:
        """Run until halted."""
        steps = 0
        while steps < max_steps and self.step(debug=debug):
            steps += 1
        return self.ax


def demo():
    """Show raw bytecode with attention-based instruction fetch."""

    # Program: return 6 * 7
    bytecode = [
        1 | (6 << 8),    # IMM 6
        13,              # PSH
        1 | (7 << 8),    # IMM 7
        27,              # MUL
        38,              # EXIT
    ]

    vm = RawBytecodeVM()
    vm.load(bytecode, data=b"Hello", argv=["test"])

    print("=" * 70)
    print("RAW BYTECODE (C4 Format + Attention-Based Instruction Fetch)")
    print("=" * 70)

    prompt = vm.get_prompt()
    print(f"Prompt length: {len(prompt)} bytes")
    print(f"Code section: bytes 0-{vm.code_end-1} ({vm.code_words} words)")
    print(f"Attention keys shape: {vm.word_keys.shape}")
    print(f"Attention values shape: {vm.word_values.shape}")
    print()

    # Show word dump
    print("Word dump:")
    op_names = {
        0: 'LEA', 1: 'IMM', 2: 'JMP', 3: 'JSR', 4: 'BZ', 5: 'BNZ', 6: 'ENT', 7: 'ADJ',
        8: 'LEV', 9: 'LI', 10: 'LC', 11: 'SI', 12: 'SC', 13: 'PSH',
        14: 'OR', 15: 'XOR', 16: 'AND', 17: 'EQ', 18: 'NE', 19: 'LT', 20: 'GT', 21: 'LE', 22: 'GE',
        23: 'SHL', 24: 'SHR', 25: 'ADD', 26: 'SUB', 27: 'MUL', 28: 'DIV', 29: 'MOD',
        38: 'EXIT'
    }

    w = 0
    while w < vm.code_words:
        word, attn = vm._fetch_word_attention(w)
        op = word

        if vm.has_immediate(op):
            imm = vm._fetch_word_signed(w + 1)
            print(f"  [{w:2d}] {op_names.get(op, f'OP{op}'):4s} {imm:<10} (attn_max={attn.max():.4f})")
            w += 2
        else:
            print(f"  [{w:2d}] {op_names.get(op, f'OP{op}'):4s}            (attn_max={attn.max():.4f})")
            w += 1

    # Demonstrate attention sharpness
    print()
    print("Attention weight distribution (fetching word 3):")
    _, attn = vm._fetch_word_attention(3)
    for i, a in enumerate(attn):
        bar = '#' * int(a * 50)
        print(f"  word[{i}]: {a:.4f} {bar}")

    print()
    print("Running with debug (showing attention):")
    vm.reset()
    vm.load(bytecode)  # Reload to reset PC
    result = vm.run(debug=True)
    print(f"Result: {result}")

    return vm


if __name__ == "__main__":
    demo()
