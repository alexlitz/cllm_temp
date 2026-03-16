"""
Token Sequence VM - Bytecode as System Prompt Tokens.

Instead of a separate code_tokens list, bytecode is encoded as tokens
in the normal transformer context. This allows standard attention
to fetch instructions.

Token Sequence Layout:
  [CODE_START] <inst0> <inst1> ... <instN> [CODE_END] [DATA_START] ... [DATA_END] [ARGV] ... [/ARGV] [OUTPUT...]

Each instruction token contains:
  - Token type: T_CODE
  - Address: The PC value (index * 5)
  - Opcode: The operation
  - Immediate: The operand value

Instruction fetch uses standard Q/K/V attention:
  - Q = PC encoding (binary position)
  - K = instruction address encodings
  - V = (opcode, immediate) pairs
  - Output = instruction at matching PC
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from dataclasses import dataclass


# =============================================================================
# TOKEN TYPES
# =============================================================================

class TokenType:
    """Token types in the sequence."""
    # Special markers
    CODE_START = 256    # Start of bytecode section
    CODE_END = 257      # End of bytecode section
    DATA_START = 258    # Start of data section
    DATA_END = 259      # End of data section
    ARGV_START = 260    # Start of argv
    ARGV_END = 261      # End of argv

    # Content tokens
    CODE = 262          # Bytecode instruction
    DATA = 263          # Data byte
    ARGV_CHAR = 264     # Argv character
    OUTPUT = 265        # Program output
    INPUT = 266         # User input


# =============================================================================
# BINARY POSITION ENCODING (for exact PC matching)
# =============================================================================

class BinaryEncoder(nn.Module):
    """
    Encode integers as binary vectors for exact matching via dot product.

    Position n is encoded as 32-bit binary:
    - encoding[k] = +1 if bit k is set, -1 otherwise

    Dot product properties:
    - Same position: score = 32 (all bits match)
    - Different positions: score < 32 (some bits differ)
    """

    def __init__(self, num_bits: int = 32, max_value: int = 65536):
        super().__init__()
        self.num_bits = num_bits

        # Precompute encodings for common values
        values = torch.arange(max_value)
        encoding = torch.zeros(max_value, num_bits)
        for k in range(num_bits):
            bit_k = ((values >> k) & 1).float()
            encoding[:, k] = 2 * bit_k - 1  # 0 -> -1, 1 -> +1
        self.register_buffer('encoding', encoding)

    def encode(self, value: torch.Tensor) -> torch.Tensor:
        """Encode integer value(s) as binary vector(s)."""
        return self.encoding[value.long()]

    def encode_int(self, value: int) -> torch.Tensor:
        """Encode single integer."""
        if value < self.encoding.shape[0]:
            return self.encoding[value]
        # Dynamic encoding for larger values
        enc = torch.zeros(self.num_bits)
        for k in range(self.num_bits):
            enc[k] = 2 * ((value >> k) & 1) - 1
        return enc


# =============================================================================
# INSTRUCTION TOKEN
# =============================================================================

@dataclass
class InstructionToken:
    """A bytecode instruction as a token."""
    address: int        # PC value (index * 5)
    opcode: int         # The operation
    immediate: int      # The operand

    def to_tensor(self, encoder: BinaryEncoder) -> torch.Tensor:
        """Encode as tensor: [type, address_enc, opcode, immediate]."""
        # Token layout:
        # [0]: token type (T_CODE = 262)
        # [1:33]: binary address encoding (32 bits)
        # [33]: opcode (0-255)
        # [34]: immediate (as scalar, or could be 32-bit encoded)

        token = torch.zeros(256)  # Fixed token size
        token[0] = TokenType.CODE
        token[1:33] = encoder.encode_int(self.address)
        token[33] = self.opcode
        token[34] = self.immediate & 0xFFFFFFFF
        # Store immediate sign separately
        token[35] = 1.0 if self.immediate < 0 else 0.0
        return token


# =============================================================================
# TOKEN SEQUENCE STATE
# =============================================================================

class TokenSequenceState(nn.Module):
    """
    VM state as a token sequence with standard transformer attention.

    The context contains:
    1. [CODE_START] ... instructions ... [CODE_END]
    2. [DATA_START] ... data bytes ... [DATA_END]
    3. [ARGV_START] ... argv chars ... [ARGV_END]
    4. Register tokens (AX, SP, BP, PC)
    5. Memory write tokens
    6. Output tokens

    All reads use standard causal attention.
    """

    # Token dimensions
    TOKEN_DIM = 256     # Each token is 256 floats
    ADDR_DIM = 32       # Binary address encoding

    def __init__(self, max_code_size: int = 4096, max_data_size: int = 65536):
        super().__init__()
        self.max_code_size = max_code_size
        self.max_data_size = max_data_size

        # Binary encoder for exact position matching
        self.addr_encoder = BinaryEncoder(num_bits=32, max_value=max_data_size)

        # Token sequence (grows during execution)
        self.tokens: List[torch.Tensor] = []

        # Section boundaries (for fast lookup)
        self.code_start_idx = 0
        self.code_end_idx = 0
        self.data_start_idx = 0
        self.data_end_idx = 0
        self.argv_start_idx = 0
        self.argv_end_idx = 0

    def reset(self):
        """Clear all tokens."""
        self.tokens = []
        self.code_start_idx = 0
        self.code_end_idx = 0
        self.data_start_idx = 0
        self.data_end_idx = 0
        self.argv_start_idx = 0
        self.argv_end_idx = 0

    def _marker_token(self, token_type: int) -> torch.Tensor:
        """Create a marker token."""
        token = torch.zeros(self.TOKEN_DIM)
        token[0] = token_type
        return token

    def load_bytecode(self, bytecode: List[int]):
        """
        Load bytecode as tokens in the sequence.

        This is like setting the "system prompt" - it comes first.
        """
        # Add CODE_START marker
        self.code_start_idx = len(self.tokens)
        self.tokens.append(self._marker_token(TokenType.CODE_START))

        # Add each instruction as a token
        for i, instr in enumerate(bytecode):
            op = instr & 0xFF
            imm = instr >> 8
            # Sign extend 32-bit immediate
            if imm >= (1 << 31):
                imm -= (1 << 32)

            inst_token = InstructionToken(
                address=i * 5,  # 5 bytes per instruction
                opcode=op,
                immediate=imm
            )
            self.tokens.append(inst_token.to_tensor(self.addr_encoder))

        # Add CODE_END marker
        self.code_end_idx = len(self.tokens)
        self.tokens.append(self._marker_token(TokenType.CODE_END))

    def load_data(self, data: bytes):
        """Load initial data section."""
        # Add DATA_START marker
        self.data_start_idx = len(self.tokens)
        self.tokens.append(self._marker_token(TokenType.DATA_START))

        # Add data bytes as tokens
        for i, b in enumerate(data):
            token = torch.zeros(self.TOKEN_DIM)
            token[0] = TokenType.DATA
            token[1:33] = self.addr_encoder.encode_int(0x10000 + i)  # Data starts at 0x10000
            token[33] = b
            self.tokens.append(token)

        # Add DATA_END marker
        self.data_end_idx = len(self.tokens)
        self.tokens.append(self._marker_token(TokenType.DATA_END))

    def load_argv(self, argv: List[str]):
        """Load argv as tokens (like user providing arguments)."""
        # Add ARGV_START marker
        self.argv_start_idx = len(self.tokens)
        self.tokens.append(self._marker_token(TokenType.ARGV_START))

        # Add argv strings
        char_idx = 0
        for arg in argv:
            for c in arg:
                token = torch.zeros(self.TOKEN_DIM)
                token[0] = TokenType.ARGV_CHAR
                token[1:33] = self.addr_encoder.encode_int(char_idx)
                token[33] = ord(c)
                self.tokens.append(token)
                char_idx += 1
            # Null terminator between args
            token = torch.zeros(self.TOKEN_DIM)
            token[0] = TokenType.ARGV_CHAR
            token[1:33] = self.addr_encoder.encode_int(char_idx)
            token[33] = 0
            self.tokens.append(token)
            char_idx += 1

        # Add ARGV_END marker
        self.argv_end_idx = len(self.tokens)
        self.tokens.append(self._marker_token(TokenType.ARGV_END))

    def get_tokens_tensor(self) -> torch.Tensor:
        """Get all tokens as a single tensor."""
        if not self.tokens:
            return torch.zeros(0, self.TOKEN_DIM)
        return torch.stack(self.tokens)

    def fetch_instruction(self, pc: int) -> Tuple[int, int]:
        """
        Fetch instruction at PC using standard attention.

        Q = binary encoding of PC
        K = binary encodings of instruction addresses (from CODE section)
        V = (opcode, immediate) pairs

        Returns (opcode, immediate).
        """
        tokens = self.get_tokens_tensor()

        # Get code section tokens (between CODE_START and CODE_END)
        code_tokens = tokens[self.code_start_idx + 1:self.code_end_idx]

        if code_tokens.shape[0] == 0:
            return 38, 0  # EXIT

        # Query: PC address encoding
        q = self.addr_encoder.encode_int(pc)  # [32]

        # Keys: instruction address encodings (columns 1:33)
        k = code_tokens[:, 1:33]  # [num_instr, 32]

        # Standard attention: scores = Q · K^T
        scores = torch.matmul(k, q)  # [num_instr]

        # Scale for sharp attention (binary match gives score=32)
        # Use temperature to make it sharp
        temperature = 0.1
        weights = F.softmax(scores / temperature, dim=0)

        # Values: opcode and immediate (columns 33, 34)
        opcodes = code_tokens[:, 33]  # [num_instr]
        immediates = code_tokens[:, 34]  # [num_instr]
        signs = code_tokens[:, 35]  # [num_instr]

        # Weighted sum (sharp attention means mostly one instruction)
        opcode = (weights * opcodes).sum()
        immediate = (weights * immediates).sum()
        sign = (weights * signs).sum()

        # Decode
        op = int(round(opcode.item()))
        imm = int(round(immediate.item()))
        if sign.item() > 0.5:
            imm = -imm

        return op, imm

    def read_memory(self, addr: int) -> int:
        """
        Read memory via attention over data/memory-write tokens.

        Searches:
        1. Memory write tokens (newest first - later in sequence wins)
        2. Data section tokens
        """
        tokens = self.get_tokens_tensor()

        # Query: address encoding
        q = self.addr_encoder.encode_int(addr)

        # Find all memory-related tokens (DATA or memory writes)
        # For now, scan data section
        data_tokens = tokens[self.data_start_idx + 1:self.data_end_idx]

        if data_tokens.shape[0] == 0:
            return 0

        # Keys: address encodings
        k = data_tokens[:, 1:33]

        # Attention
        scores = torch.matmul(k, q)

        # Add position bias (later tokens win for same address)
        positions = torch.arange(data_tokens.shape[0], dtype=torch.float32)
        scores = scores + positions * 0.01

        weights = F.softmax(scores / 0.1, dim=0)

        # Values: data bytes
        values = data_tokens[:, 33]
        result = (weights * values).sum()

        return int(round(result.item())) & 0xFF

    def write_memory(self, addr: int, value: int):
        """Append a memory write token (later writes shadow earlier)."""
        token = torch.zeros(self.TOKEN_DIM)
        token[0] = TokenType.DATA  # Reuse DATA type for memory
        token[1:33] = self.addr_encoder.encode_int(addr)
        token[33] = value & 0xFF
        self.tokens.append(token)

    def write_output(self, char: int):
        """Append an output token."""
        token = torch.zeros(self.TOKEN_DIM)
        token[0] = TokenType.OUTPUT
        token[33] = char & 0xFF
        self.tokens.append(token)


# =============================================================================
# TOKEN SEQUENCE VM
# =============================================================================

class TokenSequenceVM(nn.Module):
    """
    VM where all state is in the token sequence.

    - Bytecode is tokens (like system prompt)
    - Data/memory is tokens
    - Argv is tokens
    - Registers are tracked (could also be tokens)
    - All reads use standard transformer attention
    """

    # Opcodes (same as C4)
    LEA, IMM, JMP, JSR, BZ, BNZ, ENT, ADJ, LEV = 0, 1, 2, 3, 4, 5, 6, 7, 8
    LI, LC, SI, SC, PSH = 9, 10, 11, 12, 13
    OR, XOR, AND, EQ, NE, LT, GT, LE, GE = 14, 15, 16, 17, 18, 19, 20, 21, 22
    SHL, SHR, ADD, SUB, MUL, DIV, MOD = 23, 24, 25, 26, 27, 28, 29
    EXIT = 38

    def __init__(self):
        super().__init__()
        self.state = TokenSequenceState()
        self.reset()

    def reset(self):
        """Reset VM state."""
        self.state.reset()
        self.ax = 0
        self.sp = 0x10000
        self.bp = 0x10000
        self.pc = 0
        self.halted = False
        self.stack = {}  # SP-indexed memory

    def load(self, bytecode: List[int], data: Optional[bytes] = None,
             argv: Optional[List[str]] = None):
        """Load program as tokens."""
        self.reset()

        # Bytecode becomes the "system prompt"
        self.state.load_bytecode(bytecode)

        # Data section
        if data:
            self.state.load_data(data)

        # Argv (like user input before program starts)
        if argv:
            self.state.load_argv(argv)

    def _push(self, value: int):
        """Push to stack."""
        self.sp -= 8
        self.stack[self.sp] = value

    def _pop(self) -> int:
        """Pop from stack."""
        value = self.stack.get(self.sp, 0)
        self.sp += 8
        return value

    def _signed(self, val: int) -> int:
        """Convert to signed 32-bit."""
        if val >= 0x80000000:
            return val - 0x100000000
        return val

    def step(self) -> bool:
        """Execute one instruction using attention over token sequence."""
        if self.halted:
            return False

        # Fetch instruction via attention
        op, imm = self.state.fetch_instruction(self.pc)
        self.pc += 5  # 5 bytes per instruction

        # Execute
        if op == self.IMM:
            self.ax = imm

        elif op == self.LEA:
            self.ax = self.bp + imm

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
            self.ax = self.state.read_memory(self.ax)
            # For int, read 4 bytes
            val = 0
            for i in range(4):
                val |= self.state.read_memory(self.ax + i) << (i * 8)
            self.ax = val

        elif op == self.LC:
            self.ax = self.state.read_memory(self.ax) & 0xFF

        elif op == self.SI:
            addr = self._pop()
            # Store 4 bytes
            for i in range(4):
                self.state.write_memory(addr + i, (self.ax >> (i * 8)) & 0xFF)

        elif op == self.SC:
            addr = self._pop()
            self.state.write_memory(addr, self.ax & 0xFF)

        elif op == self.PSH:
            self._push(self.ax)

        elif op == self.ADD:
            self.ax = (self._pop() + self.ax) & 0xFFFFFFFF

        elif op == self.SUB:
            self.ax = (self._pop() - self.ax) & 0xFFFFFFFF

        elif op == self.MUL:
            self.ax = (self._pop() * self.ax) & 0xFFFFFFFF

        elif op == self.DIV:
            b = self.ax
            a = self._pop()
            self.ax = a // b if b != 0 else 0

        elif op == self.MOD:
            b = self.ax
            a = self._pop()
            self.ax = a % b if b != 0 else 0

        elif op == self.AND:
            self.ax = self._pop() & self.ax

        elif op == self.OR:
            self.ax = self._pop() | self.ax

        elif op == self.XOR:
            self.ax = self._pop() ^ self.ax

        elif op == self.SHL:
            self.ax = (self._pop() << self.ax) & 0xFFFFFFFF

        elif op == self.SHR:
            self.ax = self._pop() >> self.ax

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

        elif op == self.EXIT:
            self.halted = True
            return False

        return True

    def run(self, max_steps: int = 1000000) -> int:
        """Run until halted."""
        steps = 0
        while steps < max_steps and self.step():
            steps += 1
        return self.ax


# =============================================================================
# EXAMPLE: Show token sequence layout
# =============================================================================

def demo_token_sequence():
    """Show what the token sequence looks like."""

    # Simple program: return 42
    # IMM 42; EXIT
    bytecode = [
        1 | (42 << 8),   # IMM 42
        38 | (0 << 8),   # EXIT
    ]

    vm = TokenSequenceVM()
    vm.load(bytecode, data=b"Hello", argv=["prog", "arg1"])

    print("=" * 70)
    print("TOKEN SEQUENCE LAYOUT")
    print("=" * 70)

    tokens = vm.state.get_tokens_tensor()
    print(f"Total tokens: {tokens.shape[0]}")

    type_names = {
        TokenType.CODE_START: "CODE_START",
        TokenType.CODE_END: "CODE_END",
        TokenType.DATA_START: "DATA_START",
        TokenType.DATA_END: "DATA_END",
        TokenType.ARGV_START: "ARGV_START",
        TokenType.ARGV_END: "ARGV_END",
        TokenType.CODE: "CODE",
        TokenType.DATA: "DATA",
        TokenType.ARGV_CHAR: "ARGV_CHAR",
    }

    for i, token in enumerate(vm.state.tokens):
        ttype = int(token[0].item())
        name = type_names.get(ttype, f"TYPE_{ttype}")

        if ttype == TokenType.CODE:
            op = int(token[33].item())
            imm = int(token[34].item())
            print(f"  [{i:3d}] {name:12s} op={op:2d} imm={imm}")
        elif ttype == TokenType.DATA:
            val = int(token[33].item())
            print(f"  [{i:3d}] {name:12s} val={val} ({chr(val) if 32 <= val < 127 else '?'})")
        elif ttype == TokenType.ARGV_CHAR:
            val = int(token[33].item())
            print(f"  [{i:3d}] {name:12s} char={val} ({chr(val) if val > 0 else 'NUL'})")
        else:
            print(f"  [{i:3d}] {name}")

    print()
    print("Running program...")
    result = vm.run()
    print(f"Result: {result}")

    return vm


if __name__ == "__main__":
    demo_token_sequence()
