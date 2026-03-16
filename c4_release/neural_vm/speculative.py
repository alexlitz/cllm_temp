"""
Speculative decoding for the autoregressive Neural VM.

DraftVM executes C4 bytecode via Python arithmetic and produces 35 draft
tokens per step — the same format the transformer generates. A single
forward pass through the transformer verifies all 35 tokens at once,
giving a 35x reduction in forward passes per step.

Token format per VM step (35 tokens):
    REG_PC  + 4 value bytes     (5 tokens)
    REG_AX  + 4 value bytes     (5 tokens)
    REG_SP  + 4 value bytes     (5 tokens)
    REG_BP  + 4 value bytes     (5 tokens)
    STACK0  + 4 value bytes     (5 tokens)  — value at *SP (stack top)
    MEM     + 4 addr + 4 value  (9 tokens)
    STEP_END or HALT            (1 token)
"""

from .vm_step import Token


def _le_bytes(val):
    """32-bit value -> 4 little-endian byte tokens."""
    v = val & 0xFFFFFFFF
    return [(v >> (i * 8)) & 0xFF for i in range(4)]


def _binop(op, a, b):
    """Execute binary op (pop OP AX), return 32-bit result."""
    if op == 14:    # OR
        return (a | b) & 0xFFFFFFFF
    elif op == 15:  # XOR
        return (a ^ b) & 0xFFFFFFFF
    elif op == 16:  # AND
        return (a & b) & 0xFFFFFFFF
    elif op == 17:  # EQ
        return 1 if a == b else 0
    elif op == 18:  # NE
        return 1 if a != b else 0
    elif op == 19:  # LT
        return 1 if a < b else 0
    elif op == 20:  # GT
        return 1 if a > b else 0
    elif op == 21:  # LE
        return 1 if a <= b else 0
    elif op == 22:  # GE
        return 1 if a >= b else 0
    elif op == 23:  # SHL
        return (a << b) & 0xFFFFFFFF
    elif op == 24:  # SHR
        return (a >> b) & 0xFFFFFFFF
    elif op == 25:  # ADD
        return (a + b) & 0xFFFFFFFF
    elif op == 26:  # SUB
        return (a - b) & 0xFFFFFFFF
    elif op == 27:  # MUL
        return (a * b) & 0xFFFFFFFF
    elif op == 28:  # DIV
        return (a // b) & 0xFFFFFFFF if b != 0 else 0
    elif op == 29:  # MOD
        return (a % b) & 0xFFFFFFFF if b != 0 else 0
    else:
        raise ValueError(f"Unknown binary op: {op}")


class DraftVM:
    """Lightweight C4 VM for speculative token prediction.

    Executes one instruction per step(), producing 35 draft tokens
    matching the autoregressive transformer's output format.
    """

    def __init__(self, bytecode):
        self.code = bytecode  # list of packed instructions
        self.idx = 0          # instruction index
        self.pc = 2           # PC = idx * 5 + 2
        self.ax = 0
        self.sp = 0
        self.bp = 0
        self.memory = {}      # sparse memory dict (addr -> 32-bit value)
        self.halted = False
        self._last_mem_addr = 0
        self._last_mem_val = 0

    def _mem_read(self, addr):
        """Read 32-bit value from memory (ZFOD: uninitialized -> 0)."""
        return self.memory.get(addr & 0xFFFFFFFF, 0)

    def _mem_write(self, addr, val):
        """Write 32-bit value to memory. Track last write for MEM field."""
        addr = addr & 0xFFFFFFFF
        val = val & 0xFFFFFFFF
        self.memory[addr] = val
        self._last_mem_addr = addr
        self._last_mem_val = val

    def step(self):
        """Execute one instruction. Returns True if executed, False if halted/done."""
        if self.idx >= len(self.code) or self.halted:
            return False

        instr = self.code[self.idx]
        op = instr & 0xFF
        imm = instr >> 8
        # Sign extend 24-bit immediate
        if imm >= 0x800000:
            imm -= 0x1000000

        self.idx += 1
        self.pc = self.idx * 5 + 2  # default: advance to next instruction
        self._last_mem_addr = 0
        self._last_mem_val = 0

        if op == 0:    # LEA
            self.ax = (self.bp + imm) & 0xFFFFFFFF
        elif op == 1:  # IMM
            self.ax = imm & 0xFFFFFFFF
        elif op == 2:  # JMP
            self.pc = imm & 0xFFFFFFFF
            self.idx = (imm - 2) // 5
        elif op == 3:  # JSR
            self.sp = (self.sp - 8) & 0xFFFFFFFF
            self._mem_write(self.sp, self.pc)  # push return address
            self.pc = imm & 0xFFFFFFFF
            self.idx = (imm - 2) // 5
        elif op == 4:  # BZ
            if self.ax == 0:
                self.pc = imm & 0xFFFFFFFF
                self.idx = (imm - 2) // 5
        elif op == 5:  # BNZ
            if self.ax != 0:
                self.pc = imm & 0xFFFFFFFF
                self.idx = (imm - 2) // 5
        elif op == 6:  # ENT
            self.sp = (self.sp - 8) & 0xFFFFFFFF
            self._mem_write(self.sp, self.bp)
            self.bp = self.sp
            self.sp = (self.sp - imm) & 0xFFFFFFFF
        elif op == 7:  # ADJ
            self.sp = (self.sp + imm) & 0xFFFFFFFF
        elif op == 8:  # LEV
            self.sp = self.bp
            self.bp = self._mem_read(self.sp)
            self.sp = (self.sp + 8) & 0xFFFFFFFF
            ret_addr = self._mem_read(self.sp)
            self.sp = (self.sp + 8) & 0xFFFFFFFF
            self.pc = ret_addr & 0xFFFFFFFF
            self.idx = (ret_addr - 2) // 5
        elif op == 9:  # LI
            self.ax = self._mem_read(self.ax)
        elif op == 11:  # SI
            addr = self._mem_read(self.sp)
            self.sp = (self.sp + 8) & 0xFFFFFFFF
            self._mem_write(addr, self.ax)
        elif op == 13:  # PSH
            self.sp = (self.sp - 8) & 0xFFFFFFFF
            self._mem_write(self.sp, self.ax)
        # Binary ops that pop stack
        elif 14 <= op <= 29:
            a = self._mem_read(self.sp)
            self.sp = (self.sp + 8) & 0xFFFFFFFF
            self.ax = _binop(op, a, self.ax)
        elif op == 38:  # EXIT
            self.halted = True
        # NOP and other unhandled ops: do nothing (advance PC only)

        return True

    def draft_tokens(self):
        """Encode current state as 35 tokens."""
        tokens = []
        tokens.append(Token.REG_PC)
        tokens.extend(_le_bytes(self.pc))
        tokens.append(Token.REG_AX)
        tokens.extend(_le_bytes(self.ax))
        tokens.append(Token.REG_SP)
        tokens.extend(_le_bytes(self.sp))
        tokens.append(Token.REG_BP)
        tokens.extend(_le_bytes(self.bp))
        tokens.append(Token.STACK0)
        tokens.extend(_le_bytes(self._mem_read(self.sp)))
        tokens.append(Token.MEM)
        tokens.extend(_le_bytes(self._last_mem_addr))
        tokens.extend(_le_bytes(self._last_mem_val))
        tokens.append(Token.HALT if self.halted else Token.STEP_END)
        assert len(tokens) == 35
        return tokens
