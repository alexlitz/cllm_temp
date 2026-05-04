"""
Software reference VM (8-bit). Used for testing and bytecode generation.

Instruction format: 2 bytes per instruction [opcode, immediate].
Memory: 256 bytes, stack grows down from 0xFE.
"""

from .config import Op, INSTR_WIDTH, VALUE_MASK, STACK_INIT, HEAP_START


class SoftVM:
    def __init__(self, memory_size=256):
        self.memory = bytearray(memory_size)
        self.ax = 0
        self.pc = 0
        self.sp = STACK_INIT
        self.bp = STACK_INIT
        self.running = True
        self.exit_code = 0
        self.output = []
        self._input_buf = []

    def load(self, bytecode, data=None, data_addr=None):
        for i, b in enumerate(bytecode):
            if i < len(self.memory):
                self.memory[i] = b & 0xFF
        if data is not None and data_addr is not None:
            for i, b in enumerate(data):
                addr = data_addr + i
                if addr < len(self.memory):
                    self.memory[addr] = b & 0xFF

    def set_input(self, chars):
        self._input_buf = list(chars)

    def _push(self, val):
        self.sp = (self.sp - 1) & 0xFF
        self.memory[self.sp] = val & 0xFF

    def _pop(self):
        val = self.memory[self.sp]
        self.sp = (self.sp + 1) & 0xFF
        return val

    def _read16(self, addr):
        return self.memory[addr & 0xFF]

    def _read(self, addr):
        return self.memory[addr & 0xFF]

    def _write(self, addr, val):
        self.memory[addr & 0xFF] = val & 0xFF

    def step(self):
        if not self.running:
            return False
        opcode = self.memory[self.pc]
        imm = self.memory[(self.pc + 1) & 0xFF]

        if opcode == Op.IMM:
            self.ax = imm
            self.pc = (self.pc + INSTR_WIDTH) & 0xFF
        elif opcode == Op.JMP:
            self.pc = imm
        elif opcode == Op.JSR:
            self._push((self.pc + INSTR_WIDTH) & 0xFF)
            self.pc = imm
        elif opcode == Op.BZ:
            if self.ax == 0:
                self.pc = imm
            else:
                self.pc = (self.pc + INSTR_WIDTH) & 0xFF
        elif opcode == Op.BNZ:
            if self.ax != 0:
                self.pc = imm
            else:
                self.pc = (self.pc + INSTR_WIDTH) & 0xFF
        elif opcode == Op.ENT:
            self._push(self.bp)
            self.bp = self.sp
            self.sp = (self.sp - imm) & 0xFF
            self.pc = (self.pc + INSTR_WIDTH) & 0xFF
        elif opcode == Op.ADJ:
            self.sp = (self.sp + imm) & 0xFF
            self.pc = (self.pc + INSTR_WIDTH) & 0xFF
        elif opcode == Op.LEV:
            self.sp = self.bp
            self.bp = self._pop()
            self.pc = self._pop()
        elif opcode == Op.LI:
            self.ax = self._read(self.ax)
            self.pc = (self.pc + INSTR_WIDTH) & 0xFF
        elif opcode == Op.LC:
            self.ax = self._read(self.ax)
            self.pc = (self.pc + INSTR_WIDTH) & 0xFF
        elif opcode == Op.SI:
            addr = self._pop()
            self._write(addr, self.ax)
            self.pc = (self.pc + INSTR_WIDTH) & 0xFF
        elif opcode == Op.SC:
            addr = self._pop()
            self._write(addr, self.ax & 0xFF)
            self.pc = (self.pc + INSTR_WIDTH) & 0xFF
        elif opcode == Op.PSH:
            self._push(self.ax)
            self.pc = (self.pc + INSTR_WIDTH) & 0xFF
        elif opcode == Op.OR:
            b = self._pop()
            self.ax = (b | self.ax) & 0xFF
            self.pc = (self.pc + INSTR_WIDTH) & 0xFF
        elif opcode == Op.XOR:
            b = self._pop()
            self.ax = (b ^ self.ax) & 0xFF
            self.pc = (self.pc + INSTR_WIDTH) & 0xFF
        elif opcode == Op.AND:
            b = self._pop()
            self.ax = (b & self.ax) & 0xFF
            self.pc = (self.pc + INSTR_WIDTH) & 0xFF
        elif opcode == Op.EQ:
            b = self._pop()
            self.ax = 1 if b == self.ax else 0
            self.pc = (self.pc + INSTR_WIDTH) & 0xFF
        elif opcode == Op.NE:
            b = self._pop()
            self.ax = 1 if b != self.ax else 0
            self.pc = (self.pc + INSTR_WIDTH) & 0xFF
        elif opcode == Op.LT:
            b = self._pop()
            self.ax = 1 if b < self.ax else 0
            self.pc = (self.pc + INSTR_WIDTH) & 0xFF
        elif opcode == Op.GT:
            b = self._pop()
            self.ax = 1 if b > self.ax else 0
            self.pc = (self.pc + INSTR_WIDTH) & 0xFF
        elif opcode == Op.LE:
            b = self._pop()
            self.ax = 1 if b <= self.ax else 0
            self.pc = (self.pc + INSTR_WIDTH) & 0xFF
        elif opcode == Op.GE:
            b = self._pop()
            self.ax = 1 if b >= self.ax else 0
            self.pc = (self.pc + INSTR_WIDTH) & 0xFF
        elif opcode == Op.SHL:
            b = self._pop()
            self.ax = (b << self.ax) & 0xFF
            self.pc = (self.pc + INSTR_WIDTH) & 0xFF
        elif opcode == Op.SHR:
            b = self._pop()
            self.ax = (b >> self.ax) & 0xFF
            self.pc = (self.pc + INSTR_WIDTH) & 0xFF
        elif opcode == Op.ADD:
            b = self._pop()
            self.ax = (b + self.ax) & 0xFF
            self.pc = (self.pc + INSTR_WIDTH) & 0xFF
        elif opcode == Op.SUB:
            b = self._pop()
            self.ax = (b - self.ax) & 0xFF
            self.pc = (self.pc + INSTR_WIDTH) & 0xFF
        elif opcode == Op.MUL:
            b = self._pop()
            self.ax = (b * self.ax) & 0xFF
            self.pc = (self.pc + INSTR_WIDTH) & 0xFF
        elif opcode == Op.DIV:
            b = self._pop()
            self.ax = (b // self.ax) & 0xFF if self.ax != 0 else 0
            self.pc = (self.pc + INSTR_WIDTH) & 0xFF
        elif opcode == Op.MOD:
            b = self._pop()
            self.ax = (b % self.ax) & 0xFF if self.ax != 0 else 0
            self.pc = (self.pc + INSTR_WIDTH) & 0xFF
        elif opcode == Op.EXIT:
            self.exit_code = self.ax
            self.running = False
        elif opcode == Op.GETCHAR:
            if self._input_buf:
                self.ax = ord(self._input_buf.pop(0)) & 0xFF
            else:
                self.ax = 0xFF
            self.pc = (self.pc + INSTR_WIDTH) & 0xFF
        elif opcode == Op.PUTCHAR:
            self.output.append(chr(self.ax & 0xFF))
            self.pc = (self.pc + INSTR_WIDTH) & 0xFF
        else:
            self.running = False
        return self.running

    def run(self, max_steps=10000):
        for _ in range(max_steps):
            if not self.step():
                break
        return "".join(self.output), self.exit_code


def assemble(instructions):
    bc = bytearray()
    for op, imm in instructions:
        bc.append(op & 0xFF)
        bc.append(imm & 0xFF)
    return bytes(bc)
