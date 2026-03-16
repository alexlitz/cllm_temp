#!/usr/bin/env python3
"""
LLM I/O for C4 VM

Simple tag-based I/O protocol for LLM generation:
  <NEED_INPUT/>  - Pause, wait for user input
  <PROGRAM_END/> - Program finished

The token stream looks like:
  Hello! What's your name?
  > <NEED_INPUT/>Alice
  Nice to meet you, Alice!
  <PROGRAM_END/>
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import Optional, List, Tuple
from dataclasses import dataclass, field
from neural_vm import Opcode


class ExtendedOpcode:
    """Opcodes from compiler."""
    GETCHAR = 64
    PUTCHAR = 65
    EXIT = 38
    MALC = 34


@dataclass
class LLMSession:
    """
    VM session with LLM-style I/O.

    Generates output until input is needed, then pauses.
    User input is appended to the stream and generation continues.
    """

    # VM state
    memory: dict = field(default_factory=dict)
    sp: int = 0x10000
    bp: int = 0x10000
    ax: int = 0
    pc: int = 0
    halted: bool = False
    code: List[tuple] = field(default_factory=list)
    heap: int = 0x20000

    # I/O state
    output_buffer: str = ""
    input_buffer: List[int] = field(default_factory=list)
    waiting_for_input: bool = False
    finished: bool = False

    # LLM mode
    llm_mode: bool = True

    def load(self, bytecode: List[int], data: Optional[bytes] = None):
        """Load bytecode into VM."""
        self.code = []
        for instr in bytecode:
            op = instr & 0xFF
            imm = instr >> 8
            if imm >= (1 << 55):
                imm -= (1 << 56)
            self.code.append((op, imm))

        if data:
            for i, b in enumerate(data):
                self.memory[0x10000 + i] = b

    def _signed(self, val):
        """Convert to signed 32-bit."""
        if val >= 0x80000000:
            return val - 0x100000000
        return val

    def run_until_input_or_done(self) -> str:
        """
        Run VM until it needs input or finishes.

        Returns the output generated (with <NEED_INPUT/> or <PROGRAM_END/> tag).
        """
        self.output_buffer = ""
        self.waiting_for_input = False
        max_steps = 1_000_000
        steps = 0

        while steps < max_steps and not self.halted:
            result = self._step()

            if result == "NEED_INPUT":
                self.waiting_for_input = True
                if self.llm_mode:
                    self.output_buffer += "<NEED_INPUT/>"
                return self.output_buffer

            elif result == "PROGRAM_END":
                self.finished = True
                self.halted = True
                if self.llm_mode:
                    self.output_buffer += "<PROGRAM_END/>"
                return self.output_buffer

            steps += 1

        return self.output_buffer

    def provide_input(self, text: str) -> str:
        """
        Provide user input and continue running.

        Args:
            text: User's input (without newline - we add it)

        Returns:
            Output generated until next input needed or program ends.
        """
        if not self.waiting_for_input:
            return "[Not waiting for input]"

        if self.finished:
            return "[Program ended]"

        # Buffer the input characters
        for c in text + '\n':
            self.input_buffer.append(ord(c))

        self.waiting_for_input = False
        return self.run_until_input_or_done()

    def _step(self) -> Optional[str]:
        """Execute one instruction. Returns 'NEED_INPUT' or 'PROGRAM_END' or None."""

        if self.halted:
            return "PROGRAM_END"

        pc_idx = self.pc // 8
        if pc_idx >= len(self.code):
            self.halted = True
            return "PROGRAM_END"

        op, imm = self.code[pc_idx]
        self.pc += 8

        # Standard opcodes
        if op == Opcode.LEA:
            self.ax = self.bp + imm
        elif op == Opcode.IMM:
            self.ax = imm
        elif op == Opcode.JMP:
            self.pc = imm
        elif op == Opcode.JSR:
            self.sp -= 8
            self.memory[self.sp] = self.pc
            self.pc = imm
        elif op == Opcode.BZ:
            if self.ax == 0:
                self.pc = imm
        elif op == Opcode.BNZ:
            if self.ax != 0:
                self.pc = imm
        elif op == Opcode.ENT:
            self.sp -= 8
            self.memory[self.sp] = self.bp
            self.bp = self.sp
            self.sp -= imm
        elif op == Opcode.ADJ:
            self.sp += imm
        elif op == Opcode.LEV:
            self.sp = self.bp
            self.bp = self.memory.get(self.sp, 0)
            self.sp += 8
            self.pc = self.memory.get(self.sp, 0)
            self.sp += 8
        elif op == Opcode.LI:
            self.ax = self.memory.get(self.ax, 0)
        elif op == Opcode.LC:
            self.ax = self.memory.get(self.ax, 0) & 0xFF
        elif op == Opcode.SI:
            addr = self.memory.get(self.sp, 0)
            self.sp += 8
            self.memory[addr] = self.ax
        elif op == Opcode.SC:
            addr = self.memory.get(self.sp, 0)
            self.sp += 8
            self.memory[addr] = self.ax & 0xFF
        elif op == Opcode.PSH:
            self.sp -= 8
            self.memory[self.sp] = self.ax

        # Arithmetic
        elif op == Opcode.ADD:
            a = self.memory.get(self.sp, 0)
            self.sp += 8
            self.ax = (a + self.ax) & 0xFFFFFFFF
        elif op == Opcode.SUB:
            a = self.memory.get(self.sp, 0)
            self.sp += 8
            self.ax = (a - self.ax) & 0xFFFFFFFF
        elif op == Opcode.MUL:
            a = self.memory.get(self.sp, 0)
            self.sp += 8
            self.ax = (a * self.ax) & 0xFFFFFFFF
        elif op == Opcode.DIV:
            a = self.memory.get(self.sp, 0)
            self.sp += 8
            self.ax = a // self.ax if self.ax != 0 else 0
        elif op == Opcode.MOD:
            a = self.memory.get(self.sp, 0)
            self.sp += 8
            self.ax = a % self.ax if self.ax != 0 else 0

        # Bitwise
        elif op == Opcode.AND:
            a = self.memory.get(self.sp, 0)
            self.sp += 8
            self.ax = a & self.ax
        elif op == Opcode.OR:
            a = self.memory.get(self.sp, 0)
            self.sp += 8
            self.ax = a | self.ax
        elif op == Opcode.XOR:
            a = self.memory.get(self.sp, 0)
            self.sp += 8
            self.ax = a ^ self.ax
        elif op == Opcode.SHL:
            a = self.memory.get(self.sp, 0)
            self.sp += 8
            self.ax = (a << self.ax) & 0xFFFFFFFF
        elif op == Opcode.SHR:
            a = self.memory.get(self.sp, 0)
            self.sp += 8
            self.ax = a >> self.ax

        # Comparison (signed)
        elif op == Opcode.EQ:
            a = self.memory.get(self.sp, 0)
            self.sp += 8
            self.ax = 1 if a == self.ax else 0
        elif op == Opcode.NE:
            a = self.memory.get(self.sp, 0)
            self.sp += 8
            self.ax = 1 if a != self.ax else 0
        elif op == Opcode.LT:
            a = self._signed(self.memory.get(self.sp, 0))
            self.sp += 8
            self.ax = 1 if a < self._signed(self.ax) else 0
        elif op == Opcode.GT:
            a = self._signed(self.memory.get(self.sp, 0))
            self.sp += 8
            self.ax = 1 if a > self._signed(self.ax) else 0
        elif op == Opcode.LE:
            a = self._signed(self.memory.get(self.sp, 0))
            self.sp += 8
            self.ax = 1 if a <= self._signed(self.ax) else 0
        elif op == Opcode.GE:
            a = self._signed(self.memory.get(self.sp, 0))
            self.sp += 8
            self.ax = 1 if a >= self._signed(self.ax) else 0

        # === I/O ===
        elif op == Opcode.GETC or op == ExtendedOpcode.GETCHAR:
            # Need input
            if self.input_buffer:
                self.ax = self.input_buffer.pop(0)
            else:
                # No buffered input - need to pause
                self.pc -= 8  # Re-execute this instruction after input
                return "NEED_INPUT"

        elif op == Opcode.PUTC or op == ExtendedOpcode.PUTCHAR:
            char = self.memory.get(self.sp, 0) & 0xFF
            self.output_buffer += chr(char)

        elif op == Opcode.EXIT or op == ExtendedOpcode.EXIT:
            self.ax = self.memory.get(self.sp, 0) if self.sp in self.memory else self.ax
            self.halted = True
            return "PROGRAM_END"

        elif op == Opcode.MALC or op == ExtendedOpcode.MALC:
            size = self.memory.get(self.sp, 0)
            self.ax = self.heap
            self.heap += size

        return None


def run_eliza_llm():
    """Run Eliza with LLM-style I/O."""
    from src.compiler import compile_c

    c_file = os.path.join(os.path.dirname(__file__), 'eliza_simple.c')
    with open(c_file, 'r') as f:
        source = f.read()

    bytecode, data = compile_c(source)

    session = LLMSession()
    session.load(bytecode, data)

    # Run until first input needed
    output = session.run_until_input_or_done()
    print(output, end='')

    # Chat loop
    while not session.finished:
        if session.waiting_for_input:
            try:
                user_input = input()
            except (EOFError, KeyboardInterrupt):
                break

            # Show input in stream (for demonstration)
            print(user_input)

            output = session.provide_input(user_input)
            print(output, end='')

    print("\n[Session complete]")


def demo_token_stream():
    """Show what the token stream looks like."""
    from src.compiler import compile_c

    c_file = os.path.join(os.path.dirname(__file__), 'eliza_simple.c')
    with open(c_file, 'r') as f:
        source = f.read()

    bytecode, data = compile_c(source)

    session = LLMSession()
    session.load(bytecode, data)

    # Simulated conversation
    inputs = ["hello", "i feel sad", "bye"]
    input_idx = 0

    full_stream = ""

    output = session.run_until_input_or_done()
    full_stream += output

    while not session.finished and input_idx < len(inputs):
        if session.waiting_for_input:
            user_input = inputs[input_idx]
            input_idx += 1
            full_stream += user_input + "\n"
            output = session.provide_input(user_input)
            full_stream += output

    print("=" * 70)
    print("COMPLETE TOKEN STREAM")
    print("=" * 70)
    print(full_stream)
    print("=" * 70)


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo_token_stream()
    else:
        run_eliza_llm()
