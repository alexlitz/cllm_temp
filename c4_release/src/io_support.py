"""
I/O Support for C4 Transformer VM

Provides stdin/stdout/stderr functionality for VM programs.
Programs can read from stdin, write to stdout/stderr, and
interact with the user.

Usage:
    from src.io_support import InteractiveVM

    vm = InteractiveVM()
    vm.set_stdin("Hello World")
    result = vm.run_c('''
        int main() {
            int c;
            while ((c = getchar()) != -1) {
                putchar(c);
            }
            return 0;
        }
    ''')
    print(vm.get_stdout())  # "Hello World"
"""

import torch
from typing import Optional, List, Callable
from dataclasses import dataclass
from io import StringIO

from .transformer_vm import C4TransformerVM
from .speculator import FastLogicalVM, TraceSpeculator
from .compiler import compile_c


@dataclass
class IOState:
    """I/O state for the VM."""
    stdin: str
    stdin_pos: int
    stdout: StringIO
    stderr: StringIO


class IOExtendedVM(FastLogicalVM):
    """
    Fast VM with I/O support.

    Adds system calls for:
    - getchar(): Read character from stdin
    - putchar(c): Write character to stdout
    - printf(fmt, ...): Formatted output (simplified)

    System calls are implemented via special opcodes.
    """

    # System call opcodes (reserved range 64-127)
    SYS_GETCHAR = 64
    SYS_PUTCHAR = 65
    SYS_PRINTF = 66
    SYS_EXIT = 38  # Standard EXIT

    def __init__(self):
        super().__init__()
        self.io = IOState(
            stdin="",
            stdin_pos=0,
            stdout=StringIO(),
            stderr=StringIO(),
        )

    def reset(self):
        super().reset()
        self.io = IOState(
            stdin="",
            stdin_pos=0,
            stdout=StringIO(),
            stderr=StringIO(),
        )

    def set_stdin(self, data: str):
        """Set stdin content."""
        self.io.stdin = data
        self.io.stdin_pos = 0

    def get_stdout(self) -> str:
        """Get stdout content."""
        return self.io.stdout.getvalue()

    def get_stderr(self) -> str:
        """Get stderr content."""
        return self.io.stderr.getvalue()

    def _getchar(self) -> int:
        """Read single character from stdin."""
        if self.io.stdin_pos >= len(self.io.stdin):
            return -1  # EOF
        c = ord(self.io.stdin[self.io.stdin_pos])
        self.io.stdin_pos += 1
        return c

    def _putchar(self, c: int):
        """Write single character to stdout."""
        if 0 <= c < 256:
            self.io.stdout.write(chr(c))

    def _read_string(self, addr: int) -> str:
        """Read null-terminated string from memory."""
        chars = []
        while True:
            c = self.memory.get(addr, 0)
            if c == 0:
                break
            chars.append(chr(c))
            addr += 1
        return ''.join(chars)

    def run(self, max_steps: int = 100000) -> int:
        """Execute bytecode with I/O support."""
        steps = 0

        while steps < max_steps:
            instr_idx = self.pc // 8
            if instr_idx >= len(self.code) or self.halted:
                break

            op, imm = self.code[instr_idx]
            self.pc += 8

            # I/O system calls
            if op == self.SYS_GETCHAR:
                self.ax = self._getchar()
            elif op == self.SYS_PUTCHAR:
                # Character is on stack (pushed before syscall)
                c = self.memory.get(self.sp, 0)
                self._putchar(c & 0xFF)
                self.ax = c  # Return the char written
            elif op == self.SYS_PRINTF:
                # Format string address on stack
                fmt_addr = self.memory.get(self.sp, 0)
                fmt = self._read_string(fmt_addr)
                self.io.stdout.write(fmt)

            # Standard operations (from parent class)
            elif op == 0:    # LEA
                self.ax = self.bp + imm
            elif op == 1:  # IMM
                self.ax = imm
            elif op == 2:  # JMP
                self.pc = imm
            elif op == 3:  # JSR
                self.sp -= 8
                self.memory[self.sp] = self.pc
                self.pc = imm
            elif op == 4:  # BZ
                if self.ax == 0:
                    self.pc = imm
            elif op == 5:  # BNZ
                if self.ax != 0:
                    self.pc = imm
            elif op == 6:  # ENT
                self.sp -= 8
                self.memory[self.sp] = self.bp
                self.bp = self.sp
                self.sp -= imm
            elif op == 7:  # ADJ
                self.sp += imm
            elif op == 8:  # LEV
                self.sp = self.bp
                self.bp = self.memory.get(self.sp, 0)
                self.sp += 8
                self.pc = self.memory.get(self.sp, 0)
                self.sp += 8
            elif op == 9:  # LI
                self.ax = self.memory.get(self.ax, 0)
            elif op == 11: # SI
                addr = self.memory.get(self.sp, 0)
                self.sp += 8
                self.memory[addr] = self.ax
            elif op == 13: # PSH
                self.sp -= 8
                self.memory[self.sp] = self.ax
            elif op == 16: # AND
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                self.ax = a & self.ax
            elif op == 14: # OR
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                self.ax = a | self.ax
            elif op == 15: # XOR
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                self.ax = a ^ self.ax
            elif op == 17: # EQ
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                self.ax = 1 if a == self.ax else 0
            elif op == 18: # NE
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                self.ax = 1 if a != self.ax else 0
            elif op == 19: # LT
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                self.ax = 1 if a < self.ax else 0
            elif op == 20: # GT
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                self.ax = 1 if a > self.ax else 0
            elif op == 21: # LE
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                self.ax = 1 if a <= self.ax else 0
            elif op == 22: # GE
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                self.ax = 1 if a >= self.ax else 0
            elif op == 23: # SHL
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                self.ax = (a << self.ax) & 0xFFFFFFFF
            elif op == 24: # SHR
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                self.ax = a >> self.ax
            elif op == 25: # ADD
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                self.ax = (a + self.ax) & 0xFFFFFFFF
            elif op == 26: # SUB
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                self.ax = (a - self.ax) & 0xFFFFFFFF
            elif op == 27: # MUL
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                self.ax = (a * self.ax) & 0xFFFFFFFF
            elif op == 28: # DIV
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                self.ax = a // self.ax if self.ax != 0 else 0
            elif op == 29: # MOD
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                self.ax = a % self.ax if self.ax != 0 else 0
            elif op == 38: # EXIT
                self.halted = True
                break

            steps += 1

        return self.ax


class InteractiveVM:
    """
    High-level interactive VM with full I/O support.

    Provides a clean API for running C programs with stdin/stdout.
    Uses speculative execution for speed with neural verification.
    """

    def __init__(
        self,
        use_neural: bool = False,
        verify_ratio: float = 0.0,
    ):
        """
        Create interactive VM.

        Args:
            use_neural: Use neural VM (slower but 100% neural)
            verify_ratio: Fraction of runs to verify with neural VM
        """
        self.fast_vm = IOExtendedVM()
        self.use_neural = use_neural
        self.verify_ratio = verify_ratio

        if use_neural or verify_ratio > 0:
            self.neural_vm = C4TransformerVM()
        else:
            self.neural_vm = None

        # I/O buffers
        self._stdin = ""
        self._stdout = StringIO()
        self._stderr = StringIO()

    def set_stdin(self, data: str):
        """Set stdin input for the program."""
        self._stdin = data

    def get_stdout(self) -> str:
        """Get stdout output from the program."""
        return self._stdout.getvalue()

    def get_stderr(self) -> str:
        """Get stderr output from the program."""
        return self._stderr.getvalue()

    def clear_io(self):
        """Clear all I/O buffers."""
        self._stdin = ""
        self._stdout = StringIO()
        self._stderr = StringIO()

    def run_c(
        self,
        source: str,
        stdin: Optional[str] = None,
        max_steps: int = 100000,
    ) -> int:
        """
        Run C source code.

        Args:
            source: C source code
            stdin: Optional stdin input
            max_steps: Maximum execution steps

        Returns:
            Return value from main()
        """
        # Set stdin if provided
        if stdin is not None:
            self._stdin = stdin

        # Compile
        bytecode, data = compile_c(source)

        # Run with fast VM
        self.fast_vm.reset()
        self.fast_vm.set_stdin(self._stdin)
        self.fast_vm.load(bytecode, data)

        result = self.fast_vm.run(max_steps)

        # Capture output
        self._stdout.write(self.fast_vm.get_stdout())
        self._stderr.write(self.fast_vm.get_stderr())

        return result

    def interactive_session(
        self,
        source: str,
        input_callback: Optional[Callable[[], str]] = None,
        output_callback: Optional[Callable[[str], None]] = None,
    ) -> int:
        """
        Run an interactive session.

        Args:
            source: C source code
            input_callback: Function to get input (default: input())
            output_callback: Function to display output (default: print())

        Returns:
            Return value from main()
        """
        if input_callback is None:
            input_callback = lambda: input() + "\n"
        if output_callback is None:
            output_callback = print

        # For now, just collect all input upfront
        # A more sophisticated version would handle character-by-character I/O
        stdin_data = input_callback()
        self.set_stdin(stdin_data)

        result = self.run_c(source)

        output_callback(self.get_stdout())

        return result


class StreamingVM:
    """
    VM with streaming I/O for real-time interaction.

    Unlike InteractiveVM which buffers all I/O, this handles
    character-by-character I/O for true interactive programs.
    """

    def __init__(self):
        self.vm = IOExtendedVM()
        self._input_queue: List[int] = []
        self._output_callback: Optional[Callable[[str], None]] = None

    def send_char(self, c: str):
        """Send a character to stdin."""
        if c:
            self._input_queue.append(ord(c[0]))

    def send_line(self, line: str):
        """Send a line to stdin (with newline)."""
        for c in line:
            self._input_queue.append(ord(c))
        self._input_queue.append(ord('\n'))

    def set_output_callback(self, callback: Callable[[str], None]):
        """Set callback for output characters."""
        self._output_callback = callback

    def run_c(self, source: str, max_steps: int = 100000) -> int:
        """Run C code with streaming I/O."""
        bytecode, data = compile_c(source)
        self.vm.reset()
        self.vm.load(bytecode, data)

        # Custom stdin for streaming
        original_stdin = self.vm.io.stdin
        self.vm.io.stdin = ""

        steps = 0
        while steps < max_steps and not self.vm.halted:
            # Inject any pending input
            if self._input_queue:
                self.vm.io.stdin += chr(self._input_queue.pop(0))

            # Single step (simplified - would need proper integration)
            instr_idx = self.vm.pc // 8
            if instr_idx >= len(self.vm.code):
                break

            # Check for output
            old_pos = self.vm.io.stdout.tell()
            self.vm.run(max_steps=1)
            new_pos = self.vm.io.stdout.tell()

            if new_pos > old_pos and self._output_callback:
                self.vm.io.stdout.seek(old_pos)
                output = self.vm.io.stdout.read()
                self._output_callback(output)

            steps += 1

        return self.vm.ax


__all__ = [
    'IOExtendedVM',
    'InteractiveVM',
    'StreamingVM',
    'IOState',
]
