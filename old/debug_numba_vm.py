#!/usr/bin/env python3
"""Debug Numba vs Python VM divergence."""

import sys
import os
import numpy as np
from numba import jit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.compiler import compile_c

# Simple Python VM for comparison
class SimpleVM:
    def __init__(self):
        self.memory = {}
        self.pc = 0
        self.sp = 0x30000
        self.bp = 0x30000
        self.ax = 0
        self.code = []
        self.stdout = []
        self.steps = 0

    def load(self, bytecode, data=None):
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

    def mem_read(self, addr):
        return self.memory.get(addr, 0)

    def mem_write(self, addr, val):
        self.memory[addr] = val

    def step(self):
        if self.pc // 8 >= len(self.code):
            return False

        op, imm = self.code[self.pc // 8]
        self.pc += 8
        self.steps += 1

        if op == 65:  # PUTCHAR
            c = self.mem_read(self.sp)
            self.stdout.append(c & 0xFF)
            self.ax = c
        elif op == 0:  # LEA
            self.ax = self.bp + imm
        elif op == 1:  # IMM
            self.ax = imm
        elif op == 2:  # JMP
            self.pc = imm
        elif op == 3:  # JSR
            self.sp -= 8
            self.mem_write(self.sp, self.pc)
            self.pc = imm
        elif op == 4:  # BZ
            if self.ax == 0:
                self.pc = imm
        elif op == 5:  # BNZ
            if self.ax != 0:
                self.pc = imm
        elif op == 6:  # ENT
            self.sp -= 8
            self.mem_write(self.sp, self.bp)
            self.bp = self.sp
            self.sp -= imm
        elif op == 7:  # ADJ
            self.sp += imm
        elif op == 8:  # LEV
            self.sp = self.bp
            self.bp = self.mem_read(self.sp)
            self.sp += 8
            self.pc = self.mem_read(self.sp)
            self.sp += 8
        elif op == 9:  # LI
            self.ax = self.mem_read(self.ax)
        elif op == 10:  # LC
            self.ax = self.mem_read(self.ax) & 0xFF
        elif op == 11:  # SI
            addr = self.mem_read(self.sp)
            self.sp += 8
            self.mem_write(addr, self.ax)
        elif op == 12:  # SC
            addr = self.mem_read(self.sp)
            self.sp += 8
            self.mem_write(addr, self.ax & 0xFF)
        elif op == 13:  # PSH
            self.sp -= 8
            self.mem_write(self.sp, self.ax)
        elif op == 14:  # OR
            a = self.mem_read(self.sp)
            self.sp += 8
            self.ax = a | self.ax
        elif op == 15:  # XOR
            a = self.mem_read(self.sp)
            self.sp += 8
            self.ax = a ^ self.ax
        elif op == 16:  # AND
            a = self.mem_read(self.sp)
            self.sp += 8
            self.ax = a & self.ax
        elif op == 17:  # EQ
            a = self.mem_read(self.sp)
            self.sp += 8
            self.ax = 1 if a == self.ax else 0
        elif op == 18:  # NE
            a = self.mem_read(self.sp)
            self.sp += 8
            self.ax = 1 if a != self.ax else 0
        elif op == 19:  # LT
            a = self.mem_read(self.sp)
            self.sp += 8
            self.ax = 1 if a < self.ax else 0
        elif op == 20:  # GT
            a = self.mem_read(self.sp)
            self.sp += 8
            self.ax = 1 if a > self.ax else 0
        elif op == 21:  # LE
            a = self.mem_read(self.sp)
            self.sp += 8
            self.ax = 1 if a <= self.ax else 0
        elif op == 22:  # GE
            a = self.mem_read(self.sp)
            self.sp += 8
            self.ax = 1 if a >= self.ax else 0
        elif op == 23:  # SHL
            a = self.mem_read(self.sp)
            self.sp += 8
            self.ax = a << self.ax
        elif op == 24:  # SHR
            a = self.mem_read(self.sp)
            self.sp += 8
            self.ax = a >> self.ax
        elif op == 25:  # ADD
            a = self.mem_read(self.sp)
            self.sp += 8
            self.ax = a + self.ax
        elif op == 26:  # SUB
            a = self.mem_read(self.sp)
            self.sp += 8
            self.ax = a - self.ax
        elif op == 27:  # MUL
            a = self.mem_read(self.sp)
            self.sp += 8
            self.ax = a * self.ax
        elif op == 28:  # DIV
            a = self.mem_read(self.sp)
            b = self.ax
            self.sp += 8
            if b != 0:
                # Python integer division
                self.ax = int(a / b) if (a < 0) != (b < 0) else a // b
            else:
                self.ax = 0
        elif op == 29:  # MOD
            a = self.mem_read(self.sp)
            b = self.ax
            self.sp += 8
            if b != 0:
                quot = int(a / b) if (a < 0) != (b < 0) else a // b
                self.ax = a - quot * b
            else:
                self.ax = 0
        elif op == 38:  # EXIT
            return False

        return True


@jit(nopython=True, cache=True)
def run_vm_numba(code_ops, code_imms, data, max_steps):
    """Numba-JIT compiled VM loop."""
    memory = np.zeros(0x40000, dtype=np.int64)

    for i in range(len(data)):
        memory[0x10000 + i] = data[i]

    pc = 0
    sp = 0x30000
    bp = 0x30000
    ax = 0

    stdout = []
    steps = 0
    n_code = len(code_ops)

    while steps < max_steps:
        idx = pc >> 3
        if idx >= n_code:
            break

        op = code_ops[idx]
        imm = code_imms[idx]
        pc += 8
        steps += 1

        if op == 65:  # PUTCHAR
            c = memory[sp]
            stdout.append(c & 0xFF)
            ax = c
        elif op == 0:  # LEA
            ax = bp + imm
        elif op == 1:  # IMM
            ax = imm
        elif op == 2:  # JMP
            pc = imm
        elif op == 3:  # JSR
            sp -= 8
            memory[sp] = pc
            pc = imm
        elif op == 4:  # BZ
            if ax == 0:
                pc = imm
        elif op == 5:  # BNZ
            if ax != 0:
                pc = imm
        elif op == 6:  # ENT
            sp -= 8
            memory[sp] = bp
            bp = sp
            sp -= imm
        elif op == 7:  # ADJ
            sp += imm
        elif op == 8:  # LEV
            sp = bp
            bp = memory[sp]
            sp += 8
            pc = memory[sp]
            sp += 8
        elif op == 9:  # LI
            ax = memory[ax]
        elif op == 10:  # LC
            ax = memory[ax] & 0xFF
        elif op == 11:  # SI
            addr = memory[sp]
            sp += 8
            memory[addr] = ax
        elif op == 12:  # SC
            addr = memory[sp]
            sp += 8
            memory[addr] = ax & 0xFF
        elif op == 13:  # PSH
            sp -= 8
            memory[sp] = ax
        elif op == 14:  # OR
            a = memory[sp]
            sp += 8
            ax = a | ax
        elif op == 15:  # XOR
            a = memory[sp]
            sp += 8
            ax = a ^ ax
        elif op == 16:  # AND
            a = memory[sp]
            sp += 8
            ax = a & ax
        elif op == 17:  # EQ
            a = memory[sp]
            sp += 8
            ax = 1 if a == ax else 0
        elif op == 18:  # NE
            a = memory[sp]
            sp += 8
            ax = 1 if a != ax else 0
        elif op == 19:  # LT
            a = memory[sp]
            sp += 8
            ax = 1 if a < ax else 0
        elif op == 20:  # GT
            a = memory[sp]
            sp += 8
            ax = 1 if a > ax else 0
        elif op == 21:  # LE
            a = memory[sp]
            sp += 8
            ax = 1 if a <= ax else 0
        elif op == 22:  # GE
            a = memory[sp]
            sp += 8
            ax = 1 if a >= ax else 0
        elif op == 23:  # SHL
            a = memory[sp]
            sp += 8
            ax = a << ax
        elif op == 24:  # SHR
            a = memory[sp]
            sp += 8
            ax = a >> ax
        elif op == 25:  # ADD
            a = memory[sp]
            sp += 8
            ax = a + ax
        elif op == 26:  # SUB
            a = memory[sp]
            sp += 8
            ax = a - ax
        elif op == 27:  # MUL
            a = memory[sp]
            sp += 8
            ax = a * ax
        elif op == 28:  # DIV
            a = memory[sp]
            b = ax
            sp += 8
            if b != 0:
                # C-style truncation toward zero
                if (a < 0) != (b < 0):
                    ax = -(-a // b) if a < 0 else -(a // -b)
                else:
                    ax = a // b
            else:
                ax = 0
        elif op == 29:  # MOD
            a = memory[sp]
            b = ax
            sp += 8
            if b != 0:
                if (a < 0) != (b < 0):
                    quot = -(-a // b) if a < 0 else -(a // -b)
                else:
                    quot = a // b
                ax = a - quot * b
            else:
                ax = 0
        elif op == 38:  # EXIT
            break

    return steps, stdout


def main():
    # Simple test program with negative numbers
    test_source = """
int main() {
    int a, b, c;
    a = -100;
    b = 7;
    c = a / b;
    putchar(48 + (c / -10));  // Should be '1' (14)
    putchar(48 + (-(c % -10)));  // Should be '4'
    return 0;
}
"""

    print("Testing negative number division...")
    bytecode, data = compile_c(test_source)

    # Python VM
    py_vm = SimpleVM()
    py_vm.load(bytecode, data)
    while py_vm.step():
        pass
    py_out = bytes(py_vm.stdout)
    print(f"Python: {py_out} (steps: {py_vm.steps})")

    # Numba VM
    code_ops = np.array([b & 0xFF for b in bytecode], dtype=np.int32)
    code_imms = np.array([b >> 8 if b >> 8 < (1 << 55) else (b >> 8) - (1 << 56)
                          for b in bytecode], dtype=np.int64)
    data_arr = np.array(data if data else [], dtype=np.int64)

    steps, stdout = run_vm_numba(code_ops, code_imms, data_arr, 1000000)
    numba_out = bytes(stdout)
    print(f"Numba:  {numba_out} (steps: {steps})")

    # Test Mandelbrot with small size
    print("\nTesting Mandelbrot 8x8...")
    c_file = os.path.join(os.path.dirname(__file__), "mandelbrot_putchar_c4.c")
    with open(c_file) as f:
        source = f.read()

    source = source.replace("width = 32;", "width = 8;")
    source = source.replace("height = 32;", "height = 8;")

    bytecode, data = compile_c(source)

    # Python VM
    py_vm = SimpleVM()
    py_vm.load(bytecode, data)
    while py_vm.step():
        pass
    py_out = bytes(py_vm.stdout)
    print(f"Python: {len(py_out)} bytes, steps: {py_vm.steps}")

    # Numba VM
    code_ops = np.array([b & 0xFF for b in bytecode], dtype=np.int32)
    code_imms = np.array([b >> 8 if b >> 8 < (1 << 55) else (b >> 8) - (1 << 56)
                          for b in bytecode], dtype=np.int64)
    data_arr = np.array(data if data else [], dtype=np.int64)

    steps, stdout = run_vm_numba(code_ops, code_imms, data_arr, 100000000)
    numba_out = bytes(stdout)
    print(f"Numba:  {len(numba_out)} bytes, steps: {steps}")

    if py_out == numba_out:
        print("MATCH!")
    else:
        print("MISMATCH!")
        print(f"Python first 20: {py_out[:20]}")
        print(f"Numba first 20:  {numba_out[:20]}")


if __name__ == "__main__":
    main()
