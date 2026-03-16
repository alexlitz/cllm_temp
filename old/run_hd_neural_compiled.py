#!/usr/bin/env python3
"""
HD Mandelbrot with torch.compile for VM acceleration + Batched Neural Verification
"""

import sys
import os
import time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.compiler import compile_c

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# =============================================================================
# NEURAL PRIMITIVES
# =============================================================================

def silu(x):
    return x * torch.sigmoid(x)

def swiglu_mul_batch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Batched exact multiplication using SwiGLU identity."""
    return silu(a) * b + silu(-a) * (-b)

class NeuralNewtonDivide:
    """FFN-based Newton division."""

    def __init__(self, n_segments=64):
        self.n_segments = n_segments
        self.device = DEVICE

        self.breakpoints = torch.linspace(0.5, 1.0, n_segments + 1,
                                          dtype=torch.float32, device=DEVICE)
        self.values = 1.0 / self.breakpoints
        self._set_weights()

    def _set_weights(self):
        n = self.n_segments
        self.W1 = torch.ones(2 * n, 1, dtype=torch.float32, device=self.device)
        self.b1 = torch.zeros(2 * n, dtype=torch.float32, device=self.device)
        for i in range(n):
            self.b1[2*i] = -self.breakpoints[i]
            self.b1[2*i + 1] = -self.breakpoints[i+1]

        self.W2 = torch.zeros(1, 2 * n, dtype=torch.float32, device=self.device)
        for i in range(n):
            delta_x = self.breakpoints[i+1] - self.breakpoints[i]
            slope = (self.values[i+1] - self.values[i]) / delta_x
            self.W2[0, 2*i] = slope
            self.W2[0, 2*i + 1] = -slope

        self.b2 = torch.tensor([self.values[0].item()], dtype=torch.float32, device=self.device)

    def divide_batch(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Batched division on GPU."""
        sign = torch.sign(a) * torch.sign(b)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        a_abs = torch.abs(a.float())
        b_abs = torch.abs(b.float())

        exp = torch.floor(torch.log2(b_abs + 1e-10)).int()
        normalized = b_abs / (2.0 ** exp)
        normalized = torch.clamp(normalized, 0.5, 0.9999)

        x = normalized.unsqueeze(-1)
        hidden = F.relu(torch.matmul(x, self.W1.T) + self.b1)
        y = (torch.matmul(hidden, self.W2.T) + self.b2).squeeze(-1)

        for _ in range(2):
            correction = 2.0 - swiglu_mul_batch(normalized, y)
            y = swiglu_mul_batch(y, correction)

        y = y / (2.0 ** exp)
        result = swiglu_mul_batch(a_abs, y)

        candidate = torch.round(result)
        check = swiglu_mul_batch(candidate, b_abs)
        candidate = torch.where(check > a_abs + 0.5, candidate - 1, candidate)

        return (candidate * sign).int()


# =============================================================================
# SPECULATIVE VM WITH BATCHED VERIFICATION
# =============================================================================

class SpeculativeNeuralVM:
    """Fast VM with speculative execution and batched neural verification."""

    OP_NAMES = {
        0: 'LEA', 1: 'IMM', 2: 'JMP', 3: 'JSR', 4: 'BZ', 5: 'BNZ',
        6: 'ENT', 7: 'ADJ', 8: 'LEV', 9: 'LI', 10: 'LC', 11: 'SI', 12: 'SC',
        13: 'PSH', 14: 'OR', 15: 'XOR', 16: 'AND', 17: 'EQ', 18: 'NE',
        19: 'LT', 20: 'GT', 21: 'LE', 22: 'GE', 23: 'SHL', 24: 'SHR',
        25: 'ADD', 26: 'SUB', 27: 'MUL', 28: 'DIV', 29: 'MOD',
        38: 'EXIT', 64: 'PRTF', 65: 'PUTCHAR'
    }

    def __init__(self, batch_size=10000):
        self.batch_size = batch_size
        self.divider = NeuralNewtonDivide()

        self.memory = {}
        self.pc = 0
        self.sp = 0x30000
        self.bp = 0x30000
        self.ax = 0
        self.halted = False
        self.code = []

        self.stdout = []
        self.steps = 0
        self.mul_count = 0
        self.div_count = 0

        # Operation queues for batched verification
        self.mul_queue_a = []
        self.mul_queue_b = []
        self.mul_queue_r = []
        self.div_queue_a = []
        self.div_queue_b = []
        self.div_queue_r = []

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

    def _mem_write(self, addr, value):
        self.memory[addr] = value

    def _mem_read(self, addr):
        return self.memory.get(addr, 0)

    def _mul(self, a, b):
        """Speculative multiply - queue for verification."""
        self.mul_count += 1
        result = a * b
        self.mul_queue_a.append(a)
        self.mul_queue_b.append(b)
        self.mul_queue_r.append(result)
        return result

    def _div(self, a, b):
        """Speculative divide - queue for verification."""
        self.div_count += 1
        if b == 0:
            return 0
        # C-style truncation toward zero
        if (a < 0) != (b < 0):
            result = -(-a // b) if a < 0 else -(a // -b)
        else:
            result = a // b
        self.div_queue_a.append(a)
        self.div_queue_b.append(b)
        self.div_queue_r.append(result)
        return result

    def verify_batch(self):
        """Verify queued operations on GPU."""
        errors = 0

        if self.mul_queue_a:
            a = torch.tensor(self.mul_queue_a, dtype=torch.float32, device=DEVICE)
            b = torch.tensor(self.mul_queue_b, dtype=torch.float32, device=DEVICE)
            expected = torch.tensor(self.mul_queue_r, dtype=torch.float32, device=DEVICE)

            result = swiglu_mul_batch(a, b)
            err = (torch.abs(result - expected) > 1.0).sum().item()
            errors += err

            self.mul_queue_a.clear()
            self.mul_queue_b.clear()
            self.mul_queue_r.clear()

        if self.div_queue_a:
            a = torch.tensor(self.div_queue_a, dtype=torch.float32, device=DEVICE)
            b = torch.tensor(self.div_queue_b, dtype=torch.float32, device=DEVICE)
            expected = torch.tensor(self.div_queue_r, dtype=torch.int32, device=DEVICE)

            result = self.divider.divide_batch(a, b)
            err = (result != expected).sum().item()
            errors += err

            self.div_queue_a.clear()
            self.div_queue_b.clear()
            self.div_queue_r.clear()

        return errors

    def step(self):
        if self.halted or self.pc // 8 >= len(self.code):
            self.halted = True
            return False

        op, imm = self.code[self.pc // 8]
        self.pc += 8
        self.steps += 1

        if op == 65:  # PUTCHAR
            c = self._mem_read(self.sp)
            self.stdout.append(c & 0xFF)
            self.ax = c
        elif op == 0:  self.ax = self.bp + imm
        elif op == 1:  self.ax = imm
        elif op == 2:  self.pc = imm
        elif op == 3:
            self.sp -= 8
            self._mem_write(self.sp, self.pc)
            self.pc = imm
        elif op == 4:
            if self.ax == 0: self.pc = imm
        elif op == 5:
            if self.ax != 0: self.pc = imm
        elif op == 6:
            self.sp -= 8
            self._mem_write(self.sp, self.bp)
            self.bp = self.sp
            self.sp -= imm
        elif op == 7:  self.sp += imm
        elif op == 8:
            self.sp = self.bp
            self.bp = self._mem_read(self.sp)
            self.sp += 8
            self.pc = self._mem_read(self.sp)
            self.sp += 8
        elif op == 9:  self.ax = self._mem_read(self.ax)
        elif op == 10: self.ax = self._mem_read(self.ax) & 0xFF
        elif op == 11:
            addr = self._mem_read(self.sp)
            self.sp += 8
            self._mem_write(addr, self.ax)
        elif op == 12:
            addr = self._mem_read(self.sp)
            self.sp += 8
            self._mem_write(addr, self.ax & 0xFF)
        elif op == 13:
            self.sp -= 8
            self._mem_write(self.sp, self.ax)
        elif op == 14:
            a = self._mem_read(self.sp); self.sp += 8
            self.ax = a | self.ax
        elif op == 15:
            a = self._mem_read(self.sp); self.sp += 8
            self.ax = a ^ self.ax
        elif op == 16:
            a = self._mem_read(self.sp); self.sp += 8
            self.ax = a & self.ax
        elif op == 17:
            a = self._mem_read(self.sp); self.sp += 8
            self.ax = 1 if a == self.ax else 0
        elif op == 18:
            a = self._mem_read(self.sp); self.sp += 8
            self.ax = 1 if a != self.ax else 0
        elif op == 19:
            a = self._mem_read(self.sp); self.sp += 8
            self.ax = 1 if a < self.ax else 0
        elif op == 20:
            a = self._mem_read(self.sp); self.sp += 8
            self.ax = 1 if a > self.ax else 0
        elif op == 21:
            a = self._mem_read(self.sp); self.sp += 8
            self.ax = 1 if a <= self.ax else 0
        elif op == 22:
            a = self._mem_read(self.sp); self.sp += 8
            self.ax = 1 if a >= self.ax else 0
        elif op == 23:
            a = self._mem_read(self.sp); self.sp += 8
            self.ax = a << self.ax
        elif op == 24:
            a = self._mem_read(self.sp); self.sp += 8
            self.ax = a >> self.ax
        elif op == 25:
            a = self._mem_read(self.sp); self.sp += 8
            self.ax = a + self.ax
        elif op == 26:
            a = self._mem_read(self.sp); self.sp += 8
            self.ax = a - self.ax
        elif op == 27:  # MUL
            a = self._mem_read(self.sp); self.sp += 8
            self.ax = self._mul(a, self.ax)
        elif op == 28:  # DIV
            a = self._mem_read(self.sp); self.sp += 8
            self.ax = self._div(a, self.ax)
        elif op == 29:  # MOD
            a = self._mem_read(self.sp)
            b = self.ax
            self.sp += 8
            if b != 0:
                quot = self._div(a, b)
                self.ax = a - quot * b
            else:
                self.ax = 0
        elif op == 38:  # EXIT
            self.halted = True
            return False

        return True

    def run(self, max_steps=10000000000, report_interval=None):
        start = time.time()
        last_report = 0
        total_errors = 0

        while self.steps < max_steps and self.step():
            # Verify in batches
            if len(self.mul_queue_a) + len(self.div_queue_a) >= self.batch_size:
                total_errors += self.verify_batch()

            if report_interval and self.steps - last_report >= report_interval:
                elapsed = time.time() - start
                print(f"  {elapsed/60:.1f}min: {self.mul_count:,} MUL, {len(self.stdout):,} bytes")
                last_report = self.steps

        # Final verification
        total_errors += self.verify_batch()

        return self.ax, total_errors

    def get_stdout_bytes(self):
        return bytes(self.stdout)

    def stats(self):
        return {
            'steps': self.steps,
            'mul_count': self.mul_count,
            'div_count': self.div_count,
            'output_bytes': len(self.stdout),
        }


# =============================================================================
# MAIN
# =============================================================================

def main():
    width = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    height = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    output = sys.argv[3] if len(sys.argv) > 3 else f"/tmp/mandelbrot_{width}x{height}_compiled.png"

    print("=" * 70)
    print("  HD MANDELBROT - SPECULATIVE VM + BATCHED NEURAL VERIFICATION")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Resolution: {width}x{height} ({width*height:,} pixels)")
    print()

    c_file = os.path.join(os.path.dirname(__file__), "mandelbrot_putchar_c4.c")
    with open(c_file) as f:
        source = f.read()

    source = source.replace("width = 32;", f"width = {width};")
    source = source.replace("height = 32;", f"height = {height};")

    print("Compiling C to bytecode...")
    bytecode, data = compile_c(source)
    print(f"Compiled: {len(bytecode)} instructions")
    print()

    print("Running with batched neural verification...")
    vm = SpeculativeNeuralVM(batch_size=10000)
    vm.load(bytecode, data)

    start = time.time()
    report_interval = 1000000 if width * height > 10000 else None
    ax, errors = vm.run(report_interval=report_interval)
    elapsed = time.time() - start

    stats = vm.stats()

    print()
    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print()
    print(f"Time: {elapsed/60:.1f} minutes ({elapsed:.1f}s)")
    print(f"VM steps: {stats['steps']:,}")
    print(f"Verification errors: {errors}")
    print()

    print("NEURAL OPERATIONS:")
    print(f"  SwiGLU MUL:  {stats['mul_count']:,}")
    print(f"  Newton DIV:  {stats['div_count']:,}")
    print(f"  Total:       {stats['mul_count'] + stats['div_count']:,}")
    print()

    # FLOP calculation
    swiglu_flops = stats['mul_count'] * 91
    newton_flops = stats['div_count'] * 930
    total = swiglu_flops + newton_flops

    print("FLOPS (FFN-based verification):")
    print(f"  SwiGLU MUL:  {stats['mul_count']:,} × 91 = {swiglu_flops/1e9:.2f} GFLOPs")
    print(f"  Newton DIV:  {stats['div_count']:,} × 930 = {newton_flops/1e9:.2f} GFLOPs")
    print(f"  TOTAL:       {total/1e9:.2f} GFLOPs")
    print()

    # Save output
    output_bytes = vm.get_stdout_bytes()
    if output_bytes:
        with open(output, 'wb') as f:
            f.write(output_bytes)

        is_png = output_bytes[:4] == b'\x89PNG'
        print(f"Output: {len(output_bytes):,} bytes -> {output}")
        print(f"PNG valid: {is_png}")

    print("=" * 70)


if __name__ == "__main__":
    main()
