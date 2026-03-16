#!/usr/bin/env python3
"""
HD Mandelbrot with BATCHED Neural Verification

Speculate at full speed, batch verify on GPU every N operations.
"""

import sys
import os
import time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.compiler import compile_c

# Use MPS if available
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# =============================================================================
# BATCHED NEURAL VERIFICATION
# =============================================================================

class BatchedNeuralVerifier:
    """Batch verify MUL and DIV operations on GPU."""

    def __init__(self, batch_size=10000):
        self.batch_size = batch_size
        self.device = DEVICE

        # Queues for pending verification
        self.mul_queue = []  # [(a, b, expected_result), ...]
        self.div_queue = []  # [(a, b, expected_result), ...]

        # FFN weights for Newton division
        self._init_ffn_weights()

        # Stats
        self.total_mul_verified = 0
        self.total_div_verified = 0
        self.verification_errors = 0

    def _init_ffn_weights(self, n_segments=64):
        """Initialize FFN weights for reciprocal lookup."""
        breakpoints = torch.linspace(0.5, 1.0, n_segments + 1,
                                     dtype=torch.float32, device=self.device)
        values = 1.0 / breakpoints

        n = n_segments
        self.W1 = torch.ones(2 * n, 1, dtype=torch.float32, device=self.device)
        self.b1 = torch.zeros(2 * n, dtype=torch.float32, device=self.device)
        for i in range(n):
            self.b1[2*i] = -breakpoints[i]
            self.b1[2*i + 1] = -breakpoints[i+1]

        self.W2 = torch.zeros(1, 2 * n, dtype=torch.float32, device=self.device)
        for i in range(n):
            delta_x = breakpoints[i+1] - breakpoints[i]
            slope = (values[i+1] - values[i]) / delta_x
            self.W2[0, 2*i] = slope
            self.W2[0, 2*i + 1] = -slope

        self.b2 = torch.tensor([values[0].item()], dtype=torch.float32, device=self.device)

    def queue_mul(self, a, b, result):
        """Queue a multiplication for batch verification."""
        self.mul_queue.append((a, b, result))
        if len(self.mul_queue) >= self.batch_size:
            self._verify_mul_batch()

    def queue_div(self, a, b, result):
        """Queue a division for batch verification."""
        self.div_queue.append((a, b, result))
        if len(self.div_queue) >= self.batch_size:
            self._verify_div_batch()

    @staticmethod
    @torch.jit.script
    def _swiglu_mul_batch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Batched SwiGLU multiply (JIT compiled)."""
        silu_a = a * torch.sigmoid(a)
        silu_neg_a = (-a) * torch.sigmoid(-a)
        return silu_a * b + silu_neg_a * (-b)

    def _verify_mul_batch(self):
        """Verify queued multiplications."""
        if not self.mul_queue:
            return

        # Convert to tensors
        a = torch.tensor([x[0] for x in self.mul_queue], dtype=torch.float32, device=self.device)
        b = torch.tensor([x[1] for x in self.mul_queue], dtype=torch.float32, device=self.device)
        expected = torch.tensor([x[2] for x in self.mul_queue], dtype=torch.float32, device=self.device)

        # Batch verify with SwiGLU
        result = self._swiglu_mul_batch(a, b)

        # Check results (allow small tolerance for float32)
        errors = torch.abs(result - expected) > 1.0
        n_errors = errors.sum().item()

        if n_errors > 0:
            self.verification_errors += n_errors

        self.total_mul_verified += len(self.mul_queue)
        self.mul_queue = []

    def _verify_div_batch(self):
        """Verify queued divisions using FFN-based Newton."""
        if not self.div_queue:
            return

        # Convert to tensors
        a = torch.tensor([x[0] for x in self.div_queue], dtype=torch.float32, device=self.device)
        b = torch.tensor([x[1] for x in self.div_queue], dtype=torch.float32, device=self.device)
        expected = torch.tensor([x[2] for x in self.div_queue], dtype=torch.float32, device=self.device)

        # Handle signs
        sign = torch.sign(a) * torch.sign(b)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        a_abs = torch.abs(a)
        b_abs = torch.abs(b) + 1e-10  # avoid div by zero

        # Normalize b to [0.5, 1.0)
        exp = torch.floor(torch.log2(b_abs + 1e-10))
        normalized = b_abs / (2.0 ** exp)
        normalized = torch.clamp(normalized, 0.5, 0.9999)

        # FFN lookup
        x = normalized.unsqueeze(-1)
        hidden = F.relu(torch.matmul(x, self.W1.T) + self.b1)
        y = torch.matmul(hidden, self.W2.T) + self.b2
        y = y.squeeze(-1)

        # Newton iterations
        for _ in range(2):
            correction = 2.0 - self._swiglu_mul_batch(normalized, y)
            y = self._swiglu_mul_batch(y, correction)

        # Denormalize and compute result
        y = y / (2.0 ** exp)
        result = self._swiglu_mul_batch(a_abs, y)
        result = torch.round(result) * sign

        # Check results
        errors = torch.abs(result - expected) > 1.0
        n_errors = errors.sum().item()

        if n_errors > 0:
            self.verification_errors += n_errors

        self.total_div_verified += len(self.div_queue)
        self.div_queue = []

    def flush(self):
        """Verify any remaining queued operations."""
        self._verify_mul_batch()
        self._verify_div_batch()

    def stats(self):
        return {
            'mul_verified': self.total_mul_verified,
            'div_verified': self.total_div_verified,
            'errors': self.verification_errors,
        }


# =============================================================================
# SPECULATIVE VM WITH BATCHED VERIFICATION
# =============================================================================

class SpeculativeVMBatched:
    """Fast speculative execution with batched neural verification."""

    OP_MUL = 27
    OP_DIV = 28
    OP_MOD = 29
    OP_PUTCHAR = 65
    OP_EXIT = 38

    def __init__(self, verify_interval=10000):
        self.verifier = BatchedNeuralVerifier(batch_size=verify_interval)

        # State
        self.memory = {}
        self.pc = 0
        self.sp = 0x30000
        self.bp = 0x30000
        self.ax = 0
        self.halted = False
        self.code = []
        self.stdout = []

        # Counters
        self.steps = 0
        self.mul_count = 0
        self.div_count = 0
        self.total_writes = 0

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
        self.total_writes += 1

    def _mem_read(self, addr):
        return self.memory.get(addr, 0)

    def _mul(self, a, b):
        """Fast speculative multiply, queue for verification."""
        self.mul_count += 1
        result = (a * b) & 0xFFFFFFFF
        self.verifier.queue_mul(float(a), float(b), float(result))
        return result

    def _div(self, a, b):
        """Fast speculative divide, queue for verification."""
        self.div_count += 1
        if b == 0:
            return 0
        result = a // b
        self.verifier.queue_div(float(a), float(b), float(result))
        return result

    def step(self):
        if self.halted or self.pc // 8 >= len(self.code):
            self.halted = True
            return False

        op, imm = self.code[self.pc // 8]
        self.pc += 8
        self.steps += 1

        if op == self.OP_PUTCHAR:
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
            self.ax = (a << self.ax) & 0xFFFFFFFF
        elif op == 24:
            a = self._mem_read(self.sp); self.sp += 8
            self.ax = a >> self.ax
        elif op == 25:
            a = self._mem_read(self.sp); self.sp += 8
            self.ax = (a + self.ax) & 0xFFFFFFFF
        elif op == 26:
            a = self._mem_read(self.sp); self.sp += 8
            self.ax = (a - self.ax) & 0xFFFFFFFF
        elif op == self.OP_MUL:
            a = self._mem_read(self.sp); self.sp += 8
            self.ax = self._mul(a, self.ax)
        elif op == self.OP_DIV:
            a = self._mem_read(self.sp); self.sp += 8
            self.ax = self._div(a, self.ax)
        elif op == self.OP_MOD:
            a = self._mem_read(self.sp)
            b = self.ax
            self.sp += 8
            if b != 0:
                quot = self._div(a, b)
                self.ax = a - quot * b
            else:
                self.ax = 0
        elif op == self.OP_EXIT:
            self.halted = True
            return False

        return True

    def run(self, max_steps=10000000000, report_interval=None):
        start = time.time()
        last_report = 0

        while self.steps < max_steps and self.step():
            if report_interval and self.steps - last_report >= report_interval:
                elapsed = time.time() - start
                print(f"  {elapsed/60:.1f}min: {self.mul_count:,} MUL, {len(self.stdout):,} bytes")
                last_report = self.steps

        # Flush remaining verification queue
        self.verifier.flush()
        return self.ax

    def get_stdout_bytes(self):
        return bytes(self.stdout)

    def stats(self):
        v_stats = self.verifier.stats()
        return {
            'steps': self.steps,
            'mul_count': self.mul_count,
            'div_count': self.div_count,
            'total_writes': self.total_writes,
            'live_memory': len(self.memory),
            'output_bytes': len(self.stdout),
            'verified_mul': v_stats['mul_verified'],
            'verified_div': v_stats['div_verified'],
            'verification_errors': v_stats['errors'],
        }


# =============================================================================
# MAIN
# =============================================================================

def main():
    width = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    height = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    output = sys.argv[3] if len(sys.argv) > 3 else f"/tmp/mandelbrot_{width}x{height}_batched.png"

    print("=" * 70)
    print("  HD MANDELBROT - BATCHED NEURAL VERIFICATION")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Resolution: {width}x{height} ({width*height:,} pixels)")
    print(f"Output: {output}")
    print()

    c_file = os.path.join(os.path.dirname(__file__), "mandelbrot_putchar_c4.c")
    with open(c_file) as f:
        source = f.read()

    source = source.replace("width = 32;", f"width = {width};")
    source = source.replace("height = 32;", f"height = {height};")

    print("Compiling...")
    bytecode, data = compile_c(source)
    print(f"Compiled: {len(bytecode)} instructions")
    print()

    print("Running with batched neural verification...")
    vm = SpeculativeVMBatched(verify_interval=50000)  # Larger batches = better GPU utilization
    vm.load(bytecode, data)

    start = time.time()
    report_interval = 1000000 if width * height > 10000 else None
    vm.run(report_interval=report_interval)
    elapsed = time.time() - start

    stats = vm.stats()

    print()
    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print()
    print(f"Time: {elapsed/60:.1f} minutes ({elapsed:.1f}s)")
    print(f"VM steps: {stats['steps']:,}")
    print()

    print("OPERATIONS:")
    print(f"  MUL: {stats['mul_count']:,}")
    print(f"  DIV: {stats['div_count']:,}")
    print()

    print("VERIFICATION:")
    print(f"  MUL verified: {stats['verified_mul']:,}")
    print(f"  DIV verified: {stats['verified_div']:,}")
    print(f"  Errors: {stats['verification_errors']}")
    print()

    print("MEMORY:")
    print(f"  Total writes: {stats['total_writes']:,}")
    print(f"  Live entries: {stats['live_memory']:,}")
    print()

    # FLOP calculation
    swiglu_flops = stats['mul_count'] * 91
    newton_flops = stats['div_count'] * 930
    total_verify = swiglu_flops + newton_flops

    print("FLOPS (FFN-based verification):")
    print(f"  SwiGLU MUL: {stats['mul_count']:,} × 91 = {swiglu_flops/1e9:.2f} GFLOPs")
    print(f"  Newton DIV: {stats['div_count']:,} × 930 = {newton_flops/1e9:.2f} GFLOPs")
    print(f"  TOTAL: {total_verify/1e9:.2f} GFLOPs")
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
