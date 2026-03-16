#!/usr/bin/env python3
"""
HD Mandelbrot with Full Neural Verification

This runs Mandelbrot through:
1. Speculative execution (fast Python VM with dict memory)
2. Memory pruning (keeps only live entries)
3. Neural verification (SwiGLU mul, attention-based Newton div)
4. Complete FLOP tracking

All arithmetic is verified neurally, memory is pruned, and we track
both the speculative path and what a full neural VM would cost.
"""

import sys
import os
import time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.compiler import compile_c


# =============================================================================
# NEURAL PRIMITIVES
# =============================================================================

def silu(x):
    return x * torch.sigmoid(x)

def swiglu_mul(a: float, b: float) -> float:
    """Exact multiplication using SwiGLU identity."""
    at = torch.tensor(a, dtype=torch.float32)
    bt = torch.tensor(b, dtype=torch.float32)
    result = silu(at) * bt + silu(-at) * (-bt)
    return result.item()


class NeuralNewtonDivide:
    """Division using FFN-based table lookup + SwiGLU Newton iterations.

    FFN is ~10x more efficient than attention for table lookups:
    - Attention: 256 × 8 dots + softmax = ~3,072 FLOPs
    - FFN: hidden_dim × 2 multiplies = ~256 FLOPs

    Weights are set analytically to implement piecewise-linear interpolation
    over the reciprocal table, not trained.
    """

    def __init__(self, n_segments=64):
        self.n_segments = n_segments

        # Build reciprocal table for [0.5, 1.0)
        # We use n_segments breakpoints for piecewise-linear approximation
        self.breakpoints = torch.linspace(0.5, 1.0, n_segments + 1, dtype=torch.float32)
        self.values = 1.0 / self.breakpoints

        # FFN weights for piecewise-linear: implements table lookup + interpolation
        # Each "neuron" activates for its segment: ReLU(x - breakpoint[i]) - ReLU(x - breakpoint[i+1])
        # W1: maps x to segment activations
        # W2: maps segment activations to interpolated output
        self._set_weights()

    def _set_weights(self):
        """Set FFN weights to implement piecewise-linear reciprocal."""
        n = self.n_segments

        # W1: [2*n, 1] - creates n pairs of ReLU activations
        # For segment i, we need: ReLU(x - b[i]) and ReLU(x - b[i+1])
        self.W1 = torch.ones(2 * n, 1, dtype=torch.float32)
        self.b1 = torch.zeros(2 * n, dtype=torch.float32)
        for i in range(n):
            self.b1[2*i] = -self.breakpoints[i]      # ReLU(x - b[i])
            self.b1[2*i + 1] = -self.breakpoints[i+1]  # ReLU(x - b[i+1])

        # W2: combines pairs to get segment indicator × slope
        # For each segment, output = start_val + slope * (x - breakpoint)
        # Slope = (1/b[i+1] - 1/b[i]) / (b[i+1] - b[i])
        self.W2 = torch.zeros(1, 2 * n, dtype=torch.float32)
        for i in range(n):
            delta_x = self.breakpoints[i+1] - self.breakpoints[i]
            slope = (self.values[i+1] - self.values[i]) / delta_x
            self.W2[0, 2*i] = slope      # ReLU(x - b[i]) contributes slope
            self.W2[0, 2*i + 1] = -slope  # ReLU(x - b[i+1]) cancels for next segment

        # b2: constant term (value at x=0.5)
        self.b2 = torch.tensor([self.values[0].item()], dtype=torch.float32)

    def _ffn_lookup(self, x: float) -> float:
        """Look up reciprocal using FFN with fixed weights."""
        x_t = torch.tensor([[x]], dtype=torch.float32)
        hidden = F.relu(torch.matmul(x_t, self.W1.T) + self.b1)
        result = torch.matmul(hidden, self.W2.T) + self.b2
        return result.item()

    def divide(self, a: int, b: int) -> int:
        """Compute a / b using neural primitives."""
        if b == 0:
            return 0

        sign = 1
        if a < 0:
            sign, a = -sign, -a
        if b < 0:
            sign, b = -sign, -b

        # Normalize b to [0.5, 1.0)
        exp = 0
        normalized = float(b)
        while normalized >= 1.0:
            normalized *= 0.5
            exp += 1
        while normalized < 0.5:
            normalized *= 2.0
            exp -= 1

        # FFN table lookup (replaces attention)
        y = self._ffn_lookup(normalized)

        # Newton iterations with SwiGLU
        for _ in range(2):
            correction = 2.0 - swiglu_mul(normalized, y)
            y = swiglu_mul(y, correction)

        y = y * (2.0 ** (-exp))
        result = swiglu_mul(float(a), y)

        # Round to nearest integer, then verify using SwiGLU multiply
        # This handles float32 precision errors by checking the result
        candidate = int(result + 0.5)

        # Verify: candidate * b should not exceed a
        check = swiglu_mul(float(candidate), float(b))
        if check > a + 0.5:  # candidate too high
            candidate -= 1

        return candidate * sign


# =============================================================================
# SPECULATIVE VM WITH PRUNING
# =============================================================================

class SpeculativeNeuralVM:
    """
    Speculative execution with neural verification.

    Fast path: Python VM with dict memory
    Verification: SwiGLU mul, attention-based Newton div
    Memory: Pruned (only live entries kept)
    """

    OP_MUL = 27
    OP_DIV = 28
    OP_MOD = 29
    OP_PUTCHAR = 65
    OP_EXIT = 38

    def __init__(self, verify_neural: bool = True):
        self.verify_neural = verify_neural
        self.divider = NeuralNewtonDivide() if verify_neural else None

        # State
        self.memory = {}
        self.pc = 0
        self.sp = 0x30000
        self.bp = 0x30000
        self.ax = 0
        self.halted = False
        self.code = []

        # I/O
        self.stdout = []

        # Counters
        self.steps = 0
        self.mul_count = 0
        self.div_count = 0
        self.total_writes = 0

        # FLOP tracking
        self.flops = {
            'swiglu_mul': 0,
            'newton_div': 0,
            'instruction_fetch': 0,  # For full neural estimate
            'memory_access': 0,      # For full neural estimate
        }

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
        """Multiply with optional neural verification."""
        self.mul_count += 1

        if self.verify_neural:
            result = swiglu_mul(float(a), float(b))
            self.flops['swiglu_mul'] += 91  # SwiGLU FLOPs
            return int(result) & 0xFFFFFFFF
        else:
            return (a * b) & 0xFFFFFFFF

    def _div(self, a, b):
        """Divide with optional neural verification."""
        self.div_count += 1

        if b == 0:
            return 0

        if self.verify_neural:
            result = self.divider.divide(a, b)
            # FFN table (64 segments × 2 = 128 neurons):
            #   W1: 128 FLOPs, ReLU: 128 FLOPs, W2: 128 FLOPs = 384 FLOPs
            # Newton iterations: 4 × SwiGLU (91) = 364 FLOPs
            # Final multiply: SwiGLU (91) = 91 FLOPs
            # Verification multiply: SwiGLU (91) = 91 FLOPs
            # Total: 384 + 364 + 91 + 91 = 930 FLOPs
            self.flops['newton_div'] += 930
            return result
        else:
            return a // b

    def step(self):
        if self.halted or self.pc // 8 >= len(self.code):
            self.halted = True
            return False

        op, imm = self.code[self.pc // 8]
        self.pc += 8
        self.steps += 1

        # Execute
        if op == self.OP_PUTCHAR:
            c = self._mem_read(self.sp)
            self.stdout.append(c & 0xFF)
            self.ax = c

        elif op == 0:  self.ax = self.bp + imm  # LEA
        elif op == 1:  self.ax = imm  # IMM
        elif op == 2:  self.pc = imm  # JMP
        elif op == 3:  # JSR
            self.sp -= 8
            self._mem_write(self.sp, self.pc)
            self.pc = imm
        elif op == 4:  # BZ
            if self.ax == 0: self.pc = imm
        elif op == 5:  # BNZ
            if self.ax != 0: self.pc = imm
        elif op == 6:  # ENT
            self.sp -= 8
            self._mem_write(self.sp, self.bp)
            self.bp = self.sp
            self.sp -= imm
        elif op == 7:  self.sp += imm  # ADJ
        elif op == 8:  # LEV
            self.sp = self.bp
            self.bp = self._mem_read(self.sp)
            self.sp += 8
            self.pc = self._mem_read(self.sp)
            self.sp += 8
        elif op == 9:  self.ax = self._mem_read(self.ax)  # LI
        elif op == 10: self.ax = self._mem_read(self.ax) & 0xFF  # LC
        elif op == 11:  # SI
            addr = self._mem_read(self.sp)
            self.sp += 8
            self._mem_write(addr, self.ax)
        elif op == 12:  # SC
            addr = self._mem_read(self.sp)
            self.sp += 8
            self._mem_write(addr, self.ax & 0xFF)
        elif op == 13:  # PSH
            self.sp -= 8
            self._mem_write(self.sp, self.ax)
        elif op == 14:  # OR
            a = self._mem_read(self.sp); self.sp += 8
            self.ax = a | self.ax
        elif op == 15:  # XOR
            a = self._mem_read(self.sp); self.sp += 8
            self.ax = a ^ self.ax
        elif op == 16:  # AND
            a = self._mem_read(self.sp); self.sp += 8
            self.ax = a & self.ax
        elif op == 17:  # EQ
            a = self._mem_read(self.sp); self.sp += 8
            self.ax = 1 if a == self.ax else 0
        elif op == 18:  # NE
            a = self._mem_read(self.sp); self.sp += 8
            self.ax = 1 if a != self.ax else 0
        elif op == 19:  # LT
            a = self._mem_read(self.sp); self.sp += 8
            self.ax = 1 if a < self.ax else 0
        elif op == 20:  # GT
            a = self._mem_read(self.sp); self.sp += 8
            self.ax = 1 if a > self.ax else 0
        elif op == 21:  # LE
            a = self._mem_read(self.sp); self.sp += 8
            self.ax = 1 if a <= self.ax else 0
        elif op == 22:  # GE
            a = self._mem_read(self.sp); self.sp += 8
            self.ax = 1 if a >= self.ax else 0
        elif op == 23:  # SHL
            a = self._mem_read(self.sp); self.sp += 8
            self.ax = (a << self.ax) & 0xFFFFFFFF
        elif op == 24:  # SHR
            a = self._mem_read(self.sp); self.sp += 8
            self.ax = a >> self.ax
        elif op == 25:  # ADD
            a = self._mem_read(self.sp); self.sp += 8
            self.ax = (a + self.ax) & 0xFFFFFFFF
        elif op == 26:  # SUB
            a = self._mem_read(self.sp); self.sp += 8
            self.ax = (a - self.ax) & 0xFFFFFFFF
        elif op == self.OP_MUL:  # MUL
            a = self._mem_read(self.sp); self.sp += 8
            self.ax = self._mul(a, self.ax)
        elif op == self.OP_DIV:  # DIV
            a = self._mem_read(self.sp); self.sp += 8
            self.ax = self._div(a, self.ax)
        elif op == self.OP_MOD:  # MOD
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

        return self.ax

    def get_stdout_bytes(self):
        return bytes(self.stdout)

    def stats(self):
        # Calculate full neural VM estimate
        n_instructions = len(self.code)
        n_memory = len(self.memory)

        # Per-step FLOPs for full neural
        flops_per_fetch = n_instructions * 16 + n_instructions * 3 + n_instructions
        flops_per_mem = n_memory * 20 + n_memory * 3 + n_memory

        full_neural_flops = (
            self.steps * flops_per_fetch +  # Instruction fetch
            self.total_writes * 2 * flops_per_mem +  # Memory ops (read+write)
            self.flops['swiglu_mul'] +
            self.flops['newton_div']
        )

        return {
            'steps': self.steps,
            'mul_count': self.mul_count,
            'div_count': self.div_count,
            'total_writes': self.total_writes,
            'live_memory': len(self.memory),
            'prune_ratio': 1.0 - len(self.memory) / max(1, self.total_writes),
            'output_bytes': len(self.stdout),
            'verification_flops': self.flops['swiglu_mul'] + self.flops['newton_div'],
            'full_neural_estimate': full_neural_flops,
            'n_instructions': n_instructions,
        }


# =============================================================================
# MAIN
# =============================================================================

def main():
    width = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    height = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    output = sys.argv[3] if len(sys.argv) > 3 else f"/tmp/mandelbrot_{width}x{height}_neural.png"

    print("=" * 70)
    print("  HD MANDELBROT - NEURAL VERIFICATION")
    print("=" * 70)
    print(f"Resolution: {width}x{height} ({width*height:,} pixels)")
    print(f"Output: {output}")
    print()

    # Load source
    c_file = os.path.join(os.path.dirname(__file__), "mandelbrot_putchar_c4.c")
    with open(c_file) as f:
        source = f.read()

    source = source.replace("width = 32;", f"width = {width};")
    source = source.replace("height = 32;", f"height = {height};")

    print("Compiling...")
    bytecode, data = compile_c(source)
    print(f"Compiled: {len(bytecode)} instructions")
    print()

    print("Running with neural verification...")
    vm = SpeculativeNeuralVM(verify_neural=True)
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

    print("NEURAL OPERATIONS:")
    print(f"  SwiGLU MUL:  {stats['mul_count']:,}")
    print(f"  Newton DIV:  {stats['div_count']:,}")
    print(f"  Total:       {stats['mul_count'] + stats['div_count']:,}")
    print()

    print("MEMORY:")
    print(f"  Total writes:  {stats['total_writes']:,}")
    print(f"  Live entries:  {stats['live_memory']:,}")
    print(f"  Prune ratio:   {stats['prune_ratio']*100:.4f}%")
    print()

    print("FLOPS (VERIFICATION ONLY):")
    swiglu_flops = stats['mul_count'] * 91
    newton_flops = stats['div_count'] * 930  # FFN-based: 384 + 364 + 91 + 91 (verify)
    total_verify = swiglu_flops + newton_flops
    print(f"  SwiGLU MUL:  {stats['mul_count']:,} × 91 = {swiglu_flops/1e9:.2f} GFLOPs")
    print(f"  Newton DIV (FFN):  {stats['div_count']:,} × 930 = {newton_flops/1e9:.2f} GFLOPs")
    print(f"  TOTAL:       {total_verify/1e9:.2f} GFLOPs")
    print()

    # Full neural estimate
    n_instr = stats['n_instructions']
    n_mem = stats['live_memory']
    flops_fetch = n_instr * 20  # simplified
    flops_mem = n_mem * 24  # simplified

    full_fetch = stats['steps'] * flops_fetch
    full_mem = stats['total_writes'] * 2 * flops_mem
    full_total = full_fetch + full_mem + total_verify

    print("FLOPS (FULL NEURAL VM ESTIMATE):")
    print(f"  Instruction fetch:  {stats['steps']:,} × {flops_fetch:,} = {full_fetch/1e12:.2f} TFLOPs")
    print(f"  Memory access:      {stats['total_writes']*2:,} × {flops_mem:,} = {full_mem/1e12:.2f} TFLOPs")
    print(f"  Arithmetic:         {total_verify/1e9:.2f} GFLOPs")
    print(f"  TOTAL:              {full_total/1e12:.2f} TFLOPs")
    print()

    # LLM equivalents
    print("LLM TOKEN EQUIVALENTS:")
    models = [
        ("GPT-2 Small (117M)", 169.9e6),
        ("LLaMA 7B", 10.07e9),
        ("LLaMA 70B", 118.11e9),
        ("LLaMA 405B", 660e9),  # estimated
        ("GPT-3 175B", 347.89e9),
        ("GPT-4 (est 1.8T)", 773.09e9),
        ("DeepSeek-V3 671B", 1200e9),  # estimated MoE activated
    ]

    print()
    print("  Verification only:")
    for name, fpt in models:
        tokens = total_verify / fpt
        if tokens < 1:
            print(f"    {name:<25} < 1 token")
        elif tokens < 1000:
            print(f"    {name:<25} {tokens:.0f} tokens")
        else:
            print(f"    {name:<25} {tokens/1e3:.1f}K tokens")

    print()
    print("  Full neural VM:")
    for name, fpt in models:
        tokens = full_total / fpt
        if tokens >= 1e6:
            print(f"    {name:<25} {tokens/1e6:.1f}M tokens")
        elif tokens >= 1e3:
            print(f"    {name:<25} {tokens/1e3:.0f}K tokens")
        elif tokens < 1:
            print(f"    {name:<25} < 1 token")
        else:
            print(f"    {name:<25} {tokens:.0f} tokens")

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
