#!/usr/bin/env python3
"""
Neural VM Lockstep Test

Runs the pure autoregressive neural VM (nn.Sequential forward pass)
alongside a fast integer VM for lockstep verification.

The neural VM's forward pass is 100% neural (SwiGLU FFN + MoE routing).
The runner handles I/O boundaries (instruction fetch, memory, halt) —
analogous to an LLM inference loop handling tokenization and KV cache.
"""

import sys
import os
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compiler import compile_c  # noqa: direct import avoids broken src/__init__.py
from neural_vm.run_vm import NeuralVMRunner, IMMEDIATE_OPS, BINARY_OPS, STORE_OPS, LOAD_OPS
from neural_vm.numpy_runner import NumpyVMRunner
from neural_vm.embedding import E, Opcode
from neural_vm.vm_step import (
    set_register, get_register, set_opcode, set_immediate,
    set_nib_a, set_position_encoding, get_result,
    PC_BASE, SP_BASE, BP_BASE, AX_BASE,
)
from collections import defaultdict
from io import StringIO


# =============================================================================
# Fast VM for verification (unbounded memory)
# =============================================================================

class FastVM:
    def __init__(self):
        self.memory = defaultdict(int)
        self.sp = 0x10000
        self.bp = 0x10000
        self.ax = 0
        self.pc = 0
        self.halted = False
        self.code = []
        self.heap_ptr = 0x200000
        self.step_count = 0
        self.stdout = StringIO()

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

    def snapshot(self):
        return {'ax': self.ax, 'sp': self.sp, 'bp': self.bp, 'pc': self.pc}

    def step(self):
        idx = self.pc >> 3
        if idx >= len(self.code):
            self.halted = True
            return False
        op, imm = self.code[idx]
        mem = self.memory
        self.pc += 8

        if   op == 1:  self.ax = imm
        elif op == 9:  self.ax = mem[self.ax]
        elif op == 13: self.sp -= 8; mem[self.sp] = self.ax
        elif op == 0:  self.ax = self.bp + imm
        elif op == 4:
            if self.ax == 0: self.pc = imm
        elif op == 5:
            if self.ax != 0: self.pc = imm
        elif op == 11: addr = mem[self.sp]; self.sp += 8; mem[addr] = self.ax
        elif op == 7:  self.sp += imm
        elif op == 3:  self.sp -= 8; mem[self.sp] = self.pc; self.pc = imm
        elif op == 8:
            self.sp = self.bp; self.bp = mem[self.sp]; self.sp += 8
            self.pc = mem[self.sp]; self.sp += 8
        elif op == 6:
            self.sp -= 8; mem[self.sp] = self.bp; self.bp = self.sp
            self.sp -= imm
        elif op == 2:  self.pc = imm
        elif op == 10: self.ax = mem[self.ax] & 0xFF
        elif op == 12: addr = mem[self.sp]; self.sp += 8; mem[addr] = self.ax & 0xFF
        elif op == 65:
            self.stdout.write(chr(mem[self.sp] & 0xFF))
            self.ax = mem[self.sp]
        elif op == 34:
            size = mem[self.sp]; self.ax = self.heap_ptr
            self.heap_ptr += size
            if self.heap_ptr & 7:
                self.heap_ptr += 8 - (self.heap_ptr & 7)
        elif op == 35: pass
        elif op == 38: self.halted = True; return False
        else:
            a = mem[self.sp]; self.sp += 8
            if   op == 25: self.ax = a + self.ax
            elif op == 26: self.ax = a - self.ax
            elif op == 17: self.ax = 1 if a == self.ax else 0
            elif op == 16: self.ax = a & self.ax
            elif op == 14: self.ax = a | self.ax
            elif op == 23: self.ax = a << self.ax
            elif op == 19: self.ax = 1 if a < self.ax else 0
            elif op == 27: self.ax = a * self.ax
            elif op == 28: self.ax = a // self.ax if self.ax != 0 else 0
            elif op == 29: self.ax = a % self.ax if self.ax != 0 else 0
            elif op == 18: self.ax = 1 if a != self.ax else 0
            elif op == 20: self.ax = 1 if a > self.ax else 0
            elif op == 21: self.ax = 1 if a <= self.ax else 0
            elif op == 22: self.ax = 1 if a >= self.ax else 0
            elif op == 15: self.ax = a ^ self.ax
            elif op == 24: self.ax = a >> self.ax
        self.step_count += 1
        return True


# =============================================================================
# Neural VM wrapper for lockstep testing
# =============================================================================

class NeuralVMWrapper:
    """Wraps NeuralVMRunner for step-by-step lockstep comparison."""

    def __init__(self, num_carry_iters=7):
        self.runner = NeuralVMRunner(num_carry_iters=num_carry_iters)
        self.halted = False

    def load(self, bytecode, data=None):
        self.runner.load_packed_program(bytecode)
        if data:
            for i, b in enumerate(data):
                self.runner.memory[0x10000 + i] = b

    def snapshot(self):
        return {
            'ax': get_register(self.runner.embedding, AX_BASE),
            'sp': get_register(self.runner.embedding, SP_BASE),
            'bp': get_register(self.runner.embedding, BP_BASE),
            'pc': get_register(self.runner.embedding, PC_BASE),
        }

    def step(self):
        """Execute one instruction. Returns True to continue."""
        if self.halted:
            return False

        runner = self.runner
        pc = runner._read_pc()
        instr_idx = pc // 8
        if instr_idx >= len(runner.code):
            self.halted = True
            return False

        op, imm = runner.code[instr_idx]

        # Save pre-step register values (needed for JSR/ENT post-step)
        pre_pc = pc
        pre_bp = get_register(runner.embedding, BP_BASE)
        pre_sp = get_register(runner.embedding, SP_BASE)

        # Setup embedding for this instruction
        runner._setup_instruction(op, imm)

        # Neural forward pass (the PURE part)
        with torch.no_grad():
            runner.embedding = runner.step(runner.embedding)

        # Post-step I/O handling
        runner._handle_post_step(op, pre_pc, pre_bp, pre_sp)

        if runner.halted:
            self.halted = True
            return False

        return True


# =============================================================================
# Main: lockstep execution
# =============================================================================

def main():
    src_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'demos', 'sudoku.c'
    )
    with open(src_path) as f:
        source = f.read()

    # Parse args: [steps] [--numpy] [--mul-only] [--no-neural]
    use_numpy = '--numpy' in sys.argv or '--mul-only' in sys.argv or '--no-neural' in sys.argv
    mul_only = '--mul-only' in sys.argv
    no_neural = '--no-neural' in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    max_steps = int(args[0]) if args else 1000

    if no_neural:
        mode = "no-neural (integer only)"
    elif mul_only:
        mode = "neural MUL only"
    elif use_numpy:
        mode = "numpy-extracted (all ops)"
    else:
        mode = "torch"
    print("=" * 60)
    print(f"  Neural VM — Lockstep Test ({mode})")
    print("=" * 60)
    print()

    bytecode, data = compile_c(source)
    print(f"Compiled: {len(bytecode)} instructions")

    if use_numpy:
        neural_ops = None  # all ops
        if mul_only:
            neural_ops = {Opcode.MUL}
        elif no_neural:
            neural_ops = set()  # empty = skip all neural
        # Numpy-accelerated neural VM
        print(f"Building numpy neural VM ({mode})...", end=" ", flush=True)
        t0 = time.time()
        nvm_np = NumpyVMRunner(num_carry_iters=7, neural_ops=neural_ops)
        nvm_np.load_packed_program(bytecode)
        if data:
            for i, b in enumerate(data):
                nvm_np.memory[0x10000 + i] = b
        build_time = time.time() - t0
        num_opcodes = len(nvm_np.opcode_forwards)
        total_layers = sum(n for _, n in nvm_np.opcode_forwards.values())
        print(f"done ({build_time:.1f}s, {num_opcodes} opcodes extracted)")
    else:
        # Torch neural VM
        print("Building neural VM (nn.Sequential)...", end=" ", flush=True)
        t0 = time.time()
        nvm = NeuralVMWrapper(num_carry_iters=7)
        nvm.load(bytecode, data)
        build_time = time.time() - t0
        num_layers = len(nvm.runner.step)
        num_params = sum(p.numel() for p in nvm.runner.step.parameters())
        print(f"done ({build_time:.1f}s, {num_layers} layers, {num_params:,} params)")

    # Build fast VM
    fast = FastVM()
    fast.load(bytecode, data)

    check_every = max(1, max_steps // 20)

    print(f"Running {max_steps:,} steps (lockstep)")
    print(f"Comparing every {check_every:,} steps")
    print()

    divergences = 0
    first_diverge_step = None
    t0 = time.time()

    op_names = {0:'LEA',1:'IMM',2:'JMP',3:'JSR',4:'BZ',5:'BNZ',6:'ENT',7:'ADJ',8:'LEV',
                9:'LI',10:'LC',11:'SI',12:'SC',13:'PSH',14:'OR',15:'XOR',16:'AND',
                17:'EQ',18:'NE',19:'LT',20:'GT',21:'LE',22:'GE',23:'SHL',24:'SHR',
                25:'ADD',26:'SUB',27:'MUL',28:'DIV',29:'MOD',34:'MALC',35:'FREE',38:'EXIT',65:'PUTCHAR'}

    for i in range(max_steps):
        # Snapshot BEFORE stepping
        f_snap = fast.snapshot()
        if use_numpy:
            n_snap = nvm_np.snapshot()
        else:
            n_snap = nvm.snapshot()

        # Check for divergence before stepping
        match = all(f_snap[k] == n_snap[k] for k in ['ax', 'sp', 'bp', 'pc'])
        if not match and first_diverge_step is None:
            first_diverge_step = i
            idx = f_snap['pc'] >> 3
            if idx < len(fast.code):
                op = fast.code[idx][0]
                print(f"FIRST DIVERGENCE at step {i} (before stepping):")
                for k in ['ax', 'sp', 'bp', 'pc']:
                    if f_snap[k] != n_snap[k]:
                        print(f"  {k}: fast={f_snap[k]:#x} neural={n_snap[k]:#x}")
                    else:
                        print(f"  {k}: both={f_snap[k]:#x}")
                print(f"  fast opcode: {op_names.get(op, op)}")
                print()

        # Step both
        tok = fast.step()
        if not tok:
            print(f"Fast VM halted at step {i}")
            break

        if use_numpy:
            tok2 = nvm_np.step_one()
        else:
            tok2 = nvm.step()
        if not tok2:
            print(f"Neural VM halted at step {i}")
            break

        # Periodic reporting
        if (i + 1) % check_every == 0 or i == max_steps - 1:
            f_snap = fast.snapshot()
            if use_numpy:
                n_snap = nvm_np.snapshot()
            else:
                n_snap = nvm.snapshot()
            match = all(f_snap[k] == n_snap[k] for k in ['ax', 'sp', 'bp', 'pc'])

            elapsed = time.time() - t0
            rate = (i + 1) / elapsed

            status = "OK" if match else "DIVERGE"
            if not match:
                divergences += 1
                diffs = " ".join(f"{k}:{f_snap[k]}!={n_snap[k]}"
                                 for k in ['ax','sp','bp','pc'] if f_snap[k] != n_snap[k])
                status += " " + diffs

            print(f"  step {i+1:>10,}  {rate:>8.1f} steps/s  {status}")

    total = time.time() - t0
    steps_done = i + 1

    print()
    print("=" * 60)
    print(f"  Steps:          {steps_done:,}")
    print(f"  Time:           {total:.1f}s ({steps_done/total:.1f} steps/s)")
    print(f"  First diverge:  {first_diverge_step}")
    print(f"  Divergences:    {divergences}")
    if use_numpy:
        print(f"  Mode:           numpy-extracted ({num_opcodes} opcodes)")
    else:
        print(f"  Neural layers:  {num_layers}")
        print(f"  Parameters:     {num_params:,}")
    print("=" * 60)


if __name__ == "__main__":
    main()
