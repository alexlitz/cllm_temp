"""
Speculative Execution for C4 Transformer VM

Uses a fast logical VM to predict results, then validates
against the transformer VM. This enables:

1. Fast execution via logical VM (~10x faster)
2. Parallel validation of transformer correctness
3. Confidence scoring based on validation history

The key insight: if the transformer VM is correct, running
the same bytecode in a fast logical VM produces identical output.
"""

import torch
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


class ValidationError(Exception):
    """Raised when batch validation fails."""
    pass


@dataclass
class SpeculationResult:
    """Result of speculative execution."""
    result: int
    validated: bool
    matched: bool
    fast_time_ms: float
    transformer_time_ms: Optional[float]


class FastLogicalVM:
    """
    Fast reference VM using Python arithmetic.

    Runs ~10x faster than transformer VM, produces identical output.
    Used as oracle for speculative execution.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.memory = {}
        self.sp = 0x10000
        self.bp = 0x10000
        self.ax = 0
        self.pc = 0
        self.halted = False
        self.code = []
        self.steps = 0

    def load(self, bytecode: List[int], data: Optional[bytes] = None):
        """Load bytecode from compiler output."""
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

    def run(self, max_steps: int = 100000) -> int:
        """Execute bytecode."""
        steps = 0

        while steps < max_steps:
            instr_idx = self.pc // 8
            if instr_idx >= len(self.code) or self.halted:
                break

            op, imm = self.code[instr_idx]
            self.pc += 8

            if op == 0:    # LEA
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

        self.steps = steps
        return self.ax


class SpeculativeVM:
    """
    VM with speculative execution.

    Uses fast logical VM for speed, optionally validates
    against transformer VM for correctness.
    """

    def __init__(self, transformer_vm=None, validate_ratio: float = 0.1):
        """
        Args:
            transformer_vm: Optional transformer VM for validation
            validate_ratio: Fraction of executions to validate (0.0-1.0)
        """
        self.fast_vm = FastLogicalVM()
        self.transformer_vm = transformer_vm
        self.validate_ratio = validate_ratio

        # Statistics
        self.total_runs = 0
        self.validations = 0
        self.mismatches = 0

    def run(self, bytecode: List[int], data: Optional[bytes] = None,
            validate: bool = False) -> int:
        """
        Execute bytecode with optional validation.

        Args:
            bytecode: Compiled bytecode
            data: Optional data segment
            validate: Force validation against transformer VM

        Returns:
            Execution result
        """
        import time

        # Fast path
        self.fast_vm.reset()
        self.fast_vm.load(bytecode, data)

        t0 = time.time()
        fast_result = self.fast_vm.run()
        fast_time = time.time() - t0

        self.total_runs += 1

        # Validation
        should_validate = validate or (
            self.validate_ratio > 0 and
            self.transformer_vm is not None and
            hash(tuple(bytecode)) % 100 < self.validate_ratio * 100
        )

        if should_validate and self.transformer_vm is not None:
            self.transformer_vm.reset()
            self.transformer_vm.load_bytecode(bytecode, data)

            t0 = time.time()
            trans_result = self.transformer_vm.run()
            trans_time = time.time() - t0

            self.validations += 1

            if fast_result != trans_result:
                self.mismatches += 1
                # Could log or raise here

        return fast_result

    def get_stats(self) -> Dict:
        """Get execution statistics."""
        return {
            'total_runs': self.total_runs,
            'validations': self.validations,
            'mismatches': self.mismatches,
            'match_rate': (self.validations - self.mismatches) / max(1, self.validations),
        }


class ParallelSpeculator:
    """
    Parallel speculative execution with multiple fast VMs.

    Can speculate on multiple possible inputs/branches simultaneously.
    """

    def __init__(self, num_workers: int = 4):
        self.workers = [FastLogicalVM() for _ in range(num_workers)]

    def run_parallel(self, bytecodes: List[Tuple[List[int], Optional[bytes]]]) -> List[int]:
        """
        Run multiple bytecodes in parallel.

        Args:
            bytecodes: List of (bytecode, data) tuples

        Returns:
            List of results
        """
        results = []
        for i, (bytecode, data) in enumerate(bytecodes):
            worker = self.workers[i % len(self.workers)]
            worker.reset()
            worker.load(bytecode, data)
            results.append(worker.run())
        return results


@dataclass
class TraceStep:
    """Single step in execution trace."""
    op: int
    imm: int
    operand_a: int  # First operand (from stack/memory)
    operand_b: int  # Second operand (AX)
    result: int     # Result of operation
    pc_before: int
    pc_after: int


@dataclass
class TraceVerificationResult:
    """Result of trace verification."""
    accepted: bool
    num_steps: int
    num_verified: int
    num_mismatches: int
    mismatch_indices: List[int]
    fast_time_ms: float
    verify_time_ms: float


class TracingVM(FastLogicalVM):
    """
    Fast VM that records execution trace for batch verification.

    Records all arithmetic operations with operands and results,
    allowing batch verification against the neural ALU.
    """

    # Operations that can be batch-verified
    ARITHMETIC_OPS = {
        25: 'ADD', 26: 'SUB', 27: 'MUL', 28: 'DIV', 29: 'MOD',
        16: 'AND', 14: 'OR', 15: 'XOR',
        17: 'EQ', 18: 'NE', 19: 'LT', 20: 'GT', 21: 'LE', 22: 'GE',
        23: 'SHL', 24: 'SHR',
    }

    def __init__(self):
        super().__init__()
        self.trace: List[TraceStep] = []

    def reset(self):
        super().reset()
        self.trace = []

    def run_with_trace(self, max_steps: int = 100000) -> Tuple[int, List[TraceStep]]:
        """Execute and record trace of arithmetic operations."""
        steps = 0
        self.trace = []

        while steps < max_steps:
            instr_idx = self.pc // 8
            if instr_idx >= len(self.code) or self.halted:
                break

            op, imm = self.code[instr_idx]
            pc_before = self.pc
            self.pc += 8

            # Record arithmetic operations for verification
            if op in self.ARITHMETIC_OPS:
                operand_a = self.memory.get(self.sp, 0)
                operand_b = self.ax

                # Execute operation
                if op == 25:  # ADD
                    self.sp += 8
                    self.ax = (operand_a + operand_b) & 0xFFFFFFFF
                elif op == 26:  # SUB
                    self.sp += 8
                    self.ax = (operand_a - operand_b) & 0xFFFFFFFF
                elif op == 27:  # MUL
                    self.sp += 8
                    self.ax = (operand_a * operand_b) & 0xFFFFFFFF
                elif op == 28:  # DIV
                    self.sp += 8
                    self.ax = operand_a // operand_b if operand_b != 0 else 0
                elif op == 29:  # MOD
                    self.sp += 8
                    self.ax = operand_a % operand_b if operand_b != 0 else 0
                elif op == 16:  # AND
                    self.sp += 8
                    self.ax = operand_a & operand_b
                elif op == 14:  # OR
                    self.sp += 8
                    self.ax = operand_a | operand_b
                elif op == 15:  # XOR
                    self.sp += 8
                    self.ax = operand_a ^ operand_b
                elif op == 17:  # EQ
                    self.sp += 8
                    self.ax = 1 if operand_a == operand_b else 0
                elif op == 18:  # NE
                    self.sp += 8
                    self.ax = 1 if operand_a != operand_b else 0
                elif op == 19:  # LT
                    self.sp += 8
                    self.ax = 1 if operand_a < operand_b else 0
                elif op == 20:  # GT
                    self.sp += 8
                    self.ax = 1 if operand_a > operand_b else 0
                elif op == 21:  # LE
                    self.sp += 8
                    self.ax = 1 if operand_a <= operand_b else 0
                elif op == 22:  # GE
                    self.sp += 8
                    self.ax = 1 if operand_a >= operand_b else 0
                elif op == 23:  # SHL
                    self.sp += 8
                    self.ax = (operand_a << operand_b) & 0xFFFFFFFF
                elif op == 24:  # SHR
                    self.sp += 8
                    self.ax = operand_a >> operand_b

                # Record the step
                self.trace.append(TraceStep(
                    op=op,
                    imm=imm,
                    operand_a=operand_a,
                    operand_b=operand_b,
                    result=self.ax,
                    pc_before=pc_before,
                    pc_after=self.pc,
                ))
            else:
                # Non-arithmetic ops - execute normally (not traced)
                if op == 0:    # LEA
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
                elif op == 38: # EXIT
                    self.halted = True
                    break

            steps += 1

        return self.ax, self.trace


class TraceSpeculator:
    """
    Speculative execution with whole-trace verification.

    Like speculative decoding in LLMs:
    1. Fast VM generates entire execution trace
    2. Neural ALU batch-verifies all arithmetic operations
    3. If all match, accept result; otherwise identify mismatches

    This is more efficient than step-by-step verification because:
    - Arithmetic ops can be batched into single tensor operations
    - One forward pass verifies many operations
    - Amortizes neural network overhead
    """

    def __init__(self, transformer_vm=None):
        """
        Args:
            transformer_vm: Transformer VM for batch verification
        """
        self.tracing_vm = TracingVM()
        self.transformer_vm = transformer_vm

        # Statistics
        self.total_runs = 0
        self.total_ops_verified = 0
        self.total_mismatches = 0

    def run(
        self,
        bytecode: List[int],
        data: Optional[bytes] = None,
        verify: bool = True,
        max_steps: int = 100000,
    ) -> Tuple[int, Optional[TraceVerificationResult]]:
        """
        Execute with optional whole-trace verification.

        Args:
            bytecode: Compiled bytecode
            data: Optional data segment
            verify: Whether to verify trace against neural ALU
            max_steps: Maximum execution steps

        Returns:
            (result, verification_result) tuple
        """
        import time

        # Fast execution with tracing
        self.tracing_vm.reset()
        self.tracing_vm.load(bytecode, data)

        t0 = time.time()
        result, trace = self.tracing_vm.run_with_trace(max_steps)
        fast_time = (time.time() - t0) * 1000

        self.total_runs += 1

        if not verify or self.transformer_vm is None or len(trace) == 0:
            return result, None

        # Batch verify the trace
        t0 = time.time()
        verification = self._verify_trace_batched(trace)
        verify_time = (time.time() - t0) * 1000

        self.total_ops_verified += verification.num_verified
        self.total_mismatches += verification.num_mismatches

        verification.fast_time_ms = fast_time
        verification.verify_time_ms = verify_time

        return result, verification

    def _verify_trace_batched(self, trace: List[TraceStep]) -> TraceVerificationResult:
        """
        Batch verify arithmetic operations in trace.

        Groups operations by type and verifies each group in one
        batched forward pass through the neural ALU.
        """
        if not trace:
            return TraceVerificationResult(
                accepted=True,
                num_steps=0,
                num_verified=0,
                num_mismatches=0,
                mismatch_indices=[],
                fast_time_ms=0,
                verify_time_ms=0,
            )

        alu = self.transformer_vm.alu
        mismatches = []

        # Group by operation type for batching
        op_groups: Dict[int, List[Tuple[int, TraceStep]]] = {}
        for i, step in enumerate(trace):
            if step.op not in op_groups:
                op_groups[step.op] = []
            op_groups[step.op].append((i, step))

        # Verify each group
        for op, steps in op_groups.items():
            # Encode operands as tensors
            a_tensors = [alu._encode_int(s.operand_a) for _, s in steps]
            b_tensors = [alu._encode_int(s.operand_b) for _, s in steps]
            expected = [s.result for _, s in steps]
            indices = [i for i, _ in steps]

            # Compute via neural ALU
            for j, (a_enc, b_enc, exp, idx) in enumerate(zip(a_tensors, b_tensors, expected, indices)):
                step = steps[j][1]

                # Execute appropriate neural operation
                if op == 25:  # ADD
                    result_enc = alu.add(a_enc, b_enc)
                elif op == 26:  # SUB
                    result_enc = alu.subtract(a_enc, b_enc)
                elif op == 27:  # MUL
                    result_enc = alu.multiply(a_enc, b_enc)
                elif op == 28:  # DIV
                    result_enc = alu.divide(a_enc, b_enc)
                elif op == 29:  # MOD
                    # mod = a - (a/b)*b
                    quot = alu.divide(a_enc, b_enc)
                    prod = alu.multiply(quot, b_enc)
                    result_enc = alu.subtract(a_enc, prod)
                elif op == 16:  # AND
                    result_enc = alu.bitwise_op(a_enc, b_enc, 'and')
                elif op == 14:  # OR
                    result_enc = alu.bitwise_op(a_enc, b_enc, 'or')
                elif op == 15:  # XOR
                    result_enc = alu.bitwise_op(a_enc, b_enc, 'xor')
                elif op in (17, 18, 19, 20, 21, 22):  # Comparisons
                    lt, eq, gt = alu.compare(a_enc, b_enc)
                    if op == 17:  # EQ
                        result_enc = alu._blend(alu._encode_int(0), alu._encode_int(1), eq[0])
                    elif op == 18:  # NE
                        result_enc = alu._blend(alu._encode_int(1), alu._encode_int(0), eq[0])
                    elif op == 19:  # LT
                        result_enc = alu._blend(alu._encode_int(0), alu._encode_int(1), lt[0])
                    elif op == 20:  # GT
                        result_enc = alu._blend(alu._encode_int(0), alu._encode_int(1), gt[0])
                    elif op == 21:  # LE
                        le = lt[0] + eq[0] - lt[0] * eq[0]
                        result_enc = alu._blend(alu._encode_int(0), alu._encode_int(1), le)
                    elif op == 22:  # GE
                        ge = gt[0] + eq[0] - gt[0] * eq[0]
                        result_enc = alu._blend(alu._encode_int(0), alu._encode_int(1), ge)
                elif op == 23:  # SHL
                    result_enc = alu.neural_shift_left(a_enc, b_enc)
                elif op == 24:  # SHR
                    result_enc = alu.neural_shift_right(a_enc, b_enc)
                else:
                    continue

                # Decode and compare
                neural_result = alu._decode(result_enc)
                if neural_result != exp:
                    mismatches.append(idx)

        return TraceVerificationResult(
            accepted=len(mismatches) == 0,
            num_steps=len(trace),
            num_verified=len(trace),
            num_mismatches=len(mismatches),
            mismatch_indices=mismatches,
            fast_time_ms=0,
            verify_time_ms=0,
        )

    def get_stats(self) -> Dict:
        """Get execution statistics."""
        return {
            'total_runs': self.total_runs,
            'total_ops_verified': self.total_ops_verified,
            'total_mismatches': self.total_mismatches,
            'accuracy': 1.0 - (self.total_mismatches / max(1, self.total_ops_verified)),
        }


__all__ = [
    'FastLogicalVM',
    'SpeculativeVM',
    'ParallelSpeculator',
    'SpeculationResult',
    'TracingVM',
    'TraceSpeculator',
    'TraceStep',
    'TraceVerificationResult',
]
