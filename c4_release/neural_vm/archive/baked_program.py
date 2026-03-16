"""
Baked Program - Bytecode compiled directly into transformer weights.

Instead of attention over bytecode tokens, the program IS the network.
Instruction fetch becomes FFN evaluation with baked weights.

Architecture:
  Layer 1 (NibbleEqualityFFN): Check each nibble of PC against all values
  Layer 2 (AddressMatchFFN): AND nibble matches to get exact address match
  Layer 3 (InstructionMoE): Route to expert that returns (opcode, immediate)

This turns a program into a transformer - the weights encode the code.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


# =============================================================================
# NIBBLE EQUALITY FFN (Layer 1)
# =============================================================================

class NibbleEqualityFFN(nn.Module):
    """
    Check if each nibble of the input equals each possible value (0-15).

    For an 8-nibble (32-bit) address, produces 8 × 16 = 128 outputs.
    Output[nibble_pos * 16 + value] = 1.0 iff nibble_pos has that value.

    Uses SwiGLU: silu(W_up @ x) * (W_gate @ x)
    - W_up extracts the nibble and checks equality
    - W_gate provides sharp activation via threshold
    """

    def __init__(self, num_nibbles: int = 8):
        super().__init__()
        self.num_nibbles = num_nibbles
        self.num_outputs = num_nibbles * 16

        # Input: one-hot nibbles [num_nibbles, 16] flattened to [num_nibbles * 16]
        input_dim = num_nibbles * 16

        # W_up: For each (nibble_pos, target_value) pair, extract that nibble
        # W_gate: Threshold to get sharp 0/1
        # W_down: Identity (pass through the match indicators)

        self.W_up = nn.Parameter(torch.zeros(self.num_outputs, input_dim))
        self.b_up = nn.Parameter(torch.zeros(self.num_outputs))
        self.W_gate = nn.Parameter(torch.zeros(self.num_outputs, input_dim))
        self.b_gate = nn.Parameter(torch.zeros(self.num_outputs))

        self._bake_weights()

    def _bake_weights(self):
        """Bake equality check weights."""
        S = 100.0  # Scale for sharp activation

        with torch.no_grad():
            for nib_pos in range(self.num_nibbles):
                for target_val in range(16):
                    out_idx = nib_pos * 16 + target_val
                    in_idx = nib_pos * 16 + target_val

                    # W_up reads the specific nibble value
                    # If input has 1.0 at in_idx, W_up @ x = S
                    self.W_up[out_idx, in_idx] = S

                    # W_gate also reads the same position
                    # silu(S) * S ≈ S^2 when match, ≈ 0 when no match
                    self.W_gate[out_idx, in_idx] = S

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, num_nibbles * 16] one-hot encoded nibbles

        Returns:
            [batch, num_outputs] match indicators (≈1.0 where nibble equals value)
        """
        up = F.linear(x, self.W_up, self.b_up)
        gate = F.linear(x, self.W_gate, self.b_gate)
        hidden = F.silu(up) * gate
        # Normalize to 0-1 range
        return torch.clamp(hidden / 10000.0, 0, 1)


# =============================================================================
# ADDRESS MATCH FFN (Layer 2)
# =============================================================================

class AddressMatchFFN(nn.Module):
    """
    AND together nibble matches to get exact address match.

    For N nibble matches, all must be 1 for the address to match.

    Implements N-way AND via:
    1. Sum the N required nibble indicators (W_select @ x)
    2. If sum == N, all matched. Use threshold at N-0.5.
    3. Apply sharp sigmoid to get 0/1 output.

    Alternative (MoE-style): Use top-1 routing which naturally selects max.
    """

    def __init__(self, addresses: List[int], num_nibbles: int = 8):
        super().__init__()
        self.addresses = addresses
        self.num_nibbles = num_nibbles
        self.num_addresses = len(addresses)

        input_dim = num_nibbles * 16

        # W_select: For each address, select (sum) the required nibble indicators
        # bias: Threshold at -(N - 0.5) so output > 0 iff all N match
        W_select = torch.zeros(self.num_addresses, input_dim)
        bias = torch.zeros(self.num_addresses)

        for addr_idx, addr in enumerate(addresses):
            nibbles = [(addr >> (i * 4)) & 0xF for i in range(num_nibbles)]
            for nib_pos, nib_val in enumerate(nibbles):
                in_idx = nib_pos * 16 + nib_val
                W_select[addr_idx, in_idx] = 1.0
            # Threshold: need exactly N matches
            bias[addr_idx] = -(num_nibbles - 0.5)

        self.register_buffer('W_select', W_select)
        self.register_buffer('bias', bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, num_nibbles * 16] nibble match indicators (0 or 1)

        Returns:
            [batch, num_addresses] address match indicators (one-hot)
        """
        # Sum required nibble matches, subtract threshold
        scores = F.linear(x, self.W_select, self.bias)
        # Sharp sigmoid to get 0/1
        # score > 0 means all nibbles matched
        return torch.sigmoid(scores * 100)


# =============================================================================
# INSTRUCTION MOE (Layer 3)
# =============================================================================

class InstructionMoE(nn.Module):
    """
    Mixture of Experts where each expert returns one instruction.

    Router = address match indicators (one-hot from AddressMatchFFN)
    Experts = fixed (opcode, immediate) outputs stored as weight matrix

    This is implemented as:
    - W_values: [num_experts, 2] matrix where each row is (opcode, immediate)
    - output = router_weights @ W_values

    Since router is one-hot, this selects exactly one row.
    Equivalent to: output = W_values[argmax(router_weights)]
    """

    def __init__(self, instructions: Dict[int, Tuple[int, int]]):
        """
        Args:
            instructions: {address: (opcode, immediate)} mapping
        """
        super().__init__()
        self.addresses = sorted(instructions.keys())
        self.num_experts = len(self.addresses)

        # W_values: Each row is the (opcode, immediate) for that address
        # This IS the "expert" - just a bias term essentially
        W_values = torch.zeros(self.num_experts, 2)
        for i, addr in enumerate(self.addresses):
            op, imm = instructions[addr]
            W_values[i, 0] = float(op)
            W_values[i, 1] = float(imm)

        # Use Parameter so it shows in parameter count
        self.W_values = nn.Parameter(W_values, requires_grad=False)

    def forward(self, router_weights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            router_weights: [batch, num_experts] routing weights (one-hot)

        Returns:
            [batch, 2] (opcode, immediate)
        """
        # Matrix multiply: router_weights @ W_values
        # Since router_weights is one-hot, this selects exactly one row
        return torch.matmul(router_weights, self.W_values)


# =============================================================================
# BAKED PROGRAM (Complete instruction fetch as FFN)
# =============================================================================

class BakedProgram(nn.Module):
    """
    A program baked into transformer weights.

    Instruction fetch is purely FFN evaluation:
      pc_nibbles -> NibbleEquality -> AddressMatch -> InstructionMoE -> (op, imm)

    No attention needed - the program IS the network.
    """

    def __init__(self, instructions: Dict[int, Tuple[int, int]], num_nibbles: int = 8):
        """
        Args:
            instructions: {address: (opcode, immediate)} mapping
            num_nibbles: Nibbles per address (8 for 32-bit)
        """
        super().__init__()
        self.instructions = instructions
        self.addresses = sorted(instructions.keys())
        self.num_nibbles = num_nibbles

        # Layer 1: Nibble equality checks
        self.nibble_eq = NibbleEqualityFFN(num_nibbles)

        # Layer 2: Address matching (AND of nibble matches)
        self.addr_match = AddressMatchFFN(self.addresses, num_nibbles)

        # Layer 3: MoE to return instruction
        self.instruction_moe = InstructionMoE(instructions)

    def encode_pc(self, pc: int) -> torch.Tensor:
        """Encode PC as one-hot nibbles."""
        encoding = torch.zeros(self.num_nibbles * 16)
        for i in range(self.num_nibbles):
            nibble = (pc >> (i * 4)) & 0xF
            encoding[i * 16 + nibble] = 1.0
        return encoding

    def forward(self, pc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fetch instruction at PC.

        Args:
            pc: [batch, num_nibbles * 16] one-hot encoded PC

        Returns:
            (opcode, immediate) tensors
        """
        # Layer 1: Check nibble equalities
        nibble_matches = self.nibble_eq(pc)

        # Layer 2: AND to get address matches
        addr_matches = self.addr_match(nibble_matches)

        # Layer 3: Route to instruction expert
        instruction = self.instruction_moe(addr_matches)

        return instruction[:, 0], instruction[:, 1]

    def fetch(self, pc: int) -> Tuple[int, int]:
        """Convenience method: fetch instruction at integer PC."""
        pc_encoded = self.encode_pc(pc).unsqueeze(0)
        with torch.no_grad():
            op, imm = self.forward(pc_encoded)
        return int(round(op.item())), int(round(imm.item()))


# =============================================================================
# BAKED VM (VM using baked program)
# =============================================================================

class BakedVM:
    """
    VM where the program is baked into FFN weights.

    This is a transformer that IS the program, not one that runs bytecode.
    """

    # Opcodes (C4 compatible)
    LEA, IMM, JMP, JSR, BZ, BNZ, ENT, ADJ, LEV = 0, 1, 2, 3, 4, 5, 6, 7, 8
    LI, LC, SI, SC, PSH = 9, 10, 11, 12, 13
    OR, XOR, AND, EQ, NE, LT, GT, LE, GE = 14, 15, 16, 17, 18, 19, 20, 21, 22
    SHL, SHR, ADD, SUB, MUL, DIV, MOD = 23, 24, 25, 26, 27, 28, 29
    EXIT = 38

    def __init__(self, program: BakedProgram):
        self.program = program
        self.reset()

    def reset(self):
        self.ax = 0
        self.sp = 0x10000
        self.bp = 0x10000
        self.pc = 0
        self.halted = False
        self.stack = {}
        self.memory = {}

    def _push(self, v):
        self.sp -= 8
        self.stack[self.sp] = v

    def _pop(self):
        v = self.stack.get(self.sp, 0)
        self.sp += 8
        return v

    def _signed(self, v):
        return v - 0x100000000 if v >= 0x80000000 else v

    def step(self) -> bool:
        if self.halted:
            return False

        # Instruction fetch via baked FFN (not attention!)
        op, imm = self.program.fetch(self.pc)
        self.pc += 5  # 5 bytes per instruction

        # Execute (same as before)
        if op == self.IMM:
            self.ax = imm
        elif op == self.LEA:
            self.ax = self.bp + imm
        elif op == self.JMP:
            self.pc = imm
        elif op == self.JSR:
            self._push(self.pc)
            self.pc = imm
        elif op == self.BZ:
            if self.ax == 0:
                self.pc = imm
        elif op == self.BNZ:
            if self.ax != 0:
                self.pc = imm
        elif op == self.ENT:
            self._push(self.bp)
            self.bp = self.sp
            self.sp -= imm
        elif op == self.ADJ:
            self.sp += imm
        elif op == self.LEV:
            self.sp = self.bp
            self.bp = self._pop()
            self.pc = self._pop()
        elif op == self.LI:
            self.ax = self.memory.get(self.ax, 0)
        elif op == self.LC:
            self.ax = self.memory.get(self.ax, 0) & 0xFF
        elif op == self.SI:
            self.memory[self._pop()] = self.ax
        elif op == self.SC:
            self.memory[self._pop()] = self.ax & 0xFF
        elif op == self.PSH:
            self._push(self.ax)
        elif op == self.ADD:
            self.ax = (self._pop() + self.ax) & 0xFFFFFFFF
        elif op == self.SUB:
            self.ax = (self._pop() - self.ax) & 0xFFFFFFFF
        elif op == self.MUL:
            self.ax = (self._pop() * self.ax) & 0xFFFFFFFF
        elif op == self.DIV:
            b, a = self.ax, self._pop()
            self.ax = a // b if b else 0
        elif op == self.MOD:
            b, a = self.ax, self._pop()
            self.ax = a % b if b else 0
        elif op == self.AND:
            self.ax = self._pop() & self.ax
        elif op == self.OR:
            self.ax = self._pop() | self.ax
        elif op == self.XOR:
            self.ax = self._pop() ^ self.ax
        elif op == self.SHL:
            self.ax = (self._pop() << self.ax) & 0xFFFFFFFF
        elif op == self.SHR:
            self.ax = self._pop() >> self.ax
        elif op == self.EQ:
            self.ax = 1 if self._pop() == self.ax else 0
        elif op == self.NE:
            self.ax = 1 if self._pop() != self.ax else 0
        elif op == self.LT:
            self.ax = 1 if self._signed(self._pop()) < self._signed(self.ax) else 0
        elif op == self.GT:
            self.ax = 1 if self._signed(self._pop()) > self._signed(self.ax) else 0
        elif op == self.LE:
            self.ax = 1 if self._signed(self._pop()) <= self._signed(self.ax) else 0
        elif op == self.GE:
            self.ax = 1 if self._signed(self._pop()) >= self._signed(self.ax) else 0
        elif op == self.EXIT:
            self.halted = True
            return False

        return True

    def run(self, max_steps: int = 1000000) -> int:
        steps = 0
        while steps < max_steps and self.step():
            steps += 1
        return self.ax


# =============================================================================
# COMPILER: Bytecode -> BakedProgram
# =============================================================================

def compile_to_baked(bytecode: List[int]) -> BakedProgram:
    """
    Compile C4 bytecode into a BakedProgram (weights).

    This transforms the program into the network itself.
    """
    instructions = {}

    # C4 format: opcode and immediate are separate words
    # We pack them as: instructions[addr] = (opcode, immediate)
    pc = 0
    i = 0
    while i < len(bytecode):
        op = bytecode[i]

        # Opcodes that have immediates
        if op <= 8:  # LEA, IMM, JMP, JSR, BZ, BNZ, ENT, ADJ, LEV
            if op != 8:  # LEV has no immediate
                imm = bytecode[i + 1] if i + 1 < len(bytecode) else 0
                instructions[pc] = (op, imm)
                pc += 5  # 5 bytes per instruction
                i += 2
            else:
                instructions[pc] = (op, 0)
                pc += 5
                i += 1
        else:
            # No immediate
            instructions[pc] = (op, 0)
            pc += 5
            i += 1

    return BakedProgram(instructions)


def compile_packed_to_baked(bytecode: List[int]) -> BakedProgram:
    """
    Compile packed bytecode (op|imm<<8) into BakedProgram.
    """
    instructions = {}

    for i, instr in enumerate(bytecode):
        op = instr & 0xFF
        imm = instr >> 8
        if imm >= (1 << 31):  # Sign extend 32-bit
            imm -= (1 << 32)
        instructions[i * 5] = (op, imm)  # 5 bytes per instruction

    return BakedProgram(instructions)


# =============================================================================
# DEMO
# =============================================================================

def demo():
    print("=" * 70)
    print("BAKED PROGRAM DEMO")
    print("Program compiled into transformer weights")
    print("=" * 70)

    # Simple program: return 6 * 7
    # Using packed format: op | (imm << 8)
    bytecode = [
        1 | (6 << 8),    # IMM 6
        13,              # PSH
        1 | (7 << 8),    # IMM 7
        27,              # MUL
        38,              # EXIT
    ]

    print("\nBytecode:")
    for i, instr in enumerate(bytecode):
        op = instr & 0xFF
        imm = instr >> 8
        op_names = {1: 'IMM', 13: 'PSH', 27: 'MUL', 38: 'EXIT'}
        print(f"  {i*5:3d}: {op_names.get(op, f'OP{op}'):4s} {imm if imm else ''}")

    # Compile to baked weights
    program = compile_packed_to_baked(bytecode)

    print(f"\nBaked Program:")
    print(f"  Addresses: {program.addresses}")
    print(f"  NibbleEqualityFFN: {program.nibble_eq.num_outputs} outputs")
    print(f"  AddressMatchFFN: {program.addr_match.num_addresses} addresses")
    print(f"  InstructionMoE: {program.instruction_moe.num_experts} experts")

    # Test instruction fetch
    print(f"\nInstruction Fetch (via FFN, no attention):")
    for addr in program.addresses:
        op, imm = program.fetch(addr)
        op_names = {1: 'IMM', 13: 'PSH', 27: 'MUL', 38: 'EXIT'}
        print(f"  PC={addr:3d} -> op={op:2d} ({op_names.get(op, '?'):4s}) imm={imm}")

    # Run the baked VM
    vm = BakedVM(program)
    result = vm.run()

    print(f"\nExecution result: {result}")
    print(f"Expected: 42")

    # Count parameters
    total_params = sum(p.numel() for p in program.parameters())
    print(f"\nTotal parameters in baked program: {total_params}")

    return program, vm


if __name__ == "__main__":
    demo()
