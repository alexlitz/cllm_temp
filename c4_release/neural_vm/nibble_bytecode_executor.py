"""
Nibble-Based Bytecode Executor

Executes C4 bytecode programs through the neural VM with nibble-based arithmetic.
"""

import torch
from typing import List, Dict, Optional, Tuple
from neural_vm.vm_step import AutoregressiveVM
from neural_vm.nibble_embedding import NibbleVMEmbedding
from neural_vm.embedding import Opcode


class NibbleBytecodeExecutor:
    """Executes C4 bytecode using the neural VM."""

    def __init__(self, vm: AutoregressiveVM, embed: NibbleVMEmbedding):
        self.vm = vm
        self.embed = embed
        self.max_cycles = 10000  # Safety limit

    def execute(self, code: List[int], data: List[int], verbose: bool = False) -> int:
        """
        Execute a compiled program.

        Args:
            code: List of instruction words (opcode + immediate)
            data: Data segment
            verbose: Print execution trace

        Returns:
            Exit code (value in AX when program terminates)
        """
        # Initialize VM state
        pc = 0
        ax = 0
        sp = 0x100000  # Stack grows downward from 1MB
        bp = sp
        stack = {}     # Simulated stack memory
        memory = {}    # Simulated data memory
        cycle = 0

        # Initialize data segment
        data_base = 0x10000  # Data starts at 64KB
        for i, byte_val in enumerate(data):
            memory[data_base + i] = byte_val

        if verbose:
            print("=" * 70)
            print("STARTING EXECUTION")
            print("=" * 70)
            print(f"PC: {pc:08x}, SP: {sp:08x}, BP: {bp:08x}")
            print()

        while cycle < self.max_cycles:
            cycle += 1

            # Check if PC is in valid range
            if pc >= len(code) * 8:
                if verbose:
                    print(f"PC out of bounds: {pc:08x}")
                break

            # Fetch instruction
            instr_idx = pc // 8
            instr = code[instr_idx]
            opcode = instr & 0xFF
            imm = instr >> 8

            if verbose:
                print(f"[{cycle:04d}] PC: {pc:08x}, OP: {opcode:02d}, IMM: {imm:08x}, AX: {ax:08x}, SP: {sp:08x}")

            # Increment PC for next instruction
            pc += 8

            # Execute instruction
            if opcode == Opcode.IMM:
                # Load immediate
                ax = imm

            elif opcode == Opcode.LEA:
                # Load effective address (BP + offset)
                ax = bp + imm

            elif opcode == Opcode.JMP:
                # Unconditional jump
                pc = imm

            elif opcode == Opcode.JSR:
                # Jump subroutine (call function)
                sp -= 8
                stack[sp] = pc
                pc = imm

            elif opcode == Opcode.BZ:
                # Branch if zero
                if ax == 0:
                    pc = imm

            elif opcode == Opcode.BNZ:
                # Branch if not zero
                if ax != 0:
                    pc = imm

            elif opcode == Opcode.ENT:
                # Enter function (allocate stack frame)
                sp -= 8
                stack[sp] = bp
                bp = sp
                sp -= imm  # Allocate local variables

            elif opcode == Opcode.ADJ:
                # Adjust stack (pop arguments)
                sp += imm

            elif opcode == Opcode.LEV:
                # Leave function (return)
                sp = bp
                if sp in stack:
                    bp = stack[sp]
                    sp += 8
                if sp in stack:
                    pc = stack[sp]
                    sp += 8
                else:
                    # Return from main - exit
                    if verbose:
                        print(f"\nProgram exited with code: {ax}")
                    return ax & 0xFFFFFFFF

            elif opcode == Opcode.LI:
                # Load int from memory
                addr = ax
                if addr in memory:
                    # Load 8 bytes (64-bit int)
                    val = 0
                    for i in range(8):
                        if addr + i in memory:
                            val |= (memory[addr + i] << (i * 8))
                    ax = val & 0xFFFFFFFF
                elif addr in stack:
                    ax = stack[addr] & 0xFFFFFFFF
                else:
                    ax = 0

            elif opcode == Opcode.LC:
                # Load char from memory
                addr = ax
                if addr in memory:
                    ax = memory[addr] & 0xFF
                elif addr in stack:
                    ax = stack[addr] & 0xFF
                else:
                    ax = 0

            elif opcode == Opcode.SI:
                # Store int to memory
                # Stack top has address, AX has value
                if sp in stack:
                    addr = stack[sp]
                    sp += 8
                    # Store 8 bytes
                    val = ax
                    for i in range(8):
                        byte_val = (val >> (i * 8)) & 0xFF
                        if addr >= data_base:
                            memory[addr + i] = byte_val
                        else:
                            stack[addr + i] = byte_val

            elif opcode == Opcode.SC:
                # Store char to memory
                if sp in stack:
                    addr = stack[sp]
                    sp += 8
                    byte_val = ax & 0xFF
                    if addr >= data_base:
                        memory[addr] = byte_val
                    else:
                        stack[addr] = byte_val

            elif opcode == Opcode.PSH:
                # Push AX to stack
                sp -= 8
                stack[sp] = ax

            elif opcode == Opcode.ADD:
                # Add: AX = stack_top + AX (neural execution)
                if sp in stack:
                    stack_top = stack[sp]
                    sp += 8
                    ax = self._execute_arithmetic(Opcode.ADD, stack_top, ax)

            elif opcode == Opcode.SUB:
                # Subtract: AX = stack_top - AX (neural execution)
                if sp in stack:
                    stack_top = stack[sp]
                    sp += 8
                    ax = self._execute_arithmetic(Opcode.SUB, stack_top, ax)

            elif opcode == Opcode.MUL:
                # Multiply (Python fallback for now)
                if sp in stack:
                    stack_top = stack[sp]
                    sp += 8
                    ax = (stack_top * ax) & 0xFFFFFFFF

            elif opcode == Opcode.DIV:
                # Divide (Python fallback for now)
                if sp in stack:
                    stack_top = stack[sp]
                    sp += 8
                    if ax != 0:
                        ax = stack_top // ax
                    else:
                        ax = 0

            elif opcode == Opcode.MOD:
                # Modulo (Python fallback for now)
                if sp in stack:
                    stack_top = stack[sp]
                    sp += 8
                    if ax != 0:
                        ax = stack_top % ax
                    else:
                        ax = 0

            elif opcode == Opcode.OR:
                # Bitwise OR
                if sp in stack:
                    stack_top = stack[sp]
                    sp += 8
                    ax = (stack_top | ax) & 0xFFFFFFFF

            elif opcode == Opcode.XOR:
                # Bitwise XOR
                if sp in stack:
                    stack_top = stack[sp]
                    sp += 8
                    ax = (stack_top ^ ax) & 0xFFFFFFFF

            elif opcode == Opcode.AND:
                # Bitwise AND
                if sp in stack:
                    stack_top = stack[sp]
                    sp += 8
                    ax = (stack_top & ax) & 0xFFFFFFFF

            elif opcode == Opcode.EQ:
                # Equal
                if sp in stack:
                    stack_top = stack[sp]
                    sp += 8
                    ax = 1 if stack_top == ax else 0

            elif opcode == Opcode.NE:
                # Not equal
                if sp in stack:
                    stack_top = stack[sp]
                    sp += 8
                    ax = 1 if stack_top != ax else 0

            elif opcode == Opcode.LT:
                # Less than
                if sp in stack:
                    stack_top = stack[sp]
                    sp += 8
                    ax = 1 if stack_top < ax else 0

            elif opcode == Opcode.GT:
                # Greater than
                if sp in stack:
                    stack_top = stack[sp]
                    sp += 8
                    ax = 1 if stack_top > ax else 0

            elif opcode == Opcode.LE:
                # Less than or equal
                if sp in stack:
                    stack_top = stack[sp]
                    sp += 8
                    ax = 1 if stack_top <= ax else 0

            elif opcode == Opcode.GE:
                # Greater than or equal
                if sp in stack:
                    stack_top = stack[sp]
                    sp += 8
                    ax = 1 if stack_top >= ax else 0

            elif opcode == Opcode.SHL:
                # Shift left
                if sp in stack:
                    stack_top = stack[sp]
                    sp += 8
                    ax = (stack_top << ax) & 0xFFFFFFFF

            elif opcode == Opcode.SHR:
                # Shift right
                if sp in stack:
                    stack_top = stack[sp]
                    sp += 8
                    ax = (stack_top >> ax) & 0xFFFFFFFF

            elif opcode == Opcode.EXIT:
                # Exit program
                if verbose:
                    print(f"\nProgram exited with code: {ax}")
                return ax & 0xFFFFFFFF

            else:
                if verbose:
                    print(f"Unknown opcode: {opcode}")
                break

        if verbose:
            print(f"\nExecution limit reached ({self.max_cycles} cycles)")

        return ax & 0xFFFFFFFF

    def _execute_arithmetic(self, opcode: int, operand_a: int, operand_b: int) -> int:
        """
        Execute ADD or SUB through the neural VM.

        Args:
            opcode: Opcode.ADD or Opcode.SUB
            operand_a: First operand (stack top)
            operand_b: Second operand (AX)

        Returns:
            Result as 32-bit integer
        """
        # For SUB: The ALU computes NIB_A - NIB_B
        # For ADD: The ALU computes NIB_A + NIB_B (commutative, order doesn't matter)
        #
        # C4 bytecode: "a - b" → PSH a, IMM b, SUB
        # At SUB: stack_top=a, ax=b, result should be a-b
        #
        # SUB computes NIB_A - NIB_B, so we need:
        #   NIB_A = a (first operand, operand_a)
        #   NIB_B = b (second operand, operand_b)
        #
        # But encode_vm_state sets:
        #   NIB_A = ax parameter
        #   NIB_B = stack_top parameter
        #
        # So for SUB we must swap: ax=operand_a, stack_top=operand_b
        # For ADD, order doesn't matter, but keep consistent

        if opcode == Opcode.SUB:
            # Swap operands for SUB: encode_vm_state(ax=A, stack_top=B) → NIB_A=A, NIB_B=B
            # Then SUB computes NIB_A - NIB_B = A - B ✓
            input_embedding = self.embed.encode_vm_state(
                pc=0,
                ax=operand_a,        # First operand → NIB_A
                sp=0,
                bp=0,
                opcode=opcode,
                stack_top=operand_b, # Second operand → NIB_B
                batch_size=1,
            )
        else:
            # For ADD: A + B = B + A, so order doesn't matter
            input_embedding = self.embed.encode_vm_state(
                pc=0,
                ax=operand_b,
                sp=0,
                bp=0,
                opcode=opcode,
                stack_top=operand_a,
                batch_size=1,
            )

        # Execute through 3-layer pipeline
        with torch.no_grad():
            x = input_embedding.unsqueeze(1)  # [1, 1, 1280]

            # Layer 9: Raw + Generate
            x = self.vm.blocks[9](x)

            # Layer 10: Carry Lookahead
            x = self.vm.blocks[10](x)

            # Layer 11: Finalize
            x = self.vm.blocks[11](x)

            output_embedding = x.squeeze(1)  # [1, 1280]

        # Decode result
        result = self.embed.decode_result_nibbles(output_embedding)

        return result & 0xFFFFFFFF


def create_executor(verbose: bool = False) -> NibbleBytecodeExecutor:
    """Create an executor with loaded weights."""
    from neural_vm.weight_loader import CompiledWeightLoader

    if verbose:
        print("Creating VM and loading weights...")

    vm = AutoregressiveVM(d_model=1280, n_layers=16, ffn_hidden=4096)
    vm.eval()

    loader = CompiledWeightLoader()
    loader.load_all_weights(vm, verbose=verbose)

    embed = NibbleVMEmbedding(d_model=1280)

    if verbose:
        print("VM ready.")
        print()

    return NibbleBytecodeExecutor(vm, embed)
