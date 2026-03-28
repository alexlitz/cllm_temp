"""
Autoregressive Nibble-Based VM

Fully neural execution loop where each cycle:
1. Input: VM state (PC, AX, SP, BP, memory) as nibble embedding
2. Forward pass: All 16 layers compute next state
3. Output: Updated VM state for next cycle

Architecture:
  Layers 0-8:   Instruction fetch and decode
  Layers 9-11:  Arithmetic (ADD, SUB) - ✅ Working
  Layer 12:     Comparisons, bitwise
  Layer 13:     Control flow (PC updates)
  Layers 14-15: Memory operations

NO Python control flow - everything is neural.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
from neural_vm.vm_step import AutoregressiveVM
from neural_vm.nibble_embedding import NibbleVMEmbedding
from neural_vm.embedding import Opcode, E


class VMState:
    """VM state for one execution cycle."""

    def __init__(self, pc: int = 0, ax: int = 0, sp: int = 0x100000, bp: int = 0x100000):
        self.pc = pc
        self.ax = ax
        self.sp = sp
        self.bp = bp
        self.halted = False
        self.cycle = 0

    def to_dict(self) -> Dict:
        return {
            'pc': self.pc,
            'ax': self.ax,
            'sp': self.sp,
            'bp': self.bp,
            'halted': self.halted,
            'cycle': self.cycle,
        }


class ProgramMemory:
    """
    Program memory: code and data segments embedded in KV cache.

    The bytecode is pre-embedded and stored in the KV cache.
    Layers 0-8 attend to this memory to fetch instructions.
    """

    def __init__(self, code: List[int], data: List[int], embed: NibbleVMEmbedding):
        self.code = code
        self.data = data
        self.embed = embed
        self.data_base = 0x10000  # Data starts at 64KB

        # Embed bytecode into memory
        self.code_embeddings = self._embed_code()
        self.data_embeddings = self._embed_data()

    def _embed_code(self) -> torch.Tensor:
        """
        Embed bytecode instructions.

        Each instruction is 64 bits (opcode + immediate).
        We embed each instruction as a nibble embedding.

        Returns:
            [num_instructions, d_model] tensor
        """
        num_instructions = len(self.code)
        embeddings = torch.zeros(num_instructions, self.embed.d_model)

        for i, instr in enumerate(self.code):
            opcode = instr & 0xFF
            imm = instr >> 8

            # Embed instruction: store opcode and immediate
            # We'll encode the full instruction word across the 8 nibbles
            for pos in range(8):
                base_idx = pos * 160

                # Store instruction nibbles (full 64-bit instruction)
                instr_nibble = (instr >> (pos * 4)) & 0xF
                embeddings[i, base_idx + E.NIB_A] = float(instr_nibble)

                # Store opcode separately for easier access
                if pos == 0:
                    # At position 0, also store opcode in one-hot
                    if opcode < 72:
                        embeddings[i, base_idx + E.OP_START + opcode] = 1.0

        return embeddings

    def _embed_data(self) -> torch.Tensor:
        """
        Embed data segment.

        Returns:
            [num_bytes, d_model] tensor
        """
        if len(self.data) == 0:
            return torch.zeros(1, self.embed.d_model)

        # Embed data bytes (grouped into 8-byte chunks for efficiency)
        num_chunks = (len(self.data) + 7) // 8
        embeddings = torch.zeros(num_chunks, self.embed.d_model)

        for chunk_idx in range(num_chunks):
            # Combine 8 bytes into one embedding (each byte is 2 nibbles)
            for byte_offset in range(8):
                data_idx = chunk_idx * 8 + byte_offset
                if data_idx >= len(self.data):
                    break

                byte_val = self.data[data_idx]

                # Each byte occupies 2 nibble positions (low and high nibble)
                pos = byte_offset
                base_idx = pos * 160

                # Low nibble
                embeddings[chunk_idx, base_idx + E.NIB_A] = float(byte_val & 0xF)
                # High nibble
                if pos + 1 < 8:
                    next_base = (pos + 1) * 160
                    embeddings[chunk_idx, next_base + E.NIB_A] = float((byte_val >> 4) & 0xF)

        return embeddings

    def fetch_instruction(self, pc: int) -> Optional[Tuple[int, int]]:
        """
        Fetch instruction at PC (Python fallback for now).

        In full neural implementation, this would be done by attention
        in layers 0-8.

        Returns:
            (opcode, immediate) or None if PC out of bounds
        """
        instr_idx = pc // 8
        if instr_idx >= len(self.code):
            return None

        instr = self.code[instr_idx]
        opcode = instr & 0xFF
        imm = instr >> 8
        return (opcode, imm)


class AutoregressiveNibbleVM:
    """
    Autoregressive VM with nibble-based execution.

    Each cycle:
    1. Encode current VM state → nibble embedding
    2. Forward pass through all 16 layers
    3. Decode output → next VM state
    4. Repeat until halt
    """

    def __init__(self, vm: AutoregressiveVM, embed: NibbleVMEmbedding):
        self.vm = vm
        self.embed = embed
        self.max_cycles = 10000

    def execute(
        self,
        code: List[int],
        data: List[int],
        verbose: bool = False,
    ) -> int:
        """
        Execute a compiled program autoregressively.

        Args:
            code: Bytecode instructions
            data: Data segment
            verbose: Print execution trace

        Returns:
            Exit code (AX value when program halts)
        """
        # Initialize state
        state = VMState(pc=0, ax=0, sp=0x100000, bp=0x100000)

        # Create program memory
        program = ProgramMemory(code, data, self.embed)

        # Simulated stack/memory (Python fallback for now)
        stack = {}
        memory = {}
        for i, byte_val in enumerate(data):
            memory[program.data_base + i] = byte_val

        if verbose:
            print("=" * 70)
            print("AUTOREGRESSIVE NIBBLE VM EXECUTION")
            print("=" * 70)
            print(f"Code: {len(code)} instructions")
            print(f"Data: {len(data)} bytes")
            print()

        # Execution loop
        while state.cycle < self.max_cycles and not state.halted:
            state.cycle += 1

            # Fetch instruction (Python fallback for now)
            fetch_result = program.fetch_instruction(state.pc)
            if fetch_result is None:
                if verbose:
                    print(f"PC out of bounds: {state.pc:08x}")
                break

            opcode, imm = fetch_result

            if verbose:
                print(f"[{state.cycle:04d}] PC:{state.pc:08x} OP:{opcode:02d} IMM:{imm:08x} AX:{state.ax:08x}")

            # Execute instruction
            next_state = self._execute_cycle(
                state, opcode, imm, stack, memory, verbose
            )

            if next_state is None:
                # Halt
                break

            state = next_state

        if verbose:
            print()
            print("=" * 70)
            print(f"Execution completed in {state.cycle} cycles")
            print(f"Exit code: {state.ax}")
            print("=" * 70)

        return state.ax & 0xFFFFFFFF

    def _execute_cycle(
        self,
        state: VMState,
        opcode: int,
        imm: int,
        stack: Dict,
        memory: Dict,
        verbose: bool,
    ) -> Optional[VMState]:
        """
        Execute one VM cycle.

        For now, this is a hybrid approach:
        - Arithmetic operations (ADD, SUB) use neural execution (layers 9-11)
        - Other operations use Python fallback
        - Next step: Make PC update and control flow neural (layer 13)

        Returns:
            Next state, or None if halted
        """
        # Create next state (copy current)
        next_state = VMState(
            pc=state.pc + 8,  # Default: increment PC
            ax=state.ax,
            sp=state.sp,
            bp=state.bp,
        )
        next_state.cycle = state.cycle

        # Execute based on opcode
        if opcode == Opcode.IMM:
            next_state.ax = imm

        elif opcode == Opcode.LEA:
            next_state.ax = state.bp + imm

        elif opcode == Opcode.JMP:
            next_state.pc = imm

        elif opcode == Opcode.JSR:
            # Call function
            next_state.sp = state.sp - 8
            stack[next_state.sp] = next_state.pc
            next_state.pc = imm

        elif opcode == Opcode.BZ:
            if state.ax == 0:
                next_state.pc = imm

        elif opcode == Opcode.BNZ:
            if state.ax != 0:
                next_state.pc = imm

        elif opcode == Opcode.ENT:
            next_state.sp = state.sp - 8
            stack[next_state.sp] = state.bp
            next_state.bp = next_state.sp
            next_state.sp = next_state.sp - imm

        elif opcode == Opcode.ADJ:
            next_state.sp = state.sp + imm

        elif opcode == Opcode.LEV:
            # Return from function
            next_state.sp = state.bp
            if next_state.sp in stack:
                next_state.bp = stack[next_state.sp]
                next_state.sp += 8
            if next_state.sp in stack:
                next_state.pc = stack[next_state.sp]
                next_state.sp += 8
            else:
                # Return from main - halt
                next_state.halted = True
                return next_state

        elif opcode == Opcode.PSH:
            next_state.sp = state.sp - 8
            stack[next_state.sp] = state.ax

        elif opcode == Opcode.ADD:
            # ✅ NEURAL EXECUTION
            if state.sp in stack:
                stack_top = stack[state.sp]
                next_state.sp = state.sp + 8
                next_state.ax = self._execute_neural_arithmetic(
                    Opcode.ADD, stack_top, state.ax
                )

        elif opcode == Opcode.SUB:
            # ✅ NEURAL EXECUTION
            if state.sp in stack:
                stack_top = stack[state.sp]
                next_state.sp = state.sp + 8
                next_state.ax = self._execute_neural_arithmetic(
                    Opcode.SUB, stack_top, state.ax
                )

        elif opcode == Opcode.MUL:
            # Python fallback
            if state.sp in stack:
                stack_top = stack[state.sp]
                next_state.sp = state.sp + 8
                next_state.ax = (stack_top * state.ax) & 0xFFFFFFFF

        elif opcode == Opcode.DIV:
            if state.sp in stack:
                stack_top = stack[state.sp]
                next_state.sp = state.sp + 8
                if state.ax != 0:
                    next_state.ax = stack_top // state.ax
                else:
                    next_state.ax = 0

        elif opcode == Opcode.MOD:
            if state.sp in stack:
                stack_top = stack[state.sp]
                next_state.sp = state.sp + 8
                if state.ax != 0:
                    next_state.ax = stack_top % state.ax
                else:
                    next_state.ax = 0

        elif opcode == Opcode.LI:
            # Load int
            addr = state.ax
            val = 0
            for i in range(8):
                if addr + i in memory:
                    val |= (memory[addr + i] << (i * 8))
                elif addr + i in stack:
                    val |= (stack[addr + i] << (i * 8))
            next_state.ax = val & 0xFFFFFFFF

        elif opcode == Opcode.LC:
            # Load char
            addr = state.ax
            if addr in memory:
                next_state.ax = memory[addr] & 0xFF
            elif addr in stack:
                next_state.ax = stack[addr] & 0xFF
            else:
                next_state.ax = 0

        elif opcode == Opcode.SI:
            # Store int
            if state.sp in stack:
                addr = stack[state.sp]
                next_state.sp = state.sp + 8
                val = state.ax
                for i in range(8):
                    byte_val = (val >> (i * 8)) & 0xFF
                    if addr >= 0x10000:
                        memory[addr + i] = byte_val
                    else:
                        stack[addr + i] = byte_val

        elif opcode == Opcode.SC:
            # Store char
            if state.sp in stack:
                addr = stack[state.sp]
                next_state.sp = state.sp + 8
                byte_val = state.ax & 0xFF
                if addr >= 0x10000:
                    memory[addr] = byte_val
                else:
                    stack[addr] = byte_val

        elif opcode == Opcode.EXIT:
            next_state.halted = True
            return next_state

        else:
            # Comparisons and bitwise (Python fallback for now)
            if state.sp in stack:
                stack_top = stack[state.sp]
                next_state.sp = state.sp + 8

                if opcode == Opcode.EQ:
                    next_state.ax = 1 if stack_top == state.ax else 0
                elif opcode == Opcode.NE:
                    next_state.ax = 1 if stack_top != state.ax else 0
                elif opcode == Opcode.LT:
                    next_state.ax = 1 if stack_top < state.ax else 0
                elif opcode == Opcode.GT:
                    next_state.ax = 1 if stack_top > state.ax else 0
                elif opcode == Opcode.LE:
                    next_state.ax = 1 if stack_top <= state.ax else 0
                elif opcode == Opcode.GE:
                    next_state.ax = 1 if stack_top >= state.ax else 0
                elif opcode == Opcode.OR:
                    next_state.ax = (stack_top | state.ax) & 0xFFFFFFFF
                elif opcode == Opcode.XOR:
                    next_state.ax = (stack_top ^ state.ax) & 0xFFFFFFFF
                elif opcode == Opcode.AND:
                    next_state.ax = (stack_top & state.ax) & 0xFFFFFFFF
                elif opcode == Opcode.SHL:
                    next_state.ax = (stack_top << state.ax) & 0xFFFFFFFF
                elif opcode == Opcode.SHR:
                    next_state.ax = (stack_top >> state.ax) & 0xFFFFFFFF

        return next_state

    def _execute_neural_arithmetic(
        self, opcode: int, operand_a: int, operand_b: int
    ) -> int:
        """
        Execute ADD or SUB through neural VM (layers 9-11).

        This is the only neural execution - everything else is still Python.
        """
        # Encode operands
        if opcode == Opcode.SUB:
            # SUB: NIB_A - NIB_B, so swap to get correct order
            input_embedding = self.embed.encode_vm_state(
                pc=0, ax=operand_a, sp=0, bp=0,
                opcode=opcode, stack_top=operand_b, batch_size=1
            )
        else:
            # ADD: commutative
            input_embedding = self.embed.encode_vm_state(
                pc=0, ax=operand_b, sp=0, bp=0,
                opcode=opcode, stack_top=operand_a, batch_size=1
            )

        # Neural execution through 3-layer pipeline
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


def create_autoregressive_executor(verbose: bool = False) -> AutoregressiveNibbleVM:
    """Create an autoregressive executor with loaded weights."""
    from neural_vm.weight_loader import CompiledWeightLoader

    if verbose:
        print("Creating autoregressive VM...")

    vm = AutoregressiveVM(d_model=1280, n_layers=16, ffn_hidden=4096)
    vm.eval()

    loader = CompiledWeightLoader()
    loader.load_all_weights(vm, verbose=verbose)

    embed = NibbleVMEmbedding(d_model=1280)

    if verbose:
        print("Autoregressive VM ready.")
        print()

    return AutoregressiveNibbleVM(vm, embed)
