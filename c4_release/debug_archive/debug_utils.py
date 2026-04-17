"""
Debug Utilities for C4 Neural VM

Provides common debugging functions for analyzing VM execution, inspecting model
internals, and tracing program flow.

Usage:
    from debug_utils import quick_test, inspect_tokens, trace_execution

    # Quick test a C program
    result, tokens = quick_test('int main() { return 42; }')

    # Inspect token output
    inspect_tokens(tokens[0])  # First step

    # Trace full execution
    trace_execution('int main() { int x = 10; return x * 2; }')
"""

import sys
import torch
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.compiler import compile_c
from src.speculator import FastLogicalVM
from neural_vm.vm_step import AutoregressiveVM, Token
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.speculative import DraftVM
from neural_vm.embedding import Opcode
from neural_vm.constants import STACK_INIT
from neural_vm.token_layout import (
    POS_PC_MARKER, POS_PC_BYTE0,
    POS_AX_MARKER, POS_AX_BYTE0,
    POS_SP_MARKER, POS_SP_BYTE0,
    POS_BP_MARKER, POS_BP_BYTE0,
    POS_STACK0_MARKER, POS_STACK0_BYTE0,
    bytes_to_int32, get_pc_bytes, get_ax_bytes, get_sp_bytes, get_bp_bytes
)


# =============================================================================
# Quick Testing Functions
# =============================================================================

def quick_test(source: str, max_steps: int = 1000, mode: str = 'draft') -> Tuple[int, List[List[int]]]:
    """
    Compile and run a C program quickly.

    Args:
        source: C source code
        max_steps: Maximum VM steps
        mode: 'draft' (fast) or 'neural' (transformer-based)

    Returns:
        (exit_code, token_sequence) tuple

    Example:
        result, tokens = quick_test('int main() { return 42; }')
        print(f'Result: {result}')
        print(f'Steps: {len(tokens)}')
    """
    bytecode, data = compile_c(source)

    if mode == 'draft':
        vm = DraftVM(bytecode)
        if data:
            vm.load_data(data)
        tokens = []
        while not vm.halted and len(tokens) < max_steps:
            vm.step()
            step_tokens = vm.draft_tokens()
            tokens.append(step_tokens)
        return vm.ax, tokens

    elif mode == 'neural':
        runner = AutoregressiveVMRunner()
        result = runner.run(bytecode, data, max_steps=max_steps)
        # For neural mode, we don't have easy access to token sequence
        return result, []

    else:
        raise ValueError(f"Unknown mode: {mode}")


def quick_compare(source: str, max_steps: int = 100) -> Dict[str, Any]:
    """
    Run a program in both draft and neural modes and compare results.

    Args:
        source: C source code
        max_steps: Maximum VM steps

    Returns:
        Dictionary with comparison results

    Example:
        comparison = quick_compare('int main() { return 42; }')
        print(f"Match: {comparison['match']}")
        print(f"Draft result: {comparison['draft_result']}")
        print(f"Neural result: {comparison['neural_result']}")
    """
    bytecode, data = compile_c(source)

    # Run draft VM
    draft_vm = DraftVM(bytecode)
    if data:
        draft_vm.load_data(data)
    draft_tokens = []
    while not draft_vm.halted and len(draft_tokens) < max_steps:
        draft_vm.step()
        step_tokens = draft_vm.draft_tokens()
        draft_tokens.append(step_tokens)
    draft_result = draft_vm.ax

    # Run neural VM
    runner = AutoregressiveVMRunner()
    neural_result = runner.run(bytecode, data, max_steps=max_steps)

    return {
        'match': draft_result == neural_result,
        'draft_result': draft_result,
        'neural_result': neural_result,
        'draft_steps': len(draft_tokens),
        'source': source
    }


# =============================================================================
# Token Inspection
# =============================================================================

def inspect_tokens(tokens: List[int], show_bytes: bool = False) -> Dict[str, Any]:
    """
    Parse and display VM step tokens.

    Args:
        tokens: 35-token output from a VM step
        show_bytes: If True, show raw byte values

    Returns:
        Dictionary with parsed values

    Example:
        result, all_tokens = quick_test('int main() { return 42; }')
        info = inspect_tokens(all_tokens[0])
        print(f"PC: 0x{info['pc']:08x}")
        print(f"AX: {info['ax']}")
    """
    if len(tokens) < 35:
        return {'error': f'Expected 35 tokens, got {len(tokens)}'}

    pc_bytes = get_pc_bytes(tokens)
    ax_bytes = get_ax_bytes(tokens)
    sp_bytes = get_sp_bytes(tokens)
    bp_bytes = get_bp_bytes(tokens)

    pc = bytes_to_int32(pc_bytes)
    ax = bytes_to_int32(ax_bytes)
    sp = bytes_to_int32(sp_bytes)
    bp = bytes_to_int32(bp_bytes)

    # STACK0 bytes
    stack0_bytes = [tokens[POS_STACK0_BYTE0 + i] for i in range(4)]
    stack0 = bytes_to_int32(stack0_bytes)

    # Terminator
    terminator = tokens[34]
    terminator_name = f'TOKEN_{terminator}'
    for name in dir(Token):
        if not name.startswith('_') and getattr(Token, name) == terminator:
            terminator_name = name
            break

    result = {
        'pc': pc,
        'ax': ax,
        'sp': sp,
        'bp': bp,
        'stack0': stack0,
        'terminator': terminator_name,
        'halted': terminator == Token.HALT,
    }

    if show_bytes:
        result['pc_bytes'] = pc_bytes
        result['ax_bytes'] = ax_bytes
        result['sp_bytes'] = sp_bytes
        result['bp_bytes'] = bp_bytes
        result['stack0_bytes'] = stack0_bytes

    return result


def print_tokens(tokens: List[int]):
    """
    Pretty-print VM step tokens.

    Args:
        tokens: 35-token output from a VM step

    Example:
        result, all_tokens = quick_test('int main() { return 42; }')
        for i, step_tokens in enumerate(all_tokens):
            print(f'\\nStep {i}:')
            print_tokens(step_tokens)
    """
    info = inspect_tokens(tokens)

    if 'error' in info:
        print(f"ERROR: {info['error']}")
        return

    print(f"  PC:     0x{info['pc']:08x} ({info['pc']:10d})")
    print(f"  AX:     0x{info['ax']:08x} ({info['ax']:10d})")
    print(f"  SP:     0x{info['sp']:08x} ({info['sp']:10d})")
    print(f"  BP:     0x{info['bp']:08x} ({info['bp']:10d})")
    print(f"  STACK0: 0x{info['stack0']:08x} ({info['stack0']:10d})")
    print(f"  Status: {info['terminator']}")


# =============================================================================
# Execution Tracing
# =============================================================================

def trace_execution(source: str, max_steps: int = 100, show_all: bool = False) -> List[Dict[str, Any]]:
    """
    Trace VM execution step-by-step.

    Args:
        source: C source code
        max_steps: Maximum VM steps to trace
        show_all: If True, print all steps; if False, just return data

    Returns:
        List of dictionaries, one per step, with parsed token info

    Example:
        trace = trace_execution('int main() { int x = 10; return x; }', show_all=True)
    """
    bytecode, data = compile_c(source)
    vm = DraftVM(bytecode)
    if data:
        vm.load_data(data)

    trace = []
    step = 0

    if show_all:
        print(f"=== Tracing Program ===")
        print(f"Bytecode: {len(bytecode)} instructions")
        print(f"Data: {len(data) if data else 0} bytes")
        print()

    while not vm.halted and step < max_steps:
        vm.step()
        tokens = vm.draft_tokens()
        info = inspect_tokens(tokens)
        info['step'] = step
        trace.append(info)

        if show_all:
            print(f"Step {step}:")
            print_tokens(tokens)
            print()

        step += 1

    if show_all:
        print(f"=== Execution Complete ===")
        print(f"Total steps: {step}")
        print(f"Final AX: {vm.ax}")
        print(f"Halted: {vm.halted}")

    return trace


# =============================================================================
# Bytecode Inspection
# =============================================================================

def disassemble(bytecode, max_instructions: int = 50) -> List[Tuple[int, str, int]]:
    """
    Disassemble bytecode into readable instructions.

    Args:
        bytecode: Raw bytecode (list of ints or bytes)
        max_instructions: Maximum number of instructions to disassemble

    Returns:
        List of (address, opcode_name, immediate) tuples

    Example:
        bytecode, data = compile_c('int main() { return 42; }')
        instructions = disassemble(bytecode)
        for addr, op, imm in instructions:
            print(f"{addr:04x}: {op:8s} {imm}")
    """
    instructions = []

    # bytecode is a list of packed instructions (each is a single int)
    # Each instruction is: opcode (1 byte) + immediate (4 bytes) packed into an int
    for idx, packed_instr in enumerate(bytecode):
        if idx >= max_instructions:
            break

        # Extract opcode (lowest byte)
        opcode = packed_instr & 0xFF

        # Extract immediate (next 4 bytes)
        immediate = (packed_instr >> 8) & 0xFFFFFFFF

        # Get opcode name from Opcode class attributes
        opcode_name = f"OP_{opcode:02x}"
        for name in dir(Opcode):
            if not name.startswith('_') and getattr(Opcode, name) == opcode:
                opcode_name = name
                break

        # Calculate address (idx * 8 for instruction width)
        addr = idx * 8

        instructions.append((addr, opcode_name, immediate))

    return instructions


def print_disassembly(bytecode: bytes, max_instructions: int = 50):
    """
    Pretty-print disassembled bytecode.

    Args:
        bytecode: Raw bytecode bytes
        max_instructions: Maximum number of instructions to show

    Example:
        bytecode, data = compile_c('int main() { return 42; }')
        print_disassembly(bytecode)
    """
    instructions = disassemble(bytecode, max_instructions)

    print("Address  Opcode    Immediate")
    print("-------  --------  ----------")
    for addr, op, imm in instructions:
        if imm < 256:
            imm_str = f"{imm}"
        else:
            imm_str = f"0x{imm:08x}"
        print(f"{addr:04x}     {op:8s}  {imm_str:>10s}")


# =============================================================================
# Register/Memory Inspection
# =============================================================================

def get_register_state(vm: DraftVM) -> Dict[str, int]:
    """
    Get current register state from a VM.

    Args:
        vm: DraftVM instance

    Returns:
        Dictionary with register values
    """
    return {
        'pc': vm.pc,
        'ax': vm.ax,
        'sp': vm.sp,
        'bp': vm.bp,
    }


def print_register_state(vm: DraftVM):
    """
    Pretty-print VM register state.

    Args:
        vm: DraftVM instance
    """
    state = get_register_state(vm)
    print("Registers:")
    print(f"  PC = 0x{state['pc']:08x} ({state['pc']:10d})")
    print(f"  AX = 0x{state['ax']:08x} ({state['ax']:10d})")
    print(f"  SP = 0x{state['sp']:08x} ({state['sp']:10d})")
    print(f"  BP = 0x{state['bp']:08x} ({state['bp']:10d})")


# =============================================================================
# Main Entry Point (for command-line usage)
# =============================================================================

def main():
    """
    Command-line interface for debug utilities.

    Usage:
        python debug_utils.py "int main() { return 42; }"
    """
    import sys

    if len(sys.argv) < 2:
        print("Usage: python debug_utils.py '<C source code>'")
        print("\nExample:")
        print("  python debug_utils.py 'int main() { return 42; }'")
        return

    source = sys.argv[1]

    print("=== Quick Test ===")
    result, tokens = quick_test(source)
    print(f"Result: {result}")
    print(f"Steps: {len(tokens)}\n")

    print("=== Bytecode ===")
    bytecode, data = compile_c(source)
    print_disassembly(bytecode)
    print()

    print("=== Execution Trace ===")
    trace_execution(source, show_all=True)


if __name__ == '__main__':
    main()
