#!/usr/bin/env python3
"""
Full Opcode Table for Neural VM.

Shows all C4 opcodes with:
- Implementation status (Implemented/Partial/Not Implemented)
- Number of layers used
- Number of non-zero parameters
- Description
"""

import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .embedding import E, Opcode, IOToolCallType


@dataclass
class OpcodeInfo:
    """Information about a single opcode."""
    number: int
    name: str
    description: str
    status: str  # 'Implemented', 'Partial', 'Not Implemented'
    num_layers: int
    num_ffns: int
    num_attention: int
    nonzero_params: int
    notes: str


# C4 Opcode definitions with descriptions
OPCODE_DEFINITIONS = {
    # Stack/Address operations (0-8)
    0: ("LEA", "Load effective address: AX = BP + imm", "Not Implemented", "Requires stack frame"),
    1: ("IMM", "Load immediate: AX = imm", "Partial", "Handled by embedding"),
    2: ("JMP", "Unconditional jump: PC = imm", "Implemented", ""),
    3: ("JSR", "Jump subroutine: push PC, PC = imm", "Implemented", ""),
    4: ("BZ", "Branch if zero: if AX == 0, PC = imm", "Implemented", ""),
    5: ("BNZ", "Branch if not zero: if AX != 0, PC = imm", "Implemented", ""),
    6: ("ENT", "Enter function: push BP, BP = SP, SP -= imm", "Not Implemented", "Requires stack frame"),
    7: ("ADJ", "Adjust stack: SP += imm", "Not Implemented", "Requires stack"),
    8: ("LEV", "Leave function: SP = BP, BP = pop, PC = pop", "Not Implemented", "Requires stack frame"),

    # Memory operations (9-13)
    9: ("LI", "Load int: AX = *AX (load from address)", "Implemented", "Via KV memory"),
    10: ("LC", "Load char: AX = *(char*)AX", "Implemented", ""),
    11: ("SI", "Store int: *pop = AX", "Implemented", ""),
    12: ("SC", "Store char: *(char*)pop = AX", "Implemented", ""),
    13: ("PSH", "Push: SP -= 8, *SP = AX", "Implemented", ""),

    # Bitwise operations (14-16)
    14: ("OR", "AX = pop | AX", "Implemented", "Bit extraction + OR"),
    15: ("XOR", "AX = pop ^ AX", "Implemented", "Bit extraction + XOR"),
    16: ("AND", "AX = pop & AX", "Implemented", "Bit extraction + AND"),

    # Comparison operations (17-22)
    17: ("EQ", "AX = (pop == AX)", "Implemented", "Subtraction + zero check"),
    18: ("NE", "AX = (pop != AX)", "Implemented", "Subtraction + non-zero check"),
    19: ("LT", "AX = (pop < AX)", "Implemented", "Subtraction + borrow check"),
    20: ("GT", "AX = (pop > AX)", "Implemented", "Swap + LT"),
    21: ("LE", "AX = (pop <= AX)", "Implemented", "NOT GT"),
    22: ("GE", "AX = (pop >= AX)", "Implemented", "NOT LT"),

    # Shift operations (23-24)
    23: ("SHL", "AX = pop << AX", "Implemented", "Position-based attention"),
    24: ("SHR", "AX = pop >> AX", "Implemented", "Position-based attention"),

    # Arithmetic operations (25-29)
    25: ("ADD", "AX = pop + AX", "Implemented", "Carry propagation"),
    26: ("SUB", "AX = pop - AX", "Implemented", "Borrow propagation"),
    27: ("MUL", "AX = pop * AX", "Implemented", "Partial products + carry"),
    28: ("DIV", "AX = pop / AX", "Implemented", "Iterative subtraction"),
    29: ("MOD", "AX = pop % AX", "Implemented", "Division remainder"),

    # System calls (30-33)
    30: ("OPEN", "open(filename, mode)", "Partial", "Tool-use only"),
    31: ("READ", "read(fd, buf, n)", "Partial", "Tool-use only"),
    32: ("CLOS", "close(fd)", "Partial", "Tool-use only"),
    33: ("PRTF", "printf(fmt, ...)", "Implemented", "Native + tool-use"),

    # Memory subroutines (36-37)
    36: ("MSET", "memset(ptr, val, n)", "Implemented", "Loop-based"),
    37: ("MCMP", "memcmp(a, b, n)", "Implemented", "Loop-based"),

    # Control (38-42)
    38: ("HALT", "exit(code)", "Implemented", "Sets IO_PROGRAM_END"),
    39: ("NOP", "No operation", "Implemented", "Identity"),
    40: ("POP", "Pop stack", "Implemented", ""),
    41: ("BLT", "Branch if less than", "Implemented", ""),
    42: ("BGE", "Branch if greater or equal", "Implemented", ""),

    # I/O operations (64-66)
    64: ("GETCHAR", "getchar()", "Implemented", "Native + streaming + tool-use"),
    65: ("PUTCHAR", "putchar(c)", "Implemented", "Native + streaming + tool-use"),
    66: ("PRINTF2", "printf variant", "Not Implemented", ""),
}


def get_implementation_stats() -> Dict[int, Tuple[int, int, int, int]]:
    """
    Get implementation stats from SparseMoEALU.

    Returns:
        Dict mapping opcode to (num_ffns, num_attn, total_params, nonzero_params)
    """
    try:
        from .sparse_moe_alu import SparseMoEALU
        from .analyze_ops import analyze_sparse_moe_alu

        alu = SparseMoEALU()
        stats = analyze_sparse_moe_alu(alu)

        result = {}
        for op, s in stats.items():
            result[op] = (s.num_ffns, s.num_attention, s.total_params, s.nonzero_params)
        return result
    except Exception as e:
        print(f"Warning: Could not load ALU stats: {e}")
        return {}


def generate_opcode_table(include_stats: bool = True) -> List[OpcodeInfo]:
    """
    Generate complete opcode table with implementation info.

    Args:
        include_stats: Whether to include layer/param stats from ALU

    Returns:
        List of OpcodeInfo for all opcodes
    """
    # Get stats if available
    stats = get_implementation_stats() if include_stats else {}

    opcodes = []
    for op_num, (name, desc, status, notes) in OPCODE_DEFINITIONS.items():
        if op_num in stats:
            num_ffns, num_attn, total_params, nonzero_params = stats[op_num]
        else:
            num_ffns, num_attn, total_params, nonzero_params = 0, 0, 0, 0

        opcodes.append(OpcodeInfo(
            number=op_num,
            name=name,
            description=desc,
            status=status,
            num_layers=num_ffns + num_attn,
            num_ffns=num_ffns,
            num_attention=num_attn,
            nonzero_params=nonzero_params,
            notes=notes
        ))

    return sorted(opcodes, key=lambda x: x.number)


def print_opcode_table(verbose: bool = False):
    """Print formatted opcode table."""
    opcodes = generate_opcode_table()

    print("=" * 100)
    print("NEURAL VM OPCODE TABLE - Full C4 Compatibility")
    print("=" * 100)
    print()

    # Count by status
    implemented = sum(1 for o in opcodes if o.status == 'Implemented')
    partial = sum(1 for o in opcodes if o.status == 'Partial')
    not_impl = sum(1 for o in opcodes if o.status == 'Not Implemented')

    print(f"Implementation Status: {implemented} Implemented, {partial} Partial, {not_impl} Not Implemented")
    print()

    # Header
    print(f"{'#':<4} {'Name':<10} {'Status':<15} {'Layers':<8} {'FFN':<5} {'Attn':<5} {'Params':<10} Description")
    print("-" * 100)

    # Group by category
    categories = [
        ("Stack/Address", range(0, 9)),
        ("Memory", range(9, 14)),
        ("Bitwise", range(14, 17)),
        ("Comparison", range(17, 23)),
        ("Shift", range(23, 25)),
        ("Arithmetic", range(25, 30)),
        ("System Calls", range(30, 34)),
        ("Mem Subroutines", [36, 37]),
        ("Control", range(38, 43)),
        ("I/O", range(64, 67)),
    ]

    for cat_name, op_range in categories:
        print(f"\n{cat_name}:")
        for op in opcodes:
            if op.number in op_range:
                status_marker = {
                    'Implemented': '[OK]',
                    'Partial': '[--]',
                    'Not Implemented': '[  ]'
                }[op.status]

                params_str = f"{op.nonzero_params:,}" if op.nonzero_params > 0 else "-"
                layers_str = str(op.num_layers) if op.num_layers > 0 else "-"
                ffn_str = str(op.num_ffns) if op.num_ffns > 0 else "-"
                attn_str = str(op.num_attention) if op.num_attention > 0 else "-"

                print(f"{op.number:<4} {op.name:<10} {status_marker:<15} {layers_str:<8} {ffn_str:<5} {attn_str:<5} {params_str:<10} {op.description}")

                if verbose and op.notes:
                    print(f"     -> {op.notes}")

    print()
    print("=" * 100)

    # Summary by category
    print("\nSummary by Category:")
    for cat_name, op_range in categories:
        cat_ops = [o for o in opcodes if o.number in op_range]
        impl = sum(1 for o in cat_ops if o.status == 'Implemented')
        total_layers = sum(o.num_layers for o in cat_ops)
        total_params = sum(o.nonzero_params for o in cat_ops)
        print(f"  {cat_name:<20}: {impl}/{len(cat_ops)} implemented, {total_layers} total layers, {total_params:,} non-zero params")

    print()

    # Neural implementation details
    print("Neural Implementation Details:")
    print("  - FFN: SwiGLU feedforward - out = x + W_down @ (silu(W_up @ x + b_up) * (W_gate @ x + b_gate))")
    print("  - Attention: Multi-head attention for cross-position communication (carry propagation)")
    print("  - Weights: Baked at initialization, not learned")
    print("  - Sparsity: ~99% of parameters are zero (operation-specific)")
    print()
    print("I/O Modes:")
    print("  - Native: Direct stdin/stdout (NativeIOHandler)")
    print("  - Streaming: LLM-style with <NEED_INPUT/>, <PROGRAM_END/> markers")
    print("  - Tool-Use: Agentic with TOOL_CALL:type:id:{params} protocol")
    print("  - LLM-Native: Binary position addressing for user input/model output")
    print()
    print("=" * 100)


def generate_markdown_table() -> str:
    """Generate markdown table for documentation."""
    opcodes = generate_opcode_table()

    lines = [
        "# Neural VM Opcode Table",
        "",
        "| # | Name | Status | Layers | Non-Zero Params | Description |",
        "|---|------|--------|--------|-----------------|-------------|",
    ]

    for op in opcodes:
        status_emoji = {
            'Implemented': '✅',
            'Partial': '⚠️',
            'Not Implemented': '❌'
        }[op.status]

        params = f"{op.nonzero_params:,}" if op.nonzero_params > 0 else "-"
        layers = str(op.num_layers) if op.num_layers > 0 else "-"

        lines.append(f"| {op.number} | {op.name} | {status_emoji} {op.status} | {layers} | {params} | {op.description} |")

    return "\n".join(lines)


def main():
    print_opcode_table(verbose=True)


if __name__ == "__main__":
    main()
