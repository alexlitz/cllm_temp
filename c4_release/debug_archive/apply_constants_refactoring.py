#!/usr/bin/env python3
"""
Script to refactor magic numbers to use constants from neural_vm/constants.py

This demonstrates the refactoring but does NOT automatically apply it due to
the other Claude session modifying files. Review and apply manually.
"""

import re
from pathlib import Path

def show_refactoring_examples():
    """Show examples of what the refactoring would change."""

    examples = [
        ("speculative.py", [
            ("Magic number in _le_bytes",
             "return [(v >> (i * 8)) & 0xFF for i in range(4)]",
             "return [(v >> (i * BITS_PER_BYTE)) & BYTE_MASK for i in range(VALUE_BYTES)]"),

            ("PC initialization",
             "self.pc = 2",
             "self.pc = PC_OFFSET"),

            ("PC increment formula",
             "self.pc = self.idx * 8 + 2",
             "self.pc = idx_to_pc(self.idx)"),

            ("JMP PC to idx conversion",
             "self.idx = (imm - 2) // 8",
             "self.idx = pc_to_idx(imm)"),

            ("Stack alignment",
             "self.sp = (self.sp - 8) & 0xFFFFFFFF",
             "self.sp = (self.sp - STACK_ALIGNMENT) & WORD_MASK"),

            ("Token count assertion",
             "assert len(tokens) == 35",
             "assert len(tokens) == TOKENS_PER_STEP"),
        ]),

        ("vm_step.py", [
            ("L3 FFN PC default",
             "ffn.W_down[BD.OUTPUT_LO + 2, unit] = 2.0 / S  # PC=2",
             "ffn.W_down[BD.OUTPUT_LO + PC_OFFSET, unit] = 2.0 / S  # PC=PC_OFFSET"),

            ("Nibble range iteration",
             "for k in range(16):",
             "for k in range(NIBBLE_RANGE):"),

            ("ADDR_KEY nibble extraction",
             "lo = addr & 0xF",
             "lo = addr & NIBBLE_MASK"),

            ("ADDR_KEY shift amount",
             "hi = (addr >> 4) & 0xF",
             "hi = (addr >> NIBBLE_SIZE) & NIBBLE_MASK"),

            ("ADDR_KEY PC offset",
             "addr = i - cs_pos - 1 + 2",
             "addr = i - cs_pos - 1 + PC_OFFSET"),

            ("Opcode threshold",
             "T = 4.0  # opcode threshold",
             "# Opcode threshold defined in constants"),

            ("Anti-leak gate",
             "attn.W_q[base + GATE, BD.CONST] = -5000.0",
             "attn.W_q[base + GATE, BD.CONST] = ANTI_LEAK_GATE"),

            ("Fetch attention strength",
             "L = 20.0",
             "L = FETCH_ATTENTION_L"),
        ]),
    ]

    print("=" * 80)
    print("CONSTANTS REFACTORING EXAMPLES")
    print("=" * 80)
    print()
    print("NOTE: Due to the other Claude session modifying files, this script")
    print("      shows examples rather than applying changes automatically.")
    print()

    for filename, changes in examples:
        print(f"\n{'=' * 80}")
        print(f"FILE: {filename}")
        print(f"{'=' * 80}\n")

        for i, (description, before, after) in enumerate(changes, 1):
            print(f"{i}. {description}")
            print(f"   BEFORE: {before}")
            print(f"   AFTER:  {after}")
            print()

def show_import_statements():
    """Show the required import statements."""
    print("\n" + "=" * 80)
    print("REQUIRED IMPORT STATEMENTS")
    print("=" * 80)
    print()

    print("For speculative.py:")
    print("-" * 80)
    print("""
from .constants import (
    INSTR_WIDTH, PC_OFFSET,
    BITS_PER_BYTE, BYTE_MASK, WORD_MASK,
    STACK_ALIGNMENT, VALUE_BYTES,
    TOKENS_PER_STEP,
    pc_to_idx, idx_to_pc,
)
""")

    print("\nFor vm_step.py:")
    print("-" * 80)
    print("""
from .constants import (
    INSTR_WIDTH, PC_OFFSET,
    NIBBLE_RANGE, NIBBLE_MASK, NIBBLE_SIZE,
    BYTE_MASK, WORD_MASK,
    OPCODE_THRESHOLD, PC_INCREMENT_THRESHOLD, CARRY_THRESHOLD,
    FETCH_ATTENTION_L, ATTENTION_SCALE_L,
    ANTI_LEAK_GATE,
)
""")

def show_benefits():
    """Show the benefits of this refactoring."""
    print("\n" + "=" * 80)
    print("BENEFITS OF USING CONSTANTS")
    print("=" * 80)
    print()

    benefits = [
        ("Self-Documenting Code",
         "STACK_ALIGNMENT is clearer than magic number 8"),

        ("Single Source of Truth",
         "Change PC_OFFSET once, updates everywhere"),

        ("Easy Experimentation",
         "Toggle PC_OFFSET between 0 and 2 to compare conventions"),

        ("Type Safety",
         "Helper functions like idx_to_pc() prevent errors"),

        ("Better Comments",
         "Constants replace repeated explanatory comments"),

        ("Refactoring Safety",
         "Find all usages with IDE 'find references'"),

        ("Validation",
         "Assert formulas in constants.py to catch inconsistencies"),
    ]

    for i, (benefit, explanation) in enumerate(benefits, 1):
        print(f"{i}. {benefit}")
        print(f"   → {explanation}\n")

def main():
    """Main entry point."""
    show_refactoring_examples()
    show_import_statements()
    show_benefits()

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print()
    print("1. Review the constants defined in neural_vm/constants.py")
    print("2. Coordinate with other Claude session to pause file modifications")
    print("3. Add import statements to speculative.py and vm_step.py")
    print("4. Systematically replace magic numbers using examples above")
    print("5. Run tests after each file is updated")
    print("6. Once both files use constants, toggle PC_OFFSET to switch conventions")
    print()

if __name__ == "__main__":
    main()
