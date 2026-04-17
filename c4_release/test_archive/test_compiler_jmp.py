#!/usr/bin/env python3
"""Verify C compiler produces correct JMP PC values."""

import sys
sys.path.insert(0, '.')

from src.compiler import compile_c
from neural_vm.embedding import Opcode
from neural_vm.constants import PC_OFFSET, INSTR_WIDTH

def decode_instruction(word):
    """Decode a bytecode word into opcode + immediate."""
    opcode = word & 0xFF
    immediate = (word >> 8) & 0xFFFFFF
    return opcode, immediate

opcode_names = {
    1: "IMM", 2: "LEA", 3: "JMP", 4: "JSR", 5: "BZ", 6: "BNZ",
    7: "ENT", 8: "ADJ", 9: "LEV", 10: "LI", 11: "LC",
    12: "SI", 13: "SC", 14: "PSH", 37: "NOP", 38: "EXIT"
}

def test_compiler_jmp():
    """Test that compiler produces correct JMP targets."""
    print("=== Compiler JMP Encoding Test ===")
    print(f"PC_OFFSET = {PC_OFFSET}, INSTR_WIDTH = {INSTR_WIDTH}")
    print()

    # Use if-else which generates BZ/JMP
    source = """
    int main() {
        int x;
        x = 1;
        if (x) {
            return 42;
        } else {
            return 99;
        }
    }
    """

    bytecode, data = compile_c(source)

    print("Bytecode instructions:")
    print("-" * 50)
    for idx, word in enumerate(bytecode):
        opcode, imm = decode_instruction(word)
        pc = idx * INSTR_WIDTH + PC_OFFSET
        name = opcode_names.get(opcode, f"OP_{opcode}")
        print(f"idx={idx} PC={pc:3d} (0x{pc:02x}): {name:5s} {imm} (0x{imm:04x})")

    # Find JMP instruction
    print("\n=== JMP Analysis ===")
    for idx, word in enumerate(bytecode):
        opcode, imm = decode_instruction(word)
        if opcode == Opcode.JMP:
            jmp_pc = idx * INSTR_WIDTH + PC_OFFSET
            print(f"JMP found at idx={idx}, PC={jmp_pc}")
            print(f"JMP target = {imm} (0x{imm:02x})")

            # Verify target is a valid PC (divisible by 8 with offset 2)
            target_idx = (imm - PC_OFFSET) // INSTR_WIDTH if imm >= PC_OFFSET else -1
            if (imm - PC_OFFSET) % INSTR_WIDTH == 0 and target_idx >= 0:
                print(f"Target instruction index: {target_idx}")
                print("Target PC is VALID (properly encoded with PC_OFFSET)")
            else:
                print(f"WARNING: Target {imm} is NOT a valid PC!")
                print(f"  Expected: idx * {INSTR_WIDTH} + {PC_OFFSET}")

if __name__ == "__main__":
    test_compiler_jmp()
