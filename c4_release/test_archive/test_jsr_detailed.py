#!/usr/bin/env python3
"""Test JSR with detailed PC tracking."""

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode
from neural_vm.vm_step import Token
import sys

print("=" * 80)
print("JSR DETAILED TEST")
print("=" * 80)

bytecode = [
    Opcode.JSR | (25 << 8),  # 0: Call function at byte 25
    Opcode.EXIT,              # 1: Exit
    Opcode.NOP,               # 2: Padding
    Opcode.NOP,               # 3: Padding
    Opcode.NOP,               # 4: Padding
    Opcode.IMM | (42 << 8),   # 5: Function - load 42
    Opcode.EXIT,              # 6: Function - exit
]

print("\nBytecode map:")
print("  Byte 0 (instr 0): JSR 25")
print("  Byte 5 (instr 1): EXIT")
print("  Byte 10 (instr 2): NOP")
print("  Byte 15 (instr 3): NOP")
print("  Byte 20 (instr 4): NOP")
print("  Byte 25 (instr 5): IMM 42  <- JSR target")
print("  Byte 30 (instr 6): EXIT")

print("\nCreating runner...")
runner = AutoregressiveVMRunner()

# Extract PC from context
def extract_pc_from_tokens(tokens):
    """Extract PC value from last step in token sequence."""
    if len(tokens) < 35:
        return None

    # PC is tokens 0-4 in each 35-token step
    # Token 0: PC marker (257)
    # Tokens 1-4: PC bytes (little endian)
    last_step = tokens[-35:]
    if last_step[0] != Token.REG_PC:
        return None

    pc_bytes = last_step[1:5]
    pc = sum((b & 0xFF) << (i * 8) for i, b in enumerate(pc_bytes))
    return pc

# Extract AX from context
def extract_ax_from_tokens(tokens):
    """Extract AX value from last step."""
    if len(tokens) < 35:
        return None

    last_step = tokens[-35:]
    if len(last_step) < 10:
        return None

    # AX is tokens 5-9 in each step
    # Token 5: AX marker (258)
    # Tokens 6-9: AX bytes
    if last_step[5] != Token.REG_AX:
        return None

    ax_bytes = last_step[6:10]
    ax = sum((b & 0xFF) << (i * 8) for i, b in enumerate(ax_bytes))
    return ax

# Track execution
tokens = []
step_count = [0]
original_generate = runner.model.generate_next

def tracking_generate(context, **kwargs):
    step_count[0] += 1
    result = original_generate(context, **kwargs)
    tokens.append(result)

    # After each complete step, show PC and AX
    if step_count[0] % 35 == 0:
        step_num = step_count[0] // 35
        pc = extract_pc_from_tokens(tokens)
        ax = extract_ax_from_tokens(tokens)

        # Decode instruction at PC
        if pc is not None and pc // 5 < len(bytecode):
            instr_idx = pc // 5
            instr = bytecode[instr_idx]
            opcode = instr & 0xFF
            imm = (instr >> 8) & 0xFFFFFF

            op_names = {
                Opcode.JSR: "JSR",
                Opcode.EXIT: "EXIT",
                Opcode.NOP: "NOP",
                Opcode.IMM: "IMM",
            }
            op_name = op_names.get(opcode, f"OP{opcode}")

            if imm:
                instr_str = f"{op_name} {imm}"
            else:
                instr_str = op_name

            print(f"Step {step_num}: PC=0x{pc:08x} (byte {pc}, instr {instr_idx}={instr_str}), AX=0x{ax:08x} ({ax})")
        else:
            print(f"Step {step_num}: PC=0x{pc:08x if pc else 0:08x}, AX=0x{ax:08x if ax else 0:08x}")

    if step_count[0] >= 150:
        raise RuntimeError("Max tokens")

    return result

runner.model.generate_next = tracking_generate

print("\nRunning...\n")

try:
    result = runner.run(bytecode, [], max_steps=10)

    if isinstance(result, tuple):
        _, exit_code = result
    else:
        exit_code = result

    print(f"\n{'=' * 80}")
    print(f"Final result: exit code = {exit_code}")
    print(f"Total steps: {step_count[0] // 35}")

    if exit_code == 42:
        print("\n✅ SUCCESS! JSR worked!")
    else:
        print(f"\n❌ FAILED - Expected 42, got {exit_code}")
        print("\nAnalysis:")
        print("  - If JSR worked, should have jumped from byte 0 to byte 25 (instr 5)")
        print("  - Then IMM 42 at instr 5 should set AX=42")
        print("  - Then EXIT at instr 6 with AX=42")

    sys.exit(0 if exit_code == 42 else 1)

except RuntimeError:
    print(f"\n❌ Timeout after {step_count[0] // 35} steps")
    sys.exit(1)

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
