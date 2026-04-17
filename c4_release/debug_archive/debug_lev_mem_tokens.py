#!/usr/bin/env python3
"""Debug LEV - inspect MEM tokens to see if JSR/ENT are writing to memory correctly."""

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token
from src.compiler import compile_c
from pathlib import Path
import tempfile
import torch

# Simple function that triggers JSR, ENT, and LEV
code = """
int helper(int x) {
    int local;
    local = x * 2;
    return local;
}

int main() {
    return helper(21);
}
"""

print("=" * 70)
print("LEV Memory Debug - Inspecting MEM Tokens")
print("=" * 70)

# Compile
with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
    f.write(code)
    f.flush()
    temp_path = f.name

bytecode, data = compile_c(Path(temp_path).read_text())
Path(temp_path).unlink()

print(f"Bytecode: {len(bytecode)} instructions")

# Create runner
runner = AutoregressiveVMRunner()

# Monkey-patch the step function to inspect tokens
original_step = runner._step

step_count = 0
def debug_step(context):
    global step_count
    step_count += 1

    # Call original step
    result = original_step(context)

    # After step, inspect tokens for MEM sections
    if hasattr(runner, '_context') and runner._context is not None:
        ctx = runner._context
        seq_len = ctx.size(1)

        # Look for MEM tokens
        mem_positions = []
        for i in range(seq_len):
            token_id = int(torch.argmax(ctx[0, i, :Token.VOCAB_SIZE]).item())
            if token_id == Token.MEM:
                mem_positions.append(i)

        # On steps with JSR, ENT, or LEV, show MEM tokens
        exec_pc = runner._exec_pc()
        exec_idx = exec_pc // 5
        if 0 <= exec_idx < len(bytecode):
            opcode = bytecode[exec_idx] & 0xFF
            opcode_names = {1: 'IMM', 2: 'LEA', 15: 'JSR', 16: 'ENT', 17: 'LEV'}

            if opcode in [15, 16, 17]:  # JSR, ENT, LEV
                op_name = opcode_names.get(opcode, f'OP{opcode}')
                print(f"\n[STEP {step_count}] Executing {op_name} at PC=0x{exec_pc:08x}")
                print(f"  Found {len(mem_positions)} MEM tokens in context")

                # Show last 3 MEM tokens
                for pos in mem_positions[-3:]:
                    # Extract address bytes (positions 1-4 after MEM)
                    if pos + 8 < seq_len:
                        addr_bytes = []
                        val_bytes = []
                        for j in range(4):
                            addr_token = int(torch.argmax(ctx[0, pos + 1 + j, :Token.VOCAB_SIZE]).item())
                            val_token = int(torch.argmax(ctx[0, pos + 5 + j, :Token.VOCAB_SIZE]).item())
                            addr_bytes.append(addr_token)
                            val_bytes.append(val_token)

                        # Reconstruct address and value
                        addr = (addr_bytes[0] | (addr_bytes[1] << 8) |
                               (addr_bytes[2] << 16) | (addr_bytes[3] << 24))
                        val = (val_bytes[0] | (val_bytes[1] << 8) |
                              (val_bytes[2] << 16) | (val_bytes[3] << 24))

                        print(f"    MEM[0x{addr:08x}] = 0x{val:08x} (pos {pos})")

                # Stop after first LEV
                if opcode == 17:
                    print(f"\n{'=' * 70}")
                    print(f"Stopping after first LEV to analyze MEM tokens")
                    print(f"{'=' * 70}")
                    return None  # Signal to stop

    return result

runner._step = debug_step

print("\nRunning program with MEM token inspection...\n")
try:
    result = runner.run(bytecode, data, [], max_steps=50)
    print(f"\nProgram result: {result}")
except:
    print(f"\nStopped after LEV for analysis")
