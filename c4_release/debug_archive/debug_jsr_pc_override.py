#!/usr/bin/env python3
"""Debug JSR handler PC override mechanism."""

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token
from neural_vm.embedding import Opcode
from neural_vm.constants import INSTR_WIDTH, PC_OFFSET
from src.compiler import compile_c

# Simple function call test
code = '''
int helper(int x) {
    return x * 2;
}

int main() {
    return helper(21);
}
'''

print("Compiling...")
bytecode, data = compile_c(code)
print(f"Bytecode: {len(bytecode)} instructions")
print(f"First instruction: 0x{bytecode[0]:08x}")
print(f"  Opcode: {bytecode[0] & 0xFF}")
print(f"  Immediate: {bytecode[0] >> 8} (PC target)")

# Create runner with JSR handler
runner = AutoregressiveVMRunner()
print(f"\nHandlers: {list(runner._func_call_handlers.keys())}")

# Build initial context
ctx = runner._build_context(bytecode, data, [])
print(f"\nInitial context length: {len(ctx)}")

# Execute first step manually
print("\n--- First Step (should be JSR) ---")
runner._bytecode = bytecode

# Check exec_pc
exec_pc = runner._exec_pc()
print(f"exec_pc: {exec_pc}")
exec_idx = exec_pc // INSTR_WIDTH
print(f"exec_idx: {exec_idx}")

# Get the instruction
instr = bytecode[exec_idx]
opcode = instr & 0xFF
target = instr >> 8
print(f"Instruction at index {exec_idx}: 0x{instr:08x}")
print(f"  Opcode: {opcode}")
print(f"  Target PC: {target}")

# Generate step - need to call generate_next in a loop until STEP_END
print("\nGenerating step tokens...")
runner.model.set_active_opcode(opcode)
prev_len = len(ctx)

# Generate 35 tokens (TOKENS_PER_STEP) or until STEP_END
for i in range(35):
    next_token = runner.model.generate_next(ctx)
    ctx.append(next_token)
    print(f"  Token {i}: {next_token}", end="")
    if next_token == Token.STEP_END:
        print(" (STEP_END)")
        break
    elif next_token == Token.HALT:
        print(" (HALT)")
        break
    else:
        print()

print(f"Generated {len(ctx) - prev_len} tokens total")

# Extract what model output for PC
model_pc = runner._extract_register(ctx, Token.REG_PC)
print(f"\nModel output PC (before handler): {model_pc}")

# Now the handler should be invoked
# Let's manually call it to see what happens
print("\n--- JSR Handler ---")

# First, find where REG_PC is in context
scan_back = Token.STEP_TOKENS + 5
pc_positions = []
for i in range(len(ctx) - 1, max(0, len(ctx) - scan_back), -1):
    if ctx[i] == Token.REG_PC:
        pc_positions.append(i)
        print(f"Found REG_PC at position {i}")
        # Show bytes
        if i + 4 < len(ctx):
            bytes_before = [ctx[i+1+j] for j in range(4)]
            val = sum(b << (j*8) for j, b in enumerate(bytes_before))
            print(f"  Bytes: {bytes_before} = {val}")

# Call handler
print("\nCalling _handler_jsr...")
handler = runner._func_call_handlers.get(Opcode.JSR)
if handler:
    handler(ctx, None)  # output not used by JSR handler
    print("Handler completed")
else:
    print("No JSR handler!")

# Extract PC after handler
pc_after = runner._extract_register(ctx, Token.REG_PC)
print(f"\nPC after handler: {pc_after}")

# Check if override worked
if pc_after == target:
    print(f"  SUCCESS: PC was set to target {target}")
else:
    print(f"  FAILED: PC is {pc_after}, expected {target}")

# Show bytes again
for i in pc_positions:
    if i + 4 < len(ctx):
        bytes_after = [ctx[i+1+j] for j in range(4)]
        val = sum(b << (j*8) for j, b in enumerate(bytes_after))
        print(f"REG_PC at {i} bytes: {bytes_after} = {val}")
