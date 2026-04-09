"""Debug why PRTF isn't being executed."""

import sys
import os
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token

c_code = '''
int main() {
    printf("Hello");
    return 42;
}
'''

print("Compiling...")
code, data = compile_c(c_code)
print(f"Instructions: {len(code)}")

# Show what instructions we have
print("\nProgram instructions:")
for i, instr in enumerate(code):
    op = instr & 0xFF
    imm = instr >> 8
    op_names = {1: "IMM", 2: "LEA", 3: "JMP", 4: "JSR", 15: "PSH",
                33: "PRTF", 34: "EXIT", 18: "ENT", 19: "ADJ", 20: "LEV"}
    op_name = op_names.get(op, f"op_{op}")
    print(f"  {i}: {op_name:8s} (0x{op:02x}) imm=0x{imm:08x}")

print(f"\nData section: {len(data)} bytes")
if data:
    print(f"  First 20 bytes: {data[:20]}")

print("\nCreating runner...")
runner = AutoregressiveVMRunner(conversational_io=True)
if torch.cuda.is_available():
    runner.model = runner.model.cuda()

# Track everything
step_count = 0
opcodes_set = []

original_set = runner.model.set_active_opcode
def track_opcode(op):
    opcodes_set.append(op)
    print(f"  Step {step_count}: set_active_opcode({op}) - {op_names.get(op, f'op_{op}') if op else 'None'}")
    original_set(op)

runner.model.set_active_opcode = track_opcode

tokens_generated = []
original_gen = runner.model.generate_next
def track_gen(context):
    token = original_gen(context)
    tokens_generated.append(token)
    return token
runner.model.generate_next = track_gen

print("\nRunning (max 15 steps)...")
output, exit_code = runner.run(code, data, [], max_steps=15)

print(f"\nResults:")
print(f"  Steps executed: {step_count}")
print(f"  Opcodes set: {len(opcodes_set)}")
print(f"  Tokens generated: {len(tokens_generated)}")
print(f"  Exit code: {exit_code}")
print(f"  Output: {repr(output)}")

print(f"\nOpcodes set during execution:")
for i, op in enumerate(opcodes_set[:20]):
    op_name = op_names.get(op, f"op_{op}") if op else "None"
    marker = " <-- PRTF!" if op == 33 else ""
    print(f"  Step {i}: {op_name}{marker}")

# Check if PRTF was ever set
prtf_set = 33 in opcodes_set
print(f"\n{'✅' if prtf_set else '❌'} PRTF (33) was {'SET' if prtf_set else 'NOT SET'}")

# Check token types
step_ends = tokens_generated.count(Token.STEP_END)
thinking_ends = tokens_generated.count(Token.THINKING_END)
print(f"\nToken counts:")
print(f"  STEP_END: {step_ends}")
print(f"  THINKING_END: {thinking_ends}")

if not prtf_set:
    print("\n⚠️  PRTF opcode was never set - program didn't reach printf!")
    print("   Possible reasons:")
    print("   - Program halted early")
    print("   - Infinite loop before printf")
    print("   - Jump/branch issue")
