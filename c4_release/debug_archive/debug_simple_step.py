#!/usr/bin/env python3
"""Debug simple step generation to verify model works."""

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token
from neural_vm.embedding import Opcode
from src.compiler import compile_c

# Super simple program - just IMM + EXIT
code = '''
int main() {
    return 42;
}
'''

print("Compiling...")
bytecode, data = compile_c(code)
print(f"Bytecode: {len(bytecode)} instructions")
print(f"First 5 instructions:")
for i in range(min(5, len(bytecode))):
    instr = bytecode[i]
    opcode = instr & 0xFF
    imm = instr >> 8
    opname = "?"
    for name in dir(Opcode):
        if not name.startswith('_') and getattr(Opcode, name) == opcode:
            opname = name
            break
    print(f"  {i}: {opname} (op={opcode}, imm={imm})")

# Create runner
runner = AutoregressiveVMRunner()
print(f"\nHandlers: {list(runner._func_call_handlers.keys())}")

# Add token tracking
token_count = [0]
original_generate = runner.model.generate_next
def traced_generate(context, *args, **kwargs):
    token = original_generate(context, *args, **kwargs)
    token_count[0] += 1
    if token_count[0] <= 70:  # First 70 tokens (2 steps)
        print(f"[TOKEN {token_count[0]}] = {token}", end="", flush=True)
        if token == Token.STEP_END:
            print(" (STEP_END)", flush=True)
        elif token == Token.HALT:
            print(" (HALT)", flush=True)
        elif token == Token.REG_PC:
            print(" (REG_PC)", flush=True)
        elif token == Token.REG_AX:
            print(" (REG_AX)", flush=True)
        elif token == Token.REG_SP:
            print(" (REG_SP)", flush=True)
        elif token == Token.REG_BP:
            print(" (REG_BP)", flush=True)
        elif token == Token.STACK0:
            print(" (STACK0)", flush=True)
        elif token == Token.MEM:
            print(" (MEM)", flush=True)
        else:
            print(flush=True)
    return token
runner.model.generate_next = traced_generate

print("\nRunning (max 3 steps)...")
try:
    result = runner.run(bytecode, data, max_steps=3)
    print(f"\nResult: {result}")
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
