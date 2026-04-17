#!/usr/bin/env python3
"""Debug JSR handler in full run() flow."""

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

# Create runner with JSR handler
runner = AutoregressiveVMRunner()
print(f"\nHandlers: {list(runner._func_call_handlers.keys())}")

# Monkey-patch to trace PC flow
original_extract_register = runner._extract_register
def traced_extract_register(context, marker_token):
    result = original_extract_register(context, marker_token)
    if marker_token == Token.REG_PC:
        print(f"  [TRACE] _extract_register(REG_PC) = {result}", flush=True)
    return result
runner._extract_register = traced_extract_register

original_override_register = runner._override_register_in_last_step
def traced_override_register(context, marker_token, value):
    if marker_token == Token.REG_PC:
        print(f"  [TRACE] _override_register_in_last_step(REG_PC, {value})", flush=True)
    original_override_register(context, marker_token, value)
runner._override_register_in_last_step = traced_override_register

original_handler_jsr = runner._handler_jsr
def traced_handler_jsr(context, output):
    exec_pc = runner._exec_pc()
    exec_idx = exec_pc // INSTR_WIDTH
    instr = bytecode[exec_idx] if 0 <= exec_idx < len(bytecode) else 0
    target = instr >> 8
    print(f"  [TRACE] _handler_jsr called: exec_pc={exec_pc}, target={target}", flush=True)
    original_handler_jsr(context, output)
runner._handler_jsr = traced_handler_jsr
runner._func_call_handlers[Opcode.JSR] = traced_handler_jsr

# Add step tracking
step_count = [0]
token_count = [0]
original_generate = runner.model.generate_next
def traced_generate(context, *args, **kwargs):
    token = original_generate(context, *args, **kwargs)
    token_count[0] += 1
    if token_count[0] <= 100:  # First 100 tokens
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
        else:
            print(flush=True)
    if token == Token.STEP_END:
        step_count[0] += 1
        if step_count[0] <= 5:
            exec_pc = runner._exec_pc()
            exec_idx = exec_pc // INSTR_WIDTH
            exec_op = bytecode[exec_idx] & 0xFF if 0 <= exec_idx < len(bytecode) else 0
            opname = "?"
            for name in dir(Opcode):
                if not name.startswith('_') and getattr(Opcode, name) == exec_op:
                    opname = name
                    break
            pc = runner._extract_register(context + [token], Token.REG_PC)
            print(f"\n[STEP {step_count[0]}] exec={opname} @ PC={exec_pc}, model_output_PC={pc}", flush=True)
    return token
runner.model.generate_next = traced_generate

print("\nRunning (max 5 steps)...")
try:
    result = runner.run(bytecode, data, max_steps=5)
    print(f"\nResult: {result}")
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()

print(f"\nFinal _last_pc: {runner._last_pc}")
