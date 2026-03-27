#!/usr/bin/env python3
"""Detailed trace of what the model generates."""

from neural_vm.run_vm import AutoregressiveVMRunner, Token
from neural_vm.vm_step import set_vm_weights
from src.compiler import compile_c

# Compile simple program
code = "int main() { return 42; }"
bytecode, data = compile_c(code)

print(f"Program: {code}")
print(f"Bytecode: {bytecode}")
print()

# Token name mapping
TOKEN_NAMES = {
    Token.CODE_START: "CODE_START",
    Token.DATA_START: "DATA_START",
    Token.DATA_END: "DATA_END",
    Token.SEP: "SEP",
    Token.REG_PC: "PC",
    Token.REG_AX: "AX",
    Token.REG_SP: "SP",
    Token.REG_BP: "BP",
    Token.MEM: "MEM",
    Token.STACK0: "STACK0",
    Token.STEP_END: "STEP_END",
    Token.HALT: "HALT",
}

def token_name(tok):
    if tok in TOKEN_NAMES:
        return TOKEN_NAMES[tok]
    elif 0 <= tok <= 255:
        return f"0x{tok:02x}"
    else:
        return str(tok)

# Patch runner to capture generated tokens
class TracingRunner(AutoregressiveVMRunner):
    def run(self, bytecode, data=b"", argv=None, stdin="", max_steps=100000, tool_handler=None):
        self._generated_tokens = []

        # Capture original generate_next
        original_generate = self.model.generate_next

        def trace_generate(context):
            token = original_generate(context)
            self._generated_tokens.append(token)
            return token

        self.model.generate_next = trace_generate

        try:
            result = super().run(bytecode, data, argv, stdin, max_steps, tool_handler)
        finally:
            self.model.generate_next = original_generate

        return result

# Create and run
runner = TracingRunner()
set_vm_weights(runner.model)
runner.model.compact(block_size=32)
runner.model.compact_moe()

print("Running...\n")
output, exit_code = runner.run(bytecode, data, max_steps=10)

print(f"Generated {len(runner._generated_tokens)} tokens:\n")

# Print tokens grouped by step
step = 0
i = 0
while i < len(runner._generated_tokens):
    tok = runner._generated_tokens[i]

    if tok == Token.STEP_END:
        step += 1
        print(f"\n--- STEP_END (step {step}) ---\n")
        i += 1
    elif tok == Token.HALT:
        print(f"\n*** HALT ***\n")
        i += 1
        break
    elif tok in [Token.REG_PC, Token.REG_AX, Token.REG_SP, Token.REG_BP, Token.STACK0, Token.MEM]:
        # Register marker - print with its 4 bytes
        name = token_name(tok)
        if i + 4 < len(runner._generated_tokens):
            b0 = runner._generated_tokens[i+1]
            b1 = runner._generated_tokens[i+2]
            b2 = runner._generated_tokens[i+3]
            b3 = runner._generated_tokens[i+4]
            value = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
            print(f"{name} = 0x{value:08x} ({value})")
            i += 5
        else:
            print(f"{name} (incomplete)")
            i += 1
    else:
        print(f"  {token_name(tok)}")
        i += 1

print(f"\nExit code: {exit_code}")
print(f"Expected: 42")
