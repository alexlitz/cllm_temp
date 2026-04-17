#!/usr/bin/env python3
"""Check context windowing issue - bytecode might be outside the attention window."""

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token
from neural_vm.embedding import Opcode
from src.compiler import compile_c

# Simple program
code = 'int main() { return 42; }'
bytecode, data = compile_c(code)

# Create runner
runner = AutoregressiveVMRunner()
runner._bytecode = bytecode
runner._last_sp = 0x1F800
runner._last_bp = 0x10000
ctx = runner._build_context(bytecode, data, [])

print(f"Total context length: {len(ctx)}")

# Find CODE section
code_start_pos = -1
code_end_pos = -1
data_start_pos = -1
data_end_pos = -1

for i, tok in enumerate(ctx):
    if tok == Token.CODE_START:
        code_start_pos = i
    elif tok == Token.CODE_END:
        code_end_pos = i
    elif tok == Token.DATA_START:
        data_start_pos = i
    elif tok == Token.DATA_END:
        data_end_pos = i

print(f"CODE section: positions {code_start_pos} to {code_end_pos} (length {code_end_pos - code_start_pos + 1})")
print(f"DATA section: positions {data_start_pos} to {data_end_pos} (length {data_end_pos - data_start_pos + 1})")

# Show tokens in CODE section
print(f"\nCODE section tokens:")
for i in range(code_start_pos, min(code_end_pos + 1, code_start_pos + 20)):
    tok = ctx[i]
    print(f"  Pos {i:4d}: {tok:3d}", end="")
    if tok == Token.CODE_START:
        print(" (CODE_START)")
    elif tok == Token.CODE_END:
        print(" (CODE_END)")
    elif i == code_start_pos + 1:
        print(f" (JSR opcode)")
    elif i == code_start_pos + 2:
        print(f" (imm byte 0 = 0x{tok:02x})")
    else:
        print()

# Check what's at position 0 in ctx[-512:]
window_start = len(ctx) - 512
print(f"\n512-token window: positions {window_start} to {len(ctx)}")
print(f"CODE section ends at: {code_end_pos}")
print(f"Window includes CODE? {window_start <= code_end_pos}")

if window_start > code_end_pos:
    print(f"\n*** PROBLEM: Bytecode is NOT in the 512-token window! ***")
    print(f"    Window starts at {window_start}, but CODE ends at {code_end_pos}")
    print(f"    Need to include at least {code_end_pos + 1} tokens from start")
