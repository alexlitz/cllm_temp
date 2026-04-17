#!/usr/bin/env python3
"""Debug LEV - check memory state after JSR/ENT to see if values are being stored."""

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode
from src.compiler import compile_c
from pathlib import Path
import tempfile

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
print("LEV Memory State Debug")
print("=" * 70)

# Compile
with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
    f.write(code)
    f.flush()
    temp_path = f.name

bytecode, data = compile_c(Path(temp_path).read_text())
Path(temp_path).unlink()

print(f"\nBytecode: {len(bytecode)} instructions")
print(f"Data: {len(data)} bytes")

# Look for JSR/ENT/LEV in bytecode
for i, instr in enumerate(bytecode):
    op = instr & 0xFF
    if op in [Opcode.JSR, Opcode.ENT, Opcode.LEV]:
        imm = instr >> 8
        op_names = {Opcode.JSR: 'JSR', Opcode.ENT: 'ENT', Opcode.LEV: 'LEV'}
        print(f"  [{i:4d}] PC=0x{i*5:08x}: {op_names[op]:3s} 0x{imm:08x}")

print()

# Create runner with debug enabled
runner = AutoregressiveVMRunner()
runner._debug_lev = True
runner._debug_ent = True

# Monkey-patch to inspect memory state after JSR/ENT
original_mem_store = runner._mem_store_word

def debug_mem_store(addr, value):
    result = original_mem_store(addr, value)
    print(f"  [MEM WRITE] addr=0x{addr:08x}, value=0x{value:08x}", flush=True)
    return result

runner._mem_store_word = debug_mem_store

# Run with limited steps
print("Running program (max 30 steps)...\n")
try:
    result = runner.run(bytecode, data, [], max_steps=30)
    print(f"\nProgram result: {result}")
except Exception as e:
    print(f"\nProgram stopped: {e}")

# Inspect memory state
print(f"\n{'=' * 70}")
print("Final Memory State:")
print(f"{'=' * 70}")
print(f"Memory entries: {len(runner._mem_history)}")
for addr in sorted(runner._mem_history.keys()):
    value = runner._mem_load_word(addr)
    print(f"  [0x{addr:08x}] = 0x{value:08x}")
