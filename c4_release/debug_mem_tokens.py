#!/usr/bin/env python3
"""
Debug script to trace MEM token generation in detail.
"""

import sys
sys.path.insert(0, '/home/alexlitz/Documents/misc/c4_release/c4_release')

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token

# Simple function with ENT
code = '''
int helper(int x) {
    int local;
    local = x * 2;
    return local;
}

int main() {
    return helper(21);
}
'''

print("Compiling code...")
bytecode, data = compile_c(code, link_stdlib=False)
print(f"✓ Compiled: {len(bytecode)} bytes of bytecode\n")

print("Creating runner...")
runner = AutoregressiveVMRunner(
    n_layers=17,
    pure_attention_memory=False,
)

# Monkey-patch the MEM extraction to add detailed logging
original_extract = runner._extract_mem_section

def debug_extract_mem_section(context):
    """Enhanced version with detailed logging."""
    scan_back = Token.STEP_TOKENS + 5
    print(f"\n[DEBUG] Scanning for MEM in last {scan_back} tokens of context (len={len(context)})")

    for i in range(len(context) - 1, max(0, len(context) - scan_back), -1):
        if context[i] == Token.MEM:
            print(f"  Found Token.MEM at index {i}")
            if i + 8 < len(context):
                mem_section = list(context[i:i + 9])
                print(f"  Raw MEM section tokens: {mem_section}")

                # Decode address
                addr_bytes = [mem_section[1 + j] & 0xFF for j in range(4)]
                addr = sum(addr_bytes[j] << (j * 8) for j in range(4))
                print(f"  Address bytes: {[f'{b:02x}' for b in addr_bytes]} → 0x{addr:08x}")

                # Decode value
                val_bytes = [mem_section[5 + j] & 0xFF for j in range(4)]
                value = sum(val_bytes[j] << (j * 8) for j in range(4))
                print(f"  Value bytes:   {[f'{b:02x}' for b in val_bytes]} → 0x{value:08x}")

                return mem_section
            else:
                print(f"  ERROR: Not enough tokens after MEM marker (need 9, have {len(context) - i})")

    print(f"  No Token.MEM found in scan range")
    return None

runner._extract_mem_section = debug_extract_mem_section

print("Running VM with MEM debug tracing...\n")
print("="*70)

try:
    result = runner.run(bytecode, data, max_steps=100)
    print("\n" + "="*70)
    print(f"Exit code: {result.exit_code} (expected 42)")
    print(f"Steps: {result.steps}")
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
