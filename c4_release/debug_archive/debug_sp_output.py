#!/usr/bin/env python3
"""
Debug script to check what raw tokens are in the last step context.
"""

import sys
sys.path.insert(0, '/home/alexlitz/Documents/misc/c4_release/c4_release')

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token

# Simple JSR operation
code = '''
int helper() {
    return 42;
}

int main() {
    return helper();
}
'''

print("Compiling code...")
bytecode, data = compile_c(code, link_stdlib=False)
print(f"✓ Compiled: {len(bytecode)} bytes\n")

print("Creating runner...")
runner = AutoregressiveVMRunner(
    n_layers=17,
    pure_attention_memory=False,
)

# Monkey-patch to intercept context after first JSR
step_count = [0]
jsr_found = [False]

original_generate = runner.model.generate

def debug_generate(*args, **kwargs):
    result = original_generate(*args, **kwargs)
   
    step_count[0] += 1
    if result == Token.STEP_END and step_count[0] == 2:  # After first real operation (JSR)
        # Extract the last 35 tokens (one step)
        context = args[0][0].tolist()  # input_ids tensor
        last_step = context[-35:]
        
        print(f"\n{'='*70}")
        print(f"STEP {step_count[0]}: Last 35 tokens (one execution step)")
        print(f"{'='*70}\n")
        
        # Decode token sequence
        markers = {
            257: "REG_PC",
            258: "REG_AX",
            259: "REG_SP",
            260: "REG_BP",
            261: "STACK0",
            262: "MEM",
            263: "STEP_END",
        }
        
        for i, token in enumerate(last_step):
            if token in markers:
                print(f"  [{i:2d}] {markers[token]}")
            elif i in [1,2,3,4]:
                print(f"  [{i:2d}] PC byte {i-1}: 0x{token:02x}")
            elif i in [6,7,8,9]:
                print(f"  [{i:2d}] AX byte {i-6}: 0x{token:02x}")
            elif i in [11,12,13,14]:
                print(f"  [{i:2d}] SP byte {i-11}: 0x{token:02x}")
            elif i in [16,17,18,19]:
                print(f"  [{i:2d}] BP byte {i-16}: 0x{token:02x}")
            elif i in [21,22,23,24]:
                print(f"  [{i:2d}] STACK0 byte {i-21}: 0x{token:02x}")
            elif i in [26,27,28,29]:
                print(f"  [{i:2d}] MEM addr byte {i-26}: 0x{token:02x}")
            elif i in [30,31,32,33]:
                print(f"  [{i:2d}] MEM val byte {i-30}: 0x{token:02x}")
            else:
                print(f"  [{i:2d}] {token}")
        
        # Reconstruct values
        sp_val = sum(last_step[11+i] << (i*8) for i in range(4))
        mem_addr = sum(last_step[26+i] << (i*8) for i in range(4))
        mem_val = sum(last_step[30+i] << (i*8) for i in range(4))
        
        print(f"\nDecoded values:")
        print(f"  SP = 0x{sp_val:08x}")
        print(f"  MEM addr = 0x{mem_addr:08x}")
        print(f"  MEM val = 0x{mem_val:08x}")
        print()
        jsr_found[0] = True
        
    return result

runner.model.generate = debug_generate

print("Running VM...\n")

try:
    result = runner.run(bytecode, data, max_steps=20)
    print(f"\n{'='*70}")
    print(f"Exit code: {result.exit_code} (expected 42)")
    print(f"Steps: {result.steps}")
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
