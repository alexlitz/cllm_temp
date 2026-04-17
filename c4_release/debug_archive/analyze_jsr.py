#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')

from neural_vm.speculative import DraftVM

# JSR 16 instruction
bytecode = [0x1003]  # JSR 16

print("JSR INSTRUCTION ANALYSIS")
print("=" * 60)

vm = DraftVM(bytecode)
print(f"\nBefore JSR:")
print(f"  idx = {vm.idx}")
print(f"  pc  = {vm.pc}")

# Manually trace through JSR execution
print(f"\nJSR execution steps:")
print(f"  1. idx += 1  →  idx = {vm.idx + 1}")
print(f"  2. pc = idx_to_pc({vm.idx + 1})  →  pc = {(vm.idx + 1) * 8}")  # INSTR_WIDTH=8
print(f"  3. Push return address (pc={(vm.idx + 1) * 8}) to stack")
print(f"  4. Jump: idx = imm // 8 = 16 // 8 = 2")
print(f"  5. pc = idx_to_pc(2) = 2 * 8 = 16")

vm.step()

print(f"\nAfter JSR:")
print(f"  idx = {vm.idx}")
print(f"  pc  = {vm.pc}")
print(f"  sp  = 0x{vm.sp:08x}")
print(f"  Stack[sp] = {vm.memory.get(vm.sp, 'N/A')} (return address)")

print(f"\n" + "=" * 60)
print("KEY INSIGHT:")
print("=" * 60)
print(f"""
The return address pushed to stack is: {vm.memory.get(vm.sp, 'N/A')}
The jump target (new PC) is:          {vm.pc}

If the transformer predicts PC=8, it's predicting the RETURN ADDRESS,
not the JUMP TARGET!

This suggests the transformer may have learned:
  "Output the return address" 
instead of:
  "Output the new PC after jump"
""")
