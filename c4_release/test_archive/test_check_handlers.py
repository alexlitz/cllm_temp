#!/usr/bin/env python3
"""Quick check: which handlers are registered?"""

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode

runner = AutoregressiveVMRunner()

print("=" * 80)
print("Handler Registration Status")
print("=" * 80)

handlers = runner._func_call_handlers

print("\nRegistered handlers:")
for opcode, handler in handlers.items():
    opcode_name = opcode.name if hasattr(opcode, 'name') else str(opcode)
    handler_name = handler.__name__
    print(f"  {opcode_name}: {handler_name}")

print(f"\nTotal: {len(handlers)} handlers")

print("\n" + "=" * 80)
print("Status by operation:")
print("=" * 80)

ops_to_check = [
    ('IMM', Opcode.IMM),
    ('LEA', Opcode.LEA),
    ('ENT', Opcode.ENT),
    ('JSR', Opcode.JSR),
    ('LEV', Opcode.LEV),
    ('PSH', Opcode.PSH),
]

for name, opcode in ops_to_check:
    if opcode in handlers:
        print(f"  {name}: ❌ HAS HANDLER (not fully neural)")
    else:
        print(f"  {name}: ✅ NO HANDLER (fully neural)")

print("\n" + "=" * 80)
if len(handlers) == 0:
    print("✅ 100% NEURAL - No handlers registered!")
else:
    print(f"⚠️  {len(handlers)} handlers still active - not 100% neural yet")
print("=" * 80)
