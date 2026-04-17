#!/usr/bin/env python3
"""Test that JSR/LEV handlers are removed."""

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode

runner = AutoregressiveVMRunner()

print("=" * 80)
print("Handler Status After Removal")
print("=" * 80)

handlers = runner._func_call_handlers

print(f"\nTotal handlers: {len(handlers)}")

if len(handlers) == 0:
    print("\n✅ SUCCESS! No VM operation handlers registered!")
    print("   VM should now be 100% neural (excluding I/O boundaries)")
else:
    print(f"\n⚠️  {len(handlers)} handlers still registered:")
    for op, handler in handlers.items():
        print(f"  - {op.name if hasattr(op, 'name') else op}: {handler.__name__}")

# Check specific ops
ops_to_check = [
    ('JSR', Opcode.JSR),
    ('LEV', Opcode.LEV),
    ('ENT', Opcode.ENT),
    ('PSH', Opcode.PSH),
]

print("\n" + "=" * 80)
print("VM Operation Status:")
print("=" * 80)
for name, opcode in ops_to_check:
    if opcode in handlers:
        print(f"  {name}: ❌ HAS HANDLER")
    else:
        print(f"  {name}: ✅ FULLY NEURAL")

print("=" * 80)
