"""Full end-to-end integration test for conversational I/O.

This test actually runs a program with printf through the runner
and verifies that:
1. THINKING_END is generated during execution
2. Output is emitted by the runner
3. THINKING_START is emitted to resume
4. Execution continues after printf
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token

print("=" * 70)
print("FULL INTEGRATION TEST - Printf with Conversational I/O")
print("=" * 70)

c_code = '''
int main() {
    printf("Hello");
    return 42;
}
'''

print("\n1. Compiling program...")
code, data = compile_c(c_code)
print(f"   ✓ {len(code)} instructions, {len(data)} data bytes")

# Show what opcodes we have
print("\n2. Program opcodes:")
for i, instr in enumerate(code[:10]):
    op = instr & 0xFF
    op_name = "?"
    # Common opcodes
    op_names = {1: "IMM", 2: "LEA", 3: "JMP", 4: "JSR",
                15: "PSH", 33: "PRTF", 34: "EXIT"}
    op_name = op_names.get(op, str(op))
    marker = " <-- PRTF" if op == 33 else ""
    print(f"   {i}: {op_name} (0x{op:02x}){marker}")

print("\n3. Running with conversational_io=True...")
runner = AutoregressiveVMRunner(conversational_io=True)

# Track what happens
events = []

# Wrap generate_next to track THINKING tokens
original_generate = runner.model.generate_next
def track_generate(context):
    token = original_generate(context)
    if token == Token.THINKING_END:
        events.append("THINKING_END generated")
        print("   [EVENT] THINKING_END generated!")
    elif token == Token.THINKING_START:
        events.append("THINKING_START generated")
        print("   [EVENT] THINKING_START generated!")
    return token
runner.model.generate_next = track_generate

# Wrap set_active_opcode to track PRTF
original_set_opcode = runner.model.set_active_opcode
def track_opcode(opcode):
    if opcode == 33:
        events.append("PRTF opcode set")
        print("   [EVENT] PRTF opcode (33) set!")
    original_set_opcode(opcode)
runner.model.set_active_opcode = track_opcode

try:
    output, exit_code = runner.run(code, data, [], max_steps=15)

    print(f"\n4. Execution Results:")
    print(f"   Exit code: {exit_code}")
    print(f"   Output: {repr(output)}")
    print(f"   Events captured: {len(events)}")

    print(f"\n5. Event Timeline:")
    for i, event in enumerate(events):
        print(f"   {i+1}. {event}")

    print(f"\n6. Verification:")
    checks = {
        "PRTF opcode set": "PRTF opcode set" in events,
        "THINKING_END generated": "THINKING_END generated" in events,
        "THINKING_START generated": "THINKING_START generated" in events,
        "Output produced": len(output) > 0 or "Hello" in output,
        "Program completed": exit_code == 42,
    }

    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"   {status} {check}")

    print("\n" + "=" * 70)
    if all(checks.values()):
        print("✅ FULL INTEGRATION TEST PASSED!")
        print("   Conversational I/O is working end-to-end!")
    else:
        print("⚠️  PARTIAL SUCCESS - Some components need work")
        failed = [k for k, v in checks.items() if not v]
        print(f"   Failed checks: {', '.join(failed)}")
    print("=" * 70)

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
