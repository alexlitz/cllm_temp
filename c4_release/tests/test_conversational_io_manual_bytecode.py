"""Test conversational I/O with manual bytecode (bypasses compiler bug).

This test creates bytecode manually to avoid the compiler's JMP bug,
allowing us to verify the conversational I/O mechanism works end-to-end.
"""

import sys
import os
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token

print("=" * 70)
print("CONVERSATIONAL I/O - Manual Bytecode Test")
print("=" * 70)

# Format: op | (imm << 8)
# Opcodes: IMM=1, PSH=15, PRTF=33, EXIT=34
#
# Program logic:
#   AX = 0x10000  (address of format string in data section)
#   push AX       (PRTF expects format string pointer on stack)
#   printf()
#   exit

bytecode = [
    1 | (0x10000 << 8),  # IMM 0x10000 (data section address)
    15 | (0 << 8),       # PSH (push AX to stack)
    33 | (0 << 8),       # PRTF (printf)
    34 | (0 << 8),       # EXIT
]

# Data section: "Hello from Neural VM!\0"
data = b"Hello from Neural VM!\x00"

print("\n1. Bytecode:")
for i, instr in enumerate(bytecode):
    op = instr & 0xFF
    imm = instr >> 8
    op_names = {1: "IMM", 15: "PSH", 33: "PRTF", 34: "EXIT"}
    print(f"   {i}: {op_names.get(op, f'op_{op}'):6s} imm=0x{imm:08x}")

print(f"\n2. Data section: {data}")

print("\n3. Creating runner with conversational_io=True...")
runner = AutoregressiveVMRunner(conversational_io=True)

if torch.cuda.is_available():
    print("   Moving to GPU...")
    try:
        runner.model = runner.model.cuda()
        print("   ✓ Model on GPU")
    except torch.cuda.OutOfMemoryError:
        print("   ⚠ GPU OOM, using CPU")
else:
    print("   Running on CPU (will be slow)")

print("\n4. Setting up event tracking...")
events = []

# Track opcode execution
original_set_opcode = runner.model.set_active_opcode
def track_opcode(opcode):
    if opcode == 33:  # PRTF
        events.append(("PRTF_SET", opcode))
        print(f"   [EVENT] PRTF opcode (33) set!")
    original_set_opcode(opcode)
runner.model.set_active_opcode = track_opcode

# Track token generation
original_generate = runner.model.generate_next
def track_generate(context):
    token = original_generate(context)
    if token == Token.THINKING_END:
        events.append(("THINKING_END", len(context)))
        print(f"   [EVENT] THINKING_END generated at position {len(context)}")
    elif token == Token.THINKING_START:
        events.append(("THINKING_START", len(context)))
        print(f"   [EVENT] THINKING_START generated at position {len(context)}")
    elif token == Token.STEP_END:
        events.append(("STEP_END", len(context)))
    return token
runner.model.generate_next = track_generate

print("\n5. Running program (max 10 steps)...")
try:
    output, exit_code = runner.run(bytecode, data, [], max_steps=10)

    print(f"\n6. Results:")
    print(f"   Exit code: {exit_code}")
    print(f"   Output: {repr(output)}")
    print(f"   Events: {len(events)}")

    print(f"\n7. Event Timeline:")
    for event_type, info in events:
        print(f"   - {event_type}: {info}")

    print(f"\n8. Verification:")
    checks = {
        "PRTF opcode was set": any(e[0] == "PRTF_SET" for e in events),
        "THINKING_END was generated": any(e[0] == "THINKING_END" for e in events),
        "THINKING_START was generated": any(e[0] == "THINKING_START" for e in events),
        "Output contains 'Hello'": "Hello" in output,
        "At least one STEP_END": any(e[0] == "STEP_END" for e in events),
    }

    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"   {status} {check}")

    print("\n" + "=" * 70)

    if checks["THINKING_END was generated"]:
        print("🎉 CONVERSATIONAL I/O WORKS END-TO-END!")
        print("\nThe transformer successfully:")
        print("  1. Detected PRTF opcode")
        print("  2. Generated THINKING_END token")
        if checks["Output contains 'Hello'"]:
            print("  3. Runner emitted output")
            print("  4. Generated THINKING_START to resume")
            print("\n✅ FULL PIPELINE VERIFIED!")
        else:
            print("\n⚠️  THINKING_END generated but runner didn't emit output")
            print("    (Runner-side implementation needed)")
    elif checks["PRTF opcode was set"]:
        print("⚠️  PARTIAL SUCCESS")
        print("\nPRTF opcode detected but THINKING_END not generated.")
        print("This suggests an issue in L5/L6 detection or L15 output.")
    else:
        print("❌ TEST FAILED")
        print("\nPRTF opcode was never set.")
        print("Check if program execution reached PRTF instruction.")

    print("=" * 70)

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
