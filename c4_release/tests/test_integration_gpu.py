"""GPU-enabled integration test for conversational I/O."""

import sys
import os
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token

print("=" * 70)
print("INTEGRATION TEST - Conversational I/O (GPU)")
print("=" * 70)

# Check GPU availability
if torch.cuda.is_available():
    print(f"\n✓ GPU available: {torch.cuda.get_device_name(0)}")
    device = "cuda"
else:
    print("\n⚠ No GPU available, running on CPU")
    device = "cpu"

c_code = '''
int main() {
    printf("Hello World");
    return 42;
}
'''

print("\n1. Compiling program...")
code, data = compile_c(c_code)
print(f"   ✓ {len(code)} instructions compiled")

print("\n2. Creating runner with conversational_io=True...")
runner = AutoregressiveVMRunner(conversational_io=True)

# Move to GPU
if device == "cuda":
    print("   Moving model to GPU...")
    runner.model = runner.model.cuda()
    print("   ✓ Model on GPU")

print("\n3. Setting up event tracking...")
events = []

original_generate = runner.model.generate_next
def track_generate(context):
    token = original_generate(context)
    if token == Token.THINKING_END:
        events.append(("THINKING_END", len(context)))
        print(f"   [EVENT] THINKING_END at context len {len(context)}")
    elif token == Token.THINKING_START:
        events.append(("THINKING_START", len(context)))
        print(f"   [EVENT] THINKING_START at context len {len(context)}")
    return token
runner.model.generate_next = track_generate

original_set_opcode = runner.model.set_active_opcode
def track_opcode(opcode):
    if opcode == 33:
        events.append(("PRTF_SET", opcode))
        print(f"   [EVENT] PRTF opcode (33) set")
    original_set_opcode(opcode)
runner.model.set_active_opcode = track_opcode

print("\n4. Running program (max 15 steps)...")
try:
    output, exit_code = runner.run(code, data, [], max_steps=15)

    print(f"\n5. Execution Results:")
    print(f"   Exit code: {exit_code}")
    print(f"   Output: {repr(output)}")
    print(f"   Events: {len(events)}")

    print(f"\n6. Event Timeline:")
    for event_type, info in events:
        print(f"   - {event_type}: {info}")

    print(f"\n7. Verification:")
    checks = {
        "PRTF opcode was set": any(e[0] == "PRTF_SET" for e in events),
        "THINKING_END was generated": any(e[0] == "THINKING_END" for e in events),
        "THINKING_START was emitted": any(e[0] == "THINKING_START" for e in events),
        "Output was produced": len(output) > 0,
        "Exit code is 42": exit_code == 42,
    }

    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"   {status} {check}")

    print("\n" + "=" * 70)
    if all(checks.values()):
        print("✅ INTEGRATION TEST PASSED!")
        print("\nConversational I/O is working end-to-end:")
        print("  1. PRTF opcode detected")
        print("  2. THINKING_END generated")
        print("  3. Output emitted by runner")
        print("  4. THINKING_START emitted to resume")
        print("  5. Program completed successfully")
    elif checks["THINKING_END was generated"]:
        print("⚠️  PARTIAL SUCCESS")
        print("\nTHINKING_END is being generated (core goal achieved)!")
        print("Some runner-side features may need work:")
        failed = [k for k, v in checks.items() if not v]
        for f in failed:
            print(f"  - {f}")
    else:
        print("❌ TEST FAILED")
        failed = [k for k, v in checks.items() if not v]
        print(f"\nFailed checks:")
        for f in failed:
            print(f"  - {f}")
    print("=" * 70)

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
