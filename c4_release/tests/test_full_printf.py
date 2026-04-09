"""Test full printf with conversational I/O enabled."""

import sys
import os
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token

c_code = '''
int main() {
    printf("Hello\\n");
    return 0;
}
'''

print("Compiling...")
code, data = compile_c(c_code)
print(f"✓ Compiled: {len(code)} instructions, {len(data)} data bytes")

# Show data section content
print("\nData section:")
for i, b in enumerate(data):
    if 32 <= b < 127:
        print(f"  0x{i:04x}: {b:3d} (0x{b:02x}) '{chr(b)}'")
    else:
        print(f"  0x{i:04x}: {b:3d} (0x{b:02x})")

print("\nRunning with conversational_io=True...")
runner = AutoregressiveVMRunner(conversational_io=True)
if torch.cuda.is_available():
    runner.model = runner.model.cuda()

# Track tokens to see what's generated
generated_tokens = []
original_generate = runner.model.generate_next

def track_all_tokens(context):
    token = original_generate(context)
    generated_tokens.append(token)
    if token == Token.THINKING_END:
        print(f"[TRACK] THINKING_END at context len {len(context)}")
    elif token == Token.THINKING_START:
        print(f"[TRACK] THINKING_START at context len {len(context)}")
    return token

runner.model.generate_next = track_all_tokens

try:
    output_str, exit_code = runner.run(code, data, [], max_steps=20)
    print(f"\n✓ Execution complete")
    print(f"Exit code: {exit_code}")
    print(f"Output: {repr(output_str)}")

    # Check what tokens were generated
    thinking_end_count = generated_tokens.count(Token.THINKING_END)
    thinking_start_count = generated_tokens.count(Token.THINKING_START)

    print(f"\nToken counts:")
    print(f"  THINKING_END: {thinking_end_count}")
    print(f"  THINKING_START: {thinking_start_count}")

    # Check for byte tokens (output)
    byte_tokens = [t for t in generated_tokens if 0 <= t < 256]
    if byte_tokens:
        print(f"  Byte tokens generated: {len(byte_tokens)}")
        # Show first few bytes
        byte_str = ''.join(chr(b) if 32 <= b < 127 else f'\\x{b:02x}' for b in byte_tokens[:20])
        print(f"  First bytes: {repr(byte_str)}")

    if thinking_end_count > 0:
        print(f"\n✅ SUCCESS: Conversational I/O triggered!")
    else:
        print(f"\n❌ No THINKING_END generated")

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
