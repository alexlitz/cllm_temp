"""Check what token is generated at position 113 (where NEXT_THINKING_END is set)."""

import sys
import os
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token

c_code = '''
int main() {
    printf("H\\n");
    return 0;
}
'''

print("Compiling...")
code, data = compile_c(c_code)

runner = AutoregressiveVMRunner(conversational_io=True)
if torch.cuda.is_available():
    runner.model = runner.model.cuda()

# Track all generated tokens
all_tokens = []
original_generate = runner.model.generate_next

def track_tokens(context):
    token = original_generate(context)
    all_tokens.append((len(context), token, runner.model._active_opcode))
    return token

runner.model.generate_next = track_tokens

print("\nRunning VM...")
output_str, exit_code = runner.run(code, data, [], max_steps=10)

print(f"\n✓ Execution complete, exit code: {exit_code}")

# Find tokens around position 113-120
print(f"\nTokens generated when context reached 113-120:")
for pos, tok, opcode in all_tokens:
    if 113 <= pos <= 120:
        tok_name = "?"
        if tok == Token.THINKING_END:
            tok_name = "THINKING_END ✓✓✓"
        elif tok == Token.THINKING_START:
            tok_name = "THINKING_START"
        elif tok == Token.STEP_END:
            tok_name = "STEP_END"
        elif tok == Token.REG_PC:
            tok_name = "REG_PC"
        elif 0 <= tok < 256:
            tok_name = f"byte_{tok:02x}"
        else:
            tok_name = f"tok_{tok}"

        op_str = f"0x{opcode:02x}" if opcode is not None else "None"
        print(f"  Context len {pos}: token={tok} ({tok_name}), active_opcode={op_str}")

# Check if THINKING_END was ever generated
if Token.THINKING_END in [t[1] for t in all_tokens]:
    print(f"\n✅ SUCCESS: THINKING_END was generated!")
    for pos, tok, _ in all_tokens:
        if tok == Token.THINKING_END:
            print(f"   At context length: {pos}")
else:
    print(f"\n❌ FAILURE: THINKING_END was never generated")
