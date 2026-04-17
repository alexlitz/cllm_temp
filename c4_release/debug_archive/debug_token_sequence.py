"""Debug: What tokens are actually generated during execution?"""

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token, Opcode
import sys

code = "int main() { return 10 + 32; }"
bytecode, data = compile_c(code)

print("=" * 70, file=sys.stderr)
print("CHECKING TOKEN SEQUENCE DURING EXECUTION", file=sys.stderr)
print("=" * 70, file=sys.stderr)

runner = AutoregressiveVMRunner()
if Opcode.ADD in runner._func_call_handlers:
    del runner._func_call_handlers[Opcode.ADD]

# Capture generated tokens
generated_tokens = []
original_generate = runner.model.generate_next

def capture_generate(context):
    token = original_generate(context)
    generated_tokens.append(token)
    return token

runner.model.generate_next = capture_generate

print("Running VM...", file=sys.stderr)
_, exit_code = runner.run(bytecode, max_steps=2)
print(f"Exit code: {exit_code}\n", file=sys.stderr)

# Analyze token sequence
print(f"Total tokens generated: {len(generated_tokens)}", file=sys.stderr)
print("\nFirst 70 tokens:", file=sys.stderr)

marker_tokens = {
    Token.REG_PC: "REG_PC",
    Token.REG_AX: "REG_AX",
    Token.REG_SP: "REG_SP",
    Token.REG_BP: "REG_BP",
    Token.MEM: "MEM",
    Token.STACK0: "STACK0",
    Token.CODE_START: "CODE_START",
    Token.STEP_END: "STEP_END",
}

marker_positions = []
for i, tok in enumerate(generated_tokens[:70]):
    if tok in marker_tokens:
        tok_name = marker_tokens[tok]
        marker_positions.append((i, tok_name))
        print(f"  Token {i}: {tok_name} ← MARKER", file=sys.stderr)
    elif tok < 256:
        print(f"  Token {i}: byte 0x{tok:02x}", file=sys.stderr)
    else:
        print(f"  Token {i}: token {tok}", file=sys.stderr)

print("", file=sys.stderr)
print(f"Marker tokens found: {len(marker_positions)}", file=sys.stderr)
print("", file=sys.stderr)

# Check step format
if len(generated_tokens) >= 35:
    print("First step (tokens 0-34):", file=sys.stderr)
    step1_markers = [(i, name) for i, name in marker_positions if i < 35]
    print(f"  Markers in step 1: {len(step1_markers)}", file=sys.stderr)
    for i, name in step1_markers:
        print(f"    Position {i}: {name}", file=sys.stderr)
else:
    print(f"⚠ WARNING: Only {len(generated_tokens)} tokens generated (expected multiples of 35)", file=sys.stderr)

print("", file=sys.stderr)
print("=" * 70, file=sys.stderr)
print("DIAGNOSIS", file=sys.stderr)
print("=" * 70, file=sys.stderr)

expected_markers_per_step = 6  # REG_PC, REG_AX, REG_SP, REG_BP, STACK0, MEM (maybe multiple)
if len(marker_positions) == 0:
    print("❌ NO marker tokens generated!", file=sys.stderr)
    print("   This means the model is NOT generating REG_PC, REG_AX, etc.", file=sys.stderr)
    print("   Step format may be completely different than expected!", file=sys.stderr)
elif len(marker_positions) < expected_markers_per_step:
    print(f"⚠ WARNING: Only {len(marker_positions)} markers found (expected ~{expected_markers_per_step} per step)", file=sys.stderr)
    print("   Some markers are missing from the generated sequence", file=sys.stderr)
else:
    print(f"✓ Found {len(marker_positions)} marker tokens in first 70 tokens", file=sys.stderr)
    print("   Markers ARE being generated - issue must be elsewhere", file=sys.stderr)

print("=" * 70, file=sys.stderr)

del runner
import torch
torch.cuda.empty_cache()
