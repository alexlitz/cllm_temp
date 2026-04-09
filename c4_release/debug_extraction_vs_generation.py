#!/usr/bin/env python3
"""Compare step generation vs runner extraction to understand the discrepancy."""
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token
from neural_vm.embedding import Opcode
from neural_vm.run_vm import AutoregressiveVMRunner

model = AutoregressiveVM()
set_vm_weights(model)
model = model.cuda()
model.eval()

runner = AutoregressiveVMRunner()
runner.model = model

bytecode = [Opcode.IMM | (42 << 8), Opcode.EXIT]
context = runner._build_context(bytecode, b'', [])

print("=" * 80)
print("INVESTIGATING EXTRACTION VS GENERATION DISCREPANCY")
print("=" * 80)
print(f"Program: IMM 42, EXIT\n")

# Run to completion
for i in range(100):
    tok = model.generate_next(context)
    context.append(tok)
    if tok == Token.HALT:
        break

print(f"Generated {len(context)} tokens total\n")

# Find step boundaries
step_ends = [i for i, tok in enumerate(context) if tok == Token.STEP_END]
print(f"Step boundaries at positions: {step_ends}\n")

# Analyze each step
for step_idx, end_pos in enumerate(step_ends):
    start_pos = step_ends[step_idx - 1] + 1 if step_idx > 0 else 20  # Skip prompt
    step_tokens = context[start_pos:end_pos+1]

    print(f"Step {step_idx} (positions {start_pos}-{end_pos}):")
    print(f"  Tokens: {step_tokens[:10]}..." if len(step_tokens) > 10 else f"  Tokens: {step_tokens}")

    # Find REG_AX in this step
    reg_ax_pos = None
    for i, tok in enumerate(step_tokens):
        if tok == 260:  # REG_AX
            reg_ax_pos = start_pos + i
            break

    if reg_ax_pos:
        ax_bytes = [context[reg_ax_pos + 1 + i] for i in range(4)]
        ax_value = sum(b << (i*8) for i, b in enumerate(ax_bytes))
        print(f"  REG_AX at position {reg_ax_pos}: {ax_bytes} = 0x{ax_value:08x} ({ax_value})")
    else:
        print(f"  REG_AX not found in step output")

    print()

print("=" * 80)
print("RUNNER EXTRACTION")
print("=" * 80)

# Now check what runner extracts
output, exit_code = runner.run(bytecode, b'', [], max_steps=5)

print(f"Runner extracted exit code: {exit_code}")
print(f"Runner output steps: {len(output)}")

for step_idx, step in enumerate(output):
    print(f"\nStep {step_idx}:")
    for key, val in step.items():
        if key == 'ax':
            print(f"  AX: 0x{val:08x} ({val})")
        elif key == 'pc':
            print(f"  PC: 0x{val:08x} ({val})")

print("\n" + "=" * 80)
print("CHECKING RUNNER._EXTRACT_REGISTER METHOD")
print("=" * 80)

# Manually call _extract_register on the generated context
print("\nManually extracting from generated context:")

# Find last REG_AX before HALT
reg_ax_positions = []
for i in range(len(context)):
    if context[i] == 260:  # REG_AX
        reg_ax_positions.append(i)

if reg_ax_positions:
    last_ax_pos = reg_ax_positions[-1]
    print(f"Last REG_AX at position {last_ax_pos}")

    # Extract using runner's method
    extracted_value = runner._extract_register(context, 260)  # REG_AX token
    print(f"Runner._extract_register(context, REG_AX) = {extracted_value} (0x{extracted_value:08x})")

    # Manual extraction
    ax_bytes_manual = [context[last_ax_pos + 1 + i] for i in range(4)]
    ax_value_manual = sum(b << (i*8) for i, b in enumerate(ax_bytes_manual))
    print(f"Manual extraction from last REG_AX: {ax_bytes_manual} = {ax_value_manual} (0x{ax_value_manual:08x})")

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)
print("""
If the runner extracts 42 but step output shows 0:
1. Runner may search beyond step boundaries
2. Runner may use a different extraction logic
3. There may be multiple REG_AX markers in the sequence

Let's check the _extract_register implementation.
""")
