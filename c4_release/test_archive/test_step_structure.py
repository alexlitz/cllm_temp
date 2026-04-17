#!/usr/bin/env python3
"""Examine step 0 structure and PC values."""
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim
from neural_vm.embedding import Opcode
from neural_vm.run_vm import AutoregressiveVMRunner

BD = _SetDim

print("Building model...")
model = AutoregressiveVM()
set_vm_weights(model)
model = model.cuda()
model.eval()

runner = AutoregressiveVMRunner()
runner.model = model
bytecode = [Opcode.IMM | (42 << 8), Opcode.EXIT]

# Use runner's context builder
context = runner._build_context(bytecode, b'', [])
initial_len = len(context)
print(f"Initial context: {initial_len} tokens\n")

# Generate step 0
print("Generating step 0...")
for i in range(100):
    tok = model.generate_next(context)
    context.append(tok)
    if tok == Token.STEP_END:
        print(f"Step 0 complete\n")
        break

# Print step 0 structure
print("Step 0 tokens:")
print("Pos  | Token      | Name")
print("-" * 40)

token_names = {
    257: "REG_PC",
    258: "REG_BP",
    259: "REG_SP",
    260: "REG_AX",
    261: "REG_CYCLE",
    262: "STEP_END",
    263: "MEM",
    264: "CODE_START",
}

for pos in range(initial_len, len(context)):
    tok = context[pos]
    if tok in token_names:
        name = token_names[tok]
    elif tok < 256:
        name = f"0x{tok:02x}"
    else:
        name = str(tok)

    print(f"{pos:4} | {tok:10} | {name}")

# Now check PC at REG_PC marker
print("\n" + "=" * 60)
print("Checking PC value at REG_PC marker")
print("=" * 60)

token_ids = torch.tensor([context], dtype=torch.long, device='cuda')

with torch.no_grad():
    logits = model.forward(token_ids)

    # Find REG_PC position
    reg_pc_pos = None
    for pos in range(initial_len, len(context)):
        if context[pos] == 257:  # REG_PC
            reg_pc_pos = pos
            break

    if reg_pc_pos is None:
        print("REG_PC not found!")
    else:
        print(f"\nREG_PC at position {reg_pc_pos}")

        # Predict PC bytes after REG_PC marker
        print("\nPredicted PC bytes (using logits after REG_PC):")
        pc_bytes = []
        for i in range(4):
            next_pos = reg_pc_pos + i
            prediction = logits[0, reg_pc_pos, :256].argmax(-1).item()
            actual = context[next_pos + 1] if next_pos + 1 < len(context) else None

            pc_bytes.append(prediction)
            print(f"  Byte {i}: predicted=0x{prediction:02x}, actual={f'0x{actual:02x}' if actual is not None and actual < 256 else actual}")

        pc_predicted = sum(b << (i*8) for i, b in enumerate(pc_bytes))
        print(f"\nPredicted PC: 0x{pc_predicted:08x}")

        # Check actual PC bytes in context
        actual_pc_bytes = []
        for i in range(4):
            next_pos = reg_pc_pos + i + 1
            if next_pos < len(context) and context[next_pos] < 256:
                actual_pc_bytes.append(context[next_pos])
            else:
                break

        if len(actual_pc_bytes) == 4:
            actual_pc = sum(b << (i*8) for i, b in enumerate(actual_pc_bytes))
            print(f"Actual PC:    0x{actual_pc:08x}")

            if actual_pc == 0x0a:
                print("\n✓ PC IS CORRECT (0x0a)!")
            else:
                print(f"\n✗ PC IS WRONG (expected 0x0a, got 0x{actual_pc:08x})")
