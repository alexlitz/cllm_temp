#!/usr/bin/env python3
"""Check actual AX bytes in step output."""
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token
from neural_vm.embedding import Opcode
from neural_vm.run_vm import AutoregressiveVMRunner

print("Building model...")
model = AutoregressiveVM()
set_vm_weights(model)
model = model.cuda()
model.eval()

runner = AutoregressiveVMRunner()
runner.model = model
bytecode = [Opcode.IMM | (42 << 8), Opcode.EXIT]

context = runner._build_context(bytecode, b'', [])
print(f"Program: IMM 42, EXIT\n")

# Generate step 0
for i in range(100):
    tok = model.generate_next(context)
    context.append(tok)
    if tok == Token.STEP_END:
        break

# Find REG_AX and print raw bytes
print("Step 0 tokens around REG_AX:")
for pos in range(20, len(context)):
    if context[pos] == 260:  # REG_AX
        print(f"Position {pos}: REG_AX (260)")
        for i in range(1, 5):
            byte_val = context[pos + i]
            print(f"Position {pos+i}: byte {i-1} = {byte_val} (0x{byte_val:02x})")

        # Decode as little-endian
        ax_bytes = [context[pos+i] for i in range(1, 5)]
        ax_le = sum(b << (i*8) for i, b in enumerate(ax_bytes))
        print(f"\nLittle-endian: 0x{ax_le:08x} = {ax_le}")

        # Decode as big-endian
        ax_be = sum(b << ((3-i)*8) for i, b in enumerate(ax_bytes))
        print(f"Big-endian:    0x{ax_be:08x} = {ax_be}")

        print(f"\nExpected: 0x0000002a (42)")
        if ax_le == 0x2a:
            print("✓ Little-endian matches!")
        elif ax_be == 0x2a:
            print("✓ Big-endian matches!")
        else:
            print(f"✗ Neither matches (LE={hex(ax_le)}, BE={hex(ax_be)})")
        break

# Check what the model predicts at REG_AX marker
print("\n" + "=" * 60)
print("Checking model prediction at REG_AX marker")
print("=" * 60)

for pos in range(20, len(context)):
    if context[pos] == 260:  # REG_AX
        token_ids = torch.tensor([context], dtype=torch.long, device='cuda')
        with torch.no_grad():
            logits = model.forward(token_ids)

        print(f"\nPredictions at REG_AX position {pos}:")
        for i in range(4):
            pred = logits[0, pos, :256].argmax(-1).item()
            actual = context[pos + i + 1] if pos + i + 1 < len(context) else None

            print(f"  Byte {i}: predicted=0x{pred:02x}, actual={f'0x{actual:02x}' if actual < 256 else actual}")

            # Show top 5 predictions
            top5 = torch.topk(logits[0, pos, :256], 5)
            print(f"    Top 5: {', '.join(f'0x{idx.item():02x}={val.item():.1f}' for val, idx in zip(top5.values, top5.indices))}")
        break
