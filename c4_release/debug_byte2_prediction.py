#!/usr/bin/env python3
"""Debug what causes byte 2 to be 0x01."""
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim
from neural_vm.embedding import Opcode
from neural_vm.run_vm import AutoregressiveVMRunner

BD = _SetDim

model = AutoregressiveVM()
set_vm_weights(model)
model = model.cuda()
model.eval()

runner = AutoregressiveVMRunner()
runner.model = model
bytecode = [Opcode.IMM | (42 << 8), Opcode.EXIT]
context = runner._build_context(bytecode, b'', [])

# Generate up to position 38 (2 bytes after REG_AX)
for i in range(100):
    tok = model.generate_next(context)
    context.append(tok)
    if len(context) == 38:  # Stop at position 37 (before byte 2)
        break

print(f"Context up to position {len(context)-1}:")
print(f"Last 10 tokens: {context[-10:]}")

# Find REG_AX
reg_ax_pos = None
for pos in range(len(context)):
    if context[pos] == 260:  # REG_AX
        reg_ax_pos = pos
        break

print(f"\nREG_AX at position {reg_ax_pos}")
print(f"  Byte 0 (pos {reg_ax_pos+1}): 0x{context[reg_ax_pos+1]:02x}")
print(f"  Byte 1 (pos {reg_ax_pos+2}): 0x{context[reg_ax_pos+2]:02x}")
print(f"  About to predict byte 2 at pos {reg_ax_pos+3}")

print("\n" + "=" * 80)
print(f"ANALYZING POSITION {len(context)-1} (predicting byte 2)")
print("=" * 80)

token_ids = torch.tensor([context], dtype=torch.long, device='cuda')

with torch.no_grad():
    x = model.embed(token_ids)

    # Run through all layers
    for layer_idx in range(16):
        x = model.blocks[layer_idx](x, kv_cache=None)

    # Check the last position (where we're predicting byte 2)
    pred_pos = len(context) - 1
    print(f"\nAfter all 16 layers, at position {pred_pos}:")

    # Check OUTPUT dimensions
    output_lo_vals = []
    for i in range(16):
        val = x[0, pred_pos, BD.OUTPUT_LO + i].item()
        if abs(val) > 0.1:
            output_lo_vals.append((i, val))

    if output_lo_vals:
        print(f"  OUTPUT_LO non-zero nibbles: {output_lo_vals}")
        # Find which byte this encodes
        max_idx, max_val = max(output_lo_vals, key=lambda x: abs(x[1]))
        print(f"  OUTPUT_LO max nibble: {max_idx} (0x{max_idx:x}) with value {max_val:.3f}")

    # Check if there's a pattern in higher nibbles
    output_hi_vals = []
    for i in range(16):
        val = x[0, pred_pos, BD.OUTPUT_HI + i].item()
        if abs(val) > 0.1:
            output_hi_vals.append((i, val))

    if output_hi_vals:
        print(f"  OUTPUT_HI non-zero nibbles: {output_hi_vals}")

    # Get actual logits
    logits = model.head(x)
    pred = logits[0, pred_pos, :256].argmax(-1).item()

    print(f"\n  Final prediction: 0x{pred:02x}")

    # Show top 5
    top5 = torch.topk(logits[0, pred_pos, :256], 5)
    print(f"  Top 5 predictions:")
    for val, idx in zip(top5.values, top5.indices):
        print(f"    0x{idx.item():02x}: {val.item():.3f}")

    # Check specific dimensions that might contribute to 0x01
    print(f"\n  Checking dimensions that could produce 0x01:")
    print(f"    EMBED_LO[1]: {x[0, pred_pos, BD.EMBED_LO + 1].item():.3f}")
    print(f"    OUTPUT_LO[1]: {x[0, pred_pos, BD.OUTPUT_LO + 1].item():.3f}")
    print(f"    FETCH_LO[1]: {x[0, pred_pos, BD.FETCH_LO + 1].item():.3f}")

print("\n" + "=" * 80)
print("POSITION TRACE")
print("=" * 80)

# Check multiple byte positions
for byte_idx in range(4):
    pos = reg_ax_pos + 1 + byte_idx
    if pos < len(context):
        actual = context[pos]
        print(f"\nByte {byte_idx} at position {pos}:")
        print(f"  Actual value: 0x{actual:02x}")

        # Get prediction
        token_ids_up_to = torch.tensor([context[:pos]], dtype=torch.long, device='cuda')
        with torch.no_grad():
            logits = model.forward(token_ids_up_to)
            pred = logits[0, -1, :256].argmax(-1).item()
            print(f"  Predicted: 0x{pred:02x}")
