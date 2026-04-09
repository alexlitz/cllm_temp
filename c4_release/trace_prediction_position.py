"""
Trace OUTPUT at the position where first AX byte is predicted.
"""

import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim
from neural_vm.embedding import Opcode
from neural_vm.run_vm import AutoregressiveVMRunner

BD = _SetDim

# Build model
model = AutoregressiveVM()
set_vm_weights(model)
model.compact(block_size=32)
model.compact_moe()
model = model.cuda()
model.eval()

# Build bytecode and generate to step 1 REG_AX marker (but NOT past it)
bytecode = [Opcode.IMM | (42 << 8), Opcode.EXIT]
runner = AutoregressiveVMRunner()
runner.model = model
context = runner._build_context(bytecode, b'', [])

for _ in range(100):
    tok = model.generate_next(context)
    context.append(tok)
    if tok == Token.STEP_END:
        break

# Generate up to REG_AX marker (not past it)
for _ in range(6):
    tok = model.generate_next(context)
    context.append(tok)
    if tok == Token.REG_AX:
        break

# Now context ends with REG_AX marker
# The NEXT position will predict the first AX byte
marker_pos = len(context) - 1
byte_predict_pos = len(context)  # This is where we'll predict the first byte

print(f"REG_AX marker at position {marker_pos}")
print(f"First AX byte will be predicted at position {byte_predict_pos}")

token_ids = torch.tensor([context], dtype=torch.long, device='cuda')

with torch.no_grad():
    x = model.embed(token_ids)

    # Forward to Layer 7
    for i in range(7):
        x = model.blocks[i](x, kv_cache=None)

    print(f"\nAfter L6 at marker position {marker_pos}:")
    print(f"  OUTPUT_LO[0]: {x[0, marker_pos, BD.OUTPUT_LO + 0].item():.1f}")
    print(f"  OUTPUT_LO[10]: {x[0, marker_pos, BD.OUTPUT_LO + 10].item():.1f}")
    print(f"  OUTPUT_HI[0]: {x[0, marker_pos, BD.OUTPUT_HI + 0].item():.1f}")
    print(f"  OUTPUT_HI[2]: {x[0, marker_pos, BD.OUTPUT_HI + 2].item():.1f}")

    print("\nLayer-by-layer at marker position:")
    for layer_idx in range(7, 16):
        x_before = x.clone()
        x = model.blocks[layer_idx](x, kv_cache=None)

        lo0 = x[0, marker_pos, BD.OUTPUT_LO + 0].item()
        lo10 = x[0, marker_pos, BD.OUTPUT_LO + 10].item()
        hi0 = x[0, marker_pos, BD.OUTPUT_HI + 0].item()
        hi2 = x[0, marker_pos, BD.OUTPUT_HI + 2].item()

        lo0_delta = lo0 - x_before[0, marker_pos, BD.OUTPUT_LO + 0].item()
        lo10_delta = lo10 - x_before[0, marker_pos, BD.OUTPUT_LO + 10].item()

        marker = ""
        if abs(lo0_delta) > 5:
            marker = " ← L%d adds OUTPUT[0]!" % layer_idx

        print(f"After L{layer_idx}: LO[0]={lo0:.1f} (Δ{lo0_delta:+.1f}), LO[10]={lo10:.1f} (Δ{lo10_delta:+.1f}), HI[0]={hi0:.1f}, HI[2]={hi2:.1f}{marker}")

    # Final prediction
    logits = model.head(x)
    prediction = logits[0, marker_pos, :].argmax(-1).item()
    print(f"\nPredicted byte from marker position: 0x{prediction:02x}")
    if prediction == 0x2a:
        print("  ✓ CORRECT!")
    else:
        print(f"  ✗ WRONG! Expected 0x2a")
        # Show top predictions
        top5 = torch.topk(logits[0, marker_pos, :256], 5)
        print(f"\n  Top 5 predictions:")
        for val, idx in zip(top5.values, top5.indices):
            print(f"    0x{idx.item():02x}: {val.item():.1f}")
