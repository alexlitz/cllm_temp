"""
Debug IMM+EXIT with both fixes applied.
Trace layer-by-layer to see where contamination occurs.
"""

import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, _SetDim, Token

BD = _SetDim

# Build model
print("Building model...")
model = AutoregressiveVM()
set_vm_weights(model)
model = model.cuda()
model.eval()

# Build bytecode manually: IMM 42; EXIT
# IMM = 0x01, value goes in next 4 bytes
# EXIT = 0x02, exit code from AX
bytecode = [
    0x01, 0x2a, 0x00, 0x00, 0x00,  # IMM 42
    0x02, 0x00, 0x00, 0x00, 0x00,  # EXIT
]
print(f"Bytecode: IMM 42; EXIT")

# Build context (just CODE section, minimal)
context = [
    Token.CODE_START,
    bytecode[0], bytecode[1], bytecode[2], bytecode[3], bytecode[4],  # IMM 42
    bytecode[5], bytecode[6], bytecode[7], bytecode[8], bytecode[9],  # EXIT
    Token.CODE_END,
]

token_ids = torch.tensor([context], dtype=torch.long, device='cuda')
print(f"Context: {len(context)} tokens\n")

# Forward pass with layer tracing
with torch.no_grad():
    x = model.embed(token_ids)

    # Check after embedding
    print("After embedding (position 0 = CODE_START):")
    print(f"  Position 0:")
    print(f"    OP_IMM: {x[0, 0, BD.OP_IMM].item():.3f}")
    print(f"    MARK_PC: {x[0, 0, BD.MARK_PC].item():.3f}")
    print(f"    MARK_AX: {x[0, 0, BD.MARK_AX].item():.3f}")
    print(f"    HAS_SE: {x[0, 0, BD.HAS_SE].item():.3f}")

    # Position 1 = IMM opcode byte
    print(f"  Position 1 (IMM opcode byte 0x01):")
    print(f"    OP_IMM: {x[0, 1, BD.OP_IMM].item():.3f}")
    print(f"    ADDR_KEY+0x02 (should be 1.0): {x[0, 1, BD.ADDR_KEY + 2].item():.3f}")
    print()

    # Layer-by-layer
    for i, block in enumerate(model.blocks):
        x_before = x.clone()
        x = block(x, kv_cache=None)

        # Check key positions after each layer
        if i in [5, 6, 14]:  # Layer 6, 7, 15 (most relevant)
            print(f"After Layer {i+1}:")
            print(f"  Position 0:")
            print(f"    OUTPUT_HI[2]: {x[0, 0, BD.OUTPUT_HI + 2].item():.3f}")
            print(f"    FORMAT_PTR_HI+2: {x[0, 0, BD.FORMAT_PTR_HI + 2].item():.3f}")
            print(f"    HAS_SE: {x[0, 0, BD.HAS_SE].item():.3f}")
            print()

    # Final prediction
    logits = model.head(x)
    pred_token = logits[0, 0, :].argmax(-1).item()

    print(f"Predicted first output token: {pred_token}")
    if pred_token == Token.REG_PC:
        print("  ✓ Correct! (REG_PC)")
    else:
        print(f"  ✗ Wrong! Expected REG_PC ({Token.REG_PC}), got {pred_token}")

    # Check what PC byte would be predicted
    if len(model.blocks) > 0:
        print(f"\nOutput dimensions at position 0:")
        for k in range(8):
            lo_val = x[0, 0, BD.OUTPUT_LO + k].item()
            hi_val = x[0, 0, BD.OUTPUT_HI + k].item()
            if abs(lo_val) > 0.1 or abs(hi_val) > 0.1:
                print(f"  OUTPUT_LO[{k}]: {lo_val:.3f}, OUTPUT_HI[{k}]: {hi_val:.3f}")
