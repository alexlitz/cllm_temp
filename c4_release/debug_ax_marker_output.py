"""Debug OUTPUT at AX marker position."""
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim as BD
from neural_vm.embedding import Opcode

def build_context(bytecode):
    tokens = [Token.CODE_START]
    for instr in bytecode:
        op = instr & 0xFF
        imm = instr >> 8
        tokens.append(op)
        for i in range(4):
            tokens.append((imm >> (i * 8)) & 0xFF)
    tokens.append(Token.CODE_END)
    tokens.append(Token.DATA_START)
    tokens.append(Token.DATA_END)
    return tokens

model = AutoregressiveVM()
set_vm_weights(model)
model.compact(block_size=32)
model.compact_moe()
model.eval()

bytecode = [Opcode.IMM | (42 << 8)]
context = build_context(bytecode)
# Add tokens up to AX marker
context.extend([Token.REG_PC, 10, 0, 0, 0, Token.REG_AX])

print(f"Context ends with REG_AX marker")
print(f"Predicting AX_b0 (should be 42)")

ctx_tensor = torch.tensor([context], dtype=torch.long)
with torch.no_grad():
    x = model.embed(ctx_tensor)
    model._add_code_addr_keys(ctx_tensor, x)
    model._inject_mem_store(ctx_tensor, x)

    ax_marker_pos = len(context) - 1  # Position of REG_AX marker

    print(f"\n=== TRACKING OUTPUT AT AX MARKER (pos {ax_marker_pos}) ===")

    for i, block in enumerate(model.blocks):
        x_before = x.clone()
        x = block(x)

        lo_before = [x_before[0, ax_marker_pos, BD.OUTPUT_LO + k].item() for k in range(16)]
        lo_after = [x[0, ax_marker_pos, BD.OUTPUT_LO + k].item() for k in range(16)]
        hi_before = [x_before[0, ax_marker_pos, BD.OUTPUT_HI + k].item() for k in range(16)]
        hi_after = [x[0, ax_marker_pos, BD.OUTPUT_HI + k].item() for k in range(16)]

        if lo_before != lo_after or hi_before != hi_after:
            lo_max = max(range(16), key=lambda k: lo_after[k])
            hi_max = max(range(16), key=lambda k: hi_after[k])
            val = lo_max | (hi_max << 4)

            lo_diff = sum(1 for k in range(16) if lo_before[k] != lo_after[k])
            hi_diff = sum(1 for k in range(16) if hi_before[k] != hi_after[k])

            print(f"Layer {i:2d}: OUTPUT changed ({lo_diff} LO + {hi_diff} HI dims), now encodes {val}")
            if val != 0:
                print(f"  OUTPUT_LO[{lo_max}] = {lo_after[lo_max]:.3f}")
                print(f"  OUTPUT_HI[{hi_max}] = {hi_after[hi_max]:.3f}")

    # Final check
    logits = model.head(x)
    predicted = logits[0, ax_marker_pos, :].argmax().item()
    print(f"\nFinal prediction at AX marker: {predicted} (expected: 42)")

    # Also check FETCH_LO/HI which Layer 6 reads for IMM
    print(f"\nFETCH values at AX marker:")
    for k in range(16):
        if x[0, ax_marker_pos, BD.FETCH_LO + k].item() > 0.1:
            print(f"  FETCH_LO[{k}] = {x[0, ax_marker_pos, BD.FETCH_LO + k].item():.3f}")
    for k in range(16):
        if x[0, ax_marker_pos, BD.FETCH_HI + k].item() > 0.1:
            print(f"  FETCH_HI[{k}] = {x[0, ax_marker_pos, BD.FETCH_HI + k].item():.3f}")
