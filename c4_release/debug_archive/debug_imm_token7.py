"""Debug token 7 (PC_b1) prediction for IMM 42."""
import sys
for mod in list(sys.modules.keys()):
    if 'neural_vm' in mod:
        del sys.modules[mod]

import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim as BD
from neural_vm.embedding import Opcode
from neural_vm.speculative import DraftVM

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

print("Token 7 (PC_b1) Debug for IMM 42")
print("=" * 70)
print()

bytecode = [Opcode.IMM | (42 << 8)]
context = build_context(bytecode)

draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

print("Expected sequence:")
for i in range(10):
    print(f"  Token {i}: {draft_tokens[i]}")
print()

# Build context up to token 7
ctx_7 = context + draft_tokens[:7]
print(f"Context for token 7 (length {len(ctx_7)}): {ctx_7}")
print(f"Last 5 tokens: {ctx_7[-5:]}")
print()

# Capture Layer 15 output
l15_out = None
def hook(module, input, output):
    global l15_out
    l15_out = output.detach().clone()

model.blocks[15].register_forward_hook(hook)

ctx_tensor = torch.tensor([ctx_7], dtype=torch.long)
with torch.no_grad():
    logits = model.forward(ctx_tensor)

# Check OUTPUT values at last position
if l15_out is not None:
    pos = len(ctx_7) - 1
    output_lo = l15_out[0, pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    output_hi = l15_out[0, pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16]

    print(f"OUTPUT values at position {pos} (after L15):")
    print(f"  OUTPUT_LO argmax: {torch.argmax(output_lo).item()} (expected: 0)")
    print(f"  OUTPUT_HI argmax: {torch.argmax(output_hi).item()} (expected: 0)")
    print()
    print(f"  OUTPUT_LO top values: {output_lo.topk(5)}")
    print(f"  OUTPUT_HI top values: {output_hi.topk(5)}")
    print()

# Check logits
predicted = torch.argmax(logits[0, -1]).item()
print(f"Predicted: {predicted}")
print(f"Expected:  {draft_tokens[7]} (PC_b1 = byte 0)")
print()

top_10_vals, top_10_idx = torch.topk(logits[0, -1], 10)
print("Top 10 predictions:")
for i, (val, idx) in enumerate(zip(top_10_vals, top_10_idx)):
    print(f"  {i+1:2d}. Token {idx.item():3d}: logit={val.item():8.2f}")
print()

# Analysis
print("Analysis:")
print("  PC value after IMM 42: 42 (0x2A)")
print("  PC bytes: [42, 0, 0, 0]")
print("  PC nibbles: [0xA, 0x2, 0, 0, 0, 0, 0, 0]")
print()
print("  Token 6 (PC_b0): Should output low nibble of byte 0 = 0xA = 10")
print("  Token 7 (PC_b1): Should output high nibble of byte 0 = 0x2 = 2")
print()
print("  But model predicts 42 for token 7, same as PC_b0!")
print("  This suggests PC_b0 value is being relayed/carried incorrectly.")
