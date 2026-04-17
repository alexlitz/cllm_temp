"""Debug IMM 1 failure at token 6."""
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

print("IMM 1 Debug - Token 6 Failure")
print("=" * 70)

bytecode = [Opcode.IMM | (1 << 8)]
context = build_context(bytecode)

draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

# Build context up to and including REG_AX (token 5)
ctx = context + draft_tokens[:6]  # Include REG_AX
pos = len(ctx) - 1

print(f"Context last 3: {ctx[-3:]}")
print(f"Position {pos}: token {ctx[pos]} (should be REG_AX = 258)")
print(f"Predicting token 6 (AX_b0): expected {draft_tokens[6]}, should be 1")
print()

# Capture L3 and L6 outputs
l3_out = None
l6_out = None

def l3_hook(module, input, output):
    global l3_out
    l3_out = output.detach().clone()

def l6_hook(module, input, output):
    global l6_out
    l6_out = output.detach().clone()

model.blocks[3].register_forward_hook(l3_hook)
model.blocks[6].register_forward_hook(l6_hook)

ctx_tensor = torch.tensor([ctx], dtype=torch.long)
with torch.no_grad():
    logits = model.forward(ctx_tensor)
    pred = torch.argmax(logits[0, -1, :]).item()

print(f"Prediction: {pred} (expected {draft_tokens[5]})")
print()

if l3_out is not None:
    print("After L3:")
    h1_ax = l3_out[0, pos, BD.H1 + 1].item()
    byte_idx_0 = l3_out[0, pos, BD.BYTE_INDEX_0].item()
    output_lo = l3_out[0, pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    output_hi = l3_out[0, pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16]

    print(f"  H1[AX]: {h1_ax:.3f}")
    print(f"  BYTE_INDEX_0: {byte_idx_0:.3f}")
    print(f"  OUTPUT: lo={torch.argmax(output_lo).item()}, hi={torch.argmax(output_hi).item()}")
    output_val = torch.argmax(output_lo).item() + 16 * torch.argmax(output_hi).item()
    print(f"  OUTPUT value: {output_val}")
    print()

if l6_out is not None:
    print("After L6:")
    op_imm = l6_out[0, pos, BD.OP_IMM].item()
    mark_ax = l6_out[0, pos, BD.MARK_AX].item()
    output_lo = l6_out[0, pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    output_hi = l6_out[0, pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16]

    print(f"  OP_IMM: {op_imm:.3f}")
    print(f"  MARK_AX: {mark_ax:.3f}")
    print(f"  OUTPUT: lo={torch.argmax(output_lo).item()}, hi={torch.argmax(output_hi).item()}")
    output_val = torch.argmax(output_lo).item() + 16 * torch.argmax(output_hi).item()
    print(f"  OUTPUT value: {output_val}")
    print()

print("Analysis:")
print("  At REG_AX marker, BYTE_INDEX_0 should be ~0")
print("  But if BYTE_INDEX_0 ≈ 1, my AX byte logic fires and sets OUTPUT=0")
print("  This would override the IMM value")
