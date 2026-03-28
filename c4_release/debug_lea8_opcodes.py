"""Debug LEA 8 - check if OP_LEA is set correctly."""
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

bytecode = [Opcode.LEA | (8 << 8)]
context = build_context(bytecode)

draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

# Check at AX marker
ctx = context + draft_tokens[:6]
pos = len(ctx) - 1

# Capture L6 and L10 outputs
l6_out = None
l10_out = None

def l6_hook(module, input, output):
    global l6_out
    l6_out = output.detach().clone()

def l10_hook(module, input, output):
    global l10_out
    l10_out = output.detach().clone()

model.blocks[6].register_forward_hook(l6_hook)
model.blocks[10].register_forward_hook(l10_hook)

ctx_tensor = torch.tensor([ctx], dtype=torch.long)
with torch.no_grad():
    logits = model.forward(ctx_tensor)
    pred = torch.argmax(logits[0, -1, :]).item()

print("LEA 8 Opcode Flags at AX Marker")
print("=" * 70)
print()

if l6_out is not None:
    print("After L6 (opcode decode):")
    mark_ax = l6_out[0, pos, BD.MARK_AX].item()
    op_lea = l6_out[0, pos, BD.OP_LEA].item()
    op_imm = l6_out[0, pos, BD.OP_IMM].item()
    ax_carry_lo = l6_out[0, pos, BD.AX_CARRY_LO:BD.AX_CARRY_LO+16]
    ax_carry_hi = l6_out[0, pos, BD.AX_CARRY_HI:BD.AX_CARRY_HI+16]

    print(f"  MARK_AX: {mark_ax:.3f}")
    print(f"  OP_LEA: {op_lea:.3f}")
    print(f"  OP_IMM: {op_imm:.3f}")
    print(f"  AX_CARRY: {torch.argmax(ax_carry_lo).item()} + 16*{torch.argmax(ax_carry_hi).item()} = {torch.argmax(ax_carry_lo).item() + 16*torch.argmax(ax_carry_hi).item()}")
    print()

if l10_out is not None:
    print("After L10:")
    mark_ax = l10_out[0, pos, BD.MARK_AX].item()
    op_lea = l10_out[0, pos, BD.OP_LEA].item()
    ax_carry_lo = l10_out[0, pos, BD.AX_CARRY_LO:BD.AX_CARRY_LO+16]
    ax_carry_hi = l10_out[0, pos, BD.AX_CARRY_HI:BD.AX_CARRY_HI+16]
    output_lo = l10_out[0, pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    output_hi = l10_out[0, pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16]

    print(f"  MARK_AX: {mark_ax:.3f}")
    print(f"  OP_LEA: {op_lea:.3f}")
    print(f"  AX_CARRY: {torch.argmax(ax_carry_lo).item()} + 16*{torch.argmax(ax_carry_hi).item()} = {torch.argmax(ax_carry_lo).item() + 16*torch.argmax(ax_carry_hi).item()}")
    print(f"  OUTPUT: {torch.argmax(output_lo).item()} + 16*{torch.argmax(output_hi).item()} = {torch.argmax(output_lo).item() + 16*torch.argmax(output_hi).item()}")
    print()

print(f"Final prediction: {pred} (expected 8)")
print()
print("Analysis:")
print("  AX passthrough should be suppressed when OP_LEA ≈ 5")
print("  Unit activation: up = S*(MARK_AX - OP_LEA) - S*0.5 = S*(1-5) - S*0.5 = -4.5*S < 0")
print("  Should NOT activate!")
