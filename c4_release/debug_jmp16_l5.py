"""Debug JMP 16 - check what L5 does to AX_CARRY."""
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

print("JMP 16 - AX_CARRY Through L5")
print("=" * 70)
print()

bytecode = [Opcode.JMP | (16 << 8)]
context = build_context(bytecode)

draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

ctx = context + [draft_tokens[0]]

# Capture L4, L5 attn, L5 outputs
l4_out = None
l5_attn_out = None
l5_out = None

def l4_hook(module, input, output):
    global l4_out
    l4_out = output.detach().clone()

def l5_attn_hook(module, input, output):
    global l5_attn_out
    l5_attn_out = output.detach().clone()

def l5_hook(module, input, output):
    global l5_out
    l5_out = output.detach().clone()

model.blocks[4].register_forward_hook(l4_hook)
model.blocks[5].attn.register_forward_hook(l5_attn_hook)
model.blocks[5].register_forward_hook(l5_hook)

ctx_tensor = torch.tensor([ctx], dtype=torch.long)
with torch.no_grad():
    logits = model.forward(ctx_tensor)

pos = len(ctx) - 1
print(f"Position {pos} (REG_PC marker)")
print()

if l4_out is not None:
    print("After L4:")
    ax_carry_lo = l4_out[0, pos, BD.AX_CARRY_LO:BD.AX_CARRY_LO+16]
    ax_carry_hi = l4_out[0, pos, BD.AX_CARRY_HI:BD.AX_CARRY_HI+16]
    print(f"  AX_CARRY_LO: max={torch.max(ax_carry_lo).item():.6f}, argmax={torch.argmax(ax_carry_lo).item()}")
    print(f"  AX_CARRY_HI: max={torch.max(ax_carry_hi).item():.6f}, argmax={torch.argmax(ax_carry_hi).item()}")
    print()

if l5_attn_out is not None:
    print("After L5 attention:")
    ax_carry_lo = l5_attn_out[0, pos, BD.AX_CARRY_LO:BD.AX_CARRY_LO+16]
    ax_carry_hi = l5_attn_out[0, pos, BD.AX_CARRY_HI:BD.AX_CARRY_HI+16]
    print(f"  AX_CARRY_LO: {ax_carry_lo}")
    print(f"  AX_CARRY_HI: {ax_carry_hi}")
    print(f"  AX_CARRY_LO argmax: {torch.argmax(ax_carry_lo).item()}")
    print(f"  AX_CARRY_HI argmax: {torch.argmax(ax_carry_hi).item()}")
    print()

    if l4_out is not None:
        delta_lo = ax_carry_lo - l4_out[0, pos, BD.AX_CARRY_LO:BD.AX_CARRY_LO+16]
        delta_hi = ax_carry_hi - l4_out[0, pos, BD.AX_CARRY_HI:BD.AX_CARRY_HI+16]
        print("  Delta (L5 attn contribution):")
        print(f"    AX_CARRY_LO: {delta_lo}")
        print(f"    AX_CARRY_HI: {delta_hi}")
        print()

if l5_out is not None:
    print("After L5 (after FFN):")
    ax_carry_lo = l5_out[0, pos, BD.AX_CARRY_LO:BD.AX_CARRY_LO+16]
    ax_carry_hi = l5_out[0, pos, BD.AX_CARRY_HI:BD.AX_CARRY_HI+16]
    print(f"  AX_CARRY_LO: {ax_carry_lo}")
    print(f"  AX_CARRY_HI: {ax_carry_hi}")
    print(f"  AX_CARRY_LO argmax: {torch.argmax(ax_carry_lo).item()}")
    print(f"  AX_CARRY_HI argmax: {torch.argmax(ax_carry_hi).item()}")
    print()

print("Analysis:")
print("  L5 attention or FFN must be setting AX_CARRY for first-step JMP")
print("  Expected: AX_CARRY_LO[0]=1, AX_CARRY_HI[1]=1")
print("  If still near-zero after L5: wrong layer is responsible")
