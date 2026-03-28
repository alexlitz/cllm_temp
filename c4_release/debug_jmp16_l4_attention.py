"""Debug JMP 16 - trace AX_CARRY through L4."""
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

print("JMP 16 - AX_CARRY Through L4")
print("=" * 70)
print()

bytecode = [Opcode.JMP | (16 << 8)]
context = build_context(bytecode)

draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

ctx = context + [draft_tokens[0]]

# Capture L3 output (before L4) and L4 output (after L4)
l3_out = None
l4_attn_out = None
l4_out = None

def l3_hook(module, input, output):
    global l3_out
    l3_out = output.detach().clone()

def l4_attn_hook(module, input, output):
    global l4_attn_out
    l4_attn_out = output.detach().clone()

def l4_hook(module, input, output):
    global l4_out
    l4_out = output.detach().clone()

model.blocks[3].register_forward_hook(l3_hook)
model.blocks[4].attn.register_forward_hook(l4_attn_hook)
model.blocks[4].register_forward_hook(l4_hook)

ctx_tensor = torch.tensor([ctx], dtype=torch.long)
with torch.no_grad():
    logits = model.forward(ctx_tensor)

pos = len(ctx) - 1  # PC marker position
print(f"Position {pos} (REG_PC marker)")
print()

if l3_out is not None:
    print("After L3 (before L4):")
    ax_carry_lo = l3_out[0, pos, BD.AX_CARRY_LO:BD.AX_CARRY_LO+16]
    ax_carry_hi = l3_out[0, pos, BD.AX_CARRY_HI:BD.AX_CARRY_HI+16]
    print(f"  AX_CARRY_LO: {ax_carry_lo}")
    print(f"  AX_CARRY_HI: {ax_carry_hi}")
    print(f"  AX_CARRY_LO argmax: {torch.argmax(ax_carry_lo).item()}")
    print(f"  AX_CARRY_HI argmax: {torch.argmax(ax_carry_hi).item()}")
    print()

if l4_attn_out is not None:
    print("After L4 attention (before L4 FFN):")
    ax_carry_lo = l4_attn_out[0, pos, BD.AX_CARRY_LO:BD.AX_CARRY_LO+16]
    ax_carry_hi = l4_attn_out[0, pos, BD.AX_CARRY_HI:BD.AX_CARRY_HI+16]
    print(f"  AX_CARRY_LO: {ax_carry_lo}")
    print(f"  AX_CARRY_HI: {ax_carry_hi}")
    print(f"  AX_CARRY_LO argmax: {torch.argmax(ax_carry_lo).item()}")
    print(f"  AX_CARRY_HI argmax: {torch.argmax(ax_carry_hi).item()}")
    print()

    # Check if residual is adding to the fetched value
    if l3_out is not None:
        delta_lo = ax_carry_lo - l3_out[0, pos, BD.AX_CARRY_LO:BD.AX_CARRY_LO+16]
        delta_hi = ax_carry_hi - l3_out[0, pos, BD.AX_CARRY_HI:BD.AX_CARRY_HI+16]
        print("  Delta from L3 (attention contribution):")
        print(f"    AX_CARRY_LO delta: {delta_lo}")
        print(f"    AX_CARRY_HI delta: {delta_hi}")
        print()

if l4_out is not None:
    print("After L4 (after L4 FFN):")
    ax_carry_lo = l4_out[0, pos, BD.AX_CARRY_LO:BD.AX_CARRY_LO+16]
    ax_carry_hi = l4_out[0, pos, BD.AX_CARRY_HI:BD.AX_CARRY_HI+16]
    print(f"  AX_CARRY_LO: {ax_carry_lo}")
    print(f"  AX_CARRY_HI: {ax_carry_hi}")
    print(f"  AX_CARRY_LO argmax: {torch.argmax(ax_carry_lo).item()}")
    print(f"  AX_CARRY_HI argmax: {torch.argmax(ax_carry_hi).item()}")
    print()

print("Analysis:")
print("  L4 attention head 3 should fetch byte 16 from address 1")
print("  Expected: AX_CARRY_LO[0]=1, AX_CARRY_HI[1]=1 (clean)")
print("  If fuzzy after attention: softmax spreading to multiple positions")
print("  If fuzzy after FFN: FFN is contaminating the values")
