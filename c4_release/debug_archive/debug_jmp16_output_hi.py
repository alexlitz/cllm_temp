"""Debug JMP 16 - check OUTPUT_HI values in detail."""
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

print("JMP 16 - OUTPUT_HI Detail")
print("=" * 70)
print()

bytecode = [Opcode.JMP | (16 << 8)]
context = build_context(bytecode)

draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

ctx = context + [draft_tokens[0]]

# Capture L6 input/output
l6_in = None
l6_out = None

def l6_in_hook(module, input, output):
    global l6_in
    if isinstance(input, tuple):
        l6_in = input[0].detach().clone()
    else:
        l6_in = input.detach().clone()

def l6_out_hook(module, input, output):
    global l6_out
    l6_out = output.detach().clone()

model.blocks[6].register_forward_hook(l6_in_hook)
model.blocks[6].register_forward_hook(l6_out_hook)

ctx_tensor = torch.tensor([ctx], dtype=torch.long)
with torch.no_grad():
    logits = model.forward(ctx_tensor)

pos = len(ctx) - 1
print(f"Position {pos} (last token = {ctx[-1]} = REG_PC)")
print()

if l6_in is not None:
    print("L6 Input - AX_CARRY_HI values:")
    ax_carry_hi = l6_in[0, pos, BD.AX_CARRY_HI:BD.AX_CARRY_HI+16]
    print(f"  Full tensor: {ax_carry_hi}")
    print(f"  Argmax: {torch.argmax(ax_carry_hi).item()}")
    print(f"  Expected: index 1 should be high (hi nibble of 16)")
    print()

    print("L6 Input - OUTPUT_HI before:")
    output_hi_in = l6_in[0, pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16]
    print(f"  Full tensor: {output_hi_in}")
    print(f"  Argmax: {torch.argmax(output_hi_in).item()}")
    print()

if l6_out is not None:
    print("L6 Output - OUTPUT_HI after:")
    output_hi_out = l6_out[0, pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16]
    print(f"  Full tensor: {output_hi_out}")
    print(f"  Argmax: {torch.argmax(output_hi_out).item()}")
    print(f"  Expected: index 1 should be high")
    print()

    # Check the change
    if l6_in is not None:
        delta = output_hi_out - output_hi_in
        print("OUTPUT_HI delta (L6 out - L6 in):")
        print(f"  Full tensor: {delta}")
        print(f"  Index 0 change: {delta[0].item():.3f}")
        print(f"  Index 1 change: {delta[1].item():.3f}")
        print()

print("Analysis:")
print("  AX_CARRY_HI[1] should be 1.0 (hi nibble of 16)")
print("  First-step JMP units should add AX_CARRY_HI to OUTPUT_HI")
print("  OUTPUT_HI[1] should increase, making hi nibble = 1")
print("  But OUTPUT_HI[0] stays highest instead of OUTPUT_HI[1]")
