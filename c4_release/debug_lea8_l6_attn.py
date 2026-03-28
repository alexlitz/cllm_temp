"""Debug LEA 8 - check L6 attention vs FFN contributions."""
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

print("LEA 8 - L6 Attention vs FFN")
print("=" * 70)
print()

# Capture L6 attention input, output, and FFN output
l6_attn_in = None
l6_attn_out = None
l6_ffn_out = None

def l6_attn_hook(module, input, output):
    global l6_attn_in, l6_attn_out
    if isinstance(input, tuple):
        l6_attn_in = input[0].detach().clone()
    else:
        l6_attn_in = input.detach().clone()
    l6_attn_out = output.detach().clone()

def l6_ffn_hook(module, input, output):
    global l6_ffn_out
    l6_ffn_out = output.detach().clone()

model.blocks[6].attn.register_forward_hook(l6_attn_hook)
model.blocks[6].ffn.register_forward_hook(l6_ffn_hook)

ctx_tensor = torch.tensor([ctx], dtype=torch.long)
with torch.no_grad():
    logits = model.forward(ctx_tensor)

print(f"Position {pos} (REG_AX marker):")
print()

# Check OUTPUT at each stage
if l6_attn_in is not None:
    x = l6_attn_in[0, pos, :]
    output_lo = x[BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    output_hi = x[BD.OUTPUT_HI:BD.OUTPUT_HI+16]
    lo = torch.argmax(output_lo).item()
    hi = torch.argmax(output_hi).item()
    print(f"Before L6 Attention: OUTPUT = {lo + 16*hi} (lo={lo}, hi={hi})")

if l6_attn_out is not None:
    x = l6_attn_out[0, pos, :]
    output_lo = x[BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    output_hi = x[BD.OUTPUT_HI:BD.OUTPUT_HI+16]
    lo = torch.argmax(output_lo).item()
    hi = torch.argmax(output_hi).item()
    print(f"After L6 Attention:  OUTPUT = {lo + 16*hi} (lo={lo}, hi={hi})")

    # Check delta from attention
    if l6_attn_in is not None:
        delta_lo = x[BD.OUTPUT_LO:BD.OUTPUT_LO+16] - l6_attn_in[0, pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
        delta_hi = x[BD.OUTPUT_HI:BD.OUTPUT_HI+16] - l6_attn_in[0, pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16]
        max_delta = max(torch.max(torch.abs(delta_lo)).item(), torch.max(torch.abs(delta_hi)).item())
        if max_delta > 0.01:
            print(f"  → Attention changed OUTPUT (max delta={max_delta:.3f})")
        else:
            print(f"  → Attention did NOT change OUTPUT")

if l6_ffn_out is not None:
    x = l6_ffn_out[0, pos, :]
    output_lo = x[BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    output_hi = x[BD.OUTPUT_HI:BD.OUTPUT_HI+16]
    lo = torch.argmax(output_lo).item()
    hi = torch.argmax(output_hi).item()
    print(f"After L6 FFN:        OUTPUT = {lo + 16*hi} (lo={lo}, hi={hi})")

    # Check delta from FFN
    if l6_attn_out is not None:
        delta_lo = x[BD.OUTPUT_LO:BD.OUTPUT_LO+16] - l6_attn_out[0, pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
        delta_hi = x[BD.OUTPUT_HI:BD.OUTPUT_HI+16] - l6_attn_out[0, pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16]
        max_delta = max(torch.max(torch.abs(delta_lo)).item(), torch.max(torch.abs(delta_hi)).item())
        if max_delta > 0.01:
            print(f"  → FFN changed OUTPUT (max delta={max_delta:.3f})")
            print(f"  → Delta OUTPUT_LO: {delta_lo.tolist()}")
            print(f"  → Delta OUTPUT_HI: {delta_hi.tolist()}")
        else:
            print(f"  → FFN did NOT change OUTPUT")
