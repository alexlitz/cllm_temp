"""Debug LEA 8 - check L9 attention vs FFN for OUTPUT_HI."""
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

print("LEA 8 - L9 Attention vs FFN for OUTPUT_HI")
print("=" * 70)
print()

# Capture L9 attention input, output, and FFN output
l9_attn_in = None
l9_attn_out = None
l9_ffn_out = None

def l9_attn_hook(module, input, output):
    global l9_attn_in, l9_attn_out
    if isinstance(input, tuple):
        l9_attn_in = input[0].detach().clone()
    else:
        l9_attn_in = input.detach().clone()
    l9_attn_out = output.detach().clone()

def l9_ffn_hook(module, input, output):
    global l9_ffn_out
    l9_ffn_out = output.detach().clone()

model.blocks[9].attn.register_forward_hook(l9_attn_hook)
model.blocks[9].ffn.register_forward_hook(l9_ffn_hook)

ctx_tensor = torch.tensor([ctx], dtype=torch.long)
with torch.no_grad():
    logits = model.forward(ctx_tensor)

print(f"Position {pos} (REG_AX marker):")
print()

# Check OUTPUT_HI at each stage
if l9_attn_in is not None:
    x = l9_attn_in[0, pos, :]
    output_hi = x[BD.OUTPUT_HI:BD.OUTPUT_HI+16]
    mean_val = output_hi.mean().item()
    std_val = output_hi.std().item()
    max_val = output_hi.max().item()
    print(f"Before L9 Attention: OUTPUT_HI")
    print(f"  Mean: {mean_val:.1f}, Std: {std_val:.3f}, Max: {max_val:.1f}")
    if std_val < 0.1:
        print(f"  ⚠ All dims equal!")
    print()

if l9_attn_out is not None:
    x = l9_attn_out[0, pos, :]
    output_hi = x[BD.OUTPUT_HI:BD.OUTPUT_HI+16]
    mean_val = output_hi.mean().item()
    std_val = output_hi.std().item()
    max_val = output_hi.max().item()
    print(f"After L9 Attention: OUTPUT_HI")
    print(f"  Mean: {mean_val:.1f}, Std: {std_val:.3f}, Max: {max_val:.1f}")
    if std_val < 0.1:
        print(f"  ⚠ All dims equal!")

    # Check delta from attention
    if l9_attn_in is not None:
        delta_hi = output_hi - l9_attn_in[0, pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16]
        max_delta = torch.max(torch.abs(delta_hi)).item()
        if max_delta > 100:
            print(f"  ❌ Attention changed OUTPUT_HI by {max_delta:.1f}!")
            print(f"  Delta: {delta_hi.tolist()}")
        elif max_delta > 0.01:
            print(f"  → Attention changed OUTPUT_HI (max delta={max_delta:.3f})")
        else:
            print(f"  → Attention did NOT change OUTPUT_HI")
    print()

if l9_ffn_out is not None:
    x = l9_ffn_out[0, pos, :]
    output_hi = x[BD.OUTPUT_HI:BD.OUTPUT_HI+16]
    mean_val = output_hi.mean().item()
    std_val = output_hi.std().item()
    max_val = output_hi.max().item()
    print(f"After L9 FFN: OUTPUT_HI")
    print(f"  Mean: {mean_val:.1f}, Std: {std_val:.3f}, Max: {max_val:.1f}")
    if std_val < 0.1:
        print(f"  ❌ All dims equal! This is the corruption!")

    # Check delta from FFN
    if l9_attn_out is not None:
        delta_hi = output_hi - l9_attn_out[0, pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16]
        max_delta = torch.max(torch.abs(delta_hi)).item()
        if max_delta > 100:
            print(f"  ❌ FFN changed OUTPUT_HI by {max_delta:.1f}!")
            print(f"  Delta: {delta_hi.tolist()}")
        elif max_delta > 0.01:
            print(f"  → FFN changed OUTPUT_HI (max delta={max_delta:.3f})")
        else:
            print(f"  → FFN did NOT change OUTPUT_HI")
