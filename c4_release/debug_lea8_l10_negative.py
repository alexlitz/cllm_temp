"""Debug LEA 8 - check if L10 writes negative OUTPUT values."""
import sys
for mod in list(sys.modules.keys()):
    if 'neural_vm' in mod:
        del sys.modules[mod]

import torch
import torch.nn.functional as F
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

print("LEA 8 - L10 OUTPUT Contributions")
print("=" * 70)
print()

# Capture L10 FFN input
l10_ffn_in = None
l10_ffn_out = None

def l10_ffn_hook(module, input, output):
    global l10_ffn_in, l10_ffn_out
    if isinstance(input, tuple):
        l10_ffn_in = input[0].detach().clone()
    else:
        l10_ffn_in = input.detach().clone()
    l10_ffn_out = output.detach().clone()

model.blocks[10].ffn.register_forward_hook(l10_ffn_hook)

ctx_tensor = torch.tensor([ctx], dtype=torch.long)
with torch.no_grad():
    logits = model.forward(ctx_tensor)

if l10_ffn_in is not None and l10_ffn_out is not None:
    ffn = model.blocks[10].ffn
    x_in = l10_ffn_in[0, pos, :]
    x_out = l10_ffn_out[0, pos, :]

    print(f"Position {pos} (REG_AX marker):")
    print()

    # Check OUTPUT before and after L10 FFN
    output_lo_in = x_in[BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    output_hi_in = x_in[BD.OUTPUT_HI:BD.OUTPUT_HI+16]
    output_lo_out = x_out[BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    output_hi_out = x_out[BD.OUTPUT_HI:BD.OUTPUT_HI+16]

    print("OUTPUT before L10 FFN:")
    print(f"  LO: {output_lo_in.tolist()}")
    print(f"  HI: {output_hi_in.tolist()}")
    print()

    print("OUTPUT after L10 FFN:")
    print(f"  LO: {output_lo_out.tolist()}")
    print(f"  HI: {output_hi_out.tolist()}")
    print()

    # Check delta
    delta_lo = output_lo_out - output_lo_in
    delta_hi = output_hi_out - output_hi_in

    print("Delta (FFN contribution):")
    print(f"  LO: {delta_lo.tolist()}")
    print(f"  HI: {delta_hi.tolist()}")
    print()

    # Find units with strong negative output
    up = F.linear(x_in.unsqueeze(0), ffn.W_up, ffn.b_up)[0]
    gate = F.linear(x_in.unsqueeze(0), ffn.W_gate, ffn.b_gate)[0]
    hidden = F.silu(up) * gate

    W_down = ffn.W_down

    # Check OUTPUT_LO[8] (should be strong, but being reduced)
    units_lo8 = torch.where(torch.abs(W_down[BD.OUTPUT_LO + 8, :]) > 0.01)[0]
    print(f"Units affecting OUTPUT_LO[8]: {len(units_lo8)}")
    for unit_idx in units_lo8:
        weight = W_down[BD.OUTPUT_LO + 8, unit_idx].item()
        h = hidden[unit_idx].item()
        contrib = h * weight
        if abs(contrib) > 0.1:
            print(f"  Unit {unit_idx}: hidden={h:.3f}, weight={weight:.3f}, contrib={contrib:.3f}")
