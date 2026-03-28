"""Debug L15 nibble copy for IMM."""
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

print("L15 Nibble Copy for IMM")
print("=" * 70)
print()

bytecode = [Opcode.IMM | (42 << 8)]
context = build_context(bytecode)

draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

# Build context up to AX byte 0
ctx = context + draft_tokens[:7]  # Include AX byte 0
pos = len(ctx) - 1

# Capture L14, L15 FFN input and output
l14_out = None
l15_ffn_in = None
l15_out = None

def l14_hook(module, input, output):
    global l14_out
    l14_out = output.detach().clone()

def l15_ffn_hook(module, input, output):
    global l15_ffn_in
    if isinstance(input, tuple):
        l15_ffn_in = input[0].detach().clone()
    else:
        l15_ffn_in = input.detach().clone()

def l15_hook(module, input, output):
    global l15_out
    l15_out = output.detach().clone()

model.blocks[14].register_forward_hook(l14_hook)
model.blocks[15].ffn.register_forward_hook(l15_ffn_hook)
model.blocks[15].register_forward_hook(l15_hook)

ctx_tensor = torch.tensor([ctx], dtype=torch.long)
with torch.no_grad():
    logits = model.forward(ctx_tensor)
    pred = torch.argmax(logits[0, -1, :]).item()

print(f"Position {pos} (token {ctx[pos]} = AX byte 0 = 42):")
print()

# Check L14 output
if l14_out is not None:
    print("After L14:")
    is_byte = l14_out[0, pos, BD.IS_BYTE].item()
    h1_ax = l14_out[0, pos, BD.H1 + 1].item()
    embed_lo = l14_out[0, pos, BD.EMBED_LO:BD.EMBED_LO+16]
    embed_hi = l14_out[0, pos, BD.EMBED_HI:BD.EMBED_HI+16]
    output_lo = l14_out[0, pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    output_hi = l14_out[0, pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16]
    print(f"  IS_BYTE: {is_byte:.3f}")
    print(f"  H1[AX]: {h1_ax:.3f}")
    print(f"  EMBED: lo={torch.argmax(embed_lo).item()}, hi={torch.argmax(embed_hi).item()}")
    print(f"  OUTPUT: lo={torch.argmax(output_lo).item()}, hi={torch.argmax(output_hi).item()}")
    embed_val = torch.argmax(embed_lo).item() + 16 * torch.argmax(embed_hi).item()
    output_val = torch.argmax(output_lo).item() + 16 * torch.argmax(output_hi).item()
    print(f"  EMBED value: {embed_val}")
    print(f"  OUTPUT value: {output_val}")
    print()

# Check L15 FFN input
if l15_ffn_in is not None:
    print("L15 FFN input (after L15 attention):")
    is_byte = l15_ffn_in[0, pos, BD.IS_BYTE].item()
    h1_pc = l15_ffn_in[0, pos, BD.H1 + 0].item()
    h1_sp = l15_ffn_in[0, pos, BD.H1 + 2].item()
    h1_bp = l15_ffn_in[0, pos, BD.H1 + 3].item()
    embed_lo = l15_ffn_in[0, pos, BD.EMBED_LO:BD.EMBED_LO+16]
    embed_hi = l15_ffn_in[0, pos, BD.EMBED_HI:BD.EMBED_HI+16]
    output_lo = l15_ffn_in[0, pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    output_hi = l15_ffn_in[0, pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16]

    print(f"  IS_BYTE: {is_byte:.3f}")
    print(f"  H1[PC]: {h1_pc:.3f}")
    print(f"  H1[SP]: {h1_sp:.3f}")
    print(f"  H1[BP]: {h1_bp:.3f}")
    print(f"  EMBED_LO[10]: {embed_lo[10].item():.3f}")
    print(f"  EMBED_HI[2]: {embed_hi[2].item():.3f}")

    # Check up calculation
    S = 20.0
    up_expected = S * (is_byte - h1_pc - h1_sp - h1_bp) - S * 0.5
    print(f"  Expected 'up' for nibble copy: {up_expected:.3f}")
    print()

# Check L15 output
if l15_out is not None:
    print("After L15:")
    output_lo = l15_out[0, pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    output_hi = l15_out[0, pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16]
    print(f"  OUTPUT: lo={torch.argmax(output_lo).item()}, hi={torch.argmax(output_hi).item()}")
    output_val = torch.argmax(output_lo).item() + 16 * torch.argmax(output_hi).item()
    print(f"  OUTPUT value: {output_val}")
    print()

print(f"Final prediction: {pred} (expected {draft_tokens[7]} = 0)")
