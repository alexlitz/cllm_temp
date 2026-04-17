"""Debug EMBED values at AX byte positions for IMM 42."""
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

print("EMBED Values for IMM 42 at AX Byte Positions")
print("=" * 70)
print()

bytecode = [Opcode.IMM | (42 << 8)]
context = build_context(bytecode)

draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

print("Expected AX value: 42 = 0x2A")
print("Expected AX bytes: [42, 0, 0, 0]")
print()

# Test position 15 (after AX_b0 = 42)
ctx = context + draft_tokens[:7]  # Include tokens 0-6

# Capture L6 FFN input
l6_ffn_in = None
def l6_hook(module, input, output):
    global l6_ffn_in
    if isinstance(input, tuple):
        l6_ffn_in = input[0].detach().clone()
    else:
        l6_ffn_in = input.detach().clone()

model.blocks[6].ffn.register_forward_hook(l6_hook)

ctx_tensor = torch.tensor([ctx], dtype=torch.long)
with torch.no_grad():
    logits = model.forward(ctx_tensor)

pos = len(ctx) - 1  # Position 15
print(f"Position {pos} (last token = {ctx[-1]} = AX_b0):")
print()

if l6_ffn_in is not None:
    # Check EMBED values
    embed_lo = l6_ffn_in[0, pos, BD.EMBED_LO:BD.EMBED_LO+16]
    embed_hi = l6_ffn_in[0, pos, BD.EMBED_HI:BD.EMBED_HI+16]
    embed_lo_val = torch.argmax(embed_lo).item()
    embed_hi_val = torch.argmax(embed_hi).item()
    embed_byte = embed_lo_val + (embed_hi_val << 4)

    print(f"EMBED values at L6 FFN input:")
    print(f"  EMBED_LO argmax: {embed_lo_val} (0x{embed_lo_val:X})")
    print(f"  EMBED_HI argmax: {embed_hi_val} (0x{embed_hi_val:X})")
    print(f"  Decoded byte: {embed_byte} (0x{embed_byte:02X})")
    print()

    # Check if EMBED matches the token value
    token_val = ctx[-1]
    print(f"Token value: {token_val}")
    print(f"Match: {embed_byte == token_val}")
    print()

    # Check OUTPUT values
    output_lo = l6_ffn_in[0, pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    output_hi = l6_ffn_in[0, pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16]
    output_lo_val = torch.argmax(output_lo).item()
    output_hi_val = torch.argmax(output_hi).item()
    output_byte = output_lo_val + (output_hi_val << 4)

    print(f"OUTPUT values at L6 FFN input:")
    print(f"  OUTPUT_LO argmax: {output_lo_val} (0x{output_lo_val:X})")
    print(f"  OUTPUT_HI argmax: {output_hi_val} (0x{output_hi_val:X})")
    print(f"  Decoded byte: {output_byte} (0x{output_byte:02X})")
    print()

    print(f"Expected OUTPUT for next token: 0 (AX byte 1)")
    print(f"Actual OUTPUT: {output_byte}")
    print(f"Problem: OUTPUT has not been updated for byte relay!")
