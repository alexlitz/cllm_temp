"""Debug JMP 16 - check CMP[0] and OUTPUT at token 1."""
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

print("JMP 16 Token 1 Debug - CMP[0] Analysis")
print("=" * 70)
print()

bytecode = [Opcode.JMP | (16 << 8)]
context = build_context(bytecode)

draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

print(f"Expected token 1 (PC_b0): {draft_tokens[1]}")
print()

# Test token 1 (PC_b0)
ctx = context + [draft_tokens[0]]

# Capture at multiple layers
embed_out = None
l6_ffn_in = None
l6_ffn_out = None
l15_out = None

def embed_hook(module, input, output):
    global embed_out
    embed_out = output.detach().clone()

def l6_ffn_in_hook(module, input, output):
    global l6_ffn_in
    if isinstance(input, tuple):
        l6_ffn_in = input[0].detach().clone()
    else:
        l6_ffn_in = input.detach().clone()

def l6_ffn_out_hook(module, input, output):
    global l6_ffn_out
    l6_ffn_out = output.detach().clone()

def l15_hook(module, input, output):
    global l15_out
    l15_out = output.detach().clone()

model.embed.register_forward_hook(embed_hook)
model.blocks[6].ffn.register_forward_hook(l6_ffn_in_hook)
model.blocks[6].register_forward_hook(l6_ffn_out_hook)
model.blocks[15].register_forward_hook(l15_hook)

ctx_tensor = torch.tensor([ctx], dtype=torch.long)
with torch.no_grad():
    logits = model.forward(ctx_tensor)

predicted = torch.argmax(logits[0, -1]).item()
pos = len(ctx) - 1

print(f"Position {pos} (last token = {ctx[-1]} = REG_PC):")
print()

# Check CMP[0] at different layers
if embed_out is not None:
    cmp0_embed = embed_out[0, pos, BD.CMP + 0].item()
    print(f"Embedding CMP[0]: {cmp0_embed:.3f}")

if l6_ffn_in is not None:
    cmp0_l6_in = l6_ffn_in[0, pos, BD.CMP + 0].item()
    mark_pc = l6_ffn_in[0, pos, BD.MARK_PC].item()
    mark_sp = l6_ffn_in[0, pos, BD.MARK_SP].item()
    mark_stack0 = l6_ffn_in[0, pos, BD.MARK_STACK0].item()

    print(f"L6 FFN input CMP[0]: {cmp0_l6_in:.3f}")
    print(f"L6 FFN input MARK_PC: {mark_pc:.3f}")
    print(f"L6 FFN input MARK_SP: {mark_sp:.3f}")
    print(f"L6 FFN input MARK_STACK0: {mark_stack0:.3f}")
    print()

    # Check OUTPUT before L6 FFN
    output_lo_in = l6_ffn_in[0, pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    output_hi_in = l6_ffn_in[0, pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16]
    output_lo_val = torch.argmax(output_lo_in).item()
    output_hi_val = torch.argmax(output_hi_in).item()
    output_byte_in = output_lo_val + (output_hi_val << 4)

    print(f"OUTPUT before L6 FFN: {output_byte_in} (lo={output_lo_val}, hi={output_hi_val})")

if l6_ffn_out is not None:
    # Check OUTPUT after L6 FFN
    output_lo_out = l6_ffn_out[0, pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    output_hi_out = l6_ffn_out[0, pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16]
    output_lo_val = torch.argmax(output_lo_out).item()
    output_hi_val = torch.argmax(output_hi_out).item()
    output_byte_out = output_lo_val + (output_hi_val << 4)

    print(f"OUTPUT after L6 FFN: {output_byte_out} (lo={output_lo_val}, hi={output_hi_val})")
    print()

if l15_out is not None:
    # Check OUTPUT after L15
    output_lo_final = l15_out[0, pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    output_hi_final = l15_out[0, pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16]
    output_lo_val = torch.argmax(output_lo_final).item()
    output_hi_val = torch.argmax(output_hi_final).item()
    output_byte_final = output_lo_val + (output_hi_val << 4)

    print(f"OUTPUT after L15: {output_byte_final} (lo={output_lo_val}, hi={output_hi_val})")
    print()
    print(f"OUTPUT_LO top 5: {output_lo_final.topk(5)}")
    print(f"OUTPUT_HI top 5: {output_hi_final.topk(5)}")
    print()

print(f"Predicted: {predicted}")
print(f"Expected:  {draft_tokens[1]}")
match = "✓" if predicted == draft_tokens[1] else "✗"
print(f"Match: {match}")
print()

print("Analysis:")
print("  JMP 16 = 0x10: lo nibble = 0, hi nibble = 1")
print("  Expected OUTPUT: lo=0, hi=1 → byte value 16")
print()
print("  If CMP[0] is high at PC marker, PSH units might be interfering.")
print("  PSH units check: CMP[0] + MARK_SP/MARK_STACK0 > threshold")
print("  With negative weight, high CMP[0] would suppress PSH units.")
