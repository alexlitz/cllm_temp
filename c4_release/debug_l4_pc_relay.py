"""Check if L4 relays PC correctly to AX marker."""
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

# Test JMP 12
bytecode = [Opcode.JMP | (12 << 8), Opcode.EXIT]
draft_vm = DraftVM(bytecode)
context = build_context(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

print("L4 PC Relay Check")
print("="*70)

context_with_draft = context + draft_tokens
print(f"Full sequence (context + draft):")
for i, tok in enumerate(context_with_draft[:25]):
    if tok == Token.REG_PC:
        print(f"  {i}: REG_PC")
    elif tok == Token.REG_AX:
        print(f"  {i}: REG_AX")
    elif tok < 256:
        print(f"  {i}: byte {tok}")
    else:
        print(f"  {i}: token {tok}")

ctx_len = len(context)
pc_marker_pos = ctx_len
pc_byte0_pos = ctx_len + 1
ax_marker_pos = ctx_len + 5

print(f"\nKey positions:")
print(f"  PC marker: {pc_marker_pos}")
print(f"  PC byte 0: {pc_byte0_pos} (token={draft_tokens[1]})")
print(f"  AX marker: {ax_marker_pos}")

# Load model
model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

# Hook L3 output (input to L4)
l3_output = {}
def capture_l3(module, input, output):
    l3_output['x'] = output.clone()
model.blocks[3].register_forward_hook(capture_l3)

# Hook L4 attention output
l4_attn_output = {}
def capture_l4_attn(module, input, output):
    l4_attn_output['x'] = output.clone()
model.blocks[4].attn.register_forward_hook(capture_l4_attn)

# Hook L4 FFN output (final L4 output)
l4_output = {}
def capture_l4(module, input, output):
    l4_output['x'] = output.clone()
model.blocks[4].register_forward_hook(capture_l4)

ctx_tensor = torch.tensor([context_with_draft], dtype=torch.long)

with torch.no_grad():
    _ = model.forward(ctx_tensor)

print("\n" + "="*70)
print("EMBED values at PC byte 0 position (after embedding)")
print("-"*70)
# Check the embedding layer output
x_embed = model.embed(ctx_tensor)
embed_lo_pc = x_embed[0, pc_byte0_pos, BD.EMBED_LO:BD.EMBED_LO+16]
embed_hi_pc = x_embed[0, pc_byte0_pos, BD.EMBED_HI:BD.EMBED_HI+16]

embed_lo_idx = (embed_lo_pc > 0.5).nonzero(as_tuple=True)[0]
embed_hi_idx = (embed_hi_pc > 0.5).nonzero(as_tuple=True)[0]

print(f"At position {pc_byte0_pos} (PC byte 0, token={draft_tokens[1]}):")
print(f"  EMBED_LO[{embed_lo_idx.tolist()}]=1.0")
print(f"  EMBED_HI[{embed_hi_idx.tolist()}]=1.0")

if len(embed_lo_idx) > 0:
    byte_val = embed_lo_idx[0].item() + (embed_hi_idx[0].item() << 4)
    print(f"  Byte value: {byte_val} {'✓' if byte_val == 12 else '✗'}")

print("\n" + "="*70)
print("After L4 Attention: EMBED at AX marker")
print("-"*70)

x_l4_attn = l4_attn_output['x']
embed_lo_ax = x_l4_attn[0, ax_marker_pos, BD.EMBED_LO:BD.EMBED_LO+16]
embed_hi_ax = x_l4_attn[0, ax_marker_pos, BD.EMBED_HI:BD.EMBED_HI+16]

embed_lo_ax_idx = (embed_lo_ax > 0.5).nonzero(as_tuple=True)[0]
embed_hi_ax_idx = (embed_hi_ax > 0.5).nonzero(as_tuple=True)[0]

print(f"At position {ax_marker_pos} (AX marker):")
print(f"  EMBED_LO[{embed_lo_ax_idx.tolist()}]=1.0")
print(f"  EMBED_HI[{embed_hi_ax_idx.tolist()}]=1.0")

if len(embed_lo_ax_idx) > 0:
    byte_val = embed_lo_ax_idx[0].item() + (embed_hi_ax_idx[0].item() << 4)
    print(f"  Relayed PC value: {byte_val} {'✓ (correct)' if byte_val == 12 else '✗ (wrong)'}")
else:
    print(f"  ✗ No EMBED values set (relay failed)")

print("\n" + "="*70)
print("After L4 FFN: TEMP (PC+1) at AX marker")
print("-"*70)

x_l4 = l4_output['x']
temp_lo = x_l4[0, ax_marker_pos, BD.TEMP:BD.TEMP+16]
temp_hi = x_l4[0, ax_marker_pos, BD.TEMP+16:BD.TEMP+32]

temp_lo_idx = (temp_lo > 0.5).nonzero(as_tuple=True)[0]
temp_hi_idx = (temp_hi > 0.5).nonzero(as_tuple=True)[0]

print(f"At position {ax_marker_pos} (AX marker):")
print(f"  TEMP lo[{temp_lo_idx.tolist()}], hi[{temp_hi_idx.tolist()}]")

if len(temp_lo_idx) > 0:
    pc_plus1_val = temp_lo_idx[0].item() + (temp_hi_idx[0].item() << 4) if len(temp_hi_idx) > 0 else temp_lo_idx[0].item()
    expected = 13 if byte_val == 12 else "unknown"
    print(f"  PC+1 value: {pc_plus1_val} (expected: {expected})")
    if pc_plus1_val == 13:
        print("  ✓ PC+1 computed correctly")
    else:
        print(f"  ✗ Wrong PC+1 value (got {pc_plus1_val}, expected 13)")
