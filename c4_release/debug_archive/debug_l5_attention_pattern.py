"""Check L5 attention patterns to see what it's fetching from."""
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

print("L5 Attention Pattern Analysis")
print("="*70)
print("Context tokens:", context)
print()
print("Token positions:")
for i, tok in enumerate(context):
    if tok == Token.CODE_START:
        print(f"  {i}: CODE_START")
    elif tok == Token.CODE_END:
        print(f"  {i}: CODE_END")
    elif tok == Token.DATA_START:
        print(f"  {i}: DATA_START")
    elif tok == Token.DATA_END:
        print(f"  {i}: DATA_END")
    elif tok < 256:
        print(f"  {i}: byte {tok}")
    else:
        print(f"  {i}: token {tok}")

print()
print("Expected: L5 head 0 at AX marker should attend to position 2 (byte 12)")
print()

# Load model with attention hooks
model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

# Hook to capture L5 attention weights
l5_attn_weights = {}
def capture_l5_attn_weights(module, input, output):
    # Attention module forward returns (output, attn_weights) in some implementations
    # But PureAttention doesn't return attn_weights by default
    # We need to hook into the attention computation
    pass

# Actually, let's compute attention manually by checking the inputs to L5
# We'll look at the hidden state after L4, which is the input to L5

l4_output = {}
def capture_l4_output(module, input, output):
    l4_output['x'] = output.clone()
model.blocks[4].register_forward_hook(capture_l4_output)

l5_attn_output = {}
def capture_l5_attn(module, input, output):
    l5_attn_output['x'] = output.clone()
model.blocks[5].attn.register_forward_hook(capture_l5_attn)

context_with_draft = context + draft_tokens
ctx_tensor = torch.tensor([context_with_draft], dtype=torch.long)

with torch.no_grad():
    logits = model.forward(ctx_tensor)

ctx_len = len(context)
ax_marker_pos = ctx_len + 5

print("="*70)
print("Checking inputs and outputs at AX marker")
print("-"*70)

# Check what's at the code byte positions after L4
x_l4 = l4_output['x']
print(f"\nAfter L4 (input to L5 attention):")
for i in range(len(context)):
    tok = context[i]
    if tok < 256 and tok != Token.CODE_END:
        clean_embed_lo = x_l4[0, i, BD.CLEAN_EMBED_LO:BD.CLEAN_EMBED_LO+16]
        clean_embed_hi = x_l4[0, i, BD.CLEAN_EMBED_HI:BD.CLEAN_EMBED_HI+16]
        addr_key_lo = x_l4[0, i, BD.ADDR_KEY:BD.ADDR_KEY+16]
        addr_key_mid = x_l4[0, i, BD.ADDR_KEY+16:BD.ADDR_KEY+32]

        # Find which indices are set
        clean_lo_idx = (clean_embed_lo > 0.5).nonzero(as_tuple=True)[0]
        clean_hi_idx = (clean_embed_hi > 0.5).nonzero(as_tuple=True)[0]
        addr_lo_idx = (addr_key_lo > 0.5).nonzero(as_tuple=True)[0]
        addr_mid_idx = (addr_key_mid > 0.5).nonzero(as_tuple=True)[0]

        if len(clean_lo_idx) > 0:
            print(f"  Pos {i} (byte {tok}):")
            print(f"    CLEAN_EMBED_LO[{clean_lo_idx.tolist()}]=1.0")
            print(f"    CLEAN_EMBED_HI[{clean_hi_idx.tolist()}]=1.0")
            print(f"    ADDR_KEY lo[{addr_lo_idx.tolist()}], mid[{addr_mid_idx.tolist()}]")

# Check what L5 wrote to AX marker
x_l5 = l5_attn_output['x']
print(f"\n" + "="*70)
print(f"After L5 attention at AX marker (pos {ax_marker_pos}):")
print("-"*70)

fetch_lo = x_l5[0, ax_marker_pos, BD.FETCH_LO:BD.FETCH_LO+16]
fetch_hi = x_l5[0, ax_marker_pos, BD.FETCH_HI:BD.FETCH_HI+16]

fetch_lo_idx = (fetch_lo > 0.5).nonzero(as_tuple=True)[0]
fetch_hi_idx = (fetch_hi > 0.5).nonzero(as_tuple=True)[0]

print(f"FETCH_LO[{fetch_lo_idx.tolist()}]=1.0")
print(f"FETCH_HI[{fetch_hi_idx.tolist()}]=1.0")

if len(fetch_lo_idx) > 0:
    fetched_lo = fetch_lo_idx[0].item()
    fetched_hi = fetch_hi_idx[0].item() if len(fetch_hi_idx) > 0 else 0
    fetched_byte = fetched_lo + (fetched_hi << 4)
    print(f"\nFetched byte value: {fetched_byte} (expected: 12)")

    if fetched_byte == 0:
        print("✗ Fetched byte 0 instead of 12!")
        print("  Likely attending to wrong position or position has wrong embedding")
    elif fetched_byte == 12:
        print("✓ Correct byte fetched!")
    else:
        print(f"✗ Unexpected byte value!")

# Check TEMP at AX marker (should have PC+1 address)
temp = x_l4[0, ax_marker_pos, BD.TEMP:BD.TEMP+32]
temp_lo_idx = (temp[:16] > 0.5).nonzero(as_tuple=True)[0]
temp_mid_idx = (temp[16:32] > 0.5).nonzero(as_tuple=True)[0]

print(f"\nTEMP at AX marker (PC+1 address for fetch):")
print(f"  TEMP lo[{temp_lo_idx.tolist()}], mid[{temp_mid_idx.tolist()}]")
if len(temp_lo_idx) > 0:
    addr_lo = temp_lo_idx[0].item()
    addr_mid = temp_mid_idx[0].item() if len(temp_mid_idx) > 0 else 0
    addr = addr_lo + (addr_mid << 4)
    print(f"  Address value: {addr} (expected: 1 for PC+1 when PC=0)")
