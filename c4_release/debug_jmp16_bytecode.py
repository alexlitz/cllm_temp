"""Debug JMP 16 - check bytecode and CLEAN_EMBED at address 1."""
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

print("JMP 16 - Bytecode and CLEAN_EMBED Check")
print("=" * 70)
print()

bytecode = [Opcode.JMP | (16 << 8)]
context = build_context(bytecode)

print("Context tokens:")
for i, tok in enumerate(context):
    if tok == Token.CODE_START:
        print(f"  {i}: CODE_START ({tok})")
    elif tok == Token.CODE_END:
        print(f"  {i}: CODE_END ({tok})")
    elif tok == Token.DATA_START:
        print(f"  {i}: DATA_START ({tok})")
    elif tok == Token.DATA_END:
        print(f"  {i}: DATA_END ({tok})")
    elif i >= 1 and i <= 5:  # Code section
        print(f"  {i}: byte {tok} = 0x{tok:02X}  ← Address {i-1}")
print()

draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

ctx = context + [draft_tokens[0]]

# Capture embedding output
embed_out = None
def embed_hook(module, input, output):
    global embed_out
    embed_out = output.detach().clone()

model.embed.register_forward_hook(embed_hook)

ctx_tensor = torch.tensor([ctx], dtype=torch.long)
with torch.no_grad():
    logits = model.forward(ctx_tensor)

print("Checking CLEAN_EMBED at address 1 (position 2 in context):")
if embed_out is not None:
    pos = 2  # Position 2 = address 1 (first immediate byte)
    token_val = context[pos]

    clean_embed_lo = embed_out[0, pos, BD.CLEAN_EMBED_LO:BD.CLEAN_EMBED_LO+16]
    clean_embed_hi = embed_out[0, pos, BD.CLEAN_EMBED_HI:BD.CLEAN_EMBED_HI+16]

    print(f"  Token at position {pos}: {token_val} (0x{token_val:02X})")
    print(f"  Expected nibbles: lo={token_val & 0xF}, hi={(token_val >> 4) & 0xF}")
    print()
    print(f"  CLEAN_EMBED_LO: {clean_embed_lo}")
    print(f"  CLEAN_EMBED_HI: {clean_embed_hi}")
    print()
    print(f"  CLEAN_EMBED_LO argmax: {torch.argmax(clean_embed_lo).item()}")
    print(f"  CLEAN_EMBED_HI argmax: {torch.argmax(clean_embed_hi).item()}")
    print()

    # Check if it's a clean one-hot
    clean_lo_max = torch.max(clean_embed_lo).item()
    clean_lo_second = torch.topk(clean_embed_lo, 2).values[1].item()
    clean_hi_max = torch.max(clean_embed_hi).item()
    clean_hi_second = torch.topk(clean_embed_hi, 2).values[1].item()

    print(f"  CLEAN_EMBED_LO max: {clean_lo_max:.3f}, 2nd: {clean_lo_second:.3f}")
    print(f"  CLEAN_EMBED_HI max: {clean_hi_max:.3f}, 2nd: {clean_hi_second:.3f}")
    print()

    if clean_lo_max > 0.99 and clean_lo_second < 0.01:
        print("  ✓ CLEAN_EMBED_LO is clean one-hot")
    else:
        print("  ✗ CLEAN_EMBED_LO is NOT clean (fuzzy encoding)")

    if clean_hi_max > 0.99 and clean_hi_second < 0.01:
        print("  ✓ CLEAN_EMBED_HI is clean one-hot")
    else:
        print("  ✗ CLEAN_EMBED_HI is NOT clean (fuzzy encoding)")

print()
print("Analysis:")
print("  For JMP 16, immediate value is 16 = 0x10")
print("  At address 1, bytecode should have byte 16")
print("  CLEAN_EMBED should have clean one-hot: lo[0]=1, hi[1]=1")
print("  If CLEAN_EMBED is fuzzy, that explains why AX_CARRY is fuzzy")
