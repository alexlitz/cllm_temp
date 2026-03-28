"""Debug token 1 prediction for IMM."""
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

print("Token 1 Debug for IMM 42")
print("=" * 70)
print()

bytecode = [Opcode.IMM | (42 << 8)]
context = build_context(bytecode)

draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

print(f"Draft token 0: {draft_tokens[0]} (REG_PC)")
print(f"Draft token 1: {draft_tokens[1]} (should be PC_b0 or AX_b0?)")
print()

# IMM 42 = 0x2A
# In bytes: [42, 0, 0, 0]
# Nibbles: [0xA, 0x2, 0, 0, 0, 0, 0, 0]
print(f"IMM value: 42 = 0x{42:02X}")
print(f"Low nibble: {42 & 0xF} = 0x{42 & 0xF:X}")
print(f"High nibble: {(42 >> 4) & 0xF} = 0x{(42 >> 4) & 0xF:X}")
print()

# Build context for token 1
ctx_1 = context + [draft_tokens[0]]
print(f"Context for token 1: {ctx_1}")
print(f"Last 5 tokens: {ctx_1[-5:]}")
print()

# Capture Layer 15 output (before final head)
l15_out = None
def hook(module, input, output):
    global l15_out
    l15_out = output.detach().clone()

model.blocks[15].register_forward_hook(hook)

ctx_tensor = torch.tensor([ctx_1], dtype=torch.long)
with torch.no_grad():
    logits = model.forward(ctx_tensor)

# Check OUTPUT values at last position
if l15_out is not None:
    pos = len(ctx_1) - 1
    output_lo = l15_out[0, pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    output_hi = l15_out[0, pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16]

    print("OUTPUT values at last position (after L15):")
    print(f"  OUTPUT_LO argmax: {torch.argmax(output_lo).item()} (values: {output_lo.tolist()[:5]}...)")
    print(f"  OUTPUT_HI argmax: {torch.argmax(output_hi).item()} (values: {output_hi.tolist()[:5]}...)")
    print()

# Check logits
top_5_vals, top_5_idx = torch.topk(logits[0, -1], 10)
print("Top 10 predictions:")
for i, (val, idx) in enumerate(zip(top_5_vals, top_5_idx)):
    tok = idx.item()
    if tok == 257:
        name = "REG_PC"
    elif tok == 258:
        name = "REG_AX"
    elif tok < 256:
        name = f"byte_{tok}"
    else:
        name = f"token_{tok}"
    print(f"  {i+1:2d}. {name:15s} (tok={tok:3d}): logit={val.item():8.2f}")

print()
print(f"Expected: {draft_tokens[1]}")
print(f"Predicted: {torch.argmax(logits[0, -1]).item()}")
