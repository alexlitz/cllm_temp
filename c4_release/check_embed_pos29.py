"""Check EMBED values at position 29."""
import sys
if 'neural_vm.vm_step' in sys.modules:
    del sys.modules['neural_vm.vm_step']

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

bytecode = [Opcode.JMP | (16 << 8)]
context = build_context(bytecode)

draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

current_context = context[:]
for i in range(21):
    current_context.append(draft_tokens[i])

ctx_tensor = torch.tensor([current_context], dtype=torch.long)
pos = 29

# Hook to capture Layer 6 FFN input
l6_ffn_in = None
def hook(module, input, output):
    global l6_ffn_in
    l6_ffn_in = input[0].detach().clone()

model.blocks[6].ffn.register_forward_hook(hook)

with torch.no_grad():
    _ = model.forward(ctx_tensor)

print("EMBED values at position 29 (input to L6 FFN)")
print("=" * 70)
print()

if l6_ffn_in is not None:
    embed_lo = l6_ffn_in[0, pos, BD.EMBED_LO:BD.EMBED_LO+16]
    embed_hi = l6_ffn_in[0, pos, BD.EMBED_HI:BD.EMBED_HI+16]

    print("EMBED_LO:")
    print(f"  argmax: {torch.argmax(embed_lo).item()}")
    print(f"  values: {embed_lo.tolist()}")
    print()

    print("EMBED_HI:")
    print(f"  argmax: {torch.argmax(embed_hi).item()}")
    print(f"  values: {embed_hi.tolist()}")
    print()

    print("Expected for token 0 (ST_b0):")
    print("  EMBED_LO argmax should be 0 (low nibble of 0x00)")
    print("  EMBED_HI argmax should be 0 (high nibble of 0x00)")
    print()

    if torch.argmax(embed_hi).item() == 15:
        print("❌ EMBED_HI argmax is 15, not 0!")
        print("   This suggests EMBED is corrupted or relayed incorrectly")
