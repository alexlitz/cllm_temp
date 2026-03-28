"""Debug AX markers for tokens 5-7 in IMM 42."""
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

print("AX Marker Propagation for IMM 42")
print("=" * 70)
print()

bytecode = [Opcode.IMM | (42 << 8)]
context = build_context(bytecode)

draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

print("Expected sequence:")
print(f"  Token 5 (REG_AX):  {draft_tokens[5]} - AX marker")
print(f"  Token 6 (AX_b0):   {draft_tokens[6]} - AX byte 0 = 42")
print(f"  Token 7 (AX_b1):   {draft_tokens[7]} - AX byte 1 = 0")
print()

# Test tokens 5, 6, 7
for test_token_idx in [5, 6, 7]:
    ctx = context + draft_tokens[:test_token_idx]

    # Capture embedding and L6 FFN input
    embed_out = None
    l6_ffn_in = None

    def embed_hook(module, input, output):
        global embed_out
        embed_out = output.detach().clone()

    def l6_hook(module, input, output):
        global l6_ffn_in
        if isinstance(input, tuple):
            l6_ffn_in = input[0].detach().clone()
        else:
            l6_ffn_in = input.detach().clone()

    model.embed.register_forward_hook(embed_hook)
    model.blocks[6].ffn.register_forward_hook(l6_hook)

    ctx_tensor = torch.tensor([ctx], dtype=torch.long)
    with torch.no_grad():
        logits = model.forward(ctx_tensor)

    pos = len(ctx) - 1
    last_token = ctx[-1]

    print(f"Position {pos} (token {test_token_idx-1}, value={last_token}):")

    if embed_out is not None:
        mark_ax = embed_out[0, pos, BD.MARK_AX].item()
        is_byte = embed_out[0, pos, BD.IS_BYTE].item()
        byte_idx_0 = embed_out[0, pos, BD.BYTE_INDEX_0].item()
        print(f"  Embedding: MARK_AX={mark_ax:.3f}, IS_BYTE={is_byte:.3f}, BYTE_IDX[0]={byte_idx_0:.3f}")

    if l6_ffn_in is not None:
        mark_ax = l6_ffn_in[0, pos, BD.MARK_AX].item()
        byte_idx_0 = l6_ffn_in[0, pos, BD.BYTE_INDEX_0].item()
        print(f"  L6 FFN in: MARK_AX={mark_ax:.3f}, BYTE_IDX[0]={byte_idx_0:.3f}")

    predicted = torch.argmax(logits[0, -1]).item()
    expected = draft_tokens[test_token_idx]
    match = "✓" if predicted == expected else "✗"
    print(f"  Predicts: {predicted} (expected: {expected}) {match}")
    print()

print()
print("Analysis:")
print("  Token 5 (REG_AX = 258): Should have MARK_AX=1, IS_BYTE=0")
print("  Token 6 (AX_b0 = 42):   Should have MARK_AX=1, IS_BYTE=1, BYTE_INDEX_0=0")
print("  Token 7 (AX_b1 = 0):    Should have MARK_AX=1, IS_BYTE=1, BYTE_INDEX_0=1")
print()
print("  If MARK_AX is not propagating to byte positions, the relay won't work!")
