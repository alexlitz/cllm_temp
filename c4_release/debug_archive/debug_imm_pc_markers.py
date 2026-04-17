"""Debug PC markers for tokens 5-7 in IMM 42."""
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

print("PC Marker Propagation for IMM 42")
print("=" * 70)
print()

bytecode = [Opcode.IMM | (42 << 8)]
context = build_context(bytecode)

draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

print("Expected sequence:")
print(f"  Token 5 (REG_PC):  {draft_tokens[5]} - PC marker")
print(f"  Token 6 (PC_b0):   {draft_tokens[6]} - PC byte 0 = 42")
print(f"  Token 7 (PC_b1):   {draft_tokens[7]} - PC byte 1 = 0")
print()

# Test tokens 5, 6, 7
for test_token_idx in [5, 6, 7]:
    ctx = context + draft_tokens[:test_token_idx]

    # Capture embedding output
    embed_out = None
    def embed_hook(module, input, output):
        global embed_out
        embed_out = output.detach().clone()

    model.embed.register_forward_hook(embed_hook)

    ctx_tensor = torch.tensor([ctx], dtype=torch.long)
    with torch.no_grad():
        logits = model.forward(ctx_tensor)

    pos = len(ctx) - 1
    if embed_out is not None:
        mark_pc = embed_out[0, pos, BD.MARK_PC].item()
        is_byte = embed_out[0, pos, BD.IS_BYTE].item()
        byte_idx_0 = embed_out[0, pos, BD.BYTE_INDEX_0].item()

        last_token = ctx[-1]
        print(f"Position {pos} (token {test_token_idx-1}, value={last_token}):")
        print(f"  MARK_PC:      {mark_pc:.3f}")
        print(f"  IS_BYTE:      {is_byte:.3f}")
        print(f"  BYTE_INDEX_0: {byte_idx_0:.3f}")

        predicted = torch.argmax(logits[0, -1]).item()
        expected = draft_tokens[test_token_idx]
        match = "✓" if predicted == expected else "✗"
        print(f"  Predicts: {predicted} (expected: {expected}) {match}")
        print()

print()
print("Analysis:")
print("  Token 5 (REG_PC = 258): Should have MARK_PC=1, IS_BYTE=0")
print("  Token 6 (PC_b0 = 42):   Should have MARK_PC=1, IS_BYTE=1, BYTE_INDEX_0=0")
print("  Token 7 (PC_b1 = 0):    Should have MARK_PC=1, IS_BYTE=1, BYTE_INDEX_0=1")
print()
print("  If MARK_PC is not propagating to byte positions, the relay won't work!")
