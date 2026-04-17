"""Debug PC byte prediction issue."""
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token
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

print("JMP 12 Debug")
print("="*70)
print(f"Initial PC: {draft_vm.pc}")

draft_vm.step()
print(f"After step PC: {draft_vm.pc}")

draft_tokens = draft_vm.draft_tokens()
print(f"\nDraftVM tokens (first 10): {draft_tokens[:10]}")
print(f"  Token 0 (marker): {draft_tokens[0]} (REG_PC={Token.REG_PC})")
print(f"  Token 1 (PC byte 0): {draft_tokens[1]}")
print(f"  Token 2 (PC byte 1): {draft_tokens[2]}")
print(f"  Token 3 (PC byte 2): {draft_tokens[3]}")
print(f"  Token 4 (PC byte 3): {draft_tokens[4]}")

# Show PC as 4-byte little-endian
pc = draft_vm.pc
pc_bytes = [(pc >> (8*i)) & 0xFF for i in range(4)]
print(f"\nPC = {pc} = 0x{pc:08x}")
print(f"PC bytes (LE): {pc_bytes}")

# Now test neural prediction
model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

context_with_draft = context + draft_tokens
ctx_tensor = torch.tensor([context_with_draft], dtype=torch.long)

with torch.no_grad():
    logits = model.forward(ctx_tensor)

ctx_len = len(context)
print(f"\nNeural predictions:")
for i in range(5):
    pos = ctx_len + i
    expected = draft_tokens[i]
    predicted = logits[0, pos - 1, :].argmax(-1).item()
    match = "✓" if expected == predicted else "✗"
    print(f"  Position {i}: expected {expected:3d}, predicted {predicted:3d} {match}")

# Check logits for PC byte 0
print(f"\nLogits for PC byte 0 (position {ctx_len}):")
logits_byte0 = logits[0, ctx_len, :]
top5_vals, top5_idx = logits_byte0.topk(5)
for val, idx in zip(top5_vals, top5_idx):
    marker = " ← expected" if idx.item() == draft_tokens[1] else ""
    print(f"  {idx.item():3d}: {val.item():8.2f}{marker}")
