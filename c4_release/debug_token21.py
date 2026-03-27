"""Debug token 21 (STACK0_b0) prediction for JMP."""
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

# Token positions:
# 0-4: REG_PC (marker + 4 bytes)
# 5-9: REG_AX (marker + 4 bytes)
# 10-14: REG_SP (marker + 4 bytes)
# 15-19: REG_BP (marker + 4 bytes)
# 20-24: STACK0 (marker + 4 bytes)
# Token 21 = STACK0_b0

print("Token predictions up to token 21:")
current_context = context[:]
for i in range(22):  # 0-21
    ctx_tensor = torch.tensor([current_context], dtype=torch.long)
    with torch.no_grad():
        logits = model.forward(ctx_tensor)
        predicted = logits[0, -1, :].argmax().item()

    match = "✓" if predicted == draft_tokens[i] else "✗"

    # Token names
    if i == 0:
        name = "REG_PC"
    elif 1 <= i <= 4:
        name = f"PC_b{i-1}"
    elif i == 5:
        name = "REG_AX"
    elif 6 <= i <= 9:
        name = f"AX_b{i-6}"
    elif i == 10:
        name = "REG_SP"
    elif 11 <= i <= 14:
        name = f"SP_b{i-11}"
    elif i == 15:
        name = "REG_BP"
    elif 16 <= i <= 19:
        name = f"BP_b{i-16}"
    elif i == 20:
        name = "STACK0"
    elif i == 21:
        name = "ST_b0"

    print(f"Token {i:2d} ({name:8s}): draft={draft_tokens[i]:3d}, predicted={predicted:3d} {match}")

    current_context.append(draft_tokens[i])

print(f"\nExpected ST_b0 = 0 (no stack operations in JMP)")
print(f"Actual prediction = 32")
print(f"\nNote: 32 = 0x20 = space character in ASCII")
