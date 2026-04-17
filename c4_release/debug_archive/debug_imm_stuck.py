"""Debug why IMM gets stuck predicting 257."""
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

print("IMM Stuck Loop Investigation")
print("=" * 70)
print()

# Test IMM 42
bytecode = [Opcode.IMM | (42 << 8)]
context = build_context(bytecode)

print("Context tokens:", context)
print(f"Token.REG_AX = {Token.REG_AX}")
print(f"Token.REG_PC = {Token.REG_PC}")
print()

draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

print("Expected draft tokens (first 10):")
for i in range(10):
    print(f"  {i}: {draft_tokens[i]}")
print()

# Test prediction for token 0 (should predict REG_AX = 257)
ctx_tensor = torch.tensor([context], dtype=torch.long)
with torch.no_grad():
    logits = model.forward(ctx_tensor)

predicted = torch.argmax(logits[0, -1]).item()
print(f"Token 0 prediction: {predicted} (expected {Token.REG_AX}={Token.REG_AX})")
print()

# Now test token 1 (should predict AX_b0)
# Context: original context + token 0
ctx_1 = context + [draft_tokens[0]]
print(f"Context for token 1: {ctx_1}")
print(f"Length: {len(ctx_1)}")
print()

ctx_tensor_1 = torch.tensor([ctx_1], dtype=torch.long)
with torch.no_grad():
    logits_1 = model.forward(ctx_tensor_1)

# Get top 5 predictions
top_5_vals, top_5_idx = torch.topk(logits_1[0, -1], 5)
print("Token 1 predictions (top 5):")
for i, (val, idx) in enumerate(zip(top_5_vals, top_5_idx)):
    print(f"  {i+1}. Token {idx.item():3d}: logit={val.item():8.2f}")
print()
print(f"Expected: {draft_tokens[1]} (AX_b0 = byte 42 & 0xF = 10 = 0xA)")
print()

# Check what's different about the sequence
print("Analyzing sequence structure:")
print(f"  Context length: {len(context)}")
print(f"  After token 0: {len(ctx_1)}")
print(f"  Last token in context: {context[-1]}")
print(f"  Token 0 (REG_AX): {draft_tokens[0]}")
print()

# Check if compaction is working
print(f"Model compacted: {hasattr(model, 'compacted')}")
print(f"Block size: {model.blocks[0].ffn.hidden_size if hasattr(model.blocks[0].ffn, 'hidden_size') else 'N/A'}")
