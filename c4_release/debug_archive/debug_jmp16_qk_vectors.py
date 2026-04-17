"""Debug JMP 16 - Check Q and K vectors for L5 head 3."""
import sys
for mod in list(sys.modules.keys()):
    if 'neural_vm' in mod:
        del sys.modules[mod]

import torch
import torch.nn.functional as F
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

print("JMP 16 - L5 Head 3 Q/K Vectors")
print("=" * 70)
print()

bytecode = [Opcode.JMP | (16 << 8)]
context = build_context(bytecode)

draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

ctx = context + [draft_tokens[0]]

# Get L5 input
l5_in = None
def l5_in_hook(module, input, output):
    global l5_in
    if isinstance(input, tuple):
        l5_in = input[0].detach().clone()
    else:
        l5_in = input.detach().clone()

model.blocks[5].attn.register_forward_hook(l5_in_hook)

ctx_tensor = torch.tensor([ctx], dtype=torch.long)
with torch.no_grad():
    logits = model.forward(ctx_tensor)

# Get attention parameters
attn = model.blocks[5].attn
HD = attn.head_dim
head_idx = 3
base = head_idx * HD

# Get W_q, W_k for head 3
W_q = attn.W_q[base:base+HD, :]
W_k = attn.W_k[base:base+HD, :]

# Compute Q and K
Q = F.linear(l5_in, W_q)  # [1, S, HD]
K = F.linear(l5_in, W_k)  # [1, S, HD]

pc_pos = 9
print(f"Q vector at PC marker (position {pc_pos}):")
q_vec = Q[0, pc_pos, :]
print(f"  Shape: {q_vec.shape}")
print(f"  Norm: {torch.norm(q_vec).item():.2f}")
print(f"  Max value: {torch.max(q_vec).item():.2f}")
print(f"  Min value: {torch.min(q_vec).item():.2f}")
print(f"  Non-zero dims: {torch.sum(torch.abs(q_vec) > 0.01).item()}/{HD}")
print()

# Show first few dims
print("  First 40 dims:")
for i in range(0, 40, 4):
    vals = [q_vec[j].item() for j in range(i, min(i+4, 40))]
    print(f"    [{i:2d}-{i+3:2d}]: {vals}")
print()

print(f"K vector at position 1 (address 0, opcode):")
k1_vec = K[0, 1, :]
print(f"  Non-zero dims: {torch.sum(torch.abs(k1_vec) > 0.01).item()}/{HD}")
print(f"  Norm: {torch.norm(k1_vec).item():.2f}")
print()

print(f"K vector at position 2 (address 1, immediate):")
k2_vec = K[0, 2, :]
print(f"  Non-zero dims: {torch.sum(torch.abs(k2_vec) > 0.01).item()}/{HD}")
print(f"  Norm: {torch.norm(k2_vec).item():.2f}")
print()

# Compute dot products
dot1 = torch.dot(q_vec, k1_vec).item()
dot2 = torch.dot(q_vec, k2_vec).item()

print(f"Dot products:")
print(f"  Q · K1 (position 1): {dot1:.2f}")
print(f"  Q · K2 (position 2): {dot2:.2f}")
print(f"  Difference: {dot2 - dot1:.2f}")
print()

# Check if there's residual contamination
print("Checking for residual contamination:")
print(f"  Q vector has large values in unexpected dims?")
expected_dims = [1, 16, 32, 33]  # dims that should be set by design
for i in range(HD):
    if i not in expected_dims and abs(q_vec[i].item()) > 1.0:
        print(f"    ⚠️  Dim {i}: {q_vec[i].item():.2f} (unexpected!)")
