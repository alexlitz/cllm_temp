"""Check raw attention scores before softmax."""
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

bytecode = [Opcode.IMM | (5 << 8), Opcode.EXIT]
draft_vm = DraftVM(bytecode)
context = build_context(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

l9_out = {}
def hook(m, i, o):
    l9_out['x'] = o.clone()
model.blocks[9].register_forward_hook(hook)

context_with_draft = context + draft_tokens
ctx_tensor = torch.tensor([context_with_draft], dtype=torch.long)
ctx_len = len(context)
pc_byte0_pos = ctx_len + 1

with torch.no_grad():
    _ = model.forward(ctx_tensor)

x = l9_out['x'][0, :, :]
attn = model.blocks[10].attn
head_idx = 1
base = 64 * head_idx
d_k = 64

# Compute Q and K
Q = x[pc_byte0_pos, :] @ attn.W_q[base:base+64, :].T
K_all = x @ attn.W_k[base:base+64, :].T

# Raw scores before scaling
raw_scores = Q @ K_all.T
scaled_scores = raw_scores / (d_k ** 0.5)

# Check position 2
print("Attention score breakdown for position 2:")
print("=" * 70)
print(f"Query: PC byte 0 (pos {pc_byte0_pos})")
print(f"Key: context position 2 (immediate byte, value=5)")
print()

# Check Q and K values
Q_vals = Q
K2_vals = K_all[2, :]

print("Q and K dimensions that might contribute:")
print("-" * 70)

# Check key dimensions
dims_to_check = [0, 1, 2, 3, 4, 5, 33]
for dim in dims_to_check:
    q_val = Q_vals[dim].item()
    k_val = K2_vals[dim].item()
    contrib = q_val * k_val
    print(f"  dim {dim:2d}: Q={q_val:8.2f}, K={k_val:8.2f}, contrib={contrib:10.2f}")

print()
print(f"Raw score (sum of all 64 dims): {raw_scores[2].item():.2f}")
print(f"Scaled score (/ sqrt({d_k})): {scaled_scores[2].item():.2f}")

# ALiBi bias
alibi_slope = attn.alibi_slopes[head_idx].item()
dist = abs(2 - pc_byte0_pos)
alibi_bias = -alibi_slope * dist
print(f"ALiBi bias (-{alibi_slope} * {dist}): {alibi_bias:.2f}")
print()
print(f"Final score (before softmax): {(scaled_scores[2] + alibi_bias).item():.2f}")
