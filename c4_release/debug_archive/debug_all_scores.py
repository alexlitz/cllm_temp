"""Check all attention scores."""
import torch
import torch.nn.functional as F
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

Q = x[pc_byte0_pos, :] @ attn.W_q[base:base+64, :].T
K_all = x @ attn.W_k[base:base+64, :].T

scores = (Q @ K_all.T) / (d_k ** 0.5)

# ALiBi
alibi_slope = attn.alibi_slopes[head_idx].item()
positions = torch.arange(scores.shape[0])
alibi_bias = -alibi_slope * abs(positions - pc_byte0_pos).float()
scores = scores + alibi_bias

# Causal mask
mask = positions <= pc_byte0_pos
scores_masked = scores.clone()
scores_masked[~mask] = float('-inf')

# Softmax
attn_weights = F.softmax(scores_masked, dim=-1)

print("All scores at PC byte 0:")
print("=" * 70)

# Show top 10 scores (before masking)
top_scores, top_indices = torch.topk(scores, min(10, len(scores)))

print("\nTop 10 scores (before causal mask, before softmax):")
for i, (score, idx) in enumerate(zip(top_scores, top_indices)):
    idx_val = idx.item()
    score_val = score.item()
    tok = context_with_draft[idx_val]
    causal_ok = "✓" if idx_val <= pc_byte0_pos else "✗ masked"
    attn_wt = attn_weights[idx_val].item()
    print(f"  {i+1:2d}. pos {idx_val:3d} (tok={tok:3d}): score={score_val:8.2f} {causal_ok:10s} attn_wt={attn_wt:.6f}")

print()
print(f"Total attention weight (should be 1.0): {attn_weights.sum().item():.6f}")
