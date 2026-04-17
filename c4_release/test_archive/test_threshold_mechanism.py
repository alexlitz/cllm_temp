#!/usr/bin/env python3
"""Deep dive into threshold attention mechanism - inspect attention patterns."""

from neural_vm.vm_step import AutoregressiveVM, Token, _SetDim as BD, set_vm_weights
import torch
import torch.nn.functional as F

# Create model
model = AutoregressiveVM(n_layers=17)
set_vm_weights(model)  # IMPORTANT: Configure weights!

# Create a simple context with just SP marker and bytes
context = [
    Token.REG_SP,  # Position 0: SP marker
    0xF8,          # Position 1: SP byte 0 (d=1 from SP)
    0xF7,          # Position 2: SP byte 1 (d=2 from SP)
    0x01,          # Position 3: SP byte 2 (d=3 from SP)
    0x00,          # Position 4: SP byte 3 (d=4 from SP)
]

input_ids = torch.tensor([context])  # [1, 5]
B, S = 1, 5

print("=" * 80)
print("Threshold Attention Mechanism Deep Dive")
print("=" * 80)

# Get embedding
x = model.embed(input_ids)
print(f"\nEmbedding shape: {x.shape}")
print(f"Context: SP marker + 4 byte tokens")

# Check IS_MARK values
print("\nIS_MARK values:")
for i in range(S):
    is_mark = x[0, i, BD.IS_MARK].item()
    mark_sp = x[0, i, BD.MARK_SP].item()
    tok = context[i]
    print(f"  pos {i} (token {tok:3d}): IS_MARK={is_mark:.2f}, MARK_SP={mark_sp:.2f}")

# Get L0 attention layer
attn0 = model.blocks[0].attn
print(f"\nL0 attention: {attn0.num_heads} heads, ALiBi slope={attn0.alibi_slopes[0].item():.1f}")

# Manually compute attention for head 1 (L1H1, threshold=1.5)
head_idx = 1
threshold = 1.5
slope = attn0.alibi_slopes[head_idx].item()
HD = attn0.head_dim

print(f"\n{'=' * 80}")
print(f"Analyzing Head {head_idx} (threshold={threshold}, slope={slope})")
print(f"{'=' * 80}")

# Compute Q, K, V
Q = F.linear(x, attn0.W_q).view(B, S, attn0.num_heads, HD).transpose(1, 2)  # [B, H, S, HD]
K = F.linear(x, attn0.W_k).view(B, S, attn0.num_heads, HD).transpose(1, 2)
V = F.linear(x, attn0.W_v).view(B, S, attn0.num_heads, HD).transpose(1, 2)

# Extract head 1
Q_h = Q[0, head_idx, :, :]  # [S, HD]
K_h = K[0, head_idx, :, :]  # [S, HD]
V_h = V[0, head_idx, :, :]  # [S, HD]

print("\nQ values (head dimension 0):")
for i in range(S):
    tok = context[i]
    q_val = Q_h[i, 0].item()
    print(f"  pos {i} (token {tok:3d}): Q[0]={q_val:8.2f}")

print("\nK values (head dimension 0):")
for i in range(S):
    tok = context[i]
    k_val = K_h[i, 0].item()
    is_mark = x[0, i, BD.IS_MARK].item()
    print(f"  pos {i} (token {tok:3d}): K[0]={k_val:8.2f}, IS_MARK={is_mark:.2f}")

# Compute attention scores
scores = torch.matmul(Q_h, K_h.transpose(0, 1)) * (HD ** -0.5)  # [S, S]

# Add ALiBi bias
q_positions = torch.arange(S).unsqueeze(1)  # [S, 1]
k_positions = torch.arange(S).unsqueeze(0)  # [1, S]
dist = (q_positions - k_positions).abs().float()  # [S, S]
alibi = -slope * dist  # [S, S]

scores_with_alibi = scores + alibi

# Add causal mask
causal_mask = torch.triu(torch.full((S, S), float("-inf")), diagonal=1)
scores_masked = scores_with_alibi + causal_mask

print("\nAttention scores for position 1 (SP byte 0, d=1):")
print(f"{'Pos':<5} {'Token':<6} {'Q·K/√d':<10} {'ALiBi':<10} {'Total':<10} {'After mask':<12}")
print("-" * 70)
for j in range(S):
    qk_score = scores[1, j].item()
    alibi_score = alibi[1, j].item()
    total_score = scores_with_alibi[1, j].item()
    masked_score = scores_masked[1, j].item()
    tok = context[j]
    print(f"{j:<5} {tok:<6} {qk_score:10.2f} {alibi_score:10.2f} {total_score:10.2f} {masked_score:12.2f}")

# Compute softmax
from neural_vm.vm_step import softmax1
attn_weights = softmax1(scores_masked, dim=-1)  # [S, S]

print("\nAttention weights after softmax (position 1):")
for j in range(S):
    weight = attn_weights[1, j].item()
    tok = context[j]
    marker = "  ← attending here" if weight > 0.1 else ""
    print(f"  pos {j} (token {tok:3d}): {weight:.4f}{marker}")

# Apply attention to V
out = torch.matmul(attn_weights, V_h)  # [S, HD]

print("\nOutput values at each position:")
print(f"{'Pos':<5} {'Token':<6} {'Output[0]':<12} {'Output[1]':<12}")
print("-" * 40)
for i in range(S):
    tok = context[i]
    out0 = out[i, 0].item()
    out1 = out[i, 1].item() if HD > 1 else 0
    print(f"{i:<5} {tok:<6} {out0:12.4f} {out1:12.4f}")

# Check V matrix configuration
print(f"\nV matrix configuration (dims 1-7 should copy marker flags):")
SP_I = 2
for dim_offset in range(7):
    v_weight = attn0.W_v[head_idx * HD + 1 + dim_offset, BD.MARK_SP].item()
    print(f"  V[{head_idx * HD + 1 + dim_offset}, MARK_SP] = {v_weight:.2f}")

print("\n" + "=" * 80)
print("Analysis:")
print("- Q should be constant (80.0 with slope=10)")
print("- K should be non-zero only at markers (threshold * IS_MARK)")
print("- ALiBi adds -slope * distance")
print("- Softmax normalizes across all positions (markers + non-markers)")
print("- This causes dilution - output is weighted average, not binary")
print("=" * 80)
