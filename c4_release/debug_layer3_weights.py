"""Check Layer 3 attention weights to see if AX carry is configured."""

from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, _SetDim
import torch
import sys

BD = _SetDim

print("=" * 70, file=sys.stderr)
print("LAYER 3 ATTENTION WEIGHT INSPECTION", file=sys.stderr)
print("=" * 70, file=sys.stderr)

# Create model and set weights
model = AutoregressiveVM()
set_vm_weights(model)

# Get Layer 3 attention
attn3 = model.blocks[3].attn

print("Layer 3 Attention Configuration:", file=sys.stderr)
print(f"  W_q shape: {attn3.W_q.shape}", file=sys.stderr)
print(f"  W_k shape: {attn3.W_k.shape}", file=sys.stderr)
print(f"  W_v shape: {attn3.W_v.shape}", file=sys.stderr)
print(f"  W_o shape: {attn3.W_o.shape}", file=sys.stderr)
print("", file=sys.stderr)

num_heads = attn3.W_q.shape[0]
d_model = attn3.W_o.shape[0]
print(f"  Inferred d_model: {d_model}", file=sys.stderr)
print(f"  Inferred num_heads: {num_heads}", file=sys.stderr)
print("", file=sys.stderr)

# Check if there are any non-zero weights in W_q, W_k, W_v, W_o
# that involve MARK_AX or AX_CARRY dimensions

print("Checking for AX-related attention patterns:", file=sys.stderr)
print("", file=sys.stderr)

# W_q: (num_heads, d_model, head_dim)
# W_k: (num_heads, d_model, head_dim)
# W_v: (num_heads, d_model, head_dim)
# W_o: (d_model, num_heads * head_dim)

# Check W_q for queries on MARK_AX
print("W_q (Query) weights involving MARK_AX:", file=sys.stderr)
for head in range(num_heads):
    q_weights = attn3.W_q.data[head, BD.MARK_AX, :].abs()
    if q_weights.sum() > 0.01:
        print(f"  Head {head}: sum={q_weights.sum().item():.3f}, max={q_weights.max().item():.3f}", file=sys.stderr)

if all(attn3.W_q.data[head, BD.MARK_AX, :].abs().sum() < 0.01 for head in range(num_heads)):
    print("  (No significant W_q weights on MARK_AX)", file=sys.stderr)

print("", file=sys.stderr)

# Check W_k for keys on MARK_AX
print("W_k (Key) weights involving MARK_AX:", file=sys.stderr)
for head in range(num_heads):
    k_weights = attn3.W_k.data[head, BD.MARK_AX, :].abs()
    if k_weights.sum() > 0.01:
        print(f"  Head {head}: sum={k_weights.sum().item():.3f}, max={k_weights.max().item():.3f}", file=sys.stderr)

if all(attn3.W_k.data[head, BD.MARK_AX, :].abs().sum() < 0.01 for head in range(num_heads)):
    print("  (No significant W_k weights on MARK_AX)", file=sys.stderr)

print("", file=sys.stderr)

# Check W_v for values involving AX dimensions
print("W_v (Value) weights reading from AX dimensions (EMBED_LO, EMBED_HI):", file=sys.stderr)
for head in range(num_heads):
    v_weights_lo = attn3.W_v.data[head, BD.EMBED_LO:BD.EMBED_LO+16, :].abs().sum()
    v_weights_hi = attn3.W_v.data[head, BD.EMBED_HI:BD.EMBED_HI+16, :].abs().sum()
    if v_weights_lo > 0.01 or v_weights_hi > 0.01:
        print(f"  Head {head}: EMBED_LO sum={v_weights_lo.item():.3f}, EMBED_HI sum={v_weights_hi.item():.3f}", file=sys.stderr)

if all(attn3.W_v.data[head, BD.EMBED_LO:BD.EMBED_LO+16, :].abs().sum() < 0.01 and
       attn3.W_v.data[head, BD.EMBED_HI:BD.EMBED_HI+16, :].abs().sum() < 0.01
       for head in range(num_heads)):
    print("  (No significant W_v weights reading from AX value dimensions)", file=sys.stderr)

print("", file=sys.stderr)

# Check W_o for outputs to AX_CARRY dimensions
print("W_o (Output) weights writing to AX_CARRY_LO:", file=sys.stderr)
for i in range(16):
    dim = BD.AX_CARRY_LO + i
    w_o_weights = attn3.W_o.data[dim, :].abs()
    if w_o_weights.sum() > 0.01:
        print(f"  AX_CARRY_LO[{i}]: sum={w_o_weights.sum().item():.3f}, max={w_o_weights.max().item():.3f}", file=sys.stderr)

if all(attn3.W_o.data[BD.AX_CARRY_LO + i, :].abs().sum() < 0.01 for i in range(16)):
    print("  ⚠ WARNING: No W_o weights writing to AX_CARRY_LO!", file=sys.stderr)
    print("  This means Layer 3 cannot output to AX_CARRY_LO dimensions", file=sys.stderr)

print("", file=sys.stderr)

# Similarly check AX_CARRY_HI
print("W_o (Output) weights writing to AX_CARRY_HI:", file=sys.stderr)
for i in range(16):
    dim = BD.AX_CARRY_HI + i
    w_o_weights = attn3.W_o.data[dim, :].abs()
    if w_o_weights.sum() > 0.01:
        print(f"  AX_CARRY_HI[{i}]: sum={w_o_weights.sum().item():.3f}, max={w_o_weights.max().item():.3f}", file=sys.stderr)

if all(attn3.W_o.data[BD.AX_CARRY_HI + i, :].abs().sum() < 0.01 for i in range(16)):
    print("  ⚠ WARNING: No W_o weights writing to AX_CARRY_HI!", file=sys.stderr)

print("", file=sys.stderr)
print("=" * 70, file=sys.stderr)
print("SUMMARY", file=sys.stderr)
print("=" * 70, file=sys.stderr)

# Check the function that sets Layer 3 weights
print("Checking vm_step.py for Layer 3 weight initialization...", file=sys.stderr)
print("", file=sys.stderr)

# Count total non-zero weights
total_weights = attn3.W_q.numel() + attn3.W_k.numel() + attn3.W_v.numel() + attn3.W_o.numel()
nonzero_weights = (attn3.W_q.data.abs() > 1e-6).sum() + \
                  (attn3.W_k.data.abs() > 1e-6).sum() + \
                  (attn3.W_v.data.abs() > 1e-6).sum() + \
                  (attn3.W_o.data.abs() > 1e-6).sum()

print(f"Total Layer 3 attention weights: {total_weights}", file=sys.stderr)
print(f"Non-zero weights: {nonzero_weights} ({100.0 * nonzero_weights / total_weights:.2f}%)", file=sys.stderr)

if nonzero_weights == 0:
    print("", file=sys.stderr)
    print("❌ CRITICAL: Layer 3 attention has NO non-zero weights!", file=sys.stderr)
    print("This means Layer 3 is not configured at all.", file=sys.stderr)
    print("It's just passing through the residual connection.", file=sys.stderr)
elif all(attn3.W_o.data[BD.AX_CARRY_LO + i, :].abs().sum() < 0.01 for i in range(16)):
    print("", file=sys.stderr)
    print("❌ ISSUE: Layer 3 has weights but doesn't output to AX_CARRY_LO", file=sys.stderr)
    print("The AX carry mechanism is not implemented in the weights.", file=sys.stderr)
else:
    print("", file=sys.stderr)
    print("✓ Layer 3 has weights that could potentially write to AX_CARRY", file=sys.stderr)

print("=" * 70, file=sys.stderr)
