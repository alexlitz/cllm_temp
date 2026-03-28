"""Debug LEA 8 - detailed L10 attention analysis."""
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

bytecode = [Opcode.LEA | (8 << 8)]
context = build_context(bytecode)

draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

# Check at AX marker
ctx = context + draft_tokens[:6]
pos = len(ctx) - 1

print("LEA 8 - L10 Detailed Analysis")
print("=" * 70)
print()

# Capture L10 attention input and output
l10_attn_in = None
l10_attn_out = None
l10_ffn_in = None

def l10_attn_hook(module, input, output):
    global l10_attn_in, l10_attn_out
    if isinstance(input, tuple):
        l10_attn_in = input[0].detach().clone()
    else:
        l10_attn_in = input.detach().clone()
    l10_attn_out = output.detach().clone()

def l10_ffn_hook(module, input, output):
    global l10_ffn_in
    if isinstance(input, tuple):
        l10_ffn_in = input[0].detach().clone()
    else:
        l10_ffn_in = input.detach().clone()

model.blocks[10].attn.register_forward_hook(l10_attn_hook)
model.blocks[10].ffn.register_forward_hook(l10_ffn_hook)

ctx_tensor = torch.tensor([ctx], dtype=torch.long)
with torch.no_grad():
    logits = model.forward(ctx_tensor)

print(f"Position {pos} (REG_AX marker, token {ctx[pos]}):")
print()

# Check L10 attention input
if l10_attn_in is not None:
    print("Before L10 attention:")
    output_lo = l10_attn_in[0, pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    output_hi = l10_attn_in[0, pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16]
    lo_val = torch.argmax(output_lo).item()
    hi_val = torch.argmax(output_hi).item()
    print(f"  OUTPUT: {lo_val} + 16*{hi_val} = {lo_val + 16*hi_val}")
    print()

# Check L10 attention output
if l10_attn_out is not None:
    print("After L10 attention:")
    output_lo = l10_attn_out[0, pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    output_hi = l10_attn_out[0, pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16]
    lo_val = torch.argmax(output_lo).item()
    hi_val = torch.argmax(output_hi).item()
    print(f"  OUTPUT: {lo_val} + 16*{hi_val} = {lo_val + 16*hi_val}")

    # Check what attention added
    delta_lo = l10_attn_out[0, pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16] - l10_attn_in[0, pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    delta_hi = l10_attn_out[0, pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16] - l10_attn_in[0, pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16]
    print(f"  Delta from attention: lo_max={torch.max(torch.abs(delta_lo)).item():.3f}, hi_max={torch.max(torch.abs(delta_hi)).item():.3f}")

    # Which dims changed?
    changed_dims_lo = torch.nonzero(torch.abs(delta_lo) > 0.01).squeeze()
    changed_dims_hi = torch.nonzero(torch.abs(delta_hi) > 0.01).squeeze()
    if len(changed_dims_lo) > 0:
        print(f"  OUTPUT_LO dims changed: {changed_dims_lo.tolist() if changed_dims_lo.dim() > 0 else [changed_dims_lo.item()]}")
    if len(changed_dims_hi) > 0:
        print(f"  OUTPUT_HI dims changed: {changed_dims_hi.tolist() if changed_dims_hi.dim() > 0 else [changed_dims_hi.item()]}")
    print()

# Manually compute attention to see which head is responsible
if l10_attn_in is not None:
    attn = model.blocks[10].attn
    H = attn.num_heads
    HD = attn.head_dim

    # Compute Q, K, V for each head
    Q = F.linear(l10_attn_in, attn.W_q).view(1, -1, H, HD).transpose(1, 2)
    K = F.linear(l10_attn_in, attn.W_k).view(1, -1, H, HD).transpose(1, 2)
    V = F.linear(l10_attn_in, attn.W_v).view(1, -1, H, HD).transpose(1, 2)

    # Compute scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) * attn.scale

    # Add ALiBi
    S = l10_attn_in.shape[1]
    q_positions = torch.arange(S, device=l10_attn_in.device)
    kv_positions = torch.arange(S, device=l10_attn_in.device)
    dist = (q_positions.unsqueeze(1) - kv_positions.unsqueeze(0)).abs().float()

    print("L10 Attention per head (at AX marker):")
    for h in range(H):
        alibi_slope = attn.alibi_slopes[h].item()
        alibi = -alibi_slope * dist
        head_scores = scores[0, h, pos, :] + alibi[pos, :]
        head_weights = F.softmax(head_scores, dim=-1)

        # Top 3 attended positions
        top3 = torch.topk(head_weights, 3)
        print(f"  Head {h} (slope={alibi_slope:.1f}):")
        for i in range(3):
            attn_pos = top3.indices[i].item()
            weight = top3.values[i].item()
            if weight > 0.01:
                print(f"    Pos {attn_pos}: weight={weight:.3f}")
    print()

# Check L10 FFN input
if l10_ffn_in is not None:
    print("Before L10 FFN:")
    mark_ax = l10_ffn_in[0, pos, BD.MARK_AX].item()
    op_lea = l10_ffn_in[0, pos, BD.OP_LEA].item()
    ax_carry_lo = l10_ffn_in[0, pos, BD.AX_CARRY_LO:BD.AX_CARRY_LO+16]
    ax_carry_hi = l10_ffn_in[0, pos, BD.AX_CARRY_HI:BD.AX_CARRY_HI+16]
    output_lo = l10_ffn_in[0, pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    output_hi = l10_ffn_in[0, pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16]

    print(f"  MARK_AX: {mark_ax:.3f}")
    print(f"  OP_LEA: {op_lea:.3f}")
    print(f"  AX_CARRY: {torch.argmax(ax_carry_lo).item()} + 16*{torch.argmax(ax_carry_hi).item()}")
    print(f"  OUTPUT: {torch.argmax(output_lo).item()} + 16*{torch.argmax(output_hi).item()}")

    # Check if AX passthrough should activate
    # up = S*(MARK_AX - OP_LEA) - S*0.5
    S = 20.0
    up_expected = S * (mark_ax - op_lea) - S * 0.5
    print(f"  AX passthrough 'up' = {up_expected:.1f} (should be < 0 to NOT activate)")
    print()
