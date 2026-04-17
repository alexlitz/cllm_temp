"""Debug Layer 10 attention head 1 scores at PC marker."""
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

bytecode = [Opcode.JMP | (16 << 8)]
context = build_context(bytecode)

draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

current_context = context + [draft_tokens[0]]  # Just REG_PC marker
ctx_tensor = torch.tensor([current_context], dtype=torch.long)

with torch.no_grad():
    # Run through embedding and first 9 layers
    logits = model.forward(ctx_tensor)

    # Now manually recompute to get intermediate activations
    x = model.embed(ctx_tensor)

    pc_marker_pos = len(current_context) - 1

    # Run through Layer 9
    for layer_idx in range(10):
        x = model.blocks[layer_idx](x)

    # Manually run Layer 10 attention head 1 to inspect scores
    attn = model.blocks[10].attn
    B, S, D = x.shape
    H = attn.num_heads
    HD = attn.head_dim

    Q = F.linear(x, attn.W_q).view(B, S, H, HD).transpose(1, 2)
    K = F.linear(x, attn.W_k).view(B, S, H, HD).transpose(1, 2)

    # Head 1 (byte passthrough)
    head_idx = 1
    Q_h1 = Q[0, head_idx, :, :]  # [S, HD]
    K_h1 = K[0, head_idx, :, :]  # [S, HD]

    print(f"=== Layer 10 Head 1 (Byte Passthrough) ===")
    print(f"PC marker position: {pc_marker_pos}")
    print(f"\nQ vector at PC marker (dim 0-10 and 33):")
    for d in list(range(11)) + [33]:
        print(f"  Q[{d}] = {Q_h1[pc_marker_pos, d].item():.2f}")

    print(f"\nK vector at first position (dim 0-5 and 33):")
    for d in list(range(6)) + [33]:
        print(f"  K[0][{d}] = {K_h1[0, d].item():.2f}")

    # Compute scores for PC marker attending to all positions
    scores = torch.matmul(Q_h1[pc_marker_pos:pc_marker_pos+1, :], K_h1.transpose(-2, -1)) * attn.scale

    print(f"\nScores (PC marker attending to all positions):")
    print(f"  Raw scores: {scores[0, :].tolist()[:20]}")  # First 20 positions
    print(f"  Max score: {scores.max().item():.2f}")
    print(f"  Min score: {scores.min().item():.2f}")

    # Note: scores already include all biases from Q/K projections
    # ALiBi is added in the forward pass, but for this debug we just check raw scores

    # Compute attention weights (without additional masks)
    attn_weights = F.softmax(scores, dim=-1)

    print(f"\nAttention weights (top 5):")
    top_indices = torch.topk(attn_weights[0, :], k=5).indices
    for idx in top_indices:
        print(f"  Position {idx.item()}: weight = {attn_weights[0, idx].item():.6f}")

    print(f"\nExpected: All weights near zero (gate should suppress PC marker)")
