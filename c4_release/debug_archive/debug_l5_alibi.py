#!/usr/bin/env python3
"""Debug L5 attention with ALiBi penalty to understand why FETCH has wrong values."""

import torch
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token, _SetDim as BD
from neural_vm.embedding import Opcode
from neural_vm.kv_cache_eviction import softmax1
from src.compiler import compile_c

code = 'int main() { return 42; }'
bytecode, data = compile_c(code)
target_byte0 = (bytecode[0] >> 8) & 0xFF
print(f"Expected byte: 0x{target_byte0:02x} (lo={target_byte0 & 0xF}, hi={target_byte0 >> 4})")

runner = AutoregressiveVMRunner()
runner._func_call_handlers = {}
runner._bytecode = bytecode
runner._last_sp = 0x1F800
runner._last_bp = 0x10000
ctx = runner._build_context(bytecode, data, [])

runner.model.set_active_opcode(Opcode.JSR)
next_token = runner.model.generate_next(ctx)
ctx.append(next_token)

with torch.no_grad():
    device = next(runner.model.parameters()).device
    token_ids = torch.tensor([ctx], dtype=torch.long, device=device)

    # Pass through layers 0-4
    x = runner.model.embed(token_ids)
    for i in range(5):
        x = runner.model.blocks[i](x)

    # L5 attention
    L5_attn = runner.model.blocks[5].attn
    HD = 64
    head = 3
    base = head * HD
    H = 8  # number of heads

    # Get Q, K, V
    q = x @ L5_attn.W_q.T
    k = x @ L5_attn.W_k.T
    v = x @ L5_attn.W_v.T

    B, S, D = q.shape
    print(f"\nSequence length: {S}")
    print(f"Bytecode positions: 0-5")
    print(f"Query position: {S-1}")

    # Reshape for multi-head attention
    q = q.view(B, S, H, HD).transpose(1, 2)  # [B, H, S, HD]
    k = k.view(B, S, H, HD).transpose(1, 2)
    v = v.view(B, S, H, HD).transpose(1, 2)

    # Attention scores (before ALiBi)
    scale = HD ** -0.5
    scores_all = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, S, S]

    # Check if L5 uses ALiBi
    print(f"\n=== L5 Positional Encoding ===")
    print(f"_positional_encoding: {L5_attn._positional_encoding}")
    print(f"alibi_slopes: {L5_attn.alibi_slopes}")
    print(f"_rope_cos: {L5_attn._rope_cos is not None}")

    # Head 3 scores at last position
    scores_h3 = scores_all[0, head, -1, :]  # [S]

    print(f"\n=== Head 3 Scores (last pos → all) BEFORE ALiBi ===")
    top_scores, top_positions = torch.topk(scores_h3, min(10, len(scores_h3)))
    for i, (score, pos) in enumerate(zip(top_scores, top_positions)):
        print(f"  {i}: pos={pos.item()}, score={score.item():.2f}")
    print(f"Score at position 2 (bytecode): {scores_h3[2].item():.2f}")

    # Apply ALiBi if present
    if L5_attn.alibi_slopes is not None:
        q_positions = torch.arange(S, device=device).unsqueeze(1)  # [S, 1]
        k_positions = torch.arange(S, device=device).unsqueeze(0)  # [1, S]
        dist = (q_positions - k_positions).abs().float()  # [S, S]
        alibi = -L5_attn.alibi_slopes.view(1, H, 1, 1) * dist  # [1, H, S, S]

        print(f"\n=== ALiBi Penalty ===")
        print(f"Head 3 slope: {L5_attn.alibi_slopes[head].item():.6f}")
        distance_to_pos2 = abs(S - 1 - 2)
        penalty = -L5_attn.alibi_slopes[head].item() * distance_to_pos2
        print(f"Distance from pos {S-1} to pos 2: {distance_to_pos2}")
        print(f"ALiBi penalty for pos 2: {penalty:.2f}")

        scores_with_alibi = scores_all + alibi
        scores_h3_alibi = scores_with_alibi[0, head, -1, :]

        print(f"\n=== Head 3 Scores AFTER ALiBi ===")
        top_scores_alibi, top_positions_alibi = torch.topk(scores_h3_alibi, min(10, len(scores_h3_alibi)))
        for i, (score, pos) in enumerate(zip(top_scores_alibi, top_positions_alibi)):
            print(f"  {i}: pos={pos.item()}, score={score.item():.2f}")
        print(f"Score at position 2 (bytecode): {scores_h3_alibi[2].item():.2f}")

        # Apply causal mask
        causal_mask = torch.triu(torch.full((S, S), float('-inf'), device=device), diagonal=1)
        scores_masked = scores_with_alibi + causal_mask

        # Apply softmax1
        attn_weights = softmax1(scores_masked, dim=-1)

        h3_weights = attn_weights[0, head, -1, :]
        print(f"\n=== Head 3 Attention Weights (softmax1) ===")
        top_weights, top_pos = torch.topk(h3_weights, min(10, len(h3_weights)))
        for i, (w, pos) in enumerate(zip(top_weights, top_pos)):
            print(f"  {i}: pos={pos.item()}, weight={w.item():.6f}")
        print(f"Weight at position 2: {h3_weights[2].item():.6f}")
        print(f"Sum of all weights: {h3_weights.sum().item():.4f}")
    else:
        print("\nNo ALiBi - using regular scores")
        # Apply causal mask
        causal_mask = torch.triu(torch.full((S, S), float('-inf'), device=device), diagonal=1)
        scores_masked = scores_all + causal_mask
        attn_weights = softmax1(scores_masked, dim=-1)
        h3_weights = attn_weights[0, head, -1, :]

        print(f"\n=== Head 3 Attention Weights (softmax1, no ALiBi) ===")
        top_weights, top_pos = torch.topk(h3_weights, min(10, len(h3_weights)))
        for i, (w, pos) in enumerate(zip(top_weights, top_pos)):
            print(f"  {i}: pos={pos.item()}, weight={w.item():.6f}")

    # Check V values at top weighted positions
    print(f"\n=== V values at top positions ===")
    v_h3 = v[0, head, :, :]  # [S, HD]
    for pos_tensor in top_pos[:5]:
        pos = pos_tensor.item()
        v_pos = v_h3[pos, :]
        # Check nibbles 32-48 (FETCH_LO source) and 48-64 (FETCH_HI source)
        lo_vals = v_pos[32:48].tolist()
        hi_vals = v_pos[48:64].tolist()
        lo_max_idx = lo_vals.index(max(lo_vals)) if max(lo_vals) > 0.1 else -1
        hi_max_idx = hi_vals.index(max(hi_vals)) if max(hi_vals) > 0.1 else -1
        print(f"  Pos {pos}: V lo_nibble={lo_max_idx}, hi_nibble={hi_max_idx}")

    # Compute actual attention output for head 3
    attn_out_h3 = torch.matmul(h3_weights.unsqueeze(0), v_h3).squeeze(0)  # [HD]
    print(f"\n=== Head 3 Attention Output ===")
    print(f"V dims 32-40 (lo nibbles 0-7): {[attn_out_h3[32+k].item() for k in range(8)]}")
    print(f"V dims 48-56 (hi nibbles 0-7): {[attn_out_h3[48+k].item() for k in range(8)]}")

    # Now run actual L5 block and compare
    print(f"\n=== Actual L5 Block Output ===")
    x_out = runner.model.blocks[5](x)
    print(f"FETCH_LO: {[x_out[0, -1, BD.FETCH_LO + k].item() for k in range(8)]}")
    print(f"FETCH_HI: {[x_out[0, -1, BD.FETCH_HI + k].item() for k in range(8)]}")
