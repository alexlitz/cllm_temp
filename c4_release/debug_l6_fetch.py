#!/usr/bin/env python3
"""Debug L6 attention FETCH_LO[1] spurious contribution in Step 3."""

import torch
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, _SetDim as BD, Token
from neural_vm.speculative import DraftVM
from neural_vm.embedding import Opcode
from neural_vm.constants import IMMEDIATE_SIZE, PADDING_SIZE

# Test program: IMM 1, PSH, IMM 0, MUL, EXIT
BYTECODE = [Opcode.IMM | (1 << 8), Opcode.PSH, Opcode.IMM | (0 << 8), Opcode.MUL, Opcode.EXIT]

def softmax1(x, dim=-1, anchor=0.0):
    """softmax1 for ZFOD."""
    max_val = torch.max(x.max(dim=dim, keepdim=True).values,
                        torch.tensor(anchor, device=x.device))
    exp_x = torch.exp(x - max_val)
    exp_anchor = torch.exp(torch.tensor(anchor, device=x.device) - max_val)
    return exp_x / (exp_anchor + exp_x.sum(dim=dim, keepdim=True))

def build_context(bytecode, data=b''):
    """Build initial context."""
    context = []

    # Code section
    context.append(Token.CODE_START)
    for instr in bytecode:
        # Extract opcode (low byte) and immediate (top 24 bits)
        op = instr & 0xFF
        imm = instr >> 8
        context.append(op)
        # Add 4 immediate bytes (little-endian)
        for i in range(IMMEDIATE_SIZE):
            context.append((imm >> (i * 8)) & 0xFF)
        # Add 3 padding bytes for 8-byte instruction alignment
        for _ in range(PADDING_SIZE):
            context.append(0)
    context.append(Token.CODE_END)

    # Data section
    context.append(Token.DATA_START)
    context.extend(list(data))
    context.append(Token.DATA_END)

    return context

def trace_l6_fetch_lo(model, context_list, step_num, token_idx):
    """Trace L6 attention contribution to FETCH_LO[1] at specific position."""
    print(f"\n=== L6 FETCH_LO[1] Trace: Step {step_num}, Token {token_idx} ===")

    # Get L6 attention layer
    attn = model.blocks[6].attn

    # Convert context to tensor
    token_ids = torch.tensor([context_list], dtype=torch.long)
    S = len(context_list)

    # Get embedding
    x = model.embed(token_ids)  # [1, S, D]
    B, _, D = x.shape
    H = attn.num_heads
    HD = attn.head_dim

    print(f"Context length: {S}, Heads: {H}, Head_dim: {HD}")

    # Run layers 0-5 to get input to L6
    with torch.no_grad():
        for i in range(6):
            x = model.blocks[i](x)

    print(f"\n--- Input to L6 attention at position {S-1} ---")
    print(f"FETCH_LO[0] = {x[0, S-1, BD.FETCH_LO].item():.4f}")
    print(f"FETCH_LO[1] = {x[0, S-1, BD.FETCH_LO + 1].item():.4f}")

    # Now trace L6 attention in detail
    is_compact = getattr(attn, "_is_compact", False)

    with torch.no_grad():
        if is_compact:
            in_idx = attn._compact_in_idx
            x_in = x[:, :, in_idx]
            Q = F.linear(x_in, attn.W_q).view(B, S, H, -1).transpose(1, 2)
            K = F.linear(x_in, attn.W_k).view(B, S, H, -1).transpose(1, 2)
            V = F.linear(x_in, attn.W_v).view(B, S, H, -1).transpose(1, 2)
            n_out = len(attn._compact_out_idx)
            print(f"Compact mode: n_in={len(in_idx)}, n_out={n_out}")
        else:
            Q = F.linear(x, attn.W_q).view(B, S, H, HD).transpose(1, 2)
            K = F.linear(x, attn.W_k).view(B, S, H, HD).transpose(1, 2)
            V = F.linear(x, attn.W_v).view(B, S, H, HD).transpose(1, 2)

        # Apply RoPE if enabled
        if attn._rope_cos is not None:
            from neural_vm.base_layers import rotate_half
            S_q = Q.shape[2]
            S_kv = K.shape[2]
            cos_q = attn._rope_cos[:S_q].unsqueeze(0).unsqueeze(0)
            sin_q = attn._rope_sin[:S_q].unsqueeze(0).unsqueeze(0)
            cos_k = attn._rope_cos[:S_kv].unsqueeze(0).unsqueeze(0)
            sin_k = attn._rope_sin[:S_kv].unsqueeze(0).unsqueeze(0)
            Q = (Q * cos_q) + (rotate_half(Q) * sin_q)
            K = (K * cos_k) + (rotate_half(K) * sin_k)
            print("RoPE applied")

        # ALiBi slopes
        if attn.alibi_slopes is not None:
            print(f"ALiBi slopes: {attn.alibi_slopes}")
        else:
            print("No ALiBi (using RoPE)")

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * attn.scale

        # Add ALiBi if present
        if attn.alibi_slopes is not None:
            S_q, S_kv = Q.shape[2], K.shape[2]
            q_positions = torch.arange(S_q, device=x.device).unsqueeze(1)
            k_positions = torch.arange(S_kv, device=x.device).unsqueeze(0)
            dist = (q_positions - k_positions).abs().float()
            alibi = -attn.alibi_slopes.view(1, H, 1, 1) * dist
            scores = scores + alibi

        # Causal mask
        S_q = Q.shape[2]
        S_kv = K.shape[2]
        causal_mask = torch.triu(
            torch.full((S_q, S_kv), float("-inf"), device=x.device), diagonal=1
        )
        scores = scores + causal_mask

        # Softmax1
        attn_weights = softmax1(scores, dim=-1)

        # Compute attention output
        out = torch.matmul(attn_weights, V)  # [B, H, S, HD]

        # Reshape and apply W_o
        if is_compact:
            out_flat = out.transpose(1, 2).contiguous().view(B, S, n_out)
            attn_out = F.linear(out_flat, attn.W_o)
        else:
            out_flat = out.transpose(1, 2).contiguous().view(B, S, D)
            attn_out = F.linear(out_flat, attn.W_o)

        # Query position = last position
        query_pos = S - 1

        # Print per-head analysis
        print(f"\n--- Per-head analysis at position {query_pos} ---")

        for h in range(H):
            weights = attn_weights[0, h, query_pos, :]  # [S_kv]

            # Find top-3 positions by attention weight
            top_weights, top_positions = weights.topk(min(3, len(weights)))

            if top_weights[0].item() > 0.01:  # Only show heads with significant attention
                print(f"\nHead {h}:")
                print(f"  Top attended positions: {top_positions.tolist()}")
                print(f"  Attention weights: {[f'{w:.4f}' for w in top_weights.tolist()]}")

        # Special trace for head 5 (the problematic one)
        print(f"\n=== DETAILED HEAD 5 TRACE ===")
        h5_scores = scores[0, 5, query_pos, :]  # Raw scores before softmax1
        h5_weights = attn_weights[0, 5, query_pos, :]  # After softmax1
        h5_v = V[0, 5, :, :]  # [S, HD]
        h5_out = out[0, 5, query_pos, :]  # [HD]

        # Top raw scores
        top_raw_scores, top_raw_pos = h5_scores.topk(5)
        print(f"Head 5 top raw scores: {[f'{s:.4f}' for s in top_raw_scores.tolist()]}")
        print(f"  at positions: {top_raw_pos.tolist()}")

        # Top attention weights
        top_h5_weights, top_h5_pos = h5_weights.topk(5)
        print(f"Head 5 top attention weights: {[f'{w:.4f}' for w in top_h5_weights.tolist()]}")
        print(f"  at positions: {top_h5_pos.tolist()}")

        # Q vector for head 5 at query position
        h5_q = Q[0, 5, query_pos, :]  # [HD]
        print(f"Head 5 Q vector (query pos {query_pos}):")
        nonzero_q = (h5_q.abs() > 0.1).nonzero().squeeze(-1).tolist()
        if isinstance(nonzero_q, int):
            nonzero_q = [nonzero_q]
        for qi in nonzero_q[:10]:
            print(f"  Q[{qi}] = {h5_q[qi].item():.4f}")

        # K vector for head 5 at PC marker (should be in current step)
        # Find PC marker position in current step
        step_start = len(context_list) - 35 if len(context_list) >= 35 else 0
        for pos in range(step_start, len(context_list)):
            # PC marker has MARK_PC = 1
            mark_pc = x[0, pos, BD.MARK_PC].item()
            if mark_pc > 0.5:
                pc_marker_pos = pos
                print(f"PC marker at position {pc_marker_pos}")
                h5_k_pc = K[0, 5, pc_marker_pos, :]
                nonzero_k = (h5_k_pc.abs() > 0.1).nonzero().squeeze(-1).tolist()
                if isinstance(nonzero_k, int):
                    nonzero_k = [nonzero_k]
                print(f"Head 5 K vector at PC marker (pos {pc_marker_pos}):")
                for ki in nonzero_k[:10]:
                    print(f"  K[{ki}] = {h5_k_pc[ki].item():.4f}")
                break

        # Output contribution through V
        print(f"\nHead 5 output vector (slot 17 = FETCH_LO offset 0, slot 18 = offset 1):")
        for slot in [17, 18, 19, 20]:
            print(f"  out[{slot}] = {h5_out[slot].item():.4f}")

        # Check V values at attended positions for slots 17-20
        print(f"\nHead 5 V values at top positions for slots 17-20:")
        for pos in top_h5_pos.tolist()[:3]:
            v_vals = h5_v[pos, 17:21]
            print(f"  Pos {pos}: V[17:21] = {[f'{v:.4f}' for v in v_vals.tolist()]}")

        # Check FETCH_LO + 1 contribution
        debug_dim = BD.FETCH_LO + 1
        print(f"\n--- Contribution to FETCH_LO + 1 (dim {debug_dim}) ---")
        output_delta = attn_out[0, query_pos, debug_dim]
        print(f"Total attention contribution: {output_delta.item():.4f}")

        # Trace through W_o
        print(f"is_compact: {is_compact}")
        print(f"W_o shape: {attn.W_o.shape}")

        w_o_row = attn.W_o[debug_dim, :]
        out_at_qpos = out_flat[0, query_pos, :]

        nonzero_mask = w_o_row.abs() > 0.0001
        n_nonzero = nonzero_mask.sum().item()
        print(f"Non-zero W_o entries for dim {debug_dim}: {n_nonzero}")

        if nonzero_mask.any():
            nonzero_idxs = nonzero_mask.nonzero().squeeze(-1).tolist()
            if isinstance(nonzero_idxs, int):
                nonzero_idxs = [nonzero_idxs]

            # Sort by absolute contribution
            contribs = [(idx, w_o_row[idx].item() * out_at_qpos[idx].item()) for idx in nonzero_idxs]
            contribs.sort(key=lambda x: abs(x[1]), reverse=True)

            print("Top contributors:")
            for idx, contrib in contribs[:15]:
                print(f"  W_o[{debug_dim}, {idx}] = {w_o_row[idx].item():.4f}, out[{idx}] = {out_at_qpos[idx].item():.4f}, contrib = {contrib:.4f}")

                # Map compact index to original head/slot
                if is_compact:
                    orig_idx = attn._compact_out_idx[idx].item()
                    h = orig_idx // HD
                    slot = orig_idx % HD
                    print(f"    -> Original index {orig_idx}: Head {h}, Slot {slot}")

        # Check what L6 adds via residual
        x_out = x + attn_out
        print(f"\n--- After L6 attention (via residual) ---")
        print(f"FETCH_LO[0] = {x_out[0, query_pos, BD.FETCH_LO].item():.4f}")
        print(f"FETCH_LO[1] = {x_out[0, query_pos, BD.FETCH_LO + 1].item():.4f}")

def main():
    # Create and set up model
    model = AutoregressiveVM()
    set_vm_weights(model)
    model.compact(block_size=32)
    model.eval()

    # Build context
    context = build_context(BYTECODE)
    print(f"Initial context length: {len(context)}")

    # Create DraftVM
    draft = DraftVM(BYTECODE)

    # Run 3 steps to reach IMM 0
    for step in range(3):
        draft.step()
        draft_tokens = draft.draft_tokens()
        context.extend(draft_tokens)
        print(f"After step {step+1}: PC={draft.pc:#x}, AX={draft.ax:#x}, SP={draft.sp:#x}, context_len={len(context)}")

    # Now we're about to predict step 3 (IMM 0) output
    # Token 6 in the step output is AX byte 0

    # For Step 3, we need to trace what happens at each token position
    # Token positions in step output:
    # 0-4: PC bytes
    # 5: AX marker
    # 6-10: AX bytes (0-4)
    # 11: SP marker
    # 12-16: SP bytes
    # etc.

    # To predict token 6 (AX byte 0), we need context up to position (context_len - 35 + 5)
    # Actually, the draft tokens are already appended, so the position predicting AX byte 0
    # is at context_len - 35 + 6 - 1 = context_len - 30

    # Let me trace what the model predicts for each of the first few step tokens
    step_start = len(context) - 35

    print(f"\n=== Tracing Step 3 token predictions ===")
    print(f"Step starts at context position {step_start}")
    print(f"Draft tokens: {context[step_start:step_start+10]}")  # First 10 tokens

    # Trace prediction for token 6 (AX byte 0)
    # To predict token 6, we use context up to step_start + 5 (inclusive)
    target_token = 6  # AX byte 0
    context_for_pred = context[:step_start + target_token]  # Context up to AX marker

    print(f"\nContext length for predicting token {target_token}: {len(context_for_pred)}")
    print(f"Draft says AX byte 0 should be: {context[step_start + target_token]}")

    # Trace L6 attention contribution
    trace_l6_fetch_lo(model, context_for_pred, 3, target_token)

if __name__ == "__main__":
    main()
