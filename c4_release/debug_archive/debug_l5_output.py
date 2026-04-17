#!/usr/bin/env python3
"""Debug L5 output projection to see why FETCH has wrong values."""

import torch
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token, _SetDim as BD
from neural_vm.embedding import Opcode
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

    # Manual L5 attention computation
    L5_block = runner.model.blocks[5]
    L5_attn = L5_block.attn
    HD = 64
    n_heads = 8
    head = 3
    base = head * HD

    # Full QKV projection
    q = x @ L5_attn.W_q.T  # [B, S, D]
    k = x @ L5_attn.W_k.T
    v = x @ L5_attn.W_v.T

    B, S, D = q.shape

    # Reshape for multi-head attention
    q = q.view(B, S, n_heads, HD).transpose(1, 2)  # [B, H, S, HD]
    k = k.view(B, S, n_heads, HD).transpose(1, 2)
    v = v.view(B, S, n_heads, HD).transpose(1, 2)

    # Attention scores
    scores = torch.matmul(q, k.transpose(-2, -1)) / (HD ** 0.5)  # [B, H, S, S]

    # Apply causal mask
    mask = torch.triu(torch.ones(S, S, device=device) * float('-inf'), diagonal=1)
    scores = scores + mask

    # Softmax
    weights = torch.softmax(scores, dim=-1)  # [B, H, S, S]

    # Attention output
    attn_out = torch.matmul(weights, v)  # [B, H, S, HD]

    # Concatenate heads
    attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)  # [B, S, D]

    print(f"\n=== Attention output for head 3 at last position ===")
    head3_out = attn_out[0, -1, base:base+HD]
    print(f"head3_out[32:40]: {[head3_out[32+k].item() for k in range(8)]}")
    print(f"head3_out[48:56]: {[head3_out[48+k].item() for k in range(8)]}")

    # Output projection
    out = attn_out @ L5_attn.W_o.T  # [B, S, D]

    print(f"\n=== After output projection at last position ===")
    print(f"FETCH_LO dims: {[out[0, -1, BD.FETCH_LO + k].item() for k in range(8)]}")
    print(f"FETCH_HI dims: {[out[0, -1, BD.FETCH_HI + k].item() for k in range(8)]}")

    # Check what W_o does for head 3
    # W_o[FETCH_LO + k, base + 32 + k] = 40.0 for k in range(16)
    print(f"\n=== W_o weights for head 3 → FETCH ===")
    for k in range(4):
        w = L5_attn.W_o[BD.FETCH_LO + k, base + 32 + k].item()
        print(f"  W_o[FETCH_LO+{k}={BD.FETCH_LO + k}, {base + 32 + k}] = {w:.1f}")

    # Check contribution from head3_out[34] (lo nibble 2) to FETCH_LO[2]
    print(f"\n=== Manual FETCH_LO[2] calculation ===")
    # FETCH_LO[2] = sum over all dims of: out[d] * W_o[FETCH_LO+2, d]
    fetch_lo_2_contrib = 0
    for d in range(D):
        w = L5_attn.W_o[BD.FETCH_LO + 2, d].item()
        if abs(w) > 0.01:
            contrib = attn_out[0, -1, d].item() * w
            print(f"  dim {d}: attn_out={attn_out[0, -1, d].item():.2f}, W_o={w:.1f}, contrib={contrib:.2f}")
            fetch_lo_2_contrib += contrib
    print(f"  Total FETCH_LO[2]: {fetch_lo_2_contrib:.2f}")

    # Also check residual connection
    print(f"\n=== Full forward through L5 ===")
    x_out = L5_block(x)
    print(f"After L5 block (with residual):")
    print(f"  FETCH_LO: {[x_out[0, -1, BD.FETCH_LO + k].item() for k in range(8)]}")
    print(f"  FETCH_HI: {[x_out[0, -1, BD.FETCH_HI + k].item() for k in range(8)]}")
