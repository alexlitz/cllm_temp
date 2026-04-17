#!/usr/bin/env python3
"""Debug L5 head 3 softmax weights and output."""

import torch
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token, _SetDim as BD
from neural_vm.embedding import Opcode
from neural_vm.constants import PC_OFFSET
from src.compiler import compile_c

# Simple program
code = 'int main() { return 42; }'
bytecode, data = compile_c(code)
target_byte0 = (bytecode[0] >> 8) & 0xFF
print(f"Expected byte: 0x{target_byte0:02x} (lo={target_byte0 & 0xF}, hi={target_byte0 >> 4})")

# Create runner and context
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

    # Get Q, K, V
    q = x @ L5_attn.W_q.T
    k = x @ L5_attn.W_k.T
    v = x @ L5_attn.W_v.T

    # Extract head 3
    q_h3 = q[0, :, base:base+HD]
    k_h3 = k[0, :, base:base+HD]
    v_h3 = v[0, :, base:base+HD]

    # Attention scores and weights
    q_last = q_h3[-1, :]
    scores = torch.matmul(q_last, k_h3.T) / (HD ** 0.5)

    weights = torch.softmax(scores, dim=-1)

    print(f"\n=== Softmax Weights (top 10) ===")
    top_weights, top_positions = torch.topk(weights, min(10, len(weights)))
    for i, (w, pos) in enumerate(zip(top_weights, top_positions)):
        tok = token_ids[0, pos].item()
        print(f"  {i}: pos={pos.item()}, tok={tok}, weight={w.item():.6f}")

    print(f"\nWeight at position 2 (target): {weights[2].item():.6f}")
    print(f"Sum of top 10 weights: {top_weights.sum().item():.4f}")

    # V at position 2
    v_pos2 = v_h3[2, :]
    print(f"\n=== V at position 2 (byte 0x{target_byte0:02x}) ===")
    print(f"V[32:40]: {[v_pos2[32+k].item() for k in range(8)]}")
    print(f"V[48:56]: {[v_pos2[48+k].item() for k in range(8)]}")

    # Expected: V[32+2]=1.0 (lo nibble=2), V[48+7]=1.0 (hi nibble=7)
    print(f"\nExpected active: V[34]={v_pos2[34].item():.2f} (lo=2), V[55]={v_pos2[55].item():.2f} (hi=7)")

    # Weighted output
    attn_output = weights @ v_h3  # [HD]
    print(f"\n=== Weighted attention output ===")
    print(f"output[32:40]: {[attn_output[32+k].item() for k in range(8)]}")
    print(f"output[48:56]: {[attn_output[48+k].item() for k in range(8)]}")

    # Check V at other high-weighted positions
    print(f"\n=== V values at top weighted positions ===")
    for pos_tensor in top_positions[:5]:
        pos = pos_tensor.item()
        tok = token_ids[0, pos].item()
        v_pos = v_h3[pos, :]
        # Find which nibbles are active
        max_lo = max(v_pos[32:48])
        max_hi = max(v_pos[48:64])
        lo_idx = -1
        hi_idx = -1
        for k in range(16):
            if v_pos[32+k] > 0.5:
                lo_idx = k
            if v_pos[48+k] > 0.5:
                hi_idx = k
        print(f"  Pos {pos}: tok={tok}, V nibbles: lo={lo_idx}, hi={hi_idx} → byte 0x{lo_idx + (hi_idx<<4):02x}" if lo_idx >= 0 else f"  Pos {pos}: tok={tok}, V weak")
