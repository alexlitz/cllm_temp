#!/usr/bin/env python3
"""Debug L3 head 5 - AX_FULL relay not working."""

import torch
from neural_vm.run_vm import AutoregressiveVMRunner, Token
from neural_vm.vm_step import _SetDim as BD
from neural_vm.embedding import Opcode

bytecode = [
    Opcode.IMM | (42 << 8),  # [0] IMM 42
    Opcode.EXIT,             # [1] EXIT
]

runner = AutoregressiveVMRunner()
runner._func_call_handlers = {}
runner._bytecode = bytecode
runner._last_sp = 0x1F800
runner._last_bp = 0x10000

ctx = runner._build_context(bytecode, b"", [])
runner.model.set_active_opcode(bytecode[0] & 0xFF)

# Step 0: IMM 42
for _ in range(35):
    next_token = runner.model.generate_next(ctx, max_context_window=2048)
    ctx.append(next_token)
    if next_token == Token.HALT:
        break

# Set opcode for step 1 (EXIT)
runner.model.set_active_opcode(Opcode.EXIT)

# Generate tokens until AX marker for step 1
for _ in range(20):
    next_token = runner.model.generate_next(ctx, max_context_window=2048)
    ctx.append(next_token)
    if next_token == Token.REG_AX:
        break

# Find positions of AX markers in context
ax_positions = [i for i, t in enumerate(ctx) if t == Token.REG_AX]
print(f"AX marker positions: {ax_positions}")
print(f"Context length: {len(ctx)}")

if len(ax_positions) >= 2:
    step0_ax = ax_positions[-2]
    step1_ax = ax_positions[-1]
    distance = step1_ax - step0_ax
    print(f"Step 0 AX position: {step0_ax}")
    print(f"Step 1 AX position: {step1_ax}")
    print(f"Distance: {distance}")

# Check L3 attention head 5
with torch.no_grad():
    device = next(runner.model.parameters()).device
    token_ids = torch.tensor([ctx], dtype=torch.long, device=device)

    # Run through L0-L2 (blocks 0,1,2) before L3 (block 3)
    x = runner.model.embed(token_ids)
    for i in range(3):  # blocks[0], blocks[1], blocks[2]
        x = runner.model.blocks[i](x)

    # Check L3 attention - blocks[3] is "L3" per code convention
    attn3 = runner.model.blocks[3].attn

    # Check ALiBi slopes
    if hasattr(attn3, 'alibi_slopes') and attn3.alibi_slopes is not None:
        print(f"\nL3 ALiBi slopes: {attn3.alibi_slopes.tolist()}")
        print(f"  Head 5 slope: {attn3.alibi_slopes[5].item():.4f}")

    # Compute Q, K, V for head 5
    HD = 64
    base = 5 * HD

    Q = x @ attn3.W_q.T
    K = x @ attn3.W_k.T
    V = x @ attn3.W_v.T

    # Q at step 1 AX marker (last position)
    q_head5 = Q[0, -1, base:base+HD]
    print(f"\nHead 5 Q at step 1 AX marker (pos {step1_ax}):")
    nonzero_q = [(i, q_head5[i].item()) for i in range(HD) if abs(q_head5[i].item()) > 0.5]
    print(f"  Non-zero Q: {nonzero_q[:5]}")

    # K at step 0 AX marker
    k_head5_step0 = K[0, step0_ax, base:base+HD]
    print(f"\nHead 5 K at step 0 AX marker (pos {step0_ax}):")
    nonzero_k = [(i, k_head5_step0[i].item()) for i in range(HD) if abs(k_head5_step0[i].item()) > 0.5]
    print(f"  Non-zero K: {nonzero_k[:5]}")

    # Raw attention score
    score = (q_head5 @ k_head5_step0).item() / (HD ** 0.5)
    print(f"\nRaw attention score (before ALiBi): {score:.4f}")

    # ALiBi penalty
    slope = attn3.alibi_slopes[5].item()
    alibi_penalty = slope * distance
    print(f"ALiBi penalty (slope={slope:.4f} * distance={distance}): {alibi_penalty:.4f}")

    final_score = score - alibi_penalty
    print(f"Final score: {final_score:.4f}")

    # V at step 0 AX marker (should have OUTPUT values)
    v_head5_step0 = V[0, step0_ax, base:base+HD]
    print(f"\nHead 5 V at step 0 AX marker:")
    # V dims 1-16: OUTPUT_LO, V dims 17-32: OUTPUT_HI
    v_output_lo = [v_head5_step0[1+k].item() for k in range(16)]
    v_output_hi = [v_head5_step0[17+k].item() for k in range(16)]
    max_v_lo = v_output_lo.index(max(v_output_lo)) if max(v_output_lo) > 0.1 else -1
    max_v_hi = v_output_hi.index(max(v_output_hi)) if max(v_output_hi) > 0.1 else -1
    print(f"  V OUTPUT_LO max: idx={max_v_lo}, val={max(v_output_lo):.2f}")
    print(f"  V OUTPUT_HI max: idx={max_v_hi}, val={max(v_output_hi):.2f}")
    if max_v_lo >= 0 and max_v_hi >= 0:
        v_byte = max_v_lo | (max_v_hi << 4)
        print(f"  → V encodes byte 0x{v_byte:02x} = {v_byte}")

    # Check what OUTPUT values are at step 0 AX marker (before attention)
    print(f"\nOUTPUT at step 0 AX marker (pos {step0_ax}) before L3:")
    pos_step0 = x[0, step0_ax, :]
    out_lo = [pos_step0[BD.OUTPUT_LO + k].item() for k in range(16)]
    out_hi = [pos_step0[BD.OUTPUT_HI + k].item() for k in range(16)]
    max_out_lo = out_lo.index(max(out_lo)) if max(out_lo) > 0.1 else -1
    max_out_hi = out_hi.index(max(out_hi)) if max(out_hi) > 0.1 else -1
    print(f"  OUTPUT_LO max: idx={max_out_lo}, val={max(out_lo):.2f}")
    print(f"  OUTPUT_HI max: idx={max_out_hi}, val={max(out_hi):.2f}")
