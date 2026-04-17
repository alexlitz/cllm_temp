#!/usr/bin/env python3
"""Debug L5 head 3 attention scores to see which position it's attending to."""

import torch
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token, _SetDim as BD
from neural_vm.embedding import Opcode
from neural_vm.constants import INSTR_WIDTH, PC_OFFSET
from src.compiler import compile_c

# Simple program
code = 'int main() { return 42; }'
bytecode, data = compile_c(code)
target_byte0 = (bytecode[0] >> 8) & 0xFF
print(f"Expected byte: 0x{target_byte0:02x}")

# Create runner
runner = AutoregressiveVMRunner()
runner._func_call_handlers = {}
runner._bytecode = bytecode
runner._last_sp = 0x1F800
runner._last_bp = 0x10000
ctx = runner._build_context(bytecode, data, [])

# Generate REG_PC marker
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

    # Now compute L5 attention manually
    L5_attn = runner.model.blocks[5].attn
    HD = 64
    head = 3
    base = head * HD

    # Get Q, K, V
    q = x @ L5_attn.W_q.T  # [B, S, D]
    k = x @ L5_attn.W_k.T
    v = x @ L5_attn.W_v.T

    # Extract head 3
    q_h3 = q[0, :, base:base+HD]
    k_h3 = k[0, :, base:base+HD]
    v_h3 = v[0, :, base:base+HD]

    # Q at last position, K at all positions
    q_last = q_h3[-1, :]
    scores = torch.matmul(q_last, k_h3.T) / (HD ** 0.5)

    print(f"\n=== L5 Head 3 Attention Scores (last pos → all) ===")
    print(f"Sequence length: {len(scores)}")

    # Show top 10 positions
    top_scores, top_positions = torch.topk(scores, min(10, len(scores)))
    print(f"\nTop 10 positions:")
    for i, (score, pos) in enumerate(zip(top_scores, top_positions)):
        tok = token_ids[0, pos].item()
        # Get ADDR_KEY at this position
        addr_key_lo = [x[0, pos, BD.ADDR_KEY + k].item() for k in range(16)]
        addr_key_hi = [x[0, pos, BD.ADDR_KEY + 16 + k].item() for k in range(16)]
        lo_idx = addr_key_lo.index(max(addr_key_lo)) if max(addr_key_lo) > 0.5 else -1
        hi_idx = addr_key_hi.index(max(addr_key_hi)) if max(addr_key_hi) > 0.5 else -1

        print(f"  {i}: pos={pos.item()}, tok={tok}, addr={lo_idx + hi_idx*16 if lo_idx >= 0 else 'none'}, score={score.item():.2f}")

    # Show score at position 2 (where the target byte is)
    print(f"\nPosition 2 (expected target):")
    print(f"  score={scores[2].item():.2f}")
    print(f"  Q@K: {torch.dot(q_last, k_h3[2]).item():.2f}")

    # Check Q values
    imm_addr = PC_OFFSET + 1  # = 3
    print(f"\nQ at last position (querying for addr={imm_addr}):")
    print(f"  Q[{imm_addr & 0xF}] (lo nibble {imm_addr & 0xF}): {q_last[imm_addr & 0xF].item():.2f}")
    print(f"  Q[16] (hi nibble 0): {q_last[16].item():.2f}")
    print(f"  Q[32] (MARK_PC gate): {q_last[32].item():.2f}")

    # Check K at position 2
    print(f"\nK at position 2 (addr=3):")
    print(f"  K[3]: {k_h3[2, 3].item():.2f}")
    print(f"  K[16]: {k_h3[2, 16].item():.2f}")
    print(f"  K[32]: {k_h3[2, 32].item():.2f}")
