#!/usr/bin/env python3
"""Debug JMP relay - which block has the head 2 JMP relay configured?"""

import torch
from neural_vm.run_vm import AutoregressiveVMRunner, Token
from neural_vm.vm_step import _SetDim as BD
from neural_vm.embedding import Opcode

runner = AutoregressiveVMRunner()
runner._func_call_handlers = {}

# Print which blocks have head 2 configured for JMP relay
print("=== Checking head 2 Q weights across all blocks ===")
print("Looking for: W_q[128, MARK_PC]=50, W_q[128, HAS_SE]=-50, W_q[128, MARK_AX]=-50")
print()

HD = 64
base = 2 * HD  # head 2 base

for i, block in enumerate(runner.model.blocks):
    attn = block.attn
    q_mark_pc = attn.W_q.data[base, BD.MARK_PC].item()
    q_has_se = attn.W_q.data[base, BD.HAS_SE].item()
    q_mark_ax = attn.W_q.data[base, BD.MARK_AX].item()
    q_op_jmp = attn.W_q.data[base, BD.OP_JMP].item() if hasattr(BD, 'OP_JMP') else 0

    k_mark_pc = attn.W_k.data[base, BD.MARK_PC].item()

    # Check if this block has JMP relay-like Q weights
    is_jmp_like = abs(q_mark_pc) > 10 and abs(q_has_se) > 10

    if is_jmp_like or i in [5, 6, 7]:  # Always show L5-L7
        print(f"blocks[{i}] (L{i+1} or 'LAYER {i}'):")
        print(f"  W_q[128, MARK_PC]: {q_mark_pc:.2f}")
        print(f"  W_q[128, HAS_SE]: {q_has_se:.2f}")
        print(f"  W_q[128, MARK_AX]: {q_mark_ax:.2f}")
        print(f"  W_q[128, OP_JMP]: {q_op_jmp:.2f}")
        print(f"  W_k[128, MARK_PC]: {k_mark_pc:.2f}")

        # Check V for FETCH
        v_fetch_lo_4 = attn.W_v.data[base + 2 + 4, BD.FETCH_LO + 4].item()
        v_op_jmp = attn.W_v.data[base + 1, BD.OP_JMP].item() if hasattr(BD, 'OP_JMP') else 0
        print(f"  W_v[{base+6}, FETCH_LO+4]: {v_fetch_lo_4:.2f}")
        print(f"  W_v[{base+1}, OP_JMP]: {v_op_jmp:.2f}")

        # Check O for AX_CARRY
        o_ax_carry_lo_4 = attn.W_o.data[BD.AX_CARRY_LO + 4, base + 2 + 4].item()
        o_cmp_0 = attn.W_o.data[BD.CMP + 0, base + 1].item()
        print(f"  W_o[AX_CARRY_LO+4, {base+6}]: {o_ax_carry_lo_4:.2f}")
        print(f"  W_o[CMP+0, {base+1}]: {o_cmp_0:.2f}")
        print()

# Now run a simple JMP test
print("=" * 60)
print("=== Running simple JMP 4 test ===")
print()

bytecode = [
    Opcode.JMP | (4 << 8),    # [0] JMP to instruction 4
    Opcode.IMM | (99 << 8),   # [1] IMM 99 (should be skipped)
    Opcode.EXIT,              # [2] EXIT 99 (should be skipped)
    Opcode.NOP,               # [3] NOP padding
    Opcode.IMM | (42 << 8),   # [4] IMM 42 (target)
    Opcode.EXIT,              # [5] EXIT with 42
]

runner._bytecode = bytecode
runner._last_sp = 0x1F800
runner._last_bp = 0x10000

ctx = runner._build_context(bytecode, b"", [])
runner.model.set_active_opcode(Opcode.JMP)

# Generate REG_PC marker
next_token = runner.model.generate_next(ctx, max_context_window=2048)
ctx.append(next_token)

print(f"Context length: {len(ctx)} tokens")
print(f"Generated token: {next_token} (expected REG_PC={Token.REG_PC})")

with torch.no_grad():
    device = next(runner.model.parameters()).device
    token_ids = torch.tensor([ctx], dtype=torch.long, device=device)

    # Check state at different layer boundaries
    x = runner.model.embed(token_ids)

    for layer_idx in [5, 6, 7]:  # Check after blocks[5], blocks[6], blocks[7]
        # Run through blocks up to layer_idx (inclusive)
        x_test = runner.model.embed(token_ids)
        for i in range(layer_idx + 1):
            x_test = runner.model.blocks[i](x_test)

        pos = x_test[0, -1, :]

        ax_carry_lo = [pos[BD.AX_CARRY_LO + k].item() for k in range(16)]
        ax_carry_hi = [pos[BD.AX_CARRY_HI + k].item() for k in range(16)]
        max_axc_lo = ax_carry_lo.index(max(ax_carry_lo)) if max(ax_carry_lo) > 0.1 else -1

        fetch_lo = [pos[BD.FETCH_LO + k].item() for k in range(16)]
        max_fetch = fetch_lo.index(max(fetch_lo)) if max(fetch_lo) > 0.1 else -1

        cmp0 = pos[BD.CMP + 0].item()

        print(f"After blocks[0:{layer_idx}] (through 'LAYER {layer_idx}'):")
        print(f"  FETCH_LO max: idx={max_fetch}, val={max(fetch_lo):.2f}")
        print(f"  AX_CARRY_LO max: idx={max_axc_lo}, val={max(ax_carry_lo):.2f}")
        print(f"  CMP[0]: {cmp0:.2f}")
        print()
