#!/usr/bin/env python3
"""Debug L10 FFN for LEA 16 at PC marker - detailed unit analysis."""
import torch
import sys
sys.path.insert(0, '.')

from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim as BD
from neural_vm.speculative import DraftVM
from neural_vm.embedding import Opcode
from neural_vm.constants import IMMEDIATE_SIZE, PADDING_SIZE

BYTECODE = [Opcode.LEA | (16 << 8)]

def build_context(bytecode, data=b""):
    context = [Token.CODE_START]
    for instr in bytecode:
        op = instr & 0xFF
        imm = instr >> 8
        context.append(op)
        for i in range(IMMEDIATE_SIZE):
            context.append((imm >> (i * 8)) & 0xFF)
        for _ in range(PADDING_SIZE):
            context.append(0)
    context.append(Token.CODE_END)
    context.append(Token.DATA_START)
    context.extend(list(data))
    context.append(Token.DATA_END)
    return context

model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

S = 100

context = build_context(BYTECODE)
draft = DraftVM(BYTECODE)
draft.step()
step1_tokens = draft.draft_tokens()

ctx = context + step1_tokens[:1]  # Just PC marker
pc_marker_pos = len(ctx) - 1

token_ids = torch.tensor([ctx], dtype=torch.long)

with torch.no_grad():
    x = model.embed(token_ids)
    for i in range(10):
        x = model.blocks[i](x)

    # Get values at PC marker
    print(f"At PC marker (pos {pc_marker_pos}):")
    print(f"  MARK_AX = {x[0, pc_marker_pos, BD.MARK_AX].item():.4f}")

    # Check OP flags
    op_and = x[0, pc_marker_pos, BD.OP_AND].item()
    op_or = x[0, pc_marker_pos, BD.OP_OR].item()
    op_xor = x[0, pc_marker_pos, BD.OP_XOR].item()
    print(f"  OP_AND = {op_and:.4f}")
    print(f"  OP_OR = {op_or:.4f}")
    print(f"  OP_XOR = {op_xor:.4f}")

    # Check ALU_LO values
    print(f"\n  ALU_LO values (non-zero):")
    alu_lo = [x[0, pc_marker_pos, BD.ALU_LO + k].item() for k in range(16)]
    for k, v in enumerate(alu_lo):
        if abs(v) > 0.5:
            print(f"    ALU_LO[{k}] = {v:.4f}")

    # Check AX_CARRY_LO values
    print(f"\n  AX_CARRY_LO values (non-zero):")
    ax_carry_lo = [x[0, pc_marker_pos, BD.AX_CARRY_LO + k].item() for k in range(16)]
    for k, v in enumerate(ax_carry_lo):
        if abs(v) > 0.5:
            print(f"    AX_CARRY_LO[{k}] = {v:.4f}")

    # Calculate which bitwise units would fire
    mark_ax = x[0, pc_marker_pos, BD.MARK_AX].item()

    print(f"\n  Analyzing L10 bitwise units that fire (up > 0):")
    print(f"  Formula: MARK_AX*10 + ALU_LO[a] + AX_CARRY_LO[b] - 10.5")

    firing_units = []
    for a in range(16):
        for b in range(16):
            up = mark_ax * 10 + alu_lo[a] + ax_carry_lo[b] - 10.5
            if up > 0:
                silu_val = torch.nn.functional.silu(torch.tensor(up * S)).item()
                # AND result
                result = a & b
                firing_units.append((a, b, result, up, silu_val, op_and))

    print(f"  {len(firing_units)} units fire (up > 0)")
    if len(firing_units) > 0:
        # Show top 5 by contribution
        firing_units.sort(key=lambda x: -x[4] * x[5])
        print(f"\n  Top 5 firing units (by contrib to OUTPUT_LO):")
        for a, b, result, up, silu_val, gate in firing_units[:5]:
            contrib = (2.0 / S) * silu_val * gate
            print(f"    a={a}, b={b}: up={up:.2f}, silu={silu_val:.2f}, gate={gate:.4f}, result={result}, contrib={contrib:.4f}")

        # Calculate total contribution to OUTPUT_LO[0]
        total_output_lo_0 = 0
        for a in range(16):
            for b in range(16):
                up = mark_ax * 10 + alu_lo[a] + ax_carry_lo[b] - 10.5
                if up > 0:
                    silu_val = torch.nn.functional.silu(torch.tensor(up * S)).item()
                    result = a & b
                    if result == 0:
                        total_output_lo_0 += (2.0 / S) * silu_val * op_and

        print(f"\n  Total contribution to OUTPUT_LO[0] from AND units: {total_output_lo_0:.4f}")

        # What about OR and XOR?
        total_or = 0
        total_xor = 0
        for a in range(16):
            for b in range(16):
                up = mark_ax * 10 + alu_lo[a] + ax_carry_lo[b] - 10.5
                if up > 0:
                    silu_val = torch.nn.functional.silu(torch.tensor(up * S)).item()
                    if (a | b) == 0:
                        total_or += (2.0 / S) * silu_val * op_or
                    if (a ^ b) == 0:
                        total_xor += (2.0 / S) * silu_val * op_xor

        print(f"  Total contribution to OUTPUT_LO[0] from OR units: {total_or:.4f}")
        print(f"  Total contribution to OUTPUT_LO[0] from XOR units: {total_xor:.4f}")

    # What threshold is needed?
    print(f"\n  Required threshold to block all units:")
    max_up = 0
    for a in range(16):
        for b in range(16):
            up_val = mark_ax * 10 + alu_lo[a] + ax_carry_lo[b]
            if up_val > max_up:
                max_up = up_val
    print(f"  Max up value (before threshold): {max_up:.2f}")
    print(f"  Current threshold: 10.5")
    print(f"  Suggested threshold: {max_up + 1:.2f}")
