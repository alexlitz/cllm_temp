"""Debug LEA 8 - check what's activating in L9 and contributing to OUTPUT_HI[0]."""
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

ctx = context + draft_tokens[:6]
pos = len(ctx) - 1

print("LEA 8 - L9 Activations Contributing to OUTPUT_HI[0]")
print("=" * 70)
print()

l9_ffn_in = None

def l9_ffn_hook(module, input, output):
    global l9_ffn_in
    if isinstance(input, tuple):
        l9_ffn_in = input[0].detach().clone()
    else:
        l9_ffn_in = input.detach().clone()

model.blocks[9].ffn.register_forward_hook(l9_ffn_hook)

ctx_tensor = torch.tensor([ctx], dtype=torch.long)
with torch.no_grad():
    logits = model.forward(ctx_tensor)

if l9_ffn_in is not None:
    ffn = model.blocks[9].ffn
    x = l9_ffn_in[0, pos, :]

    # Manually compute FFN
    up = F.linear(x.unsqueeze(0), ffn.W_up, ffn.b_up)[0]
    gate = F.linear(x.unsqueeze(0), ffn.W_gate, ffn.b_gate)[0]
    hidden = F.silu(up) * gate

    W_down = ffn.W_down

    print(f"Position {pos} (REG_AX marker):")
    print()

    # Find units contributing to OUTPUT_HI[0]
    output_hi_0_col = W_down[BD.OUTPUT_HI + 0, :]
    strong_writers = torch.where(torch.abs(output_hi_0_col) > 0.01)[0]

    active_units = []
    for unit_idx in strong_writers:
        h = hidden[unit_idx].item()
        weight = output_hi_0_col[unit_idx].item()
        contrib = h * weight

        if abs(contrib) > 0.1:
            active_units.append((unit_idx.item(), h, weight, contrib))

    print(f"Found {len(active_units)} active units writing to OUTPUT_HI[0]:")
    print()

    total_contribution = 0.0
    for unit_idx, h, weight, contrib in sorted(active_units, key=lambda x: -abs(x[3]))[:10]:
        total_contribution += contrib
        print(f"Unit {unit_idx}: h={h:.1f}, w={weight:.3f}, contrib={contrib:.1f}")

        # Analyze this unit
        W_up_unit = ffn.W_up[unit_idx, :]
        b_up = ffn.b_up[unit_idx].item()
        W_gate_unit = ffn.W_gate[unit_idx, :]
        b_gate = ffn.b_gate[unit_idx].item()

        # Check what activated this unit
        strong_up_dims = []
        up_val = b_up
        for dim in range(x.shape[0]):
            w = W_up_unit[dim].item()
            if abs(w) > 0.5:
                val = x[dim].item()
                contribution = val * w
                up_val += contribution
                if abs(contribution) > 1.0:
                    strong_up_dims.append((dim, val, w, contribution))

        strong_gate_dims = []
        gate_val = b_gate
        for dim in range(x.shape[0]):
            w = W_gate_unit[dim].item()
            if abs(w) > 0.5:
                val = x[dim].item()
                contribution = val * w
                gate_val += contribution
                if abs(contribution) > 0.5:
                    strong_gate_dims.append((dim, val, w, contribution))

        print(f"  up={up_val:.1f} (bias={b_up:.1f})")
        if len(strong_up_dims) > 0:
            for dim, val, w, c in sorted(strong_up_dims, key=lambda x: -abs(x[3]))[:3]:
                print(f"    x[{dim}]={val:.1f} * {w:.2f} = {c:.1f}")

        print(f"  gate={gate_val:.1f} (bias={b_gate:.1f})")
        if len(strong_gate_dims) > 0:
            for dim, val, w, c in sorted(strong_gate_dims, key=lambda x: -abs(x[3]))[:3]:
                print(f"    x[{dim}]={val:.1f} * {w:.2f} = {c:.1f}")

        print()

    print(f"Total contribution to OUTPUT_HI[0]: {total_contribution:.1f}")
    print()

    # Show relevant input dims
    print("Key input dimensions:")
    print(f"  MARK_AX: {x[BD.MARK_AX].item():.1f}")
    print(f"  ALU_HI[0-7]: {x[BD.ALU_HI:BD.ALU_HI+8].tolist()}")
    print(f"  AX_CARRY_HI[0-7]: {x[BD.AX_CARRY_HI:BD.AX_CARRY_HI+8].tolist()}")
    print(f"  CARRY[0]: {x[BD.CARRY+0].item():.1f}")
    print(f"  OP_ADD: {x[BD.OP_ADD].item():.1f}")
    print(f"  OP_LEA: {x[BD.OP_LEA].item():.1f}")
