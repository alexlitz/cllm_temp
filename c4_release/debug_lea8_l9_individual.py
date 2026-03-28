"""Debug LEA 8 - check units writing to individual OUTPUT_HI dims."""
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

print("LEA 8 - L9 FFN Individual OUTPUT_HI Writers")
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

    up = F.linear(x.unsqueeze(0), ffn.W_up, ffn.b_up)[0]
    gate = F.linear(x.unsqueeze(0), ffn.W_gate, ffn.b_gate)[0]
    hidden = F.silu(up) * gate

    W_down = ffn.W_down

    print(f"Position {pos} (REG_AX marker):")
    print()

    # For each OUTPUT_HI dim, find contributing units
    print("Checking each OUTPUT_HI dimension:")
    print()

    for hi_dim in range(16):
        output_dim = BD.OUTPUT_HI + hi_dim

        # Find units writing to this dim
        W_down_col = W_down[output_dim, :]
        strong_writers = torch.where(torch.abs(W_down_col) > 0.01)[0]

        total_contribution = 0.0
        active_units = []

        for unit_idx in strong_writers:
            h = hidden[unit_idx].item()
            weight = W_down_col[unit_idx].item()
            contrib = h * weight

            if abs(contrib) > 0.1:
                active_units.append((unit_idx.item(), h, weight, contrib))
                total_contribution += contrib

        if len(active_units) > 0:
            print(f"OUTPUT_HI[{hi_dim}]: {len(active_units)} active units, total contrib = {total_contribution:.1f}")
            for unit_idx, h, weight, contrib in sorted(active_units, key=lambda x: -abs(x[3]))[:3]:
                print(f"  Unit {unit_idx}: h={h:.2f}, w={weight:.3f}, contrib={contrib:.1f}")

            if len(active_units) == 1:
                # Single unit - investigate it
                unit_idx = active_units[0][0]
                print(f"  → Investigating unit {unit_idx}...")

                W_up_unit = ffn.W_up[unit_idx, :]
                b_up_unit = ffn.b_up[unit_idx].item()
                W_gate_unit = ffn.W_gate[unit_idx, :]
                b_gate_unit = ffn.b_gate[unit_idx].item()

                strong_up = torch.where(torch.abs(W_up_unit) > 50)[0]
                if len(strong_up) > 0:
                    print(f"     W_up: ", end="")
                    for dim in strong_up[:3]:
                        val = x[dim].item()
                        w = W_up_unit[dim].item()
                        print(f"x[{dim.item()}]={val:.1f}*{w:.0f} ", end="")
                    print(f"+ b={b_up_unit:.1f}")

                strong_gate = torch.where(torch.abs(W_gate_unit) > 0.5)[0]
                if len(strong_gate) > 0:
                    print(f"     W_gate: ", end="")
                    for dim in strong_gate[:3]:
                        val = x[dim].item()
                        w = W_gate_unit[dim].item()
                        print(f"x[{dim.item()}]={val:.1f}*{w:.1f} ", end="")
                    print(f"+ b={b_gate_unit:.1f}")

            print()
