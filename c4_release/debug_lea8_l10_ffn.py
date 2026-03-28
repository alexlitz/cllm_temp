"""Debug LEA 8 - check which L10 FFN unit is activating."""
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

# Check at AX marker
ctx = context + draft_tokens[:6]
pos = len(ctx) - 1

print("LEA 8 - L10 FFN Unit Activation Analysis")
print("=" * 70)
print()

# Capture L10 FFN input
l10_ffn_in = None

def l10_ffn_hook(module, input, output):
    global l10_ffn_in
    if isinstance(input, tuple):
        l10_ffn_in = input[0].detach().clone()
    else:
        l10_ffn_in = input.detach().clone()

model.blocks[10].ffn.register_forward_hook(l10_ffn_hook)

ctx_tensor = torch.tensor([ctx], dtype=torch.long)
with torch.no_grad():
    logits = model.forward(ctx_tensor)

if l10_ffn_in is not None:
    ffn = model.blocks[10].ffn
    x = l10_ffn_in[0, pos, :]

    # Compute FFN activation
    up = F.linear(x.unsqueeze(0), ffn.W_up, ffn.b_up)[0]
    gate = F.linear(x.unsqueeze(0), ffn.W_gate, ffn.b_gate)[0]
    hidden = F.silu(up) * gate

    # Find highly activated units
    activated = torch.where(hidden > 0.1)[0]

    print(f"Position {pos} (REG_AX marker):")
    print(f"  Total units with activation > 0.1: {len(activated)}")
    print()

    if len(activated) > 0:
        print("Top activated units:")
        top_acts = torch.topk(hidden, min(10, len(hidden)))
        for i in range(min(10, len(top_acts.values))):
            unit_idx = top_acts.indices[i].item()
            activation = top_acts.values[i].item()
            if activation > 0.01:
                # Check what this unit writes to
                w_down = ffn.W_down[:, unit_idx]
                output_dims = torch.where(torch.abs(w_down) > 0.01)[0]

                print(f"  Unit {unit_idx}: activation={activation:.3f}")

                # Check up and gate values
                up_val = up[unit_idx].item()
                gate_val = gate[unit_idx].item()
                print(f"    up={up_val:.2f}, gate={gate_val:.3f}, hidden={activation:.3f}")

                # What does it write to?
                if len(output_dims) > 0:
                    for out_dim in output_dims[:5]:  # Show first 5
                        out_idx = out_dim.item()
                        weight = w_down[out_idx].item()

                        # Identify dimension
                        dim_name = "?"
                        if BD.OUTPUT_LO <= out_idx < BD.OUTPUT_LO + 16:
                            dim_name = f"OUTPUT_LO[{out_idx - BD.OUTPUT_LO}]"
                        elif BD.OUTPUT_HI <= out_idx < BD.OUTPUT_HI + 16:
                            dim_name = f"OUTPUT_HI[{out_idx - BD.OUTPUT_HI}]"
                        elif BD.AX_CARRY_LO <= out_idx < BD.AX_CARRY_LO + 16:
                            dim_name = f"AX_CARRY_LO[{out_idx - BD.AX_CARRY_LO}]"
                        elif BD.AX_CARRY_HI <= out_idx < BD.AX_CARRY_HI + 16:
                            dim_name = f"AX_CARRY_HI[{out_idx - BD.AX_CARRY_HI}]"

                        print(f"      → {dim_name}: {weight:.3f}")
                print()

    # Check specific dimensions
    print("Input dimensions at marker:")
    print(f"  MARK_AX: {x[BD.MARK_AX].item():.3f}")
    print(f"  OP_LEA: {x[BD.OP_LEA].item():.3f}")
    print(f"  IS_BYTE: {x[BD.IS_BYTE].item():.3f}")
    print(f"  H1[AX]: {x[BD.H1 + 1].item():.3f}")
    print(f"  BYTE_INDEX_0: {x[BD.BYTE_INDEX_0].item():.3f}")
    print(f"  BYTE_INDEX_1: {x[BD.BYTE_INDEX_1].item():.3f}")

    ax_carry_lo = x[BD.AX_CARRY_LO:BD.AX_CARRY_LO+16]
    ax_carry_hi = x[BD.AX_CARRY_HI:BD.AX_CARRY_HI+16]
    print(f"  AX_CARRY: {torch.argmax(ax_carry_lo).item()} + 16*{torch.argmax(ax_carry_hi).item()}")
