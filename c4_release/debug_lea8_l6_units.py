"""Debug LEA 8 - check which L6 FFN units activate to produce OUTPUT=105."""
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

print("LEA 8 - L6 FFN Unit Activation Analysis")
print("=" * 70)
print()

# Capture L6 FFN input
l6_ffn_in = None

def l6_ffn_hook(module, input, output):
    global l6_ffn_in
    if isinstance(input, tuple):
        l6_ffn_in = input[0].detach().clone()
    else:
        l6_ffn_in = input.detach().clone()

model.blocks[6].ffn.register_forward_hook(l6_ffn_hook)

ctx_tensor = torch.tensor([ctx], dtype=torch.long)
with torch.no_grad():
    logits = model.forward(ctx_tensor)

if l6_ffn_in is not None:
    ffn = model.blocks[6].ffn
    x = l6_ffn_in[0, pos, :]

    print(f"Position {pos} (REG_AX marker):")
    print()

    # Manually compute FFN
    up = F.linear(x.unsqueeze(0), ffn.W_up, ffn.b_up)[0]
    gate = F.linear(x.unsqueeze(0), ffn.W_gate, ffn.b_gate)[0]
    hidden = F.silu(up) * gate

    # Find units writing to OUTPUT_LO[9] or OUTPUT_HI[6]
    W_down = ffn.W_down

    # Check OUTPUT_LO[9]
    units_lo9 = torch.where(torch.abs(W_down[BD.OUTPUT_LO + 9, :]) > 0.01)[0]
    print(f"Units writing to OUTPUT_LO[9]: {len(units_lo9)}")
    for unit_idx in units_lo9:
        if hidden[unit_idx].item() > 0.1:
            weight = W_down[BD.OUTPUT_LO + 9, unit_idx].item()
            contribution = hidden[unit_idx].item() * weight
            print(f"  Unit {unit_idx}: hidden={hidden[unit_idx].item():.3f}, weight={weight:.3f}, contrib={contribution:.3f}")
            print(f"    up={up[unit_idx].item():.2f}, gate={gate[unit_idx].item():.3f}")

            # Check which dimensions this unit reads
            W_up_unit = ffn.W_up[unit_idx, :]
            W_gate_unit = ffn.W_gate[unit_idx, :]
            strong_up = torch.where(torch.abs(W_up_unit) > 50)[0]
            strong_gate = torch.where(torch.abs(W_gate_unit) > 0.5)[0]

            if len(strong_up) > 0:
                print(f"    Strong W_up dims: ", end="")
                for dim in strong_up[:5]:
                    print(f"{dim.item()}:{x[dim].item():.2f} ", end="")
                print()
            if len(strong_gate) > 0:
                print(f"    Strong W_gate dims: ", end="")
                for dim in strong_gate[:5]:
                    print(f"{dim.item()}:{x[dim].item():.2f} ", end="")
                print()

    print()

    # Check OUTPUT_HI[6]
    units_hi6 = torch.where(torch.abs(W_down[BD.OUTPUT_HI + 6, :]) > 0.01)[0]
    print(f"Units writing to OUTPUT_HI[6]: {len(units_hi6)}")
    for unit_idx in units_hi6:
        if hidden[unit_idx].item() > 0.1:
            weight = W_down[BD.OUTPUT_HI + 6, unit_idx].item()
            contribution = hidden[unit_idx].item() * weight
            print(f"  Unit {unit_idx}: hidden={hidden[unit_idx].item():.3f}, weight={weight:.3f}, contrib={contribution:.3f}")
            print(f"    up={up[unit_idx].item():.2f}, gate={gate[unit_idx].item():.3f}")

            # Check which dimensions this unit reads
            W_up_unit = ffn.W_up[unit_idx, :]
            W_gate_unit = ffn.W_gate[unit_idx, :]
            strong_up = torch.where(torch.abs(W_up_unit) > 50)[0]
            strong_gate = torch.where(torch.abs(W_gate_unit) > 0.5)[0]

            if len(strong_up) > 0:
                print(f"    Strong W_up dims: ", end="")
                for dim in strong_up[:5]:
                    print(f"{dim.item()}:{x[dim].item():.2f} ", end="")
                print()
            if len(strong_gate) > 0:
                print(f"    Strong W_gate dims: ", end="")
                for dim in strong_gate[:5]:
                    print(f"{dim.item()}:{x[dim].item():.2f} ", end="")
                print()

    print()
    print("Input flags:")
    print(f"  MARK_AX: {x[BD.MARK_AX].item():.3f}")
    print(f"  MARK_PC: {x[BD.MARK_PC].item():.3f}")
    print(f"  OP_LEA: {x[BD.OP_LEA].item():.3f}")
    print(f"  OP_JSR: {x[BD.OP_JSR].item():.3f}")
    print(f"  OP_ENT: {x[BD.OP_ENT].item():.3f}")
    print(f"  OP_LEV: {x[BD.OP_LEV].item():.3f}")
    print()
    print(f"  TEMP[0]: {x[BD.TEMP + 0].item():.3f}")
    print(f"  CMP[2]: {x[BD.CMP + 2].item():.3f}")
