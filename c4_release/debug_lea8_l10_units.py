"""Debug LEA 8 - find which L10 units are writing huge values to OUTPUT_HI."""
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

print("LEA 8 - L10 Units Writing to OUTPUT_HI")
print("=" * 70)
print()

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

    # Manually compute FFN
    up = F.linear(x.unsqueeze(0), ffn.W_up, ffn.b_up)[0]
    gate = F.linear(x.unsqueeze(0), ffn.W_gate, ffn.b_gate)[0]
    hidden = F.silu(up) * gate

    W_down = ffn.W_down

    print(f"Position {pos} (REG_AX marker):")
    print()

    # Find units contributing significantly to OUTPUT_HI
    print("Units writing significantly to any OUTPUT_HI dim:")
    print()

    active_units = []
    for unit_idx in range(hidden.shape[0]):
        h = hidden[unit_idx].item()
        if abs(h) > 1.0:
            # Check contribution to OUTPUT_HI
            output_hi_weights = W_down[BD.OUTPUT_HI:BD.OUTPUT_HI+16, unit_idx]
            max_contrib = torch.max(torch.abs(output_hi_weights * h)).item()
            if max_contrib > 100:
                active_units.append((unit_idx, h, output_hi_weights, max_contrib))

    print(f"Found {len(active_units)} units with large OUTPUT_HI contributions:")
    print()

    for unit_idx, h, output_hi_weights, max_contrib in sorted(active_units, key=lambda x: -x[3])[:5]:
        print(f"Unit {unit_idx}: hidden={h:.1f}, max_contrib={max_contrib:.1f}")

        # Show which OUTPUT_HI dims this unit writes to
        for i in range(16):
            w = output_hi_weights[i].item()
            contrib = h * w
            if abs(contrib) > 100:
                print(f"  OUTPUT_HI[{i}]: w={w:.3f}, contrib={contrib:.1f}")

        # Analyze what activated this unit
        W_up_unit = ffn.W_up[unit_idx, :]
        b_up = ffn.b_up[unit_idx].item()
        W_gate_unit = ffn.W_gate[unit_idx, :]
        b_gate = ffn.b_gate[unit_idx].item()

        # Compute up value
        up_val = b_up
        strong_up = []
        for dim in range(x.shape[0]):
            w = W_up_unit[dim].item()
            if abs(w) > 0.5:
                val = x[dim].item()
                c = val * w
                up_val += c
                if abs(c) > 5.0:
                    strong_up.append((dim, val, w, c))

        # Compute gate value
        gate_val = b_gate
        strong_gate = []
        for dim in range(x.shape[0]):
            w = W_gate_unit[dim].item()
            if abs(w) > 0.5:
                val = x[dim].item()
                c = val * w
                gate_val += c
                if abs(c) > 0.5:
                    strong_gate.append((dim, val, w, c))

        print(f"  up={up_val:.1f} (bias={b_up:.1f}), gate={gate_val:.1f} (bias={b_gate:.1f})")

        if len(strong_up) > 0:
            print(f"  Strong W_up:")
            for dim, val, w, c in sorted(strong_up, key=lambda x: -abs(x[3]))[:5]:
                print(f"    x[{dim}]={val:.1f} * {w:.2f} = {c:.1f}")

        if len(strong_gate) > 0:
            print(f"  Strong W_gate:")
            for dim, val, w, c in sorted(strong_gate, key=lambda x: -abs(x[3]))[:5]:
                print(f"    x[{dim}]={val:.1f} * {w:.2f} = {c:.1f}")

        print()

    # Show key input dimensions
    print("Key L10 input dimensions:")
    print(f"  OUTPUT_HI[0-3]: {x[BD.OUTPUT_HI:BD.OUTPUT_HI+4].tolist()}")
    print(f"  OUTPUT_LO[0-3]: {x[BD.OUTPUT_LO:BD.OUTPUT_LO+4].tolist()}")
    print(f"  CARRY[0-3]: {x[BD.CARRY:BD.CARRY+4].tolist()}")
