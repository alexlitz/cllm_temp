"""Debug LEA 8 - find which L9 units write to OUTPUT_HI."""
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

print("LEA 8 - L9 FFN Units Writing to OUTPUT_HI")
print("=" * 70)
print()

# Capture L9 FFN input
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

    print(f"Position {pos} (REG_AX marker):")
    print()

    # Manually compute FFN activations
    up = F.linear(x.unsqueeze(0), ffn.W_up, ffn.b_up)[0]
    gate = F.linear(x.unsqueeze(0), ffn.W_gate, ffn.b_gate)[0]
    hidden = F.silu(up) * gate

    W_down = ffn.W_down

    # Check which units write to ALL OUTPUT_HI dimensions
    print("Looking for units that write to multiple OUTPUT_HI dims...")
    print()

    # Find units that write to OUTPUT_HI
    output_hi_writers = {}
    for unit_idx in range(hidden.shape[0]):
        if hidden[unit_idx].item() > 0.1:
            # Check how many OUTPUT_HI dims this unit writes to
            weights_to_hi = W_down[BD.OUTPUT_HI:BD.OUTPUT_HI+16, unit_idx]
            nonzero_hi = torch.where(torch.abs(weights_to_hi) > 0.01)[0]

            if len(nonzero_hi) > 5:  # Writes to many dims
                output_hi_writers[unit_idx] = {
                    'hidden': hidden[unit_idx].item(),
                    'up': up[unit_idx].item(),
                    'gate': gate[unit_idx].item(),
                    'num_dims': len(nonzero_hi),
                    'weights': weights_to_hi.tolist()
                }

    if len(output_hi_writers) > 0:
        print(f"Found {len(output_hi_writers)} units writing to many OUTPUT_HI dims:")
        print()

        for unit_idx, info in sorted(output_hi_writers.items(), key=lambda x: -x[1]['hidden'])[:5]:
            print(f"Unit {unit_idx}:")
            print(f"  hidden={info['hidden']:.3f}, up={info['up']:.2f}, gate={info['gate']:.3f}")
            print(f"  Writes to {info['num_dims']} OUTPUT_HI dims")

            # Show contribution
            total_contrib = info['hidden'] * sum(abs(w) for w in info['weights'])
            print(f"  Total contribution: {total_contrib:.1f}")

            # Check what activates this unit (strong W_up dims)
            W_up_unit = ffn.W_up[unit_idx, :]
            W_gate_unit = ffn.W_gate[unit_idx, :]

            strong_up = torch.where(torch.abs(W_up_unit) > 50)[0]
            strong_gate = torch.where(torch.abs(W_gate_unit) > 0.5)[0]

            if len(strong_up) > 0:
                print(f"  Strong W_up dims ({len(strong_up)}): ", end="")
                for dim in strong_up[:5]:
                    val = x[dim].item()
                    weight = W_up_unit[dim].item()
                    print(f"{dim.item()}({val:.1f}*{weight:.0f}) ", end="")
                print()

            if len(strong_gate) > 0:
                print(f"  Strong W_gate dims ({len(strong_gate)}): ", end="")
                for dim in strong_gate[:5]:
                    val = x[dim].item()
                    weight = W_gate_unit[dim].item()
                    print(f"{dim.item()}({val:.1f}*{weight:.1f}) ", end="")
                print()
            print()
    else:
        print("No units found writing to many OUTPUT_HI dims")
        print("This is unexpected - investigating further...")
