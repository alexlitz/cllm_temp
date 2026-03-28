"""Debug LEA 8 - precisely capture OUTPUT before/after L10 FFN."""
import sys
for mod in list(sys.modules.keys()):
    if 'neural_vm' in mod:
        del sys.modules[mod]

import torch
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

print("LEA 8 - L10 FFN Precise OUTPUT Trace")
print("=" * 70)
print()

# Capture before/after L10 FFN
l10_ffn_input = None
l10_ffn_output = None

def l10_ffn_hook(module, input, output):
    global l10_ffn_input, l10_ffn_output
    if isinstance(input, tuple):
        l10_ffn_input = input[0].detach().clone()
    else:
        l10_ffn_input = input.detach().clone()
    l10_ffn_output = output.detach().clone()

# Register hook on the FFN module (which is EfficientALU_L10_Neural)
model.blocks[10].ffn.register_forward_hook(l10_ffn_hook)

ctx_tensor = torch.tensor([ctx], dtype=torch.long)
with torch.no_grad():
    logits = model.forward(ctx_tensor)

print(f"Position {pos} (REG_AX marker):")
print()

if l10_ffn_input is not None:
    x_in = l10_ffn_input[0, pos, :]
    output_lo_in = x_in[BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    output_hi_in = x_in[BD.OUTPUT_HI:BD.OUTPUT_HI+16]
    lo_in = torch.argmax(output_lo_in).item()
    hi_in = torch.argmax(output_hi_in).item()
    val_in = lo_in + 16 * hi_in

    print(f"OUTPUT BEFORE L10 FFN: {val_in}")
    print(f"  lo={lo_in}, hi={hi_in}")
    print(f"  lo_probs: {output_lo_in.tolist()}")
    print()

if l10_ffn_output is not None:
    x_out = l10_ffn_output[0, pos, :]
    output_lo_out = x_out[BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    output_hi_out = x_out[BD.OUTPUT_HI:BD.OUTPUT_HI+16]
    lo_out = torch.argmax(output_lo_out).item()
    hi_out = torch.argmax(output_hi_out).item()
    val_out = lo_out + 16 * hi_out

    print(f"OUTPUT AFTER L10 FFN: {val_out}")
    print(f"  lo={lo_out}, hi={hi_out}")
    print(f"  lo_probs: {output_lo_out.tolist()}")
    print()

    # Check what changed
    delta = x_out - x_in
    output_lo_delta = delta[BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    output_hi_delta = delta[BD.OUTPUT_HI:BD.OUTPUT_HI+16]

    print("Delta (output - input):")
    print(f"  OUTPUT_LO: {output_lo_delta.tolist()}")
    print(f"  OUTPUT_HI: {output_hi_delta.tolist()}")
    print()

    max_delta_lo = torch.max(torch.abs(output_lo_delta)).item()
    max_delta_hi = torch.max(torch.abs(output_hi_delta)).item()

    if max_delta_lo > 0.01 or max_delta_hi > 0.01:
        print(f"❌ L10 FFN CHANGED OUTPUT from {val_in} to {val_out}")
        print(f"   Max delta: lo={max_delta_lo:.3f}, hi={max_delta_hi:.3f}")
    else:
        print(f"✓ L10 FFN did NOT change OUTPUT (max delta < 0.01)")

print()
print("Module type:", type(model.blocks[10].ffn).__name__)
