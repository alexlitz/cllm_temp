"""Debug LEA 8 - check OUTPUT_HI at final prediction step."""
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

# Check at OUTPUT prediction (after AX marker)
ctx = context + draft_tokens[:6]
pos = len(ctx) - 1

print("LEA 8 - OUTPUT_HI at Prediction Step")
print("=" * 70)
print()

# Capture L15 input (final layer before logits)
l15_in = None

def l15_hook(module, input, output):
    global l15_in
    if isinstance(input, tuple):
        l15_in = input[0].detach().clone()
    else:
        l15_in = input.detach().clone()

model.blocks[15].attn.register_forward_hook(l15_hook)

ctx_tensor = torch.tensor([ctx], dtype=torch.long)
with torch.no_grad():
    logits = model.forward(ctx_tensor)

predicted = logits[0, -1, :].argmax().item()

print(f"Position {pos} (REG_AX marker):")
print(f"Expected: 8 (OUTPUT_HI[0])")
print(f"Predicted: {predicted}")
print()

if l15_in is not None:
    x = l15_in[0, pos, :]
    output_hi = x[BD.OUTPUT_HI:BD.OUTPUT_HI+16]

    print("OUTPUT_HI values at L15 input:")
    for i, val in enumerate(output_hi):
        print(f"  OUTPUT_HI[{i}]: {val.item():.1f}")
    print()

    mean_val = output_hi.mean().item()
    std_val = output_hi.std().item()
    max_idx = output_hi.argmax().item()
    max_val = output_hi.max().item()

    print(f"Mean: {mean_val:.1f}, Std: {std_val:.3f}")
    print(f"Max: OUTPUT_HI[{max_idx}] = {max_val:.1f}")

    if std_val < 0.1:
        print("❌ All OUTPUT_HI dims are equal - this causes argmax to pick first!")
        print("   This is why we predict OUTPUT_HI[0] + BYTECODE_START = 1 + 256 = 257")
    else:
        print("✅ OUTPUT_HI dims are different")
        print(f"   Should predict OUTPUT_HI[{max_idx}] + BYTECODE_START = {max_idx} + 256 = {max_idx + 256}")
