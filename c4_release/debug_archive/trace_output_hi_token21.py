"""Trace OUTPUT_HI through all layers for token 21."""
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

bytecode = [Opcode.JMP | (16 << 8)]
context = build_context(bytecode)

draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

# Build sequence up to token 20
current_context = context[:]
for i in range(21):
    current_context.append(draft_tokens[i])

ctx_tensor = torch.tensor([current_context], dtype=torch.long)
pos = len(current_context) - 1  # position 29

# Capture OUTPUT_HI after each layer
layer_outputs = {}

for layer_idx in range(16):
    def make_hook(idx):
        def hook(module, input, output):
            layer_outputs[idx] = output.detach().clone()
        return hook

    model.blocks[layer_idx].register_forward_hook(make_hook(layer_idx))

# Forward pass
with torch.no_grad():
    logits = model.forward(ctx_tensor)

predicted = torch.argmax(logits[0, -1]).item()

print("OUTPUT_HI Trace for Token 21 (ST_b0)")
print("=" * 70)
print(f"Position: {pos}, Predicted: {predicted} (expected: 0)")
print()

print("OUTPUT_HI[2] value after each layer:")
print("(Looking for where it becomes > 0)")
print()

for layer_idx in range(16):
    if layer_idx in layer_outputs:
        output_hi = layer_outputs[layer_idx][0, pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16]
        argmax = torch.argmax(output_hi).item()
        max_val = torch.max(output_hi).item()
        val_2 = output_hi[2].item()

        marker = ""
        if val_2 > 1.0:
            marker = " ← HIGH VALUE"
        elif val_2 > 0.1:
            marker = " ← increasing"

        print(f"L{layer_idx:2d}: argmax={argmax}, max={max_val:6.2f}, OUTPUT_HI[2]={val_2:6.2f}{marker}")

print()
print("=" * 70)

# Check final logits
print("Final prediction analysis:")
final_output_hi = layer_outputs[15][0, pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16]
print(f"OUTPUT_HI argmax: {torch.argmax(final_output_hi).item()}")
print(f"OUTPUT_HI values: {final_output_hi.tolist()}")
