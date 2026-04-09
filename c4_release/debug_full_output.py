#\!/usr/bin/env python3
"""Debug full OUTPUT_LO/HI values at byte 1 position."""
import os, sys
sys.path.insert(0, os.getcwd())
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, _SetDim, Token
from neural_vm.embedding import Opcode
from neural_vm.speculative import DraftVM
from neural_vm.constants import IMMEDIATE_SIZE, PADDING_SIZE

bytecode = [Opcode.IMM | (1 << 8), Opcode.PSH, Opcode.IMM | (0 << 8), Opcode.MUL, Opcode.EXIT]

def build_context(bytecode):
    context = [Token.CODE_START]
    for instr in bytecode:
        op, imm = instr & 0xFF, instr >> 8
        context.append(op)
        for i in range(IMMEDIATE_SIZE):
            context.append((imm >> (i * 8)) & 0xFF)
        for _ in range(PADDING_SIZE):
            context.append(0)
    context.extend([Token.CODE_END, Token.DATA_START, Token.DATA_END])
    return context

context = build_context(bytecode)
model = AutoregressiveVM(vocab_size=512, n_layers=16, n_heads=8, ffn_hidden=4096)
set_vm_weights(model)
model.eval()
BD = _SetDim

vm = DraftVM(bytecode)
all_tokens = context[:]
for i in range(3):
    vm.step()
    all_tokens.extend(vm.draft_tokens())
vm.step()
step3_tokens = vm.draft_tokens()

activations = {}
def hook(name):
    def fn(module, input, output):
        out = output[0] if isinstance(output, tuple) else output
        activations[name] = out.detach().clone()
    return fn

for i, block in enumerate(model.blocks):
    block.attn.register_forward_hook(hook(f'L{i}_attn'))
    block.ffn.register_forward_hook(hook(f'L{i}_ffn'))

# At byte 1 position (predicting byte 2)
test_context = all_tokens + step3_tokens[:13]
token_ids = torch.tensor([test_context], dtype=torch.long)

with torch.no_grad():
    logits = model(token_ids)

print("=== Full OUTPUT_LO/HI at byte 1 position ===")
h_before = activations['L10_attn'][0, -1, :]
h_after = activations['L10_ffn'][0, -1, :]

print("\nAfter L10 attn (passthrough):")
lo = h_before[BD.OUTPUT_LO:BD.OUTPUT_LO+16].tolist()
hi = h_before[BD.OUTPUT_HI:BD.OUTPUT_HI+16].tolist()
print(f"  LO: {[f'{v:.2f}' for v in lo]}")
print(f"  HI: {[f'{v:.2f}' for v in hi]}")
print(f"  LO argmax={torch.argmax(h_before[BD.OUTPUT_LO:BD.OUTPUT_LO+16]).item()}, HI argmax={torch.argmax(h_before[BD.OUTPUT_HI:BD.OUTPUT_HI+16]).item()}")

print("\nAfter L10 FFN (carry application):")
lo = h_after[BD.OUTPUT_LO:BD.OUTPUT_LO+16].tolist()
hi = h_after[BD.OUTPUT_HI:BD.OUTPUT_HI+16].tolist()
print(f"  LO: {[f'{v:.2f}' for v in lo]}")
print(f"  HI: {[f'{v:.2f}' for v in hi]}")
lo_idx = torch.argmax(h_after[BD.OUTPUT_LO:BD.OUTPUT_LO+16]).item()
hi_idx = torch.argmax(h_after[BD.OUTPUT_HI:BD.OUTPUT_HI+16]).item()
print(f"  LO argmax={lo_idx}, HI argmax={hi_idx} → value={lo_idx + hi_idx * 16}")

predicted = logits[0, -1, :].argmax().item()
expected = step3_tokens[13]
print(f"\n  Expected: {expected}, Predicted: {predicted}")
