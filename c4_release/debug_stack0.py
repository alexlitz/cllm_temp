#\!/usr/bin/env python3
"""Debug STACK0 byte 0 prediction for IMM 1, PSH, IMM 0, MUL, EXIT."""
import os, sys
sys.path.insert(0, os.getcwd())
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, _SetDim, Token
from neural_vm.embedding import Opcode
from neural_vm.speculative import DraftVM
from neural_vm.constants import IMMEDIATE_SIZE, PADDING_SIZE

# Program: IMM 1, PSH, IMM 0, MUL, EXIT
bytecode = [Opcode.IMM | (1 << 8), Opcode.PSH, Opcode.IMM | (0 << 8), Opcode.MUL, Opcode.EXIT]

vm = DraftVM(bytecode)
vm.step(); draft_step0 = vm.draft_tokens()
vm.step(); draft_step1 = vm.draft_tokens()
vm.step(); draft_step2 = vm.draft_tokens()

print("Step 1 (PSH): STACK0 bytes =", draft_step1[21:25])
print("Step 2 (IMM): STACK0 bytes =", draft_step2[21:25])
print("Expected STACK0 byte 0 at step 2:", draft_step2[21])

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
# Context up to step 2 token 20 (STACK0 marker) - predict token 21 (STACK0 byte 0)
full_context = context + draft_step0 + draft_step1 + draft_step2[:21]

model = AutoregressiveVM(vocab_size=512, n_layers=16, n_heads=8, ffn_hidden=4096)
set_vm_weights(model)
model.eval()
BD = _SetDim

activations = {}
def hook_fn(name):
    def fn(module, input, output):
        activations[name] = output[0].detach().clone() if isinstance(output, tuple) else output.detach().clone()
    return fn

for i, block in enumerate(model.blocks):
    block.attn.register_forward_hook(hook_fn(f'L{i}_attn'))
    block.ffn.register_forward_hook(hook_fn(f'L{i}_ffn'))

token_ids = torch.tensor([full_context], dtype=torch.long)
with torch.no_grad():
    logits = model(token_ids)

predicted = logits[0, -1, :].argmax().item()
expected = draft_step2[21]
print(f"\nPrediction for STACK0 byte 0:")
print(f"  Expected: {expected}")
print(f"  Predicted: {predicted}")
print(f"  Match: {predicted == expected}")

# Analyze at STACK0 marker position
h = activations['L15_ffn'][0, -1, :]
print(f"\n--- At STACK0 marker (last token) ---")
print(f"MARK_STACK0 = {h[BD.MARK_STACK0].item():.3f}")
print(f"IS_BYTE = {h[BD.IS_BYTE].item():.3f}")
print(f"PSH_AT_SP = {h[BD.PSH_AT_SP].item():.3f}")

# Check EMBED and OUTPUT
embed_lo_max = h[BD.EMBED_LO:BD.EMBED_LO+16].argmax().item()
embed_hi_max = h[BD.EMBED_HI:BD.EMBED_HI+16].argmax().item()
out_lo_max = h[BD.OUTPUT_LO:BD.OUTPUT_LO+16].argmax().item()
out_hi_max = h[BD.OUTPUT_HI:BD.OUTPUT_HI+16].argmax().item()
print(f"\nEMBED: lo={embed_lo_max}, hi={embed_hi_max}")
print(f"OUTPUT: lo={out_lo_max}, hi={out_hi_max}")

# Expected STACK0 byte 0 = 1 -> lo=1, hi=0
print(f"\nExpected: lo=1, hi=0 (for STACK0 byte 0 = 1)")

# Trace OUTPUT through layers
print(f"\n--- OUTPUT at STACK0 marker through layers ---")
for layer in [3, 6, 10, 15]:
    h = activations[f'L{layer}_ffn'][0, -1, :]
    out_lo_max = h[BD.OUTPUT_LO:BD.OUTPUT_LO+16].argmax().item()
    out_hi_max = h[BD.OUTPUT_HI:BD.OUTPUT_HI+16].argmax().item()
    out_lo_val = h[BD.OUTPUT_LO:BD.OUTPUT_LO+16].max().item()
    print(f"L{layer:2d}: OUTPUT lo={out_lo_max} ({out_lo_val:.3f}), hi={out_hi_max}")
