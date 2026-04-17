#\!/usr/bin/env python3
"""Debug L10 at SP marker."""
import os, sys
sys.path.insert(0, os.getcwd())
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, _SetDim, Token
from neural_vm.embedding import Opcode
from neural_vm.speculative import DraftVM
from neural_vm.constants import IMMEDIATE_SIZE, PADDING_SIZE

bytecode = [Opcode.IMM | (0 << 8), Opcode.PSH, Opcode.IMM | (0 << 8), Opcode.MUL, Opcode.EXIT]
vm = DraftVM(bytecode)
vm.step(); draft_step0 = vm.draft_tokens()
vm.step(); draft_step1 = vm.draft_tokens()
vm.step(); draft_step2 = vm.draft_tokens()

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
full_context = context + draft_step0 + draft_step1 + draft_step2[:11]

model = AutoregressiveVM(vocab_size=512, n_layers=16, n_heads=8, ffn_hidden=4096)
set_vm_weights(model)
model.eval()
BD = _SetDim

activations = {}
def hook_fn(name):
    def fn(module, input, output):
        activations[name] = output[0].detach().clone() if isinstance(output, tuple) else output.detach().clone()
    return fn

model.blocks[9].ffn.register_forward_hook(hook_fn('L9_ffn'))
model.blocks[10].attn.register_forward_hook(hook_fn('L10_attn'))
model.blocks[10].ffn.register_forward_hook(hook_fn('L10_ffn'))

token_ids = torch.tensor([full_context], dtype=torch.long)
with torch.no_grad():
    logits = model(token_ids)

print("--- At SP marker (last token) ---")
print()

# After L9 FFN (before L10)
h = activations['L9_ffn'][0, -1, :]
out_lo_9 = [h[BD.OUTPUT_LO + i].item() for i in range(16)]
print(f"After L9 FFN - OUTPUT_LO: max={out_lo_9.index(max(out_lo_9))} ({max(out_lo_9):.3f})")
print(f"  Significant: {[(i, f'{v:.3f}') for i, v in enumerate(out_lo_9) if abs(v) > 0.1]}")

# After L10 attn
h = activations['L10_attn'][0, -1, :]
out_lo_10a = [h[BD.OUTPUT_LO + i].item() for i in range(16)]
print(f"\nAfter L10 attn - OUTPUT_LO: max={out_lo_10a.index(max(out_lo_10a))} ({max(out_lo_10a):.3f})")
print(f"  Significant: {[(i, f'{v:.3f}') for i, v in enumerate(out_lo_10a) if abs(v) > 0.1]}")

# After L10 FFN
h = activations['L10_ffn'][0, -1, :]
out_lo_10f = [h[BD.OUTPUT_LO + i].item() for i in range(16)]
print(f"\nAfter L10 FFN - OUTPUT_LO: max={out_lo_10f.index(max(out_lo_10f))} ({max(out_lo_10f):.3f})")
print(f"  Significant: {[(i, f'{v:.3f}') for i, v in enumerate(out_lo_10f) if abs(v) > 0.1]}")

# Check what L10 attention added
delta_attn = [out_lo_10a[i] - out_lo_9[i] for i in range(16)]
print(f"\nL10 attn delta: {[(i, f'{v:.3f}') for i, v in enumerate(delta_attn) if abs(v) > 0.1]}")

# Check what L10 FFN added
delta_ffn = [out_lo_10f[i] - out_lo_10a[i] for i in range(16)]
print(f"L10 FFN delta: {[(i, f'{v:.3f}') for i, v in enumerate(delta_ffn) if abs(v) > 0.1]}")

# Check PSH_AT_SP and other flags at SP marker
print(f"\n--- Flags at SP marker ---")
print(f"MARK_SP = {h[BD.MARK_SP].item():.3f}")
print(f"IS_BYTE = {h[BD.IS_BYTE].item():.3f}")
print(f"H1[SP] = {h[BD.H1 + 2].item():.3f}")
print(f"PSH_AT_SP = {h[BD.PSH_AT_SP].item():.3f}")
print(f"OP_PSH = {h[BD.OP_PSH].item():.3f}")
