#!/usr/bin/env python3
"""Debug MEM_STORE flag through layers."""
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
vm.step()
step1_tokens = vm.draft_tokens()
all_tokens.extend(step1_tokens)
vm.step()
step2_tokens = vm.draft_tokens()

print("=== Step 2 token layout ===")
print("Index 25: STACK0 marker =", step2_tokens[25])
print("Index 26: MEM addr byte 0 =", step2_tokens[26])
print("Index 27: MEM addr byte 1 =", step2_tokens[27])
print()

# Test at different positions
activations = {}
def hook(name):
    def fn(module, input, output):
        out = output[0] if isinstance(output, tuple) else output
        activations[name] = out.detach().clone()
    return fn

for i, block in enumerate(model.blocks):
    block.attn.register_forward_hook(hook(f'L{i}_attn'))
    block.ffn.register_forward_hook(hook(f'L{i}_ffn'))

# Test at STACK0 marker position (index 25)
print("=== At STACK0 marker position (index 25) ===")
test_context = all_tokens + step2_tokens[:26]  # Include STACK0 marker
token_ids = torch.tensor([test_context], dtype=torch.long)
with torch.no_grad():
    logits = model(token_ids)

# Trace MEM_STORE at STACK0 marker position
print("MEM_STORE trace at STACK0 marker:")
for layer in range(8):
    h_attn = activations[f'L{layer}_attn'][0, -1, :]
    h_ffn = activations[f'L{layer}_ffn'][0, -1, :]
    ms_attn = h_attn[BD.MEM_STORE].item()
    ms_ffn = h_ffn[BD.MEM_STORE].item()
    delta = ms_ffn - ms_attn
    flag = " <-- CHANGE" if abs(delta) > 0.01 else ""
    print(f"L{layer}: attn={ms_attn:6.3f}, ffn={ms_ffn:6.3f}{flag}")

print("\nOP_PSH at AX marker (should be ~1.0):")
# Find AX marker position - it's at step2_tokens[5] = 258 (REG_AX)
# Format: PC_marker(0) + PC_bytes(1-4) + AX_marker(5) + AX_bytes(6-9) + ...
ax_marker_pos = len(all_tokens) + 5  # Position in full context
print(f"  len(all_tokens) = {len(all_tokens)}")
print(f"  ax_marker_pos = {ax_marker_pos}")
print(f"  len(test_context) = {len(test_context)}")
print(f"  Token at ax_marker_pos: {test_context[ax_marker_pos]}")
print(f"  step2_tokens[10] = {step2_tokens[10]}")
h = activations['L5_ffn'][0, ax_marker_pos, :]
print(f"  OP_PSH at AX marker: {h[BD.OP_PSH].item():.3f}")

# Check OPCODE_BYTE values at AX marker
print(f"\nOPCODE_BYTE at AX marker (for PSH=13=0x0D, expect LO[13]=1, HI[0]=1):")
h_before = activations['L5_attn'][0, ax_marker_pos, :]
lo = h_before[BD.OPCODE_BYTE_LO:BD.OPCODE_BYTE_LO+16].tolist()
hi = h_before[BD.OPCODE_BYTE_HI:BD.OPCODE_BYTE_HI+16].tolist()
print(f"  OPCODE_BYTE_LO: {[f'{v:.2f}' for v in lo]}")
print(f"  OPCODE_BYTE_HI: {[f'{v:.2f}' for v in hi]}")
print(f"  LO argmax: {lo.index(max(lo))}, HI argmax: {hi.index(max(hi))}")
print(f"  MARK_AX: {h_before[BD.MARK_AX].item():.3f}")

print("\n=== At MEM addr byte 0 position (index 26) ===")
test_context = all_tokens + step2_tokens[:27]  # Predicting byte 1
token_ids = torch.tensor([test_context], dtype=torch.long)
with torch.no_grad():
    logits = model(token_ids)

print("MEM_STORE trace at MEM addr byte 0 (all layers):")
for layer in range(16):
    h_attn = activations[f'L{layer}_attn'][0, -1, :]
    h_ffn = activations[f'L{layer}_ffn'][0, -1, :]
    ms_attn = h_attn[BD.MEM_STORE].item()
    ms_ffn = h_ffn[BD.MEM_STORE].item()
    delta = ms_ffn - ms_attn
    flag = " <-- CHANGE" if abs(delta) > 0.01 else ""
    print(f"L{layer:2d}: attn={ms_attn:6.3f}, ffn={ms_ffn:6.3f}{flag}")

# Check STACK0 marker in context for MEM_STORE
stack0_marker_pos = len(all_tokens) + 25
h = activations['L6_attn'][0, stack0_marker_pos, :]
print(f"\nMEM_STORE at STACK0 marker after L6 attn: {h[BD.MEM_STORE].item():.3f}")
h = activations['L7_ffn'][0, stack0_marker_pos, :]
print(f"MEM_STORE at STACK0 marker after L7 FFN: {h[BD.MEM_STORE].item():.3f}")
