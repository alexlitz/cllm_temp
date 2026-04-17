#\!/usr/bin/env python3
"""Debug the L3 default cancel unit."""
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

ffn10 = model.blocks[10].ffn

# The cancel unit should be after the byte 1 carry units
# Byte 0: 33 units (16 LO + 16 HI + 1 carry_1 flag)
# Byte 1: 32 units (16 LO + 16 HI)
# Cancel units: 2 (SP and BP)
# So cancel starts at 1600 + 33 + 32 = 1665

print("=== L10 FFN cancel unit weights ===")
for i, (name, reg_idx) in enumerate([("SP", 2), ("BP", 3)]):
    unit = 1665 + i
    print(f"\nCancel unit {unit} ({name}):")
    print(f"  W_up[H1+{reg_idx}]: {ffn10.W_up[unit, BD.H1 + reg_idx].item():.0f}")
    print(f"  W_up[BYTE_INDEX_1]: {ffn10.W_up[unit, BD.BYTE_INDEX_1].item():.0f}")
    print(f"  W_up[CMP+3]: {ffn10.W_up[unit, BD.CMP + 3].item():.0f}")
    print(f"  b_up: {ffn10.b_up[unit].item():.0f}")
    print(f"  b_gate: {ffn10.b_gate[unit].item():.2f}")
    print(f"  W_down[OUTPUT_LO+1]: {ffn10.W_down[BD.OUTPUT_LO + 1, unit].item():.4f}")

# Also check what the byte 1 carry units do at 1633
print("\n=== Byte 1 carry units (1633+) for reference ===")
for k in range(3):
    unit = 1633 + k
    print(f"  Unit {unit}: W_up sum = {ffn10.W_up[unit, :].sum().item():.0f}, b_up = {ffn10.b_up[unit].item():.0f}")

# Run the model and check activations at byte 1 position
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

h = activations['L10_attn'][0, -1, :]  # Before FFN
print("\n=== Cancel unit activation inputs (after L10 attn) ===")
print(f"  H1[SP]: {h[BD.H1 + 2].item():.3f}")
print(f"  H1[BP]: {h[BD.H1 + 3].item():.3f}")
print(f"  BYTE_INDEX_1: {h[BD.BYTE_INDEX_1].item():.3f}")
print(f"  CMP[3]: {h[BD.CMP + 3].item():.3f}")

# Compute expected cancel activation
S = 100.0
T_cancel = 2.5
h1_sp = h[BD.H1 + 2].item()
bi1 = h[BD.BYTE_INDEX_1].item()
cmp3 = h[BD.CMP + 3].item()
up_act = S * (h1_sp + bi1 + cmp3) - S * T_cancel
print(f"\nExpected SP cancel up activation: {S}*({h1_sp:.2f} + {bi1:.2f} + {cmp3:.2f}) - {S}*{T_cancel} = {up_act:.1f}")
print(f"Should fire: {up_act > 0}")

# Check OUTPUT_LO[1] before and after L10 FFN
out_lo1_before = h[BD.OUTPUT_LO + 1].item()
h_after = activations['L10_ffn'][0, -1, :]
out_lo1_after = h_after[BD.OUTPUT_LO + 1].item()
print(f"\nOUTPUT_LO[1] before L10 FFN: {out_lo1_before:.3f}")
print(f"OUTPUT_LO[1] after L10 FFN: {out_lo1_after:.3f}")
print(f"Delta: {out_lo1_after - out_lo1_before:.3f}")
