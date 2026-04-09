#!/usr/bin/env python3
"""Debug what EMBED contains at SP marker for carry detection."""
import os, sys
sys.path.insert(0, os.getcwd())
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, _SetDim, Token
from neural_vm.embedding import Opcode
from neural_vm.speculative import DraftVM
from neural_vm.constants import IMMEDIATE_SIZE, PADDING_SIZE

bytecode = [Opcode.IMM | (1 << 8), Opcode.PSH, Opcode.IMM | (0 << 8), Opcode.MUL, Opcode.EXIT]
vm = DraftVM(bytecode)

# Run to step 3
for i in range(4):
    vm.step()
    if i == 2:
        draft_step2 = vm.draft_tokens()
    if i == 3:
        draft_step3 = vm.draft_tokens()

print("Step 2 SP bytes:", draft_step2[11:15], "(prev SP for step 3)")
print("Step 3 SP bytes:", draft_step3[11:15], "(expected new SP)")

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

# Build context up to step 3 SP marker
vm2 = DraftVM(bytecode)
all_tokens = context[:]
for i in range(3):
    vm2.step()
    all_tokens.extend(vm2.draft_tokens())

vm2.step()
step3_tokens = vm2.draft_tokens()

# Test at SP marker position (index 10 in step)
test_context = all_tokens + step3_tokens[:11]  # Up to SP marker
token_ids = torch.tensor([test_context], dtype=torch.long)

print(f"\nContext length at SP marker: {len(test_context)}")
print(f"Last 5 tokens: {test_context[-5:]}")
print(f"Token at SP marker (should be 271): {test_context[-1]}")

activations = {}
def hook_fn(name):
    def fn(module, input, output):
        activations[name] = output[0].detach().clone() if isinstance(output, tuple) else output.detach().clone()
    return fn

for i, block in enumerate(model.blocks):
    block.attn.register_forward_hook(hook_fn(f'L{i}_attn'))
    block.ffn.register_forward_hook(hook_fn(f'L{i}_ffn'))

with torch.no_grad():
    logits = model(token_ids)

# Check EMBED_LO/HI at SP marker after L3
print("\n=== At SP marker position (after L3) ===")
h3 = activations['L3_ffn'][0, -1, :]

embed_lo = h3[BD.EMBED_LO:BD.EMBED_LO+16]
embed_hi = h3[BD.EMBED_HI:BD.EMBED_HI+16]
embed_lo_idx = embed_lo.argmax().item()
embed_hi_idx = embed_hi.argmax().item()
embed_val = embed_lo_idx + embed_hi_idx * 16

print(f"EMBED_LO: argmax={embed_lo_idx}, val={embed_lo.max().item():.3f}")
print(f"EMBED_HI: argmax={embed_hi_idx}, val={embed_hi.max().item():.3f}")
print(f"EMBED value: {embed_val} (0x{embed_val:02X})")
print(f"Expected prev_byte0: 248 (0xF8)")

# Check if this matches prev SP byte 0
prev_sp_byte0 = draft_step2[11]
print(f"Actual prev_byte0: {prev_sp_byte0} (0x{prev_sp_byte0:02X})")

# For carry_0 detection: prev_byte0 >= 248 means hi nibble = 15 AND lo nibble >= 8
if embed_val >= 248:
    print(f"CARRY_0 should be 1 (prev_byte0 = {embed_val} >= 248)")
else:
    print(f"CARRY_0 should be 0 (prev_byte0 = {embed_val} < 248)")

# Check CMP[3] for binary pop flag
print(f"\nCMP[3] (binary pop flag): {h3[BD.CMP + 3].item():.3f}")
print(f"MARK_SP: {h3[BD.MARK_SP].item():.3f}")

# Also check after L6 where we'd add carry detection
h6 = activations['L6_ffn'][0, -1, :]
print(f"\n=== After L6 FFN ===")
print(f"EMBED_LO argmax: {h6[BD.EMBED_LO:BD.EMBED_LO+16].argmax().item()}")
print(f"EMBED_HI argmax: {h6[BD.EMBED_HI:BD.EMBED_HI+16].argmax().item()}")
print(f"CMP[3]: {h6[BD.CMP + 3].item():.3f}")
