#!/usr/bin/env python3
"""Debug SP pop carry detection after unit range fix."""
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

print("Step 2 (IMM 0):")
print(f"  SP bytes: {draft_step2[11:15]}")
print("Step 3 (MUL):")
print(f"  SP bytes: {draft_step3[11:15]} (expected: [0, 0, 1, 0] for SP=0x10000)")

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
S = 100.0

# Check the actual weights at unit 1200
ffn6 = model.blocks[6].ffn
print("\n=== L6 FFN carry detection weights (unit 1200) ===")
print(f"  W_up[1200, MARK_SP]: {ffn6.W_up[1200, BD.MARK_SP].item():.1f}")
print(f"  W_up[1200, CMP+3]: {ffn6.W_up[1200, BD.CMP + 3].item():.1f}")
print(f"  W_up[1200, EMBED_HI+15]: {ffn6.W_up[1200, BD.EMBED_HI + 15].item():.1f}")
print(f"  b_up[1200]: {ffn6.b_up[1200].item():.1f}")
print(f"  W_gate[1200, EMBED_LO+8]: {ffn6.W_gate[1200, BD.EMBED_LO + 8].item():.1f}")
print(f"  W_down[SP_POP_CARRY_0, 1200]: {ffn6.W_down[BD.SP_POP_CARRY_0, 1200].item():.4f}")

# Build context up to step 3
vm2 = DraftVM(bytecode)
all_tokens = context[:]
for i in range(3):
    vm2.step()
    all_tokens.extend(vm2.draft_tokens())

vm2.step()  # Step 3 (MUL)
step3_tokens = vm2.draft_tokens()

activations = {}
def hook_fn(name):
    def fn(module, input, output):
        activations[name] = output[0].detach().clone() if isinstance(output, tuple) else output.detach().clone()
    return fn

for i, block in enumerate(model.blocks):
    block.ffn.register_forward_hook(hook_fn(f'L{i}_ffn'))

# Test at SP marker position (index 10 in step)
test_context = all_tokens + step3_tokens[:11]  # Up to SP marker
token_ids = torch.tensor([test_context], dtype=torch.long)

print(f"\n=== At SP marker (step 3) ===")
print(f"Context length: {len(test_context)}")
print(f"Last token (SP marker): {test_context[-1]} (expected 271)")

with torch.no_grad():
    logits = model(token_ids)

h3 = activations['L3_ffn'][0, -1, :]
h6 = activations['L6_ffn'][0, -1, :]

print("\n--- Input dimensions at SP marker (after L3) ---")
print(f"MARK_SP: {h3[BD.MARK_SP].item():.3f}")
print(f"CMP[3] (binary pop): {h3[BD.CMP + 3].item():.3f}")
print(f"EMBED_HI[15]: {h3[BD.EMBED_HI + 15].item():.3f}")
print(f"EMBED_LO[8]: {h3[BD.EMBED_LO + 8].item():.3f}")

# Compute expected activation at unit 1200
h3_mark_sp = h3[BD.MARK_SP].item()
h3_cmp3 = h3[BD.CMP + 3].item()
h3_embed_hi15 = h3[BD.EMBED_HI + 15].item()
expected_up = S * (h3_mark_sp + h3_cmp3 + h3_embed_hi15) - S * 2.0
print(f"\nExpected W_up activation: S*({h3_mark_sp:.1f} + {h3_cmp3:.1f} + {h3_embed_hi15:.1f}) - S*2.0 = {expected_up:.1f}")

print("\n--- SP_POP_CARRY_0 values ---")
print(f"After L6: {h6[BD.SP_POP_CARRY_0].item():.3f}")

# Test predictions for SP bytes
print("\n=== Testing SP byte predictions ===")
for byte_idx in range(4):
    test_ctx = all_tokens + step3_tokens[:11 + byte_idx]
    token_ids = torch.tensor([test_ctx], dtype=torch.long)

    with torch.no_grad():
        logits = model(token_ids)

    predicted = logits[0, -1, :].argmax().item()
    expected = step3_tokens[11 + byte_idx]

    h = activations['L15_ffn'][0, -1, :]
    out_lo = h[BD.OUTPUT_LO:BD.OUTPUT_LO+16].argmax().item()
    out_hi = h[BD.OUTPUT_HI:BD.OUTPUT_HI+16].argmax().item()
    carry_0 = h[BD.SP_POP_CARRY_0].item()

    match = "✓" if predicted == expected else "✗"
    print(f"Byte {byte_idx}: expected={expected}, predicted={predicted} {match}, OUT_LO={out_lo}, OUT_HI={out_hi}, carry_0={carry_0:.2f}")
