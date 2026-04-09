#!/usr/bin/env python3
"""Debug MUL SP byte prediction issues."""
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
print(f"  AX bytes: {draft_step3[6:10]}")
print(f"  STACK0: {draft_step3[20:25]}")

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

# Test at SP byte 1 position (predicting SP byte 2)
# SP marker is at index 10 in step tokens, SP byte 0 at 11
test_context = all_tokens + step3_tokens[:12]  # Up to SP byte 1 (position 11 in step)
token_ids = torch.tensor([test_context], dtype=torch.long)

print(f"\nContext length: {len(test_context)}")
print(f"Last 5 tokens: {test_context[-5:]}")

with torch.no_grad():
    logits = model(token_ids)

predicted = logits[0, -1, :].argmax().item()
expected = step3_tokens[12]  # SP byte 2

print(f"\n=== At SP byte 1 position (predicting byte 2) ===")
print(f"Expected: {expected}")
print(f"Predicted: {predicted}")
print(f"Match: {predicted == expected}")

h = activations['L15_ffn'][0, -1, :]

# Check transition flags
print(f"\n--- Transition flags ---")
print(f"NEXT_PC: {h[BD.NEXT_PC].item():.3f}")
print(f"NEXT_AX: {h[BD.NEXT_AX].item():.3f}")
print(f"NEXT_SP: {h[BD.NEXT_SP].item():.3f}")
print(f"NEXT_BP: {h[BD.NEXT_BP].item():.3f}")
print(f"NEXT_STACK0: {h[BD.NEXT_STACK0].item():.3f}")
print(f"NEXT_MEM: {h[BD.NEXT_MEM].item():.3f}")
print(f"NEXT_SE: {h[BD.NEXT_SE].item():.3f}")

# Check OUTPUT values
out_lo = h[BD.OUTPUT_LO:BD.OUTPUT_LO+16]
out_hi = h[BD.OUTPUT_HI:BD.OUTPUT_HI+16]
print(f"\n--- OUTPUT values ---")
print(f"OUTPUT_LO: argmax={out_lo.argmax().item()}, max={out_lo.max().item():.3f}")
print(f"OUTPUT_HI: argmax={out_hi.argmax().item()}, max={out_hi.max().item():.3f}")
print(f"OUTPUT_LO[0]: {out_lo[0].item():.3f}")
print(f"OUTPUT_HI[0]: {out_hi[0].item():.3f}")

# Check H values for MEM (to understand NEXT_SE)
print(f"\n--- H values for NEXT_SE check ---")
MEM_I = 4
print(f"H2[MEM]: {h[BD.H2 + MEM_I].item():.3f} (threshold 7.5)")
print(f"H3[MEM]: {h[BD.H3 + MEM_I].item():.3f} (threshold 8.5)")

# Check IS_BYTE
print(f"\n--- Byte position flags ---")
print(f"IS_BYTE: {h[BD.IS_BYTE].item():.3f}")
print(f"IS_MARK: {h[BD.IS_MARK].item():.3f}")
print(f"HAS_SE: {h[BD.HAS_SE].item():.3f}")

# Check logits for specific tokens
print(f"\n--- Final logits ---")
print(f"Token 0 logit: {logits[0, -1, 0].item():.3f}")
print(f"Token 1 logit: {logits[0, -1, 1].item():.3f}")
print(f"Token 276 (STEP_END) logit: {logits[0, -1, 276].item():.3f}")
print(f"Token 271 (REG_SP) logit: {logits[0, -1, 271].item():.3f}")

# Trace OUTPUT_HI through layers
print(f"\n--- OUTPUT_HI[0] trace ---")
for layer in [3, 6, 10, 15]:
    h_l = activations[f'L{layer}_ffn'][0, -1, :]
    hi_val = h_l[BD.OUTPUT_HI + 0].item()
    print(f"L{layer:2d}: OUTPUT_HI[0] = {hi_val:.3f}")

# Trace NEXT_SE through layers
print(f"\n--- NEXT_SE trace ---")
for layer in [2, 3, 6, 10, 15]:
    h_l = activations[f'L{layer}_ffn'][0, -1, :]
    next_se = h_l[BD.NEXT_SE].item()
    h3_mem = h_l[BD.H3 + MEM_I].item()
    h2_mem = h_l[BD.H2 + MEM_I].item()
    print(f"L{layer:2d}: NEXT_SE={next_se:.3f}, H3[MEM]={h3_mem:.3f}, H2[MEM]={h2_mem:.3f}")
