#!/usr/bin/env python3
"""Debug MEM addr byte 1 prediction issue."""
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

print("=== Step 2 (PSH) tokens ===")
print(f"Full: {step2_tokens}")
print(f"MEM section (indices 26-34): {step2_tokens[26:35]}")
print(f"  Addr: {step2_tokens[26:30]} (expected [248, 255, 0, 0])")
print(f"  Val: {step2_tokens[30:34]} (expected [1, 0, 0, 0])")
print()

# Test at MEM addr byte 0 position (index 26) - predicting addr byte 1
# Context = all_tokens + step2_tokens[:27]
test_idx = 27  # At this position, we predict step2_tokens[27] = 255

activations = {}
def hook(name):
    def fn(module, input, output):
        out = output[0] if isinstance(output, tuple) else output
        activations[name] = out.detach().clone()
    return fn

for i, block in enumerate(model.blocks):
    block.attn.register_forward_hook(hook(f'L{i}_attn'))
    block.ffn.register_forward_hook(hook(f'L{i}_ffn'))

test_context = all_tokens + step2_tokens[:test_idx]
token_ids = torch.tensor([test_context], dtype=torch.long)

with torch.no_grad():
    logits = model(token_ids)

print(f"=== At MEM addr byte 0 position (predicting byte 1) ===")
print(f"Context length: {len(test_context)}")
print(f"Last 5 tokens: {test_context[-5:]}")

# Check key values before L14
h = activations['L13_ffn'][0, -1, :]
print(f"\n=== Before L14 (after L13 FFN) ===")
print(f"MEM_STORE: {h[BD.MEM_STORE].item():.3f}")
print(f"MEM_ADDR_SRC: {h[BD.MEM_ADDR_SRC].item():.3f}")
print(f"MARK_MEM: {h[BD.MARK_MEM].item():.3f}")
print(f"L1H0[MEM=4]: {h[BD.L1H0 + 4].item():.3f}")
print(f"L1H1[MEM=4]: {h[BD.L1H1 + 4].item():.3f}")

# This is d=1 from MEM marker (we're at addr byte 0, which is d=1)
# L14 head 1 should fire here with Q = L1H1[MEM] - L1H0[MEM] = 1 - 0 = 1

print(f"\nL14 addr head 1 Q position:")
print(f"  Q should be: L1H1[MEM] - L1H0[MEM] = {h[BD.L1H1 + 4].item():.3f} - {h[BD.L1H0 + 4].item():.3f}")

# Check what K should match for SP byte 1
# The head should attend to SP byte 1 position and copy its value
print(f"\nBYTE_INDEX_1 at current position: {h[BD.BYTE_INDEX_1].item():.3f}")
print(f"H1[SP=2] at current position: {h[BD.H1 + 2].item():.3f}")

# Check L14 output
h_after = activations['L14_attn'][0, -1, :]
print(f"\n=== After L14 attention ===")
lo = h_after[BD.OUTPUT_LO:BD.OUTPUT_LO+16].tolist()
hi = h_after[BD.OUTPUT_HI:BD.OUTPUT_HI+16].tolist()
print(f"OUTPUT_LO: {[f'{v:.2f}' for v in lo]}")
print(f"OUTPUT_HI: {[f'{v:.2f}' for v in hi]}")
lo_idx = torch.argmax(h_after[BD.OUTPUT_LO:BD.OUTPUT_LO+16]).item()
hi_idx = torch.argmax(h_after[BD.OUTPUT_HI:BD.OUTPUT_HI+16]).item()
print(f"LO argmax={lo_idx}, HI argmax={hi_idx} → value={lo_idx + hi_idx * 16}")
print(f"Expected: 255 = 15 + 15*16")

# Final output
h_final = activations['L15_ffn'][0, -1, :]
lo = h_final[BD.OUTPUT_LO:BD.OUTPUT_LO+16].tolist()
hi = h_final[BD.OUTPUT_HI:BD.OUTPUT_HI+16].tolist()
print(f"\n=== Final OUTPUT (after L15 FFN) ===")
print(f"LO: {[f'{v:.2f}' for v in lo]}")
print(f"HI: {[f'{v:.2f}' for v in hi]}")
lo_idx = torch.argmax(h_final[BD.OUTPUT_LO:BD.OUTPUT_LO+16]).item()
hi_idx = torch.argmax(h_final[BD.OUTPUT_HI:BD.OUTPUT_HI+16]).item()
print(f"LO argmax={lo_idx}, HI argmax={hi_idx} → value={lo_idx + hi_idx * 16}")

predicted = logits[0, -1, :].argmax().item()
expected = step2_tokens[test_idx]
print(f"\nExpected: {expected}, Predicted: {predicted}")
