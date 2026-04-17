#!/usr/bin/env python3
"""Debug L10 head 3 STACK0 attention weights."""
import os, sys
sys.path.insert(0, os.getcwd())
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, _SetDim, Token
from neural_vm.embedding import Opcode
from neural_vm.speculative import DraftVM
from neural_vm.constants import IMMEDIATE_SIZE, PADDING_SIZE

bytecode = [Opcode.IMM | (1 << 8), Opcode.PSH, Opcode.IMM | (0 << 8), Opcode.MUL, Opcode.EXIT]
vm = DraftVM(bytecode)
vm.step(); draft_step0 = vm.draft_tokens()
vm.step(); draft_step1 = vm.draft_tokens()
vm.step(); draft_step2 = vm.draft_tokens()

print(f"Step 1 (PSH): STACK0 = {draft_step1[21:25]}")
print(f"Step 2 (IMM): STACK0 = {draft_step2[21:25]}")

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
# Position: at STACK0 byte 0 (predicting byte 1)
full_context = context + draft_step0 + draft_step1 + draft_step2[:22]

# Compute positions
step0_start = len(context)
step1_start = step0_start + 35
step2_start = step1_start + 35
step1_stack0_byte0 = step1_start + 21
step1_stack0_byte1 = step1_start + 22
step2_stack0_byte0 = step2_start + 21
current_pos = len(full_context) - 1

print(f"\nPositions:")
print(f"  Step 1 STACK0 byte 0: {step1_stack0_byte0} (value {draft_step1[21]})")
print(f"  Step 1 STACK0 byte 1: {step1_stack0_byte1} (value {draft_step1[22]})")
print(f"  Step 2 STACK0 byte 0: {step2_stack0_byte0} (current, value {draft_step2[21]})")
print(f"  Current position: {current_pos}")

model = AutoregressiveVM(vocab_size=512, n_layers=16, n_heads=8, ffn_hidden=4096)
set_vm_weights(model)
model.eval()
BD = _SetDim

# Hook to get attention weights
attn_weights = {}
def attn_hook(name):
    def fn(module, input, output):
        # output is (out, attn_weights) or just out depending on implementation
        # Let's capture the intermediate Q, K, V
        pass
    return fn

# Get pre-attention values and manually compute attention
pre_l10 = {}
def pre_hook(name):
    def fn(module, input):
        pre_l10[name] = input[0].detach().clone()
    return fn

model.blocks[10].attn.register_forward_pre_hook(pre_hook('L10_attn_input'))

token_ids = torch.tensor([full_context], dtype=torch.long)
with torch.no_grad():
    logits = model(token_ids)

# Get L10 attention weights
attn10 = model.blocks[10].attn
h = pre_l10['L10_attn_input']
HD = 64  # head dim

# Manually compute Q and K for head 3
head = 3
base = head * HD
Q = torch.zeros(len(full_context), HD)
K = torch.zeros(len(full_context), HD)

for pos in range(len(full_context)):
    for d in range(HD):
        Q[pos, d] = (h[0, pos, :] @ attn10.W_q[base + d, :]).item()
        K[pos, d] = (h[0, pos, :] @ attn10.W_k[base + d, :]).item()

# Add biases
for d in range(HD):
    if hasattr(attn10, 'b_q') and attn10.b_q is not None:
        Q[:, d] += attn10.b_q[base + d].item()
    if hasattr(attn10, 'b_k') and attn10.b_k is not None:
        K[:, d] += attn10.b_k[base + d].item()

# Show Q at current position
print(f"\nQ at current position (STACK0 byte 0, head 3):")
print(f"  dim 0 (IS_BYTE*L + HAS_SE*2L - 2.5L): {Q[current_pos, 0]:.3f}")
print(f"  dim 1 (L1H4[BP]*L - H1[BP]*L - 0.5L): {Q[current_pos, 1]:.3f}")
print(f"  dim 2 (BYTE_INDEX_3 suppress): {Q[current_pos, 2]:.3f}")
print(f"  dim 3 (BYTE_INDEX_0): {Q[current_pos, 3]:.3f}")
print(f"  dim 33 (gate): {Q[current_pos, 33]:.3f}")

# Show K at step 1 STACK0 byte 1
print(f"\nK at step 1 STACK0 byte 1 (pos {step1_stack0_byte1}):")
print(f"  dim 0 (IS_BYTE*L): {K[step1_stack0_byte1, 0]:.3f}")
print(f"  dim 1 (L1H4[BP]*L - H1[BP]*L): {K[step1_stack0_byte1, 1]:.3f}")
print(f"  dim 2 (BYTE_INDEX_0 suppress): {K[step1_stack0_byte1, 2]:.3f}")
print(f"  dim 3 (BYTE_INDEX_1): {K[step1_stack0_byte1, 3]:.3f}")
print(f"  dim 33 (gate): {K[step1_stack0_byte1, 33]:.3f}")

# Compute Q*K scores for nearby positions
print(f"\nQ*K scores (from current to various K positions):")
for pos in [step1_stack0_byte0, step1_stack0_byte1, step1_stack0_byte1+1, step1_stack0_byte1+2, current_pos-1, current_pos]:
    if pos < len(full_context):
        score = (Q[current_pos, :] @ K[pos, :]).item() / (HD ** 0.5)
        print(f"  pos {pos}: {score:.1f}")

# Check what values are at top K positions
# Get input values at step 1 positions
print(f"\nCLEAN_EMBED_LO at step 1 STACK0 positions:")
for i, pos in enumerate([step1_stack0_byte0, step1_stack0_byte1, step1_stack0_byte1+1, step1_stack0_byte1+2]):
    if pos < len(full_context):
        clean_lo = h[0, pos, BD.CLEAN_EMBED_LO:BD.CLEAN_EMBED_LO+16]
        max_idx = clean_lo.argmax().item()
        max_val = clean_lo.max().item()
        print(f"  byte {i}: argmax={max_idx} (val={max_val:.3f}) - token value {full_context[pos] if pos < len(full_context) else '?'}")
