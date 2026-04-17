#!/usr/bin/env python3
"""Trace which layer adds ADDR_B0_LO[9] contamination at the AX marker."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from neural_vm.vm_step import AutoregressiveVM, Token, _SetDim, set_vm_weights
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode

BD = _SetDim

# Build model and run program to get context
runner = AutoregressiveVMRunner()
model = runner.model
set_vm_weights(model)

bytecode = [Opcode.IMM | (42 << 8), Opcode.PSH, Opcode.LI, Opcode.EXIT]
context = runner._build_context(bytecode, b'', [])

# Generate up to LI step (step 2)
step_num = 0
step_tokens = []
for i in range(3 * Token.STEP_TOKENS + 10):
    tok = model.generate_next(context)
    context.append(tok)
    step_tokens.append(tok)
    if tok in (Token.STEP_END, Token.HALT):
        # Apply runner corrections
        pc = runner._extract_register(context, Token.REG_PC)
        instr_idx = pc // 5 if pc else None
        op = bytecode[instr_idx] & 0xFF if instr_idx is not None and 0 <= instr_idx < len(bytecode) else None
        if op and not runner.pure_attention_memory:
            if op == Opcode.PSH:
                new_sp = (runner._last_sp - 8) & 0xFFFFFFFF
                runner._override_register_in_last_step(context, Token.REG_SP, new_sp)
        if op:
            runner._track_memory_write(context, op)
        sp = runner._extract_register(context, Token.REG_SP)
        runner._last_sp = sp or 0
        runner._last_bp = runner._extract_register(context, Token.REG_BP) or 0
        step_num += 1
        step_tokens = []
        if tok == Token.HALT or step_num >= 3:
            break

# Build context up to AX marker of step 2
# Find step 2 start
li_step_start = None
step_count = 0
for i in range(len(context)):
    if context[i] == Token.STEP_END:
        step_count += 1
    if step_count == 2 and context[i] == Token.REG_PC:
        li_step_start = i
        break

ax_marker_pos = li_step_start + 5
print(f"AX marker at pos {ax_marker_pos}, token={context[ax_marker_pos]}")
assert context[ax_marker_pos] == Token.REG_AX

# Context up to AX marker
ctx = context[:ax_marker_pos + 1]
x_tensor = torch.tensor([ctx], dtype=torch.long)

# Hook each layer's attn to trace dim 21 = ADDR_B0_LO[9]
DIM_TARGET = BD.ADDR_B0_LO + 9  # dim 21
DIM_TARGET2 = BD.ADDR_B0_LO + 10  # dim 22 (correct value)

residuals = {}

def make_pre_hook(name):
    def hook(module, input):
        residuals[name] = input[0][0, -1, DIM_TARGET].item()
        residuals[name + '_10'] = input[0][0, -1, DIM_TARGET2].item()
    return hook

hooks = []
for i in range(16):
    h = model.blocks[i].attn.register_forward_pre_hook(make_pre_hook(f'pre_L{i}_attn'))
    hooks.append(h)
    h = model.blocks[i].ffn.register_forward_pre_hook(make_pre_hook(f'pre_L{i}_ffn'))
    hooks.append(h)

# Also hook after last layer
def final_hook(module, input, output):
    residuals['post_L15_ffn'] = output[0, -1, DIM_TARGET].item()
    residuals['post_L15_ffn_10'] = output[0, -1, DIM_TARGET2].item()
hooks.append(model.blocks[15].ffn.register_forward_hook(final_hook))

with torch.no_grad():
    _ = model(x_tensor)

for h in hooks:
    h.remove()

print(f"\n=== Tracing dim {DIM_TARGET} (ADDR_B0_LO[9]) at AX marker ===")
for i in range(16):
    pre_a = residuals.get(f'pre_L{i}_attn', 0)
    pre_f = residuals.get(f'pre_L{i}_ffn', 0)
    delta_a = pre_f - pre_a
    next_pre = residuals.get(f'pre_L{i+1}_attn', residuals.get('post_L15_ffn', 0)) if i < 15 else residuals.get('post_L15_ffn', 0)
    delta_f = next_pre - pre_f
    if abs(delta_a) > 0.001 or abs(delta_f) > 0.001:
        print(f"  L{i}: attn→{pre_f:.6f} (delta={delta_a:+.6f})  ffn→{next_pre:.6f} (delta={delta_f:+.6f})")

print(f"\n=== Tracing dim {DIM_TARGET2} (ADDR_B0_LO[10]) at AX marker ===")
for i in range(16):
    pre_a = residuals.get(f'pre_L{i}_attn_10', 0)
    pre_f = residuals.get(f'pre_L{i}_ffn_10', 0)
    delta_a = pre_f - pre_a
    next_pre = residuals.get(f'pre_L{i+1}_attn_10', residuals.get('post_L15_ffn_10', 0)) if i < 15 else residuals.get('post_L15_ffn_10', 0)
    delta_f = next_pre - pre_f
    if abs(delta_a) > 0.001 or abs(delta_f) > 0.001:
        print(f"  L{i}: attn→{pre_f:.6f} (delta={delta_a:+.6f})  ffn→{next_pre:.6f} (delta={delta_f:+.6f})")

# Also trace B1_LO[0] = dim 28 (ADDR_B1_LO[0])
DIM_B1 = BD.ADDR_B1_LO + 0
residuals2 = {}

hooks2 = []
def make_pre_hook2(name):
    def hook(module, input):
        residuals2[name] = input[0][0, -1, DIM_B1].item()
    return hook
for i in range(16):
    h = model.blocks[i].attn.register_forward_pre_hook(make_pre_hook2(f'pre_L{i}_attn'))
    hooks2.append(h)
    h = model.blocks[i].ffn.register_forward_pre_hook(make_pre_hook2(f'pre_L{i}_ffn'))
    hooks2.append(h)
def final_hook2(module, input, output):
    residuals2['post_L15_ffn'] = output[0, -1, DIM_B1].item()
hooks2.append(model.blocks[15].ffn.register_forward_hook(final_hook2))

with torch.no_grad():
    _ = model(x_tensor)

for h in hooks2:
    h.remove()

print(f"\n=== Tracing dim {DIM_B1} (ADDR_B1_LO[0]) at AX marker ===")
for i in range(16):
    pre_a = residuals2.get(f'pre_L{i}_attn', 0)
    pre_f = residuals2.get(f'pre_L{i}_ffn', 0)
    delta_a = pre_f - pre_a
    next_pre = residuals2.get(f'pre_L{i+1}_attn', residuals2.get('post_L15_ffn', 0)) if i < 15 else residuals2.get('post_L15_ffn', 0)
    delta_f = next_pre - pre_f
    if abs(delta_a) > 0.001 or abs(delta_f) > 0.001:
        print(f"  L{i}: attn→{pre_f:.6f} (delta={delta_a:+.6f})  ffn→{next_pre:.6f} (delta={delta_f:+.6f})")
