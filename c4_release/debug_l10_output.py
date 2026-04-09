#\!/usr/bin/env python3
"""Debug L10 OUTPUT_LO values in detail."""
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

activations = {}
def hook(name):
    def fn(module, input, output):
        out = output[0] if isinstance(output, tuple) else output
        activations[name] = out.detach().clone()
    return fn

for i, block in enumerate(model.blocks):
    block.attn.register_forward_hook(hook(f'L{i}_attn'))
    block.ffn.register_forward_hook(hook(f'L{i}_ffn'))

# Test at SP byte 1 position (predicting byte 2)
test_context = all_tokens + step3_tokens[:13]
token_ids = torch.tensor([test_context], dtype=torch.long)

with torch.no_grad():
    logits = model(token_ids)

# Show OUTPUT_LO values before and after L10 FFN
print("=== OUTPUT_LO values at byte 1 position (predicting byte 2) ===")
h_before = activations['L10_attn'][0, -1, :]
h_after = activations['L10_ffn'][0, -1, :]

print("\nAfter L10 attn (before FFN):")
out_lo_before = h_before[BD.OUTPUT_LO:BD.OUTPUT_LO+16].tolist()
out_hi_before = h_before[BD.OUTPUT_HI:BD.OUTPUT_HI+16].tolist()
print(f"  OUTPUT_LO: {[f'{v:.3f}' for v in out_lo_before]}")
print(f"  OUTPUT_HI: {[f'{v:.3f}' for v in out_hi_before]}")
print(f"  argmax: lo={h_before[BD.OUTPUT_LO:BD.OUTPUT_LO+16].argmax().item()}, hi={h_before[BD.OUTPUT_HI:BD.OUTPUT_HI+16].argmax().item()}")

print("\nAfter L10 FFN:")
out_lo_after = h_after[BD.OUTPUT_LO:BD.OUTPUT_LO+16].tolist()
out_hi_after = h_after[BD.OUTPUT_HI:BD.OUTPUT_HI+16].tolist()
print(f"  OUTPUT_LO: {[f'{v:.3f}' for v in out_lo_after]}")
print(f"  OUTPUT_HI: {[f'{v:.3f}' for v in out_hi_after]}")
print(f"  argmax: lo={h_after[BD.OUTPUT_LO:BD.OUTPUT_LO+16].argmax().item()}, hi={h_after[BD.OUTPUT_HI:BD.OUTPUT_HI+16].argmax().item()}")

# Show EMBED values
print("\n=== EMBED values (input to passthrough) ===")
embed_lo = h_before[BD.EMBED_LO:BD.EMBED_LO+16].tolist()
embed_hi = h_before[BD.EMBED_HI:BD.EMBED_HI+16].tolist()
print(f"  EMBED_LO: {[f'{v:.3f}' for v in embed_lo]}")
print(f"  EMBED_HI: {[f'{v:.3f}' for v in embed_hi]}")

# Check if any other dimensions could explain the 28
print("\n=== Other relevant dimensions ===")
print(f"  H1[SP]: {h_before[BD.H1 + 2].item():.3f}")
print(f"  IS_BYTE: {h_before[BD.IS_BYTE].item():.3f}")
print(f"  CMP[3]: {h_before[BD.CMP + 3].item():.3f}")
print(f"  BYTE_INDEX_1: {h_before[BD.BYTE_INDEX_1].item():.3f}")
print(f"  SP_POP_CARRY_0: {h_before[BD.SP_POP_CARRY_0].item():.3f}")

# Check what the carry application weights are
ffn10 = model.blocks[10].ffn
print("\n=== L10 FFN carry application weights (unit 1600-1631, byte 1 LO) ===")
unit = 1632  # byte 1 LO starts after byte 0 (32 units)
for k in range(16):
    w_up_sum = (ffn10.W_up[unit, BD.H1 + 2].item() + 
                ffn10.W_up[unit, BD.IS_BYTE].item() + 
                ffn10.W_up[unit, BD.CMP + 3].item() +
                ffn10.W_up[unit, BD.SP_POP_CARRY_0].item() +
                ffn10.W_up[unit, BD.BYTE_INDEX_1].item())
    b_up = ffn10.b_up[unit].item()
    w_gate = ffn10.W_gate[unit, BD.OUTPUT_LO + k].item()
    new_k = (k + 1) % 16
    w_down_new = ffn10.W_down[BD.OUTPUT_LO + new_k, unit].item()
    w_down_old = ffn10.W_down[BD.OUTPUT_LO + k, unit].item()
    print(f"  unit {unit}: k={k}, up_sum={w_up_sum:.0f}, b_up={b_up:.0f}, gate_in={w_gate:.2f}, down[{new_k}]={w_down_new:.4f}, down[{k}]={w_down_old:.4f}")
    unit += 1
