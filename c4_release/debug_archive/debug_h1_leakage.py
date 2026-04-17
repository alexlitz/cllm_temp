"""Check if H1[AX_IDX] is preventing leakage at PC byte positions."""
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim as BD
from neural_vm.embedding import Opcode
from neural_vm.speculative import DraftVM

def build_context(bytecode):
    tokens = [Token.CODE_START]
    for instr in bytecode:
        op = instr & 0xFF
        imm = instr >> 8
        tokens.append(op)
        for i in range(4):
            tokens.append((imm >> (i * 8)) & 0xFF)
    tokens.append(Token.CODE_END)
    tokens.append(Token.DATA_START)
    tokens.append(Token.DATA_END)
    return tokens

# Test IMM 5
bytecode = [Opcode.IMM | (5 << 8), Opcode.EXIT]
draft_vm = DraftVM(bytecode)
context = build_context(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

# Load model
model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

# Hook layers
layer_outputs = {}
def make_hook(layer_name):
    def hook(module, input, output):
        layer_outputs[layer_name] = output.clone()
    return hook

for i in range(16):
    model.blocks[i].register_forward_hook(make_hook(f"L{i}"))

context_with_draft = context + draft_tokens
ctx_tensor = torch.tensor([context_with_draft], dtype=torch.long)
ctx_len = len(context)

with torch.no_grad():
    _ = model.forward(ctx_tensor)

print("Checking H1 dimension at PC and AX byte positions:")
print("=" * 70)

# PC byte 0 position (should have H1[0]=1, H1[1]=0)
pc_byte0_pos = ctx_len + 1
x_pc = layer_outputs["L9"][0, pc_byte0_pos, :]  # Before L10
h1_pc = x_pc[BD.H1:BD.H1+8]
print(f"PC byte 0 position (pos {pc_byte0_pos}):")
print(f"  H1: {h1_pc.tolist()}")
print(f"  H1[0] (PC_IDX): {h1_pc[0].item():.3f}")
print(f"  H1[1] (AX_IDX): {h1_pc[1].item():.3f}")
print()

# AX byte 0 position (should have H1[0]=0, H1[1]=1)
ax_byte0_pos = ctx_len + 6
x_ax = layer_outputs["L9"][0, ax_byte0_pos, :]  # Before L10
h1_ax = x_ax[BD.H1:BD.H1+8]
print(f"AX byte 0 position (pos {ax_byte0_pos}):")
print(f"  H1: {h1_ax.tolist()}")
print(f"  H1[0] (PC_IDX): {h1_ax[0].item():.3f}")
print(f"  H1[1] (AX_IDX): {h1_ax[1].item():.3f}")
print()

# Check L10 attention head 1 at PC byte 0
print("=" * 70)
print("Checking L10 attention head 1 at PC byte 0:")
print("-" * 70)

attn10 = model.blocks[10].attn
base = 64  # HD = 64 for head 1

# Check Q at PC byte 0
x_in = layer_outputs["L9"][0, pc_byte0_pos, :]
q_vals = []
for dim_idx in range(64):
    q_val = (attn10.W_q[base + dim_idx, :] @ x_in + attn10.b_q[base + dim_idx]).item()
    q_vals.append(q_val)

print(f"Q[33] at PC byte 0: {q_vals[33]:.3f}")
print(f"  Expected: -500 (to suppress non-AX positions)")
print()

# Check if attention is actually suppressed
# We need to compute the full attention pattern
# For simplicity, just check if OUTPUT_LO is being written by L10
x_before = layer_outputs["L9"][0, pc_byte0_pos, :]
x_after = layer_outputs["L10"][0, pc_byte0_pos, :]
output_lo_before = x_before[BD.OUTPUT_LO:BD.OUTPUT_LO+16]
output_lo_after = x_after[BD.OUTPUT_LO:BD.OUTPUT_LO+16]
output_lo_delta = output_lo_after - output_lo_before

print("OUTPUT_LO change from L9 to L10:")
active_changes = (output_lo_delta.abs() > 0.3).nonzero(as_tuple=True)[0].tolist()
if active_changes:
    for idx in active_changes:
        print(f"  OUTPUT_LO[{idx}]: {output_lo_before[idx].item():.3f} → {output_lo_after[idx].item():.3f} (Δ={output_lo_delta[idx].item():+.3f})")
else:
    print("  No significant changes (good - suppression working\!)")
