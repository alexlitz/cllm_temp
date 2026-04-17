"""Debug PC byte 1 prediction issue."""
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

print("Expected output:")
print(f"  PC = {draft_vm.pc} = {draft_vm.pc:#x}")
print(f"  PC bytes: {draft_tokens[1:5]}")
print(f"  AX = {draft_vm.ax}")
print(f"  AX bytes: {draft_tokens[6:10]}")
print()

# Load model
model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

# Hook to capture OUTPUT dimensions at PC marker
layer_outputs = {}
def make_hook(layer_name):
    def hook(module, input, output):
        layer_outputs[layer_name] = output.clone()
    return hook

# Hook all layers
for i in range(16):
    model.blocks[i].register_forward_hook(make_hook(f"L{i}"))

context_with_draft = context + draft_tokens
ctx_tensor = torch.tensor([context_with_draft], dtype=torch.long)
ctx_len = len(context)
pc_marker_pos = ctx_len  # Position of REG_PC marker
pc_byte1_pos = ctx_len + 2  # Position of PC byte 1

with torch.no_grad():
    logits = model.forward(ctx_tensor)

# Check OUTPUT dimensions at PC marker (predicts PC byte 0)
print("OUTPUT dimensions at PC marker (predicting PC byte 0):")
print("-" * 70)
x_pc_marker = layer_outputs["L15"][0, pc_marker_pos, :]
output_lo = x_pc_marker[BD.OUTPUT_LO:BD.OUTPUT_LO+16]
output_hi = x_pc_marker[BD.OUTPUT_HI:BD.OUTPUT_HI+16]

print(f"  OUTPUT_LO active indices: {(output_lo.abs() > 0.5).nonzero(as_tuple=True)[0].tolist()}")
print(f"  OUTPUT_HI active indices: {(output_hi.abs() > 0.5).nonzero(as_tuple=True)[0].tolist()}")
print()

# Check what PC byte 0 is predicted as
pred_byte0 = logits[0, pc_marker_pos, :].argmax().item()
print(f"  Prediction for PC byte 0: {pred_byte0} (expected {draft_tokens[1]})")
print()

# Check OUTPUT dimensions at PC byte 0 position (predicts PC byte 1)
print("OUTPUT dimensions at PC byte 0 position (predicting PC byte 1):")
print("-" * 70)
x_pc_byte0 = layer_outputs["L15"][0, pc_marker_pos + 1, :]
output_lo2 = x_pc_byte0[BD.OUTPUT_LO:BD.OUTPUT_LO+16]
output_hi2 = x_pc_byte0[BD.OUTPUT_HI:BD.OUTPUT_HI+16]

print(f"  OUTPUT_LO active indices: {(output_lo2.abs() > 0.5).nonzero(as_tuple=True)[0].tolist()}")
print(f"  OUTPUT_HI active indices: {(output_hi2.abs() > 0.5).nonzero(as_tuple=True)[0].tolist()}")
print()

# Check what PC byte 1 is predicted as
pred_byte1 = logits[0, pc_byte1_pos - 1, :].argmax().item()
print(f"  Prediction for PC byte 1: {pred_byte1} (expected {draft_tokens[2]})")
print()

# Check if OUTPUT dimensions are being converted correctly by the head
print("Checking head conversion:")
print("-" * 70)
# The head should convert OUTPUT_LO[lo] + OUTPUT_HI[hi] -> byte value (lo + hi*16)
# For byte 0, we expect lo=10 (0xA), hi=0, so byte = 10
# For byte 1, we expect lo=0, hi=0, so byte = 0

# Manually compute what the head should produce from OUTPUT dimensions
for byte_pos, pos_name in [(pc_marker_pos + 1, "PC byte 0"), (pc_byte1_pos - 1, "PC byte 1")]:
    x = layer_outputs["L15"][0, byte_pos, :]
    lo_vec = x[BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    hi_vec = x[BD.OUTPUT_HI:BD.OUTPUT_HI+16]

    lo_idx = lo_vec.argmax().item()
    hi_idx = hi_vec.argmax().item()
    byte_val = lo_idx + hi_idx * 16

    pred = logits[0, byte_pos, :].argmax().item()

    print(f"  {pos_name}:")
    print(f"    OUTPUT_LO.argmax = {lo_idx}, OUTPUT_HI.argmax = {hi_idx}")
    print(f"    Expected byte = {lo_idx} + {hi_idx}*16 = {byte_val}")
    print(f"    Predicted = {pred}")
    print()
