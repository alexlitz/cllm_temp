"""Check OUTPUT_LO and OUTPUT_HI values at PC marker."""
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

# Test JMP 12
bytecode = [Opcode.JMP | (12 << 8), Opcode.EXIT]
draft_vm = DraftVM(bytecode)
context = build_context(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

print("Checking OUTPUT dimensions for JMP 12")
print("="*70)
print(f"Expected PC after JMP: 12 (0x0C = lo:12, hi:0)")

# Load model
model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

# Hook to capture final hidden state
final_hidden = {}
def capture_final(module, input, output):
    final_hidden['x'] = output.clone()

hook = model.blocks[-1].register_forward_hook(capture_final)

context_with_draft = context + draft_tokens
ctx_tensor = torch.tensor([context_with_draft], dtype=torch.long)

with torch.no_grad():
    logits = model.forward(ctx_tensor)

hook.remove()

ctx_len = len(context)
pc_byte0_pos = ctx_len  # Position where PC byte 0 is predicted

# Get hidden state
x = final_hidden['x']

print(f"\nHidden state at position {pc_byte0_pos} (predicting PC byte 0):")
print(f"  Shape: {x.shape}")

# Check OUTPUT_LO and OUTPUT_HI
output_lo = x[0, pc_byte0_pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
output_hi = x[0, pc_byte0_pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16]

print(f"\nOUTPUT_LO (should have [12]≈high for byte=12):")
for i in range(16):
    val = output_lo[i].item()
    marker = " ← should be high" if i == 12 else ""
    if abs(val) > 0.1:
        print(f"  [{i:2d}]: {val:7.2f}{marker}")

print(f"\nOUTPUT_HI (should have [0]≈high for byte=12):")
for i in range(4):
    val = output_hi[i].item()
    marker = " ← should be high" if i == 0 else ""
    if abs(val) > 0.1:
        print(f"  [{i:2d}]: {val:7.2f}{marker}")

# Check what the output head will compute
print(f"\nOutput head computation for byte 12:")
print(f"  head.weight[12, OUTPUT_LO+12] * OUTPUT_LO[12] = 5.0 * {output_lo[12].item():.2f} = {5.0 * output_lo[12].item():.2f}")
print(f"  head.weight[12, OUTPUT_HI+0]  * OUTPUT_HI[0]  = 5.0 * {output_hi[0].item():.2f} = {5.0 * output_hi[0].item():.2f}")
print(f"  Total contribution: {5.0 * (output_lo[12].item() + output_hi[0].item()):.2f}")

print(f"\nOutput head computation for byte 0:")
print(f"  head.weight[0, OUTPUT_LO+0] * OUTPUT_LO[0] = 5.0 * {output_lo[0].item():.2f} = {5.0 * output_lo[0].item():.2f}")
print(f"  head.weight[0, OUTPUT_HI+0] * OUTPUT_HI[0] = 5.0 * {output_hi[0].item():.2f} = {5.0 * output_hi[0].item():.2f}")
print(f"  Total contribution: {5.0 * (output_lo[0].item() + output_hi[0].item()):.2f}")

# Actual logits
print(f"\nActual logits:")
print(f"  Byte 12: {logits[0, pc_byte0_pos, 12].item():.2f}")
print(f"  Byte  0: {logits[0, pc_byte0_pos, 0].item():.2f}")
