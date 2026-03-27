"""Check EMBED_LO/HI values at PC marker."""
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

print("Checking EMBED dimensions for JMP 12")
print("="*70)
print(f"Expected PC after JMP: 12 (0x0C = lo:12, hi:0)")

# Load model
model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

# Hook L6 block output (after attention, before FFN) to see EMBED values
l6_attn_output = {}
def capture_l6_attn(module, input, output):
    l6_attn_output['x'] = output.clone()

model.blocks[6].attn.register_forward_hook(capture_l6_attn)

context_with_draft = context + draft_tokens
ctx_tensor = torch.tensor([context_with_draft], dtype=torch.long)

with torch.no_grad():
    logits = model.forward(ctx_tensor)

ctx_len = len(context)

# Find PC marker position in the generated tokens
# The first token after context should be REG_PC marker
pc_marker_pos = ctx_len  # Position where we're predicting PC marker
pc_byte0_pos = ctx_len + 1  # Position where we're predicting PC byte 0

# Check the hidden state after L6 attention (input to L6 FFN)
x = l6_attn_output['x']

print(f"\nL6 FFN input at position {pc_marker_pos} (PC marker):")

# Check EMBED_LO and EMBED_HI
embed_lo = x[0, pc_marker_pos, BD.EMBED_LO:BD.EMBED_LO+16]
embed_hi = x[0, pc_marker_pos, BD.EMBED_HI:BD.EMBED_HI+16]

print(f"\nEMBED_LO (source for PC OUTPUT):")
for i in range(16):
    val = embed_lo[i].item()
    marker = " ← expect this for PC=12" if i == 12 else ""
    marker = " ← currently set!" if abs(val) > 0.5 and i == 0 else marker
    if abs(val) > 0.1:
        print(f"  [{i:2d}]: {val:7.2f}{marker}")

print(f"\nEMBED_HI (source for PC OUTPUT):")
for i in range(4):
    val = embed_hi[i].item()
    marker = " ← expect this for PC=12" if i == 0 else ""
    if abs(val) > 0.1:
        print(f"  [{i:2d}]: {val:7.2f}{marker}")

# Also check MARK_PC
mark_pc = x[0, pc_marker_pos, BD.MARK_PC].item()
print(f"\nMARK_PC: {mark_pc:.2f} (should be ~1.0 at PC marker)")

print(f"\nDiagnosis:")
if embed_lo[0].item() > 0.5:
    print(f"  BUG: EMBED_LO[0]={embed_lo[0].item():.2f} is set (should be 0)")
    print(f"       EMBED_LO[12]={embed_lo[12].item():.2f} should be ~1.0")
    print(f"  → Some layer is writing PC byte value to EMBED_LO[0] instead of EMBED_LO[12]")
