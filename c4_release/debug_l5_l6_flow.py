"""Trace FETCH → AX_CARRY → OUTPUT flow for JMP 12."""
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

print("Tracing JMP 12 Flow")
print("="*70)
print(f"Expected: JMP to PC=12 (0x0C = lo:12, hi:0)")
print()

# Load model
model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

# Hook to capture L5 attention output (after L5 attn, before L5 FFN)
l5_attn_output = {}
def capture_l5_attn(module, input, output):
    l5_attn_output['x'] = output.clone()
model.blocks[5].attn.register_forward_hook(capture_l5_attn)

# Hook to capture L6 attention output (after L6 attn, before L6 FFN)
l6_attn_output = {}
def capture_l6_attn(module, input, output):
    l6_attn_output['x'] = output.clone()
model.blocks[6].attn.register_forward_hook(capture_l6_attn)

# Hook to capture L6 FFN output (final)
l6_ffn_output = {}
def capture_l6_ffn(module, input, output):
    l6_ffn_output['x'] = output.clone()
model.blocks[6].ffn.register_forward_hook(capture_l6_ffn)

context_with_draft = context + draft_tokens
ctx_tensor = torch.tensor([context_with_draft], dtype=torch.long)

with torch.no_grad():
    logits = model.forward(ctx_tensor)

ctx_len = len(context)

# Find AX marker in context (it's part of the first draft step)
# Context ends at position ctx_len-1
# Position ctx_len: REG_PC marker
# Positions ctx_len+1 to ctx_len+4: PC bytes
# Position ctx_len+5: REG_AX marker
# Positions ctx_len+6 to ctx_len+9: AX bytes

ax_marker_pos = ctx_len + 5
pc_marker_pos = ctx_len

print("Step 1: L5 Attention Output (FETCH_LO/HI at AX marker)")
print("-"*70)
x_l5 = l5_attn_output['x']
fetch_lo = x_l5[0, ax_marker_pos, BD.FETCH_LO:BD.FETCH_LO+16]
fetch_hi = x_l5[0, ax_marker_pos, BD.FETCH_HI:BD.FETCH_HI+16]

print(f"At position {ax_marker_pos} (AX marker):")
print(f"\nFETCH_LO (should have [12]≈1.0):")
for i in range(16):
    val = fetch_lo[i].item()
    if abs(val) > 0.1:
        marker = " ← expected for IMM=12" if i == 12 else ""
        marker = " ← WRONG INDEX!" if i == 0 and abs(val) > 0.5 else marker
        print(f"  [{i:2d}]: {val:7.2f}{marker}")

print(f"\nFETCH_HI (should have [0]≈1.0):")
for i in range(4):
    val = fetch_hi[i].item()
    if abs(val) > 0.1:
        marker = " ← expected for IMM=12" if i == 0 else ""
        print(f"  [{i:2d}]: {val:7.2f}{marker}")

print("\n" + "="*70)
print("Step 2: L6 Attention Output (AX_CARRY_LO/HI at PC marker)")
print("-"*70)
x_l6_attn = l6_attn_output['x']
ax_carry_lo = x_l6_attn[0, pc_marker_pos, BD.AX_CARRY_LO:BD.AX_CARRY_LO+16]
ax_carry_hi = x_l6_attn[0, pc_marker_pos, BD.AX_CARRY_HI:BD.AX_CARRY_HI+16]

print(f"At position {pc_marker_pos} (PC marker):")
print(f"\nAX_CARRY_LO (should have [12]≈1.0):")
for i in range(16):
    val = ax_carry_lo[i].item()
    if abs(val) > 0.1:
        marker = " ← expected" if i == 12 else ""
        marker = " ← WRONG INDEX!" if i == 0 and abs(val) > 0.5 else marker
        print(f"  [{i:2d}]: {val:7.2f}{marker}")

print(f"\nAX_CARRY_HI (should have [0]≈1.0):")
for i in range(4):
    val = ax_carry_hi[i].item()
    if abs(val) > 0.1:
        marker = " ← expected" if i == 0 else ""
        print(f"  [{i:2d}]: {val:7.2f}{marker}")

cmp0 = x_l6_attn[0, pc_marker_pos, BD.CMP + 0].item()
print(f"\nCMP[0] (JMP flag): {cmp0:.2f} (should be ~1.0 for JMP)")

print("\n" + "="*70)
print("Step 3: L6 FFN Output (OUTPUT_LO/HI at PC marker)")
print("-"*70)
x_l6_ffn = l6_ffn_output['x']
output_lo = x_l6_ffn[0, pc_marker_pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
output_hi = x_l6_ffn[0, pc_marker_pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16]

print(f"At position {pc_marker_pos} (PC marker):")
print(f"\nOUTPUT_LO (should have [12]≈1.0):")
for i in range(16):
    val = output_lo[i].item()
    if abs(val) > 0.1:
        marker = " ← expected for PC=12" if i == 12 else ""
        marker = " ← WRONG! Predicts byte 0" if i == 0 and abs(val) > 0.5 else marker
        print(f"  [{i:2d}]: {val:7.2f}{marker}")

print(f"\nOUTPUT_HI (should have [0]≈1.0):")
for i in range(4):
    val = output_hi[i].item()
    if abs(val) > 0.1:
        marker = " ← expected for PC=12" if i == 0 else ""
        print(f"  [{i:2d}]: {val:7.2f}{marker}")

print("\n" + "="*70)
print("DIAGNOSIS:")
print("-"*70)
if abs(fetch_lo[0].item()) > 0.5:
    print("BUG at L5: FETCH_LO[0] is set instead of FETCH_LO[12]")
    print("  → L5 fetch head is writing to wrong FETCH_LO index")
elif abs(ax_carry_lo[0].item()) > 0.5:
    print("BUG at L6 attn: AX_CARRY_LO[0] is set instead of AX_CARRY_LO[12]")
    print("  → L6 attention W_o is writing to wrong AX_CARRY_LO index")
elif abs(output_lo[0].item()) > 0.5:
    print("BUG at L6 FFN: OUTPUT_LO[0] is set instead of OUTPUT_LO[12]")
    print("  → L6 FFN is writing to wrong OUTPUT_LO index")
else:
    print("No obvious bug found - values may be too small to detect")

print()
print("Actual prediction:")
pred = logits[0, pc_marker_pos, :].argmax().item()
print(f"  PC byte 0 predicted: {pred} (expected: 12)")
