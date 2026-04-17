"""Check if L6 JMP relay fires in step 2."""
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

# Test JMP 12, then NOP
bytecode = [Opcode.JMP | (12 << 8), Opcode.NOP]
draft_vm = DraftVM(bytecode)
context = build_context(bytecode)

# Run two steps
draft_vm.step()
step1_tokens = draft_vm.draft_tokens()
draft_vm.step()
step2_tokens = draft_vm.draft_tokens()

print("Checking L6 JMP Relay in Step 2")
print("="*70)
print(f"Step 1 AX (should contain JMP target 12): {step1_tokens[6:10]}")
print(f"Step 2 PC (should be 12 if JMP relay works): {step2_tokens[1:5]}")
print()

# Load model
model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

# Hook L6 attention output for step 2
l6_attn_output = {}
def capture_l6_attn(module, input, output):
    l6_attn_output['x'] = output.clone()
model.blocks[6].attn.register_forward_hook(capture_l6_attn)

context_with_drafts = context + step1_tokens + step2_tokens
ctx_tensor = torch.tensor([context_with_drafts], dtype=torch.long)

with torch.no_grad():
    logits = model.forward(ctx_tensor)

ctx_len = len(context)
step1_start = ctx_len
step1_ax_marker = step1_start + 5  # AX marker in step 1
step2_start = ctx_len + 35
step2_pc_marker = step2_start  # PC marker in step 2

print("Step 1 positions:")
print(f"  AX marker: {step1_ax_marker}")
print(f"  AX bytes: {step1_ax_marker+1} to {step1_ax_marker+4}")

print(f"\nStep 2 positions:")
print(f"  PC marker: {step2_pc_marker}")
print(f"  PC bytes: {step2_pc_marker+1} to {step2_pc_marker+4}")

print(f"\nDistance from step 2 PC marker to step 1 AX marker: {step2_pc_marker - step1_ax_marker}")

print("\n" + "="*70)
print("L6 Attention Output at Step 2 PC Marker")
print("-"*70)

x = l6_attn_output['x']
ax_carry_lo = x[0, step2_pc_marker, BD.AX_CARRY_LO:BD.AX_CARRY_LO+16]
ax_carry_hi = x[0, step2_pc_marker, BD.AX_CARRY_HI:BD.AX_CARRY_HI+16]
cmp0 = x[0, step2_pc_marker, BD.CMP + 0].item()

print(f"At position {step2_pc_marker} (step 2 PC marker):")
print(f"\nAX_CARRY_LO (should have [12]≈1.0 if relay works):")
for i in range(16):
    val = ax_carry_lo[i].item()
    if abs(val) > 0.1:
        marker = " ← expected for JMP target 12" if i == 12 else ""
        print(f"  [{i:2d}]: {val:7.2f}{marker}")

print(f"\nAX_CARRY_HI (should have [0]≈1.0):")
for i in range(4):
    val = ax_carry_hi[i].item()
    if abs(val) > 0.1:
        marker = " ← expected" if i == 0 else ""
        print(f"  [{i:2d}]: {val:7.2f}{marker}")

print(f"\nCMP[0] (JMP flag): {cmp0:.2f} (should be ~1.0 if JMP relay active)")

# Check what's at step 1 AX marker
print("\n" + "="*70)
print("What's available at Step 1 AX Marker for Relay")
print("-"*70)
print(f"Step 1 AX tokens (from DraftVM): {step1_tokens[5:10]}")
print(f"  AX marker: {step1_tokens[5]} (should be {Token.REG_AX})")
print(f"  AX byte 0: {step1_tokens[6]}")
print(f"  AX bytes: {step1_tokens[6:10]}")

# Check FETCH and OP_JMP at step 1 AX marker
# We need to look at the hidden state AFTER L5 but BEFORE L6 attention
# Actually, we need L5 output at step 1 AX marker
# Let me create another hook for L5

model2 = AutoregressiveVM()
set_vm_weights(model2)
model2.eval()

l5_attn_output = {}
def capture_l5_attn(module, input, output):
    l5_attn_output['x'] = output.clone()
model2.blocks[5].attn.register_forward_hook(capture_l5_attn)

with torch.no_grad():
    _ = model2.forward(ctx_tensor)

x_l5 = l5_attn_output['x']
fetch_lo = x_l5[0, step1_ax_marker, BD.FETCH_LO:BD.FETCH_LO+16]
fetch_hi = x_l5[0, step1_ax_marker, BD.FETCH_HI:BD.FETCH_HI+16]

# Check for OP_JMP flag
# OP_JMP should be set by L5 FFN after decoding the opcode
# Actually, let me check L5 FFN output instead
l5_ffn_output = {}
def capture_l5_ffn(module, input, output):
    l5_ffn_output['x'] = output.clone()

model3 = AutoregressiveVM()
set_vm_weights(model3)
model3.eval()
model3.blocks[5].ffn.register_forward_hook(capture_l5_ffn)

with torch.no_grad():
    _ = model3.forward(ctx_tensor)

x_l5_ffn = l5_ffn_output['x']
op_jmp = x_l5_ffn[0, step1_ax_marker, BD.OP_JMP].item()

print(f"\nAt step 1 AX marker (position {step1_ax_marker}):")
print(f"  OP_JMP flag (L5 FFN output): {op_jmp:.2f}")

print(f"\n  FETCH_LO (L5 attn output):")
for i in range(16):
    val = fetch_lo[i].item()
    if abs(val) > 0.1:
        marker = " ← JMP immediate" if i == 12 else ""
        print(f"    [{i:2d}]: {val:7.2f}{marker}")

print(f"\n  FETCH_HI (L5 attn output):")
for i in range(4):
    val = fetch_hi[i].item()
    if abs(val) > 0.1:
        print(f"    [{i:2d}]: {val:7.2f}")

print("\n" + "="*70)
print("Diagnosis:")
print("-"*70)
if cmp0 < 0.5:
    print("✗ JMP relay DID NOT fire in step 2 (CMP[0] ≈ 0)")
    if op_jmp < 0.5:
        print("  → OP_JMP flag not set at step 1 AX marker")
        print("  → Opcode decoder may not be working")
    else:
        print(f"  → OP_JMP={op_jmp:.2f} IS set at step 1 AX marker")
        print("  → But L6 attention head 0 didn't copy it to step 2")
        print("  → Check attention weights or ALiBi distance penalty")
else:
    print(f"✓ JMP relay fired in step 2 (CMP[0]={cmp0:.2f})")
    if max(ax_carry_lo).item() < 0.5:
        print("  ✗ But AX_CARRY_LO is not set correctly")
