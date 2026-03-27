"""Check final OUTPUT values at PC marker before output head."""
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

print("Final OUTPUT Values at PC Marker")
print("="*70)

context_with_draft = context + draft_tokens
ctx_len = len(context)
pc_marker_pos = ctx_len

# Load model
model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

# Hook final layer output (L15 output, input to output head)
final_output = {}
def capture_final(module, input, output):
    final_output['x'] = output.clone()
model.blocks[-1].register_forward_hook(capture_final)

ctx_tensor = torch.tensor([context_with_draft], dtype=torch.long)

with torch.no_grad():
    logits = model.forward(ctx_tensor)

x_final = final_output['x']
output_lo = x_final[0, pc_marker_pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
output_hi = x_final[0, pc_marker_pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16]

print(f"At position {pc_marker_pos} (PC marker, final layer output):")
print(f"\nOUTPUT_LO values:")
for i in range(16):
    val = output_lo[i].item()
    if abs(val) > 0.1:
        print(f"  [{i:2d}]: {val:7.2f}")

print(f"\nOUTPUT_HI values:")
for i in range(16):
    val = output_hi[i].item()
    if abs(val) > 0.1:
        print(f"  [{i:2d}]: {val:7.2f}")

# Manually compute logits for a few key bytes
print(f"\n" + "="*70)
print("Manual Logit Computation:")
print("-"*70)

head = model.head

for byte_val in [0, 2, 7, 12]:
    lo = byte_val & 0xF
    hi = (byte_val >> 4) & 0xF

    # Compute logit contribution from OUTPUT_LO and OUTPUT_HI
    logit_contrib = 0.0
    logit_contrib += head.weight[byte_val, BD.OUTPUT_LO + lo].item() * output_lo[lo].item()
    logit_contrib += head.weight[byte_val, BD.OUTPUT_HI + hi].item() * output_hi[hi].item()

    actual_logit = logits[0, pc_marker_pos, byte_val].item()

    print(f"\nByte {byte_val} (lo={lo}, hi={hi}):")
    print(f"  head.weight[{byte_val}, OUTPUT_LO+{lo}] * OUTPUT_LO[{lo}] = {head.weight[byte_val, BD.OUTPUT_LO + lo].item():.2f} * {output_lo[lo].item():.2f} = {head.weight[byte_val, BD.OUTPUT_LO + lo].item() * output_lo[lo].item():.2f}")
    print(f"  head.weight[{byte_val}, OUTPUT_HI+{hi}] * OUTPUT_HI[{hi}] = {head.weight[byte_val, BD.OUTPUT_HI + hi].item():.2f} * {output_hi[hi].item():.2f} = {head.weight[byte_val, BD.OUTPUT_HI + hi].item() * output_hi[hi].item():.2f}")
    print(f"  Computed logit contribution from OUTPUT: {logit_contrib:.2f}")
    print(f"  Actual logit: {actual_logit:.2f}")
    if abs(actual_logit - logit_contrib) > 0.1:
        print(f"  (Difference {actual_logit - logit_contrib:.2f} from other dimensions)")

# Find top 5 predictions
print(f"\n" + "="*70)
print("Top 5 Predictions:")
print("-"*70)
top5_vals, top5_idx = logits[0, pc_marker_pos, :].topk(5)
for val, idx in zip(top5_vals, top5_idx):
    marker = " ← expected" if idx.item() == 12 else ""
    marker = " ← PREDICTED" if idx.item() == 7 else marker
    print(f"  Byte {idx.item():3d}: logit {val.item():8.2f}{marker}")
