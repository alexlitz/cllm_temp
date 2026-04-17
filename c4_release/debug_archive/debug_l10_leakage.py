"""Check if L10 is leaking to PC byte positions."""
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
pc_byte0_pos = ctx_len + 1

with torch.no_grad():
    _ = model.forward(ctx_tensor)

print("Checking L10 leakage at PC byte 0:")
print("=" * 70)

# Check H1 values
x_before = layer_outputs["L9"][0, pc_byte0_pos, :]
h1_ax = x_before[BD.H1 + 1].item()
print(f"H1[AX_IDX] at PC byte 0: {h1_ax:.6f}")
print(f"  (Should be 0.0 to suppress L10 AX passthrough)")
print()

# Check OUTPUT_LO changes
output_lo_before = x_before[BD.OUTPUT_LO:BD.OUTPUT_LO+16]
x_after = layer_outputs["L10"][0, pc_byte0_pos, :]
output_lo_after = x_after[BD.OUTPUT_LO:BD.OUTPUT_LO+16]
output_lo_delta = output_lo_after - output_lo_before

print("OUTPUT_LO changes from L9 → L10:")
active = (output_lo_delta.abs() > 0.3).nonzero(as_tuple=True)[0].tolist()
if active:
    for idx in active:
        print(f"  OUTPUT_LO[{idx}]: {output_lo_before[idx].item():.3f} → {output_lo_after[idx].item():.3f} (Δ={output_lo_delta[idx].item():+.3f})")
    print()
    print("  ❌ LEAKAGE DETECTED\! L10 should not write OUTPUT at PC byte positions.")
else:
    print("  ✓ No leakage - suppression working correctly")
