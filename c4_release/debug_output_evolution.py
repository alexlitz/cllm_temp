"""Trace OUTPUT_LO evolution at PC byte 0 position."""
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

# Hook to capture all layer outputs
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
pc_byte0_pos = ctx_len + 1  # Position of PC byte 0 token (value 10)

with torch.no_grad():
    _ = model.forward(ctx_tensor)

print("Layer-by-layer OUTPUT_LO at PC byte 0 position:")
print("=" * 70)
print(f"Token at this position: {draft_tokens[1]} (PC byte 0 = 10)")
print(f"Expected next token (PC byte 1): {draft_tokens[2]} (should be 0)")
print()

for i in range(16):
    x = layer_outputs[f"L{i}"][0, pc_byte0_pos, :]
    output_lo = x[BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    active_indices = (output_lo.abs() > 0.3).nonzero(as_tuple=True)[0].tolist()
    active_values = [output_lo[idx].item() for idx in active_indices]

    if active_indices:
        print(f"L{i:2d}: OUTPUT_LO{active_indices} = {[f'{v:.3f}' for v in active_values]}")
    else:
        print(f"L{i:2d}: OUTPUT_LO all zero/small")

print()
print("=" * 70)
print("Diagnosis:")
print("-" * 70)

# Check where OUTPUT_LO[5] first appears
for i in range(16):
    x = layer_outputs[f"L{i}"][0, pc_byte0_pos, :]
    output_lo_5 = x[BD.OUTPUT_LO + 5].item()
    if abs(output_lo_5) > 0.3:
        print(f"OUTPUT_LO[5] first becomes significant at L{i}: {output_lo_5:.3f}")
        break

# Also check EMBED and other dimensions at this position
print()
print("Other dimensions at PC byte 0 position:")
print("-" * 70)
x = layer_outputs["L2"][0, pc_byte0_pos, :]  # After embedding
embed_lo = x[BD.EMBED_LO:BD.EMBED_LO+16]
embed_hi = x[BD.EMBED_HI:BD.EMBED_HI+16]
print(f"  EMBED_LO active: {(embed_lo.abs() > 0.3).nonzero(as_tuple=True)[0].tolist()}")
print(f"  EMBED_HI active: {(embed_hi.abs() > 0.3).nonzero(as_tuple=True)[0].tolist()}")
print(f"  (PC byte 0 = 10 = 0xA, so EMBED_LO[10], EMBED_HI[0] should be active)")
