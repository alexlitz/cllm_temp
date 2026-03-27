"""Debug Layer 10 in actual forward pass with hooks."""
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

def decode_nibbles(lo_vals, hi_vals):
    lo_max = max(range(16), key=lambda k: lo_vals[k])
    hi_max = max(range(16), key=lambda k: hi_vals[k])
    return lo_max | (hi_max << 4)

model = AutoregressiveVM()
set_vm_weights(model)
model.compact(block_size=32)
model.compact_moe()
model.eval()

bytecode = [Opcode.JMP | (16 << 8)]
context = build_context(bytecode)

draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

current_context = context + [draft_tokens[0]]  # Just REG_PC marker
ctx_tensor = torch.tensor([current_context], dtype=torch.long)
pc_marker_pos = len(current_context) - 1

# Store activations using hooks
activations = {}

def get_hook(name):
    def hook(module, input, output):
        activations[name] = output.detach().clone()
    return hook

# Register hooks
model.blocks[9].register_forward_hook(get_hook('layer9_out'))
model.blocks[10].attn.register_forward_hook(get_hook('layer10_attn_out'))
model.blocks[10].ffn.register_forward_hook(get_hook('layer10_ffn_out'))
model.blocks[10].register_forward_hook(get_hook('layer10_out'))

with torch.no_grad():
    logits = model.forward(ctx_tensor)
    predicted = logits[0, -1, :].argmax().item()

print(f"=== LAYER 9 OUTPUT ===")
x = activations['layer9_out']
output_lo = [x[0, pc_marker_pos, BD.OUTPUT_LO + k].item() for k in range(16)]
output_hi = [x[0, pc_marker_pos, BD.OUTPUT_HI + k].item() for k in range(16)]
output_val = decode_nibbles(output_lo, output_hi)
print(f"OUTPUT = {output_val}")
print(f"HAS_SE = {x[0, pc_marker_pos, BD.HAS_SE].item():.3f}")

print(f"\n=== LAYER 10 ATTENTION OUTPUT ===")
x = activations['layer10_attn_out']
output_lo = [x[0, pc_marker_pos, BD.OUTPUT_LO + k].item() for k in range(16)]
output_hi = [x[0, pc_marker_pos, BD.OUTPUT_HI + k].item() for k in range(16)]
output_val = decode_nibbles(output_lo, output_hi)
print(f"OUTPUT = {output_val}")
print(f"OUTPUT_LO[0] = {output_lo[0]:.2f}")
print(f"OUTPUT_HI[0] = {output_hi[0]:.2f}")

print(f"\n=== LAYER 10 FFN OUTPUT ===")
x = activations['layer10_ffn_out']
output_lo = [x[0, pc_marker_pos, BD.OUTPUT_LO + k].item() for k in range(16)]
output_hi = [x[0, pc_marker_pos, BD.OUTPUT_HI + k].item() for k in range(16)]
output_val = decode_nibbles(output_lo, output_hi)
print(f"OUTPUT = {output_val}")

print(f"\n=== LAYER 10 FINAL OUTPUT ===")
x = activations['layer10_out']
output_lo = [x[0, pc_marker_pos, BD.OUTPUT_LO + k].item() for k in range(16)]
output_hi = [x[0, pc_marker_pos, BD.OUTPUT_HI + k].item() for k in range(16)]
output_val = decode_nibbles(output_lo, output_hi)
print(f"OUTPUT = {output_val}")

print(f"\n=== FINAL PREDICTION ===")
print(f"Predicted: {predicted}")
print(f"Expected:  {draft_tokens[1]} (JMP target)")
