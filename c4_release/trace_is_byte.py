"""Trace IS_BYTE through all layers to find where it's being cleared."""
import sys
for mod in ['neural_vm.vm_step', 'neural_vm.embedding', 'neural_vm.speculative']:
    if mod in sys.modules:
        del sys.modules[mod]

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

current_context = context[:]
for i in range(21):
    current_context.append(draft_tokens[i])

ctx_tensor = torch.tensor([current_context], dtype=torch.long)
pos = 29  # Position of token 21 (byte 0)

# Capture IS_BYTE after each layer
layer_outputs = {}

for layer_idx in range(16):
    def make_hook(idx):
        def hook(module, input, output):
            layer_outputs[idx] = output.detach().clone()
        return hook
    model.blocks[layer_idx].register_forward_hook(make_hook(layer_idx))

# Also capture embedding output
embed_out = None
def hook_embed(module, input, output):
    global embed_out
    embed_out = output.detach().clone()

model.embed.register_forward_hook(hook_embed)

with torch.no_grad():
    _ = model.forward(ctx_tensor)

print("IS_BYTE Trace at Position 29 (Token 21 = byte 0)")
print("=" * 70)
print()

# Check embedding
if embed_out is not None:
    is_byte_embed = embed_out[0, pos, BD.IS_BYTE].item()
    print(f"Embedding:  IS_BYTE = {is_byte_embed:.3f}")
    print()

# Check each layer
print("After each layer:")
for layer_idx in range(16):
    if layer_idx in layer_outputs:
        is_byte = layer_outputs[layer_idx][0, pos, BD.IS_BYTE].item()

        marker = ""
        if is_byte < 0.5 and layer_idx == 0:
            marker = " ← FIRST DROP TO 0"
        elif is_byte < 0.5:
            marker = " ← Still 0"

        print(f"L{layer_idx:2d}: IS_BYTE = {is_byte:6.3f}{marker}")

print()
print("Expected: IS_BYTE should remain 1.000 at byte positions throughout")
print("Actual: IS_BYTE is being cleared, likely by attention writing to it")
