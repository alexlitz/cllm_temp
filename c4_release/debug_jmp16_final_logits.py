"""Debug JMP 16 - check final logits and OUTPUT."""
import sys
for mod in list(sys.modules.keys()):
    if 'neural_vm' in mod:
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

print("JMP 16 - Final Logits and OUTPUT")
print("=" * 70)
print()

bytecode = [Opcode.JMP | (16 << 8)]
context = build_context(bytecode)

draft_vm = DraftVM(bytecode)
draft_vm.step()

# Capture final layer output
l15_out = None
def l15_hook(module, input, output):
    global l15_out
    l15_out = output.detach().clone()

model.blocks[15].register_forward_hook(l15_hook)

ctx_tensor = torch.tensor([context], dtype=torch.long)
with torch.no_grad():
    logits = model.forward(ctx_tensor)

pos = len(context) - 1
print(f"Position {pos} (last token = {context[pos]}):")
print()

# Check OUTPUT in final hidden state
if l15_out is not None:
    output_lo = l15_out[0, pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    output_hi = l15_out[0, pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16]
    lo_idx = torch.argmax(output_lo).item()
    hi_idx = torch.argmax(output_hi).item()
    output_val = lo_idx + 16 * hi_idx
    print(f"OUTPUT in hidden state (L15 output):")
    print(f"  LO: argmax={lo_idx}, max={torch.max(output_lo).item():.3f}")
    print(f"  HI: argmax={hi_idx}, max={torch.max(output_hi).item():.3f}")
    print(f"  Value: {output_val}")
    print()

# Check logits
pred = torch.argmax(logits[0, pos, :]).item()
print(f"Final prediction: {pred}")
print()

# Check top 5 logits
top5 = torch.topk(logits[0, pos, :], 5)
print("Top 5 logits:")
for i in range(5):
    tok = top5.indices[i].item()
    val = top5.values[i].item()
    print(f"  {tok:3d}: {val:8.3f}")
print()

# Check specific tokens
print("Specific token logits:")
print(f"  Token 16: {logits[0, pos, 16].item():.3f}")
print(f"  Token 257 (REG_AX+1): {logits[0, pos, 257].item():.3f}")
print(f"  Token 256 (REG_AX): {logits[0, pos, 256].item():.3f}")
