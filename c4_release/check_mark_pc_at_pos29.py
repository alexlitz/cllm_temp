"""Check if MARK_PC could be used as discriminator instead of IS_BYTE."""
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

# Hook to capture Layer 6 FFN input
l6_ffn_in = None
def hook(module, input, output):
    global l6_ffn_in
    l6_ffn_in = input[0].detach().clone()

model.blocks[6].ffn.register_forward_hook(hook)

with torch.no_grad():
    _ = model.forward(ctx_tensor)

print("Marker Analysis for PSH Discriminator")
print("=" * 70)
print()

# Check positions 28 (STACK0 marker) and 29 (byte position)
for pos, desc in [(28, "Position 28 (Token 20 = STACK0 marker)"),
                   (29, "Position 29 (Token 21 = ST_b0 byte)")]:
    print(desc)
    print("-" * 70)

    if l6_ffn_in is not None:
        mark_pc = l6_ffn_in[0, pos, BD.MARK_PC].item()
        mark_stack0 = l6_ffn_in[0, pos, BD.MARK_STACK0].item()
        mark_sp = l6_ffn_in[0, pos, BD.MARK_SP].item()
        mark_bp = l6_ffn_in[0, pos, BD.MARK_BP].item()
        is_byte = l6_ffn_in[0, pos, BD.IS_BYTE].item()
        cmp0 = l6_ffn_in[0, pos, BD.CMP + 0].item()

        print(f"  MARK_PC:     {mark_pc:.3f}")
        print(f"  MARK_STACK0: {mark_stack0:.3f}")
        print(f"  MARK_SP:     {mark_sp:.3f}")
        print(f"  MARK_BP:     {mark_bp:.3f}")
        print(f"  IS_BYTE:     {is_byte:.3f}")
        print(f"  CMP[0]:      {cmp0:.3f}")
        print()

        # Test different discriminator strategies
        print("  Discriminator Tests:")

        # Strategy 1: MARK_STACK0 AND NOT IS_BYTE
        disc1 = (mark_stack0 > 0.5) and (is_byte < 0.5)
        print(f"    MARK_STACK0 AND NOT IS_BYTE: {disc1}")

        # Strategy 2: MARK_STACK0 AND NOT MARK_PC
        disc2 = (mark_stack0 > 0.5) and (mark_pc < 0.5)
        print(f"    MARK_STACK0 AND NOT MARK_PC: {disc2}")

        # Strategy 3: MARK_STACK0 AND (CMP[0] < 2.0)
        disc3 = (mark_stack0 > 0.5) and (cmp0 < 2.0)
        print(f"    MARK_STACK0 AND (CMP[0] < 2.0): {disc3}")

        print()

print("Analysis:")
print("-" * 70)
print("At STACK0 marker (pos 28):")
print("  - Should activate PSH units: YES")
print("  - Expected discriminators: All should be True (if PSH active)")
print()
print("At byte position (pos 29):")
print("  - Should activate PSH units: NO")
print("  - Expected discriminators: All should be False")
print()
print("Best discriminator: One that is True at pos 28, False at pos 29")
