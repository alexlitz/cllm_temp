"""Debug LEA 8 - check why L10 AX passthrough activates."""
import sys
for mod in list(sys.modules.keys()):
    if 'neural_vm' in mod:
        del sys.modules[mod]

import torch
import torch.nn.functional as F
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

bytecode = [Opcode.LEA | (8 << 8)]
context = build_context(bytecode)

draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

# Check at AX marker
ctx = context + draft_tokens[:6]
pos = len(ctx) - 1

print("LEA 8 - L10 AX Passthrough Debug")
print("=" * 70)
print()

# Capture L10 FFN input
l10_ffn_in = None

def l10_ffn_hook(module, input, output):
    global l10_ffn_in
    if isinstance(input, tuple):
        l10_ffn_in = input[0].detach().clone()
    else:
        l10_ffn_in = input.detach().clone()

model.blocks[10].ffn.register_forward_hook(l10_ffn_hook)

ctx_tensor = torch.tensor([ctx], dtype=torch.long)
with torch.no_grad():
    logits = model.forward(ctx_tensor)

if l10_ffn_in is not None:
    x = l10_ffn_in[0, pos, :]

    print(f"Position {pos} (REG_AX marker):")
    print()

    # Check AX passthrough activation conditions
    mark_ax = x[BD.MARK_AX].item()
    op_lea = x[BD.OP_LEA].item()
    ax_carry_lo = x[BD.AX_CARRY_LO:BD.AX_CARRY_LO+16]

    print("AX Passthrough activation conditions:")
    print(f"  MARK_AX: {mark_ax:.3f}")
    print(f"  OP_LEA: {op_lea:.3f}")
    print()

    # The AX passthrough logic (lines 4013-4028):
    # up = S*MARK_AX + (-S)*OP_LEA + ... - S*0.5
    # For LEA with OP_LEA active, up should be negative
    S = 100.0
    up_expected = S*mark_ax - S*op_lea - S*0.5
    print(f"Expected 'up' for AX passthrough: {up_expected:.1f}")
    if up_expected > 0:
        print(f"  ❌ Positive! AX passthrough WILL activate")
        print(f"     MARK_AX({mark_ax:.1f}) - OP_LEA({op_lea:.1f}) - 0.5 = {mark_ax - op_lea - 0.5:.1f}")
        print(f"     → Scaled: {up_expected:.1f}")
    else:
        print(f"  ✓ Negative! AX passthrough should NOT activate")

    print()
    print("Analysis:")
    if op_lea < 1.0:
        print(f"  ❌ OP_LEA = {op_lea:.3f} is too low!")
        print(f"     Expected ≈ 5.0 for LEA opcode")
        print(f"     → Suppression fails because -S*OP_LEA is too small")
    else:
        print(f"  ✓ OP_LEA = {op_lea:.3f} is reasonable")
