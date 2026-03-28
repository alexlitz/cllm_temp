"""Debug LEA 8 - check ALL opcodes at L10 input."""
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

bytecode = [Opcode.LEA | (8 << 8)]
context = build_context(bytecode)

draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

# Check at AX marker
ctx = context + draft_tokens[:6]
pos = len(ctx) - 1

print("LEA 8 - Opcode Flags at L10 Input (AX marker)")
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

    # Check all relevant opcodes
    print("Opcode flags:")
    print(f"  OP_IMM:  {x[BD.OP_IMM].item():.3f}")
    print(f"  OP_ADD:  {x[BD.OP_ADD].item():.3f}")
    print(f"  OP_SUB:  {x[BD.OP_SUB].item():.3f}")
    print(f"  OP_MUL:  {x[BD.OP_MUL].item():.3f}")
    print(f"  OP_AND:  {x[BD.OP_AND].item():.3f}")
    print(f"  OP_OR:   {x[BD.OP_OR].item():.3f}")
    print(f"  OP_XOR:  {x[BD.OP_XOR].item():.3f}")
    print(f"  OP_SHL:  {x[BD.OP_SHL].item():.3f}")
    print(f"  OP_SHR:  {x[BD.OP_SHR].item():.3f}")
    print(f"  OP_LEA:  {x[BD.OP_LEA].item():.3f}")
    print(f"  OP_PSH:  {x[BD.OP_PSH].item():.3f}")
    print(f"  OP_POP:  {x[BD.OP_POP].item():.3f}")
    print()

    # Check marker flags
    print("Marker flags:")
    print(f"  MARK_AX: {x[BD.MARK_AX].item():.3f}")
    print(f"  MARK_PC: {x[BD.MARK_PC].item():.3f}")
    print()

    # Check ALU inputs
    alu_lo = x[BD.ALU_LO:BD.ALU_LO+16]
    alu_hi = x[BD.ALU_HI:BD.ALU_HI+16]
    print(f"ALU input: {torch.argmax(alu_lo).item()} + 16*{torch.argmax(alu_hi).item()}")

    # Check AX_CARRY
    ax_carry_lo = x[BD.AX_CARRY_LO:BD.AX_CARRY_LO+16]
    ax_carry_hi = x[BD.AX_CARRY_HI:BD.AX_CARRY_HI+16]
    print(f"AX_CARRY: {torch.argmax(ax_carry_lo).item()} + 16*{torch.argmax(ax_carry_hi).item()}")

    # Check OUTPUT before L10 FFN
    output_lo = x[BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    output_hi = x[BD.OUTPUT_HI:BD.OUTPUT_HI+16]
    print(f"OUTPUT (before L10 FFN): {torch.argmax(output_lo).item()} + 16*{torch.argmax(output_hi).item()}")
    print()

    # Check if any bitwise opcode is active (should be 0 for LEA)
    bitwise_sum = x[BD.OP_AND].item() + x[BD.OP_OR].item() + x[BD.OP_XOR].item()
    print(f"Bitwise opcode sum (AND+OR+XOR): {bitwise_sum:.3f}")
    print(f"  → Should be 0 for LEA (no bitwise ops)")
    print()

    # Check what the EfficientALU_L10_Neural sees
    print("Analysis:")
    if bitwise_sum > 0.5:
        print("  ❌ ERROR: Bitwise opcodes are active when they shouldn't be!")
    else:
        print("  ✓ Bitwise opcodes are correctly inactive")
        print("  → But OUTPUT still changes from 8 to 1")
        print("  → This suggests the issue is NOT in opcode activation")
        print("  → Possible causes:")
        print("    1. GEToBDConverter writing OUTPUT even with opcode_mask=0")
        print("    2. Different logic in EfficientALU_L10_Neural interfering")
        print("    3. Attention residual from L10.attn (but already checked - no change)")
