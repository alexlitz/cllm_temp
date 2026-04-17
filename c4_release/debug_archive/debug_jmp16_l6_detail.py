"""Debug JMP 16 L6 - check JMP relay and opcode flags."""
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

print("JMP 16 L6 Detail - Opcode Flags and JMP Logic")
print("=" * 70)
print()

bytecode = [Opcode.JMP | (16 << 8)]
context = build_context(bytecode)

draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

ctx = context + [draft_tokens[0]]

# Capture L6 input/output
l6_in = None
l6_out = None

def l6_in_hook(module, input, output):
    global l6_in
    if isinstance(input, tuple):
        l6_in = input[0].detach().clone()
    else:
        l6_in = input.detach().clone()

def l6_out_hook(module, input, output):
    global l6_out
    l6_out = output.detach().clone()

model.blocks[6].register_forward_hook(l6_in_hook)
model.blocks[6].register_forward_hook(l6_out_hook)

ctx_tensor = torch.tensor([ctx], dtype=torch.long)
with torch.no_grad():
    logits = model.forward(ctx_tensor)

pos = len(ctx) - 1
print(f"Position {pos} (last token = {ctx[-1]} = REG_PC)")
print()

if l6_in is not None:
    print("L6 FFN Input:")
    print("-" * 70)

    # Check opcode flags
    op_jmp = l6_in[0, pos, BD.OP_JMP].item()
    has_se = l6_in[0, pos, BD.HAS_SE].item()
    mark_pc = l6_in[0, pos, BD.MARK_PC].item()
    cmp0 = l6_in[0, pos, BD.CMP + 0].item()
    cmp1 = l6_in[0, pos, BD.CMP + 1].item()

    print(f"  OP_JMP: {op_jmp:.3f}")
    print(f"  HAS_SE: {has_se:.3f}")
    print(f"  MARK_PC: {mark_pc:.3f}")
    print(f"  CMP[0]: {cmp0:.3f}")
    print(f"  CMP[1]: {cmp1:.3f}")
    print()

    # Check FETCH (immediate value)
    fetch_lo = l6_in[0, pos, BD.FETCH_LO:BD.FETCH_LO+16]
    fetch_hi = l6_in[0, pos, BD.FETCH_HI:BD.FETCH_HI+16]
    fetch_lo_val = torch.argmax(fetch_lo).item()
    fetch_hi_val = torch.argmax(fetch_hi).item()
    fetch_byte = fetch_lo_val + (fetch_hi_val << 4)

    print(f"  FETCH value: {fetch_byte} (lo={fetch_lo_val}, hi={fetch_hi_val})")
    print(f"    Expected: 16 for JMP 16")
    print()

    # Check AX_CARRY (JMP uses this for target)
    ax_carry_lo = l6_in[0, pos, BD.AX_CARRY_LO:BD.AX_CARRY_LO+16]
    ax_carry_hi = l6_in[0, pos, BD.AX_CARRY_HI:BD.AX_CARRY_HI+16]
    ax_carry_lo_val = torch.argmax(ax_carry_lo).item()
    ax_carry_hi_val = torch.argmax(ax_carry_hi).item()
    ax_carry_byte = ax_carry_lo_val + (ax_carry_hi_val << 4)

    print(f"  AX_CARRY value: {ax_carry_byte} (lo={ax_carry_lo_val}, hi={ax_carry_hi_val})")
    print()

    # Check OUTPUT before L6
    output_lo_in = l6_in[0, pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    output_hi_in = l6_in[0, pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16]
    output_lo_val = torch.argmax(output_lo_in).item()
    output_hi_val = torch.argmax(output_hi_in).item()
    output_byte_in = output_lo_val + (output_hi_val << 4)

    print(f"  OUTPUT before: {output_byte_in} (lo={output_lo_val}, hi={output_hi_val})")

if l6_out is not None:
    print()
    print("L6 Output:")
    print("-" * 70)

    output_lo_out = l6_out[0, pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    output_hi_out = l6_out[0, pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16]
    output_lo_val = torch.argmax(output_lo_out).item()
    output_hi_val = torch.argmax(output_hi_out).item()
    output_byte_out = output_lo_val + (output_hi_val << 4)

    print(f"  OUTPUT after: {output_byte_out} (lo={output_lo_val}, hi={output_hi_val})")
    print()

print()
print("Analysis:")
print("  JMP in L6 should:")
print("    1. Detect OP_JMP + MARK_PC + HAS_SE")
print("    2. Cancel default PC increment (OUTPUT=8)")
print("    3. Write AX_CARRY (jump target=16) to OUTPUT")
print()
print("  If OP_JMP is set but OUTPUT becomes 0:")
print("    - JMP units are canceling but not adding target")
print("    - Or another unit is interfering")
