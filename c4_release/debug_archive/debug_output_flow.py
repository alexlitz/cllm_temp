"""Track OUTPUT flow through AX marker and bytes for IMM 42."""
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

print("OUTPUT Flow for IMM 42")
print("=" * 70)
print()

bytecode = [Opcode.IMM | (42 << 8)]
context = build_context(bytecode)

draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

# Track OUTPUT at positions 13, 14, 15, 16
for test_idx in [4, 5, 6, 7]:  # Tokens 4, 5, 6, 7
    ctx = context + draft_tokens[:test_idx]

    # Capture L6 FFN input and L15 output
    l6_ffn_in = None
    l15_out = None

    def l6_hook(module, input, output):
        global l6_ffn_in
        if isinstance(input, tuple):
            l6_ffn_in = input[0].detach().clone()
        else:
            l6_ffn_in = input.detach().clone()

    def l15_hook(module, input, output):
        global l15_out
        l15_out = output.detach().clone()

    model.blocks[6].ffn.register_forward_hook(l6_hook)
    model.blocks[15].register_forward_hook(l15_hook)

    ctx_tensor = torch.tensor([ctx], dtype=torch.long)
    with torch.no_grad():
        logits = model.forward(ctx_tensor)

    pos = len(ctx) - 1
    token_val = ctx[-1]
    expected_next = draft_tokens[test_idx] if test_idx < len(draft_tokens) else None
    predicted = torch.argmax(logits[0, -1]).item()

    print(f"Position {pos} (token {test_idx-1} = {token_val}):")

    # Get OUTPUT at L6 FFN input (before L6 processes it)
    if l6_ffn_in is not None:
        output_lo = l6_ffn_in[0, pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
        output_hi = l6_ffn_in[0, pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16]
        output_lo_val = torch.argmax(output_lo).item()
        output_hi_val = torch.argmax(output_hi).item()
        output_byte = output_lo_val + (output_hi_val << 4)
        print(f"  L6 FFN input OUTPUT: {output_byte} (0x{output_byte:02X})")

    # Get OUTPUT at L15 output (after all layers)
    if l15_out is not None:
        output_lo = l15_out[0, pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
        output_hi = l15_out[0, pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16]
        output_lo_val = torch.argmax(output_lo).item()
        output_hi_val = torch.argmax(output_hi).item()
        output_byte = output_lo_val + (output_hi_val << 4)
        print(f"  L15 output OUTPUT: {output_byte} (0x{output_byte:02X})")

    print(f"  Predicted next: {predicted}")
    if expected_next is not None:
        print(f"  Expected next: {expected_next}")
        match = "✓" if predicted == expected_next else "✗"
        print(f"  Match: {match}")
    print()

print("Analysis:")
print("  Position 13 (PC_b3): Should predict REG_AX (258)")
print("  Position 14 (REG_AX): Should predict AX_b0 (42)")
print("  Position 15 (AX_b0): Should predict AX_b1 (0)")
