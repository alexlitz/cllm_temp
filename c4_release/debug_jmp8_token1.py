"""Debug JMP 8 token 1 (PC_b0) failure."""
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

print("JMP 8 Token 1 Debug")
print("=" * 70)
print()

# Compare JMP 16 (works) vs JMP 8 (fails)
for jmp_val in [16, 8]:
    print(f"JMP {jmp_val}:")
    print("-" * 70)

    bytecode = [Opcode.JMP | (jmp_val << 8)]
    context = build_context(bytecode)

    draft_vm = DraftVM(bytecode)
    draft_vm.step()
    draft_tokens = draft_vm.draft_tokens()

    print(f"JMP target: {jmp_val} = 0x{jmp_val:02X}")
    print(f"PC bytes: {[draft_tokens[i] for i in range(1, 5)]}")
    print()

    # Test token 1 (PC_b0)
    ctx = context + [draft_tokens[0]]

    # Capture L15 output
    l15_out = None
    def hook(module, input, output):
        global l15_out
        l15_out = output.detach().clone()

    model.blocks[15].register_forward_hook(hook)

    ctx_tensor = torch.tensor([ctx], dtype=torch.long)
    with torch.no_grad():
        logits = model.forward(ctx_tensor)

    predicted = torch.argmax(logits[0, -1]).item()
    expected = draft_tokens[1]

    if l15_out is not None:
        pos = len(ctx) - 1
        output_lo = l15_out[0, pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
        output_hi = l15_out[0, pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16]
        output_lo_val = torch.argmax(output_lo).item()
        output_hi_val = torch.argmax(output_hi).item()
        output_byte = output_lo_val + (output_hi_val << 4)

        print(f"OUTPUT after L15:")
        print(f"  OUTPUT_LO argmax: {output_lo_val} (0x{output_lo_val:X})")
        print(f"  OUTPUT_HI argmax: {output_hi_val} (0x{output_hi_val:X})")
        print(f"  Decoded byte: {output_byte} (0x{output_byte:02X})")
        print()
        print(f"  OUTPUT_LO top 5: {output_lo.topk(5)}")
        print()

    print(f"Predicted: {predicted}")
    print(f"Expected:  {expected}")
    match = "✓" if predicted == expected else "✗"
    print(f"Match: {match}")
    print()
    print()

print("Analysis:")
print("  JMP 16 = 0x10: nibbles [0, 1], expected PC_b0 = 16")
print("  JMP 8 = 0x08: nibbles [8, 0], expected PC_b0 = 8")
print()
print("  If JMP 8 predicts 2, that suggests:")
print("    - OUTPUT_LO might be wrong (2 instead of 8)")
print("    - Or nibble encoding/decoding issue")
