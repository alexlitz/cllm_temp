"""Debug PC byte 0 prediction issue."""
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

# Test NOP (simplest case)
bytecode = [Opcode.NOP]
draft_vm = DraftVM(bytecode)
context = build_context(bytecode)

draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

print("DraftVM after NOP step:")
print(f"  PC should be 8 (advanced from 0)")
print(f"  draft_tokens[0] = {draft_tokens[0]} (REG_PC marker)")
print(f"  draft_tokens[1] = {draft_tokens[1]} (PC byte 0, should be 8)")
print(f"  draft_tokens[2] = {draft_tokens[2]} (PC byte 1, should be 0)")
print()

model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

ctx_tensor = torch.tensor([context + draft_tokens], dtype=torch.long)
ctx_len = len(context)

with torch.no_grad():
    # Get hidden states
    x = model.embed(ctx_tensor)
    model._inject_code_addr_keys(ctx_tensor, x)
    model._inject_mem_store(ctx_tensor, x)

    for i, block in enumerate(model.blocks):
        x = block(x)
        if i in [5, 6, 10, 15]:
            # Check OUTPUT at PC marker position
            pc_marker_pos = ctx_len
            output_lo = x[0, pc_marker_pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
            output_hi = x[0, pc_marker_pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16]
            lo_val = output_lo.argmax().item()
            hi_val = output_hi.argmax().item()
            byte_val = lo_val | (hi_val << 4)
            print(f"After L{i}: PC marker OUTPUT = {byte_val} (lo={lo_val}, hi={hi_val}, expect 8)")

    # Final logits
    logits = model.head(x)

    # Check PC byte 0 prediction
    pc_byte0_logits = logits[0, ctx_len, :]  # logits at marker position predict byte 0
    predicted = pc_byte0_logits.argmax().item()
    expected = draft_tokens[1]

    print()
    print(f"PC byte 0 prediction:")
    print(f"  Expected: {expected}")
    print(f"  Predicted: {predicted}")
    print(f"  Logits for 0: {pc_byte0_logits[0].item():.2f}")
    print(f"  Logits for 8: {pc_byte0_logits[8].item():.2f}")

    # Check what the final OUTPUT dimensions are
    output_lo_final = x[0, ctx_len, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    output_hi_final = x[0, ctx_len, BD.OUTPUT_HI:BD.OUTPUT_HI+16]
    print()
    print(f"Final OUTPUT_LO: {output_lo_final}")
    print(f"Final OUTPUT_HI: {output_hi_final}")
