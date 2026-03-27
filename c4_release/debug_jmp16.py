"""Debug JMP 16 - why does PC byte 0 predict 8 instead of 16?"""
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
model.eval()

# Test both JMP 8 (passes) and JMP 16 (fails)
import sys
jmp_target = int(sys.argv[1]) if len(sys.argv) > 1 else 16
bytecode = [Opcode.JMP | (jmp_target << 8)]
print(f"Testing JMP {jmp_target}")
context = build_context(bytecode)

draft_vm = DraftVM(bytecode)
draft_vm.step()
draft1 = draft_vm.draft_tokens()

print("Expected PC bytes:", [257, 16, 0, 0, 0])
print("Draft VM PC tokens:", draft1[0:5])

context_step1 = context + draft1
ctx_tensor = torch.tensor([context_step1], dtype=torch.long)
ctx_len = len(context)
pc_marker_pos = ctx_len
pc_byte0_pos = ctx_len + 1

with torch.no_grad():
    x = model.embed(ctx_tensor)
    model._inject_code_addr_keys(ctx_tensor, x)
    model._inject_mem_store(ctx_tensor, x)

    # Check if code injection worked
    print("\n=== CODE INJECTION CHECK ===")
    # JMP 16 instruction at PC=0 should have immediate=16 at offset +1
    # CODE_START is at position 0, instruction starts at position 1
    # Format: [opcode_byte, imm_b0, imm_b1, imm_b2, imm_b3]
    instr_start = 1  # Position of opcode byte
    print(f"Instruction bytes in context: {ctx_tensor[0, instr_start:instr_start+5].tolist()}")
    print(f"Expected: [2, 16, 0, 0, 0] (JMP opcode=2, imm=16)")

    # Check ADDR_KEY at the immediate byte position (should encode address 1)
    imm_pos = instr_start + 1  # First immediate byte
    addr_key_val = x[0, imm_pos, BD.ADDR_KEY]
    print(f"\nADDR_KEY at position {imm_pos} (immediate byte): {addr_key_val.item():.3f}")
    print(f"This position should have address=1 encoded in ADDR_KEY")

    # Check key layers
    for layer_idx in [3, 4, 5, 6]:
        if layer_idx > 0:
            for i in range(layer_idx):
                x = model.blocks[i](x)

        print(f"\n=== AFTER L{layer_idx} ===")
        output_lo = x[0, pc_marker_pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
        output_hi = x[0, pc_marker_pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16]
        fetch_lo = x[0, pc_marker_pos, BD.FETCH_LO:BD.FETCH_LO+16]
        fetch_hi = x[0, pc_marker_pos, BD.FETCH_HI:BD.FETCH_HI+16]
        embed_lo = x[0, pc_marker_pos, BD.EMBED_LO:BD.EMBED_LO+16]
        embed_hi = x[0, pc_marker_pos, BD.EMBED_HI:BD.EMBED_HI+16]
        pc_lo = output_lo.argmax().item()
        pc_hi = output_hi.argmax().item()
        embed_pc_lo = embed_lo.argmax().item()
        embed_pc_hi = embed_hi.argmax().item()
        print(f"PC marker OUTPUT: lo={pc_lo}, hi={pc_hi} → PC={pc_lo | (pc_hi << 4)}")
        print(f"PC marker EMBED: lo={embed_pc_lo}, hi={embed_pc_hi} → PC={embed_pc_lo | (embed_pc_hi << 4)}")
        print(f"PC marker FETCH_LO: max={fetch_lo.max().item():.3f} at index {fetch_lo.argmax().item()}")
        print(f"PC marker FETCH_HI: max={fetch_hi.max().item():.3f} at index {fetch_hi.argmax().item()}")

        # Reset for next iteration
        x = model.embed(ctx_tensor)
        model._inject_code_addr_keys(ctx_tensor, x)
        model._inject_mem_store(ctx_tensor, x)

    # Full forward pass
    logits = model.forward(ctx_tensor)
    predicted_pc_byte0 = logits[0, ctx_len, :].argmax().item()
    print(f"\n=== PREDICTION ===")
    print(f"PC byte 0: expected=16, predicted={predicted_pc_byte0}")
