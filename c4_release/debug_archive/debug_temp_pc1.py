"""Check if TEMP contains PC+1 correctly."""
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim as BD
from neural_vm.embedding import Opcode

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

bytecode = [Opcode.IMM | (42 << 8)]
context = build_context(bytecode)

# Add draft tokens for first step  
from neural_vm.speculative import DraftVM
draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

ctx_tensor = torch.tensor([context + draft_tokens], dtype=torch.long)
ctx_len = len(context)
ax_marker_pos = ctx_len + 5

print(f"Context length: {ctx_len}")
print(f"AX marker position: {ax_marker_pos}")
print(f"\nPC for first step should be 0 (before execution)")
print(f"PC+1 should be 1 (where immediate is stored)")
print(f"Immediate byte in context at position 2: {context[2]} (expecting 42)")

with torch.no_grad():
    x = model.embed(ctx_tensor)
    model._inject_code_addr_keys(ctx_tensor, x)
    model._inject_mem_store(ctx_tensor, x)

    # After L4 (which should set TEMP to PC+1)
    for i in range(5):
        x = model.blocks[i](x)
        if i == 3:  # After L4
            print(f"\nAfter L4 FFN (should set TEMP to PC+1):")
            temp = x[0, ax_marker_pos, BD.TEMP:BD.TEMP+32]
            temp_lo = temp[:16].argmax().item()
            temp_hi = temp[16:32].argmax().item()
            temp_val = temp_lo | (temp_hi << 4)
            print(f"  TEMP[0..15] (lo nibble): {temp_lo}")
            print(f"  TEMP[16..31] (hi nibble): {temp_hi}")
            print(f"  TEMP as byte: {temp_val} (expected 1 for PC+1)")
            
            # Also check what PC is
            embed_lo = x[0, ax_marker_pos, BD.EMBED_LO:BD.EMBED_LO+16]
            embed_hi = x[0, ax_marker_pos, BD.EMBED_HI:BD.EMBED_HI+16]
            pc_lo = embed_lo.argmax().item()
            pc_hi = embed_hi.argmax().item()
            pc_val = pc_lo | (pc_hi << 4)
            print(f"  PC (EMBED): {pc_val} (expected 0)")

