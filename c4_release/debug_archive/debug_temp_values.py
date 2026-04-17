"""Check actual TEMP values."""
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

bytecode = [Opcode.IMM | (42 << 8)]
context = build_context(bytecode)

draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

ctx_tensor = torch.tensor([context + draft_tokens], dtype=torch.long)
ctx_len = len(context)
ax_marker_pos = ctx_len + 5

with torch.no_grad():
    x = model.embed(ctx_tensor)
    model._inject_code_addr_keys(ctx_tensor, x)
    model._inject_mem_store(ctx_tensor, x)

    # After L4
    for i in range(5):
        x = model.blocks[i](x)
        if i == 3:  # After L4
            print("After L4 - checking TEMP and EMBED:")
            temp = x[0, ax_marker_pos, BD.TEMP:BD.TEMP+32]
            embed_lo = x[0, ax_marker_pos, BD.EMBED_LO:BD.EMBED_LO+16]
            
            print(f"\nEMBED_LO (PC): {embed_lo}")
            print(f"EMBED_LO max at index {embed_lo.argmax().item()} with value {embed_lo.max().item():.3f}")
            
            print(f"\nTEMP[0..15] (PC+1 lo nibble): {temp[:16]}")
            print(f"TEMP[0..15] max at index {temp[:16].argmax().item()} with value {temp[:16].max().item():.3f}")
            
            # Check if TEMP has any non-zero values
            if temp.abs().max().item() < 0.01:
                print("\n⚠️  TEMP is all zeros! L4 FFN is not working.")
            else:
                print(f"\n✓ TEMP has non-zero values (max: {temp.abs().max().item():.3f})")

