"""Check what L3 actually sets at PC marker."""
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
pc_marker_pos = ctx_len

with torch.no_grad():
    x = model.embed(ctx_tensor)
    model._inject_code_addr_keys(ctx_tensor, x)
    model._inject_mem_store(ctx_tensor, x)

    # After L3
    for i in range(3):
        x = model.blocks[i](x)

    print("After L3 FFN at PC marker position:")
    output_lo = x[0, pc_marker_pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    output_hi = x[0, pc_marker_pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16]
    
    print(f"OUTPUT_LO: {output_lo}")
    print(f"OUTPUT_LO argmax: {output_lo.argmax().item()} (value: {output_lo.max().item():.3f})")
    print(f"OUTPUT_HI argmax: {output_hi.argmax().item()} (value: {output_hi.max().item():.3f})")
    
    output_byte = output_lo.argmax().item() | (output_hi.argmax().item() << 4)
    print(f"OUTPUT as byte: {output_byte} (expected 8 from PC_OFFSET=0 fix)")
    
    # Check HAS_SE
    has_se = x[0, pc_marker_pos, BD.HAS_SE]
    mark_pc = x[0, pc_marker_pos, BD.MARK_PC]
    print(f"\nHAS_SE: {has_se.item():.3f} (should be ~0 for first step)")
    print(f"MARK_PC: {mark_pc.item():.3f} (should be ~1)")

