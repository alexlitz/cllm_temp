"""Check what L15 outputs before head projection."""
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

bytecode = [Opcode.IMM | (42 << 8), Opcode.EXIT]
context = build_context(bytecode)

draft_vm = DraftVM(bytecode)
draft_vm.step()
draft1 = draft_vm.draft_tokens()
context_step1 = context + draft1
draft_vm.step()
draft2 = draft_vm.draft_tokens()
context_step2 = context_step1 + draft2

ctx_tensor = torch.tensor([context_step2], dtype=torch.long)
ctx_len = len(context_step1)
pc_marker_pos = ctx_len
pc_byte0_pos = ctx_len + 1

with torch.no_grad():
    x = model.embed(ctx_tensor)
    model._inject_code_addr_keys(ctx_tensor, x)
    model._inject_mem_store(ctx_tensor, x)

    for block in model.blocks:
        x = block(x)

    print("=== AFTER L15 (before head projection) ===")
    print(f"\nAt PC marker position {pc_marker_pos}:")
    output_lo = x[0, pc_marker_pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    output_hi = x[0, pc_marker_pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16]
    print(f"OUTPUT_LO: max={output_lo.max().item():.3f} at index {output_lo.argmax().item()}")
    print(f"OUTPUT_HI: max={output_hi.max().item():.3f} at index {output_hi.argmax().item()}")

    print(f"\nAt PC byte 0 position {pc_byte0_pos}:")
    output_lo_byte = x[0, pc_byte0_pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    output_hi_byte = x[0, pc_byte0_pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16]
    print(f"OUTPUT_LO: max={output_lo_byte.max().item():.3f} at index {output_lo_byte.argmax().item()}")
    print(f"OUTPUT_HI: max={output_hi_byte.max().item():.3f} at index {output_hi_byte.argmax().item()}")

    # Manually compute logits for PC byte 0
    head = model.head
    x_pc_byte0 = x[0, pc_byte0_pos, :]
    logits_manual = head.weight @ x_pc_byte0 + head.bias

    print(f"\n=== Manual logit computation for PC byte 0 ===")
    print(f"Top 5 logits:")
    top5 = torch.topk(logits_manual, 5)
    for val, idx in zip(top5.values, top5.indices):
        print(f"  Token {idx.item():3d}: {val.item():.3f}")

    print(f"\nExpected: Token 16")
    print(f"logit[16] = {logits_manual[16].item():.3f}")
    print(f"logit[0] = {logits_manual[0].item():.3f}")
