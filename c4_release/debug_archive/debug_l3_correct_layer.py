"""Debug L3 FFN - using the CORRECT layer (blocks[3])!"""
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

# Check that Layer 3 is blocks[3]
l3_ffn = model.blocks[3].ffn  # ← CORRECT!

print("Layer 3 FFN weights (blocks[3]):")
print(f"Unit 0 - W_up[MARK_PC={BD.MARK_PC}]: {l3_ffn.W_up[0, BD.MARK_PC].item():.1f}")
print(f"Unit 0 - b_up: {l3_ffn.b_up[0].item():.1f}")
print(f"Unit 0 - b_gate: {l3_ffn.b_gate[0].item():.3f}")

with torch.no_grad():
    x = model.embed(ctx_tensor)
    model._inject_code_addr_keys(ctx_tensor, x)
    model._inject_mem_store(ctx_tensor, x)

    # Run through layers 0-2
    for i in range(3):
        x = model.blocks[i](x)

    print("\nBefore Layer 3 (after blocks[0-2]):")
    mark_pc = x[0, pc_marker_pos, BD.MARK_PC]
    has_se = x[0, pc_marker_pos, BD.HAS_SE]
    print(f"MARK_PC: {mark_pc.item():.3f}")
    print(f"HAS_SE: {has_se.item():.3f}")

    # Run Layer 3 (blocks[3])
    x = model.blocks[3](x)

    print("\nAfter Layer 3 (blocks[3]):")
    output_lo = x[0, pc_marker_pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    output_hi = x[0, pc_marker_pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16]
    print(f"OUTPUT_LO argmax: {output_lo.argmax().item()} (expected 8)")
    print(f"OUTPUT_LO[8]: {output_lo[8].item():.6f}")
    print(f"OUTPUT_HI argmax: {output_hi.argmax().item()} (expected 0)")
