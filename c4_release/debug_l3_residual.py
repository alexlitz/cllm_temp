"""Check residual connection issue in L3 FFN."""
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
ctx_len_step2 = len(context_step1)
pc_marker_pos = ctx_len_step2

with torch.no_grad():
    x = model.embed(ctx_tensor)
    model._inject_code_addr_keys(ctx_tensor, x)
    model._inject_mem_store(ctx_tensor, x)

    for i in range(3):
        x = model.blocks[i](x)
    x = x + model.blocks[3].attn(x)

    print("=== INPUT to L3 FFN (after L3 attention) ===")
    output_lo = x[0, pc_marker_pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    print(f"OUTPUT_LO at PC marker:")
    print(f"  Max value: {output_lo.max().item():.6f} at index {output_lo.argmax().item()}")
    print(f"  OUTPUT_LO[0]: {output_lo[0].item():.6f}")
    print(f"  OUTPUT_LO[8]: {output_lo[8].item():.6f}")

    # Run FFN
    l3_ffn = model.blocks[3].ffn
    ffn_delta = l3_ffn(x)

    print("\n=== FFN DELTA (what FFN adds to residual) ===")
    output_lo_delta = ffn_delta[0, pc_marker_pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    print(f"OUTPUT_LO delta at PC marker:")
    print(f"  Max value: {output_lo_delta.max().item():.6f} at index {output_lo_delta.argmax().item()}")
    print(f"  OUTPUT_LO[0] delta: {output_lo_delta[0].item():.6f}")
    print(f"  OUTPUT_LO[8] delta: {output_lo_delta[8].item():.6f}")

    # Final output
    x_final = x + ffn_delta

    print("\n=== OUTPUT (input + ffn_delta) ===")
    output_lo_final = x_final[0, pc_marker_pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    print(f"OUTPUT_LO at PC marker:")
    print(f"  Max value: {output_lo_final.max().item():.6f} at index {output_lo_final.argmax().item()}")
    print(f"  OUTPUT_LO[0]: {output_lo_final[0].item():.6f}")
    print(f"  OUTPUT_LO[8]: {output_lo_final[8].item():.6f}")
