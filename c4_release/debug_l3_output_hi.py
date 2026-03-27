"""Check if L3 FFN sets OUTPUT_HI correctly for carry."""
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

    print("=== INPUT to L3 FFN ===")
    output_hi = x[0, pc_marker_pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16]
    embed_lo = x[0, pc_marker_pos, BD.EMBED_LO:BD.EMBED_LO+16]
    embed_hi = x[0, pc_marker_pos, BD.EMBED_HI:BD.EMBED_HI+16]
    print(f"EMBED: lo={embed_lo.argmax().item()}, hi={embed_hi.argmax().item()} → PC=8")
    print(f"OUTPUT_HI: max={output_hi.max().item():.6f} at index {output_hi.argmax().item()}")

    # Run FFN
    l3_ffn = model.blocks[3].ffn
    ffn_delta = l3_ffn(x)

    print("\n=== FFN DELTA ===")
    output_lo_delta = ffn_delta[0, pc_marker_pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    output_hi_delta = ffn_delta[0, pc_marker_pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16]
    print(f"OUTPUT_LO delta: max={output_lo_delta.max().item():.6f} at index {output_lo_delta.argmax().item()}")
    print(f"OUTPUT_HI delta: max={output_hi_delta.max().item():.6f} at index {output_hi_delta.argmax().item()}")
    print(f"  OUTPUT_HI[0] delta: {output_hi_delta[0].item():.6f}")
    print(f"  OUTPUT_HI[1] delta: {output_hi_delta[1].item():.6f}")

    # Final output
    x_final = x + ffn_delta

    print("\n=== FINAL OUTPUT ===")
    output_lo_final = x_final[0, pc_marker_pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    output_hi_final = x_final[0, pc_marker_pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16]
    pc_lo = output_lo_final.argmax().item()
    pc_hi = output_hi_final.argmax().item()
    pc_byte = pc_lo | (pc_hi << 4)
    print(f"OUTPUT: lo={pc_lo}, hi={pc_hi} → PC={pc_byte}")
    print(f"Expected: PC=16")
