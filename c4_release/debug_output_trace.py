"""Trace OUTPUT_LO values through layers."""
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

bytecode = [Opcode.IMM | (42 << 8)]
context = build_context(bytecode)

# Get draft tokens
draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

# Build context up to PC_b0
current_context = context + [draft_tokens[0], draft_tokens[1]]  # REG_PC + PC_b0

ctx_tensor = torch.tensor([current_context], dtype=torch.long)
with torch.no_grad():
    x = model.embed(ctx_tensor)
    model._add_code_addr_keys(ctx_tensor, x)
    model._inject_mem_store(ctx_tensor, x)

    pc_b0_pos = len(current_context) - 1

    print(f"Token at position {pc_b0_pos}: {current_context[pc_b0_pos]} (PC_b0=10)")
    print(f"\nTracing OUTPUT_LO channels:")

    for layer_idx in range(16):
        x = model.blocks[layer_idx](x)

        # Check OUTPUT_LO[0] and OUTPUT_LO[10]
        out_lo_0 = x[0, pc_b0_pos, BD.OUTPUT_LO + 0].item()
        out_lo_10 = x[0, pc_b0_pos, BD.OUTPUT_LO + 10].item()

        if abs(out_lo_0) > 0.01 or abs(out_lo_10) > 0.01:
            print(f"\nLayer {layer_idx:2d}:")
            print(f"  OUTPUT_LO[0]  = {out_lo_0:7.3f}")
            print(f"  OUTPUT_LO[10] = {out_lo_10:7.3f}")

            # Show all OUTPUT_LO values if any are non-zero
            all_vals = [x[0, pc_b0_pos, BD.OUTPUT_LO + k].item() for k in range(16)]
            if any(abs(v) > 0.01 for v in all_vals):
                print(f"  All OUTPUT_LO: ", end="")
                for k, v in enumerate(all_vals):
                    if abs(v) > 0.01:
                        print(f"[{k}]={v:.2f} ", end="")
                print()
