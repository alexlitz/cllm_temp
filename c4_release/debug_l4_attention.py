"""Check if L4 attention relays PC correctly."""
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
pc_marker_pos = ctx_len  # Token 0 is REG_PC marker
ax_marker_pos = ctx_len + 5  # Token 5 is REG_AX marker

print(f"PC marker at position {pc_marker_pos}")
print(f"AX marker at position {ax_marker_pos}")

with torch.no_grad():
    x = model.embed(ctx_tensor)
    model._inject_code_addr_keys(ctx_tensor, x)
    model._inject_mem_store(ctx_tensor, x)

    print("\nAfter embedding + injection:")
    print(f"At PC marker - EMBED_LO: {x[0, pc_marker_pos, BD.EMBED_LO:BD.EMBED_LO+16].max().item():.6f}")
    print(f"At AX marker - EMBED_LO: {x[0, ax_marker_pos, BD.EMBED_LO:BD.EMBED_LO+16].max().item():.6f}")

    # After L3 (sets first-step PC)
    for i in range(4):
        x = model.blocks[i](x)
        if i == 2:  # After L3
            print("\nAfter L3 (should set PC=8 at PC marker):")
            output_lo = x[0, pc_marker_pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
            print(f"At PC marker - OUTPUT_LO max at index {output_lo.argmax().item()}")
            print(f"At PC marker - EMBED_LO: {x[0, pc_marker_pos, BD.EMBED_LO:BD.EMBED_LO+16].max().item():.6f}")

        if i == 3:  # After L4
            print("\nAfter L4 attention (should relay PC to AX marker):")
            print(f"At PC marker - EMBED_LO max: {x[0, pc_marker_pos, BD.EMBED_LO:BD.EMBED_LO+16].max().item():.6f}")
            print(f"At AX marker - EMBED_LO max: {x[0, ax_marker_pos, BD.EMBED_LO:BD.EMBED_LO+16].max().item():.6f}")
            
            embed_lo = x[0, ax_marker_pos, BD.EMBED_LO:BD.EMBED_LO+16]
            if embed_lo.max().item() < 0.01:
                print("⚠️  L4 attention did NOT relay PC to AX marker!")
            else:
                print(f"✓ L4 relayed PC value: {embed_lo.argmax().item()}")

