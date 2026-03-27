"""Debug L15 OUTPUT relay to see why PC bytes are affected."""
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

bytecode = [Opcode.LEA | (8 << 8)]
context = build_context(bytecode)

draft_vm = DraftVM(bytecode)
draft_vm.step()
draft1 = draft_vm.draft_tokens()
context_step1 = context + draft1

ctx_tensor = torch.tensor([context_step1], dtype=torch.long)
ctx_len = len(context)

with torch.no_grad():
    x = model.embed(ctx_tensor)
    model._inject_code_addr_keys(ctx_tensor, x)
    model._inject_mem_store(ctx_tensor, x)

    # Run through L14 (before L15)
    for i in range(15):
        x = model.blocks[i](x)

    print("=== BEFORE L15 (after L14) ===")
    # Check OUTPUT at different positions
    for i, name in [(ctx_len, "PC marker"), (ctx_len + 1, "PC byte 0"),
                     (ctx_len + 5, "AX marker"), (ctx_len + 6, "AX byte 0")]:
        output_lo = x[0, i, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
        output_hi = x[0, i, BD.OUTPUT_HI:BD.OUTPUT_HI+16]
        print(f"\n{name} (pos {i}):")
        print(f"  OUTPUT_LO: max={output_lo.max().item():.3f} at index {output_lo.argmax().item()}")
        print(f"  OUTPUT_HI: max={output_hi.max().item():.3f} at index {output_hi.argmax().item()}")

        # Check marker flags
        mark_pc = x[0, i, BD.MARK_PC].item()
        mark_ax = x[0, i, BD.MARK_AX].item()
        byte_idx_0 = x[0, i, BD.BYTE_INDEX_0].item()
        op_li_relay = x[0, i, BD.OP_LI_RELAY].item() if hasattr(BD, 'OP_LI_RELAY') else 0.0
        print(f"  MARK_PC={mark_pc:.2f}, MARK_AX={mark_ax:.2f}, BYTE_INDEX_0={byte_idx_0:.2f}, OP_LI_RELAY={op_li_relay:.2f}")

    # Run L15 attention only
    x_after_attn = x + model.blocks[15].attn(x)

    print("\n=== AFTER L15 ATTENTION ===")
    for i, name in [(ctx_len, "PC marker"), (ctx_len + 1, "PC byte 0"),
                     (ctx_len + 5, "AX marker"), (ctx_len + 6, "AX byte 0")]:
        output_lo = x_after_attn[0, i, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
        output_hi = x_after_attn[0, i, BD.OUTPUT_HI:BD.OUTPUT_HI+16]
        print(f"\n{name} (pos {i}):")
        print(f"  OUTPUT_LO: max={output_lo.max().item():.3f} at index {output_lo.argmax().item()}")
        print(f"  OUTPUT_HI: max={output_hi.max().item():.3f} at index {output_hi.argmax().item()}")
