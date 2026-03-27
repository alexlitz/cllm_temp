"""Debug IMM FETCH mechanism."""
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

def test_imm_fetch(value):
    print(f"\n{'='*70}")
    print(f"IMM {value} - FETCH debugging")
    print('='*70)

    bytecode = [Opcode.IMM | (value << 8)]
    context = build_context(bytecode)

    draft_vm = DraftVM(bytecode)
    draft_vm.step()
    draft_tokens = draft_vm.draft_tokens()

    ctx_tensor = torch.tensor([context + draft_tokens], dtype=torch.long)
    ctx_len = len(context)
    ax_marker_pos = ctx_len + 5  # REG_AX marker

    with torch.no_grad():
        x = model.embed(ctx_tensor)
        model._inject_code_addr_keys(ctx_tensor, x)
        model._inject_mem_store(ctx_tensor, x)

        # After L5 (fetch)
        for i in range(6):
            x = model.blocks[i](x)
            if i == 4:  # After L5
                print(f"\nAfter L5 (fetch layer):")
                fetch_lo = x[0, ax_marker_pos, BD.FETCH_LO:BD.FETCH_LO+16]
                fetch_hi = x[0, ax_marker_pos, BD.FETCH_HI:BD.FETCH_HI+16]

                fetch_lo_val = fetch_lo.argmax().item()
                fetch_hi_val = fetch_hi.argmax().item()
                fetched_value = fetch_lo_val | (fetch_hi_val << 4)

                print(f"  FETCH_LO: {fetch_lo_val}")
                print(f"  FETCH_HI: {fetch_hi_val}")
                print(f"  Fetched byte: {fetched_value} (expected {value})")

            if i == 5:  # After L6
                print(f"\nAfter L6 (should copy FETCH → OUTPUT):")
                output_lo = x[0, ax_marker_pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
                output_hi = x[0, ax_marker_pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16]

                output_lo_val = output_lo.argmax().item()
                output_hi_val = output_hi.argmax().item()
                output_value = output_lo_val | (output_hi_val << 4)

                print(f"  OUTPUT_LO: {output_lo_val}")
                print(f"  OUTPUT_HI: {output_hi_val}")
                print(f"  Output byte: {output_value} (expected {value})")

test_imm_fetch(0)
test_imm_fetch(42)
