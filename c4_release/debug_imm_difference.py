"""Debug why IMM 0 works but IMM 42 fails."""
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

def test_imm(value, name):
    print(f"\n{'='*70}")
    print(f"Testing IMM {value} ({name})")
    print('='*70)

    bytecode = [Opcode.IMM | (value << 8)]
    draft_vm = DraftVM(bytecode)
    context = build_context(bytecode)

    print(f"Bytecode: {[hex(x) for x in bytecode]}")
    print(f"Instruction bytes: op={bytecode[0] & 0xFF}, imm={bytecode[0] >> 8}")
    print(f"Context tokens: {context}")
    print(f"  Immediate bytes in context: {context[2:6]}")  # Bytes after opcode

    draft_vm.step()
    draft_tokens = draft_vm.draft_tokens()

    print(f"\nDraftVM result:")
    print(f"  AX = {draft_vm.ax}")
    print(f"  Draft token[6] (AX_b0) = {draft_tokens[6]}")

    # Get transformer predictions
    ctx_tensor = torch.tensor([context + draft_tokens], dtype=torch.long)
    ctx_len = len(context)

    with torch.no_grad():
        # Trace through layers
        x = model.embed(ctx_tensor)
        model._inject_code_addr_keys(ctx_tensor, x)
        model._inject_mem_store(ctx_tensor, x)

        # Check key layers for IMM handling
        print(f"\nTransformer layer analysis:")

        for layer_idx in [0, 3, 5, 6, 10, 15]:
            x = model.blocks[layer_idx](x)

            # Check AX marker position OUTPUT
            ax_marker_pos = ctx_len + 5  # Token 5 is REG_AX marker
            output_lo = x[0, ax_marker_pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
            output_hi = x[0, ax_marker_pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16]

            lo_val = output_lo.argmax().item()
            hi_val = output_hi.argmax().item()
            ax_byte0 = lo_val | (hi_val << 4)

            print(f"  After L{layer_idx:2d}: OUTPUT at AX marker = {ax_byte0:3d} (lo={lo_val}, hi={hi_val})")

        # Final prediction
        logits = model.head(x)
        predicted_ax_b0 = logits[0, ctx_len + 5, :].argmax().item()

        print(f"\nFinal prediction:")
        print(f"  AX byte 0 predicted = {predicted_ax_b0}")
        print(f"  AX byte 0 expected  = {draft_tokens[6]}")
        print(f"  Match: {'✓' if predicted_ax_b0 == draft_tokens[6] else '✗'}")

# Test both cases
test_imm(0, "WORKS")
test_imm(42, "FAILS")
