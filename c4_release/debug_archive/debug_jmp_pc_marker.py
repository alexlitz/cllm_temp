"""Debug JMP at PC marker position."""
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

bytecode = [Opcode.JMP | (16 << 8)]
context = build_context(bytecode)

# Get draft tokens
draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

# Build context up to REG_PC marker
current_context = context + [draft_tokens[0]]  # Just REG_PC marker

ctx_tensor = torch.tensor([current_context], dtype=torch.long)
with torch.no_grad():
    x = model.embed(ctx_tensor)
    model._add_code_addr_keys(ctx_tensor, x)
    model._inject_mem_store(ctx_tensor, x)

    pc_marker_pos = len(current_context) - 1

    print(f"PC marker at position {pc_marker_pos}")
    print(f"\nTracing through layers:")

    for layer_idx in range(16):
        x = model.blocks[layer_idx](x)

        if layer_idx in [5, 6, 12, 13, 15]:
            print(f"\n=== AFTER LAYER {layer_idx} at PC marker ===")
            print(f"Opcode flags:")
            print(f"  OP_JMP: {x[0, pc_marker_pos, BD.OP_JMP].item():.3f}")
            print(f"  OP_IMM: {x[0, pc_marker_pos, BD.OP_IMM].item():.3f}")

            # Check FETCH values (immediate)
            fetch_lo_vals = [x[0, pc_marker_pos, BD.FETCH_LO + k].item() for k in range(16)]
            fetch_hi_vals = [x[0, pc_marker_pos, BD.FETCH_HI + k].item() for k in range(16)]
            fetch_lo_max = max(range(16), key=lambda k: fetch_lo_vals[k])
            fetch_hi_max = max(range(16), key=lambda k: fetch_hi_vals[k])
            fetch_val = fetch_lo_max | (fetch_hi_max << 4)
            if fetch_val > 0:
                print(f"  FETCH value: {fetch_val}")

            # Check OUTPUT values
            lo_vals = [x[0, pc_marker_pos, BD.OUTPUT_LO + k].item() for k in range(16)]
            hi_vals = [x[0, pc_marker_pos, BD.OUTPUT_HI + k].item() for k in range(16)]
            lo_max = max(range(16), key=lambda k: lo_vals[k])
            hi_max = max(range(16), key=lambda k: hi_vals[k])
            output_val = lo_max | (hi_max << 4)
            print(f"  OUTPUT value: {output_val}")

    print(f"\n=== FINAL PREDICTION ===")
    logits = model.head(x)
    predicted = logits[0, -1, :].argmax().item()
    print(f"Predicted PC_b0: {predicted}")
    print(f"Expected: {draft_tokens[1]} (18 = 16 + PC_OFFSET)")
