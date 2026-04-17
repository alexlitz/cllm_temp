"""Full trace of IMM 42 execution with CORRECT layer indices."""
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

print("Context tokens:")
print(f"  CODE_START, IMM={Opcode.IMM}, 42, 0, 0, 0, CODE_END, DATA_START, DATA_END")
print(f"  Immediate byte at position 2: {context[2]} (should be 42)")

draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

ctx_tensor = torch.tensor([context + draft_tokens], dtype=torch.long)
ctx_len = len(context)
pc_marker_pos = ctx_len      # Token 0 = REG_PC
ax_marker_pos = ctx_len + 5  # Token 5 = REG_AX

print(f"\nMarker positions:")
print(f"  PC marker: {pc_marker_pos}")
print(f"  AX marker: {ax_marker_pos}")

with torch.no_grad():
    x = model.embed(ctx_tensor)
    model._inject_code_addr_keys(ctx_tensor, x)
    model._inject_mem_store(ctx_tensor, x)

    # === LAYER 3: PC first-step default ===
    for i in range(4):  # Layers 0-3
        x = model.blocks[i](x)

    print("\n=== After Layer 3 (blocks[3]) ===")
    output_lo = x[0, pc_marker_pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    output_hi = x[0, pc_marker_pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16]
    pc_lo = output_lo.argmax().item()
    pc_hi = output_hi.argmax().item()
    print(f"PC marker OUTPUT: lo={pc_lo}, hi={pc_hi} → PC={pc_lo | (pc_hi << 4)} (expected 8)")

    # === LAYER 4: PC relay + PC+1 ===
    x = model.blocks[4](x)

    print("\n=== After Layer 4 (blocks[4]) ===")
    embed_lo = x[0, ax_marker_pos, BD.EMBED_LO:BD.EMBED_LO+16]
    embed_hi = x[0, ax_marker_pos, BD.EMBED_HI:BD.EMBED_HI+16]
    pc_lo = embed_lo.argmax().item()
    pc_hi = embed_hi.argmax().item()
    print(f"AX marker EMBED (relayed PC): lo={pc_lo}, hi={pc_hi} → PC={pc_lo | (pc_hi << 4)}")

    temp = x[0, ax_marker_pos, BD.TEMP:BD.TEMP+32]
    temp_lo = temp[:16].argmax().item()
    temp_hi = temp[16:32].argmax().item()
    print(f"AX marker TEMP (PC+1): lo={temp_lo}, hi={temp_hi} → PC+1={temp_lo | (temp_hi << 4)} (expected 1)")

    # === LAYER 5: Fetch ===
    x = model.blocks[5](x)

    print("\n=== After Layer 5 (blocks[5]) ===")
    fetch_lo = x[0, ax_marker_pos, BD.FETCH_LO:BD.FETCH_LO+16]
    fetch_hi = x[0, ax_marker_pos, BD.FETCH_HI:BD.FETCH_HI+16]
    imm_lo = fetch_lo.argmax().item()
    imm_hi = fetch_hi.argmax().item()
    fetched_imm = imm_lo | (imm_hi << 4)
    print(f"AX marker FETCH (immediate): lo={imm_lo}, hi={imm_hi} → IMM={fetched_imm} (expected 42)")

    # === LAYER 6: IMM routing ===
    x = model.blocks[6](x)

    print("\n=== After Layer 6 (blocks[6]) ===")
    output_lo_ax = x[0, ax_marker_pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    output_hi_ax = x[0, ax_marker_pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16]
    ax_lo = output_lo_ax.argmax().item()
    ax_hi = output_hi_ax.argmax().item()
    ax_val = ax_lo | (ax_hi << 4)
    print(f"AX marker OUTPUT: lo={ax_lo}, hi={ax_hi} → AX={ax_val} (expected 42)")

    # === Check final prediction at token position ===
    logits = model.forward(ctx_tensor)
    predicted_ax_byte0 = logits[0, ctx_len + 6, :].argmax().item()  # Token 6 = AX byte 0
    print(f"\n=== Final Prediction ===")
    print(f"Predicted AX byte 0: {predicted_ax_byte0} (expected 42)")
    print(f"DraftVM AX: {draft_vm.ax}")
