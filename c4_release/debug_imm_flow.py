"""Comprehensive trace of IMM immediate fetch data flow."""
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

# Test: IMM 5
bytecode = [Opcode.IMM | (5 << 8)]
draft_vm = DraftVM(bytecode)
context = build_context(bytecode)

print("="*70)
print("IMM 5 Data Flow Trace")
print("="*70)
print(f"Bytecode: IMM 5 (opcode=1, immediate=5)")
print(f"Context length: {len(context)}")
print()

# Execute DraftVM
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()
draft_ax = draft_tokens[6] | (draft_tokens[7] << 8) | (draft_tokens[8] << 16) | (draft_tokens[9] << 24)
print(f"Expected result (DraftVM): AX = {draft_ax}")
print()

# Neural VM
model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

ctx_tensor = torch.tensor([context + draft_tokens], dtype=torch.long)
ctx_len = len(context)
# Output structure: REG_PC(1) + PC_bytes(4) + REG_AX(1) + AX_bytes(4) + ...
ax_marker_pos = ctx_len + 5  # REG_AX marker position (PC marker + 4 PC bytes + AX marker)
ax_byte0_pos = ctx_len + 6  # AX byte 0 position in output

print("Layer-by-layer trace:")
print("-" * 70)

with torch.no_grad():
    x = model.embed(ctx_tensor)
    model._inject_code_addr_keys(ctx_tensor, x)

    # Through L0-L4
    for i in range(5):
        x = model.blocks[i](x)

    print(f"After L4 (PC relay):")
    print(f"  Position {ax_marker_pos} (REG_AX marker):")
    print(f"    MARK_AX = {x[0, ax_marker_pos, BD.MARK_AX]:.2f} (expect ~1.0)")

    # Check EMBED_LO one-hot encoding
    embed_lo_vals = x[0, ax_marker_pos, BD.EMBED_LO:BD.EMBED_LO+16]
    pc_lo_nibble = embed_lo_vals.argmax().item()
    print(f"    EMBED_LO nibble: {pc_lo_nibble} (expect 0 for PC=0)")
    print(f"      EMBED_LO[0] = {embed_lo_vals[0]:.2f} (expect ~1.0)")

    # Check TEMP one-hot encoding
    temp_vals = x[0, ax_marker_pos, BD.TEMP:BD.TEMP+16]
    temp_lo_nibble = temp_vals.argmax().item()
    print(f"    TEMP nibble: {temp_lo_nibble} (expect 1 for PC+1)")
    print(f"      TEMP[1] = {temp_vals[1]:.2f} (expect ~1.0)")
    print()

    # L5 attention
    x_before_l5 = x.clone()
    x = model.blocks[5].attn(x)

    print(f"After L5 attention:")
    print(f"  Position {ax_marker_pos} (REG_AX marker):")

    # Check OPCODE_BYTE (from head 1 opcode fetch)
    opcode_lo_vals = x[0, ax_marker_pos, BD.OPCODE_BYTE_LO:BD.OPCODE_BYTE_LO+16]
    opcode_hi_vals = x[0, ax_marker_pos, BD.OPCODE_BYTE_HI:BD.OPCODE_BYTE_HI+16]
    op_lo = opcode_lo_vals.argmax().item()
    op_hi = opcode_hi_vals.argmax().item()
    opcode_val = op_lo | (op_hi << 4)
    print(f"    OPCODE_BYTE decoded: {opcode_val} (expect 1 for IMM)")
    print(f"      LO={op_lo}, HI={op_hi}")

    # Check FETCH (from head 0 immediate fetch)
    fetch_lo_vals = x[0, ax_marker_pos, BD.FETCH_LO:BD.FETCH_LO+16]
    fetch_hi_vals = x[0, ax_marker_pos, BD.FETCH_HI:BD.FETCH_HI+16]
    fetch_lo = fetch_lo_vals.argmax().item()
    fetch_hi = fetch_hi_vals.argmax().item()
    fetch_val = fetch_lo | (fetch_hi << 4)
    print(f"    FETCH decoded: {fetch_val} (expect 5 for immediate)")
    print(f"      LO={fetch_lo}, HI={fetch_hi}")
    print()

    # L5 FFN (opcode decode)
    x = model.blocks[5].ffn(x)

    print(f"After L5 FFN (opcode decode):")
    print(f"  Position {ax_marker_pos} (REG_AX marker):")
    print(f"    OP_IMM = {x[0, ax_marker_pos, BD.OP_IMM]:.2f} (expect ~5.0)")
    print()

    # L6 attention (JMP relay, etc)
    x = model.blocks[6].attn(x)

    # L6 FFN (output routing)
    x_before_l6_ffn = x.clone()
    x = model.blocks[6].ffn(x)

    print(f"After L6 FFN (output routing):")
    print(f"  Position {ax_marker_pos} (REG_AX marker):")
    output_lo_vals = x[0, ax_marker_pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    output_hi_vals = x[0, ax_marker_pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16]
    out_lo = output_lo_vals.argmax().item()
    out_hi = output_hi_vals.argmax().item()
    output_val = out_lo | (out_hi << 4)
    print(f"    OUTPUT decoded: {output_val} (expect 5)")
    print(f"      LO={out_lo}, HI={out_hi}")

    print(f"  Position {ax_byte0_pos} (AX byte 0):")
    output_lo_vals_b0 = x[0, ax_byte0_pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    output_hi_vals_b0 = x[0, ax_byte0_pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16]
    out_lo_b0 = output_lo_vals_b0.argmax().item()
    out_hi_b0 = output_hi_vals_b0.argmax().item()
    output_val_b0 = out_lo_b0 | (out_hi_b0 << 4)
    print(f"    OUTPUT decoded: {output_val_b0} (expect 5)")
    print(f"      LO={out_lo_b0}, HI={out_hi_b0}")
    print()

    # Check activation conditions for L6 FFN IMM routing
    print("L6 FFN IMM routing activation check:")
    op_imm_val = x_before_l6_ffn[0, ax_marker_pos, BD.OP_IMM].item()
    mark_ax_val = x_before_l6_ffn[0, ax_marker_pos, BD.MARK_AX].item()
    activation_sum = op_imm_val + mark_ax_val
    threshold = 4.0
    print(f"  OP_IMM = {op_imm_val:.2f}")
    print(f"  MARK_AX = {mark_ax_val:.2f}")
    print(f"  Sum = {activation_sum:.2f} (threshold = {threshold})")
    print(f"  Active? {activation_sum > threshold}")

    if activation_sum > threshold:
        print("  ✓ IMM routing should be active")
    else:
        print("  ✗ IMM routing NOT active - this is the problem!")
    print()

    # Continue through L7-L10 to see if byte 0 gets OUTPUT
    for i in range(7, 11):
        x = model.blocks[i](x)

    print(f"After L10 (byte passthrough + AX passthrough):")
    print(f"  Position {ax_byte0_pos} (AX byte 0):")
    output_lo_vals_b0_l10 = x[0, ax_byte0_pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    output_hi_vals_b0_l10 = x[0, ax_byte0_pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16]
    out_lo_b0_l10 = output_lo_vals_b0_l10.argmax().item()
    out_hi_b0_l10 = output_hi_vals_b0_l10.argmax().item()
    output_val_b0_l10 = out_lo_b0_l10 | (out_hi_b0_l10 << 4)
    print(f"    OUTPUT decoded: {output_val_b0_l10} (expect 5)")
    print(f"      LO={out_lo_b0_l10}, HI={out_hi_b0_l10}")
