#!/usr/bin/env python3
"""Trace JSR fetch mechanism at each layer to understand why FETCH/OUTPUT are wrong."""

import torch
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token, _SetDim as BD
from neural_vm.embedding import Opcode
from neural_vm.constants import INSTR_WIDTH, PC_OFFSET
from src.compiler import compile_c

# Simple program - JSR to main
code = 'int main() { return 42; }'
bytecode, data = compile_c(code)

# Extract JSR target
jsr_instr = bytecode[0]
jsr_opcode = jsr_instr & 0xFF
jsr_target = jsr_instr >> 8  # 24-bit immediate
target_byte0 = jsr_target & 0xFF
target_byte1 = (jsr_target >> 8) & 0xFF
target_byte2 = (jsr_target >> 16) & 0xFF

print(f"JSR instruction: 0x{jsr_instr:08x}")
print(f"  Opcode: {jsr_opcode} (JSR)")
print(f"  Target: {jsr_target} = 0x{jsr_target:06x}")
print(f"  Byte 0: 0x{target_byte0:02x} (lo={target_byte0 & 0xF}, hi={target_byte0 >> 4})")
print(f"  Byte 1: 0x{target_byte1:02x}")
print(f"  Byte 2: 0x{target_byte2:02x}")

# Show the immediate byte that L5 should fetch
# L5 head 3 fetches at address PC_OFFSET + 1 = 3 (immediate byte 0)
# But for non-first steps it fetches at PC + 1
print(f"\nBytecode bytes at first instruction:")
for i in range(8):
    addr = i
    byte_val = bytecode[0] >> (i * 8) & 0xFF if i < 4 else 0
    print(f"  Addr {addr}: 0x{byte_val:02x}")

# Create runner
runner = AutoregressiveVMRunner()
runner._func_call_handlers = {}  # Remove handlers for pure neural

# Build context
runner._bytecode = bytecode
runner._last_sp = 0x1F800
runner._last_bp = 0x10000
ctx = runner._build_context(bytecode, data, [])

# Set active opcode
runner.model.set_active_opcode(Opcode.JSR)

print(f"\nContext length: {len(ctx)}")
print(f"Last 10 tokens: {ctx[-10:]}")

# Now trace through each layer at the position where REG_PC marker will be generated
# Actually, we need to generate up to REG_PC position first
print(f"\n=== Generating up to REG_PC marker ===")
with torch.no_grad():
    device = next(runner.model.parameters()).device
    token_ids = torch.tensor([ctx[-512:] if len(ctx) > 512 else ctx], dtype=torch.long, device=device)

    # Forward pass through each layer
    x = runner.model.embed(token_ids)
    last_pos_idx = -1  # last position

    print(f"\nEmbedding position info (last position):")
    print(f"  IS_BYTE: {x[0, last_pos_idx, BD.IS_BYTE].item():.2f}")
    print(f"  CONST: {x[0, last_pos_idx, BD.CONST].item():.2f}")

    # After embedding, check if STEP_END marker is present
    print(f"  MARK_SE: {x[0, last_pos_idx, BD.MARK_SE].item():.2f}")

    # Check ADDR_KEY values to understand what addresses are being encoded
    addr_key_vals = [x[0, last_pos_idx, BD.ADDR_KEY + k].item() for k in range(32)]
    print(f"  ADDR_KEY[0:16]: {addr_key_vals[:16]}")
    print(f"  ADDR_KEY[16:32]: {addr_key_vals[16:32]}")

    # Check what immediate byte 0 (at address 3) looks like in the context
    # Look for position with ADDR_KEY encoding 3
    print(f"\n=== Finding bytecode positions ===")
    for pos in range(min(20, token_ids.shape[1])):
        addr_key = [x[0, pos, BD.ADDR_KEY + k].item() for k in range(48)]
        # Decode address from ADDR_KEY (one-hot encoding per nibble)
        lo_nibble = -1
        mid_nibble = -1
        hi_nibble = -1
        for k in range(16):
            if addr_key[k] > 0.5:
                lo_nibble = k
            if addr_key[16 + k] > 0.5:
                mid_nibble = k
            if addr_key[32 + k] > 0.5:
                hi_nibble = k
        if lo_nibble >= 0:
            addr = lo_nibble + (mid_nibble << 4) + (hi_nibble << 8)
            clean_lo = [x[0, pos, BD.CLEAN_EMBED_LO + k].item() for k in range(16)]
            clean_hi = [x[0, pos, BD.CLEAN_EMBED_HI + k].item() for k in range(16)]
            clean_lo_idx = max(range(16), key=lambda k: clean_lo[k])
            clean_hi_idx = max(range(16), key=lambda k: clean_hi[k])
            byte_val = clean_lo_idx + (clean_hi_idx << 4)
            print(f"  Pos {pos}: addr={addr}, byte=0x{byte_val:02x} (lo_idx={clean_lo_idx}, hi_idx={clean_hi_idx})")

    print(f"\n=== Layer-by-layer trace at last position ===")
    x = runner.model.embed(token_ids)
    for layer_idx, block in enumerate(runner.model.blocks):
        x = block(x)

        # Check key dimensions at last position after this layer
        last = x[0, -1]

        if layer_idx in [4, 5, 6, 7]:  # L5 is fetch, L6 is routing
            fetch_lo = [last[BD.FETCH_LO + k].item() for k in range(8)]
            fetch_hi = [last[BD.FETCH_HI + k].item() for k in range(8)]
            output_lo = [last[BD.OUTPUT_LO + k].item() for k in range(8)]
            output_hi = [last[BD.OUTPUT_HI + k].item() for k in range(8)]
            temp_0 = last[BD.TEMP + 0].item()  # IS_JSR flag
            op_jsr = last[BD.OP_JSR].item()
            mark_pc = last[BD.MARK_PC].item()
            mark_ax = last[BD.MARK_AX].item()
            next_pc = last[BD.NEXT_PC].item()
            has_se = last[BD.HAS_SE].item()

            print(f"\nAfter L{layer_idx}:")
            print(f"  FETCH_LO[0:8]: {[f'{v:.2f}' for v in fetch_lo]}")
            print(f"  FETCH_HI[0:8]: {[f'{v:.2f}' for v in fetch_hi]}")
            print(f"  OUTPUT_LO[0:8]: {[f'{v:.2f}' for v in output_lo]}")
            print(f"  OUTPUT_HI[0:8]: {[f'{v:.2f}' for v in output_hi]}")
            print(f"  TEMP[0] (IS_JSR): {temp_0:.2f}")
            print(f"  OP_JSR: {op_jsr:.2f}")
            print(f"  MARK_PC: {mark_pc:.2f}, MARK_AX: {mark_ax:.2f}")
            print(f"  NEXT_PC: {next_pc:.2f}, HAS_SE: {has_se:.2f}")

# Now generate the first token (should be REG_PC marker)
print(f"\n=== Generating first token ===")
next_token = runner.model.generate_next(ctx)
ctx.append(next_token)
print(f"Token 0: {next_token} (expected: {Token.REG_PC})")

# Now we're at position where PC byte 0 will be generated
# Trace the forward pass at this position
print(f"\n=== State at PC byte 0 generation position ===")
with torch.no_grad():
    token_ids = torch.tensor([ctx[-512:] if len(ctx) > 512 else ctx], dtype=torch.long, device=device)

    x = runner.model.embed(token_ids)
    for layer_idx, block in enumerate(runner.model.blocks):
        x = block(x)

        if layer_idx in [4, 5, 6]:
            last = x[0, -1]
            fetch_lo = [last[BD.FETCH_LO + k].item() for k in range(8)]
            output_lo = [last[BD.OUTPUT_LO + k].item() for k in range(8)]
            temp_0 = last[BD.TEMP + 0].item()
            op_jsr = last[BD.OP_JSR].item()
            is_byte = last[BD.IS_BYTE].item()
            mark_pc = last[BD.MARK_PC].item()
            byte_idx0 = last[BD.BYTE_INDEX_0].item()

            print(f"\nAfter L{layer_idx}:")
            print(f"  FETCH_LO[0:8]: {[f'{v:.2f}' for v in fetch_lo]}")
            print(f"  OUTPUT_LO[0:8]: {[f'{v:.2f}' for v in output_lo]}")
            print(f"  TEMP[0]: {temp_0:.2f}, OP_JSR: {op_jsr:.2f}")
            print(f"  IS_BYTE: {is_byte:.2f}, MARK_PC: {mark_pc:.2f}")
            print(f"  BYTE_INDEX_0: {byte_idx0:.2f}")

    # Final output projection
    logits = runner.model.head(x[0, -1:])
    top5_vals, top5_ids = torch.topk(logits[0, 0], 5)
    print(f"\nTop 5 predictions: {[(idx.item(), f'{val.item():.2f}') for idx, val in zip(top5_ids, top5_vals)]}")
    print(f"Expected byte: 0x{target_byte0:02x} = {target_byte0}")

# Generate second token
next_token = runner.model.generate_next(ctx)
print(f"Token 1 (PC byte 0): {next_token} (expected: {target_byte0})")
