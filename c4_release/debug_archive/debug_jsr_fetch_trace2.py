#!/usr/bin/env python3
"""Trace JSR fetch mechanism - focus on bytecode positions and ADDR_KEY."""

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
jsr_target = jsr_instr >> 8  # 24-bit immediate
target_byte0 = jsr_target & 0xFF
print(f"JSR target: {jsr_target} = 0x{jsr_target:06x}")
print(f"  Byte 0: 0x{target_byte0:02x} (lo={target_byte0 & 0xF}, hi={target_byte0 >> 4})")

# Show expected addresses
print(f"\nExpected addresses (PC_OFFSET={PC_OFFSET}, INSTR_WIDTH={INSTR_WIDTH}):")
print(f"  Instr 0 opcode (JSR): addr = 0 * {INSTR_WIDTH} + {PC_OFFSET} + 0 = {PC_OFFSET}")
print(f"  Instr 0 imm[0] (0x{target_byte0:02x}): addr = 0 * {INSTR_WIDTH} + {PC_OFFSET} + 1 = {PC_OFFSET + 1}")
print(f"  L5 head 3 fetches at address PC_OFFSET + 1 = {PC_OFFSET + 1} (imm byte 0)")

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

print(f"\nContext tokens near beginning (should have CODE):")
print(f"  Token 0: {ctx[0]} (CODE_START={Token.CODE_START})")
print(f"  Token 1: {ctx[1]} (opcode, expected JSR=3)")
print(f"  Token 2: {ctx[2]} (imm byte 0, expected 0x{target_byte0:02x}={target_byte0})")
print(f"  Token 3: {ctx[3]} (imm byte 1)")

# Trace the embedding to check ADDR_KEY
print(f"\n=== Checking ADDR_KEY in bytecode positions ===")
with torch.no_grad():
    device = next(runner.model.parameters()).device
    token_ids = torch.tensor([ctx[-2048:] if len(ctx) > 2048 else ctx], dtype=torch.long, device=device)

    # Get embeddings (this includes ADDR_KEY augmentation)
    x = runner.model.embed(token_ids)

    # Check ADDR_KEY at first few positions
    for pos in range(min(12, token_ids.shape[1])):
        tok = token_ids[0, pos].item()
        addr_key = [x[0, pos, BD.ADDR_KEY + k].item() for k in range(48)]

        # Decode address from ADDR_KEY
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

        addr = lo_nibble + (mid_nibble * 16) + (hi_nibble * 256) if lo_nibble >= 0 else -1

        # Also check CLEAN_EMBED values
        clean_lo = [x[0, pos, BD.CLEAN_EMBED_LO + k].item() for k in range(16)]
        clean_hi = [x[0, pos, BD.CLEAN_EMBED_HI + k].item() for k in range(16)]

        print(f"  Pos {pos:2d}: tok={tok:3d}", end="")
        if lo_nibble >= 0:
            print(f", addr={addr:3d} (lo={lo_nibble}, mid={mid_nibble}, hi={hi_nibble})", end="")
            # Check CLEAN_EMBED values
            max_lo = max(clean_lo)
            max_hi = max(clean_hi)
            lo_idx = clean_lo.index(max_lo)
            hi_idx = clean_hi.index(max_hi)
            byte_val = lo_idx + (hi_idx << 4)
            print(f", byte=0x{byte_val:02x} (CLEAN_EMBED)")
        else:
            print(" (no ADDR_KEY)")

# Now trace through layers at the position for PC byte 0 generation
# First, generate up to that position
print(f"\n=== Generating up to PC byte 0 position ===")

# Generate tokens until we have REG_PC marker
next_token = runner.model.generate_next(ctx)
ctx.append(next_token)
print(f"Token 0: {next_token} (expected REG_PC={Token.REG_PC})")

# Now we're at position to generate PC byte 0
with torch.no_grad():
    token_ids = torch.tensor([ctx[-2048:] if len(ctx) > 2048 else ctx], dtype=torch.long, device=device)

    # Trace through layers
    x = runner.model.embed(token_ids)

    print(f"\n=== Layer-by-layer at PC byte 0 position ===")
    for layer_idx, block in enumerate(runner.model.blocks):
        x = block(x)

        if layer_idx == 4:  # L5 is the fetch layer
            last = x[0, -1]
            print(f"\nAfter L5 (fetch layer):")
            print(f"  HAS_SE: {last[BD.HAS_SE].item():.2f}")
            print(f"  MARK_PC: {last[BD.MARK_PC].item():.2f}")
            print(f"  MARK_AX: {last[BD.MARK_AX].item():.2f}")

            # Check if L5 head 3 is working
            # It should query for address PC_OFFSET+1=3 and find the immediate byte
            # The FETCH_LO/HI should have the byte value
            fetch_lo = [last[BD.FETCH_LO + k].item() for k in range(16)]
            fetch_hi = [last[BD.FETCH_HI + k].item() for k in range(16)]
            print(f"  FETCH_LO: {[f'{v:.1f}' for v in fetch_lo]}")
            print(f"  FETCH_HI: {[f'{v:.1f}' for v in fetch_hi]}")

            # Try to decode FETCH as one-hot
            max_fetch_lo = max(fetch_lo)
            max_fetch_hi = max(fetch_hi)
            if max_fetch_lo > 0.5:
                lo_idx = fetch_lo.index(max_fetch_lo)
                hi_idx = fetch_hi.index(max_fetch_hi)
                byte_val = lo_idx + (hi_idx << 4)
                print(f"  Decoded FETCH byte: 0x{byte_val:02x} (expected 0x{target_byte0:02x})")
            else:
                print(f"  FETCH appears empty (max_lo={max_fetch_lo:.2f})")

            # Also check TEMP[0] (IS_JSR flag)
            temp_0 = last[BD.TEMP + 0].item()
            print(f"  TEMP[0] (IS_JSR): {temp_0:.2f}")

            # Check OP_JSR flag
            op_jsr = last[BD.OP_JSR].item()
            print(f"  OP_JSR: {op_jsr:.2f}")

        if layer_idx == 5:  # L6 is the routing layer
            last = x[0, -1]
            print(f"\nAfter L6 (routing layer):")
            output_lo = [last[BD.OUTPUT_LO + k].item() for k in range(16)]
            output_hi = [last[BD.OUTPUT_HI + k].item() for k in range(16)]
            print(f"  OUTPUT_LO: {[f'{v:.1f}' for v in output_lo]}")
            print(f"  OUTPUT_HI: {[f'{v:.1f}' for v in output_hi]}")

            # Decode OUTPUT as byte
            max_out_lo = max(output_lo)
            max_out_hi = max(output_hi)
            if max_out_lo > 0.5:
                lo_idx = output_lo.index(max_out_lo)
                hi_idx = output_hi.index(max_out_hi)
                byte_val = lo_idx + (hi_idx << 4)
                print(f"  Decoded OUTPUT byte: 0x{byte_val:02x} (expected 0x{target_byte0:02x})")
            else:
                print(f"  OUTPUT appears empty or wrong")

            # Check TEMP[0] propagation
            temp_0 = last[BD.TEMP + 0].item()
            print(f"  TEMP[0] (IS_JSR): {temp_0:.2f}")

    # Generate and check token
    logits = runner.model.head(x[0, -1:])
    top5_vals, top5_ids = torch.topk(logits[0], 5)
    print(f"\nTop 5 predictions:")
    for i in range(5):
        tok = top5_ids[i].item()
        val = top5_vals[i].item()
        if tok < 256:
            print(f"  {i}: byte 0x{tok:02x}={tok} (score={val:.2f})")
        else:
            print(f"  {i}: tok {tok} (score={val:.2f})")
    print(f"Expected: byte 0x{target_byte0:02x}={target_byte0}")

# Generate actual token
next_token = runner.model.generate_next(ctx)
print(f"\nGenerated PC byte 0: {next_token} (expected: {target_byte0})")
