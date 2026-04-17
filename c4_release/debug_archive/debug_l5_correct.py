#!/usr/bin/env python3
"""Debug L5 (blocks[5]) attention for JSR immediate fetch."""

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
jsr_target = jsr_instr >> 8
target_byte0 = jsr_target & 0xFF
print(f"JSR target byte 0: 0x{target_byte0:02x}")

# Create runner
runner = AutoregressiveVMRunner()
runner._func_call_handlers = {}
runner._bytecode = bytecode
runner._last_sp = 0x1F800
runner._last_bp = 0x10000
ctx = runner._build_context(bytecode, data, [])

# Generate REG_PC marker
runner.model.set_active_opcode(Opcode.JSR)
next_token = runner.model.generate_next(ctx)
ctx.append(next_token)
print(f"Generated REG_PC marker: {next_token}")

# Trace through layers using CORRECT layer indices
print(f"\n=== Layer-by-layer trace at PC marker position ===")
with torch.no_grad():
    device = next(runner.model.parameters()).device
    token_ids = torch.tensor([ctx], dtype=torch.long, device=device)

    x = runner.model.embed(token_ids)

    for layer_idx in range(8):  # Layers 0-7
        x = runner.model.blocks[layer_idx](x)

        if layer_idx == 4:  # This is L4 (not L5!)
            last = x[0, -1]
            print(f"\nAfter L4 (blocks[4]):")
            print(f"  MARK_PC: {last[BD.MARK_PC].item():.2f}")
            print(f"  HAS_SE: {last[BD.HAS_SE].item():.2f}")

        if layer_idx == 5:  # This is L5 (fetch layer)
            last = x[0, -1]
            fetch_lo = [last[BD.FETCH_LO + k].item() for k in range(16)]
            fetch_hi = [last[BD.FETCH_HI + k].item() for k in range(16)]
            print(f"\nAfter L5 (blocks[5]):")
            print(f"  FETCH_LO: {[f'{v:.1f}' for v in fetch_lo]}")
            print(f"  FETCH_HI: {[f'{v:.1f}' for v in fetch_hi]}")
            
            max_lo = max(fetch_lo)
            max_hi = max(fetch_hi)
            if max_lo > 0.5:
                lo_idx = fetch_lo.index(max_lo)
                hi_idx = fetch_hi.index(max_hi)
                print(f"  Decoded FETCH byte: 0x{lo_idx + (hi_idx << 4):02x} (expected 0x{target_byte0:02x})")
            else:
                print(f"  FETCH is empty (max_lo={max_lo:.2f})")

            # Check TEMP[0] (IS_JSR flag set by L5 FFN)
            temp_0 = last[BD.TEMP + 0].item()
            print(f"  TEMP[0] (IS_JSR): {temp_0:.2f}")

        if layer_idx == 6:  # L6 (routing layer)
            last = x[0, -1]
            output_lo = [last[BD.OUTPUT_LO + k].item() for k in range(16)]
            output_hi = [last[BD.OUTPUT_HI + k].item() for k in range(16)]
            print(f"\nAfter L6 (blocks[6]):")
            print(f"  OUTPUT_LO: {[f'{v:.1f}' for v in output_lo]}")
            print(f"  OUTPUT_HI: {[f'{v:.1f}' for v in output_hi]}")
            
            max_lo = max(output_lo)
            max_hi = max(output_hi)
            if max_lo > 0.5:
                lo_idx = output_lo.index(max_lo)
                hi_idx = output_hi.index(max_hi)
                print(f"  Decoded OUTPUT byte: 0x{lo_idx + (hi_idx << 4):02x} (expected 0x{target_byte0:02x})")
            else:
                print(f"  OUTPUT is weak (max_lo={max_lo:.2f})")

            temp_0 = last[BD.TEMP + 0].item()
            print(f"  TEMP[0] (IS_JSR): {temp_0:.2f}")

# Generate PC byte 0
next_token = runner.model.generate_next(ctx)
print(f"\n=== Generation Result ===")
print(f"Generated PC byte 0: {next_token} (expected: {target_byte0})")
