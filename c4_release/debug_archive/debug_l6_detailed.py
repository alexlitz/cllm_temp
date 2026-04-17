#\!/usr/bin/env python3
"""Detailed L6 trace at step 2 PC marker."""

import torch
from neural_vm.vm_step import AutoregressiveVM, Token, _SetDim as BD
from neural_vm.embedding import Opcode
from neural_vm.run_vm import AutoregressiveVMRunner

bytecode = [
    Opcode.JSR | (26 << 8),
    Opcode.EXIT,
    Opcode.NOP,
    Opcode.IMM | (42 << 8),
    Opcode.EXIT,
]

runner = AutoregressiveVMRunner()
model = runner.model
device = next(model.parameters()).device

# Build context with step 1 complete and step 2 PC marker
context = runner._build_context(bytecode, b"", [], "")
for _ in range(35):
    token = model.generate_next(context)
    context.append(token)
    if token == Token.STEP_END:
        break

token = model.generate_next(context)
context.append(token)
step2_pc_marker_idx = len(context) - 1

# Trace L6 in detail
with torch.no_grad():
    input_ids = torch.tensor([context], dtype=torch.long, device=device)
    x = model.embed(input_ids, active_opcode=model._active_opcode)

    # Process L0-L5
    for i in range(6):
        x = model.blocks[i](x, kv_cache=None)

    print("=== Before L6 ===")
    state = x[0, step2_pc_marker_idx, :]
    print(f"OUTPUT_LO: {state[BD.OUTPUT_LO:BD.OUTPUT_LO+16].tolist()[:8]}...")
    print(f"OUTPUT_HI: {state[BD.OUTPUT_HI:BD.OUTPUT_HI+16].tolist()[:8]}...")
    out_lo = state[BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    out_hi = state[BD.OUTPUT_HI:BD.OUTPUT_HI+16]
    print(f"Decoded: {out_lo.argmax().item() + 16*out_hi.argmax().item()}")

    # Run L6 attention only
    block6 = model.blocks[6]
    attn_out = block6.attn(x, None)
    if isinstance(attn_out, tuple):
        attn_out = attn_out[0]
    x_after_attn = x + attn_out

    print("\n=== After L6 attn ===")
    state = x_after_attn[0, step2_pc_marker_idx, :]
    print(f"OUTPUT_LO: {state[BD.OUTPUT_LO:BD.OUTPUT_LO+16].tolist()[:8]}...")
    print(f"OUTPUT_HI: {state[BD.OUTPUT_HI:BD.OUTPUT_HI+16].tolist()[:8]}...")
    out_lo = state[BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    out_hi = state[BD.OUTPUT_HI:BD.OUTPUT_HI+16]
    print(f"Decoded: {out_lo.argmax().item() + 16*out_hi.argmax().item()}")

    # Check some key dimensions
    print(f"\nKey dims at PC marker:")
    print(f"  OP_IMM: {state[BD.OP_IMM].item():.2f}")
    print(f"  OP_JSR: {state[BD.OP_JSR].item():.2f}")
    print(f"  OP_JMP: {state[BD.OP_JMP].item():.2f}")
    print(f"  MARK_PC: {state[BD.MARK_PC].item():.2f}")
    print(f"  MARK_AX: {state[BD.MARK_AX].item():.2f}")
    print(f"  HAS_SE: {state[BD.HAS_SE].item():.2f}")
    print(f"  TEMP[0] (IS_JSR): {state[BD.TEMP + 0].item():.2f}")

    # Run L6 FFN
    ffn_out = block6.ffn(x_after_attn)
    x_after_ffn = x_after_attn + ffn_out

    print("\n=== After L6 FFN ===")
    state = x_after_ffn[0, step2_pc_marker_idx, :]
    print(f"OUTPUT_LO: {state[BD.OUTPUT_LO:BD.OUTPUT_LO+16].tolist()[:8]}...")
    print(f"OUTPUT_HI: {state[BD.OUTPUT_HI:BD.OUTPUT_HI+16].tolist()[:8]}...")
    out_lo = state[BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    out_hi = state[BD.OUTPUT_HI:BD.OUTPUT_HI+16]
    print(f"Decoded: {out_lo.argmax().item() + 16*out_hi.argmax().item()}")
    
    # Check what changed
    print("\n=== FFN delta ===")
    delta = ffn_out[0, step2_pc_marker_idx, :]
    delta_lo = delta[BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    delta_hi = delta[BD.OUTPUT_HI:BD.OUTPUT_HI+16]
    print(f"FFN OUTPUT_LO delta: {delta_lo.tolist()[:8]}...")
    print(f"FFN OUTPUT_HI delta: {delta_hi.tolist()[:8]}...")
