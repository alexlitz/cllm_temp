#!/usr/bin/env python3
"""Debug PC OUTPUT corruption after L3."""

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

# Trace OUTPUT through all layers
with torch.no_grad():
    input_ids = torch.tensor([context], dtype=torch.long, device=device)
    x = model.embed(input_ids, active_opcode=model._active_opcode)

    for layer_idx in range(len(model.blocks)):
        block = model.blocks[layer_idx]
        x = block(x, kv_cache=None)

        pc_state = x[0, step2_pc_marker_idx, :]
        out_lo = pc_state[BD.OUTPUT_LO:BD.OUTPUT_LO+16]
        out_hi = pc_state[BD.OUTPUT_HI:BD.OUTPUT_HI+16]
        out_lo_max_idx = out_lo.argmax().item()
        out_hi_max_idx = out_hi.argmax().item()
        out_lo_max_val = out_lo.max().item()
        out_hi_max_val = out_hi.max().item()
        decoded = out_lo_max_idx + 16 * out_hi_max_idx

        # Show all layers, highlight changes
        if decoded != 34:
            marker = "  *** WRONG"
        else:
            marker = "  OK"
        print(f"L{layer_idx:2d}: OUTPUT_LO[{out_lo_max_idx:2d}]={out_lo_max_val:6.2f}, OUTPUT_HI[{out_hi_max_idx:2d}]={out_hi_max_val:6.2f} -> {decoded:3d}{marker}")

    # Check final logits
    print(f"\n=== Final OUTPUT: {decoded} (expected 34) ===")
