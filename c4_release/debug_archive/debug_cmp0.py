#!/usr/bin/env python3
"""Debug CMP[0] at PC marker for step 2."""

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

# Trace CMP[0] through layers
with torch.no_grad():
    input_ids = torch.tensor([context], dtype=torch.long, device=device)
    x = model.embed(input_ids, active_opcode=model._active_opcode)

    print("Tracing CMP[0] at step 2 PC marker:")
    print("Layer | CMP[0] | OUTPUT decoded | Note")
    print("-" * 50)

    for layer_idx in range(len(model.blocks)):
        block = model.blocks[layer_idx]
        x = block(x, kv_cache=None)

        pc_state = x[0, step2_pc_marker_idx, :]
        cmp0 = pc_state[BD.CMP + 0].item()
        out_lo = pc_state[BD.OUTPUT_LO:BD.OUTPUT_LO+16]
        out_hi = pc_state[BD.OUTPUT_HI:BD.OUTPUT_HI+16]
        decoded = out_lo.argmax().item() + 16 * out_hi.argmax().item()

        # JMP threshold is 5.5, so CMP[0] > 4.5 triggers override
        if cmp0 > 4.5:
            note = "CMP[0] > 4.5 - JMP override may fire!"
        elif layer_idx == 6 and decoded != 34:
            note = "L6 corrupted OUTPUT!"
        else:
            note = ""
        print(f"  L{layer_idx:2d} | {cmp0:6.2f} | {decoded:3d}           | {note}")

    print(f"\nFinal OUTPUT: {decoded} (expected 34)")
    print(f"CMP[0] at L6: {cmp0:.2f} (threshold for JMP override is 4.5)")
