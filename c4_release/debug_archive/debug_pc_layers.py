#!/usr/bin/env python3
"""Debug PC carry-forward by tracing layer-by-layer state."""

import torch
from neural_vm.vm_step import AutoregressiveVM, Token, _SetDim as BD
from neural_vm.embedding import Opcode
from neural_vm.run_vm import AutoregressiveVMRunner

bytecode = [
    Opcode.JSR | (26 << 8),  # Instr 0: Jump to PC=26
    Opcode.EXIT,
    Opcode.NOP,
    Opcode.IMM | (42 << 8),
    Opcode.EXIT,
]

runner = AutoregressiveVMRunner()
model = runner.model
device = next(model.parameters()).device

# Build initial context
context = runner._build_context(bytecode, b"", [], "")

# Generate step 1 completely
for _ in range(35):
    token = model.generate_next(context)
    context.append(token)
    if token == Token.STEP_END:
        break

step1_end = len(context) - 1
print(f"Step 1 ended at idx {step1_end}")

# Find step 1's PC byte 0 position
pc_byte0_idx = None
for i in range(len(context)):
    if context[i] == Token.REG_PC:
        pc_byte0_idx = i + 1
        break

print(f"Step 1 PC byte 0 at idx {pc_byte0_idx}, value = {context[pc_byte0_idx]}")

# Generate step 2's PC marker
token = model.generate_next(context)
context.append(token)
print(f"Step 2 first token: {token} ({'PC marker' if token == Token.REG_PC else 'ERROR'})")

step2_pc_marker_idx = len(context) - 1

# Now trace layer-by-layer state at step 2 PC marker position
print(f"\n=== Layer-by-layer state at step 2 PC marker (idx {step2_pc_marker_idx}) ===")

with torch.no_grad():
    input_ids = torch.tensor([context], dtype=torch.long, device=device)

    # Get embedding
    x = model.embed(input_ids, active_opcode=model._active_opcode)  # [1, seq_len, d]
    print(f"\nAfter embedding:")
    state = x[0, step2_pc_marker_idx, :]
    print(f"  EMBED_LO argmax: [{state[BD.EMBED_LO:BD.EMBED_LO+16].argmax().item()}]")
    print(f"  EMBED_HI argmax: [{state[BD.EMBED_HI:BD.EMBED_HI+16].argmax().item()}]")
    print(f"  MARK_PC: {state[BD.MARK_PC].item():.2f}")
    print(f"  HAS_SE: {state[BD.HAS_SE].item():.2f}")

    # Check step 1 PC byte 0 position embedding
    prev_state = x[0, pc_byte0_idx, :]
    print(f"\n  (Step 1 PC byte 0 embedding at idx {pc_byte0_idx}):")
    print(f"    EMBED_LO argmax: [{prev_state[BD.EMBED_LO:BD.EMBED_LO+16].argmax().item()}] = {prev_state[BD.EMBED_LO:BD.EMBED_LO+16].max().item():.2f}")
    print(f"    EMBED_HI argmax: [{prev_state[BD.EMBED_HI:BD.EMBED_HI+16].argmax().item()}] = {prev_state[BD.EMBED_HI:BD.EMBED_HI+16].max().item():.2f}")
    print(f"    L1H0[PC=0]: {prev_state[BD.L1H0 + 0].item():.2f}")
    print(f"    L1H1[PC=0]: {prev_state[BD.L1H1 + 0].item():.2f}")

    # Run through layers
    for layer_idx, block in enumerate(model.blocks):
        # Attention
        attn_out, _ = block.attn(block.ln1(x), None)
        x = x + attn_out

        state = x[0, step2_pc_marker_idx, :]
        prev_state = x[0, pc_byte0_idx, :]

        if layer_idx in [0, 1, 2, 3, 4, 5, 6]:
            print(f"\nAfter L{layer_idx} attn:")
            print(f"  PC marker - EMBED_LO argmax: [{state[BD.EMBED_LO:BD.EMBED_LO+16].argmax().item()}] val={state[BD.EMBED_LO:BD.EMBED_LO+16].max().item():.2f}")
            print(f"  PC marker - EMBED_HI argmax: [{state[BD.EMBED_HI:BD.EMBED_HI+16].argmax().item()}] val={state[BD.EMBED_HI:BD.EMBED_HI+16].max().item():.2f}")
            print(f"  PC marker - L1H1[PC]: {state[BD.L1H1 + 0].item():.2f}")
            print(f"  Step1 PC byte0 - L1H1[PC]: {prev_state[BD.L1H1 + 0].item():.2f}")
            print(f"  Step1 PC byte0 - L1H0[PC]: {prev_state[BD.L1H0 + 0].item():.2f}")

        # FFN
        ffn_out = block.ffn(block.ln2(x))
        x = x + ffn_out

        state = x[0, step2_pc_marker_idx, :]

        if layer_idx in [0, 1, 2, 3, 4, 5, 6]:
            print(f"\nAfter L{layer_idx} FFN:")
            print(f"  PC marker - EMBED_LO argmax: [{state[BD.EMBED_LO:BD.EMBED_LO+16].argmax().item()}] val={state[BD.EMBED_LO:BD.EMBED_LO+16].max().item():.2f}")
            print(f"  PC marker - EMBED_HI argmax: [{state[BD.EMBED_HI:BD.EMBED_HI+16].argmax().item()}] val={state[BD.EMBED_HI:BD.EMBED_HI+16].max().item():.2f}")
            print(f"  PC marker - OUTPUT_LO argmax: [{state[BD.OUTPUT_LO:BD.OUTPUT_LO+16].argmax().item()}] val={state[BD.OUTPUT_LO:BD.OUTPUT_LO+16].max().item():.2f}")
            print(f"  PC marker - OUTPUT_HI argmax: [{state[BD.OUTPUT_HI:BD.OUTPUT_HI+16].argmax().item()}] val={state[BD.OUTPUT_HI:BD.OUTPUT_HI+16].max().item():.2f}")

print("\n=== Final state ===")
state = x[0, step2_pc_marker_idx, :]
out_lo = state[BD.OUTPUT_LO:BD.OUTPUT_LO+16]
out_hi = state[BD.OUTPUT_HI:BD.OUTPUT_HI+16]
print(f"OUTPUT_LO: argmax [{out_lo.argmax().item()}] = {out_lo.max().item():.2f}")
print(f"OUTPUT_HI: argmax [{out_hi.argmax().item()}] = {out_hi.max().item():.2f}")
print(f"Decoded OUTPUT: {out_lo.argmax().item() + 16 * out_hi.argmax().item()}")
print(f"Expected: 34")

# Now generate PC byte 0 and see what we get
token = model.generate_next(context)
context.append(token)
print(f"\nGenerated PC byte 0: {token}")
print(f"Expected: 2 (34 % 16 = 2, but full byte should be 34)")
