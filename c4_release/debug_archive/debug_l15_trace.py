#\!/usr/bin/env python3
"""Trace L15 at step 2 PC marker."""

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

context = runner._build_context(bytecode, b"", [], "")
for _ in range(35):
    token = model.generate_next(context)
    context.append(token)
    if token == Token.STEP_END:
        break

token = model.generate_next(context)
context.append(token)
step2_pc_marker_idx = len(context) - 1

with torch.no_grad():
    input_ids = torch.tensor([context], dtype=torch.long, device=device)
    x = model.embed(input_ids, active_opcode=model._active_opcode)

    # Run L0-L14
    for i in range(15):
        x = model.blocks[i](x, kv_cache=None)

    print("=== Before L15 (after L14) ===")
    state = x[0, step2_pc_marker_idx, :]
    out_lo = state[BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    out_hi = state[BD.OUTPUT_HI:BD.OUTPUT_HI+16]
    print(f"OUTPUT_LO: argmax [{out_lo.argmax().item()}] = {out_lo.max().item():.2f}")
    print(f"OUTPUT_HI: argmax [{out_hi.argmax().item()}] = {out_hi.max().item():.2f}")
    print(f"Decoded: {out_lo.argmax().item() + 16*out_hi.argmax().item()}")
    
    print(f"\nKey dims for L15 Q activation:")
    print(f"  CONST: {state[BD.CONST].item():.2f}")
    print(f"  OP_LI_RELAY: {state[BD.OP_LI_RELAY].item():.2f}")
    print(f"  OP_LC_RELAY: {state[BD.OP_LC_RELAY].item():.2f}")
    print(f"  MARK_STACK0: {state[BD.MARK_STACK0].item():.2f}")
    print(f"  OP_LEV: {state[BD.OP_LEV].item():.2f}")
    print(f"  MARK_BP: {state[BD.MARK_BP].item():.2f}")
    print(f"  MARK_PC: {state[BD.MARK_PC].item():.2f}")
    print(f"  L1H4[BP=3]: {state[BD.L1H4 + 3].item():.2f}")
    print(f"  H1[BP=3]: {state[BD.H1 + 3].item():.2f}")
    print(f"  CMP[0]: {state[BD.CMP + 0].item():.2f}")
    
    # Compute expected Q[0] for head 0
    q0 = -2000 * state[BD.CONST].item()
    q0 += 2000 * state[BD.OP_LI_RELAY].item()
    q0 += 2000 * state[BD.OP_LC_RELAY].item()
    q0 += 2000 * state[BD.MARK_STACK0].item()
    q0 -= 2000 * state[BD.CMP + 0].item()
    print(f"\n  Head 0 Q[0] (bias dim): {q0:.0f} (should be << 0 to suppress)")

    # Run L15
    block15 = model.blocks[15]
    attn_out = block15.attn(x, None)
    if isinstance(attn_out, tuple):
        attn_out = attn_out[0]
    x_after_attn = x + attn_out

    print("\n=== After L15 attn ===")
    state = x_after_attn[0, step2_pc_marker_idx, :]
    out_lo = state[BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    out_hi = state[BD.OUTPUT_HI:BD.OUTPUT_HI+16]
    print(f"OUTPUT_LO: argmax [{out_lo.argmax().item()}] = {out_lo.max().item():.2f}")
    print(f"OUTPUT_HI: argmax [{out_hi.argmax().item()}] = {out_hi.max().item():.2f}")
    print(f"Decoded: {out_lo.argmax().item() + 16*out_hi.argmax().item()}")
    
    # Check delta
    delta = attn_out[0, step2_pc_marker_idx, :]
    delta_lo = delta[BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    delta_hi = delta[BD.OUTPUT_HI:BD.OUTPUT_HI+16]
    print(f"\nL15 attn delta to OUTPUT_LO: {delta_lo.tolist()[:8]}...")
    print(f"L15 attn delta to OUTPUT_HI: {delta_hi.tolist()[:8]}...")

    # Run L15 FFN
    ffn_out = block15.ffn(x_after_attn)
    x_after_ffn = x_after_attn + ffn_out

    print("\n=== After L15 FFN ===")
    state = x_after_ffn[0, step2_pc_marker_idx, :]
    out_lo = state[BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    out_hi = state[BD.OUTPUT_HI:BD.OUTPUT_HI+16]
    print(f"OUTPUT_LO: argmax [{out_lo.argmax().item()}] = {out_lo.max().item():.2f}")
    print(f"OUTPUT_HI: argmax [{out_hi.argmax().item()}] = {out_hi.max().item():.2f}")
    print(f"Decoded: {out_lo.argmax().item() + 16*out_hi.argmax().item()}")
