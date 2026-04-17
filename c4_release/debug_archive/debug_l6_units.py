"""
Check which Layer 6 FFN units are firing at REG_AX marker.
"""

import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim
from neural_vm.embedding import Opcode
from neural_vm.run_vm import AutoregressiveVMRunner

BD = _SetDim

# Build model
model = AutoregressiveVM()
set_vm_weights(model)
model.compact(block_size=32)
model.compact_moe()
model = model.cuda()
model.eval()

# Build bytecode and generate to step 1 REG_AX marker
bytecode = [Opcode.IMM | (42 << 8), Opcode.EXIT]
runner = AutoregressiveVMRunner()
runner.model = model
context = runner._build_context(bytecode, b'', [])

for _ in range(100):
    tok = model.generate_next(context)
    context.append(tok)
    if tok == Token.STEP_END:
        break

for _ in range(6):
    tok = model.generate_next(context)
    context.append(tok)
    if tok == Token.REG_AX:
        break

marker_pos = len(context) - 1
print(f"REG_AX marker at position {marker_pos}")

token_ids = torch.tensor([context], dtype=torch.long, device='cuda')

with torch.no_grad():
    x = model.embed(token_ids)

    # Forward to Layer 6
    for i in range(6):
        x = model.blocks[i](x, kv_cache=None)

    x_before_l6_ffn = x + model.blocks[6].attn(x, kv_cache=None)

    print(f"\nBefore Layer 6 FFN at marker position {marker_pos}:")
    print(f"  OP_EXIT: {x_before_l6_ffn[0, marker_pos, BD.OP_EXIT].item():.2f}")
    print(f"  OP_IMM: {x_before_l6_ffn[0, marker_pos, BD.OP_IMM].item():.2f}")
    print(f"  OP_NOP: {x_before_l6_ffn[0, marker_pos, BD.OP_NOP].item():.2f}")
    print(f"  MARK_AX: {x_before_l6_ffn[0, marker_pos, BD.MARK_AX].item():.2f}")
    print(f"  MARK_PC: {x_before_l6_ffn[0, marker_pos, BD.MARK_PC].item():.2f}")
    print(f"  IS_BYTE: {x_before_l6_ffn[0, marker_pos, BD.IS_BYTE].item():.2f}")

    print(f"\n  AX_CARRY values:")
    for k in range(16):
        lo = x_before_l6_ffn[0, marker_pos, BD.AX_CARRY_LO + k].item()
        hi = x_before_l6_ffn[0, marker_pos, BD.AX_CARRY_HI + k].item()
        if abs(lo) > 0.1 or abs(hi) > 0.1:
            print(f"    LO[{k}]: {lo:.2f}, HI[{k}]: {hi:.2f}")

    print(f"\n  FETCH values:")
    for k in range(16):
        lo = x_before_l6_ffn[0, marker_pos, BD.FETCH_LO + k].item()
        hi = x_before_l6_ffn[0, marker_pos, BD.FETCH_HI + k].item()
        if abs(lo) > 0.1 or abs(hi) > 0.1:
            print(f"    LO[{k}]: {lo:.2f}, HI[{k}]: {hi:.2f}")

    # Check unit activations (assuming S=16, T=0.5)
    S = 16.0
    T = 0.5
    op_exit = x_before_l6_ffn[0, marker_pos, BD.OP_EXIT].item()
    op_imm = x_before_l6_ffn[0, marker_pos, BD.OP_IMM].item()
    mark_ax = x_before_l6_ffn[0, marker_pos, BD.MARK_AX].item()
    mark_pc = x_before_l6_ffn[0, marker_pos, BD.MARK_PC].item()
    is_byte = x_before_l6_ffn[0, marker_pos, BD.IS_BYTE].item()

    exit_activation = S * op_exit + S * mark_ax - S * mark_pc - S * is_byte - S * T
    imm_activation = S * op_imm + S * mark_ax - S * mark_pc - S * is_byte - S * T

    print(f"\nUnit activations:")
    print(f"  EXIT: {S}*{op_exit:.2f} + {S}*{mark_ax:.2f} - {S}*{mark_pc:.2f} - {S}*{is_byte:.2f} - {S}*{T} = {exit_activation:.1f}")
    print(f"    Should fire: {exit_activation > 0}")
    print(f"  IMM: {S}*{op_imm:.2f} + {S}*{mark_ax:.2f} - {S}*{mark_pc:.2f} - {S}*{is_byte:.2f} - {S}*{T} = {imm_activation:.1f}")
    print(f"    Should fire: {imm_activation > 0}")

    # Run Layer 6 FFN
    x_after_l6_ffn = x_before_l6_ffn + model.blocks[6].ffn(x_before_l6_ffn)

    print(f"\nAfter Layer 6 FFN:")
    print(f"  OUTPUT_LO[0]: {x_after_l6_ffn[0, marker_pos, BD.OUTPUT_LO + 0].item():.1f}")
    print(f"  OUTPUT_LO[10]: {x_after_l6_ffn[0, marker_pos, BD.OUTPUT_LO + 10].item():.1f}")
    print(f"  OUTPUT_HI[0]: {x_after_l6_ffn[0, marker_pos, BD.OUTPUT_HI + 0].item():.1f}")
    print(f"  OUTPUT_HI[2]: {x_after_l6_ffn[0, marker_pos, BD.OUTPUT_HI + 2].item():.1f}")
