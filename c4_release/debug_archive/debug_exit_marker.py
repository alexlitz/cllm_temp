"""
Check EXIT units at the REG_AX MARKER position.
"""

import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim
from neural_vm.embedding import Opcode
from neural_vm.run_vm import AutoregressiveVMRunner

BD = _SetDim

# Build model
print("Building model...")
model = AutoregressiveVM()
set_vm_weights(model)
model.compact(block_size=32)
model.compact_moe()
model = model.cuda()
model.eval()

# Build bytecode
bytecode = [Opcode.IMM | (42 << 8), Opcode.EXIT]
runner = AutoregressiveVMRunner()
runner.model = model
context = runner._build_context(bytecode, b'', [])

# Generate step 0
for _ in range(100):
    tok = model.generate_next(context)
    context.append(tok)
    if tok == Token.STEP_END:
        break

# Generate step 1 up to REG_AX marker (not past it)
for _ in range(6):
    tok = model.generate_next(context)
    context.append(tok)
    if tok == Token.REG_AX:
        break

reg_ax_pos = len(context) - 1
print(f"REG_AX marker at position {reg_ax_pos}")

# Forward pass
token_ids = torch.tensor([context], dtype=torch.long, device='cuda')

with torch.no_grad():
    x = model.embed(token_ids)
    
    # Forward to Layer 6
    for i in range(6):
        x = model.blocks[i](x, kv_cache=None)
    
    x_before_l6_ffn = x + model.blocks[6].attn(x, kv_cache=None)
    
    print(f"\nAt REG_AX marker (position {reg_ax_pos}), before Layer 6 FFN:")
    print(f"  OP_EXIT: {x_before_l6_ffn[0, reg_ax_pos, BD.OP_EXIT].item():.3f}")
    print(f"  MARK_AX: {x_before_l6_ffn[0, reg_ax_pos, BD.MARK_AX].item():.3f}")
    print(f"  MARK_PC: {x_before_l6_ffn[0, reg_ax_pos, BD.MARK_PC].item():.3f}")
    print(f"  IS_BYTE: {x_before_l6_ffn[0, reg_ax_pos, BD.IS_BYTE].item():.3f}")
    print(f"  AX_CARRY_LO[10]: {x_before_l6_ffn[0, reg_ax_pos, BD.AX_CARRY_LO + 10].item():.3f}")
    print(f"  AX_CARRY_HI[2]: {x_before_l6_ffn[0, reg_ax_pos, BD.AX_CARRY_HI + 2].item():.3f}")
    
    # Check EXIT unit activation
    S = 16.0
    T = 0.5
    op_exit = x_before_l6_ffn[0, reg_ax_pos, BD.OP_EXIT].item()
    mark_ax = x_before_l6_ffn[0, reg_ax_pos, BD.MARK_AX].item()
    mark_pc = x_before_l6_ffn[0, reg_ax_pos, BD.MARK_PC].item()
    is_byte = x_before_l6_ffn[0, reg_ax_pos, BD.IS_BYTE].item()
    
    activation = S * op_exit + S * mark_ax - S * mark_pc - S * is_byte - S * T
    print(f"\n  EXIT unit activation:")
    print(f"    {S}*{op_exit:.3f} + {S}*{mark_ax:.3f} - {S}*{mark_pc:.3f} - {S}*{is_byte:.3f} - {S}*{T}")
    print(f"    = {activation:.3f}")
    print(f"    Should fire: {activation > 0}")
    
    # Run Layer 6 FFN
    x_after_l6_ffn = x_before_l6_ffn + model.blocks[6].ffn(x_before_l6_ffn)
    
    print(f"\n  After Layer 6 FFN:")
    print(f"    OUTPUT_LO[10]: {x_after_l6_ffn[0, reg_ax_pos, BD.OUTPUT_LO + 10].item():.3f}")
    print(f"    OUTPUT_HI[2]: {x_after_l6_ffn[0, reg_ax_pos, BD.OUTPUT_HI + 2].item():.3f}")
    
    expected_output = 2.0 / S  # EXIT uses gate weight 1.0, down weight 2.0/S
    if abs(x_after_l6_ffn[0, reg_ax_pos, BD.OUTPUT_LO + 10].item()) > 0.05 and \
       abs(x_after_l6_ffn[0, reg_ax_pos, BD.OUTPUT_HI + 2].item()) > 0.05:
        print(f"    ✓ EXIT wrote to OUTPUT\!")
    else:
        print(f"    ✗ EXIT did not write to OUTPUT\!")
    
    # Continue to Layer 15
    x_after_all = x_after_l6_ffn
    for i in range(7, 16):
        x_after_all = model.blocks[i](x_after_all, kv_cache=None)
    
    print(f"\n  After Layer 15 (all layers):")
    print(f"    OUTPUT_LO[10]: {x_after_all[0, reg_ax_pos, BD.OUTPUT_LO + 10].item():.3f}")
    print(f"    OUTPUT_HI[2]: {x_after_all[0, reg_ax_pos, BD.OUTPUT_HI + 2].item():.3f}")
    
    # Predict
    logits = model.head(x_after_all)
    next_token = logits[0, reg_ax_pos, :].argmax(-1).item()
    print(f"\n  Predicted next token: 0x{next_token:02x}")
    if next_token == 0x2a:
        print(f"    ✓ CORRECT!")
    else:
        print(f"    ✗ WRONG! Expected 0x2a")
