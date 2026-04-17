"""
Debug Layer 6 FFN input/output to find which unit writes OUTPUT_HI[2].
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
model = model.cuda()
model.eval()

# Build initial context
value = 42
bytecode = [
    Opcode.IMM | (value << 8),
    Opcode.EXIT,
]

runner = AutoregressiveVMRunner()
runner.model = model
context = runner._build_context(bytecode, b'', [])
context.append(Token.REG_PC)

# Forward pass
token_ids = torch.tensor([context], dtype=torch.long, device='cuda')

with torch.no_grad():
    x = model.embed(token_ids)
    pos = len(context) - 1

    # Forward to Layer 6
    for i in range(6):
        x = model.blocks[i](x, kv_cache=None)

    # Layer 6 attention
    x_before_attn = x.clone()
    attn_out = model.blocks[6].attn(x, kv_cache=None)
    x_after_attn = x + attn_out

    print(f"Position {pos} (REG_PC):")
    print(f"\nBefore Layer 6 attention:")
    print(f"  OUTPUT_HI[2]: {x_before_attn[0, pos, BD.OUTPUT_HI + 2].item():.3f}")

    print(f"\nAfter Layer 6 attention:")
    print(f"  OUTPUT_HI[2]: {x_after_attn[0, pos, BD.OUTPUT_HI + 2].item():.3f}")

    # Check FFN input values
    S = 16.0
    mark_pc = x_after_attn[0, pos, BD.MARK_PC].item()
    mark_ax = x_after_attn[0, pos, BD.MARK_AX].item()
    op_imm = x_after_attn[0, pos, BD.OP_IMM].item()

    print(f"\nFFN input values (what FFN sees):")
    print(f"  MARK_PC: {mark_pc:.3f}")
    print(f"  MARK_AX: {mark_ax:.3f}")
    print(f"  OP_IMM: {op_imm:.3f}")

    # Check all CMP values
    print(f"  CMP values:")
    for k in range(8):
        cmp_val = x_after_attn[0, pos, BD.CMP + k].item()
        if abs(cmp_val) > 0.01:
            print(f"    CMP[{k}]: {cmp_val:.3f}")

    # Now FFN
    ffn_out = model.blocks[6].ffn(x_after_attn)
    x_after_ffn = x_after_attn + ffn_out

    print(f"\nAfter Layer 6 FFN:")
    print(f"  OUTPUT_HI[2]: {x_after_ffn[0, pos, BD.OUTPUT_HI + 2].item():.3f}")

    # Check the FFN output itself (before residual)
    print(f"\nFFN output (delta, before residual):")
    print(f"  OUTPUT_HI[2]: {ffn_out[0, pos, BD.OUTPUT_HI + 2].item():.3f}")

    # Manual activation check (with -MARK_PC fix)
    T = 5.5
    activation_imm = S * op_imm + S * mark_ax - S * mark_pc - S * T
    print(f"\nManual IMM unit activation (with -MARK_PC guard):")
    print(f"  {S} * {op_imm:.3f} + {S} * {mark_ax:.3f} - {S} * {mark_pc:.3f} - {S} * {T}")
    print(f"  = {activation_imm:.3f}")
    print(f"  Should fire: {activation_imm > 0}")

    if activation_imm > 0:
        # Compute expected output
        fetch_hi_2 = x_after_attn[0, pos, BD.FETCH_HI + 2].item()
        # Unit: W_gate[unit, FETCH_HI+2] = 1.0
        #       W_down[OUTPUT_HI+2, unit] = 2.0 / S
        # Output: swish(activation) * gate_input * W_down
        #       = swish(activation) * FETCH_HI[2] * (2.0 / S)
        swish = activation_imm / (1 + torch.exp(-torch.tensor(activation_imm))).item()
        expected = swish * fetch_hi_2 * (2.0 / S)
        print(f"\n  Expected FFN output:")
        print(f"    FETCH_HI[2]: {fetch_hi_2:.3f}")
        print(f"    swish({activation_imm:.3f}): {swish:.3f}")
        print(f"    Output: {swish:.3f} * {fetch_hi_2:.3f} * {2.0/S:.3f} = {expected:.3f}")
