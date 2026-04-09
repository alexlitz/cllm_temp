"""
Check which ALU opcodes are active at REG_AX marker.
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

    # Forward to Layer 10
    for i in range(10):
        x = model.blocks[i](x, kv_cache=None)

    x_before_l10 = x + model.blocks[10].attn(x, kv_cache=None)

    print(f"\nBefore Layer 10 FFN at marker position {marker_pos}:")

    # Check all ALU opcodes
    alu_opcodes = [
        ('OP_ADD', BD.OP_ADD),
        ('OP_SUB', BD.OP_SUB),
        ('OP_MUL', BD.OP_MUL),
        ('OP_DIV', BD.OP_DIV),
        ('OP_MOD', BD.OP_MOD),
        ('OP_AND', BD.OP_AND),
        ('OP_OR', BD.OP_OR),
        ('OP_XOR', BD.OP_XOR),
        ('OP_SHL', BD.OP_SHL),
        ('OP_SHR', BD.OP_SHR),
    ]

    print("\nALU opcode flags:")
    for name, dim in alu_opcodes:
        val = x_before_l10[0, marker_pos, dim].item()
        if abs(val) > 0.01:
            print(f"  {name}: {val:.3f}")

    print(f"\nALU operand values:")
    for k in range(16):
        lo = x_before_l10[0, marker_pos, BD.ALU_LO + k].item()
        hi = x_before_l10[0, marker_pos, BD.ALU_HI + k].item()
        if abs(lo) > 0.1 or abs(hi) > 0.1:
            print(f"  ALU_LO[{k}]: {lo:.2f}, ALU_HI[{k}]: {hi:.2f}")

    print(f"\nAX_CARRY operand values:")
    for k in range(16):
        lo = x_before_l10[0, marker_pos, BD.AX_CARRY_LO + k].item()
        hi = x_before_l10[0, marker_pos, BD.AX_CARRY_HI + k].item()
        if abs(lo) > 0.1 or abs(hi) > 0.1:
            print(f"  AX_CARRY_LO[{k}]: {lo:.2f}, AX_CARRY_HI[{k}]: {hi:.2f}")

    print(f"\nI/O instruction flags:")
    print(f"  OP_EXIT: {x_before_l10[0, marker_pos, BD.OP_EXIT].item():.3f}")
    print(f"  OP_IMM: {x_before_l10[0, marker_pos, BD.OP_IMM].item():.3f}")
