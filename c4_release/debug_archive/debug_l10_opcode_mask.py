"""
Debug Layer 10 opcode mask value at REG_AX marker.
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

    x_before_l10_ffn = x + model.blocks[10].attn(x, kv_cache=None)

    # Get the ALU FFN module
    alu_ffn = model.blocks[10].ffn

    # Convert to GE format
    x_ge = alu_ffn.bd_to_ge(x_before_l10_ffn)  # [B, seq_len, 8, 160]

    print(f"\nAfter BD→GE conversion at marker position {marker_pos}:")
    print(f"  x_ge shape: {x_ge.shape}")

    # Check opcode flags in GE format
    ge = alu_ffn.ge
    print(f"\n  GE opcode flags (position 0):")
    print(f"    OP_AND (ge.OP_START+30={ge.OP_START + 30}): {x_ge[0, marker_pos, 0, ge.OP_START + 30].item():.3f}")
    print(f"    OP_OR (ge.OP_START+28={ge.OP_START + 28}): {x_ge[0, marker_pos, 0, ge.OP_START + 28].item():.3f}")
    print(f"    OP_XOR (ge.OP_START+29={ge.OP_START + 29}): {x_ge[0, marker_pos, 0, ge.OP_START + 29].item():.3f}")

    # Compute opcode_mask as ALU does
    x_ge_flat = x_ge.view(1 * len(context), 8, ge.DIM)
    op_and = x_ge_flat[:, 0, ge.OP_START + 30]  # [seq_len]
    op_or = x_ge_flat[:, 0, ge.OP_START + 28]
    op_xor = x_ge_flat[:, 0, ge.OP_START + 29]

    opcode_mask_flat = op_and + op_or + op_xor  # [seq_len]
    opcode_mask = opcode_mask_flat.view(1, len(context))  # [1, seq_len]

    print(f"\n  opcode_mask at marker position: {opcode_mask[0, marker_pos].item():.6f}")

    if abs(opcode_mask[0, marker_pos].item()) < 0.001:
        print(f"    → opcode_mask is effectively 0, ALU should NOT write OUTPUT!")
    else:
        print(f"    → opcode_mask is non-zero, ALU WILL write OUTPUT")

    # Check what RESULT values are in GE format
    print(f"\n  GE RESULT values:")
    print(f"    RESULT at position 0 (lo byte): {x_ge[0, marker_pos, 0, ge.RESULT].item():.2f}")
    print(f"    RESULT at position 1 (hi byte): {x_ge[0, marker_pos, 1, ge.RESULT].item():.2f}")
