"""
Debug what Layer 10 FFN is adding to OUTPUT[0].
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

    print(f"\nBefore Layer 10 FFN at marker position {marker_pos}:")
    print(f"  OUTPUT_LO[0]: {x_before_l10_ffn[0, marker_pos, BD.OUTPUT_LO + 0].item():.1f}")
    print(f"  OUTPUT_LO[10]: {x_before_l10_ffn[0, marker_pos, BD.OUTPUT_LO + 10].item():.1f}")

    # Run Layer 10 FFN
    ffn_output = model.blocks[10].ffn(x_before_l10_ffn)
    x_after_l10_ffn = x_before_l10_ffn + ffn_output

    print(f"\nAfter Layer 10 FFN:")
    print(f"  OUTPUT_LO[0]: {x_after_l10_ffn[0, marker_pos, BD.OUTPUT_LO + 0].item():.1f}")
    print(f"  OUTPUT_LO[10]: {x_after_l10_ffn[0, marker_pos, BD.OUTPUT_LO + 10].item():.1f}")

    # Check what FFN added
    delta_lo0 = ffn_output[0, marker_pos, BD.OUTPUT_LO + 0].item()
    delta_lo10 = ffn_output[0, marker_pos, BD.OUTPUT_LO + 10].item()

    print(f"\nLayer 10 FFN deltas:")
    print(f"  OUTPUT_LO[0] delta: {delta_lo0:.2f}")
    print(f"  OUTPUT_LO[10] delta: {delta_lo10:.2f}")

    # Check activation dims that might trigger Layer 10 units
    print(f"\nActivation dims before Layer 10 FFN:")
    print(f"  OP_EXIT: {x_before_l10_ffn[0, marker_pos, BD.OP_EXIT].item():.2f}")
    print(f"  OP_IMM: {x_before_l10_ffn[0, marker_pos, BD.OP_IMM].item():.2f}")
    print(f"  MARK_AX: {x_before_l10_ffn[0, marker_pos, BD.MARK_AX].item():.2f}")
    print(f"  IS_BYTE: {x_before_l10_ffn[0, marker_pos, BD.IS_BYTE].item():.2f}")

    # Check if there are any other dims with large values
    print(f"\nTop 10 largest activation dims:")
    vals, indices = torch.topk(x_before_l10_ffn[0, marker_pos, :].abs(), 10)
    for val, idx in zip(vals, indices):
        print(f"    Dim {idx.item()}: {x_before_l10_ffn[0, marker_pos, idx].item():.2f}")
