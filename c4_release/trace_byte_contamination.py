"""
Trace OUTPUT[0] contamination at byte position through layers 7-15.
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

# Build bytecode and generate to step 1 first AX byte position
bytecode = [Opcode.IMM | (42 << 8), Opcode.EXIT]
runner = AutoregressiveVMRunner()
runner.model = model
context = runner._build_context(bytecode, b'', [])

for _ in range(100):
    tok = model.generate_next(context)
    context.append(tok)
    if tok == Token.STEP_END:
        break

# Generate 7 more: REG_PC + 4 bytes + REG_AX + first byte
for _ in range(7):
    tok = model.generate_next(context)
    context.append(tok)

byte_pos = len(context) - 1  # Position of first AX byte
print(f"Tracing OUTPUT at byte position {byte_pos} (first AX byte = 0x{context[byte_pos]:02x})")

token_ids = torch.tensor([context], dtype=torch.long, device='cuda')

with torch.no_grad():
    x = model.embed(token_ids)

    # Forward to Layer 7
    for i in range(7):
        x = model.blocks[i](x, kv_cache=None)

    print("\nLayer-by-layer OUTPUT[0] and OUTPUT[10] values at byte position:")
    print(f"After L6: LO[0]={x[0, byte_pos, BD.OUTPUT_LO + 0].item():.1f}, LO[10]={x[0, byte_pos, BD.OUTPUT_LO + 10].item():.1f}")

    for layer_idx in range(7, 16):
        x_before = x.clone()
        x = model.blocks[layer_idx](x, kv_cache=None)

        lo0_before = x_before[0, byte_pos, BD.OUTPUT_LO + 0].item()
        lo10_before = x_before[0, byte_pos, BD.OUTPUT_LO + 10].item()
        lo0_after = x[0, byte_pos, BD.OUTPUT_LO + 0].item()
        lo10_after = x[0, byte_pos, BD.OUTPUT_LO + 10].item()

        lo0_delta = lo0_after - lo0_before
        lo10_delta = lo10_after - lo10_before

        marker = ""
        if abs(lo0_delta) > abs(lo10_delta) + 5:
            marker = " ← OUTPUT[0] CONTAMINATION!"

        print(f"After L{layer_idx}: LO[0]={lo0_after:.1f} (Δ{lo0_delta:+.1f}), LO[10]={lo10_after:.1f} (Δ{lo10_delta:+.1f}){marker}")
