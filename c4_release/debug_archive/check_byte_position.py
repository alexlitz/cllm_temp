"""
Check OUTPUT at the byte position where prediction happens.
"""

import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim
from neural_vm.embedding import Opcode
from neural_vm.run_vm import AutoregressiveVMRunner

BD = _SetDim

# Build and run
model = AutoregressiveVM()
set_vm_weights(model)
model.compact(block_size=32)
model.compact_moe()
model = model.cuda()
model.eval()

bytecode = [Opcode.IMM | (42 << 8), Opcode.EXIT]
runner = AutoregressiveVMRunner()
runner.model = model
context = runner._build_context(bytecode, b'', [])

# Generate to step 1 first AX byte position
for _ in range(100):
    tok = model.generate_next(context)
    context.append(tok)
    if tok == Token.STEP_END:
        break

for _ in range(6):  # REG_PC + 4 bytes + REG_AX
    tok = model.generate_next(context)
    context.append(tok)

byte_pos = len(context)
print(f"At byte position {byte_pos} (should predict first AX byte)")

token_ids = torch.tensor([context], dtype=torch.long, device='cuda')

with torch.no_grad():
    logits = model.forward(token_ids)
    
    # Get activations at this position
    x = model.embed(token_ids)
    for i in range(16):
        x = model.blocks[i](x, kv_cache=None)
    
    print(f"\nAfter all layers at byte position {byte_pos}:")
    print(f"  IS_BYTE: {x[0, byte_pos, BD.IS_BYTE].item():.3f}")
    print(f"  OUTPUT_LO values:")
    for k in range(16):
        val = x[0, byte_pos, BD.OUTPUT_LO + k].item()
        if abs(val) > 0.5:
            print(f"    [{k}]: {val:.2f}")
    
    print(f"  OUTPUT_HI values:")
    for k in range(16):
        val = x[0, byte_pos, BD.OUTPUT_HI + k].item()
        if abs(val) > 0.5:
            print(f"    [{k}]: {val:.2f}")
    
    # Prediction
    next_token = logits[0, byte_pos, :].argmax(-1).item()
    print(f"\n  Predicted: 0x{next_token:02x}")
    if next_token == 0x2a:
        print(f"    ✓ CORRECT\!")
    else:
        print(f"    ✗ WRONG\! Expected 0x2a")
