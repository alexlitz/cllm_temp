"""
Trace OUTPUT contamination through layers 7-15.
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

reg_ax_pos = len(context) - 1
token_ids = torch.tensor([context], dtype=torch.long, device='cuda')

with torch.no_grad():
    x = model.embed(token_ids)
    
    # Forward to Layer 7
    for i in range(7):
        x = model.blocks[i](x, kv_cache=None)
    
    print("Layer-by-layer OUTPUT values at REG_AX marker:")
    print(f"After L6: LO[10]={x[0, reg_ax_pos, BD.OUTPUT_LO + 10].item():.1f}, HI[2]={x[0, reg_ax_pos, BD.OUTPUT_HI + 2].item():.1f}")
    
    for layer_idx in range(7, 16):
        x_before = x.clone()
        x = model.blocks[layer_idx](x, kv_cache=None)
        
        lo_before = x_before[0, reg_ax_pos, BD.OUTPUT_LO + 10].item()
        hi_before = x_before[0, reg_ax_pos, BD.OUTPUT_HI + 2].item()
        lo_after = x[0, reg_ax_pos, BD.OUTPUT_LO + 10].item()
        hi_after = x[0, reg_ax_pos, BD.OUTPUT_HI + 2].item()
        
        lo_delta = lo_after - lo_before
        hi_delta = hi_after - hi_before
        
        marker = ""
        if abs(lo_delta) > 10 or abs(hi_delta) > 10:
            marker = " ← CONTAMINATION\!"
        
        print(f"After L{layer_idx}: LO[10]={lo_after:.1f} (Δ{lo_delta:+.1f}), HI[2]={hi_after:.1f} (Δ{hi_delta:+.1f}){marker}")
