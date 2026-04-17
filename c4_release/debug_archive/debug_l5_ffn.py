#!/usr/bin/env python3
"""Debug L5 FFN to see if it's overwriting FETCH values."""

import torch
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token, _SetDim as BD
from neural_vm.embedding import Opcode
from src.compiler import compile_c

code = 'int main() { return 42; }'
bytecode, data = compile_c(code)

runner = AutoregressiveVMRunner()
runner._func_call_handlers = {}
runner._bytecode = bytecode
runner._last_sp = 0x1F800
runner._last_bp = 0x10000
ctx = runner._build_context(bytecode, data, [])

runner.model.set_active_opcode(Opcode.JSR)
next_token = runner.model.generate_next(ctx)
ctx.append(next_token)

with torch.no_grad():
    device = next(runner.model.parameters()).device
    token_ids = torch.tensor([ctx], dtype=torch.long, device=device)

    # Pass through layers 0-4
    x = runner.model.embed(token_ids)
    for i in range(5):
        x = runner.model.blocks[i](x)

    # L5 = blocks[5]
    L5_block = runner.model.blocks[5]

    # Manually compute attention output
    attn_out = L5_block.attn(x)

    # Residual after attention
    x_after_attn = x + attn_out

    print(f"=== After L5 attention (x + attn_out) ===")
    print(f"FETCH_LO: {[x_after_attn[0, -1, BD.FETCH_LO + k].item() for k in range(8)]}")
    print(f"FETCH_HI: {[x_after_attn[0, -1, BD.FETCH_HI + k].item() for k in range(8)]}")

    # Now FFN
    ffn_out = L5_block.ffn(x_after_attn)

    print(f"\n=== FFN output (before adding to residual) ===")
    print(f"FETCH_LO dims in FFN output: {[ffn_out[0, -1, BD.FETCH_LO + k].item() for k in range(8)]}")
    print(f"FETCH_HI dims in FFN output: {[ffn_out[0, -1, BD.FETCH_HI + k].item() for k in range(8)]}")

    # Final output after residual
    x_final = x_after_attn + ffn_out

    print(f"\n=== After L5 FFN residual (x_after_attn + ffn_out) ===")
    print(f"FETCH_LO: {[x_final[0, -1, BD.FETCH_LO + k].item() for k in range(8)]}")
    print(f"FETCH_HI: {[x_final[0, -1, BD.FETCH_HI + k].item() for k in range(8)]}")

    # Compare with full block output
    x_block = L5_block(x)
    print(f"\n=== Full L5 block output (should match above) ===")
    print(f"FETCH_LO: {[x_block[0, -1, BD.FETCH_LO + k].item() for k in range(8)]}")
    print(f"FETCH_HI: {[x_block[0, -1, BD.FETCH_HI + k].item() for k in range(8)]}")
