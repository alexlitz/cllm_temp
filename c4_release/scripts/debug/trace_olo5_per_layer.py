"""Trace OUTPUT_LO[5] at REG_AX_mark layer by layer to find the polluter."""
import torch
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode
from neural_vm.vm_step import _SetDim as BD, Token

runner = AutoregressiveVMRunner(trust_neural_alu=True, pure_neural=True)
runner._func_call_handlers = {}
runner._syscall_handlers = {}
model = runner.model
model.eval()
device = next(model.parameters()).device

bytecode = [(5 << 8) | 1, 38]
context = runner._build_context(bytecode, b"", [], b"")
prefix_len = len(context)
with torch.no_grad():
    for i in range(Token.STEP_TOKENS):
        tok = model.generate_next(context, use_incremental=False)
        context.append(tok)
token_ids = torch.tensor([context], dtype=torch.long, device=device)

import sys
pos = prefix_len + (int(sys.argv[1]) if len(sys.argv) > 1 else 5)  # pos within step

print(f"Tracking OLO[4], OLO[5], OHI[0], OLO[0], CARRY[2] at pos {pos} per layer")
print(f"{'layer':<8s} {'OLO[4]':>10s} {'OLO[5]':>10s} {'OHI[0]':>10s} {'OLO[0]':>10s} {'CARRY[2]':>10s}")
with torch.no_grad():
    x = model.embed(token_ids, active_opcode=model._active_opcode)
    v = x[0, pos]
    print(f"{'embed':<8s} {v[BD.OUTPUT_LO+4].item():>10.2f} {v[BD.OUTPUT_LO+5].item():>10.2f} {v[BD.OUTPUT_HI].item():>10.2f} {v[BD.OUTPUT_LO].item():>10.2f} {v[BD.CARRY+2].item():>10.2f}")
    for i, block in enumerate(model.blocks):
        x_attn = block.attn(x)
        v = x_attn[0, pos]
        print(f"L{i:02d}-attn {v[BD.OUTPUT_LO+4].item():>10.2f} {v[BD.OUTPUT_LO+5].item():>10.2f} {v[BD.OUTPUT_HI].item():>10.2f} {v[BD.OUTPUT_LO].item():>10.2f} {v[BD.CARRY+2].item():>10.2f}")
        x_ffn = block.ffn(x_attn)
        for op in block.post_ops:
            x_ffn = op(x_ffn)
        v = x_ffn[0, pos]
        print(f"L{i:02d}-ffn  {v[BD.OUTPUT_LO+4].item():>10.2f} {v[BD.OUTPUT_LO+5].item():>10.2f} {v[BD.OUTPUT_HI].item():>10.2f} {v[BD.OUTPUT_LO].item():>10.2f} {v[BD.CARRY+2].item():>10.2f}")
        x = x_ffn
