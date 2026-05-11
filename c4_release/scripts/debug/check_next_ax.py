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
with torch.no_grad():
    x = model.embed(token_ids, active_opcode=model._active_opcode)
    for block in model.blocks:
        x = block(x)
    pos = prefix_len + 5
    v = x[0, pos]
    print(f"At REG_AX_mark (pos {pos}):")
    print(f"  NEXT_PC={v[BD.NEXT_PC].item():.2f}, NEXT_AX={v[BD.NEXT_AX].item():.2f}, NEXT_SP={v[BD.NEXT_SP].item():.2f}")
    print(f"  NEXT_BP={v[BD.NEXT_BP].item():.2f}, NEXT_STACK0={v[BD.NEXT_STACK0].item():.2f}, NEXT_MEM={v[BD.NEXT_MEM].item():.2f}")
    print(f"  NEXT_SE={v[BD.NEXT_SE].item():.2f}, NEXT_HALT={v[BD.NEXT_HALT].item():.2f}")
    print(f"  OUTPUT_LO[5]={v[BD.OUTPUT_LO+5].item():.2f}, OUTPUT_HI[0]={v[BD.OUTPUT_HI+0].item():.2f}")
    print(f"  OUTPUT_LO[0..15]: {[v[BD.OUTPUT_LO+k].item() for k in range(16)]}")
    print(f"  OUTPUT_HI[0..15]: {[v[BD.OUTPUT_HI+k].item() for k in range(16)]}")
