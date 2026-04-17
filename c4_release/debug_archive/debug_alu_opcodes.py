import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim
from neural_vm.embedding import Opcode
from neural_vm.run_vm import AutoregressiveVMRunner

BD = _SetDim

model = AutoregressiveVM()
set_vm_weights(model)
model = model.cuda()
model.eval()

runner = AutoregressiveVMRunner()
runner.model = model
bytecode = [Opcode.IMM | (42 << 8), Opcode.EXIT]
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
token_ids = torch.tensor([context], dtype=torch.long, device='cuda')

with torch.no_grad():
    x = model.embed(token_ids)
    for i in range(10):
        x = model.blocks[i](x, kv_cache=None)

    print(f"Before Layer 10 FFN at position {marker_pos}:")
    print(f"  OP_EXIT: {x[0, marker_pos, BD.OP_EXIT].item():.2f}")
    print(f"  OP_IMM: {x[0, marker_pos, BD.OP_IMM].item():.2f}")
    print(f"  OP_AND: {x[0, marker_pos, BD.OP_AND].item():.2f}")
    print(f"  OP_OR: {x[0, marker_pos, BD.OP_OR].item():.2f}")
    print(f"  OP_XOR: {x[0, marker_pos, BD.OP_XOR].item():.2f}")

    # Check what the ALU sees
    alu = model.blocks[10].ffn
    x_ge = alu.bd_to_ge(x)
    ge = alu.ge
    
    print(f"\nAfter BD->GE conversion (GE format):")
    print(f"  GE OP_AND (idx {ge.OP_START + 30}): {x_ge[0, marker_pos, 0, ge.OP_START + 30].item():.2f}")
    print(f"  GE OP_OR (idx {ge.OP_START + 28}): {x_ge[0, marker_pos, 0, ge.OP_START + 28].item():.2f}")
    print(f"  GE OP_XOR (idx {ge.OP_START + 29}): {x_ge[0, marker_pos, 0, ge.OP_START + 29].item():.2f}")
