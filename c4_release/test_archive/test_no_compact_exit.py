import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim
from neural_vm.embedding import Opcode
from neural_vm.run_vm import AutoregressiveVMRunner

BD = _SetDim

print("Building model WITHOUT compaction...")
model = AutoregressiveVM()
set_vm_weights(model)
# Don't compact
model = model.cuda()
model.eval()

runner = AutoregressiveVMRunner()
runner.model = model
bytecode = [Opcode.IMM | (42 << 8), Opcode.EXIT]
context = runner._build_context(bytecode, b'', [])

print("Generating step 0...")
for _ in range(100):
    tok = model.generate_next(context)
    context.append(tok)
    if tok == Token.STEP_END:
        break

print("Generating step 1 up to REG_AX marker...")
for _ in range(6):
    tok = model.generate_next(context)
    context.append(tok)
    if tok == Token.REG_AX:
        break

marker_pos = len(context) - 1
print(f"REG_AX marker at position {marker_pos}")

token_ids = torch.tensor([context], dtype=torch.long, device='cuda')

with torch.no_grad():
    logits = model.forward(token_ids)
    prediction = logits[0, marker_pos, :].argmax(-1).item()
    print(f"\nPredicted byte: 0x{prediction:02x}")
    if prediction == 0x2a:
        print("✓ CORRECT\!")
    else:
        print(f"✗ WRONG\! Expected 0x2a")
        top5 = torch.topk(logits[0, marker_pos, :256], 5)
        print("Top 5:")
        for val, idx in zip(top5.values, top5.indices):
            print(f"  0x{idx.item():02x}: {val.item():.1f}")
