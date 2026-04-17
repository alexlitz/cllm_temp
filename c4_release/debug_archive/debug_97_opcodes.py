import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim as BD
from neural_vm.embedding import Opcode
from neural_vm.speculative import DraftVM

def build_context(bytecode):
    tokens = [Token.CODE_START]
    for instr in bytecode:
        op = instr & 0xFF
        imm = instr >> 8
        tokens.append(op)
        for i in range(4):
            tokens.append((imm >> (i * 8)) & 0xFF)
    tokens.append(Token.CODE_END)
    tokens.append(Token.DATA_START)
    tokens.append(Token.DATA_END)
    return tokens

model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

# Test all 97.1% opcodes
test_cases = [
    ('NOP', [Opcode.NOP]),
    ('EXIT', [Opcode.EXIT]),
    ('JMP', [Opcode.JMP | (12 << 8)]),
    ('BZ (taken)', [Opcode.IMM | (0 << 8), Opcode.BZ | (12 << 8)]),
    ('BNZ (not taken)', [Opcode.IMM | (0 << 8), Opcode.BNZ | (12 << 8)]),
    ('SHR', [Opcode.IMM | (16 << 8), Opcode.PSH, Opcode.IMM | (2 << 8), Opcode.SHR]),
]

# Position names for reference
pos_names = ['PC', 'PC.0', 'PC.1', 'PC.2', 'PC.3',
             'AX', 'AX.0', 'AX.1', 'AX.2', 'AX.3',
             'SP', 'SP.0', 'SP.1', 'SP.2', 'SP.3',
             'BP', 'BP.0', 'BP.1', 'BP.2', 'BP.3',
             'STACK0', 'STACK0.0', 'STACK0.1', 'STACK0.2', 'STACK0.3',
             'MEM', 'MEM.addr.0', 'MEM.addr.1', 'MEM.addr.2', 'MEM.addr.3',
             'MEM.val.0', 'MEM.val.1', 'MEM.val.2', 'MEM.val.3', 'SE']

print('=== Failure Analysis ===')
for name, bytecode in test_cases:
    context = build_context(bytecode)
    draft_vm = DraftVM(bytecode)
    draft_vm.step()
    draft_tokens = draft_vm.draft_tokens()

    ctx_tensor = torch.tensor([context + draft_tokens], dtype=torch.long)
    ctx_len = len(context)

    with torch.no_grad():
        logits = model.forward(ctx_tensor)

    failures = []
    for i in range(35):
        expected = draft_tokens[i]
        predicted = logits[0, ctx_len - 1 + i, :].argmax().item()
        if expected != predicted:
            failures.append((i, expected, predicted))

    print(f'{name}: {35-len(failures)}/35')
    for pos, exp, pred in failures:
        pos_name = pos_names[pos] if pos < len(pos_names) else f'pos{pos}'
        print(f'  {pos_name} (pos {pos}): expected {exp}, got {pred}')
