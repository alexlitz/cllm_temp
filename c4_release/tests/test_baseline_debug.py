"""Debug baseline without conversational_io."""

import sys
import os
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token, _SetDim as BD

c_code = 'int main() { return 0; }'

code, data = compile_c(c_code)

# Test both modes
for conv_io in [False, True]:
    print(f"\n{'='*60}")
    print(f"Testing with conversational_io={conv_io}")
    print('='*60)

    runner = AutoregressiveVMRunner(conversational_io=conv_io)

    # Build prefix
    prefix = [Token.CODE_START]
    for instr in code:
        op = instr & 0xFF
        imm = [(instr >> (8 + i*8)) & 0xFF for i in range(4)]
        prefix.append(op)
        prefix.extend(imm)
    prefix.append(Token.CODE_END)
    prefix.append(Token.DATA_START)
    prefix.extend(data)
    prefix.append(Token.DATA_END)
    prefix.append(Token.THINKING_START)

    context = torch.tensor([prefix], dtype=torch.long)
    runner.model.set_active_opcode(code[0] & 0xFF)

    with torch.no_grad():
        logits = runner.model.forward(context)[0, -1, :]

    print(f"\nTop 5 logits:")
    top5 = torch.topk(logits, 5)
    for val, idx in zip(top5.values, top5.indices):
        tok_name = "?"
        for attr in dir(Token):
            if not attr.startswith('_') and getattr(Token, attr) == idx.item():
                tok_name = attr
                break
        if idx < 256:
            tok_name = f"byte_{idx.item()}"
        print(f"  {idx.item():3d} ({tok_name:20s}): {val.item():7.2f}")

    winner = logits.argmax().item()
    expected = Token.REG_PC
    status = "✅" if winner == expected else "❌"
    print(f"\n{status} Winner: {winner}, Expected: {expected}")
