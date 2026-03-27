"""Identify which token position is consistently failing across opcodes."""
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token
from neural_vm.embedding import Opcode
from neural_vm.run_vm import AutoregressiveVMRunner

def test_opcode_with_runner(name, bytecode):
    """Test using AutoregressiveVMRunner (with overrides)."""
    runner = AutoregressiveVMRunner()
    runner.load_program(bytecode)

    # Run one step
    runner.step()

    # Get the context (includes bytecode + first step output)
    context = runner._context

    # Find where output starts (after DATA_END)
    output_start = None
    for i, tok in enumerate(context):
        if tok == Token.DATA_END:
            output_start = i + 1
            break

    if output_start is None:
        print(f"{name}: Could not find output start")
        return

    # Check first step tokens (35 tokens)
    model = runner._model
    ctx_tensor = torch.tensor([context], dtype=torch.long)

    with torch.no_grad():
        logits = model.forward(ctx_tensor)

    # Extract first step (35 tokens)
    token_names = (
        ['REG_PC'] + [f'PC_b{i}' for i in range(4)] +
        ['REG_AX'] + [f'AX_b{i}' for i in range(4)] +
        ['REG_SP'] + [f'SP_b{i}' for i in range(4)] +
        ['REG_BP'] + [f'BP_b{i}' for i in range(4)] +
        ['STACK0'] + [f'ST_b{i}' for i in range(4)] +
        ['MEM'] + [f'MEM_a{i}' for i in range(4)] + [f'MEM_v{i}' for i in range(4)] +
        ['END']
    )

    mismatches = []
    for i in range(35):
        expected = context[output_start + i]
        predicted = logits[0, output_start + i, :].argmax().item()
        if expected != predicted:
            mismatches.append((i, token_names[i], expected, predicted))

    if mismatches:
        print(f"\n{name}: {len(mismatches)}/35 mismatches")
        for pos, name_str, exp, pred in mismatches:
            print(f"  Position {pos:2d} ({name_str:8s}): expected={exp:3d}, predicted={pred:3d}")
    else:
        print(f"{name}: ✓ All 35 tokens match")

    return mismatches

print("Testing opcodes to find consistent mismatch pattern:")
print("=" * 70)

# Test various opcodes
test_opcode_with_runner('IMM', [Opcode.IMM | (5 << 8)])
test_opcode_with_runner('NOP', [Opcode.NOP])
test_opcode_with_runner('EXIT', [Opcode.EXIT])
test_opcode_with_runner('JMP', [Opcode.JMP | (16 << 8)])
test_opcode_with_runner('ADD', [Opcode.IMM | (10 << 8), Opcode.PSH, Opcode.IMM | (5 << 8), Opcode.ADD])
