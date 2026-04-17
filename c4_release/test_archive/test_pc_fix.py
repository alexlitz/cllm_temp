"""Quick test to verify PC byte 0 fix achieves 100% with runner."""
import torch
from neural_vm.embedding import Opcode
from neural_vm.run_vm import AutoregressiveVMRunner

def test_with_runner(name, bytecode):
    """Test using AutoregressiveVMRunner (with overrides)."""
    runner = AutoregressiveVMRunner()
    runner.load_program(bytecode)

    # Get initial context length
    initial_len = len(runner._context)

    # Run one step
    runner.step()

    # Check the added tokens (should be 35)
    added_tokens = len(runner._context) - initial_len
    if added_tokens != 35:
        print(f"{name}: ERROR - expected 35 tokens, got {added_tokens}")
        return False

    # Verify tokens match what the model would predict
    ctx_tensor = torch.tensor([runner._context], dtype=torch.long)

    with torch.no_grad():
        logits = runner._model.forward(ctx_tensor)

    # Check last 35 tokens (the step output)
    mismatches = 0
    for i in range(35):
        pos = initial_len + i
        expected = runner._context[pos]
        predicted = logits[0, pos, :].argmax().item()
        if expected != predicted:
            mismatches += 1

    accuracy = (35 - mismatches) / 35 * 100
    status = "✓" if mismatches == 0 else "⚠"
    print(f"{status} {name}: {35-mismatches}/35 tokens ({accuracy:.1f}%)")
    return mismatches == 0

print("Testing PC byte 0 fix with runner overrides:")
print("=" * 60)

all_pass = True
all_pass &= test_with_runner('NOP', [Opcode.NOP])
all_pass &= test_with_runner('IMM', [Opcode.IMM | (42 << 8)])
all_pass &= test_with_runner('LEA', [Opcode.LEA | (8 << 8)])
all_pass &= test_with_runner('PSH', [Opcode.IMM | (5 << 8), Opcode.PSH])
all_pass &= test_with_runner('ADD', [Opcode.IMM | (10 << 8), Opcode.PSH, Opcode.IMM | (5 << 8), Opcode.ADD])
all_pass &= test_with_runner('JMP', [Opcode.JMP | (16 << 8)])
all_pass &= test_with_runner('EXIT', [Opcode.EXIT])

print()
if all_pass:
    print("🎉 ALL TESTS PASSED - 100% ACCURACY ACHIEVED!")
else:
    print("Some tests still have mismatches")
