"""Test if token validation actually catches mismatches."""
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights
from neural_vm.batch_runner_v2 import UltraBatchRunner
from neural_vm.embedding import Opcode

print("Testing if speculative validation actually rejects bad tokens...\n")

# Test IMM 42; EXIT
bytecode = [Opcode.IMM | (42 << 8), Opcode.EXIT]

runner = UltraBatchRunner(batch_size=1, device='cpu')
results = runner.run_batch([bytecode], max_steps=10)

print(f"IMM 42; EXIT result: {results[0]}")
print(f"Expected: 42")
print()

# Check how many tokens were accepted in the first step
# We can check this by looking at the draft VM's execution
from neural_vm.speculative import DraftVM
from neural_vm.vm_step import Token

def build_context(bytecode):
    tokens = [Token.CODE_START]
    for instr in bytecode:
        for i in range(8):
            tokens.append((instr >> (i * 8)) & 0xFF)
    tokens.append(Token.CODE_END)
    tokens.append(Token.DATA_START)
    tokens.append(Token.DATA_END)
    return tokens

# Manually test validation
draft_vm = DraftVM(bytecode)
context = build_context(bytecode)

# Execute first step
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

print(f"DraftVM produced {len(draft_tokens)} tokens")
print(f"AX from DraftVM: token[6] = {draft_tokens[6]} (expecting 42)")
print()

# Now check what transformer predicts
model = runner.model
ctx_with_draft = [context + draft_tokens]
token_ids = torch.tensor(ctx_with_draft, dtype=torch.long)

with torch.no_grad():
    logits = model.forward(token_ids)

ctx_len = len(context)
print("Checking transformer predictions vs DraftVM tokens:")
print("=" * 70)

mismatches = []
for i in range(35):
    predicted = logits[0, ctx_len - 1 + i, :].argmax().item()
    expected = draft_tokens[i]
    match = "✓" if predicted == expected else "✗"
    if predicted != expected:
        mismatches.append((i, expected, predicted))
        if len(mismatches) <= 5:  # Only print first 5 mismatches
            print(f"{match} Token {i:2d}: predicted={predicted:3d}, expected={expected:3d}")

print()
if mismatches:
    print(f"Found {len(mismatches)} mismatches out of 35 tokens")
    print("\nNow checking what verify_speculative_batch returns...")

    accepted = model.verify_speculative_batch(ctx_with_draft, [35])
    print(f"Tokens accepted: {accepted[0]}/35")

    if accepted[0] == 35:
        print("\n🚨 BUG: All 35 tokens accepted despite mismatches!")
        print("This means validation is not working correctly.")
    else:
        print(f"\n✓ Validation working: stopped at first mismatch (token {accepted[0]})")
        print(f"   Expected exit code: 42")
        print(f"   Actual exit code: {results[0]}")
        if results[0] == 42:
            print("\n🤔 Exit code is correct despite token rejection.")
            print("   This means DraftVM keeps executing even when tokens are rejected.")
else:
    print("✓ All 35 tokens match - no mismatches found!")
