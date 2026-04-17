"""Test if alternative toggle configurations actually execute correctly."""

import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights
from neural_vm.embedding import Opcode
from neural_vm.speculative import DraftVM

def test_simple_program_with_config(use_softmax1, pos_encoding):
    """Test a simple IMM 42; EXIT program with specific configuration."""

    print(f"\nTesting: use_softmax1={use_softmax1}, pos_encoding={pos_encoding}")

    # Simple program: IMM 42; EXIT
    bytecode = [Opcode.IMM | (42 << 8), Opcode.EXIT]

    # Create model with this configuration
    model = AutoregressiveVM(
        use_softmax1=use_softmax1,
        pos_encoding=pos_encoding
    )

    # Try to set weights (should fail for non-default configs)
    try:
        set_vm_weights(model)
        print(f"  ✓ set_vm_weights succeeded")
    except ValueError as e:
        print(f"  ✗ set_vm_weights failed: {e}")
        print(f"  → This configuration requires training from scratch")
        return False

    # Run with DraftVM to get expected result
    draft_vm = DraftVM(bytecode)
    draft_vm.step()
    expected_ax = draft_vm.ax
    print(f"  Expected AX (from DraftVM): {expected_ax}")

    # Build context
    from neural_vm.vm_step import Token
    context = [Token.CODE_START]
    for instr in bytecode:
        for i in range(8):
            context.append((instr >> (i * 8)) & 0xFF)
    context.append(Token.CODE_END)
    context.append(Token.DATA_START)
    context.append(Token.DATA_END)

    # Try to generate one step with the transformer
    model.eval()
    device = 'cpu'  # Use CPU for simplicity
    model = model.to(device)

    # Generate tokens autoregressively for one VM step (35 tokens)
    input_ids = torch.tensor([context], dtype=torch.long, device=device)

    with torch.no_grad():
        for i in range(35):  # One VM step = 35 tokens
            logits = model(input_ids)
            next_token = logits[0, -1, :].argmax().item()
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)

    # Extract the generated tokens
    generated = input_ids[0, len(context):].tolist()
    print(f"  Generated {len(generated)} tokens")

    # Check if it matches what DraftVM produced
    draft_tokens = draft_vm.draft_tokens()
    matches = sum(1 for i in range(min(len(generated), len(draft_tokens)))
                  if generated[i] == draft_tokens[i])
    print(f"  Matches: {matches}/{len(draft_tokens)} tokens")

    if matches == len(draft_tokens):
        print(f"  ✓ Perfect match! Configuration works correctly.")
        return True
    else:
        print(f"  ✗ Mismatch! This configuration doesn't work correctly.")
        # Show first mismatch
        for i in range(len(draft_tokens)):
            if i >= len(generated) or generated[i] != draft_tokens[i]:
                print(f"    First mismatch at token {i}: expected {draft_tokens[i]}, got {generated[i] if i < len(generated) else 'N/A'}")
                break
        return False

if __name__ == "__main__":
    print("="*70)
    print("Testing Toggle Correctness")
    print("="*70)

    configs = [
        (True, 'alibi'),   # Default - should work
        (False, 'alibi'),  # F.softmax - won't work without training
        (True, 'rope'),    # RoPE - won't work without training
        (False, 'rope'),   # Both changed - won't work
        (True, 'none'),    # No pos encoding - won't work
    ]

    results = []
    for use_softmax1, pos_encoding in configs:
        result = test_simple_program_with_config(use_softmax1, pos_encoding)
        results.append((use_softmax1, pos_encoding, result))

    print("\n" + "="*70)
    print("Summary:")
    print("="*70)
    for use_softmax1, pos_encoding, result in results:
        status = "✓ WORKS" if result else "✗ DOESN'T WORK"
        print(f"  softmax1={use_softmax1}, pos_encoding={pos_encoding}: {status}")
