"""Trace exactly what happens during batch execution."""
import torch
from neural_vm.batch_runner_v2 import UltraBatchRunner
from neural_vm.embedding import Opcode

print("Tracing execution flow for IMM 42; EXIT\n")
print("=" * 70)

# Patch the runner to add logging
class DebugBatchRunner(UltraBatchRunner):
    def run_batch(self, bytecodes, data_list=None, argv_list=None, max_steps=100):
        print("\n1. INITIALIZATION")
        print(f"   Creating {len(bytecodes)} DraftVMs")

        result = super().run_batch(bytecodes, data_list, argv_list, max_steps)

        print(f"\n6. FINAL RESULT")
        print(f"   Exit code from DraftVM.ax: {result[0]}")
        print(f"   (Note: This comes from DraftVM, NOT transformer predictions)")

        return result

    def _validate_batch_parallel(self, draft_tokens_batch):
        print("\n3. VALIDATION")
        print(f"   DraftVM executed and produced {len(draft_tokens_batch[0])} tokens")
        print(f"   Token 6 (AX byte 0) from DraftVM: {draft_tokens_batch[0][6]}")

        # Call parent validation
        super()._validate_batch_parallel(draft_tokens_batch)

        # Check how many were accepted
        from neural_vm.vm_step import Token
        ctx_len = len(self.contexts[0])
        draft_len = len(draft_tokens_batch[0])
        tokens_in_context = len(self.contexts[0])

        # The context now has the accepted tokens appended
        # But wait, let me check what the parent did...
        print(f"   Context length after validation: {len(self.contexts[0])}")

        # Try to figure out how many tokens were accepted
        # by seeing how much the context grew
        # Actually, let me just call verify_speculative_batch directly
        ctx_with_draft = [self.contexts[0][:ctx_len] + draft_tokens_batch[0]]
        accepted = self.model.verify_speculative_batch(ctx_with_draft, [35])
        print(f"   Transformer accepted: {accepted[0]}/35 tokens")

        if accepted[0] < 35:
            print(f"   ⚠️  Transformer REJECTED tokens after position {accepted[0]}")
            print(f"       But DraftVM has already executed the full step!")

# Create test program
bytecode = [Opcode.IMM | (42 << 8), Opcode.EXIT]

print("\n2. EXECUTION BEGINS")
runner = DebugBatchRunner(batch_size=1, device='cpu')
results = runner.run_batch([bytecode], max_steps=5)

print("\n" + "=" * 70)
print("\nKEY INSIGHT:")
print("  - DraftVM executes in Python (line 105 in batch_runner_v2.py)")
print("  - Transformer validates tokens AFTER DraftVM already executed")
print("  - Final result comes from DraftVM.ax, NOT transformer predictions")
print("  - So tests pass even if transformer predictions are wrong!")
print("\nThis is why:")
print("  ✓ Tests pass (DraftVM execution is correct)")
print("  ✗ Transformer predictions can be wrong (but doesn't affect tests)")
