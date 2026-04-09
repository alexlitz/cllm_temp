"""Trace bytecode execution to identify which step executes ADD."""

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Opcode
from neural_vm.embedding import Token
import torch
import sys

# Test programs
test_add = "int main() { return 10 + 32; }"
test_imm = "int main() { return 42; }"

def trace_execution(code, desc, disable_handler=None):
    """Trace VM execution and show which opcodes execute in which steps."""
    print(f"\n{'=' * 70}", file=sys.stderr)
    print(f"TRACING: {desc}", file=sys.stderr)
    print(f"Code: {code}", file=sys.stderr)
    print(f"{'=' * 70}", file=sys.stderr)

    bytecode, data = compile_c(code)
    print(f"Bytecode: {len(bytecode)} bytes", file=sys.stderr)

    runner = AutoregressiveVMRunner()
    if disable_handler and disable_handler in runner._func_call_handlers:
        del runner._func_call_handlers[disable_handler]
        print(f"✓ Disabled {disable_handler} handler", file=sys.stderr)

    runner.model.cuda()

    # Hook to capture generated tokens
    generated_tokens = []

    original_generate = runner.model.forward
    def capture_forward(input_ids, *args, **kwargs):
        result = original_generate(input_ids, *args, **kwargs)
        # result is logits (batch, seq, vocab)
        if result.shape[1] == input_ids.shape[1] + 1:
            # Generated one new token
            next_token = result[0, -1, :].argmax().item()
            generated_tokens.append(next_token)
        return result

    runner.model.forward = capture_forward

    print("Running VM...", file=sys.stderr)
    _, exit_code = runner.run(bytecode, max_steps=50)

    print(f"\nExit code: {exit_code}", file=sys.stderr)
    print(f"Total tokens generated: {len(generated_tokens)}", file=sys.stderr)

    # Parse generated tokens into VM steps
    # Each step is 35 tokens: PC(5) + AX(5) + SP(5) + BP(5) + STACK0(5) + MEM(9) + SE(1)
    print(f"\nVM Steps:", file=sys.stderr)

    step = 0
    token_idx = 0
    while token_idx < len(generated_tokens):
        # Step starts with REG_PC
        if token_idx >= len(generated_tokens):
            break

        step_tokens = generated_tokens[token_idx:min(token_idx + 35, len(generated_tokens))]

        # Extract PC value (tokens 1-4 after REG_PC)
        if len(step_tokens) >= 5:
            pc_bytes = step_tokens[1:5]
            # Convert byte tokens to value
            # Tokens are like: byte_00, byte_0a, etc.
            # Need to map token IDs back to values

            # Extract AX value (tokens 6-9 after REG_AX at position 5)
            if len(step_tokens) >= 10:
                ax_bytes = step_tokens[6:10]

                print(f"  Step {step}: tokens {token_idx}-{token_idx + len(step_tokens) - 1}", file=sys.stderr)

        token_idx += 35
        step += 1

    del runner
    torch.cuda.empty_cache()

    return exit_code

# Trace ADD execution
print("\n" + "=" * 70, file=sys.stderr)
print("EXECUTION TRACES", file=sys.stderr)
print("=" * 70, file=sys.stderr)

exit_code_add = trace_execution(test_add, "ADD test (handler disabled)", disable_handler=Opcode.ADD)
exit_code_imm = trace_execution(test_imm, "IMM test (baseline)", disable_handler=None)

print("\n" + "=" * 70, file=sys.stderr)
print("COMPARISON", file=sys.stderr)
print("=" * 70, file=sys.stderr)
print(f"ADD result: {exit_code_add} (expected 42)", file=sys.stderr)
print(f"IMM result: {exit_code_imm} (expected 42)", file=sys.stderr)

if exit_code_add == 42:
    print("✓ ADD works neurally!", file=sys.stderr)
elif exit_code_add == 10:
    print("✗ ADD returns first operand only", file=sys.stderr)
else:
    print(f"✗ ADD returns unexpected value: {exit_code_add}", file=sys.stderr)
