"""Check if IS_MARK is set for marker tokens during generation."""

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Opcode, Token, _SetDim
import torch
import sys

BD = _SetDim

code = "int main() { return 10 + 32; }"
bytecode, data = compile_c(code)

print("=" * 70, file=sys.stderr)
print("CHECKING IS_MARK DURING ALL EMBEDDING PASSES", file=sys.stderr)
print("=" * 70, file=sys.stderr)

runner = AutoregressiveVMRunner()
if Opcode.ADD in runner._func_call_handlers:
    del runner._func_call_handlers[Opcode.ADD]

runner.model.cuda()

# Capture ALL embedding passes
embed_passes = []

def embed_hook(module, input, output):
    # Store: (input_ids, output_embeddings)
    if len(input) > 0:
        embed_passes.append((input[0].detach().cpu(), output.detach().cpu()))

h1 = runner.model.embed.register_forward_hook(embed_hook)

print("Running VM (max 1 step)...", file=sys.stderr)
_, exit_code = runner.run(bytecode, max_steps=1)

h1.remove()

print(f"Exit code: {exit_code}", file=sys.stderr)
print(f"Total embedding passes: {len(embed_passes)}\n", file=sys.stderr)

# Analyze embedding passes
markers = {
    Token.REG_PC: "REG_PC",
    Token.REG_AX: "REG_AX",
    Token.REG_SP: "REG_SP",
    Token.REG_BP: "REG_BP",
    Token.MEM: "MEM",
    Token.STACK0: "STACK0",
    Token.STEP_END: "STEP_END",
}

# Check the last few passes (should be generation of step tokens)
print("Last 10 embedding passes (generation):", file=sys.stderr)
for pass_idx in range(max(0, len(embed_passes) - 10), len(embed_passes)):
    input_ids, embed_out = embed_passes[pass_idx]
    seq_len = input_ids.shape[1]

    # Check last token in sequence (the one just added)
    if seq_len > 0:
        last_tok = input_ids[0, -1].item()
        is_mark_val = embed_out[0, -1, BD.IS_MARK].item()

        tok_name = markers.get(last_tok, f"token_{last_tok}" if last_tok >= 256 else f"byte_0x{last_tok:02x}")

        if last_tok in markers:
            # This is a marker token - check if IS_MARK is set
            print(f"  Pass {pass_idx}: seq_len={seq_len}, last_tok={tok_name}, IS_MARK={is_mark_val:.2f}", file=sys.stderr)

            if is_mark_val < 0.5:
                print(f"    ❌ IS_MARK NOT SET for {tok_name}!", file=sys.stderr)
            else:
                print(f"    ✓ IS_MARK set correctly", file=sys.stderr)

print("", file=sys.stderr)
print("=" * 70, file=sys.stderr)
print("DETAILED ANALYSIS", file=sys.stderr)
print("=" * 70, file=sys.stderr)

# Find the pass where REG_AX was generated
print("\nLooking for REG_AX token generation:", file=sys.stderr)
for pass_idx, (input_ids, embed_out) in enumerate(embed_passes):
    seq_len = input_ids.shape[1]
    if seq_len > 0:
        last_tok = input_ids[0, -1].item()
        if last_tok == Token.REG_AX:
            is_mark_val = embed_out[0, -1, BD.IS_MARK].item()
            print(f"  Pass {pass_idx}: REG_AX generated (position {seq_len-1})", file=sys.stderr)
            print(f"  IS_MARK value: {is_mark_val:.3f}", file=sys.stderr)

            # Check all embedding dimensions for this token
            print(f"  Non-zero dimensions at this position:", file=sys.stderr)
            for dim in range(embed_out.shape[2]):
                val = embed_out[0, -1, dim].item()
                if abs(val) > 0.1:
                    print(f"    Dim {dim}: {val:.2f}", file=sys.stderr)
                    if dim == BD.IS_MARK:
                        print(f"      ^ This is IS_MARK (dim {BD.IS_MARK})", file=sys.stderr)
            break

print("", file=sys.stderr)
print("=" * 70, file=sys.stderr)

del runner
torch.cuda.empty_cache()
