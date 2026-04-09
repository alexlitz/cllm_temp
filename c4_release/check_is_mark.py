"""Check if IS_MARK flag is set in embeddings."""

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Opcode, _SetDim
from neural_vm.vm_step import Token
import torch
import sys

BD = _SetDim

code = "int main() { return 10 + 32; }"
bytecode, data = compile_c(code)

print("=" * 70, file=sys.stderr)
print("CHECKING IS_MARK FLAG IN EMBEDDINGS", file=sys.stderr)
print("=" * 70, file=sys.stderr)

runner = AutoregressiveVMRunner()
if Opcode.ADD in runner._func_call_handlers:
    del runner._func_call_handlers[Opcode.ADD]

runner.model.cuda()

# Capture embedding output
embed_out = None

def embed_hook(module, input, output):
    global embed_out
    if embed_out is None:
        embed_out = output.detach().cpu()

h1 = runner.model.embed.register_forward_hook(embed_hook)

print("Running VM...", file=sys.stderr)
_, exit_code = runner.run(bytecode, max_steps=50)

h1.remove()

print(f"Exit code: {exit_code}\n", file=sys.stderr)

if embed_out is not None:
    seq_len = embed_out.shape[1]
    print(f"Sequence length: {seq_len} tokens\n", file=sys.stderr)

    # Check if IS_MARK is set anywhere
    is_mark_positions = []
    for pos in range(min(seq_len, 200)):
        is_mark = embed_out[0, pos, BD.IS_MARK].item()
        if is_mark > 0.5:
            is_mark_positions.append(pos)

    print(f"Positions with IS_MARK=1: {len(is_mark_positions)}", file=sys.stderr)

    # Always show all tokens to debug
    print("\nAll tokens in sequence:", file=sys.stderr)
    for pos in range(min(seq_len, 60)):  # Show first 60
        # Check which flags are set
        flags = []
        if embed_out[0, pos, BD.MARK_PC].item() > 0.5:
            flags.append("MARK_PC")
        if embed_out[0, pos, BD.MARK_AX].item() > 0.5:
            flags.append("MARK_AX")
        if embed_out[0, pos, BD.MARK_SP].item() > 0.5:
            flags.append("MARK_SP")
        if embed_out[0, pos, BD.MARK_BP].item() > 0.5:
            flags.append("MARK_BP")
        if embed_out[0, pos, BD.MARK_MEM].item() > 0.5:
            flags.append("MARK_MEM")
        if embed_out[0, pos, BD.MARK_SE].item() > 0.5:
            flags.append("MARK_SE")
        if embed_out[0, pos, BD.IS_BYTE].item() > 0.5:
            flags.append("IS_BYTE")
        if embed_out[0, pos, BD.IS_MARK].item() > 0.5:
            flags.append("IS_MARK")

        if pos < 20 or flags:  # Always show first 20, then only non-empty
            print(f"  Position {pos}: {flags if flags else '[none]'}", file=sys.stderr)

    print("", file=sys.stderr)

    if len(is_mark_positions) > 0:
        print(f"✓ IS_MARK set at {len(is_mark_positions)} positions", file=sys.stderr)

        # Show first few IS_MARK positions
        print("\nFirst few IS_MARK positions:", file=sys.stderr)
        for pos in is_mark_positions[:10]:
            # Check which marker it is
            marker_name = "UNKNOWN"
            if embed_out[0, pos, BD.MARK_PC].item() > 0.5:
                marker_name = "PC"
            elif embed_out[0, pos, BD.MARK_AX].item() > 0.5:
                marker_name = "AX"
            elif embed_out[0, pos, BD.MARK_SP].item() > 0.5:
                marker_name = "SP"
            elif embed_out[0, pos, BD.MARK_BP].item() > 0.5:
                marker_name = "BP"
            elif embed_out[0, pos, BD.MARK_MEM].item() > 0.5:
                marker_name = "MEM"
            elif embed_out[0, pos, BD.MARK_SE].item() > 0.5:
                marker_name = "SE"

            print(f"  Position {pos}: {marker_name} marker", file=sys.stderr)

print("", file=sys.stderr)
print("=" * 70, file=sys.stderr)
print("DIAGNOSIS", file=sys.stderr)
print("=" * 70, file=sys.stderr)

if embed_out is not None and len(is_mark_positions) == 0:
    print("❌ IS_MARK flag not set in embeddings", file=sys.stderr)
    print("", file=sys.stderr)
    print("Impact:", file=sys.stderr)
    print("  - Threshold attention uses W_k[base, IS_MARK] = threshold", file=sys.stderr)
    print("  - Without IS_MARK, attention scores are zero", file=sys.stderr)
    print("  - L1H0/L1H1/L1H2 outputs stay zero", file=sys.stderr)
    print("  - Layer 3 cannot identify byte positions", file=sys.stderr)
    print("  - AX carry-forward broken", file=sys.stderr)
    print("", file=sys.stderr)
    print("Fix: Check embedding initialization in vm_step.py", file=sys.stderr)
    print("     IS_MARK should be set for REG_PC, REG_AX, REG_SP, REG_BP, MEM, etc.", file=sys.stderr)
elif embed_out is not None:
    print("✓ IS_MARK is set correctly", file=sys.stderr)
    print("  Issue must be elsewhere in threshold attention mechanism", file=sys.stderr)

print("=" * 70, file=sys.stderr)

del runner
torch.cuda.empty_cache()
