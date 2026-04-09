"""Check what token IDs are in the context during execution."""

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Opcode, Token
import torch
import sys

code = "int main() { return 10 + 32; }"
bytecode, data = compile_c(code)

print("=" * 70, file=sys.stderr)
print("CHECKING TOKEN IDs IN CONTEXT", file=sys.stderr)
print("=" * 70, file=sys.stderr)

runner = AutoregressiveVMRunner()
if Opcode.ADD in runner._func_call_handlers:
    del runner._func_call_handlers[Opcode.ADD]

runner.model.cuda()

# Capture input_ids
captured_ids = None

def embed_hook(module, input, output):
    global captured_ids
    if captured_ids is None and len(input) > 0:
        captured_ids = input[0].detach().cpu()

h1 = runner.model.embed.register_forward_hook(embed_hook)

print("Running VM (max 2 steps)...", file=sys.stderr)
_, exit_code = runner.run(bytecode, max_steps=2)

h1.remove()

print(f"Exit code: {exit_code}\n", file=sys.stderr)

if captured_ids is not None:
    seq_len = captured_ids.shape[1]
    print(f"Sequence length: {seq_len} tokens\n", file=sys.stderr)

    # Define marker tokens
    markers = {
        Token.REG_PC: "REG_PC",
        Token.REG_AX: "REG_AX",
        Token.REG_SP: "REG_SP",
        Token.REG_BP: "REG_BP",
        Token.MEM: "MEM",
        Token.STACK0: "STACK0",
        Token.CODE_START: "CODE_START",
        Token.STEP_END: "STEP_END",
    }

    # Check first 70 tokens
    print("First 70 tokens in sequence:", file=sys.stderr)
    marker_count = 0
    for i in range(min(70, seq_len)):
        tok_id = captured_ids[0, i].item()

        if tok_id in markers:
            print(f"  Position {i}: token {tok_id} = {markers[tok_id]} ← MARKER", file=sys.stderr)
            marker_count += 1
        elif tok_id < 256:
            print(f"  Position {i}: token {tok_id} = byte 0x{tok_id:02x}", file=sys.stderr)
        else:
            print(f"  Position {i}: token {tok_id}", file=sys.stderr)

    print(f"\nTotal marker tokens in first 70: {marker_count}", file=sys.stderr)

    # Check expected step format
    if seq_len >= 35:
        print("\nAnalyzing first step (positions 0-34):", file=sys.stderr)
        step1_markers = []
        for i in range(min(35, seq_len)):
            tok_id = captured_ids[0, i].item()
            if tok_id in markers:
                step1_markers.append((i, markers[tok_id]))

        print(f"  Markers in step 1: {len(step1_markers)}", file=sys.stderr)
        for pos, name in step1_markers:
            print(f"    Position {pos}: {name}", file=sys.stderr)

        # Expected: 6 markers (PC, AX, SP, BP, STACK0, MEM) + STEP_END = 7
        if len(step1_markers) < 6:
            print(f"\n⚠ WARNING: Expected 6-7 markers per step, found {len(step1_markers)}", file=sys.stderr)

print("\n" + "=" * 70, file=sys.stderr)

del runner
torch.cuda.empty_cache()
