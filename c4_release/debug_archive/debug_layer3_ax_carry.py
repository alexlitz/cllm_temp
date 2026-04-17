"""Debug Layer 3 attention to see why AX_CARRY_LO is not being populated."""

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Opcode, _SetDim
import torch
import sys

BD = _SetDim

code = "int main() { return 10 + 32; }"
bytecode, data = compile_c(code)

print("=" * 70, file=sys.stderr)
print("DEBUGGING LAYER 3 AX CARRY", file=sys.stderr)
print("=" * 70, file=sys.stderr)
print(f"Test: {code}", file=sys.stderr)
print("Expected: Layer 3 should carry AX value to AX_CARRY_LO", file=sys.stderr)
print("", file=sys.stderr)

runner = AutoregressiveVMRunner()
if Opcode.ADD in runner._func_call_handlers:
    del runner._func_call_handlers[Opcode.ADD]
    print("✓ Disabled ADD handler", file=sys.stderr)

runner.model.cuda()

# Capture embeddings and Layer 3 output
embed_out = None
l3_attn_out = None
l3_ffn_out = None

def embed_hook(module, input, output):
    global embed_out
    if embed_out is None:
        embed_out = output.detach().cpu()

def l3_attn_hook(module, input, output):
    global l3_attn_out
    if l3_attn_out is None:
        l3_attn_out = output.detach().cpu()

def l3_ffn_hook(module, input, output):
    global l3_ffn_out
    if l3_ffn_out is None:
        l3_ffn_out = output.detach().cpu()

h1 = runner.model.embed.register_forward_hook(embed_hook)
h2 = runner.model.blocks[3].attn.register_forward_hook(l3_attn_hook)
h3 = runner.model.blocks[3].ffn.register_forward_hook(l3_ffn_hook)

print("Running VM...", file=sys.stderr)
_, exit_code = runner.run(bytecode, max_steps=50)

h1.remove()
h2.remove()
h3.remove()

print(f"Exit code: {exit_code}\n", file=sys.stderr)

# Analyze Layer 3 behavior
if embed_out is not None and l3_attn_out is not None:
    seq_len = embed_out.shape[1]
    print(f"Sequence length: {seq_len} tokens", file=sys.stderr)
    print("", file=sys.stderr)

    # Find AX markers in embedding
    print("AX markers in embedding:", file=sys.stderr)
    ax_positions = []
    for pos in range(seq_len):
        if embed_out[0, pos, BD.MARK_AX].item() > 0.5:
            ax_positions.append(pos)
            # Get AX value at this position
            ax_lo_vals = []
            for i in range(16):
                if embed_out[0, pos + 1, BD.EMBED_LO + i].item() > 0.5:  # Next token has value
                    ax_lo_vals.append(i)
            print(f"  Position {pos}: AX marker (next token AX_LO = {ax_lo_vals})", file=sys.stderr)

    print("", file=sys.stderr)

    # Check what Layer 3 attention does at these positions
    print("Layer 3 attention output at AX positions:", file=sys.stderr)
    for pos in ax_positions:
        if pos < l3_attn_out.shape[1]:
            # Check AX_CARRY_LO at this position
            ax_carry_vals = []
            for i in range(16):
                val = l3_attn_out[0, pos, BD.AX_CARRY_LO + i].item()
                if val > 0.1:
                    ax_carry_vals.append((i, val))

            print(f"  Position {pos}:", file=sys.stderr)
            print(f"    AX_CARRY_LO: {ax_carry_vals if ax_carry_vals else 'NONE'}", file=sys.stderr)

            # Also check if MARK_AX is preserved
            mark_ax = l3_attn_out[0, pos, BD.MARK_AX].item()
            print(f"    MARK_AX: {mark_ax:.3f}", file=sys.stderr)

    print("", file=sys.stderr)

    # Check Layer 3 FFN output
    print("Layer 3 FFN output at AX positions:", file=sys.stderr)
    if l3_ffn_out is not None:
        for pos in ax_positions:
            if pos < l3_ffn_out.shape[1]:
                ax_carry_vals = []
                for i in range(16):
                    val = l3_ffn_out[0, pos, BD.AX_CARRY_LO + i].item()
                    if val > 0.1:
                        ax_carry_vals.append((i, val))

                print(f"  Position {pos}:", file=sys.stderr)
                print(f"    AX_CARRY_LO: {ax_carry_vals if ax_carry_vals else 'NONE'}", file=sys.stderr)

                mark_ax = l3_ffn_out[0, pos, BD.MARK_AX].item()
                print(f"    MARK_AX: {mark_ax:.3f}", file=sys.stderr)

    print("", file=sys.stderr)

    # Now check subsequent steps - where does AX_CARRY_LO appear?
    print("Scanning all positions for AX_CARRY_LO activity:", file=sys.stderr)
    positions_with_ax_carry = []
    if l3_attn_out is not None:
        for pos in range(min(seq_len, 200)):
            max_ax_carry = l3_attn_out[0, pos, BD.AX_CARRY_LO:BD.AX_CARRY_LO+16].max().item()
            if max_ax_carry > 0.1:
                positions_with_ax_carry.append((pos, max_ax_carry))

    if positions_with_ax_carry:
        print(f"  Found AX_CARRY_LO at {len(positions_with_ax_carry)} positions:", file=sys.stderr)
        for pos, val in positions_with_ax_carry[:10]:
            print(f"    Position {pos}: {val:.3f}", file=sys.stderr)
    else:
        print(f"  ⚠ WARNING: No AX_CARRY_LO found anywhere!", file=sys.stderr)
        print(f"  This means Layer 3 attention is not carrying AX values forward.", file=sys.stderr)

print("", file=sys.stderr)
print("=" * 70, file=sys.stderr)
print("DIAGNOSIS", file=sys.stderr)
print("=" * 70, file=sys.stderr)

if not positions_with_ax_carry:
    print("❌ ISSUE FOUND: Layer 3 attention is not populating AX_CARRY_LO", file=sys.stderr)
    print("", file=sys.stderr)
    print("Expected behavior:", file=sys.stderr)
    print("  - Layer 3 attention should look back at previous AX markers", file=sys.stderr)
    print("  - Copy AX value to AX_CARRY_LO dimensions", file=sys.stderr)
    print("  - This makes AX available for ALU operations", file=sys.stderr)
    print("", file=sys.stderr)
    print("Actual behavior:", file=sys.stderr)
    print("  - AX_CARRY_LO is empty (all zeros)", file=sys.stderr)
    print("  - ADD operation cannot access second operand", file=sys.stderr)
    print("  - Returns first operand only (from ALU_LO)", file=sys.stderr)
else:
    print("✓ Layer 3 is populating AX_CARRY_LO at some positions", file=sys.stderr)
    print("Further investigation needed to see if it's at the right positions", file=sys.stderr)

print("=" * 70, file=sys.stderr)

del runner
torch.cuda.empty_cache()
