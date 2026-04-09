"""Debug Layer 1 threshold attention mechanism in detail."""

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Opcode, Token, _SetDim
import torch
import sys

BD = _SetDim

code = "int main() { return 10 + 32; }"
bytecode, data = compile_c(code)

print("=" * 70, file=sys.stderr)
print("DEBUGGING LAYER 1 THRESHOLD MECHANISM", file=sys.stderr)
print("=" * 70, file=sys.stderr)

runner = AutoregressiveVMRunner()
if Opcode.ADD in runner._func_call_handlers:
    del runner._func_call_handlers[Opcode.ADD]

runner.model.cuda()

# Capture Layer 1 attention scores and outputs
l1_attn_scores = []
l1_attn_out = []
l1_input = []

def l1_input_hook(module, input, output):
    if len(input) > 0:
        l1_input.append(input[0].detach().cpu())

def l1_attn_hook(module, input, output):
    l1_attn_out.append(output.detach().cpu())

# Hook Layer 1 attention
h_in = runner.model.blocks[1].register_forward_hook(l1_input_hook)
h_attn = runner.model.blocks[1].attn.register_forward_hook(l1_attn_hook)

print("Running VM (max 1 step)...", file=sys.stderr)
_, exit_code = runner.run(bytecode, max_steps=1)

h_in.remove()
h_attn.remove()

print(f"Exit code: {exit_code}", file=sys.stderr)
print(f"Layer 1 forward passes: {len(l1_attn_out)}\n", file=sys.stderr)

# Get Layer 1 attention
attn1 = runner.model.blocks[1].attn

# Check weight configuration
print("Layer 1 Attention Configuration:", file=sys.stderr)
print(f"  Number of heads: {attn1.num_heads}", file=sys.stderr)
print(f"  Head dim: {attn1.head_dim}", file=sys.stderr)
if hasattr(attn1, 'alibi_slopes') and attn1.alibi_slopes is not None:
    print(f"  ALiBi slopes: {attn1.alibi_slopes.cpu().numpy()}", file=sys.stderr)
else:
    print(f"  Positional encoding: RoPE (no ALiBi slopes)", file=sys.stderr)
print("", file=sys.stderr)

# Check head 0 (threshold 0.5) and head 1 (threshold 1.5) specifically
print("Threshold Head Configuration:", file=sys.stderr)
for h, threshold in [(0, 0.5), (1, 1.5)]:
    base = h * attn1.head_dim
    wq_const = attn1.W_q[base, BD.CONST].item()
    wk_is_mark = attn1.W_k[base, BD.IS_MARK].item()
    print(f"  Head {h} (threshold {threshold}):", file=sys.stderr)
    print(f"    W_q[{base}, CONST={BD.CONST}] = {wq_const:.2f}", file=sys.stderr)
    print(f"    W_k[{base}, IS_MARK={BD.IS_MARK}] = {wk_is_mark:.2f}", file=sys.stderr)

print("", file=sys.stderr)

# Analyze the LAST attention pass (should include full generated step)
if len(l1_attn_out) > 0:
    last_out = l1_attn_out[-1]
    seq_len = last_out.shape[1]
    print(f"Last Layer 1 attention output:", file=sys.stderr)
    print(f"  Sequence length: {seq_len}", file=sys.stderr)

    # Check if L1H0 and L1H1 outputs are non-zero
    AX_I = 1  # AX index in MARKS array
    l1h0_ax_vals = last_out[0, :, BD.L1H0 + AX_I]
    l1h1_ax_vals = last_out[0, :, BD.L1H1 + AX_I]

    l1h0_max = l1h0_ax_vals.max().item()
    l1h1_max = l1h1_ax_vals.max().item()

    print(f"  Max L1H0[AX]: {l1h0_max:.3f}", file=sys.stderr)
    print(f"  Max L1H1[AX]: {l1h1_max:.3f}", file=sys.stderr)

    if l1h0_max < 0.1 and l1h1_max < 0.1:
        print("\n❌ L1H0/L1H1 outputs are zero!", file=sys.stderr)
        print("   Threshold attention is not working.", file=sys.stderr)

        # Check if IS_MARK is present in the input to Layer 1
        if len(l1_input) > 0:
            last_input = l1_input[-1]
            is_mark_vals = last_input[0, :, BD.IS_MARK]
            is_mark_count = (is_mark_vals > 0.5).sum().item()
            print(f"\n  Input to Layer 1:", file=sys.stderr)
            print(f"    IS_MARK positions: {is_mark_count} / {last_input.shape[1]}", file=sys.stderr)

            if is_mark_count == 0:
                print("    ❌ No IS_MARK in input to Layer 1!", file=sys.stderr)
            else:
                print("    ✓ IS_MARK present in input", file=sys.stderr)

                # Show where IS_MARK is set
                print("\n    Positions with IS_MARK:", file=sys.stderr)
                for pos in range(min(last_input.shape[1], 100)):
                    if is_mark_vals[pos] > 0.5:
                        const_val = last_input[0, pos, BD.CONST].item()
                        print(f"      Position {pos}: CONST={const_val:.2f}", file=sys.stderr)
    else:
        print(f"\n✓ L1H0/L1H1 ARE producing output!", file=sys.stderr)

        # Show where they're active
        print("\nPositions with L1H1[AX] > 0.5:", file=sys.stderr)
        for pos in range(min(seq_len, 100)):
            if l1h1_ax_vals[pos] > 0.5:
                l1h0_val = l1h0_ax_vals[pos].item()
                print(f"  Position {pos}: L1H1[AX]={l1h1_ax_vals[pos]:.2f}, L1H0[AX]={l1h0_val:.2f}", file=sys.stderr)

print("\n" + "=" * 70, file=sys.stderr)

del runner
torch.cuda.empty_cache()
