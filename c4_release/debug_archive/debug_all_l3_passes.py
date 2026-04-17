"""Check ALL Layer 3 attention passes to find when AX_CARRY is set."""

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Opcode, _SetDim, Token
import torch
import sys

BD = _SetDim

code = "int main() { return 10 + 32; }"
bytecode, data = compile_c(code)

print("=" * 70, file=sys.stderr)
print("CHECKING ALL LAYER 3 ATTENTION PASSES", file=sys.stderr)
print("=" * 70, file=sys.stderr)

runner = AutoregressiveVMRunner()
if Opcode.ADD in runner._func_call_handlers:
    del runner._func_call_handlers[Opcode.ADD]

runner.model.cuda()

# Capture ALL Layer 3 attention outputs
l3_attn_passes = []
step_count = [0]

def count_steps(module, input, output):
    if len(input) > 0 and input[0].shape[1] > 0:
        last_tok = input[0][0, -1].item()
        if last_tok == Token.STEP_END:
            step_count[0] += 1

def attn_hook(module, input, output):
    l3_attn_passes.append((step_count[0], output.detach().cpu()))

h_embed = runner.model.embed.register_forward_hook(count_steps)
h_attn = runner.model.blocks[3].attn.register_forward_hook(attn_hook)

print("Running VM (4 steps to reach ADD)...", file=sys.stderr)
_, exit_code = runner.run(bytecode, max_steps=5)

h_embed.remove()
h_attn.remove()

print(f"Exit code: {exit_code}", file=sys.stderr)
print(f"Total Layer 3 attention passes: {len(l3_attn_passes)}", file=sys.stderr)
print(f"Steps completed: {step_count[0]}\n", file=sys.stderr)

# Find passes where AX_CARRY is set
print("Scanning for AX_CARRY_LO activity:", file=sys.stderr)
print("-" * 70, file=sys.stderr)

passes_with_carry = []
for pass_idx, (step_num, attn_out) in enumerate(l3_attn_passes):
    seq_len = attn_out.shape[1]
    ax_carry_lo = attn_out[0, -1, BD.AX_CARRY_LO:BD.AX_CARRY_LO+16]
    max_lo = ax_carry_lo.max().item()

    if max_lo > 0.5:
        passes_with_carry.append((pass_idx, step_num, max_lo, seq_len))

print(f"Found {len(passes_with_carry)} passes with AX_CARRY_LO > 0.5", file=sys.stderr)

if len(passes_with_carry) > 0:
    print("\nPasses with AX_CARRY set:", file=sys.stderr)
    for pass_idx, step_num, max_val, seq_len in passes_with_carry[:10]:
        print(f"  Pass {pass_idx} (step {step_num}, seq_len={seq_len}): AX_CARRY_LO max={max_val:.3f}", file=sys.stderr)
else:
    print("\n❌ NO passes have AX_CARRY_LO set!", file=sys.stderr)
    print("   Layer 3 attention is not populating AX_CARRY at all.", file=sys.stderr)

    # Check if the attention weights are configured
    attn3 = runner.model.blocks[3].attn
    head1_base = 64  # head 1 is for AX carry-forward

    AX_I = 1
    wk_l1h1 = attn3.W_k[head1_base, BD.L1H1 + AX_I].item()
    wk_l1h0 = attn3.W_k[head1_base, BD.L1H0 + AX_I].item()
    wq_mark_ax = attn3.W_q[head1_base, BD.MARK_AX].item()

    print(f"\n  Layer 3 Head 1 configuration:", file=sys.stderr)
    print(f"    W_q[{head1_base}, MARK_AX={BD.MARK_AX}] = {wq_mark_ax:.2f}", file=sys.stderr)
    print(f"    W_k[{head1_base}, L1H1+AX={BD.L1H1 + AX_I}] = {wk_l1h1:.2f}", file=sys.stderr)
    print(f"    W_k[{head1_base}, L1H0+AX={BD.L1H0 + AX_I}] = {wk_l1h0:.2f}", file=sys.stderr)

    if abs(wk_l1h1) < 0.1:
        print(f"\n  ❌ W_k[L1H1] not configured!", file=sys.stderr)
    else:
        print(f"\n  ✓ Weights configured correctly", file=sys.stderr)
        print(f"    But attention is not producing AX_CARRY output.", file=sys.stderr)
        print(f"    This means L1H1/L1H0 inputs are not reaching Layer 3.", file=sys.stderr)

print("\n" + "=" * 70, file=sys.stderr)

del runner
torch.cuda.empty_cache()
