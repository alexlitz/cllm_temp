"""Find which layer loses AX_CARRY_LO/HI values."""

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Opcode, _SetDim
import torch
import sys

BD = _SetDim

code = "int main() { return 10 + 32; }"
bytecode, data = compile_c(code)

print("=" * 70, file=sys.stderr)
print("FINDING WHERE AX_CARRY IS LOST", file=sys.stderr)
print("=" * 70, file=sys.stderr)

runner = AutoregressiveVMRunner()
if Opcode.ADD in runner._func_call_handlers:
    del runner._func_call_handlers[Opcode.ADD]

runner.model.cuda()

# Capture output from each layer
layer_outputs = {}

def make_hook(layer_idx):
    def hook(module, input, output):
        layer_outputs[layer_idx] = output.detach().cpu()
    return hook

# Hook Layers 3-7 (after Layer 3 sets AX_CARRY, before Layer 7 needs it)
hooks = []
for i in range(3, 8):
    h = runner.model.blocks[i].register_forward_hook(make_hook(i))
    hooks.append(h)

print("Running VM...", file=sys.stderr)
_, exit_code = runner.run(bytecode, max_steps=10)

for h in hooks:
    h.remove()

print(f"Exit code: {exit_code}\n", file=sys.stderr)

# Check AX_CARRY_LO max value after each layer
print("AX_CARRY_LO progression through layers:", file=sys.stderr)
print("-" * 70, file=sys.stderr)

for layer_idx in range(3, 8):
    if layer_idx in layer_outputs:
        out = layer_outputs[layer_idx]
        seq_len = out.shape[1]

        # Get AX_CARRY_LO at last position
        ax_carry_lo = out[0, -1, BD.AX_CARRY_LO:BD.AX_CARRY_LO+16]
        ax_carry_hi = out[0, -1, BD.AX_CARRY_HI:BD.AX_CARRY_HI+16]

        max_lo = ax_carry_lo.max().item()
        max_hi = ax_carry_hi.max().item()

        print(f"  Layer {layer_idx}: AX_CARRY_LO max={max_lo:.3f}, AX_CARRY_HI max={max_hi:.3f}", file=sys.stderr)

        if max_lo < 0.1:
            print(f"             ^ ❌ LOST! (was non-zero before this layer)", file=sys.stderr)

print("", file=sys.stderr)
print("-" * 70, file=sys.stderr)

# Find the transition
print("\nDiagnosis:", file=sys.stderr)
prev_max = None
for layer_idx in range(3, 8):
    if layer_idx in layer_outputs:
        out = layer_outputs[layer_idx]
        ax_carry_lo = out[0, -1, BD.AX_CARRY_LO:BD.AX_CARRY_LO+16]
        max_lo = ax_carry_lo.max().item()

        if prev_max is not None:
            if prev_max > 0.5 and max_lo < 0.1:
                print(f"  AX_CARRY lost between Layer {layer_idx-1} and Layer {layer_idx}", file=sys.stderr)
                print(f"  This means Layer {layer_idx} is overwriting AX_CARRY dimensions!", file=sys.stderr)
                break

        prev_max = max_lo

print("\n" + "=" * 70, file=sys.stderr)

del runner
torch.cuda.empty_cache()
