#!/usr/bin/env python3
"""Find WHICH physical block injects the 0x0a flood into BP byte0 (var_simple_12).

Builds the final spec_k=0 context once, then runs ONE forward per block with
stop_after_block and reads the full [D] residual at the BP-byte0 prediction
row. Reports, per block, the top-magnitude residual dims and the running
delta, to pin the block where the giant value first appears.
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import torch  # noqa
from tools.probe_groundtruth import build_groundtruth_probe  # noqa
from neural_vm.batched_pure_neural import Token  # noqa
from tests.test_suite_1000 import generate_test_programs  # noqa
from src.compiler import compile_c  # noqa

probe = build_groundtruth_probe()
tests = generate_test_programs()
src, exp, _ = tests[262]
bc, _ = compile_c(src)
ctx = probe._final_context(bc)          # build ONCE
pl = len(probe._build_context(bc))
dev = next(probe.model.parameters()).device

# step-0 REG_BP marker
i = pl
while ctx[i] != int(Token.REG_BP):
    i += 1
bp_idx = i
# BP byte0 token is at ctx[bp_idx+1]; the row that PRODUCES it is position
# bp_idx (forward over full ctx, read resid[0, bp_idx], LM head of that row
# predicts ctx[bp_idx+1]).
read_pos = bp_idx
print(f"pl={pl} bp_idx={bp_idx} read_pos={read_pos} "
      f"(produces BP byte0 = ctx[{bp_idx+1}]=0x{ctx[bp_idx+1]:02x})", flush=True)

padded = torch.tensor([ctx], dtype=torch.long, device=dev)
bl = probe.block_layer_map()
prev = None
nblocks = len(probe.model.blocks)
print(f"\n{'phys':>4} {'log':>3} {'max|dim|':>10} {'argmax_dim':>10} "
      f"{'val':>14}  {'biggest_delta_dim':>16} {'delta':>14}", flush=True)
with torch.no_grad():
    for phys in range(nblocks):
        resid = probe.model.forward(padded, stop_after_block=phys)  # [1,S,D]
        row = resid[0, read_pos].float()  # [D]
        amax = int(row.abs().argmax().item())
        amax_v = float(row[amax].item())
        if prev is None:
            delta_dim, delta_v = amax, amax_v
        else:
            d = (row - prev)
            delta_dim = int(d.abs().argmax().item())
            delta_v = float(d[delta_dim].item())
        log = bl[phys]["logical"]
        print(f"{phys:>4} {log:>3} {row.abs().max().item():>10.1f} "
              f"{amax:>10} {amax_v:>14.2f}  {delta_dim:>16} {delta_v:>14.2f}",
              flush=True)
        prev = row.clone()

# Final: which dims are huge after the last block?
print("\n=== top-8 |dim| at final block (block {}) ===".format(nblocks-1),
      flush=True)
row = prev
top = torch.topk(row.abs(), 8)
for v, d in zip(top.values.tolist(), top.indices.tolist()):
    print(f"  dim {int(d):4d}  val={float(row[int(d)]):.3f}", flush=True)
