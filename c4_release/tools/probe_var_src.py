#!/usr/bin/env python3
"""Locate the SOURCE of the block-29 explosion in dims 69/79/85 for var_simple_12.

After block 28 (pre-explosion), scan all positions for large values in the
H2/H3 dims and the AX/MEM region. Then after block 29, confirm explosion is
localized to the BP-byte0 prediction row. Determines attention-gather vs
FFN-local amplification.
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
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
ctx = probe._final_context(bc)
pl = len(probe._build_context(bc))
dev = next(probe.model.parameters()).device
i = pl
while ctx[i] != int(Token.REG_BP): i += 1
bp_idx = i
read_pos = bp_idx  # produces BP byte0
padded = torch.tensor([ctx], dtype=torch.long, device=dev)

WATCH = [69, 79, 85, 70, 71, 72, 76, 77, 78]

with torch.no_grad():
    r28 = probe.model.forward(padded, stop_after_block=28)[0].float()  # [S,D]
    r29 = probe.model.forward(padded, stop_after_block=29)[0].float()

print(f"pl={pl} bp_idx={bp_idx} read_pos={read_pos} S={r28.shape[0]}", flush=True)

# Max abs over all positions in WATCH dims after block 28
print("\n=== after block 28 (PRE-explosion): max|val| per watched dim over all positions ===", flush=True)
for d in WATCH:
    col = r28[:, d]
    mp = int(col.abs().argmax().item())
    print(f"  dim {d:3d}: max|val|={col.abs().max().item():.3f} at pos {mp} "
          f"(val={float(col[mp]):.3f})  @read_pos={float(r28[read_pos,d]):.3f}", flush=True)

print("\n=== after block 29 (POST-explosion): same ===", flush=True)
for d in WATCH:
    col = r29[:, d]
    mp = int(col.abs().argmax().item())
    print(f"  dim {d:3d}: max|val|={col.abs().max().item():.3f} at pos {mp} "
          f"(val={float(col[mp]):.3f})  @read_pos={float(r29[read_pos,d]):.3f}", flush=True)

# Is the explosion ONLY at read_pos, or many positions?
print("\n=== block-29 |residual| total per position (top 8) ===", flush=True)
mags = r29.abs().max(dim=1).values  # [S]
top = torch.topk(mags, 8)
for v, pidx in zip(top.values.tolist(), top.indices.tolist()):
    pidx = int(pidx)
    tok = ctx[pidx] if pidx < len(ctx) else -1
    print(f"  pos {pidx:3d} (tok {tok}): max|dim|={v:.1f}", flush=True)

# decompose block 29: is it attn or ffn? Read block 28 output, run block 29
# attn only vs full. We approximate by checking if r29 explosion magnitude
# matches an attention copy of some pos in r28 (it won't if FFN-generated).
print("\n=== block 28 global max|val| (any pos,dim) ===", flush=True)
mx = r28.abs().max().item()
where = (r28.abs() == r28.abs().max()).nonzero()[0].tolist()
print(f"  {mx:.3f} at (pos,dim)={where}", flush=True)
print("=== block 29 global max|val| ===", flush=True)
mx = r29.abs().max().item()
where = (r29.abs() == r29.abs().max()).nonzero()[0].tolist()
print(f"  {mx:.3f} at (pos,dim)={where}", flush=True)
