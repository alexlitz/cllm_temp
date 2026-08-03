"""Does block i's FFN read dims that block j<i's FFN WRITES? If not, blocks are
independent and can be fused into one grid. If yes, sequential."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
import torch
from c4_min.compact_alloc import build_compact_sparse_streaming
from c4_min.sparse_coo_spmm import _dense_of

dev = torch.device("cuda:0")
torch.cuda.set_device(dev); torch.zeros(1).to(dev)
model, L, _ = build_compact_sparse_streaming(code_size=44, compute_mode="sparse_mm")
model.to(str(dev))
D = model.dim

# For each block, which residual dims does W_up/W_gate READ (input cols),
# and which residual dims does W_down WRITE (output rows)?
read_dims = []
write_dims = []
for b in model.blocks:
    ffn = b.ffn
    if getattr(ffn, "W_up", None) is None:
        read_dims.append(set()); write_dims.append(set()); continue
    Wu = _dense_of(ffn.W_up); Wg = _dense_of(ffn.W_gate); Wd = _dense_of(ffn.W_down)
    rd = set(torch.nonzero(Wu.abs().sum(0)).flatten().tolist()) | set(torch.nonzero(Wg.abs().sum(0)).flatten().tolist())
    wr = set(torch.nonzero(Wd.abs().sum(1)).flatten().tolist())  # output residual dims (rows of down are Dff, cols mapped... wait)
    # W_down: [D, Dff], row=output residual dim, col=hidden. writes to residual dims = rows with any nonzero
    wr = set(torch.nonzero((Wd != 0).any(dim=1)).flatten().tolist())
    read_dims.append(rd); write_dims.append(wr)

# Check: does block i read any dim that some EARLIER block j<i writes?
# (residual is additive; a write by j changes the value block i reads)
deps = 0
indep_blocks = 0
cumulative_writes = set()
for i in range(len(model.blocks)):
    if read_dims[i] & cumulative_writes:
        deps += 1
    else:
        indep_blocks += 1
    cumulative_writes |= write_dims[i]
print(f"blocks that READ a dim an EARLIER block WROTE (sequential-dep): {deps}")
print(f"blocks independent of all earlier writes: {indep_blocks}")

# The 3 live-attn blocks also mix dims via attention. But the big question:
# can we GROUP blocks into independent waves? Build a dependency: block i depends
# on block j (j<i) if read_dims[i] intersects write_dims[j].
import numpy as np
n = len(model.blocks)
# Greedy level assignment (longest-path layering)
level = [0]*n
for i in range(n):
    lv = 0
    for j in range(i):
        if read_dims[i] & write_dims[j]:
            lv = max(lv, level[j]+1)
    level[i] = lv
from collections import Counter
c = Counter(level)
print(f"max dependency level = {max(level)}  (n levels = {max(level)+1})")
print(f"level sizes (blocks per parallel wave): {sorted(c.items())[:20]}")
print(f"blocks in largest wave: {max(c.values())}")
