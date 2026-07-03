#!/usr/bin/env python3
"""Pin the L6 (block 6) writer of dims 79/85 (H2/H3) at the REG_PC marker row.

spec_k=0, hook-free. For var_simple_12 (id 262):
  - Genesis probe says dim 79 jumps 1.0 -> 507 at block 6, pos 92 (REG_PC).
  - H2/H3 are L0-attention-output residual dims; no named L6 op writes them.
  - So either block 6 attention W_o[:, 79/85] or FFN W_down[79/85, :] is nonzero.

This probe:
  1. Reads the residual into block 6 (post block 5) at pos 92 in dims 79/85.
  2. Decomposes block 6's delta into attention vs FFN contribution.
  3. Dumps which W_o columns / W_down rows are nonzero for dims 79/85 at block 6,
     and which input dims drive them.
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
padded = torch.tensor([ctx], dtype=torch.long, device=dev)
bl = probe.block_layer_map()

POS = 92  # REG_PC marker row (genesis seed)
DIMS = [79, 85]
BLK = 6

print(f"pl={pl} S={len(ctx)} POS={POS} tok@POS={ctx[POS]} DIMS={DIMS} BLK={BLK}",
      flush=True)

# Residual before block 6 (post block 5) and after block 6.
with torch.no_grad():
    r5 = probe.model.forward(padded, stop_after_block=BLK - 1)[0].float()  # [S,D]
    r6 = probe.model.forward(padded, stop_after_block=BLK)[0].float()
for d in DIMS:
    print(f"  dim {d}: post-blk5={float(r5[POS,d]):+.4f}  "
          f"post-blk6={float(r6[POS,d]):+.4f}  "
          f"delta={float(r6[POS,d]-r5[POS,d]):+.4f}", flush=True)

# --- Decompose block6 delta into attn vs ffn by manual replay ---
# r5 is the input to block 6. Run block 6's attn then ffn manually if possible.
blk = probe.model.blocks[BLK]
with torch.no_grad():
    x_in = r5.unsqueeze(0)  # [1,S,D]
    try:
        a_out = blk.attn(x_in)  # includes residual add
        for d in DIMS:
            print(f"  [decomp] dim {d}: after attn(+resid)={float(a_out[0,POS,d]):+.4f}",
                  flush=True)
        f_out = blk.ffn(a_out)
        for d in DIMS:
            print(f"  [decomp] dim {d}: after ffn(+resid)={float(f_out[0,POS,d]):+.4f}",
                  flush=True)
    except Exception as e:
        print(f"  [decomp] manual replay failed: {e}", flush=True)

print(f"\nblock {BLK} type={type(blk).__name__}", flush=True)
print(f"  attrs: {[a for a in dir(blk) if not a.startswith('_') and a in ('attn','ffn','attention')]}", flush=True)

# --- Attention contribution: W_o columns hitting dims 79/85 ---
def _dense(t):
    if t.is_sparse or t.layout in (torch.sparse_csr, torch.sparse_coo):
        return t.to_dense().float()
    return t.float()

attn = getattr(blk, "attn", None) or getattr(blk, "attention", None)
if attn is not None and hasattr(attn, "W_o"):
    Wo = _dense(attn.W_o.data)  # [dim_out, dim_in_concat]
    print(f"\n[ATTN] W_o shape={tuple(Wo.shape)}", flush=True)
    for d in DIMS:
        row = Wo[d]  # how dim d is written from concat(head outputs)
        nz = (row.abs() > 1e-9).nonzero().flatten().tolist()
        print(f"  W_o[dim={d}] nonzero src-slots: {nz[:20]} "
              f"(vals {[round(float(row[i]),3) for i in nz[:20]]})", flush=True)
else:
    print("\n[ATTN] no W_o on block", flush=True)

# --- FFN contribution: W_down rows hitting dims 79/85 ---
ffn = getattr(blk, "ffn", None)
if ffn is not None and hasattr(ffn, "W_down"):
    Wd = _dense(ffn.W_down.data)  # [dim_out, hidden]
    Wu = _dense(ffn.W_up.data)    # [hidden, dim_in]
    print(f"\n[FFN] W_down shape={tuple(Wd.shape)} W_up shape={tuple(Wu.shape)}", flush=True)
    for d in DIMS:
        col = Wd[d]  # which hidden units write dim d
        hu = (col.abs() > 1e-9).nonzero().flatten().tolist()
        print(f"  W_down[dim={d}] driven by {len(hu)} hidden units: {hu[:30]}", flush=True)
        for h in hu[:12]:
            # which input dims drive hidden unit h
            urow = Wu[h]
            src = (urow.abs() > 1e-6).nonzero().flatten().tolist()
            print(f"    hidden {h}: W_down={float(col[h]):+.3f}  "
                  f"<- W_up src dims {src[:12]} "
                  f"vals {[round(float(urow[i]),2) for i in src[:12]]}", flush=True)
else:
    print("\n[FFN] no W_down on block", flush=True)
