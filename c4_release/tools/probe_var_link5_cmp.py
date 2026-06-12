#!/usr/bin/env python3
"""Compare the step-3 SP byte3 prediction row (pred_row=212, BUG: emits BP
marker 260 instead of byte 0x00) against the step-4 SP byte3 prediction row
(pred_row=245, CORRECT: emits byte 0x00). Find the dim that flips the
marker-vs-byte decision, and walk per-block to find the genesis block/op.

READ-ONLY. spec_k=0, hook-free."""
from __future__ import annotations
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import torch  # noqa
from tools.probe_groundtruth import build_groundtruth_probe  # noqa
from tests.test_suite_1000 import generate_test_programs  # noqa
from src.compiler import compile_c  # noqa

probe = build_groundtruth_probe()
m = probe.model
dp = m.dim_positions
dev = next(m.parameters()).device
bl = probe.block_layer_map()
rev = {}
for n, d in dp.items():
    if isinstance(d, int):
        rev.setdefault(d, []).append(n)

tests = generate_test_programs()
src, exp, desc = tests[262]
bc, data = compile_c(src)
ctx = probe._final_context(bc)
padded = torch.tensor([ctx], dtype=torch.long, device=dev)

BUG = 212   # step3 SP byte3 -> emits BP marker (260)
GOOD = 245  # step4-dump1 SP byte3 -> emits byte 0x00

with torch.no_grad():
    blocks = {}
    for phys in range(len(m.blocks)):
        r = m.forward(padded, stop_after_block=phys)[0].float()
        blocks[phys] = (r[BUG].clone(), r[GOOD].clone())
    final = m.forward(padded)[0].float()
    # also the FULL residual at the input (block -1) = embedding
    emb = m.forward(padded, stop_after_block=0)[0].float()

# final residual diff
fb = m.forward(padded, stop_after_block=len(m.blocks) - 1)[0].float()
dbug = fb[BUG]; dgood = fb[GOOD]
diff = (dbug - dgood)
idx = torch.topk(diff.abs(), 25).indices.tolist()
print("=== FINAL residual diff (BUG row 212 - GOOD row 245), top 25 |dim| ===")
for d in idx:
    print(f"  {rev.get(d,[f'd{d}'])[0]:>22} (d{d}): bug={float(dbug[d]):10.3f} "
          f"good={float(dgood[d]):10.3f} diff={float(diff[d]):10.3f}")

# Per-block: track the marker-deciding dims. Print IS_BYTE/IS_MARK + the dims
# from the final-diff top set, when they change between BUG and GOOD.
track = ["IS_BYTE", "IS_MARK", "MARK_SP", "MARK_BP", "BYTE_INDEX_0",
         "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3"]
track = [t for t in track if t in dp]
print("\n=== per-block IS_BYTE/IS_MARK/byte-index for BUG vs GOOD ===")
print(f"{'blk':>3}{'L':>3} | " + " ".join(f"{t[:9]:>9}" for t in track))
prev = None
for phys in range(len(m.blocks)):
    rb, rg = blocks[phys]
    vals = []
    for t in track:
        d = dp[t]
        vals.append(f"{float(rb[d]):4.1f}/{float(rg[d]):4.1f}")
    cur = " ".join(vals)
    if cur != prev:
        print(f"{phys:>3}{bl[phys]['logical']:>3} | " + " ".join(
            f"{v:>9}" for v in vals))
        prev = cur

# Which block first makes the BUG row's marker-token (260=BP) logit exceed the
# byte-token (0) logit? Track head readout per block via final head on partial.
print("\n=== per-block LM argmax(BUG row 212) (marker 260 vs byte 0) ===")
head = m.head
prevd = None
with torch.no_grad():
    for phys in range(len(m.blocks)):
        r = m.forward(padded, stop_after_block=phys)[0].float()
        lg = head(r[BUG].unsqueeze(0)).squeeze(0).float()
        am = int(lg.argmax())
        l260 = float(lg[260]); l0 = float(lg[0])
        cur = (am, round(l260, 1), round(l0, 1))
        if cur != prevd:
            print(f"  blk{phys:2d}/L{bl[phys]['logical']:2d}: argmax={am} "
                  f"L[260BP]={l260:.2f} L[0byte]={l0:.2f}")
            prevd = cur
