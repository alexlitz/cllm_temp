#!/usr/bin/env python3
"""Inc-3 ROOT B: per-hidden-unit attribution of the blk34 (L20) OUTPUT
explosion at the BP[0]-predictor row in the 30-tok campaign config.

blk34 turns OUTPUT_LO+15 from +42 into -1.29e7. This captures the resid AFTER
blk33 (= input to blk34's FFN), runs blk34.ffn's silu(up)*gate by hand, and
ranks the hidden units by their |W_down[OUTPUT_LO+15, u] * hidden[u]|
contribution. Prints the top contributors so we can name the owning rule.

Run in CAMPAIGN:
  C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 python tools/probe_inc3_bp_blk34_units.py
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_TEST_SPEC_K"] = "0"
os.environ["C4_SKIP_DIM_INTEGRITY"] = "1"
os.environ["C4_SKIP_GATE_CHECK"] = "1"
import warnings
warnings.filterwarnings("ignore")
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from src.compiler import compile_c  # noqa: E402
from neural_vm.batched_pure_neural import Token  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402

SRC = "int main() { int x; x = 28; return x; }"
BLK = int(os.environ.get("PROBE_BLK", "34"))
OUTDIM = os.environ.get("PROBE_OUTDIM", "OUTPUT_LO+15")


def main():
    bytecode, _ = compile_c(SRC)
    p = build_groundtruth_probe()
    from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic)
    _m, _l = compile_full_vm_dynamic(disk_cache=True)
    dp = dict(_l.dim_positions)

    def D(n):
        if "+" in n:
            b, o = n.rsplit("+", 1)
            return dp[b] + int(o)
        return dp[n]

    od = D(OUTDIM)
    STEP = int(Token.STEP_TOKENS)
    pl = len(p._build_context(bytecode))
    bp0 = pl + 6 * STEP + 16
    ctx = p._final_context(bytecode, max_steps=10)
    padded = torch.tensor([ctx], dtype=torch.long, device=p._device)
    # resid INTO blk BLK = output of block BLK-1
    with torch.no_grad():
        xin = p.model.forward(padded, stop_after_block=BLK - 1)[0, bp0]
    blk = p.model.blocks[BLK]
    ffn = blk.ffn

    def dense(w):
        return w.to_dense().float() if w.layout != torch.strided else w.float()
    Wup = dense(ffn.W_up)
    Wgate = dense(ffn.W_gate)
    Wdown = dense(ffn.W_down)
    bup = ffn.b_up.float()
    bgate = ffn.b_gate.float()
    x = xin.float()
    up = F.linear(x, Wup, bup)
    gate = F.linear(x, Wgate, bgate)
    hidden = F.silu(up) * gate
    contrib = Wdown[od] * hidden  # per-unit contribution to OUTDIM
    print(f"=== blk{BLK} BP[0] pos={bp0} dim={OUTDIM}({od}) inval={float(xin[od]):+.3e} ===")
    order = contrib.abs().argsort(descending=True)
    tot = float(contrib.sum())
    print(f"  total down contribution to {OUTDIM} = {tot:+.3e}")
    for u in order[:20].tolist():
        c = float(contrib[u])
        if abs(c) < 1.0:
            break
        print(f"  unit {u:5d}  contrib={c:+.3e}  hidden={float(hidden[u]):+.3e}  "
              f"up={float(up[u]):+.3e} gate={float(gate[u]):+.3e} "
              f"Wdown={float(Wdown[od, u]):+.3e}")


if __name__ == "__main__":
    main()
