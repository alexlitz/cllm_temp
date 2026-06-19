#!/usr/bin/env python3
"""Inc-3 if_gt: per-hidden-unit attribution of the blk34 (L20 layer16_lev_routing)
OUTPUT crush at the branch-step AX-marker row (off=5, the predictor for the AX
value-byte 0x00). The axcrush probe localized OUTPUT_LO+0/HI+0 -> -4300 at blk34
when AX==0. This ranks blk34.ffn hidden units by |W_down[OUTPUT_LO+0, u]*hidden[u]|
so we can NAME the owning lev_routing rule. Uses the PROBE model's OWN dims and
forward consistently (no cross-layout mixing).

  C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 python tools/probe_inc3_ifgt_axunit.py
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
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

SRC = os.environ.get("PROBE_SRC", "int main() { if (35 > 43) return 1; return 0; }")
BLK = int(os.environ.get("PROBE_BLK", "34"))
OUTDIM = os.environ.get("PROBE_OUTDIM", "OUTPUT_LO+0")
TARGET_STEP = int(os.environ.get("PROBE_STEP", "4"))
OFF = int(os.environ.get("PROBE_OFF", "5"))


def main():
    bytecode, data = compile_c(SRC)
    p = build_groundtruth_probe()
    runner = p.runner
    dp = dict(p.model.dim_positions)

    def D(n):
        if "+" in n:
            b, o = n.rsplit("+", 1)
            return dp[b] + int(o)
        return dp[n]

    od = D(OUTDIM)
    STEP = int(Token.STEP_TOKENS)
    # ORACLE clean tape (stable positions).
    _, oracle_tokens = runner._oracle_pc_ax_steps(
        bytecode, data or b"", "", expected_steps=None, with_tokens=True)
    prompt = runner._build_element(bytecode, data or b"", [], "", spec_k=1,
                                   adaptive_start_k=0, expected_steps=None)
    prefix = list(prompt.context)
    tape = list(prefix)
    for stp in oracle_tokens:
        tape.extend(stp)
    row = len(prefix) + TARGET_STEP * STEP + OFF
    padded = torch.tensor([tape], dtype=torch.long, device=p._device)
    with torch.no_grad():
        xin = p.model.forward(padded, stop_after_block=BLK - 1)[0, row]
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
    contrib = Wdown[od] * hidden
    print(f"=== blk{BLK} step{TARGET_STEP} off={OFF} row={row} dim={OUTDIM}({od}) "
          f"inval={float(xin[od]):+.3e} src={SRC!r} ===")
    order = contrib.abs().argsort(descending=True)
    tot = float(contrib.sum())
    print(f"  total blk{BLK} down contribution to {OUTDIM} = {tot:+.3e}")
    for u in order[:20].tolist():
        c = float(contrib[u])
        if abs(c) < 1.0:
            break
        print(f"  unit {u:5d}  contrib={c:+.3e}  hidden={float(hidden[u]):+.3e}  "
              f"up={float(up[u]):+.3e} gate={float(gate[u]):+.3e} "
              f"Wdown={float(Wdown[od, u]):+.3e}")


if __name__ == "__main__":
    main()
