#!/usr/bin/env python3
"""Inc-3 ROOT B: dump the gate/up/down weights of the explosion unit (blk34
unit 236) + the input residual at each read dim, in the campaign config, to
name the owning rule and find WHY its gate blows up to -1e6.

Run in CAMPAIGN:
  C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 python tools/probe_inc3_unit236.py
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
from src.compiler import compile_c  # noqa: E402
from neural_vm.batched_pure_neural import Token  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402

SRC = "int main() { int x; x = 28; return x; }"
BLK = int(os.environ.get("PROBE_BLK", "34"))
UNIT = int(os.environ.get("PROBE_UNIT", "236"))


def main():
    bytecode, _ = compile_c(SRC)
    p = build_groundtruth_probe()
    from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic)
    _m, _l = compile_full_vm_dynamic(disk_cache=True)
    dp = dict(_l.dim_positions)
    inv = {}
    for name, idx in dp.items():
        inv.setdefault(idx, name)

    def dname(i):
        # find the band base <= i with the largest base
        best = None
        for name, base in dp.items():
            if base <= i and (best is None or base > best[1]):
                best = (name, base)
        if best:
            off = i - best[1]
            return f"{best[0]}+{off}" if off else best[0]
        return f"dim{i}"

    STEP = int(Token.STEP_TOKENS)
    pl = len(p._build_context(bytecode))
    bp0 = pl + 6 * STEP + 16
    ctx = p._final_context(bytecode, max_steps=10)
    padded = torch.tensor([ctx], dtype=torch.long, device=p._device)
    with torch.no_grad():
        xin = p.model.forward(padded, stop_after_block=BLK - 1)[0, bp0].float()
    blk = p.model.blocks[BLK]
    ffn = blk.ffn

    def dense(w):
        return w.to_dense().float() if w.layout != torch.strided else w.float()
    Wup = dense(ffn.W_up)[UNIT]
    Wgate = dense(ffn.W_gate)[UNIT]
    bup = float(ffn.b_up[UNIT])
    bgate = float(ffn.b_gate[UNIT])
    print(f"=== blk{BLK} unit{UNIT} BP[0] pos={bp0}  b_up={bup:+.2f} b_gate={bgate:+.2f} ===")
    print("  -- W_gate nonzeros (weight, input_val, product) --")
    nz = (Wgate.abs() > 1e-9).nonzero().flatten().tolist()
    gsum = bgate
    for i in nz:
        w = float(Wgate[i]); v = float(xin[i])
        gsum += w * v
        print(f"   {dname(i):28s}({i:4d}) w={w:+.3e} inval={v:+.3e} prod={w*v:+.3e}")
    print(f"   GATE total = {gsum:+.3e}")
    print("  -- W_up nonzeros --")
    nz = (Wup.abs() > 1e-9).nonzero().flatten().tolist()
    usum = bup
    for i in nz:
        w = float(Wup[i]); v = float(xin[i])
        usum += w * v
        print(f"   {dname(i):28s}({i:4d}) w={w:+.3e} inval={v:+.3e} prod={w*v:+.3e}")
    print(f"   UP total = {usum:+.3e}")


if __name__ == "__main__":
    main()
