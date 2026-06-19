#!/usr/bin/env python3
"""Inc-3 ROOT B: block-by-block OUTPUT-magnitude trace at the BP[0]-predictor
row in the 30-tok campaign config — find WHERE the OUTPUT band explodes.

At the BP[0] row (step6 off16) of ``int main(){int x; x=28; return x;}`` the
``l16_bp_frame_byte1_ff`` 0xff emitter writes +-0.5 but is swamped by an
OUTPUT band that explodes to ~6.5e13 in CAMPAIGN (golden = +48.3 clean). This
walks every physical block, printing the OUTPUT_LO band (the full vector
abs-max over the 16-wide OUTPUT_LO band + the two index dims +0/+15) per block,
flagging the block where the magnitude first crosses 1e3 (the explosion onset)
and where it crosses 1e9 (the runaway). The owning op = block_layer_map[blk].

Run TWICE (clear cache between):
  C4_NO_STACK0_EMIT=0  python tools/probe_inc3_bp_explosion_trace.py
  C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 python tools/probe_inc3_bp_explosion_trace.py
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


def main():
    nostk = os.environ.get("C4_NO_STACK0_EMIT", "0") != "0"
    cfg = "CAMPAIGN(30-tok)" if nostk else "GOLDEN(35-tok)"
    bytecode, _ = compile_c(SRC)
    p = build_groundtruth_probe()
    from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic)
    _m, _l = compile_full_vm_dynamic(disk_cache=True)
    dp = dict(_l.dim_positions)
    olo = dp["OUTPUT_LO"]
    ohi = dp["OUTPUT_HI_THIS_STEP"]
    STEP = int(Token.STEP_TOKENS)
    pl = len(p._build_context(bytecode))
    bp0 = pl + 6 * STEP + 16  # BP[0]-predictor row
    nblk = len(p.runner.model.blocks)
    bl_map = p.block_layer_map()
    print(f"=== {cfg} STEP={STEP} blocks={nblk} BP[0] pos={bp0} ===")
    ctx = p._final_context(bytecode, max_steps=10)
    padded = torch.tensor([ctx], dtype=torch.long, device=p._device)
    prev_mag = 0.0
    with torch.no_grad():
        for blk in range(nblk):
            resid = p.model.forward(padded, stop_after_block=blk)[0, bp0]
            band_lo = resid[olo:olo + 16].abs()
            band_hi = resid[ohi:ohi + 16].abs()
            mag = float(max(band_lo.max().item(), band_hi.max().item()))
            o15 = float(resid[olo + 15].item())
            o0 = float(resid[olo + 0].item())
            ratio = mag / max(prev_mag, 1e-9)
            # Flag any block that multiplies the band by >5x OR adds >1e3 OR is a
            # big absolute jump.
            big = (mag > 1e3 and ratio > 5.0) or (mag - prev_mag) > 1e3 or \
                  (prev_mag < 1e3 <= mag)
            if big or blk == nblk - 1:
                lm = bl_map[blk]
                print(f" blk{blk:2d} (L{lm['logical']:2d} {lm['ffn'][:28]:28s}) "
                      f"mag {prev_mag:+.2e} -> {mag:+.2e} (x{ratio:.1f})  "
                      f"O15={o15:+.2e} O0={o0:+.2e}")
            prev_mag = mag


if __name__ == "__main__":
    main()
