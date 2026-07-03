#!/usr/bin/env python3
"""Inc-3 ROOT B: read the e8-marker up-path input dims at the BP[0]-predictor
row, in WHICHEVER config, taken BEFORE blk34 (the explosion block). Confirms
the address signature (ADDR_B0_LO+8, ADDR_B0_HI+14) that drives the e8 ALU
materializer's silu(up) to fire is present in CAMPAIGN but absent in GOLDEN.

Run TWICE (clear cache between):
  C4_NO_STACK0_EMIT=0  python tools/probe_inc3_bp_addr_inputs.py
  C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 python tools/probe_inc3_bp_addr_inputs.py
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
# Read just before the L20/blk34 explosion. In golden the same op runs at a
# different block index, so read at the block index BEFORE the explosion.
DIMS = ["ADDR_B0_LO+8", "ADDR_B0_HI+14", "ADDR_B0_HI+15", "HAS_SE",
        "MARK_STACK0", "IS_BYTE", "MARK_BP", "ALU_LO+15", "OUTPUT_LO+15"]


def main():
    nostk = os.environ.get("C4_NO_STACK0_EMIT", "0") != "0"
    cfg = "CAMPAIGN(30-tok)" if nostk else "GOLDEN(35-tok)"
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

    STEP = int(Token.STEP_TOKENS)
    pl = len(p._build_context(bytecode))
    bp0 = pl + 6 * STEP + 16
    ctx = p._final_context(bytecode, max_steps=10)
    padded = torch.tensor([ctx], dtype=torch.long, device=p._device)
    # Read at L19 (one block before L20 in both configs). L19 = blk33 (campaign)
    # / blk31 (golden 35-tok)? Find by logical layer.
    blm = p.block_layer_map()
    blk = None
    for b, lm in enumerate(blm):
        if lm["logical"] == 19:
            blk = b
            break
    with torch.no_grad():
        resid = p.model.forward(padded, stop_after_block=blk)[0, bp0].float()
    print(f"=== {cfg} STEP={STEP} BP[0] pos={bp0} after L19=blk{blk} ===")
    for n in DIMS:
        print(f"   {n:22s} = {float(resid[D(n)]):+.4e}")


if __name__ == "__main__":
    main()
