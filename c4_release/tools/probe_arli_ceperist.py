#!/usr/bin/env python3
"""AR-LI blueprint: does the store value row's CLEAN_EMBED (the LI head's V
source) carry the stored value at the LI head's READ time, or is it cleared
between the store step and the LI step in AR? Trace CLEAN_EMBED byte value at the
store value-byte-0 row (pos given by PROBE_POS) block-by-block.
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
from src.compiler import compile_c  # noqa: E402
from neural_vm.batched_pure_neural import Token  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402

SRC = os.environ.get("PROBE_SRC", "int main() { int x; x = 28; return x; }")
POS = int(os.environ.get("PROBE_POS", "267"))


def td(w):
    return (w.to_dense() if w.layout != torch.strided else w).detach().cpu().float()


def main():
    bytecode, _ = compile_c(SRC)
    p = build_groundtruth_probe()
    from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic)
    _m, _l = compile_full_vm_dynamic(disk_cache=True)
    dp = dict(_l.dim_positions)
    STEP = int(Token.STEP_TOKENS)
    ctx = p._final_context(bytecode, max_steps=8)
    padded = torch.tensor([ctx], device=p._device)
    nblk = len(p.model.blocks)
    blk_map = p.block_layer_map()
    CE_LO = dp["CLEAN_EMBED_LO"]; CE_HI = dp["CLEAN_EMBED_HI"]
    print(f"=== CLEAN_EMBED at pos{POS} (token={ctx[POS]}) per block (changes) ===")
    prev = None
    for b in range(nblk):
        x = td(p.model.forward(padded, stop_after_block=b)[0])
        row = x[POS]
        lo = int(torch.argmax(row[CE_LO:CE_LO+16])); lov = float(row[CE_LO+lo])
        hi = int(torch.argmax(row[CE_HI:CE_HI+16])); hiv = float(row[CE_HI+hi])
        val = hi*16+lo
        l2 = float(row[dp["L2H0"]+4]); msv = float(row[dp.get("MEM_STORE_AT_VAL", CE_LO)])
        key = (val, round(lov,1), round(hiv,1), round(l2,1), round(msv,1))
        if key != prev:
            lg = blk_map[b].get("logical")
            print(f"  blk{b:2d}(L{lg}): CE_val={val} (lo_c{lov:.1f} hi_c{hiv:.1f}) "
                  f"L2H0[MEM]={l2:.1f} MEM_STORE_AT_VAL={msv:.1f}")
            prev = key


if __name__ == "__main__":
    main()
