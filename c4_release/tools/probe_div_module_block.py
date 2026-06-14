#!/usr/bin/env python3
"""Find which physical block holds the FlattenedDivMod post_op module, and
which holds the layer10_psh_ax_broadcast heads. Settles the timing.

Usage: CUDA_VISIBLE_DEVICES=0 python tools/probe_div_module_block.py
"""
import os, sys
os.environ.setdefault("C4_TEST_SPEC_K","0"); os.environ.setdefault("C4_SMOKE_SPEC_K","0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings("ignore")
from tools.probe_groundtruth import build_groundtruth_probe

def main():
    probe=build_groundtruth_probe(); model=probe.model
    for phys, blk in enumerate(model.blocks):
        logical=getattr(blk,"_logical_layer",phys)
        ffn=type(getattr(blk,"ffn",None)).__name__
        attn=type(getattr(blk,"attn",None)).__name__
        post=[type(p).__name__ for p in getattr(blk,"post_ops",[])] if hasattr(blk,"post_ops") else []
        nheads=getattr(getattr(blk,"attn",None),"num_heads",None)
        marker=""
        if "DivMod" in ffn or any("DivMod" in p for p in post): marker += " <<DIVMOD"
        if 9 <= phys <= 18:
            print(f"blk{phys:2d} L{logical:2d} attn={attn}(h={nheads}) ffn={ffn} post={post}{marker}")

if __name__=="__main__":
    main()
