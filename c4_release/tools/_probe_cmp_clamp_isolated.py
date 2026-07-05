#!/usr/bin/env python3
"""Isolated verification of the C4_CMP_COMBINE_MARGIN OUTPUT-HI clamp.

Builds the model (flag as set in env), then locates the cmp_hi_clamp FFN
units in the L10-main and L17 comparison_combine banks and reports their
weights so we can confirm (a) the +6 clamp units exist flag-ON, (b) each
writes -8/S on OUTPUT_HI_THIS_STEP+1..15 and +6/S on +0, gated on
MARK_SE_ONLY + SE_OP_<cmp>/OP_<cmp>. Cheap: one build, no decode.

Run:
  CUDA_VISIBLE_DEVICES="" C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
    C4_CMP_COMBINE_MARGIN=1 C4_VM_CACHE_DIR=/tmp/c4cache_flagon2 \
    python tools/_probe_cmp_clamp_isolated.py
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings("ignore")

import torch  # noqa: E402
torch.set_num_threads(int(os.environ.get("PROBE_THREADS", "4")))

from neural_vm.unified_compiler.full_vm_compiler_dynamic import (  # noqa: E402
    compile_full_vm_dynamic,
)


def main():
    model, layout = compile_full_vm_dynamic(disk_cache=True)
    dp = layout.dim_positions
    hi0 = dp["OUTPUT_HI_THIS_STEP"]
    lo0 = dp["OUTPUT_LO"]
    se = dp.get("MARK_SE_ONLY")
    # Scan every block's FFN W_down for units that write a strong negative on
    # OUTPUT_HI_THIS_STEP+1..15 AND positive on +0 (the clamp signature).
    n_found = 0
    for bi, blk in enumerate(model.blocks):
        ffn = getattr(blk, "ffn", None)
        if ffn is None or not hasattr(ffn, "W_down"):
            continue
        Wd = ffn.W_down  # [d_model, hidden]
        H = Wd.shape[1]
        for u in range(H):
            hi_neg = [float(Wd[hi0 + h, u].item()) for h in range(1, 16)]
            hi_pos = float(Wd[hi0 + 0, u].item())
            # clamp signature: +0 positive, all +1..15 negative, no OUTPUT_LO write
            lo_writes = [abs(float(Wd[lo0 + k, u].item())) for k in range(16)]
            if hi_pos > 0.01 and all(v < -0.01 for v in hi_neg) and max(lo_writes) < 1e-6:
                n_found += 1
                # report the up-gate conditions (which markers/opcodes it reads)
                Wu = ffn.W_up
                se_w = float(Wu[u, se].item()) if se is not None else 0.0
                print(f"  blk{bi} unit{u}: HI+0={hi_pos:+.4f} "
                      f"HI+1..15=[{hi_neg[0]:+.4f}..] MARK_SE_ONLY_up={se_w:+.3f} "
                      f"(clamp signature)", flush=True)
    print(f"\nTOTAL clamp units found: {n_found} "
          f"(expect 12 flag-ON = 6 L10-main + 6 L17; 0 flag-OFF)", flush=True)


if __name__ == "__main__":
    main()
