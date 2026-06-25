#!/usr/bin/env python3
"""Map physical blocks 42..60 to their owning post-op / FFN by inspecting the
built model block structure (ffn type, hidden dim, presence of attn).
"""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ.setdefault("C4_NO_STACK0_EMIT", "1")
os.environ.setdefault("C4_OPERAND_FROM_MEMSP", "1")
os.environ.setdefault("C4_SKIP_DIM_INTEGRITY", "1")
os.environ.setdefault("C4_SKIP_GATE_CHECK", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings("ignore")

import torch  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402


def main():
    p = build_groundtruth_probe()
    model = p.model
    for b, blk in enumerate(model.blocks):
        if b < 40:
            continue
        attn = getattr(blk, "attn", None)
        ffn = getattr(blk, "ffn", None)
        post = getattr(blk, "post_ops", None) or []
        ffn_type = type(ffn).__name__ if ffn is not None else None
        hid = None
        if ffn is not None and hasattr(ffn, "W_up"):
            try:
                hid = tuple(ffn.W_up.shape)
            except Exception:
                hid = None
        has_attn = attn is not None and getattr(attn, "W_q", None) is not None
        print(f"block {b}: ffn={ffn_type} W_up={hid} attn={has_attn} npost={len(post)}")


if __name__ == "__main__":
    main()
