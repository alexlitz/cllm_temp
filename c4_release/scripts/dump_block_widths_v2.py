#!/usr/bin/env python3
"""Properly read W_up shape per block to match the diag label."""
from __future__ import annotations

import os
import sys

os.environ.setdefault("C4_DECLARATIONS_ONLY_BAKE", "1")
os.environ.setdefault("C4_SPEC_K", "0")
os.environ.setdefault("C4_BATCH_USE_KV_CACHE", "0")

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
if PKG not in sys.path:
    sys.path.insert(0, PKG)

from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
    compile_full_vm_dynamic,
)


def main() -> int:
    model, layout = compile_full_vm_dynamic(strict=False)
    n_blocks = len(model.blocks)
    print(f"n_blocks={n_blocks}")
    for i, block in enumerate(model.blocks):
        attn = getattr(block, "attn", None)
        layer = getattr(attn, "layer_idx", None)
        ffn = getattr(block, "ffn", None)
        w_up = getattr(ffn, "W_up", None) if ffn is not None else None
        if w_up is not None:
            try:
                width = int(w_up.shape[0])
            except Exception:
                width = "?"
        else:
            width = None
        print(f"  block{i}: width={width} layer={layer}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
