"""Probe: which PRE-EXPANSION layer does the divmod install op land at?

Prints the ``ops_per_layer`` assignment for the key ops, plus the
post-expansion physical-block mapping, so we can confirm whether
``l10_alu_divmod_install`` (and the divmod cleanup/ge FFNs it appends to
``block.post_ops``) lands at the intended logical L10 or drifts (like MUL
did) to a later layer.

Run: CUDA_VISIBLE_DEVICES=1 python tools/probe_divmod_assignment.py
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU compile is fine

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_vm.unified_compiler.full_vm_compiler_dynamic import compile_full_vm_dynamic

KEY = {
    "layer10_carry_relay",
    "l10_alu_divmod_bdtoge",
    "l10_alu_divmod_longdiv",
    "l10_alu_divmod_getobd",
    "l10_alu_divmod_install",
    "efficient_l11_alumul_wrap",
    "layer10_alu",
    "layer9_marker_suppress",
}


def main():
    model, layout = compile_full_vm_dynamic(alu_mode="efficient", disk_cache=False)
    print("=== PRE-EXPANSION ops_per_layer for key ops ===")
    for li, ops in enumerate(layout.ops_per_layer):
        names = [o.name for o in ops]
        hits = [n for n in names if n in KEY]
        if hits:
            print(f"  pre-exp layer {li:>2}: {hits}")
    print()
    print("=== POST-EXPANSION block -> logical map (key regions) ===")
    for phys, blk in enumerate(model.blocks):
        logical = getattr(blk, "_logical_layer", phys)
        is_exp = getattr(blk, "_is_post_op_expansion", False)
        # Flag blocks that host divmod / mul compute by hidden size
        ffn = getattr(blk, "ffn", None)
        W = getattr(ffn, "W_up", None) if ffn is not None else None
        hidden = int(W.shape[0]) if (W is not None and hasattr(W, "shape") and W.ndim == 2) else None
        tag = ""
        if hidden == 131072:
            tag = " <== GE_FFN (DIVMOD LOOKUP)"
        elif hidden == 4:
            tag = " <== DIV cleanup"
        elif hidden == 256:
            tag = " <== wide_mul(width=1)"
        if 8 <= logical <= 16 or tag:
            print(f"  phys {phys:>2} logical {logical:>2} exp={str(is_exp)[:1]} "
                  f"hidden={hidden} {type(ffn).__name__}{tag}")


if __name__ == "__main__":
    main()
