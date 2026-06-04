#!/usr/bin/env python3
"""One-shot: compile the VM, enumerate per-block FFN unit counts, and
identify the op(s) bound at L34.ffn (the 192-unit block flagged in
STATUS_1096_2026_06_04.md). Read-only attribution helper.
"""
from __future__ import annotations

import os
import sys

# Match the 1096 diag chunk env (declarations-only, spec_k=0, kv off).
os.environ.setdefault("C4_DECLARATIONS_ONLY_BAKE", "1")
os.environ.setdefault("C4_SPEC_K", "0")
os.environ.setdefault("C4_BATCH_USE_KV_CACHE", "0")

# Add the c4_release dir to sys.path so ``import neural_vm`` resolves.
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
        ffn = getattr(block, "ffn", None)
        if ffn is None:
            print(f"  L{i}: no ffn")
            continue
        # Walk for hidden dim
        hidden = getattr(ffn, "hidden_dim", None)
        cls_name = type(ffn).__name__
        if hidden is None and hasattr(ffn, "W_up"):
            try:
                hidden = ffn.W_up.shape[0]
            except Exception:
                hidden = "?"
        print(f"  L{i}: {cls_name} hidden_dim={hidden}")

    # Layout is a ModelLayout with ops_per_layer: List[List[Operation]]
    # and ffn_widths: Dict[int, int]. Print FFN width per pre-expansion
    # layer and which named ops live there.
    print("\n--- layout.n_layers / ffn_widths ---")
    print(f"layout.n_layers = {getattr(layout, 'n_layers', '?')}")
    ffn_widths = getattr(layout, "ffn_widths", {})
    if isinstance(ffn_widths, dict):
        for layer in sorted(ffn_widths.keys()):
            print(f"  ffn_widths[{layer}] = {ffn_widths[layer]}")

    print("\n--- layout.ops_per_layer (FFN ops) ---")
    ops_per_layer = getattr(layout, "ops_per_layer", [])
    for layer_idx, ops in enumerate(ops_per_layer):
        ffn_ops = [o for o in ops if getattr(o, "kind", None) == "ffn"]
        attn_ops = [o for o in ops if getattr(o, "kind", None) == "attn"]
        other_ops = [o for o in ops
                     if getattr(o, "kind", None) not in ("ffn", "attn")]
        ffn_names = [o.name for o in ffn_ops]
        attn_names = [o.name for o in attn_ops]
        other_names = [
            f"{o.name}({getattr(o, 'kind', '?')})" for o in other_ops
        ]
        print(
            f"  L{layer_idx} ffn={ffn_names} attn={attn_names} "
            f"other={other_names}"
        )

    # Print block-level ops with their resolved target layer.
    print("\n--- layout.block_ops (target_op_name -> resolved layer) ---")
    block_ops = getattr(layout, "block_ops", [])
    for op in block_ops:
        target = getattr(op, "target_op_name", None)
        try:
            resolved = layout.resolve_block_op_layer(op)
        except Exception as e:
            resolved = f"<err {e!r}>"
        print(
            f"  name={op.name!r} kind={getattr(op, 'kind', '?')} "
            f"target_op_name={target!r} resolved_layer={resolved}"
        )

    # And model-level
    print("\n--- layout.model_ops ---")
    for op in getattr(layout, "model_ops", []):
        print(f"  name={op.name!r} kind={getattr(op, 'kind', '?')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
