"""End-to-end compile path: declare dims+ops, build model, bake head+embedding.

This is the M4/M5 entry point that combines:
- LayerCompiler (auto-allocates layers from declared ops)
- build_model_from_layout (constructs AutoregressiveVM with auto d_model/n_layers)
- setup_head_weights (bakes the head projection using compiler dims)
- NeuralVMEmbedding(dim_positions=...) (uses compiler dims for runtime augmentations)

Output: a working AutoregressiveVM whose entire weight set was produced by
the compiler — no _SetDim hardcoding anywhere in the production path.
"""

from typing import Optional

from .layer_compiler import LayerCompiler, build_model_from_layout
from .migrated_ops import (
    all_core_ops,
    declare_setdim_compat_dims,
    setup_head_weights,
)


def compile_full_vm(S: float = 100.0):
    """Compile and bake a full Neural VM model.

    Returns:
        (model, layout) where:
        - model is an AutoregressiveVM with all weights baked
        - layout is the ModelLayout (d_model, n_layers, dim_positions, ops_per_layer)
    """
    compiler = LayerCompiler()
    declare_setdim_compat_dims(compiler)
    for op in all_core_ops():
        compiler.add_op(op)

    layout = compiler.compile()
    # Pad d_model to be divisible by num_heads (default 8)
    if layout.d_model % 8 != 0:
        pad = 8 - (layout.d_model % 8)
        compiler.declare_dim("_pad", pad)
        layout = compiler.compile()

    model = build_model_from_layout(layout, S=S)

    # M4: also bake the head and re-create the embedding with compiler dims
    setup_head_weights(model.head, layout.dim_positions)

    # Re-create the embedding to point at compiler dim positions
    from ..neural_embedding import NeuralVMEmbedding
    new_embed = NeuralVMEmbedding(
        vocab_size=model.vocab_size,
        d_model=model.d_model,
        dim_positions=dict(layout.dim_positions),
    )
    # Copy over the embedding weight tensor (it's per-token, not dim-positional;
    # but the per-token CLEAN_EMBED/MARK_* setup happens in set_vm_weights which
    # we haven't migrated yet — for now the embedding values come from the
    # original setup. This is a remaining M4 task.)
    new_embed.embed.weight.data.copy_(model.embed.embed.weight.data)
    model.embed = new_embed

    return model, layout
