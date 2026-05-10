"""End-to-end compile path: declare dims+ops, build model, bake all weights.

This is THE production entry point for building a Neural VM. The compiler is
the bake authority: every weight that goes into the model is set by an
Operation registered with `LayerCompiler`.

Pipeline:
  1. Declare all dims (positions pinned to `_SetDim` for backward-compat)
  2. Add per-layer ops from `all_core_ops()` — these drive layout (d_model,
     n_layers) via dependency analysis
  3. Add `legacy_bake` model-level op — bridges the migration: while individual
     ops are being split out of `set_vm_weights` into their own `Operation`
     instances, this op runs the legacy pipeline. As ops migrate out, the
     legacy pipeline shrinks.
  4. `build_model_from_layout` constructs the model and dispatches all ops in
     dependency / phase order

Output: an AutoregressiveVM with all weights baked via the compiler. No
direct call to `set_vm_weights` from outside the compiler module.
"""

from .layer_compiler import LayerCompiler, build_model_from_layout
from .migrated_ops import (
    all_core_ops,
    declare_setdim_compat_dims,
    make_l10_post_op_attach_op,
    make_legacy_bake_op,
)


def derive_layout(num_heads: int = 8):
    """Run the LayerCompiler over `all_core_ops` to produce a ModelLayout.

    Returns a layout whose d_model is divisible by `num_heads`, padding via
    a synthetic `_pad` dim if needed.
    """
    compiler = LayerCompiler()
    declare_setdim_compat_dims(compiler)
    for op in all_core_ops():
        compiler.add_op(op)
    layout = compiler.compile()
    if layout.d_model % num_heads != 0:
        pad = num_heads - (layout.d_model % num_heads)
        compiler.declare_dim("_pad", pad)
        layout = compiler.compile()
    return layout


def compile_full_vm(
    S: float = 100.0,
    *,
    enable_conversational_io: bool = False,
    enable_tool_calling: bool = False,
    alu_mode: str = "lookup",
    n_heads: int = 8,
    ffn_hidden: int = 4096,
    max_seq_len: int = 8192,
):
    """Compile and bake a full Neural VM model via the compiler.

    The compiler is the single bake authority. Internally, the legacy
    `set_vm_weights` pipeline is wrapped as one model-level Operation
    (`legacy_bake`) that the compiler dispatches. As individual ops migrate
    out of `set_vm_weights` into their own per-layer Operation instances, the
    legacy_bake op shrinks and eventually disappears.

    Returns:
        (model, layout) where:
        - model is an AutoregressiveVM with all weights baked
        - layout is the ModelLayout (d_model, n_layers, dim_positions)
    """
    compiler = LayerCompiler()
    declare_setdim_compat_dims(compiler)

    # Per-layer ops drive the layout (d_model, n_layers, dim_positions).
    # Forward alu_mode so SHL/SHR (and any future alu_mode-aware migrated op)
    # can branch between the legacy lookup-table bake and the efficient
    # neural-ALU bake.
    for op in all_core_ops(alu_mode=alu_mode):
        compiler.add_op(op)

    # L10 post_op attach: runs as a migrated block op (phase=10.7) before the
    # legacy bake pipeline. Replaces the inline post_op appends previously
    # done in set_vm_weights. alu_mode-dependent, so it lives outside
    # all_core_ops().
    compiler.add_op(make_l10_post_op_attach_op(alu_mode=alu_mode))

    # Bridge: legacy bake runs the full set_vm_weights pipeline. Marked as
    # phase=999 so it runs after all migrated bakes. Build_model_from_layout
    # detects its presence and skips per-layer bakes (legacy_bake owns those
    # for now).
    compiler.add_op(
        make_legacy_bake_op(
            alu_mode=alu_mode,
            enable_conversational_io=enable_conversational_io,
            enable_tool_calling=enable_tool_calling,
        )
    )

    layout = compiler.compile()
    if layout.d_model % n_heads != 0:
        pad = n_heads - (layout.d_model % n_heads)
        compiler.declare_dim("_pad", pad)
        layout = compiler.compile()

    # Build the model with d_model/n_layers from the compiler. We override
    # the default size from build_model_from_layout to set ffn_hidden,
    # n_heads, and max_seq_len that AutoregressiveVM expects.
    from ..vm_step import AutoregressiveVM

    model = AutoregressiveVM(
        d_model=layout.d_model,
        n_layers=layout.n_layers,
        n_heads=n_heads,
        ffn_hidden=ffn_hidden,
        max_seq_len=max_seq_len,
    )

    # Run all model-level ops via the compiler dispatch. Per-layer ops are
    # skipped because legacy_bake is present and owns them, UNLESS the op is
    # marked `migrated=True`, in which case the compiler runs its bake_fn and
    # the corresponding direct call has been removed from set_vm_weights.
    # Block ops with migrated=True are dispatched before legacy_bake (sorted
    # by phase) so their bakes run on the freshly-built model before
    # set_vm_weights edits.
    import torch as _torch
    with _torch.no_grad():
        # Migrated per-layer attn/ffn ops fire before block ops and model ops.
        # legacy_bake (a model op at phase=999) won't re-bake these because
        # set_vm_weights no longer contains their direct calls.
        for layer_idx, ops_at_layer in enumerate(layout.ops_per_layer):
            block = model.blocks[layer_idx]
            for op in ops_at_layer:
                if not op.migrated:
                    continue
                target = block.attn if op.kind == "attn" else block.ffn
                op.bake_fn(target, layout.dim_positions, S)

        # Migrated block ops fire before model ops (which include legacy_bake).
        for op in sorted(
            layout.block_ops, key=lambda o: (o.layer_idx, o.phase or 0)
        ):
            if not op.migrated:
                continue
            block = model.blocks[op.layer_idx]
            op.bake_fn(block, layout.dim_positions, S)

        for op in sorted(layout.model_ops, key=lambda o: (o.phase or 0)):
            op.bake_fn(model, layout.dim_positions, S)

    return model, layout
