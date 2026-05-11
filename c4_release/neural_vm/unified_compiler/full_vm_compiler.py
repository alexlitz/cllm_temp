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
    all_hybrid_alu_wrap_ops,
    declare_setdim_compat_dims,
    make_alu_divmod_composite_ops,
    make_contract_validation_op,
    make_efficient_l8_addsub_wrap_op,
    make_efficient_l10_andorxor_wrap_op,
    make_efficient_l11_alumul_wrap_op,
    make_l10_post_op_attach_op,
    make_l11_alu_mul_bdtoge_op,
    make_l11_alu_mul_carrypass1_op,
    make_l11_alu_mul_carrypass2_op,
    make_l11_alu_mul_carrypass3_op,
    make_l11_alu_mul_schoolbook_op,
    make_l12_alu_mul_binarylookahead_op,
    make_l12_alu_mul_finalcorrection_op,
    make_l12_alu_mul_genprop_op,
    make_l12_alu_mul_getobd_op,
    make_layer8_op_imm_relay_op,
    make_layer10_residual_alibi_slopes_op,
    make_residual_alibi_slopes_op,
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
    pin_io_only: bool = False,
):
    """Compile and bake a full Neural VM model via the compiler.

    The compiler is the single bake authority. Internally, the legacy
    `set_vm_weights` pipeline is wrapped as one model-level Operation
    (`legacy_bake`) that the compiler dispatches. As individual ops migrate
    out of `set_vm_weights` into their own per-layer Operation instances, the
    legacy_bake op shrinks and eventually disappears.

    Args:
        pin_io_only: when True, only IO-required dims (the externally-
            observable ones read/written by token embedding, the head, and
            `_inject_*` runtime injectors) are pinned, and they are pinned to
            a *compact contiguous block* starting at position 0. Every other
            dim is bump-pointer-allocated above the IO block, shrinking
            d_model relative to the legacy `_SetDim`-pinned layout. See
            `declare_setdim_compat_dims` for details. Defaults to False for
            backward compatibility.

    Returns:
        (model, layout) where:
        - model is an AutoregressiveVM with all weights baked
        - layout is the ModelLayout (d_model, n_layers, dim_positions)
    """
    compiler = LayerCompiler()
    declare_setdim_compat_dims(compiler, pin_io_only=pin_io_only)

    # Per-layer ops drive the layout (d_model, n_layers, dim_positions).
    # Forward alu_mode so SHL/SHR (and any future alu_mode-aware migrated op)
    # can branch between the legacy lookup-table bake and the efficient
    # neural-ALU bake. Forward enable_conversational_io / enable_tool_calling
    # to flag-gated ops (registered unconditionally to keep the dep graph
    # stable) so they fire their bakes when the corresponding flag is on.
    for op in all_core_ops(
        alu_mode=alu_mode,
        enable_conversational_io=enable_conversational_io,
        enable_tool_calling=enable_tool_calling,
    ):
        compiler.add_op(op)

    # L10 post_op attach: runs as a migrated block op (phase=10.7) before the
    # legacy bake pipeline. Replaces the inline post_op appends previously
    # done in set_vm_weights. alu_mode-dependent, so it lives outside
    # all_core_ops().
    compiler.add_op(make_l10_post_op_attach_op(alu_mode=alu_mode))

    # L11/L12 MUL ALU flattening: 9 phase-ordered block ops at L11 install
    # the FlattenedALUMul wrapper. Phases 11.0..12.3 split the previous
    # monolithic `model.blocks[11].ffn = ALUMul(...)` assignment in
    # set_vm_weights into discrete BD↔GE conversion + 7 sub-FFN MUL pipeline
    # stages. All bind to layer_idx=11 (the runtime forward collapses them
    # into one block call). Only registered in efficient mode — lookup mode
    # keeps the `_set_layer11_mul_partial` / `_set_layer12_mul_combine`
    # lookup tables that are wrapped by HybridALUBlock instead.
    if alu_mode == "efficient":
        compiler.add_op(make_l11_alu_mul_bdtoge_op())
        compiler.add_op(make_l11_alu_mul_schoolbook_op())
        compiler.add_op(make_l11_alu_mul_carrypass1_op())
        compiler.add_op(make_l11_alu_mul_carrypass2_op())
        compiler.add_op(make_l11_alu_mul_carrypass3_op())
        compiler.add_op(make_l12_alu_mul_genprop_op())
        compiler.add_op(make_l12_alu_mul_binarylookahead_op())
        compiler.add_op(make_l12_alu_mul_finalcorrection_op())
        compiler.add_op(make_l12_alu_mul_getobd_op())

        # Efficient-mode ALU wrapper installs — replace the inline
        # ``model.blocks[N].ffn = ...`` assignments previously in legacy_bake's
        # efficient branch. See migrated_ops for ordering notes.
        compiler.add_op(make_efficient_l8_addsub_wrap_op(alu_mode=alu_mode))
        compiler.add_op(make_efficient_l10_andorxor_wrap_op(alu_mode=alu_mode))
        compiler.add_op(make_efficient_l11_alumul_wrap_op(alu_mode=alu_mode))

    # L10 DIV/MOD ALU flattening: 4 cooperating block ops install the
    # FlattenedDivMod composite (BD→GE, long-division pipeline, GE→BD,
    # plus an install op that appends to model.blocks[10].post_ops).
    # Replaces the previous EfficientDivMod_Neural runtime instantiations
    # (lookup-mode override at vm_step.py and efficient-mode append in
    # make_l10_post_op_attach_op). All 4 ops share a builder so the install
    # op (phase=10.8) gets the fully-assembled composite. Both alu_modes
    # use the flattened composite — its forward is byte-identical to the
    # previous EfficientDivMod_Neural.
    for op in make_alu_divmod_composite_ops():
        compiler.add_op(op)

    # Residual ALiBi-slope bakes (previously inline in set_vm_weights):
    #   phase=999 :  L6/L8/L14/L15 alibi_slopes (mode-agnostic)
    #   phase=999.1: L10 alibi_slopes (mode-conditional: lookup vs efficient)
    # phase=8.4 (block op): L8 head 4 OP_IMM relay (previously inline)
    # phase=1199 (model op): contract validation diagnostic
    # The hybrid_alu_wrap_ops are added below in the lookup branch.
    compiler.add_op(make_residual_alibi_slopes_op())
    compiler.add_op(make_layer10_residual_alibi_slopes_op(alu_mode=alu_mode))
    compiler.add_op(make_layer8_op_imm_relay_op())
    compiler.add_op(make_contract_validation_op())

    # Hybrid ALU wrap ops (lookup mode only): wrap each L8-L13 FFN with a
    # HybridALUBlock that runs a structural neural ALU on top of the
    # baked lookup-table FFN. Must run AFTER the L8-L13 FFN bakes complete,
    # which is naturally the case here because phase=L+0.5 sits AFTER the
    # block-op phases (8.0-8.5, 9, 10.0-10.85, 11, 12, 13) and the
    # phase-999 residuals.
    if alu_mode == 'lookup':
        for op in all_hybrid_alu_wrap_ops():
            compiler.add_op(op)

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
        dim_positions=layout.dim_positions,
    )

    # Run all model-level ops via the compiler dispatch. Per-layer ops are
    # skipped (by default) because legacy_bake is present and owns them.
    # Per-layer ops with migrated=True are dispatched before legacy_bake so
    # their bakes run on the freshly-built model before set_vm_weights edits.
    # Block ops with migrated=True are dispatched similarly.
    import torch as _torch
    with _torch.no_grad():
        # Migrated per-layer ops fire before block/model ops.
        for layer_idx, ops_at_layer in enumerate(layout.ops_per_layer):
            block = model.blocks[layer_idx]
            for op in ops_at_layer:
                if not op.migrated:
                    continue
                if op.kind == "attn":
                    target = block.attn
                elif op.kind == "ffn":
                    target = block.ffn
                else:
                    raise ValueError(
                        f"Op {op.name!r} in ops_per_layer has kind={op.kind!r}; "
                        "expected 'attn' or 'ffn'"
                    )
                op.bake_fn(target, layout.dim_positions, S)

        # Migrated block ops fire before model ops (which include legacy_bake).
        # `_n_layers_hint` lets block bake_fns gate on total layer count (e.g.
        # the L15 attention resize only fires for >=17-layer models).
        # Block op binding via target_op_name (resolved from layout) takes
        # precedence over the legacy layer_idx field.
        for op in sorted(
            layout.block_ops,
            key=lambda o: (layout.resolve_block_op_layer(o), o.phase or 0),
        ):
            if not op.migrated:
                continue
            block = model.blocks[layout.resolve_block_op_layer(op)]
            block._n_layers_hint = len(model.blocks)
            op.bake_fn(block, layout.dim_positions, S)

        for op in sorted(layout.model_ops, key=lambda o: (o.phase or 0)):
            op.bake_fn(model, layout.dim_positions, S)

    return model, layout
