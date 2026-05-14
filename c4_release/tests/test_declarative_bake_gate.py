import pytest

from c4_release.neural_vm.unified_compiler import full_vm_compiler as fvc
from c4_release.neural_vm.unified_compiler.decl_verifier import _build_layout_only
from c4_release.neural_vm.unified_compiler.layer_compiler import (
    DeclarationsOnlyBakeError,
    ModelLayout,
    Operation,
    dispatch_operation_bake,
)


def _op(name, *, kind="attn", migrated=True):
    return Operation(
        name=name,
        reads=set(),
        writes=set(),
        kind=kind,
        bake_fn=lambda module, dim_positions, S: None,
        migrated=migrated,
    )


def _authority_op(name, authority, *, kind="attn", migrated=True):
    op = _op(name, kind=kind, migrated=migrated)
    op.declarative_authority = authority
    return op


def _layout(
    *,
    layer_ops=None,
    block_ops=None,
    model_ops=None,
):
    return ModelLayout(
        d_model=8,
        n_layers=1,
        ops_per_layer=[list(layer_ops or [])],
        dim_positions={},
        dim_sizes={},
        block_ops=list(block_ops or []),
        model_ops=list(model_ops or []),
    )


def test_declarative_bake_authority_report_allows_owned_layout():
    report = fvc.inspect_declarative_bake_authority(
        _layout(
            layer_ops=[_op("owned_attn")],
            block_ops=[_op("owned_block", kind="block")],
            model_ops=[
                _op("head_bake", kind="model"),
                _authority_op(
                    "expand_wrapper_blocks",
                    "structural_model",
                    kind="model",
                ),
            ],
        )
    )

    assert report.ok


def test_declarations_only_dispatch_uses_declarative_generator():
    class Target:
        pass

    target = Target()

    def imperative_bake(module, dim_positions, S):
        raise AssertionError("imperative bake_fn should not run")

    def declarative_bake(module, dim_positions, S):
        module.generated = (dict(dim_positions), S)

    op = Operation(
        name="decl_only_op",
        reads=set(),
        writes=set(),
        kind="model",
        bake_fn=imperative_bake,
        declarative_bake_fn=declarative_bake,
        declarative_authority="spec_generated",
    )

    dispatch_operation_bake(
        op,
        target,
        {"A": 1},
        123.0,
        declarations_only=True,
    )

    assert target.generated == ({"A": 1}, 123.0)


def test_declarations_only_dispatch_rejects_imperative_only_op():
    op = Operation(
        name="imperative_only",
        reads=set(),
        writes=set(),
        kind="model",
        bake_fn=lambda module, dim_positions, S: None,
        declarative_authority="declarative",
    )

    with pytest.raises(DeclarationsOnlyBakeError, match="imperative_only"):
        dispatch_operation_bake(
            op,
            object(),
            {},
            100.0,
            declarations_only=True,
        )


def test_declarations_only_dispatch_skips_topology_anchor():
    op = Operation(
        name="anchor",
        reads=set(),
        writes=set(),
        kind="attn",
        bake_fn=lambda module, dim_positions, S: (_ for _ in ()).throw(
            AssertionError("topology anchor should not bake")
        ),
        migrated=True,
        declarative_authority="topology_anchor",
    )

    dispatch_operation_bake(
        op,
        object(),
        {},
        100.0,
        declarations_only=True,
    )


def test_compile_full_vm_declarations_only_reports_unsupported_ops():
    with pytest.raises(DeclarationsOnlyBakeError) as exc:
        fvc.compile_full_vm(declarations_only=True, disk_cache=False)

    message = str(exc.value)
    assert "declarative_bake_fn" in message
    assert exc.value.unsupported_ops
    assert "embedding_bake" not in exc.value.unsupported_ops
    assert "head_bake" not in exc.value.unsupported_ops
    assert "initial_pc_bake" not in exc.value.unsupported_ops
    assert "l10_post_ops_combined" not in exc.value.unsupported_ops
    assert "l10_post_op_attach" not in exc.value.unsupported_ops
    assert "l10_alu_divmod_bdtoge" not in exc.value.unsupported_ops
    assert "l10_alu_divmod_longdiv" not in exc.value.unsupported_ops
    assert "l10_alu_divmod_getobd" not in exc.value.unsupported_ops
    assert "l10_alu_divmod_install" not in exc.value.unsupported_ops
    assert "l10_alu_postop_attach" not in exc.value.unsupported_ops
    assert "l11_alu_postop_attach" not in exc.value.unsupported_ops
    assert "l12_alu_postop_attach" not in exc.value.unsupported_ops
    assert "l13_alu_postop_attach" not in exc.value.unsupported_ops
    assert "layer10_psh_stack0_passthrough_bake" not in exc.value.unsupported_ops
    assert "layer10_stack0_byte_relay_bake" not in exc.value.unsupported_ops
    assert "layer10_alu" in exc.value.unsupported_ops
    assert "layer11_mul_partial" in exc.value.unsupported_ops
    assert "layer12_mul_combine" in exc.value.unsupported_ops
    assert "layer13_shifts" in exc.value.unsupported_ops


def test_declarative_bake_authority_report_allows_topology_anchors():
    report = fvc.inspect_declarative_bake_authority(
        _layout(
            layer_ops=[
                _authority_op(
                    "_layer5_fetch_dep_anchor",
                    "topology_anchor",
                    kind="attn",
                    migrated=True,
                )
            ],
        )
    )

    assert report.ok


def test_declarative_bake_authority_enforcement_reports_blockers():
    layout = _layout(
        layer_ops=[_op("dep_anchor", migrated=False)],
        block_ops=[_op("block_anchor", kind="block", migrated=False)],
        model_ops=[
            _op("legacy_bake", kind="model"),
            _op("expand_wrapper_blocks", kind="model"),
            _authority_op(
                "real_legacy_wrapper",
                "legacy_wrapper",
                kind="model",
            ),
        ],
    )

    with pytest.raises(fvc.DeclarativeBakeRequirementError) as exc:
        fvc.enforce_declarative_bake_authority(layout)

    message = str(exc.value)
    assert "legacy_bake" in message
    assert "dep_anchor" in message
    assert "block_anchor" in message
    assert "expand_wrapper_blocks" in message
    assert "real_legacy_wrapper" in message


def test_l10_and_nibble_copy_anchors_do_not_block_authority_gate():
    layout = _build_layout_only(
        alu_mode="lookup",
        enable_conversational_io=False,
        enable_tool_calling=False,
        n_heads=8,
    )
    report = fvc.inspect_declarative_bake_authority(layout)

    migrated_anchor_names = {
        "l10_post_ops_combined",
        "layer10_byte_passthrough",
        "layer10_carry_relay",
        "layer10_psh_stack0_passthrough",
        "layer10_sp_byte_passthrough",
        "layer10_stack0_byte_relay",
        "nibble_copy_ffn",
    }

    assert not (migrated_anchor_names & set(report.non_migrated_layer_ops))
    assert not (migrated_anchor_names & set(report.non_migrated_block_ops))


def test_compile_full_vm_env_gate_fails_before_model_bake(monkeypatch):
    bad_layout = _layout(model_ops=[_op("legacy_bake", kind="model")])

    class FakeCompiler:
        def __init__(self):
            self.ops = []

        def add_op(self, op):
            self.ops.append(op)

        def compile(self):
            return bad_layout

    noop_model_op = _op("noop_model_op", kind="model")
    monkeypatch.setenv("C4_REQUIRE_DECLARATIVE_BAKE", "1")
    monkeypatch.setattr(fvc, "LayerCompiler", FakeCompiler)
    monkeypatch.setattr(fvc, "declare_setdim_compat_dims", lambda *args, **kwargs: None)
    monkeypatch.setattr(fvc, "all_core_ops", lambda *args, **kwargs: [])
    monkeypatch.setattr(fvc, "make_l10_post_op_attach_op", lambda *args, **kwargs: noop_model_op)
    monkeypatch.setattr(fvc, "make_alu_divmod_composite_ops", lambda *args, **kwargs: [])
    monkeypatch.setattr(fvc, "make_residual_alibi_slopes_op", lambda *args, **kwargs: noop_model_op)
    monkeypatch.setattr(fvc, "make_layer10_residual_alibi_slopes_op", lambda *args, **kwargs: noop_model_op)
    monkeypatch.setattr(fvc, "make_layer8_op_imm_relay_op", lambda *args, **kwargs: noop_model_op)
    monkeypatch.setattr(fvc, "make_contract_validation_op", lambda *args, **kwargs: noop_model_op)
    monkeypatch.setattr(fvc, "all_alu_postop_attach_ops", lambda *args, **kwargs: [])

    with pytest.raises(fvc.DeclarativeBakeRequirementError) as exc:
        fvc.compile_full_vm(disk_cache=True)

    assert "legacy_bake" in str(exc.value)


def test_layer6_dep_anchors_are_not_non_migrated_blockers():
    report = fvc.inspect_declarative_bake_authority(fvc.derive_layout())

    assert "layer6_attn" not in report.non_migrated_layer_ops
    assert "layer6_relay_heads" not in report.non_migrated_layer_ops
