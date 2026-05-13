import pytest

from c4_release.neural_vm.unified_compiler import full_vm_compiler as fvc
from c4_release.neural_vm.unified_compiler.layer_compiler import ModelLayout, Operation


def _op(name, *, kind="attn", migrated=True):
    return Operation(
        name=name,
        reads=set(),
        writes=set(),
        kind=kind,
        bake_fn=lambda module, dim_positions, S: None,
        migrated=migrated,
    )


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
            model_ops=[_op("head_bake", kind="model")],
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
        ],
    )

    with pytest.raises(fvc.DeclarativeBakeRequirementError) as exc:
        fvc.enforce_declarative_bake_authority(layout)

    message = str(exc.value)
    assert "legacy_bake" in message
    assert "dep_anchor" in message
    assert "block_anchor" in message
    assert "expand_wrapper_blocks" in message


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
