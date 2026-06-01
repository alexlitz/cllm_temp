"""F-6: FFNRule.scope field threading tests."""
from neural_vm.unified_compiler.ir import FFNRule


def test_default_scope_is_none():
    r = FFNRule.constant_write(
        conditions=(("X", 1.0),),
        threshold=0.5,
        writes=(("Y", 1.0),),
    )
    assert r.scope is None


def test_constant_write_accepts_scope():
    r = FFNRule.constant_write(
        conditions=(("X", 1.0),),
        threshold=0.5,
        writes=(("Y", 1.0),),
        scope="mark == SP",
    )
    assert r.scope == "mark == SP"


def test_gated_write_accepts_scope():
    r = FFNRule.gated_write(
        conditions=(("X", 1.0),),
        threshold=0.5,
        writes=(("Y", 1.0),),
        gate="G",
        scope="mark == AX AND step_is_fresh",
    )
    assert r.scope == "mark == AX AND step_is_fresh"


def test_scope_preserved_through_immutability():
    """FFNRule is frozen — scope must be set at construction."""
    r = FFNRule.constant_write(
        conditions=(("X", 1.0),),
        threshold=0.5,
        writes=(("Y", 1.0),),
        scope="mark == SP",
    )
    import dataclasses
    with __import__('pytest').raises(dataclasses.FrozenInstanceError):
        r.scope = "different"


def test_exact_output_byte_rules_threads_scope():
    """If exact_output_byte_rules helper exists, scope should thread
    through to every generated FFNRule."""
    try:
        from neural_vm.unified_compiler.ops.l10_ops import exact_output_byte_rules
    except ImportError:
        import pytest
        pytest.skip("exact_output_byte_rules not importable from l10_ops")
    rules = list(exact_output_byte_rules(
        name="test",
        expected_byte=0xF8,
        conditions=(("MARK_SP", 10.0),),
        threshold=10.0,
        max_abs_weight=1e6,
        scope="mark == SP AND step_index == 0",
    ))
    assert len(rules) > 0
    for r in rules:
        assert r.scope == "mark == SP AND step_index == 0", (
            f"rule {r.name} did not inherit scope: got {r.scope!r}"
        )
