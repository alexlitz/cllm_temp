"""S-4: writer index tests."""
import pytest

from neural_vm.dim_registry import DimRegistry
from neural_vm.unified_compiler.ir import FFNOp, FFNRule
from neural_vm.verification.writer_index import (
    WriterEntry,
    build_writer_index,
)


class _FakeOp:
    def __init__(self, name, rules):
        self.name = name
        self.compiler_ir = FFNOp(rules=list(rules))


@pytest.fixture
def reg():
    r = DimRegistry(d_model=64)
    r.alloc("MARK_SP", 0, 1, "SP", semantics="mark == SP")
    r.alloc("MARK_AX", 1, 1, "AX", semantics="mark == AX")
    r.alloc("OUT_LO", 16, 16, "out lo", semantics="is_byte OR NOT is_byte")
    r.alloc("OUT_HI", 32, 16, "out hi", semantics="is_byte OR NOT is_byte")
    return r


def test_empty_index(reg):
    idx = build_writer_index([], reg)
    assert idx == {}


def test_single_rule_one_write(reg):
    rule = FFNRule.constant_write(
        conditions=(("MARK_SP", 10.0),),
        threshold=5.0,
        writes=(("OUT_LO+3", 100.0),),
        name="r1",
        scope="mark == SP",
    )
    op = _FakeOp("op1", [rule])
    idx = build_writer_index([op], reg)

    key = ("OUT_LO", 3)
    assert key in idx
    assert len(idx[key]) == 1
    entry = idx[key][0]
    assert entry.op_name == "op1"
    assert entry.rule is rule
    assert entry.max_contribution == 100.0 * (10.0 - 5.0)  # 500


def test_multiple_writers_same_dim(reg):
    r1 = FFNRule.constant_write(
        conditions=(("MARK_SP", 100.0),),
        threshold=10.0,
        writes=(("OUT_LO+0", 1.0),),
        name="r1", scope="mark == SP",
    )
    r2 = FFNRule.constant_write(
        conditions=(("MARK_SP", 20.0),),
        threshold=10.0,
        writes=(("OUT_LO+0", 1.0),),
        name="r2", scope="mark == SP",
    )
    op = _FakeOp("op", [r1, r2])
    idx = build_writer_index([op], reg)

    assert len(idx[("OUT_LO", 0)]) == 2


def test_separate_offsets_separate_entries(reg):
    r = FFNRule.constant_write(
        conditions=(("MARK_SP", 10.0),),
        threshold=5.0,
        writes=(("OUT_LO+0", 50.0), ("OUT_LO+1", 50.0)),
        name="r", scope="mark == SP",
    )
    op = _FakeOp("op", [r])
    idx = build_writer_index([op], reg)

    assert ("OUT_LO", 0) in idx
    assert ("OUT_LO", 1) in idx
    assert idx[("OUT_LO", 0)][0].rule is r
    assert idx[("OUT_LO", 1)][0].rule is r


def test_zero_weight_writes_excluded(reg):
    r = FFNRule.constant_write(
        conditions=(("MARK_SP", 10.0),),
        threshold=5.0,
        writes=(("OUT_LO+0", 50.0), ("OUT_LO+1", 0.0)),
        name="r", scope="mark == SP",
    )
    op = _FakeOp("op", [r])
    idx = build_writer_index([op], reg)

    assert ("OUT_LO", 0) in idx
    assert ("OUT_LO", 1) not in idx


def test_unknown_dim_skipped_silently(reg):
    """Rule with a condition dim that's not in the registry should be
    skipped, not crash the whole build."""
    r = FFNRule.constant_write(
        conditions=(("UNKNOWN_DIM", 10.0),),
        threshold=5.0,
        writes=(("OUT_LO+0", 50.0),),
        name="r", scope="mark == SP",
    )
    op = _FakeOp("op", [r])
    # Should not raise
    idx = build_writer_index([op], reg)
    # The skipped rule may still index OR be omitted — both OK; just
    # confirm no crash.
    assert isinstance(idx, dict)


def test_compiler_ir_with_layers(reg):
    """Test that walker handles CompilerIR.layers[].ffn shape."""
    rule = FFNRule.constant_write(
        conditions=(("MARK_SP", 10.0),),
        threshold=5.0,
        writes=(("OUT_LO+0", 100.0),),
        name="r", scope="mark == SP",
    )

    # Mimic CompilerIR shape
    class _MockLayer:
        def __init__(self, ffn):
            self.ffn = ffn

    class _MockCompilerIR:
        def __init__(self, layers):
            self.layers = layers

    op = type('Op', (), {})()
    op.name = "mock"
    op.compiler_ir = _MockCompilerIR([_MockLayer(FFNOp(rules=[rule]))])

    idx = build_writer_index([op], reg)
    assert ("OUT_LO", 0) in idx
    assert idx[("OUT_LO", 0)][0].rule is rule
