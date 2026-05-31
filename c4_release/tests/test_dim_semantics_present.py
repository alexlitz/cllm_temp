"""Tests for the DimSlot.semantics field (F-3 phase: tolerant shim)."""
import warnings
import pytest
from neural_vm.dim_registry import DimRegistry, DimSlot


def test_dimslot_carries_semantics():
    s = DimSlot("X", 0, 1, "desc", semantics="mark == SP")
    assert s.semantics == "mark == SP"


def test_dimslot_semantics_default_none():
    s = DimSlot("X", 0, 1, "desc")
    assert s.semantics is None


def test_alloc_accepts_semantics():
    reg = DimRegistry(d_model=16)
    slot = reg.alloc("X", 0, 1, "desc", semantics="mark == SP")
    assert slot.semantics == "mark == SP"
    assert reg.semantics("X") == "mark == SP"


def test_alloc_warns_when_missing_semantics():
    reg = DimRegistry(d_model=16)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        reg.alloc("X", 0, 1, "desc")
        assert any("semantics" in str(rec.message) for rec in w), (
            "expected DeprecationWarning mentioning semantics"
        )


def test_alloc_with_semantics_does_not_warn():
    reg = DimRegistry(d_model=16)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        reg.alloc("X", 0, 1, "desc", semantics="mark == SP")
        assert not any("semantics" in str(rec.message) for rec in w)


def test_semantics_lookup_unknown_dim_raises():
    reg = DimRegistry(d_model=16)
    with pytest.raises(KeyError):
        reg.semantics("nonexistent")
