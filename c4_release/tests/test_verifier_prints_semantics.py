"""F-11: verifier output should include dim semantics where available."""
import pytest
from neural_vm.dim_registry import DimRegistry


def test_format_dim_with_semantics_basic():
    from neural_vm.verification.decl_verifier import _format_dim_with_semantics
    reg = DimRegistry(d_model=16)
    reg.alloc("MARK_SP", 0, 1, "SP marker", semantics="mark == SP")
    out = _format_dim_with_semantics("MARK_SP", reg)
    assert "mark == SP" in out
    assert "MARK_SP" in out


def test_format_dim_with_semantics_missing():
    from neural_vm.verification.decl_verifier import _format_dim_with_semantics
    reg = DimRegistry(d_model=16)
    reg.alloc("X", 0, 1, "X")  # no semantics -- emits DeprecationWarning, that's fine
    out = _format_dim_with_semantics("X", reg)
    assert out == "X"  # no brackets when semantics is None


def test_format_dim_with_semantics_no_registry():
    from neural_vm.verification.decl_verifier import _format_dim_with_semantics
    out = _format_dim_with_semantics("MARK_SP", None)
    assert out == "MARK_SP"


def test_format_dim_with_semantics_unknown_dim():
    from neural_vm.verification.decl_verifier import _format_dim_with_semantics
    reg = DimRegistry(d_model=16)
    # Don't allocate "MYSTERY"; lookup will raise KeyError internally
    out = _format_dim_with_semantics("MYSTERY", reg)
    assert out == "MYSTERY"  # helper swallows the KeyError
