"""F-4 acceptance test: every default dim has a parseable semantics."""
import pytest
from neural_vm.dim_registry import build_default_registry
from neural_vm.unified_compiler.predicates import parse


def test_every_default_dim_has_semantics():
    reg = build_default_registry()
    missing = [n for n, s in reg.slots.items() if s.semantics is None]
    assert not missing, f"slots without semantics: {missing}"


def test_every_default_dim_semantics_parses():
    reg = build_default_registry()
    bad = []
    for n, s in reg.slots.items():
        try:
            parse(s.semantics)
        except Exception as e:
            bad.append((n, s.semantics, str(e)))
    assert not bad, f"unparseable semantics: {bad}"
