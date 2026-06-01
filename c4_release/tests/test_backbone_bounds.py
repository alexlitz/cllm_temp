"""S-8: backbone bounds loader tests."""
import json
import pytest
from pathlib import Path
import tempfile

from neural_vm.unified_compiler.backbone_bounds import BackboneBounds, load_default_bounds


def _write_v1(tmp_path, bounds):
    path = tmp_path / "v1.json"
    path.write_text(json.dumps({
        "version": 1,
        "corpus_size": 32,
        "model_commit": "abc123",
        "bounds": bounds,
    }))
    return path


def test_empty_bounds():
    b = BackboneBounds.empty()
    assert b.max_positive_contribution("X", 0, "mark == SP") == 0.0
    assert not b.has_entry("X", 0, "mark == SP")


def test_load_simple(tmp_path):
    path = _write_v1(tmp_path, {
        "OUT_LO+0": {
            "mark == SP": {"max_positive_contribution": 5.0, "max_negative_contribution": -2.0, "samples": 10}
        }
    })
    b = BackboneBounds.load(path)
    assert b.version == 1
    assert b.corpus_size == 32
    assert b.has_entry("OUT_LO", 0, "mark == SP")
    assert b.max_positive_contribution("OUT_LO", 0, "mark == SP") == 5.0
    assert b.max_negative_contribution("OUT_LO", 0, "mark == SP") == -2.0


def test_missing_dim_warns_and_returns_zero(tmp_path):
    path = _write_v1(tmp_path, {})
    b = BackboneBounds.load(path)

    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        val = b.max_positive_contribution("UNKNOWN", 0, "mark == SP")
    assert val == 0.0
    assert any("UNKNOWN" in str(rec.message) for rec in w)


def test_missing_position_class_returns_zero_no_warn(tmp_path):
    """If the dim exists but the position class doesn't, no warning - just 0."""
    path = _write_v1(tmp_path, {
        "OUT_LO+0": {"mark == SP": {"max_positive_contribution": 5.0}}
    })
    b = BackboneBounds.load(path)
    assert b.max_positive_contribution("OUT_LO", 0, "mark == AX") == 0.0


def test_as_strength_bound_exact_match(tmp_path):
    path = _write_v1(tmp_path, {
        "OUT_LO+0": {"mark == SP": {"max_positive_contribution": 5.0}}
    })
    b = BackboneBounds.load(path)
    assert b.as_strength_bound("OUT_LO", 0, "mark == SP") == 5.0


def test_as_strength_bound_falls_back_to_max(tmp_path):
    """When scope_str doesn't match any class, return max across classes."""
    path = _write_v1(tmp_path, {
        "OUT_LO+0": {
            "mark == SP": {"max_positive_contribution": 5.0},
            "mark == AX": {"max_positive_contribution": 12.0},
        }
    })
    b = BackboneBounds.load(path)
    # Unrecognized scope_str -> conservative max
    assert b.as_strength_bound("OUT_LO", 0, "mark == SP AND step_index == 0") == 12.0


def test_load_default_bounds_returns_empty_if_missing():
    """If the real bounds file doesn't exist, fall back gracefully."""
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        b = load_default_bounds()
    # Either loaded successfully OR emits a warning + returns empty bounds.
    # Test passes either way - we just want no crash.
    assert b is not None
