"""Tests for the B15-prep ``Operation.phase`` deprecation-warning hook.

The hook in ``Operation.__post_init__`` is triple-gated: it only emits a
``DeprecationWarning`` when ALL of the following hold:

  1. ``op.phase is not None``
  2. ``C4_PHASE_STRICT_MODE=1`` is set in the environment
  3. ``C4_PHASE_DEPRECATION_WARN=1`` is set in the environment

Under any other combination it must stay silent. These tests pin that
behavior so we can flip the env flags once B14 strict mode stabilizes
and surface remaining ``phase=N.M`` users for the B15 removal pass.
"""

import os
import sys
import warnings

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neural_vm.unified_compiler.layer_compiler import Operation  # noqa: E402


PHASE_ENVS = ("C4_PHASE_STRICT_MODE", "C4_PHASE_DEPRECATION_WARN")


@pytest.fixture
def clean_phase_env(monkeypatch):
    """Strip both phase-related env flags so each test starts from a known state."""
    for key in PHASE_ENVS:
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


def _make_op(*, name="dummy_op", phase=None):
    """Construct a minimal Operation with optional ``phase``."""
    return Operation(
        name=name,
        reads=set(),
        writes=set(),
        kind="ffn",
        bake_fn=lambda module, dims, S: None,
        phase=phase,
    )


def _instantiate_capturing(maker):
    """Instantiate inside catch_warnings; return list of DeprecationWarnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        op = maker()
        # u32 invariant sanity: phase is a plain Python float/None; nothing the
        # hook does touches tensor dtype paths, so this test purely exercises
        # the gate logic and does not need a u32-typed payload.
        assert isinstance(op, Operation)
    return [w for w in caught if issubclass(w.category, DeprecationWarning)]


def test_default_no_warning_even_with_phase(clean_phase_env):
    """Default behavior: phase set, no env flags → completely silent."""
    deps = _instantiate_capturing(lambda: _make_op(phase=3.5))
    assert deps == [], (
        f"Expected no DeprecationWarning under default env, "
        f"got: {[str(w.message) for w in deps]}"
    )


def test_strict_mode_alone_no_warning(clean_phase_env):
    """Gate 2 alone is not enough: strict mode without the warn flag is silent."""
    clean_phase_env.setenv("C4_PHASE_STRICT_MODE", "1")
    deps = _instantiate_capturing(lambda: _make_op(phase=3.5))
    assert deps == []


def test_warn_flag_alone_no_warning(clean_phase_env):
    """Gate 3 alone is not enough: warn flag without strict mode is silent."""
    clean_phase_env.setenv("C4_PHASE_DEPRECATION_WARN", "1")
    deps = _instantiate_capturing(lambda: _make_op(phase=3.5))
    assert deps == []


def test_both_flags_with_phase_emits_warning(clean_phase_env):
    """All three gates: phase set + strict + warn flag → DeprecationWarning."""
    clean_phase_env.setenv("C4_PHASE_STRICT_MODE", "1")
    clean_phase_env.setenv("C4_PHASE_DEPRECATION_WARN", "1")
    deps = _instantiate_capturing(lambda: _make_op(name="phase_user", phase=7.0))
    assert len(deps) == 1, (
        f"Expected exactly one DeprecationWarning, got {len(deps)}: "
        f"{[str(w.message) for w in deps]}"
    )
    msg = str(deps[0].message)
    assert "phase_user" in msg
    assert "phase=7.0" in msg
    assert "dep-based ordering" in msg


def test_no_phase_never_warns(clean_phase_env):
    """Gate 1 short-circuit: phase=None → silent regardless of env flags."""
    clean_phase_env.setenv("C4_PHASE_STRICT_MODE", "1")
    clean_phase_env.setenv("C4_PHASE_DEPRECATION_WARN", "1")
    deps = _instantiate_capturing(lambda: _make_op(phase=None))
    assert deps == []


def test_non_one_values_treated_as_off(clean_phase_env):
    """Only the literal string '1' enables a gate — 'true'/'yes'/'0' do not."""
    clean_phase_env.setenv("C4_PHASE_STRICT_MODE", "true")
    clean_phase_env.setenv("C4_PHASE_DEPRECATION_WARN", "1")
    deps = _instantiate_capturing(lambda: _make_op(phase=3.5))
    assert deps == []

    clean_phase_env.setenv("C4_PHASE_STRICT_MODE", "1")
    clean_phase_env.setenv("C4_PHASE_DEPRECATION_WARN", "yes")
    deps = _instantiate_capturing(lambda: _make_op(phase=3.5))
    assert deps == []
