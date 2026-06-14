"""Contract tests for the op-local residual-band registry.

The registry replaces the centralized ``_PRODUCTION_EXTRA_RESIDUAL_DIMS`` dict.
These tests lock in the byte-identity-critical invariants: the auto-collected
band set (default config), its ORDER (which fixes tail dim_positions), the
flag-gating of the MUL band, the never-share names, and the registration-API
guards (collision / idempotency / validation). Cheap (no model bake).
"""

import os

import pytest

# Importing all_core_ops triggers import-time registration from every lN_ops
# module (wildcard import), which populates the registry.
from neural_vm.unified_compiler.ops import all_core_ops  # noqa: F401
from neural_vm.unified_compiler.ops.residual_band_registry import (
    register_residual_band,
    collect_registered_residual_bands,
    collect_never_share_band_names,
    registered_band_specs,
)


# The legacy ``_PRODUCTION_EXTRA_RESIDUAL_DIMS`` for the DEFAULT config
# (C4_MUL_WIDTH2 on), in declaration order. The auto-collected set MUST equal
# this (set + ORDER) for the produced model to be byte-identical.
_EXPECTED_DEFAULT_ORDER = [
    ("H1_PREV_STEP", 7),
    ("H1_DUMP_OUT", 7),
    ("AX_CARRY_OVERFLOW", 1),
    ("STACK0_B0_H1_PREV", 7),
    ("STACK0_B0_H3_PREV", 7),
    ("STACK0_B0_DUMP_H1", 7),
    ("STACK0_B0_DUMP_H3", 7),
    ("STACK0_B0_CARRIED", 1),
    ("STACK0_B0_SHARP", 1),
    ("STACK0_B0_PREV_DOM", 1),
    ("STACK0_B0_NOT_CMP", 1),
    ("MUL_RESULT_HI_LO", 16),
    ("MUL_RESULT_HI_HI", 16),
]

_NEVER_SHARE = {
    "H1_PREV_STEP", "H1_DUMP_OUT", "AX_CARRY_OVERFLOW",
    "STACK0_B0_H1_PREV", "STACK0_B0_H3_PREV",
    "STACK0_B0_DUMP_H1", "STACK0_B0_DUMP_H3",
    "STACK0_B0_CARRIED", "STACK0_B0_SHARP",
    "STACK0_B0_PREV_DOM", "STACK0_B0_NOT_CMP",
}


def test_default_collected_set_matches_legacy_dict_and_order():
    """Byte-identity gate: collected bands == legacy dict, same ORDER."""
    if os.environ.get("C4_MUL_WIDTH2", "1") == "0":
        pytest.skip("MUL band off; this test asserts the default (MUL-on) set")
    bands = collect_registered_residual_bands()
    assert list(bands.items()) == _EXPECTED_DEFAULT_ORDER


def test_never_share_names_are_the_carry_bands_not_mul():
    names = collect_never_share_band_names()
    assert names == _NEVER_SHARE
    # MUL is liveness-shareable (matches legacy: MUL was never in the set).
    assert "MUL_RESULT_HI_LO" not in names
    assert "MUL_RESULT_HI_HI" not in names


def test_mul_band_is_flag_gated_off():
    """With C4_MUL_WIDTH2=0 the MUL band drops; AX/Root2 bands stay."""
    prev = os.environ.get("C4_MUL_WIDTH2")
    os.environ["C4_MUL_WIDTH2"] = "0"
    try:
        bands = collect_registered_residual_bands()
        assert "MUL_RESULT_HI_LO" not in bands
        assert "MUL_RESULT_HI_HI" not in bands
        # Always-present carry bands survive the flag flip.
        assert "H1_PREV_STEP" in bands
        assert "STACK0_B0_NOT_CMP" in bands
    finally:
        if prev is None:
            os.environ.pop("C4_MUL_WIDTH2", None)
        else:
            os.environ["C4_MUL_WIDTH2"] = prev


def test_every_expected_band_is_registered_with_an_owner():
    specs = {s.name: s for s in registered_band_specs()}
    for name, size in _EXPECTED_DEFAULT_ORDER:
        assert name in specs, f"{name} not registered"
        assert specs[name].size == size
        assert specs[name].owner  # non-empty owner


def test_idempotent_reregistration_is_noop():
    spec = next(s for s in registered_band_specs() if s.name == "H1_PREV_STEP")
    # Re-registering the SAME (name, size, owner, never_share) must not raise
    # and must not duplicate.
    before = len(registered_band_specs())
    register_residual_band(
        spec.name, spec.size, owner=spec.owner, never_share=spec.never_share,
    )
    assert len(registered_band_specs()) == before


def test_conflicting_reregistration_raises():
    with pytest.raises(ValueError):
        register_residual_band("H1_PREV_STEP", 99, owner="someone_else")


def test_validation_guards():
    with pytest.raises(ValueError):
        register_residual_band("", 4, owner="x")
    with pytest.raises(ValueError):
        register_residual_band("OK_NAME_A", 0, owner="x")
    with pytest.raises(ValueError):
        register_residual_band("OK_NAME_B", -1, owner="x")
    with pytest.raises(ValueError):
        register_residual_band("OK_NAME_C", True, owner="x")  # bool is not int-size
    with pytest.raises(ValueError):
        register_residual_band("OK_NAME_D", 4, owner="")
