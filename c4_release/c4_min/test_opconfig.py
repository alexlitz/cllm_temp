"""Unit tests for the per-op config toggle system (``c4_min.opconfig``).

Covers the two things the consolidation gate cares about:
  1. the VALIDATOR (the precision<->radix COUPLING table) rejects a radix that
     overflows the precision's exact-integer ceiling for the op, per the accMax
     formulas, and accepts the boundary + whole-value cases;
  2. the DEFAULT config round-trips to the nibble / fp32 build (== golden
     174ece66) — DEFAULT.is_default(), empty-env resolve == DEFAULT, and the
     C4_OPCFG_* flag grammar.

Pure CPU, no model materialised.  Run:
    OMP_NUM_THREADS=4 python -m pytest c4_min/test_opconfig.py -v
"""
from __future__ import annotations

import math

import pytest

from c4_min import opconfig as OC


# =========================================================================== #
# DEFAULT == the nibble / fp32 (golden) build round-trip.
# =========================================================================== #
def test_default_is_the_nibble_fp32_build():
    assert OC.DEFAULT.base == OC.DEFAULT_AXES
    assert OC.DEFAULT_AXES == OC.AxisConfig(
        precision="fp32", radix=16, extraction="nibble", recurrence="unrolled")
    assert OC.DEFAULT.is_default()
    for op in OC.ALL_OPS:
        assert OC.DEFAULT.for_op(op) == OC.DEFAULT_AXES
    assert OC.DEFAULT.non_default_ops() == {}


def test_empty_env_resolves_to_default():
    cfg = OC.resolve(env={})
    assert cfg.is_default()
    assert cfg.for_op("DIV") == OC.DEFAULT_AXES
    # A totally unrelated C4_* flag must NOT perturb the config.
    assert OC.resolve(env={"C4_DOOM_FAST": "1", "PATH": "/x"}).is_default()


def test_default_validates_clean():
    OC.validate(OC.DEFAULT)     # nibble fp32 radix-16 is exact for every op


# =========================================================================== #
# The VALIDATOR — the precision<->radix coupling table.
# =========================================================================== #
def test_acc_max_formulas_match_the_surface():
    # PRECISION_RADIX_SURFACE.md §1: add/sub 2r · cmp ~r(+1) · div/mod r^2 · mul peak
    assert OC.acc_max("ADD", 16) == 32          # 2r
    assert OC.acc_max("SUB", 16) == 32
    assert OC.acc_max("EQ", 16) == 17           # r+1 (cmp)
    assert OC.acc_max("DIV", 16) == 256         # r^2
    assert OC.acc_max("MOD", 16) == 256
    # MUL uses the exact schoolbook column peak (not the loose r^2*L).
    assert OC.acc_max("MUL", 16) == 1904        # the surface's measured MUL peak


def test_max_safe_radix_matches_surface_table():
    # The full (op-class x precision) -> max power-of-2 radix table, as 2^k, must
    # equal PRECISION_RADIX_SURFACE.md §2 exactly.
    expect = {
        "int8": {"ADD/SUB": 5, "CMP": 6, "MUL": 1, "DIV/MOD": 3},
        "bf16": {"ADD/SUB": 7, "CMP": 7, "MUL": 2, "DIV/MOD": 4},
        "fp16": {"ADD/SUB": 10, "CMP": 10, "MUL": 4, "DIV/MOD": 5},
        "fp32": {"ADD/SUB": 23, "CMP": 23, "MUL": 11, "DIV/MOD": 12},
        "fp64": {"ADD/SUB": 52, "CMP": 52, "MUL": 26, "DIV/MOD": 26},
    }
    for prec, row in expect.items():
        for cls, k in row.items():
            r = OC.max_safe_radix(cls, prec)
            assert r == (1 << k), (prec, cls, r, 1 << k)


@pytest.mark.parametrize("op,prec,radix,ok", [
    # DIV r^2: bf16 ceiling 256.  r=16 -> 256 == ceiling (boundary OK); r=32 -> 1024 reject.
    ("DIV", "bf16", 16, True),
    ("DIV", "bf16", 32, False),
    ("MOD", "bf16", 16, True),
    ("MOD", "bf16", 32, False),
    # ADD 2r: fp16 ceiling 2048.  r=1024 -> 2048 (OK); r=2048 -> 4096 reject.
    ("ADD", "fp16", 1024, True),
    ("ADD", "fp16", 2048, False),
    # CMP r+1: int8 ceiling 127.  r=64 -> 65 (OK); r=128 -> 129 reject.
    ("EQ", "int8", 64, True),
    ("EQ", "int8", 128, False),
    # MUL peak: int8 ceiling 127.  r=2 -> 62 (OK); r=4 -> peak 254 reject.
    ("MUL", "int8", 2, True),
    ("MUL", "int8", 4, False),
    # fp32 nibble radix-16 (the DEFAULT) is exact for every op.
    ("MUL", "fp32", 16, True),
    ("DIV", "fp32", 16, True),
])
def test_validator_coupling_rejects_overflow(op, prec, radix, ok):
    axes = OC.AxisConfig(precision=prec, radix=radix, extraction="digit_extract")
    if ok:
        OC.validate_axes(op, axes)       # no raise
    else:
        with pytest.raises(OC.OpConfigError):
            OC.validate_axes(op, axes)


def test_whole_value_skips_the_radix_coupling():
    # whole_value holds the whole value in one scalar: radix does NOT bound the
    # datapath, so an otherwise-overflowing radix is allowed.
    OC.validate_axes("MUL", OC.AxisConfig(
        precision="fp128", radix=10, extraction="whole_value", recurrence="tied"))
    OC.validate_axes("DIV", OC.AxisConfig(
        precision="fp64", radix=10, extraction="whole_value", recurrence="tied"))
    # but a limb-decomposing extraction at the same radix/precision would need the
    # coupling to hold (fp128 ceiling 2^64 is huge, so radix 10 is fine there too;
    # use a tight case: DIV fp16 radix 64 -> 4096 > 2048 rejected under digit).
    with pytest.raises(OC.OpConfigError):
        OC.validate_axes("DIV", OC.AxisConfig(
            precision="fp16", radix=64, extraction="digit_extract"))


def test_validator_rejects_bad_axis_values():
    for bad in (
        OC.AxisConfig(precision="fp99"),
        OC.AxisConfig(extraction="bits"),
        OC.AxisConfig(recurrence="folded"),
        OC.AxisConfig(radix=1),
        OC.AxisConfig(radix=0),
    ):
        with pytest.raises(OC.OpConfigError):
            OC.validate_axes("ADD", bad)


# =========================================================================== #
# The C4_OPCFG_* resolver.
# =========================================================================== #
def test_resolve_per_op_flags():
    env = {
        "C4_OPCFG_DIV_PRECISION": "fp64",
        "C4_OPCFG_DIV_RADIX": "16",
        "C4_OPCFG_DIV_EXTRACTION": "whole_value",
        "C4_OPCFG_DIV_RECURRENCE": "tied",
    }
    cfg = OC.resolve(env=env)
    div = cfg.for_op("DIV")
    assert div.precision == "fp64" and div.radix == 16
    assert div.extraction == "whole_value" and div.recurrence == "tied"
    # every OTHER op stays at DEFAULT.
    assert cfg.for_op("ADD") == OC.DEFAULT_AXES
    assert not cfg.is_default()
    assert set(cfg.non_default_ops()) == {"DIV"}


def test_resolve_all_flag_sets_the_base_and_per_op_wins():
    env = {"C4_OPCFG_ALL_PRECISION": "bf16",
           "C4_OPCFG_ALL_RADIX": "16",
           "C4_OPCFG_ALL_EXTRACTION": "digit_extract",
           "C4_OPCFG_ALL_RECURRENCE": "tied",
           "C4_OPCFG_MUL_PRECISION": "fp16"}       # per-op wins over ALL
    cfg = OC.resolve(env=env)
    assert cfg.for_op("ADD").precision == "bf16"
    assert cfg.for_op("MUL").precision == "fp16"   # override beats ALL
    assert cfg.for_op("MUL").radix == 16           # inherited from ALL base


def test_resolve_validates_by_default_and_rejects_overflow():
    with pytest.raises(OC.OpConfigError):
        # DIV bf16 radix 32 overflows the ceiling -> resolve validates -> raise.
        OC.resolve(env={"C4_OPCFG_DIV_PRECISION": "bf16",
                        "C4_OPCFG_DIV_RADIX": "32",
                        "C4_OPCFG_DIV_EXTRACTION": "digit_extract"})
    # validate_result=False lets an invalid combo through (for the fitter to size).
    cfg = OC.resolve(env={"C4_OPCFG_DIV_PRECISION": "bf16",
                          "C4_OPCFG_DIV_RADIX": "32",
                          "C4_OPCFG_DIV_EXTRACTION": "digit_extract"},
                     validate_result=False)
    assert cfg.for_op("DIV").radix == 32


def test_resolve_rejects_malformed_flags():
    for bad in ({"C4_OPCFG_DIV": "fp64"},               # no axis
                {"C4_OPCFG_DIV_WIDTH": "8"},            # unknown axis
                {"C4_OPCFG_ZZZ_PRECISION": "fp64"},     # unknown op
                {"C4_OPCFG_DIV_RADIX": "sixteen"}):     # non-int radix
        with pytest.raises(OC.OpConfigError):
            OC.resolve(env=bad)


# =========================================================================== #
# The two Pareto-corner named configs.
# =========================================================================== #
def test_min_params_config_is_fp64_fp128_whole_value_tied():
    cfg = OC.min_params_config()
    OC.validate(cfg)
    assert cfg.for_op("ADD").precision == "fp64"
    assert cfg.for_op("ADD").extraction == "whole_value"
    assert cfg.for_op("ADD").recurrence == "tied"
    assert cfg.for_op("MUL").precision == "fp128"   # 64-bit product needs fp128
    assert not cfg.is_default()


def test_min_walltime_config_is_bf16_radix16_digit_tied():
    cfg = OC.min_walltime_config()
    OC.validate(cfg)
    assert cfg.for_op("ADD").precision == "bf16"
    assert cfg.for_op("ADD").radix == 16
    assert cfg.for_op("ADD").extraction == "digit_extract"
    assert cfg.for_op("ADD").recurrence == "tied"
    assert cfg.for_op("MUL").precision == "fp16"    # MUL column peak needs fp16
    assert cfg.for_op("DIV").precision == "bf16"    # DIV r^2=256 boundary OK
    assert not cfg.is_default()
