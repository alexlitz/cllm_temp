"""Unit tests for ``tools/lint_positional_invariants.py``.

The positional-invariant audit is the static catalog of every rule / head /
imperative weight-write that references a STEP_TOKENS-dependent positional
frame dim (``BYTE_INDEX_*`` / ``STACK0_BYTE*`` / ``MEM_VAL_B*`` /
``H*`` / ``L1H*`` / ``L2H0``) and reports whether it is campaign-guarded.

Contracts:
  1. ``--prove`` exits 0: the catalog includes BOTH confirmed campaign roots
     (the div/mod ``STACK0_BYTE1`` cummax anchor + the operand-CAM
     ``L2H0``/``H1``/``MEM_VAL_B*`` distance markers) and flags them as
     shift-risk.
  2. The classifier distinguishes UNGUARDED from CAMPAIGN_AWARE: a planted
     positional ref inside a guard-consulting function is CAMPAIGN_AWARE,
     one outside any guard is UNGUARDED.
  3. The registry-derive stays in sync (independently re-confirms the
     seeded positional dims from ``dim_registry.py`` descriptions).
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest


_THIS_FILE = Path(__file__).resolve()
_C4_RELEASE = _THIS_FILE.parent.parent
_REPO_ROOT = _C4_RELEASE.parent
_TOOL = _C4_RELEASE / "tools" / "lint_positional_invariants.py"

# Import the module directly for unit-level assertions.
sys.path.insert(0, str(_C4_RELEASE / "tools"))
import lint_positional_invariants as L  # noqa: E402


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_TOOL), *args],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )


def test_prove_exits_zero_and_catches_both_roots():
    """``--prove`` must pass: both confirmed roots in the catalog."""
    res = _run("--prove")
    assert res.returncode == 0, res.stdout + res.stderr
    assert "PROOF PASSED" in res.stdout
    # div/mod root, flagged UNGUARDED.
    assert "efficient_alu_neural.py" in res.stdout
    assert "STACK0_BYTE1" in res.stdout
    assert "UNGUARDED" in res.stdout
    # operand-CAM root, flagged with a distance offset.
    assert "make_layer8_mem_to_alu" in res.stdout
    assert "DISTANCE_OFFSET" in res.stdout


def test_catalog_runs_and_reports_unguarded_surface():
    res = _run()
    assert res.returncode == 0, res.stderr
    assert "POSITIONAL-INVARIANT AUDIT" in res.stdout
    assert "UNGUARDED" in res.stdout
    # The catalog is non-trivial (the whole point is a big shift-risk surface).
    assert "references:" in res.stdout


def test_dim_filter():
    res = _run("--dim", "STACK0_BYTE1")
    assert res.returncode == 0, res.stderr
    assert "efficient_alu_neural.py" in res.stdout
    # The divmod declarative op also references STACK0_BYTE1.
    assert "make_layer10_divmod_op" in res.stdout


def test_registry_derive_in_sync():
    """The registry-derive independently re-confirms the seeded positional
    dims, so a stale seed cannot silently drop a real distance-bank dim."""
    derived = L.derive_positional_dims_from_registry(_C4_RELEASE)
    # Core seed is a subset of the derived union.
    assert L._CORE_POSITIONAL_DIMS <= derived
    # The registry text actually drives the matcher (not just the seed):
    for known in ("STACK0_BYTE1", "MEM_VAL_B1", "BYTE_INDEX_0", "L2H0", "H1"):
        assert known in derived


def test_classifier_unguarded_vs_campaign_aware(tmp_path):
    """A planted positional ref outside any guard is UNGUARDED; inside a
    guard-consulting function it is CAMPAIGN_AWARE."""
    src = (
        "from .shared import no_stack0_emit_enabled\n"
        "\n"
        "def make_unguarded_op():\n"
        "    rules = [('STACK0_BYTE1', 1.0)]\n"
        "    return rules\n"
        "\n"
        "def make_guarded_op():\n"
        "    if no_stack0_emit_enabled():\n"
        "        return []\n"
        "    rules = [('STACK0_BYTE1', 1.0)]\n"
        "    return rules\n"
    )
    f = tmp_path / "planted_ops.py"
    f.write_text(src)
    dims = {"STACK0_BYTE1"}
    refs = L.scan_file(f, dims)
    by_func = {r.func: r for r in refs}
    assert "make_unguarded_op" in by_func
    assert "make_guarded_op" in by_func
    assert by_func["make_unguarded_op"].risk == "UNGUARDED"
    assert by_func["make_guarded_op"].risk == "CAMPAIGN_AWARE"


def test_distance_offset_flag(tmp_path):
    """A ``+offset`` into a distance bank is flagged DISTANCE_OFFSET in both
    the string and the attribute reference form."""
    src = (
        "def make_offset_op(BD, MEM_I):\n"
        "    a = ('L2H0+4', 1.0)\n"
        "    b = BD.L2H0 + MEM_I\n"
        "    c = ('STACK0_BYTE1', 1.0)\n"
        "    return a, b, c\n"
    )
    f = tmp_path / "offset_ops.py"
    f.write_text(src)
    dims = {"L2H0", "STACK0_BYTE1"}
    refs = L.scan_file(f, dims)
    offset_refs = [r for r in refs if r.has_offset]
    flat_refs = [r for r in refs if not r.has_offset]
    # Both L2H0 refs carry an offset; STACK0_BYTE1 does not.
    assert len(offset_refs) == 2
    assert all(r.dim == "L2H0" for r in offset_refs)
    assert any(r.dim == "STACK0_BYTE1" for r in flat_refs)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
