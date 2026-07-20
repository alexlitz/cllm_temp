"""pytest wrapper for the C4 lib-program corpus (``libprog_corpus.py``).

Phase 1 (NOW): compile each corpus .c with the real compiler (src.compiler,
stdlib-linked) and run its bytecode on the REFERENCE VM, asserting stdout is
byte-exact against the gcc golden.  These tests need no model build (fast, CPU).

Phase 2 (PENDING on #646): the same corpus through the NEURAL model.  Those tests
are ``skip``-marked until #646's runtime library (``c4_min.nibble_runtime`` on the
unified model) lands, at which point they run the SAME byte-exact assertion via
``model.forward``.  They are NOT faked while pending.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("OMP_NUM_THREADS", "4")

from . import libprog_corpus as LC


# ---------------------------------------------------------------------------
# Corpus parametrization.
# ---------------------------------------------------------------------------
_ENTRIES = LC.CORPUS
_IDS = [e.name for e in _ENTRIES]


def _has_gcc() -> bool:
    return LC.gcc_available()


def _has_compiler() -> bool:
    try:
        LC.compile_c_source("int main(){return 0;}")
        return True
    except Exception:
        return False


needs_compiler = pytest.mark.skipif(
    not _has_compiler(), reason="src.compiler.compile_c not importable")


# ---------------------------------------------------------------------------
# The goldens are frozen in libprog/GOLDENS.txt.  Assert they exist AND (when gcc
# is available) that they still match a fresh gcc build — so a drifted golden is
# caught, not silently trusted.
# ---------------------------------------------------------------------------
def test_goldens_file_present():
    goldens = LC.load_goldens()
    assert goldens, "libprog/GOLDENS.txt missing or empty (run --regen-goldens)"
    for e in _ENTRIES:
        assert e.name in goldens, f"no frozen golden for {e.name}"


@pytest.mark.skipif(not _has_gcc(), reason="gcc not available")
@pytest.mark.parametrize("entry", _ENTRIES, ids=_IDS)
def test_frozen_golden_matches_gcc(entry):
    """The frozen golden equals a fresh real-gcc build (no drift)."""
    frozen = LC.load_goldens().get(entry.name)
    fresh = LC.gen_golden(entry)
    assert frozen == fresh, (
        f"{entry.name}: frozen golden != fresh gcc\n"
        f"  frozen={frozen!r}\n  gcc   ={fresh!r}")


# ---------------------------------------------------------------------------
# PHASE 1 — reference VM byte-exact.  This is the gate that runs now.
# ---------------------------------------------------------------------------
@needs_compiler
@pytest.mark.parametrize("entry", _ENTRIES, ids=_IDS)
def test_reference_byte_exact(entry):
    """Compile -> run on the reference C4 VM -> stdout == golden (byte-exact)."""
    want = LC.golden_for(entry)
    got = LC.run_reference(entry).encode("latin-1")
    assert got == want, (
        f"{entry.name}: reference stdout != golden\n"
        f"  lib funcs : {entry.lib_funcs}\n"
        f"  want={want!r}\n  got ={got!r}")


# ---------------------------------------------------------------------------
# PHASE 2 — neural model byte-exact.  PENDING on #646.
# ---------------------------------------------------------------------------
_LIB_READY = LC.lib_integrated()


@pytest.mark.skipif(
    not _LIB_READY,
    reason="PENDING #646: c4_min.nibble_runtime (runtime lib on the unified "
           "model) not yet in this checkout — neural engine not runnable")
@needs_compiler
@pytest.mark.parametrize("entry", _ENTRIES, ids=_IDS)
def test_model_byte_exact(entry):
    """Compile -> run THROUGH THE TRANSFORMER -> stdout == golden (byte-exact).

    Active only once #646 lands; until then it is skipped (reported PENDING), and
    NEVER faked.
    """
    want = LC.golden_for(entry)
    got = LC.run_model(entry).encode("latin-1")
    assert got == want, (
        f"{entry.name}: model stdout != golden\n"
        f"  want={want!r}\n  got ={got!r}")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
