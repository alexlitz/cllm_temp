"""pytest wrapper for the C4 lib-program corpus (``libprog_corpus.py``).

Phase 1: compile each corpus .c with the real compiler (src.compiler,
stdlib-linked) and run its bytecode on the REFERENCE VM, asserting stdout is
byte-exact against the gcc golden.  These tests need no model build (fast, CPU).

Phase 2: the SAME corpus THROUGH THE NEURAL MODEL (``model.forward``).  Each
program is retargeted to the neural ABI (``retarget_to_neural_abi``), run through
the streaming lib model (``build_lib_model_streaming``, ~4 GB peak, KV-cache
eviction ON) via the KV-cached driver, and its PRTF stdout is asserted byte-exact
against the SAME gcc golden.

The Phase-2 tests split into two tiers by executed-step count (neural wall-clock
is ~linear in steps at a few seconds/step on CPU):
  * FAST tier (printf_str/hex/int, <=27 steps): runs by default (minutes).
  * HEAVY tier (malloc/memset/memcmp/filecat, ~220-1551 steps): gated behind
    ``C4_LIB_NEURAL_HEAVY=1`` so plain ``pytest`` stays tractable.  These are
    real (byte-exact vs gcc), NOT faked — set the flag to run them.

Memory: single-process, ``OMP_NUM_THREADS=4``, ONE shared streaming model (~4 GB)
grown to fit the largest program, KV eviction on.
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
# PHASE 2 — neural model byte-exact (through model.forward).
# ---------------------------------------------------------------------------
_LIB_READY = LC.lib_integrated()

# The printf-only programs are the FAST neural tier (<=27 executed steps); the
# heap / file / deep-loop programs are the HEAVY tier (~220-1551 steps, minutes
# each) gated behind C4_LIB_NEURAL_HEAVY so plain pytest stays tractable.
_FAST_NEURAL = {"printf_str", "printf_hex", "printf_int"}
_HEAVY = os.environ.get("C4_LIB_NEURAL_HEAVY", "") not in ("", "0")


def _neural_ids():
    return _FAST_NEURAL | ({e.name for e in _ENTRIES} if _HEAVY else set())


@pytest.mark.skipif(
    not _LIB_READY,
    reason="c4_min.nibble_runtime + lib_neural.build_lib_model_streaming not in "
           "this checkout — neural engine not runnable")
@needs_compiler
@pytest.mark.parametrize(
    "entry",
    [e for e in _ENTRIES if e.name in _FAST_NEURAL or _HEAVY],
    ids=[e.name for e in _ENTRIES if e.name in _FAST_NEURAL or _HEAVY])
def test_model_byte_exact(entry):
    """Compile -> RETARGET to neural ABI -> run THROUGH THE TRANSFORMER -> stdout
    == golden (byte-exact).  FAST tier by default; HEAVY tier needs
    ``C4_LIB_NEURAL_HEAVY=1`` (real, not faked).  Uses the shared streaming model
    so the tier amortises ONE ~4 GB build."""
    want = LC.golden_for(entry)
    got = LC.run_model(entry, shared=True).encode("latin-1")
    assert got == want, (
        f"{entry.name}: model stdout != golden\n"
        f"  want={want!r}\n  got ={got!r}")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
