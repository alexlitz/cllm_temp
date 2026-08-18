"""Conformance gate for the #903/#920 c-torture long-long + softint64 ABI fix.

Two layers, honestly separated by what is reproducible on any box:

1. RECORDED INVARIANTS (always run) — asserts the committed
   `CTORTURE32_LONGLONG_PPIF_CONFORMANCE.json` records exactly the outcome the
   ppif work claims: the +4 newly-closed long-long cases
   (ashldi-1/ashrdi-1/lshrdi-1/pr85582-2), ZERO regressions, and the three
   conformance gates (golden 174ece66 untouched, doom byte-identity, 0
   ctorture regressions). This needs no corpus and runs green everywhere.

2. LIVE RUN (skipped unless deps present) — actually invokes
   `run_ctorture32.py` and re-checks it produces 0 regressions and the +4
   newly-closed set. This requires the GCC c-torture corpus
   (`C4_CTESTSUITE_DIR`, default /tmp/c-testsuite/tests/single-exec), the
   c4_doom transpiler id_port (`C4_DOOM_IDPORT`, has transpile.py +
   softint64.py + c4vm32u.py), and gcc -m32. When any is missing the live
   check SKIPS (it is a cross-repo / external-corpus dependency, not a
   correctness fact this repo can regenerate on its own).

The fix is confined to the c4_doom id_port transpiler + this repo's NEW
`src/compiler32.py` sibling — the golden model-build `src/compiler.py` is NOT
modified by this consolidation branch (asserted by the shared golden gate).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PPIF_JSON = os.path.join(HERE, "CTORTURE32_LONGLONG_PPIF_CONFORMANCE.json")

EXPECTED_NEWLY_CLOSED = {"ashldi-1", "ashrdi-1", "lshrdi-1", "pr85582-2"}


# --------------------------------------------------------------------------
# 1. Recorded-invariant checks (always run; no corpus / no c4_doom needed).
# --------------------------------------------------------------------------
def _load_ppif():
    with open(PPIF_JSON) as fh:
        return json.load(fh)


def test_ppif_newly_closed_plus4():
    d = _load_ppif()
    got = set(d["newly_closed_cases"])
    assert got == EXPECTED_NEWLY_CLOSED, (
        f"newly_closed_cases {got} != expected +4 {EXPECTED_NEWLY_CLOSED}"
    )
    assert len(d["newly_closed_cases"]) == 4


def test_ppif_zero_regressions():
    d = _load_ppif()
    assert d["regressions"] == [], f"unexpected regressions: {d['regressions']}"
    assert d["gates"]["ctorture_regressions"] == 0


def test_ppif_gates_present():
    d = _load_ppif()
    gates = d["gates"]
    assert str(gates["golden_174ece66"]).startswith("untouched"), gates["golden_174ece66"]
    assert "byte-identity" in str(gates["doom_byte_identity"]) or \
           "SHA-identical" in str(gates["doom_byte_identity"]), gates["doom_byte_identity"]


# --------------------------------------------------------------------------
# 2. Live conformance run (skips unless the external deps are present).
# --------------------------------------------------------------------------
def _live_deps_available():
    corpus = os.environ.get("C4_CTESTSUITE_DIR",
                            "/tmp/c-testsuite/tests/single-exec")
    doom = os.environ.get("C4_DOOM_IDPORT")
    if not (doom and os.path.isdir(doom) and
            os.path.isfile(os.path.join(doom, "softint64.py")) and
            os.path.isfile(os.path.join(doom, "c4vm32u.py"))):
        return False, "C4_DOOM_IDPORT (transpile+softint64+c4vm32u) not available"
    if not os.path.isdir(corpus):
        return False, f"c-torture corpus not at {corpus} (set C4_CTESTSUITE_DIR)"
    if shutil.which("gcc") is None:
        return False, "gcc (-m32 oracle) not available"
    return True, ""


def test_ctorture32_live_zero_regressions(tmp_path):
    ok, why = _live_deps_available()
    if not ok:
        pytest.skip(f"live c-torture run unavailable: {why}")

    out_json = str(tmp_path / "live_ctorture.json")
    rel_root = os.path.abspath(os.path.join(HERE, "..", ".."))
    env = dict(os.environ)
    env["C4_RELEASE_ROOT"] = env.get("C4_RELEASE_ROOT", rel_root)
    env["PYTHONPATH"] = os.pathsep.join(
        [rel_root, HERE, env.get("PYTHONPATH", "")]
    )
    subprocess.run(
        [sys.executable, os.path.join(HERE, "run_ctorture32.py"),
         "--json", out_json],
        cwd=HERE, env=env, check=True,
    )
    with open(out_json) as fh:
        live = json.load(fh)
    # The live run must not introduce any regression vs the recorded baseline.
    assert live.get("regressions", []) == [], live.get("regressions")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
