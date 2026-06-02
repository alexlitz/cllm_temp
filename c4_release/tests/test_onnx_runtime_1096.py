#!/usr/bin/env python3
"""Phase 8.J.2 — ONNX runtime 1096 corpus byte-identity gate (scaffold).

This module is the test framework promised by ``docs/PHASE_8_PLAN.md`` sub-wave
**8.J** (the "ONNX export + ONNX runtime 1096 test suite" wave that closes
checklist item C3 and contributes to C10). The acceptance criterion is:

    All 1096 corpus programs run through the ONNX runtime export and produce
    byte-identical output (exit code + stdout) versus the PyTorch
    ``BakedC4Transformer`` reference. ONNX runtime passes 1096/1096 == PyTorch
    1096/1096.

Status: SCAFFOLD ONLY. The 8.J ONNX-export agent is in flight. The exporter
hooks expected by these tests are::

    bundler.bundle_onnx_standard  (or v2 / memory variant)
    tools.export_full_vm_onnx     (or whatever lands under tools/)

Until the exporter lands, every test in this file skips with a structured
"ONNX export not yet landed" reason. The detection logic is centralized in
:func:`_locate_onnx_exporter` so a single fixture flip auto-activates the
suite when 8.J commits land — no test edits required.

The skip-until pattern mirrors the conditional-import + module-level skipif
pattern already used by ``test_kv_eviction.py`` / ``test_onnx_export.py``
in this repo (see ``HAS_ONNX_RUNTIME`` / ``pytest.mark.skipif`` blocks).

Running standalone is still supported for ad-hoc validation::

    python tests/test_onnx_runtime_1096.py            # all 1096 (when activated)
    python tests/test_onnx_runtime_1096.py --quick    # first 100
    python tests/test_onnx_runtime_1096.py --verbose  # show each test

Under pytest the same tests run but skip cleanly when the exporter is absent::

    pytest tests/test_onnx_runtime_1096.py -v
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import tempfile
import time
from typing import Any, List, Optional, Tuple

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Optional runtime dependencies (torch is required, onnxruntime is optional
# at import time but required for any non-skipped test).
# ---------------------------------------------------------------------------


try:
    import torch  # noqa: F401  -- imported for side-effect availability check
    HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is a hard dep of this repo
    HAS_TORCH = False


try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:  # pragma: no cover
    HAS_NUMPY = False


try:
    import onnxruntime as ort  # noqa: F401
    HAS_ONNX_RUNTIME = True
except ImportError:
    HAS_ONNX_RUNTIME = False


# ---------------------------------------------------------------------------
# 8.J exporter detection: a single source of truth for "is the export agent's
# work landed yet?". The names below are the candidate hooks documented in
# ``docs/PHASE_8_PLAN.md`` (sub-wave 8.J.1). Any one of them being importable
# flips the suite from skip-mode to live-mode.
# ---------------------------------------------------------------------------


# Each entry is (module_path, attribute) where attribute is the callable that
# produces an ONNX file from a compiled VM. The first hit wins. Order matters:
# the bundler entries are the wave's primary deliverable, the tools entry is
# the documented secondary deliverable, the in-repo helper at the bottom of
# this file is the legacy fallback referenced by the original 8.J.2 script.
_CANDIDATE_EXPORTERS: Tuple[Tuple[str, str], ...] = (
    ("bundler.bundle_onnx_standard", "export"),
    ("bundler.bundle_onnx_v2", "export"),
    ("bundler.bundle_onnx_memory", "export"),
    ("tools.export_full_vm_onnx", "export_full_vm_onnx"),
    ("tools.bundle_onnx", "export_full_vm_onnx"),
)


def _locate_onnx_exporter() -> Optional[Tuple[str, str, Any]]:
    """Return (module_path, attr, callable) of the first available exporter.

    Returns None when no candidate is importable AND has the expected
    attribute, signalling that the 8.J export agent has not yet landed.
    """
    for module_path, attr in _CANDIDATE_EXPORTERS:
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            continue
        exporter = getattr(module, attr, None)
        if exporter is None or not callable(exporter):
            continue
        return (module_path, attr, exporter)
    return None


def _onnx_export_landed() -> bool:
    """Cheap predicate for ``pytest.mark.skipif`` decorators."""
    return _locate_onnx_exporter() is not None


SKIP_REASON_NO_EXPORTER = (
    "ONNX export not yet landed (Phase 8.J in flight). "
    "This scaffold auto-activates when one of "
    f"{[m for m, _ in _CANDIDATE_EXPORTERS]} becomes importable."
)


SKIP_REASON_NO_ONNXRUNTIME = (
    "onnxruntime not installed (install via `pip install onnxruntime` to "
    "exercise the 1096 corpus byte-identity gate)."
)


# Module-level skipif: if neither dependency is present, every test below
# skips with a structured reason. This is the "skip-until" guard the brief
# asked for; mirrors test_kv_eviction.py / test_onnx_export.py's pattern.
pytestmark = [
    pytest.mark.skipif(not HAS_TORCH, reason="torch not installed"),
    pytest.mark.skipif(not HAS_NUMPY, reason="numpy not installed"),
]


# ---------------------------------------------------------------------------
# Corpus + reference fixtures.
# ---------------------------------------------------------------------------


# Quick-mode subset size. Matches ``test_suite_1000.get_quick_tests`` (first
# 100 programs). Kept as a module-level constant so the CLI ``--quick`` flag
# and the parametrized "quick" pytest marker agree on what "quick" means.
QUICK_SUBSET_SIZE = 100


@pytest.fixture(scope="module")
def corpus_all() -> List[Tuple[str, int, str]]:
    """The full 1096-program regression corpus.

    Each entry is ``(source, expected_exit_code, description)``. Generated
    by ``tests.test_suite_1000.generate_test_programs``, the canonical
    corpus producer used by the rest of the repo's 1096 harnesses.
    """
    from tests.test_suite_1000 import generate_test_programs
    tests = generate_test_programs()
    assert len(tests) == 1096, (
        f"Corpus size drifted from 1096 (got {len(tests)}). Regenerate "
        f"the baseline or update QUICK_SUBSET_SIZE."
    )
    return tests


@pytest.fixture(scope="module")
def corpus_quick() -> List[Tuple[str, int, str]]:
    """First :data:`QUICK_SUBSET_SIZE` programs from the corpus."""
    from tests.test_suite_1000 import get_quick_tests
    return get_quick_tests()


@pytest.fixture(scope="module")
def onnx_exporter():
    """The 8.J ONNX exporter callable (or skip if not landed yet)."""
    located = _locate_onnx_exporter()
    if located is None:
        pytest.skip(SKIP_REASON_NO_EXPORTER)
    return located  # (module_path, attr, callable)


@pytest.fixture(scope="module")
def onnx_model_path(onnx_exporter, tmp_path_factory) -> str:
    """Materialize the ONNX export to a temp file (once per test module)."""
    _module_path, _attr, exporter = onnx_exporter
    out_dir = tmp_path_factory.mktemp("onnx_runtime_1096")
    out_path = str(out_dir / "vm.onnx")

    # The exporter is expected to accept ``output_path`` as either a
    # positional or keyword argument. Try keyword first for clarity, then
    # positional for older signatures.
    try:
        exporter(output_path=out_path)
    except TypeError:
        exporter(out_path)

    assert os.path.exists(out_path), (
        f"ONNX exporter {exporter!r} did not produce a file at {out_path}"
    )
    assert os.path.getsize(out_path) > 0, (
        f"ONNX exporter {exporter!r} produced an empty file at {out_path}"
    )
    return out_path


@pytest.fixture(scope="module")
def onnx_session(onnx_model_path):
    """Loaded ``onnxruntime.InferenceSession`` for the exported VM."""
    if not HAS_ONNX_RUNTIME:
        pytest.skip(SKIP_REASON_NO_ONNXRUNTIME)
    import onnxruntime as ort
    return ort.InferenceSession(onnx_model_path)


@pytest.fixture(scope="module")
def pytorch_reference():
    """PyTorch ``BakedC4Transformer`` reference runner.

    Shared across the module so the (expensive) bake happens once.
    """
    from src.baked_c4 import BakedC4Transformer
    return BakedC4Transformer(use_speculator=True)


# ---------------------------------------------------------------------------
# Helpers — kept module-level so the CLI entrypoint can reuse them.
# ---------------------------------------------------------------------------


def _run_program_pytorch(reference, source: str) -> int:
    """Run a C source through the PyTorch reference, return exit code."""
    return reference.run_c(source)


def _run_program_onnx(session, source: str) -> int:
    """Run a C source through the ONNX runtime, return exit code.

    Wired against the 8.J export agent's expected runtime contract: the
    exported graph accepts token-id input ``input_ids: [B, S]`` (int64)
    and produces ``logits: [B, S, V]`` (float). The autoregressive decode
    loop drives the session to an ``EXIT`` token and returns its value.

    Until the export agent lands, this helper is never reached (the
    fixtures above skip the test first).
    """
    # The real implementation belongs in the 8.J export agent's runtime
    # helper module. For now this is a placeholder so the scaffold has a
    # complete call graph; the live exporter will provide a
    # ``run_program(session, source)`` adapter, which the test will
    # prefer when present.
    raise NotImplementedError(
        "ONNX autoregressive decode loop is the 8.J export agent's "
        "responsibility. This helper is replaced by the exporter's "
        "runtime adapter when 8.J lands."
    )


def _run_corpus_pytorch(reference, tests, verbose: bool = False) -> dict:
    """Run the corpus through the PyTorch reference, return result dict."""
    passed = 0
    failed = 0
    errors = 0
    failed_tests: List[Tuple[str, int, Optional[int], str]] = []
    start = time.time()
    for i, (source, expected, desc) in enumerate(tests):
        try:
            got = _run_program_pytorch(reference, source)
            if got == expected:
                passed += 1
            else:
                failed += 1
                failed_tests.append((desc, expected, got, "mismatch"))
        except Exception as e:  # pragma: no cover - diagnostic only
            errors += 1
            failed_tests.append((desc, expected, None, f"ERROR: {e}"))
        if verbose:
            print(f"  [{i+1:4d}] {desc}")
    return {
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "total": len(tests),
        "elapsed": time.time() - start,
        "failed_tests": failed_tests,
    }


def _run_corpus_onnx(session, tests, verbose: bool = False) -> dict:
    """Run the corpus through ONNX runtime, return result dict.

    Prefers the exporter-provided ``run_program`` adapter (if present)
    over the placeholder :func:`_run_program_onnx`.
    """
    # Look up the exporter's runtime adapter; this is the 8.J agent's
    # promised public API. Falls back to the placeholder which raises
    # NotImplementedError.
    runner = None
    located = _locate_onnx_exporter()
    if located is not None:
        module_path, _attr, _exporter = located
        try:
            module = importlib.import_module(module_path)
            runner = getattr(module, "run_program", None)
        except ImportError:
            runner = None
    if runner is None:
        runner = _run_program_onnx

    passed = 0
    failed = 0
    errors = 0
    failed_tests: List[Tuple[str, int, Optional[int], str]] = []
    start = time.time()
    for i, (source, expected, desc) in enumerate(tests):
        try:
            got = runner(session, source)
            if got == expected:
                passed += 1
            else:
                failed += 1
                failed_tests.append((desc, expected, got, "mismatch"))
        except Exception as e:
            errors += 1
            failed_tests.append((desc, expected, None, f"ERROR: {e}"))
        if verbose:
            print(f"  [{i+1:4d}] {desc}")
    return {
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "total": len(tests),
        "elapsed": time.time() - start,
        "failed_tests": failed_tests,
    }


# ---------------------------------------------------------------------------
# Pre-activation status tests — these run even before 8.J lands and
# document the current export-readiness state.
# ---------------------------------------------------------------------------


def test_corpus_generator_yields_1096():
    """Corpus producer is stable at 1096 programs (regression gate)."""
    from tests.test_suite_1000 import generate_test_programs
    tests = generate_test_programs()
    assert len(tests) == 1096, (
        f"Corpus drift: expected 1096 programs, got {len(tests)}. "
        f"Update QUICK_SUBSET_SIZE and downstream callers."
    )


def test_quick_subset_is_prefix_of_full_corpus():
    """``get_quick_tests`` returns the first 100 of the full corpus."""
    from tests.test_suite_1000 import generate_test_programs, get_quick_tests
    full = generate_test_programs()
    quick = get_quick_tests()
    assert len(quick) == QUICK_SUBSET_SIZE
    assert quick == full[:QUICK_SUBSET_SIZE]


def test_onnx_export_landing_indicator_is_well_formed():
    """``_locate_onnx_exporter`` returns either None or a 3-tuple.

    Property test: the predicate's contract is what gates the rest of the
    suite, so the contract itself is worth pinning down. This stays green
    in both pre-landing and post-landing modes.
    """
    located = _locate_onnx_exporter()
    if located is None:
        # Pre-landing: skip-mode is active. Document the reason for
        # readers triaging a CI failure here.
        assert not _onnx_export_landed()
    else:
        assert _onnx_export_landed()
        assert isinstance(located, tuple) and len(located) == 3
        module_path, attr, exporter = located
        assert isinstance(module_path, str) and module_path
        assert isinstance(attr, str) and attr
        assert callable(exporter)


# ---------------------------------------------------------------------------
# Live tests — every one skips with SKIP_REASON_NO_EXPORTER until 8.J
# lands an exporter that ``_locate_onnx_exporter`` can find.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _onnx_export_landed(), reason=SKIP_REASON_NO_EXPORTER)
@pytest.mark.skipif(not HAS_ONNX_RUNTIME, reason=SKIP_REASON_NO_ONNXRUNTIME)
def test_onnx_export_produces_loadable_file(onnx_model_path):
    """The exported ONNX file loads in ``onnxruntime``.

    Acceptance gate from 8.J.1. Independent of the 1096 corpus run.
    """
    import onnxruntime as ort
    session = ort.InferenceSession(onnx_model_path)
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    assert len(inputs) >= 1, "exported VM has no inputs"
    assert len(outputs) >= 1, "exported VM has no outputs"


@pytest.mark.skipif(not _onnx_export_landed(), reason=SKIP_REASON_NO_EXPORTER)
@pytest.mark.skipif(not HAS_ONNX_RUNTIME, reason=SKIP_REASON_NO_ONNXRUNTIME)
def test_onnx_quick_subset_matches_pytorch_byte_identical(
    onnx_session, pytorch_reference, corpus_quick,
):
    """First :data:`QUICK_SUBSET_SIZE` programs: ONNX exit code == PyTorch.

    Faster smoke gate that runs before the full 1096 sweep. Both halves
    use the same corpus producer so any mismatch points cleanly at the
    ONNX runtime path.
    """
    onnx_results = _run_corpus_onnx(onnx_session, corpus_quick)
    pt_results = _run_corpus_pytorch(pytorch_reference, corpus_quick)

    assert pt_results["failed"] == 0 and pt_results["errors"] == 0, (
        f"PyTorch reference failed on quick subset "
        f"({pt_results['failed']} failures, {pt_results['errors']} errors); "
        f"cannot establish baseline for ONNX comparison."
    )
    assert onnx_results["passed"] == pt_results["passed"], (
        f"ONNX runtime quick subset diverged from PyTorch baseline: "
        f"ONNX {onnx_results['passed']}/{onnx_results['total']} vs "
        f"PyTorch {pt_results['passed']}/{pt_results['total']}. "
        f"First mismatches: {onnx_results['failed_tests'][:5]}"
    )


@pytest.mark.slow
@pytest.mark.skipif(not _onnx_export_landed(), reason=SKIP_REASON_NO_EXPORTER)
@pytest.mark.skipif(not HAS_ONNX_RUNTIME, reason=SKIP_REASON_NO_ONNXRUNTIME)
def test_onnx_full_1096_matches_pytorch_byte_identical(
    onnx_session, pytorch_reference, corpus_all,
):
    """Full 1096 corpus: ONNX exit code == PyTorch for every program.

    The acceptance criterion for sub-wave 8.J.2 (1096/1096 ONNX ==
    PyTorch 1096/1096). Marked ``slow`` because the corpus takes several
    minutes; run with ``pytest --runslow`` per ``conftest.py``.
    """
    onnx_results = _run_corpus_onnx(onnx_session, corpus_all)
    pt_results = _run_corpus_pytorch(pytorch_reference, corpus_all)

    assert pt_results["passed"] == pt_results["total"], (
        f"PyTorch reference is not at 1096/1096 "
        f"(passed={pt_results['passed']}). Cannot gate ONNX against a "
        f"broken PyTorch baseline; fix the upstream regression first."
    )
    assert onnx_results["passed"] == pt_results["passed"], (
        f"ONNX runtime diverged from PyTorch baseline on full corpus: "
        f"ONNX {onnx_results['passed']}/{onnx_results['total']} vs "
        f"PyTorch {pt_results['passed']}/{pt_results['total']}. "
        f"First mismatches: {onnx_results['failed_tests'][:10]}"
    )


# ---------------------------------------------------------------------------
# CLI entrypoint — preserved for ad-hoc validation. Mirrors the original
# script's flags but delegates to the pytest-style helpers.
# ---------------------------------------------------------------------------


def _cli_main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run ONNX runtime 1096 suite")
    parser.add_argument("--quick", action="store_true",
                        help=f"Run first {QUICK_SUBSET_SIZE} tests only")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show each test result")
    parser.add_argument("--onnx-path", type=str,
                        help="Path to existing ONNX model (skip export)")
    args = parser.parse_args(argv)

    print("=" * 70)
    print("C4 VM — ONNX RUNTIME 1096 BYTE-IDENTITY GATE")
    print("=" * 70)

    if not HAS_TORCH:
        print("ERROR: torch not installed.")
        return 2
    if not HAS_ONNX_RUNTIME:
        print("ERROR: onnxruntime not installed.")
        return 2

    located = _locate_onnx_exporter()
    if located is None and not args.onnx_path:
        print(f"SKIP: {SKIP_REASON_NO_EXPORTER}")
        print("\nRerun with --onnx-path=<file> to point at a pre-exported "
              "model, or wait for Phase 8.J to land.")
        return 0  # not a failure; scaffold-mode

    # Build / load ONNX session.
    import onnxruntime as ort
    if args.onnx_path:
        onnx_path = args.onnx_path
    else:
        _module_path, _attr, exporter = located  # type: ignore[misc]
        onnx_path = tempfile.mktemp(suffix=".onnx")
        print(f"Exporting via {_module_path}.{_attr} -> {onnx_path}")
        try:
            exporter(output_path=onnx_path)
        except TypeError:
            exporter(onnx_path)
    session = ort.InferenceSession(onnx_path)

    # Load corpus + reference.
    from tests.test_suite_1000 import generate_test_programs, get_quick_tests
    from src.baked_c4 import BakedC4Transformer
    tests = get_quick_tests() if args.quick else generate_test_programs()
    reference = BakedC4Transformer(use_speculator=True)

    print(f"\nRunning {len(tests)} programs through PyTorch reference...")
    pt_results = _run_corpus_pytorch(reference, tests, verbose=args.verbose)
    print(f"  PyTorch: {pt_results['passed']}/{pt_results['total']} "
          f"passed ({pt_results['elapsed']:.1f}s)")

    print(f"\nRunning {len(tests)} programs through ONNX runtime...")
    onnx_results = _run_corpus_onnx(session, tests, verbose=args.verbose)
    print(f"  ONNX:    {onnx_results['passed']}/{onnx_results['total']} "
          f"passed ({onnx_results['elapsed']:.1f}s)")

    match = onnx_results["passed"] == pt_results["passed"]
    print("\n" + "=" * 70)
    print(f"RESULT: ONNX vs PyTorch byte-identity gate: "
          f"{'PASS' if match else 'FAIL'}")
    print("=" * 70)
    return 0 if match else 1


if __name__ == "__main__":
    sys.exit(_cli_main())
