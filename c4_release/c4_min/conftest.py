"""pytest fixtures + machine-safety net for the c4_min test suite.

The MEMORY hazard this guards: the DENSE complete-model build
(``build_pure_forward_complete_model``) pads every block to the ~160k-row
MUL/DIV/MOD FFN and peaks at 54-108 GB RSS, which repeatedly starved the box
into swap when a guard/vanilla test triggered it.  The tests are routed to the
memory-safe streaming build (``build_compact_sparse_streaming``, peak ~5 GB);
this conftest is the LAST line of defence — an autouse watchdog that HARD-ABORTS
the process before the catastrophic dense build swaps the machine.

Two ceilings, deliberately different:

* ``guarded_build`` / ``guarded_complete_build`` (``_build_guard``) — the PER
  BUILD assertion for the ROUTED complete-model tests: peak must stay under
  20 GB (the streaming build peaks ~5 GB, so a >20 GB peak means the routing
  regressed back to the dense path).  This is a soft ``AssertionError`` failing
  that one test.
* This conftest's autouse watchdog — the MACHINE-SAFETY hard-abort.  Its ceiling
  is set ABOVE the documented compact-dense path (``build_compact_pure_forward_model``
  peaks ~48 GB — a separate, pre-existing family of compact/onnx/quine/moe tests)
  and BELOW the swap-inducing dense catastrophe (105 GB on this 125 GB box), so
  it kills a runaway dense build before it takes the box down without breaking
  the legitimate ~48 GB compact-build tests.

A test that GENUINELY needs the dense build opts in with
``C4_ALLOW_DENSE_BUILD=1`` (see ``_build_guard.require_dense_build``); the
watchdog then relaxes its ceiling so the sanctioned dense build can run.
"""
from __future__ import annotations

import pytest

from c4_min._build_guard import RSSWatchdog, dense_build_allowed

# Machine-safety hard-abort ceiling: above the documented ~48 GB compact-dense
# path, below the ~105 GB dense-complete catastrophe (which swaps a 125 GB box).
_MACHINE_SAFETY_GB = 60.0
# When a test opts into the dense build, give it enough headroom to finish.
_DENSE_OPT_IN_GB = 120.0


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "dense_build: test builds the DENSE complete model (54-108 GB RSS). "
        "Opt-in only; set C4_ALLOW_DENSE_BUILD=1 to run.")


@pytest.fixture(autouse=True)
def _rss_ceiling_guard():
    """Arm a MACHINE-SAFETY RSS watchdog around EVERY c4_min test.

    A default test that accidentally routes back to the DENSE complete build
    (105 GB) is hard-aborted before it swaps the machine.  The per-build 20 GB
    correctness assertion lives in ``guarded_build`` (used by the routed
    complete-model tests); this ceiling is the coarser machine-safety net that
    tolerates the pre-existing ~48 GB compact-build tests."""
    max_gb = _DENSE_OPT_IN_GB if dense_build_allowed() else _MACHINE_SAFETY_GB
    wd = RSSWatchdog(max_gb=max_gb, hard_abort=True).start()
    try:
        yield
    finally:
        wd.stop()
