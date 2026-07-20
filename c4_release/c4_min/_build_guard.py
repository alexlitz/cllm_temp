"""Memory-safety guard for the c4_min full-VM model builds.

Background
----------
``nibble_pure_forward_complete.build_pure_forward_complete_model`` (the DENSE
complete model) pads EVERY one of its ~37/305 transformer blocks to the GLOBAL
max hidden width — which is the ~160k-row 8-bit MUL/DIV/MOD select FFN
(``nibble_unified.compile_mdm_select``: one hidden unit per non-zero ``op(a,b)``
across MUL/DIV/MOD x 256 x 256).  So the dense build peaks at 54-108 GB RSS and
has repeatedly starved the box into swap when a guard / vanilla / smoke test
built it (e.g. ``test_exec_path_vanilla`` at ``code_size=16`` ballooned to
105 GB).

This module gives tests + tools a memory-SAFE way to obtain the SAME
byte-identical full-op-set interpreter, plus an RSS watchdog they can arm
around any build.

The memory-safe builds
----------------------
* ``build_compact_sparse_streaming`` (compact_alloc) — builds the FULL op set
  (every opcode, byte-identical L-inf=0 in ``dense_kernel`` mode) block-at-a-time
  as a streaming ``SparseTransformer``.  Peak RSS ~5 GB at ``code_size=44``.
  This is the drop-in replacement: the returned model is driven by the SAME
  ``run_pure_forward_complete`` runner as the dense model.
* ``build_pure_forward_model(include_muldiv=False)`` — the LEAN (no MUL/DIV/MOD)
  model.  A DIFFERENT (smaller) layout + its own ``run_pure_forward`` runner;
  use only when a test genuinely does not need the complete-model layout.

Use ``guarded_complete_build(code_size=...)`` to get the streaming full-op model
with an RSS watchdog armed for free.  A test that GENUINELY needs the dense
build (e.g. a dense-vs-sparse equivalence proof) must opt in via
``C4_ALLOW_DENSE_BUILD=1`` and is otherwise skipped by ``require_dense_build()``.
"""
from __future__ import annotations

import os
import threading
from typing import Callable, Optional, Tuple

# A DEFAULT (non-opt-in) model build must never cross this RSS ceiling.
DEFAULT_MAX_BUILD_GB = 20.0


def rss_gb(pid: Optional[int] = None) -> float:
    """Current resident-set size of ``pid`` (default: this process) in GiB."""
    pid = pid if pid is not None else os.getpid()
    try:
        with open(f"/proc/{pid}/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        pass
    return 0.0


def dense_build_allowed() -> bool:
    """True iff the opt-in ``C4_ALLOW_DENSE_BUILD`` env flag is set.

    The dense complete model peaks at 54-108 GB; it must never build on a
    default test/tool run.  A test that genuinely needs it (dense-vs-sparse
    equivalence) opts in with ``C4_ALLOW_DENSE_BUILD=1``.
    """
    return os.environ.get("C4_ALLOW_DENSE_BUILD", "0") not in ("0", "", "false", "False")


def require_dense_build():
    """pytest gate: SKIP unless the caller opted into the dense build.

    Returns a ``pytest.mark.skipif`` a module/test uses as::

        pytestmark = require_dense_build()   # module-level skipif
    """
    import pytest
    return pytest.mark.skipif(
        not dense_build_allowed(),
        reason="dense complete-model build (54-108 GB RSS) is opt-in: "
               "set C4_ALLOW_DENSE_BUILD=1 to run (memory hazard — see "
               "c4_min/_build_guard.py)",
    )


class RSSWatchdog:
    """Daemon-thread RSS watchdog.  Polls this process's RSS; if it crosses
    ``max_gb`` it records the trip AND (optionally) hard-aborts the process, so a
    runaway build can never take the box down.

    Preferred use is the ``guarded_build`` helper below, which also records the
    peak and, on a clean exit, ASSERTS the peak stayed under the ceiling (so a
    build that quietly grew past it fails the test rather than the machine)."""

    def __init__(self, max_gb: float = DEFAULT_MAX_BUILD_GB,
                 poll_s: float = 0.25, hard_abort: bool = False):
        self.max_gb = float(max_gb)
        self.poll_s = poll_s
        self.hard_abort = hard_abort
        self.peak_gb = 0.0
        self.tripped = False
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _run(self):
        base = rss_gb()
        while not self._stop.is_set():
            cur = rss_gb()
            if cur > self.peak_gb:
                self.peak_gb = cur
            if cur > self.max_gb and not self.tripped:
                self.tripped = True
                if self.hard_abort:
                    import sys
                    sys.stderr.write(
                        f"\n[C4-MEM-GUARD] RSS {cur:.1f} GB > {self.max_gb:.1f} GB "
                        f"cap (base {base:.1f}) — HARD ABORT\n")
                    sys.stderr.flush()
                    os._exit(137)
            self._stop.wait(self.poll_s)

    def start(self) -> "RSSWatchdog":
        self.peak_gb = rss_gb()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False


def guarded_build(fn: Callable, *args, max_gb: float = DEFAULT_MAX_BUILD_GB,
                  hard_abort: bool = False, **kwargs):
    """Run ``fn(*args, **kwargs)`` (a model build) under an RSS watchdog and
    ASSERT the peak RSS stayed under ``max_gb``.

    Returns ``fn``'s result.  Raises ``AssertionError`` (or hard-aborts if
    ``hard_abort``) if the build crossed the ceiling — so a default test can
    never silently trigger a >20 GB build."""
    wd = RSSWatchdog(max_gb=max_gb, hard_abort=hard_abort).start()
    try:
        result = fn(*args, **kwargs)
    finally:
        wd.stop()
    assert wd.peak_gb <= max_gb, (
        f"model build exceeded the {max_gb:.0f} GB RSS ceiling "
        f"(peak {wd.peak_gb:.1f} GB) — did you build the DENSE complete model? "
        f"Route to build_compact_sparse_streaming / include_muldiv=False.")
    return result


# The streaming build's empirical value-liveness pass is LOAD-BEARING for
# correctness (it separates values that are live at overlapping times so the
# dim-colouring does not over-share a slot — skipping it corrupts decode to
# 0xFF).  That pass runs the default probe programs (compiled C sources up to
# ~19 instructions), and ``overlay`` indexes ``L.CODE_OP[k]`` for the whole
# program, so ``code_size`` must be able to hold them.  A test that only runs
# tiny programs can pass a small ``code_size``; we transparently clamp it up to
# this floor (harmless — a larger CODE band, same op set) so the liveness pass
# runs and the model is correct.
_MIN_STREAM_CODE_SIZE = 20


def guarded_complete_build(code_size: int = 44,
                           recurrent_divmod: bool = False,
                           max_gb: float = DEFAULT_MAX_BUILD_GB) -> Tuple:
    """Memory-SAFE full-op-set complete model, drop-in for
    ``build_pure_forward_complete_model``.

    Builds via ``build_compact_sparse_streaming`` (peak ~5 GB), returns
    ``(model, L)`` — a streaming ``SparseTransformer`` + its remapped layout,
    driven by the SAME ``run_pure_forward_complete`` runner and byte-identical
    (L-inf=0, ``dense_kernel``) to the dense complete model.  The build runs
    under an RSS watchdog asserting the peak stays under ``max_gb``.

    ``code_size`` is clamped up to a floor (~20) so the load-bearing empirical
    value-liveness pass can run its probe programs; the extra CODE capacity is
    inert (the same op set, byte-identical results for any program that fits)."""
    from .compact_alloc import build_compact_sparse_streaming
    eff_code_size = max(int(code_size), _MIN_STREAM_CODE_SIZE)
    model, L, _stats = guarded_build(
        build_compact_sparse_streaming,
        code_size=eff_code_size, recurrent_divmod=recurrent_divmod,
        max_gb=max_gb)
    return model, L
