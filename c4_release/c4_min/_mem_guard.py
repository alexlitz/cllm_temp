#!/usr/bin/env python3
"""Run a target module in-process under an RSS watchdog thread.

Usage:
    python -m c4_min._mem_guard <max_gb> <module> [args...]

Spawns a daemon thread that polls this process's RSS every 2s; if it exceeds
``max_gb`` it prints a red banner and hard-kills the process (os._exit) so a
runaway model build can NEVER take the box down.  The target module's ``main``
(or ``__main__`` block) runs in the main thread with ``sys.argv`` rewritten to
``[module, *args]``.
"""
from __future__ import annotations
import os
import runpy
import sys
import threading
import time


def _rss_gb() -> float:
    try:
        with open(f"/proc/{os.getpid()}/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        pass
    return 0.0


def _watch(max_gb: float):
    peak = 0.0
    while True:
        rss = _rss_gb()
        peak = max(peak, rss)
        if rss > max_gb:
            sys.stderr.write(
                f"\n[MEM-GUARD] RSS {rss:.1f} GB > {max_gb:.1f} GB cap — ABORT\n")
            sys.stderr.flush()
            os._exit(137)
        time.sleep(2.0)


def main() -> int:
    max_gb = float(sys.argv[1])
    module = sys.argv[2]
    argv = sys.argv[3:]
    t = threading.Thread(target=_watch, args=(max_gb,), daemon=True)
    t.start()
    sys.argv = [module, *argv]
    runpy.run_module(module, run_name="__main__", alter_sys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
