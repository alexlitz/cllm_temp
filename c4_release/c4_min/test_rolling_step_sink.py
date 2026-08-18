"""Pytest wrapper for the ROLLING PER-STEP SINK evidence battery.

Wraps `verify_rolling_step_sink.run()` (BLOG_SPEC §Memory, MEASURED against
the real §Memory CAM head — plain softmax + ALiBi, no softmax1/exemption for
the rolling variant). The battery emits 23 named checks; this asserts all
23/23 PASS and that `run()` returns True.

CPU-safe: no model rebuild, peak RSS ~760 MB (well under the 4 GB cap).
"""
from __future__ import annotations

import io
import re
from contextlib import redirect_stdout

from c4_min import verify_rolling_step_sink as vrss


def test_rolling_step_sink_23_of_23():
    buf = io.StringIO()
    with redirect_stdout(buf):
        ok = vrss.run()
    out = buf.getvalue()

    # No individual FAIL line.
    fails = [ln for ln in out.splitlines() if "[FAIL]" in ln]
    assert not fails, "rolling-step-sink checks failed:\n" + "\n".join(fails)

    # Explicit N/N cited from the battery's own summary line.
    m = re.search(r"====\s*(\d+)/(\d+)\s+PASS", out)
    assert m is not None, "no PASS summary line found:\n" + out
    npass, ntot = int(m.group(1)), int(m.group(2))
    assert (npass, ntot) == (23, 23), f"expected 23/23, got {npass}/{ntot}"
    assert ok is True


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
