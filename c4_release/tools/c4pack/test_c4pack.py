"""Pytest wrapper for the c4pack byte-exact verification harness.

Wraps `verify_c4pack`: for each demo program it packages the source into a
STANDALONE static binary (no Python/torch), runs the actual compiled binary,
and compares its stdout byte-for-byte against gcc -m32 -std=c90 (always) and
the RefVM oracle (pure-compute/printf programs). Nine programs yield 11
(program, oracle) checks; this asserts all 11/11 are byte-exact and the
harness exit code is 0.

Requires gcc with -m32 multilib (the harness falls back to native gcc only
if -m32 is unavailable). CPU-safe: no model load.
"""
from __future__ import annotations

import io
import re
import shutil
from contextlib import redirect_stdout

import pytest

from tools.c4pack import verify_c4pack

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None,
                                reason="gcc required to build the reference binaries")


def test_c4pack_11_of_11_byte_exact():
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = verify_c4pack.main()
    out = buf.getvalue()

    fails = [ln for ln in out.splitlines() if ln.startswith("[FAIL]")]
    assert not fails, "c4pack byte-exact checks failed:\n" + "\n".join(fails)

    m = re.search(r"byte-exact checks:\s*(\d+)/(\d+)", out)
    assert m is not None, "no tally line found:\n" + out
    passed, total = int(m.group(1)), int(m.group(2))
    assert (passed, total) == (11, 11), f"expected 11/11, got {passed}/{total}\n{out}"
    assert rc == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
