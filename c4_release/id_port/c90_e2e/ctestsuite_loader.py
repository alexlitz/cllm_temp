#!/usr/bin/env python3
"""Load the c-testsuite single-exec C90 cases as BROAD-corpus entries.

The c-testsuite (https://github.com/c-testsuite/c-testsuite) is the standard
collaborative C-compiler conformance database.  Its ``tests/single-exec/*.c``
suite is exactly the shape our SOURCE-path transpiler harness needs: each case
is a self-contained C program whose entry is ``main`` and whose observable is
stdout+stderr with exit 0 on success.  We reuse it here to turn the 117
hand-written cases into a BROAD (hundreds-of-cases) standard C90 conformance
suite for the ``transpile.py -> compile_c -> c4vm`` port pipeline.

Ground truth = ``gcc -m32 -std=c90`` (the same oracle the hand-written corpus
uses), NOT the shipped ``.c.expected`` — so a case whose observable legitimately
depends on native behaviour is still measured against the same x86 reference the
Doom port targets.

This loader:
  * reads every ``NNNNN.c`` under a c-testsuite checkout,
  * auto-classifies each by DOMINANT construct (multidim / structarr / union /
    enum / fnptr / varargs / bitwise / loops / switch / goto / recursion /
    strings / storage / preproc / ptrarith / misc),
  * flags OUT-OF-SUBSET cases (function pointers, user varargs, float/double,
    long-long) so they are counted as by-design boundary, not transpiler bugs,
  * emits ``(name, category, source, expect_mod256, sizeof_gap)`` tuples in the
    exact shape ``transpiler_corpus.CORPUS`` uses.

Pure ADDITION under id_port/c90_e2e/; no model-build file touched.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

# default checkout location (override with C4_CTESTSUITE_DIR)
CTESTSUITE_DIR = os.environ.get(
    "C4_CTESTSUITE_DIR",
    "/tmp/c-testsuite/tests/single-exec")


# ---- out-of-subset feature detectors (by-design boundary, not bugs) --------
_RE_FLOAT   = re.compile(r'\b(float|double)\b')
_RE_LONGLONG = re.compile(r'\blong\s+long\b')
_RE_FNPTR   = re.compile(r'\(\s*\*\s*[A-Za-z_]\w*\s*\)\s*\(')          # (*f)(...)
_RE_FNPTR2  = re.compile(r'\(\s*\*\s*\)\s*\(')                          # (*)(...)  in casts/typedefs
_RE_VARARG  = re.compile(r'\.\.\.\s*\)')
_RE_VA      = re.compile(r'\bva_(list|start|arg|end)\b')
_RE_TYPEDEF_FNPTR = re.compile(r'typedef\b[^;]*\(\s*\*\s*[A-Za-z_]\w*\s*\)\s*\(')


# long-long is NOW ADDRESSABLE (#920): the transpiler's softint64 pass
# (id_port/softint64.py) lowers `long long`/`unsigned long long`/`int64_t`/
# `uint64_t` to TWO 32-bit words on the native-32-bit machine (add/sub-carry,
# schoolbook mul, binary long-division, word-boundary shifts, per-word bitwise,
# hi-first signed+unsigned compares, int<->ll conversions).  So `longlong` is no
# longer an auto-OUT-OF-SUBSET boundary -- it counts IN-SUBSET.  Set
# ``C4_LONGLONG_OOB=1`` to restore the legacy (pre-softint64) classification for a
# before/after A-B comparison.
_LONGLONG_IS_OOB = bool(os.environ.get("C4_LONGLONG_OOB"))


def out_of_subset_reason(src: str) -> Optional[str]:
    """Return a short OUT-OF-SUBSET reason (B3/B9/float) or None.

    These are the constructs Doom deliberately sidesteps; a failure on them is
    a documented subset-boundary, NOT a transpiler bug.  ``long long`` is NO
    LONGER here -- softint64 makes it addressable (see the module note)."""
    if _RE_FNPTR.search(src) or _RE_FNPTR2.search(src) or _RE_TYPEDEF_FNPTR.search(src):
        # function pointers -> B3 (Doom uses the FN_ID __actions__ dispatch)
        return "B3_fnptr"
    if _RE_VA.search(src) or (_RE_VARARG.search(src) and "va_list" in src):
        return "B9_varargs"
    if _RE_VARARG.search(src) and not re.search(r'\b(printf|scanf|fprintf|sprintf|snprintf|fscanf|sscanf)\b', src):
        # user-declared vararg fn (not a libc call)
        return "B9_varargs"
    if _RE_FLOAT.search(src):
        return "float"          # c4 is an integer word VM; floats unsupported
    if _LONGLONG_IS_OOB and _RE_LONGLONG.search(src):
        # legacy A-B mode only (C4_LONGLONG_OOB=1): pre-softint64 classification.
        return "longlong"
    return None


# ---- dominant-construct classifier -----------------------------------------
def classify(src: str) -> str:
    """Assign the DOMINANT construct category (matches transpiler_corpus cats)."""
    s = src
    # order matters: most-specific first
    if re.search(r'\[\s*\w+\s*\]\s*\[\s*\w+\s*\]', s) or re.search(r'\]\s*\[\s*\w', s):
        return "multidim"
    if re.search(r'\bunion\b', s):
        return "union"
    if re.search(r'\benum\b', s):
        return "enum"
    if re.search(r'\bstruct\b', s):
        return "structarr"
    if re.search(r'\bswitch\b', s):
        return "switch"
    if re.search(r'\bgoto\b', s):
        return "goto"
    if re.search(r'\bva_arg\b|\.\.\.', s):
        return "varargs"
    if re.search(r'\(\s*\*\s*\w', s):
        return "fnptr"
    if re.search(r'[|&^]=|<<=|>>=|\+=|-=|\*=|/=|%=', s):
        return "compound"
    if re.search(r'<<|>>|(?<![&|])&(?![&])|(?<![|])\|(?![|])|\^|~', s):
        return "bitwise"
    if re.search(r'\bfor\b|\bwhile\b|\bdo\b', s):
        # recursion vs generic loop
        # crude self-call detection
        fns = re.findall(r'\b([A-Za-z_]\w*)\s*\([^;{)]*\)\s*\{', s)
        for fn in fns:
            body_call = re.search(r'\b%s\s*\(' % re.escape(fn), s)
            if fn != 'main' and len(re.findall(r'\b%s\s*\(' % re.escape(fn), s)) >= 2:
                return "recursion"
        return "loops"
    if re.search(r'"[^"]*"', s) and re.search(r'\bchar\b', s):
        return "strings"
    if re.search(r'\bstatic\b|=\s*\{|^\s*(int|char|long|short|unsigned)\s+\w+\s*=', s, re.M):
        return "storage"
    if re.search(r'\*\s*\w+|\[\s*\w*\s*\]', s):
        return "ptrarith"
    return "misc"


def _needs_preproc(src: str, tags: str) -> bool:
    return "needs-cpp" in tags or "#define" in src or "#include" in src


def load_cases(directory: Optional[str] = None,
               limit: Optional[int] = None) -> List[Tuple]:
    """Return CORPUS-shaped tuples for the c-testsuite single-exec cases.

    Each: (name, category, source, expect_mod256=None, sizeof_gap=False).
    expect is None -> the harness uses the gcc oracle for the expected value.
    The out-of-subset reason is folded into the NAME suffix so the runner can
    bucket by-design fails (name form: ``cts_NNNNN`` or ``cts_NNNNN__B3_fnptr``).
    """
    d = Path(directory or CTESTSUITE_DIR)
    if not d.exists():
        return []
    out = []
    files = sorted(d.glob("*.c"))
    for f in files:
        base = f.name[:-2]           # strip ".c"
        src = f.read_text(errors="replace")
        tags = ""
        tagf = d / (f.name + ".tags")
        if tagf.exists():
            tags = tagf.read_text()
        oos = out_of_subset_reason(src)
        cat = classify(src)
        name = "cts_" + base
        if oos:
            name = name + "__" + oos
        # sizeof gap: cases that print or return sizeof(int)/pointer-size-dependent
        sizeof_gap = bool(re.search(r'sizeof\s*\(\s*(int|long|unsigned|void\s*\*|\w+\s*\*)', src))
        out.append((name, cat, src, None, sizeof_gap))
        if limit and len(out) >= limit:
            break
    return out


if __name__ == "__main__":
    import collections
    cases = load_cases()
    print(f"loaded {len(cases)} c-testsuite cases from {CTESTSUITE_DIR}")
    bycat = collections.Counter(c[1] for c in cases)
    oos = collections.Counter(c[0].split("__", 1)[1] for c in cases if "__" in c[0])
    print("by category:", dict(bycat))
    print("out-of-subset:", dict(oos))
    print("in-subset:", sum(1 for c in cases if "__" not in c[0]))
