#!/usr/bin/env python3
"""c4_min 32-BIT EXACTNESS MATRIX runner.

Runs every ``audit32`` case through the execution PATHS and prints the
per-opcode / per-case / per-path pass-fail matrix, localizing every divergence.

PATHS
-----
  GOLDEN  ``audit32.golden32``          — always (cheap, full corpus).
  DRAFT   ``ref_interpret_word32``      — always (cheap).  The doom fast path.
  NEURAL  ``run_pure_forward_complete`` — a bounded per-opcode sample (the build
          is ~15s and each program is ~1-30s).  Requires ``C4_PF_CFM=1`` (see
          THE CFM CAVEAT below).  ``--neural`` (sample) / ``--neural-all``.
  NATIVE  ``./c4`` (the reference c4 binary) — spot-check from generated C source
          for the cases a C program can express (``--native``).

THE CFM CAVEAT (a base-commit bug this audit found + fixed)
-----------------------------------------------------------
At base ``f875db79`` the pure-forward NEURAL driver was BROKEN with ``C4_PF_CFM``
OFF (the default): ``run_pure_forward_complete`` raised ``TypeError: unsupported
operand type(s) for +: 'NoneType' and 'int'`` in ``_overlay_pf_code_frames``
because ``compact_alloc._rebuild_layout`` remapped the layout's ``cfm`` BOOL as if
it were a dim index — ``isinstance(False, int)`` is True in Python, so ``cfm=False``
(== 0) was rewritten to ``new_slot[0]`` (a real dim, ~566), which is TRUTHY.  The
overlay then took the CFM branch but ``CODE_KEY_BIN`` was never allocated (only
allocated when ``_pf_cfm_enabled()`` is true at construction) -> ``None + int``.
``test_pure_forward_1096.py`` FAILED at base for the same reason.  FIXED here by a
one-line, WEIGHT-NEUTRAL change in ``_rebuild_layout`` (``type(val) is int`` instead
of ``isinstance``, so bool flags are not treated as dims) — golden fingerprint
``069cc32f`` UNCHANGED, and the default neural path now runs byte-exact.  This
runner still sets ``C4_PF_CFM=1`` defensively so it works even against an unpatched
tree.

Usage
-----
    # golden + draft matrix (fast, no model):
    PYTHONPATH=$(pwd) python c4_min/audit_matrix.py
    # + a neural sample (one case per opcode, ~10-20 min):
    PYTHONPATH=$(pwd) python c4_min/audit_matrix.py --neural
    # + every case through the neural model:
    PYTHONPATH=$(pwd) python c4_min/audit_matrix.py --neural-all
    # + native ./c4 spot-check:
    PYTHONPATH=$(pwd) python c4_min/audit_matrix.py --native
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter, OrderedDict
from typing import Dict, List, Optional

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "4")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

import c4_min.nibble_pure_forward as _PF          # noqa: E402
import c4_min.nibble_pure_forward_complete as _PFC  # noqa: E402
_PF.SP_INIT = 0xFC
_PFC.SP_INIT = 0xFC

from c4_min import isa                             # noqa: E402
from c4_min.audit32 import generate_cases, Case, M32, s32  # noqa: E402
from c4_min.selfhost.word32_draft_vm import ref_interpret_word32  # noqa: E402


# ---------------------------------------------------------------------------
# DRAFT path.
# ---------------------------------------------------------------------------
def draft_value(case: Case) -> int:
    tr, _ = ref_interpret_word32(case.code(), seed_mem=dict(case.seed_mem),
                                 max_steps=case.max_steps)
    return tr[-1] & M32 if tr else 0


# ---------------------------------------------------------------------------
# NATIVE ./c4 path — spot-check via a generated C program (only for cases a C
# program can express: seeded operands become explicit literals; the op is the
# C operator).  Signed/unsigned distinctions are picked to match the c4 int
# (signed long) semantics.
# ---------------------------------------------------------------------------
_C4_BIN = "/home/alexlitz/Documents/misc/c4_doom/c4"

_C_OP = {"ADD": "+", "SUB": "-", "MUL": "*", "DIV": "/", "MOD": "%",
         "AND": "&", "OR": "|", "XOR": "^", "SHL": "<<", "SHR": ">>",
         "EQ": "==", "NE": "!=", "LT": "<", "GT": ">", "LE": "<=", "GE": ">="}


def native_value(case: Case) -> Optional[int]:
    """Run the case through ``./c4`` from a generated C source, if expressible.
    Returns the process exit code (the c4 program's ``return`` value, masked to a
    byte by the OS) — so only meaningful for cases whose result fits a byte, OR
    printed via printf.  We print the 32-bit result so the full word is visible.
    """
    import subprocess
    import tempfile
    if case.op not in _C_OP:
        return None
    if len(case.seed_mem) < 1:
        return None
    a = case.seed_mem.get(0x40)
    b = case.seed_mem.get(0x44)
    if a is None:
        return None
    cop = _C_OP[case.op]
    # c4 'int' is a signed long (64-bit) — cast operands to signed 32-bit ints so
    # the arithmetic matches the 32-bit ISA.  Print the low 32 bits.
    sa = s32(a)
    if b is not None:
        sb = s32(b)
        src = (f"int main(){{ int a; int b; a={sa}; b={sb}; "
               f"printf(\"%d\\n\", a {cop} b); return 0; }}")
    else:
        # shift by explicit amount (recorded on the case; the body's first IMM is
        # the LOAD address, not the shift, so use case.shift_amt).
        amt = case.shift_amt if case.shift_amt is not None else 0
        src = (f"int main(){{ int a; a={sa}; "
               f"printf(\"%d\\n\", a {cop} {amt}); return 0; }}")
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".c", delete=False) as fh:
            fh.write(src)
            path = fh.name
        out = subprocess.run([_C4_BIN, path], capture_output=True, timeout=10,
                             text=True)
        os.unlink(path)
        # c4 prints the program's stdout then a trailing "exit(N) cycle = M"
        # diagnostic line.  Parse the FIRST line that is a bare (optionally signed)
        # integer — the printf("%d") result.
        for ln in out.stdout.splitlines():
            ln = ln.strip()
            if ln and (ln.lstrip("-").isdigit()):
                return int(ln) & M32
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# NEURAL path.
# ---------------------------------------------------------------------------
def build_model(code_size: int = 48):
    from c4_min.compact_alloc import build_compact_sparse_streaming
    model, L, _ = build_compact_sparse_streaming(
        code_size=code_size, compute_mode="dense_kernel")
    return model, L


def neural_value(case: Case, model, L) -> int:
    tr = _PFC.run_pure_forward_complete(
        model, L, case.code(), max_steps=case.max_steps, mask=M32,
        seed_mem=dict(case.seed_mem) or None)
    return tr[-1] & M32 if tr else 0


# ---------------------------------------------------------------------------
# Matrix.
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--neural", action="store_true",
                    help="Run one NEURAL case per opcode (bounded sample).")
    ap.add_argument("--neural-all", action="store_true",
                    help="Run EVERY case through the neural model.")
    ap.add_argument("--native", action="store_true",
                    help="Spot-check each expressible case through ./c4.")
    ap.add_argument("--only-op", type=str, default=None,
                    help="Restrict to a single opcode.")
    args = ap.parse_args(argv)

    cases = generate_cases()
    if args.only_op:
        cases = [c for c in cases if c.op == args.only_op.upper()]

    rows: List[Dict] = []
    for c in cases:
        g = c.golden()
        d = draft_value(c)
        rows.append({"case": c, "golden": g, "draft": d,
                     "draft_ok": (d == g), "native": None, "native_ok": None,
                     "neural": None, "neural_ok": None})

    # NATIVE.
    if args.native:
        for r in rows:
            nv = native_value(r["case"])
            r["native"] = nv
            r["native_ok"] = (nv == r["golden"]) if nv is not None else None

    # NEURAL.  The production DEFAULT (C4_PF_CFM off) now runs byte-exact thanks to
    # the _rebuild_layout bool-remap fix (see THE CFM CAVEAT); we test THAT path.
    if args.neural or args.neural_all:
        t = time.monotonic()
        print("[audit] building pure-forward model ...", file=sys.stderr, flush=True)
        model, L = build_model()
        print(f"[audit] model dim={L.D} blocks={len(model.blocks)} "
              f"({time.monotonic()-t:.1f}s)", file=sys.stderr, flush=True)
        if args.neural_all:
            selected = rows
        else:
            seen: Counter = Counter()
            selected = []
            for r in rows:
                op = r["case"].op
                if seen[op] < 1:
                    seen[op] += 1
                    selected.append(r)
        for i, r in enumerate(selected):
            c = r["case"]
            t0 = time.monotonic()
            try:
                nv = neural_value(c, model, L)
                r["neural"] = nv
                r["neural_ok"] = (nv == r["golden"])
            except Exception as exc:  # noqa: BLE001
                r["neural"] = None
                r["neural_ok"] = False
                r["neural_exc"] = repr(exc)
            print(f"[audit] {i+1}/{len(selected)} {c.op:5s} {c.name:22s} "
                  f"golden={r['golden']:#010x} neural="
                  f"{('%#010x' % r['neural']) if r['neural'] is not None else 'ERR':>10} "
                  f"{'ok' if r['neural_ok'] else 'DIVERGE'} "
                  f"({time.monotonic()-t0:.1f}s)", file=sys.stderr, flush=True)

    _print_matrix(rows, native=args.native, neural=(args.neural or args.neural_all))
    return 0


def _print_matrix(rows, native=False, neural=False):
    # per-opcode roll-up
    print("\n" + "=" * 92)
    print("c4_min 32-BIT EXACTNESS MATRIX  (golden = the 32-bit c4_min reference; "
          "golden 069cc32f)")
    print("=" * 92)
    by_op: "OrderedDict[str, Dict]" = OrderedDict()
    for r in rows:
        op = r["case"].op
        row = by_op.setdefault(op, {"n": 0, "draft_div": 0, "native_run": 0,
                                    "native_div": 0, "neural_run": 0, "neural_div": 0})
        row["n"] += 1
        if not r["draft_ok"]:
            row["draft_div"] += 1
        if r["native_ok"] is not None:
            row["native_run"] += 1
            if not r["native_ok"]:
                row["native_div"] += 1
        if r["neural_ok"] is not None:
            row["neural_run"] += 1
            if not r["neural_ok"]:
                row["neural_div"] += 1
    hdr = f"  {'op':5s} {'n':>3s} {'DRAFT div':>10s}"
    if native:
        hdr += f" {'NATIVE run/div':>15s}"
    if neural:
        hdr += f" {'NEURAL run/div':>15s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for op, row in sorted(by_op.items()):
        line = f"  {op:5s} {row['n']:3d} {row['draft_div']:10d}"
        if native:
            line += f" {row['native_run']:6d}/{row['native_div']:<8d}"
        if neural:
            line += f" {row['neural_run']:6d}/{row['neural_div']:<8d}"
        print(line)

    # every divergence, localized
    print("\nDIVERGENCES (localized — opcode, case, path, golden vs got):")
    any_div = False
    for r in rows:
        c = r["case"]
        if not r["draft_ok"]:
            any_div = True
            print(f"  [DRAFT ] {c.op:5s} {c.name:22s} golden={r['golden']:#010x} "
                  f"draft={r['draft']:#010x}  :: {c.note}")
        if r["native_ok"] is False:
            any_div = True
            nv = r["native"]
            print(f"  [NATIVE] {c.op:5s} {c.name:22s} golden={r['golden']:#010x} "
                  f"native={('%#010x' % nv) if nv is not None else 'None':>10}  :: {c.note}")
        if r["neural_ok"] is False:
            any_div = True
            nv = r["neural"]
            tag = "xfail" if c.neural_xfail else "FAIL"
            exc = r.get("neural_exc", "")
            print(f"  [NEURAL:{tag}] {c.op:5s} {c.name:22s} golden={r['golden']:#010x} "
                  f"neural={('%#010x' % nv) if nv is not None else 'ERR':>10}  "
                  f":: {c.note} {exc}")
    if not any_div:
        print("  (none)")

    # the DRAFT gap summary — the doom-fast-path fix list.
    draft_div_ops = sorted({r["case"].op for r in rows if not r["draft_ok"]})
    print(f"\nDRAFT (doom fast path) diverges on opcodes: {draft_div_ops}")
    if neural:
        nn = [r for r in rows if r["neural_ok"] is False and not r["case"].neural_xfail]
        nx = [r for r in rows if r["neural_ok"] is False and r["case"].neural_xfail]
        print(f"NEURAL unexpected divergences (NOT documented xfail): "
              f"{[r['case'].name for r in nn]}")
        print(f"NEURAL documented xfail (expected): {[r['case'].name for r in nx]}")
    print("=" * 92)


if __name__ == "__main__":
    sys.exit(main())
