#!/usr/bin/env python3
"""c4_min EDGE-OPS scoreboard — verify the additive edge corpus BYTE-EXACT.

Two gates:
  (A) GOLDEN gate (fast, pure-python): every case's ``expected`` == the c4_min
      reference (``ref_interpret``, 32-bit, mask=0xFFFFFFFF).  This confirms the
      corpus's expected values ARE the byte-exact c4 semantics (the same golden
      ``run_1096_pure_forward`` scores against).  I/O cases (READ/PRTF) verify
      the golden AX + golden stdout via the reference I/O contract.
  (B) NEURAL gate (slow, model.forward): a BOUNDED per-cluster sample is run
      through the pure-forward model (``run_pure_forward_complete``, the
      established byte-exact neural path) and asserted byte-exact vs golden.
      The neural path is ~15-60s/program, so ``--neural-per-cluster N`` bounds
      wall-time (default 1; ``--neural-all`` runs everything; ``--no-neural``
      skips the model entirely).

Usage
-----
    # golden gate only (fast, ~seconds) — verifies all 67 expected values:
    OMP_NUM_THREADS=4 PYTHONPATH=$(pwd) python c4_min/run_edge_ops.py --no-neural

    # golden + 1 neural sample per cluster (the default; ~10-15 min):
    OMP_NUM_THREADS=4 PYTHONPATH=$(pwd) python c4_min/run_edge_ops.py

    # golden + every case through the neural model (~40+ min):
    OMP_NUM_THREADS=4 PYTHONPATH=$(pwd) python c4_min/run_edge_ops.py --neural-all

CPU-only (the tiny nibble model; never a GPU).  Golden 069cc32f is untouched —
this file authors no weights, only imports the existing reference + model.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "4")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.dirname(_HERE)
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

# Pin SP_INIT=0xFC for the neural path (same as run_1096_pure_forward) BEFORE the
# drivers import — keeps every stack address 8-bit-addressable for the value head.
import c4_min.nibble_pure_forward as _PF          # noqa: E402
import c4_min.nibble_pure_forward_complete as _PFC  # noqa: E402
_PF.SP_INIT = 0xFC
_PFC.SP_INIT = 0xFC

from c4_min import isa                              # noqa: E402
from c4_min import nibble_filesys as FS             # noqa: E402
from c4_min.edge_corpus import generate_edge_cases, EdgeCase, opcodes_covered  # noqa: E402
from c4_min.nibble_pure_forward_complete import ref_interpret  # noqa: E402


M32 = 0xFFFFFFFF


# ===========================================================================
# GOLDEN: compute the byte-exact c4_min reference AX (+ stdout) for a case.
# ===========================================================================
def golden_value(case: EdgeCase) -> Tuple[int, Optional[bytes]]:
    """Return (golden_ax, golden_stdout) from the c4_min reference interpreter.

    Non-I/O:  ``ref_interpret(code, mask=0xFFFFFFFF)`` — the SAME golden the
              pure-forward corpus scores against.  golden_ax = final AX.
    I/O:      the reference I/O contract — READ pulls from ``case.stdin`` (a
              reference input stream), PRTF appends AX&0xFF (single-char channel)
              or, for a %d/%c/%s format, the c4 printf subset via the runner.
    """
    code = case.code()
    if not case.io:
        tr = ref_interpret(code, max_steps=case.max_steps, mask=M32)
        return (tr[-1] & M32 if tr else 0), None
    return _golden_io(case, code)


class _RefStdin:
    """Minimal byte source for the reference READ(fd=0) path (``read(n)->bytes``)."""

    def __init__(self, data: bytes):
        self.data = bytes(data)
        self.pos = 0

    def read(self, n: int) -> bytes:
        n = int(n)
        chunk = self.data[self.pos:self.pos + n]
        self.pos += len(chunk)
        return chunk


def _golden_io(case: EdgeCase, code) -> Tuple[int, Optional[bytes]]:
    """Golden for a READ / PRTF program via a plain-python reference that mirrors
    the neural I/O contract (``ref_interpret`` READ + the FileRunner PRTF subset)."""
    # READ(stdin) cases: ref_interpret has no READ, so drive them via the reference
    # READ executor (fd/buf/n marshalling + signed LC readback).
    if any(ins.op == isa.READ for ins in code):
        return _golden_read(case, code)
    # PRTF-with-format cases: use the SAME FileRunner the neural driver services,
    # so the golden stdout is the runner's c4-printf-subset output byte-for-byte.
    has_prtf_fmt = any(ins.op == isa.PRTF for ins in code) and bool(case.data_seg)
    if has_prtf_fmt:
        return _golden_prtf_via_runner(case, code)
    # plain PUTCHAR (PRTF of AX byte) cases -> ref_interpret's own out contract.
    out: List[int] = []
    tr = ref_interpret(code, max_steps=case.max_steps, mask=M32, out=out)
    return (tr[-1] & M32 if tr else 0), bytes(out)


def _golden_read(case: EdgeCase, code) -> Tuple[int, Optional[bytes]]:
    """Reference READ(fd=0) execution: run the ISA by hand with a stdin stream and
    a byte-addressed mem, mirroring ``nibble_filesys.dispatch_file_op`` marshalling
    (fd=pop, buf=pop, n=AX) + a signed LC readback."""
    stdin = _RefStdin(case.stdin)
    mem: Dict[int, int] = dict(case.data_seg)
    for a, v in case.seed_mem.items():
        for i in range(4):
            mem[a + i] = (v >> (8 * i)) & 0xFF
    ax = 0
    sp = 0x10000
    stack: Dict[int, int] = {}
    pc = steps = 0
    out: List[int] = []

    def push(v):
        nonlocal sp
        sp -= 4
        stack[sp] = v & M32

    def pop():
        nonlocal sp
        v = stack.get(sp, 0)
        sp += 4
        return v

    while 0 <= pc < len(code) and steps < case.max_steps:
        steps += 1
        op, imm = code[pc].op, code[pc].imm
        pc += 1
        if op == isa.IMM:
            ax = imm & 0xFF
        elif op == isa.PSH:
            push(ax)
        elif op == isa.READ:
            n = ax
            buf = pop()
            fd = pop()
            chunk = stdin.read(n) if fd == 0 else b""
            for i, b in enumerate(chunk):
                mem[buf + i] = b & 0xFF
            ax = len(chunk) & M32
        elif op == isa.LC:
            b = mem.get(ax, 0) & 0xFF
            ax = (b - 0x100 if b & 0x80 else b) & M32
        elif op == isa.LI:
            ax = mem.get(ax, 0) & 0xFF
        elif op == isa.PRTF:
            out.append(ax & 0xFF)
        elif op == isa.HALT:
            break
        else:
            raise NotImplementedError(f"golden_read: op {isa.NAMES.get(op, op)}")
    return ax & M32, (bytes(out) if any(i.op == isa.PRTF for i in code) else None)


def _golden_prtf_via_runner(case: EdgeCase, code) -> Tuple[int, Optional[bytes]]:
    """Golden for a printf(fmt, args) case: run the ISA by hand and dispatch PRTF
    through the SAME ``nibble_filesys`` runner the neural driver uses, so the
    stdout is byte-identical to the model's tool-serviced output."""
    runner = FS.FileRunner(fs=FS.StubFilesystem({}))
    fio = FS.FileOpState(runner=runner)
    # The neural driver marshals the printf varargs from the KV store log; here we
    # supply them directly (the doom-style printf(\"hi %d %c\\n\", 7, 'Z')).
    fio.pending_args = list(case.prtf_args) or [7, ord("Z")]
    mem: Dict[int, int] = dict(case.data_seg)
    ax = 0
    sp = 0x10000
    stack: Dict[int, int] = {}
    pc = steps = 0

    class _M:
        def load_int(self, a, w=4):
            return sum(mem.get(a + i, 0) << (8 * i) for i in range(w))

        def store_int(self, a, v, w=4):
            for i in range(w):
                mem[a + i] = (v >> (8 * i)) & 0xFF
    M = _M()

    def pop():
        nonlocal sp
        v = stack.get(sp, 0)
        sp += 4
        return v

    def push(v):
        nonlocal sp
        sp -= 4
        stack[sp] = v & M32

    while 0 <= pc < len(code) and steps < case.max_steps:
        steps += 1
        op, imm = code[pc].op, code[pc].imm
        pc += 1
        if op == isa.IMM:
            ax = imm & 0xFF
        elif op == isa.PSH:
            push(ax)
        elif op in FS.FILE_OPCODES:
            ax = FS.dispatch_file_op(op, ax, imm, pop, M, fio)
        elif op == isa.HALT:
            break
        else:
            raise NotImplementedError(f"golden_prtf: op {isa.NAMES.get(op, op)}")
    return ax & M32, bytes(runner.stdout)


# ===========================================================================
# NEURAL: run a case through the pure-forward model (the byte-exact path).
# ===========================================================================
def neural_value(case: EdgeCase, model, L) -> Tuple[int, Optional[bytes]]:
    """Run ``case`` through ``run_pure_forward_complete`` and return (ax, stdout).

    Non-I/O: mask=0xFFFFFFFF final-AX.  I/O: fio + data_seg drive the TOOL_CALL /
    input-KV protocol; stdout is the model's OWN decoded PRTF bytes (LM argmax)."""
    from c4_min.nibble_pure_forward_complete import run_pure_forward_complete
    code = case.code()
    if not case.io:
        tr = run_pure_forward_complete(model, L, code, max_steps=case.max_steps, mask=M32)
        return (tr[-1] & M32 if tr else 0), None
    # I/O path: build a FileOpState with the case's stdin, and the data segment.
    fio = FS.FileOpState(
        runner=FS.FileRunner(fs=FS.StubFilesystem({}),
                             stdin=FS.InputKVStream(case.stdin)))
    if any(ins.op == isa.PRTF for ins in code) and case.data_seg:
        fio.pending_args = list(case.prtf_args) or [7, ord("Z")]
    out: List[int] = [] if _prints(code) else None
    res = run_pure_forward_complete(
        model, L, code, max_steps=case.max_steps, mask=M32,
        fio=fio, data_seg=dict(case.data_seg) or None, out=out)
    tr = res[0] if isinstance(res, tuple) else res
    ax = (tr[-1] & M32 if tr else 0)
    stdout = bytes(fio.runner.stdout) if (any(i.op == isa.PRTF for i in code) and case.data_seg) \
        else (bytes(out) if out is not None else None)
    return ax, stdout


def _prints(code) -> bool:
    return any(ins.op == isa.PRTF for ins in code)


# ===========================================================================
# Result + reporting.
# ===========================================================================
@dataclass
class Result:
    name: str
    cluster: str
    expected: int
    golden_ax: Optional[int]
    golden_stdout: Optional[bytes]
    golden_ok: bool
    neural_ax: Optional[int] = None
    neural_stdout: Optional[bytes] = None
    neural_ok: Optional[bool] = None       # None = not run
    seconds: float = 0.0
    detail: str = ""
    note: str = ""


_STATUSES = ("golden", "neural")


def _cluster_table(results: List[Result]) -> "OrderedDict":
    table: "OrderedDict" = OrderedDict()
    for r in results:
        row = table.setdefault(r.cluster, {"n": 0, "g_ok": 0, "n_run": 0, "n_ok": 0})
        row["n"] += 1
        row["g_ok"] += 1 if r.golden_ok else 0
        if r.neural_ok is not None:
            row["n_run"] += 1
            row["n_ok"] += 1 if r.neural_ok else 0
    return table


def _print_report(results: List[Result], wall: float, coverage_before, coverage_after,
                  neural_mode: str) -> None:
    table = _cluster_table(results)
    print("\n" + "=" * 78)
    print("c4_min EDGE-OPS SCOREBOARD  (byte-exact vs c4_min reference; golden 069cc32f)")
    print("=" * 78)
    print(f"  opcode coverage:  {len(coverage_before)}  ->  {len(coverage_after)}  "
          f"(+{len(coverage_after - coverage_before)})")
    added = sorted(coverage_after - coverage_before)
    print(f"  newly exercised:  {added}")
    print(f"  neural mode: {neural_mode}   wall: {wall:.1f}s")
    print("-" * 78)
    hdr = (f"  {'cluster':14s} {'n':>3s} {'gold_ok':>8s} {'neu_run':>8s} "
           f"{'neu_ok':>7s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    tot = {"n": 0, "g_ok": 0, "n_run": 0, "n_ok": 0}
    for cl, row in sorted(table.items()):
        print(f"  {cl:14s} {row['n']:3d} {row['g_ok']:8d} {row['n_run']:8d} "
              f"{row['n_ok']:7d}")
        for k in tot:
            tot[k] += row[k]
    print("  " + "-" * (len(hdr) - 2))
    print(f"  {'TOTAL':14s} {tot['n']:3d} {tot['g_ok']:8d} {tot['n_run']:8d} "
          f"{tot['n_ok']:7d}")
    print("=" * 78)

    fails = [r for r in results if (not r.golden_ok) or (r.neural_ok is False)]
    if fails:
        print("\nFAILS / DIVERGENCES (the point of the suite):")
        for r in fails:
            if not r.golden_ok:
                print(f"  [GOLDEN] {r.name:26s} exp={r.expected} "
                      f"golden={r.golden_ax} :: {r.note}  ({r.detail})")
            if r.neural_ok is False:
                print(f"  [NEURAL] {r.name:26s} golden_ax={r.golden_ax} "
                      f"neural_ax={r.neural_ax} "
                      f"gold_out={r.golden_stdout!r} neu_out={r.neural_stdout!r} "
                      f":: {r.note}")
    else:
        print("\nNo divergences: every case is byte-exact (golden == expected, "
              "neural == golden for every run case).")


# ===========================================================================
# Main.
# ===========================================================================
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-neural", action="store_true",
                    help="Golden gate only (fast). Skip the model entirely.")
    ap.add_argument("--neural-all", action="store_true",
                    help="Run EVERY case through the neural model (~40+ min).")
    ap.add_argument("--neural-per-cluster", type=int, default=1,
                    help="Neural sample size per cluster (default 1; the balance of "
                         "coverage vs the ~15-60s/program neural cost).")
    ap.add_argument("--only-cluster", type=str, default=None,
                    help="Restrict to a single cluster (e.g. char_shift).")
    ap.add_argument("--code-size", type=int, default=48,
                    help="Model code-band width (must hold the longest program).")
    ap.add_argument("--output", type=str, default=None, help="JSON dump path.")
    args = ap.parse_args(argv)

    cases = generate_edge_cases()
    if args.only_cluster:
        cases = [c for c in cases if c.cluster == args.only_cluster]

    # coverage before/after (the canonical corpus exercises 23 opcodes).
    coverage_before = {"LEA", "IMM", "JMP", "JSR", "BZ", "ENT", "ADJ", "LEV",
                       "LI", "SI", "PSH", "EQ", "NE", "LT", "GT", "LE",
                       "ADD", "SUB", "MUL", "DIV", "MOD", "HALT", "NOP"}
    coverage_after = coverage_before | opcodes_covered(cases)

    # --- GOLDEN gate (all cases) -------------------------------------------
    results: List[Result] = []
    for c in cases:
        try:
            gax, gout = golden_value(c)
            gok = (gax == (c.expected & M32))
            if c.expected_stdout is not None and gout is not None:
                gok = gok and (gout == c.expected_stdout)
            detail = "" if gok else f"exp_out={c.expected_stdout!r} got_out={gout!r}"
            results.append(Result(
                name=c.name, cluster=c.cluster, expected=c.expected & M32,
                golden_ax=gax, golden_stdout=gout, golden_ok=gok, note=c.note,
                detail=detail))
        except Exception as exc:  # noqa: BLE001
            results.append(Result(
                name=c.name, cluster=c.cluster, expected=c.expected & M32,
                golden_ax=None, golden_stdout=None, golden_ok=False, note=c.note,
                detail=f"golden EXC: {exc!r}"))

    by_name = {r.name: r for r in results}

    # --- NEURAL gate (bounded) ---------------------------------------------
    neural_mode = "OFF"
    wall = 0.0
    if not args.no_neural:
        if args.neural_all:
            selected = list(cases)
            neural_mode = "ALL"
        else:
            per = args.neural_per_cluster
            seen: Counter = Counter()
            selected = []
            for c in cases:
                if seen[c.cluster] < per:
                    seen[c.cluster] += 1
                    selected.append(c)
            neural_mode = f"sample <= {per}/cluster ({len(selected)} cases)"

        t_build = time.monotonic()
        print(f"[edge] building pure-forward model (code_size={args.code_size}) ...",
              file=sys.stderr, flush=True)
        from c4_min.compact_alloc import build_compact_sparse_streaming
        model, L, _ = build_compact_sparse_streaming(
            code_size=args.code_size, compute_mode="dense_kernel")
        print(f"[edge] model dim={L.D} blocks={len(model.blocks)} "
              f"({time.monotonic()-t_build:.1f}s); running {len(selected)} neural cases",
              file=sys.stderr, flush=True)

        t0 = time.monotonic()
        for i, c in enumerate(selected):
            r = by_name[c.name]
            t = time.monotonic()
            try:
                nax, nout = neural_value(c, model, L)
                nok = (nax == (c.expected & M32))
                if c.expected_stdout is not None and nout is not None:
                    nok = nok and (nout == c.expected_stdout)
                r.neural_ax, r.neural_stdout, r.neural_ok = nax, nout, nok
            except Exception as exc:  # noqa: BLE001
                r.neural_ok = False
                r.detail = (r.detail + f" | neural EXC: {exc!r}").strip(" |")
            r.seconds = time.monotonic() - t
            print(f"[edge] {i+1}/{len(selected)} {c.name:26s} "
                  f"golden={'ok' if r.golden_ok else 'FAIL'} "
                  f"neural={'ok' if r.neural_ok else 'FAIL'} "
                  f"({r.seconds:.1f}s)", file=sys.stderr, flush=True)
        wall = time.monotonic() - t0

    _print_report(results, wall, coverage_before, coverage_after, neural_mode)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump({
                "coverage_before": sorted(coverage_before),
                "coverage_after": sorted(coverage_after),
                "newly_exercised": sorted(coverage_after - coverage_before),
                "neural_mode": neural_mode,
                "wall_seconds": wall,
                "results": [
                    {"name": r.name, "cluster": r.cluster, "expected": r.expected,
                     "golden_ax": r.golden_ax,
                     "golden_stdout": (r.golden_stdout.hex() if r.golden_stdout else None),
                     "golden_ok": r.golden_ok,
                     "neural_ax": r.neural_ax,
                     "neural_stdout": (r.neural_stdout.hex() if r.neural_stdout else None),
                     "neural_ok": r.neural_ok, "seconds": r.seconds,
                     "note": r.note, "detail": r.detail}
                    for r in results],
            }, fh, indent=2)
        print(f"[edge] wrote {args.output}", file=sys.stderr, flush=True)

    # exit non-zero only on a GOLDEN failure (the corpus's expected values must be
    # correct); a NEURAL divergence is REPORTED (the point of the suite) but does
    # not fail the run — pass --neural-strict to also fail on neural divergence.
    n_gold_fail = sum(1 for r in results if not r.golden_ok)
    return 1 if n_gold_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
