#!/usr/bin/env python3
"""c4_min PURE-FORWARD VM — FULL 1096 SCOREBOARD (the CHK-1 deliverable).

The whole C4 VM runs 100% through the vanilla transformer forward
(``model.forward`` + argmax-generate), NO Python compute — scored on the FULL
1096 corpus.  This is the scoreboard for checklist items #2/#10 ("100%
autoregressive, no external memory or logic, only standard layers" / "100%
vanilla transformer, none of the operations performed any other way").

ONE persistent :class:`Transformer` (``build_pure_forward_complete_model``) does a
full VM step per ``model.forward``: in-model MoE opcode dispatch (no Python
if/elif), softmax1-KV memory (no Python dict), a multi-slot stack via a KV head,
the full calling convention (JSR/ENT/LEV/ADJ/LEA), and the **fp32-exact 32-bit
ALU** (ADD/SUB per-byte carry chain, MUL nibble schoolbook, DIV/MOD base-16 long
division — ``nibble_alu32``).  The only Python on the compute path is the
argmax-generate-append emit; the ``assert_no_python_compute`` settrace guard
(``--guard``) is the machine proof.

Corpus + expected values are the SAME as the shared corpus loader:
``tests.test_suite_1000.generate_test_programs`` (source, expected, description)
compiled by ``src.compiler.compile_c``.  The C4 compiler encodes ``int`` as an
8-byte word (``elem_size = 8``); the pure-forward ISA addresses the stack in
4-byte slots, so LEA/ENT/ADJ byte-offset immediates are re-encoded to slot units
(``imm // WORD``) — a faithful frame-layout isomorphism (both describe the same
frame, ``byte_off = WORD * slot``).  The final AX (decoded from the canonical
32-bit AX nibble band via the LM byte-head argmax, no ``torch.round``) is compared
to ``expected & 0xFFFFFFFF``.

Categorisation (mirrors ``run_1096_canonical``):
  PASS      — final AX == expected.
  FAIL      — halted, final AX != expected.
  TIMEOUT   — ran to ``--step-cap`` without HALT (deep-loop / non-halt).
  ERROR     — compile / run exception.

Usage
-----
    OMP_NUM_THREADS=4 PYTHONPATH=$(pwd) python c4_min/run_1096_pure_forward.py --limit 64
    OMP_NUM_THREADS=4 PYTHONPATH=$(pwd) python c4_min/run_1096_pure_forward.py \
        --step-cap 6000 --output /tmp/pf_1096.json
    # stratified sample across ALL clusters (bounds wall-time; NO silent truncation):
    OMP_NUM_THREADS=4 PYTHONPATH=$(pwd) python c4_min/run_1096_pure_forward.py \
        --per-cluster 4 --output /tmp/pf_sample.json

MEMORY DISCIPLINE: LEAN/SPARSE model (``include_bitwise=False`` — the corpus uses
NO bitwise ops; ``include_divmod`` folds the 32-bit long-division blocks needed by
div/mod/expr_mul_div).  Runs SEQUENTIALLY.  Set ``OMP_NUM_THREADS=4``.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

# CPU-only, low-footprint (tiny nibble foundation model): never touch a GPU.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "4")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.dirname(_HERE)  # .../c4_release
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

# small stack base so frame-relative LEA (the frame lives in a small byte window)
# reaches the frame; MUST be set before importing the driver modules.
# 0xFC (not 0xF0): the stack grows DOWN from here, so the base must exceed the
# deepest program's stack depth or SP underflows below 0 — which the LM value-head
# (``_snap_lane``, argmax over v>=0) CANNOT represent, so a negative SP snaps to 0
# and the ENT/LEV frame collapses.  rec_sum(14) needs 248 bytes; 0xF0=240 underflows
# to -8, 0xFC=252 keeps the whole 1096 corpus in [4, 252] (deep-recursion fix
# e52ab0c5).  NOTE: build_compact_pure_forward_model transitively imports THIS
# module, which re-pins SP_INIT — so this value is the one in force at draft time.
import c4_min.nibble_pure_forward as _PF        # noqa: E402
import c4_min.nibble_pure_forward_complete as _PFC  # noqa: E402
_PF.SP_INIT = 0xFC
_PFC.SP_INIT = 0xFC

from c4_min import isa  # noqa: E402
from c4_min.nibble_pure_forward_complete import (  # noqa: E402
    build_pure_forward_complete_model, run_pure_forward_complete, ref_interpret,
)
from c4_min.nibble_pure_forward import assert_no_python_compute  # noqa: E402


# ---------------------------------------------------------------------------
# Bytecode -> isa.Instr : re-encode the compiler's 8-byte-word frame offsets to
# the pure-forward ISA's 4-byte slot units (a faithful frame isomorphism).
# ---------------------------------------------------------------------------
# The C4 compiler (src/compiler.py) sizes ``int`` at WORD = 8 bytes.  LEA/ENT/ADJ
# immediates are BYTE offsets into that frame; the pure-forward ISA (isa.py /
# nibble_pure_forward_complete.ref_interpret) addresses the stack in 4-byte slots
# and scales its own SP/BP motion by 4, so we divide those byte offsets by WORD to
# recover the slot index (LEA slot k => BP + 4k in the pure-forward frame).  Value
# immediates (IMM) and PC targets (JMP/BZ/BNZ/JSR) are copied verbatim.
_WORD = 8
_SLOT_SCALED_OPS = frozenset({isa.LEA, isa.ENT, isa.ADJ})


def _sign32(imm: int) -> int:
    return imm if imm < (1 << 31) else imm - (1 << 32)


def bytecode_to_isa(bytecode) -> List[isa.Instr]:
    """Decode c4 bytecode words -> [isa.Instr], re-encoding LEA/ENT/ADJ byte
    offsets to slot units.  Word format: ``op = word & 0xFF``, ``imm = word >> 8``."""
    out: List[isa.Instr] = []
    for word in bytecode:
        op = int(word) & 0xFF
        imm = int(word) >> 8
        if op in _SLOT_SCALED_OPS:
            simm = _sign32(imm)
            assert simm % _WORD == 0, f"unaligned {isa.NAMES.get(op)} imm {simm}"
            out.append(isa.Instr(op, simm // _WORD))
        else:
            out.append(isa.Instr(op, imm & 0xFFFFFFFF))
    return out


# ---------------------------------------------------------------------------
# Per-program result + cluster key (matches run_1096_canonical).
# ---------------------------------------------------------------------------
@dataclass
class Result:
    idx: int
    description: str
    cluster: str
    expected: int
    got_exit: Optional[int]
    got_steps: Optional[int]
    status: str                      # PASS | FAIL | TIMEOUT | ERROR
    guard_clean: Optional[bool] = None
    n_instrs: int = 0
    detail: str = ""


def cluster_of(description: str) -> str:
    """Stable cluster key (matches run_1096_canonical.cluster_of)."""
    base = description.split(":", 1)[0].strip()
    base = re.sub(r"_\d+$", "", base)
    base = re.sub(r"\d+$", "", base)
    base = base.rstrip("_")
    return base or "misc"


# ---------------------------------------------------------------------------
# The scoreboard core.  ONE program: compile, translate, run pure-forward.
# ---------------------------------------------------------------------------
def score_program(idx: int, source: str, expected: int, description: str,
                  *, model, L, compile_c, step_cap: int,
                  guard: bool, ref_steps: Optional[int] = None) -> Result:
    cluster = cluster_of(description)
    exp = expected & 0xFFFFFFFF
    base = dict(idx=idx, description=description, cluster=cluster, expected=exp)

    try:
        bytecode, _data = compile_c(source)
        code = bytecode_to_isa(bytecode)
    except Exception as exc:  # noqa: BLE001
        return Result(got_exit=None, got_steps=None, status="ERROR",
                      detail=f"compile/translate: {exc!r}", **base)

    n_instrs = len(code)
    # Right-size the per-program cap: a program halts at its reference step count,
    # so cap the model run at ref_steps + a small headroom (a program that diverges
    # more than that is a genuine TIMEOUT).  This avoids wasting the global cap (and
    # the quadratic stream growth) on a program that should halt in a dozen steps.
    # ref_interpret is pure-python HARNESS bookkeeping (sizing), NOT model compute —
    # it is NOT inside the guarded run and does not touch model.forward.
    if ref_steps is not None:
        step_cap = min(step_cap, ref_steps + 6)
    try:
        if guard:
            gclean = True
            try:
                trace = assert_no_python_compute(
                    run_pure_forward_complete, model, L, code,
                    max_steps=step_cap, mask=0xFFFFFFFF)
            except AssertionError as gexc:
                gclean = False
                # run once more WITHOUT the guard to still get a verdict.
                trace = run_pure_forward_complete(model, L, code,
                                                  max_steps=step_cap, mask=0xFFFFFFFF)
                return Result(got_exit=(trace[-1] if trace else None),
                              got_steps=len(trace), status="ERROR",
                              guard_clean=False, n_instrs=n_instrs,
                              detail=f"GUARD LEAK: {gexc}", **base)
        else:
            gclean = None
            trace = run_pure_forward_complete(model, L, code,
                                              max_steps=step_cap, mask=0xFFFFFFFF)
    except Exception as exc:  # noqa: BLE001
        return Result(got_exit=None, got_steps=None, status="ERROR",
                      guard_clean=(guard and False), n_instrs=n_instrs,
                      detail=f"run: {exc!r}", **base)

    if not trace:
        return Result(got_exit=None, got_steps=0, status="ERROR",
                      guard_clean=gclean, n_instrs=n_instrs,
                      detail="no frame emitted", **base)

    got = int(trace[-1]) & 0xFFFFFFFF
    steps = len(trace)
    # A run that hit the step cap almost certainly never halted (the driver breaks
    # the loop on HALT / out-of-range PC), so call it a TIMEOUT not a FAIL.
    if steps >= step_cap:
        return Result(got_exit=got, got_steps=steps, status="TIMEOUT",
                      guard_clean=gclean, n_instrs=n_instrs,
                      detail=f"no HALT within {step_cap} steps", **base)

    if got == exp:
        return Result(got_exit=got, got_steps=steps, status="PASS",
                      guard_clean=gclean, n_instrs=n_instrs, **base)
    return Result(got_exit=got, got_steps=steps, status="FAIL",
                  guard_clean=gclean, n_instrs=n_instrs,
                  detail=f"exit mismatch: exp {exp} got {got}", **base)


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------
_STATUSES = ("PASS", "FAIL", "TIMEOUT", "ERROR", "DEEP")


def _cluster_table(results: List[Result]) -> "OrderedDict[str, Dict[str, int]]":
    table: "OrderedDict[str, Dict[str, int]]" = OrderedDict()
    for r in results:
        row = table.setdefault(r.cluster, {"n": 0, **{s: 0 for s in _STATUSES}})
        row["n"] += 1
        row[r.status] += 1
    return table


def _print_cluster_table(table, fh=sys.stdout) -> None:
    print("\nPER-CLUSTER BREAKDOWN", file=fh)
    hdr = (f"  {'cluster':18s} {'n':>4s} {'PASS':>5s} {'FAIL':>5s} "
           f"{'TMOUT':>6s} {'ERR':>4s} {'DEEP':>5s} {'pass%':>6s} {'pass%run':>8s}")
    print(hdr, file=fh)
    print("  " + "-" * (len(hdr) - 2), file=fh)
    for cluster, row in sorted(table.items()):
        n = row["n"]
        run_n = n - row["DEEP"]
        pct = (100.0 * row["PASS"] / n) if n else 0.0
        pct_run = (100.0 * row["PASS"] / run_n) if run_n else 0.0
        print(f"  {cluster:18s} {n:4d} {row['PASS']:5d} {row['FAIL']:5d} "
              f"{row['TIMEOUT']:6d} {row['ERROR']:4d} {row['DEEP']:5d} "
              f"{pct:6.1f} {pct_run:8.1f}", file=fh)


def _print_summary(results: List[Result], wall: float, step_cap: int,
                   coverage: str, guard: bool, fh=sys.stdout) -> None:
    counts = Counter(r.status for r in results)
    total = len(results)
    n_pass = counts.get("PASS", 0)
    print("\n" + "=" * 72, file=fh)
    print("c4_min PURE-FORWARD VM SCOREBOARD  (100% model.forward, no python compute)",
          file=fh)
    print("=" * 72, file=fh)
    print(f"  coverage: {coverage}", file=fh)
    print(f"  step cap: {step_cap}   wall: {wall:.1f}s", file=fh)
    if guard:
        n_guard = sum(1 for r in results if r.guard_clean is True)
        n_leak = sum(1 for r in results if r.guard_clean is False)
        print(f"  purity guard (assert_no_python_compute): "
              f"{n_guard} clean, {n_leak} leaked", file=fh)
    print("-" * 72, file=fh)
    for s in _STATUSES:
        print(f"  {s:10s} {counts.get(s, 0):5d}", file=fh)
    print("-" * 72, file=fh)
    print(f"  SCORE:  {n_pass}/{total}  ({100.0 * n_pass / total:.2f}%)  "
          f"[pure-forward VM, 32-bit]", file=fh)
    print("=" * 72, file=fh)


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=None,
                    help="Run only the first N programs (default: all 1096).")
    ap.add_argument("--offset", type=int, default=0,
                    help="Skip the first M programs before --limit.")
    ap.add_argument("--per-cluster", type=int, default=None,
                    help="Stratified sample: at most N programs per cluster (bounds "
                         "wall-time; NO silent truncation — coverage is reported).")
    ap.add_argument("--step-cap", type=int, default=10000,
                    help="Global autoregressive step cap (bounds non-halting deep "
                         "loops). Per-program cap is min(this, ref_steps+6). "
                         "Default 10000.")
    ap.add_argument("--max-ref-steps", type=int, default=None,
                    help="Run to completion only programs whose reference step count "
                         "is <= this (bounds the QUADRATIC stream-growth wall-time on "
                         "deep loops); the rest are reported as DEEP (not run). "
                         "Coverage is reported explicitly. Default: run all.")
    ap.add_argument("--deep-per-cluster", type=int, default=0,
                    help="Of the programs EXCEEDING --max-ref-steps, still run this "
                         "many per cluster (stratified deep-loop sample). Default 0.")
    ap.add_argument("--code-size", type=int, default=64,
                    help="Max program length the model's code band holds. Default 64.")
    ap.add_argument("--no-divmod", action="store_true",
                    help="LEAN model WITHOUT the 32-bit long-division blocks (~10x "
                         "faster forward; div/mod/expr_mul_div then FAIL).")
    ap.add_argument("--guard", action="store_true",
                    help="Wrap every run in assert_no_python_compute (the purity "
                         "proof). Roughly doubles wall-time (runs a settrace).")
    ap.add_argument("--output", type=str, default=None,
                    help="Dump full per-program results + tables as JSON.")
    ap.add_argument("--print-nonpass", action="store_true",
                    help="Print each non-PASS program row.")
    ap.add_argument("--progress", type=int, default=25,
                    help="Print a progress line every N programs.")
    args = ap.parse_args(argv)

    from src.compiler import compile_c
    from tests.test_suite_1000 import generate_test_programs

    all_tests = generate_test_programs()

    # Select the window (offset/limit) then, optionally, a stratified per-cluster
    # sample (deterministic: the first N of each cluster in corpus order).
    indexed = list(enumerate(all_tests))[args.offset:]
    if args.limit is not None:
        indexed = indexed[:args.limit]
    if args.per_cluster is not None:
        seen: Counter = Counter()
        sampled = []
        for idx, tp in indexed:
            cl = cluster_of(tp[2])
            if seen[cl] < args.per_cluster:
                seen[cl] += 1
                sampled.append((idx, tp))
        indexed = sampled

    # Reference step counts (pure-python HARNESS sizing; not model compute) — used
    # to right-size each program's cap and to split off the deep-loop tail whose
    # quadratic stream growth is the wall-time hazard.
    ref_steps_by_idx: Dict[int, Optional[int]] = {}
    if args.max_ref_steps is not None or True:
        for idx, (source, _exp, _desc) in indexed:
            try:
                bc, _d = compile_c(source)
                tr = ref_interpret(bytecode_to_isa(bc), max_steps=200000,
                                   mask=0xFFFFFFFF)
                ref_steps_by_idx[idx] = len(tr)
            except Exception:  # noqa: BLE001
                ref_steps_by_idx[idx] = None

    # Split into RUN (ref_steps <= max_ref_steps) + a stratified DEEP sample.
    deep_reported: List[Tuple[int, tuple]] = []
    if args.max_ref_steps is not None:
        run_set, deep_set = [], []
        for item in indexed:
            idx = item[0]
            rs = ref_steps_by_idx.get(idx)
            if rs is not None and rs > args.max_ref_steps:
                deep_set.append(item)
            else:
                run_set.append(item)
        # keep a per-cluster deep sample IN the run set; the rest are reported DEEP.
        if args.deep_per_cluster > 0:
            seen: Counter = Counter()
            kept = []
            for item in deep_set:
                cl = cluster_of(item[1][2])
                if seen[cl] < args.deep_per_cluster:
                    seen[cl] += 1
                    run_set.append(item)
                else:
                    kept.append(item)
            deep_reported = kept
        else:
            deep_reported = deep_set
        indexed = sorted(run_set, key=lambda it: it[0])

    coverage = (f"{len(indexed)}/{len(all_tests)} run"
                + (f" (stratified: <= {args.per_cluster}/cluster)"
                   if args.per_cluster is not None else "")
                + (f"; {len(deep_reported)} deep-loop (ref_steps > "
                   f"{args.max_ref_steps}) NOT run"
                   if deep_reported else ""))

    include_divmod = not args.no_divmod
    t_build = time.monotonic()
    print(f"[pf-1096] building pure-forward model (code_size={args.code_size}, "
          f"include_divmod={include_divmod}, include_bitwise=False) ...",
          file=sys.stderr, flush=True)
    model, L = build_pure_forward_complete_model(
        code_size=args.code_size, include_bitwise=False, include_divmod=include_divmod)
    print(f"[pf-1096] model: dim={L.D} blocks={len(model.blocks)} "
          f"heads={model.blocks[0].attn.n_heads} ({time.monotonic()-t_build:.1f}s)",
          file=sys.stderr, flush=True)
    print(f"[pf-1096] scoring {coverage} | step_cap={args.step_cap} | "
          f"guard={'ON' if args.guard else 'off'}", file=sys.stderr, flush=True)

    t0 = time.monotonic()
    results: List[Result] = []
    for i, (idx, (source, expected, description)) in enumerate(indexed):
        r = score_program(idx, source, expected, description,
                          model=model, L=L, compile_c=compile_c,
                          step_cap=args.step_cap, guard=args.guard,
                          ref_steps=ref_steps_by_idx.get(idx))
        results.append(r)
        if args.progress and (i + 1) % args.progress == 0:
            npass = sum(1 for x in results if x.status == "PASS")
            print(f"[pf-1096] {i + 1}/{len(indexed)} done (pass so far: {npass}) "
                  f"[{time.monotonic()-t0:.0f}s]", file=sys.stderr, flush=True)
    wall = time.monotonic() - t0

    # Record the deep-loop tail (NOT run) transparently as DEEP rows.
    for idx, (source, expected, description) in deep_reported:
        results.append(Result(
            idx=idx, description=description, cluster=cluster_of(description),
            expected=expected & 0xFFFFFFFF, got_exit=None,
            got_steps=ref_steps_by_idx.get(idx), status="DEEP",
            n_instrs=0,
            detail=f"deep loop (ref_steps={ref_steps_by_idx.get(idx)} > "
                   f"{args.max_ref_steps}) — not run (quadratic wall-time)"))

    results.sort(key=lambda r: r.idx)

    _print_summary(results, wall, args.step_cap, coverage, args.guard)
    table = _cluster_table(results)
    _print_cluster_table(table)

    if args.print_nonpass:
        print("\nNON-PASS PROGRAMS", file=sys.stdout)
        for r in results:
            if r.status != "PASS":
                print(f"  id={r.idx:04d} [{r.status:7s}] {r.cluster:16s} "
                      f"exp={r.expected} got={r.got_exit} {r.detail}  "
                      f"({r.description})")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump({
                "wall_seconds": wall,
                "coverage": coverage,
                "step_cap": args.step_cap,
                "include_divmod": include_divmod,
                "guard": args.guard,
                "summary": {s: sum(1 for r in results if r.status == s)
                            for s in _STATUSES} | {"total": len(results)},
                "clusters": {k: v for k, v in table.items()},
                "results": [asdict(r) for r in results],
            }, fh, indent=2)
        print(f"[pf-1096] wrote {args.output}", file=sys.stderr, flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
