#!/usr/bin/env python3
"""c4_min NIBBLE UNIVERSAL VM — 1096 SCOREBOARD (the progress meter to FULL 1096).

This is the authoritative scoreboard for the push to FULL 1096 on the **nibble
universal VM** built on the BLOG_SPEC foundation lineage (``c4_min/blogspec_*``).
It is the nibble-substrate analogue of the green-field recurrent scoreboard
(``git show greenfield-scoreboard:c4_release/c4_min/run_1096.py``): same corpus,
same reference oracle, same PASS/FAIL/NOT_IMPL/TIMEOUT/ERROR categorisation and
the same ``op X unlocks N programs`` unlock map — but the model side drives the
spec-faithful nibble transformer:

  * register values are carried as **16 4-bit nibbles** per register
    (``blogspec_layout.NibbleLayout``);
  * every VM step emits the **30-token register frame**
    (``blogspec_vocab.build_step_frame``) and the emit->re-embed of those
    exact-integer byte tokens IS the re-quantization (no ``torch.round``);
  * the runtime is the **softmax1 + ALiBi** vanilla transformer
    (``blogspec_model.Transformer``), stepped by the standard autoregressive
    loop (``blogspec_run.run_program``); the register nibbles are decoded back to
    integers only via the LM byte-head argmax.

Pipeline per program (mirrors ``tools/run_1096_canonical.py`` on the reference
side, and the green-field scoreboard on the categorisation side):

  1. ``compile_c(source)`` -> c4 bytecode + data (the SAME compiler the canonical
     runner uses). A compile failure is ``ERROR``.
  2. ``declarative_oracle_for_program`` -> the REFERENCE exit code + step count
     (the SAME 32-bit symbolic VM). A reference non-halt / disagreement is
     ``REF_ERROR`` (says nothing about the model; never a model PASS/FAIL).
  3. Decode the bytecode to ``[(op_name, imm), ...]``. If ANY instruction uses an
     opcode that is NOT in the c4_min ISA, or is in the ISA but has NO nibble
     step-transition rule yet (the ``BUILT`` set, DERIVED by probing the actual
     ``blogspec_run._apply_op`` dispatch), the program is ``NOT_IMPL`` and we
     record EVERY missing opcode it needs.
  4. Otherwise bake the nibble step-model (``blogspec_compiler.build_step_model``)
     and run it autoregressively with a generous step cap
     (``blogspec_run.run_program``). Decode the final-step AX out of the emitted
     nibble frame and compare it, masked to the substrate's OWN width (auto-
     detected: 8-bit today, 32-bit as the nibble ALU stops folding), to the
     reference exit code. Match -> ``PASS``; mismatch -> ``FAIL``; no HALT within
     the cap -> ``TIMEOUT``; a run exception -> ``ERROR``.

The BUILT set and the compare width are both DERIVED from the substrate (probe
``_apply_op`` for dispatch, probe the ALU for width), so re-running this file
auto-rescores as each fan-out opcode / the 32-bit ALU lands — it is designed to
be the single re-runnable meter to 1096.

Usage
-----
    OMP_NUM_THREADS=4 PYTHONPATH=$(pwd) python c4_min/run_1096_nibble.py
    OMP_NUM_THREADS=4 PYTHONPATH=$(pwd) python c4_min/run_1096_nibble.py --limit 128
    OMP_NUM_THREADS=4 PYTHONPATH=$(pwd) python c4_min/run_1096_nibble.py --output /tmp/nib_1096.json
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

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.dirname(_HERE)  # .../c4_release
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from c4_min import isa  # noqa: E402
from c4_min import blogspec_run as R  # noqa: E402
from c4_min import blogspec_compiler as C  # noqa: E402
from c4_min import blogspec_vocab as _V  # noqa: E402

_HALT_TOKEN = _V.HALT   # end-of-generation terminator (§Exiting)


# ---------------------------------------------------------------------------
# The BUILT op set: opcodes the nibble step-transition actually implements.
#
# Derived by probing ``blogspec_run._apply_op`` (the single-step VM transition
# through the nibble ALU gadgets). An opcode that is in the c4_min ISA but has no
# transition rule falls into ``_apply_op``'s ``else`` branch and raises
# ``NotImplementedError`` — so the BUILT set tracks the real capability instead
# of a stale hardcoded list, and auto-grows as fan-out ops are wired in.
# ---------------------------------------------------------------------------
def _built_op_names() -> frozenset:
    built = set()
    # generic register state that keeps stack/branch ops from erroring for the
    # wrong reason (SP/BP init per §Registers; a non-empty stack0).
    pc, ax, sp, bp, stack0 = 0, 1, 0x10000, 0x10000, 1
    for op_id, name in isa.NAMES.items():
        try:
            R._apply_op(isa.Instr(op_id, 1), pc, ax, sp, bp, stack0)
        except NotImplementedError:
            continue
        except Exception:
            # any other exception means the transition rule EXISTS (it ran) but
            # tripped on the probe inputs — still counts as BUILT (dispatched).
            pass
        built.add(name)
    return frozenset(built)


BUILT_OPS = _built_op_names()


# ---------------------------------------------------------------------------
# Compare width: DERIVED from the substrate's own ALU, not hardcoded.
#
# The nibble representation carries a full 32-bit value (16 nibbles), but the
# foundation *ALU* currently folds AX/results to 8 bits. Probe the actual add
# gadget with an operand pair that overflows 8 bits: if 200+200 comes back 144
# the substrate is 8-bit; if it comes back 400 (or wider) the mask auto-widens.
# The reference exit codes are 32-bit; we compare ``got & MASK == ref & MASK``.
# ---------------------------------------------------------------------------
def _detect_width_mask() -> Tuple[int, int]:
    """Return (mask, n_bits) of the substrate's effective ALU width.

    Probes the REAL per-step transition (``R._apply_op``), not the raw gadget: an
    ADD whose true sum overflows N bits (``IMM (2^N-1); PSH; IMM 1; ADD``) tells
    us whether the substrate folds at N. The fan-out ops run AX/STACK0 at the
    32-bit nibble width, so this now auto-detects 32-bit (the corpus's multi-byte
    immediates and mul/div products need the full width to score correctly).
    """
    def _add_via_transition(a: int, b: int) -> int:
        # push ``a`` then ADD ``b``: stack0=a, ax=b -> ax = a + b (substrate width)
        _, ax, *_ = R._apply_op(isa.Instr(isa.ADD, 0), 0, b, 0x10000, 0x10000, a)
        return ax

    for n_bits in (8, 16, 32):
        mask = (1 << n_bits) - 1
        # a value that just overflows n_bits: (2^n_bits - 1) + 1 == 2^n_bits.
        a = mask
        try:
            got = _add_via_transition(a, 1)
        except Exception:  # noqa: BLE001
            continue
        # if the transition folded exactly at n_bits, got == 0; if it carried past
        # n_bits (wider substrate) got == 2^n_bits. Take the widest that carries.
        wide = None
        for wider in (16, 32):
            if wider <= n_bits:
                continue
            try:
                g2 = _add_via_transition(mask, 1)
            except Exception:  # noqa: BLE001
                g2 = None
            if g2 == ((mask + 1) & ((1 << wider) - 1)) and g2 != 0:
                wide = wider
        if wide is not None:
            return (1 << wide) - 1, wide
        if got == ((a + 1) & mask):
            return mask, n_bits
    return 0xFFFFFFFF, 32  # the substrate carries full 32-bit values.


WIDTH_MASK, WIDTH_BITS = _detect_width_mask()


# ---------------------------------------------------------------------------
# Bytecode decode -> [(op_name, imm)] + missing-op set.
# ---------------------------------------------------------------------------
def _decode_bytecode(bytecode) -> Tuple[List[Tuple[str, int]], List[str]]:
    """Decode c4 bytecode -> ([(op_name, imm), ...], [missing_op_name, ...]).

    ``imm`` is the FULL (unmasked) immediate word — for jump/branch ops it is a
    PC target that can exceed one byte, so we must not truncate it here. Value
    immediates (IMM/LEA) are folded to the substrate width inside ``_apply_op``.

    ``missing`` is the sorted set of opcodes this program needs that are NOT yet
    runnable on the nibble substrate — either not in the c4_min ISA at all
    (resolved to the reference-VM name / ``<id:NN>``) or in the ISA but with no
    ``_apply_op`` transition rule yet. Empty ``missing`` == fully runnable.
    """
    ops: List[Tuple[str, int]] = []
    missing: set = set()
    for word in bytecode:
        op = int(word) & 0xFF
        imm = int(word) >> 8            # full immediate (may be a >8-bit PC target)
        name = isa.NAMES.get(op)
        if name is None:
            missing.add(_ref_op_name(op))
            ops.append((f"<{op}>", imm))
            continue
        ops.append((name, imm))
        if name not in BUILT_OPS:
            missing.add(name)
    return ops, sorted(missing)


_REF_NAME_CACHE: Dict[int, str] = {}


def _ref_op_name(op_id: int) -> str:
    """Resolve an opcode id NOT in the c4_min ISA to its reference-VM name
    (ADJ/MUL/DIV/MOD/NOP/...), falling back to ``<id:NN>``."""
    if not _REF_NAME_CACHE:
        try:
            from neural_vm.embedding import Opcode  # noqa: E402

            for attr in dir(Opcode):
                if attr.startswith("_"):
                    continue
                try:
                    _REF_NAME_CACHE[int(getattr(Opcode, attr))] = attr
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            pass
    return _REF_NAME_CACHE.get(op_id, f"<id:{op_id}>")


# ---------------------------------------------------------------------------
# Per-program result + cluster key.
# ---------------------------------------------------------------------------
@dataclass
class Result:
    idx: int
    description: str
    cluster: str
    suite_expected: int
    ref_exit: Optional[int]          # reference exit (masked to substrate width)
    ref_steps: Optional[int]
    got_exit: Optional[int]          # nibble-VM exit (masked to substrate width)
    got_steps: Optional[int]
    status: str                      # PASS | FAIL | NOT_IMPL | TIMEOUT | ERROR | REF_ERROR
    missing_ops: List[str] = field(default_factory=list)
    detail: str = ""


def cluster_of(description: str) -> str:
    """Stable cluster key (matches ``tools/run_1096_canonical.cluster_of``)."""
    base = description.split(":", 1)[0].strip()
    base = re.sub(r"_\d+$", "", base)
    base = re.sub(r"\d+$", "", base)
    base = base.rstrip("_")
    return base or "misc"


# ---------------------------------------------------------------------------
# The scoreboard core.
# ---------------------------------------------------------------------------
def score_program(
    idx: int,
    source: str,
    suite_expected: int,
    description: str,
    *,
    compile_c,
    oracle_fn,
    step_cap: int,
) -> Result:
    """Compile, reference-oracle, and (if fully built) nibble-VM one program."""
    cluster = cluster_of(description)
    base = dict(idx=idx, description=description, cluster=cluster,
                suite_expected=suite_expected)

    # 1. compile C -> bytecode
    try:
        bytecode, data = compile_c(source)
    except Exception as exc:  # noqa: BLE001
        return Result(ref_exit=None, ref_steps=None, got_exit=None,
                      got_steps=None, status="ERROR",
                      detail=f"compile error: {exc!r}", **base)

    # 2. reference oracle (ground truth). A ref non-halt/disagreement is a
    #    REF_ERROR — it says nothing about the nibble model, so we never count it
    #    as a model PASS/FAIL.
    try:
        ref = oracle_fn(bytecode, data, suite_expected=suite_expected,
                        label=f"id={idx:04d}", max_steps=None)
    except Exception as exc:  # noqa: BLE001
        return Result(ref_exit=None, ref_steps=None, got_exit=None,
                      got_steps=None, status="REF_ERROR",
                      detail=f"oracle exception: {exc!r}", **base)
    if ref.error is not None or ref.exit_code is None or ref.steps is None:
        return Result(ref_exit=None, ref_steps=None, got_exit=None,
                      got_steps=None, status="REF_ERROR",
                      detail=ref.error or "reference did not halt", **base)

    ref_exit = int(ref.exit_code) & WIDTH_MASK
    ref_steps = int(ref.steps)

    # 3. which ops does it need, and are they all built on the nibble substrate?
    ops, missing = _decode_bytecode(bytecode)
    if missing:
        return Result(ref_exit=ref_exit, ref_steps=ref_steps, got_exit=None,
                      got_steps=None, status="NOT_IMPL", missing_ops=missing,
                      detail=f"needs {', '.join(missing)}", **base)

    # 4. fully built -> bake the nibble step-model and run it autoregressively.
    try:
        model, L, code = C.build_step_model(ops)
        _tokens, frames = R.run_program(model, L, code, max_steps=step_cap)
    except Exception as exc:  # noqa: BLE001
        return Result(ref_exit=ref_exit, ref_steps=ref_steps, got_exit=None,
                      got_steps=None, status="ERROR",
                      detail=f"nibble run error: {exc!r}", **base)

    trace = R.decode_trace(frames)
    if not trace:
        return Result(ref_exit=ref_exit, ref_steps=ref_steps, got_exit=None,
                      got_steps=0, status="TIMEOUT",
                      detail="nibble VM produced no frame", **base)

    got_exit = int(trace[-1]) & WIDTH_MASK
    got_steps = len(frames)

    # A TIMEOUT is "ran to the cap without HALT". ``run_program`` breaks the loop
    # on the HALT op; a run that used the full cap almost certainly never halted.
    halted = _tokens and _tokens[-1] == _HALT_TOKEN
    if not halted and got_steps >= step_cap:
        return Result(ref_exit=ref_exit, ref_steps=ref_steps, got_exit=got_exit,
                      got_steps=got_steps, status="TIMEOUT",
                      detail=f"no HALT within {step_cap} steps", **base)

    if got_exit == ref_exit:
        return Result(ref_exit=ref_exit, ref_steps=ref_steps, got_exit=got_exit,
                      got_steps=got_steps, status="PASS", **base)
    return Result(ref_exit=ref_exit, ref_steps=ref_steps, got_exit=got_exit,
                  got_steps=got_steps, status="FAIL",
                  detail=f"exit mismatch: ref {ref_exit} got {got_exit}", **base)


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------
_STATUSES = ("PASS", "FAIL", "NOT_IMPL", "TIMEOUT", "ERROR", "REF_ERROR")


def _cluster_table(results: List[Result]) -> "OrderedDict[str, Dict[str, int]]":
    table: "OrderedDict[str, Dict[str, int]]" = OrderedDict()
    for r in results:
        row = table.setdefault(
            r.cluster, {"n": 0, **{s: 0 for s in _STATUSES}}
        )
        row["n"] += 1
        row[r.status] += 1
    return table


def _print_cluster_table(table, fh=sys.stdout) -> None:
    print("\nPER-CLUSTER BREAKDOWN", file=fh)
    hdr = (f"  {'cluster':16s} {'n':>4s} {'PASS':>5s} {'FAIL':>5s} "
           f"{'NIMPL':>6s} {'TMOUT':>6s} {'ERR':>4s} {'REFE':>5s} {'pass%':>6s}")
    print(hdr, file=fh)
    print("  " + "-" * (len(hdr) - 2), file=fh)
    for cluster, row in sorted(table.items()):
        n = row["n"]
        pct = (100.0 * row["PASS"] / n) if n else 0.0
        print(
            f"  {cluster:16s} {n:4d} {row['PASS']:5d} {row['FAIL']:5d} "
            f"{row['NOT_IMPL']:6d} {row['TIMEOUT']:6d} {row['ERROR']:4d} "
            f"{row['REF_ERROR']:5d} {pct:6.1f}",
            file=fh,
        )


def _unlock_map(results: List[Result]) -> Tuple[Counter, Counter, Counter]:
    """Build the NOT_IMPL-by-opcode unlock map.

    Returns three counters over opcode name:
      * ``needed``  — programs referencing that op at all (any NOT_IMPL program
        whose ``missing_ops`` contains it),
      * ``sole``    — programs blocked SOLELY by that op (it is the ONLY missing
        op) — building JUST that op unlocks the program IF the run is then
        correct (an upper bound on the immediate unlock),
      * ``ref_ok``  — of the ``sole`` programs, how many the reference actually
        halts on (a REF-halting sole-blocked program is a candidate PASS once the
        op lands; non-halting ones can't be scored anyway).
    """
    needed: Counter = Counter()
    sole: Counter = Counter()
    ref_ok: Counter = Counter()
    for r in results:
        if r.status != "NOT_IMPL":
            continue
        for op in r.missing_ops:
            needed[op] += 1
        if len(r.missing_ops) == 1:
            op = r.missing_ops[0]
            sole[op] += 1
            if r.ref_exit is not None:
                ref_ok[op] += 1
    return needed, sole, ref_ok


def _print_unlock_map(results: List[Result], fh=sys.stdout) -> None:
    needed, sole, ref_ok = _unlock_map(results)
    print("\nNOT_IMPL BY OPCODE — UNLOCK MAP", file=fh)
    print("  (needed = progs referencing the op; sole = progs whose ONLY missing "
          "op is this one;", file=fh)
    print("   sole&ref-halts = sole-blocked progs the reference halts on = "
          "candidate PASSes once built)", file=fh)
    print(f"  {'opcode':8s} {'needed':>7s} {'sole':>6s} {'sole&ref-halts':>15s}",
          file=fh)
    print("  " + "-" * 40, file=fh)
    for op, _ in needed.most_common():
        print(f"  {op:8s} {needed[op]:7d} {sole.get(op, 0):6d} "
              f"{ref_ok.get(op, 0):15d}", file=fh)

    # The single most useful line for the fan-out: rank the ops by SOLE unlock.
    print("\n  NEXT-OP PRIORITY (by sole-blocked programs, ref-halting):", file=fh)
    ranked = sorted(sole.items(), key=lambda kv: ref_ok.get(kv[0], 0),
                    reverse=True)
    for op, n_sole in ranked:
        print(f"    build {op:6s} -> unlocks up to {ref_ok.get(op, 0):4d} progs "
              f"(sole-blocked & ref-halts; {n_sole} sole total)", file=fh)
    if not ranked:
        print("    (no sole-blocked programs — every NOT_IMPL needs 2+ ops)",
              file=fh)


def _print_summary(results: List[Result], wall: float, step_cap: int,
                   fh=sys.stdout) -> None:
    counts = Counter(r.status for r in results)
    total = len(results)
    n_pass = counts.get("PASS", 0)
    print("\n" + "=" * 72, file=fh)
    print("c4_min NIBBLE UNIVERSAL VM SCOREBOARD  (vs ALL 1096)", file=fh)
    print("=" * 72, file=fh)
    print(f"  BUILT nibble ops: {', '.join(sorted(BUILT_OPS))}", file=fh)
    print(f"  compare width: {WIDTH_BITS}-bit (mask 0x{WIDTH_MASK:X}, "
          f"auto-detected from the nibble ALU)", file=fh)
    print(f"  step cap: {step_cap}   wall: {wall:.1f}s", file=fh)
    print("-" * 72, file=fh)
    for s in _STATUSES:
        print(f"  {s:10s} {counts.get(s, 0):5d}", file=fh)
    print("-" * 72, file=fh)
    print(f"  SCORE:  {n_pass}/{total}  ({100.0 * n_pass / total:.2f}%)  "
          f"[nibble universal VM, {WIDTH_BITS}-bit width]", file=fh)
    ref_scorable = sum(1 for r in results if r.status != "REF_ERROR")
    if ref_scorable:
        print(f"  (of {ref_scorable} reference-scorable programs: "
              f"{100.0 * n_pass / ref_scorable:.2f}%)", file=fh)
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
    ap.add_argument("--step-cap", type=int, default=4096,
                    help="Autoregressive step cap (bounds pathological "
                         "non-halting runs). Default 4096.")
    ap.add_argument("--output", type=str, default=None,
                    help="Dump full per-program results + tables as JSON.")
    ap.add_argument("--print-nonpass", action="store_true",
                    help="Print each non-PASS program row.")
    args = ap.parse_args(argv)

    from src.compiler import compile_c
    from tests.declarative_oracle import declarative_oracle_for_program
    from tests.test_suite_1000 import generate_test_programs

    all_tests = generate_test_programs()
    window = list(enumerate(all_tests))[args.offset:]
    if args.limit is not None:
        window = window[:args.limit]

    print(f"[nib-1096] scoring {len(window)} of {len(all_tests)} programs "
          f"| BUILT ops: {', '.join(sorted(BUILT_OPS))} "
          f"| width={WIDTH_BITS}-bit | step_cap={args.step_cap}",
          file=sys.stderr, flush=True)

    t0 = time.monotonic()
    results: List[Result] = []
    for i, (idx, (source, expected, description)) in enumerate(window):
        r = score_program(idx, source, expected, description,
                          compile_c=compile_c,
                          oracle_fn=declarative_oracle_for_program,
                          step_cap=args.step_cap)
        results.append(r)
        if (i + 1) % 100 == 0:
            npass = sum(1 for x in results if x.status == "PASS")
            print(f"[nib-1096] {i + 1}/{len(window)} done "
                  f"(pass so far: {npass})", file=sys.stderr, flush=True)
    wall = time.monotonic() - t0

    results.sort(key=lambda r: r.idx)

    _print_summary(results, wall, args.step_cap)
    table = _cluster_table(results)
    _print_cluster_table(table)
    _print_unlock_map(results)

    if args.print_nonpass:
        print("\nNON-PASS PROGRAMS", file=sys.stdout)
        for r in results:
            if r.status != "PASS":
                print(f"  id={r.idx:04d} [{r.status:9s}] {r.cluster:14s} "
                      f"ref={r.ref_exit} got={r.got_exit} {r.detail}  "
                      f"({r.description})")

    if args.output:
        needed, sole, ref_ok = _unlock_map(results)
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump({
                "wall_seconds": wall,
                "built_ops": sorted(BUILT_OPS),
                "width_bits": WIDTH_BITS,
                "width_mask": WIDTH_MASK,
                "step_cap": args.step_cap,
                "summary": {s: sum(1 for r in results if r.status == s)
                            for s in _STATUSES} | {"total": len(results)},
                "clusters": {k: v for k, v in table.items()},
                "unlock_map": {
                    op: {"needed": needed[op], "sole": sole.get(op, 0),
                         "sole_ref_halts": ref_ok.get(op, 0)}
                    for op in needed
                },
                "results": [asdict(r) for r in results],
            }, fh, indent=2)
        print(f"[nib-1096] wrote {args.output}", file=sys.stderr, flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
