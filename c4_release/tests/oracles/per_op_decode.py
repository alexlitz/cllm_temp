"""Per-op decode ORACLE — the fast, verdict-faithful, per-op-class unit gate.

Why this exists
---------------
Every decode-preserving change (generator-collapse, corrector-delete-with-clean-
root, abstraction/rebuild) has to prove it did not move the per-op decode. The
authoritative signal for that is the full-1096 ``run_1096_canonical --criterion
full_trace`` GPU run — a ~30-min SERIAL gate. That gate is the timeline
bottleneck for op-class work: it forces every small, decode-local change through
a whole-corpus GPU integration run.

This module is the FAST substitute for that gate on decode-preserving,
per-op-class work. It rests on ONE proven foundation:

  ``neural_vm.verification.faithful_autoregressive.FaithfulAutoregressiveRunner``

which drives the PRODUCTION fail-fast / full_trace verdict machinery
(``BatchedPureNeuralRunner``) with a per-token CPU forward
(``ModelExactForward``) that runs the REAL block ``nn.Module``s — i.e. it IS
``model.forward`` over one row. So its per-token argmax is BIT-EXACT to the
neural model, and its per-program ``status`` (pass/fail/error) is byte-identical
to ``tools/run_1096_canonical.py --criterion full_trace --spec-k 0`` — including
the saturated-tie decode positions where the recovered-weight
``CachedFaithfulForward`` false-fails (per the ``faithful_autoregressive``
docstring, this is the WHOLE reason ``ModelExactForward`` exists). NO GPU is
needed — that is the point: this replaces the GPU integration gate for
decode-preserving op-class work.

What "faithful verdict" means here
----------------------------------
For each program the oracle:

  1. Runs the pure-Python ISA VM (``SymbolicDeclarativeProgramRunner`` via
     ``declarative_oracle_for_program``) to get the GROUND-TRUTH exit code +
     step count + per-step (PC, AX) trace. This is the spec, independent of the
     neural model.
  2. Runs ``FaithfulAutoregressiveRunner.run_batch_fail_fast(..., criterion=
     "full_trace", spec_k=32)`` — the production decode — and reads its
     per-program ``status``. ``status == "pass"`` means the neural model's
     decoded (PC, AX) matched the oracle's trace at EVERY completed step (the
     exact full_trace pass criterion the 30-min gate uses).

A per-op-class test PASSES iff every representative program for that op-class
decodes ``pass``. That is the UNIT-level verdict for the op-class.

Op-class coverage
-----------------
The ISA op-classes the mission enumerates are (opcode table in
``neural_vm.verification.symbolic_forward._OPCODE_NAMES``):

    ADD SUB MUL DIV MOD | EQ NE LT GT LE GE | AND OR XOR | SHL SHR |
    LI LC SI SC PSH | LEA IMM JMP JSR ENT ADJ LEV | BZ BNZ

The C test corpus (``tests.test_suite_1000``) EMITS most of these (ADD/SUB/MUL/
DIV/MOD/EQ/LT/GT/LI/SI/PSH/LEA/IMM/JMP/JSR/ENT/ADJ/LEV/BZ) — for those the
oracle draws SHORT representative programs from the corpus (already-compiled,
already-oracled). But the corpus C programs never emit the bitwise/shift ops,
NE/LE/GE, BNZ, or char load/store (LC/SC) — the C compiler canonicalises
comparisons and has no bitwise/char programs. For those op-classes the oracle
carries HAND-AUTHORED bytecode programs (``IMM a; PSH; IMM b; <OP>; EXIT`` for
the binops, an explicit ``BNZ``/``LC``/``SC`` sequence for the rest), which the
ISA VM oracle validates just the same. Every op-class thus has >=1 decode-checked
representative.

Usage
-----
    # Pytest (parametrised over op-classes; ~40-60s per short program):
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" pytest -q \
        tests/oracles/per_op_decode.py

    # Standalone, one shared model build, parallel across op-classes:
    OMP_NUM_THREADS=4 python tests/oracles/per_op_decode.py --op-classes ADD,SUB
    OMP_NUM_THREADS=4 python tests/oracles/per_op_decode.py --list

See ``tools/run_per_op_oracle.py`` for the parallel runner + baseline recorder,
and ``docs/PER_OP_DECODE_ORACLE_2026_07_14.md`` for the gate protocol.

Tooling only: this module never writes weights (it reads the SAME cached baked
model the smoke gate builds), so the golden model is byte-identical.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# The oracle runs a per-token CPU forward; keep it off any GPU and cap threads
# (the memory discipline — one build is ~1.2GB resident).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("C4_SKIP_DIM_INTEGRITY", "1")
os.environ.setdefault("C4_SKIP_GATE_CHECK", "1")
os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")


# ---------------------------------------------------------------------------
# ISA opcode ids (mirrors src.compiler.Op / symbolic_program).
# ---------------------------------------------------------------------------
OP_LEA, OP_IMM, OP_JMP, OP_JSR, OP_BZ, OP_BNZ = 0, 1, 2, 3, 4, 5
OP_ENT, OP_ADJ, OP_LEV = 6, 7, 8
OP_LI, OP_LC, OP_SI, OP_SC, OP_PSH = 9, 10, 11, 12, 13
OP_OR, OP_XOR, OP_AND = 14, 15, 16
OP_EQ, OP_NE, OP_LT, OP_GT, OP_LE, OP_GE = 17, 18, 19, 20, 21, 22
OP_SHL, OP_SHR = 23, 24
OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_MOD = 25, 26, 27, 28, 29
OP_EXIT = 38


def _i(op: int, imm: int = 0) -> int:
    """Encode one instruction: ``opcode | (imm << 8)`` (the c4 encoding)."""
    return int(op) | (int(imm) << 8)


def _binop_prog(a: int, opcode: int, b: int) -> List[int]:
    """``IMM a; PSH; IMM b; <opcode>; EXIT`` -> AX = (a <op> b), the EXIT code.

    This is the exact shape the C compiler emits for ``return a <op> b`` (see
    ``add_0``: ``IMM 654; PSH; IMM 114; ADD; EXIT``), so a raw binop program
    decodes through the SAME per-step path as a corpus arith program.
    """
    return [_i(OP_IMM, a), _i(OP_PSH), _i(OP_IMM, b), _i(opcode), _i(OP_EXIT)]


# ---------------------------------------------------------------------------
# Representative programs.
#
# A representative is either a CORPUS id (drawn from tests.test_suite_1000 —
# already compiled + oracled) or a RAW bytecode program (list of encoded
# instructions + data bytes). Corpus ids are preferred where the C compiler
# emits the op; raw programs cover the op-classes the corpus never reaches.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RawProgram:
    """A hand-authored bytecode program that directly exercises one op-class."""

    label: str
    bytecode: Tuple[int, ...]
    data: bytes = b""


@dataclass(frozen=True)
class OpClassSpec:
    """Everything needed to decode-check one ISA op-class."""

    op: str
    #: SHORT corpus ids that emit this op (preferred; already oracled).
    corpus_ids: Tuple[int, ...] = ()
    #: Hand-authored raw programs (for ops the corpus never emits).
    raw: Tuple[RawProgram, ...] = ()
    note: str = ""


# The op-classes covered by SHORT corpus members (<=~9 oracle steps). The ids
# are chosen from the op->cluster map (see tools/run_per_op_oracle.py --map) as
# the CHEAPEST members that emit the op, so the decode cost stays low. Arith +
# PSH ids are 5-step; the frame/branch ops (LEA/JSR/ENT/ADJ/LEV/LI/SI) live in
# the 7-14-step var_/func_/if_ programs (the shortest members that emit them).
_CORPUS = {
    # --- ALU binops: the 5-step arith clusters ---
    "ADD": (0, 1),          # add_0/add_1
    "SUB": (50, 51),        # sub_0/sub_1
    "MUL": (100, 101),      # mul_0/mul_1
    "DIV": (150, 151),      # div_0/div_1
    "MOD": (200, 201),      # mod_0/mod_1
    # --- comparisons the C compiler DOES emit (branch-compare) ---
    "EQ": (400,),           # if_eq_0 (7 steps)
    "LT": (375,),           # if_lt_0 (7 steps)
    "GT": (350,),           # if_gt_0 (7 steps)
    # --- IMM / EXIT: every program; use the 2-step edge literals ---
    "IMM": (1031, 1022),    # edge_literal / edge (2 steps)
    # --- BZ: the 4-step edge if-programs ---
    "BZ": (1026, 1027),     # edge_if_zero / edge_if_one
    # --- frame + memory ops: shortest var_/func_/if_ members that emit them ---
    "PSH": (0, 100),        # any arith program pushes an operand
    "LEA": (250,),          # var_simple_0 (9 steps)
    "LI": (250,),           # var_simple_0
    "SI": (250,),           # var_simple_0 (store then load)
    "JSR": (250,),          # var_simple_0 (calls main)
    "ENT": (250,),          # var_simple_0 (frame enter)
    "JMP": (1029,),         # edge_loop_never (15 steps)
    "ADJ": (550,),          # func_identity_0 (11 steps)
    "LEV": (550,),          # func_identity_0 (function return)
}

# The op-classes the C corpus never emits — carried as hand-authored raw
# bytecode. Each is a minimal, deterministic program whose ISA-VM exit code the
# declarative oracle computes, so the faithful decode is checked against a real
# spec. Kept short (5-7 steps) so each decodes in ~20-40s.
_RAW: Dict[str, Tuple[RawProgram, ...]] = {
    "AND": (
        RawProgram("and_108_58", tuple(_binop_prog(0x6C, OP_AND, 0x3A))),   # -> 40
        RawProgram("and_255_15", tuple(_binop_prog(0xFF, OP_AND, 0x0F))),   # -> 15
    ),
    "OR": (
        RawProgram("or_108_58", tuple(_binop_prog(0x6C, OP_OR, 0x3A))),     # -> 126
        RawProgram("or_16_1", tuple(_binop_prog(16, OP_OR, 1))),            # -> 17
    ),
    "XOR": (
        RawProgram("xor_108_58", tuple(_binop_prog(0x6C, OP_XOR, 0x3A))),   # -> 86
        RawProgram("xor_255_255", tuple(_binop_prog(0xFF, OP_XOR, 0xFF))),  # -> 0
    ),
    "SHL": (
        RawProgram("shl_5_3", tuple(_binop_prog(5, OP_SHL, 3))),            # -> 40
        RawProgram("shl_1_8", tuple(_binop_prog(1, OP_SHL, 8))),            # -> 256
    ),
    "SHR": (
        RawProgram("shr_200_2", tuple(_binop_prog(200, OP_SHR, 2))),        # -> 50
        RawProgram("shr_255_4", tuple(_binop_prog(0xFF, OP_SHR, 4))),       # -> 15
    ),
    "NE": (
        RawProgram("ne_7_9", tuple(_binop_prog(7, OP_NE, 9))),              # -> 1
        RawProgram("ne_5_5", tuple(_binop_prog(5, OP_NE, 5))),              # -> 0
    ),
    "LE": (
        RawProgram("le_7_9", tuple(_binop_prog(7, OP_LE, 9))),              # -> 1
        RawProgram("le_9_7", tuple(_binop_prog(9, OP_LE, 7))),              # -> 0
    ),
    "GE": (
        RawProgram("ge_9_7", tuple(_binop_prog(9, OP_GE, 7))),             # -> 1
        RawProgram("ge_7_9", tuple(_binop_prog(7, OP_GE, 9))),             # -> 0
    ),
    "BNZ": (
        # IMM 1 leaves AX!=0 so BNZ (rel +2 instr) is TAKEN, skipping the
        # IMM 99; the taken path sets AX=7 and EXITs. Exercises the BNZ
        # branch-taken decode explicitly (the C corpus emits BZ, never BNZ).
        RawProgram(
            "bnz_taken",
            (
                _i(OP_IMM, 1),      # [0] AX=1
                _i(OP_BNZ, 3),      # [1] AX!=0 -> jump to instr idx 3
                _i(OP_IMM, 99),     # [2] (skipped)
                _i(OP_IMM, 7),      # [3] AX=7
                _i(OP_EXIT),        # [4] exit 7
            ),
        ),
        RawProgram(
            "bnz_nottaken",
            (
                _i(OP_IMM, 0),      # [0] AX=0
                _i(OP_BNZ, 3),      # [1] AX==0 -> fall through
                _i(OP_IMM, 42),     # [2] AX=42
                _i(OP_EXIT),        # [3] exit 42
            ),
        ),
    ),
    "LC": (
        # Load a char from the data segment. data[0]=0x5A ('Z'=90). LC reads a
        # single byte at the address in AX; LEA/IMM the data address first.
        # Program: IMM <data_addr>; LC; EXIT. The data segment base is the ISA's
        # data segment base is 0x10000 (SymbolicProgramState.load_data). Load
        # the first data byte (0x5A='Z'=90) and EXIT with it, so the LC decode
        # is checked against a NONZERO spec value.
        RawProgram(
            "lc_data0",
            (_i(OP_IMM, 0x10000), _i(OP_LC), _i(OP_EXIT)),
            data=bytes([0x5A, 0x00, 0x00, 0x00]),
        ),
    ),
    "SC": (
        # Store a char then load it back: IMM addr; PSH; IMM val; SC; ... ; LC.
        # SC stores AX's low byte to *[sp]. We store 0x41 then LC it back and
        # EXIT with it. Exercises the SC store-char decode path.
        RawProgram(
            "sc_then_lc",
            (
                _i(OP_IMM, 0),      # addr
                _i(OP_PSH),         # push addr
                _i(OP_IMM, 0x41),   # val 'A'
                _i(OP_SC),          # store char to *[sp]
                _i(OP_IMM, 0),      # addr again
                _i(OP_LC),          # load char back
                _i(OP_EXIT),
            ),
            data=bytes([0x00, 0x00, 0x00, 0x00]),
        ),
    ),
}


def op_class_specs() -> "Dict[str, OpClassSpec]":
    """Return the full op-class -> spec map (corpus ids + raw programs)."""
    ops = [
        "ADD", "SUB", "MUL", "DIV", "MOD",
        "EQ", "NE", "LT", "GT", "LE", "GE",
        "AND", "OR", "XOR",
        "SHL", "SHR",
        "LI", "LC", "SI", "SC", "PSH",
        "LEA", "IMM", "JMP", "JSR", "ENT", "ADJ", "LEV",
        "BZ", "BNZ",
    ]
    out: "Dict[str, OpClassSpec]" = {}
    for op in ops:
        out[op] = OpClassSpec(
            op=op,
            corpus_ids=tuple(_CORPUS.get(op, ())),
            raw=tuple(_RAW.get(op, ())),
        )
    return out


ALL_OP_CLASSES = tuple(op_class_specs().keys())


# ---------------------------------------------------------------------------
# The verdict engine.
# ---------------------------------------------------------------------------


@dataclass
class ProgramVerdict:
    """One representative program's decode verdict."""

    op: str
    label: str
    steps: int
    oracle_exit: int
    status: str                     # "pass" | "fail" | "error" | "oracle_error"
    divergence_step: Optional[int] = None
    expected_pc: Optional[int] = None
    got_pc: Optional[int] = None
    expected_ax: Optional[int] = None
    got_ax: Optional[int] = None
    note: str = ""
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == "pass"


@dataclass
class OpClassVerdict:
    """Aggregate verdict for one op-class."""

    op: str
    programs: List[ProgramVerdict] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.programs) and all(p.ok for p in self.programs)

    @property
    def n_pass(self) -> int:
        return sum(1 for p in self.programs if p.ok)


# A module-level cached runner + resolved corpus, so a pytest session or the
# standalone runner build the model ONCE.
_RUNNER = None
_CORPUS_CACHE: Optional[Dict[int, Tuple[list, bytes, int, int]]] = None


def _get_runner():
    global _RUNNER
    if _RUNNER is None:
        import contextlib
        import io

        from neural_vm.verification.faithful_autoregressive import (
            FaithfulAutoregressiveRunner,
        )

        with contextlib.redirect_stderr(io.StringIO()):
            _RUNNER = FaithfulAutoregressiveRunner()
    return _RUNNER


def _resolve_corpus(ids: Sequence[int]) -> Dict[int, Tuple[list, bytes, int, int]]:
    """Compile + oracle the requested corpus ids: id -> (bc, data, steps, exit).

    Cached across op-classes so a shared id (e.g. var_simple_0 emits LEA/LI/SI/
    JSR/ENT) is compiled once.
    """
    global _CORPUS_CACHE
    if _CORPUS_CACHE is None:
        _CORPUS_CACHE = {}
    need = [i for i in ids if i not in _CORPUS_CACHE]
    if need:
        import contextlib
        import io

        from tests.test_suite_1000 import generate_test_programs
        from src.compiler import compile_c

        progs = generate_test_programs()
        for i in need:
            src, exp, _desc = progs[i]
            bc, data = compile_c(src)
            with contextlib.redirect_stderr(io.StringIO()):
                o = declarative_oracle_for_program(
                    bc, data, suite_expected=exp, label=f"id={i}", max_steps=60,
                )
            if o.error or o.steps is None:
                _CORPUS_CACHE[i] = (list(bc), bytes(data), -1, -1)
            else:
                _CORPUS_CACHE[i] = (
                    list(bc), bytes(data), int(o.steps), int(o.exit_code),
                )
    return {i: _CORPUS_CACHE[i] for i in ids}


# Imported lazily-usable name (declarative_oracle_for_program) — placed here so
# _resolve_corpus / decode_program can reference it without a top-level heavy
# import at module load (keeps ``--list`` instant).
from tests.declarative_oracle import declarative_oracle_for_program  # noqa: E402


def decode_program(
    bytecode: Sequence[int],
    data: bytes,
    *,
    label: str,
    op: str,
    suite_expected: Optional[int] = None,
    max_steps_cap: int = 60,
) -> ProgramVerdict:
    """Faithfully decode one program and return its full_trace verdict.

    Runs the ISA-VM oracle for the ground-truth (steps + exit + per-step trace),
    then the production ``FaithfulAutoregressiveRunner`` full_trace decode. The
    returned ``status`` is byte-identical to ``run_1096_canonical --criterion
    full_trace --spec-k 0`` (the 30-min gate's per-program criterion).
    """
    import contextlib
    import io
    import time

    with contextlib.redirect_stderr(io.StringIO()):
        o = declarative_oracle_for_program(
            bytecode, data, suite_expected=suite_expected,
            label=label, max_steps=max_steps_cap + 2,
        )
    if o.error or o.steps is None:
        return ProgramVerdict(
            op=op, label=label, steps=-1, oracle_exit=-1,
            status="oracle_error", note=str(o.error),
        )
    steps = int(o.steps)
    if steps > max_steps_cap:
        return ProgramVerdict(
            op=op, label=label, steps=steps, oracle_exit=int(o.exit_code),
            status="skipped", note=f"steps {steps} > cap {max_steps_cap}",
        )
    runner = _get_runner()
    t0 = time.monotonic()
    with contextlib.redirect_stderr(io.StringIO()):
        v = runner.run_batch_fail_fast(
            [list(bytecode)], data_list=[bytes(data)], max_steps=None,
            expected_steps_list=[steps], spec_k=32, criterion="full_trace",
        )[0]
    dt = time.monotonic() - t0
    return ProgramVerdict(
        op=op, label=label, steps=steps, oracle_exit=int(o.exit_code),
        status=str(v.get("status")),
        divergence_step=v.get("divergence_step"),
        expected_pc=v.get("expected_pc"), got_pc=v.get("got_pc"),
        expected_ax=v.get("expected_ax"), got_ax=v.get("got_ax"),
        seconds=dt,
    )


def run_op_class(op: str, *, max_steps_cap: int = 60) -> OpClassVerdict:
    """Decode-check every representative program for one op-class."""
    spec = op_class_specs()[op]
    verdict = OpClassVerdict(op=op)
    # Corpus representatives.
    if spec.corpus_ids:
        resolved = _resolve_corpus(spec.corpus_ids)
        for i in spec.corpus_ids:
            bc, data, steps, exit_ = resolved[i]
            if steps < 0:
                verdict.programs.append(ProgramVerdict(
                    op=op, label=f"corpus#{i}", steps=-1, oracle_exit=-1,
                    status="oracle_error", note="corpus oracle failed",
                ))
                continue
            verdict.programs.append(decode_program(
                bc, data, label=f"corpus#{i}", op=op,
                max_steps_cap=max_steps_cap,
            ))
    # Raw representatives.
    for raw in spec.raw:
        verdict.programs.append(decode_program(
            list(raw.bytecode), raw.data, label=raw.label, op=op,
            max_steps_cap=max_steps_cap,
        ))
    return verdict


# ---------------------------------------------------------------------------
# Pytest entry points (parametrised over op-classes).
# ---------------------------------------------------------------------------

try:
    import pytest

    @pytest.mark.parametrize("op", ALL_OP_CLASSES)
    def test_per_op_decode(op: str) -> None:
        """Each op-class: every representative program decodes ``pass``."""
        v = run_op_class(op)
        assert v.programs, f"{op}: no representative programs configured"
        fails = [p for p in v.programs if not p.ok and p.status != "skipped"]
        detail = "; ".join(
            f"{p.label}[{p.status}"
            + (f" @step {p.divergence_step} exp(pc={p.expected_pc},ax={p.expected_ax})"
               f" got(pc={p.got_pc},ax={p.got_ax})" if p.status == "fail" else "")
            + (f" {p.note}" if p.note else "")
            + "]"
            for p in fails
        )
        assert not fails, (
            f"{op}: {len(fails)}/{len(v.programs)} representative programs "
            f"did NOT decode pass: {detail}"
        )
except ImportError:  # pytest not installed — module still importable/runnable.
    pass


# ---------------------------------------------------------------------------
# Standalone CLI.
# ---------------------------------------------------------------------------


def _main(argv: Optional[List[str]] = None) -> int:
    import argparse
    import json
    import sys
    import time

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--op-classes", default=None,
                    help="comma-separated op-classes (default: all).")
    ap.add_argument("--list", action="store_true",
                    help="print the op-class -> representative map and exit.")
    ap.add_argument("--map", action="store_true",
                    help="print the op->corpus-cluster derivation map and exit.")
    ap.add_argument("--max-steps-cap", type=int, default=60)
    ap.add_argument("--json", default=None, help="write per-program verdicts to JSON.")
    args = ap.parse_args(argv)

    specs = op_class_specs()

    if args.list:
        for op, s in specs.items():
            reps = [f"corpus#{i}" for i in s.corpus_ids] + [r.label for r in s.raw]
            print(f"  {op:5s}: {reps}")
        return 0

    ops = (
        [o.strip().upper() for o in args.op_classes.split(",") if o.strip()]
        if args.op_classes else list(specs)
    )
    unknown = [o for o in ops if o not in specs]
    if unknown:
        print(f"unknown op-classes: {unknown}", file=sys.stderr)
        return 2

    t0 = time.monotonic()
    results: List[OpClassVerdict] = []
    for op in ops:
        v = run_op_class(op, max_steps_cap=args.max_steps_cap)
        results.append(v)
        secs = sum(p.seconds for p in v.programs)
        badge = "PASS" if v.ok else "FAIL"
        print(f"[{badge}] {op:5s} {v.n_pass}/{len(v.programs)} progs "
              f"({secs:.0f}s):", flush=True)
        for p in v.programs:
            extra = ""
            if p.status == "fail":
                extra = (f" @step {p.divergence_step} "
                         f"exp(pc={p.expected_pc},ax={p.expected_ax}) "
                         f"got(pc={p.got_pc},ax={p.got_ax})")
            elif p.note:
                extra = f" {p.note}"
            print(f"       {p.label:16s} steps={p.steps:3d} "
                  f"oracle_exit={p.oracle_exit:<6} -> {p.status}{extra}")

    n_ok = sum(1 for v in results if v.ok)
    print(f"\nOP-CLASS PASS: {n_ok}/{len(results)}   "
          f"(total wall {time.monotonic() - t0:.0f}s)")

    if args.json:
        payload = {
            "op_classes": {
                v.op: {
                    "ok": v.ok,
                    "programs": [
                        {
                            "label": p.label, "steps": p.steps,
                            "oracle_exit": p.oracle_exit, "status": p.status,
                            "divergence_step": p.divergence_step,
                            "seconds": round(p.seconds, 2),
                        }
                        for p in v.programs
                    ],
                }
                for v in results
            }
        }
        json.dump(payload, open(args.json, "w"), indent=1)
        print(f"wrote {args.json}")

    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(_main())
