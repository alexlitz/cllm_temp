#!/usr/bin/env python3
"""Program-level DSL spec-conformance gate.

The MISSING structural-correctness gate. ``tests/test_dsl_interpreter.py``
checks *op-level* semantics in isolation; nothing runs whole 1096/smoke
PROGRAMS through the declarative op pipeline and compares the result to
the declarative oracle. This driver closes that gap.

What it does
------------

For a program's bytecode it:

1. Drives the autoregressive VM loop SYMBOLICALLY — for every VM step it
   builds the step's AX-marker residual (register/opcode/operand context
   via :func:`symbolic_state_builder.state_after_program`), runs the full
   per-step declarative op pipeline (``all_core_ops`` placed into
   ``ops_per_layer`` by the real ``LayerCompiler``, declarations-only so
   no GPU/weights/cache are touched) through :class:`DSLInterpreter`, and
   inspects the resulting OUTPUT residual.
2. Computes the declarative oracle's exit code
   (:func:`declarative_oracle_for_program`) as ground truth.
3. Classifies each program:

   * **UNINTERPRETABLE** — the program's op path hits an *opaque*
     (imperative ``bake_fn``-only, no IR) op the interpreter cannot
     execute. A coverage gap the 100%-declarative migration closes.
   * **AUTHORING-MISMATCH** — every op on its path is declarative, yet
     the declared rules provably disagree with C4 (no rule on the path
     writes the oracle-expected OUTPUT nibble — a real rule-level /
     structural bug).
   * **PASS** — interpretable AND the declared rules do produce the
     oracle-expected OUTPUT nibble.

Why the structural (not numeric) authoring check
-------------------------------------------------

This gate is CACHE-IMMUNE and GPU-free by construction: it never lowers
to weights. :class:`DSLInterpreter` approximates attention as a
context-free V→O copy and applies every FFN rule whose linear
condition-sum clears threshold — it does NOT model the softmax / argmax
competition that selects the winning OUTPUT one-hot in the real
transformer. So a naive "argmax OUTPUT == oracle exit code" decode is
NOT faithful (see ``--faithfulness`` — the argmax is dominated by
static-default rules). The gate's authoring verdict therefore rests on a
RULE-COVERAGE invariant that the interpreter CAN evaluate faithfully:

    Does ANY declared rule on the program's path write the
    oracle-expected OUTPUT nibble cell at all?

If the expected cell receives zero declared writers, no rule produces the
value C4 needs — a provable authoring/structural bug (this is exactly how
the known multi-byte sub/add/div operand-relay/cascade gaps manifest: the
result nibble cell is simply never written). It is a NECESSARY condition
for correctness, not a sufficient one — a cell can receive a write yet
lose the argmax race. The gate reports both the coverage verdict and the
faithfulness probe so the distinction is explicit.

Usage
-----

    CUDA_VISIBLE_DEVICES="" python tools/dsl_spec_gate.py --smoke
    CUDA_VISIBLE_DEVICES="" python tools/dsl_spec_gate.py --sample-1096 60
    CUDA_VISIBLE_DEVICES="" python tools/dsl_spec_gate.py --opaque-inventory
    CUDA_VISIBLE_DEVICES="" python tools/dsl_spec_gate.py --faithfulness
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Make ``c4_release`` importable when run directly.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)               # .../c4_release
_ROOT = os.path.dirname(_PKG)               # repo root
for p in (_PKG, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

# Silence the (expected) compile-time integrity / cross-step / gate-audit
# warnings — they fire on every declarations-only build and are not part of
# this gate's signal.
os.environ.setdefault("C4_SKIP_DIM_INTEGRITY", "1")
os.environ.setdefault("C4_SKIP_GATE_CHECK", "1")
warnings.filterwarnings("ignore")

from neural_vm.unified_compiler.dsl_interpreter import (  # noqa: E402
    DSLInterpreter,
)
from neural_vm.unified_compiler.symbolic_forward import (  # noqa: E402
    default_embedding_for_instruction,
)
from neural_vm.unified_compiler.symbolic_program import (  # noqa: E402
    SymbolicDeclarativeProgramRunner,
)
from tests.declarative_oracle import (  # noqa: E402
    declarative_oracle_for_program,
)


# ---------------------------------------------------------------------------
# Opcode -> indicator-dim name (the OP_<NAME> dispatch flag the op pipeline
# gates on). Mirrors symbolic_forward._OPCODE_NAMES.
# ---------------------------------------------------------------------------

_OPCODE_FLAG = {
    0: "OP_LEA", 1: "OP_IMM", 2: "OP_JMP", 3: "OP_JSR", 4: "OP_BZ",
    5: "OP_BNZ", 6: "OP_ENT", 7: "OP_ADJ", 8: "OP_LEV", 9: "OP_LI",
    10: "OP_LC", 11: "OP_SI", 12: "OP_SC", 13: "OP_PSH",
    14: "OP_OR", 15: "OP_XOR", 16: "OP_AND", 17: "OP_EQ", 18: "OP_NE",
    19: "OP_LT", 20: "OP_GT", 21: "OP_LE", 22: "OP_GE", 23: "OP_SHL",
    24: "OP_SHR", 25: "OP_ADD", 26: "OP_SUB", 27: "OP_MUL", 28: "OP_DIV",
    29: "OP_MOD", 38: "OP_EXIT",
}

# Classification labels.
#
# PASS / AUTHORING_MISMATCH are the brief's interpretable verdicts. Because
# DSLInterpreter cannot faithfully decode the OUTPUT *value* (attention /
# softmax / argmax abstraction — see --faithfulness), the interpretable
# verdict rests on the RULE-COVERAGE necessary condition: PASS = the
# oracle-expected OUTPUT nibble cell receives a declared writer; AUTHORING-
# MISMATCH = it receives NONE (no rule produces the value C4 needs). PASS is
# a necessary-not-sufficient verdict and is reported as such.
PASS = "PASS"
AUTHORING_MISMATCH = "AUTHORING-MISMATCH"
UNINTERPRETABLE = "UNINTERPRETABLE"
ORACLE_ERROR = "ORACLE-ERROR"


# ---------------------------------------------------------------------------
# Layout / op-index — built once, reused across every program.
# ---------------------------------------------------------------------------


@dataclass
class _OpInfo:
    name: str
    kind: str
    has_ir: bool
    reads: frozenset
    writes: frozenset


@dataclass
class GateContext:
    """Built-once context: the declarations-only layout, a flat op list in
    block order, and the per-op IR/opaque classification."""

    layout: Any
    dim_positions: Dict[str, int]
    flat_ops: List[Any]                       # Operation objects, block order
    op_infos: List[_OpInfo]
    # opcode_flag -> list of op infos that GATE on that flag (reads OP_<NAME>)
    ops_by_opcode: Dict[str, List[_OpInfo]] = field(default_factory=dict)

    def opaque_ops(self) -> List[_OpInfo]:
        return [oi for oi in self.op_infos if not oi.has_ir]


def _op_has_ir(op: Any, dim_positions: Dict[str, int]) -> bool:
    """An op is interpretable iff the DSLInterpreter can extract a
    ``CompilerIR`` from it (pre-built ``compiler_ir`` or a
    ``compiler_ir_factory`` that builds without error). Otherwise the op's
    bake is purely imperative and opaque to symbolic interpretation."""
    if getattr(op, "compiler_ir", None) is not None:
        return True
    factory = getattr(op, "compiler_ir_factory", None)
    if factory is not None:
        try:
            factory(dim_positions, 8)
            return True
        except Exception:
            return False
    return False


def build_gate_context(*, verbose: bool = True) -> GateContext:
    """Compile the declarations-only layout and index its ops.

    declarations_only=True + disk_cache=False mirrors
    ``replay_expected_diff._build_compiler``: we only need
    ``ops_per_layer`` + ``dim_positions`` (no weights), and the on-disk
    pickle drops ``compiler_ir_factory`` lambdas — which would wrongly
    mark factory-only ops as opaque. In-memory compile is CPU-only (~8s).
    """
    import contextlib
    import io

    from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic,
    )

    if verbose:
        print("[dsl_spec_gate] compiling declarations-only layout "
              "(CPU, ~8s)...", file=sys.stderr)
    # The compiler prints a verbose right-sizing / contract report to
    # stdout; redirect it so the gate's report stays clean.
    with contextlib.redirect_stdout(io.StringIO()):
        _model, layout = compile_full_vm_dynamic(
            declarations_only=True, disk_cache=False,
        )
    dim_positions = dict(layout.dim_positions)

    flat_ops: List[Any] = []
    for block in layout.ops_per_layer:
        flat_ops.extend(block)
    flat_ops.extend(list(getattr(layout, "block_ops", []) or []))
    flat_ops.extend(list(getattr(layout, "model_ops", []) or []))

    op_infos: List[_OpInfo] = []
    by_op: Dict[str, List[_OpInfo]] = {}
    for op in flat_ops:
        info = _OpInfo(
            name=getattr(op, "name", "<anon>"),
            kind=getattr(op, "kind", "ffn"),
            has_ir=_op_has_ir(op, dim_positions),
            reads=frozenset(getattr(op, "reads", ()) or ()),
            writes=frozenset(getattr(op, "writes", ()) or ()),
        )
        op_infos.append(info)
        for flag in _OPCODE_FLAG.values():
            if flag in info.reads:
                by_op.setdefault(flag, []).append(info)

    ctx = GateContext(
        layout=layout,
        dim_positions=dim_positions,
        flat_ops=flat_ops,
        op_infos=op_infos,
        ops_by_opcode=by_op,
    )
    if verbose:
        n_ir = sum(1 for oi in op_infos if oi.has_ir)
        print(f"[dsl_spec_gate] layout: {len(op_infos)} ops, "
              f"{n_ir} interpretable, {len(op_infos) - n_ir} opaque; "
              f"{len(layout.ops_per_layer)} blocks.", file=sys.stderr)
    return ctx


# ---------------------------------------------------------------------------
# Per-program classification
# ---------------------------------------------------------------------------


@dataclass
class GateResult:
    name: str
    classification: str
    oracle_exit: Optional[int]
    n_steps: int
    opcodes: List[str]
    # Opaque ops that gate on an opcode actually used by the program.
    opaque_on_path: List[str] = field(default_factory=list)
    # Authoring detail (interpretable programs only).
    expected_lo_cell: Optional[int] = None
    expected_hi_cell: Optional[int] = None
    lo_writers: int = 0
    hi_writers: int = 0
    # Faithfulness probe (argmax of accumulated OUTPUT contributions).
    decoded_argmax: Optional[int] = None
    note: str = ""


_ORACLE_RUNNER = SymbolicDeclarativeProgramRunner()


def _oracle_trace(bytecode: Sequence[int], data: Sequence[int] | bytes = b""):
    """Run the program through the real declarative oracle and return its
    per-step trace. The oracle (``SymbolicDeclarativeProgramRunner``)
    resolves branches correctly (PC<->index via ``resolve_static_target_idx``),
    so the trace is the authoritative reachable-opcode path AND the
    register-state source for the exit-step seed. This sidesteps the
    ``dim_oracle.ReferenceOracle``'s naive JMP (which loops on small
    immediate targets)."""
    state = _ORACLE_RUNNER.run(bytecode, data, max_steps=4000)
    return state.trace


def _opcodes_used(trace) -> Tuple[List[str], List[int]]:
    """Return (flag-name list, opcode-int list) for every executed step
    from the oracle trace (taken/untaken branches honoured)."""
    flags: List[str] = []
    ints: List[int] = []
    for tr in trace:
        ints.append(tr.opcode)
        flags.append(_OPCODE_FLAG.get(tr.opcode, f"OP_{tr.opcode}"))
    return flags, ints


def _exit_trace_step(trace):
    """Return the trace entry whose AX-marker residual carries the exit
    code: the halting step (``halted=True``) or the last executed step."""
    for tr in trace:
        if tr.halted:
            return tr
    return trace[-1] if trace else None


def _seed_ax_marker_from_trace(tr) -> Dict[str, float]:
    """Build the AX-marker-position residual for one trace step.

    Mirrors ``symbolic_state_builder.state_after_program`` (opcode flag +
    IMM payload + register/operand nibble one-hots) but sourced from the
    oracle trace's register state, so it is correct through branches.
    ``ax_before`` is the operand/result the EXIT step's compute sees; for
    the halting EXIT step the AX value IS the exit code (= ``ax_after``)."""
    ax = tr.ax_after & 0xFFFFFFFF
    sp = tr.sp_after & 0xFFFFFFFF
    bp = tr.bp_after & 0xFFFFFFFF
    state: Dict[str, float] = dict(
        default_embedding_for_instruction(tr.opcode, tr.imm & 0xFFFFFF,
                                          pc=tr.pc_after)
    )

    def emit(reg: str, value: int) -> None:
        for h in range(4):
            byte_val = (value >> (h * 8)) & 0xFF
            state[f"REG_{reg}_BYTE{h}_LO+{byte_val & 0x0F}"] = 1.0
            state[f"REG_{reg}_BYTE{h}_HI+{(byte_val >> 4) & 0x0F}"] = 1.0

    emit("AX", ax)
    emit("SP", sp)
    emit("BP", bp)
    emit("PC", tr.pc_after & 0xFFFFFFFF)
    # STACK0 / AX_CARRY operand staging — AX byte values feed OUTPUT.
    for h in range(4):
        byte_val = (ax >> (h * 8)) & 0xFF
        state[f"STACK0_BYTE{h}_LO+{byte_val & 0x0F}"] = 1.0
        state[f"STACK0_BYTE{h}_HI+{(byte_val >> 4) & 0x0F}"] = 1.0
    b0 = ax & 0xFF
    state[f"AX_CARRY_LO+{b0 & 0x0F}"] = 1.0
    state[f"AX_CARRY_HI+{(b0 >> 4) & 0x0F}"] = 1.0
    return state


def _output_writers_for_cell(
    seed: Dict[str, float], ctx: GateContext, cell: int, family: str,
) -> Tuple[int, Dict[str, float]]:
    """Count declared op-pipeline writers to ``family+cell`` when the
    EXIT-step AX-marker residual ``seed`` is run through the full op
    pipeline. Returns (writer_count, final_state)."""
    interp = DSLInterpreter(initial_state=dict(seed),
                            dim_positions=ctx.dim_positions)
    result = interp.run(ctx.flat_ops)
    key = f"{family}+{cell}"
    writers = [(n, v) for n, v in result.writes_to(key) if abs(v) > 1e-9]
    return len(writers), interp.state


def _argmax_byte0(state: Dict[str, float]) -> Optional[int]:
    """Decode AX byte 0 from accumulated OUTPUT_LO/HI contributions
    (faithfulness probe — known to be dominated by static-default rules)."""
    def amax(fam: str) -> Optional[int]:
        cells = {
            int(k.split("+")[1]): v
            for k, v in state.items()
            if k.startswith(fam + "+") and ".*" not in k
        }
        if not cells:
            return None
        return max(cells, key=lambda c: cells[c])

    lo = amax("OUTPUT_LO")
    hi = amax("OUTPUT_HI")
    if lo is None or hi is None:
        return None
    return (hi << 4) | lo


def classify_program(
    ctx: GateContext,
    name: str,
    bytecode: Sequence[int],
    data: Sequence[int] | bytes = b"",
    *,
    suite_check=None,
) -> GateResult:
    """Run one program through the gate and classify it."""
    # 1. Oracle ground truth.
    oracle_res = declarative_oracle_for_program(
        bytecode, data, suite_check=suite_check, label=name,
    )
    trace = _oracle_trace(bytecode, data)
    flags, ints = _opcodes_used(trace)

    if oracle_res.error is not None or oracle_res.exit_code is None:
        return GateResult(
            name=name, classification=ORACLE_ERROR,
            oracle_exit=oracle_res.exit_code, n_steps=len(ints),
            opcodes=flags, note=oracle_res.error or "no exit code",
        )

    oracle_exit = oracle_res.exit_code & 0xFFFFFFFF

    # 2. Coverage: opaque ops gating on opcodes the program actually runs.
    used_flags = set(flags)
    opaque_on_path: List[str] = []
    for flag in used_flags:
        for oi in ctx.ops_by_opcode.get(flag, ()):
            if not oi.has_ir:
                opaque_on_path.append(f"{oi.name}({flag})")
    opaque_on_path = sorted(set(opaque_on_path))

    if opaque_on_path:
        return GateResult(
            name=name, classification=UNINTERPRETABLE,
            oracle_exit=oracle_exit, n_steps=len(ints), opcodes=flags,
            opaque_on_path=opaque_on_path,
            note=("opcode path hits opaque (imperative bake_fn) op(s): "
                  + ", ".join(opaque_on_path)),
        )

    # 3. Authoring (rule-coverage) check: does any declared rule write the
    #    oracle-expected OUTPUT byte-0 nibble cell?
    exit_tr = _exit_trace_step(trace)
    if exit_tr is None:
        return GateResult(
            name=name, classification=ORACLE_ERROR,
            oracle_exit=oracle_exit, n_steps=len(ints), opcodes=flags,
            note="could not locate exit step",
        )

    byte0 = oracle_exit & 0xFF
    lo_cell = byte0 & 0x0F
    hi_cell = (byte0 >> 4) & 0x0F

    seed = _seed_ax_marker_from_trace(exit_tr)
    lo_writers, _state = _output_writers_for_cell(
        seed, ctx, lo_cell, "OUTPUT_LO",
    )
    hi_writers, final_state = _output_writers_for_cell(
        seed, ctx, hi_cell, "OUTPUT_HI",
    )
    decoded = _argmax_byte0(final_state)

    # AUTHORING-MISMATCH: the expected result nibble is never written by any
    # declared rule on the path -> no rule produces the value C4 needs. This
    # is the only value-level structural bug the interpreter can prove for
    # byte-0 (the byte-1+ result of a multi-byte op is NOT representable —
    # OUTPUT_LO/HI is byte-0-only; multi-byte byte-h routing is positional
    # and collapses in the bag-of-dims interpreter). PASS is a rule-COVERAGE
    # verdict (the expected nibble is reachable by some declared rule), NOT a
    # value-verified pass: byte-0 OUTPUT is flooded by ~15 static-default
    # writers per cell so the winning one-hot is an attention/argmax
    # phenomenon the interpreter abstracts away (see --faithfulness).
    if lo_writers == 0 or hi_writers == 0:
        classification = AUTHORING_MISMATCH
        note = (f"expected OUTPUT byte0=0x{byte0:02x} "
                f"(LO+{lo_cell}: {lo_writers} writers, "
                f"HI+{hi_cell}: {hi_writers} writers) — "
                f"expected nibble cell has NO declared writer "
                f"(no rule produces C4's value)")
    else:
        classification = PASS
        note = (f"coverage-PASS: expected OUTPUT byte0=0x{byte0:02x} "
                f"reachable (LO+{lo_cell}: {lo_writers} writers, "
                f"HI+{hi_cell}: {hi_writers} writers); value-decode "
                f"unfaithful (static-default flood)")

    return GateResult(
        name=name, classification=classification,
        oracle_exit=oracle_exit, n_steps=len(ints), opcodes=flags,
        expected_lo_cell=lo_cell, expected_hi_cell=hi_cell,
        lo_writers=lo_writers, hi_writers=hi_writers,
        decoded_argmax=decoded, note=note,
    )


# ---------------------------------------------------------------------------
# Program sources
# ---------------------------------------------------------------------------


def _smoke_programs() -> List[Dict[str, Any]]:
    """The ~41 smoke programs from tests/test_smoke.py (bytecode + check)."""
    from tests.test_smoke import _SMOKE_GROUPS

    out: List[Dict[str, Any]] = []
    for _group, tests in _SMOKE_GROUPS.items():
        for t in tests:
            out.append({
                "name": t["name"].split("::")[-1],
                "bytecode": t["bytecode"],
                "check": t.get("check"),
            })
    return out


def _sample_1096_programs(n: int) -> List[Dict[str, Any]]:
    """Compile a representative sample of the 1096 corpus to bytecode.

    Prioritises the clusters the task flags as having KNOWN structural
    bugs (add / sub / div / mod / if_*), then fills out with the rest.
    """
    from tests.test_suite_1000 import generate_test_programs
    from src.compiler import compile_c

    all_tests = generate_test_programs()  # (source, expected, description)

    def cluster(desc: str) -> str:
        d = desc.lower()
        for key in ("subtract", "divide", "modulo", "add", "sub", "div",
                    "mod", "mul", "multipl", "if", "var", "comparison",
                    "ternary", "loop", "func"):
            if key in d:
                return {"subtract": "sub", "divide": "div", "modulo": "mod",
                        "multipl": "mul"}.get(key, key)
        return "other"

    # Prioritise the known-bug clusters.
    priority = ("sub", "div", "mod", "add", "if", "mul", "var")
    buckets: Dict[str, List[Tuple[str, int, str]]] = {}
    for src, expected, desc in all_tests:
        buckets.setdefault(cluster(desc), []).append((src, expected, desc))

    selected: List[Tuple[str, int, str]] = []
    order = list(priority) + [c for c in buckets if c not in priority]
    idx = {c: 0 for c in buckets}
    while len(selected) < n:
        progressed = False
        for c in order:
            if c not in buckets:
                continue
            if idx[c] < len(buckets[c]):
                selected.append(buckets[c][idx[c]])
                idx[c] += 1
                progressed = True
                if len(selected) >= n:
                    break
        if not progressed:
            break

    out: List[Dict[str, Any]] = []
    for src, expected, desc in selected:
        try:
            bytecode, prog_data = compile_c(src)
        except Exception as exc:
            out.append({"name": desc[:48], "compile_error": repr(exc)})
            continue
        out.append({
            "name": desc[:48],
            "bytecode": bytecode,
            "data": prog_data,
            "expected": expected,
        })
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _run_set(ctx: GateContext, programs: List[Dict[str, Any]], title: str):
    print("=" * 72)
    print(f"  {title}  ({len(programs)} programs)")
    print("=" * 72)
    results: List[GateResult] = []
    for p in programs:
        if "compile_error" in p:
            print(f"  COMPILE-ERR  {p['name']}: {p['compile_error']}")
            continue
        try:
            r = classify_program(
                ctx, p["name"], p["bytecode"], p.get("data", b""),
                suite_check=p.get("check"),
            )
        except Exception as exc:
            print(f"  GATE-ERR     {p['name']}: {exc!r}")
            continue
        results.append(r)
        print(f"  {r.classification:<20} {r.name}  ({r.note})")

    _summary(results, title)
    return results


def _summary(results: List[GateResult], title: str) -> None:
    n = len(results)
    if n == 0:
        return
    counts: Dict[str, int] = {}
    for r in results:
        counts[r.classification] = counts.get(r.classification, 0) + 1
    interpretable = [
        r for r in results
        if r.classification in (PASS, AUTHORING_MISMATCH)
    ]
    n_interp = len(interpretable)
    n_pass = counts.get(PASS, 0)
    print("-" * 72)
    print(f"  SUMMARY [{title}]: {n} programs")
    for k in (PASS, AUTHORING_MISMATCH, UNINTERPRETABLE, ORACLE_ERROR):
        if counts.get(k):
            print(f"    {k:<20} {counts[k]}")
    cov = (100.0 * n_interp / n) if n else 0.0
    pr = (100.0 * n_pass / n_interp) if n_interp else 0.0
    print(f"    coverage (interpretable / total):  {n_interp}/{n} "
          f"({cov:.0f}%)   [FAITHFUL signal]")
    print(f"    interpretable coverage-PASS rate:  {n_pass}/{n_interp} "
          f"({pr:.0f}%)   [necessary-condition; value-decode unfaithful]")
    mism = [r for r in results if r.classification == AUTHORING_MISMATCH]
    if mism:
        print(f"  AUTHORING-MISMATCH list ({len(mism)} — declared rules "
              f"provably disagree with C4; expected nibble has no writer):")
        for r in mism:
            print(f"    - {r.name}: exit=0x{(r.oracle_exit or 0):x} "
                  f"opcodes={'/'.join(dict.fromkeys(r.opcodes))} :: {r.note}")
    else:
        print("  AUTHORING-MISMATCH list: (none with a zero-writer expected "
              "nibble — see report notes on why byte-0 cannot surface the "
              "multi-byte routing bugs)")
    unint = [r for r in results if r.classification == UNINTERPRETABLE]
    if unint:
        opc = sorted({o.split("(")[-1].rstrip(")")
                      for r in unint for o in r.opaque_on_path})
        print(f"  UNINTERPRETABLE: {len(unint)} programs gated on opaque "
              f"ALU compute (opcodes: {', '.join(opc)}) — the coverage gap "
              f"the 100%-declarative migration closes.")
    print()


def _opaque_inventory(ctx: GateContext) -> None:
    print("=" * 72)
    print("  OPAQUE-OP INVENTORY (imperative bake_fn-only; no IR — the "
          "coverage")
    print("  gap the 100%-declarative migration must close for full gate "
          "coverage)")
    print("=" * 72)
    opaque = ctx.opaque_ops()
    n = len(ctx.op_infos)
    print(f"  {len(opaque)}/{n} ops are opaque "
          f"({100.0 * len(opaque) / n:.1f}%).\n")
    by_kind: Dict[str, List[_OpInfo]] = {}
    for oi in opaque:
        by_kind.setdefault(oi.kind, []).append(oi)
    for kind in sorted(by_kind):
        print(f"  kind={kind}: {len(by_kind[kind])}")
        for oi in by_kind[kind]:
            gated = sorted(f for f in oi.reads if f.startswith("OP_"))
            gate_desc = (" gates_on=" + ",".join(gated)) if gated else ""
            print(f"      {oi.name}{gate_desc}")
    print()


def _faithfulness_probe(ctx: GateContext) -> None:
    """Validate the interpreter's faithfulness on a few PASSING neural
    tests: the OUTPUT argmax SHOULD recover the oracle exit code if the
    interpreter modelled attention/argmax faithfully. It does not — this
    documents the divergence (attention softmax/ALiBi approximation)."""
    print("=" * 72)
    print("  INTERPRETER FAITHFULNESS PROBE")
    print("=" * 72)
    print("  Decoding AX byte0 from accumulated OUTPUT_LO/HI contributions")
    print("  (argmax) and comparing to the oracle exit code. A faithful")
    print("  interpreter would match; divergence => the V→O / softmax / "
          "argmax")
    print("  abstraction loses value-routing (documented, expected).\n")
    wanted = ("test_imm_exit", "test_and_basic", "test_sub_basic",
              "test_add_basic", "test_or_basic", "test_mul_basic")
    by_name = {p["name"]: p for p in _smoke_programs()}
    for nm in wanted:
        p = by_name.get(nm)
        if p is None:
            continue
        r = classify_program(
            ctx, nm, p["bytecode"], suite_check=p.get("check"),
        )
        exp = r.oracle_exit & 0xFF if r.oracle_exit is not None else None
        got = r.decoded_argmax
        match = "MATCH" if (got is not None and exp == got) else "DIVERGE"
        print(f"  {match:<8} {nm}: oracle byte0=0x{(exp or 0):02x} "
              f"argmax-decode=0x{(got or 0):02x}  "
              f"[expected-cell writers LO={r.lo_writers} HI={r.hi_writers}]")
    print("\n  FINDING: the argmax decode diverges because OUTPUT one-hot")
    print("  selection is an attention/softmax phenomenon the interpreter")
    print("  abstracts away. The gate's authoring verdict therefore uses the")
    print("  rule-COVERAGE invariant (is the expected nibble written at all),")
    print("  which the interpreter evaluates faithfully.\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke", action="store_true",
                    help="Run the ~41 smoke programs.")
    ap.add_argument("--sample-1096", type=int, default=0, metavar="N",
                    help="Compile + gate N sampled 1096 programs "
                         "(add/sub/div/mod/if_* clusters first).")
    ap.add_argument("--opaque-inventory", action="store_true",
                    help="Dump the opaque-op inventory and exit.")
    ap.add_argument("--faithfulness", action="store_true",
                    help="Run the interpreter-faithfulness probe.")
    ap.add_argument("--all", action="store_true",
                    help="opaque-inventory + faithfulness + smoke + "
                         "sample-1096 40.")
    args = ap.parse_args(argv)

    if not any([args.smoke, args.sample_1096, args.opaque_inventory,
                args.faithfulness, args.all]):
        args.all = True

    ctx = build_gate_context()
    print()

    if args.opaque_inventory or args.all:
        _opaque_inventory(ctx)
    if args.faithfulness or args.all:
        _faithfulness_probe(ctx)
    if args.smoke or args.all:
        _run_set(ctx, _smoke_programs(), "SMOKE PROGRAMS")
    n1096 = args.sample_1096 or (40 if args.all else 0)
    if n1096:
        _run_set(ctx, _sample_1096_programs(n1096),
                 f"1096 SAMPLE (n={n1096})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
