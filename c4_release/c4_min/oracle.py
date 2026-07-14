"""c4_min per-op ORACLE — expected-output computer + per-opcode program generator.

This is the green-field counterpart to ``tests/oracles/per_op_decode.py`` (the
reference model's verdict-faithful per-op gate). It has two jobs:

  1. GENERATE a small program per ISA opcode (a few 8-bit operand cases each),
     as ``Program`` objects (``bytecode`` + ``data``).
  2. COMPUTE the EXPECTED ISA output for each program — the ground truth — by
     reusing the SAME reference semantics the per-op oracle trusts:
     ``neural_vm.verification.symbolic_program.SymbolicDeclarativeProgramRunner``.

The reference ISA VM is imported for the EXPECTED side ONLY. The
model-under-test side (``c4_min.compile.compile_program`` +
``c4_min.model.run``) is c4_min-only; the runner (``run_oracle.py``) never lets
neural_vm touch the model-under-test path.

Nothing here builds or reads any neural weights — it is pure Python and fast.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# ISA opcode ids (the c4 encoding; mirrors symbolic_program's opcode table and
# tests/oracles/per_op_decode.py).
# ---------------------------------------------------------------------------
OP_LEA, OP_IMM, OP_JMP, OP_JSR, OP_BZ, OP_BNZ = 0, 1, 2, 3, 4, 5
OP_ENT, OP_ADJ, OP_LEV = 6, 7, 8
OP_LI, OP_LC, OP_SI, OP_SC, OP_PSH = 9, 10, 11, 12, 13
OP_OR, OP_XOR, OP_AND = 14, 15, 16
OP_EQ, OP_NE, OP_LT, OP_GT, OP_LE, OP_GE = 17, 18, 19, 20, 21, 22
OP_SHL, OP_SHR = 23, 24
OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_MOD = 25, 26, 27, 28, 29
OP_EXIT = 38

#: The data segment base the reference VM loads ``data`` at (see
#: ``SymbolicProgramState.load_data`` / ``constants.STACK_INIT``).
DATA_BASE = 0x10000

# The op-classes the harness enumerates (the same 30 the reference oracle uses).
ALL_OP_CLASSES: Tuple[str, ...] = (
    "ADD", "SUB", "MUL", "DIV", "MOD",
    "EQ", "NE", "LT", "GT", "LE", "GE",
    "AND", "OR", "XOR",
    "SHL", "SHR",
    "LI", "LC", "SI", "SC", "PSH",
    "LEA", "IMM", "JMP", "JSR", "ENT", "ADJ", "LEV",
    "BZ", "BNZ",
)

_BINOP_ID: Dict[str, int] = {
    "OR": OP_OR, "XOR": OP_XOR, "AND": OP_AND,
    "EQ": OP_EQ, "NE": OP_NE, "LT": OP_LT, "GT": OP_GT, "LE": OP_LE, "GE": OP_GE,
    "SHL": OP_SHL, "SHR": OP_SHR,
    "ADD": OP_ADD, "SUB": OP_SUB, "MUL": OP_MUL, "DIV": OP_DIV, "MOD": OP_MOD,
}


def encode(op: int, imm: int = 0) -> int:
    """Encode one instruction: ``opcode | (imm << 8)`` (the c4 encoding)."""
    return int(op) | (int(imm) << 8)


# ---------------------------------------------------------------------------
# Program + Decoded value types (the harness/model contract — see DESIGN.md).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Program:
    """A green-field ISA program: encoded bytecode + a data segment."""

    op: str                       #: op-class this program exercises
    label: str                    #: human-readable case label
    bytecode: Tuple[int, ...]
    data: bytes = b""
    note: str = ""


@dataclass(frozen=True)
class Expected:
    """The ground-truth ISA behaviour of a ``Program`` (reference VM)."""

    exit_code: Optional[int]      #: AX (mod 2**32) at EXIT, or None if no halt
    steps: Optional[int]
    halted: bool
    trace: Tuple[Tuple[int, int], ...] = ()   #: per-step (pc_after, ax_after)
    error: Optional[str] = None


@dataclass(frozen=True)
class Decoded:
    """A model-under-test's autoregressive decode of a ``Program``.

    The green-field ``c4_min.model.run`` returns one of these (see DESIGN.md).
    ``trace`` is OPTIONAL: when present the harness does a per-step full-trace
    comparison; when empty it compares exit_code + steps only.
    """

    exit_code: Optional[int]
    steps: Optional[int] = None
    halted: bool = True
    trace: Tuple[Tuple[int, int], ...] = ()


# ---------------------------------------------------------------------------
# Program generator: a few 8-bit operand cases per opcode.
# ---------------------------------------------------------------------------


def _binop_prog(op: str, a: int, b: int) -> Program:
    """``IMM a; PSH; IMM b; <op>; EXIT`` -> exit code = (a <op> b).

    The exact shape the c4 compiler emits for ``return a <op> b``, so the model
    decodes it through the same per-step path as a real arith program.
    """
    opcode = _BINOP_ID[op]
    bc = (
        encode(OP_IMM, a), encode(OP_PSH), encode(OP_IMM, b),
        encode(opcode), encode(OP_EXIT),
    )
    return Program(op=op, label=f"{op.lower()}_{a}_{b}", bytecode=bc)


# Per-binop 8-bit operand cases. Chosen so the expected exit code is nonzero
# where the semantics allow (a nonzero spec value is a stronger decode check)
# and to hit boundaries (0, identity, overflow, div-by-zero-adjacent).
_BINOP_CASES: Dict[str, Tuple[Tuple[int, int], ...]] = {
    "ADD": ((3, 4), (100, 27), (200, 55), (255, 1)),
    "SUB": ((9, 4), (200, 55), (7, 9), (0, 1)),
    "MUL": ((6, 7), (12, 12), (15, 17), (0, 200)),
    "DIV": ((84, 7), (100, 3), (9, 4), (7, 9)),
    "MOD": ((84, 5), (100, 7), (9, 4), (7, 9)),
    "AND": ((0x6C, 0x3A), (0xFF, 0x0F), (0xF0, 0x0F)),
    "OR":  ((0x6C, 0x3A), (16, 1), (0xF0, 0x0F)),
    "XOR": ((0x6C, 0x3A), (0xFF, 0xFF), (0xAA, 0x55)),
    "SHL": ((5, 3), (1, 7), (3, 0)),
    "SHR": ((200, 2), (0xFF, 4), (7, 0)),
    "EQ":  ((5, 5), (7, 9)),
    "NE":  ((7, 9), (5, 5)),
    "LT":  ((7, 9), (9, 7), (5, 5)),
    "GT":  ((9, 7), (7, 9), (5, 5)),
    "LE":  ((7, 9), (9, 7), (5, 5)),
    "GE":  ((9, 7), (7, 9), (5, 5)),
}


def _programs_for(op: str) -> List[Program]:
    """Build the representative program(s) for one op-class."""
    if op in _BINOP_CASES:
        return [_binop_prog(op, a, b) for a, b in _BINOP_CASES[op]]

    if op == "IMM":
        # IMM v; EXIT -> exit code = v. The most direct AX-load check.
        return [
            Program("IMM", f"imm_{v}",
                    (encode(OP_IMM, v), encode(OP_EXIT)))
            for v in (0, 42, 200)
        ]

    if op == "PSH":
        # PSH pushes AX; verify the pushed value survives a round-trip through
        # an ADD with 0: IMM v; PSH; IMM 0; ADD; EXIT -> v.
        return [
            Program("PSH", f"psh_{v}",
                    (encode(OP_IMM, v), encode(OP_PSH), encode(OP_IMM, 0),
                     encode(OP_ADD), encode(OP_EXIT)))
            for v in (42, 200)
        ]

    if op == "BZ":
        return [
            # AX==0 -> branch to idx 3 (skip IMM 99), exit 7.
            Program("BZ", "bz_taken",
                    (encode(OP_IMM, 0), encode(OP_BZ, 3), encode(OP_IMM, 99),
                     encode(OP_IMM, 7), encode(OP_EXIT))),
            # AX!=0 -> fall through, exit 42.
            Program("BZ", "bz_nottaken",
                    (encode(OP_IMM, 1), encode(OP_BZ, 3), encode(OP_IMM, 42),
                     encode(OP_EXIT))),
        ]

    if op == "BNZ":
        return [
            # AX!=0 -> branch to idx 3 (skip IMM 99), exit 7.
            Program("BNZ", "bnz_taken",
                    (encode(OP_IMM, 1), encode(OP_BNZ, 3), encode(OP_IMM, 99),
                     encode(OP_IMM, 7), encode(OP_EXIT))),
            # AX==0 -> fall through, exit 42.
            Program("BNZ", "bnz_nottaken",
                    (encode(OP_IMM, 0), encode(OP_BNZ, 3), encode(OP_IMM, 42),
                     encode(OP_EXIT))),
        ]

    if op == "JMP":
        return [
            # JMP over IMM 99 to IMM 5; EXIT -> 5.
            Program("JMP", "jmp_fwd",
                    (encode(OP_JMP, 2), encode(OP_IMM, 99), encode(OP_IMM, 5),
                     encode(OP_EXIT))),
        ]

    if op == "LI":
        # Load a 32-bit word from data[0..3]. data little-endian = 0x0000005A=90.
        return [
            Program("LI", "li_data0",
                    (encode(OP_IMM, DATA_BASE), encode(OP_LI), encode(OP_EXIT)),
                    data=bytes([0x5A, 0x00, 0x00, 0x00])),
        ]

    if op == "LC":
        # Load a single char (low byte) from data[0]=0x5A=90.
        return [
            Program("LC", "lc_data0",
                    (encode(OP_IMM, DATA_BASE), encode(OP_LC), encode(OP_EXIT)),
                    data=bytes([0x5A, 0x00, 0x00, 0x00])),
        ]

    if op == "SI":
        # Store a word to data, then load it back: exit = stored value.
        return [
            Program("SI", "si_then_li",
                    (encode(OP_IMM, DATA_BASE), encode(OP_PSH),   # push addr
                     encode(OP_IMM, 0x23), encode(OP_SI),          # *[addr]=0x23
                     encode(OP_IMM, DATA_BASE), encode(OP_LI),     # AX = *[addr]
                     encode(OP_EXIT)),
                    data=bytes([0, 0, 0, 0])),
        ]

    if op == "SC":
        # Store a char then load it back: exit = 0x41.
        return [
            Program("SC", "sc_then_lc",
                    (encode(OP_IMM, DATA_BASE), encode(OP_PSH),
                     encode(OP_IMM, 0x41), encode(OP_SC),
                     encode(OP_IMM, DATA_BASE), encode(OP_LC),
                     encode(OP_EXIT)),
                    data=bytes([0, 0, 0, 0])),
        ]

    if op == "LEA":
        # LEA loads BP+imm. After ENT, BP=SP. Frame-relative address is a big
        # pointer; to get a deterministic small exit we LEA then subtract BP by
        # loading it another way is awkward, so exercise LEA's effect indirectly:
        # ENT 8; LEA -8 gives a slot address; store 0x37 there via SI; LI it back.
        return [
            Program("LEA", "lea_local",
                    (encode(OP_ENT, 8),          # frame: reserve 1 local
                     encode(OP_LEA, -8),         # AX = &local0
                     encode(OP_PSH),             # push &local0
                     encode(OP_IMM, 0x37),       # AX = 0x37
                     encode(OP_SI),              # local0 = 0x37
                     encode(OP_LEA, -8),         # AX = &local0
                     encode(OP_LI),              # AX = local0
                     encode(OP_EXIT))),
        ]

    if op == "ENT":
        # ENT reserves frame; ADJ/LEV tear down. Store into a local and read it
        # back to prove the frame is consistent. exit = 0x2A.
        return [
            Program("ENT", "ent_local",
                    (encode(OP_ENT, 8),
                     encode(OP_LEA, -8), encode(OP_PSH),
                     encode(OP_IMM, 0x2A), encode(OP_SI),
                     encode(OP_LEA, -8), encode(OP_LI),
                     encode(OP_EXIT))),
        ]

    if op == "ADJ":
        # ADJ adjusts SP. Push a value, ADJ +8 to discard it, then exit with a
        # fresh IMM so SP hygiene is exercised without leaking. exit = 0x11.
        return [
            Program("ADJ", "adj_discard",
                    (encode(OP_IMM, 0x99), encode(OP_PSH),
                     encode(OP_ADJ, 8),                # discard the pushed word
                     encode(OP_IMM, 0x11), encode(OP_EXIT))),
        ]

    if op in ("JSR", "LEV"):
        # A tiny call: main JSRs a leaf that returns 0x2A, then EXITs with it.
        #   idx0: JSR ->4 (leaf)   idx1: (ret here) EXIT
        # Wait: JSR pushes return PC then branches. We lay out:
        #   [0] IMM 0            (clear AX)
        #   [1] JSR 3            call leaf at idx 3
        #   [2] EXIT             exit with AX from leaf
        #   [3] IMM 0x2A         leaf: AX = 0x2A
        #   [4] LEV              return to idx 2
        prog = (
            encode(OP_IMM, 0), encode(OP_JSR, 3), encode(OP_EXIT),
            encode(OP_IMM, 0x2A), encode(OP_LEV),
        )
        return [Program(op, f"{op.lower()}_call", prog)]

    raise KeyError(f"no program generator for op-class {op!r}")


def programs_by_op() -> Dict[str, List[Program]]:
    """Return the full op-class -> representative programs map."""
    return {op: _programs_for(op) for op in ALL_OP_CLASSES}


def all_programs() -> List[Program]:
    """Flat list of every generated program (across all op-classes)."""
    out: List[Program] = []
    for op in ALL_OP_CLASSES:
        out.extend(_programs_for(op))
    return out


# ---------------------------------------------------------------------------
# Expected-output computer: reuse the reference ISA semantics (EXPECTED side).
# ---------------------------------------------------------------------------

_RUNNER = None


def _reference_runner():
    """Lazily build the reference ISA VM (expected side only)."""
    global _RUNNER
    if _RUNNER is None:
        from neural_vm.verification.symbolic_program import (
            SymbolicDeclarativeProgramRunner,
        )

        _RUNNER = SymbolicDeclarativeProgramRunner()
    return _RUNNER


def expected_for_program(prog: Program, *, max_steps: int = 64) -> Expected:
    """Compute the ground-truth ISA behaviour of ``prog``.

    Uses ``SymbolicDeclarativeProgramRunner`` — the SAME reference semantics the
    reference model's per-op oracle (``tests/oracles/per_op_decode.py``) trusts.
    Returns exit code + step count + per-step (pc_after, ax_after) trace.
    """
    try:
        runner = _reference_runner()
        state = runner.run(list(prog.bytecode), prog.data, max_steps=max_steps)
    except Exception as exc:  # noqa: BLE001
        return Expected(exit_code=None, steps=None, halted=False,
                        error=f"{prog.label}: reference VM error: {exc!r}")

    if not state.halted:
        return Expected(exit_code=None, steps=None, halted=False,
                        error=f"{prog.label}: did not halt in {max_steps} steps")

    trace = tuple((t.pc_after, t.ax_after) for t in state.trace)
    return Expected(
        exit_code=int(state.ax) & 0xFFFFFFFF,
        steps=int(state.steps),
        halted=True,
        trace=trace,
    )


def expected_by_op(*, max_steps: int = 64) -> Dict[str, List[Tuple[Program, Expected]]]:
    """op-class -> [(program, expected), ...] for every generated program."""
    out: Dict[str, List[Tuple[Program, Expected]]] = {}
    for op, progs in programs_by_op().items():
        out[op] = [(p, expected_for_program(p, max_steps=max_steps)) for p in progs]
    return out


# ---------------------------------------------------------------------------
# Adapters for the model-under-test's decoded output.
# ---------------------------------------------------------------------------


def decoded_from_bytes(
    output_bytes: Sequence[int],
    *,
    steps: Optional[int] = None,
    halted: bool = True,
    trace: Sequence[Tuple[int, int]] = (),
) -> Decoded:
    """Wrap a model's flat decoded byte stream into a ``Decoded``.

    The green-field ``model.run`` is expected to return a ``Decoded`` directly,
    but a model that emits a flat little-endian byte stream for the exit code
    can be adapted here (so the harness plugs into either shape).
    """
    exit_code: Optional[int]
    if not output_bytes:
        exit_code = None
    else:
        exit_code = 0
        for i, b in enumerate(output_bytes[:4]):
            exit_code |= (int(b) & 0xFF) << (8 * i)
    return Decoded(exit_code=exit_code, steps=steps, halted=halted,
                   trace=tuple(trace))


# ---------------------------------------------------------------------------
# The verdict: compare a model's Decoded against the Expected.
# ---------------------------------------------------------------------------


@dataclass
class ProgramVerdict:
    op: str
    label: str
    expected_exit: Optional[int]
    got_exit: Optional[int]
    expected_steps: Optional[int]
    got_steps: Optional[int]
    status: str                     # "pass" | "fail" | "oracle_error" | "model_error"
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "pass"


@dataclass
class OpClassVerdict:
    op: str
    programs: List[ProgramVerdict] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.programs) and all(p.ok for p in self.programs)

    @property
    def n_pass(self) -> int:
        return sum(1 for p in self.programs if p.ok)


def compare(prog: Program, expected: Expected, decoded: Decoded) -> ProgramVerdict:
    """Compare one model decode against the reference expected output.

    PASS iff the exit code matches AND (when both sides carry a per-step trace)
    every step's (pc, ax) matches. If the reference itself errored, the verdict
    is ``oracle_error`` (not the model's fault); if the model failed to produce
    an exit code, ``model_error``.
    """
    base = dict(op=prog.op, label=prog.label,
                expected_exit=expected.exit_code, got_exit=decoded.exit_code,
                expected_steps=expected.steps, got_steps=decoded.steps)

    if expected.error is not None or expected.exit_code is None:
        return ProgramVerdict(status="oracle_error",
                              detail=expected.error or "no expected exit", **base)

    if decoded.exit_code is None:
        return ProgramVerdict(status="model_error",
                              detail="model produced no exit code", **base)

    if int(decoded.exit_code) != int(expected.exit_code):
        return ProgramVerdict(
            status="fail",
            detail=(f"exit mismatch: expected {expected.exit_code} "
                    f"got {decoded.exit_code}"),
            **base,
        )

    # Full-trace comparison when the model supplies a trace.
    if decoded.trace:
        exp_tr, got_tr = expected.trace, tuple(decoded.trace)
        n = min(len(exp_tr), len(got_tr))
        for i in range(n):
            if tuple(exp_tr[i]) != tuple(got_tr[i]):
                return ProgramVerdict(
                    status="fail",
                    detail=(f"trace diverged @step {i}: "
                            f"expected (pc,ax)={exp_tr[i]} got={got_tr[i]}"),
                    **base,
                )
        if len(got_tr) != len(exp_tr):
            return ProgramVerdict(
                status="fail",
                detail=(f"trace length mismatch: expected {len(exp_tr)} "
                        f"steps got {len(got_tr)}"),
                **base,
            )

    # Step count, when the model reports it, must agree too.
    if decoded.steps is not None and expected.steps is not None \
            and int(decoded.steps) != int(expected.steps):
        return ProgramVerdict(
            status="fail",
            detail=(f"step count mismatch: expected {expected.steps} "
                    f"got {decoded.steps}"),
            **base,
        )

    return ProgramVerdict(status="pass", **base)
