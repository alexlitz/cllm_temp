"""Edge-case ISA suite — the 17 opcodes the canonical 1096 corpus never touches.

ADDITIVE (test/tooling only).  Authors NO weights (imports the existing reference
interpreters + the already-built pure-forward model), so the golden c4_min
fingerprint ``069cc32f`` is untouched by this file.

WHY.  ``tests.test_suite_1000.generate_test_programs()`` is exactly 1096 programs
but exercises ONLY 23/40 opcodes.  This suite covers the never-exercised char-typed
+ I/O + bitwise + branch opcodes — the doom-critical path (``char*`` buffers,
fixed-point ``>>``, signed div).

Two gates (see ``c4_min.run_edge_ops`` for the scoreboard):
  * GOLDEN (fast, always run here) — every case's ``expected`` == the c4_min
    reference (``ref_interpret``, 32-bit).  This is the same golden the
    pure-forward 1096 corpus scores against.
  * NEURAL (slow, opt-in) — a bounded per-cluster sample run through the
    pure-forward model (``run_pure_forward_complete``) and asserted byte-exact vs
    golden.  ~15-60s/program, so it is marked ``slow``/``neural`` and defaults
    OFF.  ``tests/conftest.py`` DESELECTS ``slow`` tests unless ``--runslow``, so
    the neural gate runs with:
        C4_EDGE_NEURAL=1 pytest tests/test_suite_edge_ops.py --runslow -k neural
    (``C4_EDGE_NEURAL_PER_CLUSTER=N`` widens the sample.)  The standalone
    scoreboard ``python c4_min/run_edge_ops.py`` gives the full per-cluster table.

The KNOWN NEURAL divergence (signed-char SHR of a PUSHED 32-bit negative — the
1-slot STACK0 multi-byte-relay wall) is captured as an ``xfail`` so the suite
stays green while documenting the real localized bug.
"""
from __future__ import annotations

import os

import pytest

from c4_min import isa
from c4_min.edge_corpus import generate_edge_cases, opcodes_covered
from c4_min.run_edge_ops import golden_value, neural_value


_CASES = generate_edge_cases()
_M32 = 0xFFFFFFFF

# Neural cases KNOWN to diverge (documented findings, not regressions), confirmed
# empirically through the pure-forward model (2026-07-30).  ROOT: the SHR
# arithmetic sign-fill reads the operand from the 1-slot STACK0 relay, which
# carries only the LOW BYTE of a PUSHED value — so a NEGATIVE operand (from an
# earlier signed LC) looks positive and the shift produces the correct low byte
# with NO sign-extension (the #702 1-slot STACK0 multi-byte-relay wall).  Confirmed
# examples: (char -1)>>1 -> golden -1 / neural 255; (char -128)>>1 -> golden -64 /
# neural 192 (== 0xC0, the correct low byte, un-extended).  POSITIVE / UNSIGNED SHR
# and SHL are byte-exact (shr_char_pos_by3, shr_uchar_ff_by1, shl_* all PASS).
_KNOWN_NEURAL_XFAIL = {
    "shr_char_neg1_by1", "shr_char_neg1_by4", "shr_char_neg1_by7",
    "shr_char_neg1_by8", "shr_char_neg1_by31",
    "shr_char_neg128_by1", "shr_char_neg128_by7",
}

# Compiled runtime-library programs (malloc/memset/memcmp + compiled while-loops)
# lower to ~250-300 instructions (the linked stdlib), which EXCEEDS the neural
# model's fixed code band (code_size=48) -> the overlay IndexErrors.  These are
# verified vs the c4_min GOLDEN only; the neural code band would need code_size
# >= ~300 (a much larger / slower build) to run them.  NOT a neural semantic bug.
_NEURAL_CODE_BAND_SKIP = {
    "c_string_copy_scan", "char_fill_loop", "memcmp_equal",
    "malloc_write_read_free",
}


# ===========================================================================
# Coverage claim.
# ===========================================================================
def test_canonical_corpus_covers_only_23_opcodes():
    """The premise: the 1096 corpus exercises ONLY 23/40 opcodes (LC/SC/SHR/…
    never appear)."""
    from src.compiler import compile_c
    from tests.test_suite_1000 import generate_test_programs
    from c4_min.run_1096_pure_forward import bytecode_to_isa
    seen = set()
    for src, _exp, _desc in generate_test_programs():
        bc, _ = compile_c(src)
        for ins in bytecode_to_isa(bc):
            seen.add(isa.NAMES.get(ins.op, ins.op))
    # exactly the 23 the task documents (incl HALT/NOP).
    assert len(seen) == 23, sorted(seen)
    for never in ("LC", "SC", "SHR", "SHL", "AND", "OR", "XOR", "GE", "BNZ",
                  "PRTF", "READ"):
        assert never not in seen, f"{never} unexpectedly in the 1096 corpus"


def test_edge_suite_adds_opcode_coverage():
    """The edge suite exercises the previously-untested opcodes (23 -> >=34)."""
    covered = opcodes_covered(_CASES)
    for op in ("LC", "SC", "SHR", "SHL", "AND", "OR", "XOR", "GE", "BNZ",
               "PRTF", "READ"):
        assert op in covered, f"edge suite missing {op}"


# ===========================================================================
# GOLDEN gate — every case's expected == the c4_min reference (fast).
# ===========================================================================
@pytest.mark.parametrize("case", _CASES, ids=[c.name for c in _CASES])
def test_golden_byte_exact(case):
    """``expected`` is the byte-exact c4_min reference value (and stdout for PRTF)."""
    gax, gout = golden_value(case)
    assert gax == (case.expected & _M32), (
        f"{case.name}: golden AX {gax} != expected {case.expected & _M32} "
        f"({case.note})")
    if case.expected_stdout is not None and gout is not None:
        assert gout == case.expected_stdout, (
            f"{case.name}: golden stdout {gout!r} != {case.expected_stdout!r}")


# ===========================================================================
# NEURAL gate — bounded per-cluster sample through the pure-forward model.
# Opt-in: `-m neural` (or C4_EDGE_NEURAL=1).  ~15-60s/program.
# ===========================================================================
def _neural_sample(per_cluster: int = 1):
    """A per-cluster sample of cases that FIT the neural code band (skip the
    compiled ~250-300-instruction runtime programs that IndexError the overlay)."""
    from collections import Counter
    seen: Counter = Counter()
    out = []
    for c in _CASES:
        if c.name in _NEURAL_CODE_BAND_SKIP:
            continue
        if seen[c.cluster] < per_cluster:
            seen[c.cluster] += 1
            out.append(c)
    return out


_NEURAL_ON = os.environ.get("C4_EDGE_NEURAL") == "1"
_NEURAL_SAMPLE = _neural_sample(int(os.environ.get("C4_EDGE_NEURAL_PER_CLUSTER", "1")))


@pytest.mark.slow
@pytest.mark.neural
@pytest.mark.skipif(not _NEURAL_ON,
                    reason="neural pure-forward is ~15-60s/prog; set C4_EDGE_NEURAL=1")
@pytest.mark.parametrize("case", _NEURAL_SAMPLE, ids=[c.name for c in _NEURAL_SAMPLE])
def test_neural_byte_exact(case, edge_model):
    """The pure-forward NEURAL model reproduces the golden byte-exactly.

    Known-divergent cases (documented findings) are xfail'd, not skipped, so a
    surprise PASS surfaces as XPASS."""
    if case.name in _NEURAL_CODE_BAND_SKIP:
        pytest.skip("compiled program exceeds the neural code band (code_size=48); "
                    "golden-verified only")
    model, L = edge_model
    nax, nout = neural_value(case, model, L)
    ok = (nax == (case.expected & _M32))
    if case.expected_stdout is not None and nout is not None:
        ok = ok and (nout == case.expected_stdout)
    if case.name in _KNOWN_NEURAL_XFAIL and not ok:
        pytest.xfail(f"known neural divergence: {case.name} "
                     f"golden={case.expected & _M32} neural={nax} ({case.note})")
    assert ok, (f"{case.name}: neural {nax}/{nout!r} != golden "
                f"{case.expected & _M32}/{case.expected_stdout!r} ({case.note})")


@pytest.fixture(scope="module")
def edge_model():
    """Memory-safe streaming pure-forward model (peak ~5 GB), pinned SP_INIT=0xFC."""
    import c4_min.nibble_pure_forward as _PF
    import c4_min.nibble_pure_forward_complete as _PFC
    _PF.SP_INIT = 0xFC
    _PFC.SP_INIT = 0xFC
    from c4_min._build_guard import guarded_complete_build
    model, L = guarded_complete_build(code_size=48)
    return model, L
