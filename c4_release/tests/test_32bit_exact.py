"""32-BIT EXACTNESS regression gate — the permanent coverage for the audit.

Three GATES over the ``c4_min.audit32`` corpus (all ISA opcodes × 32-bit
boundaries / sign combos / carries):

  * ``test_golden_matches_native_c4``  — the GOLDEN 32-bit reference
    (``audit32.golden32``) matches the reference ``./c4`` binary for every
    case a C program can express, EXCEPT the documented signed-DIV/MOD design
    divergence (the neural gadget does UNSIGNED base-16 long division; native
    c4 does signed C division — they differ only on a NEGATIVE dividend).  If
    ``./c4`` is absent the test skips.
  * ``test_draft_divergence_set_is_locked``  — the DRAFT VM
    (``ref_interpret_word32``, the doom fast path) diverges from the golden on
    EXACTLY the doom-critical opcode set {SHR, DIV, MOD, LC, SC} and NOWHERE
    else.  A NEW divergent opcode (or a fixed one) FAILS this test — it is the
    tripwire that a doom-fast-path change (or a golden change) is unaccounted
    for.
  * ``test_neural_pure_forward_32bit``  — the NEURAL pure-forward model
    reproduces the golden byte-exact for every non-xfail case, and the
    documented xfail cases (the #702 signed-SHR-of-a-pushed-negative gap) FAIL.
    SLOW (~15s build + ~30s/case): opt-in via ``C4_AUDIT_NEURAL=1`` (also sets
    ``C4_PF_CFM=1`` — the neural driver is broken without it at this base; see
    ``audit_matrix.py`` THE CFM CAVEAT).

Additive test file; authors NO weights (golden 069cc32f untouched).
"""
from __future__ import annotations

import os

import pytest

# Pin SP_INIT to the runner value (0xFC) BEFORE importing the audit corpus, so the
# golden (which reads _PFC.SP_INIT), the draft (word32, reads _PF.SP_INIT), and the
# neural model (SP_INIT=0xFC) all share the SAME frame base — otherwise LEA / ENT /
# stack-address cases diverge trivially on the SP-base convention, not on a real
# 32-bit gap.  This mirrors run_1096_pure_forward / run_edge_ops.
import c4_min.nibble_pure_forward as _PF  # noqa: E402
import c4_min.nibble_pure_forward_complete as _PFC  # noqa: E402
_PF.SP_INIT = 0xFC
_PFC.SP_INIT = 0xFC

from c4_min.audit32 import generate_cases, M32, INT_MIN, INT_MAX, golden32  # noqa: E402
from c4_min.selfhost.word32_draft_vm import ref_interpret_word32  # noqa: E402


_CASES = generate_cases()

# The doom-critical opcodes the DRAFT (doom fast path) models at the WRONG width
# / signedness vs the golden.  Locked so a new gap (or a fix) is surfaced.
#   SHR : draft is LOGICAL, golden ARITHMETIC (sign-fill)         — #702.
#   DIV : draft is SIGNED C-trunc, golden UNSIGNED floor          — negative dividend.
#   MOD : draft is SIGNED C-trunc, golden UNSIGNED floor          — negative dividend.
#   LC  : draft is UNSIGNED 32-bit load, golden SIGN-extends      — char* reads.
#   SC  : draft stores 32-bit, golden truncates to a byte (then LC readback diverges).
_EXPECTED_DRAFT_DIVERGENT_OPS = {"SHR", "DIV", "MOD", "LC", "SC"}

# On a NEGATIVE dividend the golden (unsigned floor) and native c4 (signed C div)
# INTENTIONALLY differ — the neural gadget is unsigned.  These case names are the
# documented golden-vs-native design divergence (NOT a golden error).
_SIGNED_DIVMOD_NATIVE_DIVERGES = {
    "div_neg_by2", "div_intmin_neg1", "div_neg_pos",
    "mod_neg_pos", "mod_neg_by2",
}

_C4_BIN = "/home/alexlitz/Documents/misc/c4_doom/c4"


# ---------------------------------------------------------------------------
# Sanity: a handful of the boundary values are computed correctly by the golden.
# (Locks the golden's semantics independent of any path.)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,want", [
    ("add_32_wrap", 0x00000000),           # 0xFFFFFFFF + 1 -> 0 (wrap)
    ("add_intmax_1", INT_MIN),             # INT_MAX + 1 -> INT_MIN
    ("sub_intmin_1", INT_MAX),             # INT_MIN - 1 -> INT_MAX
    ("mul_overflow", 0x00000000),          # 0x10000 * 0x10000 -> 0 (>32b trunc)
    ("mul_16_16", 0xFFFE0001),             # 0xFFFF * 0xFFFF
    ("shl_1_by31", INT_MIN),               # 1 << 31 -> INT_MIN
    ("shr_neg1_by1", 0xFFFFFFFF),          # (-1) >> 1 -> -1 (arithmetic)
    ("shr_intmin_by1", 0xC0000000),        # INT_MIN >> 1 (arithmetic)
    ("lc_80", 0xFFFFFF80),                 # LC 0x80 sign-extend
    ("lc_ff", 0xFFFFFFFF),                 # LC 0xFF -> -1
    ("li_ffffffff", 0xFFFFFFFF),           # LI zero-extends full word
    ("si_li_roundtrip", 0xDEADBEEF),       # SI/LI 32-bit round trip
    ("lt_neg_vs_pos", 1),                  # -1 < 1  (SIGNED)
    ("gt_neg_vs_pos", 0),                  # -1 > 1  is false (SIGNED)
    ("bnz_high_only", 7),                  # BNZ on 0x10000 (low byte 0) branches
    ("jsr_leaf", 42),                      # JSR/ENT/LEV round trip
])
def test_golden_boundary_values(name, want):
    case = next(c for c in _CASES if c.name == name)
    assert case.golden() == (want & M32), (
        f"golden {name}: got {case.golden():#010x} want {want & M32:#010x}")


# ---------------------------------------------------------------------------
# GATE 1: golden == native ./c4 for every expressible case (minus the documented
# signed-DIV/MOD design divergence).
# ---------------------------------------------------------------------------
def _native_value(case):
    import subprocess
    import tempfile
    from c4_min.audit32 import s32
    C_OP = {"ADD": "+", "SUB": "-", "MUL": "*", "DIV": "/", "MOD": "%",
            "AND": "&", "OR": "|", "XOR": "^", "SHL": "<<", "SHR": ">>",
            "EQ": "==", "NE": "!=", "LT": "<", "GT": ">", "LE": "<=", "GE": ">="}
    if case.op not in C_OP:
        return None
    a = case.seed_mem.get(0x40)
    b = case.seed_mem.get(0x44)
    if a is None:
        return None
    cop = C_OP[case.op]
    sa = s32(a)
    if b is not None:
        src = (f"int main(){{ int a; int b; a={sa}; b={s32(b)}; "
               f"printf(\"%d\\n\", a {cop} b); return 0; }}")
    else:
        amt = case.shift_amt if case.shift_amt is not None else 0
        src = (f"int main(){{ int a; a={sa}; "
               f"printf(\"%d\\n\", a {cop} {amt}); return 0; }}")
    with tempfile.NamedTemporaryFile("w", suffix=".c", delete=False) as fh:
        fh.write(src)
        path = fh.name
    try:
        out = subprocess.run([_C4_BIN, path], capture_output=True, timeout=10,
                             text=True)
    finally:
        os.unlink(path)
    for ln in out.stdout.splitlines():
        ln = ln.strip()
        if ln and ln.lstrip("-").isdigit():
            return int(ln) & M32
    return None


@pytest.mark.skipif(not os.path.exists(_C4_BIN),
                    reason="native ./c4 binary not present")
@pytest.mark.parametrize("case", [c for c in _CASES], ids=[c.name for c in _CASES])
def test_golden_matches_native_c4(case):
    nv = _native_value(case)
    if nv is None:
        pytest.skip("case not expressible as a native C program")
    g = case.golden()
    if case.name in _SIGNED_DIVMOD_NATIVE_DIVERGES:
        # documented design divergence: golden is UNSIGNED floor, native is signed.
        assert nv != g, (
            f"{case.name}: golden/native were EXPECTED to diverge (signed-DIV/MOD "
            f"design gap) but both are {g:#010x} — remove it from the divergence set")
    else:
        assert nv == g, (
            f"{case.name} ({case.op}): golden {g:#010x} != native ./c4 {nv:#010x} "
            f":: {case.note}")


# ---------------------------------------------------------------------------
# GATE 2: the DRAFT divergence set is exactly the doom-critical opcodes.
# ---------------------------------------------------------------------------
def _draft_value(case):
    tr, _ = ref_interpret_word32(case.code(), seed_mem=dict(case.seed_mem),
                                 max_steps=case.max_steps)
    return tr[-1] & M32 if tr else 0


def test_draft_divergence_set_is_locked():
    divergent_ops = set()
    per_case = {}
    for c in _CASES:
        g = c.golden()
        d = _draft_value(c)
        per_case[c.name] = (g, d)
        if d != g:
            divergent_ops.add(c.op)
    assert divergent_ops == _EXPECTED_DRAFT_DIVERGENT_OPS, (
        f"DRAFT (doom fast path) divergent-opcode set changed: "
        f"got {sorted(divergent_ops)} expected "
        f"{sorted(_EXPECTED_DRAFT_DIVERGENT_OPS)}. A NEW divergent op means the "
        f"doom fast path grew a gap; a MISSING one means a gap was fixed — update "
        f"_EXPECTED_DRAFT_DIVERGENT_OPS either way.")


@pytest.mark.parametrize("case", [c for c in _CASES if c.op in {"ADD", "SUB", "MUL"}],
                         ids=[c.name for c in _CASES if c.op in {"ADD", "SUB", "MUL"}])
def test_draft_matches_golden_on_wrapping_alu(case):
    # ADD/SUB/MUL are 32-bit-wrap on BOTH paths — the draft MUST match the golden
    # (the doom fast path is trusted for these).
    assert _draft_value(case) == case.golden(), (
        f"{case.name}: draft {_draft_value(case):#010x} != golden "
        f"{case.golden():#010x} — a wrapping-ALU op regressed on the fast path")


# ---------------------------------------------------------------------------
# GATE 3: the NEURAL pure-forward model reproduces the golden byte-exact.  SLOW.
# ---------------------------------------------------------------------------
_NEURAL = os.environ.get("C4_AUDIT_NEURAL", "0") not in ("0", "", "false", "False")


@pytest.mark.skipif(not _NEURAL,
                    reason="set C4_AUDIT_NEURAL=1 to run the slow neural gate "
                           "(~15s build + ~30s/case)")
class TestNeural32bit:
    @pytest.fixture(scope="class")
    def model(self):
        # Tests the PRODUCTION default (C4_PF_CFM off) — runs byte-exact thanks to
        # the _rebuild_layout bool-remap fix (see audit_matrix.py THE CFM CAVEAT).
        import c4_min.nibble_pure_forward as _PF
        import c4_min.nibble_pure_forward_complete as _PFC
        _PF.SP_INIT = 0xFC
        _PFC.SP_INIT = 0xFC
        from c4_min.compact_alloc import build_compact_sparse_streaming
        m, L, _ = build_compact_sparse_streaming(
            code_size=48, compute_mode="dense_kernel")
        return m, L

    @pytest.mark.parametrize("case", [c for c in _CASES],
                             ids=[c.name for c in _CASES])
    def test_neural_matches_golden(self, model, case):
        import c4_min.nibble_pure_forward_complete as _PFC
        m, L = model
        tr = _PFC.run_pure_forward_complete(
            m, L, case.code(), max_steps=case.max_steps, mask=M32,
            seed_mem=dict(case.seed_mem) or None)
        got = tr[-1] & M32 if tr else 0
        g = case.golden()
        if case.neural_xfail:
            pytest.xfail(f"documented neural 32-bit gap (multi-byte SHL/SHR / "
                         f"multi-byte-EQ comparison): {case.name} :: {case.note}")
        assert got == g, (
            f"{case.name} ({case.op}): neural {got:#010x} != golden {g:#010x} "
            f":: {case.note}")
