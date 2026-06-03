"""Step-1 safety: ``CrossStepReadWarning`` at compile time.

These tests pin the behaviour of
``_find_cross_step_reads_with_same_step_writers`` /
``_emit_cross_step_safety_warnings`` — the compile-time check that warns
when an op declares a cross-step (``X.*.-1``) read of a residual dim
whose base name ALSO has a same-step writer in the scheduled op set.

On VM step 1 (the first step) there is no previous step, so the
cross-step SSA alias resolves to 0. When the same dim has a same-step
writer, the bake author very often *meant* to consume that fresh write
(or to OR same-step ∪ prev-step), but the IR's bare ``.*.-1`` form
silently falls through to 0 on step 1. The motivating discovery is the
``OPCODE_BYTE_LO.*.-1`` read in ``opcode_decode_ffn`` whose same-step
writer is ``layer5_fetch``: on step 1 the decoder reads 0.

Tests cover:

1. **The motivating case fires.** A finding for
   ``opcode_decode_ffn`` reading ``OPCODE_BYTE_LO.*.-1`` with
   ``layer5_fetch`` as the same-step writer must appear in the analysis.
2. **The warning category is correct.** ``warnings.catch_warnings`` /
   ``warnings.simplefilter("always", CrossStepReadWarning)`` must capture
   warnings of the dedicated category when
   ``_emit_cross_step_safety_warnings`` runs.
3. **Self-writers are excluded.** An op whose own ``writes`` include the
   base dim of its own cross-step read must NOT trigger a warning (that
   is the canonical back-edge pattern, not a step-1 bug).
4. **No-finding case stays silent.** A clean synthetic op set with no
   ``.*.-1`` reads emits no warnings and returns no findings.
5. **Production op set finding count is non-trivial.** Today's
   declarations produce a non-zero baseline of findings — the check is
   actually examining something.
"""

import warnings

from c4_release.neural_vm.unified_compiler.full_vm_compiler_dynamic import (
    CrossStepReadWarning,
    _collect_ops_for_compile,
    _emit_cross_step_safety_warnings,
    _find_cross_step_reads_with_same_step_writers,
)
from c4_release.neural_vm.unified_compiler.layer_compiler import Operation


def _noop_bake(*_args, **_kwargs):
    return None


# ---------------------------------------------------------------------------
# 1. The motivating case: OPCODE_BYTE_LO.*.-1 in opcode_decode_ffn
# ---------------------------------------------------------------------------


def test_opcode_byte_lo_cross_step_finding_fires():
    """The OPCODE_BYTE_LO discovery must surface from the safety analysis.

    ``opcode_decode_ffn`` declares ``reads={"OPCODE_BYTE_LO.*.-1", ...}``
    (see ``ops/l5_ops.py``) and ``layer5_fetch`` writes ``OPCODE_BYTE_LO``
    in the same step. On VM step 1 there is no prev step, so the decoder
    reads 0 and silently misroutes.

    Pinning this finding ensures the safety check stays wired even if the
    motivating bug eventually gets a real fix (the fix may rewrite the
    read to ``OR(OPCODE_BYTE_LO.*.-1, OPCODE_BYTE_LO)``, which would NOT
    suppress this warning because the same-step writer is still present).
    """
    ops = _collect_ops_for_compile(
        alu_mode="lookup",
        enable_conversational_io=False,
        enable_tool_calling=False,
        enable_neural_io_think_protocol=False,
    )
    findings = _find_cross_step_reads_with_same_step_writers(ops)
    matched = [
        (c, r, ws) for (c, r, ws) in findings
        if "OPCODE_BYTE_LO" in r and c == "opcode_decode_ffn"
    ]
    assert matched, (
        "Expected the canonical OPCODE_BYTE_LO cross-step finding for "
        "opcode_decode_ffn to surface. If it disappeared, either the "
        "underlying bake was fixed (re-pin the test to the next motivating "
        "case) or the safety check regressed."
    )
    _consumer, _read, writers = matched[0]
    assert "layer5_fetch" in writers, (
        f"Expected layer5_fetch as a same-step writer of OPCODE_BYTE_LO; "
        f"got {writers!r}. The motivating same-step write moved or was "
        f"renamed — update the assertion or check ops/l5_ops.py."
    )


# ---------------------------------------------------------------------------
# 2. ``warnings.warn`` emission via ``CrossStepReadWarning``
# ---------------------------------------------------------------------------


def test_emit_uses_cross_step_read_warning_category():
    """``_emit_cross_step_safety_warnings`` must emit each finding as a
    ``CrossStepReadWarning``. The dedicated category lets callers filter
    or promote-to-error via the stdlib ``warnings`` mechanism without
    touching unrelated ``UserWarning`` traffic.
    """
    # Minimal synthetic op pair that triggers exactly one finding:
    # writer writes BASE in same step; reader reads BASE.*.-1.
    writer = Operation(
        name="writer_op",
        reads=set(),
        writes={"FOO"},
        kind="ffn",
        bake_fn=_noop_bake,
        phase=0.0,
    )
    reader = Operation(
        name="reader_op",
        reads={"FOO.*.-1"},
        writes={"BAR"},
        kind="ffn",
        bake_fn=_noop_bake,
        phase=1.0,
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", CrossStepReadWarning)
        findings = _emit_cross_step_safety_warnings([writer, reader])
    assert len(findings) == 1
    consumer, ssa, writers = findings[0]
    assert consumer == "reader_op"
    assert ssa == "FOO.*.-1"
    assert writers == ("writer_op",)
    cross_step_warns = [
        w for w in captured if issubclass(w.category, CrossStepReadWarning)
    ]
    assert len(cross_step_warns) == 1, (
        f"expected exactly one CrossStepReadWarning; got "
        f"{len(cross_step_warns)} of {len(captured)} total warnings"
    )
    msg = str(cross_step_warns[0].message)
    assert "reader_op" in msg
    assert "FOO.*.-1" in msg
    assert "writer_op" in msg


# ---------------------------------------------------------------------------
# 3. Self-writers are excluded (canonical back-edge pattern)
# ---------------------------------------------------------------------------


def test_self_writer_does_not_trigger_warning():
    """An op that both writes ``X`` and reads ``X.*.-1`` declares the
    canonical back-edge: it consumes its own prior-step write. That is
    NOT a step-1 zero-propagation bug — step 1 just sees the dim's
    zero-init, which is the intended starting state for a recurrence.

    The check excludes self-writers so this common pattern stays quiet.
    """
    self_referential = Operation(
        name="recurrence_op",
        reads={"STATE.*.-1"},
        writes={"STATE"},  # writes same base dim it reads (cross-step)
        kind="ffn",
        bake_fn=_noop_bake,
        phase=0.0,
    )
    findings = _find_cross_step_reads_with_same_step_writers(
        [self_referential]
    )
    assert findings == [], (
        f"self-writer of base dim must not trigger the safety check; got "
        f"{findings}"
    )


# ---------------------------------------------------------------------------
# 4. No findings -> no warnings on a clean synthetic op set
# ---------------------------------------------------------------------------


def test_clean_op_set_produces_no_findings_or_warnings():
    """A purely same-step op chain with no ``.*.-1`` reads must produce
    no findings and emit no warnings.
    """
    ops = [
        Operation(
            name="a",
            reads=set(),
            writes={"X"},
            kind="ffn",
            bake_fn=_noop_bake,
            phase=0.0,
        ),
        Operation(
            name="b",
            reads={"X"},
            writes={"Y"},
            kind="ffn",
            bake_fn=_noop_bake,
            phase=1.0,
        ),
    ]
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", CrossStepReadWarning)
        findings = _emit_cross_step_safety_warnings(ops)
    assert findings == []
    assert not [
        w for w in captured if issubclass(w.category, CrossStepReadWarning)
    ]


# ---------------------------------------------------------------------------
# 5. Production op set: baseline finding count is non-trivial
# ---------------------------------------------------------------------------


def test_production_op_set_has_nontrivial_finding_baseline():
    """Today's production op set must produce a non-zero number of
    findings — otherwise the check is silently no-op'ing and would not
    catch regressions like the OPCODE_BYTE_LO discovery.

    The exact count drifts as ops are added / refactored; pin a loose
    floor (>= 10) rather than a brittle exact number. As of the initial
    landing the count is ~82.
    """
    ops = _collect_ops_for_compile(
        alu_mode="lookup",
        enable_conversational_io=False,
        enable_tool_calling=False,
        enable_neural_io_think_protocol=False,
    )
    findings = _find_cross_step_reads_with_same_step_writers(ops)
    assert len(findings) >= 10, (
        f"expected >= 10 cross-step-read findings on today's op set "
        f"(initial landing baseline ~82); got {len(findings)}. If this "
        f"dropped to near zero, celebrate — the OUTPUT_HI / IF_VAR SCC "
        f"may have been broken — and consider tightening the floor."
    )
