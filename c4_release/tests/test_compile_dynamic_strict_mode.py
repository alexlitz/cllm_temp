"""B14 prep tests for ``compile_full_vm_dynamic(strict=...)``.

These tests pin the behaviour of the strict-mode admission gate. The
``strict`` parameter defaults to ``False`` (off) per the B14 rollout
plan in ``c4_release/docs/DYNAMIC_SCHEDULER_MIGRATION_PLAN.md``; flipping
the default is deferred until B9 (dim decomposition) and B12 (declaration
backfill) fully land. The tests here cover three regression surfaces:

1. **Default is OFF.** ``compile_full_vm_dynamic()`` with no ``strict``
   kwarg behaves exactly as before, so existing callers and the B11
   byte-identity invariant are unaffected.
2. **Strict mode fails today** on the production op set with a
   structured error citing the 70+ cycle members and 23+
   ``phase_required_but_undeclared`` ops the Phase A diagnostic found.
3. **Strict mode succeeds** on a synthetic mini-layout where every op
   declares enough deps to be either ``phase_pinned_by_deps`` or
   ``freely_placeable`` — proving the gate isn't unconditionally
   rejecting.

The tests are *fast* (no full bake) by exercising the categoriser
directly for the synthetic layout case and by relying on strict mode
raising BEFORE any LayerCompiler work happens for the failure case.
"""

import inspect

import pytest

from c4_release.neural_vm.unified_compiler.full_vm_compiler_dynamic import (
    StrictModeUnschedulableError,
    _assert_strict_mode_clean,
    _collect_ops_for_compile,
    _strict_mode_categorise,
    compile_full_vm_dynamic,
)
from c4_release.neural_vm.unified_compiler.layer_compiler import Operation


# ---------------------------------------------------------------------------
# 1. Default-off guarantee
# ---------------------------------------------------------------------------


def test_strict_kwarg_defaults_to_false():
    """``compile_full_vm_dynamic`` must keep ``strict=False`` as default.

    B14 is the unit that flips this default; until then any change to the
    default is a behavioural break for existing callers that import the
    function with positional / kwargs that don't include ``strict``.
    """
    sig = inspect.signature(compile_full_vm_dynamic)
    assert "strict" in sig.parameters, (
        "compile_full_vm_dynamic must expose a 'strict' parameter for B14 prep"
    )
    param = sig.parameters["strict"]
    assert param.default is False, (
        f"strict parameter default must be False (B14 rollout requires "
        f"off-by-default); got {param.default!r}"
    )
    assert param.kind is inspect.Parameter.KEYWORD_ONLY, (
        "strict must be keyword-only to prevent positional-argument breakage"
    )


# ---------------------------------------------------------------------------
# 2. Strict mode on current op set: must fail with structured error
# ---------------------------------------------------------------------------


def test_strict_mode_rejects_current_production_op_set():
    """On the canonical op set (today, 2026-06-01) strict mode must
    refuse to compile because of the OUTPUT_HI SCC and the 23 ops the
    Phase A diagnostic catalogued as ``phase_required_but_undeclared``.

    This test is the B14 "go" signal in reverse: when it starts
    PASSING with an empty error, the dep graph is fully declared and
    strict mode can become the default. Until then it locks in the
    expected failure.
    """
    with pytest.raises(StrictModeUnschedulableError) as exc_info:
        compile_full_vm_dynamic(strict=True, disk_cache=False)

    err = exc_info.value
    # Lower bound: Phase A reported 57 SCC members; the op set has since
    # grown so today it's ~70. Allow 50..120 to absorb minor drift.
    assert 50 <= len(err.cycle_members) <= 120, (
        f"Expected ~70 cycle members on current op set (Phase A baseline "
        f"~57, today drifted up); got {len(err.cycle_members)}. If this "
        f"is now near zero, celebrate — and re-enable the B14 default flip."
    )
    # Phase A: 27 ops in phase_required_but_undeclared. B12 wave 1
    # landed 2 declarations so today's count is ~23. Lower bound 10
    # ensures we notice when B12 finishes the backfill.
    assert len(err.phase_required_but_undeclared) >= 10, (
        f"Expected at least 10 phase_required_but_undeclared ops today "
        f"(Phase A: 27, B12 in flight); got "
        f"{len(err.phase_required_but_undeclared)}. If near zero, "
        f"flip strict default in B14."
    )
    # All offending op names are non-empty strings — the error has to
    # name them so a future agent can act on the message.
    for name in err.cycle_members + err.phase_required_but_undeclared:
        assert isinstance(name, str) and name, (
            f"Strict-mode error must cite each offending op by name; "
            f"got {name!r}"
        )


def test_strict_mode_error_message_lists_offending_classes():
    """The ``str(error)`` must mention each non-empty bucket by name so a
    bake author reading the traceback knows whether the failure is a
    cycle (needs B9) or a missing declaration (needs B12).
    """
    with pytest.raises(StrictModeUnschedulableError) as exc_info:
        compile_full_vm_dynamic(strict=True, disk_cache=False)
    msg = str(exc_info.value)
    assert "dep_graph_cycle_member" in msg
    assert "phase_required_but_undeclared" in msg
    # Counts must be present so the message is actionable at a glance.
    assert "(" in msg and ")" in msg
    # Resolution hint must point at the migration plan units.
    assert "B9" in msg
    assert "B12" in msg


def test_strict_mode_does_not_pollute_disk_cache():
    """Strict-mode failure must abort BEFORE the disk-cache lookup runs.

    If the gate ran AFTER the cache hit path, a stale cached layout
    could mask a regression. Verify by passing ``disk_cache=True`` and
    confirming the same structured error is raised.
    """
    with pytest.raises(StrictModeUnschedulableError):
        compile_full_vm_dynamic(strict=True, disk_cache=True)


# ---------------------------------------------------------------------------
# 3. Strict mode on a synthetic fully-declared mini layout: must succeed
# ---------------------------------------------------------------------------


def _make_clean_mini_op_set():
    """Return a 4-op linear chain where every op is dep-pinned.

    Layout: ``a -> b -> c -> d`` via ``writes``/``reads`` on shared dims.
    Phases match the dep depth exactly so the analyzer categoriser tags
    every op as ``phase_pinned_by_deps`` (no cycles, no gap between
    phase and dep depth).
    """
    def _noop_bake(*_args, **_kwargs):
        return None

    return [
        Operation(
            name="op_a",
            reads=set(),
            writes={"DIM_X"},
            kind="ffn",
            bake_fn=_noop_bake,
            phase=0.0,
        ),
        Operation(
            name="op_b",
            reads={"DIM_X"},
            writes={"DIM_Y"},
            kind="ffn",
            bake_fn=_noop_bake,
            phase=1.0,
        ),
        Operation(
            name="op_c",
            reads={"DIM_Y"},
            writes={"DIM_Z"},
            kind="ffn",
            bake_fn=_noop_bake,
            phase=2.0,
        ),
        Operation(
            name="op_d",
            reads={"DIM_Z"},
            writes={"DIM_W"},
            kind="ffn",
            bake_fn=_noop_bake,
            phase=3.0,
        ),
    ]


def test_strict_mode_admits_fully_declared_synthetic_layout():
    """A linear-chain op set with phases matching dep depth must pass
    the strict admission gate. This is the post-B14 success path that
    today only triggers for synthetic / freshly-declared op sets.
    """
    ops = _make_clean_mini_op_set()
    buckets = _strict_mode_categorise(ops)
    assert buckets["cycle_members"] == [], (
        f"clean synthetic layout has no cycles; got {buckets['cycle_members']}"
    )
    assert buckets["phase_required_but_undeclared"] == [], (
        f"clean synthetic layout has every phase aligned with deps; got "
        f"{buckets['phase_required_but_undeclared']}"
    )
    assert buckets["phase_inconsistent_with_deps"] == [], (
        f"clean synthetic layout has no phase/dep contradictions; got "
        f"{buckets['phase_inconsistent_with_deps']}"
    )
    assert set(buckets["ok"]) == {o.name for o in ops}

    # And the assertion-form helper must not raise on this set.
    _assert_strict_mode_clean(ops)


def test_strict_mode_classifies_synthetic_cycle_as_unschedulable():
    """Inject a back-edge into the synthetic mini layout and confirm
    strict mode catches the cycle. This guards the categoriser against
    silent false-positives on day-to-day dep edits.
    """
    def _noop_bake(*_args, **_kwargs):
        return None

    # Two ops writing each other's reads = a 2-cycle.
    op_x = Operation(
        name="cycle_x",
        reads={"DIM_Y"},
        writes={"DIM_X"},
        kind="ffn",
        bake_fn=_noop_bake,
        phase=0.0,
    )
    op_y = Operation(
        name="cycle_y",
        reads={"DIM_X"},
        writes={"DIM_Y"},
        kind="ffn",
        bake_fn=_noop_bake,
        phase=1.0,
    )

    with pytest.raises(StrictModeUnschedulableError) as exc_info:
        _assert_strict_mode_clean([op_x, op_y])
    err = exc_info.value
    assert set(err.cycle_members) == {"cycle_x", "cycle_y"}, (
        f"expected both cycle ops cited; got {err.cycle_members}"
    )
    assert err.phase_required_but_undeclared == []
    assert err.phase_inconsistent_with_deps == []


# ---------------------------------------------------------------------------
# 4. Cross-check: strict-mode categorisation agrees with the offline analyzer
# ---------------------------------------------------------------------------


def test_strict_mode_categories_agree_with_analyzer():
    """The strict-mode admission gate must mirror
    ``tools/analyze_scheduler.py:categorise`` exactly so the offline
    analyzer report and the in-process error message describe the same
    op classifications. Any drift here means the agent reading the
    analyzer output gets misled when strict mode fires.
    """
    from c4_release.tools.analyze_scheduler import categorise, topo_depth
    from c4_release.neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        _build_dep_graph,
    )

    ops = _collect_ops_for_compile(
        alu_mode="lookup",
        enable_conversational_io=False,
        enable_tool_calling=False,
        enable_neural_io_think_protocol=False,
    )
    in_edges, out_edges = _build_dep_graph(ops)
    depth, cycle = topo_depth(ops, in_edges, out_edges)
    analyzer_cats = categorise(ops, depth, cycle, in_edges, out_edges)
    strict_buckets = _strict_mode_categorise(ops)

    # Strict-mode failure buckets must be a subset of (or equal to) the
    # analyzer's failure buckets. Both ought to count cycle members
    # identically; the "phase_required_but_undeclared" count is what
    # B12 will drive to zero.
    analyzer_cycle = {n for n, c in analyzer_cats.items()
                      if c == "dep_graph_cycle_member"}
    analyzer_req = {n for n, c in analyzer_cats.items()
                    if c == "phase_required_but_undeclared"}
    analyzer_inc = {n for n, c in analyzer_cats.items()
                    if c == "phase_inconsistent_with_deps"}

    assert set(strict_buckets["cycle_members"]) == analyzer_cycle, (
        "strict-mode cycle_members must equal analyzer's "
        "dep_graph_cycle_member set"
    )
    assert set(strict_buckets["phase_required_but_undeclared"]) == analyzer_req
    assert set(strict_buckets["phase_inconsistent_with_deps"]) == analyzer_inc
