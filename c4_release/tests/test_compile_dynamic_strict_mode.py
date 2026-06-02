"""B14 prep tests for ``compile_full_vm_dynamic(strict=...)``.

These tests pin the behaviour of the strict-mode admission gate. The
``strict`` parameter defaults to ``False`` (off) per the B14 rollout
plan in ``c4_release/docs/DYNAMIC_SCHEDULER_MIGRATION_PLAN.md``; flipping
the default is deferred until B9 (dim decomposition) and B12 (declaration
backfill) fully land.

Phase 7.A.5 B14 attempt
-----------------------
Strict mode now supports a *cycle-aware* admission policy via
``allow_sealed_cycles=True`` (the strict-mode default). The SCC of
cycle members is accepted as a sealed group: the hybrid scheduler still
falls back to phase ordering INSIDE the SCC, but every op OUTSIDE the
SCC must be cleanly placeable from declared deps alone. Today's
production op set has 92 cycle members (one large OUTPUT_HI / IF_VAR
SCC) and 0 non-cycle ``phase_required_but_undeclared`` /
``phase_inconsistent_with_deps`` ops, so cycle-aware strict mode admits
the compile. Setting ``allow_sealed_cycles=False`` restores the legacy
"any cycle is a failure" behaviour.

The tests here cover four regression surfaces:

1. **Default is OFF.** ``compile_full_vm_dynamic()`` with no ``strict``
   kwarg behaves exactly as before, so existing callers and the B11
   byte-identity invariant are unaffected.
2. **Cycle-aware strict mode admits today's op set.** With
   ``allow_sealed_cycles=True`` the strict-mode admission check passes
   even though 92 ops are cycle members, because the rest of the graph
   is fully declared.
3. **Legacy strict (``allow_sealed_cycles=False``) still fails today**
   on the production op set with a structured error citing the cycle.
4. **Strict mode succeeds** on a synthetic mini-layout where every op
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
    _strict_mode_sccs,
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
# 2. Cycle-aware strict mode on current op set: must ADMIT today's set
# ---------------------------------------------------------------------------


def test_strict_mode_categoriser_is_clean_outside_the_scc():
    """On the canonical op set (today, 2026-06-01) every op outside the
    OUTPUT_HI SCC must categorise cleanly. The strict-mode admission
    gate then accepts the SCC as a sealed group and lets the hybrid
    scheduler route those members via phase tiebreaker.

    Concretely: ``phase_required_but_undeclared`` and
    ``phase_inconsistent_with_deps`` must both be empty. ``cycle_members``
    is allowed to be non-empty (the SCC is sealed).

    This is the Phase 7.A.5 B14 "go" signal — when this test passes
    cleanly, cycle-aware strict mode can become the production default.
    """
    ops = _collect_ops_for_compile(
        alu_mode="lookup",
        enable_conversational_io=False,
        enable_tool_calling=False,
        enable_neural_io_think_protocol=False,
    )
    buckets = _strict_mode_categorise(ops)
    assert buckets["phase_required_but_undeclared"] == [], (
        f"Strict mode found {len(buckets['phase_required_but_undeclared'])} "
        f"non-cycle ops that need a static phase pin to be placed: "
        f"{buckets['phase_required_but_undeclared']}. Backfill explicit "
        f"requires={{\"after\": ...}} / reads / consumes_fresh for these "
        f"ops per DYNAMIC_SCHEDULER_MIGRATION_PLAN.md B12 before flipping "
        f"strict-mode default to True."
    )
    assert buckets["phase_inconsistent_with_deps"] == [], (
        f"Strict mode found {len(buckets['phase_inconsistent_with_deps'])} "
        f"ops whose static phase is EARLIER than the declared dep DAG "
        f"requires: {buckets['phase_inconsistent_with_deps']}. This is "
        f"either a phase typo or a latent cycle whose member's depth is "
        f"the -1 sentinel."
    )
    # SCC is still expected to be non-empty: until B9 dim decomposition
    # lands, the OUTPUT_HI / IF_VAR back-edges keep ~90 ops cyclic.
    assert buckets["cycle_members"], (
        "Expected the OUTPUT_HI / IF_VAR SCC to still be present today. "
        "If empty, celebrate — and consider flipping to "
        "allow_sealed_cycles=False as the strict-mode default."
    )


def test_cycle_aware_strict_mode_admits_current_op_set():
    """The cycle-aware strict mode (``allow_sealed_cycles=True``) must
    accept today's op set without raising.

    The hybrid scheduler still uses phase ordering INSIDE the SCC, but
    every op outside the SCC is dep-derived. This is the Phase 7.A.5
    B14 attempt landing point.
    """
    ops = _collect_ops_for_compile(
        alu_mode="lookup",
        enable_conversational_io=False,
        enable_tool_calling=False,
        enable_neural_io_think_protocol=False,
    )
    # Must NOT raise.
    _assert_strict_mode_clean(ops, allow_sealed_cycles=True)


def test_legacy_strict_mode_still_fails_on_current_op_set():
    """``allow_sealed_cycles=False`` reproduces the pre-B14 strict
    behaviour: any cycle is a hard failure. Today's op set has ~92
    cycle members so the error must cite them.
    """
    with pytest.raises(StrictModeUnschedulableError) as exc_info:
        compile_full_vm_dynamic(
            strict=True,
            allow_sealed_cycles=False,
            disk_cache=False,
        )
    err = exc_info.value
    # Lower bound: Phase A reported 57 SCC members; the op set has since
    # grown so today it's ~92. Allow 50..120 to absorb minor drift.
    assert 50 <= len(err.cycle_members) <= 120, (
        f"Expected ~92 cycle members on current op set (Phase A baseline "
        f"~57, today drifted up); got {len(err.cycle_members)}. If this "
        f"is now near zero, drop allow_sealed_cycles=False from this test."
    )
    # All offending op names are non-empty strings — the error has to
    # name them so a future agent can act on the message.
    for name in err.cycle_members:
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
        compile_full_vm_dynamic(
            strict=True,
            allow_sealed_cycles=False,
            disk_cache=False,
        )
    msg = str(exc_info.value)
    assert "dep_graph_cycle_member" in msg
    # Counts must be present so the message is actionable at a glance.
    assert "(" in msg and ")" in msg
    # Resolution hint must point at the migration plan units.
    assert "B9" in msg
    assert "B12" in msg


def test_strict_mode_does_not_pollute_disk_cache():
    """Strict-mode failure must abort BEFORE the disk-cache lookup runs.

    If the gate ran AFTER the cache hit path, a stale cached layout
    could mask a regression. Verify by passing ``disk_cache=True`` and
    ``allow_sealed_cycles=False`` (the legacy strict path that still
    fails today) and confirming the same structured error is raised.
    """
    with pytest.raises(StrictModeUnschedulableError):
        compile_full_vm_dynamic(
            strict=True,
            allow_sealed_cycles=False,
            disk_cache=True,
        )


def test_strict_mode_scc_is_a_single_large_component():
    """The Phase A finding is that today's cycle members are dominated
    by ONE large SCC (the OUTPUT_HI / IF_VAR back-edge cluster), not a
    fragmented set of small cycles. This invariant is what makes the
    cycle-aware sealed-group acceptance safe: there's one well-known
    group whose internal order is phase-tiebroken, not dozens of tiny
    cycles with overlapping membership.

    Note: ``cycle_members`` in the categoriser counts both true SCC
    members AND ops transitively downstream of the SCC (Kahn never
    drains them). The SCC count itself (from Tarjan on the cycle
    sub-graph) is the true "tangled cluster" measurement.
    """
    ops = _collect_ops_for_compile(
        alu_mode="lookup",
        enable_conversational_io=False,
        enable_tool_calling=False,
        enable_neural_io_think_protocol=False,
    )
    sccs = _strict_mode_sccs(ops)
    assert sccs, "expected at least one SCC on today's op set"
    # Today: 1 SCC of size 70 swallows the OUTPUT_HI / IF_VAR cluster.
    # The other ~22 "cycle_members" are linearly downstream of it and
    # would topologically drain immediately once the SCC was broken.
    assert len(sccs) == 1, (
        f"Expected exactly 1 SCC today (the OUTPUT_HI / IF_VAR cluster); "
        f"got {len(sccs)} of sizes {[len(s) for s in sccs]}. If the SCC "
        f"is fragmenting, the sealed-group acceptance assumption needs "
        f"re-evaluation."
    )
    largest = sccs[0]
    assert len(largest) >= 50, (
        f"Largest SCC ({len(largest)}) below Phase A baseline (~67). "
        f"If it dropped a lot, celebrate and consider flipping "
        f"allow_sealed_cycles=False as the default."
    )


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

    Run with ``allow_sealed_cycles=False`` so any cycle raises (the
    legacy strict path). The cycle-aware default would seal this 2-op
    SCC and accept the input.
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
        _assert_strict_mode_clean([op_x, op_y], allow_sealed_cycles=False)
    err = exc_info.value
    assert set(err.cycle_members) == {"cycle_x", "cycle_y"}, (
        f"expected both cycle ops cited; got {err.cycle_members}"
    )
    assert err.phase_required_but_undeclared == []
    assert err.phase_inconsistent_with_deps == []


def test_cycle_aware_strict_mode_admits_synthetic_cycle():
    """The complement test: with cycle-aware admission the same 2-cycle
    is accepted as a sealed SCC group. Mirrors the production path where
    the OUTPUT_HI / IF_VAR SCC is admitted because the rest of the graph
    is clean.
    """
    def _noop_bake(*_args, **_kwargs):
        return None

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

    # Must NOT raise — the 2-cycle is a sealed group.
    _assert_strict_mode_clean([op_x, op_y], allow_sealed_cycles=True)


# ---------------------------------------------------------------------------
# 4. Cross-check: strict-mode categorisation agrees with the offline analyzer
# ---------------------------------------------------------------------------


def test_strict_mode_categories_agree_with_analyzer():
    """The strict-mode admission gate must mirror
    ``tools/analyze_scheduler.py:categorise`` so the offline analyzer
    report and the in-process error message describe the same op
    classifications. Any drift here means the agent reading the
    analyzer output gets misled when strict mode fires.

    The analyzer uses ``build_dep_graph`` (with B9 R-OH-2 suppression +
    ``same_layer_edges``) and ``topo_depth`` (peer-aware Kahn). Strict
    mode now uses the same edge model via ``_build_strict_dep_graph``
    and ``_strict_topo_depth`` so the cycle bucket agrees exactly.

    Minor semantic divergence (intentional): the strict-mode
    categoriser broadens the analyzer's ``freely_placeable``
    refinement to include ANY op with no in-edges AND no out-edges,
    even when ``current > derived``. The analyzer leaves these in
    ``phase_required_but_undeclared`` because they appear to need a
    phase pin to be later than depth 0 — but with no edges there is
    nothing for strict mode to validate, so admitting them is safe.
    This test verifies the divergence is strictly on no-edge ops only.

    Note: the analyzer enables ALL flag-gated ops via its own
    ``collect_ops`` (think-protocol, tool-call, etc), while strict
    mode by default runs against the lookup-mode default flag set. To
    compare like-for-like, we feed the analyzer the same flag-off op
    set strict mode sees.
    """
    from c4_release.tools.analyze_scheduler import (
        build_dep_graph, categorise, topo_depth,
    )

    ops = _collect_ops_for_compile(
        alu_mode="lookup",
        enable_conversational_io=False,
        enable_tool_calling=False,
        enable_neural_io_think_protocol=False,
    )
    in_edges, out_edges, _reasons, same_layer_edges = build_dep_graph(ops)
    depth, cycle = topo_depth(ops, in_edges, out_edges, same_layer_edges)
    analyzer_cats = categorise(ops, depth, cycle, in_edges, out_edges)
    strict_buckets = _strict_mode_categorise(ops)

    analyzer_cycle = {n for n, c in analyzer_cats.items()
                      if c == "dep_graph_cycle_member"}
    analyzer_req = {n for n, c in analyzer_cats.items()
                    if c == "phase_required_but_undeclared"}
    analyzer_inc = {n for n, c in analyzer_cats.items()
                    if c == "phase_inconsistent_with_deps"}

    # Cycle members: exact equality.
    assert set(strict_buckets["cycle_members"]) == analyzer_cycle, (
        "strict-mode cycle_members must equal analyzer's "
        "dep_graph_cycle_member set"
    )
    # phase_inconsistent_with_deps: exact equality.
    assert set(strict_buckets["phase_inconsistent_with_deps"]) == analyzer_inc

    # phase_required_but_undeclared: strict mode is a subset of the
    # analyzer's set; the difference must consist entirely of
    # no-in/no-out ops (which strict mode treats as freely placeable).
    strict_req = set(strict_buckets["phase_required_but_undeclared"])
    assert strict_req.issubset(analyzer_req), (
        f"strict mode's phase_required_but_undeclared must be a subset "
        f"of analyzer's. Extras in strict: {strict_req - analyzer_req}"
    )
    only_in_analyzer = analyzer_req - strict_req
    for name in only_in_analyzer:
        assert not in_edges[name] and not out_edges[name], (
            f"analyzer says {name!r} is phase_required_but_undeclared "
            f"but strict mode accepted it WITHOUT the no-edges "
            f"refinement applying. in={len(in_edges[name])} "
            f"out={len(out_edges[name])}"
        )


# ---------------------------------------------------------------------------
# 5. Byte-identity: cycle-aware strict mode must match the default layout
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_cycle_aware_strict_mode_byte_identical_to_default():
    """Cycle-aware strict mode (``strict=True, allow_sealed_cycles=True``)
    must produce a byte-identical layout to the default
    ``strict=False`` dynamic compile.

    Strict mode is purely an admission gate — the underlying
    ``compute_dynamic_schedule`` is unchanged. So when admission passes,
    the layout must agree exactly with the strict-off path.
    """
    import torch as _torch

    default_model, default_layout = compile_full_vm_dynamic(
        strict=False,
        disk_cache=False,
    )
    strict_model, strict_layout = compile_full_vm_dynamic(
        strict=True,
        allow_sealed_cycles=True,
        disk_cache=False,
    )

    sd_default = default_model.state_dict()
    sd_strict = strict_model.state_dict()
    assert set(sd_default.keys()) == set(sd_strict.keys()), (
        "state_dict keys differ between strict and default modes"
    )
    diff_keys = [
        k for k in sd_default.keys()
        if not _torch.equal(sd_default[k], sd_strict[k])
    ]
    assert diff_keys == [], (
        f"strict mode produced {len(diff_keys)} tensor differences vs "
        f"default; first 5: {diff_keys[:5]}"
    )
    for field in ("d_model", "n_layers", "dim_positions", "dim_sizes",
                  "ffn_widths"):
        s_val = getattr(default_layout, field)
        d_val = getattr(strict_layout, field)
        assert s_val == d_val, (
            f"layout field {field!r} differs: default={s_val!r} "
            f"strict={d_val!r}"
        )


def test_allow_sealed_cycles_kwarg_default_is_true():
    """The B14 attempt landed cycle-aware strict mode as the default.
    When a caller does ``compile_full_vm_dynamic(strict=True)`` they
    get cycle-aware admission, not the legacy fail-on-any-cycle path.
    """
    sig = inspect.signature(compile_full_vm_dynamic)
    assert "allow_sealed_cycles" in sig.parameters
    param = sig.parameters["allow_sealed_cycles"]
    assert param.default is True, (
        f"allow_sealed_cycles must default to True (cycle-aware strict "
        f"is the B14 acceptance path); got {param.default!r}"
    )
