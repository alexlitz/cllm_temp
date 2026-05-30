"""Shared assertions for per-layer per-op claim-verification audits.

Used by ``test_l0_marker_transitions.py``, ``test_l2_mem_byte_flags.py``,
``test_l4_pc_relay.py`` (and any future per-layer harness) to keep the
boilerplate per-file minimal while preserving layer-labelled error
messages.

All helpers take the ``static_claims_report`` session fixture (built once
per pytest session by ``conftest.py``) and a layer label like ``"L0"`` so
failure messages name the offending layer explicitly.
"""

from __future__ import annotations


def _results_by_name(static_report):
    return {r.op_name: r for r in static_report.results}


def assert_no_drift(static_report, layer_label: str, op_name: str) -> None:
    """Op must appear in the report and have no declared-but-not-written cells."""
    results = _results_by_name(static_report)
    assert op_name in results, (
        f"{layer_label} op {op_name!r} was not exercised by "
        f"verify_claims_static; the op may have been renamed, "
        f"deregistered, or had its claims emptied. Available ops in "
        f"report: {sorted(results)[:10]}..."
    )
    r = results[op_name]
    assert r.ok, (
        f"{layer_label} op {op_name!r} has declaration drift: "
        f"declared={len(r.declared)} observed={len(r.observed)} "
        f"unused_decl={sorted(r.declared_but_not_written)[:5]}"
    )


def assert_op_fires(static_report, layer_label: str, op_name: str) -> None:
    """Op must dispatch and emit at least one observable write (not INERT)."""
    r = _results_by_name(static_report)[op_name]
    assert not r.inert, (
        f"{layer_label} op {op_name!r} reported INERT: bake_fn "
        f"dispatched but produced no observable diff."
    )
    assert len(r.observed) > 0, (
        f"{layer_label} op {op_name!r} fired but wrote no observable "
        f"cells (observed=0)."
    )


def assert_op_absent(static_report, layer_label: str, op_name: str) -> None:
    """Op must NOT appear in the report (used for known-empty-claims ops).

    ``verify_claims_static`` only inspects ops with non-empty ``claims``;
    if a previously-empty-claims op shows up, somebody added claims or
    flipped an ``enable=`` gate and the audit needs to migrate that op
    into the drift-checked list.
    """
    names = {r.op_name for r in static_report.results}
    assert op_name not in names, (
        f"{layer_label} op {op_name!r} unexpectedly appeared in the "
        f"default-build verifier report. Previously shipped with empty "
        f"claims; if claims were added intentionally, move {op_name!r} "
        f"into the layer's drift-checked list."
    )
