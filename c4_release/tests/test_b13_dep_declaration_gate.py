"""B13 — CI gate that rejects new ops without sufficient dep declarations.

Part of the dynamic-scheduler migration (see
``c4_release/docs/DYNAMIC_SCHEDULER_MIGRATION_PLAN.md`` §B13). The gate
runs ``tools/analyze_scheduler.py`` against the full production op set
and demands that EVERY op land in one of the two "scheduler is happy"
buckets:

  * ``freely_placeable``     — no incoming / outgoing dep edges; dynamic
    scheduler may put it anywhere; or
  * ``phase_pinned_by_deps`` — the declared deps are sufficient to
    derive the op's current layer (or the op is structurally pinned via
    ``kind=model`` / post-pass phase).

Any op landing in ``phase_required_but_undeclared``,
``phase_inconsistent_with_deps``, or ``dep_graph_cycle_member`` means
the author relied on hardcoded ``phase=N.M`` ordering rather than
declaring the dependency. The gate fails so the author either adds the
missing ``reads`` / ``writes`` / ``produces`` / ``consumes_fresh`` /
``requires`` annotation or opts into ``freely_placeable``.

The gate is SCAFFOLDED OFF today because B12 (backfill of the existing
``phase_required_but_undeclared`` ops) has not landed yet. Running the
gate against ``main`` right now would flag 26 ops (15
``phase_required_but_undeclared`` + 3 ``phase_inconsistent_with_deps``
+ a slice of the 72 ``dep_graph_cycle_member`` SCC members). After B12
the expected number is 0 — that is when this gate is meant to flip on.

To enable the gate (after B12 lands):

  * Set ``B13_GATE_ENABLED=1`` in the CI environment, OR
  * Touch the marker file ``c4_release/docs/B13_GATE.flag`` in the
    repo. The marker is intentionally checked-in (not just a local
    sentinel) so flipping the gate is a one-line commit reviewable on
    its own.

Either signal turns the skip off. There is no other knob — the test
either runs the analyzer-derived assertion or it skips with an
explanatory message.
"""

from __future__ import annotations

import importlib.util
import os
from typing import Set

import pytest


# ---------------------------------------------------------------------------
# Sentinel: B13 gate is OFF by default. Flip exactly one of these to enable.
# ---------------------------------------------------------------------------

_B13_ENV_VAR = "B13_GATE_ENABLED"

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_B13_FLAG_FILE = os.path.join(_REPO_ROOT, "docs", "B13_GATE.flag")
_ANALYZE_PATH = os.path.join(_REPO_ROOT, "tools", "analyze_scheduler.py")


def _gate_enabled() -> bool:
    """Return True iff the B13 dep-declaration gate should run.

    The single-line flip after B12 lands is either:

      * ``export B13_GATE_ENABLED=1`` in the CI workflow, or
      * ``touch c4_release/docs/B13_GATE.flag`` and commit the marker.

    Either suffices. Keep this function intentionally trivial so the
    flip is auditable.
    """
    if os.environ.get(_B13_ENV_VAR, "").strip() not in ("", "0", "false", "False"):
        return True
    if os.path.exists(_B13_FLAG_FILE):
        return True
    return False


# Categories that pass the gate. Anything else fails it.
_ALLOWED_CATEGORIES: Set[str] = {
    "freely_placeable",
    "phase_pinned_by_deps",
}

# Categories that explicitly fail. Listed here for clarity / error messages.
_REJECTED_CATEGORIES: Set[str] = {
    "phase_required_but_undeclared",
    "phase_inconsistent_with_deps",
    "dep_graph_cycle_member",
}


def _load_analyze_scheduler():
    """Load ``tools/analyze_scheduler.py`` as a module.

    ``tools/`` is not a Python package, so use importlib to side-load the
    file. Mirrors ``tests/test_op_requires_op_name.py``.
    """
    spec = importlib.util.spec_from_file_location(
        "_b13_analyze_scheduler", _ANALYZE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_op_has_sufficient_dep_declarations():
    """B13 gate: every op must be ``freely_placeable`` OR ``phase_pinned_by_deps``.

    Skipped unless ``B13_GATE_ENABLED`` is set or
    ``c4_release/docs/B13_GATE.flag`` exists. See module docstring for
    rationale and flip instructions.
    """
    if not _gate_enabled():
        pytest.skip(
            "B13 dep-declaration gate is OFF by default. Set "
            f"{_B13_ENV_VAR}=1 or create {_B13_FLAG_FILE} after B12 "
            "backfill lands. See docs/DYNAMIC_SCHEDULER_MIGRATION_PLAN.md §B13."
        )

    analyze_scheduler = _load_analyze_scheduler()
    ops = analyze_scheduler.collect_ops()

    # Hard error first: any invalid ``requires`` op-name reference would
    # silently disappear from the dep graph and produce a misleading
    # categorisation. Mirror the analyzer's own pre-flight check.
    ref_errors = analyze_scheduler.validate_requires_op_refs(ops)
    assert not ref_errors, (
        "invalid requires op-name references (B10 schema):\n  - "
        + "\n  - ".join(ref_errors)
    )

    in_e, out_e, _reasons = analyze_scheduler.build_dep_graph(ops)
    depth, cycle_members = analyze_scheduler.topo_depth(ops, in_e, out_e)
    cats = analyze_scheduler.categorise(ops, depth, cycle_members, in_e, out_e)

    offenders = {
        name: cat for name, cat in cats.items() if cat not in _ALLOWED_CATEGORIES
    }
    if offenders:
        # Group by category for a readable failure message.
        by_cat = {}
        for name, cat in offenders.items():
            by_cat.setdefault(cat, []).append(name)
        lines = [
            "B13 gate FAILED: {n} ops lack sufficient dep declarations "
            "to be placed by the dynamic scheduler.".format(n=len(offenders)),
            "",
            "Allowed buckets: " + ", ".join(sorted(_ALLOWED_CATEGORIES)),
            "Rejected here:   " + ", ".join(sorted(_REJECTED_CATEGORIES)),
            "",
            "Add the missing reads/writes/produces/consumes_fresh/requires "
            "annotation, or — if the op truly has no order constraint — "
            "leave it dep-free so it lands in `freely_placeable`. Run "
            "``python tools/analyze_scheduler.py`` for the detailed report.",
            "",
        ]
        for cat in sorted(by_cat):
            names = sorted(by_cat[cat])
            preview = names[:10]
            more = "" if len(names) <= 10 else f" (+{len(names) - 10} more)"
            lines.append(f"  [{cat}] {len(names)} ops: {preview}{more}")
        pytest.fail("\n".join(lines))
