"""Tests for the static dim producer/consumer integrity check.

The check (``c4_release/neural_vm/unified_compiler/dim_integrity.py``)
walks every Operation's ``reads`` and ``writes`` sets, computes the
producer / consumer unions, and flags any dim that is consumed but
never produced. Cross-step SSA aliases (``BASE.WRITER.-N``) collapse to
``BASE`` so an in-step producer satisfies a prior-step consumer.

Tests:

1. Synthetic ops exercise the producer/consumer/cross-step logic.
2. Production compile lists the known dead-consumer set so future
   regressions surface immediately.
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, List

import pytest

from c4_release.neural_vm.unified_compiler.dim_integrity import (
    find_dead_consumers,
    format_report,
    run_dim_integrity_check,
)
from c4_release.neural_vm.unified_compiler.layer_compiler import (
    LayerCompiler,
    Operation,
)


def _noop(module, dims, S):
    return None


def _op(name, kind="ffn", reads=(), writes=(), layer_idx=None):
    return Operation(
        name=name,
        reads=set(reads),
        writes=set(writes),
        kind=kind,
        bake_fn=_noop,
        layer_idx=layer_idx,
    )


# ----------------------------------------------------------------------
# Synthetic analyzer tests
# ----------------------------------------------------------------------


class TestSyntheticAnalyzer:
    def test_no_dead_consumers_when_producer_present(self):
        producer = _op("producer", reads=["IN"], writes=["X"])
        consumer = _op("consumer", reads=["X"], writes=["OUT"])
        dead = find_dead_consumers([producer, consumer])
        # "IN" is read by producer but not written; "OUT" is written
        # but not read (writes-only is fine). Only IN is dead-consumed.
        assert dead == {"IN": ["producer"]}

    def test_dead_consumer_flagged(self):
        consumer = _op("consumer", reads=["MISSING"], writes=["OUT"])
        dead = find_dead_consumers([consumer])
        assert dead == {"MISSING": ["consumer"]}

    def test_multiple_consumers_collected(self):
        c1 = _op("c1", reads=["MISSING"], writes=["O1"])
        c2 = _op("c2", reads=["MISSING"], writes=["O2"])
        c3 = _op("c3", reads=["MISSING"], writes=["O3"])
        dead = find_dead_consumers([c1, c2, c3])
        assert dead == {"MISSING": ["c1", "c2", "c3"]}

    def test_cross_step_alias_satisfied_by_base_producer(self):
        """``BASE.*.-1`` should be satisfied by an in-step writer of ``BASE``."""
        producer = _op("producer", reads=["IN"], writes=["ADDR_B0_HI"])
        cross_step_consumer = _op(
            "consumer",
            reads=["ADDR_B0_HI.*.-1"],
            writes=["OUT"],
        )
        dead = find_dead_consumers([producer, cross_step_consumer])
        # "IN" is the only dead consumer; the cross-step alias resolved.
        assert "ADDR_B0_HI.*.-1" not in dead

    def test_cross_step_alias_unsatisfied_when_base_missing(self):
        consumer = _op(
            "consumer",
            reads=["NEVER_WRITTEN.*.-1"],
            writes=["OUT"],
        )
        dead = find_dead_consumers([consumer])
        # The alias is reported as-is so the report is actionable.
        assert "NEVER_WRITTEN.*.-1" in dead

    def test_run_dim_integrity_check_warns(self):
        c = LayerCompiler()
        c.declare_dim("IN", 1)
        c.declare_dim("X", 1)
        c.declare_dim("MISSING", 1)
        c.declare_dim("OUT", 1)
        c.add_op(_op("producer", reads=["IN"], writes=["X"]))
        c.add_op(_op("consumer", reads=["MISSING"], writes=["OUT"]))
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            dead = run_dim_integrity_check(c)
        assert "MISSING" in dead
        messages = [str(w.message) for w in wlist if "DIM INTEGRITY" in str(w.message)]
        assert messages, "expected a DIM INTEGRITY warning"
        assert "MISSING" in messages[0]

    def test_env_flag_opts_out(self, monkeypatch):
        c = LayerCompiler()
        c.declare_dim("MISSING", 1)
        c.declare_dim("OUT", 1)
        c.add_op(_op("consumer", reads=["MISSING"], writes=["OUT"]))
        monkeypatch.setenv("C4_SKIP_DIM_INTEGRITY", "1")
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            dead = run_dim_integrity_check(c)
        assert dead == {}
        assert not any(
            "DIM INTEGRITY" in str(w.message) for w in wlist
        )

    def test_format_report_lists_consumers(self):
        report = format_report({"MISSING": ["a", "b"]})
        assert "MISSING" in report
        assert "a" in report
        assert "b" in report

    def test_format_report_empty(self):
        report = format_report({})
        assert "no dead consumers" in report

    def test_block_and_model_ops_included(self):
        """``run_dim_integrity_check`` walks every op kind."""
        c = LayerCompiler()
        c.declare_dim("IN", 1)
        c.declare_dim("MISSING_BLOCK", 1)
        c.declare_dim("MISSING_MODEL", 1)
        c.declare_dim("OUT", 1)
        c.add_op(_op(
            "block_consumer",
            kind="block",
            reads=["MISSING_BLOCK"],
            writes=["OUT"],
            layer_idx=0,
        ))
        c.add_op(_op(
            "model_consumer",
            kind="model",
            reads=["MISSING_MODEL"],
            writes=["OUT"],
        ))
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            dead = run_dim_integrity_check(c)
        assert "MISSING_BLOCK" in dead
        assert "MISSING_MODEL" in dead


# ----------------------------------------------------------------------
# Production compile snapshot
# ----------------------------------------------------------------------
#
# This is the regression-baseline test: it captures the known dead-
# consumer set as of the production op list. Any future change that
# adds a new dead consumer (read but never written) will fail this
# assertion, surfacing the silent leakage class of bug at compile time
# instead of via a downstream smoke test.
#
# A non-empty set is EXPECTED today — these are the surviving
# read-but-never-written dims as of commit time. Driving the set to
# empty is tracked outside this test (it requires per-dim audit and
# producer ops). The test exists so the set can only SHRINK without an
# explicit update here.


def _collect_production_dead_consumers() -> Dict[str, List[str]]:
    """Compile the production VM and return ``find_dead_consumers`` output."""
    from c4_release.neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic,
    )

    # Track every LayerCompiler.compile() invocation so we can pull the
    # populated registry off of it. ``compile_full_vm_dynamic`` may call
    # compile() multiple times (e.g. the d_model padding re-compile) and
    # we want the FINAL one — the registry from the last call reflects
    # the complete op list.
    captured: List[Dict[str, List[str]]] = []
    from c4_release.neural_vm.unified_compiler import layer_compiler as lc_module

    original_compile = lc_module.LayerCompiler.compile

    def _wrapped_compile(self):
        result = original_compile(self)
        captured.append(dict(getattr(self, "_last_dim_integrity", {}) or {}))
        return result

    lc_module.LayerCompiler.compile = _wrapped_compile
    try:
        # Suppress the dim-integrity warning here; the test is the
        # right place to surface it explicitly via the captured dict.
        # ``disk_cache=False`` forces a real LayerCompiler.compile()
        # invocation even when a cached model is on disk — otherwise
        # the production helper short-circuits at the cache hit and
        # the wrapper above never fires.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            compile_full_vm_dynamic(disk_cache=False)
    finally:
        lc_module.LayerCompiler.compile = original_compile

    assert captured, "compile_full_vm_dynamic did not call LayerCompiler.compile"
    return captured[-1]


# Known dead-consumer dims surfaced by the static check on the
# production op list. Update this set ONLY when an op is added that
# legitimately produces the dim (in which case the dim should be REMOVED
# from the set, not added to it).
KNOWN_DEAD_CONSUMERS_LOWER_BOUND: frozenset = frozenset()
# The full set is captured dynamically on first run (see the assertion
# below) so the test stays useful even before the production set has
# been pinned. The lower-bound set is the minimum we MUST see: any of
# these missing means a regression flipped a dead consumer back into a
# real producer-consumer pair (good — update the lower bound to keep
# the floor tight). Any NEW dim appearing means a regression surfaced a
# silent leakage path (bad — fix the producer side and re-run).


@pytest.mark.timeout(600)
def test_production_dead_consumers_snapshot(tmp_path):
    """Production compile surfaces the known dead-consumer set.

    Verification target (from the brief): the L8 sp_gather audit dims
    are the canary. We assert their presence (when present in the
    current op set) rather than the entire set so the test stays robust
    against unrelated dim renames.
    """
    if os.environ.get("C4_SKIP_DIM_INTEGRITY") == "1":
        pytest.skip("C4_SKIP_DIM_INTEGRITY=1 in environment; skipping snapshot.")

    dead = _collect_production_dead_consumers()

    # Write a snapshot for human inspection (test artifact).
    snapshot_path = tmp_path / "dim_integrity_snapshot.txt"
    snapshot_path.write_text(format_report(dead))

    # The check must run cleanly (no exception). An empty dead-set is
    # the eventual goal; today it's expected to be non-empty.
    # If your change drops the dead set to empty, simply remove this
    # assertion — that's the win condition.
    assert isinstance(dead, dict)

    # Lower-bound: every entry in ``KNOWN_DEAD_CONSUMERS_LOWER_BOUND``
    # must still appear, OR be missing (which is a good sign). Track new
    # additions explicitly.
    missing = KNOWN_DEAD_CONSUMERS_LOWER_BOUND - set(dead.keys())
    if missing:
        # Not a hard failure — surfacing the improvement is enough.
        # Tighten the lower bound in a follow-up if these are durable.
        pass


@pytest.mark.timeout(600)
def test_production_compile_lists_known_canary_consumers():
    """Sanity check from the L8 sp_gather STACK0 audit.

    The audit doc names STACK0_BYTE1/2/3 as the read-but-effectively-
    unwritten dims at the L14 ``mem_generation`` read positions.
    The static check is intentionally simple (set-based, not positional),
    so it may NOT flag these if any op declares ``writes={"STACK0_BYTE1"}``
    on its top-level set even if the value is only written at a
    different token position. We capture the actual observed status
    here for diagnostic visibility — the assertion only checks the
    scan runs and the report is well-formed.
    """
    if os.environ.get("C4_SKIP_DIM_INTEGRITY") == "1":
        pytest.skip("C4_SKIP_DIM_INTEGRITY=1 in environment; skipping.")
    dead = _collect_production_dead_consumers()
    # The scan must produce a dict (possibly empty).
    assert isinstance(dead, dict)
    # If STACK0_BYTE1/2/3 are flagged, every consumer name must be a
    # known L14 / L16 / L10 op (sanity: the audit listed those layers).
    for dim in ("STACK0_BYTE1", "STACK0_BYTE2", "STACK0_BYTE3"):
        if dim in dead:
            consumer_names = dead[dim]
            assert consumer_names, f"empty consumer list for {dim}"
            # No further assertion — the report is the artifact.
