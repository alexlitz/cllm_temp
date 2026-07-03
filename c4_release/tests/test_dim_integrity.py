"""Tests for the static dim producer/consumer integrity check.

The check (``c4_release/neural_vm/unified_compiler/dim_integrity.py``)
walks every Operation's ``reads`` and ``writes`` sets, computes the
producer / consumer unions, and flags any dim that is consumed but
never produced. Cross-step SSA aliases (``BASE.WRITER.-N``) collapse to
``BASE`` so an in-step producer satisfies a prior-step consumer.
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, List

import pytest

from c4_release.neural_vm.verification.dim_integrity import (
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


class TestSyntheticAnalyzer:
    def test_no_dead_consumers_when_producer_present(self):
        producer = _op("producer", reads=["IN"], writes=["X"])
        consumer = _op("consumer", reads=["X"], writes=["OUT"])
        dead = find_dead_consumers([producer, consumer])
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
        producer = _op("producer", reads=["IN"], writes=["ADDR_B0_HI"])
        cross_step_consumer = _op(
            "consumer",
            reads=["ADDR_B0_HI.*.-1"],
            writes=["OUT"],
        )
        dead = find_dead_consumers([producer, cross_step_consumer])
        assert "ADDR_B0_HI.*.-1" not in dead

    def test_cross_step_alias_unsatisfied_when_base_missing(self):
        consumer = _op(
            "consumer",
            reads=["NEVER_WRITTEN.*.-1"],
            writes=["OUT"],
        )
        dead = find_dead_consumers([consumer])
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
        assert messages
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
        assert not any("DIM INTEGRITY" in str(w.message) for w in wlist)

    def test_format_report_lists_consumers(self):
        report = format_report({"MISSING": ["a", "b"]})
        assert "MISSING" in report
        assert "a" in report
        assert "b" in report

    def test_format_report_empty(self):
        assert "no dead consumers" in format_report({})

    def test_block_and_model_ops_included(self):
        c = LayerCompiler()
        c.declare_dim("IN", 1)
        c.declare_dim("MISSING_BLOCK", 1)
        c.declare_dim("MISSING_MODEL", 1)
        c.declare_dim("OUT", 1)
        c.add_op(_op(
            "block_consumer", kind="block",
            reads=["MISSING_BLOCK"], writes=["OUT"], layer_idx=0,
        ))
        c.add_op(_op(
            "model_consumer", kind="model",
            reads=["MISSING_MODEL"], writes=["OUT"],
        ))
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            dead = run_dim_integrity_check(c)
        assert "MISSING_BLOCK" in dead
        assert "MISSING_MODEL" in dead


def _collect_production_dead_consumers() -> Dict[str, List[str]]:
    """Compile the production VM and return the dead-consumer dict.

    Wraps ``LayerCompiler.compile`` so the registry captured on the
    final invocation (post any d_model-padding re-compile) is returned.
    ``disk_cache=False`` forces a real compile even on a cache hit.
    """
    from c4_release.neural_vm.unified_compiler import layer_compiler as lc_module
    from c4_release.neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic,
    )

    captured: List[Dict[str, List[str]]] = []
    original_compile = lc_module.LayerCompiler.compile

    def _wrapped_compile(self):
        result = original_compile(self)
        captured.append(dict(getattr(self, "_last_dim_integrity", {}) or {}))
        return result

    lc_module.LayerCompiler.compile = _wrapped_compile
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            compile_full_vm_dynamic(disk_cache=False)
    finally:
        lc_module.LayerCompiler.compile = original_compile

    assert captured, "compile_full_vm_dynamic did not call LayerCompiler.compile"
    return captured[-1]


# Dims observed in the production dead-consumer set at the time the
# integrity check was added (commit 26fbf73f). Most are embed-time
# durables (markers, IO state flags, CLEAN_EMBED, CONST) that are
# populated at the embedding layer and read by attn-anchor / threshold
# ops without a corresponding compiler-registered ``writes`` declaration.
# Drives the regression contract: a NEW entry means a real silent
# leakage path surfaced; a MISSING entry is expected when an op grows a
# proper ``writes={...}`` declaration for the dim (in which case remove
# it here).
PRODUCTION_DEAD_CONSUMERS_BASELINE: frozenset = frozenset({
    "ACTIVE_OPCODE_PRTF",
    "CLEAN_EMBED_HI",
    "CLEAN_EMBED_LO",
    "CONST",
    "IO_IS_PRTF",
    "IO_IS_PUTCHAR",
    "IO_IS_READ",
    "IO_IS_TOOL_CALL",
    "IS_BYTE",
    "IS_MARK",
    "MARK_AX",
    "MARK_BP",
    "MARK_CS",
    "MARK_MEM",
    "MARK_PC",
    "MARK_SE",
    "MARK_SE_ONLY",
    "MARK_SP",
    "MARK_STACK0",
    "MARK_THINKING_END",
    "MARK_THINKING_START",
    "OPCODE_BASE",
})


@pytest.mark.timeout(600)
@pytest.mark.skipif(
    os.environ.get("C4_SKIP_DIM_INTEGRITY") == "1",
    reason="C4_SKIP_DIM_INTEGRITY=1 in environment",
)
def test_production_dead_consumers_match_baseline():
    """Production compile produces a dead-consumer set that is a
    subset of the baseline.

    A new dim in ``dead - baseline`` means a regression introduced a
    new silent read-but-never-written path — fail loudly. A dim in
    ``baseline - dead`` means an op started declaring the dim in
    ``writes`` (good); update the baseline to keep the floor tight.
    """
    dead = _collect_production_dead_consumers()
    new_dead = set(dead.keys()) - PRODUCTION_DEAD_CONSUMERS_BASELINE
    assert not new_dead, (
        f"New dead-consumer dim(s) appeared: {sorted(new_dead)}. "
        "Each indicates an op reads the dim but no op declares it in "
        "writes={...}. Either add the producer, or — if the read is "
        "intentionally embed-time / cross-step durable — add the dim "
        "to PRODUCTION_DEAD_CONSUMERS_BASELINE.\n"
        f"Sample consumers: {[ (d, dead[d][:3]) for d in sorted(new_dead) ]}"
    )
