"""Tests for the residual-dim staleness invariant analyzer.

Phase 3 / Agent G of ``c4_release/docs/ARCH_LEAKAGE_FIX_PLAN.md``. Each
``Operation`` may declare:

  ``produces``       : Dict[dim_name, register_name]
  ``consumes_fresh`` : Dict[dim_name, register_name]

The analyzer scans every op that declares ``consumes_fresh`` and warns when
no earlier-phase op in the same step ``produces`` the matching
``(dim, register)`` -- catching today's AX_CARRY-stale bug (commit
``3d1b700``) at compile time instead of via failing add/sub smoke tests.

These tests verify:

1. A synthetic op pair (producer + consumer at correct phases) compiles
   without a staleness warning.
2. Removing the producer surfaces a ``STALENESS VIOLATION`` warning.
3. Cross-step-only consumers (no ``consumes_fresh`` declaration) don't
   trigger warnings even when no in-step producer exists -- mirrors the
   L3 head 1 AX_CARRY chain that legitimately relies on prev-step
   EMBED_LO/HI.
4. Producer with higher phase than consumer does NOT count as an in-step
   producer (catches the "writer fires too late" pattern).
5. Malformed ``produces`` / ``consumes_fresh`` annotations raise at
   ``add_op`` time.
6. The production model build emits no unexpected staleness warnings.
7. **Regression**: removing the ``layer8_head6_ax_carry_refresh`` op
   from the production compile surfaces a staleness violation for the
   L8 ALU's ``consumes_fresh AX_CARRY_LO``.
"""

import warnings

import pytest

from c4_release.neural_vm.unified_compiler.layer_compiler import (
    LayerCompiler,
    Operation,
)


def _noop(module, dims, S):
    return None


def _op(name, kind="attn", reads=(), writes=(), produces=None,
        consumes_fresh=None, phase=None, layer_idx=None):
    return Operation(
        name=name,
        reads=set(reads),
        writes=set(writes),
        kind=kind,
        bake_fn=_noop,
        phase=phase,
        layer_idx=layer_idx,
        produces=dict(produces or {}),
        consumes_fresh=dict(consumes_fresh or {}),
    )


def _collect_staleness_warnings(records):
    return [
        str(w.message) for w in records
        if "STALENESS VIOLATION" in str(w.message)
    ]


# ----------------------------------------------------------------------
# Validation tests (at add_op time)
# ----------------------------------------------------------------------


class TestValidation:
    def test_produces_must_be_dict(self):
        c = LayerCompiler()
        c.declare_dim("X", 1)
        op = _op("bad", reads=["X"], writes=["X"])
        op.produces = [("X", "reg")]  # type: ignore
        with pytest.raises(ValueError, match="produces must be a dict"):
            c.add_op(op)

    def test_consumes_fresh_must_be_dict(self):
        c = LayerCompiler()
        c.declare_dim("X", 1)
        op = _op("bad", reads=["X"], writes=["X"])
        op.consumes_fresh = [("X", "reg")]  # type: ignore
        with pytest.raises(ValueError, match="consumes_fresh must be a dict"):
            c.add_op(op)

    def test_produces_undeclared_dim_rejected(self):
        c = LayerCompiler()
        c.declare_dim("X", 1)
        bad = _op("bad", reads=["X"], writes=["X"],
                  produces={"NOT_DECLARED": "reg"})
        with pytest.raises(ValueError, match="references undeclared dim"):
            c.add_op(bad)

    def test_consumes_fresh_undeclared_dim_rejected(self):
        c = LayerCompiler()
        c.declare_dim("X", 1)
        bad = _op("bad", reads=["X"], writes=["X"],
                  consumes_fresh={"NOT_DECLARED": "reg"})
        with pytest.raises(ValueError, match="references undeclared dim"):
            c.add_op(bad)

    def test_register_must_be_string(self):
        c = LayerCompiler()
        c.declare_dim("X", 1)
        bad = _op("bad", reads=["X"], writes=["X"],
                  produces={"X": 42})
        with pytest.raises(ValueError, match="register must be str"):
            c.add_op(bad)


# ----------------------------------------------------------------------
# Synthetic analyzer tests
# ----------------------------------------------------------------------


def _build_compiler_with_ops(*ops):
    """Return a compiler with each Op's reads/writes mapped to distinct dims."""
    c = LayerCompiler()
    c.declare_dim("IN", 1)
    c.declare_dim("AX_CARRY_LO", 16)
    for i, op in enumerate(ops):
        c.add_op(op)
    return c


class TestSyntheticAnalyzer:
    def test_producer_before_consumer_no_warning(self):
        """Producer at phase 8.05 + consumer at phase 8.2 = OK."""
        producer = _op(
            "ax_carry_refresh", kind="block", layer_idx=8, phase=8.05,
            reads=["IN"], writes=["AX_CARRY_LO"],
            produces={"AX_CARRY_LO": "AX_byte0"},
        )
        consumer = _op(
            "alu_consumer", kind="block", layer_idx=8, phase=8.2,
            reads=["AX_CARRY_LO"], writes=["IN"],
            consumes_fresh={"AX_CARRY_LO": "AX_byte0"},
        )
        c = _build_compiler_with_ops(producer, consumer)
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            c.compile()
        assert _collect_staleness_warnings(wlist) == []

    def test_missing_producer_warns(self):
        """Consumer with no in-step producer surfaces a staleness warning."""
        consumer = _op(
            "alu_consumer", kind="block", layer_idx=8, phase=8.2,
            reads=["AX_CARRY_LO"], writes=["IN"],
            consumes_fresh={"AX_CARRY_LO": "AX_byte0"},
        )
        c = _build_compiler_with_ops(consumer)
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            c.compile()
        msgs = _collect_staleness_warnings(wlist)
        assert len(msgs) == 1, f"expected 1 warning, got {msgs!r}"
        assert "alu_consumer" in msgs[0]
        assert "AX_CARRY_LO" in msgs[0]
        assert "AX_byte0" in msgs[0]

    def test_wrong_register_does_not_satisfy(self):
        """Producer with different register name does not satisfy the consumer."""
        wrong_producer = _op(
            "wrong_register", kind="block", layer_idx=8, phase=8.05,
            reads=["IN"], writes=["AX_CARRY_LO"],
            produces={"AX_CARRY_LO": "different_register"},
        )
        consumer = _op(
            "alu_consumer", kind="block", layer_idx=8, phase=8.2,
            reads=["AX_CARRY_LO"], writes=["IN"],
            consumes_fresh={"AX_CARRY_LO": "AX_byte0"},
        )
        c = _build_compiler_with_ops(wrong_producer, consumer)
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            c.compile()
        msgs = _collect_staleness_warnings(wlist)
        assert len(msgs) == 1
        assert "AX_byte0" in msgs[0]

    def test_producer_after_consumer_warns(self):
        """A producer that fires AFTER the consumer doesn't count."""
        consumer = _op(
            "alu_consumer", kind="block", layer_idx=8, phase=8.2,
            reads=["AX_CARRY_LO"], writes=["IN"],
            consumes_fresh={"AX_CARRY_LO": "AX_byte0"},
        )
        late_producer = _op(
            "late_producer", kind="block", layer_idx=8, phase=9.0,
            reads=["IN"], writes=["AX_CARRY_LO"],
            produces={"AX_CARRY_LO": "AX_byte0"},
        )
        c = _build_compiler_with_ops(consumer, late_producer)
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            c.compile()
        msgs = _collect_staleness_warnings(wlist)
        assert len(msgs) == 1
        assert "alu_consumer" in msgs[0]

    def test_cross_step_consumer_no_warning(self):
        """An op that reads cross-step data (no consumes_fresh) doesn't warn.

        Mirrors the L3 head 1 AX_CARRY relay: it reads EMBED_LO/HI from
        the prev step's AX byte 0 token, which is intentionally cross-step
        and shouldn't claim ``consumes_fresh``.
        """
        l3_head1 = _op(
            "l3_head1_cross_step", kind="block", layer_idx=3, phase=3,
            reads=["IN"], writes=["AX_CARRY_LO"],
            # No ``consumes_fresh`` declaration -- only cross-step.
        )
        c = _build_compiler_with_ops(l3_head1)
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            c.compile()
        assert _collect_staleness_warnings(wlist) == []

    def test_self_producer_satisfies_invariant(self):
        """An op that both produces and consumes_fresh the same dim+reg is OK.

        Catches the edge case where a single op stages and consumes its
        own fresh value (e.g., a composite block-op that internally
        refreshes the dim then reads it).
        """
        self_op = _op(
            "self_producer", kind="block", layer_idx=8, phase=8.0,
            reads=["AX_CARRY_LO"], writes=["AX_CARRY_LO"],
            produces={"AX_CARRY_LO": "AX_byte0"},
            consumes_fresh={"AX_CARRY_LO": "AX_byte0"},
        )
        c = _build_compiler_with_ops(self_op)
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            c.compile()
        assert _collect_staleness_warnings(wlist) == []

    def test_same_phase_producer_satisfies_invariant(self):
        """Producer + consumer at the same phase: producer counts.

        Same-phase ties can fire in either order, but for the staleness
        analyzer we treat ``producer.phase <= consumer.phase`` as
        in-step. Tightly-paired ops sharing a phase shouldn't trigger
        spurious warnings.
        """
        producer = _op(
            "producer", kind="block", layer_idx=8, phase=8.05,
            reads=["IN"], writes=["AX_CARRY_LO"],
            produces={"AX_CARRY_LO": "AX_byte0"},
        )
        consumer = _op(
            "consumer", kind="block", layer_idx=8, phase=8.05,
            reads=["AX_CARRY_LO"], writes=["IN"],
            consumes_fresh={"AX_CARRY_LO": "AX_byte0"},
        )
        c = _build_compiler_with_ops(producer, consumer)
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            c.compile()
        assert _collect_staleness_warnings(wlist) == []


# ----------------------------------------------------------------------
# Registry inspection API
# ----------------------------------------------------------------------


def test_build_staleness_registry_aggregates_all_op_kinds():
    """produces / consumes_fresh across attn/ffn/block/model all flow in."""
    c = LayerCompiler()
    c.declare_dim("X", 1)
    c.declare_dim("Y", 1)
    c.add_op(_op(
        "attn_producer", kind="attn", reads=["X"], writes=["Y"],
        produces={"Y": "r1"},
    ))
    c.add_op(_op(
        "ffn_consumer", kind="ffn", reads=["Y"], writes=["X"],
        consumes_fresh={"Y": "r1"},
    ))
    producers, consumers = c.build_staleness_registry()
    assert ("Y", "r1") in producers
    assert ("Y", "r1") in consumers
    assert producers[("Y", "r1")][0][0] == "attn_producer"
    assert consumers[("Y", "r1")][0][0] == "ffn_consumer"


# ----------------------------------------------------------------------
# Production-model tests
# ----------------------------------------------------------------------


@pytest.mark.timeout(300)
def test_production_compile_emits_no_unexpected_staleness_warnings():
    """The production VM should compile clean for the annotated ops.

    Only a handful of ops are currently annotated with ``produces`` /
    ``consumes_fresh`` (the L8 ALU + L8 head 6 refresh + L8 multibyte
    routing). They form a consistent producer/consumer set today; any
    additional staleness warning here surfaces a real latent leakage
    bug that warrants investigation (per
    ``docs/ARCH_LEAKAGE_FIX_PLAN.md`` Phase 3 / Agent G).
    """
    from c4_release.neural_vm.unified_compiler.full_vm_compiler import (
        compile_full_vm,
    )

    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        compile_full_vm()
    msgs = _collect_staleness_warnings(wlist)
    assert msgs == [], (
        "Production compile emitted unexpected staleness warnings -- "
        "investigate the producer/consumer chain.\n  " + "\n  ".join(msgs)
    )


@pytest.mark.timeout(300)
def test_removing_l8_head6_surfaces_ax_carry_staleness():
    """Regression: removing the L8 head 6 op surfaces the stale-AX_CARRY bug.

    Builds the production op set, filters out
    ``layer8_head6_ax_carry_refresh``, and asserts the staleness analyzer
    warns that the L8 ALU's ``consumes_fresh AX_CARRY_LO`` has no
    in-step producer. This is the canonical bug-detection proof point
    for Phase 3 / Agent G of ARCH_LEAKAGE_FIX_PLAN.md (commit 3d1b700).
    """
    from c4_release.neural_vm.unified_compiler.layer_compiler import (
        LayerCompiler,
    )
    from c4_release.neural_vm.unified_compiler.migrated_ops import (
        all_core_ops,
        declare_setdim_compat_dims,
    )

    compiler = LayerCompiler()
    declare_setdim_compat_dims(compiler, pin_io_only=True)
    for op in all_core_ops():
        if op.name == "layer8_head6_ax_carry_refresh":
            continue
        compiler.add_op(op)

    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        compiler._detect_staleness_violations()
    msgs = _collect_staleness_warnings(wlist)
    matching = [
        m for m in msgs
        if "layer8_alu" in m and "AX_CARRY_LO" in m and "AX_byte0" in m
    ]
    assert matching, (
        "Expected the analyzer to warn that layer8_alu's "
        "consumes_fresh AX_CARRY_LO has no in-step producer after "
        "removing layer8_head6_ax_carry_refresh. Got messages:\n  "
        + "\n  ".join(msgs)
    )
