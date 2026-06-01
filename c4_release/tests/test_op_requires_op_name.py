"""B10 — ``Operation.requires`` op-name reference schema.

Covers the dynamic-scheduler migration B10 unit. ``requires["after"]`` and
``requires["same_layer_as"]`` accept op-name strings (or iterables of
strings) and produce dep edges that the analyzer and the layer compiler
honour.

Test surface:
  * Scheduler walks an explicit ``after`` op-name edge (B follows A even
    when no dim flows from A to B).
  * Iterable values are accepted.
  * ``same_layer_as`` enforces layer equality at compile time.
  * Unknown op-name references are surfaced as validation errors
    (analyzer-style hard error, not a silent no-op edge).
  * Self-references are rejected.
  * Pre-B10 residual-constraint string values (free-form keys) are still
    accepted and ignored by the scheduler.
  * ``analyze_scheduler.build_dep_graph`` adds the new edges.
"""

import importlib.util
import os

import pytest

from neural_vm.unified_compiler.layer_compiler import (
    LayerCompiler,
    Operation,
    REQUIRES_AFTER_KEY,
    REQUIRES_SAME_LAYER_AS_KEY,
    requires_after_ops,
    requires_same_layer_as_ops,
    validate_requires_op_refs,
)


_ANALYZE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tools",
    "analyze_scheduler.py",
)


def _load_analyze_scheduler():
    """Load ``tools/analyze_scheduler.py`` as a private module for testing.

    ``tools/`` is not a Python package, so this side-steps a real import
    while still exercising the production source.
    """
    spec = importlib.util.spec_from_file_location(
        "_test_analyze_scheduler", _ANALYZE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _noop(module, dims, S):
    return None


def _op(name, *, reads=(), writes=(), kind="ffn", requires=None, layer_idx=None):
    return Operation(
        name=name,
        reads=set(reads),
        writes=set(writes),
        kind=kind,
        bake_fn=_noop,
        requires=dict(requires) if requires else {},
        layer_idx=layer_idx,
    )


class TestRequiresHelpers:
    def test_after_single_string(self):
        op = _op("b", requires={REQUIRES_AFTER_KEY: "a"})
        assert requires_after_ops(op) == ("a",)
        assert requires_same_layer_as_ops(op) == ()

    def test_after_tuple_and_list(self):
        op_t = _op("b", requires={REQUIRES_AFTER_KEY: ("a", "x")})
        op_l = _op("b", requires={REQUIRES_AFTER_KEY: ["a", "x"]})
        assert requires_after_ops(op_t) == ("a", "x")
        assert requires_after_ops(op_l) == ("a", "x")

    def test_same_layer_as_single_and_list(self):
        op = _op("attach", requires={REQUIRES_SAME_LAYER_AS_KEY: "alu"})
        assert requires_same_layer_as_ops(op) == ("alu",)
        op2 = _op(
            "attach", requires={REQUIRES_SAME_LAYER_AS_KEY: ["alu", "mem"]}
        )
        assert requires_same_layer_as_ops(op2) == ("alu", "mem")

    def test_legacy_freeform_keys_ignored_by_op_name_helpers(self):
        # A pre-B10 residual-constraint key carries author docs only — the
        # scheduler must not interpret it as an op-name reference.
        op = _op(
            "legacy",
            requires={"OUTPUT_LO": "preserved across step", "after": "real"},
        )
        assert requires_after_ops(op) == ("real",)
        assert requires_same_layer_as_ops(op) == ()

    def test_non_string_value_raises(self):
        op = _op("b", requires={REQUIRES_AFTER_KEY: 17})
        with pytest.raises(TypeError):
            requires_after_ops(op)

    def test_iterable_with_non_string_member_raises(self):
        op = _op("b", requires={REQUIRES_AFTER_KEY: ("a", 17)})
        with pytest.raises(TypeError):
            requires_after_ops(op)

    def test_empty_string_filtered(self):
        op = _op("b", requires={REQUIRES_AFTER_KEY: ""})
        assert requires_after_ops(op) == ()


class TestValidateRequiresOpRefs:
    def test_clean_op_set(self):
        ops = [
            _op("a", writes=["X"]),
            _op("b", reads=["X"], requires={REQUIRES_AFTER_KEY: "a"}),
        ]
        assert validate_requires_op_refs(ops) == []

    def test_unknown_reference(self):
        ops = [
            _op("a"),
            _op("b", requires={REQUIRES_AFTER_KEY: "ghost"}),
        ]
        errors = validate_requires_op_refs(ops)
        assert len(errors) == 1
        assert "ghost" in errors[0]
        assert "b" in errors[0]

    def test_self_reference_rejected(self):
        ops = [_op("b", requires={REQUIRES_AFTER_KEY: "b"})]
        errors = validate_requires_op_refs(ops)
        assert any("references itself" in m for m in errors)

    def test_type_error_surfaced(self):
        ops = [_op("b", requires={REQUIRES_AFTER_KEY: 7})]
        errors = validate_requires_op_refs(ops)
        assert errors and "must be str" in errors[0]


class TestSchedulerWalksAfterEdge:
    def test_after_pushes_op_to_later_layer_without_dim_flow(self):
        # No dim flows from A to B — only an explicit requires["after"]
        # tells the scheduler to order them.
        c = LayerCompiler()
        c.declare_dim("M", 1)
        c.declare_dim("X", 4)
        c.declare_dim("Y", 4)
        c.add_op(_op("a", reads=["M"], writes=["X"], kind="ffn"))
        c.add_op(
            _op(
                "b",
                reads=["M"],
                writes=["Y"],
                kind="ffn",
                requires={REQUIRES_AFTER_KEY: "a"},
            )
        )
        layout = c.compile()
        layer_a = next(
            i
            for i, ops_l in enumerate(layout.ops_per_layer)
            if any(o.name == "a" for o in ops_l)
        )
        layer_b = next(
            i
            for i, ops_l in enumerate(layout.ops_per_layer)
            if any(o.name == "b" for o in ops_l)
        )
        assert layer_b > layer_a

    def test_after_accepts_iterable_of_refs(self):
        c = LayerCompiler()
        c.declare_dim("M", 1)
        c.declare_dim("X", 4)
        c.declare_dim("Y", 4)
        c.declare_dim("Z", 4)
        c.add_op(_op("a", reads=["M"], writes=["X"], kind="ffn"))
        c.add_op(_op("a2", reads=["M"], writes=["Y"], kind="ffn"))
        c.add_op(
            _op(
                "b",
                reads=["M"],
                writes=["Z"],
                kind="ffn",
                requires={REQUIRES_AFTER_KEY: ("a", "a2")},
            )
        )
        layout = c.compile()

        def layer_of(name):
            return next(
                i
                for i, ops_l in enumerate(layout.ops_per_layer)
                if any(o.name == name for o in ops_l)
            )

        assert layer_of("b") > layer_of("a")
        assert layer_of("b") > layer_of("a2")


class TestSameLayerAsEquality:
    def test_same_layer_as_pins_to_same_layer(self):
        # Postop attach pattern: the postop has NO dim dep on the ALU op
        # (its work is a bake-time attach inside the same TransformerBlock),
        # but it must live at the same layer for the bake to wire up.
        c = LayerCompiler()
        c.declare_dim("M", 1)
        c.declare_dim("X", 4)
        c.declare_dim("Y", 4)
        c.add_op(_op("alu", reads=["M"], writes=["X"], kind="attn"))
        c.add_op(
            _op(
                "postop",
                reads=["M"],
                writes=["Y"],
                kind="ffn",
                requires={REQUIRES_SAME_LAYER_AS_KEY: "alu"},
            )
        )
        layout = c.compile()

        def layer_of(name):
            return next(
                i
                for i, ops_l in enumerate(layout.ops_per_layer)
                if any(o.name == name for o in ops_l)
            )

        assert layer_of("postop") == layer_of("alu")

    def test_same_layer_as_conflict_with_dim_flow_raises(self):
        # If a dim dep would force a strictly later layer than the
        # same_layer_as target, the compiler must refuse silently producing
        # a wrong layout — it raises ValueError so the author re-models.
        c = LayerCompiler()
        c.declare_dim("M", 1)
        c.declare_dim("X", 4)
        c.declare_dim("Y", 4)
        c.add_op(_op("alu", reads=["M"], writes=["X"], kind="attn"))
        c.add_op(
            _op(
                "postop",
                reads=["X"],  # creates a dim-dep that forces later layer
                writes=["Y"],
                kind="ffn",
                requires={REQUIRES_SAME_LAYER_AS_KEY: "alu"},
            )
        )
        with pytest.raises(ValueError, match="same_layer_as"):
            c.compile()


class TestMissingOpNameClean:
    def test_validate_returns_clean_message(self):
        # Single helper call surfaces a structured error string the caller
        # can dump to stderr / fail CI on — no silent no-op edges.
        ops = [
            _op("a"),
            _op(
                "b",
                requires={
                    REQUIRES_AFTER_KEY: "a",
                    REQUIRES_SAME_LAYER_AS_KEY: "does_not_exist",
                },
            ),
        ]
        errors = validate_requires_op_refs(ops)
        assert any("does_not_exist" in m for m in errors)
        # The valid 'a' reference must NOT appear in error output.
        assert not any("=a" in m and "does_not_exist" not in m for m in errors)


class TestAnalyzerIntegration:
    def test_build_dep_graph_walks_after_edge(self):
        analyze_scheduler = _load_analyze_scheduler()

        a = _op("a", reads=["M"], writes=["X"])
        b = _op(
            "b",
            reads=["M"],
            writes=["Y"],
            requires={REQUIRES_AFTER_KEY: "a"},
        )
        in_e, out_e, reasons = analyze_scheduler.build_dep_graph([a, b])
        # Edge a -> b must exist from the requires["after"] reference even
        # though no dim flows from a to b.
        assert "a" in in_e["b"]
        assert "b" in out_e["a"]
        # The reason should mention requires[after]=a.
        labels = reasons.get(("a", "b"), [])
        assert any("requires[after]=a" in r for r in labels), labels

    def test_build_dep_graph_walks_same_layer_as_edge(self):
        analyze_scheduler = _load_analyze_scheduler()

        a = _op("a", reads=["M"], writes=["X"])
        b = _op(
            "b",
            reads=["M"],
            writes=["Y"],
            requires={REQUIRES_SAME_LAYER_AS_KEY: "a"},
        )
        in_e, _out_e, reasons = analyze_scheduler.build_dep_graph([a, b])
        assert "a" in in_e["b"]
        labels = reasons.get(("a", "b"), [])
        assert any("requires[same_layer_as]=a" in r for r in labels), labels
