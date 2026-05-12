"""Tests for the dim-ownership claim registry.

The registry is the Phase 1 / Agent B deliverable from
``c4_release/docs/ARCH_LEAKAGE_FIX_PLAN.md`` (extended with column-
granularity per the Phase 3 / Agent F follow-up). It detects when two
ops claim the same ``(layer_idx, scope, identifier, column)`` 4-tuple
at compile time and emits a warning (not a hard fail yet — opt-in
framework instrumentation).

These tests verify:

1. A synthetic Operation with overlapping claims triggers a warning.
2. The registry collects all opted-in claims across attn/ffn/block/model ops.
3. A non-overlapping pair triggers no warning.
4. Same row + different column does NOT warn (column-granular registry).
5. Same row + same column DOES warn.
6. Malformed claims (wrong tuple shape / unknown scope / non-int layer /
   non-str column / column on ``embed_row``) raise at ``add_op`` time.
7. 3-tuple legacy claims are auto-promoted to ``column=None`` and still
   collide with each other / with explicit ``column=None`` 4-tuples.
8. The production model build emits no claim-collision warnings — only a
   handful of ops are currently annotated and they're meant to be disjoint;
   any warning here surfaces a real latent collision that warrants
   investigation (per the plan doc). The historical row-level false
   positive at ``(5, "attn_W_v", "5_32")`` is now resolved by column
   granularity — no allowlist needed.
"""

import warnings

import pytest

from c4_release.neural_vm.unified_compiler.layer_compiler import (
    ALLOWED_CLAIM_SCOPES,
    LayerCompiler,
    Operation,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _noop(module, dims, S):
    return None


def _op(name, kind="attn", reads=(), writes=(), claims=None,
        layer_idx=None, phase=None):
    return Operation(
        name=name,
        reads=set(reads),
        writes=set(writes),
        kind=kind,
        bake_fn=_noop,
        layer_idx=layer_idx,
        phase=phase,
        claims=set(claims or ()),
    )


def _make_compiler_with_disjoint_writers(*ops):
    """Helper: build a LayerCompiler where each op writes a distinct dim.

    Each Operation has a unique read/write dim auto-allocated. This keeps
    the dep graph acyclic so we can focus tests on claim-collision
    behavior independent of layer assignment.
    """
    c = LayerCompiler()
    c.declare_dim("IN", 1)
    for i, op in enumerate(ops):
        c.declare_dim(f"OUT_{i}", 1)
        # Reach into the dataclass to swap in our per-op reads/writes
        # without exposing them to the caller.
        op.reads = {"IN"}
        op.writes = {f"OUT_{i}"}
        c.add_op(op)
    return c


def _collect_claim_warnings(records):
    return [
        str(w.message) for w in records
        if "DIM-OWNERSHIP COLLISION" in str(w.message)
    ]


# ----------------------------------------------------------------------
# Allowed-scope catalogue smoke test
# ----------------------------------------------------------------------

def test_allowed_scopes_present():
    """All scopes documented in the design doc are present."""
    expected = {
        "attn_W_v", "attn_W_k", "attn_W_q", "attn_W_o",
        "ffn_W_up", "ffn_W_down", "ffn_W_gate",
        "embed_row",
    }
    assert expected == set(ALLOWED_CLAIM_SCOPES)


# ----------------------------------------------------------------------
# Claim-validation tests (at add_op time, not compile time)
# ----------------------------------------------------------------------

class TestClaimValidation:
    def test_bad_scope_rejected(self):
        c = LayerCompiler()
        c.declare_dim("X", 1)
        bad_op = _op(
            "bad",
            kind="attn",
            reads=["X"],
            writes=["X"],
            claims={(0, "bogus_scope", "0_0")},
        )
        with pytest.raises(ValueError, match="scope 'bogus_scope' not in"):
            c.add_op(bad_op)

    def test_malformed_tuple_rejected(self):
        c = LayerCompiler()
        c.declare_dim("X", 1)
        # Wrong arity (only 2 elems — not 3-tuple or 4-tuple).
        bad_op = _op("bad", reads=["X"], writes=["X"],
                     claims={(0, "attn_W_v")})
        with pytest.raises(ValueError, match="malformed claim"):
            c.add_op(bad_op)

    def test_too_many_elements_rejected(self):
        c = LayerCompiler()
        c.declare_dim("X", 1)
        # Wrong arity (5 elems — not 3-tuple or 4-tuple).
        bad_op = _op(
            "bad", reads=["X"], writes=["X"],
            claims={(0, "attn_W_v", "0_0", "EMBED_LO+0", "extra")},
        )
        with pytest.raises(ValueError, match="malformed claim"):
            c.add_op(bad_op)

    def test_non_int_layer_rejected(self):
        c = LayerCompiler()
        c.declare_dim("X", 1)
        bad_op = _op("bad", reads=["X"], writes=["X"],
                     claims={("5", "attn_W_v", "0_0")})
        with pytest.raises(ValueError, match="layer_idx must be int"):
            c.add_op(bad_op)

    def test_non_str_identifier_rejected(self):
        c = LayerCompiler()
        c.declare_dim("X", 1)
        bad_op = _op("bad", reads=["X"], writes=["X"],
                     claims={(0, "attn_W_v", 5)})
        with pytest.raises(ValueError, match="identifier must be str"):
            c.add_op(bad_op)

    def test_non_str_column_rejected(self):
        c = LayerCompiler()
        c.declare_dim("X", 1)
        bad_op = _op(
            "bad", reads=["X"], writes=["X"],
            claims={(0, "attn_W_v", "0_0", 7)},
        )
        with pytest.raises(ValueError, match="column must be str or None"):
            c.add_op(bad_op)

    def test_embed_row_with_column_rejected(self):
        """``embed_row`` is row-granular; column must be None."""
        c = LayerCompiler()
        c.declare_dim("X", 1)
        bad_op = _op(
            "bad", reads=["X"], writes=["X"],
            claims={(0, "embed_row", "42", "TOK_FIELD+0")},
        )
        with pytest.raises(ValueError,
                           match="scope 'embed_row' must have column=None"):
            c.add_op(bad_op)

    def test_legacy_3tuple_promoted_to_4tuple(self):
        """A 3-tuple claim is auto-promoted to 4-tuple with column=None."""
        c = LayerCompiler()
        c.declare_dim("X", 1)
        c.declare_dim("OUT", 1)
        op = _op("legacy", reads=["X"], writes=["OUT"],
                 claims={(0, "attn_W_v", "0_0")})
        c.add_op(op)
        # add_op rewrites op.claims to 4-tuple form in-place.
        assert (0, "attn_W_v", "0_0", None) in op.claims
        # Registry sees the promoted form.
        reg = c.build_claim_registry()
        assert (0, "attn_W_v", "0_0", None) in reg


# ----------------------------------------------------------------------
# Column-granularity tests (the headline feature)
# ----------------------------------------------------------------------

class TestColumnGranularity:
    """Same-row, different-column claims must not collide."""

    def test_same_row_different_column_no_warn(self):
        """The motivating L5 head 5 row 32 case: column-disjoint writes."""
        a = _op("layer5_fetch_like",
                claims={(5, "attn_W_v", "5_32", "CLEAN_EMBED_LO+0")})
        b = _op("function_call_weights_like",
                claims={(5, "attn_W_v", "5_32", "EMBED_HI+15")})
        c = _make_compiler_with_disjoint_writers(a, b)
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            c.compile()
        assert _collect_claim_warnings(wlist) == [], (
            "column-disjoint same-row writes must not collide"
        )

    def test_same_row_same_column_warns(self):
        """Two ops writing identical (row, column) DO collide."""
        a = _op("writer_a",
                claims={(5, "attn_W_v", "6_5", "EMBED_LO+4")})
        b = _op("writer_b",
                claims={(5, "attn_W_v", "6_5", "EMBED_LO+4")})
        c = _make_compiler_with_disjoint_writers(a, b)
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            c.compile()
        msgs = _collect_claim_warnings(wlist)
        assert len(msgs) == 1, f"expected 1 warning, got {msgs!r}"
        assert "column='EMBED_LO+4'" in msgs[0]
        assert "writer_a" in msgs[0] and "writer_b" in msgs[0]

    def test_legacy_3tuple_collides_with_legacy_3tuple(self):
        """Two row-only legacy claims still collide (back-compat)."""
        a = _op("legacy_a", claims={(5, "attn_W_v", "6_5")})
        b = _op("legacy_b", claims={(5, "attn_W_v", "6_5")})
        c = _make_compiler_with_disjoint_writers(a, b)
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            c.compile()
        msgs = _collect_claim_warnings(wlist)
        assert len(msgs) == 1, f"expected 1 warning, got {msgs!r}"

    def test_legacy_3tuple_and_4tuple_with_none_column_collide(self):
        """3-tuple promotes to column=None; matches explicit None 4-tuple."""
        a = _op("legacy", claims={(5, "attn_W_v", "6_5")})
        b = _op("explicit_none",
                claims={(5, "attn_W_v", "6_5", None)})
        c = _make_compiler_with_disjoint_writers(a, b)
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            c.compile()
        msgs = _collect_claim_warnings(wlist)
        assert len(msgs) == 1, (
            "3-tuple and 4-tuple-with-None must collide as equivalent"
        )

    def test_legacy_3tuple_does_not_collide_with_explicit_column(self):
        """A legacy 3-tuple (column=None) and a 4-tuple with a real
        column string sit at different registry keys: column=None is a
        literal value, not a wildcard. This lets partial migration of
        peer ops to column granularity proceed without spurious warnings."""
        a = _op("legacy_row_only", claims={(5, "attn_W_v", "6_5")})
        b = _op("annotated",
                claims={(5, "attn_W_v", "6_5", "EMBED_LO+4")})
        c = _make_compiler_with_disjoint_writers(a, b)
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            c.compile()
        assert _collect_claim_warnings(wlist) == []


# ----------------------------------------------------------------------
# Synthetic collision tests
# ----------------------------------------------------------------------

class TestSyntheticCollision:
    def test_overlapping_claims_warn(self):
        """Two ops that claim the same (layer, scope, identifier) warn."""
        a = _op("writer_a", claims={(5, "attn_W_v", "6_5")})
        b = _op("writer_b", claims={(5, "attn_W_v", "6_5")})
        c = _make_compiler_with_disjoint_writers(a, b)
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            c.compile()
        msgs = _collect_claim_warnings(wlist)
        assert len(msgs) == 1, f"expected 1 warning, got {msgs!r}"
        assert "layer=5" in msgs[0]
        assert "scope='attn_W_v'" in msgs[0]
        assert "identifier='6_5'" in msgs[0]
        assert "writer_a" in msgs[0]
        assert "writer_b" in msgs[0]

    def test_disjoint_claims_no_warn(self):
        a = _op("writer_a", claims={(5, "attn_W_v", "5_1")})
        b = _op("writer_b", claims={(5, "attn_W_v", "5_2")})
        c = _make_compiler_with_disjoint_writers(a, b)
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            c.compile()
        assert _collect_claim_warnings(wlist) == []

    def test_different_layers_no_warn(self):
        """Same scope+id but different layer is not a collision."""
        a = _op("writer_a", claims={(5, "attn_W_v", "6_5")})
        b = _op("writer_b", claims={(6, "attn_W_v", "6_5")})
        c = _make_compiler_with_disjoint_writers(a, b)
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            c.compile()
        assert _collect_claim_warnings(wlist) == []

    def test_different_scopes_no_warn(self):
        """Same layer/id but different scope is not a collision."""
        a = _op("writer_a", claims={(5, "attn_W_v", "6_5")})
        b = _op("writer_b", claims={(5, "attn_W_q", "6_5")})
        c = _make_compiler_with_disjoint_writers(a, b)
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            c.compile()
        assert _collect_claim_warnings(wlist) == []

    def test_three_way_collision_single_warning(self):
        """A 3-op collision produces exactly one warning naming all 3."""
        ops = [_op(n, claims={(7, "ffn_W_up", "unit_42")})
               for n in ("alpha", "beta", "gamma")]
        c = _make_compiler_with_disjoint_writers(*ops)
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            c.compile()
        msgs = _collect_claim_warnings(wlist)
        assert len(msgs) == 1
        assert all(n in msgs[0] for n in ("alpha", "beta", "gamma"))
        assert "claimed by 3 ops" in msgs[0]

    def test_historical_l5_head6_collision_detected(self):
        """The exact historical L5 head 6 collision (c1a5398) is caught.

        Simulates the pre-fix state: both the deprecated `_set_layer5_fetch`
        head 6 V relay and `_set_function_call_weights`' ENT BP→TEMP relay
        write to row 6*HD+5 of attn5.W_v.
        """
        # Deprecated head 6 V slot 5 (one of 1..16 the deleted code wrote);
        # ENT relay writes V slots 1..32 of head 6 — slot 5 is in range.
        a = _op("layer5_fetch_legacy", claims={(5, "attn_W_v", "6_5")})
        b = _op("function_call_weights_legacy",
                claims={(5, "attn_W_v", "6_5")})
        c = _make_compiler_with_disjoint_writers(a, b)
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            c.compile()
        msgs = _collect_claim_warnings(wlist)
        assert any("6_5" in m and "layer=5" in m for m in msgs), \
            f"historical collision not detected; messages: {msgs!r}"


# ----------------------------------------------------------------------
# build_claim_registry API smoke test
# ----------------------------------------------------------------------

def test_build_claim_registry_aggregates_all_op_kinds():
    """Claims from attn/ffn/block/model ops all flow into the registry.

    All entries are 4-tuples: 3-tuple claims are promoted to
    ``(layer, scope, identifier, None)`` at ``add_op`` time.
    """
    c = LayerCompiler()
    c.declare_dim("X", 1)
    c.add_op(_op(
        "attn_op", kind="attn", reads=["X"], writes=["X"],
        claims={(0, "attn_W_q", "0_0")},  # legacy 3-tuple
    ))
    c.add_op(_op(
        "ffn_op", kind="ffn", reads=["X"], writes=["X"],
        claims={(0, "ffn_W_up", "u0", "X+0")},  # 4-tuple
    ))
    c.add_op(_op(
        "block_op", kind="block", reads=["X"], writes=["X"],
        layer_idx=0,
        claims={(0, "attn_W_v", "0_0", "X+0")},
    ))
    c.add_op(_op(
        "model_op", kind="model", reads=["X"], writes=["X"],
        claims={(0, "embed_row", "42")},  # embed_row: column must be None
    ))
    reg = c.build_claim_registry()
    keys = set(reg.keys())
    # 4-tuple form everywhere.
    assert (0, "attn_W_q", "0_0", None) in keys
    assert (0, "ffn_W_up", "u0", "X+0") in keys
    assert (0, "attn_W_v", "0_0", "X+0") in keys
    assert (0, "embed_row", "42", None) in keys


def test_4tuple_validation_smoke():
    """Smoke test: 4-tuple validation accepts well-formed shapes."""
    c = LayerCompiler()
    c.declare_dim("IN", 1)
    c.declare_dim("OUT", 1)
    # All these should validate cleanly.
    for claim in [
        (0, "attn_W_v", "0_0", "EMBED_LO+0"),
        (0, "attn_W_q", "1_5", None),
        (0, "ffn_W_up", "1700", "OPCODE_BYTE_LO+0"),
        (0, "embed_row", "42", None),
        (0, "attn_W_o", "7_31", "TEMP+15"),
    ]:
        op = _op(f"op_{hash(claim)}", reads=["IN"], writes=["OUT"],
                 claims={claim})
        # Need a fresh compiler since op names overlap with prior iterations
        cc = LayerCompiler()
        cc.declare_dim("IN", 1)
        cc.declare_dim("OUT", 1)
        cc.add_op(op)


# ----------------------------------------------------------------------
# Production-model smoke test
# ----------------------------------------------------------------------

@pytest.mark.timeout(300)
def test_production_compile_emits_no_collision_warnings():
    """Build the production VM and assert zero claim-collision warnings.

    Only a few ops are currently annotated (``layer5_fetch`` and
    ``function_call_weights``). With column-granularity claims, the
    historical row-level false positive at
    ``(5, "attn_W_v", "5_32")`` — where ``layer5_fetch`` writes
    ``CLEAN_EMBED_LO[0]`` and ``function_call_weights`` writes
    ``EMBED_HI[15]`` to the same row but disjoint columns — is no
    longer surfaced. The ``KNOWN_BENIGN_COLLISIONS`` allowlist has
    been retired. Any collision warning here is a real latent bug to
    investigate (per ARCH_LEAKAGE_FIX_PLAN.md).
    """
    from c4_release.neural_vm.unified_compiler.full_vm_compiler import (
        compile_full_vm,
    )

    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        # Default args mirror the bake path used by the headline tests.
        compile_full_vm()

    msgs = _collect_claim_warnings(wlist)
    assert msgs == [], (
        "Production compile emitted dim-ownership collision warnings "
        "(column-granular registry should resolve all known-benign "
        "row-level false positives).\n  " + "\n  ".join(msgs)
    )
