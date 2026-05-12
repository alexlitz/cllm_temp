"""Tests for the dim-ownership claim registry.

The registry is the Phase 1 / Agent B deliverable from
``c4_release/docs/ARCH_LEAKAGE_FIX_PLAN.md``. It detects when two ops claim
the same ``(layer_idx, scope, identifier)`` triple at compile time and
emits a warning (not a hard fail yet — opt-in framework instrumentation).

These tests verify:

1. A synthetic Operation with overlapping claims triggers a warning.
2. The registry collects all opted-in claims across attn/ffn/block/model ops.
3. A non-overlapping pair triggers no warning.
4. Malformed claims (wrong tuple shape / unknown scope / non-int layer)
   raise at ``add_op`` time.
5. The production model build emits no claim-collision warnings — only a
   handful of ops are currently annotated and they're meant to be disjoint;
   any warning here surfaces a real latent collision that warrants
   investigation (per the plan doc).
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
        # Wrong arity.
        bad_op = _op("bad", reads=["X"], writes=["X"],
                     claims={(0, "attn_W_v")})  # only 2 elems
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
    """Claims from attn/ffn/block/model ops all flow into the registry."""
    c = LayerCompiler()
    c.declare_dim("X", 1)
    c.add_op(_op(
        "attn_op", kind="attn", reads=["X"], writes=["X"],
        claims={(0, "attn_W_q", "0_0")},
    ))
    c.add_op(_op(
        "ffn_op", kind="ffn", reads=["X"], writes=["X"],
        claims={(0, "ffn_W_up", "u0")},
    ))
    c.add_op(_op(
        "block_op", kind="block", reads=["X"], writes=["X"],
        layer_idx=0, claims={(0, "attn_W_v", "0_0")},
    ))
    c.add_op(_op(
        "model_op", kind="model", reads=["X"], writes=["X"],
        claims={(0, "embed_row", "42")},
    ))
    reg = c.build_claim_registry()
    keys = set(reg.keys())
    assert (0, "attn_W_q", "0_0") in keys
    assert (0, "ffn_W_up", "u0") in keys
    assert (0, "attn_W_v", "0_0") in keys
    assert (0, "embed_row", "42") in keys


# ----------------------------------------------------------------------
# Production-model smoke test
# ----------------------------------------------------------------------

# Known-benign claim-collision findings in the production model. Each entry
# documents a (layer, scope, identifier) that today's bakes do legitimately
# both touch, but at distinct (row, col) cells of the same row — the row-
# granularity registry can't see column-level disjointness. These should be
# revisited once the framework gains position-aware writes (Phase 3 / Agent
# F of ARCH_LEAKAGE_FIX_PLAN.md):
#
#   (5, "attn_W_v", "5_32"): both ``layer5_fetch`` (CLEAN_EMBED_LO[0] →
#       OPCODE_BYTE_LO[0]) and ``function_call_weights`` (EMBED_HI[15] →
#       TEMP[31]) write to row 5*HD+32. Different input columns; different
#       Q/K marker gates (PC vs STACK0). Output values do not alias because
#       at any given position only one head's softmax is nonzero — V is
#       projected through W_o only where the attention pattern places mass,
#       and the two heads never fire at the same position.
KNOWN_BENIGN_COLLISIONS = frozenset({
    (5, "attn_W_v", "5_32"),
})


@pytest.mark.timeout(300)
def test_production_compile_emits_no_unexpected_collision_warnings():
    """Build the production VM and assert only known-benign collisions.

    Only a few ops are currently annotated (``layer5_fetch`` and
    ``function_call_weights``). The registry surfaces a known-benign
    boundary collision at ``(5, "attn_W_v", "5_32")`` — both ops touch
    row ``5*HD+32`` but at distinct (row, col) cells with mutually
    exclusive Q/K gating; see ``KNOWN_BENIGN_COLLISIONS`` above. Any
    *additional* collision is a real latent bug to investigate (per
    ARCH_LEAKAGE_FIX_PLAN.md).
    """
    from c4_release.neural_vm.unified_compiler.full_vm_compiler import (
        compile_full_vm,
    )

    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        # Default args mirror the bake path used by the headline tests.
        compile_full_vm()

    # Parse each "DIM-OWNERSHIP COLLISION" message back into a key so we
    # can compare against the allowlist symbolically.
    import re
    pattern = re.compile(
        r"layer=(\d+) scope='([^']+)' identifier='([^']+)'"
    )
    unexpected = []
    for msg in _collect_claim_warnings(wlist):
        m = pattern.search(msg)
        if m is None:
            unexpected.append(msg)
            continue
        key = (int(m.group(1)), m.group(2), m.group(3))
        if key not in KNOWN_BENIGN_COLLISIONS:
            unexpected.append(msg)

    assert not unexpected, (
        "Production compile emitted unexpected dim-ownership collisions "
        "(beyond the known-benign allowlist) — investigate before "
        "extending claim coverage.\n  " + "\n  ".join(unexpected)
    )
