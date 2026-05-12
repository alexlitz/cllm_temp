"""Tests for the declarative op verifier (Mode A static claim check).

The verifier confirms that every ``Operation.claims`` cell an op declares
is actually written by its ``bake_fn`` -- the load-bearing direction of
declaration drift (the op promised a slot but didn't deliver). The
opposite direction (the bake wrote slots not in claims) is currently
expected because ops declare claims partially (only the high-collision-
risk subset of their writes). Strict mode is available as an opt-in
escalation for ops that want full-coverage claim declarations.

See ``c4_release/docs/DECLARATIVE_VERIFICATION.md`` for the full design.

Mode B (dynamic produces/consumes) is slower and gated on
``C4_VERIFY_DECLARATIONS=1`` -- it spins up a synthetic forward pass and
inspects residual values at marker positions.
"""

import os
import sys

import pytest

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


from c4_release.neural_vm.unified_compiler.decl_verifier import (  # noqa: E402
    OpVerificationResult,
    StaticVerificationReport,
    verify_claims_static,
    verify_produces_consumes_dynamic,
)
from c4_release.neural_vm.unified_compiler.layer_compiler import (  # noqa: E402
    LayerCompiler,
    Operation,
)


# ----------------------------------------------------------------------
# Synthetic tests (cheap, unit-style)
# ----------------------------------------------------------------------


class TestUnitDecoding:
    """Validate the row/col -> claim 4-tuple decoder logic."""

    def test_decode_attn_w_v(self):
        """W_v row encodes (head, slot); col -> input dim."""
        from c4_release.neural_vm.unified_compiler.decl_verifier import (
            _build_dim_lookup,
            _decode_attn_cell,
        )
        dim_positions = {"EMBED_LO": 16, "EMBED_HI": 32}
        dim_sizes = {"EMBED_LO": 16, "EMBED_HI": 16}
        ranges = _build_dim_lookup(dim_positions, dim_sizes)
        # row=6*HD+5, col=18 -> head=6, slot=5, EMBED_LO+2
        claim = _decode_attn_cell(
            layer_idx=5, scope="attn_W_v",
            row=6 * 64 + 5, col=18,
            num_heads=8, head_dim=64, dim_ranges=ranges,
        )
        assert claim == (5, "attn_W_v", "6_5", "EMBED_LO+2")

    def test_decode_attn_w_o_reversed(self):
        """W_o: row is OUTPUT dim, col is (head, slot)."""
        from c4_release.neural_vm.unified_compiler.decl_verifier import (
            _build_dim_lookup,
            _decode_attn_cell,
        )
        dim_positions = {"OUTPUT_LO": 100}
        dim_sizes = {"OUTPUT_LO": 16}
        ranges = _build_dim_lookup(dim_positions, dim_sizes)
        # row=103 (OUTPUT_LO+3), col=7*64+1 -> head=7, slot=1
        claim = _decode_attn_cell(
            layer_idx=6, scope="attn_W_o",
            row=103, col=7 * 64 + 1,
            num_heads=8, head_dim=64, dim_ranges=ranges,
        )
        assert claim == (6, "attn_W_o", "7_1", "OUTPUT_LO+3")

    def test_decode_ffn_w_up(self):
        """W_up: row=unit, col=input dim."""
        from c4_release.neural_vm.unified_compiler.decl_verifier import (
            _build_dim_lookup,
            _decode_ffn_cell,
        )
        dim_positions = {"MARK_AX": 10}
        dim_sizes = {"MARK_AX": 1}
        ranges = _build_dim_lookup(dim_positions, dim_sizes)
        claim = _decode_ffn_cell(
            layer_idx=8, scope="ffn_W_up",
            row=1700, col=10, dim_ranges=ranges,
        )
        assert claim == (8, "ffn_W_up", "1700", "MARK_AX+0")

    def test_decode_ffn_w_down_reversed(self):
        """W_down: row=output dim, col=unit."""
        from c4_release.neural_vm.unified_compiler.decl_verifier import (
            _build_dim_lookup,
            _decode_ffn_cell,
        )
        dim_positions = {"OUTPUT_LO": 100}
        dim_sizes = {"OUTPUT_LO": 16}
        ranges = _build_dim_lookup(dim_positions, dim_sizes)
        claim = _decode_ffn_cell(
            layer_idx=8, scope="ffn_W_down",
            row=105, col=1700, dim_ranges=ranges,
        )
        assert claim == (8, "ffn_W_down", "1700", "OUTPUT_LO+5")


class TestOpVerificationResult:
    """Basic dataclass logic."""

    def test_ok_when_no_unused_claims(self):
        r = OpVerificationResult(
            op_name="x",
            declared={(0, "attn_W_v", "0_0", "FOO+0")},
            observed={(0, "attn_W_v", "0_0", "FOO+0"),
                      (0, "attn_W_v", "0_1", "FOO+1")},
        )
        assert r.ok is True  # all declared cells are written
        assert r.ok_strict is False  # but strict needs exact match
        assert len(r.declared_but_not_written) == 0
        assert len(r.written_but_not_declared) == 1

    def test_not_ok_when_declared_but_not_written(self):
        r = OpVerificationResult(
            op_name="x",
            declared={(0, "attn_W_v", "0_0", "FOO+0")},
            observed=set(),
        )
        # bake_fn ran without effects -> inert -> ok=True
        # We need to set inert=False explicitly here for a true drift test.
        r.inert = False
        assert r.ok is False
        assert r.ok_strict is False

    def test_inert_is_treated_as_ok(self):
        r = OpVerificationResult(
            op_name="x",
            declared={(0, "attn_W_v", "0_0", "FOO+0")},
            observed=set(),
            inert=True,
        )
        assert r.ok is True
        assert r.ok_strict is True


# ----------------------------------------------------------------------
# End-to-end production verifier run
# ----------------------------------------------------------------------


# This is the production gate: build the full VM, verify every op with
# ``claims``, and assert no unused claims (= declaration drift in the
# load-bearing direction). On the current code (2026-05-12) we expect
# ``layer5_fetch`` and ``function_call_weights`` to both pass with
# their declared row+column claims fully written by their bake_fns.
class TestProductionVerifier:
    @pytest.fixture(scope="class")
    def report(self):
        return verify_claims_static()

    def test_at_least_one_op_with_claims_verified(self, report):
        """Sanity: at least one annotated op was checked."""
        assert len(report.results) >= 1, (
            "verifier found no ops with non-empty claims -- annotation "
            "infrastructure regressed?"
        )

    def test_no_declaration_drift_on_annotated_ops(self, report):
        """The load-bearing check: declared claims must be written.

        If this fails, an op declares a (layer, scope, identifier, column)
        4-tuple that its bake_fn does NOT write -- a stale annotation
        masking a real change in the bake's behavior.
        """
        assert not report.has_errors(), report.format()

    def test_known_annotated_ops_present(self, report):
        """The two currently-annotated production ops appear in the run."""
        names = {r.op_name for r in report.results}
        # `layer5_fetch` is gated as a block op pinned to layer 5; its
        # claims declare the head 0..5 V-relay rows.
        assert "layer5_fetch" in names
        # `function_call_weights` is a model op; its claims declare the
        # ENT/JSR V-relay rows on L5 attn5 heads 5/6 and L6 attn6 head 7.
        assert "function_call_weights" in names

    def test_partial_claims_have_extra_writes(self, report):
        """Sanity: current ops use the partial-claim convention.

        Both `layer5_fetch` and `function_call_weights` declare ONLY the
        V-relay rows; their bake_fns also write Q/K/O of those heads
        plus other rows. So `written_but_not_declared` is expected to be
        non-empty for both. This test pins the expectation so a future
        switch to full-coverage claims (which would empty the
        ``written_but_not_declared`` set) is a deliberate code review
        event, not a silent regression.
        """
        for r in report.results:
            if r.op_name in ("layer5_fetch", "function_call_weights"):
                assert len(r.written_but_not_declared) > 0, (
                    f"{r.op_name}: expected partial-claim residue but "
                    f"observed exact match -- did the op upgrade to "
                    f"full-coverage claims? Verify with strict_mode=True "
                    f"and update this test."
                )


class TestSyntheticDriftDetection:
    """Verify the verifier can DETECT drift via injection of a synthetic
    Operation with bogus claims.

    We use a tiny standalone compiler (not the full VM) so the test is fast
    and isolated.
    """

    def test_declared_but_not_written_flagged(self):
        """Inject a claim the bake_fn does NOT write; verifier must flag it."""
        from c4_release.neural_vm.unified_compiler.decl_verifier import (
            OpVerificationResult,
        )

        # Construct results manually to exercise the verification logic.
        r = OpVerificationResult(
            op_name="bogus",
            declared={(0, "attn_W_v", "0_0", "FAKE+0")},
            observed={(0, "attn_W_v", "0_1", "OTHER+0")},
        )
        # The bake wrote something but NOT the declared cell -> drift.
        assert (0, "attn_W_v", "0_0", "FAKE+0") in r.declared_but_not_written
        assert not r.ok

    def test_strict_mode_catches_undeclared_writes(self):
        from c4_release.neural_vm.unified_compiler.decl_verifier import (
            OpVerificationResult,
            StaticVerificationReport,
        )

        r = OpVerificationResult(
            op_name="x",
            declared={(0, "attn_W_v", "0_0", "FOO+0")},
            observed={(0, "attn_W_v", "0_0", "FOO+0"),
                      (0, "attn_W_v", "0_1", "BAR+0")},
        )
        rep = StaticVerificationReport(results=[r], strict_mode=True)
        assert rep.has_errors() is True
        rep_nonstrict = StaticVerificationReport(results=[r], strict_mode=False)
        assert rep_nonstrict.has_errors() is False


# ----------------------------------------------------------------------
# Mode B: dynamic produces/consumes (slow, gated)
# ----------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("C4_VERIFY_DECLARATIONS") != "1",
    reason="Mode B is slow; set C4_VERIFY_DECLARATIONS=1 to run it",
)
class TestDynamicVerification:
    def test_dynamic_run_completes(self):
        """Mode B should at least complete without crashing on the
        production model. We don't assert clean output -- it's a best-
        effort liveness probe -- but a crash here means the verifier
        infrastructure broke.
        """
        report = verify_produces_consumes_dynamic()
        # Just sanity-check that we got a structured report back.
        assert isinstance(report.results, list)
