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
    MultistepDriftEntry,
    MultistepVerificationReport,
    OpVerificationResult,
    StaticVerificationReport,
    _ADD_CASCADE_PROGRAM,
    _build_multistep_probe,
    _pack_instr,
    _resolve_register_offset,
    verify_claims_static,
    verify_produces_consumes_dynamic,
    verify_produces_consumes_multistep,
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


# ----------------------------------------------------------------------
# Mode B+: multistep produces verification (validator marker)
# ----------------------------------------------------------------------


class TestUnitMultistepProbe:
    """Unit tests for the multistep probe builder / register resolver.

    These do not invoke ``compile_full_vm`` so they run in well under a
    second. The pyramid keeps the expensive validator tests below from
    blocking trivial regressions in the builder logic.
    """

    def test_resolve_register_offset_byte_names(self):
        # ``AX_byte0`` is offset 6 within a 35-token step (REG_AX at 5,
        # then 4 byte slots).
        assert _resolve_register_offset("AX_byte0") == 6
        assert _resolve_register_offset("AX_byte3") == 9
        assert _resolve_register_offset("STACK0_byte0") == 21
        assert _resolve_register_offset("STACK0_byte3") == 24

    def test_resolve_register_offset_bare_marker(self):
        # ``AX``/``AX_marker`` both fall back to REG_AX's offset (5).
        assert _resolve_register_offset("AX") == 5
        assert _resolve_register_offset("AX_marker") == 5
        assert _resolve_register_offset("STACK0") == 20
        assert _resolve_register_offset("REG_PC") == 0

    def test_resolve_register_offset_unknown_returns_none(self):
        assert _resolve_register_offset("BOGUS") is None
        assert _resolve_register_offset("") is None

    def test_pack_instr(self):
        # IMM 10 -> opcode=1, imm=10 -> 0x00000a01.
        assert _pack_instr(1, 10) == 0x0a01
        # ADD -> opcode=25 -> 0x19.
        assert _pack_instr(25, 0) == 0x19

    def test_build_multistep_probe_add_cascade(self):
        """Builder runs the canonical ADD program and emits 4 step markers
        with correct PC/AX/SP/STACK0 evolution.

        This is the smoke that confirms ``DraftVM.draft_tokens`` agrees
        with our offset table: AX hits 10 on step 1, STACK0 hits 10 on
        step 2 (after PSH), AX hits 32 on step 3 (after IMM 32), and
        AX hits 42 on step 4 (after ADD).
        """
        # Layout is unused by the builder (only by the verifier), so a
        # stub is fine.
        class _StubLayout:  # noqa: D401
            pass

        token_tensor, markers, summaries = _build_multistep_probe(
            _StubLayout(), _ADD_CASCADE_PROGRAM, n_steps=4,
        )
        assert token_tensor.shape == (1, 184), (
            "header (12) + 4 steps * 35 = 152 + 12 = 184; "
            f"got shape={tuple(token_tensor.shape)}"
        )
        assert len(markers) == 4
        assert len(summaries) == 4
        # Spot-check summaries: after step 1 (IMM 10), AX=0xa. After
        # step 4 (ADD), AX=0x2a (= 42).
        assert "AX=a" in summaries[0]
        assert "AX=2a" in summaries[3]
        # Each marker map carries 35 entries (one per step-token offset).
        assert len(markers[0]) == 35
        # AX_byte0 of step 2 should land in the second step's window.
        # Header = 1 (CODE_START) + 5*8 (40 bytes) + 1 (CODE_END) + 1
        # (DATA_START) + 1 (DATA_END) = 44. Step 1 starts at 44,
        # REG_AX at 44+5=49, step 2 starts at 44+35=79, REG_AX at 84.
        assert markers[0]["REG_AX"] == 49
        assert markers[1]["REG_AX"] == 84
        assert markers[1]["AX_byte0"] == 85
        assert markers[1]["STACK0_byte0"] == 100


@pytest.mark.validator
class TestMultistepProbe:
    """Mode B+ multistep dynamic verifier on the production model.

    These tests are expensive (~70s for the model bake plus a forward
    pass). They live behind ``-m validator`` so smoke runs skip them by
    default; CI can opt in via ``pytest -m validator``.

    Class-scoped fixture caches the compiled model so all tests in the
    class share one bake.
    """

    @pytest.fixture(scope="class")
    def compiled(self):
        # Build the model once for the whole class -- saves ~70s per
        # test case relative to letting each call rebuild.
        from c4_release.neural_vm.unified_compiler.full_vm_compiler import (
            compile_full_vm,
        )
        model, layout = compile_full_vm(
            S=100.0, alu_mode="lookup",
            enable_conversational_io=False, n_heads=8,
        )
        model.eval()
        return model, layout

    def test_clean_imm_exit_program_has_no_drift(self, compiled):
        """Multistep probe on a tiny IMM/EXIT program emits no drift.

        Step 1: IMM 7 (AX=7). Step 2: EXIT. The 3 ops currently annotated
        with ``produces`` (layer7_operand_gather, layer8_multibyte_routing,
        layer8_head6_ax_carry_refresh) all target AX_byte0 / ALU_*. On the
        IMM step the model populates AX bytes, so the produces-liveness
        check should pass for the live steps. The post-EXIT step may also
        have residue from the prior step (causal attention). This test
        pins the baseline: a benign program produces no drift entries.
        """
        model, layout = compiled
        program = [_pack_instr(1, 7), _pack_instr(38, 0)]  # IMM 7; EXIT
        report = verify_produces_consumes_multistep(
            model=model, layout=layout, program=program, n_steps=2,
        )
        # We allow some drift entries on the EXIT step (step 2) where
        # the model has nothing to compute -- but the IMM step (step 1)
        # should never drift for AX_byte0 producers.
        step1_drift = [
            d for d in report.drift_entries()
            if d.step == 1 and d.register == "AX_byte0"
        ]
        assert not step1_drift, (
            "IMM 7 step should populate AX_byte0 residual at every "
            "AX_byte0-producing op's post-layer residual. Drift on step "
            f"1 indicates a real declaration regression: {step1_drift}"
        )

    def test_add_cascade_surfaces_stack0_drift_if_annotated(self, compiled):
        """Multistep probe on IMM/PSH/IMM/ADD/EXIT surfaces STACK0 drift
        if any L6/L10/STK0-carry op declares ``produces[STACK0_byte*]``.

        At time of writing (2026-05-13) no such annotations exist, so
        this test skips with a reason that points at the Tier 1
        annotation agent. Once they land, this test asserts the probe
        actually surfaces the smoking-gun cascade.
        """
        model, layout = compiled
        # Look for any op with a STACK0 register annotation.
        all_ops = []
        for ops_at in layout.ops_per_layer:
            all_ops.extend(ops_at)
        all_ops.extend(layout.block_ops)
        all_ops.extend(layout.model_ops)
        stk0_producers = [
            op for op in all_ops
            if any(
                "STACK0" in reg or "STK0" in reg
                for reg in op.produces.values()
            )
        ]
        if not stk0_producers:
            pytest.skip(
                "no op declares produces=...STACK0_byte*...; Tier 1 "
                "annotation agent will fill these in. Until then the "
                "cascade verifier has no targets to flag."
            )
        report = verify_produces_consumes_multistep(
            model=model, layout=layout,
            program=list(_ADD_CASCADE_PROGRAM), n_steps=4,
        )
        # When STACK0 producers exist, the cascade program should
        # exercise them across steps 2/3 (PSH + IMM-after-PSH). At
        # least one drift entry on step 3 with a STACK0 register is
        # the canonical smoking-gun signature.
        stk0_step3_drift = [
            d for d in report.drift_entries()
            if d.step >= 3 and "STACK0" in d.register
        ]
        assert stk0_step3_drift, (
            "Expected cascade drift on step 3+ for STACK0 register "
            "produces annotations. None observed. Either the cascade "
            "bug is fixed (great!) or the STACK0 producers are not "
            "actually expected to fire on step 3. Report:\n"
            + report.format()
        )


# ----------------------------------------------------------------------
# Routine execution: env-gated auto-run-on-compile
# ----------------------------------------------------------------------


@pytest.mark.validator
class TestRoutineExecution:
    """Smoke that ``C4_VALIDATE_ON_COMPILE=1`` does not break compile.

    The env var asks ``compile_full_vm`` to run Mode A as a warn-only
    sanity check after baking the model. It must:

      1. Not raise (the validator is opt-in diagnostic).
      2. Not change the returned model/layout shape.

    Anything else (drift warnings, etc.) is informational only.
    """

    def test_validate_on_compile_env_gate(self, monkeypatch):
        from c4_release.neural_vm.unified_compiler.full_vm_compiler import (
            compile_full_vm,
        )

        monkeypatch.setenv("C4_VALIDATE_ON_COMPILE", "1")
        model, layout = compile_full_vm(disk_cache=False)
        assert layout.d_model > 0
        assert layout.n_layers > 0
        assert model is not None
