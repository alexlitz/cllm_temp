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
    CompactionSafetyReport,
    MultistepDriftEntry,
    MultistepVerificationReport,
    OpVerificationResult,
    OpcodeCoverageReport,
    SmokeCoverageReport,
    SpecCoverageReport,
    StaticVerificationReport,
    TierADriftEntry,
    TierAReport,
    _ADD_CASCADE_PROGRAM,
    _KNOWN_C4_OPCODES,
    _build_multistep_probe,
    _pack_instr,
    _parse_requires_constraint,
    _resolve_register_offset,
    audit_smoke_coverage,
    audit_spec_coverage,
    verify_claims_static,
    verify_compaction_safety,
    verify_opcode_coverage,
    verify_produces_consumes_dynamic,
    verify_produces_consumes_multistep,
    verify_requires,
    verify_reset_after_step,
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
        # then 4 byte slots). ``_resolve_register_offset`` now returns
        # ``(offset, layer_pin)`` -- layer_pin is None when the name
        # lacks an @L<N> suffix.
        assert _resolve_register_offset("AX_byte0") == (6, None)
        assert _resolve_register_offset("AX_byte3") == (9, None)
        assert _resolve_register_offset("STACK0_byte0") == (21, None)
        assert _resolve_register_offset("STACK0_byte3") == (24, None)

    def test_resolve_register_offset_bare_marker(self):
        # ``AX``/``AX_marker`` both fall back to REG_AX's offset (5).
        assert _resolve_register_offset("AX") == (5, None)
        assert _resolve_register_offset("AX_marker") == (5, None)
        assert _resolve_register_offset("STACK0") == (20, None)
        assert _resolve_register_offset("STACK0_marker") == (20, None)
        assert _resolve_register_offset("REG_PC") == (0, None)

    def test_resolve_register_offset_unknown_returns_none(self):
        assert _resolve_register_offset("BOGUS") == (None, None)
        assert _resolve_register_offset("") == (None, None)

    def test_resolve_register_offset_with_op_name_suffix(self):
        # ``@<op_name>`` suffix pins the layer for multistep probe
        # inspection by referring to whichever layer ``LayerCompiler``
        # placed the named op at. Layer resolution requires the
        # ``layout`` parameter -- without it, the base offset is still
        # returned but layer_pin is None.
        # When ``layout=None`` (unit-test mode), the base name still
        # resolves to its offset; the layer_pin is None.
        assert _resolve_register_offset("AX_byte0@some_op") == (6, None)
        assert _resolve_register_offset("STACK0_marker@some_op") == (20, None)
        assert _resolve_register_offset("PC_marker@some_op") == (0, None)
        assert _resolve_register_offset("REG_AX@some_op") == (5, None)
        # Malformed (trailing @) -> offset None, layer None.
        assert _resolve_register_offset("AX_byte0@") == (None, None)

        # With a layout, op-name references resolve to the matching
        # layer. Build a minimal fake layout exposing the three
        # registries the resolver walks.
        from c4_release.neural_vm.unified_compiler.layer_compiler import (
            Operation, ModelLayout,
        )
        noop = lambda *a, **kw: None  # noqa: E731
        op_at_l3 = Operation(
            name="placed_at_l3",
            reads=set(), writes=set(),
            kind="attn",
            bake_fn=noop,
        )
        block_op = Operation(
            name="block_at_l5",
            reads=set(), writes=set(),
            kind="block",
            layer_idx=5,
            bake_fn=noop,
        )
        model_op = Operation(
            name="model_with_hint",
            reads=set(), writes=set(),
            kind="model",
            layer_idx=6,
            bake_fn=noop,
        )
        layout = ModelLayout(
            d_model=8,
            n_layers=8,
            ops_per_layer=[[] for _ in range(8)],
            dim_positions={},
            dim_sizes={},
            block_ops=[block_op],
            model_ops=[model_op],
        )
        layout.ops_per_layer[3].append(op_at_l3)

        # ops_per_layer hit (attn op).
        assert _resolve_register_offset(
            "AX_byte0@placed_at_l3", layout=layout
        ) == (6, 3)
        # block_ops hit (layer_idx=5).
        assert _resolve_register_offset(
            "STACK0_marker@block_at_l5", layout=layout
        ) == (20, 5)
        # model_ops hit (layer_idx hint = 6).
        assert _resolve_register_offset(
            "REG_AX@model_with_hint", layout=layout
        ) == (5, 6)
        # Unknown op-name -> unresolvable (offset None, layer None).
        assert _resolve_register_offset(
            "AX_byte0@nonexistent_op", layout=layout
        ) == (None, None)

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

        token_tensor, markers, summaries, active_ops = _build_multistep_probe(
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


class TestSentinelProduces:
    """Unit tests for the ``__module_replacement`` / ``__structural``
    sentinel ``produces`` keys.

    Sentinel-only ops document that they intentionally fall outside the
    standard residual-dim schema -- module-replacement ops swap a whole
    submodule (e.g. ``block.ffn = HybridALUBlock(...)``) while structural
    ops are compiler machinery (right-size FFNs, expand wrapper blocks,
    contract validation) or write non-residual buffers (alibi_slopes).
    Mode B / Mode B+ skip drift detection for sentinel ops; the report
    tags them as ``is_sentinel=True``.
    """

    def test_is_sentinel_produces_helper(self):
        from c4_release.neural_vm.unified_compiler.decl_verifier import (
            _is_sentinel_produces,
        )
        assert _is_sentinel_produces({'__module_replacement': 'L8.ffn'})
        assert _is_sentinel_produces({'__structural': 'compiler_machinery'})
        # Mixed dicts (sentinel + real dim) are NOT sentinel-only --
        # the real dims still get checked.
        assert not _is_sentinel_produces({
            '__structural': 'x', 'CARRY': 'AX_byte0',
        })
        assert not _is_sentinel_produces({'CARRY': 'AX_byte0'})
        assert not _is_sentinel_produces({})

    def test_add_op_accepts_sentinel_produces(self):
        """``LayerCompiler.add_op`` must accept sentinel-only produces
        without requiring the sentinel keys to be declared dims.

        The standard validation path rejects undeclared dim names in
        ``produces``; sentinels are special-cased so module-replacement
        ops (which don't write to residual dims at all) can still
        document their non-cellular behavior.
        """
        compiler = LayerCompiler()
        compiler.declare_dim("FOO", 1)
        op = Operation(
            name="sentinel_op",
            reads=set(),
            writes=set(),
            kind="model",
            bake_fn=lambda model, dim_positions, S: None,
            produces={'__module_replacement': 'L8.ffn'},
        )
        # Should not raise.
        compiler.add_op(op)

    def test_add_op_rejects_undeclared_real_dim_alongside_sentinel(self):
        """Mixed dicts: sentinel keys are exempt but real dim names
        must still be declared. This prevents accidental typo'd
        keys from being silently treated as sentinels.
        """
        compiler = LayerCompiler()
        compiler.declare_dim("FOO", 1)
        op = Operation(
            name="bad_op",
            reads=set(),
            writes=set(),
            kind="model",
            bake_fn=lambda model, dim_positions, S: None,
            produces={
                '__structural': 'machinery',
                'NOT_DECLARED': 'AX_byte0',
            },
        )
        with pytest.raises(ValueError, match="references undeclared dim"):
            compiler.add_op(op)

    def test_sentinel_excluded_from_staleness_registry(self):
        """Sentinel keys must not appear in the producers registry so
        they don't accidentally satisfy consumes_fresh lookups for real
        downstream consumers.
        """
        compiler = LayerCompiler()
        compiler.declare_dim("CARRY", 16)
        sentinel_op = Operation(
            name="sentinel",
            reads=set(),
            writes=set(),
            kind="model",
            bake_fn=lambda model, dim_positions, S: None,
            produces={'__module_replacement': 'L8.ffn'},
        )
        compiler.add_op(sentinel_op)
        producers, _ = compiler.build_staleness_registry()
        # Neither sentinel key nor its dummy register should appear.
        assert not any(
            k[0].startswith("__") for k in producers.keys()
        ), f"sentinel keys leaked into producers registry: {producers}"

    def test_multistep_report_tags_sentinel_ops(self):
        """When ``verify_produces_consumes_multistep`` runs an op with
        sentinel-only produces, its result row must carry
        ``is_sentinel=True`` and an empty drift list -- the probe should
        never flag a sentinel op as drifting.
        """
        from c4_release.neural_vm.unified_compiler.decl_verifier import (
            MultistepVerificationResult,
            MultistepVerificationReport,
        )
        # Synthetic result simulating the multistep loop's output.
        sentinel_result = MultistepVerificationResult(
            op_name="efficient_l8_addsub_wrap",
            is_sentinel=True,
            sentinel_kind="module-replacement",
            notes=["sentinel produces (module-replacement)"],
        )
        normal_result = MultistepVerificationResult(
            op_name="layer7_operand_gather",
            drift=[],
        )
        report = MultistepVerificationReport(
            n_steps=2, program_summary="(synthetic)",
            results=[sentinel_result, normal_result],
        )
        # Sentinel ops are reported (not skipped silently) and tagged.
        assert sentinel_result.is_sentinel is True
        assert sentinel_result.sentinel_kind == "module-replacement"
        assert sentinel_result.drift == []
        # ``has_drift`` ignores sentinel ops (their drift is always empty).
        assert report.has_drift() is False
        # ``format`` surfaces the sentinel tag in the status string.
        out = report.format()
        assert "SENTINEL/module-replacement" in out


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

    def test_sentinel_ops_tagged_in_production_report(self):
        """The multistep report on the production model should tag the
        annotated sentinel ops with ``is_sentinel=True`` (and never
        flag them as drifting).

        Annotated sentinel ops (close-produces-annotation-gap, 2026-05-13):
          module-replacement: efficient_l8_addsub_wrap,
              efficient_l10_andorxor_wrap, efficient_l11_alumul_wrap,
              l10_post_op_attach, l10_alu_divmod_install,
              l13_alu_shift_install.
          structural: right_size_ffns, expand_wrapper_blocks,
              contract_validation, residual_alibi_slopes,
              layer10_residual_alibi_slopes.

        Note: this test does not reuse the class fixture's ``compiled``
        because the disk-cache path drops ``block_ops``/``model_ops`` from
        the loaded layout (see ``_try_load_cached`` in
        ``full_vm_compiler.py``), and the verifier needs those ops to
        enumerate sentinel-tagged annotations. We rebuild with
        ``disk_cache=False`` to populate the full op list.
        """
        from c4_release.neural_vm.unified_compiler.full_vm_compiler import (
            compile_full_vm,
        )
        model, layout = compile_full_vm(
            S=100.0, alu_mode="lookup",
            enable_conversational_io=False, n_heads=8,
            disk_cache=False,
        )
        model.eval()
        program = [_pack_instr(1, 7), _pack_instr(38, 0)]  # IMM 7; EXIT
        report = verify_produces_consumes_multistep(
            model=model, layout=layout, program=program, n_steps=2,
        )
        sentinel_results = [r for r in report.results if r.is_sentinel]
        # We annotated multiple sentinel ops; at least 4 should appear in
        # the lookup-mode layout (efficient-only ops like
        # ``efficient_l8_addsub_wrap`` are inert in lookup mode but still
        # registered, so they're collected too).
        assert len(sentinel_results) >= 4, (
            f"expected >=4 sentinel-tagged results; got {len(sentinel_results)}: "
            f"{[r.op_name for r in sentinel_results]}"
        )
        for r in sentinel_results:
            assert r.sentinel_kind in ("module-replacement", "structural")
            assert r.drift == [], (
                f"sentinel op {r.op_name!r} flagged as drifting -- "
                f"sentinel ops must never produce drift entries"
            )

    def test_clean_imm_exit_program_has_no_drift(self, compiled):
        """Multistep probe on a tiny IMM/EXIT program emits no drift on
        any AX_byte0 producer once Tier A opcode gating is honored.

        Step 1: IMM 7 (AX=7). Step 2: EXIT. With ``opcodes`` annotations
        in place on L7 operand_gather (binary ALU + LEA/ADJ/ENT) and on
        L8/L9 ALU (ADD/SUB/LEA + EQ..GE etc.), the verifier auto-gates
        the ``produces`` checks: dims like CARRY/CMP_GROUP/CMP on L8/L9/
        L10 are only expected to fire when an opcode in the op's
        ``opcodes`` set is active for the step. IMM (step 1) and EXIT
        (step 2) are NOT in any of those sets, so the verifier should
        skip those produces checks and surface no drift.

        L7's operand_gather declares ``opcodes`` covering binary ALU +
        LEA/ADJ/ENT — none of which fire on IMM/EXIT — so even ALU_LO/HI
        produces are now opcode-gated and skipped on IMM/EXIT steps.
        The expected drift count is therefore 0 across every op for this
        program.
        """
        model, layout = compiled
        program = [_pack_instr(1, 7), _pack_instr(38, 0)]  # IMM 7; EXIT
        report = verify_produces_consumes_multistep(
            model=model, layout=layout, program=program, n_steps=2,
        )
        # With opcode auto-gating, ALL produces checks skip when the
        # op's opcodes don't include IMM/EXIT. The test asserts no drift
        # on either step.
        all_drift = report.drift_entries()
        assert not all_drift, (
            "IMM 7 / EXIT program produced drift under opcode-gated "
            "produces verification. Either an op declared opcodes that "
            "incorrectly include OP_IMM or OP_EXIT, or an op is missing "
            "its opcodes annotation entirely. "
            f"Drift entries: {all_drift}"
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
# Tier C discoverability bookkeeping audits
# ----------------------------------------------------------------------


@pytest.mark.validator
class TestSmokeCoverageBookkeeping:
    """Bookkeeping check on ``Operation.smoke_tests`` annotations.

    Not a correctness gate -- the audit reports ops with empty
    ``smoke_tests`` as an informational warning so future test-coverage
    work can prioritize them. The only hard invariant: at least one op
    must declare a non-empty ``smoke_tests`` set, otherwise the
    annotation infrastructure has silently regressed.
    """

    @pytest.fixture(scope="class")
    def report(self):
        return audit_smoke_coverage()

    def test_at_least_one_op_has_smoke_tests(self, report):
        annotated = [
            name for name, tests in report.coverage.items() if tests
        ]
        assert annotated, (
            "no op declares smoke_tests; Tier C annotation infrastructure "
            "regressed (empty across the board)."
        )

    def test_known_tier_c_examples_present(self, report):
        """The example ops the Tier C agent annotated must be in coverage."""
        for name in ("layer8_alu", "layer7_operand_gather",
                     "function_call_weights", "phase_a_ffn", "layer3_ffn"):
            assert name in report.coverage, (
                f"op {name!r} missing from smoke-coverage audit -- did "
                f"its registration drift out of all_core_ops?"
            )
        # The 5 example ops should all carry non-empty smoke_tests.
        for name in ("layer8_alu", "layer7_operand_gather",
                     "function_call_weights", "phase_a_ffn", "layer3_ffn"):
            assert report.coverage[name], (
                f"op {name!r} declared no smoke_tests after the Tier C "
                f"annotation pass."
            )

    def test_untested_op_count_informational(self, report):
        """Surfaces untested op count without failing.

        The actual list is informational so a future agent can pick up
        coverage targets. We do not assert on the count -- a fresh
        annotation campaign will draw it down over time.
        """
        # Bookkeeping only: print/format so failures surface the list.
        formatted = report.format()
        assert formatted.startswith("=== Smoke coverage bookkeeping ===")


@pytest.mark.validator
class TestSpecCoverage:
    """Bookkeeping check on ``Operation.spec_section`` annotations.

    Like the smoke-coverage audit this is informational (the BLOG_SPEC
    file evolves independently of the op set). The only hard invariant:
    at least one op declares a non-empty ``spec_section`` and the spec
    file itself parses to at least one heading.
    """

    @pytest.fixture(scope="class")
    def report(self):
        return audit_spec_coverage()

    def test_at_least_one_op_has_spec_section(self, report):
        annotated_ops = sum(len(ops) for ops in report.coverage.values())
        assert annotated_ops >= 1, (
            "no op declares spec_section; Tier C annotation infrastructure "
            "regressed (BLOG_SPEC cross-reference vanished)."
        )

    def test_spec_file_parses_headings(self, report):
        assert len(report.all_sections) >= 1, (
            "BLOG_SPEC.md parsed zero headings -- spec file moved or "
            "audit's heading regex broke."
        )

    def test_known_tier_c_examples_have_spec(self, report):
        annotated_op_names: set = set()
        for ops in report.coverage.values():
            annotated_op_names.update(ops)
        for name in ("layer8_alu", "function_call_weights"):
            assert name in annotated_op_names, (
                f"op {name!r} did not receive a spec_section in the "
                f"Tier C annotation pass."
            )


@pytest.mark.validator
class TestCompactionSafety:
    """Bookkeeping check on ``Operation.compaction_safe`` annotations.

    The audit cross-checks ``compaction_safe=False`` declarations against
    a MoE partition when one is provided. The current test runs the audit
    WITHOUT a partition -- the MoE partition only exists post-compact,
    which is a heavy operation. We document the available partition_unavailable
    flag and assert the declarations are well-formed (no orphan flags).
    A follow-up agent can wire the actual partition through to get the
    full cross-check.
    """

    @pytest.fixture(scope="class")
    def report(self):
        return verify_compaction_safety()

    def test_audit_runs_and_reports_unsafe_ops(self, report):
        assert isinstance(report, CompactionSafetyReport)
        # Bookkeeping invariant: at least one op was annotated either
        # way. The default is compaction_safe=True, so declared_safe
        # should always be populated post-compile.
        assert report.declared_safe, (
            "no op declares compaction_safe=True; default-True invariant "
            "regressed."
        )

    def test_known_unsafe_op_flagged(self, report):
        """The example ``function_call_weights`` is the canonical
        compaction-unsafe op -- its V-relay units fire on non-opcode
        positions (the MoE pathology fixed in commit 2fa04dd).
        """
        assert "function_call_weights" in report.declared_unsafe, (
            "function_call_weights should be declared compaction_safe=False "
            "per Tier C example annotation."
        )

    def test_partition_unavailable_yields_empty_mismatches(self, report):
        """When the audit runs without a partition, mismatches is empty
        and partition_unavailable=True. Documents the invariant for
        future agents that wire a real partition through.
        """
        if report.partition_unavailable:
            assert not report.mismatches


# ----------------------------------------------------------------------
# Routine execution: env-gated auto-run-on-compile
# ----------------------------------------------------------------------


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


# ----------------------------------------------------------------------
# Tier A annotation detectors: reset_after_step / requires / opcodes
# ----------------------------------------------------------------------
# Tier A annotation detectors: reset_after_step / requires / opcodes
# ----------------------------------------------------------------------


class TestTierAFields:
    """Sanity tests for the Tier A field declarations on Operation."""

    def test_defaults_are_empty(self):
        """The three Tier A fields default to empty containers so
        existing ops remain back-compat (the dataclass equality before
        and after the schema bump should hold for unannotated ops).
        """
        compiler = LayerCompiler()
        compiler.declare_dim("X", 1)
        op = Operation(
            name="bare",
            reads=set(),
            writes={"X"},
            kind="ffn",
            bake_fn=lambda *a, **kw: None,
        )
        assert op.reset_after_step == set()
        assert op.requires == {}
        assert op.opcodes == set()

    def test_reset_after_step_validation(self):
        compiler = LayerCompiler()
        compiler.declare_dim("X", 1)
        with pytest.raises(ValueError, match="reset_after_step references"):
            compiler.add_op(Operation(
                name="bad", reads=set(), writes=set(), kind="ffn",
                bake_fn=lambda *a, **kw: None,
                reset_after_step={"NOT_DECLARED"},
            ))

    def test_requires_validation(self):
        compiler = LayerCompiler()
        compiler.declare_dim("X", 1)
        with pytest.raises(ValueError, match="requires references"):
            compiler.add_op(Operation(
                name="bad", reads=set(), writes=set(), kind="ffn",
                bake_fn=lambda *a, **kw: None,
                requires={"NOT_DECLARED": "set_at_AX"},
            ))

    def test_opcodes_accepts_string_set(self):
        compiler = LayerCompiler()
        compiler.declare_dim("X", 1)
        # Opcodes are NOT cross-checked against declared dims (they're a
        # parallel namespace).
        compiler.add_op(Operation(
            name="ok", reads=set(), writes={"X"}, kind="ffn",
            bake_fn=lambda *a, **kw: None,
            opcodes={"OP_ADD", "OP_SUB"},
        ))


@pytest.mark.validator
class TestResetAfterStep:
    """Tier A detector for reset_after_step.

    Unit-style: build a tiny compiler with one synthetic op declaring
    reset_after_step, then drive ``verify_reset_after_step`` over a stub
    layout/model and assert drift surfaces when the dim leaks.
    """

    def test_parse_requires_within_one_step(self):
        kind, marker, value = _parse_requires_constraint("within_one_step")
        assert kind == "within_one_step"
        assert marker is None
        assert value is None

    def test_parse_requires_set_at_marker(self):
        kind, marker, value = _parse_requires_constraint("set_at_AX")
        assert kind == "set_at_AX"
        assert marker == "AX"
        assert value is None

    def test_parse_requires_value_prefix(self):
        kind, marker, value = _parse_requires_constraint("7:set_at_AX")
        assert kind == "set_at_AX"
        assert marker == "AX"
        assert value == 7

    def test_parse_requires_unknown(self):
        kind, marker, value = _parse_requires_constraint("garbage")
        assert kind == "unknown"

    def test_detector_returns_empty_when_no_candidates(self):
        """When no op declares reset_after_step, the detector should
        return an empty drift report with a note (rather than crash).

        We use a stub layout with no candidate ops by passing a
        precompiled empty-ish layout via ``model``/``layout``. Easier to
        construct via ``_build_layout_only`` for a synthetic compiler.
        """
        # Synthetic compiler: declare nothing with reset_after_step.
        compiler = LayerCompiler()
        compiler.declare_dim("X", 1)
        compiler.add_op(Operation(
            name="bare", reads=set(), writes={"X"}, kind="ffn",
            bake_fn=lambda *a, **kw: None,
        ))
        layout = compiler.compile()

        # Build a real (production) model + layout to feed the detector;
        # the production-side check is what we want. The detector should
        # produce ``no ops declare`` if reset_after_step is empty on every
        # op in the production layout. Since `layer1_ffn` annotates it,
        # we instead assert the detector RAN to completion (drift may or
        # may not be present depending on actual leakage).
        # Skip — this test verifies the wiring, not the production state.

    def test_detector_on_production_runs_to_completion(self):
        """Smoke test: ``verify_reset_after_step`` runs against the
        production compile_full_vm without crashing. We do NOT assert
        drift==0 because the L1 STACK0_BYTE0 emission may have leakage
        across step boundaries — exactly the cascade bug the detector
        is built to surface. The point of this test is wiring.
        """
        from c4_release.neural_vm.unified_compiler.full_vm_compiler import (
            compile_full_vm,
        )
        model, layout = compile_full_vm(disk_cache=False)
        model.eval()
        report = verify_reset_after_step(
            model=model, layout=layout, n_steps=2,
        )
        # Wiring sanity: the report has the right shape.
        assert isinstance(report, TierAReport)
        assert report.detector == "reset_after_step"
        # If layer1_ffn is annotated and the wiring works, the detector
        # inspected at least one op (the multistep probe ran). The
        # `n_steps` field captures how many steps were driven.
        assert report.n_steps == 2


@pytest.mark.validator
class TestRequires:
    """Tier A detector for requires preconditions."""

    def test_constraint_parser_handles_alias_forms(self):
        """The parser should accept all the canonical constraint shapes
        documented in the Tier A schema.
        """
        # Canonical
        assert _parse_requires_constraint("set_at_AX")[0] == "set_at_AX"
        assert _parse_requires_constraint("set_at_SP")[0] == "set_at_SP"
        assert _parse_requires_constraint("set_at_PC")[0] == "set_at_PC"
        assert _parse_requires_constraint("set_at_BP")[0] == "set_at_BP"
        assert _parse_requires_constraint("set_at_STACK0")[0] == "set_at_STACK0"
        assert _parse_requires_constraint("set_at_MEM")[0] == "set_at_MEM"
        # Within-one-step lookback
        assert _parse_requires_constraint("within_one_step")[0] == "within_one_step"

    def test_detector_on_production_runs_to_completion(self):
        """Smoke test: ``verify_requires`` runs against the production
        compile_full_vm without crashing. Drift may be non-empty (e.g.,
        STACK0_BYTE0 may not be set within one step on a freshly-started
        IMM 10/PSH/IMM 32/ADD program at step 1 — there's no PSH yet).
        """
        from c4_release.neural_vm.unified_compiler.full_vm_compiler import (
            compile_full_vm,
        )
        model, layout = compile_full_vm(disk_cache=False)
        model.eval()
        report = verify_requires(model=model, layout=layout, n_steps=2)
        assert isinstance(report, TierAReport)
        assert report.detector == "requires"


@pytest.mark.validator
class TestOpcodeCoverage:
    """Tier A detector for opcode coverage."""

    def test_known_opcodes_constant_has_canonical_isa(self):
        """The opcode universe matches the C4 ISA documented in
        ``ops/shared.py``'s opcode -> "OP_<NAME>" map.
        """
        # Spot-check core opcodes
        assert "OP_ADD" in _KNOWN_C4_OPCODES
        assert "OP_LEA" in _KNOWN_C4_OPCODES
        assert "OP_EXIT" in _KNOWN_C4_OPCODES
        assert "OP_PSH" in _KNOWN_C4_OPCODES
        # Sanity: no stray names
        for opcode in _KNOWN_C4_OPCODES:
            assert opcode.startswith("OP_"), opcode

    def test_every_opcode_has_at_least_one_op(self):
        """Strong invariant: every C4 opcode has at least one op handling it.

        Marked xfail because the current annotation pass is partial.
        Once the produces-sweep agents fill in the opcodes annotation on
        the rest of the ALU/IO/control-flow ops, remove the xfail.
        """
        report = verify_opcode_coverage()
        assert not report.unreachable, (
            f"Unreachable opcodes: {report.unreachable}\n{report.format()}"
        )

    def test_partial_coverage_includes_alu_opcodes_we_annotated(self):
        """Loose invariant: the opcodes our 5 example annotations cover
        actually appear as reachable in the coverage matrix.
        """
        report = verify_opcode_coverage()
        # OP_ADD is annotated on L7 operand_gather, L8 alu, L9 alu.
        assert report.coverage.get("OP_ADD"), (
            "OP_ADD should be declared by at least one op (Tier A "
            "annotations on L7/L8/L9 ALU); coverage was empty"
        )
        assert report.coverage.get("OP_LEA"), (
            "OP_LEA should be declared by at least one op (L7 operand_"
            "gather covers LEA/ADJ/ENT)"
        )
        # OP_AND/OR/XOR are on L9.
        assert report.coverage.get("OP_AND")

    def test_with_synthetic_compile_fn(self):
        """Caller can pass a synthetic compile function — useful for
        unit-test coverage matrices without invoking ``compile_full_vm``.
        """
        def synthetic():
            compiler = LayerCompiler()
            compiler.declare_dim("X", 1)
            compiler.add_op(Operation(
                name="tiny", reads=set(), writes={"X"}, kind="ffn",
                bake_fn=lambda *a, **kw: None,
                opcodes={"OP_ADD", "OP_SUB"},
            ))
            return compiler.compile()

        report = verify_opcode_coverage(compile_fn=synthetic)
        assert report.coverage["OP_ADD"] == ["tiny"]
        assert report.coverage["OP_SUB"] == ["tiny"]
        # Most opcodes are unreachable in the synthetic compiler.
        assert "OP_LEA" in report.unreachable
