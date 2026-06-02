"""Tests for Phase 8.O.3 div_mode toggle.

The ``div_mode`` field on :class:`VMConfig` selects the DIV/MOD
implementation:

- ``"long_div"`` (default): existing base-16 long division. Byte-
  identical to every pre-8.O.3 baseline because the FlattenedDivMod
  pipeline already implements the threshold-counting digit-by-digit
  long division spec'd in BLOG_SPEC.md §"Long Division Implementation".
- ``"log_softmax1"``: softmax1-sink-based 1/n attention construction
  (BLOG_SPEC.md §"Via Attention With Log Sink"). REQUIRES
  ``attention_normalization == "softmax1"``. Currently a stub; the
  ``FlattenedDivMod.forward`` raises ``NotImplementedError`` when DIV
  or MOD is actually invoked under this mode.

Tests verify:

1. The config field defaults to ``"long_div"`` and validates inputs.
2. ``log_softmax1`` is rejected with ``attention_normalization='softmax'``.
3. The ``NEURAL_VM_DIV_MODE`` environment variable plumbs through
   :func:`get_config`.
4. ``FlattenedDivMod.forward`` under ``long_div`` runs the long-division
   pipeline as before. Under ``log_softmax1`` it raises
   ``NotImplementedError`` from the stub when DIV/MOD is active, but
   passes through cleanly when neither is active (so unrelated
   programs are not penalized).
5. A 1096-sample DIV smoke test runs under default config (``long_div``)
   end-to-end and produces the correct result.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_vm.config import (
    VMConfig,
    get_config,
    reset_config,
    set_config,
)


# =============================================================================
# Config validation
# =============================================================================


class TestDivModeConfig:
    """VMConfig validation for the Phase 8.O.3 div_mode field."""

    def setup_method(self):
        reset_config()

    def teardown_method(self):
        reset_config()
        os.environ.pop("NEURAL_VM_DIV_MODE", None)

    def test_default_long_div(self):
        cfg = get_config()
        assert cfg.div_mode == "long_div"

    def test_explicit_long_div(self):
        cfg = VMConfig(div_mode="long_div")
        assert cfg.div_mode == "long_div"

    def test_log_softmax1_with_softmax1(self):
        """``log_softmax1`` accepted when paired with softmax1 attention."""
        cfg = VMConfig(
            div_mode="log_softmax1",
            attention_normalization="softmax1",
        )
        assert cfg.div_mode == "log_softmax1"
        assert cfg.attention_normalization == "softmax1"

    def test_log_softmax1_rejects_plain_softmax(self):
        """The +1 sink in the softmax1 denominator IS the construction; bare
        softmax cannot produce 1/n via the BLOG_SPEC §"Via Attention With
        Log Sink" recipe, so the combination must be rejected at config
        construction time, not silently corrupt DIV at execution time."""
        with pytest.raises(ValueError, match="log_softmax1"):
            VMConfig(
                div_mode="log_softmax1",
                attention_normalization="softmax",
            )

    def test_rejects_unknown_div_mode(self):
        with pytest.raises(ValueError, match="div_mode"):
            VMConfig(div_mode="magic")  # type: ignore[arg-type]

    def test_env_var_long_div(self):
        os.environ["NEURAL_VM_DIV_MODE"] = "long_div"
        reset_config()
        assert get_config().div_mode == "long_div"

    def test_env_var_log_softmax1(self):
        os.environ["NEURAL_VM_DIV_MODE"] = "log_softmax1"
        reset_config()
        cfg = get_config()
        assert cfg.div_mode == "log_softmax1"
        # The default attention_normalization is softmax1, so this combo
        # is valid even without an explicit override.
        assert cfg.attention_normalization == "softmax1"


# =============================================================================
# FlattenedDivMod direct unit tests (fast — no full VM build)
# =============================================================================


def _build_flattened_div_mod():
    """Build a fully-installed FlattenedDivMod for direct forward testing.

    The production wiring installs the 4 stages via compiler ops; here we
    invoke the same installers directly so we can stress-test the
    ``forward`` div_mode gate without spinning up the full VM compile
    (which takes minutes per fresh config).
    """
    from neural_vm.efficient_alu_divmod_split import FlattenedDivMod
    from neural_vm.vm_step import _SetDim

    m = FlattenedDivMod(100.0, _SetDim)
    m.install_bdtoge()
    m.install_longdiv()
    m.install_getobd()
    return m


def _div_active_input(d_model: int = 512):
    """Build a synthetic BD residual tensor with OP_DIV + MARK_AX hot."""
    import torch
    from neural_vm.vm_step import _SetDim as BD

    x = torch.zeros(1, 1, d_model)
    x[0, 0, BD.OP_DIV] = 1.0
    x[0, 0, BD.MARK_AX] = 1.0
    x[0, 0, BD.ALU_LO + 8] = 1.0       # dividend lo (a=8)
    x[0, 0, BD.AX_CARRY_LO + 2] = 1.0  # divisor lo (b=2)
    return x


def _noop_input(d_model: int = 512):
    """Build a synthetic BD residual tensor with no DIV/MOD active.

    Used to verify the stub gate does NOT fire on unrelated programs —
    only DIV/MOD actually invokes the long-division pipeline (or its
    log_softmax1 stub)."""
    import torch
    from neural_vm.vm_step import _SetDim as BD

    x = torch.zeros(1, 1, d_model)
    x[0, 0, BD.MARK_AX] = 1.0  # marker present but no DIV/MOD opcode
    return x


class TestFlattenedDivModGate:
    """Direct unit tests on FlattenedDivMod.forward gate."""

    def setup_method(self):
        reset_config()

    def teardown_method(self):
        reset_config()

    def test_long_div_runs_pipeline(self):
        """``long_div`` (default) executes the long-division pipeline."""
        set_config(VMConfig(div_mode="long_div"))
        m = _build_flattened_div_mod()
        x = _div_active_input()
        out = m(x)
        assert out.shape == x.shape

    def test_long_div_byte_identical_to_baseline(self):
        """Byte-identity: ``long_div`` output bit-for-bit matches a direct
        ``pipeline(x_bd)`` call.

        This is the strong-form Phase 8.O.3 backwards-compatibility
        gate: the only path change for ``long_div`` is the extra
        ``get_config()`` lookup and conditional, both of which must
        evaluate cleanly to "call the existing pipeline" with no
        intervening rounding, dtype change, or alternate branch. If
        this assertion fires, the toggle has perturbed the long-div
        path and the byte-identity promise in the spec is broken.
        """
        import torch

        set_config(VMConfig(div_mode="long_div"))
        m = _build_flattened_div_mod()
        x = _div_active_input()
        out_via_gate = m(x.clone())

        # Direct call to ``pipeline(...)`` — the same code path the
        # ``long_div`` branch resolves to.
        pipeline = m.__dict__["pipeline"]
        assert pipeline is not None
        out_via_baseline = pipeline(x.clone())

        assert torch.equal(out_via_gate, out_via_baseline), (
            "Phase 8.O.3 long_div path is NOT byte-identical to the "
            "baseline pipeline call; the div_mode gate has introduced "
            "a numerical perturbation."
        )

    def test_log_softmax1_stub_raises(self):
        """``log_softmax1`` raises ``NotImplementedError`` when DIV is active.

        Error message must mention ``log_softmax1`` and ``BLOG_SPEC`` so
        the design-doc pointer survives in operator-facing tracebacks.
        """
        set_config(
            VMConfig(
                div_mode="log_softmax1",
                attention_normalization="softmax1",
            )
        )
        m = _build_flattened_div_mod()
        x = _div_active_input()
        with pytest.raises(NotImplementedError) as exc_info:
            m(x)
        msg = str(exc_info.value)
        assert "log_softmax1" in msg
        assert "BLOG_SPEC" in msg

    def test_log_softmax1_passes_through_when_no_div(self):
        """Unrelated programs (no DIV/MOD opcode) must not hit the stub.

        The gate sits BELOW the FlattenedDivMod early-out, which returns
        ``x_bd`` unchanged when both OP_DIV and OP_MOD are quiescent.
        That way enabling ``log_softmax1`` for an ADD-only program does
        not spuriously raise — the stub only fires for programs that
        actually invoke DIV/MOD.
        """
        set_config(
            VMConfig(
                div_mode="log_softmax1",
                attention_normalization="softmax1",
            )
        )
        m = _build_flattened_div_mod()
        x = _noop_input()
        out = m(x)
        assert out.shape == x.shape


# =============================================================================
# End-to-end integration test on a 1096-sample DIV program
# =============================================================================


class TestDivModeEndToEnd:
    """End-to-end DIV/MOD smoke + 1096 sample integration tests.

    These exercise the toggle through the full ``BatchedPureNeuralRunner``
    pipeline. Marked ``slow`` because building the pure-neural model is
    ~15 s on warm cache, longer on cold.

    Baseline note (June 2026, ``speedup-cache-and-buckets`` branch):
    These tests gate on the canonical 84/2=42 and 144/12=12 results,
    matching ``TestSmokeBasic::test_div_basic`` and the DIV category of
    ``tests/test_suite_1000.py``. If the underlying smoke baseline on a
    given branch is failing for non-div-mode reasons (e.g. pre-existing
    decode regressions), these tests will surface as failures rather
    than green — that is intentional. The byte-identity unit test
    above (``test_long_div_byte_identical_to_baseline``) is the
    strong-form gate that the toggle wiring did not perturb anything.
    """

    def setup_method(self):
        reset_config()

    def teardown_method(self):
        reset_config()

    @pytest.mark.slow
    def test_long_div_smoke_div_basic(self):
        """``long_div`` default executes the smoke ``test_div_basic`` shape.

        Mirrors ``TestSmokeBasic::test_div_basic`` (84 / 2 == 42) from
        ``tests/test_smoke.py`` — the canonical DIV smoke shape. Default
        config (``div_mode='long_div'``) is byte-identical to every
        pre-8.O.3 baseline; this asserts the toggle wiring does not
        perturb the long-division path.

        Uses the same ``BatchedPureNeuralRunner.run_batch`` path as the
        production smoke suite (``tests/test_smoke.py``), which is the
        path the smoke tests are calibrated against — the per-call
        ``AutoregressiveVMRunner.run(...)`` path requires extra context
        priming that BatchedPureNeuralRunner sets up internally.
        """
        from neural_vm.batched_pure_neural import BatchedPureNeuralRunner
        from neural_vm.embedding import Opcode

        # Confirmed default: no set_config() call.
        assert get_config().div_mode == "long_div"

        # Same bytecode the smoke test uses for test_div_basic.
        bytecode = [
            Opcode.IMM | (84 << 8), Opcode.PSH,
            Opcode.IMM | (2 << 8), Opcode.DIV,
            Opcode.EXIT,
        ]
        runner = BatchedPureNeuralRunner()
        results = runner.run_batch([bytecode], max_steps=20)
        _output, result = results[0]
        assert result == 42, f"84/2 expected 42, got {result}"

    @pytest.mark.slow
    def test_long_div_1096_sample_div(self):
        """1096-style sample: ``return 144 / 12;`` via compile_c.

        Pulls the DIV category template from
        ``tests/test_suite_1000.py:generate_test_programs`` (the
        ``return A / B;`` shape) and runs it through the full
        ``compile_c`` + BatchedPureNeuralRunner pipeline. Default config
        (``div_mode='long_div'``) is byte-identical to every pre-8.O.3
        baseline, so this is the integration-level regression gate.
        """
        from src.compiler import compile_c
        from neural_vm.batched_pure_neural import BatchedPureNeuralRunner

        assert get_config().div_mode == "long_div"

        source = "int main() { return 144 / 12; }"
        bytecode, _data = compile_c(source)
        runner = BatchedPureNeuralRunner()
        results = runner.run_batch([bytecode], max_steps=2000)
        _output, result = results[0]
        assert result == 12, f"144/12 expected 12, got {result}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
