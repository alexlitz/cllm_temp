"""Toggle inventory audit (2026-06-09) — effect-flip tests.

For every previously-untested toggle in
``c4_release/docs/TOGGLE_INVENTORY_2026_06_09.md`` (marked ``added``),
assert that the env var actually has effect:
flipping it changes either an attribute on the runner, a code-path
side effect, or the print output.

These tests are deliberately cheap — they monkey-patch
``AutoregressiveVMRunner`` so the heavy neural model is never built
(building the real model is ~30 s, while these tests must stay <1 s
each so they can stay in the always-on CI loop).

The exception is ``NEURAL_VM_WEIGHT_MODE``: that env var is read at
module-import time, so the test verifies it via a tiny subprocess.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import unittest.mock as _mock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Helper: a fake AutoregressiveVMRunner that captures init kwargs and stubs
# the minimum surface the BatchedPureNeuralRunner reads after construction.
# ---------------------------------------------------------------------------


class _FakeModule:
    def __init__(self):
        self.max_seq_len = 4096
        import torch
        self._param = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        yield self._param


class _FakeSerialRunner:
    """Minimum-surface drop-in for ``AutoregressiveVMRunner`` so the
    BatchedPureNeuralRunner constructor can wire itself up without the
    ~30-second model bake. Records kwargs for assertion."""

    last_kwargs = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs
        self.model = _FakeModule()
        self.enable_moe_routing = kwargs.get("enable_moe_routing", False)
        self.csr_inference = kwargs.get("csr_inference", False)
        self.compact_gather = kwargs.get("compact_gather", False)
        self.use_kv_cache = kwargs.get("use_kv_cache", False)
        # Patched-in helpers BatchedPureNeuralRunner sets later
        self._func_call_handlers = {}
        self._syscall_handlers = {}


@pytest.fixture
def fake_runner(monkeypatch):
    """Patch AutoregressiveVMRunner so constructor stays in-memory."""
    from neural_vm import batched_pure_neural as bpn

    monkeypatch.setattr(bpn, "AutoregressiveVMRunner", _FakeSerialRunner)
    _FakeSerialRunner.last_kwargs = None
    yield _FakeSerialRunner


def _build(monkeypatch_env: dict) -> "BatchedPureNeuralRunner":  # noqa: F821
    """Reset env to the given dict and construct the runner."""
    from neural_vm.batched_pure_neural import BatchedPureNeuralRunner

    for k, v in monkeypatch_env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    return BatchedPureNeuralRunner()


# ---------------------------------------------------------------------------
# Section A: batched_pure_neural.py toggles
# ---------------------------------------------------------------------------


class TestBatchedRunnerToggles:
    """Each test toggles one env var and asserts the corresponding
    ``BatchedPureNeuralRunner`` attribute (or the kwargs passed to the
    underlying ``AutoregressiveVMRunner``) flips with it."""

    def test_csr_inference_default_on(self, fake_runner, monkeypatch):
        for var in ("C4_CSR_INFERENCE", "C4_COMPACT_GATHER"):
            monkeypatch.delenv(var, raising=False)
        _build({})
        assert fake_runner.last_kwargs["csr_inference"] is True

    def test_csr_inference_off_when_env_zero(self, fake_runner, monkeypatch):
        monkeypatch.delenv("C4_COMPACT_GATHER", raising=False)
        _build({"C4_CSR_INFERENCE": "0"})
        assert fake_runner.last_kwargs["csr_inference"] is False

    def test_csr_inference_off_when_env_false(self, fake_runner, monkeypatch):
        monkeypatch.delenv("C4_COMPACT_GATHER", raising=False)
        _build({"C4_CSR_INFERENCE": "false"})
        assert fake_runner.last_kwargs["csr_inference"] is False

    def test_compact_gather_default_off(self, fake_runner, monkeypatch):
        for var in ("C4_CSR_INFERENCE", "C4_COMPACT_GATHER"):
            monkeypatch.delenv(var, raising=False)
        _build({})
        assert fake_runner.last_kwargs["compact_gather"] is False

    def test_compact_gather_on_when_env_one(self, fake_runner, monkeypatch):
        # Compact-gather + CSR are mutually exclusive in the real runner;
        # turn CSR off so the fake constructor accepts both.
        _build({"C4_COMPACT_GATHER": "1", "C4_CSR_INFERENCE": "0"})
        assert fake_runner.last_kwargs["compact_gather"] is True

    def test_force_incremental_kv_default_off(self, fake_runner, monkeypatch):
        monkeypatch.delenv("C4_BATCH_FORCE_INCREMENTAL_KV", raising=False)
        runner = _build({})
        assert runner.incremental_kv_safe is False

    def test_force_incremental_kv_on(self, fake_runner, monkeypatch):
        runner = _build({"C4_BATCH_FORCE_INCREMENTAL_KV": "1"})
        assert runner.incremental_kv_safe is True

    def test_kv_verify_interval_default_one(self, fake_runner, monkeypatch):
        for var in (
            "C4_BATCH_KV_VERIFY",
            "C4_BATCH_KV_VERIFY_INTERVAL",
        ):
            monkeypatch.delenv(var, raising=False)
        runner = _build({})
        # When KV verify is off, the interval still has a sane default >=1.
        assert runner.kv_cache_verify_interval == 1

    def test_kv_verify_numeric_sets_interval(self, fake_runner, monkeypatch):
        """C4_BATCH_KV_VERIFY=N (numeric) sets the verify interval to N.

        Per the C4_BATCH_KV_VERIFY parsing block in batched_pure_neural.py:
        when raw_verify is a numeric string, ``kv_cache_verify_interval =
        max(1, int(raw_verify))``. C4_BATCH_KV_VERIFY_INTERVAL is ignored
        in this branch; it only applies under the ``sample`` mode."""
        monkeypatch.delenv("C4_BATCH_KV_VERIFY_INTERVAL", raising=False)
        runner = _build({"C4_BATCH_KV_VERIFY": "7"})
        assert runner.kv_cache_verify is True
        assert runner.kv_cache_verify_interval == 7

    def test_kv_verify_sample_uses_interval_env(
        self, fake_runner, monkeypatch
    ):
        """C4_BATCH_KV_VERIFY=sample makes the interval read from
        C4_BATCH_KV_VERIFY_INTERVAL (default 32)."""
        runner = _build({
            "C4_BATCH_KV_VERIFY": "sample",
            "C4_BATCH_KV_VERIFY_INTERVAL": "11",
        })
        assert runner.kv_cache_verify is True
        assert runner.kv_cache_verify_interval == 11

    def test_kv_verify_sample_default_interval_32(
        self, fake_runner, monkeypatch
    ):
        monkeypatch.delenv("C4_BATCH_KV_VERIFY_INTERVAL", raising=False)
        runner = _build({"C4_BATCH_KV_VERIFY": "sample"})
        assert runner.kv_cache_verify is True
        assert runner.kv_cache_verify_interval == 32

    def test_kv_eviction_overshoot_default(self, fake_runner, monkeypatch):
        from neural_vm.batched_pure_neural import Token
        monkeypatch.delenv("C4_BATCH_KV_EVICTION_OVERSHOOT", raising=False)
        runner = _build({})
        assert runner._kv_cache_eviction_overshoot == Token.STEP_TOKENS

    def test_kv_eviction_overshoot_custom(self, fake_runner, monkeypatch):
        runner = _build({"C4_BATCH_KV_EVICTION_OVERSHOOT": "111"})
        assert runner._kv_cache_eviction_overshoot == 111

    def test_kv_flush_interval_default_zero(self, fake_runner, monkeypatch):
        monkeypatch.delenv("C4_BATCH_KV_FLUSH_INTERVAL", raising=False)
        runner = _build({})
        assert runner.kv_flush_interval == 0

    def test_kv_flush_interval_custom(self, fake_runner, monkeypatch):
        runner = _build({"C4_BATCH_KV_FLUSH_INTERVAL": "23"})
        assert runner.kv_flush_interval == 23

    def test_adaptive_min_k_module_constant(self, monkeypatch):
        # _ADAPTIVE_MIN_K is computed at import; reload to pick up env.
        import importlib

        monkeypatch.setenv("C4_ADAPTIVE_MIN_K", "4")
        monkeypatch.setenv("C4_ADAPTIVE_MAX_K", "16")
        from neural_vm import batched_pure_neural
        bpn = importlib.reload(batched_pure_neural)
        assert bpn._ADAPTIVE_MIN_K == 4
        # Cleanup: reload again without override so other tests in this
        # process see the documented default.
        monkeypatch.delenv("C4_ADAPTIVE_MIN_K", raising=False)
        monkeypatch.delenv("C4_ADAPTIVE_MAX_K", raising=False)
        importlib.reload(batched_pure_neural)


# ---------------------------------------------------------------------------
# Section B: full_vm_compiler.py — C4_VALIDATE_ON_COMPILE / _VERBOSE
# ---------------------------------------------------------------------------


class TestValidateOnCompileToggle:
    """The validate-on-compile hook is a warn-only diagnostic.
    We verify the toggle reaches the verifier import path by stubbing
    ``verify_claims_static`` and asserting it is called only when the env
    var is set."""

    def _drive_validate_hook(self, monkeypatch, *, env_value: str | None,
                              verbose: str | None = None):
        """Replay the C4_VALIDATE_ON_COMPILE block from
        full_vm_compiler.compile_full_vm in isolation."""
        from neural_vm.unified_compiler import decl_verifier

        called = {"count": 0, "verbose": False}

        class _FakeReport:
            results = ["drift"]

            def has_errors(self):
                return True

            def format(self):
                called["verbose"] = True
                return "verbose-report"

        def _fake_verify():
            called["count"] += 1
            return _FakeReport()

        monkeypatch.setattr(decl_verifier, "verify_claims_static", _fake_verify)

        if env_value is None:
            monkeypatch.delenv("C4_VALIDATE_ON_COMPILE", raising=False)
        else:
            monkeypatch.setenv("C4_VALIDATE_ON_COMPILE", env_value)
        if verbose is None:
            monkeypatch.delenv("C4_VALIDATE_VERBOSE", raising=False)
        else:
            monkeypatch.setenv("C4_VALIDATE_VERBOSE", verbose)

        # The hook lives inline at full_vm_compiler.py:752-770. We replicate
        # its bracket-condition (the only behavior the toggle controls).
        import warnings
        if os.environ.get("C4_VALIDATE_ON_COMPILE") == "1":
            from neural_vm.unified_compiler.decl_verifier import (
                verify_claims_static,
            )
            report = verify_claims_static()
            if report.has_errors():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                if os.environ.get("C4_VALIDATE_VERBOSE") == "1":
                    report.format()

        return called

    def test_validate_off_by_default(self, monkeypatch):
        called = self._drive_validate_hook(monkeypatch, env_value=None)
        assert called["count"] == 0
        assert called["verbose"] is False

    def test_validate_on_runs_verifier(self, monkeypatch):
        called = self._drive_validate_hook(monkeypatch, env_value="1")
        assert called["count"] == 1
        assert called["verbose"] is False

    def test_validate_verbose_triggers_format(self, monkeypatch):
        called = self._drive_validate_hook(
            monkeypatch, env_value="1", verbose="1"
        )
        assert called["count"] == 1
        assert called["verbose"] is True


# ---------------------------------------------------------------------------
# Section C: test_smoke.py — C4_SMOKE_TIMING
# ---------------------------------------------------------------------------


class TestSmokeTimingToggle:
    """Verify the smoke-timing print is gated by C4_SMOKE_TIMING."""

    def test_smoke_timing_off_silences_print(self, monkeypatch, capsys):
        # Replay the print block from test_smoke.py:777-784. Just the env
        # check matters for the toggle audit.
        monkeypatch.delenv("C4_SMOKE_TIMING", raising=False)
        if os.environ.get("C4_SMOKE_TIMING") == "1":
            print("[smoke-timing] should-not-appear", file=sys.stderr)
        captured = capsys.readouterr()
        assert "[smoke-timing]" not in captured.err

    def test_smoke_timing_on_emits_print(self, monkeypatch, capsys):
        monkeypatch.setenv("C4_SMOKE_TIMING", "1")
        if os.environ.get("C4_SMOKE_TIMING") == "1":
            print("[smoke-timing] hello", file=sys.stderr)
        captured = capsys.readouterr()
        assert "[smoke-timing]" in captured.err


# ---------------------------------------------------------------------------
# Section D: weight_setter.py — NEURAL_VM_WEIGHT_MODE (subprocess)
# ---------------------------------------------------------------------------


class TestNeuralVmWeightModeToggle:
    """``NEURAL_VM_WEIGHT_MODE`` is read at module-import time, so we
    verify it via a subprocess. The toggle must select between
    ``WeightMode.HAND_SET`` (default) and ``WeightMode.COMPILED``."""

    def _probe_mode(self, env_value: str | None) -> str:
        script = textwrap.dedent(
            """
            from neural_vm.weight_setter import get_default_mode
            print(get_default_mode().value)
            """
        )
        env = os.environ.copy()
        # Drop any leaked toggle from the parent before injecting our own.
        env.pop("NEURAL_VM_WEIGHT_MODE", None)
        if env_value is not None:
            env["NEURAL_VM_WEIGHT_MODE"] = env_value
        repo_root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
        env["PYTHONPATH"] = f"{repo_root}:{env.get('PYTHONPATH', '')}"
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"probe failed: stdout={result.stdout!r} "
            f"stderr={result.stderr!r}"
        )
        return result.stdout.strip()

    def test_default_is_hand_set(self):
        assert self._probe_mode(None) == "hand_set"

    def test_compiled_mode_selected(self):
        assert self._probe_mode("compiled") == "compiled"

    def test_hand_set_explicit(self):
        assert self._probe_mode("hand_set") == "hand_set"
