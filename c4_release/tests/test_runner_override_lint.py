"""Unit tests for ``tools/lint_runner_overrides.py``.

The lint is the vanilla-thesis-restoration ratchet. It walks the
runner files (``run_vm.py``, ``batched_pure_neural.py``,
``fast_runner.py``, ``batch_runner.py``, ``batch_runner_v2.py``,
``transformer_first_runner.py``) for forbidden ALU-recovery,
IO-shim, shadow-memory, and per-opcode-branch patterns and fails CI
when any runner grows beyond the per-file baseline triple captured at
2026-06-09 from ``docs/VANILLA_RESTORE_INVENTORY_2026_06_09.md``.

Three contracts:
  1. Current tree exits 0 (we are at-or-below the baseline shipped at
     this commit).
  2. After each Wave A-E removal, the per-file count is non-increasing
     (the ratchet itself enforces this — the test cross-checks that the
     CURRENT counts equal or undercut every baseline triple).
  3. A planted forbidden pattern (``_compute_alu_legacy``,
     ``_BINARY_POP_OPS``, ``if exec_op == Opcode.PSH``) in an arbitrary
     file (passed via ``--path``) exits 1.

The lint also has a hard <2s budget — slow lints get skipped in pre-commit.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest


# Repo layout: ``<repo_root>/c4_release/tools/lint_runner_overrides.py``.
_THIS_FILE = Path(__file__).resolve()
_C4_RELEASE = _THIS_FILE.parent.parent
_REPO_ROOT = _C4_RELEASE.parent
_LINT_TOOL = _C4_RELEASE / "tools" / "lint_runner_overrides.py"


def _run_lint(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Invoke the lint as a subprocess so we exercise the exit-code contract."""
    cmd = [sys.executable, str(_LINT_TOOL), *args]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(cwd or _REPO_ROOT),
    )


def _load_lint_module():
    """Import the lint module directly so we can read its baseline table."""
    spec = importlib.util.spec_from_file_location(
        "lint_runner_overrides", str(_LINT_TOOL)
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lint_tool_exists() -> None:
    """The lint script must be importable as a CLI."""
    assert _LINT_TOOL.exists(), f"missing lint tool at {_LINT_TOOL}"


def test_lint_exit_zero_on_current_tree() -> None:
    """Current tree is at-or-below the 2026-06-09 baseline; lint must exit 0.

    If this fails, either a new forbidden pattern was added to a runner
    (you should remove it — see VANILLA_RESTORE_INVENTORY for the
    vanilla-thesis rule), or a Wave A-E removal landed but the
    baseline in ``tools/lint_runner_overrides.py:_BASELINE`` was not
    decremented in the same commit (decrement it).
    """
    result = _run_lint(cwd=_REPO_ROOT)
    assert result.returncode == 0, (
        f"lint failed on current tree (exit {result.returncode}):\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "OK" in result.stdout, result.stdout


def test_lint_runs_under_2_seconds() -> None:
    """Acceptance: lint must run in <2s (pre-commit budget)."""
    t0 = time.perf_counter()
    result = _run_lint(cwd=_REPO_ROOT)
    elapsed = time.perf_counter() - t0
    assert result.returncode == 0, result.stdout + result.stderr
    assert elapsed < 2.0, (
        f"lint took {elapsed:.2f}s (>2s budget). See "
        f"tools/lint_runner_overrides.py — pure-AST walk should stay well under."
    )


def test_lint_reports_current_count() -> None:
    """JSON mode emits ``total_forbidden`` so the count is observable."""
    result = _run_lint("--json", cwd=_REPO_ROOT)
    assert result.returncode == 0, (
        f"json mode unexpectedly failed:\n{result.stdout}\n{result.stderr}"
    )
    payload = json.loads(result.stdout)
    assert "total_forbidden" in payload
    assert "files" in payload
    assert "regressions" in payload
    assert payload["regressions"] == []
    # Sanity: at least the dirty runners contribute non-trivially.
    total = payload["total_forbidden"]
    assert total > 0, "expected non-zero baseline at 2026-06-09"
    # The two clean runners must be at zero.
    for clean in (
        "c4_release/neural_vm/fast_runner.py",
        "c4_release/neural_vm/batch_runner.py",
        "c4_release/neural_vm/batch_runner_v2.py",
        "c4_release/neural_vm/transformer_first_runner.py",
    ):
        assert sum(payload["files"][clean]["counts"]) == 0, (
            f"clean runner {clean} gained forbidden patterns: "
            f"{payload['files'][clean]['counts']}"
        )


def test_lint_count_non_increasing_vs_baseline() -> None:
    """Acceptance: per-file ``(calls, names, branches)`` must be <= baseline.

    The ratchet enforces this from inside the CLI but we cross-check
    explicitly so a Wave A-E commit that DECREMENTS the baseline below
    the current count is also caught (i.e. the baseline must match the
    current tree, not be aspirationally higher).
    """
    module = _load_lint_module()
    baseline = module._BASELINE
    result = _run_lint("--json", cwd=_REPO_ROOT)
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    for rel, info in payload["files"].items():
        b_calls, b_names, b_branches = baseline[rel]
        c_calls, c_names, c_branches = info["counts"]
        assert c_calls <= b_calls, (
            f"{rel}: forbidden CALL count {c_calls} > baseline {b_calls} "
            f"(regression)"
        )
        assert c_names <= b_names, (
            f"{rel}: forbidden NAME count {c_names} > baseline {b_names} "
            f"(regression)"
        )
        assert c_branches <= b_branches, (
            f"{rel}: per-opcode BRANCH count {c_branches} > baseline "
            f"{b_branches} (regression)"
        )


def test_lint_baseline_matches_current_tree() -> None:
    """The baseline must equal the current count exactly.

    If a Wave A-E commit reduces forbidden patterns but forgets to
    decrement the baseline, this test fails. That keeps the baseline a
    tight ratchet — counts strictly walk downward.
    """
    module = _load_lint_module()
    baseline = module._BASELINE
    result = _run_lint("--json", cwd=_REPO_ROOT)
    payload = json.loads(result.stdout)
    for rel, info in payload["files"].items():
        current_tuple = tuple(info["counts"])
        assert current_tuple == baseline[rel], (
            f"{rel}: baseline {baseline[rel]} does not match current "
            f"{current_tuple}. Decrement (or update) "
            f"tools/lint_runner_overrides.py:_BASELINE in the same commit "
            f"that removed the override."
        )


def test_lint_catches_planted_compute_alu_legacy(tmp_path: Path) -> None:
    """Planted ``self._compute_alu_legacy(...)`` must trigger exit 1."""
    planted = tmp_path / "planted_alu.py"
    planted.write_text(
        textwrap.dedent(
            """
            \"\"\"Planted violation: _compute_alu_legacy call.\"\"\"
            class FakeRunner:
                def step(self, op, stack_val, ax_val):
                    return self._compute_alu_legacy(op, stack_val, ax_val)
            """
        )
    )
    result = _run_lint("--path", str(planted))
    assert result.returncode == 1, (
        f"lint should have failed on planted _compute_alu_legacy "
        f"(exit {result.returncode}):\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "_compute_alu_legacy" in result.stdout
    assert "FORBIDDEN" in result.stdout


def test_lint_catches_planted_binary_pop_ops(tmp_path: Path) -> None:
    """Planted reference to ``_BINARY_POP_OPS`` must trigger exit 1."""
    planted = tmp_path / "planted_binary_pop.py"
    planted.write_text(
        textwrap.dedent(
            """
            \"\"\"Planted violation: _BINARY_POP_OPS reference.\"\"\"
            from c4_release.neural_vm.run_vm import _BINARY_POP_OPS

            def is_binary(op):
                return op in _BINARY_POP_OPS
            """
        )
    )
    result = _run_lint("--path", str(planted))
    assert result.returncode == 1, (
        f"lint should have failed on planted _BINARY_POP_OPS "
        f"(exit {result.returncode}):\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "_BINARY_POP_OPS" in result.stdout


def test_lint_catches_planted_per_opcode_branch(tmp_path: Path) -> None:
    """Planted ``if exec_op == Opcode.PSH:`` branch must trigger exit 1."""
    planted = tmp_path / "planted_branch.py"
    planted.write_text(
        textwrap.dedent(
            """
            \"\"\"Planted violation: per-opcode branch.\"\"\"
            class Opcode:
                PSH = 0
                JSR = 1
                ENT = 2

            def step(exec_op, ax):
                if exec_op == Opcode.PSH:
                    return ax - 8
                elif exec_op == Opcode.JSR:
                    return ax + 4
                return ax
            """
        )
    )
    result = _run_lint("--path", str(planted))
    assert result.returncode == 1, (
        f"lint should have failed on planted per-opcode branch "
        f"(exit {result.returncode}):\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "Opcode.PSH" in result.stdout
    assert "Opcode.JSR" in result.stdout


def test_lint_catches_planted_inject_getchar(tmp_path: Path) -> None:
    """Planted ``self._inject_getchar(context)`` must trigger exit 1."""
    planted = tmp_path / "planted_getchar.py"
    planted.write_text(
        textwrap.dedent(
            """
            \"\"\"Planted violation: _inject_getchar.\"\"\"
            class FakeRunner:
                def step(self, context):
                    self._inject_getchar(context)
            """
        )
    )
    result = _run_lint("--path", str(planted))
    assert result.returncode == 1
    assert "_inject_getchar" in result.stdout


def test_lint_catches_planted_set_mem_store_positions(tmp_path: Path) -> None:
    """Planted ``embed.set_mem_store_positions(...)`` must trigger exit 1."""
    planted = tmp_path / "planted_mem_pos.py"
    planted.write_text(
        textwrap.dedent(
            """
            \"\"\"Planted violation: set_mem_store_positions.\"\"\"
            def setup(model):
                model.embed.set_mem_store_positions([1, 2, 3])
            """
        )
    )
    result = _run_lint("--path", str(planted))
    assert result.returncode == 1
    assert "set_mem_store_positions" in result.stdout


def test_lint_catches_planted_last_pushed_value(tmp_path: Path) -> None:
    """Planted ``s.last_pushed_value = ...`` must trigger exit 1."""
    planted = tmp_path / "planted_last_pushed.py"
    planted.write_text(
        textwrap.dedent(
            """
            \"\"\"Planted violation: last_pushed_value tracking.\"\"\"
            def snapshot(s, prev_ax):
                s.last_pushed_value = int(prev_ax) & 0xFFFFFFFF
            """
        )
    )
    result = _run_lint("--path", str(planted))
    assert result.returncode == 1
    assert "last_pushed_value" in result.stdout


def test_lint_clean_path_exits_zero(tmp_path: Path) -> None:
    """A file with no forbidden patterns must exit 0 via --path."""
    clean = tmp_path / "clean_runner.py"
    clean.write_text(
        textwrap.dedent(
            """
            \"\"\"Pure forward-pass runner shape.\"\"\"
            def run(model, input_ids):
                while True:
                    logits = model.forward(input_ids)
                    next_tok = int(logits.argmax(-1))
                    input_ids = append(input_ids, next_tok)
                    if next_tok == EXIT_TOKEN:
                        break
                return input_ids
            """
        )
    )
    result = _run_lint("--path", str(clean))
    assert result.returncode == 0, (
        f"lint should have passed on clean file (exit {result.returncode}):\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "0 FORBIDDEN" in result.stdout


def test_lint_json_path_mode_emits_hits(tmp_path: Path) -> None:
    """``--path --json`` returns a parseable hit list."""
    planted = tmp_path / "planted.py"
    planted.write_text(
        textwrap.dedent(
            """
            class R:
                def step(self):
                    self._compute_alu_legacy(0, 1, 2)
                    self._override_ax_in_last_step(0)
            """
        )
    )
    result = _run_lint("--path", str(planted), "--json")
    assert result.returncode == 1
    payload = json.loads(result.stdout)
    labels = sorted(h["label"] for h in payload["hits"])
    assert "_compute_alu_legacy" in labels
    assert "_override_ax_in_last_step" in labels


def test_lint_clean_runners_are_at_zero() -> None:
    """The four runners marked CLEAN in the inventory must stay at zero.

    ``fast_runner.py``, ``batch_runner.py``, ``batch_runner_v2.py``,
    ``transformer_first_runner.py`` are the "target shape" — pure
    forward-pass wrappers. Their baseline triples are (0, 0, 0) so any
    regression there is caught by the standard ratchet, but we also
    assert it explicitly here so the message is unambiguous.
    """
    module = _load_lint_module()
    for clean in (
        "c4_release/neural_vm/fast_runner.py",
        "c4_release/neural_vm/batch_runner.py",
        "c4_release/neural_vm/batch_runner_v2.py",
        "c4_release/neural_vm/transformer_first_runner.py",
    ):
        assert module._BASELINE[clean] == (0, 0, 0), (
            f"{clean} must remain at (0,0,0); it is the target runner shape "
            f"per VANILLA_RESTORE_INVENTORY 'Notes on what's already clean'."
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
