"""Unit tests for ``tools/lint_raw_ffn_rule.py``.

The lint is the V9 ratchet that keeps the building-blocks DSL the
source of truth for FFN weights. It walks ``c4_release/neural_vm/``
for ``FFNRule.constant_write(...)`` and ``FFNRule.gated_write(...)``
calls outside the DSL modules and fails CI when a non-baselined file
gains a raw call.

Two contracts:
  1. Current tree exits 0 (we are at-or-below the baseline shipped at
     the V9 commit).
  2. A planted raw ``FFNRule.gated_write(...)`` in an arbitrary file
     (passed via ``--path``) exits 1 with the standardized warning
     message.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


# Repo layout: ``<repo_root>/c4_release/tools/lint_raw_ffn_rule.py``.
# ``tests/`` and ``tools/`` are siblings under ``c4_release/``.
_THIS_FILE = Path(__file__).resolve()
_C4_RELEASE = _THIS_FILE.parent.parent
_REPO_ROOT = _C4_RELEASE.parent
_LINT_TOOL = _C4_RELEASE / "tools" / "lint_raw_ffn_rule.py"


def _run_lint(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Invoke the lint as a subprocess so we exercise the exit-code contract."""
    cmd = [sys.executable, str(_LINT_TOOL), *args]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(cwd or _REPO_ROOT),
    )


def test_lint_tool_exists() -> None:
    """The lint script must be importable as a CLI."""
    assert _LINT_TOOL.exists(), f"missing lint tool at {_LINT_TOOL}"


def test_lint_exit_zero_on_current_tree() -> None:
    """The current tree is at-or-below the V9 baseline; lint must exit 0.

    If this fails, either a raw ``FFNRule.constant_write(...)`` /
    ``gated_write(...)`` was added to a non-allow-listed file (you
    should migrate it to a ``building_blocks_dsl`` helper instead), or
    a baselined file grew (same fix). If you intentionally added a
    raw rule in a new module, add the path to ``_ALLOWED_FILES`` in
    ``tools/lint_raw_ffn_rule.py`` with a justification comment.
    """
    result = _run_lint(cwd=_REPO_ROOT)
    assert result.returncode == 0, (
        f"lint failed on current tree (exit {result.returncode}):\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "OK" in result.stdout, result.stdout


def test_lint_catches_planted_violation(tmp_path: Path) -> None:
    """A planted raw ``FFNRule.gated_write(...)`` must trigger exit 1."""
    planted = tmp_path / "planted_violation.py"
    planted.write_text(
        textwrap.dedent(
            """
            \"\"\"Planted violation for lint test.\"\"\"
            from c4_release.neural_vm.unified_compiler.ir import FFNRule

            def make_bad_rules():
                return [
                    FFNRule.gated_write(
                        name="bad",
                        writes={"FOO": 1.0},
                        conditions={"BAR": 1.0},
                        threshold=0.5,
                        gate="BAZ",
                        gate_weight=1.0,
                    ),
                    FFNRule.constant_write(
                        name="also_bad",
                        writes={"FOO": 1.0},
                        conditions={"BAR": 1.0},
                        threshold=0.5,
                    ),
                ]
            """
        )
    )
    result = _run_lint("--path", str(planted))
    assert result.returncode == 1, (
        f"lint should have failed on planted violation (exit {result.returncode}):\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "raw FFNRule constructor" in result.stdout, result.stdout
    # The output references the recommended migration target.
    assert "building_blocks_dsl helpers" in result.stdout, result.stdout
    # Both rules should be flagged.
    assert "gated_write" in result.stdout
    assert "constant_write" in result.stdout


def test_lint_clean_path_exits_zero(tmp_path: Path) -> None:
    """A file with no raw FFNRule calls must exit 0 via --path."""
    clean = tmp_path / "clean.py"
    clean.write_text(
        textwrap.dedent(
            """
            \"\"\"Clean DSL-style module.\"\"\"
            from c4_release.neural_vm.unified_compiler.building_blocks_dsl import (
                step_function_rule,
            )

            def make_good_rules():
                return [
                    step_function_rule(
                        name="good",
                        target_dim="FOO",
                        condition_dim="BAR",
                        threshold=0.5,
                    ),
                ]
            """
        )
    )
    result = _run_lint("--path", str(clean))
    assert result.returncode == 0, (
        f"lint should have passed on clean file (exit {result.returncode}):\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def test_lint_json_mode_emits_summary() -> None:
    """``--json`` returns a parseable summary; current tree has zero regressions."""
    import json

    result = _run_lint("--json", cwd=_REPO_ROOT)
    assert result.returncode == 0, (
        f"json mode unexpectedly failed:\n{result.stdout}\n{result.stderr}"
    )
    payload = json.loads(result.stdout)
    assert "total_raw_calls" in payload
    assert "regressions" in payload
    assert "new_files" in payload
    assert payload["regressions"] == []
    assert payload["new_files"] == []


def test_lint_dsl_modules_are_allow_listed() -> None:
    """Sanity check: the DSL files themselves are not flagged.

    The allow-list lives inside the lint module — if it ever drifts so
    that ``building_blocks_dsl.py`` or ``wide_alu_dsl.py`` lights up,
    the smoke matrix will hit every layer that uses them.
    """
    # Import the module under test directly to peek at the allow-list.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "lint_raw_ffn_rule", str(_LINT_TOOL)
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    allowed = module._ALLOWED_FILES  # type: ignore[attr-defined]
    assert "c4_release/neural_vm/unified_compiler/building_blocks_dsl.py" in allowed
    assert "c4_release/neural_vm/unified_compiler/wide_alu_dsl.py" in allowed
    assert "c4_release/neural_vm/unified_compiler/ir.py" in allowed


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
