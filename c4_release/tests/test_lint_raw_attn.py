"""Unit tests for ``tools/lint_raw_attn.py``.

The lint is the attention-side ratchet (sibling of
``tools/lint_raw_ffn_rule.py``) that keeps
``DeclarativeAttentionHeadSpec`` + ``Primitives.generate_attention_head``
the source of truth for attention weights. It walks
``c4_release/neural_vm/`` for imperative ``attn.W_q/W_k/W_v/W_o[...] =``
projection writes and ``attn.alibi_slopes[...] =`` / ``.fill_(...)``
slope writes outside the allow-listed DSL modules, failing CI when a
non-baselined file gains a raw write.

Contracts:
  1. The current tree exits 0 (at-or-below the 2026-06-11 baseline).
  2. A planted raw ``attn.W_q[...] = ...`` / ``attn.alibi_slopes[...] =
     ...`` in an arbitrary file (via ``--path``) exits 1 with the
     standardized migration message.
  3. ``--json`` returns a parseable summary with zero regressions.
  4. The DSL / primitives modules are allow-listed.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


_THIS_FILE = Path(__file__).resolve()
_C4_RELEASE = _THIS_FILE.parent.parent
_REPO_ROOT = _C4_RELEASE.parent
_LINT_TOOL = _C4_RELEASE / "tools" / "lint_raw_attn.py"


def _run_lint(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(_LINT_TOOL), *args]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(cwd or _REPO_ROOT),
    )


def test_lint_tool_exists() -> None:
    assert _LINT_TOOL.exists(), f"missing lint tool at {_LINT_TOOL}"


def test_lint_exit_zero_on_current_tree() -> None:
    """The current tree is at-or-below the 2026-06-11 baseline; exit 0.

    If this fails, either an imperative ``attn.W_*[...] = ...`` /
    ``attn.alibi_slopes[...] = ...`` write was added to a non-allow-listed
    file (migrate it to ``DeclarativeAttentionHeadSpec`` /
    ``Primitives.generate_attention_head`` and carry the slope in
    ``spec.alibi_slope``), or a baselined file grew (same fix). If a new
    module legitimately needs raw writes, add it to ``_ALLOWED_FILES`` in
    ``tools/lint_raw_attn.py`` with a justification comment.
    """
    result = _run_lint(cwd=_REPO_ROOT)
    assert result.returncode == 0, (
        f"lint failed on current tree (exit {result.returncode}):\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "OK" in result.stdout, result.stdout


def test_lint_catches_planted_projection_write(tmp_path: Path) -> None:
    """A planted raw ``attn.W_q[...] = ...`` must trigger exit 1."""
    planted = tmp_path / "planted_proj.py"
    planted.write_text(
        textwrap.dedent(
            """
            \"\"\"Planted projection-write violation for lint test.\"\"\"
            def bake_bad_head(attn):
                base = 0
                attn.W_q.data[base, 1] = 5.0
                attn.W_k[base, 2] = 5.0
                attn.W_v.data[base + 1, 3] = 1.0
                attn.W_o.data[4, base + 1] = 1.0
            """
        )
    )
    result = _run_lint("--path", str(planted))
    assert result.returncode == 1, (
        f"lint should have failed on planted projection write "
        f"(exit {result.returncode}):\n{result.stdout}\n{result.stderr}"
    )
    assert "raw imperative" in result.stdout, result.stdout
    assert "DeclarativeAttentionHeadSpec" in result.stdout, result.stdout
    # All four projection kinds are flagged.
    for kind in ("W_q[]", "W_k[]", "W_v[]", "W_o[]"):
        assert kind in result.stdout, f"{kind} not flagged:\n{result.stdout}"


def test_lint_catches_planted_slope_write(tmp_path: Path) -> None:
    """Planted ``attn.alibi_slopes[...] = ...`` and ``.fill_`` exit 1."""
    planted = tmp_path / "planted_slope.py"
    planted.write_text(
        textwrap.dedent(
            """
            \"\"\"Planted slope-write violation for lint test.\"\"\"
            def bake_bad_slope(attn):
                attn.alibi_slopes.fill_(0.5)
                attn.alibi_slopes[3] = 5.0
                attn.alibi_slopes.data[4] = 1.0
            """
        )
    )
    result = _run_lint("--path", str(planted))
    assert result.returncode == 1, (
        f"lint should have failed on planted slope write "
        f"(exit {result.returncode}):\n{result.stdout}\n{result.stderr}"
    )
    assert "alibi_slopes[]" in result.stdout, result.stdout
    assert "alibi_slopes.fill_" in result.stdout, result.stdout


def test_lint_clean_path_exits_zero(tmp_path: Path) -> None:
    """A file with no raw attention writes must exit 0 via --path."""
    clean = tmp_path / "clean.py"
    clean.write_text(
        textwrap.dedent(
            """
            \"\"\"Clean declarative-style module.\"\"\"
            from c4_release.neural_vm.unified_compiler.primitives import (
                DeclarativeAttentionHeadSpec, AP, AO,
            )

            def make_good_spec():
                return DeclarativeAttentionHeadSpec(
                    head_idx=0,
                    q=(AP(0, 1, 5.0),),
                    k=(AP(0, 2, 5.0),),
                    alibi_slope=0.5,
                )
            """
        )
    )
    result = _run_lint("--path", str(clean))
    assert result.returncode == 0, (
        f"lint should have passed on clean file (exit {result.returncode}):\n"
        f"{result.stdout}\n{result.stderr}"
    )


def test_lint_json_mode_emits_summary() -> None:
    """``--json`` returns a parseable summary; current tree has zero regressions."""
    import json

    result = _run_lint("--json", cwd=_REPO_ROOT)
    assert result.returncode == 0, (
        f"json mode unexpectedly failed:\n{result.stdout}\n{result.stderr}"
    )
    payload = json.loads(result.stdout)
    assert "total_raw_writes" in payload
    assert "regressions" in payload
    assert "new_files" in payload
    assert payload["regressions"] == []
    assert payload["new_files"] == []


def test_lint_dsl_modules_are_allow_listed() -> None:
    """The spec-lowering primitive + DSL modules must not be flagged."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "lint_raw_attn", str(_LINT_TOOL)
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    allowed = module._ALLOWED_FILES  # type: ignore[attr-defined]
    assert "c4_release/neural_vm/unified_compiler/primitives.py" in allowed
    assert "c4_release/neural_vm/unified_compiler/building_blocks_dsl.py" in allowed
    assert "c4_release/neural_vm/unified_compiler/ir.py" in allowed
    assert "c4_release/neural_vm/attention_head_allocator.py" in allowed


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
