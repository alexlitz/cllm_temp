"""Unit tests for ``tools/lint_position_role.py``.

The lint enforces the architectural directive that compute rules fire
at ``MARK_SE`` (step-end), token-emit rules fire at ``MARK_X`` /
``BYTE_INDEX_*``, and memory-read rules fire at ``MARK_AX`` only inside
LI/LC sub-steps.

Contracts mirrored by ``test_lint_raw_ffn_rule.py``:

  1. Current tree exits 0 (we are at-or-below the baseline shipped at
     the lint-introduction commit).
  2. Planted violations of each intent class flip the exit code.
  3. Compliant rules exit 0 via ``--path``.
  4. The baseline ratchet refuses a growing count for an existing file.
  5. The baseline ratchet refuses a new (non-baselined) file.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


_THIS_FILE = Path(__file__).resolve()
_C4_RELEASE = _THIS_FILE.parent.parent
_REPO_ROOT = _C4_RELEASE.parent
_LINT_TOOL = _C4_RELEASE / "tools" / "lint_position_role.py"


def _run_lint(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(_LINT_TOOL), *args]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(cwd or _REPO_ROOT),
    )


def _load_lint_module():
    spec = importlib.util.spec_from_file_location(
        "lint_position_role", str(_LINT_TOOL)
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 1. CLI existence + speed
# ---------------------------------------------------------------------------


def test_lint_tool_exists() -> None:
    assert _LINT_TOOL.exists(), f"missing lint tool at {_LINT_TOOL}"


def test_lint_runs_under_2s() -> None:
    """The lint must complete in <2s on the production tree."""
    import time

    start = time.monotonic()
    result = _run_lint(cwd=_REPO_ROOT)
    elapsed = time.monotonic() - start
    assert result.returncode in (0, 1), (
        f"lint exited unexpectedly (rc={result.returncode}):\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert elapsed < 2.0, (
        f"lint took {elapsed:.2f}s, over the 2s budget"
    )


# ---------------------------------------------------------------------------
# 2. Baseline contract — current tree exits 0
# ---------------------------------------------------------------------------


def test_lint_exit_zero_on_current_tree() -> None:
    """The current tree is at-or-below the captured baseline; lint must exit 0.

    If this fails, either a rule with COMPUTE / TOKEN_EMIT / MEMORY_READ
    intent gained an inappropriate position gate (the architectural
    drift this lint exists to catch), or a baselined file's count grew.
    """
    result = _run_lint(cwd=_REPO_ROOT)
    assert result.returncode == 0, (
        f"lint failed on current tree (exit {result.returncode}):\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "OK" in result.stdout, result.stdout


def test_lint_reports_baseline_total() -> None:
    """JSON mode reports baseline_total + per-file violation counts."""
    result = _run_lint("--json", cwd=_REPO_ROOT)
    assert result.returncode == 0, (
        f"json mode unexpectedly failed:\n{result.stdout}\n{result.stderr}"
    )
    payload = json.loads(result.stdout)
    assert "total_violations" in payload
    assert "regressions" in payload
    assert "new_files" in payload
    assert "baseline_total" in payload
    # Current tree's count must match baseline (ratchet held).
    assert payload["regressions"] == []
    assert payload["new_files"] == []
    # Baseline is non-trivial — captured the existing migration target list.
    assert payload["baseline_total"] >= 1, (
        f"baseline collapsed unexpectedly: {payload}"
    )


# ---------------------------------------------------------------------------
# 3. Planted-violation tests (one per intent class)
# ---------------------------------------------------------------------------


def test_lint_catches_compute_at_mark_ax(tmp_path: Path) -> None:
    """A COMPUTE-intent rule gated on MARK_AX (no MARK_SE) must fail."""
    planted = tmp_path / "compute_at_mark_ax.py"
    planted.write_text(
        textwrap.dedent(
            """
            \"\"\"Planted COMPUTE-at-MARK_AX violation.\"\"\"
            from c4_release.neural_vm.unified_compiler.building_blocks_dsl import (
                multi_way_and_rule,
            )

            def make_bad():
                return multi_way_and_rule(
                    name="lX_alu_add_a0_b0",
                    conditions=(("MARK_AX", 1.0),),
                    threshold=0.5,
                    writes=(("OUTPUT_LO+0", 1.0),),
                )
            """
        )
    )
    result = _run_lint("--path", str(planted))
    assert result.returncode == 1, (
        f"lint should have failed on planted COMPUTE-at-MARK_AX (exit "
        f"{result.returncode}):\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "COMPUTE" in result.stdout
    assert "MARK_AX" in result.stdout
    assert "MARK_SE" in result.stdout


def test_lint_catches_token_emit_at_mark_se(tmp_path: Path) -> None:
    """A TOKEN_EMIT-intent rule gated only on MARK_SE must fail.

    TOKEN_EMIT rules must fire at a token-position marker
    (``MARK_AX``/``MARK_PC``/...) or a ``BYTE_INDEX_*`` flag. Gating
    only on ``MARK_SE``/``HAS_SE`` means the rule has no token
    position to write at.
    """
    planted = tmp_path / "token_emit_at_mark_se.py"
    planted.write_text(
        textwrap.dedent(
            """
            \"\"\"Planted TOKEN_EMIT-at-MARK_SE violation.\"\"\"
            from c4_release.neural_vm.unified_compiler.building_blocks_dsl import (
                multi_way_and_rule,
            )

            def make_bad():
                return multi_way_and_rule(
                    name="lX_byte_passthrough_alpha",
                    conditions=(("HAS_SE", 1.0),),
                    threshold=0.5,
                    writes=(("OUTPUT_LO+0", 1.0),),
                )
            """
        )
    )
    result = _run_lint("--path", str(planted))
    assert result.returncode == 1, (
        f"lint should have failed on planted TOKEN_EMIT-at-MARK_SE "
        f"(exit {result.returncode}):\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "TOKEN_EMIT" in result.stdout
    assert (
        "MARK_SE" in result.stdout or "HAS_SE" in result.stdout
    ), result.stdout


def test_lint_catches_memory_read_at_mark_ax_without_guard(
    tmp_path: Path,
) -> None:
    """A MEMORY_READ rule at MARK_AX without OP_LI/OP_LC/MARK_MEM guard fails."""
    planted = tmp_path / "memory_read_at_mark_ax.py"
    planted.write_text(
        textwrap.dedent(
            """
            \"\"\"Planted MEMORY_READ-at-MARK_AX-no-guard violation.\"\"\"
            from c4_release.neural_vm.unified_compiler.building_blocks_dsl import (
                multi_way_and_rule,
            )

            def make_bad():
                return multi_way_and_rule(
                    name="lX_memory_lookup_addr0",
                    conditions=(("MARK_AX", 1.0),),
                    threshold=0.5,
                    writes=(("OUTPUT_LO+0", 1.0),),
                )
            """
        )
    )
    result = _run_lint("--path", str(planted))
    assert result.returncode == 1, (
        f"lint should have failed on planted MEMORY_READ-at-MARK_AX-"
        f"no-guard (exit {result.returncode}):\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "MEMORY_READ" in result.stdout


# ---------------------------------------------------------------------------
# 4. Compliant rules pass
# ---------------------------------------------------------------------------


def test_lint_compute_at_mark_se_passes(tmp_path: Path) -> None:
    """A COMPUTE rule gated on MARK_SE is the canonical pattern; must pass."""
    clean = tmp_path / "compute_at_mark_se.py"
    clean.write_text(
        textwrap.dedent(
            """
            \"\"\"Compliant COMPUTE-at-MARK_SE rule.\"\"\"
            from c4_release.neural_vm.unified_compiler.building_blocks_dsl import (
                multi_way_and_rule,
            )

            def make_good():
                return multi_way_and_rule(
                    name="lX_alu_add_se_a0_b0",
                    conditions=(("MARK_SE", 1.0),),
                    threshold=0.5,
                    writes=(("OUTPUT_LO+0", 1.0),),
                )
            """
        )
    )
    result = _run_lint("--path", str(clean))
    assert result.returncode == 0, (
        f"lint should have passed on compliant COMPUTE-at-MARK_SE "
        f"(exit {result.returncode}):\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def test_lint_memory_read_at_mark_ax_with_guard_passes(tmp_path: Path) -> None:
    """A MEMORY_READ rule with OP_LI guard at MARK_AX is allowed."""
    clean = tmp_path / "memory_read_with_guard.py"
    clean.write_text(
        textwrap.dedent(
            """
            \"\"\"Compliant MEMORY_READ-at-MARK_AX-with-OP_LI-guard rule.\"\"\"
            from c4_release.neural_vm.unified_compiler.building_blocks_dsl import (
                multi_way_and_rule,
            )

            def make_good():
                return multi_way_and_rule(
                    name="lX_memory_lookup_li_addr0",
                    conditions=(("MARK_AX", 1.0), ("OP_LI", 1.0)),
                    threshold=1.5,
                    writes=(("OUTPUT_LO+0", 1.0),),
                )
            """
        )
    )
    result = _run_lint("--path", str(clean))
    assert result.returncode == 0, (
        f"lint should have passed on guarded MEMORY_READ rule "
        f"(exit {result.returncode}):\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def test_lint_clean_path_with_unrecognised_intent_passes(
    tmp_path: Path,
) -> None:
    """A rule whose name doesn't match any intent pattern is ignored."""
    clean = tmp_path / "unrelated.py"
    clean.write_text(
        textwrap.dedent(
            """
            \"\"\"Name has no intent pattern -> lint ignores it.\"\"\"
            from c4_release.neural_vm.unified_compiler.building_blocks_dsl import (
                multi_way_and_rule,
            )

            def make_neutral():
                return multi_way_and_rule(
                    name="lX_misc_helper_unit",
                    conditions=(("MARK_AX", 1.0),),
                    threshold=0.5,
                    writes=(("OUTPUT_LO+0", 1.0),),
                )
            """
        )
    )
    result = _run_lint("--path", str(clean))
    assert result.returncode == 0, (
        f"lint should have ignored un-classified rule (exit "
        f"{result.returncode}):\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# 5. Baseline ratchet tests (mirrors cheat-lint style)
# ---------------------------------------------------------------------------


def test_baseline_ratchet_blocks_new_files() -> None:
    """A new file with a violation is a regression even when baseline allows 0.

    This is the cheat-lint-style baseline non-increase check: any file
    that is NOT in ``_BASELINE`` AND contains a violation must flip the
    exit code.
    """
    mod = _load_lint_module()
    # Synthesise a fake scan result for a non-baselined file.
    fake_hit = mod._Hit(  # type: ignore[attr-defined]
        lineno=10,
        name="lZ_alu_add_a0_b0",
        intent="COMPUTE",
        reason="planted",
        position_dims={"MARK_AX"},
    )
    fake_hits = {
        "c4_release/neural_vm/unified_compiler/ops/lZ_ops.py": [fake_hit],
    }
    regressions, new_files = mod.diff_against_baseline(  # type: ignore[attr-defined]
        fake_hits
    )
    assert regressions == []
    assert new_files == [
        "c4_release/neural_vm/unified_compiler/ops/lZ_ops.py"
    ], new_files


def test_baseline_ratchet_blocks_growing_file() -> None:
    """A baselined file whose count grew is a regression."""
    mod = _load_lint_module()
    # Pick the first real baselined file and synthesise N+1 hits.
    baseline = mod._BASELINE  # type: ignore[attr-defined]
    assert baseline, "expected a non-empty baseline"
    target = sorted(baseline.keys())[0]
    n = baseline[target]
    fake_hit = mod._Hit(  # type: ignore[attr-defined]
        lineno=10,
        name="lX_alu_add_a0_b0",
        intent="COMPUTE",
        reason="planted",
        position_dims={"MARK_AX"},
    )
    fake_hits = {target: [fake_hit] * (n + 1)}
    regressions, new_files = mod.diff_against_baseline(  # type: ignore[attr-defined]
        fake_hits
    )
    assert new_files == []
    assert regressions == [(target, n, n + 1)], regressions


def test_baseline_ratchet_allows_shrinking_file() -> None:
    """A baselined file whose count *dropped* is OK."""
    mod = _load_lint_module()
    baseline = mod._BASELINE  # type: ignore[attr-defined]
    assert baseline, "expected a non-empty baseline"
    target = sorted(baseline.keys())[0]
    n = baseline[target]
    if n < 2:
        pytest.skip("baseline too small to shrink")
    fake_hit = mod._Hit(  # type: ignore[attr-defined]
        lineno=10,
        name="lX_alu_add_a0_b0",
        intent="COMPUTE",
        reason="planted",
        position_dims={"MARK_AX"},
    )
    fake_hits = {target: [fake_hit] * (n - 1)}
    regressions, new_files = mod.diff_against_baseline(  # type: ignore[attr-defined]
        fake_hits
    )
    assert regressions == []
    assert new_files == []


# ---------------------------------------------------------------------------
# 6. Intent-classification unit tests
# ---------------------------------------------------------------------------


def test_intent_classification() -> None:
    """``classify_intent`` returns the expected category for each pattern."""
    mod = _load_lint_module()
    classify = mod.classify_intent  # type: ignore[attr-defined]

    # COMPUTE patterns
    assert classify("l10_cmp_default_eq") == "COMPUTE"
    assert classify("l8_alu_add_lo_a0_b0") == "COMPUTE"
    assert classify("l10_bitwise_or_a0_b0") == "COMPUTE"
    assert classify("l9_carry_propagation_unit") == "COMPUTE"

    # TOKEN_EMIT patterns
    assert classify("l10_byte_passthrough_lo") == "TOKEN_EMIT"
    assert classify("l16_stack0_marker_from_alu_lo") == "TOKEN_EMIT"
    assert classify("layer3_emit_byte_idx_1") == "TOKEN_EMIT"

    # MEMORY_READ patterns
    assert classify("layer15_memory_lookup_li") == "MEMORY_READ"
    assert classify("layer5_fetch_anchor") == "MEMORY_READ"

    # Un-classified
    assert classify("layer3_misc") is None
    assert classify("") is None


def test_intent_precedence() -> None:
    """MEMORY_READ + TOKEN_EMIT patterns take precedence over COMPUTE.

    A name like ``lX_alu_lookup_*`` *reads memory* despite containing
    ``_alu_``; the lint must classify it as MEMORY_READ so the wrong
    set of position rules isn't applied.
    """
    mod = _load_lint_module()
    classify = mod.classify_intent  # type: ignore[attr-defined]
    assert classify("lX_alu_lookup_addr0") == "MEMORY_READ"
    assert classify("lX_alu_byte_passthrough_lo") == "TOKEN_EMIT"


# ---------------------------------------------------------------------------
# 7. Position-role decision unit tests
# ---------------------------------------------------------------------------


def test_position_role_compute_requires_step_end() -> None:
    """A COMPUTE rule with only MARK_AX is a violation; adding MARK_SE clears it."""
    mod = _load_lint_module()
    check = mod._is_position_role_violation  # type: ignore[attr-defined]

    assert check("COMPUTE", {"MARK_AX"}) is not None
    assert check("COMPUTE", {"MARK_AX", "MARK_SE"}) is None
    assert check("COMPUTE", {"MARK_SE_ONLY"}) is None
    assert check("COMPUTE", {"HAS_SE"}) is None
    # No position gate at all -> can't prove violation, pass through.
    assert check("COMPUTE", set()) is None


def test_position_role_memory_read_guard() -> None:
    """MEMORY_READ at MARK_AX is OK with OP_LI; bare MARK_AX is a violation."""
    mod = _load_lint_module()
    check = mod._is_position_role_violation  # type: ignore[attr-defined]

    assert check("MEMORY_READ", {"MARK_AX"}) is not None
    assert check("MEMORY_READ", {"MARK_AX", "OP_LI"}) is None
    assert check("MEMORY_READ", {"MARK_AX", "OP_LC"}) is None
    assert check("MEMORY_READ", {"MARK_AX", "MARK_MEM"}) is None


def test_position_role_token_emit_byte_index_ok() -> None:
    """TOKEN_EMIT at BYTE_INDEX_1 is OK; only MARK_SE without a token mark fails."""
    mod = _load_lint_module()
    check = mod._is_position_role_violation  # type: ignore[attr-defined]

    assert check("TOKEN_EMIT", {"BYTE_INDEX_1"}) is None
    assert check("TOKEN_EMIT", {"MARK_AX"}) is None
    assert check("TOKEN_EMIT", {"HAS_SE"}) is not None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
