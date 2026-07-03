"""Step 7 (IR_INCREMENTAL_IMPROVEMENTS.md): unified strict-mode compile verifier.

A single pass that runs every shipped strict-mode / IR-invariant check
the codebase has accumulated, and emits a unified JSON manifest with
per-check pass/fail. Replaces the scatter of one-off scripts
(``scan_static_claims.py``, ``tools/sweep_compare_ffn.py``,
``tools/sweep_compare_attn.py``) plus the in-band warnings now emitted
during ``compile_full_vm_dynamic(strict=True)``.

Why this tool exists
--------------------
Before Step 7 the strict-mode checks lived in 6+ separate places:

* ``compile_full_vm_dynamic(strict=True)`` — the B14 admission gate
  (``StrictModeUnschedulableError`` when ops can't be placed from declared
  deps alone); cycle members get a phase fallback inside the sealed SCC
  (``allow_sealed_cycles=True`` default) and everything OUTSIDE the SCC
  must place cleanly.
* The ``CrossStepReadWarning`` step-1 safety scan (commit ``ef6ef561``) —
  82 findings on today's lookup-mode corpus; promoted to errors in strict
  verify mode.
* ``decl_verifier.verify_claims_static`` — Mode A static claim drift
  (declared but-not-written / written but-not-declared cells).
* ``decl_verifier.verify_produces_consumes_dynamic`` — Mode B 1-instruction
  produces-liveness probe.
* ``decl_verifier.verify_postconditions`` /
  ``verify_step_idx_gating`` / ``verify_alibi_consistency`` —
  Tier B dynamic invariants.
* ``decl_verifier.audit_smoke_coverage`` /
  ``audit_spec_coverage`` /
  ``verify_compaction_safety`` — Tier C discoverability audits.
* ``tools/sweep_compare_ffn.py`` /
  ``tools/sweep_compare_attn.py`` /
  ``compare_symbolic_to_lowered_embedding`` — symbolic-vs-lowered
  byte-identity gates per FFN / attn / embedding rule.

Running them one-at-a-time costs ~3-5 minutes per check (each does its
own ``compile_full_vm_dynamic`` bake, ~70-90 s warm). This tool runs ONE
compile up front (the strict admission gate also fires during that
compile, capturing both the cycle-aware ``StrictModeUnschedulableError``
path and the cross-step warning stream), then hands ``(model, layout)``
to every dynamic check that accepts it, dropping the wall-clock from
~30 min to <2 min on a warm cache.

Output schema
-------------
JSON manifest (written under ``c4_release/.agent-logs/`` by default):

    {
      "schema_version": "step7-strict-verify-v1",
      "created_at": "<ISO-8601 timestamp>",
      "config": {<kwargs used by the compile>},
      "summary": {
        "total_checks": <int>,
        "checks_pass": <int>,
        "checks_fail": <int>,
        "checks_skip": <int>,
        "wall_clock_s": <float>
      },
      "checks": [
        {
          "name": "<check id>",
          "category": "admission" | "warning" | "static" | "dynamic" |
                       "discoverability" | "byte-identity",
          "status": "pass" | "fail" | "skip" | "error",
          "duration_s": <float>,
          "findings": <int>,
          "summary": "<short human-readable>",
          "detail": {<check-specific fields>}
        }
      ]
    }

Each check carries enough detail in the ``detail`` field for downstream
agents / dashboards to drill in without re-running the analysis.

CLI
---
By default the tool runs every consolidated check on the lookup-mode
op set (``alu_mode='lookup'``) and writes the manifest to
``c4_release/.agent-logs/strict_verify_manifest.json`` (path overridable
via ``--out``). The sweep byte-identity gates (FFN / attn) are heavyweight
and are gated behind ``--include-sweeps`` — without that flag, the tool
records them as ``status='skip'`` with the reason
``'sweep gated; pass --include-sweeps to run'`` so the manifest is
fully populated even on the fast path.

Exit code is 0 on all-pass, 1 on any ``fail`` / ``error``, 2 on
``compile_failed`` (in which case downstream checks are all
``status='skip'`` with reason ``'compile_failed'``).
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import json
import os
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make the package importable when invoked as ``python tools/compile_strict_verify.py``
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Schema for individual check results.
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    name: str
    category: str
    status: str = "skip"  # pass | fail | skip | error
    duration_s: float = 0.0
    findings: int = 0
    summary: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)


def _now_iso() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Compile + per-check runners. Each runner is wrapped in ``_run_check`` so
# uncaught exceptions degrade to ``status='error'`` rather than aborting the
# whole pass.
# ---------------------------------------------------------------------------


def _run_check(name: str, category: str, fn) -> CheckResult:
    """Wrap a check callable with timing + exception capture."""
    t0 = time.time()
    try:
        result = fn()
    except Exception as exc:  # noqa: BLE001 — checks intentionally isolated
        traceback.print_exc()
        return CheckResult(
            name=name,
            category=category,
            status="error",
            duration_s=time.time() - t0,
            summary=f"uncaught: {type(exc).__name__}: {str(exc).splitlines()[0]}",
            detail={"traceback": traceback.format_exc()},
        )
    if not isinstance(result, CheckResult):
        return CheckResult(
            name=name,
            category=category,
            status="error",
            duration_s=time.time() - t0,
            summary="check returned non-CheckResult",
        )
    if result.duration_s == 0.0:
        result.duration_s = time.time() - t0
    return result


# ---------------------------------------------------------------------------
# Compile bookkeeping
# ---------------------------------------------------------------------------


def _compile_with_captures(
    alu_mode: str,
    enable_conversational_io: bool,
    enable_tool_calling: bool,
    n_heads: int,
    disk_cache: bool,
) -> Tuple[Optional[object], Optional[object], List[str], Optional[str]]:
    """Run ``compile_full_vm_dynamic(strict=True)`` and capture warnings.

    Returns ``(model, layout, cross_step_warnings, admission_error)``.

    * ``cross_step_warnings`` is the list of ``CrossStepReadWarning``
      messages emitted during compile (each message names the consumer op,
      the SSA read, and the same-step writer set).
    * ``admission_error`` is the formatted ``StrictModeUnschedulableError``
      message if strict admission failed, else ``None``.
    """
    from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        CrossStepReadWarning,
        StrictModeUnschedulableError,
        compile_full_vm_dynamic,
    )

    captured: List[str] = []

    with warnings.catch_warnings():
        warnings.simplefilter("always", CrossStepReadWarning)

        def _capture(message, category, filename, lineno, file=None, line=None):  # noqa: ARG001
            if issubclass(category, CrossStepReadWarning):
                captured.append(str(message))

        warnings.showwarning = _capture

        try:
            model, layout = compile_full_vm_dynamic(
                S=100.0,
                alu_mode=alu_mode,
                enable_conversational_io=enable_conversational_io,
                enable_tool_calling=enable_tool_calling,
                n_heads=n_heads,
                disk_cache=disk_cache,
                strict=True,
                allow_sealed_cycles=True,
            )
            return model, layout, captured, None
        except StrictModeUnschedulableError as exc:
            return None, None, captured, str(exc)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check_admission(admission_error: Optional[str]) -> CheckResult:
    """Strict admission gate: clean if compile succeeded with strict=True."""
    if admission_error is None:
        return CheckResult(
            name="strict_mode_admission",
            category="admission",
            status="pass",
            findings=0,
            summary=(
                "compile_full_vm_dynamic(strict=True, "
                "allow_sealed_cycles=True) accepted the op set"
            ),
            detail={
                "allow_sealed_cycles": True,
                "error": None,
            },
        )
    return CheckResult(
        name="strict_mode_admission",
        category="admission",
        status="fail",
        findings=1,
        summary="strict admission gate rejected one or more ops",
        detail={
            "allow_sealed_cycles": True,
            "error": admission_error,
        },
    )


def _check_cross_step(warnings_captured: List[str]) -> CheckResult:
    """Step-1 safety: CrossStepReadWarning findings from compile.

    The check is informational (status='pass' even with findings) — the
    historical baseline on lookup-mode is ~82 warnings and several of
    them are intentional back-edges. The detailed list is preserved in
    ``detail.warnings`` so a downstream agent / dashboard can sort by
    SSA read and prioritise the unaudited cases.

    Promotion to ``status='fail'`` happens via the ``--cross-step-strict``
    flag (see ``main``), which lets CI lock in a baseline count and fail
    when the warning count regresses upward.
    """
    return CheckResult(
        name="cross_step_safety_warnings",
        category="warning",
        status="pass",
        findings=len(warnings_captured),
        summary=f"{len(warnings_captured)} CrossStepReadWarning emissions captured",
        detail={
            "warnings": warnings_captured[:200],  # cap for manifest size
            "truncated": len(warnings_captured) > 200,
        },
    )


def _check_claims_static(
    *, alu_mode: str, enable_conversational_io: bool, enable_tool_calling: bool,
    n_heads: int,
) -> CheckResult:
    """Mode A: ``verify_claims_static``."""
    from neural_vm.verification.decl_verifier import verify_claims_static

    report = verify_claims_static(
        alu_mode=alu_mode,
        enable_conversational_io=enable_conversational_io,
        enable_tool_calling=enable_tool_calling,
        n_heads=n_heads,
    )
    n_results = len(report.results)
    n_with_errors = sum(1 for r in report.results if not r.ok)
    n_inert = sum(1 for r in report.results if r.inert)
    errors_by_op = report.errors_by_op()
    return CheckResult(
        name="verify_claims_static",
        category="static",
        status="pass" if n_with_errors == 0 else "fail",
        findings=n_with_errors,
        summary=(
            f"static claim drift: {n_with_errors}/{n_results} ops "
            f"(inert={n_inert})"
        ),
        detail={
            "n_results": n_results,
            "n_with_errors": n_with_errors,
            "n_inert": n_inert,
            "ops_with_errors": sorted(errors_by_op.keys())[:50],
            "ops_with_errors_truncated": len(errors_by_op) > 50,
        },
    )


def _check_produces_consumes_dynamic(
    *, alu_mode: str, enable_conversational_io: bool, n_heads: int,
) -> CheckResult:
    """Mode B: ``verify_produces_consumes_dynamic``."""
    from neural_vm.verification.decl_verifier import (
        verify_produces_consumes_dynamic,
    )

    report = verify_produces_consumes_dynamic(
        alu_mode=alu_mode,
        enable_conversational_io=enable_conversational_io,
        n_heads=n_heads,
    )
    n_results = len(report.results)
    drift_ops = [r.op_name for r in report.results if r.drift]
    return CheckResult(
        name="verify_produces_consumes_dynamic",
        category="dynamic",
        status="pass" if not drift_ops else "fail",
        findings=len(drift_ops),
        summary=(
            f"dynamic produces/consumes drift: {len(drift_ops)}/{n_results} ops"
        ),
        detail={
            "n_results": n_results,
            "ops_with_drift": drift_ops[:50],
            "ops_with_drift_truncated": len(drift_ops) > 50,
        },
    )


def _check_alibi_consistency(model, layout) -> CheckResult:
    from neural_vm.verification.decl_verifier import verify_alibi_consistency

    report = verify_alibi_consistency(model=model, layout=layout)
    findings = len(report.entries)
    return CheckResult(
        name="verify_alibi_consistency",
        category="dynamic",
        status="pass" if findings == 0 else "fail",
        findings=findings,
        summary=(
            f"alibi-slope consistency: {findings} mismatches across "
            f"{report.n_layers_inspected} layers"
        ),
        detail={
            "n_layers_inspected": report.n_layers_inspected,
            "n_ops_with_alibi": report.n_ops_with_alibi,
            "entries": [
                {
                    "layer_idx": e.layer_idx,
                    "head_idx": e.head_idx,
                    "kind": e.kind,
                    "declared": e.declared,
                    "actual": e.actual,
                    "owners": e.owners,
                }
                for e in report.entries[:50]
            ],
            "notes": report.notes[:20],
        },
    )


def _check_postconditions(model, layout) -> CheckResult:
    from neural_vm.verification.decl_verifier import verify_postconditions

    report = verify_postconditions(model=model, layout=layout)
    findings = len(report.drift)
    return CheckResult(
        name="verify_postconditions",
        category="dynamic",
        status="pass" if findings == 0 else "fail",
        findings=findings,
        summary=f"postcondition drift: {findings} entries",
        detail={
            "n_steps": report.n_steps,
            "drift": [
                {
                    "op_name": e.op_name,
                    "cell": e.cell,
                    "invariant": e.invariant,
                    "step": e.step,
                    "observed": e.observed,
                }
                for e in report.drift[:50]
            ],
            "notes": report.notes[:20],
        },
    )


def _check_step_idx_gating(model, layout) -> CheckResult:
    from neural_vm.verification.decl_verifier import verify_step_idx_gating

    report = verify_step_idx_gating(model=model, layout=layout)
    findings = len(report.drift)
    return CheckResult(
        name="verify_step_idx_gating",
        category="dynamic",
        status="pass" if findings == 0 else "fail",
        findings=findings,
        summary=f"step_idx gating drift: {findings} entries",
        detail={
            "n_steps": report.n_steps,
            "drift": [
                {
                    "op_name": e.op_name,
                    "step": e.step,
                    "kind": e.kind,
                    "dim": e.dim,
                    "position": e.position,
                    "observed": e.observed,
                }
                for e in report.drift[:50]
            ],
            "notes": report.notes[:20],
        },
    )


def _check_produces_consumes_multistep(model, layout) -> CheckResult:
    from neural_vm.verification.decl_verifier import (
        verify_produces_consumes_multistep,
    )

    report = verify_produces_consumes_multistep(model=model, layout=layout)
    drift_ops = [r.op_name for r in report.results if r.drift]
    return CheckResult(
        name="verify_produces_consumes_multistep",
        category="dynamic",
        status="pass" if not drift_ops else "fail",
        findings=len(drift_ops),
        summary=(
            f"multistep produces drift: {len(drift_ops)}/"
            f"{len(report.results)} ops"
        ),
        detail={
            "n_results": len(report.results),
            "ops_with_drift": drift_ops[:50],
            "n_steps": report.n_steps,
        },
    )


def _check_smoke_coverage() -> CheckResult:
    from neural_vm.verification.decl_verifier import audit_smoke_coverage

    report = audit_smoke_coverage()
    findings = len(report.untested_ops)
    return CheckResult(
        name="audit_smoke_coverage",
        category="discoverability",
        status="pass" if findings == 0 else "fail",
        findings=findings,
        summary=f"{findings} ops with no smoke_tests declaration",
        detail={
            "ops_declared": len(report.coverage),
            "untested_ops": report.untested_ops[:50],
            "untested_ops_truncated": len(report.untested_ops) > 50,
        },
    )


def _check_spec_coverage() -> CheckResult:
    from neural_vm.verification.decl_verifier import audit_spec_coverage

    report = audit_spec_coverage()
    findings = len(report.undocumented_ops)
    return CheckResult(
        name="audit_spec_coverage",
        category="discoverability",
        status="pass" if findings == 0 else "fail",
        findings=findings,
        summary=f"{findings} ops with no spec_section declaration",
        detail={
            "all_sections_count": len(report.all_sections),
            "covered_sections_count": len(report.coverage),
            "unreferenced_sections_count": len(report.unreferenced_sections),
            "undocumented_ops": report.undocumented_ops[:50],
            "undocumented_ops_truncated": len(report.undocumented_ops) > 50,
        },
    )


def _check_compaction_safety() -> CheckResult:
    from neural_vm.verification.decl_verifier import verify_compaction_safety

    report = verify_compaction_safety()
    findings = len(report.mismatches)
    return CheckResult(
        name="verify_compaction_safety",
        category="discoverability",
        status="pass" if findings == 0 else "fail",
        findings=findings,
        summary=(
            f"compaction_safe mismatches: {findings} "
            f"(partition_unavailable={report.partition_unavailable})"
        ),
        detail={
            "n_declared_safe": len(report.declared_safe),
            "n_declared_unsafe": len(report.declared_unsafe),
            "mismatches": report.mismatches,
            "partition_unavailable": report.partition_unavailable,
        },
    )


def _check_sweep_ffn(layout, *, include_sweeps: bool) -> CheckResult:
    """``compare_symbolic_to_lowered_ffn`` corpus sweep.

    Heavyweight (multi-minute) — gated behind ``--include-sweeps`` so
    the default fast path stays under 2 min wall-clock. When skipped,
    records the gating reason so the manifest stays fully populated.
    """
    if not include_sweeps:
        return CheckResult(
            name="sweep_compare_ffn",
            category="byte-identity",
            status="skip",
            findings=0,
            summary="sweep gated; pass --include-sweeps to run",
            detail={"reason": "sweep gated; pass --include-sweeps to run"},
        )

    # Defer imports so the fast path doesn't pay the cost.
    from tools.sweep_compare_ffn import _enumerate_ops, _sweep_op, _aggregate
    from tests._per_op_audit import compile_compact_layout

    compact_layout = compile_compact_layout()
    dim_positions = compact_layout.dim_positions
    head_dim = getattr(layout, "head_dim", 64)
    ops_and_flags = _enumerate_ops()
    rows = []
    for op, flags in ops_and_flags:
        rows.append(_sweep_op(op, flags, dim_positions, head_dim))
    agg = _aggregate(rows)
    n_real_bug = agg.get("total_real_bug", 0)
    return CheckResult(
        name="sweep_compare_ffn",
        category="byte-identity",
        status="pass" if n_real_bug == 0 else "fail",
        findings=n_real_bug,
        summary=(
            f"FFN sweep: clean={agg.get('total_clean', 0)} "
            f"real_bug={n_real_bug} "
            f"mismatch_only={agg.get('total_mismatch_only', 0)} "
            f"with_rules={agg.get('total_ops_with_ffn_rules', 0)}"
        ),
        detail={"aggregate": agg},
    )


def _check_sweep_attn(layout, *, include_sweeps: bool) -> CheckResult:
    """``compare_symbolic_to_lowered_attn`` corpus sweep (gated)."""
    if not include_sweeps:
        return CheckResult(
            name="sweep_compare_attn",
            category="byte-identity",
            status="skip",
            findings=0,
            summary="sweep gated; pass --include-sweeps to run",
            detail={"reason": "sweep gated; pass --include-sweeps to run"},
        )

    from tools.sweep_compare_attn import (
        _enumerate_ops, collect_attn_heads, run_sweep, build_dim_positions,
    )

    dim_positions = build_dim_positions()
    ops_with_flags = _enumerate_ops()
    heads = collect_attn_heads(ops_with_flags, dim_positions)
    rows, totals = run_sweep(heads)
    n_real_bug = int(totals.get("real_bug", 0))
    return CheckResult(
        name="sweep_compare_attn",
        category="byte-identity",
        status="pass" if n_real_bug == 0 else "fail",
        findings=n_real_bug,
        summary=(
            f"attn sweep: clean={totals.get('clean', 0)} "
            f"real_bug={n_real_bug} "
            f"mismatch_only={totals.get('mismatch_only', 0)}"
        ),
        detail={"totals": dict(totals)},
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_strict_verify(
    *,
    alu_mode: str = "lookup",
    enable_conversational_io: bool = False,
    enable_tool_calling: bool = False,
    n_heads: int = 8,
    include_sweeps: bool = False,
    include_dynamic_multistep: bool = True,
    disk_cache: bool = True,
    cross_step_strict_baseline: Optional[int] = None,
) -> Dict[str, Any]:
    """Run every consolidated strict-mode check and build the unified manifest.

    Args:
      alu_mode: ``'lookup'`` (default) or ``'efficient'``. Selects which
        ALU corpus is compiled.
      enable_conversational_io / enable_tool_calling: flags forwarded to
        the compile + the dynamic verifiers; default off (matches the
        production smoke baseline).
      n_heads: attention head count for the compile.
      include_sweeps: when True, runs the multi-minute
        ``compare_symbolic_to_lowered_ffn`` / ``_attn`` corpus sweeps.
        Default False keeps the fast path under ~2 min on a warm cache.
      include_dynamic_multistep: when True, runs
        ``verify_produces_consumes_multistep`` (~30 s). Default True.
      disk_cache: forwarded to ``compile_full_vm_dynamic``. Set False to
        force a cold compile (useful for CI freshness checks).
      cross_step_strict_baseline: when set, the cross-step safety check
        is promoted from ``pass`` to ``fail`` if findings exceed the
        baseline (lets CI lock the 82-warning floor).

    Returns the manifest dict (also suitable for ``json.dumps``).
    """
    t_start = time.time()
    manifest: Dict[str, Any] = {
        "schema_version": "step7-strict-verify-v1",
        "created_at": _now_iso(),
        "config": {
            "alu_mode": alu_mode,
            "enable_conversational_io": enable_conversational_io,
            "enable_tool_calling": enable_tool_calling,
            "n_heads": n_heads,
            "include_sweeps": include_sweeps,
            "include_dynamic_multistep": include_dynamic_multistep,
            "disk_cache": disk_cache,
            "cross_step_strict_baseline": cross_step_strict_baseline,
        },
        "summary": {},
        "checks": [],
    }

    # ---- THE compile. Captures both admission and cross-step warnings.
    t_compile0 = time.time()
    print(f"[strict-verify t={time.time() - t_start:.1f}s] compiling "
          f"(alu_mode={alu_mode}, n_heads={n_heads})...", flush=True)
    model, layout, cross_step_warnings, admission_error = _compile_with_captures(
        alu_mode=alu_mode,
        enable_conversational_io=enable_conversational_io,
        enable_tool_calling=enable_tool_calling,
        n_heads=n_heads,
        disk_cache=disk_cache,
    )
    compile_duration = time.time() - t_compile0
    print(
        f"[strict-verify t={time.time() - t_start:.1f}s] compile "
        f"{'OK' if admission_error is None else 'FAILED'} "
        f"in {compile_duration:.1f}s; "
        f"cross_step_warnings={len(cross_step_warnings)}",
        flush=True,
    )

    # Always-present checks (run even on compile failure).
    admission = _check_admission(admission_error)
    admission.duration_s = compile_duration
    manifest["checks"].append(asdict(admission))

    cross_step = _check_cross_step(cross_step_warnings)
    if cross_step_strict_baseline is not None:
        if cross_step.findings > cross_step_strict_baseline:
            cross_step.status = "fail"
            cross_step.summary = (
                f"cross-step warnings ({cross_step.findings}) exceed "
                f"baseline ({cross_step_strict_baseline})"
            )
            cross_step.detail["baseline_exceeded"] = True
            cross_step.detail["baseline"] = cross_step_strict_baseline
    manifest["checks"].append(asdict(cross_step))

    # If compile failed, every downstream check is skipped.
    if admission_error is not None:
        for name, category in [
            ("verify_claims_static", "static"),
            ("verify_produces_consumes_dynamic", "dynamic"),
            ("verify_alibi_consistency", "dynamic"),
            ("verify_postconditions", "dynamic"),
            ("verify_step_idx_gating", "dynamic"),
            ("verify_produces_consumes_multistep", "dynamic"),
            ("audit_smoke_coverage", "discoverability"),
            ("audit_spec_coverage", "discoverability"),
            ("verify_compaction_safety", "discoverability"),
            ("sweep_compare_ffn", "byte-identity"),
            ("sweep_compare_attn", "byte-identity"),
        ]:
            manifest["checks"].append(asdict(CheckResult(
                name=name, category=category,
                status="skip",
                summary="skipped: compile_failed",
                detail={"reason": "compile_failed"},
            )))
        manifest["summary"] = _summarise(manifest, t_start)
        return manifest

    # ---- Downstream checks. Each prints its phase to stdout so a stalled
    # check is obvious from the log.
    sequence = [
        ("verify_claims_static", "static",
         lambda: _check_claims_static(
             alu_mode=alu_mode,
             enable_conversational_io=enable_conversational_io,
             enable_tool_calling=enable_tool_calling,
             n_heads=n_heads)),
        ("verify_produces_consumes_dynamic", "dynamic",
         lambda: _check_produces_consumes_dynamic(
             alu_mode=alu_mode,
             enable_conversational_io=enable_conversational_io,
             n_heads=n_heads)),
        ("verify_alibi_consistency", "dynamic",
         lambda: _check_alibi_consistency(model, layout)),
        ("verify_postconditions", "dynamic",
         lambda: _check_postconditions(model, layout)),
        ("verify_step_idx_gating", "dynamic",
         lambda: _check_step_idx_gating(model, layout)),
        ("audit_smoke_coverage", "discoverability",
         _check_smoke_coverage),
        ("audit_spec_coverage", "discoverability",
         _check_spec_coverage),
        ("verify_compaction_safety", "discoverability",
         _check_compaction_safety),
    ]
    if include_dynamic_multistep:
        sequence.append((
            "verify_produces_consumes_multistep", "dynamic",
            lambda: _check_produces_consumes_multistep(model, layout),
        ))
    sequence.append((
        "sweep_compare_ffn", "byte-identity",
        lambda: _check_sweep_ffn(layout, include_sweeps=include_sweeps),
    ))
    sequence.append((
        "sweep_compare_attn", "byte-identity",
        lambda: _check_sweep_attn(layout, include_sweeps=include_sweeps),
    ))

    for name, category, fn in sequence:
        print(f"[strict-verify t={time.time() - t_start:.1f}s] running {name}...",
              flush=True)
        result = _run_check(name, category, fn)
        manifest["checks"].append(asdict(result))
        print(
            f"[strict-verify t={time.time() - t_start:.1f}s] {name}: "
            f"{result.status} findings={result.findings} "
            f"({result.duration_s:.1f}s)",
            flush=True,
        )

    manifest["summary"] = _summarise(manifest, t_start)
    return manifest


def _summarise(manifest: Dict[str, Any], t_start: float) -> Dict[str, Any]:
    checks = manifest["checks"]
    by_status: Dict[str, int] = {"pass": 0, "fail": 0, "skip": 0, "error": 0}
    for c in checks:
        by_status[c["status"]] = by_status.get(c["status"], 0) + 1
    return {
        "total_checks": len(checks),
        "checks_pass": by_status["pass"],
        "checks_fail": by_status["fail"],
        "checks_skip": by_status["skip"],
        "checks_error": by_status["error"],
        "wall_clock_s": round(time.time() - t_start, 2),
    }


def _default_out_path() -> Path:
    out_dir = _PROJECT_ROOT / ".agent-logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "strict_verify_manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Step 7 unified strict-mode compile verifier. Runs every "
            "shipped IR invariant check in one pass and writes a JSON "
            "manifest with per-check pass/fail."
        ),
    )
    parser.add_argument(
        "--alu-mode", default="lookup", choices=("lookup", "efficient"),
        help="ALU corpus to compile (default: lookup).",
    )
    parser.add_argument(
        "--enable-conversational-io", action="store_true",
        help="Forward enable_conversational_io=True to the compile.",
    )
    parser.add_argument(
        "--enable-tool-calling", action="store_true",
        help="Forward enable_tool_calling=True to the compile.",
    )
    parser.add_argument(
        "--n-heads", type=int, default=8,
        help="Attention head count (default: 8).",
    )
    parser.add_argument(
        "--include-sweeps", action="store_true",
        help=(
            "Run the heavyweight FFN / attn corpus sweeps "
            "(adds several minutes; default skips them with a recorded reason)."
        ),
    )
    parser.add_argument(
        "--skip-multistep", action="store_true",
        help="Skip verify_produces_consumes_multistep (~30 s).",
    )
    parser.add_argument(
        "--no-disk-cache", action="store_true",
        help="Force cold compile (disable disk_cache).",
    )
    parser.add_argument(
        "--cross-step-baseline", type=int, default=None,
        help=(
            "Lock the cross-step warning baseline. Findings strictly above "
            "this number flip the check from pass to fail (no effect on "
            "pass/fail otherwise)."
        ),
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help=(
            "Path to write the JSON manifest (default: "
            "c4_release/.agent-logs/strict_verify_manifest.json)."
        ),
    )
    args = parser.parse_args()

    manifest = run_strict_verify(
        alu_mode=args.alu_mode,
        enable_conversational_io=args.enable_conversational_io,
        enable_tool_calling=args.enable_tool_calling,
        n_heads=args.n_heads,
        include_sweeps=args.include_sweeps,
        include_dynamic_multistep=not args.skip_multistep,
        disk_cache=not args.no_disk_cache,
        cross_step_strict_baseline=args.cross_step_baseline,
    )

    out_path = args.out or _default_out_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2))
    print(f"\n[strict-verify] manifest written to {out_path}")

    summary = manifest["summary"]
    print(
        f"[strict-verify] total={summary['total_checks']} "
        f"pass={summary['checks_pass']} "
        f"fail={summary['checks_fail']} "
        f"skip={summary['checks_skip']} "
        f"error={summary['checks_error']} "
        f"wall={summary['wall_clock_s']}s"
    )

    if summary["checks_fail"] > 0 or summary["checks_error"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
