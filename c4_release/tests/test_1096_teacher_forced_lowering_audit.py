#!/usr/bin/env python3
"""Opt-in lowering-only audit for 1096 suite slices.

This test does not run programs autoregressively.  It compiles each selected C
program, builds the symbolic declarative token stream, teacher-forces that
golden stream through the lowered neural model, and checks whether the final
head still supports each expected token.  It is meant to catch lowering/tail
overwrites before final-output 1096 runs compound the first bad token.

Environment knobs:

* ``C4_1096_LOWERING_AUDIT=1`` opts into the audit.
* ``C4_1096_OFFSET`` / ``C4_1096_LIMIT`` select the 1096 slice.
* ``C4_1096_LOWERING_ASSERT_MODE`` controls the gate:
  ``fatal`` (default) fails only errors plus register-result/step-boundary
  misses, ``all`` fails any support drift, and ``off`` only prints.
  ``C4_1096_LOWERING_ASSERT`` is accepted as a legacy alias with the same
  values; ``1`` means ``all`` and ``0`` means ``off``.
* ``C4_1096_LOWERING_MIN_MARGIN`` optionally requires a positive final-head
  margin for the symbolic token. The default is ``0``.
* ``C4_1096_LOWERING_OUTPUT_BAND_MIN_MARGIN`` enables structural
  ``OUTPUT_LO``/``OUTPUT_HI`` one-hot nibble contracts when set. Use
  ``C4_1096_LOWERING_OUTPUT_BAND_MAX_INACTIVE`` and
  ``C4_1096_LOWERING_OUTPUT_BAND_TOLERANCE`` to tune the band policy. The
  default max-inactive value is intentionally huge, making this a winner-margin
  contract rather than a one-hot-zero contract unless explicitly tightened.
* ``C4_1096_LOWERING_PRINT_MODE`` controls per-row output:
  ``drift`` (default), ``all``, ``fatal``, or ``none``. Info-only drift is
  still printed in the default fatal gate so the raw support misses remain
  visible without failing CI.
* ``C4_1096_LOWERING_MAX_TRACE_TOKENS``, ``C4_1096_LOWERING_MAX_FAILURES``,
  ``C4_1096_LOWERING_CONTEXT_WINDOW``, ``C4_1096_LOWERING_CHUNK_TOKENS``,
  and ``C4_1096_LOWERING_DETAIL_LIMIT`` cap trace size, bounded audit
  windows, captured raw failures per fatal/info class, and expensive
  block-by-block detail probes. Set the context window to ``off`` to force
  the legacy full-trace audit.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import sys
from typing import Optional, Sequence

import pytest


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_vm.batched_pure_neural import BatchedPureNeuralRunner  # noqa: E402
from neural_vm.verification.decl_verifier import (  # noqa: E402
    TeacherForcedTraceAuditFailure,
    TeacherForcedSymbolicTrace,
    audit_teacher_forced_trace_final_support,
    build_teacher_forced_symbolic_trace,
    verify_teacher_forced_token_support,
)
from src.compiler import compile_c  # noqa: E402
from tests.declarative_oracle import declarative_oracle_for_program  # noqa: E402
from tests.test_1096_neural_declarative_diagnostic import (  # noqa: E402
    _selected_1096_tests,
)


pytestmark = pytest.mark.lowering


@dataclass(frozen=True)
class ClassifiedLoweringAuditFailure:
    failure: TeacherForcedTraceAuditFailure
    severity: str
    category: str
    reason: str

    @property
    def fatal(self) -> bool:
        return self.severity == "fatal"

    def format(self) -> str:
        return (
            f"severity={self.severity} category={self.category} "
            f"{self.failure.format()} reason={self.reason}"
        )


_STEP_SLOT_NAMES = (
    "REG_PC",
    "PC_byte0",
    "PC_byte1",
    "PC_byte2",
    "PC_byte3",
    "REG_AX",
    "AX_byte0",
    "AX_byte1",
    "AX_byte2",
    "AX_byte3",
    "REG_SP",
    "SP_byte0",
    "SP_byte1",
    "SP_byte2",
    "SP_byte3",
    "REG_BP",
    "BP_byte0",
    "BP_byte1",
    "BP_byte2",
    "BP_byte3",
    "STACK0",
    "STACK0_byte0",
    "STACK0_byte1",
    "STACK0_byte2",
    "STACK0_byte3",
    "MEM",
    "MEM_addr0",
    "MEM_addr1",
    "MEM_addr2",
    "MEM_addr3",
    "MEM_value0",
    "MEM_value1",
    "MEM_value2",
    "MEM_value3",
    "STEP_END",
)
_REGISTER_RESULT_MARKER_SLOTS = frozenset({
    "REG_PC",
    "REG_AX",
    "REG_SP",
    "REG_BP",
    "STACK0",
})
_REGISTER_RESULT_BYTE_PREFIXES = (
    "PC_byte",
    "AX_byte",
    "SP_byte",
    "BP_byte",
    "STACK0_byte",
)
_STEP_BOUNDARY_SLOTS = frozenset({"STEP_END"})


def _is_register_result_slot(slot: str) -> bool:
    return (
        slot in _REGISTER_RESULT_MARKER_SLOTS
        or slot.startswith(_REGISTER_RESULT_BYTE_PREFIXES)
    )


def _is_fatal_lowering_slot(slot: str) -> bool:
    return _is_register_result_slot(slot) or slot in _STEP_BOUNDARY_SLOTS


def _classify_lowering_failure(
    failure: TeacherForcedTraceAuditFailure,
) -> ClassifiedLoweringAuditFailure:
    if failure.output_band_margin_only:
        return ClassifiedLoweringAuditFailure(
            failure=failure,
            severity="info",
            category="output-band-margin",
            reason=(
                "expected token and nibble winners match; only output-band "
                "active margin contract missed"
            ),
        )
    if failure.output_band_contract_only:
        return ClassifiedLoweringAuditFailure(
            failure=failure,
            severity="info",
            category="output-band-contract",
            reason=(
                "expected token and output byte match; only structural "
                "output-band contract missed"
            ),
        )
    if _is_register_result_slot(failure.slot):
        return ClassifiedLoweringAuditFailure(
            failure=failure,
            severity="fatal",
            category="register-result",
            reason="completed register row feeds runner architectural state",
        )
    if failure.slot in _STEP_BOUNDARY_SLOTS:
        return ClassifiedLoweringAuditFailure(
            failure=failure,
            severity="fatal",
            category="step-boundary",
            reason="step terminator controls dispatch and halt progress",
        )
    return ClassifiedLoweringAuditFailure(
        failure=failure,
        severity="info",
        category="support-drift",
        reason="teacher-forced support miss outside the fatal register gate",
    )


def _slot_name_for_token_index(*, prefix_len: int, token_index: int) -> str:
    generated_index = int(token_index) - int(prefix_len)
    if generated_index < 0:
        raise ValueError("token_index is before the generated trace")
    return _STEP_SLOT_NAMES[generated_index % len(_STEP_SLOT_NAMES)]


def _trace_token_indices_by_fatality(trace, *, fatal: bool) -> tuple[int, ...]:
    selected = []
    for token_index in range(trace.prefix_len, len(trace.context)):
        slot = _slot_name_for_token_index(
            prefix_len=trace.prefix_len,
            token_index=token_index,
        )
        if _is_fatal_lowering_slot(slot) == fatal:
            selected.append(token_index)
    return tuple(selected)


@dataclass(frozen=True)
class LoweringAuditRow:
    test_idx: int
    description: str
    suite_expected: int
    trace_len: int = 0
    checked: int = 0
    failures: Sequence[TeacherForcedTraceAuditFailure] = ()
    skipped: Optional[str] = None
    error: Optional[str] = None
    detail: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.skipped is None and not self.failures

    @property
    def classified_failures(self) -> tuple[ClassifiedLoweringAuditFailure, ...]:
        return tuple(_classify_lowering_failure(failure) for failure in self.failures)

    @property
    def fatal_failures(self) -> tuple[ClassifiedLoweringAuditFailure, ...]:
        return tuple(
            classified
            for classified in self.classified_failures
            if classified.fatal
        )

    @property
    def info_failures(self) -> tuple[ClassifiedLoweringAuditFailure, ...]:
        return tuple(
            classified
            for classified in self.classified_failures
            if not classified.fatal
        )

    @property
    def wrong_token_failures(self) -> tuple[ClassifiedLoweringAuditFailure, ...]:
        return tuple(
            classified
            for classified in self.classified_failures
            if classified.failure.token_is_wrong
        )

    @property
    def output_band_contract_only_failures(
        self,
    ) -> tuple[ClassifiedLoweringAuditFailure, ...]:
        return tuple(
            classified
            for classified in self.classified_failures
            if classified.failure.output_band_contract_only
        )

    @property
    def output_band_margin_failures(
        self,
    ) -> tuple[ClassifiedLoweringAuditFailure, ...]:
        return tuple(
            classified
            for classified in self.classified_failures
            if classified.failure.output_band_margin_only
        )

    @property
    def gate_ok(self) -> bool:
        return self.error is None and not self.fatal_failures

    def format(self) -> str:
        if self.error is not None:
            return (
                f"[1096-lowering] id={self.test_idx:04d} status=error "
                f"desc={self.description!r} error={self.error}"
            )
        if self.skipped is not None:
            return (
                f"[1096-lowering] id={self.test_idx:04d} status=skipped "
                f"desc={self.description!r} trace_len={self.trace_len} "
                f"reason={self.skipped}"
            )
        fatal = self.fatal_failures
        info = self.info_failures
        if fatal:
            status = "fatal-lowering-divergence"
        elif info and len(info) == len(self.output_band_margin_failures):
            status = "info-output-band-margin"
        elif info and len(info) == len(self.output_band_contract_only_failures):
            status = "info-output-band-contract"
        elif info:
            status = "info-support-drift"
        else:
            status = "ok"
        first_parts = []
        if fatal:
            first_parts.append(f"first_fatal={fatal[0].format()}")
        if info:
            first_parts.append(f"first_info={info[0].format()}")
        first = "" if not first_parts else " " + " ".join(first_parts)
        detail = "" if self.detail is None else f" detail={self.detail}"
        return (
            f"[1096-lowering] id={self.test_idx:04d} status={status} "
            f"desc={self.description!r} trace_len={self.trace_len} "
            f"checked={self.checked} failures={len(self.failures)} "
            f"fatal={len(fatal)} info={len(info)} "
            f"wrong_token={len(self.wrong_token_failures)} "
            f"band_only={len(self.output_band_contract_only_failures)} "
            f"band_margin_only={len(self.output_band_margin_failures)}"
            f"{first}{detail}"
        )


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _env_optional_float(name: str) -> Optional[float]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    normalized = raw.strip().lower()
    if normalized in {"none", "off", "false", "no"}:
        return None
    return float(raw)


def _env_optional_int(name: str, default: Optional[int]) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    normalized = raw.strip().lower()
    if normalized in {"none", "off", "false", "no"}:
        return None
    return int(raw)


def _maybe_empty_cuda_cache() -> None:
    try:
        import torch
    except Exception:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _lowering_assert_mode() -> str:
    raw = os.environ.get(
        "C4_1096_LOWERING_ASSERT_MODE",
        os.environ.get("C4_1096_LOWERING_ASSERT", "fatal"),
    )
    normalized = raw.strip().lower().replace("_", "-")
    if normalized in {"", "0", "false", "no", "off", "none"}:
        return "off"
    if normalized in {"fatal", "fatal-only", "gate"}:
        return "fatal"
    if normalized in {"1", "true", "yes", "on", "all", "full", "strict"}:
        return "all"
    raise ValueError(
        "C4_1096_LOWERING_ASSERT_MODE must be fatal, all, or off "
        f"(got {raw!r})"
    )


def _lowering_print_mode() -> str:
    raw = os.environ.get("C4_1096_LOWERING_PRINT_MODE", "drift")
    normalized = raw.strip().lower().replace("_", "-")
    if normalized in {"", "0", "false", "no", "off", "none"}:
        return "none"
    if normalized in {"drift", "failures", "non-ok"}:
        return "drift"
    if normalized in {"fatal", "gate"}:
        return "fatal"
    if normalized in {"1", "true", "yes", "on", "all"}:
        return "all"
    raise ValueError(
        "C4_1096_LOWERING_PRINT_MODE must be drift, all, fatal, or none "
        f"(got {raw!r})"
    )


def _should_print_lowering_row(row: LoweringAuditRow, mode: str) -> bool:
    if mode == "none":
        return False
    if mode == "all":
        return True
    if mode == "fatal":
        return row.error is not None or bool(row.fatal_failures)
    if mode == "drift":
        return row.error is not None or row.skipped is not None or not row.ok
    raise ValueError(f"unknown print mode {mode!r}")


def _detail_failure(
    failures: Sequence[TeacherForcedTraceAuditFailure],
) -> Optional[TeacherForcedTraceAuditFailure]:
    if not failures:
        return None
    classified = tuple(_classify_lowering_failure(failure) for failure in failures)
    for failure in classified:
        if failure.fatal:
            return failure.failure
    return classified[0].failure


def _confirmed_lowering_failures(
    *,
    model,
    trace: TeacherForcedSymbolicTrace,
    failures: Sequence[TeacherForcedTraceAuditFailure],
    min_margin: float,
    output_band_min_margin: Optional[float],
    output_band_max_inactive_value: float,
    output_band_tolerance: float,
    max_context_window: Optional[int],
) -> tuple[TeacherForcedTraceAuditFailure, ...]:
    """Filter fast audit candidates through the exact single-token verifier.

    The windowed fast audit batches many token positions under one remapped
    memory-store visibility set. That is fast, but it can over-report when a
    chunk spans positions with different visible stores. Only reported
    candidates pay for the exact verifier, so the failure list remains
    authoritative without throwing away the fast path.
    """

    confirmed: list[TeacherForcedTraceAuditFailure] = []
    for failure in failures:
        band_margin = (
            output_band_min_margin
            if (
                failure.output_band_contract_only
                or failure.output_band_margin_only
            )
            else None
        )
        support = verify_teacher_forced_token_support(
            model,
            trace.context,
            token_index=failure.token_index,
            prefix_len=trace.prefix_len,
            mem_store_positions=trace.mem_store_positions,
            mem_addr_src_positions=trace.mem_addr_src_positions,
            min_margin=min_margin,
            output_band_min_margin=band_margin,
            output_band_max_inactive_value=output_band_max_inactive_value,
            output_band_tolerance=output_band_tolerance,
            max_context_window=max_context_window,
            probe_name=f"confirm:{failure.step}:{failure.slot}",
        )
        if not support.supported:
            confirmed.append(failure)
    return tuple(confirmed)


def _run_1096_lowering_audit_slice(
    *,
    model,
    offset: int,
    limit: Optional[int],
    max_trace_tokens: int,
    max_failures_per_test: int,
    detail_limit: int,
    min_margin: float,
    output_band_min_margin: Optional[float],
    output_band_max_inactive_value: float,
    output_band_tolerance: float,
    max_context_window: Optional[int],
    window_chunk_tokens: int,
) -> list[LoweringAuditRow]:
    rows: list[LoweringAuditRow] = []
    detailed = 0
    for test_idx, source, expected, description in _selected_1096_tests(
        offset=offset,
        limit=limit,
    ):
        try:
            bytecode, data = compile_c(source)
            declarative = declarative_oracle_for_program(
                bytecode,
                data,
                suite_expected=expected,
                label=f"id={test_idx:04d}",
                max_steps=None,
            )
            if declarative.exit_code != (expected & 0xFFFFFFFF):
                rows.append(LoweringAuditRow(
                    test_idx=test_idx,
                    description=description,
                    suite_expected=expected,
                    error=(
                        "suite/declarative mismatch: "
                        f"expected={expected & 0xFFFFFFFF} "
                        f"decl={declarative.exit_code}"
                    ),
                ))
                continue

            trace = build_teacher_forced_symbolic_trace(bytecode, data)
            trace_len = len(trace.context)
            if trace_len > max_trace_tokens:
                rows.append(LoweringAuditRow(
                    test_idx=test_idx,
                    description=description,
                    suite_expected=expected,
                    trace_len=trace_len,
                    skipped=(
                        f"trace_len>{max_trace_tokens}; raise "
                        "C4_1096_LOWERING_MAX_TRACE_TOKENS for full audit"
                    ),
                ))
                continue

            fatal_report = audit_teacher_forced_trace_final_support(
                model,
                trace,
                token_indices=_trace_token_indices_by_fatality(
                    trace,
                    fatal=True,
                ),
                min_margin=min_margin,
                output_band_min_margin=None,
                output_band_max_inactive_value=output_band_max_inactive_value,
                output_band_tolerance=output_band_tolerance,
                max_context_window=max_context_window,
                window_chunk_tokens=window_chunk_tokens,
                max_failures=max_failures_per_test,
                probe_name=f"id={test_idx:04d}:fatal",
            )
            info_report = audit_teacher_forced_trace_final_support(
                model,
                trace,
                token_indices=_trace_token_indices_by_fatality(
                    trace,
                    fatal=False,
                ),
                min_margin=min_margin,
                output_band_min_margin=None,
                output_band_max_inactive_value=output_band_max_inactive_value,
                output_band_tolerance=output_band_tolerance,
                max_context_window=max_context_window,
                window_chunk_tokens=window_chunk_tokens,
                max_failures=max_failures_per_test,
                probe_name=f"id={test_idx:04d}:info",
            )
            band_only_failures: tuple[TeacherForcedTraceAuditFailure, ...] = ()
            if output_band_min_margin is not None:
                band_failure_cap = max(
                    max_failures_per_test * 4,
                    max_failures_per_test + 8,
                )
                fatal_band_report = audit_teacher_forced_trace_final_support(
                    model,
                    trace,
                    token_indices=_trace_token_indices_by_fatality(
                        trace,
                        fatal=True,
                    ),
                    min_margin=min_margin,
                    output_band_min_margin=output_band_min_margin,
                    output_band_max_inactive_value=(
                        output_band_max_inactive_value
                    ),
                    output_band_tolerance=output_band_tolerance,
                    max_context_window=max_context_window,
                    window_chunk_tokens=window_chunk_tokens,
                    max_failures=band_failure_cap,
                    probe_name=f"id={test_idx:04d}:fatal-band",
                )
                info_band_report = audit_teacher_forced_trace_final_support(
                    model,
                    trace,
                    token_indices=_trace_token_indices_by_fatality(
                        trace,
                        fatal=False,
                    ),
                    min_margin=min_margin,
                    output_band_min_margin=output_band_min_margin,
                    output_band_max_inactive_value=(
                        output_band_max_inactive_value
                    ),
                    output_band_tolerance=output_band_tolerance,
                    max_context_window=max_context_window,
                    window_chunk_tokens=window_chunk_tokens,
                    max_failures=band_failure_cap,
                    probe_name=f"id={test_idx:04d}:info-band",
                )
                band_only_failures = tuple(
                    failure
                    for failure in (
                        *fatal_band_report.failures,
                        *info_band_report.failures,
                    )
                    if failure.output_band_contract_only
                )[:max_failures_per_test]
            detail = None
            failures = (
                *fatal_report.failures,
                *info_report.failures,
                *band_only_failures,
            )
            failures = _confirmed_lowering_failures(
                model=model,
                trace=trace,
                failures=failures,
                min_margin=min_margin,
                output_band_min_margin=output_band_min_margin,
                output_band_max_inactive_value=output_band_max_inactive_value,
                output_band_tolerance=output_band_tolerance,
                max_context_window=max_context_window,
            )
            detail_failure = _detail_failure(failures)
            if detail_failure is not None and detailed < detail_limit:
                support = verify_teacher_forced_token_support(
                    model,
                    trace.context,
                    token_index=detail_failure.token_index,
                    prefix_len=trace.prefix_len,
                    mem_store_positions=trace.mem_store_positions,
                    mem_addr_src_positions=trace.mem_addr_src_positions,
                    min_margin=min_margin,
                    output_band_min_margin=output_band_min_margin,
                    output_band_max_inactive_value=output_band_max_inactive_value,
                    output_band_tolerance=output_band_tolerance,
                    max_context_window=max_context_window,
                    probe_name=(
                        f"id={test_idx:04d}:"
                        f"{detail_failure.step}:{detail_failure.slot}"
                    ),
                )
                detail = support.format()
                detailed += 1
            rows.append(LoweringAuditRow(
                test_idx=test_idx,
                description=description,
                suite_expected=expected,
                trace_len=trace_len,
                checked=fatal_report.checked + info_report.checked,
                failures=failures,
                detail=detail,
            ))
        except Exception as exc:
            if "out of memory" in str(exc).lower():
                _maybe_empty_cuda_cache()
            rows.append(LoweringAuditRow(
                test_idx=test_idx,
                description=description,
                suite_expected=expected,
                error=repr(exc),
            ))
    return rows


def test_1096_teacher_forced_lowering_audit_slice() -> None:
    if os.environ.get("C4_1096_LOWERING_AUDIT") != "1":
        pytest.skip("set C4_1096_LOWERING_AUDIT=1 to run lowering audit")

    offset = _env_int("C4_1096_OFFSET", 0)
    limit_raw = os.environ.get("C4_1096_LIMIT", "8").strip().lower()
    limit = None if limit_raw in {"", "none", "all"} else int(limit_raw)
    max_trace_tokens = _env_int("C4_1096_LOWERING_MAX_TRACE_TOKENS", 4096)
    max_failures_per_test = _env_int("C4_1096_LOWERING_MAX_FAILURES", 4)
    detail_limit = _env_int("C4_1096_LOWERING_DETAIL_LIMIT", 2)
    min_margin = _env_float("C4_1096_LOWERING_MIN_MARGIN", 0.0)
    output_band_min_margin = _env_optional_float(
        "C4_1096_LOWERING_OUTPUT_BAND_MIN_MARGIN"
    )
    output_band_max_inactive_value = _env_float(
        "C4_1096_LOWERING_OUTPUT_BAND_MAX_INACTIVE",
        1e30,
    )
    output_band_tolerance = _env_float(
        "C4_1096_LOWERING_OUTPUT_BAND_TOLERANCE",
        1e-6,
    )
    max_context_window = _env_optional_int(
        "C4_1096_LOWERING_CONTEXT_WINDOW",
        512,
    )
    window_chunk_tokens = _env_int("C4_1096_LOWERING_CHUNK_TOKENS", 512)
    assert_mode = _lowering_assert_mode()
    print_mode = _lowering_print_mode()

    runner = BatchedPureNeuralRunner(
        max_seq_len=_env_int("C4_BATCH_MODEL_MAX_SEQ_LEN", 4096)
    )
    rows = _run_1096_lowering_audit_slice(
        model=runner.model,
        offset=offset,
        limit=limit,
        max_trace_tokens=max_trace_tokens,
        max_failures_per_test=max_failures_per_test,
        detail_limit=detail_limit,
        min_margin=min_margin,
        output_band_min_margin=output_band_min_margin,
        output_band_max_inactive_value=output_band_max_inactive_value,
        output_band_tolerance=output_band_tolerance,
        max_context_window=max_context_window,
        window_chunk_tokens=window_chunk_tokens,
    )

    for row in rows:
        if _should_print_lowering_row(row, print_mode):
            print(row.format(), file=sys.stderr, flush=True)

    clean = sum(1 for row in rows if row.ok)
    skipped = sum(1 for row in rows if row.skipped is not None)
    errors = sum(1 for row in rows if row.error is not None)
    fatal_rows = sum(1 for row in rows if row.fatal_failures)
    info_only_rows = sum(
        1 for row in rows if row.info_failures and not row.fatal_failures
    )
    fatal_failures = sum(len(row.fatal_failures) for row in rows)
    info_failures = sum(len(row.info_failures) for row in rows)
    wrong_token_failures = sum(len(row.wrong_token_failures) for row in rows)
    band_only_failures = sum(
        len(row.output_band_contract_only_failures) for row in rows
    )
    band_margin_failures = sum(
        len(row.output_band_margin_failures) for row in rows
    )
    print(
        "[1096-lowering-summary] "
        f"selected={len(rows)} clean={clean} fatal_rows={fatal_rows} "
        f"info_only_rows={info_only_rows} fatal_failures={fatal_failures} "
        f"info_failures={info_failures} errors={errors} skipped={skipped} "
        f"wrong_token_failures={wrong_token_failures} "
        f"band_only_failures={band_only_failures} "
        f"band_margin_failures={band_margin_failures} "
        f"assert_mode={assert_mode} print_mode={print_mode} "
        f"min_margin={min_margin:g} "
        f"output_band_min_margin={output_band_min_margin} "
        f"output_band_max_inactive={output_band_max_inactive_value:g} "
        f"context_window={max_context_window} "
        f"chunk_tokens={window_chunk_tokens}",
        file=sys.stderr,
        flush=True,
    )

    if assert_mode == "fatal":
        assert fatal_rows == 0 and errors == 0, (
            "teacher-forced lowering audit fatal gate failed: "
            f"clean={clean} fatal_rows={fatal_rows} "
            f"fatal_failures={fatal_failures} info_only_rows={info_only_rows} "
            f"info_failures={info_failures} errors={errors} skipped={skipped}"
        )
    elif assert_mode == "all":
        assert (
            fatal_rows == 0
            and info_only_rows == 0
            and errors == 0
        ), (
            "teacher-forced lowering audit full support gate failed: "
            f"clean={clean} fatal_rows={fatal_rows} "
            f"fatal_failures={fatal_failures} info_only_rows={info_only_rows} "
            f"info_failures={info_failures} errors={errors} skipped={skipped}"
        )


def _audit_failure(
    slot: str,
    *,
    failure_kind: str = "wrong_argmax,wrong_output_byte",
    band_kinds: Sequence[str] = (),
) -> TeacherForcedTraceAuditFailure:
    return TeacherForcedTraceAuditFailure(
        token_index=123,
        step=1,
        slot=slot,
        expected_token=0xF8,
        argmax_token=0x00,
        expected_margin=-3.25,
        output_byte=0x00,
        failure_kind=failure_kind,
        output_band_violation_kinds=tuple(band_kinds),
    )


def test_lowering_audit_classifies_register_result_failures_as_fatal() -> None:
    classified = _classify_lowering_failure(_audit_failure("AX_byte0"))

    assert classified.fatal
    assert classified.category == "register-result"
    assert "step1:AX_byte0" in classified.format()


def test_lowering_audit_classifies_step_boundary_failures_as_fatal() -> None:
    classified = _classify_lowering_failure(_audit_failure("STEP_END"))

    assert classified.fatal
    assert classified.category == "step-boundary"


def test_lowering_audit_classifies_mem_support_misses_as_info() -> None:
    row = LoweringAuditRow(
        test_idx=7,
        description="probe",
        suite_expected=0,
        trace_len=80,
        checked=35,
        failures=(_audit_failure("MEM_addr0"),),
    )

    assert row.gate_ok
    assert not row.fatal_failures
    assert len(row.info_failures) == 1
    text = row.format()
    assert "status=info-support-drift" in text
    assert "fatal=0 info=1" in text
    assert "first_info=severity=info" in text
    assert "step1:MEM_addr0" in text


def test_lowering_audit_classifies_output_band_margin_only_as_info() -> None:
    row = LoweringAuditRow(
        test_idx=8,
        description="band margin",
        suite_expected=0,
        trace_len=80,
        checked=35,
        failures=(
            _audit_failure(
                "AX_byte0",
                failure_kind="output_band_contract",
                band_kinds=("active_margin_low",),
            ),
        ),
    )

    assert row.gate_ok
    assert not row.fatal_failures
    assert len(row.output_band_margin_failures) == 1
    text = row.format()
    assert "status=info-output-band-margin" in text
    assert "wrong_token=0" in text
    assert "band_margin_only=1" in text
    assert "category=output-band-margin" in text


def test_lowering_audit_fatal_token_selection_excludes_mem_slots() -> None:
    class Trace:
        prefix_len = 10
        context = tuple(range(prefix_len + len(_STEP_SLOT_NAMES)))

    fatal_indices = _trace_token_indices_by_fatality(Trace, fatal=True)
    info_indices = _trace_token_indices_by_fatality(Trace, fatal=False)

    assert 10 + _STEP_SLOT_NAMES.index("AX_byte0") in fatal_indices
    assert 10 + _STEP_SLOT_NAMES.index("STEP_END") in fatal_indices
    assert 10 + _STEP_SLOT_NAMES.index("MEM_addr0") not in fatal_indices
    assert 10 + _STEP_SLOT_NAMES.index("MEM_addr0") in info_indices


def test_lowering_audit_windowed_final_support_bounds_sequence_length() -> None:
    import torch
    import torch.nn as nn

    class EchoEmbed(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seen_shapes: list[tuple[int, int]] = []
            self._mem_history_end = 0
            self._mem_store_positions = None

        def set_mem_history_end(self, end) -> None:
            self._mem_history_end = end

        def set_mem_store_positions(self, positions) -> None:
            self._mem_store_positions = positions

        def forward(self, token_ids):
            batch, seq_len = token_ids.shape
            self.seen_shapes.append((batch, seq_len))
            x = torch.zeros(batch, seq_len, 384, device=token_ids.device)
            token_features = 64 + token_ids
            x.scatter_(2, token_features.unsqueeze(-1), 10.0)
            lo = token_ids.clamp(max=255) & 0xF
            hi = (token_ids.clamp(max=255) >> 4) & 0xF
            x.scatter_(2, lo.unsqueeze(-1), 1.0)
            x.scatter_(2, (16 + hi).unsqueeze(-1), 1.0)
            return x

    class EchoModel(nn.Module):
        vocab_size = 300
        dim_positions = {"OUTPUT_LO": 0, "OUTPUT_HI": 16}

        def __init__(self) -> None:
            super().__init__()
            self.embed = EchoEmbed()
            self.blocks = nn.ModuleList()
            self.head = nn.Linear(384, self.vocab_size, bias=False)
            self.head.weight.data.zero_()
            for token in range(self.vocab_size):
                self.head.weight.data[token, 64 + token] = 1.0

    model = EchoModel()
    prefix = (260, 261, 42)
    generated = tuple(42 for _ in range(180))
    trace = TeacherForcedSymbolicTrace(
        context=prefix + generated,
        prefix_len=len(prefix),
        steps=len(generated) // len(_STEP_SLOT_NAMES),
        exit_code=None,
        halted=False,
    )

    report = audit_teacher_forced_trace_final_support(
        model,
        trace,
        min_margin=0.0,
        max_context_window=20,
        window_chunk_tokens=17,
        probe_name="windowed-echo",
    )

    assert report.passed, report.format()
    assert report.checked == len(generated)
    assert len(model.embed.seen_shapes) > 1
    assert max(seq_len for _, seq_len in model.embed.seen_shapes) <= (
        trace.prefix_len + 20 + 17
    )


def test_lowering_audit_assert_mode_parser(monkeypatch) -> None:
    monkeypatch.delenv("C4_1096_LOWERING_ASSERT_MODE", raising=False)
    monkeypatch.delenv("C4_1096_LOWERING_ASSERT", raising=False)
    assert _lowering_assert_mode() == "fatal"

    monkeypatch.setenv("C4_1096_LOWERING_ASSERT_MODE", "all")
    assert _lowering_assert_mode() == "all"

    monkeypatch.setenv("C4_1096_LOWERING_ASSERT_MODE", "off")
    assert _lowering_assert_mode() == "off"

    monkeypatch.delenv("C4_1096_LOWERING_ASSERT_MODE", raising=False)
    monkeypatch.setenv("C4_1096_LOWERING_ASSERT", "1")
    assert _lowering_assert_mode() == "all"


def test_lowering_audit_optional_float_parser(monkeypatch) -> None:
    monkeypatch.delenv("C4_PROBE_FLOAT", raising=False)
    assert _env_optional_float("C4_PROBE_FLOAT") is None

    monkeypatch.setenv("C4_PROBE_FLOAT", "off")
    assert _env_optional_float("C4_PROBE_FLOAT") is None

    monkeypatch.setenv("C4_PROBE_FLOAT", "0.75")
    assert _env_optional_float("C4_PROBE_FLOAT") == pytest.approx(0.75)


def test_lowering_audit_optional_int_parser(monkeypatch) -> None:
    monkeypatch.delenv("C4_PROBE_INT", raising=False)
    assert _env_optional_int("C4_PROBE_INT", 512) == 512

    monkeypatch.setenv("C4_PROBE_INT", "off")
    assert _env_optional_int("C4_PROBE_INT", 512) is None

    monkeypatch.setenv("C4_PROBE_INT", "2048")
    assert _env_optional_int("C4_PROBE_INT", 512) == 2048
