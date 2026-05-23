#!/usr/bin/env python3
"""Focused neural-vs-declarative diagnostics for the 1096 suite.

This is intentionally opt-in because it builds/runs the neural VM.  Example:

    C4_1096_DIAG=1 C4_1096_OFFSET=0 C4_1096_LIMIT=8 \
    C4_BATCH_USE_KV_CACHE=0 C4_SPEC_K=0 \
    pytest -q c4_release/tests/test_1096_neural_declarative_diagnostic.py -s

The diagnostic uses declarative symbolic execution as the oracle for both the
expected exit value and the halt horizon.  Rows are printed only for programs
where neural execution diverges from declarative execution.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import sys
from typing import Iterable, List, Optional, Sequence, TextIO

import pytest


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compiler import compile_c
from tests.test_suite_1000 import generate_test_programs


def _parse_spec_k(raw: str) -> int:
    raw = (raw or "").strip().lower()
    if raw == "adaptive":
        return -1
    try:
        return int(raw)
    except ValueError:
        return -1


def _shorten(text: str, width: int = 72) -> str:
    text = " ".join(text.split())
    if len(text) <= width:
        return text
    return text[: width - 3] + "..."


@dataclass(frozen=True)
class NeuralDeclarativeDiagnosticRow:
    """One selected 1096 program compared across declarative and neural paths."""

    test_idx: int
    description: str
    suite_expected: int
    declarative_exit: Optional[int]
    declarative_steps: Optional[int]
    neural_exit: Optional[int]
    neural_output: str = ""
    error: Optional[str] = None

    @property
    def status(self) -> str:
        if self.error is not None:
            return "error"
        if self.declarative_exit != (self.suite_expected & 0xFFFFFFFF):
            return "suite/declarative-mismatch"
        if self.neural_exit != self.declarative_exit:
            return "neural-divergence"
        return "ok"

    def format(self) -> str:
        output = (
            f" output={self.neural_output!r}"
            if self.neural_output
            else ""
        )
        error = f" error={self.error}" if self.error else ""
        return (
            f"[1096-diag] id={self.test_idx:04d} "
            f"status={self.status} "
            f"desc={_shorten(self.description)!r} "
            f"expected={self.suite_expected & 0xFFFFFFFF} "
            f"decl={self.declarative_exit} "
            f"decl_steps={self.declarative_steps} "
            f"neural={self.neural_exit}"
            f"{output}{error}"
        )


def _selected_1096_tests(
    *,
    offset: int,
    limit: Optional[int],
) -> List[tuple[int, str, int, str]]:
    tests = generate_test_programs()
    selected = list(enumerate(tests))
    if offset:
        selected = selected[offset:]
    if limit is not None:
        selected = selected[:limit]
    return [
        (idx, source, expected, description)
        for idx, (source, expected, description) in selected
    ]


def run_1096_neural_declarative_diagnostic(
    *,
    offset: int = 0,
    limit: Optional[int] = 8,
    chunk_size: int = 8,
    spec_k: int = 0,
    max_context_window: int = 512,
    model_max_seq_len: int = 4096,
) -> List[NeuralDeclarativeDiagnosticRow]:
    """Run a focused 1096 slice and return declarative/neural comparison rows."""

    from neural_vm.batched_pure_neural import BatchedPureNeuralRunner
    from neural_vm.unified_compiler.symbolic_program import (
        SymbolicDeclarativeProgramRunner,
    )

    selected = _selected_1096_tests(offset=offset, limit=limit)
    symbolic_runner = SymbolicDeclarativeProgramRunner()
    neural_runner = BatchedPureNeuralRunner(max_seq_len=model_max_seq_len)
    rows: List[NeuralDeclarativeDiagnosticRow] = []

    for start in range(0, len(selected), chunk_size):
        chunk = selected[start : start + chunk_size]
        bytecodes = []
        data_list = []
        compiled_slots = []
        expected_steps = []

        for slot, (idx, source, expected, description) in enumerate(chunk):
            try:
                bytecode, data = compile_c(source)
                declarative = symbolic_runner.run(
                    bytecode,
                    data,
                    max_steps=None,
                )
            except Exception as exc:
                rows.append(
                    NeuralDeclarativeDiagnosticRow(
                        test_idx=idx,
                        description=description,
                        suite_expected=expected,
                        declarative_exit=None,
                        declarative_steps=None,
                        neural_exit=None,
                        error=f"compile/declarative error: {exc!r}",
                    )
                )
                continue

            decl_exit = declarative.ax if declarative.halted else None
            decl_steps = declarative.steps if declarative.halted else None
            if decl_steps is None:
                rows.append(
                    NeuralDeclarativeDiagnosticRow(
                        test_idx=idx,
                        description=description,
                        suite_expected=expected,
                        declarative_exit=decl_exit,
                        declarative_steps=None,
                        neural_exit=None,
                        error="declarative execution did not halt",
                    )
                )
                continue

            compiled_slots.append(
                (
                    slot,
                    idx,
                    expected,
                    description,
                    decl_exit,
                    decl_steps,
                )
            )
            bytecodes.append(bytecode)
            data_list.append(data)
            expected_steps.append(decl_steps)

        if not bytecodes:
            continue

        try:
            neural_results = neural_runner.run_batch(
                bytecodes,
                data_list=data_list,
                max_steps=None,
                expected_steps_list=expected_steps,
                max_context_window=max_context_window,
                spec_k=spec_k,
            )
        except Exception as exc:
            for (
                _slot,
                idx,
                expected,
                description,
                decl_exit,
                decl_steps,
            ) in compiled_slots:
                rows.append(
                    NeuralDeclarativeDiagnosticRow(
                        test_idx=idx,
                        description=description,
                        suite_expected=expected,
                        declarative_exit=decl_exit,
                        declarative_steps=decl_steps,
                        neural_exit=None,
                        error=f"neural batch error: {exc!r}",
                    )
                )
            continue

        for (
            slot,
            idx,
            expected,
            description,
            decl_exit,
            decl_steps,
        ), (neural_output, neural_exit) in zip(compiled_slots, neural_results):
            del slot
            rows.append(
                NeuralDeclarativeDiagnosticRow(
                    test_idx=idx,
                    description=description,
                    suite_expected=expected,
                    declarative_exit=decl_exit,
                    declarative_steps=decl_steps,
                    neural_exit=neural_exit,
                    neural_output=neural_output,
                )
            )

    return rows


def print_divergence_rows(
    rows: Iterable[NeuralDeclarativeDiagnosticRow],
    *,
    stream: TextIO = sys.stderr,
) -> int:
    """Print concise rows for non-matching entries and return the count."""

    count = 0
    for row in rows:
        if row.status == "ok":
            continue
        print(row.format(), file=stream, flush=True)
        count += 1
    return count


def test_diagnostic_row_format_is_concise():
    row = NeuralDeclarativeDiagnosticRow(
        test_idx=7,
        description="large ADD carry case",
        suite_expected=768,
        declarative_exit=768,
        declarative_steps=5,
        neural_exit=512,
    )

    text = row.format()

    assert "id=0007" in text
    assert "status=neural-divergence" in text
    assert "decl=768" in text
    assert "decl_steps=5" in text
    assert "neural=512" in text


def test_1096_neural_declarative_diagnostic_slice():
    if os.environ.get("C4_1096_DIAG") != "1":
        pytest.skip("set C4_1096_DIAG=1 to run the neural diagnostic")

    offset = int(os.environ.get("C4_1096_OFFSET", "0"))
    limit_raw = os.environ.get("C4_1096_LIMIT", "8")
    limit = None if limit_raw.lower() in {"", "none", "all"} else int(limit_raw)
    chunk_size = int(os.environ.get("C4_BATCH_CHUNK", "8"))
    spec_k = _parse_spec_k(os.environ.get("C4_SPEC_K", "0"))
    max_context_window = int(os.environ.get("C4_BATCH_CONTEXT_WINDOW", "512"))
    model_max_seq_len = int(os.environ.get("C4_BATCH_MODEL_MAX_SEQ_LEN", "4096"))

    rows = run_1096_neural_declarative_diagnostic(
        offset=offset,
        limit=limit,
        chunk_size=chunk_size,
        spec_k=spec_k,
        max_context_window=max_context_window,
        model_max_seq_len=model_max_seq_len,
    )
    divergences = print_divergence_rows(rows)

    if os.environ.get("C4_1096_DIAG_ASSERT", "1") != "0":
        assert divergences == 0, (
            f"{divergences}/{len(rows)} selected 1096 programs diverged"
        )
