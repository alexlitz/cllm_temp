"""Teacher-forced lowering probes for final neural byte support.

These tests are narrower than final-output 1096 scans.  They pick one
declarative symbolic token row, run the compiled model on the teacher-forced
prefix, and assert the final head still supports the symbolic byte after every
compiled block/tail has run.
"""

from __future__ import annotations

import os
import sys

import pytest


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_vm.verification.decl_verifier import (  # noqa: E402
    audit_teacher_forced_trace_final_support,
    build_teacher_forced_symbolic_trace,
    verify_teacher_forced_token_support,
)
from neural_vm.embedding import Opcode  # noqa: E402
from src.compiler import compile_c  # noqa: E402


pytestmark = pytest.mark.validator


LOCAL_FRAME_SRC = "int main() { int x; x = 990; return x; }\n"

IDENTITY_SRC = (
    "int identity(int x) { return x; }\n"
    "int main() { return identity(70); }\n"
)

REC_FIB_1_SRC = (
    "int fib(int n) {\n"
    "    if (n < 2) return n;\n"
    "    return fib(n-1) + fib(n-2);\n"
    "}\n"
    "int main() { return fib(1); }\n"
)


@pytest.mark.parametrize(
    "name, source, step, slot, expected",
    [
        (
            "local_frame_push_mem_addr0_e0",
            LOCAL_FRAME_SRC,
            3,
            "MEM_addr0",
            0xE0,
        ),
        (
            "stale_stack0_marker_non_fire",
            IDENTITY_SRC,
            2,
            "STACK0_byte0",
            0x00,
        ),
        (
            "local_frame_jsr_mem_addr0_e0",
            IDENTITY_SRC,
            4,
            "MEM_addr0",
            0xE0,
        ),
        (
            "positive_saved_frame_stack0_byte0_e8",
            LOCAL_FRAME_SRC,
            4,
            "STACK0_byte0",
            0xE8,
        ),
        (
            "rec_fib_initial_call_pc_byte1_01",
            REC_FIB_1_SRC,
            0,
            "PC_byte1",
            0x01,
        ),
        (
            "rec_fib_inner_call_stack0_return_byte1_01",
            REC_FIB_1_SRC,
            4,
            "STACK0_byte1",
            0x01,
        ),
    ],
)
def test_teacher_forced_symbolic_byte_survives_final_tail(
    pure_neural_runner,
    name: str,
    source: str,
    step: int,
    slot: str,
    expected: int,
) -> None:
    if not getattr(pure_neural_runner, "pure_neural", False):
        pytest.skip("requires the compiled pure-neural model")

    bytecode, data = compile_c(source)
    trace = build_teacher_forced_symbolic_trace(bytecode, data)
    token_index = trace.token_index(step, slot)

    assert trace.context[token_index] == expected, (
        f"probe precondition drifted for {name}: symbolic {slot} at step "
        f"{step} is 0x{trace.context[token_index]:02x}, not 0x{expected:02x}"
    )

    report = verify_teacher_forced_token_support(
        pure_neural_runner.model,
        trace.context,
        token_index=token_index,
        prefix_len=trace.prefix_len,
        mem_store_positions=trace.mem_store_positions,
        probe_name=name,
    )

    assert report.supported, report.format()


def test_teacher_forced_raw_imm_exit_trace_survives_final_tail(
    pure_neural_runner,
) -> None:
    if not getattr(pure_neural_runner, "pure_neural", False):
        pytest.skip("requires the compiled pure-neural model")

    bytecode = [Opcode.IMM | (42 << 8), Opcode.EXIT]
    trace = build_teacher_forced_symbolic_trace(bytecode, b"")

    report = audit_teacher_forced_trace_final_support(
        pure_neural_runner.model,
        trace,
        probe_name="raw_imm_exit_full_trace",
    )

    assert report.passed, report.format()
