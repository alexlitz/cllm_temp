"""Shared symbolic-declarative oracle helpers for neural tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence


@dataclass(frozen=True)
class DeclarativeOracleResult:
    output: str
    exit_code: Optional[int]
    steps: Optional[int]
    halted: bool
    error: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "output": self.output,
            "exit_code": self.exit_code,
            "steps": self.steps,
            "halted": self.halted,
            "error": self.error,
        }


def declarative_oracle_for_program(
    bytecode: Sequence[int],
    data: Sequence[int] | bytes = b"",
    *,
    suite_expected: Optional[int] = None,
    suite_check: Optional[Callable[[int], None]] = None,
    label: str = "program",
    max_steps: Optional[int] = None,
    runner=None,
) -> DeclarativeOracleResult:
    """Run one program through the symbolic declarative VM and validate it.

    ``suite_expected`` and ``suite_check`` are optional test-suite contracts.
    When provided, disagreements are returned as oracle errors so neural tests
    fail before treating a bad declaration as the expected neural behavior.
    """

    if runner is None:
        from neural_vm.unified_compiler.symbolic_program import (
            SymbolicDeclarativeProgramRunner,
        )

        runner = SymbolicDeclarativeProgramRunner()

    try:
        state = runner.run(bytecode, data, max_steps=max_steps)
    except Exception as exc:
        return DeclarativeOracleResult(
            output="",
            exit_code=None,
            steps=None,
            halted=False,
            error=f"{label}: compile/declarative error: {exc!r}",
        )

    output = state.get_output()
    exit_code = state.ax if state.halted else None
    steps = state.steps if state.halted else None

    if not state.halted:
        return DeclarativeOracleResult(
            output=output,
            exit_code=exit_code,
            steps=steps,
            halted=False,
            error=f"{label}: declarative execution did not halt",
        )

    if suite_expected is not None and exit_code != (suite_expected & 0xFFFFFFFF):
        return DeclarativeOracleResult(
            output=output,
            exit_code=exit_code,
            steps=steps,
            halted=True,
            error=(
                f"{label}: suite expected disagrees with declarative result: "
                f"expected={suite_expected & 0xFFFFFFFF} declarative={exit_code} "
                f"decl_steps={steps}"
            ),
        )

    if suite_check is not None:
        try:
            suite_check(int(exit_code))
        except AssertionError as exc:
            return DeclarativeOracleResult(
                output=output,
                exit_code=exit_code,
                steps=steps,
                halted=True,
                error=(
                    f"{label}: suite check disagrees with declarative result: "
                    f"{exc}"
                ),
            )

    return DeclarativeOracleResult(
        output=output,
        exit_code=exit_code,
        steps=steps,
        halted=True,
    )
