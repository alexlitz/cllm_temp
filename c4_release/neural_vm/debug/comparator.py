"""
State Comparator for Neural VM.

Compares two execution traces to find divergence points.
"""

from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass

from .tracer import ExecutionTracer, StepState


@dataclass
class Divergence:
    """Information about a divergence between two traces."""

    step: int
    field: str
    expected: Any
    actual: Any
    expected_state: StepState
    actual_state: StepState

    def __str__(self) -> str:
        return (
            f"Divergence at step {self.step}: "
            f"{self.field} expected={self.expected}, actual={self.actual}"
        )


class StateComparator:
    """Compare two VM execution traces to find divergences.

    Usage:
        comparator = StateComparator()

        # Compare two traces
        trace1 = ExecutionTracer()
        trace2 = ExecutionTracer()
        # ... run executions ...

        divergence = comparator.find_first_divergence(trace1, trace2)
        if divergence:
            print(f"Traces diverge at step {divergence.step}")
            comparator.show_context(trace1, trace2, divergence.step)

        # Or compare all differences
        all_diffs = comparator.find_all_divergences(trace1, trace2)
    """

    def __init__(
        self,
        compare_pc: bool = True,
        compare_ax: bool = True,
        compare_sp: bool = True,
        compare_bp: bool = True,
        compare_opcode: bool = True,
    ):
        """Initialize comparator.

        Args:
            compare_pc: Compare program counter
            compare_ax: Compare accumulator
            compare_sp: Compare stack pointer
            compare_bp: Compare base pointer
            compare_opcode: Compare opcode
        """
        self.compare_pc = compare_pc
        self.compare_ax = compare_ax
        self.compare_sp = compare_sp
        self.compare_bp = compare_bp
        self.compare_opcode = compare_opcode

    def _compare_states(
        self,
        expected: StepState,
        actual: StepState,
    ) -> Optional[Divergence]:
        """Compare two states and return first difference."""
        if self.compare_pc and expected.pc != actual.pc:
            return Divergence(
                step=expected.step,
                field="pc",
                expected=expected.pc,
                actual=actual.pc,
                expected_state=expected,
                actual_state=actual,
            )

        if self.compare_opcode and expected.opcode != actual.opcode:
            return Divergence(
                step=expected.step,
                field="opcode",
                expected=f"{expected.opcode_name}({expected.opcode})",
                actual=f"{actual.opcode_name}({actual.opcode})",
                expected_state=expected,
                actual_state=actual,
            )

        if self.compare_ax and expected.ax != actual.ax:
            return Divergence(
                step=expected.step,
                field="ax",
                expected=expected.ax,
                actual=actual.ax,
                expected_state=expected,
                actual_state=actual,
            )

        if self.compare_sp and expected.sp != actual.sp:
            return Divergence(
                step=expected.step,
                field="sp",
                expected=expected.sp,
                actual=actual.sp,
                expected_state=expected,
                actual_state=actual,
            )

        if self.compare_bp and expected.bp != actual.bp:
            return Divergence(
                step=expected.step,
                field="bp",
                expected=expected.bp,
                actual=actual.bp,
                expected_state=expected,
                actual_state=actual,
            )

        return None

    def find_first_divergence(
        self,
        expected: ExecutionTracer,
        actual: ExecutionTracer,
    ) -> Optional[Divergence]:
        """Find the first step where traces diverge.

        Args:
            expected: Reference/expected trace
            actual: Trace to compare against expected

        Returns:
            Divergence info if found, None if traces match
        """
        min_len = min(len(expected), len(actual))

        for i in range(min_len):
            div = self._compare_states(expected[i], actual[i])
            if div:
                return div

        # Check for length mismatch
        if len(expected) != len(actual):
            return Divergence(
                step=min_len,
                field="length",
                expected=len(expected),
                actual=len(actual),
                expected_state=expected[-1] if expected.steps else None,
                actual_state=actual[-1] if actual.steps else None,
            )

        return None

    def find_all_divergences(
        self,
        expected: ExecutionTracer,
        actual: ExecutionTracer,
        max_divergences: int = 100,
    ) -> List[Divergence]:
        """Find all divergences between traces.

        Args:
            expected: Reference trace
            actual: Trace to compare
            max_divergences: Maximum divergences to return

        Returns:
            List of all divergences found
        """
        divergences = []
        min_len = min(len(expected), len(actual))

        for i in range(min_len):
            div = self._compare_states(expected[i], actual[i])
            if div:
                divergences.append(div)
                if len(divergences) >= max_divergences:
                    break

        return divergences

    def show_context(
        self,
        expected: ExecutionTracer,
        actual: ExecutionTracer,
        step: int,
        context: int = 3,
    ) -> None:
        """Show context around a divergence point.

        Args:
            expected: Reference trace
            actual: Actual trace
            step: Step where divergence occurred
            context: Number of steps before/after to show
        """
        start = max(0, step - context)
        end = min(min(len(expected), len(actual)), step + context + 1)

        print(f"\n{'='*80}")
        print(f"DIVERGENCE CONTEXT (step {step})")
        print(f"{'='*80}")

        print(f"\n{'EXPECTED':^40} | {'ACTUAL':^40}")
        print(f"{'-'*40} | {'-'*40}")

        for i in range(start, end):
            marker = ">>>" if i == step else "   "

            if i < len(expected):
                e = expected[i]
                exp_str = (
                    f"{marker} [{e.step:3d}] PC={e.pc:3d} {e.opcode_name:4s} "
                    f"AX={e.ax:5d} SP={e.sp:5d}"
                )
            else:
                exp_str = f"{marker} (no step {i})"

            if i < len(actual):
                a = actual[i]
                act_str = (
                    f"[{a.step:3d}] PC={a.pc:3d} {a.opcode_name:4s} "
                    f"AX={a.ax:5d} SP={a.sp:5d}"
                )
            else:
                act_str = "(no step)"

            print(f"{exp_str:40s} | {act_str:40s}")

        print(f"{'='*80}\n")

    def summarize_divergences(
        self,
        divergences: List[Divergence],
    ) -> Dict[str, int]:
        """Summarize divergences by field.

        Args:
            divergences: List of divergences

        Returns:
            Dict mapping field name to count
        """
        summary: Dict[str, int] = {}
        for div in divergences:
            summary[div.field] = summary.get(div.field, 0) + 1
        return summary

    def compare_final_state(
        self,
        expected: ExecutionTracer,
        actual: ExecutionTracer,
    ) -> Dict[str, Tuple[Any, Any]]:
        """Compare final states of two traces.

        Args:
            expected: Reference trace
            actual: Actual trace

        Returns:
            Dict of field -> (expected_value, actual_value) for differing fields
        """
        if not expected.steps or not actual.steps:
            return {"error": ("no steps", "no steps")}

        exp_final = expected[-1]
        act_final = actual[-1]

        diffs = {}

        if exp_final.pc != act_final.pc:
            diffs["pc"] = (exp_final.pc, act_final.pc)
        if exp_final.ax != act_final.ax:
            diffs["ax"] = (exp_final.ax, act_final.ax)
        if exp_final.sp != act_final.sp:
            diffs["sp"] = (exp_final.sp, act_final.sp)
        if exp_final.bp != act_final.bp:
            diffs["bp"] = (exp_final.bp, act_final.bp)
        if len(expected) != len(actual):
            diffs["steps"] = (len(expected), len(actual))

        return diffs
