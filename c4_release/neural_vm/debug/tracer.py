"""
Execution Tracer for Neural VM.

Provides step-by-step execution tracing with detailed state capture.
"""

import json
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class StepState:
    """State captured at a single execution step."""

    step: int
    pc: int
    opcode: int
    opcode_name: str
    immediate: int
    ax: int
    sp: int
    bp: int
    tokens_generated: int = 0
    memory_reads: List[Dict[str, int]] = field(default_factory=list)
    memory_writes: List[Dict[str, int]] = field(default_factory=list)
    stack_top: Optional[int] = None
    notes: str = ""


class ExecutionTracer:
    """Trace VM execution step by step.

    Captures detailed state at each step for debugging and analysis.

    Usage:
        tracer = ExecutionTracer(verbose=True)

        # Manual tracing
        tracer.trace_step(step=0, pc=0, opcode=1, ax=0, sp=1000, bp=1000)

        # Or integrate with runner
        runner = AutoregressiveVMRunner(tracer=tracer)
        runner.run(bytecode, data)

        # Analyze
        tracer.show_summary()
        tracer.dump("trace.json")
    """

    # Opcode names for readable output
    OPCODE_NAMES = {
        0: "LEA", 1: "IMM", 2: "JMP", 3: "JSR", 4: "BZ", 5: "BNZ",
        6: "ENT", 7: "ADJ", 8: "LEV", 9: "LI", 10: "LC", 11: "SI",
        12: "SC", 13: "PSH", 14: "OR", 15: "XOR", 16: "AND",
        17: "EQ", 18: "NE", 19: "LT", 20: "GT", 21: "LE", 22: "GE",
        23: "SHL", 24: "SHR", 25: "ADD", 26: "SUB", 27: "MUL",
        28: "DIV", 29: "MOD", 30: "OPEN", 31: "READ", 32: "CLOS",
        33: "PRTF", 34: "MALC", 35: "FREE", 36: "MSET", 37: "MCMP",
        38: "EXIT", 39: "NOP",
    }

    def __init__(self, verbose: bool = False, max_steps: int = 10000):
        """Initialize tracer.

        Args:
            verbose: Print each step as it's traced
            max_steps: Maximum steps to store (prevents memory issues)
        """
        self.verbose = verbose
        self.max_steps = max_steps
        self.steps: List[StepState] = []
        self.metadata: Dict[str, Any] = {}

    def trace_step(
        self,
        step: int,
        pc: int,
        opcode: int,
        ax: int,
        sp: int,
        bp: int,
        immediate: int = 0,
        tokens_generated: int = 0,
        memory_reads: Optional[List[Dict[str, int]]] = None,
        memory_writes: Optional[List[Dict[str, int]]] = None,
        stack_top: Optional[int] = None,
        notes: str = "",
    ) -> None:
        """Record a single execution step.

        Args:
            step: Step number (0-indexed)
            pc: Program counter value
            opcode: Opcode being executed
            ax: Accumulator value
            sp: Stack pointer value
            bp: Base pointer value
            immediate: Immediate value from instruction
            tokens_generated: Number of tokens generated this step
            memory_reads: List of memory reads [{addr, value}, ...]
            memory_writes: List of memory writes [{addr, value}, ...]
            stack_top: Value at top of stack
            notes: Optional notes about this step
        """
        if len(self.steps) >= self.max_steps:
            return  # Don't overflow

        opcode_name = self.OPCODE_NAMES.get(opcode, f"UNK({opcode})")

        state = StepState(
            step=step,
            pc=pc,
            opcode=opcode,
            opcode_name=opcode_name,
            immediate=immediate,
            ax=ax,
            sp=sp,
            bp=bp,
            tokens_generated=tokens_generated,
            memory_reads=memory_reads or [],
            memory_writes=memory_writes or [],
            stack_top=stack_top,
            notes=notes,
        )

        self.steps.append(state)

        if self.verbose:
            self._print_step(state)

    def _print_step(self, state: StepState) -> None:
        """Print a step in human-readable format."""
        imm_str = f" {state.immediate}" if state.immediate else ""
        print(
            f"[{state.step:4d}] "
            f"PC={state.pc:4d} "
            f"{state.opcode_name:4s}{imm_str:6s} "
            f"AX={state.ax:6d} "
            f"SP={state.sp:6d} "
            f"BP={state.bp:6d}"
            f"{' ' + state.notes if state.notes else ''}"
        )

    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata about the trace."""
        self.metadata[key] = value

    def show_summary(self) -> None:
        """Print a summary of the trace."""
        if not self.steps:
            print("No steps recorded")
            return

        print(f"\n{'='*60}")
        print("EXECUTION TRACE SUMMARY")
        print(f"{'='*60}")
        print(f"Total steps: {len(self.steps)}")
        print(f"Final PC: {self.steps[-1].pc}")
        print(f"Final AX: {self.steps[-1].ax}")
        print(f"Final SP: {self.steps[-1].sp}")

        # Opcode frequency
        opcode_counts: Dict[str, int] = {}
        for step in self.steps:
            opcode_counts[step.opcode_name] = opcode_counts.get(step.opcode_name, 0) + 1

        print(f"\nOpcode frequency:")
        for name, count in sorted(opcode_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"  {name}: {count}")

        print(f"{'='*60}\n")

    def find_opcode(self, opcode_name: str) -> List[StepState]:
        """Find all steps with a specific opcode."""
        return [s for s in self.steps if s.opcode_name == opcode_name]

    def find_pc(self, pc: int) -> List[StepState]:
        """Find all steps at a specific PC."""
        return [s for s in self.steps if s.pc == pc]

    def find_divergence(self, other: "ExecutionTracer") -> Optional[int]:
        """Find first step where traces diverge."""
        for i, (s1, s2) in enumerate(zip(self.steps, other.steps)):
            if s1.pc != s2.pc or s1.ax != s2.ax or s1.sp != s2.sp:
                return i
        return None

    def dump(self, path: str) -> None:
        """Dump trace to JSON file."""
        data = {
            "metadata": self.metadata,
            "steps": [asdict(s) for s in self.steps],
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str) -> "ExecutionTracer":
        """Load trace from JSON file."""
        data = json.loads(Path(path).read_text())
        tracer = cls()
        tracer.metadata = data.get("metadata", {})
        tracer.steps = [StepState(**s) for s in data.get("steps", [])]
        return tracer

    def clear(self) -> None:
        """Clear all recorded steps."""
        self.steps = []
        self.metadata = {}

    def __len__(self) -> int:
        return len(self.steps)

    def __getitem__(self, idx: int) -> StepState:
        return self.steps[idx]
