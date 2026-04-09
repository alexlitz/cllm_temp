"""
Step-Level Debugger - Interactive debugger for Neural VM.

Usage:
    from neural_vm.step_debugger import StepDebugger
    from neural_vm.embedding import Opcode

    debugger = StepDebugger(runner)
    debugger.add_breakpoint(opcode=Opcode.ADD)
    debugger.run_until_break(bytecode)

    # Interactive mode:
    debugger.inspect(layer=3, dim='AX_CARRY_LO')
    debugger.inspect(layer=8, dim='AX_CARRY_LO')
    debugger.continue_execution()
"""

import torch
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass
from .vm_step import Token, _SetDim
from .debugger import VMExecutionTracer, ExecutionTrace

BD = _SetDim


@dataclass
class Breakpoint:
    """Breakpoint configuration."""
    step: Optional[int] = None
    opcode: Optional[int] = None
    condition: Optional[Callable] = None  # Function that returns True to break
    description: str = ""
    enabled: bool = True


class StepDebugger:
    """Interactive debugger that stops at each VM step or specified breakpoints."""

    def __init__(self, runner, track_dims: Optional[List[str]] = None):
        """
        Args:
            runner: AutoregressiveVMRunner instance
            track_dims: List of dimension names to track
        """
        self.runner = runner
        self.track_dims = track_dims or ['AX_CARRY_LO', 'AX_CARRY_HI', 'ALU_LO', 'ALU_HI']
        self.breakpoints: List[Breakpoint] = []

        self.current_trace: Optional[ExecutionTrace] = None
        self.current_step = 0
        self.current_token_idx = 0
        self.is_running = False
        self.should_continue = False

    def add_breakpoint(self, step: Optional[int] = None, opcode: Optional[int] = None,
                      condition: Optional[Callable] = None, description: str = "") -> int:
        """Add breakpoint at specific step, opcode, or condition.

        Args:
            step: Stop at this step number
            opcode: Stop when this opcode executes
            condition: Custom function(trace, step, token_idx) -> bool
            description: Human-readable description

        Returns:
            Breakpoint ID (index in breakpoints list)
        """
        bp = Breakpoint(
            step=step,
            opcode=opcode,
            condition=condition,
            description=description or self._generate_bp_description(step, opcode),
            enabled=True
        )
        self.breakpoints.append(bp)
        bp_id = len(self.breakpoints) - 1

        print(f"Breakpoint {bp_id} added: {bp.description}")
        return bp_id

    def _generate_bp_description(self, step: Optional[int], opcode: Optional[int]) -> str:
        """Generate breakpoint description."""
        if step is not None:
            return f"Step {step}"
        elif opcode is not None:
            from .embedding import Opcode
            opcode_name = None
            for name, val in Opcode.__dict__.items():
                if val == opcode and not name.startswith('_'):
                    opcode_name = name
                    break
            return f"Opcode {opcode_name or opcode}"
        else:
            return "Custom condition"

    def remove_breakpoint(self, bp_id: int):
        """Remove breakpoint by ID."""
        if 0 <= bp_id < len(self.breakpoints):
            self.breakpoints[bp_id].enabled = False
            print(f"Breakpoint {bp_id} disabled")
        else:
            print(f"Invalid breakpoint ID: {bp_id}")

    def list_breakpoints(self):
        """List all breakpoints."""
        if not self.breakpoints:
            print("No breakpoints set")
            return

        print("\nBreakpoints:")
        for i, bp in enumerate(self.breakpoints):
            status = "enabled" if bp.enabled else "disabled"
            print(f"  {i}: {bp.description} ({status})")

    def _check_breakpoints(self, trace: ExecutionTrace, step: int, token_idx: int) -> bool:
        """Check if any breakpoint should trigger."""
        for bp in self.breakpoints:
            if not bp.enabled:
                continue

            # Check step breakpoint
            if bp.step is not None and step == bp.step:
                print(f"\n🔴 Breakpoint hit: {bp.description}")
                return True

            # Check opcode breakpoint
            if bp.opcode is not None:
                token_meta = next((t for t in trace.token_metadata if t['token_idx'] == token_idx), None)
                if token_meta and token_meta.get('opcode') == bp.opcode:
                    print(f"\n🔴 Breakpoint hit: {bp.description}")
                    return True

            # Check custom condition
            if bp.condition is not None:
                try:
                    if bp.condition(trace, step, token_idx):
                        print(f"\n🔴 Breakpoint hit: {bp.description}")
                        return True
                except Exception as e:
                    print(f"Warning: Breakpoint condition error: {e}")

        return False

    def run_until_break(self, bytecode, max_steps: int = 10, interactive: bool = True):
        """Run until breakpoint hit, then optionally enter interactive mode.

        Args:
            bytecode: Program bytecode
            max_steps: Maximum execution steps
            interactive: If True, enter interactive mode on break
        """
        print(f"Running until breakpoint (max_steps={max_steps})...")
        print("Use Ctrl+C to stop\n")

        # Run with tracer to capture all data
        tracer = VMExecutionTracer(self.runner, track_dims=self.track_dims)
        self.current_trace = tracer.trace_execution(bytecode, max_steps=max_steps)

        # Check breakpoints after execution
        # (In a more advanced version, we'd check during execution with hooks)
        for token_meta in self.current_trace.token_metadata:
            token_idx = token_meta['token_idx']
            step = token_meta['step']

            if self._check_breakpoints(self.current_trace, step, token_idx):
                self.current_step = step
                self.current_token_idx = token_idx

                if interactive:
                    self._interactive_mode()
                return

        print("No breakpoints hit, execution completed normally")

    def _interactive_mode(self):
        """Enter interactive debugging mode."""
        print(f"\n{'='*70}")
        print(f"Interactive Debugger - Step {self.current_step}, Token {self.current_token_idx}")
        print(f"{'='*70}")
        print("\nCommands:")
        print("  inspect <layer> <dim>  - Inspect dimension value at layer")
        print("  compare <l1> <l2> <dim> - Compare dimension between layers")
        print("  show <dim>             - Show dimension across all layers")
        print("  flow <dim> <from> <to> - Show dataflow from layer to layer")
        print("  continue / c           - Continue execution")
        print("  help / ?               - Show this help")
        print()

        # Simple interactive loop (would use proper REPL in production)
        # For now, just show current state
        self._show_current_state()

    def _show_current_state(self):
        """Show current execution state."""
        if self.current_trace is None:
            print("No trace data available")
            return

        token_meta = next((t for t in self.current_trace.token_metadata
                          if t['token_idx'] == self.current_token_idx), None)

        if token_meta:
            print(f"Current state:")
            print(f"  Step: {token_meta['step']}")
            print(f"  Token: {token_meta['token_idx']}")
            print(f"  Token ID: {token_meta['token_id']}")
            print(f"  Is marker: {token_meta['is_marker']}")
            if token_meta.get('opcode'):
                print(f"  Opcode: {token_meta['opcode']}")

    def inspect(self, layer: int, dim: str):
        """Inspect dimension value at specific layer.

        Args:
            layer: Layer index (0-15)
            dim: Dimension name (e.g., 'AX_CARRY_LO')
        """
        if self.current_trace is None:
            print("No trace data available. Run debugger first.")
            return

        val = self.current_trace.get_dimension_value(layer, self.current_token_idx, dim)

        if val is None:
            print(f"No data for {dim} at Layer {layer}")
            return

        print(f"\n{dim} at Layer {layer}:")
        if isinstance(val, float):
            print(f"  Value: {val:.3f}")
        else:
            print(f"  Max: {float(val.max()):.3f}")
            print(f"  Mean: {float(val.mean()):.3f}")
            print(f"  Values: {val}")

    def compare_layers(self, layer1: int, layer2: int, dim_name: str):
        """Compare dimension value between two layers.

        Args:
            layer1: First layer
            layer2: Second layer
            dim_name: Dimension to compare
        """
        if self.current_trace is None:
            print("No trace data available. Run debugger first.")
            return

        val1 = self.current_trace.get_dimension_value(layer1, self.current_token_idx, dim_name)
        val2 = self.current_trace.get_dimension_value(layer2, self.current_token_idx, dim_name)

        print(f"\nComparing {dim_name}: Layer {layer1} vs Layer {layer2}")

        if val1 is None or val2 is None:
            print("Missing data for one or both layers")
            return

        if isinstance(val1, float):
            diff = abs(val1 - val2)
            print(f"  Layer {layer1}: {val1:.3f}")
            print(f"  Layer {layer2}: {val2:.3f}")
            print(f"  Difference: {diff:.3f}")
        else:
            max1, max2 = float(val1.max()), float(val2.max())
            mean1, mean2 = float(val1.mean()), float(val2.mean())

            print(f"  Layer {layer1}: max={max1:.3f}, mean={mean1:.3f}")
            print(f"  Layer {layer2}: max={max2:.3f}, mean={mean2:.3f}")
            print(f"  Max diff: {abs(max1 - max2):.3f}")
            print(f"  Mean diff: {abs(mean1 - mean2):.3f}")

    def continue_execution(self):
        """Continue execution (placeholder for future interactive implementation)."""
        print("Continuing execution...")
        self.should_continue = True

    def show_dimension_across_layers(self, dim_name: str):
        """Show dimension value across all 16 layers."""
        if self.current_trace is None:
            print("No trace data available. Run debugger first.")
            return

        print(f"\n{dim_name} across all layers (Step {self.current_step}):")
        for layer_idx in range(16):
            val = self.current_trace.get_dimension_value(layer_idx, self.current_token_idx, dim_name)
            if val is not None:
                if isinstance(val, float):
                    print(f"  Layer {layer_idx:2d}: {val:.3f}")
                else:
                    print(f"  Layer {layer_idx:2d}: max={float(val.max()):.3f}")


if __name__ == "__main__":
    # Example usage
    from src.compiler import compile_c
    from neural_vm.run_vm import AutoregressiveVMRunner
    from neural_vm.embedding import Opcode

    code = "int main() { return 10 + 32; }"
    bytecode, data = compile_c(code)

    runner = AutoregressiveVMRunner()

    print("Starting step debugger for: 10 + 32")
    print("Setting breakpoint at ADD operation\n")

    debugger = StepDebugger(runner)
    debugger.add_breakpoint(opcode=Opcode.ADD)

    debugger.run_until_break(bytecode, max_steps=10, interactive=False)

    # Demonstrate inspection
    print("\nInspecting AX_CARRY_LO:")
    debugger.inspect(layer=3, dim='AX_CARRY_LO')
    debugger.inspect(layer=8, dim='AX_CARRY_LO')

    print("\nComparing layers:")
    debugger.compare_layers(3, 8, 'AX_CARRY_LO')

    print("\nShowing dimension across all layers:")
    debugger.show_dimension_across_layers('AX_CARRY_LO')
