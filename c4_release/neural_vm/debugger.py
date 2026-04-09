"""
Neural VM Execution Tracer - Debug tool for tracking dimension values through layers.

Usage:
    from neural_vm.debugger import VMExecutionTracer

    tracer = VMExecutionTracer(runner, track_dims=['AX_CARRY_LO', 'ALU_LO'])
    trace = tracer.trace_execution(bytecode, max_steps=10)

    # Find where AX_CARRY gets lost
    trace.show_dimension_history('AX_CARRY_LO', step=4)

    # Find ADD operation
    add_info = trace.find_operation(Opcode.ADD)
    print(f"ADD executed at step {add_info['step']}")
"""

import torch
import json
import pickle
import time
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from .vm_step import Token, _SetDim

BD = _SetDim


@dataclass
class LayerSnapshot:
    """Snapshot of layer output at a specific token."""
    layer_idx: int
    token_idx: int
    step_num: int
    output: torch.Tensor  # Shape: [1, seq_len, d_model]
    timestamp: float = 0.0  # Time when this layer completed (for profiling)


@dataclass
class PerformanceStats:
    """Performance statistics for execution."""
    total_time: float = 0.0
    layer_times: Dict[int, float] = field(default_factory=dict)
    token_times: List[float] = field(default_factory=list)
    forward_passes: int = 0


@dataclass
class ExecutionTrace:
    """Complete trace of VM execution."""
    layer_snapshots: Dict[int, List[LayerSnapshot]] = field(default_factory=dict)
    token_metadata: List[Dict[str, Any]] = field(default_factory=list)
    tracked_dims: List[str] = field(default_factory=list)
    performance: PerformanceStats = field(default_factory=PerformanceStats)
    bytecode: Optional[bytes] = None
    exit_code: Optional[int] = None

    def get_dimension_value(self, layer_idx: int, token_idx: int, dim_name: str):
        """Get value of a dimension at specific layer and token."""
        dim_range = getattr(BD, dim_name, None)
        if dim_range is None:
            raise ValueError(f"Unknown dimension: {dim_name}")

        # Find snapshot
        snapshots = self.layer_snapshots.get(layer_idx, [])
        snapshot = next((s for s in snapshots if s.token_idx == token_idx), None)

        if snapshot is None:
            return None

        # Get dimension value (last position in sequence)
        if isinstance(dim_range, int):
            # Single dimension
            return snapshot.output[0, -1, dim_range].item()
        else:
            # Dimension range (assume dim_range is start of range)
            # Most dimensions are 16-element ranges
            return snapshot.output[0, -1, dim_range:dim_range+16].cpu().numpy()

    def show_dimension_history(self, dim_name: str, step: Optional[int] = None):
        """Show how dimension value changes across layers."""
        print(f"\n{'='*70}")
        print(f"Dimension History: {dim_name}")
        if step is not None:
            print(f"Step: {step}")
        print(f"{'='*70}\n")

        # Get all snapshots for this step
        relevant_tokens = [t for t in self.token_metadata
                          if step is None or t['step'] == step]

        if not relevant_tokens:
            print(f"No tokens found for step {step}")
            return

        # For each layer, show dimension value
        for layer_idx in range(16):
            values = []
            for token in relevant_tokens[:5]:  # Limit to first 5 tokens
                val = self.get_dimension_value(layer_idx, token['token_idx'], dim_name)
                if val is not None:
                    if isinstance(val, float):
                        values.append(f"{val:.3f}")
                    else:
                        max_val = float(val.max())
                        values.append(f"max={max_val:.3f}")

            if values:
                print(f"  Layer {layer_idx:2d}: {', '.join(values)}")

    def find_operation(self, opcode: int) -> List[Dict[str, Any]]:
        """Find tokens where a specific opcode executed."""
        results = []
        for token in self.token_metadata:
            if token.get('opcode') == opcode:
                results.append(token)
        return results

    def save(self, path: str):
        """Save trace to disk for later analysis."""
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Save metadata as JSON
        metadata = {
            'token_metadata': self.token_metadata,
            'tracked_dims': self.tracked_dims,
            'exit_code': self.exit_code,
            'performance': {
                'total_time': self.performance.total_time,
                'layer_times': self.performance.layer_times,
                'token_times': self.performance.token_times,
                'forward_passes': self.performance.forward_passes,
            }
        }

        with open(path_obj.with_suffix('.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save layer snapshots as pickle (contains tensors)
        snapshot_data = {
            'layer_snapshots': self.layer_snapshots,
            'bytecode': self.bytecode,
        }

        with open(path_obj.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(snapshot_data, f)

        print(f"Trace saved to {path_obj.with_suffix('.json')} and {path_obj.with_suffix('.pkl')}")

    @classmethod
    def load(cls, path: str) -> 'ExecutionTrace':
        """Load trace from disk."""
        path_obj = Path(path)

        # Load metadata
        with open(path_obj.with_suffix('.json'), 'r') as f:
            metadata = json.load(f)

        # Load snapshots
        with open(path_obj.with_suffix('.pkl'), 'rb') as f:
            snapshot_data = pickle.load(f)

        # Reconstruct trace
        trace = cls()
        trace.token_metadata = metadata['token_metadata']
        trace.tracked_dims = metadata['tracked_dims']
        trace.exit_code = metadata.get('exit_code')
        trace.layer_snapshots = snapshot_data['layer_snapshots']
        trace.bytecode = snapshot_data.get('bytecode')

        # Restore performance stats
        perf = metadata.get('performance', {})
        trace.performance = PerformanceStats(
            total_time=perf.get('total_time', 0.0),
            layer_times=perf.get('layer_times', {}),
            token_times=perf.get('token_times', []),
            forward_passes=perf.get('forward_passes', 0),
        )

        print(f"Trace loaded from {path_obj}")
        return trace

    def visualize_dataflow(self, dim_name: str, from_layer: int, to_layer: int, step: Optional[int] = None):
        """Show how a dimension's value flows between layers.

        Args:
            dim_name: Dimension to track (e.g., 'AX_CARRY_LO')
            from_layer: Starting layer
            to_layer: Ending layer
            step: Optional step number to focus on
        """
        print(f"\n{'='*70}")
        print(f"Dataflow: {dim_name} from Layer {from_layer} to Layer {to_layer}")
        if step is not None:
            print(f"Step: {step}")
        print(f"{'='*70}\n")

        # Get relevant tokens
        relevant_tokens = [t for t in self.token_metadata
                          if step is None or t['step'] == step]

        if not relevant_tokens:
            print(f"No tokens found for step {step}")
            return

        # Track first token for simplicity
        token = relevant_tokens[0]

        print(f"Tracking token {token['token_idx']} (step {token['step']}):\n")

        # Show value at each layer
        for layer_idx in range(from_layer, to_layer + 1):
            val = self.get_dimension_value(layer_idx, token['token_idx'], dim_name)

            if val is not None:
                if isinstance(val, float):
                    val_str = f"{val:.3f}"
                else:
                    max_val = float(val.max())
                    mean_val = float(val.mean())
                    val_str = f"max={max_val:.3f}, mean={mean_val:.3f}"

                # Check if value changed significantly from previous layer
                marker = ""
                if layer_idx > from_layer:
                    prev_val = self.get_dimension_value(layer_idx - 1, token['token_idx'], dim_name)
                    if prev_val is not None:
                        if isinstance(val, float):
                            if abs(val - prev_val) > 0.1:
                                marker = " ← CHANGED"
                        else:
                            prev_max = float(prev_val.max())
                            curr_max = float(val.max())
                            if abs(curr_max - prev_max) > 0.1:
                                if curr_max > prev_max:
                                    marker = " ← SET"
                                else:
                                    marker = " ← LOST"

                print(f"  Layer {layer_idx:2d}: {val_str}{marker}")
            else:
                print(f"  Layer {layer_idx:2d}: (no data)")

        print()

    def validate_flow(self, dim_name: str, from_layer: int, to_layer: int,
                     min_value: float = 0.5, step: Optional[int] = None) -> bool:
        """Validate that a dimension value propagates from one layer to another.

        Args:
            dim_name: Dimension to check
            from_layer: Source layer that should set the value
            to_layer: Destination layer that should receive the value
            min_value: Minimum value considered "present"
            step: Optional step to check

        Returns:
            True if dimension propagates correctly, False otherwise
        """
        relevant_tokens = [t for t in self.token_metadata
                          if step is None or t['step'] == step]

        if not relevant_tokens:
            print(f"❌ No tokens found for step {step}")
            return False

        token = relevant_tokens[0]

        # Check source layer
        from_val = self.get_dimension_value(from_layer, token['token_idx'], dim_name)
        if from_val is None:
            print(f"❌ No data from Layer {from_layer}")
            return False

        from_max = from_val if isinstance(from_val, float) else float(from_val.max())

        if from_max < min_value:
            print(f"❌ Layer {from_layer} does not set {dim_name} (max={from_max:.3f} < {min_value})")
            return False

        # Check destination layer
        to_val = self.get_dimension_value(to_layer, token['token_idx'], dim_name)
        if to_val is None:
            print(f"❌ No data from Layer {to_layer}")
            return False

        to_max = to_val if isinstance(to_val, float) else float(to_val.max())

        if to_max < min_value:
            print(f"❌ {dim_name} value drops to {to_max:.3f} at Layer {to_layer}")

            # Find where it was lost
            for layer_idx in range(from_layer + 1, to_layer + 1):
                val = self.get_dimension_value(layer_idx, token['token_idx'], dim_name)
                if val is not None:
                    val_max = val if isinstance(val, float) else float(val.max())
                    if val_max < min_value:
                        print(f"   → Lost at Layer {layer_idx}")
                        break

            return False

        print(f"✓ {dim_name} successfully propagates from Layer {from_layer} to Layer {to_layer}")
        print(f"  Source (L{from_layer}): max={from_max:.3f}")
        print(f"  Dest (L{to_layer}): max={to_max:.3f}")
        return True

    def performance_summary(self):
        """Print performance statistics."""
        print(f"\n{'='*70}")
        print("Performance Summary")
        print(f"{'='*70}\n")

        print(f"Total execution time: {self.performance.total_time:.3f}s")
        print(f"Forward passes: {self.performance.forward_passes}")
        print(f"Tokens generated: {len(self.token_metadata)}")

        if self.performance.token_times:
            avg_token_time = sum(self.performance.token_times) / len(self.performance.token_times)
            print(f"Average time per token: {avg_token_time*1000:.2f}ms")

        if self.performance.layer_times:
            print(f"\nTime per layer:")
            total_layer_time = sum(self.performance.layer_times.values())
            for layer_idx in range(16):
                if layer_idx in self.performance.layer_times:
                    layer_time = self.performance.layer_times[layer_idx]
                    pct = (layer_time / total_layer_time * 100) if total_layer_time > 0 else 0
                    print(f"  Layer {layer_idx:2d}: {layer_time*1000:.2f}ms ({pct:.1f}%)")

        print()


class VMExecutionTracer:
    """Trace neural VM execution with full layer introspection."""

    def __init__(self, runner, track_dims: Optional[List[str]] = None, enable_profiling: bool = True):
        """
        Args:
            runner: AutoregressiveVMRunner instance
            track_dims: List of dimension names to track (e.g., ['AX_CARRY_LO', 'ALU_LO'])
            enable_profiling: Whether to track performance metrics
        """
        self.runner = runner
        self.track_dims = track_dims or ['AX_CARRY_LO', 'AX_CARRY_HI', 'ALU_LO', 'ALU_HI']
        self.trace = ExecutionTrace(tracked_dims=self.track_dims)
        self.enable_profiling = enable_profiling

        self.current_token = 0
        self.current_step = 0
        self.current_opcode = None
        self.start_time = 0.0
        self.last_token_time = 0.0

    def trace_execution(self, bytecode, max_steps: int = 10) -> ExecutionTrace:
        """Run VM with full tracing enabled."""
        print(f"Starting execution trace (tracking {len(self.track_dims)} dimensions)...")

        # Store bytecode for later reference
        self.trace.bytecode = bytecode

        # Install hooks on all layers
        hooks = []
        for layer_idx in range(16):
            hook = self.runner.model.blocks[layer_idx].register_forward_hook(
                self._make_layer_hook(layer_idx)
            )
            hooks.append(hook)

        # Hook embedding to track token metadata
        embed_hook = self.runner.model.embed.register_forward_hook(
            self._embedding_hook
        )
        hooks.append(embed_hook)

        try:
            # Run execution with timing
            self.start_time = time.time()
            self.last_token_time = self.start_time

            output, exit_code = self.runner.run(bytecode, max_steps=max_steps)

            # Record final metrics
            self.trace.performance.total_time = time.time() - self.start_time
            self.trace.exit_code = exit_code
            self.trace.performance.forward_passes = self.current_token

            print(f"Execution complete: exit_code={exit_code}")
            print(f"Captured {len(self.trace.token_metadata)} tokens")
            print(f"Layer snapshots: {sum(len(v) for v in self.trace.layer_snapshots.values())}")

            if self.enable_profiling:
                print(f"Total time: {self.trace.performance.total_time:.3f}s")

            return self.trace

        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()

    def _make_layer_hook(self, layer_idx: int):
        """Create hook function for a specific layer."""
        def hook(module, input, output):
            current_time = time.time() if self.enable_profiling else 0.0

            snapshot = LayerSnapshot(
                layer_idx=layer_idx,
                token_idx=self.current_token,
                step_num=self.current_step,
                output=output.detach().cpu(),
                timestamp=current_time
            )

            if layer_idx not in self.trace.layer_snapshots:
                self.trace.layer_snapshots[layer_idx] = []
            self.trace.layer_snapshots[layer_idx].append(snapshot)

            # Track per-layer timing (cumulative)
            if self.enable_profiling and layer_idx == 15:  # Last layer
                if layer_idx not in self.trace.performance.layer_times:
                    self.trace.performance.layer_times[layer_idx] = 0.0

                # Simple approximation: divide time evenly among all layers
                # (More accurate tracking would require per-layer hooks with timing)
                token_time = current_time - self.last_token_time
                for i in range(16):
                    if i not in self.trace.performance.layer_times:
                        self.trace.performance.layer_times[i] = 0.0
                    self.trace.performance.layer_times[i] += token_time / 16

                self.trace.performance.token_times.append(token_time)
                self.last_token_time = current_time

        return hook

    def _embedding_hook(self, module, input, output):
        """Hook to track token metadata."""
        if len(input) > 0 and input[0].shape[1] > 0:
            last_token = input[0][0, -1].item()

            # Track if this is a step boundary
            if last_token == Token.STEP_END:
                self.current_step += 1

            # Store metadata
            self.trace.token_metadata.append({
                'token_idx': self.current_token,
                'token_id': last_token,
                'step': self.current_step,
                'is_marker': last_token in [Token.REG_PC, Token.REG_AX, Token.REG_SP,
                                           Token.REG_BP, Token.STACK0, Token.MEM],
                'opcode': self.current_opcode,
            })

            self.current_token += 1

    def compare_with_handler(self, bytecode, opcode: int, max_steps: int = 10) -> Dict[str, Any]:
        """Compare neural execution with handler execution for a specific opcode.

        Args:
            bytecode: Program bytecode
            opcode: Opcode to compare
            max_steps: Maximum execution steps

        Returns:
            Dictionary with comparison results
        """
        from neural_vm.run_vm import AutoregressiveVMRunner

        print(f"\n{'='*70}")
        print(f"Comparing Neural vs Handler Execution")
        print(f"{'='*70}\n")

        # Run with handler
        print("Running with handler...")
        runner_with_handler = AutoregressiveVMRunner()
        output_with, exit_with = runner_with_handler.run(bytecode, max_steps=max_steps)

        # Run without handler (neural only)
        print("Running without handler (neural only)...")
        runner_without_handler = AutoregressiveVMRunner()
        if opcode in runner_without_handler._func_call_handlers:
            del runner_without_handler._func_call_handlers[opcode]

        output_without, exit_without = runner_without_handler.run(bytecode, max_steps=max_steps)

        # Compare results
        result = {
            'handler_output': output_with,
            'neural_output': output_without,
            'handler_exit_code': exit_with,
            'neural_exit_code': exit_without,
            'match': output_with == output_without and exit_with == exit_without,
        }

        # Print comparison
        print(f"\nResults:")
        print(f"  Handler output: {output_with}")
        print(f"  Neural output:  {output_without}")
        print(f"  Handler exit code: {exit_with}")
        print(f"  Neural exit code:  {exit_without}")

        if result['match']:
            print(f"\n✓ Outputs match! Neural implementation works correctly.")
        else:
            print(f"\n❌ Outputs differ! Neural implementation has issues.")

            if output_with != output_without:
                print(f"   Output mismatch: {output_with} vs {output_without}")
            if exit_with != exit_without:
                print(f"   Exit code mismatch: {exit_with} vs {exit_without}")

        return result


def quick_debug(bytecode, opcode_to_find=None, dim_to_track='AX_CARRY_LO'):
    """Quick debugging helper - trace execution and show dimension flow."""
    from neural_vm.run_vm import AutoregressiveVMRunner
    from neural_vm.embedding import Opcode

    runner = AutoregressiveVMRunner()

    # Remove handler if checking specific opcode
    if opcode_to_find is not None and opcode_to_find in runner._func_call_handlers:
        del runner._func_call_handlers[opcode_to_find]

    tracer = VMExecutionTracer(runner, track_dims=[dim_to_track])
    trace = tracer.trace_execution(bytecode, max_steps=10)

    # Show dimension history
    trace.show_dimension_history(dim_to_track)

    # If looking for specific opcode
    if opcode_to_find is not None:
        ops = trace.find_operation(opcode_to_find)
        if ops:
            print(f"\nFound {len(ops)} {Opcode.__dict__.get(opcode_to_find, opcode_to_find)} operations")
            for op in ops[:3]:
                print(f"  Token {op['token_idx']}, Step {op['step']}")
        else:
            print(f"\n{Opcode.__dict__.get(opcode_to_find, opcode_to_find)} not found in execution")

    return trace


if __name__ == "__main__":
    # Example usage
    from src.compiler import compile_c
    from neural_vm.embedding import Opcode

    code = "int main() { return 10 + 32; }"
    bytecode, data = compile_c(code)

    print("Tracing execution of: 10 + 32")
    print("Looking for where AX_CARRY gets lost...\n")

    trace = quick_debug(bytecode, opcode_to_find=Opcode.ADD, dim_to_track='AX_CARRY_LO')
