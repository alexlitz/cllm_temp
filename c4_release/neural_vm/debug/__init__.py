"""
Neural VM Debug Module.

Provides structured debugging tools for the Neural VM:
- ExecutionTracer: Step-by-step execution tracing
- WeightInspector: Inspect model weights and activations
- StateComparator: Compare two execution traces
- AttentionVisualizer: Visualize attention patterns

Usage:
    from neural_vm.debug import ExecutionTracer, WeightInspector

    # Trace execution
    tracer = ExecutionTracer(verbose=True)
    runner = AutoregressiveVMRunner(tracer=tracer)
    runner.run(bytecode, data)
    tracer.dump("trace.json")

    # Inspect weights
    inspector = WeightInspector(runner.model)
    inspector.show_embedding(Token.REG_PC)
    inspector.show_ffn_activations(layer=5, unit=100)
"""

from .tracer import ExecutionTracer
from .weight_inspector import WeightInspector
from .comparator import StateComparator
from .attention_viz import AttentionVisualizer

__all__ = [
    "ExecutionTracer",
    "WeightInspector",
    "StateComparator",
    "AttentionVisualizer",
]
