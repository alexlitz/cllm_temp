# ADR-004: Composite ALU pattern (current state vs future ideal)

Status: Accepted, partial (2026-05)

## Context

The ALU was originally instantiated at runtime by `efficient_alu_neural.py`: `ALUMul`, `ALUShift`, `ALUAndOrXor`, and `ALUAddSub` were Python objects whose `forward` methods were invoked from the VM step loop. This kept the ALU outside the bake-once-export path and forced the ONNX exporter to special-case them.

The ideal end-state is "everything is a block of attention + FFN at runtime," which would let us export the entire VM as a single transformer graph with no Python glue. Achieving that requires residual-stream BD↔GE conversion (binary-digit ↔ general-encoding) at the boundaries of each ALU op — work that is partially designed but not yet implemented.

## Decision

Adopt an intermediate "composite" pattern. `ALUMul`, `ALUShift`, `ALUAndOrXor`, `ALUAddSub` are now compiler-installed composites: declarative install-time ops produced by the `LayerCompiler` rather than runtime Python objects. Internally, each composite still wraps multiple sub-FFNs because the full multi-block factoring depends on the BD↔GE work.

This is explicitly an intermediate state, not the end state.

## Consequences

- **Pro:** ALU bakes through the same path as everything else. ONNX export no longer special-cases them.
- **Pro:** The Python step loop shrinks; one fewer place where bake-time and run-time diverge.
- **Con:** Composite internals are still multi-FFN, so the VM is not yet a clean stack of identical attn+FFN blocks. A reader looking at the bake output will see "ALU composite" rather than N homogeneous blocks.
- **Future work:** Implement BD↔GE conversion in the residual stream so each composite can decompose into standalone attn+FFN blocks. Once done, the composite class becomes a layout hint rather than a structural unit, and ADR-004 can be revised to "Accepted (final)."
