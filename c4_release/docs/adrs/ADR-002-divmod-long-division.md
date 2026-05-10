# ADR-002: DIV/MOD via long division (no fp64)

Status: Accepted (2026-05)

## Context

The original DIV/MOD implementation relied on fp64 reciprocal multiplication: compute `1/divisor` in double precision, multiply, floor. This worked at unit-test scope but introduced two systemic problems:

1. **Numeric leakage.** fp64 ops fall outside the activation budget the rest of the VM enforces (everything else lives in fp32 / SwiGLU range). Mixing precisions complicated the export path (ONNX) and the residual-stream invariants used by other layers.
2. **Non-neural escape hatch.** A reciprocal multiply is not expressible as attention + FFN with the rest of the VM's primitives. It made DIV/MOD a special case the pure-neural runner had to skirt around, blocking the broader "all ops are blocks" goal.

The alternatives considered were attention-based comparison plus log-domain subtraction, and digit-by-digit long division.

## Decision

Replace fp64 with a long-division algorithm that uses only the existing primitives: subtractive comparison, conditional carry, and nibble-shift rotation. Documented in `docs/DIV_ALGORITHM.md`.

## Consequences

- **Pro:** DIV/MOD now run end-to-end in the pure-neural path. No precision crossings.
- **Pro:** Same primitives the rest of the VM uses — no special-case bake.
- **Con:** Higher layer cost (one block per quotient bit) than fp64 reciprocal. Acceptable because layer count is not yet the binding constraint.
- **Con:** Slightly less precision headroom than the attention+log alternative would have provided, but that path required new primitives we didn't want to introduce mid-migration.
- **Tradeoff vs attention+log:** Long division is simpler to bake declaratively and reuses primitives we already have. Attention+log would be faster (O(log n) layers) but needs new range-extension primitives.
