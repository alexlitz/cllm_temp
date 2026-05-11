# V17 `_decode_exit_code` — Acceptable Per Spec

**Status:** GREEN (acceptable per `BLOG_SPEC.md`, no migration needed)
**Reviewer:** v17-decode-exit-code-review agent
**Date:** 2026-05-11
**File/lines:** `c4_release/neural_vm/run_vm.py:2001-2009`

## Function under review

```python
def _decode_exit_code(self, context):
    """Extract exit code from the last REG_AX before HALT."""
    for i in range(len(context) - 1, -1, -1):
        if context[i] == Token.REG_AX and i + 4 < len(context):
            val = 0
            for j in range(4):
                val |= (context[i + 1 + j] & 0xFF) << (j * 8)
            return val
    return 0
```

Called from exactly two places in `run_vm.py`:
- Line 669: `return "".join(output), self._decode_exit_code(context)` — the
  final `return` of `AutoregressiveVMRunner.run`, *after* the autoregressive
  generation loop has terminated (either via `Token.HALT` or via the step
  limit). The `model.forward()` is no longer called after this point.
- Mirror call sites in `batched_pure_neural.py` (lines 251, 266, 361, with
  the static method at line 398) — same post-generation pattern.

## Justification

`BLOG_SPEC.md` line 3 forbids "auxiliary memory or python variables" inside
the model's forward pass — i.e., Python state must not be folded back into
the autoregressive computation. `_decode_exit_code` does not violate this:
it runs strictly **after** the generation loop has terminated, reads
**only the token sequence the model emitted** (the `context` parameter,
which is the same array the transformer wrote its outputs into), does not
touch `self._memory` or any shadow register state, and does not modify
`context` or feed anything back into the model. It is the standard
generation-loop boundary — exactly analogous to how a real LLM serving
stack inspects the last sampled tokens to extract a final answer after
`stop` triggers. This is the same category as the "Result extraction"
exemption already documented in `PURE_NEURAL_POLICY.md` §1.1, and the
function is already at the irreducible 9-line minimum (scan backward for
the REG_AX marker, pack 4 little-endian bytes into an int). No
simplification is possible without losing correctness, and no migration is
required: this is a host-side boundary read, not Python state injection.

## Action

Reclassified from yellow (debatable) to green (acceptable per spec) in
`PURITY_AUDIT_2026_05_11.md`. No code change.
