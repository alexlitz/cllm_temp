# if_eq cluster: AX_byte0 pinned by OP_EQ default leak

## Symptom

if_eq_* 1096 tests fail 12/16. Probed cases:
- `if_eq_0` (49==49, expected=1): got 0
- `if_eq_2` (16==9, expected=0): got 9
- `ret 49`: got 0
- `ret 16`: got 0
- `ret 1+1`: got 1 (passes)

## Reframe: CMP is NOT the root cause

The brief framed this as "CMP→branch collapse" but the trace shows
AX_byte0 is broken across the board, including pure-IMM cases.
`ret 49` returning 0 has no CMP involvement.

**AX_byte0 is pinned to OUTPUT_LO+0 (giving 0) or to whatever residual
from the last IMM/FETCH_LO byte in the chain.**

## Likely mechanism

`_cmp_default("EQ", 0)` at `c4_release/neural_vm/unified_compiler/ops/l10_ops.py:665-680`:

```python
# Unconditionally writes OUTPUT_LO+0 = 2/S at MARK_AX + OP_EQ
# (no MARK_PC blocker, unlike the parallel rule at line 518)
```

The parallel `_l10_comparison_combine_rules` at line 518 protects with
`("MARK_PC", -50.0)`. The default rule lacks it.

When OP_EQ fires at a position where MARK_PC is also nonzero
(e.g., the AX-marker position bleeding into PC region), the default's
`OUTPUT_LO+0 = 2/S` clobbers the legitimate AX_byte0 write.

For "16==9 → 9": when CMP→1 doesn't fire (a≠b), the OP_EQ default
writes `OUTPUT_LO+0`. But the FETCH_LO+9 residual from `IMM 9` is also
present in the OUTPUT_LO band, and the residual sum lands on token 9
instead of 0.

## Connection to OPCODE_BYTE_LO SSA rename

The L5 `opcode_decode_ffn` reads `OPCODE_BYTE_LO.*.-1` (Phase 8.A SCC
break). If this causes OP_EQ to assert one step late, it would fire on
IMM steps too — not just EQ steps. The OPCODE_BYTE_LO runtime-trace
agent showed this rename was tested with a metadata revert that didn't
move runtime, but the cross-step alias may still be implicated
behaviorally (the residual carries over across steps).

## Proposed fixes

### Fix A: add MARK_PC blocker to _cmp_default (preferred, one-line)

Mirror line 518's protective pattern:

```diff
 # _layer10_alu_cmp_combine_rules._cmp_default
 conditions = (
     ("MARK_AX", weight),
     ("OP_EQ", weight),
+    ("MARK_PC", -50.0),
 )
-threshold = 1.5
+threshold = 2.5
```

### Fix B: investigate OPCODE_BYTE_LO temporal phase

If OP_EQ is firing one step late due to the L5 SSA rename, the alias
resolution should pin same-step rather than reading from KV cache.
Larger refactor.

## Cross-references

- `c4_release/neural_vm/unified_compiler/ops/l10_ops.py:665-680` — `_cmp_default`
- `c4_release/neural_vm/unified_compiler/ops/l10_ops.py:518-580` — parallel
  rule with the protective `MARK_PC` blocker (template)
- `c4_release/neural_vm/unified_compiler/ops/l5_ops.py:527` — OPCODE_BYTE_LO
  SSA rename (possibly implicated)

## Status

Diagnosis only; no fix applied. Wave 1 candidate (Fix A, one-line).
Connects to CMP-PREV path audit and OPCODE_BYTE_LO investigation.
