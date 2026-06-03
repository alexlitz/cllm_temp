# edge_pow2 cluster: OP_IMM gate leak in L16 PSH MEM rule

## Symptom

edge_pow2_* 1096 tests fail in lookup mode. Probed case: `int main()
{ return 8; }` → bytecode `[IMM|(8<<8), EXIT]`. Decl=8, neural=65512
(0xFFE8 = 16-bit signed -24).

All pow2 cases except 2^0 fail. Same fingerprint.

## Trace

`/tmp/canary_pow2.py` reproduces. Per-layer head argmax tracing at pass 6
(predicting AX_byte0 of IMM step):

| Block | Head argmax | Logit |
|------:|-------------|------|
|  L31  | 0x08        | +27.7  (correct) |
|  L32  | **0xE8**    | +1010 (flipped — bug) |

L32 writes **+200.0 into residual dim 99** (`ADDR_B1_VALID`, aliased
`H5+4`). `head.weight[token=232, dim=99] = +5.0` → +1000 logit lift for
token 0xE8.

## Root cause

`l16_psh_mem_addr0_e0_from_sp_no_addr_src` rule in
`c4_release/neural_vm/unified_compiler/ops/l16_ops.py:736-762`. Writes:

```python
writes=byte_value_writes(0xE0, strength=200_000)
# one-hot 0xE0 → OUTPUT_LO+0, OUTPUT_HI+14
```

Conditions include `("OP_IMM", -1_000_000.0)` as a "block", but the
comment at lines 747-756 admits the gate is intentionally crossable
for ENT-main relays — IMM and ENT are treated symmetrically.

For IMM=8 in lookup mode:
- `ALU_LO+8` activates strongly (low nibble of 8 matches).
- IMM marker at MARK_AX attenuates to ~1e-3 by upstream broadcast.
- So -1_000_000 × 1e-3 ≈ -1000, which DOESN'T overcome the positive
  signal sum crossing threshold=8.5.
- Rule fires wrongly on a literal-IMM step.
- The +200 leak into `ADDR_B1_VALID` is bias-spill from this rule's
  bake (H5+4 = OUTPUT_HI_THIS_STEP region edge case in dim packing).

## Proposed fixes

### Fix A (one-line, preferred)

Strengthen OP_IMM block: `-1_000_000.0` → `-1e9`. Even with the ~1e-3
attenuation, -1e9 × 1e-3 = -1e6, which dominates the +5 positive sum.

```diff
-    ("OP_IMM", -1_000_000.0),
+    ("OP_IMM", -1e9),
```

### Fix B (semantic, larger)

Add a positive predicate. Currently `MEM_ADDR_SRC: -1000.0` is a
negative blocker, not a positive memory-step gate. Add a positive
`MARK_MEM` or `OP_PSH` discriminator so the rule only fires when an
actual PSH-SP-no-addr-src memory write is in flight.

### Validation

Either fix should make AX_byte0=8 for `IMM 8; EXIT`. Then run the 11
edge_pow2 cases.

## Cross-references

- `c4_release/neural_vm/unified_compiler/ops/l16_ops.py:736-762` — the rule
- `c4_release/.agent-logs/` — canary scripts in `/tmp/canary_pow2*.py`
- Token table: `0xE8 = 232`, `ADDR_B1_VALID = H5+4 = dim 99`
- Head unembed: `head.weight[232, 99] = +5.0`

## Status

Diagnosis only; no fix applied. Wave 1 candidate (Fix A, 1-line).
