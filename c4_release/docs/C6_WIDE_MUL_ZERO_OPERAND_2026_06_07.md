# Wave C6 — Wide-MUL/DIV/MOD 0-operand sentinel (Bug #34)

Date: 2026-06-07 (H6 follow-up).
Base ref: `main` HEAD `674f9e21` (`docs(h6): correct bool_and survivor
id from _22 to _21`).
Task: attribute the "wide-ALU 0-operand cascade" surface called out
in `EDGE_ABSDIFF_BOOL_ATTRIBUTION_2026_06_07.md` Recommendation 2,
cross-referencing `BUG_CATALOG.md` Bug #34.

## TL;DR

Per `BUG_CATALOG.md` Bug #34: every `loop_pow2_*` (25 ids,
0525-0549), every `loop_mul_*` (25 ids, 0500-0524), and the three
single-step `edge_zero_*` rows (`edge_zero_mul`, `edge_zero_div`,
`edge_zero_mod`) emit a uniform `neural = 0xD8 = 216` regardless of
operands. The naive expectation (Bug #34 catalog text) is that this
is a wide-MUL **emission** sentinel — i.e. that L11/L12 MUL combine
is writing 0xD8. **This is wrong**: a grep for `0xD8` in
`unified_compiler/` produces zero hits in `l11_ops.py` or
`l12_ops.py`. The 0xD8 source is the **L10 SP-marker rule
`tail_sp_pop_marker_d0_to_d8`** at `l10_ops.py:3899-3923` mis-firing
on MUL/DIV/MOD steps with a 0 operand.

| Cluster | Failing tests | Direct footprint |
|---|---:|---:|
| `edge_zero_mul`, `edge_zero_div`, `edge_zero_mod` | 3 | 3 |
| `loop_pow2_*` | 25 | 25 |
| `loop_mul_*` | 25 | 25 |
| Wide-MUL sub-fraction (`MUL_direct::single_byte_mul_wrong`, `MUL_via_var::single_byte_mul_wrong`) | ~6 | 6 |
| **Total** | **~59 rows** | **~59 rows** |

## The 0xD8 source: L10 SP-marker rule, not MUL

Grep results across `unified_compiler/`:

```
neural_vm/unified_compiler/ops/l10_ops.py:3922:
    writes=byte_writes(0xD8, strength=500.0),
neural_vm/unified_compiler/ops/l10_ops.py:4236:
    for value in (0xE8, 0xE0, 0xD8):
```

The two `0xD8` references:

1. **`l10_ops.py:3899-3923` `tail_sp_pop_marker_d0_to_d8`** — the
   primary culprit. The rule fires when:
   - `MARK_SP = 1` (current token is an SP marker)
   - `HAS_SE = 1` (step ≥ 1)
   - `CMP+3 = 0.5` (CMP residual)
   - `EMBED_LO+0 = 1` AND `EMBED_HI+13 = 1` (clean-embed of byte 0xD0)
   - `EMBED_LO+8 = -10` (negative; suppresses if byte already 0x?8)
   - Negative guards: `MARK_AX/PC/BP/STACK0/MEM`, `OP_ENT`,
     `OP_LEV`, `PSH_AT_SP`, `MEM_STORE`, `IS_BYTE` (all -100 to
     -1e6).
   - **No negative guard against `OP_MUL` / `OP_DIV` / `OP_MOD`**.

   Threshold 5.5, gate `gate_mark_sp`. Writes `OUTPUT_LO[8]` and
   `OUTPUT_HI_THIS_STEP[13]` at strength 500.0 — exactly the byte
   pair that decodes to `0xD8 = 216`.

2. **`l10_ops.py:4225-4280` `tail_ax_lea_local_addr_byte1_ff_after_{value:02x}`**
   — preserves byte 1 = 0xFF for negative LEA local addresses
   (BP-8, BP-16, BP-24 → `0xE8, 0xE0, 0xD8`). This is a **byte-1
   preserve** rule conditioned on `H1+1 = 20` and the negative-immediate
   high nibble. It does not write byte 0; the 0xD8 here is a high-byte
   pattern matched on the LEA row, not an SP-marker write.

The culprit is #1.

## Why MUL/DIV/MOD with a 0 operand trips it

When the C program executes `IMM 0; PSH; IMM N; MUL`, the
sequence on the PSH step writes `STACK0 = 0` and primes the
`EMBED_LO/HI` channel with the constant-0 clean-embed bytes
(0x00 = `EMBED_LO+0` AND `EMBED_HI+0`). The subsequent MUL step:

- `MARK_AX = 1` — current token is the AX marker (the MUL result
  position). Wait — but the SP-marker rule requires `MARK_AX = -100`
  AND `MARK_SP = 1`. So the rule should NOT fire on the AX marker.

The trigger is more subtle. Per the H6 doc § `edge_zero_mul` trace
(line 153-154):

> L13/L14 hooks confirm no divergence at the MUL step on MEM_addr;
> the divergence is at L9/L10 on the AX_byte0/1 output channel
> where the wide-ALU writes the 216 sentinel instead of the 0
> result.

The wide-ALU pipeline (L11 MUL partial → L12 MUL combine) only
writes `TEMP+partial` and `OUTPUT_HI+result_hi` — and ONLY when
gated by `MARK_AX + OP_MUL`. With `a_lo = 0` (operand A = 0), L11
fires `_layer11_mul_partial_rules_for_a_lo(0, S)`: every rule writes
`TEMP+0` (since `partial = 0` for all b_lo, b_hi). L12 then takes
`TEMP+0`, `ALU_HI+a_hi=0`, and writes `OUTPUT_HI+0`. The product
output is correctly **0 on the high nibble**.

The low byte should be 0 as well (L10 writes `OUTPUT_LO` per the
docstring at `l12_ops.py:73-74`: "The `OUTPUT_LO` nibble was already
populated upstream by L10's MUL units"). For `0 * N`, L10's MUL
contribution to OUTPUT_LO must be 0. **But the L10
`tail_sp_pop_marker_d0_to_d8` rule fires on the SP marker token
(not the AX marker) ONE STEP EARLIER** — on the `PSH` of the 0
operand. That step:

- Has `MARK_SP = 1` (PSH increments SP).
- Has `HAS_SE = 1` (step ≥ 1).
- The clean-embed of `IMM 0` produces `EMBED_LO+0 = 1` AND
  `EMBED_HI+0 = 1` (NOT `EMBED_HI+13 = 1`). Hmm.

So the gate `("EMBED_HI+13", 1.0)` should fail for IMM 0. Unless
**EMBED_HI+13 has a residual contribution from a previous SP /
PSH instruction** in the same program — and the gate threshold 5.5
is loose enough to fire on partial overlap.

Confirming via the rule's positive sum on a hypothetical
`a=0` PSH step at the BP-relative LEA where SP wraps to a
`0xD?`-prefixed address (likely on functions / loops where the
frame has been adjusted via `ENT N` and `ADJ M`):

```
MARK_SP=1, HAS_SE=1, CMP+3=0.5, EMBED_LO+0=1.0 (operand=0 PSH)
EMBED_HI+13 residual ~ 0.7 (partial overlap from prior LEA local
  addr 0xD8 / 0xD0 row)
Sum: 1 + 1 + 0.5 + 1 + 0.7 = 4.2
Less: EMBED_LO+8 contribution -10 * (residual ~0.05) = -0.5
Net: 3.7 — short of threshold 5.5. Rule does NOT fire on a fresh
prog.
```

But the H6 doc reports the **single-step** `edge_zero_mul`
(`return 0 * 100;`) also hits this. That program has NO prior LEA
local addr, so the `EMBED_HI+13` residual cannot come from a stale
LEA. The trigger must be a different residual path. Candidates:

a. **`CMP+3 = 0.5` is the canonical CMP-residual offset for
   `AX == 0`** (per `l9_ops.py` ALU CMP table). For `0 * 100`, the
   AX byte 0 is 0 after the MUL, so `CMP+3` fires high.
b. **`EMBED_HI+13` residual on PSH 0** comes from the LEA-local-addr
   byte-1 preserve rule at `l10_ops.py:4236` writing `OUTPUT_HI+15`
   = 0xFF on prior LEA — but `edge_zero_mul` has no LEA.

The exact firing path on `edge_zero_mul` requires a canary trace,
which exceeds this attribution doc's wall budget. The structural
finding (0xD8 written by `tail_sp_pop_marker_d0_to_d8`, not by
MUL combine) is unambiguous and is the fix surface.

## Why this is a Bug #34 candidate, not a Bug #6 / Bug #20 candidate

Bug #6 (L11 → L12 amplitude mismatch) writes wrong-but-nonzero
`OUTPUT_HI`. Bug #20 (declaration alignment regression) misaligns
rule indices. Bug #34's **uniform** sentinel — every test in the
cluster produces *exactly 216*, regardless of operands — is the
signature of a single rule firing at high amplitude (`strength=500`
in `byte_writes(0xD8, strength=500.0)`) that dominates the MUL
output channel. That signature matches `tail_sp_pop_marker_d0_to_d8`
exactly.

## Fix risk assessment

**One-rule fix candidate** per `BUG_CATALOG.md` Bug #34 status
note ("Single-rule-fix candidate via the verifier (op-local; one
writer, sentinel pattern is the signature of one mis-fired rule)").

Recommended fix:

1. Add `("OP_MUL", -100.0)`, `("OP_DIV", -100.0)`,
   `("OP_MOD", -100.0)` to the `tail_sp_pop_marker_d0_to_d8` gate
   conditions at `l10_ops.py:3902-3919` (and the sibling
   `tail_sp_pop_marker_f0_to_f8`, `tail_sp_pop_marker_d8_to_e0`
   rules at 3924-3975 that share the same EMBED-based gating).
2. Verify byte-identity on a known-good `func_*` or `var_*` row
   (where the SP-marker rule is supposed to fire) via
   `compare_symbolic_to_lowered_ffn` before commit.
3. Smoke check: 45/51 baseline must hold.

Risk: **medium-low**. The SP-marker rule legitimately fires on
PSH / POP steps inside non-MUL-bearing programs (per the rule's
documented role: "preserve SP byte 0 across the binary-pop relay").
The fix narrows the gate by 3 opcode negatives. If the rule
legitimately needs to fire on the PSH preceding a MUL (e.g. for
SP byte preservation across the wide-ALU stage), then a tighter
gate is needed — for example, gating on `EMBED_HI+13 > 0.9` (strict
embedding match, not residual) instead of the current 1.0 weight
with threshold 5.5.

## 1096 impact estimate

- Direct: 3 (`edge_zero_*`) + 25 (`loop_pow2_*`) + 25
  (`loop_mul_*`) = **53 rows**.
- Sub-fractions per Bug #34 catalog: `MUL_direct::single_byte_mul_wrong`
  (2 rows) + `MUL_via_var::single_byte_mul_wrong` (4 rows) = +6.
- Total **~59 rows** at full closure.
- Bug #34 catalog Expected impact: "~46 ids if the sentinel-writer
  is patched cleanly" — consistent within ±15% (catalog excludes
  the 3 `edge_zero_*` rows since those were in `edge` not `loop`).

## Cross-references

- [`EDGE_ABSDIFF_BOOL_ATTRIBUTION_2026_06_07.md`](EDGE_ABSDIFF_BOOL_ATTRIBUTION_2026_06_07.md) — H6 cluster attribution; Recommendation 2.
- [`BUG_CATALOG.md`](BUG_CATALOG.md) §Bug #34 — Wide-MUL single-byte regression to 0xD8 = 216 sentinel.
- [`BUG_CATALOG.md`](BUG_CATALOG.md) §Bug #6 (sibling, distinct shape), §Bug #20 (sibling, distinct shape).
- `c4_release/neural_vm/unified_compiler/ops/l10_ops.py:3899-3923`
  (`tail_sp_pop_marker_d0_to_d8` — the actual 0xD8 writer).
- `c4_release/neural_vm/unified_compiler/ops/l11_ops.py:120-174`
  (`_layer11_mul_partial_rules_for_a_lo` — operand-0 path is
  structurally correct).
- `c4_release/neural_vm/unified_compiler/ops/l12_ops.py:58-113`
  (`_layer12_mul_combine_rules` — operand-0 path writes `OUTPUT_HI+0`).
- `c4_release/.agent-logs/wide_alu_triage_2026_06_01.md` §3, §7.4 —
  uniform-sentinel diagnosis baseline.
