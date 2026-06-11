# AND/MUL MARK_AX end-run: decode-row PROVEN, but downstream-corruption blocked (2026-06-11)

Status: **NOT Wall-4-blocked (decode reads MARK_AX, not SE), but the
div-style operand-cleanup end-run does NOT transfer to AND/MUL** — the
result is materialised and corrupted by ops DOWNSTREAM of the L10/L11/L12
install, outside the allowed surface. No behavioural change landed;
smoke held at the 26-pass baseline (8/8 guardrails). All probes spec_k=0,
hook-free, batched smoke path. Probe tools added:
`tools/probe_and_mul_decode_row.py`, `tools/probe_and_mul_operand_clean.py`.

## 1. The decode-row question — ANSWERED: AND/MUL decode from MARK_AX

`tools/probe_and_mul_decode_row.py` traces OUTPUT_LO/HI at BOTH the binop
MARK_AX row and the MARK_SE_ONLY row across blocks 8..37 (mirroring
`probe_div_decode_row.py`).

* **The MARK_AX row carries the live compute.** OP_AND / OP_MUL are 5.0
  at the AX row and 0.0 at the SE row through all install blocks. The
  AND/MUL OUTPUT materialises at the **AX row** and is what the exit
  decode (L3 head 5 → AX_FULL, prev-step OUTPUT_LO at the MARK_AX row)
  reads — confirmed because the AX-row argmax matches the wrong decoded
  byte (and_basic→0xFFF0, mul_basic→0xF0).
* **The SE row is a dead −240 floor.** SE_ALU_LO/HI are the uniform
  ~−5.0 empty floor, SE_AX_CARRY is empty, SE_OP_AND/OP_MUL = 0. The
  Wave-A `step_end_operand_relay` is **not transmitting** in this build
  (same dead-relay state the CMP Wall-2/Wall-4 frontier hit). So the
  Wave-B `MARK_SE_ONLY` migration of the l10 alu_bitwise / l11 mul_partial
  / l12 mul_combine FFN rules writes to a row that has neither operands
  nor opcode — those migrated rules are inert.

**Conclusion:** AND/MUL are NOT decode-SE-only. Wave B moved the
*declared* compute to SE, but the *live* compute + decode are both on
MARK_AX (the efficient-wrap `bitwise_rules` / `wide_mul_rules` in
`alu_ops.py`, gated MARK_AX). This is the same MARK_AX decode path div
uses — NOT the Wall-4 SE-decode frontier the CMP agents documented.

## 2. Why the div-style end-run does NOT transfer

DIV works because its clean GE-format lookup writes a **single answer
cell** to OUTPUT at the L10 install (block 24), and that one-hot survives
untouched through block 30/L20 to the decode (`probe_div_decode_row.py`:
div `OLO=[(10, 187.15)]` constant from block 24→37; **L20 never spikes
it**). The operand artifact div faced was a pure additive ~5.56 on cell 0;
a constant subtraction cleaned it.

AND/MUL differ on THREE evidenced points (`probe_and_mul_decode_row.py`,
`probe_and_mul_operand_clean.py`):

1. **The result is materialised at L15 (block 25), NOT at the L10/L11/L12
   install.** OUTPUT is empty at the AX row through blocks 12..24; at
   block 25 (logical L15) a downstream op writes the full noisy spread
   `OLO=[(0,23779),(2,2926),(6,2926),(8,2945),(10,3222),…]` — already
   carrying the block-8 operand-gather cell-0 magnitude artifact. The
   real answer cell IS present (mul_basic OLO@10, OHI@2) but is NOT the
   argmax (cell 0's 23779 artifact dominates).

2. **L20 (block 30) spikes OUTPUT cell 8 to ±150M**, flipping the argmax
   to 8/13 → decodes 0xF0 / 0xFFF0. DIV escapes this only because its
   OUTPUT has no cell-8 content for L20 to amplify.

3. **Operand A (ALU bands) is genuinely mis-encoded for high nibbles** at
   the AX row, not merely cell-0-additive. The `A`-value sweep
   (`probe_and_mul_operand_clean.py` style) shows A=0xFF lands at ALU_LO
   cell **8** (= byte-identical to A=0x08), A=0x0F at cell 15, A=0x2A at
   cell 10. So a constant cell-0 subtraction cannot recover 0xFF's
   nibble-15 — `test_and_basic` (0xFF AND 0x2A) is unrecoverable at the
   AX operand level regardless of OUTPUT cleanup. (B = AX_CARRY is clean.)
   This is the documented Wall-1 hybrid magnitude+nibble encoding, fixed
   on the SE row only.

**The install point I am allowed to touch (l10 alu_bitwise / l11+l12 mul /
alu_ops.py efficient wraps) is UPSTREAM of all three corruptors.** A clean
answer written there is either (a) overwritten by the L15 materialisation,
or (b) — if appended as a late post_op after L15 — added as a tiny
`2.0/S≈+1` residual that is invisible against the 23779 noise and then
buried by the L20 ±150M spike. Tested directly: a div-style two-stage
post_op (cell-0 cleanup + rescaled `wide_mul_rules` lookup) appended to
the L11 install **cleaned ALU_LO@0 but produced no net OUTPUT change and
zero smoke movement** (26→26, 8/8 guardrails). The cleanup fired; the
lookup write was swamped.

## 3. What a real fix needs (outside this task's surface)

The end-run cannot close at the L10/L11/L12 install. It needs ONE of:

* **Stop the L15 noisy OUTPUT materialisation for OP_AND/OP_MUL** and let
  a clean install own OUTPUT, exactly as div did (div's FlattenedDivMod
  composite was fully replaced; AND/MUL's FlattenedALUMul / ALUAndOrXor
  GEToBD still materialises OUTPUT at L15). The L15 writer is in
  `l15_ops.py` / the mul-combine GEToBD path — load-bearing for
  memory_lookup; co-design required.
* **Neutralise the L20 (block 30, logical L20) cell-8 ±150M spike** for
  bitwise/mul OUTPUT. L20 has no op file in the allowed surface.
* **Fix the block-8 AX-row operand-gather** so operand A's high nibbles
  land at the correct cells (0xFF→15, not 8). This is the Wall-1 root,
  owned by the phase-999 `residual_alibi_slopes` op (model_ops.py) /
  l7/l8 — explicitly off-limits here (a parallel batch-harden agent owns
  l6/l8/l9). The SE-row gather is already clean; the AX-row gather is not.

Any of these is a multi-op co-design touching forbidden surfaces, i.e.
the same class of "downstream decode/materialisation row mismatch" as the
CMP Wall-4 frontier — just located at L15/L20 OUTPUT rather than the SE
cmp_combine row.

## 4. Bearing on the ~400 if/expr/gcd 1096 programs

The optimistic "div-end-run → unblocks 400 programs" hypothesis does NOT
hold for AND/MUL as-is. Div was special: a self-contained replaceable
composite whose clean install output reached the decode unmolested. AND/MUL
(and by extension the bitwise/mul-heavy 1096 cluster) are gated by the
L15-materialise / L20-spike / block-8-AX-gather chain, which is the
shared operand-gather + downstream-OUTPUT-row surface the CMP/ALU Wall-1…4
analysis already identified as the real frontier.

## Artifacts

* `tools/probe_and_mul_decode_row.py` — AX vs SE decode-row trace for
  and_basic / mul_basic / and_16bit (spec_k=0, hook-free).
* `tools/probe_and_mul_operand_clean.py` — full ALU/CARRY band dump at the
  AX row; shows the operand-A high-nibble mis-encoding + cell-0 artifact.
* `wide_alu_dsl.wide_mul_rules` — parametrised with
  `operand_a_cond_weight` / `operand_b_cond_weight` / `marker_cond_weight`
  / `threshold` (defaults preserve byte-identity; 79/80 dsl tests pass,
  the 1 failure is the pre-existing `rejects_bad_args` width=2 expectation
  unrelated to this change). Mirrors the div agent's
  `wide_div_rules_ge_format` parametrisation so the eventual fix can
  rescale for the ~5-6-magnitude ALU one-hots and ~0.9 AX_CARRY one-hots.
  No behavioural consumer yet — kept as the byte-identity-safe enabler.
