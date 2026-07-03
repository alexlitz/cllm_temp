# func step-9 LI FIXED (JSR-phantom penalty) → wall moves to the step-11 RE-READ LEA (2026-06-25)

Worktree base: main `8c458f9c` (the FLIP commit). Campaign config is the
DEFAULT (`C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1` both default-ON), golden
`79292fe5`. spec_k=0, BUILT dims. GPU teacher-forced probes on GPU1.

## TL;DR — the step-9 LI IS fixed by this lane's `C4_L15_LI_JSR_PHANTOM`; the new wall is the step-11 RE-READ LEA

The brief (and the af415346 doc) say func_add/mul/max/min "diverge at step 9 =
LI: the `mem[SP]` operand-load returns ax=0". This lane **fixes that first-param
LI** with the `C4_L15_LI_JSR_PHANTOM` building block (L15 head-0 slot 104, a
MARK_AX-gated −OP_JSR K penalty, default campaign-ON — penalises the callee
JSR/ENT-step phantom MEM rows that share the queried address but carry value 0,
so the genuine PSH'd-arg store wins). **With it, the AUTHORITATIVE production AR
decode** (`tools/_probe_func_aremit.py 575`, STEP_TOKENS=30) shows the first LI
now CORRECT and the wall MOVED to the second LI:

```
step  8: PC=42  AX=65512(0xFFE8)   <- LEA imm=24 (&a) CORRECT
step  9: PC=50  AX=57              <- LI a  CORRECT (returns 57, NOT 0)
step 10: PC=58  AX=57              <- PSH a
step 11: PC=66  AX=57              <- LEA imm=16 (&b): should compute 0xFFE0; WRONG
step 12: PC=74  AX=0               <- LI b  returns 0  <<< FIRST WRONG AX
step 13: PC=82  AX=297             <- ADD  garbage (exp 68)
```

**The first LI (step 9, load `a`) returns 57 — CORRECT.** The earlier
`tools/_probe_func_li.py` "step 9 AX=0" was a STEP-COUNT artifact: that probe
counts 5-token STACK0/MEM register blocks that don't exist in the 30-token frame,
so its step index is mis-aligned. `_probe_func_aremit.py` honours STEP_TOKENS=30
and is the correct AR view.

**The real divergence is step 12 = the SECOND LI (load `b`) → 0**, caused by
**step 11 = the RE-READ LEA (`LEA 16`, &b) computing the WRONG address**
(`_probe_func_axcarry_q.py 575 11 12`): at the LEA-16 AX-marker row (491)
`AX_CARRY = lo9/hi3 = 0x39 = 57` (the LEAKED prior value 57), NOT `0xE0` (&b).
The LI-b query (row 521) then carries `ADDR_B0_LO=[] (empty)`, `ADDR_B0_HI=[3]`
= address `0x30` (wrong) → no store row matches → LI b = 0 → ADD garbage.

This is EXACTLY the af415346 doc's **Bug #2** (the re-read LEA byte-0 is
value-independent-stale). The `C4_FUNC_LEA_REREAD_BP_RESHARPEN` Bug #1 fix
(commit `2b0f8c4e`, NOT in this base `8c458f9c`) re-sharpens the L7 head-1
BP-frame ALU staleness; func_add ALSO needs the FETCH-immediate relay (Bug #2) so
the re-read LEA-16 computes `(BP_byte0 + 16) mod 256 = 0xE0` instead of leaking
0x39.

## Supporting: the FIRST LI (step 9) value path is fully clean

Three GPU teacher-forced probes confirm the step-9 LI lookup is correct (this is
why "fix the LI value delivery" is the WRONG target):

1. The genuine `a`-store value row is present & correctly tagged (probe
   `_probe_func_addr_query.py 575 3 4`): **pos 271 — tok=57, CLEAN=57,
   MEM_VAL_B1=0.97, ADDR_B0 = lo8/hi14 = 0xE8** (the `&a` frame address). The
   store is committed and addressable.
2. The LI query (the REAL LI step, AX-marker row 431) correctly carries
   **OP_LI=5.2 and the right query address ADDR_B0 = lo8/hi14 = 0xE8** (probe
   `_probe_func_axcarry_q.py 575 8 9`). The LEA-24 → LI address handoff is clean.
3. L15 head-0 **attends pos 271 with softmax1 weight = 1.000** (probe
   `_probe_func_winnerrow.py 575 9`): the genuine value row (score 709 871)
   beats the spurious same-address callee JSR/ENT-phantom row (pos 361, value 0,
   score 704 289) by ~5.5k. **This is delivered by the `C4_L15_LI_JSR_PHANTOM`
   building block (L15 head-0 slot 104, a MARK_AX-gated −OP_JSR K penalty,
   default campaign-ON) committed in THIS lane** — without it the more-recent
   value-0 phantom wins the 0.05 ALiBi recency tie by ~5 and the first LI returns
   0 (which is what the brief/af415346 observed). The `C4_L15_LI_VALROW_B1`
   slot-102 MEM_VAL_B1 lift is also load-bearing (it lifts the genuine row into
   contention) but it boosts the phantom EQUALLY, so the JSR-provenance penalty
   is the decisive discriminator.
4. The delivered value 57 SURVIVES to the emission: the AX-marker-row OUTPUT band
   decodes **0x39 = 57 from blk34 (L15) through blk58 (the end)** (probe
   `_probe_func_li_final.py 575 9`). The AX byte-0 token (offset 6) is the argmax
   of the AX-MARKER row (offset 5) logits in the 30-token frame — and that row
   carries 57.

**func_identity behaves IDENTICALLY at its LI step** (value 70 at the AX-marker
row blk34→58) and PASSES. So the LI value path is not what separates passing
func_identity from failing func_add.

## What this means

The step-9 "LI returns 0" reported by the AR full_trace verdict
(`run_1096_canonical --criterion full_trace` / `_probe_func_li.py`) is the
production fixed-N-token-slice reading the WRONG step's tokens after a CUMULATIVE
frame miscount that happens UPSTREAM of step 9. The internal LI computation is
correct (value 57 at the AX-marker row); the decoder just isn't pointed at it
because an earlier step over/under-emitted relative to the 30-token frame.

This is the SAME family as the af415346 doc's **Bug #2** (the re-read LEA byte-0
emits a stale value-independent `0xE8` stamp; the FETCH immediate relay is
recency-stale / absent at the L8 ALU gate row in the 30-token frame). The
LEA-24 → LI → LEA-16 chain has TWO LEA byte-0 computations; if either emits the
wrong byte-0 in AR mode, the emitted AX of that step is wrong and the verdict
"diverges" — and because the LEA precedes the first LI, the verdict pins the
divergence at the LI step even though the LI itself is fine.

## The re-read LEA byte-0 failure — block-by-block (the actionable root)

`tools/_probe_func_li_final.py 575 11` (OUTPUT byte-0 argmax at the LEA-16
AX-marker row 491, block-by-block):

```
blk30..42:  OUTPUT byte-0 = 0x00  [lo50/hi50]   <- lea_lo computed 0x00 (WRONG; &b=0xE0)
blk43:      OUTPUT byte-0 = 0x39  [lo370/hi370]  <- EXPLODES to 0x39 (=57, the leaked prior value)
blk44..58:  OUTPUT byte-0 = 0x39                 <- 57 survives to the emission
```

So TWO compounding faults at the re-read LEA-16 (both must be fixed):

1. **The L8 `_layer8_alu_lea_lo` arithmetic produces 0x00, not 0xE0.** lea byte-0
   `= (ALU_LO[a] + FETCH_LO[b]) mod 16` (gate `MARK_SE_ONLY and OP_LEA`,
   l8_ops.py:412). At the LEA-16 step the AX-marker row has `ALU_LO=nibble-0`,
   `FETCH_LO=nibble-0` → lo = 0 (correct: 0xE0 lo nibble = 0) but the HI nibble
   (0xE) is missing — `ALU_HI` carries the WRONG frame base because the L7 head-1
   BP-frame relay is STALE on the re-read LEA (multiple empty BP markers sit
   between the ENT frame row and the re-read LEA → ALiBi recency picks an empty
   marker). This is **af415346 Bug #1** (`C4_FUNC_LEA_REREAD_BP_RESHARPEN`,
   commit `2b0f8c4e`, NOT merged into this base `8c458f9c`). Cherry-pick/rebuild
   it FIRST — it advanced func_square step 9→11 and is a prerequisite.

2. **Block 43 stamps 0x39 (the leaked prior AX value 57) over the byte-0.** Even
   with the right ALU, a block-43 writer explodes OUTPUT byte-0 to `lo370/hi370`
   = 0x39. This is the campaign-config analogue of the af415346 "blk41 0xE8 tail
   stamp" — but here it is VALUE-DEPENDENT (leaks the actual prior step's AX =
   57), so it is a cross-step OUTPUT leak, not a static stamp. The campaign model
   is **59 physical blocks** (`_probe_func_li_final.py` prints `nblocks=59`), so
   block 43 is in the L15+/L25-tail region (analogue of the af415346 doc's
   "blk41 L25 tail" stamp in the 37-block / 35-token golden). Identify the
   block-43 op (`block_layer_map()` — likely an L25-tail OUTPUT-band corrector or
   a cross-step `AX_CARRY→OUTPUT` / `OUTPUT_HI_PREV_STEP` relay re-broadcasting
   the prior step's AX) and gate it OFF on the LEA step (it must not write OUTPUT
   byte-0 when `OP_LEA` is active and the L8 `lea_lo` rule owns the byte). NOTE
   `block_layer_map()` OOMs under GPU contention — run it uncontended to pin the
   exact owning op.

## The real lever (next lane)

Stop trying to "fix the LI value delivery" — it is already correct. The func
flip is gated by the **AR frame coherence of the re-read LEA byte-0** (the value
of the emitted AX byte-0 token at the re-read LEA-16 step), via the TWO faults
above, per af415346/Bug #2:

* The LEA byte-0 = `(BP_byte0 + imm) mod 256` must be COMPUTED per-step, not fall
  through to the value-independent `0xE8` blk41 tail stamp.
* The immediate FETCH (the per-step `imm` low nibble) is fetched correctly at the
  PC-marker row but is NOT present at the L8 ALU lea-lo gate (MARK_SE_ONLY/AX)
  row in the 30-token frame. Re-sharpen the FETCH PC-row → AX/SE-row relay with
  a query-side OP_LEA gate × a CURRENT-step key (the step's own MARK_PC within
  the 30-token window) so the LEA pulls ITS OWN immediate. This is the same
  staleness shape as the af415346 Bug #1 L7 re-sharpen — one relay over.
* The L7 head-1 re-sharpen (`C4_FUNC_LEA_REREAD_BP_RESHARPEN`, default-ON
  campaign on the af415346 branch) already fixes the BP-frame ALU staleness; it
  advanced func_square step 9→11. func_add/max/min need the FETCH relay too.

## Probes added (this lane, GPU teacher-forced, contention-independent)

* `tools/_probe_func_h0_score.py <id> <li_step>` — L15 head-0 raw-score
  decomposition at the LI query (sink-anchor diagnosis).
* `tools/_probe_func_addr_query.py <id> <lea_step> <li_step>` — ADDR_B0 query/
  store nibbles + value-row tags.
* `tools/_probe_func_axcarry_q.py <id> <lea_step> <li_step>` — every plausible
  address source (AX_CARRY/CLEAN/ALU/OUTPUT/ADDR_B0/FETCH) + OP_* flags at the
  AX-marker row. **This is the probe that shows OP_LEA vs OP_LI per step and the
  per-step ADDR_B0 — it disambiguated the step numbering.**
* `tools/_probe_func_winnerrow.py <id> <li_step>` — head-0 top causal-masked
  rows + the genuine value-row score + the softmax1 winner.
* `tools/_probe_func_li_final.py <id> <li_step>` — block-by-block OUTPUT argmax at
  the AX-marker & emit rows (value-survival trace L15→end).
* `tools/_probe_func_aremit.py <id>` — AR-emitted per-step register decode +
  ntok!=STEP_TOKENS frame-desync flag (the production decode view).

## Step-numbering gotcha (cost an hour)

The probe step index counts STEP_END boundaries in `_final_context`. For
func_add the callee body maps:
`step 8 = LEA imm=24 (&a=0xFFE8, OP_LEA=5.2, FETCH lo-nibble=8)`,
`step 9 = LI (OP_LI=5.2, ADDR_B0=0xE8, value 57)`. An earlier pass probed
"step 8" as the LI and saw OP_LEA + a 0x0B address (the LEA's marker) and
concluded head-0 was sink-dominated — a STEP-OFFSET ARTIFACT. Always confirm the
step with `_probe_func_axcarry_q.py` (it prints OP_LEA/OP_LI per AX-marker row)
before attributing.

## AUTHORITATIVE GPU verdict (`run_1096_canonical --spec-k 0 --criterion full_trace`)

`C4_L15_LI_JSR_PHANTOM` OFF vs ON (default), campaign config, GPU full_trace:

| id  | cluster        | OFF div_step                       | ON div_step                              |
|-----|----------------|------------------------------------|------------------------------------------|
| 575 | func_add       | step 9 (LI a: exp ax=57, got 0)    | **step 11** (re-read LEA &b: exp 0xFFE0, got 57) |
| 600 | func_mul       | step 9 (LI a: exp 49, got 0)       | **step 11** (re-read LEA, got 49)        |
| 650 | func_max       | step 9 (LI a: exp 36, got 0)       | **step 11** (re-read LEA, got 36)        |
| 675 | func_min       | step 9 (LI a: exp 13, got 0)       | **step 11** (re-read LEA, got 13)        |
| 625 | func_square    | step 9 (single-arg LEA)            | step 9 (single-arg LEA, OFF==ON, no JSR competitor) |
| 550-552 | func_identity (HOLD) | PASS                    | PASS (3/3 100%) ✓                       |
| 250-252 | var_simple (HOLD)    | PASS                    | PASS (3/3 100%) ✓                       |

**ADVANCE: func_add/mul/max/min step 9 → step 11.** The first-param LI now
delivers the correct arg (ax 0 → 57/49/36/13); the new wall is the re-read LEA-16
emitting the leaked first value instead of `&b`=0xFFE0 (the TWO faults above).
**0 REGRESS** (HOLD clusters 100% in both states). func_square (single arg, no
2nd PSH/JSR phantom) is OFF==ON, still at its own step-9 LEA wall (orthogonal).

## Gates (this lane) — ALL GREEN

* Byte-identity: `C4_L15_LI_JSR_PHANTOM=0` build → `state_dict_sha256 =
  79292fe58a00830a…` == golden `79292fe5` ✓ (`tools/_isa_golden_hash.py`, CPU).
* `lint_cross_op_attention --flag C4_L15_LI_JSR_PHANTOM`: PASS (0 non-local
  (opcode,context) rows changed).
* `pytest tests/test_smoke.py`: 51 passed (51/51).
* func_identity 550-… + var_simple 250-… HOLD at 100% (GPU full_trace).
