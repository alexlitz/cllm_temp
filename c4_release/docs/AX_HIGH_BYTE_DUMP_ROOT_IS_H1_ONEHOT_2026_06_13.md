# AX byte-1 DUMP truncation — CORRECTED root via logit attribution (2026-06-13)

**Status:** ROOT RE-LOCATED. This supersedes the `STACK0_BYTE_VAL` hypothesis
in `AX_HIGH_BYTE_DUMP_TRUNCATION_2026_06_13.md`. The byte-1 register-dump value
is **NOT** read from `STACK0_BYTE_VAL_*` (that band is correlated but NOT
causal). It is read by the LM head from the **`H1` band (L0 attention head 1,
dims 67..73) as a one-hot `H1+(byte_value+2)`**, materialised only on the
producing (fresh-AX) step and never carried. Because `H1` is 7-wide it can only
encode high-byte values 0..4 — this is a **narrow, autoregressive,
positional-head encoding**, not a value-general band, so the value-general
multi-byte AX carry-forward the brief asks for cannot just "re-deposit a band":
it must ALSO re-point the dump emission to read a real 8-bit value band.

The fix was NOT landed this session (no production weights changed). This doc
records the definitive attribution so the next pass builds the right thing.

## Repro (unchanged, re-confirmed spec_k=0, hook-free)

`tools/probe_ax_carry.py` on `add_0` (`int main(){return 654+114;}` →
`IMM 654; PSH; IMM 114; ADD; EXIT`):

| step | op   | AX bytes [b0,b1,b2,b3] | decoded | oracle | verdict        |
|------|------|------------------------|---------|--------|----------------|
| 0    | IMM  | [142, **2**, 0, 0]     | **654** | 654    | correct        |
| 1    | PSH  | [142, **0**, 0, 0]     | **142** | 654    | byte1 dropped  |
| 2    | IMM  | [114, 0, 0, 0]         | 114     | 114    | correct        |
| 3    | ADD  | [0, 3, 0, 0]           | 768     | 768    | correct        |

The ADD result (step 3) is byte-correct; the truncation is ONLY the
intermediate PSH-step dump of the persisted operand. full_trace fails at step 1.

## THE definitive evidence: LM-head logit attribution

The model has **no final norm** before the head (`vm_step.py:1839`:
`return sparse_linear(x, head.weight, head.bias)` reads the raw post-block-39
residual). So `logit[t] = head.weight[t] · residual + head.bias[t]` is exact and
fully attributable. `tools/probe_ax_logit_attrib.py` decomposes
`(logit[0x02] − logit[0x00])` at the byte-1 predictor row (= the AX byte-0
value token, whose next-token logits emit AX byte 1):

| step             | byte-1 emit | logit[0x02] | logit[0x00] | top driver        |
|------------------|-------------|-------------|-------------|-------------------|
| IMM (correct)    | 0x02        | 228.4       | 182.9       | **H1+4 = 12.14, contrib +60.7** |
| PSH (wrong)      | 0x00        | 84.7        | 90.4        | H1+4 = 0.0, only residual H1+2 |

The emission decision is driven **>99% by the `H1` band** (`H1+4` carries
`head.weight[0x02]−head.weight[0x00] = +5.0`). `OUTPUT_LO/HI`, `AX_CARRY_*`,
`AX_FULL_*`, and `STACK0_BYTE_VAL_*` contribute ~0 to this token decision at
this row. The earlier doc's `STACK0_BYTE_VAL` 45.74 diff is real but *acausal*
for the emission — it co-varies with the program without feeding `head.weight`.

## The `H1` value encoding (one-hot, capped at 4)

`H1` (dim 67, width 7, "L0 head 1: marker within dist 4.5") holds a one-hot of
the high-byte value on the producing step (spec_k=0, IMM-step byte-1 predictor):

| program  | byte1 | H1 band (dims 67..73)            | one-hot slot |
|----------|-------|----------------------------------|--------------|
| 300+0    | 1     | [_, _, 2.8, **12.0**, 0.1, 0, 1.5] | H1+3 |
| 654+114  | 2     | [_, _, 2.8, 0, **12.1**, 0, 1.5]   | H1+4 |
| 913+558  | 3     | [_, _, 2.8, 0, 0.1, **12.0**, 1.5] | H1+5 |
| 5000+0   | 19    | [_, _, 2.8, 0, 0.1, **12.0**, 1.5] | H1+5 (WRONG: caps at slot 5 → reads 3, not 19) |

Encoding = `H1 + (byte_value + 2) = 12.0`, valid only for `byte_value ∈ 0..4`
(7 slots). Values ≥ 5 already emit wrong (separate, pre-existing ceiling). The
whole 1096 add/sub corpus has high byte ≤ 4, which is why this narrow encoding
"works" there at all.

## Why byte 0 survives the PSH step but byte 1 does not

Byte 0 emits 142 correctly on BOTH steps and does NOT depend on `H1` (its
byte-0 predictor `H1` is `[_,_,1.3,0,0,0,0]` on the PSH step yet still emits
142). Byte 0 has a real carry path: L3 head 1 carries prev-step AX byte 0
`EMBED → AX_CARRY_LO/HI`, and opcode-specific `AX_CARRY → OUTPUT` materialisers
(`l16_lev_ax_carry_*`, the PUTCHAR units in `model_ops.py:120`) project it.
**Byte 1 has NO carry band and NO carry→dump materialiser at all** — its only
emission path is the fresh-step `H1` one-hot, which is regenerated each step
from the *emitted* byte-1 token (autoregressive). On a non-AX-writing step the
upstream state no longer holds 654, so the `H1` one-hot is absent and byte 1
emits 0x00. The doc's "no analog for AX bytes 1..3" is exactly right; the new
finding is *what the missing analog must feed* (a value-general dump band, NOT
`STACK0_BYTE_VAL`).

## Why this is a genuine multi-component re-architecture (not a single head/rule)

A correct value-general byte-1 carry needs BOTH halves, coordinated:

1. **A byte-1 carry band** (mirror byte 0's `AX_CARRY` head 1): a new L3
   carry-forward head keyed on `MARK_AX + IS_BYTE + BYTE_INDEX_1`, K-selecting
   the prev step's byte-1 row (the byte-0 head's `L1H1 vs L1H0` selector
   generalised to the byte-1 row), V-copying the prev byte-1 nibbles into a
   fresh per-byte carry band. L3 is **at its 8-head budget** (heads 0..7 all
   owned by `layer3_carry_forward_attn`); this needs a minimal widen or head
   sharing.
2. **Re-point the byte-1 dump emission to read that carry band.** This is the
   hard part: the emission currently reads the 7-wide `H1` positional head as a
   one-hot capped at 4. A value-general dump must read a real 8-bit value band
   (16+16 nibble pair) at the byte-1 predictor row, which means changing the
   LM-head-facing residual the byte-1 row carries — i.e. baking a new
   value→logit path, not just moving a number. The bounded `H1+(v+2)` form is
   exactly the brittle enumerated pattern the brief warned about
   (`_layer3_pc_byte1_output_rules` for PC byte 1), and capping at 4 leaves
   `byte1 ≥ 5` broken anyway.

A `STACK0_BYTE_VAL` re-deposit head (the L13 `sub_minuend_relay` /
`add_addend_relay` pattern, heads 6/7 free at L13) would NOT fix the dump: the
logit attribution proves the head does not read that band for the dump token.
That pattern is the right tool for the ALU *operand* relays (where the L10/L14
adders DO read `STACK0_BYTE_VAL`), but it is the wrong tool for the *register
dump* emission.

## Recommended next pass

Either (a) build the value-general two-part subsystem above (byte-1 carry band +
a new 8-bit byte-1 dump emission band read by the LM head), byte-identity gating
each half; or (b) if only the add/sub corpus matters short-term, an enumerated
`H1+(v+2)` carry for `v ∈ 0..4` gated on non-AX-writing steps — accepting it is
brittle and corpus-bounded (and still leaves `v ≥ 5`). Option (a) is the real
fix; it is multi-session per the brief and overlaps the convergent multi-byte
tasks (#206/#213/#217).

## Tools added this session (kept, reusable)

* `tools/probe_ax_carry.py` — per-step AX-byte decode + repro (REG_AX-marker
  anchored; the emitted stream is NOT 35-aligned, locate markers).
* `tools/probe_ax_logit_attrib.py` — exact LM-head logit attribution at any
  predictor row (no final norm → `head.weight·residual` is the whole logit).

Both spec_k=0, hook-free, use `tools/probe_groundtruth.GroundTruthProbe` and
`build_default_registry_dynamic()` for true (dynamic) dim positions (NOT the
legacy `embedding.E` map, whose positions differ).
