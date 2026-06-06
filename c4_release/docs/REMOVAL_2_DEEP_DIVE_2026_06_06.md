# Removal 2 deep dive — block-by-block trace of the IMM→binop collapse

Date: 2026-06-06
Base: `speedup-cache-and-buckets` HEAD `f4f9103d`
Parent finding: `docs/REMOVAL_2_FINDINGS_2026_06_05.md` (commit `2fc628e3`)
Probe scripts: `/tmp/probe_removal2_step2_emit.py`, `/tmp/probe_block_chain.py`,
`/tmp/probe_removal2_output_dims.py` (preserved this session).

## TL;DR

The Removal-2 brief framed the symptom as "the IMM step writes
`+2*INSTR_WIDTH` into NEXT_PC". The block-by-block residual trace
**refutes that framing**: when the prefix ends at the PSH STEP_END
(teacher-forced or autoregressively generated), the very next emit
predicts the **correct** PC byte (26 = idx 3 = SUB), with
`OUTPUT_LO+10` and `OUTPUT_HI+1` cleanly activated by L3's `pc_increment`
rule. There is **no extra +INSTR_WIDTH writer** in any block at that
position.

The real bug fires one step later, **inside** the IMM-8 step, at the
**STACK0 marker position** (token #132 of the actual autoregressive
context). The model emits a well-formed Block A (`REG_PC=26, REG_AX=8,
REG_SP=0xfff8, REG_BP=0x010000, STACK0`) up to and including the
STACK0 marker, then — instead of emitting STACK0 byte 0 — emits
`REG_PC` and starts a **second** register block (Block B) whose
`REG_PC=34` (idx 4 = EXIT) and `REG_AX=0`. Block B is followed by a
proper MEM/STEP_END so the runner sees ONE 35-token step block, but the
LAST `REG_PC` marker (in Block B) is what `_extract_register` reads,
giving `last_pc=34`.

So the model is not "advancing PC by +2"; it is **suppressing STACK0
byte emission** at the STACK0 marker, which forces the default
"emit next marker" path, which then emits `REG_PC` because the model has
also computed that the IMM step "consumes" the next binop (i.e. SUB)
inside the same step block.

## Probe 1 — confirm the bug at the byte level

`/tmp/probe_removal2_step2_emit.py` monkey-patches
`_dispatch_pure_neural` to snapshot `s.context` at each STEP_END for
`IMM 50; PSH; IMM 8; SUB; EXIT`. Step 2's emitted tail (tokens 112..167
inside `contexts_after_step[2]`) decodes as:

| pos | token | meaning |
|-----|-------|---------|
| 112 | 257 | REG_PC (start of Block A) |
| 113-116 | 26, 0, 0, 0 | PC = 26 (**correct**: idx 3 = SUB) |
| 117 | 258 | REG_AX |
| 118-121 | 8, 0, 0, 0 | AX = 8 |
| 122 | 259 | REG_SP |
| 123-126 | 248, 255, 0, 0 | SP = 0xfff8 |
| 127 | 260 | REG_BP |
| 128-131 | 0, 0, 1, 0 | BP = 0x010000 |
| 132 | 268 | STACK0 marker |
| **133** | **257** | **REG_PC** ← **BUG: byte expected, marker emitted** |
| 134-136 | 34, 0, 0, 0 | PC = 34 (Block B; wrong) |
| ... | | ... |
| 167 | 262 | STEP_END |

Block A's PC is **correct**. The bug is at pos 132 → 133: the model
predicts REG_PC instead of a STACK0 byte.

## Probe 2 — per-block residual at the failing position

`/tmp/probe_block_chain.py` runs the actual runner to capture the
autoregressively generated prefix `full_ctx[:133]`, then re-runs the
model with hooks that snapshot the residual at `probe_pos=132` (the
STACK0 marker) before and after each of the 36 transformer blocks.
The model has `OUTPUT_LO` at base dim 69 and `OUTPUT_HI` at base 85
(both 16-wide one-hot nibble bands). The full table (selected rows):

| Block | FFN units | Op identity                          | max\|OUTPUT_LO\| POST | delta range LO              |
|------:|----------:|--------------------------------------|----------------------:|-----------------------------|
| 26 | 1882 | (various)                             | 4.96e-5  | [+0, +0]                       |
| **27** | **42**   | **`layer15_nibble_copy`**             | **+40**     | **[+0, +40]**                    |
| **28** | **792**  | **`layer16_lev_routing`**             | **+3 907**  | **[+0, +3 868]**                 |
| 29-33 | 1..1 562  | (no writes at this pos)               | +3 907   | [+0, +0]                       |
| **34** | **2 059**| **`tail_bit32_result_correction`**    | **+6.29e8** (abs) | **[−6.29e8, −1.96e7]**     |
| 35 | 192 | `post_l9_bz_bnz_pc_override`          | +6.29e8 (abs) | [+0, +0]                |

The final residual at `OUTPUT_LO+k` and `OUTPUT_HI+k` is then in the
−6e8 range, which crushes every byte logit below −5e9. The lm-head's
marker-token logits sit at ≈−10 (a uniform floor) and the argmax
tie-break selects token 257 (`REG_PC`) because no marker dim
(`NEXT_PC`/`NEXT_STACK0`/...) was set positive at this position.

The fault chain is therefore:

1. **L27 (`layer15_nibble_copy`)** writes a +40 spike to `OUTPUT_LO+2`
   and `OUTPUT_HI+3` at the STACK0 marker (encoding byte 0x32 = 50,
   the value that was PSH'd one step ago).
2. **L28 (`layer16_lev_routing`)** reads OUTPUT_LO/HI back as gate
   evidence and **amplifies the +40 spike to +3 868** (×97). Block 28
   is dominated by the `lev_stack0_preserve_*` family at
   `l16_ops.py:285-300` and the `stack0_byte_preserve` and
   `stack0_zero` rule bands. The preserve gates are `gate=OUTPUT_LO+k`
   / `gate=OUTPUT_HI_THIS_STEP+k` (l16_ops.py:290/298) with
   `write_scale = 50.0 / S = 0.5` — pre-existing OUTPUT activations
   multiply each unit's contribution back into the same cell, which is
   the amplifier.
3. **L34 (`tail_bit32_result_correction`)** then reads the now-huge
   OUTPUT values as evidence for hundreds of `stack0_*` corrective
   rules at `l10_ops.py:3884-4150`. Each rule has conditions like
   `("OUTPUT_LO+lo", 0.1)` + `("OUTPUT_HI_THIS_STEP+hi", 0.1)`
   (l10_ops.py:3962-3963), `gate=gate_mark_stack0`, threshold 10.5,
   and **writes `byte_writes(value, strength=500.0)` — i.e. +500 to
   the matching nibble and −500 to all 15 competing nibbles per side**.
   With OUTPUT_LO+2 = 3 868 carried in, the 0.1-weight condition alone
   contributes +387 to the score, pushing many of the 255 rules in
   `stack0_pop_loaded_output_rules` past threshold simultaneously.
   255 rules × ≈770 hidden × −500 strength = ≈−1e8 per non-matching
   cell, summed across two rule families lands at the observed
   −6.29e8.

## Why "+2 * INSTR_WIDTH" was the wrong framing

The parent doc inferred the bug from `s.last_pc = 34` after step 2.
But Block A inside step 2 carried `REG_PC = 26` (the correct +1
advance). The runner's `_extract_register` scans **backwards** for the
last marker, so Block B's `REG_PC = 34` shadows Block A's value. The
model is not over-incrementing NEXT_PC; it is **emitting a second
register block** because the STACK0 byte-emit lane is being crushed.

So the load-bearing L6/L7/L9/L10/L14 positives the parent doc
suspected are red herrings. The actual bug is in the **gate-coupled
amplification cascade L27 → L28 → L34** at the STACK0 marker
position whenever STACK0 carries a nonzero pushed value.

## Why this is universal across binops

The cascade fires whenever:
- The previous step pushed a value (so STACK0 holds something
  nonzero, here 0x32 = 50), AND
- The current step's STACK0 marker position has `MARK_STACK0` and
  `HAS_SE` active (i.e. step >= 1).

ADD, SUB, MUL, DIV, MOD, SHL, SHR, EQ, NE, LT, GT, LE, GE all share
this pattern: `IMM A; PSH; IMM B; <binop>; EXIT`. The IMM-8 step is
the second-IMM step after a PSH; its STACK0 marker sees a nonzero
pushed value and the cascade fires. The Removal-2 doc's report of
"universal across binops" is therefore expected.

## Recommended fix

The cascade has three candidate cut points; ordered by surface area:

1. **L28 (smallest cut)** — `l16_ops.py:285-300`
   `l16_lev_stack0_byte0_preserve_{lo,hi}_{k}`. These 32 rules use
   `gate=f"OUTPUT_LO+{k}"` / `gate=f"OUTPUT_HI_THIS_STEP+{k}"` with
   `write_scale=50/S`, which is what produces the ×97 amplification.
   Adding an `OP_LEV` lower bound to the conditions (current
   `lev_stack0_preserve_conditions` is presumably absent of a hard
   non-LEV blocker; verify at the top of `_layer16_lev_routing_rules`)
   would prevent firing in IMM-followed-by-binop steps. **One-line
   declarative change candidate**.

2. **L34 (defense in depth)** — `l10_ops.py:3884-3970`
   `stack0_pop_loaded_output_rules` and the sibling
   `stack0_store_*` families. The 0.1-weight `OUTPUT_LO/HI`
   match conditions assume saturated one-hot values; they were not
   designed for the ~3 800 amplified values L28 produces. Capping the
   match contribution (e.g. clip `OUTPUT_LO+lo` to ≤1.0 via an
   intermediate one-hot dim) or raising threshold from 10.5 to
   ≈400 + 10.5 would prevent the false-positive over-firing.

3. **L27 (root cause but largest blast radius)** —
   `layer15_nibble_copy` writes the initial +40 spike. The op's
   semantic role is to copy MEM nibbles into OUTPUT for store/load
   relays; restricting its write at MARK_STACK0 to step-1+ paths that
   don't carry a pending binop would address the trigger but risks
   regressing PSH/POP nibble copies.

The cleanest fix is **option 1**: a single `OP_LEV` positive
condition (with matching threshold bump) on the
`lev_stack0_preserve_*` rules so they cannot fire on non-LEV steps.
Verify with `compare_symbolic_to_lowered_ffn` on L16 before/after,
then run smoke. Expected to recover the 9 binop tests covered by the
existing `f3342968` runner override and let it be removed in a
follow-up. **Bounded.**

## What I did NOT do

- No code change. Investigation-only per brief.
- No smoke run.
- Did not verify that `lev_stack0_preserve_conditions` lacks an
  `OP_LEV` blocker (the candidate fix surface). The L16 ops file is
  long; a quick `grep "lev_stack0_preserve_conditions"` should locate
  the conditions tuple.
- Did not measure whether option 1 also affects the
  `tail_stack0_*` writes downstream — possible second-order side
  effect on PSH/POP smoke.

## Cross-references

- `docs/REMOVAL_2_FINDINGS_2026_06_05.md` — parent doc with the
  "+2 * INSTR_WIDTH" framing this doc refutes.
- `docs/L34_FFN_ATTRIBUTION_2026_06_04.md` — block-index attribution
  used to identify L34 as `post_l9_bz_bnz_pc_override`. **Note**: that
  doc's L34 ↔ 192-unit mapping is for an older 35-block model; in the
  current 36-block model L34 = `tail_bit32_result_correction`
  (2 059 units) and L35 = `post_l9_bz_bnz_pc_override` (192 units).
- `neural_vm/unified_compiler/ops/l16_ops.py:285-300` — L28's
  `lev_stack0_preserve_*` amplifier.
- `neural_vm/unified_compiler/ops/l10_ops.py:3884-3970` — L34's
  `stack0_pop_loaded_output_rules` suppressor.
- `neural_vm/unified_compiler/ops/l15_ops.py:1746-1768` — L27's
  `layer15_nibble_copy` initial spike writer.

## Confidence

- **High** that the bug is byte-emit suppression at the STACK0 marker
  of step 2 (direct probe of token-byte emission + autoregressive
  context dump).
- **High** that block 34 is the dominant suppressor (delta range
  −6.29e8 vs all other blocks < ±4e3).
- **High** that block 28 is the proximate amplifier feeding block 34's
  faux-evidence (delta range +3.87e3, gate self-feedback identified).
- **Medium** that option-1 (L28 `OP_LEV` gate) is the bounded fix.
  The amplifier is identified; the exact rule conditions and
  threshold bump need byte-identity verification before commit.
- **Low** that no second-order regressions appear in the
  `tail_stack0_*` family after the L28 cut. Smoke run will tell.
