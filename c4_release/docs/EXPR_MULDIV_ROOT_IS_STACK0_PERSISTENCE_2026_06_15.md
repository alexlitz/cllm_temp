# expr-with-mul/div clusters: root is STACK0 cross-step persistence (#221), NOT the high-byte handoff

2026-06-15. Comprehensive probe lane on `expr_mul_div` (850-874),
`expr_add_mul` (800-824), `expr_mod` (875-899), `expr_paren` (825-849).
Verified spec_k=0, BUILT `model.dim_positions` (d_model=1090, n_heads=10),
hook-free, on HEAD `272e5c40` (flags at production defaults:
`C4_MUL_WIDTH2=1`, `C4_DIV_MULTIBYTE=1`, `C4_STACK0_B0_DUMP=1`).

## Full-trace baseline (HEAD 272e5c40, `tools/run_1096_canonical.py --criterion full_trace`)

| cluster        | ids     | pass |
|----------------|---------|------|
| expr_add_mul   | 800-824 | 0/25 |
| expr_paren     | 825-849 | 19/25|
| expr_mul_div   | 850-874 | 0/25 |
| expr_mod       | 875-899 | 3/25 |

## The hypothesis that was REFUTED

The brief hypothesised the break was the **intermediate's HIGH byte** not
reaching the 2nd op's operand: "MUL=784 computed correctly but DIV emits
769 = undivided", to be fixed by mirroring the DIV/MOD `STACK0_BYTE_VAL_1`
relay template.

Direct per-step register-trace + per-frame residual probes
(`tools/probe_expr_regtrace.py`, `tools/probe_stack0_b0_nibble.py`,
`tools/probe_expr_frame_select.py`) **refute this**:

* The FIRST op (MUL/ADD) computes the correct multi-byte result and the PSH
  stores it correctly. For `14*56/8`: per-step AX = `[14,14,56,784,784,8,769]`
  — MUL=**784 correct**; the STACK0 frame created by the PSH-of-784 holds
  byte0=**16 = 0x10 (correct)** at its `STACK0_BYTE0` row.
* The break is that the **SECOND op reads a DIFFERENT, STALE STACK0 frame**.
  For `14*56/8` the DIV reads a re-materialised STACK0 frame whose byte0 =
  **14** (the ORIGINAL first operand `a`), not 16. The intervening `IMM 8`
  step (which does NOT touch the stack) re-emits top-of-stack as a value
  carried from TWO frames back, not the just-pushed intermediate.
* This is a **byte-0** failure, not a high-byte one. The DIV/MOD
  `STACK0_BYTE_VAL_1` high-byte relay (the brief's template) therefore
  cannot help — even the low byte the 2nd op reads is wrong.

### Quantified (`tools/probe_expr_frame_select.py`, all 50 mul/div+add_mul)

> **45/50** programs: the 2nd op reads a STALE/WRONG `STACK0_BYTE0` frame
> (it reads the original `a`, not the pushed intermediate). The 5 "OK_b0"
> are byte-0 coincidences (stale value == intermediate byte0) that still
> fail on a later step/byte.

`C4_DIV_MULTIBYTE=1` (already the default) does NOT move expr_mul_div
(0/25 → 0/25) or expr_mod, confirming the DIV relay is not the lever.

## The actual root = STACK0 byte-0 cross-step persistence wall (#221)

This is exactly [[project_if_bool_expr_is_stack0_highnibble_framing_drift]]
(memory) / `STACK0_BYTE0_DUMP_CARRY_ROOT_2_2026_06_13.md`: the pushed
top-of-stack value emits fine at the PSH step (freshly computed) but is
LOST/replaced at the NEXT step where it must be persisted across the step
boundary. Surface: `layer10_stack0_persistence_head` /
`layer10_stack0_byte_relay` heads 4/5/6 (`l10_ops.py`), and the L3 byte-0
marker carry. The persistence head re-selects an OLDER STACK0 frame (the
first-operand push) over the most-recent (the intermediate push).

`expr_mod` (`a%b+c`, single-byte intermediates) fails the SAME way: the
MOD result is dropped and the ADD operates on the ORIGINAL dividend
(`46%6+6` → neural got 52 = 46+6, not 4+6=10). Same cross-step
top-of-stack persistence corruption.

`expr_paren` (`(a+b)*c`) mostly PASSES (19/25) because its ADD intermediate
`a+b ≤ 40` is single-byte AND the structure (`IMM, IMM, ADD, IMM, MUL`)
puts the 2nd op closer; the persistence still drifts on the both-nibbles-
nonzero values (the 6 fails), matching the #221 value-dependence.

## Why no single-rule expr-only fix exists

* The corruption is upstream of every operand-gather: the model's
  top-of-stack residual itself holds the wrong value by the 2nd-op step.
  Re-pointing `layer7_operand_gather` (ALiBi recency → most-recent
  `STACK0_BYTE0`) can't help — the most-recent frame IS the corrupt one,
  and there's no residual signal distinguishing a corrupt re-materialised
  frame from a legitimate one.
* The fix lives in the `layer10_stack0_persistence` head (load-bearing for
  var/store/SI/SC; single-rule edits here are documented zero-sum, 0/5
  historical agents) and the L16/block-38 `tail_bit32_result_correction`
  H1/H3 nuke — the SAME surface as the AX byte-1 dump wall. The
  `C4_STACK0_B0_DUMP` infra (now default-ON on main) fixes the byte-0
  EMISSION (slicer reads 35 tokens) but NOT the cross-step VALUE
  corruption probed here; and the discriminator between framing-drift rows
  and healthy carried-arithmetic rows is the open blocker (2 prior agents
  bounced — see ROOT_2 doc).

## Conclusion

The expr-with-mul/div clusters are blocked by the **cross-step STACK0
byte-0 value-persistence wall (#221)**, the dominant ~760-fail root, NOT by
an intermediate high-byte handoff. No model edit landed (smoke held 51/0;
HEAD byte-identical). The lever is the coordinated multi-session
persistence/AX-byte build, not an expr-local relay.

Reusable probes added (all spec_k=0, BUILT `dim_positions`, compile real
corpus bytecode via `compile_c`):
`tools/probe_expr_regtrace.py` (per-step AX trace vs oracle),
`tools/probe_expr_frame_select.py` (bulk stale-frame classifier — the 45/50
number), `tools/probe_stack0_b0_nibble.py` (per-frame STACK0_BYTE0 value),
`tools/probe_expr_intermediate_relay.py`,
`tools/probe_mul_psh_allbands.py` (locates the high byte across all bands),
`tools/probe_psh_axrows.py`, `tools/probe_div_operand_select.py`.

---

# UPDATE 2026-06-15 (#221 comprehensive lane): the corruptor is the C4_STACK0_B0_DUMP, and it's a confirmed +27 architectural TRADE

A deeper built-dim probe lane (HEAD `47246509`, GPU 1, spec_k=0, BUILT 49-block
build, `tools/probe_stack0_framechain.py` + direct LM-head logit attribution)
**re-localized** the expr corruptor and **measured the exact trade**. Two
corrections to the "stale frame the 2nd op reads" framing above:

## Correction 1: the value the 2nd op reads is correct UNTIL the L25 tail; the dump then over-stamps the stale one

For `14*56/8` the STACK0 frame the DIV reads (the op2-operand marker row, the
frame persisted across the intervening `IMM 8` step) holds byte0 = **16
(correct intermediate)** all the way through **block 31 (the ALU→OUTPUT
materializer) and blocks 31–37** (`OUTPUT_LO+0`/`OUTPUT_HI+1` = the 16 one-hot,
just at ~20× magnitude vs a fresh frame: 4017 vs 184). Then:

* **block 38** (`tail_bit32_result_correction`, the 2059-unit WIDTH-SENSITIVE
  bank) BLANKS all OUTPUT_LO/HI to ~−263 M on the carried frame (it does NOT
  fire on the fresh frame, whose OUTPUT is small/clean), and
* **block 46** (the `stack0_byte0_dump_repopulate` re-point, ON only under
  `C4_STACK0_B0_DUMP=1`) writes `OUTPUT_LO+14` = +67 M / `OUTPUT_HI+0` = +100 M
  → emits **14** (the ORIGINAL operand `a`).

LM-head logit attribution at the DIV-frame predictor row: `logit[14]=8.4e8`
(via `OUTPUT_HI+0=5.0e8 + OUTPUT_LO+14=3.4e8`) beats `logit[16]=3.0e7`. The
emitted byte-0 = 14 is then **baked into the context** and the operand-gather
re-reads it next pass (`ALU_LO+10`=stale for `10*9/1` from block 6 on) — a
**self-perpetuating** loop. So the "2nd op reads a stale prior frame" symptom
is REAL but its proximate cause is the dump over-stamp, not a persistence-head
mis-selection.

## Correction 2: it's the C4_STACK0_B0_DUMP, and it's NET +27 (can't disable)

Direct A/B on the target window (`tools/run_1096_canonical.py --ids
825-899,350-399,75-124 --criterion full_trace`):

| cluster      | DUMP ON (HEAD) | DUMP OFF | Δ(ON−OFF) |
|--------------|----------------|----------|-----------|
| sub          | 23/25          | 23/25    | 0         |
| mul          | 24/25          | 24/25    | 0         |
| if_gt        | **17**/25      | 4/25     | **+13**   |
| if_lt        | **21**/25      | 4/25     | **+17**   |
| expr_paren   | 19/25          | 14/25    | +5        |
| expr_mul_div | **0**/25       | **9**/25 | **−9**    |
| expr_mod     | 3/25           | 2/25     | +1        |
| **TOTAL**    | **107**/175    | 80/175   | **+27**   |

So the dump is a strong NET +27 (if_gt/if_lt +30 dominate the expr_mul_div −9).
`C4_STACK0_B0_DUMP=0` would recover expr_mul_div 0→9 but crater if_gt/if_lt
17/21→4/4. **Disabling is net −27 and is NOT the move.**

## The discriminator IS genuinely absent (built-dim confirmed, not a dim-mismap)

At the dump block (44/45) the expr op2-operand frame and the if/bool
comparison framing-drift frame are **batched-identical**: both are MARK_STACK0
rows with `STACK0_B0_CARRIED≈100`, `SHARP=0`, `PREV_DOM=21`, `NOT_CMP=0`,
`HAS_SE≈1`, and **NO per-step opcode** (the arith opcode OP_DIV/OP_MUL sits on
the *result* frame's marker, NOT the operand frame the dump corrupts —
`probe`: expr `14*56/8` marker rows: 195=OP_MUL, **268=∅** ← corrupted, 305=
OP_DIV; the existing `STACK0_B0_NOT_CMP` opcode rule fires =499 on 195/305 but
**=0 on 268** because no opcode lands there). Every other band probed
(OP_PSH/PSH_AT_SP trace, ALU presence, CMP+0..4, value magnitude, PREV
sharpness, pre-nuke OUTPUT cleanliness, and a FULL-RESIDUAL scan) is
overlapping. A full-residual scan at an EARLY block (15) surfaced ONE candidate
(`ADDR_B0_HI`≈0.13 on if-frames vs 0 on expr-frames) but it **washes out** by
the dump block (44): there `ADDR_B0_HI` band-mass ≈ 38 on EVERY carried frame,
expr and if alike — so it is NOT usable where the dump reads. The ONLY true
separator is **"the next step is arithmetic vs comparison+branch"** — a
**FUTURE opcode**, which is *causally unavailable* at the operand frame
(attention is causal; OP_DIV at row 305 follows row 268). This reproduces and
SHARPENS the ROOT_2 doc's conclusion with current numbers; it is not a
dim-mismap (read at BUILT `dim_positions`).

## Why even the "value-faithful carry" is NOT a clean win (tested, this lane)

The dump's carry head (`stack0_byte0_dump_carry`, L9 block 10) copies the prev
**marker row's** registry `H1`/`H3` one-hot. At block 9 the marker rows carry a
CONSTANT marker signature (`H1+10`, `H3+3`, `H3+10` — identical for every frame
regardless of value 14/56/16/8). So the carry re-emits a **near-constant byte**;
this happens to help the if/bool comparison rows and corrupts the per-frame
expr operands. The per-frame value DOES exist at block 9 in the **byte0 ROW**
(marker+1) `CLEAN_EMBED_LO/HI` (`14*56/8`: rows 233=16, 269=14; the byte0 rows
carry `STACK0_BYTE0≈0.97`, MARK_STACK0=0 — cleanly K-matchable). So the obvious
"value-faithful" fix is to re-point the carry head to attend the **previous
frame's byte0 row** (`STACK0_BYTE0` K-match + ALiBi recency) and copy *its*
`CLEAN_EMBED`. For expr that supplies the correct intermediate (marker 268 ->
prev byte0 row 233 = 16). **BUT a per-frame probe (`probe_stack0_framechain`
+ prev-byte0 trace) shows this REGRESSES if/bool**: the if comparison frames
need DIFFERENT per-frame values than "the previous frame's byte0", e.g.
`if_gt 28>9` carried marker 239 currently emits 28 (correct) but the prev-byte0
value is 2 — value-faithful carry would WRONGLY change it to 2; marker 203 emits
2 and prev-byte0 is 28. The if/bool comparison frames carry a mixed
operand+result value structure that "copy the prev frame's value" does not
satisfy. So the value-faithful carry trades expr-correct for if/bool-regress —
**also a trade, not a pure win**. The dump's constant-broadcast is a tuned
compromise the two clusters pull in opposite directions, NOT a fixable
mis-wiring.

The remaining (untested, higher-risk) lever is the **block-38 nuke source**
(`tail_bit32_result_correction`): the correct expr value is present at block 37
and only the nuke + dump destroy it; gating the nuke OFF the carried-operand
frame would let expr's value survive with NO dump needed and NO if/bool change.
But that touches the WIDTH-SENSITIVE 2059-unit bank
(`project_l10_tail_bank_width_sensitive`: repurpose, don't append) and the nuke
fires on the SAME carried-STACK0 signature (no opcode), so it likely inherits
the same indistinguishability. A genuine win requires a forward-propagating
control-flow band built during op1 (an arith-vs-cmp program-context the carry
propagates), which is a new multi-op head behavior, NOT a tail refinement.

New probe: `tools/probe_stack0_framechain.py` (per-frame byte0 across blocks +
op2 ALU read; the tool that localized the 16-survives-to-block-37 fact).
No model edit landed; smoke 51/0; HEAD `47246509` byte-identical.

---

# RESOLVED 2026-06-15 (#221 comprehensive lane): consumer-opcode lookahead + prior-arith latch — +28 full_trace, guards held, smoke 51/0

The "future opcode is causally unavailable" blocker is REAL within a single
forward pass, but the C4 ISA is **single-slot** (each instruction is one
`INSTR_WIDTH=8`-byte slot `op + (imm<<8)`; PC_OFFSET=2), so the consumer (next
instruction) sits at a FIXED `PC+8` in **program memory**, fetchable exactly
like the L5 opcode fetch. The landed mechanism (`C4_STACK0_NEXT_ARITH`,
DEFAULT-ON, commits acdccc68 / a8051a63 / 7675368d / 88cc5598):

1. **`lookahead_pc8_chain`** (L4 post-op FFN): build the `PC+8` address from
   `EMBED_LO/HI` via the declarative `nibble_rotation_chain` (offset=8, carry)
   -> `LOOKAHEAD_PC` band.
2. **`lookahead_opcode_fetch`** (L5 head): content-match `PC+8` vs the immutable
   per-CODE-position `ADDR_KEY`, copy op2's `CLEAN_EMBED` opcode-byte nibbles ->
   `NEXT_OPCODE`. (Mirrors `layer5_fetch` head 1.)
3. **`next_arith_flag`** (L6 FFN): decode "consumer is arithmetic"
   (OR/XOR/AND/SHL/SHR/ADD/SUB/MUL/DIV/MOD) -> bounded `STACK0_B0_NEXT_ARITH`.
4. **`next_arith_relay`** (L9 head): broadcast the flag from the step's AX row
   (where `NEXT_OPCODE` lives) to its STACK0-marker row (where the dump fires).
5. **`prior_arith_latch`** (L9 CAUSAL head): `STACK0_PRIOR_ARITH=1` iff some
   PRIOR position carried an arith opcode. **This is the lever.** Causality gives
   the single-op-vs-multi-op discriminator FOR FREE: `a*b`'s IMM-b operand frame
   sees the MUL as a FUTURE position (latch 0 -> KEEP the dump, which correctly
   re-supplies operand `a`), while `a*b/c`'s IMM-c frame sees the EARLIER MUL
   (latch 1 -> BLOCK the dump, which would emit the stale operand). The two
   frames are batched-identical in every other flag (CARRIED=100, SHARP=0,
   PREV_DOM=0, NOT_CMP=0) — the prior-arith latch is the ONLY separator.
6. **`dump_block_flag`** (L9 FFN): `STACK0_B0_DUMP_BLOCK = AND(NEXT_ARITH,
   PRIOR_ARITH)`; the L25-tail dump re-point reads it as a -2000 blocker.

## Why the naive blocker was a TRADE and the latch is a WIN

"Block the dump on ANY arith-consumer frame" was NET **-4** (expr_mul_div +9,
but expr_paren -5 / mul -3 / sub -4): the dump is LOAD-BEARING on single-op
operand frames (`a*b`: the dump re-supplies operand `a` for the MUL), which ALSO
have an arith consumer. The prior-arith latch excludes exactly those (no prior
arith on the first-operand frame), recovering all guards.

## A/B (spec_k=0, GPU0, `tools/run_1096_canonical.py --criterion full_trace`)

| cluster      | base (HEAD 072461c0) | ON  | Δ |
|--------------|----------------------|-----|----|
| expr_mod     | 3                    | 20  | **+17** |
| expr_mul_div | 0                    | 11  | **+11** |
| expr_paren   | 19                   | 19  | 0 |
| if_gt        | 17                   | 17  | 0 |
| if_lt        | 21                   | 21  | 0 |
| if_eq        | 21                   | 21  | 0 |
| bool_and     | 16                   | 16  | 0 |
| mul          | 24                   | 24  | 0 |
| sub          | 23                   | 23  | 0 |
| div          | 47                   | 47  | 0 |
| add          | 40                   | 40  | 0 |
| **expr window total** | **107**     |**135**| **+28** |

Smoke 51/0 (authoritative pytest, default-ON). `C4_STACK0_NEXT_ARITH=0` is
BYTE-IDENTICAL to HEAD (all 721 weight tensors match; verified atomically at
fixed PYTHONHASHSEED + cache cleared — the shared cross-worktree disk cache
`~/.cache/c4_release` produces spurious "block-40 diff" readings otherwise).

## What this does NOT fix

`expr_add_mul` stays 0/25 (the ADD intermediate path has a separate residual
issue) and the **var clusters stay 0/25 full_trace** (var's full_trace blocker
is the L3 SP_byte2 root per `project_var_failure_mode_shifted`, upstream of and
independent from this expr corruptor — no regression, but no conversion). The
remaining expr_mul_div / expr_mod fails (14/25, 5/25) are the MULTI-BYTE
intermediates where blocking the dump lets the post-L25-nuke OUTPUT survive as
garbage rather than the true intermediate (e.g. `14*56/8` -> 770 not 98): the
true win there needs the higher-risk **value-faithful** emission on the
blocked frames (re-supply the real prev-frame byte0), now cleanly gateable on
`STACK0_B0_DUMP_BLOCK` — a follow-up, not attempted this lane.

New probes (spec_k=0, BUILT dims): `tools/probe_consumer_opcode.py` (consumer
opcode at block 8/41), `tools/probe_temp_lookahead.py` (TEMP=PC+1 confirms the
single-slot +8 lookahead), `tools/probe_frame_discriminator.py` (confirms NO
same-frame separator), `tools/probe_next_arith_band.py` (band correctness).
