# func/var/rec saved-BP store garbage is the `l16_ent_frame_sp_byte1_ff` OP_ENT-broadcast misfire (2026-06-14)

Lane: **upstream-source cleaning** of the post-ENT 37-token desync (one of four
parallel approaches; the others rebuild L14's value-head attention / carry the
value / suppress the sentinel). spec_k=0, hook-free, BUILT-dim. Canonical
program: `func_identity_0` (id 550).

## TL;DR

The brief's premise — "L14's value heads attend the WRONG source position and
copy garbage `[1,255,18,255]` into the ENT saved-BP store" — is **REFUTED by
built-dim evidence**. The garbage is NOT from L14 (physical block 16): L14
leaves `OUTPUT ~= 0` at the ENT-step MEM value rows (pre-block-15 residual ==
post-block-16 residual there). The garbage is INJECTED downstream at **physical
block 31 = `layer16_lev_routing`** by ONE rule — **`l16_ent_frame_sp_byte1_ff`**
— misfiring on the **OP_ENT in-step broadcast**.

The clean old_BP IS available upstream and uniquely so: at the step-0 JSR's BP
value rows (built positions 124-127 in id 550) the CLEAN_EMBED bytes decode to
`[0,0,1,0] = 65536 = old_BP`, and those rows carry the EXACT markers L14's
slot-44 ENT-store value-source path keys on (`OP_JSR`, `H1+bp`, `BYTE_INDEX_k`).
So this is **NOT** an "old_BP never deposited" / attention-rebuild problem for
the saved-BP value — the source-clean approach is viable, and it is what this
commit lands.

## The corruptor, exactly

`l16_ent_frame_sp_byte1_ff` (l16_ops.py, in `_layer16_lev_routing_rules`) is the
genuine SP-byte1=0xff override for the ENT SP-byte0 prediction row. Threshold
4.5; positive gates `OP_ENT*1 + IS_BYTE*1 + HAS_SE*1 + H1+2*1 + BYTE_INDEX_0*1`.
`OP_ENT` does NOT stay one-hot at its own step — it BROADCASTS at ~10..18 onto
every row one VM step after an ENT. Because the threshold (4.5) is far below the
broadcast magnitude, **OP_ENT alone clears it** and the rule fires on rows that
are not SP byte rows:

| row (id 550, post-block-30) | markers | rule score (thr 4.5) | fires? |
|---|---|---|---|
| genuine SP byte0 (test_simple_function pos 98) | `H1+2=1, IS_BYTE, BYTE_INDEX_0` | 15.6 | YES (correct) |
| post-ENT STEP_END (pos 177) | `HAS_SE, OP_ENT=9.8`, **H1+2=0** | 10.8 | YES (MISFIRE) |
| saved-BP store MEM addr/val byte rows (169,173,174,175) | `IS_BYTE, MEM_STORE, OP_ENT~11`, **H1+2=0** | 12.5..14.6 | YES (MISFIRE) |

The misfire forces `OUTPUT_{LO,HI}+15 = 0xff` on those rows, which is amplified
by the L25 tail to ±5000..8000. Two consequences:

1. **STEP_END-row misfire** => the post-ENT step's first token is predicted as
   `255` (LM-head logit 4.4e10 -> 4.8e4 here). This is the +2 leading-0xff
   **37-token desync** (`docs/FUNC_LEV_IS_LI_FROM_FRAME_37TOKEN_DESYNC`). The
   desync drops the next-step PSH argument store, so the later frame-local `LI`
   loads 0.
2. **MEM-row misfire** => the saved-BP store emits `[1,255,18,255]` instead of
   `[0,0,1,0]`.

`test_simple_function` (JSR;ENT;IMM;LEV) shows the IDENTICAL garbage saved-BP
store yet PASSES — its return value comes from IMM, not a frame-local LI — which
is why the garbage looked load-bearing but is a SYMPTOM, not the gating root.

## The clean discriminator + the fix landed

The genuine SP byte rows carry the SP register marker-distance hint `H1+2 ~=
0.99..1.0`; EVERY misfire row carries `H1+2 == 0`. Fix (flag
`C4_ENT_SP_BYTE1_FF_H1_HARDEN`, default ON): promote `H1+2` to a HARD
requirement via a **net-zero CONST baseline** — raise its weight `1.0 -> 1.0 +
20` and add `("CONST", -20)`. On the genuine row the +20*H1+2 gain is cancelled
by the -20 CONST (|loss| <= 0.01*20 = 0.2 on an 11-pt margin); on any H1+2==0
misfire row the full -20 buries the OP_ENT broadcast (14.6 - 20 < 4.5 -> vetoed).
Pure subtractive on the misfire side. Flag-off restores the exact legacy
21-condition tuple (H1+2 weight 1.0, no CONST) => byte-identical build, 827 rules.

## Result (built-dim, id 550)

* Saved-BP MEM rows + STEP_END row: block-31 no longer flips nibble 15 (0xff
  band stays at the -240 default; was +4768/+4783). Verified per-block.
* Post-ENT step over-emit: **37 -> 36 tokens** (the two leading 0xff drop to one
  leading 0x00); the step-3 PSH argument store now MATERIALIZES at addr 65512
  (`MEMa=65512` in the fixed-offset trace; was absent).
* Smoke: **51 passed / 0 failed** (flag on). `test_declarative_ffn_bakes_l16.py`
  11f/20p and `test_l16_per_op.py` 2f/96p are IDENTICAL flag-on vs flag-off
  (all pre-existing on HEAD); zero regressions.

## Why it is net-NEUTRAL on the corpus (the SECOND corruptor)

full_trace divergence steps are UNCHANGED flag-on vs flag-off (var 250/251 step
5, func 550 step 7, func_add 575 step 6, rec 700 step 13, nested 950 step 7).
Reason: the desync has **TWO** corruptors, and this fix closes only the 0xff one.
A SECOND, smaller leak at **physical block 29 (logical L18)** perturbs the
post-ENT STEP_END row's `OUTPUT_LO[0] -240 -> -224 (+16), OUTPUT_HI[0] +4.5,
OUTPUT_HI[15] +16`. That +16 tips byte `0x00`
(`5*OUTPUT_LO[0]+5*OUTPUT_HI[0] = -2300`) just above the marker floor (-2400),
so the post-ENT step STILL over-emits one leading `0x00` token => +1 residual
desync => the PSH store value is still corrupted (`65546` not `70`) => LI still
loads wrong.

**CRUCIAL: the block-29 leak is from the ATTENTION, not an FFN rule.** Built-dim
decomposition of block 29 (post-attn == post-ffn at the STEP_END row, both
`-223.86`; block-28 input `-240.0`) localizes the +16 to block 29's
`AutoregressiveAttention` — a head writing `OUTPUT_LO[0]` under the OP_ENT
broadcast. Block 29 is the post-op expansion of L14's mem-generation attention.
So the second corruptor is an **attention misfire in the L14/L18 region** — the
**L14-attention-rebuild lane's territory**, NOT a source-clean FFN fix. The
source-clean lane (this commit) correctly handles the FFN-side 0xff corruptor;
closing the residual byte-0 desync requires the attention-rebuild lane to scope
that block-29 head's OP_ENT-broadcast firing.

## What this tells the four-lane effort

* The **source-clean approach for the saved-BP value is viable and correct** —
  old_BP is cleanly deposited upstream (JSR BP rows 124-127) with the markers
  L14 keys on; the 0xff garbage is a downstream OP_ENT-broadcast misfire, now
  fixed. The L14-attention-rebuild lane is NOT needed to make the saved-BP
  store clean.
* Full desync closure requires ALSO neutralising the **block-29 / logical-L18
  byte-0 OUTPUT leak**, which is an **ATTENTION misfire** (a block-29 head
  writing `OUTPUT_LO[0]` under the OP_ENT broadcast), i.e. the
  **attention-rebuild lane's** surface (block 29 is the post-op expansion of
  L14's mem-generation attention). It is deliberately left to that lane to avoid
  a cross-lane collision; this commit lands the byte-identity-safe source-clean
  (L16 FFN) half.

## Tools (read-only, this session)
* `tools/_probe_ent_mem_source.py <id>` — per-step MEM tokens + ENT MEM-value
  predictor-row OUTPUT across blocks + OP_JSR/OP_ENT position map.
* Existing `tools/_probe_func_li.py` / `_probe_func_fixed.py` — marker-aware /
  fixed-offset per-step trace (exposes the 37->36 token desync change).
