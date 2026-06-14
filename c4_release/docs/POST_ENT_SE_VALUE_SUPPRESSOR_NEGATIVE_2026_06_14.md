# Post-ENT 37-token framing recovery WORKS but does NOT advance programs — the LI value corruption dominates (2026-06-14)

Flag: `C4_POST_ENT_SE_SUPPRESS` (DEFAULT-OFF; opt in with `=1`).
HEAD byte-identical with the flag off. Smoke 51/0 with the flag ON and OFF.

## TL;DR

A clean, built-dim-verified **framing-recovery** fix for the post-ENT
37-token desync (docs/FUNC_LEV_IS_LI_FROM_FRAME_37TOKEN_DESYNC_2026_06_14.md):
suppress the byte-value OUTPUT decode on `MARK_SE_ONLY` (STEP_END marker) rows
so the post-ENT step re-emits 35 tokens instead of 37. It **mechanically works**
— it restores the 35-token frame AND defeats the WHOLE 3-emitter cascade at the
decode point (breaking the documented single-emitter zero-sum trap). But it
**does NOT advance any func/nested/rec/var program**: the spec_k=0 (smoke /
production) EXIT code is **byte-identical flag OFF vs ON**, because the
divergence is dominated by the **LI-from-frame value corruption** (load-indirect
of a stack-frame argument returns 0, not `mem[BP+off]`), which determines the
exit independently of the frame width.

This is the valuable negative result the task asked for: **framing-recovery
alone is insufficient** on this wall. The fix is preserved flag-gated (off) as
the FRAMING half of the project-level two-part build, for the next agent who
fixes the LI value path.

## The emitter (built-dim evidence, spec_k=0)

`tools/_probe_cascade_emitters.py 550` + `tools/_probe_spurrow_full.py 550`
(efficient-mode model, 47 physical blocks, d_model=981):

* The post-ENT STEP_END row (id550 ctx pos 177/214/319) carries
  `MARK_SE_ONLY`(dim 10)=1.0, `IS_BYTE`(dim 6)=0.0, `OP_ENT`(dim 191)≈9.76. It
  must predict the marker `REG_PC` (token 257, logit ~18).
* Physical **block 31 (logical L20)** writes the OUTPUT byte-decode dims
  `OUTPUT_LO+15`=dim 84 and `OUTPUT_HI_THIS_STEP+15`=dim 100 to ~+4767/+4782
  (`head.weight[255]` reads both at +5.0 → token 0xFF logit ~4.7e4). This is the
  `l16_ent_frame_sp_byte1_ff` SP-byte1=0xff override mis-firing on the marker
  row (its OP_ENT broadcast clears the +4.5·S threshold).
* It is a **3-emitter CASCADE**: block 31/L20 → 0xFF (dims 84/100); once 0xFF is
  blocked, block 41/L25 → 0x00 (dims 69/85/882); etc. Blocking ONE just lets the
  next win — the `C4_ENT_SP_BYTE1_ISMARK_BLOCKER` zero-sum trap (step stays 37).

## The discriminator (why the fix is byte-identity-safe)

The LM head's byte-value-decode dims (`OUTPUT_LO` 69-84, `OUTPUT_HI` 85-100, plus
the H1/H2/H3 dump bands 880-943) and its marker-decode dims (21-31) are
**DISJOINT** (zero overlap, verified over tokens 0-275). Genuine 0xFF/value-byte
rows carry `IS_BYTE`=1.0, `MARK_SE_ONLY`=0.0; only STEP_END marker rows carry
`MARK_SE_ONLY`=1.0. So a `MARK_SE_ONLY`-gated OUTPUT suppressor never touches a
legitimate value emission. On a clean STEP_END row the OUTPUT band is already a
uniform "no byte" `-240` default (the marker wins anyway), so pushing it further
negative does not change the emitted token. HEAD byte-identical.

## The fix (declarative, flag-gated, default-OFF)

`_post_ent_se_value_suppressor_rules` in
`neural_vm/unified_compiler/ops/l6_ops.py`: 32 `FFNRule.constant_write` units
(16 `OUTPUT_LO+k`, 16 `OUTPUT_HI_THIS_STEP+k`) gated on `MARK_SE_ONLY>=0.5`, each
writing a sentinel-dominating `-200·S` post-relu (~-1e4 residual delta on a
firing row). Appended to `post_l9_bz_bnz_pc_override` (the FINAL FFN, physical
block 46 / logical L26 — placed after every cascade emitter by the existing
`requires={"after": "layer10_alu"}` schedule). Cache-key snapshots in
`full_vm_compiler_dynamic.py` updated in BOTH places (in-proc memo + disk) so the
ON/OFF builds never share an entry (this was load-bearing — without it the disk
cache silently masked the flag).

## Verification (spec_k=0, the smoke/production path)

`tools/_check_counts.py 550` (per-step token counts, STEP_END-delimited):

```
flag OFF (193 units, == HEAD): counts=[35,35,37,35,35,35,37,35,35]  spurious_lead=[255,255,255]
flag ON  (225 units, the fix): counts=[35,35,35,34,35,35,35,35,35]  spurious_lead=[]
```

→ the 37-token desync (steps 2 & 6) IS fixed to 35; the 3 spurious leading 0xFF
tokens are eliminated. (Step 3=34 is the unrelated LEV/EXIT early-stop, present
either way.) The marker-aware register trace (`tools/_probe_func_li.py 550`) now
reads a clean PC sequence 58→66→74→82→26→34→42→50→10 through every step.

**BUT** the exit code is unchanged (`tools/_check_exit_spec0.py`, spec_k=0):

| id  | cluster        | flag OFF | flag ON | expected |
|-----|----------------|----------|---------|----------|
| 550 | func_identity  | 1536     | **1536**| 70       |
| 575 | func_add       | 3056     | **3056**| 68       |
| 250 | var_simple     | 768      | **768** | 990      |

The `full_trace` divergence on HEAD is at **step 7 (LI), got(pc=50, ax=0)** —
PC correct, AX wrong (the LI loads 0 from the frame). Framing recovery does not
touch that AX, so the exit is identical.

## Gate results

* func/nested/rec/var advance: **0** (spec_k=0 exit byte-identical ON vs OFF).
* Smoke: **51/0** flag ON and OFF.
* Guards (add/sub/mul/if/bool ids 0,1,2,50,51,100,101,350,351,375,400,425,1071):
  exit codes **byte-identical** ON vs OFF
  (`{0:768,1:816,2:563,50:801,51:1438,100:61655,101:93,350:1,351:0,375:0,400:1,425:0,1071:0}`).
* Byte-identity flag-off == HEAD: block 46 = 193 units default, 225 only with `=1`.

## What this means for the wall

The post-ENT framing desync is REAL and now has a clean, isolated, byte-identity-
safe FRAMING fix. But it is the FRAMING half only. The exit-determining root is
the **LI-from-frame returning 0** (the L15 content-addressable memory_lookup not
delivering the pushed argument because the upstream PSH store value is itself
desync-corrupted, per the original LI doc). The two-part build is:

1. (DONE, here, flag-gated off) restore the 35-token frame on post-ENT steps.
2. (NEXT) fix the LI lookup / PSH store value so `mem[BP+off]` delivers the
   argument — only THEN do these ~400-450 programs advance.

Turn `C4_POST_ENT_SE_SUPPRESS=1` ON together with the LI value fix.

## Tools added (diagnostic only)

* `tools/_probe_cascade_emitters.py <id>` — per-block logit-delta localiser for
  every post-SE spurious leading token (finds all cascade members + their
  value-decode dims, BUILT-dim labelled).
* `tools/_check_counts.py <id>` — per-step token counts + spurious leading
  value tokens (the 37-vs-35 frame check).
* `tools/_check_fixedslice.py`, `tools/_check_exit_spec0.py`,
  `tools/_guard_check.py`, `tools/_check_units.py` — spec_k=0 fixed-slice PC,
  exit-code, guard, and unit-count comparators.
