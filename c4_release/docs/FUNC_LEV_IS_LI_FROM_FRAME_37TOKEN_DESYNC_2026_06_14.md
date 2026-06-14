# func_*/nested_*/rec_* "LEV PC-restore" is REALLY LI-from-frame, gated by the post-ENT 37-token desync (2026-06-14)

Worktree: `agent-a362244792c427a8d` (base `20f323a9`, the var-step-1 L16 fix).
Smoke at base/HEAD: **51 passed / 0 failed** (pytest `tests/test_smoke.py`).
**No production weight change was made** (correct-dim probing showed a
multi-commit architectural wall on the documented framing-drift / 0xFF-leak
surface; per the task contract, this is documented, not patched).

## TL;DR — the prior LEV diagnosis is REFUTED

The brief targeted "LEV PC-restore from `mem[BP+8]` broken; needs an L10 PC
`byte_passthrough` head + an L9 `lev_bp_to_pc_relay` rewrite." That is **wrong**.

* **PC restore at LEV works.** In `func_identity_0` (id 550) the neural PC is
  byte-correct at EVERY step including the LEV step (step 8, PC 50→10). The
  `full_trace` first divergence is at **step 7 (the `LI` instruction), not the
  LEV**, with `expected(pc=50,ax=70)` vs `got(pc=50,ax=0)` — **PC matches, AX
  is wrong.**
* **The real bug is `LI` (load-indirect) of a function argument from the stack
  frame returning 0** instead of `mem[BP+offset]`. The callee body is
  `ENT 0; LEA <off>; LI 0; ...; LEV`. `LEA` computes the frame-local address
  correctly (neural AX = 0xFFE8 = 65512, byte-identical to oracle), but the
  immediately-following `LI` loads 0 instead of the pushed argument (70/57).
* **`test_simple_function` passes because it never does an `LI`.** Its callee is
  `ENT 0; IMM 42; LEV` — the return value comes from `IMM`, straight into AX. It
  exercises ZERO of the LI-from-frame path, so its passing told us nothing.

So the corpus `func_*`/`nested_*`/`rec_*`/`var_*` clusters fail at the
frame-local load, well *before* LEV. (After step 7's bad AX, the trace cascades —
BP/SP get corrupted at step 8+ — but that is all downstream of the LI=0 root.)

## Reproducer

```
CUDA_VISIBLE_DEVICES=1 python tools/run_1096_canonical.py --ids 550,575 \
    --criterion full_trace --print-failures
#   id=0550 func_identity step=7 expected(pc=50,ax=70) got(pc=50,ax=0)
#   id=0575 func_add      step=9 expected(pc=50,ax=57) got(pc=50,ax=0)
```

Oracle disassembly (id 550, `identity(70)`):

```
0 JSR 7   2->58     (main: call main-body)
1 ENT 0   58->66    (main frame)
2 IMM 70  66->74    AX=70
3 PSH 0   74->82    mem[65512]=70   <-- the argument is pushed here
4 JSR 3   82->26    (call identity)
5 ENT 0   26->34    (identity frame; BP=65496)
6 LEA 16  34->42    AX = BP+16 = 65512   (frame-local arg address)  *** neural OK: AX=65512
7 LI  0   42->50    AX = mem[65512] = 70                            *** neural BUG: AX=0
8 LEV 0   50->90    restore SP/BP/PC                                *** neural PC OK
9 ADJ 8 ; 10 EXIT
```

Per-step neural trace tool: `tools/_probe_func_li.py 550 575` (marker-aware
decode, matches the canonical runner's `full_trace` PC/AX).

## Root cause — the post-ENT **37-token desync** corrupts the in-frame MEM section

The smoke/canonical path is a **fixed 35-token VM step**
(`Token.STEP_TOKENS == 35`; layout in `batched_pure_neural._step_offset_field`:
`PC+4, AX+4, SP+4, BP+4, STACK0+4, MEM(marker+4 addr+4 val), STEP_END`).
The neural emission for `func_identity_0` is **not** a clean multiple of 35:

```
per-step token counts (delimited by STEP_END): [35, 35, 37, 35, 35, 35, 37, 35, 35]
                                                         ^^                ^^
                                                       step 2            step 6
```

Steps **2 and 6 emit 37 tokens** — and both are the instruction **immediately
after an `ENT`** (step 1 ENT → step 2 IMM; step 5 ENT → step 6 LEA). The 2 extra
tokens are **two leading `0xFF` (255) tokens** emitted at the *start* of the
post-ENT step:

```
step 2:  0:255  1:255  2:REG_PC 74 ...   (clean step 1 starts at 0:REG_PC)
```

The over-emission is driven by a **sentinel-magnitude logit**: at the position
right after step-1's STEP_END the model predicts token `255` with logit
**4.4e10** (`tools/_probe_func_li.py` + a logit dump confirm pos 178 top-1 =
255 @ 4.4e10). This is the **0xFF-emitter / sentinel-inversion family**.

**Why the ENT step triggers it:** the `ENT` step's own MEM section (the
saved-BP store) is emitted as **0xFF garbage** instead of the clean store:

```
oracle ENT step1: mem_addr=65520, mem_value=65536  =>  MEM = [261, 240,255,0,0,  0,0,1,0]
neural ENT step1:                                       MEM = [261, 255,255,18,0, 1,255,18,255]   <-- garbage, 0xFF-laden
```

The `0xFF` MEM val bytes (offsets 31/33 = 255) at the ENT step are what the
next-step 0xFF over-emitter latches onto, so each ENT pushes the following step
to 37 tokens.

## Why `LI` then returns 0 (the consumer side)

At the `LI` step (id550 step 7) the lookup **query is correct** — the binary
address bands at the AX row decode to byte0=0xE8, byte1=0xFF → **0xFFE8 = 65512**
(`tools/_probe_li_resid2.py`, post-block-17 ADDR_B0/B1 one-hots). The L15
content-addressable memory_lookup (`primitives.memory_lookup_attention`;
imperative ref `_set_layer15_memory_lookup`) should match the
`MEM_STORE`-tagged token whose ADDR == 65512 and copy its `CLEAN_EMBED` value to
`OUTPUT_LO`.

But the residual at the AX-value row (`ax_marker+1`) across ALL blocks shows
`OUTPUT_LO ≈ 0.9` — **the L15 lookup never delivers 70**
(`tools/_probe_li_l15.py 550 8 7`; block 27 = logical L15 output OUTPUT_LO=0.9).
The matching store is the **PSH at step 3**, whose neural MEM section is itself
corrupted by the upstream desync (addr=65512 OK, but **val = 0x1000A = 65546**,
not 70 — the low byte 10 is leaked frame residue). So:

* the store entry's VALUE is corrupted (desync-driven), and/or
* the store's `MEM_STORE`/`ADDR_KEY` embedding tag lands on a shifted position,

→ the LI lookup finds no usable value → `OUTPUT_LO ≈ 0` → AX = 0.

**Pure-neural `LI` is healthy in isolation:** the `IMM 0x200; PSH; IMM 42; SI;
IMM 0x200; LI; EXIT` roundtrip returns AX=42 in spec_k=0 (`tools/_probe_func_li`
analog). The difference is *frame context*: the SI/LI test has clean 35-token
steps; the func/var path has the post-ENT 37-token desync poisoning every
in-frame MEM section.

## This is the SAME mega-root the project already tracks

* memory `project_if_bool_expr_is_stack0_highnibble_framing_drift` — "comparison
  step emits 57 tokens not 35 … fixed-35 slicer misreads … SAME surface as AX
  byte-1 … sequence with the AX agents." Here the over-emission is +2 (37) on the
  post-ENT step, mechanism identical (sentinel-magnitude 0xFF over-emit).
* task #228 — "var step-2 LEA frame-local **37-token desync** (~100)". `var_simple`
  (id 250) reproduces the identical 37-token step right after ENT.
* task #227 — "func/nested step-1 ENT→next-instr".
* memory `project_ax_byte1_dump_is_h1_onehot_wall` / `project_ax_bytes_1_3_ff_leak_root`
  — the 0xFF-emitter surface; "don't throw single agents at it … two-part build."
* memory `feedback_single_rule_fixes_are_zero_sum` — single-rule fixes on the
  0xFF surface net zero.

The var-step-1 L16 fix (`20f323a9`, base of this worktree) already removed ONE
member of the OP_ENT-broadcast 0xFF family (the STACK0 leak) but did not touch
the **MEM-section** producers, which are what poison the LI lookup here.

## Cluster scope (the wall's size)

Every cluster whose body does `ENT … LEA <frame off> … LI` (load an argument or
local from the frame) is gated by this one root:

| cluster        | ids       | n   | pattern                                   |
|----------------|-----------|-----|-------------------------------------------|
| func_identity  | 550–574   | 25  | ENT; LEA; LI; LEV                         |
| func_add       | 575–599   | 25  | ENT; LEA; LI; PSH; LEA; LI; ADD; LEV      |
| func_mul/square/max/min | 600–699 | 100 | same LEA+LI frame loads          |
| rec_*          | 700–799   | 100 | recursive frame loads                     |
| nested_*       | 950–999   | 50  | nested frame loads                        |
| var_*          | 250–349   | ~100| local-var SI-to-frame then LEA+LI         |

≈ **400–450 programs** diverge at the first frame-local `LI`. (All 550/575/250
confirmed at the LI step in this session.)

## Why it is a multi-commit wall (NOT a single blocker)

The minimal trigger is clear (post-ENT 0xFF over-emit @ 4.4e10 logit, fed by the
ENT step's 0xFF-garbage MEM section). But fixing it needs a **deliberate two-part
build**, because the two natural single-rule patches are each zero-sum:

1. **Suppress the post-ENT 0xFF over-emit** (the immediate desync trigger). The
   emitter is a sentinel-magnitude (4.4e10) member of the SAME 0xFF/tail family
   that legitimately writes `0xFF` for genuine high SP/BP bytes elsewhere
   (`project_ax_ff_leak_is_tail_byte1_ff_emitters` — "partial fix relocates the
   leak"). Blocking it without the right OP_ENT-frame discriminator relocates the
   0xFF, regressing the SP/BP-byte rows that need it.
2. **Make the ENT MEM section emit the clean saved-BP store** instead of 0xFF
   garbage (kills the trigger at the source). This is the L14 `mem_gen`
   OP_ENT producers (`l14_ops.py` ~616–966; extensive "OP_ENT does NOT stay
   one-hot at its own step" notes). The OP_ENT broadcast (~9.8–17.5 one step
   later, per the var fix commit) makes a clean per-step gate non-trivial, and
   this is the surface the var/if/bool agents are already on.

Either alone is the documented zero-sum trap. The correct fix is the
project-level "two-part build" (carry-band + re-pointed dump / clean ENT MEM
section), sequenced WITH the AX-byte-1 and var/if/bool framing-drift work — not a
solo corrector, and not anything in the L9/L10 LEV-relay region (which is
already correct).

## What to do next (for the agent who picks up the two-part build)

* Probe the **ENT step's MEM section** producers in `l14_ops.py` (the saved-BP
  store) at BUILT dims; make them emit `addr=SP, val=old_BP` cleanly so the MEM
  val bytes are not 0xFF. Verify with `tools/_probe_func_li.py` that the post-ENT
  step returns to 35 tokens.
* THEN re-confirm the LI lookup (`tools/_probe_li_l15.py 550 8 7`) delivers 70 to
  `OUTPUT_LO` at the AX-value row.
* Byte-identity gate against `test_simple_function` (no LI, must stay green) and
  the memory SI/LI/SC/LC smoke (clean 35-token, must stay green).

## Tools added this session (diagnostic only, `tools/_probe_*`)

* `_probe_func_li.py <ids>` — marker-aware per-step neural PC/AX/SP/BP/STACK0/MEM.
* `_probe_func_fixed.py <id>` — same via FIXED 35-offset (exposes the desync).
* `_probe_func_tokens.py <id>` — raw token stream per step.
* `_probe_li_resid2.py <id> <ms> <li_step>` — LI-step AX-row ADDR-query +
  value-band residual across blocks (one context, per-block forward).
* `_probe_li_l15.py <id> <ms> <li_step>` — L15 lookup output at the AX-value row
  (`OUTPUT_LO` per block; shows the lookup never delivers the loaded value).

## PARTIAL LANDING — `BP_SAVE_PREV` cross-step carry (2026-06-14)

A dedicated cross-step carry-band lane (the PROVEN AX-byte-1 / STACK0-byte-0
pattern) was built and landed **flag-gated default-ON** (`C4_BP_SAVE_DUMP`). It
fixes the **VALUE** half of the ENT saved-BP store: instead of repairing the
failing same-step L14 content-addressing, it CARRIES `old_BP` across the step
boundary and re-emits it into the ENT-step MEM val rows.

**What it achieves (verified spec_k=0 on id550/575):**
* `func_identity_0` ENT saved-BP store now emits the **clean old_BP**: the
  callee ENT (step 5) val bytes go `[0,255,18,255]` (0xFF garbage) → `[240,255,
  0,0]` = 65520; the main ENT (step 1) val bytes → `[0,0,1,0]` = 65536. This is
  exactly the "clean `[0,0,1,0]` not `[1,255,18,255]`" target above. ✓
* Smoke **51/0** (SI/SC/LI/LC stay green — the dump fires ONLY on the ENT-store
  val rows, gated on the high `OP_ENT` broadcast + the `MEM_VAL_B{k}` markers).
* Byte-identity flag-off (`C4_BP_SAVE_DUMP=0`) == HEAD (weight-hash identical;
  d_model 981, the band/head/dump are omitted).
* No guard regression: add/sub/mul/div/if/var are byte-for-byte identical
  flag-off vs default-on (the dump never fires outside ENT-bearing programs).

**What it does NOT yet fix (why the cluster is still 0/N full_trace):** the LI
still returns 0 and the post-ENT **37-token desync persists** — because the ENT
store's **ADDR bytes are STILL 0xFF garbage** (`[255,255,17,0]`) and the
post-ENT `0xFF` over-emit (the sentinel-magnitude trigger) latches onto THOSE,
not the val bytes this lane cleaned. So the desync chain has ≥2 more members
beyond the val store:
  1. the ENT-store **ADDR** bytes (a SECOND carry — `SP`, not `BP`), and
  2. the post-ENT `0xFF` over-emitter suppression (other lanes own this).

This confirms the doc's "two-part build" is really a THREE+-part build: the val
carry (this lane) is necessary but not sufficient; the ADDR carry + the
over-emit suppression must land together to close the LI. The val carry is a
clean, isolated, byte-identity-gated building block the next agent can build the
ADDR carry on top of (mirror `make_bp_save_prev_carry_op` with `SP`-byte source
and an `ADDR`-row dump gate).

**Implementation** (`C4_BP_SAVE_DUMP`, default ON):
* `BP_SAVE_PREV` band (32 = 16 LO + 16 HI nibbles), `register_residual_band(...,
  flag=_bp_save_dump_enabled, never_share=True)` in `ops/l11_ops.py`.
* `make_bp_save_prev_carry_op` — L13 block-16 head 8 (a widen-padding slot). Q
  at each ENT val-byte-k PREDICTOR row (`MEM_VAL_B{k}`), K at the prev step's BP
  byte-k row (`BYTE_INDEX_{k}`) with an `OP_JSR` prev-prologue preference, a
  same-step-`OP_ENT` reject, and the DECISIVE `STACK0_BYTE{0..3}` reject (the
  prev JSR step's pushed-arg STACK0 byte rows share the BP byte rows' signature
  and are NEARER → recency would otherwise carry the arg, not old_BP). V copies
  `CLEAN_EMBED_LO/HI` (the clean old_BP nibbles — pristine in the token
  embedding even though the OUTPUT residual is later nuked) into `BP_SAVE_PREV`
  with a boosted O weight (so the dump gate magnitude dominates the corruptor).
* `make_bp_save_dump_repopulate_op` — L25 tail PureFFN after
  `tail_bit32_result_correction`; gate-copies `BP_SAVE_PREV` → `OUTPUT_LO/HI` at
  the ENT val rows with a SENTINEL-magnitude write_scale (the corruptor pre-loads
  the rows with ±7e7 0xFF garbage INCLUDING a negative on byte-0's cell, so the
  additive re-supply must clear both).
* ALL gate dims resolve from the declarative LAYOUT (`dim_positions`), NOT the
  registry — the OPPOSITE of the AX/STACK0 carries; verified the residual
  carries `MEM_VAL_B*/OP_ENT/CLEAN_EMBED/OUTPUT` at LAYOUT positions in this
  build (e.g. `OUTPUT_LO` layout=69 vs registry=174).

New probes: `tools/_probe_bp_carry{,2,3,4,_check,_attn}.py` (the source-location,
gate-discriminator, and band/OUTPUT verification chain).
