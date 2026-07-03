# var_three step-6 multi-local LEA &b root + the next (step-9 LI) blueprint

Date: 2026-06-25. Campaign config (`C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1`,
now the production default since the FLIP `8c458f9c`). spec_k=0, BUILT
`layout.dim_positions`, alu_mode="efficient". Built from main `03e3ea24`.
Vehicle: `tools/cpu_full_trace.py` (bit-exact AR verdict) + the AR block-level
residual probes `tools/_probe_vt_step6_ar.py` / `tools/_probe_vt_step6_tail.py`
+ GPU `run_1096_canonical --spec-k 0 --criterion full_trace`.

## What landed: `C4_LEA_E0D8_FETCH_DOMINATE` (default campaign-ON)

var_three (ids 300-324, `a@BP-8 / b@BP-16 / c@BP-24`) full_trace diverged at
**step 6** — the `LEA &b` (BP-16, imm=-16, correct AX byte-0 = 0xE0) emitted
0xE8 (= &a, BP-8) so the 2nd local ALIASED the 1st (`got_ax 0xFFE8` vs oracle
`0xFFE0`). PC was correct throughout (no framing desync).

### Root (AR block-by-block residual, `tools/_probe_vt_step6_tail.py`)

The OUTPUT byte-0 at the step-6 LEA AX-marker row is the **correct 0xE0 through
physical block 41**, then FLIPS to 0xE8 at **block 42** (the L25 tail bank):
`OUTPUT_LO[0]` +758 → −829M, `OUTPUT_LO[8]` +558 → +829M.

The culprit is the campaign keystone's BP-8 0xE8 writer
`tail_lea_local_ax_byte0_e8_alubp_memsp` (l10, strength 1e6). It keys ONLY on
`0.2 * ALU_HI+15`. The keystone (`_lea_byte0_memsp_relay_enabled`) ASSUMED
`ALU_HI+15 ~ +5.5` on BP-16/BP-24 frames (so `0.2*5.5 ≈ 1.1` keeps the writer
below threshold 17 on non-BP-8 locals). **That assumption is false in
var_three's deeper 3-local frame**: the `&b` LEA's `ALU_HI+15` is **+72.92**
(measured, identical magnitude to `&a`'s), so `0.2*72.92 + (MARK_AX+HAS_SE+
OP_LEA ~3) ≈ 17.6 ≥ 17` → the 0xE8 writer FIRES on `&b` and out-votes the
correct `tail_lea_local_ax_byte0_e0_fetch_memsp` 0xE0 writer (also 1e6).

(The `_lea_local_e8_multilocal_guard_enabled` guard — `C4_LEA_LOCAL_E8_MULTILOCAL
_GUARD`, default ON — was a NO-OP here: it guards the LEGACY
`tail_lea_local_ax_marker_byte0_e8` rule, which is dead in the campaign frame
(CMP+7=0, MEM_ADDR_SRC=0). Forcing it `=0` left step-6 byte-identical. The LIVE
0xE8 writer is `e8_alubp_memsp`.)

### REJECTED first attempt (the func collision — important lesson)

The natural fix — add `FETCH_LO+0` / `FETCH_HI+14` NOT-blockers to `e8_alubp_memsp`
so it only fires on imm=-8 — **regressed func_identity** (550: baseline PASS →
FAIL step 6, `&x` LEA got 70 instead of 0xFFE8). Cause: `func_identity`'s `&x`
does NOT have a dead FETCH band (contra the keystone doc) — it carries
`FETCH_LO+0` (FETCH_HI nibble **1**, not F). So a `FETCH_LO+0` blocker silences
the 0xE8 writer func RELIES on. var&b and func&x BOTH carry `FETCH_LO+0`; the
only thing separating them is `FETCH_HI` (var&b = nib F/15, func&x = nib 1) — an
AND a single additive blocker cannot express.

### The fix (strength DOMINANCE, func-safe)

Rather than block `e8_alubp`, BOOST the strength of the CORRECT
`e0_fetch_memsp` (0xE0) and `d8_fetch_memsp` (0xD8) writers from **1e6 → 4e6**
(`_lea_e0d8_fetch_strength`). These writers ALREADY carry the precise multi-local
discriminator that excludes func: `e0_fetch` REQUIRES `FETCH_LO+0` AND
`FETCH_HI+15` (func's `&x` has FETCH_HI nib 1, so e0_fetch is DARK there). So the
boost out-votes the 0xE8 1e6 tie ONLY on the rows e0/d8 already fire:

| local | imm | FETCH sig         | e0/d8 fires? | result after fix |
|-------|-----|-------------------|--------------|------------------|
| var &a| -8  | LO+8, HI+15       | no           | e8_alubp → 0xE8 ✓ |
| var &b| -16 | LO+0, HI+15       | e0 (4e6)     | 0xE0 WINS ✓       |
| var &c| -24 | LO+8, HI+14       | d8 (4e6)     | 0xD8 WINS ✓       |
| func&x| -8  | LO+0, **HI+1**    | no (needs HI+15) | e8_alubp → 0xE8 ✓ |

Keying off the e0/d8 FIRE condition (which needs `FETCH_HI+15`, absent on func)
is the func-safe discriminator the blocker approach lacked.

### Result (GPU `run_1096_canonical --spec-k 0 --criterion full_trace`)

| id  | cluster        | baseline (flag-OFF) | fix-ON      |
|-----|----------------|---------------------|-------------|
| 250 | var_simple     | PASS                | PASS        |
| 300 | var_three      | FAIL step 6 (LEA &b)| FAIL step 9 (LI &b value) — ADVANCED |
| 275 | var_mul        | FAIL step 6 (LEA &b)| FAIL step 11 (LI &a value) — ADVANCED |
| 550 | func_identity  | PASS                | PASS        |

The multi-local LEA byte-0 alias is RESOLVED for BOTH var_three and var_mul; PC
stays correct throughout. Flag-OFF / golden byte-identical (`_lea_e0d8_fetch_
strength` returns 1e6 off the campaign branch → the writes tuple is identical).

## Next root (the new first divergence) — step-9 LI &b value blueprint

`step=9 expected(pc=98,ax=6) got(pc=98,ax=0)` — the **LI that loads `b`'s value
(6)** now returns **0**. PC is correct; this is an AX-VALUE root, the multilocal
LI value-load CAM the brief anticipated (R1). It is the SAME L15 head-0 / value
CAM surface as **var_mul step-11** (task #329) and **nested_sumsq step-9 LI=0**
(task #342) — coordinate; do NOT regress var_simple / func_identity (shared L15
head-0).

Blueprint: now that the `&b` ADDRESS is correct (0xFFE0), the LI's memory CAM
must content-address the `b`-store VALUE row by its byte-0 store address (0xE0)
rather than tie/recency-pick the `a`-store (0xE8) or the AX-null. This is the
`_l15_li_addr_cam_discriminator_on` / `_l15_li_zeroaddr_cam_on` family
(l15_ops.py) — the ADDR-CAM keys on `ADDR_B0_LO`/`ADDR_B0_HI`; verify the `&b`
store value row's address nibbles (lo=0, hi=E) vs the operand LI query row, and
why the 0xE0-addressed store loses to 0/the a-store in the AR decode. Probe with
`tools/probe_li_value_cam.py` adapted to var_three step-9. The `&b` value is 6
(low nibble 6, hi 0); the load returning 0 is byte-0 = 0 — a value-row-selection
miss, not a high-byte issue.
