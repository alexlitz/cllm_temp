# L15 LEV PC-restore: address-widening BREAKS the CAM aliasing in isolation, but the framing single-store fix is the hard prerequisite (2026-06-16)

Worktree HEAD: the L15 head-14 LEV PC-restore scaffold (`5b6bb7dd`, ==
`d2be82b7`, already on this branch — no cherry-pick needed). This lane owns the
MEMORY/CAM **address path** + the LEV value path (root #1's value half); the
STACK0-emission / framing surface (offsets 20-24, the dump/materialization, the
35-token layout) is a SEPARATE lane's territory and is NOT touched here.

Commit: `feat(l15): LEV return-addr address-widening on head 14
(C4_L15_LEV_ADDR_WIDEN, default OFF)`.

## TL;DR

The prior func-LEV lane (`docs/FUNC_LEV_PC_RESTORE_ADDRESS_ALIAS_WALL...`)
concluded the LEV gather is walled by same-address aliasing resolvable only
OUTSIDE the L15 head. **This lane found a clean VALUE-INDEPENDENT discriminator
that DOES separate the genuine return store from the same-address stale store
inside the head — the store's writing-opcode tag (`OP_JSR` vs `OP_ENT`).** With
it (+ a byte-0 address boost + K-side byte-0 selection) head 14 attends the
correct JSR-return byte-0 store with attention weight **1.000** in the isolated
gather. So the address-aliasing wall **is breakable in the head**.

BUT enabling the widening is **net-negative end-to-end** and is therefore
DEFAULT-OFF: (1) it regresses func_identity **exit_code 12/12 → 0/12**, and
(2) it regresses the SI/SC/LI/LC smoke CAM tests 6/6 → 1/6. Both regressions
trace to the SAME framing/STACK0 desync that gates func `full_trace` at step 4
(BEFORE the step-8 LEV is ever reached). The single clean store / clean decode
that the framing lane delivers is the hard prerequisite to turn the widening on.

## Ground truth (spec_k=0, BUILT dims, func_identity_0 id550 step-8 LEV)

`tools/_probe_lev_realattn.py 550 8` reads head-14's REAL attention (with the
`head_dim^-0.5` scale + the baked ALiBi mask — the prior `_probe_lev_cam`
score-only re-score is misleading; it omits the scale and the self/marker rows).
The gather loses in THREE distinct layers, each fixed by one widening term:

| # | competitor that wins | why | widening fix |
|---|----------------------|-----|--------------|
| 1 | the LEV PC-marker row **itself** (self-match, score ~2.1e5) | the query lives at the PC marker whose own ADDR_B0/B1/B2 IS the gather key | slot 31 suppressor 200×−200=−4e4 → **2000×−2000=−4e6** |
| 2 | the WRONG-FRAME saved return `mem[outer_BP+8]=0xFFF8` (value 10) | genuine `mem[inner_BP+8]=0xFFF0` (value 90) and 0xFFF8 differ ONLY in addr **byte 0**; query byte-0 lo-nibble is weak (pk~0.41) and the genuine store's byte-1 one-hot is softer (pk~1.74 vs 2.0) → the byte-1 softness (−3053) beats the byte-0 win (+139) | **C4_L15_LEV_B0_BOOST** (default 8): boost ONLY the byte-0 address bits so the same-byte-0 match dominates the byte-1 softness |
| 3 | the SAME-ADDRESS saved-BP word `mem[inner_BP+8]=0xFFF0` (value 0) | both at 0xFFF0; the saved-BP word has a higher `MEM_STORE` anchor (1.70 vs 1.53). **This is the prior lane's documented wall.** | **OP_JSR discriminator**: the return word was pushed by the caller's `JSR` (its store rows carry `OP_JSR`~18, `OP_ENT`~0); the saved-BP word by the callee's `ENT` (`OP_ENT`~18, `OP_JSR`~2). slot 64 keys `+OP_JSR / −OP_ENT`; slot 65 keys K-side byte-0 selection so the OP_JSR boost lands on the byte-0 row, not the byte-1/2/3 rows of the same JSR store |

After all three: `tools/_probe_lev_realattn.py 550 8` (C4_L15_LEV_ADDR_WIDEN=1)
→ `head14 attn top: p268(addr0xfff0, ..., w1.000)` — the genuine JSR-return
byte-0 store at 0xFFF0, attention weight 1.000.

### Why OP_JSR is the address-widening the brief asked for

The brief asked to make the return-addr store **uniquely addressable** so
same-address stores stop aliasing. `OP_JSR` is exactly that: a store-time tag of
the writing opcode that makes the return store's IDENTITY unique **without
changing its 24-bit address** — so the existing BP+8 query still address-matches
it, and the tag breaks the tie that the address alone cannot. It is
value-independent (not circular like matching on value 90). The prior lane only
tested `BP_SAVE_PREV` (a red herring — also strong at the adjacent value-store
byte rows) and recency (overshoots: the genuine store is NOT the most-recent
0xFFF0 store, later stale re-emissions exist); it did not test the
writing-opcode tag, which is the clean separator.

## Why it is DEFAULT-OFF — the framing prerequisite (the honest wall)

Two hard regressions when `C4_L15_LEV_ADDR_WIDEN=1`, both rooted in the framing
desync, NOT in the widening logic:

1. **func_identity exit_code 12/12 → 0/12.** With the head dark (default) the
   body halts at the LEV step with the correct AX and the benign wrong PC never
   gates exit_code (the prior `C4_LEV_AX_BYTE1_KILL` lane). Once head 14
   delivers ANY value to OUTPUT/PC at the LEV step it perturbs that
   previously-benign exit path; and because the DECODE is diverged (see #3) the
   value it delivers is wrong anyway, so it breaks the halt. Delivering the
   *correct* PC would require the LEV step to be reached on a CLEAN decode.

2. **SI/SC/LI/LC smoke CAM tests 6/6 → 1/6.** The byte-0 boost that is REQUIRED
   for the LEV byte-0 discrimination (layer #2) also makes the strong
   (`value_scale=40`, needed so the delivered byte registers as a clean OUTPUT
   one-hot) head fire on plain LI/LC store rows — the AX-marker query
   self-/store-matches the boosted address — clobbering the head-0 CAM load. A
   slot-66 `MEM_STORE` dark gate narrows it but softmax always re-normalises to
   attend SOMETHING, so a leak remains. Dropping the byte-0 boost fixes LI/LC
   but then the head attends junk at the LEV (layer #2 returns). The two cannot
   be decoupled while the decode is unstable.

3. **`full_trace` never reaches the LEV.** func_identity / func_add /
   nested_quad ALL diverge at **step 4** (`expected(pc=26,ax=<correct>)
   got(pc=None,ax=<correct>)`) — the 37-token framing desync (the JSR step emits
   a malformed PC) — BEFORE the step-8 LEV. So the value path is unreachable via
   full_trace regardless of the widening, and the isolated gather is probed on a
   DIVERGED decode whose return-store positions/values are unstable run-to-run.
   That instability is exactly why a clean LEV-only gate cannot be tuned (#1/#2).

## Verdict

The CAM same-address-aliasing wall **is breakable inside the L15 head** via the
`OP_JSR` writing-opcode discriminator (the address-widening this lane was asked
to pursue) — head 14 attends the genuine return store, weight 1.0. But the
widening is net-harmful end-to-end and is committed flag-gated DEFAULT-OFF
because the **framing single-store / clean-decode fix is the hard prerequisite**:
without it the LEV step is unreached (full_trace), the gather operates on an
unstable diverged decode, and enabling delivery regresses both func exit_code
(12/12→0/12) and the LI/LC CAM tests (6/6→1/6). This lane's contribution is the
proven discriminator + the flag-gated mechanism the framing lane can switch on
once the single clean store / clean decode lands.

## Gates (committed default, C4_L15_LEV_ADDR_WIDEN OFF)

* **Smoke 50/1** (the 1 fail is `test_simple_function`, pre-existing +
  flag-independent per the scaffold doc; SI/SC/LI/LC all green).
* **Byte-identity:** default build PARAM_HASH `2f722f61…45180` == the scaffold
  HEAD; head-14-off (`C4_L15_LEV_PC_RESTORE=0`) PARAM_HASH `88ae0573…` == the
  no-head baseline. The widening collapses to scaffold values when the master
  flag is off.
* **func_identity exit_code 12/12** (ids 550-561, default) — unchanged.

## Reproducers

```
# head 14 attends the genuine JSR-return store, w=1.0 (widening on):
CUDA_VISIBLE_DEVICES=0 C4_L15_LEV_ADDR_WIDEN=1 python tools/_probe_lev_realattn.py 550 8

# the OP_JSR vs OP_ENT discriminator at the candidate stores:
CUDA_VISIBLE_DEVICES=0 python tools/_probe_lev_discriminator.py 550 8 268 164

# the exit_code regression (12/12 off -> 0/12 on) — clear cache between A/B:
rm -rf ~/.cache/c4_release/compiled_vm/
CUDA_VISIBLE_DEVICES=0 python tools/run_1096_canonical.py --ids 550-561 --criterion exit_code            # 12/12
rm -rf ~/.cache/c4_release/compiled_vm/
CUDA_VISIBLE_DEVICES=0 C4_L15_LEV_ADDR_WIDEN=1 python tools/run_1096_canonical.py --ids 550-561 --criterion exit_code  # 0/12

# byte-identity (default == scaffold; head-off == no-head):
rm -rf ~/.cache/c4_release/compiled_vm/; CUDA_VISIBLE_DEVICES=0 python tools/probe_model_param_hash.py                       # 2f722f61
rm -rf ~/.cache/c4_release/compiled_vm/; CUDA_VISIBLE_DEVICES=0 C4_L15_LEV_PC_RESTORE=0 python tools/probe_model_param_hash.py  # 88ae0573
```
