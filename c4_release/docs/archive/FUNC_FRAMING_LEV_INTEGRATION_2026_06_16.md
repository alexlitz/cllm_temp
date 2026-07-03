# func/nested/rec: framing + LEV PC-restore INTEGRATED — PC now correct end-to-end; sole remaining wall is the step-6 AX-clobber (2026-06-16)

This lane integrated the two proven building blocks for the func/nested/rec call
machinery and measured them together for the first time:

1. **Part A — the post-ENT framing desync (the unlock).** Cherry-picked
   `C4_PSH_STACK0_BYTE3_RELAY_DARKEN` (commit `9139db7d` → `cda712fc` here),
   the correctly-localized fix for the **34-token SHORT step** at the PSH-of-arg
   store (func/nested/rec **step 3/4**). Root: `layer10_psh_stack0_passthrough`
   head 3 relays the `NEXT_SP`-darkened AX byte-3 OUTPUT into the STACK0 byte-3
   decode, crushing the value-0 token so the MEM marker emits one position early.
   Fix = a `BYTE_INDEX_2`-keyed hard NOT-blocker on a free Q/K slot of head 3.
   **Verified:** `tools/_probe_freerun_tokens.py 550` step 3 = **34 → 35** tokens
   with the flag on. The step-4 `got pc=None` desync is gone.

2. **Part B — the LEV PC-restore mechanism.** Merged branch
   `worktree-agent-aa6d8f840ccc6cb06` (`C4_L15_LEV_PC_RESTORE` head-14 scaffold
   `5b6bb7dd` + `C4_L15_LEV_ADDR_WIDEN` `f17ffcaf`). The merge shipped
   `C4_L15_LEV_PC_RESTORE` **default-ON**, which grows L15 to 15 heads and is
   output-affecting; flipped it to **default-OFF** (`33b8ab56`) so the default
   build is byte-identical (golden `7474f26955e79619`).

## Measured result (spec_k=0, `run_1096_canonical --criterion full_trace`)

| cluster | OFF (baseline) | ALL-3 ON | what changed |
|---|---|---|---|
| func_identity (550-574) | div step 4, **pc=None** (framing desync) | div step 7, **pc CORRECT 25/25**, ax=0 | framing fixed; LEV delivers PC |
| func_add (575-599) | div step 4, pc=None | div step 5/9; 5/25 pc-correct | framing fixed; step-5 multi-arg JSR PC root surfaces |
| nested_quad (950-974) | div step 3, pc=186 wrong | div step 3, **pc CORRECT 25/25**, ax=0 | framing fixed |
| rec_factorial (700-709) | div, pc wrong | 4/4 pc-correct at div | framing fixed |

**Both halves work.** Framing is clean (no more `pc=None`) and with
`C4_L15_LEV_ADDR_WIDEN` head-14 delivers the correct return PC (func_identity PC
correct end-to-end through the LEV epilogue — was `pc=None`).

## Why full_trace PASS count is still 0 — the THIRD root: the step-6 AX-clobber

`func_identity_0` (`identity(70)`) free-run AX trace, all 3 flags ON
(`tools/_probe_freerun_step.py 550 <step>`, AX bytes at offsets 6-9):

| step | PC | AX bytes | note |
|---|---|---|---|
| 5 | 34 | **[70,0,0,0] = 70** | the LI of the argument loads the value CORRECTLY |
| 6 | 42 | **[232,255,0,0] = 0xFFE8 = 65512** | AX CLOBBERED by frame-pointer garbage (logits ~1.9e9/3.3e9) |
| 7 | 50 | [0,0,0,0] = 0 | divergence: expected ax=70, got 0 |

So the LI value path is NOT the blocker (it loads 70 at step 5). The blocker is
a **step-6 AX-dump clobber**: AX is overwritten with the BP/frame-pointer value
`0xFFE8` (65512) right after the correct LI, then decays to 0 by the LEV return.
This is the secondary LEV root documented in
`project_func_lev_pc_restore_missing_l15_heads`: the AX byte-dump
(`H2_DUMP_OUT` / register dump) reading a stale `H2_PREV_STEP` / BP-relative
frame one-hot injected at the ENT step, amplified by the LEV's large ΣAX_CARRY.
NEXT LANE: gate the step-6 AX dump so the LI-loaded AX (70) survives to the LEV
return marker. (`tools/interp_oracle_gate.py 550` flags this as CROSS-STEP /
"value-corruption@step1" — its low-confidence attention/relay zone, consistent
with an attention-dump root, not a single FFN writer.)

## Why the flags ship DEFAULT-OFF (the trade is heavy, dominated by ADDR_WIDEN)

`C4_L15_LEV_ADDR_WIDEN` is REQUIRED for the func PC win (without it func_identity
PC is wrong at step 8 — the value_scale-1.0 scaffold head doesn't pick the unique
return store) but its byte-0 boost + value_scale-40 head-14 **clobbers the LI/LC
CAM** during plain loads:

| config | var_simple (250-274) exit_code | smoke |
|---|---|---|
| all OFF (default) | **17/25** | **51/0** |
| byte3 only | 14/25 (−3) | — |
| byte3 + LEV_PC_RESTORE (no widen) | 14/25 (−3) | — |
| byte3 + LEV_PC_RESTORE + **ADDR_WIDEN** | **0/25 (−17)** | **45/6** (the 5 SI/SC/LI/LC + test_simple_function) |

The control arithmetic clusters (add 7/10, sub 10/10, div 10/10, mod 8/10) are
**0-regression** under all configs. The damage is entirely `ADDR_WIDEN` →
LI/LC-CAM clobber. Since func/nested still get **0 full_trace passes either way**
(blocked by the step-6 AX-clobber above), enabling `ADDR_WIDEN` is strictly
net-negative today. All three flags therefore stay DEFAULT-OFF; the default build
is byte-identical (`7474f26955e79619`), smoke 51/0.

## The closure sequence for the next lane

1. **Fix the step-6 AX-dump clobber** (the new precise root) so the LI-loaded
   AX=70 survives to the LEV return marker. THEN func_identity passes full_trace
   (framing + LEV-PC already deliver the correct PC).
2. **Decouple `ADDR_WIDEN` from the LI/LC CAM** (gate head-14's value_scale-40
   delivery strictly to the LEV PC marker so it stops firing on LI/LC store rows)
   — restores var_simple/SI/SC/LI/LC so the flags can ship ON.
3. func_add additionally needs the **step-5 multi-arg JSR PC root** (got_pc=26
   vs exp_pc=130 — the call branches into the function body instead of the
   caller's continuation).

Overlap note: this lane touched L10 (byte3 head) + L15 (LEV head). The
STACK0-emission framing lane touches L4/L8/L11 — different region.
