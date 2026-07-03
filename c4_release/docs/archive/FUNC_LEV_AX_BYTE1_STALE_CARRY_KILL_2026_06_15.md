# func LEV epilogue: the exit_code lever is the AX byte-1 stale-carry dump, NOT the PC restore (2026-06-15)

Worktree base / HEAD: `072461c0`. One production commit landed: `461f1f47`
(`C4_LEV_AX_BYTE1_KILL`, default ON) + `b4614cbb` (diagnostic probes). Smoke
**51/0**; flag-off byte-identical to HEAD
(`PARAM_HASH=0791d32e...5965042`, `total_elems=458018907`).

## TL;DR — the brief's PC-restore model is for full_trace; exit_code is gated by AX

The brief targeted the LEV PC-restore (the missing L15 return-addr lookup heads
8-11, the ENT-frame BP, the multi-byte store disambiguation). All three are
**full_trace** concerns. The cluster-cracking metric is **exit_code**
(`neural EXIT value == decl EXIT`), and **the PC restore does NOT gate it**:

* `func_identity_0` (id550, FAIL) and `func_identity_11` (id561, PASS) **both**
  halt at step-8 LEV with `PC=10` (the "wrong" restore). id561 PASSES because
  its restored `AX=96` is clean; id550 FAILS because its `AX=0x646` (byte-1
  leak). The model HALTS at the LEV step with the restored AX as the EXIT value;
  `PC=10` is benign for exit_code.

So the exit_code lever for the entire func/nested LEV cluster is the **LEV-step
AX byte-1 value**, which the byte-1 register-dump corrupts.

## Root (spec_k=0, BUILT dims, func_identity_0 id550 step-8 LEV)

Exact LM-head logit attribution at the LEV-step AX byte-1 predictor row
(`tools/_probe_lev_byte1_attrib.py`, names resolved from the BUILT layout — the
static registry mislabels these dims as `SE_AX_CARRY_HI`):

```
logit[6]=80.06  logit[0]=34.09  diff=45.97
  dim 896 H2_DUMP_OUT+1   res=16.22  dW=+5.0  contrib=+81.1   <-- THE leak
  dim 882 H1_DUMP_OUT+2   res= 5.15  dW=-5.0  contrib=-25.8
```

`H2_DUMP_OUT+1` is the `ax_byte1_dump_repopulate` band (`l11_ops.py`). It is the
deliberate AX byte-1 cross-step carry (re-emits a multi-byte high byte the
normal H1 path truncates to 0 on a carried step). At the LEV step it **misfires**
and re-emits a STALE carried one-hot (value 6 → `H2_DUMP_OUT+(6-5)=+1` → AX
`70`→`0x646`). The dump fires because its lower-bound gate `Σ AX_CARRY` sits in
the carry band.

## Why a single magnitude discriminator IS clean here (the separability proof)

`tools/_probe_dump_gate.py` over the full add_0..11 + sub_0..7 corpus:

| step class                         | Σ AX_CARRY @ byte-1 row | dump should |
|------------------------------------|-------------------------|-------------|
| genuine multi-byte carry (ADD/SUB) | **2.65** (value-indep.) | FIRE (real high byte) |
| LEV func-return stale-carry        | **3.66**                | NOT fire    |
| SHL                                | ~12.85                  | NOT fire (unit 0 catches) |
| JMP                                | ~47.86                  | NOT fire (unit 0 catches) |

The genuine-carry cluster (1.98–2.65) and the LEV step (3.66) have a clean gap.
`OP_LEV` is NOT usable (it broadcasts to 0 at the dump row; `OP_PSH=1.0` there
instead — opcode contamination), but `Σ AX_CARRY` separates them cleanly.

## Fix (`C4_LEV_AX_BYTE1_KILL`, default-ON)

A 3rd `AX_CARRY_OVERFLOW` dump-kill unit (`_ax_byte1_carry_overflow_flag_rules`,
`l11_ops.py`): `multi_way_and_rule` over all 32 `AX_CARRY_{LO,HI}+k` cells at
weight 1.0, threshold **3.0**. On a genuine carry (2.65 < 3.0) it is dark
(byte-identical to the prior 2-unit design); on a LEV row (3.66 ≥ 3.0) it fires
→ the dump's `-1000` read of `AX_CARRY_OVERFLOW` darkens its AND →
`H*_DUMP_OUT == 0` → AX byte-1 falls back to the clean normal H1 path (value 0).

**SAFETY**: every func_*/nested_* return value is ≤ 255 (verified 0/75 in
550-599 + 950-974 exceed 255), so the dump never genuinely needs to fire at a
func LEV step — killing it there is exactly correct, not a trade. The op reads
`AX_CARRY_LO` only when the flag is on; the hidden-unit count + both compile
cache keys are flag-aware.

## Results (exit_code, spec_k=0)

| cluster                | flag OFF (= HEAD) | flag ON   |
|------------------------|-------------------|-----------|
| **func_identity 550-574** | **3/25**       | **25/25** ✓ |
| var_simple 250-274     | 15/25 (doc)       | **17/25** (+2) |
| add 0-24 + sub 50-74   | 40/50             | **40/50** (no regression) |
| if 400-449             | 33/50             | **33/50** (identical) |
| func_add 575-599       | 0/25              | 0/25 (NOT the lever) |
| nested_quad 950+       | 0/N               | 0/N (NOT the lever) |

func_identity is fully cracked on exit_code (the biggest clean win). var_simple
improves +2 (its frame steps also benefit). add/sub genuine carries are
byte-identical flag-on/off (the threshold preserves them). if/bool identical.

## What is STILL blocked (precise next step) — the step-4 37-token framing desync

`func_add` and `nested_quad` fail at **step 4** (`full_trace`:
`expected(pc=122,ax=11) got(pc=None,ax=11)` — AX correct, PC marker decode
returns `None`). This is the documented **37-token framing desync** (the callee
ENT step's MEM section emits 0xFF garbage → a sentinel-magnitude 0xFF
over-emitter latches on → the post-ENT step emits 37 tokens not 35 → the
fixed-35 marker slicer desyncs). It corrupts the JSR/ENT framing so the
multi-load + ADD/MUL body never computes the right result.

* `func_identity` survives this for **exit_code** only because it has ONE LI
  load and returns a single clean byte — the model halts at LEV with the right
  AX despite the framing being desynced (full_trace still fails at step 4).
* `func_add` (two frame loads + ADD) and `nested_quad` (frame load + MUL +
  nested call) need the body to compute correctly, which the desync breaks.

The 37-token desync is the THREE+-part wall documented in
`FUNC_LEV_IS_LI_FROM_FRAME_37TOKEN_DESYNC_2026_06_14.md`: (1) the val-store
carry (`C4_BP_SAVE_DUMP`, landed), (2) the ENT-store ADDR-byte carry (NOT built),
(3) the post-ENT 0xFF over-emit suppression (other lanes). It is the SAME
surface as the var/if/bool framing-drift and the AX-byte-1 H1-one-hot wall, and
must be sequenced with those lanes — NOT a solo corrector.

NEXT for the func cluster: build the ENT-store **ADDR**-byte cross-step carry
(mirror `make_bp_save_prev_carry_op` with an `SP`-byte source + an ADDR-row dump
gate) AND the post-ENT 0xFF over-emit suppression, together, so the post-ENT
step returns to 35 tokens and the JSR/ENT framing stays in sync.

## The LEV PC restore (sub-roots 1-3) — confirmed present but full_trace-only

* Production L15 `num_heads=10` (head_dim=109), LEV return-addr heads 8-11
  ABSENT (only emitted when `num_heads >= 12`) — confirmed
  (`tools/_probe_l15_lev_heads.py`). The L9 `bp_plus8_shift` +
  `addr_b1_set_and_cascade` FFNs that build the `mem[BP+8]` query DO exist.
* At the LEV step the L9-relayed query is `0xfff0` (BP+0), not `0xfff8` (BP+8):
  the +8 shift now fires at `MARK_SE_ONLY` (Wave-B migration) but the L15
  return-addr heads query at the PC marker, so even if heads 8-11 were enabled
  the query row would be unshifted. AND the JSR return-address STORE itself is
  0xFF-garbage-corrupted by the same 37-token desync (`mem` addr bytes
  `[240,90,10,10]`), so no clean `mem[BP+8]=90` store exists to look up.
* Therefore enabling heads 8-11 alone cannot restore PC=90 until the store is
  clean (desync) AND the query reaches the right row — a multi-part build
  downstream of the desync wall, and it only affects `full_trace`, not the
  exit_code cluster prize.

## Reproducers

```
# func_identity exit_code 3/25 -> 25/25:
CUDA_VISIBLE_DEVICES=1 python tools/run_1096_canonical.py --ids 550-574 --criterion exit_code

# LEV byte-1 now clean (was 0x646):
CUDA_VISIBLE_DEVICES=1 python tools/_probe_func_li.py 550   # step 8 AX=0x46=70

# the leak attribution (flag off shows H2_DUMP_OUT+1=16.2):
CUDA_VISIBLE_DEVICES=1 C4_LEV_AX_BYTE1_KILL=0 python tools/_probe_lev_byte1_attrib.py 550 8

# byte-identity (all new flags off == HEAD):
CUDA_VISIBLE_DEVICES=1 C4_BP_SAVE_DUMP=0 C4_ENT_SP_BYTE1_FF_H1_HARDEN=0 \
  C4_PSH_ARG_VAL_AX=0 C4_AX_BYTE23_DUMP=0 C4_POST_ENT_SE_SUPPRESS=0 \
  C4_L15_LI_SUPPR_INERT=0 C4_LEV_AX_BYTE1_KILL=0 python tools/probe_model_param_hash.py
```
