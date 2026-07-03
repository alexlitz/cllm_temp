# Post-ENT MEM-desync INTEGRATION TEST — the 3-flag combination does NOT advance; the residual root is the PSH-argument STORE VALUE relay (2026-06-14)

Worktree: `agent-a3bd21ffd07e87dd8` (base `main` = `0339c2e7`). This is the
DECISIVE integration test that all four post-ENT lanes pointed to: combine the
three proven, flag-gated building blocks and measure func/nested/rec/var.

**Result: the combination does NOT advance a single program.** The remaining
exit-determining root is now precisely localized (built-dim, spec_k=0): the
**caller's PSH of the function ARGUMENT stores a corrupted VALUE** (the store
value thermometer is near-empty), so the callee's frame-local `LI` loads 0/garbage.
This is INDEPENDENT of the post-ENT 37-token framing desync (proven below).

## What was integrated (cherry-picks onto main HEAD)

| block | commit | flag (default) | files |
|-------|--------|----------------|-------|
| R1 store-value carry (ENT saved-BP) | `70c810e6` | `C4_BP_SAVE_DUMP` (ON) | l11_ops, all_core_ops, full_vm_compiler_dynamic |
| R3#1 arg-store SP-byte1 hardening   | `eba857ea` | `C4_ENT_SP_BYTE1_FF_H1_HARDEN` (ON) | l16_ops |
| R3 framing (37→35 SE suppressor)    | `0cd4f251` | `C4_POST_ENT_SE_SUPPRESS` (OFF) | l6_ops, l16_ops, full_vm_compiler_dynamic |

Cherry-pick conflicts: `l16_ops.py` (R3#1 vs R3-framing) — both touch the file
header (`import os` / `import os as _os`), add sibling flag helpers, and both
finalize `sp_frame_byte1_ff_conditions`. Resolved by KEEPING BOTH: R3#1's
net-zero `CONST` baseline (`+ ((("CONST", -_h1_harden_w),) if _h1_harden_w else ())`)
AND R3-framing's conditional `IS_MARK` blocker append. The
`full_vm_compiler_dynamic.py` cache-key additions from R1 (auto via the
`BP_SAVE_PREV` `register_residual_band`) and R3-framing (`C4_POST_ENT_SE_SUPPRESS`
in both snapshots) auto-merged cleanly.

### Cache-key confound FIXED (commit `7ff67266`)

R3#1 (`eba857ea`) shipped the default-ON, **output-affecting**
`C4_ENT_SP_BYTE1_FF_H1_HARDEN` flag but NEVER registered it in the compile
cache-key snapshots (it only touched `l16_ops.py`). ON and OFF builds would
therefore SHARE a disk/memo entry — the exact silent cache trap a prior agent
hit. Added the flag to BOTH snapshots in `full_vm_compiler_dynamic.py`
(in-process memo + disk). All A/B measurements below use these honest cache keys
(verified by build wall-time deltas — ON and OFF compile separately).

## Integration-test results (spec_k=0, GPU 0)

`tools/run_1096_canonical.py --ids <range> --criterion {full_trace,exit_code}`

### full_trace (per-step PC/AX), ALL THREE ON (`C4_BP_SAVE_DUMP=1 C4_ENT_SP_BYTE1_FF_H1_HARDEN=1 C4_POST_ENT_SE_SUPPRESS=1`)

```
550-560 func_identity : 0/11  — every program step=7 (LI) expected(pc=50,ax=val) got(pc=50,ax=0)
575-580 func_add      : 0/6   — step=9 (first LI) ax=0  -> returns arg b (0+b), neural=11 for add(57,11)
250-255 var_simple    : 0/6   — step=4 PC desync (expected pc=58 got pc=66); exit garbage
700-710 rec_factorial : 0/11  — 0! returns 1 (PC ok) but full_trace fails on the LI sub-steps
950-955 nested_quad   : 0/6   — ax=0
TOTAL 0/40 (6 deep rec skipped >cap)
```

### exit_code, ALL ON vs ALL OFF (byte-identity baseline)

| id | cluster | expected | ALL-ON exit | ALL-OFF exit |
|----|---------|----------|-------------|--------------|
| 250-251 | var_simple | 990/791 | 768/768 | 768/768 (identical) |
| 252-253 | var_simple | 285/359 | 256/256 | 256/256 (identical) |
| 550 | func_identity | 70 | 0 (full_trace) / 1536 (free-run) | 1536 |
| 575 | func_add | 68 | 11 | 3056 |

**0/18 PASS on exit_code for ALL-ON; 0/18 for ALL-OFF.** The framing flags DO
move bytes (func exit shifts 1536→0/garbage), but NEVER to the correct value.
**No func/nested/rec/var program advances under any flag combination.**

## Why it does NOT advance — the PSH-argument STORE VALUE is corrupted (the residual root)

The oracle for `func_identity_0` (id 550, `identity(70)`):
```
2 IMM 70  AX=70                                  *** neural AX=70 OK (step 2)
3 PSH 0   mem[65512]=70  (push the argument)     *** neural STORES val=65546 (low byte 10) NOT 70  <<< ROOT
5 ENT 0   identity frame; BP=65496
6 LEA 16  AX = BP+16 = 65512  (frame-local arg)  *** neural AX=65512 OK (step 6)
7 LI  0   AX = mem[65512] = 70                   *** neural AX=0   (the consumer of the bad store)
```

STEP-3 diagnosis questions answered at BUILT dims (`tools/_probe_func_tokens.py`,
`tools/_probe_storeval_path.py`, spec_k=0, d_model=1090, 48 blocks):

* **(a) Is the arg stored at its frame address by the caller's PSH?**
  ADDRESS yes, VALUE no. The step-3 MEM section decodes
  `addr = [232,255,0,0] = 65512` (CORRECT) but `val = [10,0,1,0] = 65546`
  (WRONG — should be 70). Low byte = `10` (0x0A, leaked frame residue), spurious
  byte-2 = 1.
* **(b) Does the LI compute the right load address [BP+offset]?**
  YES — step-6 LEA gives AX = 65512, relayed to STACK0 = 65512 at the LI step.
* **(c) Does the L15 memory_lookup read the right stored value?**
  NO — returns 0 (the store entry's value is the corrupted 65546).

### The exact corruption, traced block-by-block (the single most useful artifact)

`tools/_probe_storeval_path.py {si|550} 3 4` traces the store-VALUE relay at the
`val_b0` prediction row across all blocks. Compare the WORKING SI store (val=42)
to the BROKEN func PSH-arg store (val should be 70):

| | block 12 (logical L10) STACK0_B0 thermometer | block 29 (logical L18) OUTPUT byte |
|---|---|---|
| **SI store, val=42** | `B0_H1_PREV` max=160.0, `B0_H3_PREV` max=170.2 (encodes 42) | `OUT_byte=42` ✓ |
| **func PSH-arg, val=70** | `B0_H1_PREV` max=**40.1**, `B0_H3_PREV` max=**4.8** (near-empty) | `OUT_byte=10` ✗ |

The store value enters via the **L10 (block 12) STACK0 byte-0 thermometer**
(`STACK0_B0_H1_PREV` dim 916, `STACK0_B0_H3_PREV` dim 923; the Root-2
`C4_STACK0_B0_DUMP` carry bands) and is decoded to the MEM val byte at **L18
(block 29)**. For the SI store the byte-0 value (42) populates the thermometer
and decodes; for the func PSH-of-argument the thermometer is near-empty
(40.1/4.8 = noise) so L18 emits a default garbage byte (`10` = 0x0A = the leaked
residue).

### This root is INDEPENDENT of the post-ENT 37-token framing desync

Decisive control: the storeval trace and the token dump are **byte-identical
under ALL-ON (`C4_POST_ENT_SE_SUPPRESS=1`, 35-token frame) and ALL-OFF
(suppressor off, 37-token frame)**:
```
ALL-ON  step-3 val_bytes=[10,0,1,0]  blk12 B0H1=40.1 B0H3=4.8  blk29 OUT_byte=10
ALL-OFF step-3 val_bytes=[10,0,1,0]  blk12 B0H1=40.1 B0H3=4.8  blk29 OUT_byte=10
```
So the framing recovery (R3) — which mechanically restores 35-token steps and a
clean PC trace (`docs/POST_ENT_SE_VALUE_SUPPRESSOR_NEGATIVE_2026_06_14.md`) —
does NOT touch the PSH-arg store value. The store value corruption is upstream
of and independent from the desync. THIS is why all three flags together still
fail: R1 fixed the ENT saved-BP store value, R3 fixed framing, R3#1 hardened the
SP-byte1 emitter — but NONE of them touch the **PSH-of-argument** store value.

## Why the three flags are individually correct but jointly net-neutral

* **R1 (`C4_BP_SAVE_DUMP`)** carries `old_BP` into the ENT saved-BP store VALUE
  (a DIFFERENT store from the arg PSH). Verified: it changes the step-1 ENT MEM
  section (BP byte pattern moves). But the LI loads the ARG store (step 3), not
  the ENT store, so the ENT-store fix is invisible to the LI.
* **R3#1 (`C4_ENT_SP_BYTE1_FF_H1_HARDEN`)** suppresses the OP_ENT-broadcast
  SP-byte1=0xff misfire (a framing/MEM-row 0xFF emitter). Correct, but it is a
  framing-side fix; the arg store value is unchanged.
* **R3 (`C4_POST_ENT_SE_SUPPRESS`)** restores 35-token framing. Per its own
  negative-result doc the exit is byte-identical ON vs OFF because the LI value
  is wrong regardless of frame width — re-confirmed here.

## Final flag defaults (chosen)

* `C4_BP_SAVE_DUMP` = **ON** (default). Clean, byte-identity-off, smoke 51/0;
  it correctly fixes the ENT saved-BP store VALUE (a necessary building block for
  the eventual full frame even though it does not advance alone).
* `C4_ENT_SP_BYTE1_FF_H1_HARDEN` = **ON** (default). Clean, byte-identity-off,
  smoke 51/0; correct hardening kept on.
* `C4_POST_ENT_SE_SUPPRESS` = **OFF** (default, unchanged). It is the framing
  half; per the negative result it does not change any exit and ADDS 32 FFN
  units to the final block. Keep it default-OFF (opt-in) — flipping it ON is
  pure cost with zero exit benefit until the LI value path is fixed. Turn it ON
  together with the value fix (it will then be load-bearing for a clean trace).

## GATES (all green)

* **Smoke**: `CUDA_VISIBLE_DEVICES=0 pytest tests/test_smoke.py` = **51 passed /
  0 failed** at the chosen defaults (SI/SC/LI/LC green).
* **Byte-identity**: integrated worktree with ALL THREE flags OFF
  (`C4_BP_SAVE_DUMP=0 C4_ENT_SP_BYTE1_FF_H1_HARDEN=0 C4_POST_ENT_SE_SUPPRESS=0`)
  produces `PARAM_HASH=0791d32e...5965042`, `total_elems=458018907` — **IDENTICAL**
  to the `main` (`0339c2e7`) baseline build (`tools/probe_model_param_hash.py`).
* **Guards (add/sub/mul/if/bool, ids 0,1,2,50,51,100,101,350,351,1071,1072)**:
  exit codes BYTE-IDENTICAL between DEFAULT and ALL-OFF (7/11 pass both; the 4
  fails {1:816, 2:563, 100:61655, 350:1} are pre-existing baseline issues
  unrelated to this work). The flags are inert outside ENT-bearing programs.

## The exact NEXT fix (a genuine further multi-commit piece)

The residual root is the **byte-0 store value of a PSH that pushes a function
argument**. The fix is a new carry-band lane mirroring the proven R1/Root-2
pattern, but for the PSH-ARG store value (NOT old_BP):

1. **Producer to repair**: the **L10 (block 12) STACK0 byte-0 store-value
   thermometer** (`STACK0_B0_H1_PREV` dim 916 / `STACK0_B0_H3_PREV` dim 923),
   read by the **L18 (block 29)** OUTPUT byte decoder. For the func PSH-arg the
   thermometer is near-empty (40.1/4.8) — the pushed AX value (70) is not
   relayed into it.
2. **Source of the value**: at the PSH step, AX holds the argument (step-2 IMM
   set AX=70, confirmed in the trace). The relay must content-address AX's
   byte-0 H1/H3 nibble one-hots into the store-value thermometer at the MEM
   val_b0 predictor row. The existing `_layer10_psh_ax_broadcast_head_spec`
   (l10_ops.py:2528) already relays AX bytes **1/2/3** to `STACK0_BYTE_VAL_h`
   during OP_PSH — but byte-**0** (the bulk of small arg values) goes through the
   STACK0_B0 thermometer, which is where the gap is. Probe whether the func PSH
   row's AX byte-0 one-hot is present (it should be, AX=70) and why the carry
   head does not pick it up (likely: the carry head's K-row content-addressing
   selects frame residue — value 10 — over the AX=70 row, the same recency/frame
   aliasing R1's STACK0_BYTE reject solved for the ENT store).
3. **Gate it** flag-default-ON via `register_residual_band` (a new
   `PSH_ARG_VAL_PREV`-style band) so flag-off stays byte-identical; verify
   `tools/_probe_storeval_path.py 550 3 4` decodes `OUT_byte=70` at block 29,
   then re-run STEP 2 (the LI should return 70 and func_identity advance).
4. Byte-identity gate against `test_simple_function` (no PSH-arg/LI, must stay
   green) and SI/SC/LI/LC smoke (clean 35-token, must stay green).

Sequence this WITH the R3 framing flag flipped ON (it becomes load-bearing for a
clean PC trace once the value is correct). var_simple (250-255) has an additional
PC-desync at step 4 (expected pc=58 got pc=66) that is a separate framing member
(`task #228`); but its exit is determined by the same value path.

## Reproducers / tools

```
# the integration test
CUDA_VISIBLE_DEVICES=0 C4_BP_SAVE_DUMP=1 C4_ENT_SP_BYTE1_FF_H1_HARDEN=1 \
  C4_POST_ENT_SE_SUPPRESS=1 python tools/run_1096_canonical.py \
  --ids 550-560,575-580,250-255,700-710,950-955 --criterion full_trace

# the root (block-12/29 store-value thermometer, SI vs func-PSH)
CUDA_VISIBLE_DEVICES=0 python tools/_probe_storeval_path.py si  3 4   # working: OUT_byte=42
CUDA_VISIBLE_DEVICES=0 python tools/_probe_storeval_path.py 550 3 4   # broken:  OUT_byte=10

# the framing-independence control (identical ON vs OFF)
CUDA_VISIBLE_DEVICES=0 C4_POST_ENT_SE_SUPPRESS=1 python tools/_probe_func_tokens.py 550 12
CUDA_VISIBLE_DEVICES=0 C4_POST_ENT_SE_SUPPRESS=0 python tools/_probe_func_tokens.py 550 12

# byte-identity (all flags off == main)
CUDA_VISIBLE_DEVICES=0 C4_BP_SAVE_DUMP=0 C4_ENT_SP_BYTE1_FF_H1_HARDEN=0 \
  C4_POST_ENT_SE_SUPPRESS=0 python tools/probe_model_param_hash.py
```
