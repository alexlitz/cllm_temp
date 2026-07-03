# PSH-of-argument store-value AX lock — the LI-arg-store root is FIXED; the wall does NOT yet crack (a second piece remains) (2026-06-14)

Base `main` = `e0665ca8` (the post-ENT integration-test HEAD). Flag
`C4_PSH_ARG_VAL_AX` (DEFAULT-ON). This lands the precise residual root that the
five post-ENT lanes pointed to
(`docs/POST_ENT_INTEGRATION_TEST_LI_ARG_STORE_2026_06_14.md`): the caller's PSH
of a function argument stored a CORRUPTED byte-0, so the callee's frame-local
`LI` loaded garbage.

## The exact root (built dims, spec_k=0, d_model=1090, 48 blocks)

`func_identity(x){return x;}` (id 550, `identity(70)`):
```
2 IMM 70  AX=70
3 PSH 0   mem[65512]=70  (push the argument)   *** stored byte-0 = 10 (0x0A) NOT 70  <<< ROOT
5 ENT 0   identity frame
6 LEA 16  AX = BP+16 = 65512
7 LI  0   AX = mem[65512] = 70                  *** returned 0 (consumes the bad store)
```

The store VALUE is decoded by **`layer14_mem_generation` head 4 (MEM val
byte 0)**, which runs at **logical layer 18 / physical block 29** (NOT block 16
— the "L14" name is historical; `layout.ops_at(18)` returns
`layer14_mem_generation`). Head 4 selects its value SOURCE via slots 1/2/36
(STACK0 source for JSR/ENT, AX source otherwise) and slots 44/45 (the ENT
old-BP source override). Per-slot attention decomposition on the func PSH-arg
val row (block-28 input, head 4):

| K row | token | total score | dominant slot |
|-------|-------|-------------|---------------|
| 129 (garbage) | 10 (0x0A) | ~104k | **slot 44 = +35580** |
| 220 (AX byte-0, the arg) | 70 (0x46) | ~72.6k | slot 44 = +2053 |

**Root mechanism (the OP_ENT-broadcast misfire, same class as l14_ops.py:947
for var):** the func program is `... PSH; JSR; ENT; ...`. The ENT step's
`OP_ENT` BROADCASTS a ~0.30 residue back onto the PSH val row via the KV cache.
Slot 44 (the ENT old-BP value-SOURCE override) is Q-gated ONLY on `OP_ENT`; its
existing broadcast hardening blocks only REGISTER-emit rows
(MARK_PC/AX/SP/BP), leaving the MEM value-target row unguarded. So
`Q_slot44 = 80*0.30 = 24`, multiplied by the prologue old-BP row's GENUINE
`OP_JSR=10.79` (`K_slot44 ~ 1482`), gives a **+35580** pull onto the garbage
old-BP row (value 0x0A), burying the AX byte-0 row (the pushed argument). On a
WORKING SI store `OP_JSR==OP_ENT==0`, so slot 44 is inert and head 4 cleanly
locks onto AX byte 0 (value 42). The STACK0_B0 thermometer (dims 916/923) the
prior doc flagged is the Root-2 LM-head register-dump path (read only by
`stack0_byte0_dump_*` + `model_ops`), a CORRELATE of the value chain, not the
per-row MEM store-value decoder — the true driver is slot 44.

## The fix (`C4_PSH_ARG_VAL_AX`, DEFAULT-ON; head 4 / val byte 0 only)

Two fresh attention slots on `layer14_mem_generation` value head 4, injected in
`_layer14_mem_generation_head_specs_with_overrides` (l14_ops.py), flag-gated so
flag-OFF omits them entirely (byte-identical):

* **Slot 46 — ENT-source cancel.** `K46 @ -OP_JSR` (negates the slot-44
  ENT-prologue selector; `OP_JSR` is large ONLY on real JSR rows and is exactly
  0 in non-call programs, so SI/SC/normal-PSH contexts are untouched).
  `Q46 = 50*MEM_STORE - 25*OP_ENT` — sign-flips at `OP_ENT = 2.0`: on a PSH
  store (OP_ENT 0.30) `Q46 ~ +42.5` so `Q46*K46 ~ -36.7k` CANCELS the +35.6k
  slot-44 pull; on a GENUINE ENT store (OP_ENT 10.79) `Q46 ~ -382` flips sign
  so `Q46*K46 = (neg)*(neg)` REINFORCES slot 44 (ENT old-BP source stays
  authoritative).
* **Slot 47 — AX byte-0 in-step boost.** With slot 44's cross-step garbage
  cancelled, the residual competition is in-step (AX byte-0 row vs the step's
  frame/SP byte rows, within ~300). `K47 @ (H1+AX one-hot AND BYTE_INDEX_0)`
  (= 2 on the AX byte-0 row). `Q47 = 2000*MEM_STORE - 400*(OP_JSR+OP_ENT)`:
  PSH/SI/SC (OP sum <= ~0.34) → +3.7k onto AX byte-0 (wins); real JSR/ENT
  (>= 10.79) → `Q47 < 0` repels (correct, JSR/ENT source STACK0). Position-only
  K, so ALiBi recency keeps the CURRENT step's AX row.

SI/SC already pick AX so both slots are no-ops there (byte-identical OUTPUT).

## Result — the LI-arg-store root is FIXED; the wall does NOT yet crack

Built-dim verification (`C4_BP_SAVE_DUMP=1 C4_ENT_SP_BYTE1_FF_H1_HARDEN=1
C4_POST_ENT_SE_SUPPRESS=1 C4_PSH_ARG_VAL_AX=1`):

* **Store value:** func 550 step-3 PSH-arg val byte-0 = **10 → 70 (0x46)**
  (SI store stays 42 — byte-identical).
* **full_trace (the consumer):** the func/func_add clusters' divergence moves
  from **step-7/9 (LI)** to **step-1**, CLUSTER-WIDE:
  ```
  FLAG OFF: id=0550 step=7 expected(pc=50,ax=70) got(pc=50,ax=0)   <- broken LI store value
  FLAG ON : id=0550 step=1 expected(pc=66,ax=0)  got(pc=66,ax=168427520)  <- LI fix consumed; new earlier root
  ```
  The step-7 LI failure is ELIMINATED for every func_identity_0..10 and
  func_add_0..5 (the LI now reads the correct value). This is the documented
  root, fixed.

* **exit_code:** still **0/N** — the programs do NOT advance, because the new
  step-1 divergence (`ax=168427520 = 0x0A090200`, expected 0) is the
  **ENT-step AX-register high-byte DUMP garbage** — the pre-existing AX byte-1
  H1-one-hot wall (memory `project_ax_byte1_dump_is_h1_onehot_wall`, tasks
  #222/#227), INDEPENDENT of the store value and OUT OF SCOPE here. AX byte-0 is
  correct (0); bytes 1/2/3 = 0x02/0x09/0x0A leak.

**Verdict: CORRECT-BUT-INSUFFICIENT.** The PSH-arg store-value root is
genuinely fixed (a necessary building block for the eventual frame); the wall
now reduces to ONE remaining piece — the step-1 ENT AX-register-dump high-byte
leak. Land that (deliberate two-part build per the AX-byte-1 wall note) WITH
this fix and `C4_POST_ENT_SE_SUPPRESS` flipped ON, and func/nested/rec/var
should advance.

## Final flag defaults (chosen)

* `C4_PSH_ARG_VAL_AX` = **ON** (default). Smoke 51/0, byte-identity-OFF,
  guards no-regression; resolves a real documented root (the LI-arg-store
  divergence) cluster-wide.
* `C4_POST_ENT_SE_SUPPRESS` = **OFF** (unchanged). Per its negative-result doc
  it is pure cost (adds 32 FFN units) with zero exit benefit until the AX-dump
  piece also lands; flip it ON together with that fix.
* `C4_BP_SAVE_DUMP`, `C4_ENT_SP_BYTE1_FF_H1_HARDEN` = **ON** (unchanged).

## GATES (all green)

* **Smoke**: `CUDA_VISIBLE_DEVICES=0 pytest tests/test_smoke.py` =
  **51 passed / 0 failed** (SI/SC/LI/LC green) at the chosen defaults.
* **Byte-identity**: ALL flags OFF (`C4_PSH_ARG_VAL_AX=0` + the three building
  blocks OFF) → `PARAM_HASH=0791d32e...5965042`, `total_elems=458018907` —
  **IDENTICAL** to HEAD (`e0665ca8`) (`tools/probe_model_param_hash.py`).
* **Guards** (add/sub/mul/if/bool, ids 0,1,2,50,51,100,101,350,351,1071,1072):
  exit-code **7/11 PASS both DEFAULT and ALL-OFF** (the 4 fails {1,2,100,350}
  are pre-existing). The flag is inert outside PSH-arg-before-JSR/ENT programs
  (OP_JSR==OP_ENT==0 → SI/SC/PSH path unchanged).

## Reproducers

```
# the store-value fix (block-29 head-4 val_b0): 10 -> 70
CUDA_VISIBLE_DEVICES=0 C4_BP_SAVE_DUMP=1 C4_ENT_SP_BYTE1_FF_H1_HARDEN=1 \
  C4_POST_ENT_SE_SUPPRESS=1 python tools/_probe_storeval_path.py 550 3 4  # OUT_byte=70

# the divergence-step shift (LI root fixed -> step-1 AX-dump remains)
CUDA_VISIBLE_DEVICES=0 C4_BP_SAVE_DUMP=1 C4_ENT_SP_BYTE1_FF_H1_HARDEN=1 \
  C4_POST_ENT_SE_SUPPRESS=1 C4_PSH_ARG_VAL_AX=1 python tools/run_1096_canonical.py \
  --ids 550-560,575-580,950-955 --criterion full_trace

# byte-identity (all flags off == main)
CUDA_VISIBLE_DEVICES=0 C4_BP_SAVE_DUMP=0 C4_ENT_SP_BYTE1_FF_H1_HARDEN=0 \
  C4_POST_ENT_SE_SUPPRESS=0 C4_PSH_ARG_VAL_AX=0 python tools/probe_model_param_hash.py
```
