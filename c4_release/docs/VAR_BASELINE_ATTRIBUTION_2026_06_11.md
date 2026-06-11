# var stack-store baseline — fresh attribution (2026-06-11)

Date: 2026-06-11
HEAD: `fc8a27a168b3dcb7af10c978a9ff1028c1303009` (worktree `agent-abb1fc87c752ac720`)
Path: spec_k=0, hook-free (`tools/probe_groundtruth.py`). GPU 1.
Probe tools added (read-only): `tools/probe_var_baseline.py`,
`probe_var_raw.py`, `probe_var_bp2.py`, `probe_var_block.py`,
`probe_var_src.py`, `probe_var_genesis.py`.

## TL;DR — the prior premise is WRONG

The triage doc (`docs/1096_TRIAGE_2026_06_11.md`, cluster A) and the memory
note frame the var failure as a **stack-relative MEM store / MEM-address**
bug (`int x = N` → `step0:MEM_addr1` / `SP_byte2`). **On current HEAD that is
not where it breaks.** The divergence happens at the **JSR→ENT prologue
(step 0), BEFORE any store, load, or stack-relative MEM address is ever
computed.** The stack store (SI) at step 5 is never reached with a clean
state — the frame registers are already garbage.

## Representative program

`var_simple_12` (id **262**): `int main() { int x; x = 28; return x; }`,
expected 28. Picked as the MINIMAL var case: single-byte value (28 ≤ 255, so
no 16-bit ALU confound), one stack-relative store + one load-back. Bytecode:

```
[0] JSR 3   [1] HALT  [2] NOP
[3] ENT 8   [4] LEA -8  [5] PUSH  [6] IMM 28  [7] STORE
[8] LEA -8  [9] LOAD    [10] HALT
```

spec_k=0 result: neural exit **0**, expected 28 (`run_1096_fast --ids 262
--spec-k 0`). The SAME signature holds for the 16-bit cases id 250 (x=990)
and id 271 (x=39) — so it is value-independent (not 16-bit-ALU).

## EXACT current-HEAD attribution

**First divergence (neural vs symbolic oracle, after-state):**

| field | value |
|---|---|
| step | **0** |
| opcode | **JSR** |
| register | **REG_BP** |
| byte | **byte0** (also byte1, byte3) |
| expected | `0x00` (BP=65536=`00 00 01 00` after JSR) |
| neural | **`0x0a`** (=10) |

The model floods the BP register byte slots with the value **`0x0a` = 10**,
which is exactly the **JSR-pushed return value** (oracle step0 JSR
`mem_val=10`, the return-PC index). i.e. the stack-pushed PC leaks into the
BP register emission. The flood is so strong it also displaces the `STACK0`
(268) record marker that should follow BP, desyncing the entire step framing;
by step 5 the stream is a runaway flood of `0x02` tokens.

**Logit evidence at the step-0 REG_BP byte0 prediction row** (ctx idx 107):

```
BP byte0: top = [(0x0a, 2354051.25), (0x00, 6.41), ...]
BP byte1: top = [(0x0a, 453946880.0), (0x00, 12.0), ...]
```

`0x0a` wins by ~2.35e6 logits; the correct `0x00` sits at 6.41. This is a
catastrophically over-driven emission, not a near-miss.

## Physical-block + dim pin

The LM head reads residual dims **79 and 85** (+5.0 each) to emit token
`0x0a`. Those dims live in the **H2 (74–80) / H3 (81–87) byte-decoder slots**
(`dim_registry_dynamic.py:143-144`, "L0 head N marker-within-dist" attention
outputs, the per-byte register-emit decoder band — NOT operand one-hots, NOT
MEM-address dims).

Residual magnitude in dims 69–85 across blocks (genesis cascade, at the
position that seeds each block; `probe_var_genesis.py`):

| phys | logical | global max\|val\| | seed pos / token |
|---:|---:|---:|---|
| 5 | L5 | 1.0 | — |
| **6** | **L6** | **507** | pos 92 = first REG_PC marker |
| 10 | L9 | 788 | REG_AX (step 2) |
| **11** | **L10** | **48 651** | REG_AX (step 2) — first catastrophic jump |
| 27–28 | L18–L19 | 7.6e6 | step-1 region |
| 29 | L20 | 1.3e7 | gathers the giant value into the BP-byte0 row |
| 35–36 | L25–L26 | 3.3e14 | total numerical blowup |

- **Genesis (crosses 1.0 → 500):** physical block **6 / logical L6**, seeded
  at the first `REG_PC` marker row (pos 92).
- **First catastrophic amplification (→ ~5e4):** physical block **11 /
  logical L10**.
- **Injection into the BP-byte0 prediction row:** block **29 / L20**
  attention gathers the upstream giant value into read_pos 107.
- The downstream blocks (L20→L26) just propagate/blow up an already-broken
  residual; they are carriers, not the root (consistent with the memory
  note's warning that L34/block-35 attributions were mis-attributed carriers).

## Decision questions (the brief's acceptance)

1. **Operand-gather (block 8) downstream? NO.** Block 8 is in the path but the
   seed is at **L6** and the dims are the **H2/H3 register-emit decoder slots**,
   not the operand magnitude+nibble one-hots that gate CMP/ALU. The
   operand-gather clean-one-hot fix (clusters C/D/H/I/K/L) will **not** unblock
   the var/frame baseline. This is a **distinct bug**.

2. **Stack-relative MEM-address bug (L3/L10/L34)? NO.** The store/load never
   execute with a clean state; the failure is the **JSR/ENT frame-register
   emission** corrupting BP (and cascading to SP) at step 0/1. It is NOT the
   `mem_byte_0_default` / L13-witness / SP_byte2-nibble path the prior notes
   chased. Those were stale carrier attributions.

3. **Shared baseline?** YES — identical `step0 JSR REG_BP byte0 exp=0x00
   neu=0x0a` signature on id 250/262/271. Because every func/loop/rec program
   opens with the same `JSR main … ENT` frame prologue, this single
   frame-register-emit instability is the prerequisite gate for clusters
   A/B/F/G/J (~525 programs). Flat programs with NO frame (e.g. `IMM 28; EXIT`,
   the compiled form of `int main(){return 28;}`) PASS — confirming the break
   is the frame prologue, not arithmetic or I/O.

## What a fix must target (for the future fix agent — diagnostic only here)

The fix surface is the **register-emit decoder for BP at the JSR/ENT
prologue**, specifically the H2/H3 byte-decoder band (dims 74–87) that picks
up the JSR-pushed stack value (`0x0a`) instead of the BP value. Root genesis
is **L6** (first amplification past unity at the REG_PC marker row); the first
runaway is **L10**. Do NOT fix at L20/L25/L34 (carriers) and do NOT chase
the L3 SP_byte2 / L13-witness / MEM-addr paths (not on the failing path at
this HEAD). Per `feedback_single_rule_fixes_are_zero_sum`, use
`decl_verifier.py` + block residual decomposition at L6/L10 before any rule
change. This is a multi-commit frame-prologue surface, not a single rule.
