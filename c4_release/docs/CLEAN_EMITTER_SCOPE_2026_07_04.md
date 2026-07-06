# CLEAN EMITTER — scope + first increment (the ≠STEP_TOKENS framing megaroot)

**Status:** LANDED (multi-session megaroot-fix). Increments 0-1 shipped: the
`C4_CLEAN_EMITTER` op is now **DEFAULT ON** (PROVEN net-positive on the full
1096: 516 → 525, +9, full_trace spec_k=0 cap-600) and the two subsumed point-fix
correctors (`no_stack0_se_output_clear` + `no_stack0_mem_marker_output_clear`,
plus the `C4_MEM_MARKER_OUTPUT_CLEAR` kill-switch) are DELETED. Feeds
`docs/semantic_spec_EMIT_FRAMING.md` §G5.
**Golden gate:** the bare-env `tools/_isa_golden_hash.py` golden moved from
`91f55411` (pre-flip) to
`b1dcae630381bbe93ece7a53efbeadf4a6fefef0227db8fa17fba81d76ad53f5` (`b1dcae63`)
— an INTENDED verdict-change (because `C4_NO_STACK0_EMIT` defaults ON, the bare
env now bakes the emitter). On the pre-deletion tree `C4_CLEAN_EMITTER=0`
reproduced `91f55411`; that escape hatch is retired now the correctors are gone.
The narrative below is the ORIGINAL increment-0 scope (default-OFF pilot); the
"default OFF" language is historical.

This is the SURVEY-R1 biggest verdict lever (~760 programs: var / expr / if /
bool / nested / loops / func). It documents the EXACT drift mechanism from the
authoritative autoregressive (AR) oracle, then builds the FIRST flag-gated
increment of the principled fix: a CLEAN generic R-FRAME marker-dominance
emitter that makes the `NEXT_*` marker win its row UNCONDITIONALLY, so the drift
class is impossible by construction (and the per-marker-row point-fix correctors
become deletable — the biggest LOC lever in the EMIT/FRAMING family).

---

## 1. The drift mechanism (from the AR oracle + the head-bake algebra)

### 1.a What a "step" is, and where the fixed-stride decode breaks

Every executed VM step emits exactly `Token.STEP_TOKENS` tokens (30 campaign /
35 golden) — a fixed sequence of register blocks
(`neural_vm/token_layout.py`; `docs/semantic_spec_EMIT_FRAMING.md` §2.a):

```
30-token step = [ PC_mark, PC_b0..b3,  AX_mark, AX_b0..b3,  SP_mark, SP_b0..b3,
                  BP_mark, BP_b0..b3,  MEM_mark, MEM_addr_b0..b3, MEM_val_b0..b3,
                  STEP_END ]
    offsets      0  1..4     5  6..9    10 11..14  15 16..19    20  21..24  25..28   29
```

The production decode is a FIXED-`STEP_TOKENS`-stride slice
(`neural_vm/batched_pure_neural.py` `_UNSAFE_OFFSETS` + `_step_one`): it reads
PC from offset 0, AX from offset 5, etc. of each 30-token window. The MEM
addr/val bytes (offsets 21..28) are teacher-forced from the DraftVM oracle
(`_UNSAFE_OFFSETS = range(21,29)`); every OTHER offset — the 6 marker rows
(0,5,10,15,20,29) and the register VALUE bytes — is model-DECODED.

If a step emits `31` tokens (a stray byte before a marker) or `29` (a marker
where a value byte was due), the window slips: the NEXT step's PC marker lands
at the wrong absolute offset, and `got_pc` / `got_ax` are read from garbage for
every remaining step → frame collapse. That is the ≠STEP_TOKENS drift.

### 1.b Why a stray byte wins a marker row (the head-bake algebra)

The marker chain is opcode-agnostic: L0 `phase_a_ffn` (`l0_ops.py:93`) fires
exactly ONE `NEXT_<REG>` flag at each marker-PREDICTING row (the row one past the
previous block's value bytes):

| marker row (predicts)   | offset | `NEXT_*` fired at that row | L0 transition source        |
|-------------------------|-------:|----------------------------|------------------------------|
| PC marker (next step)   |     29 | `NEXT_PC`  (via `MARK_SE_ONLY` / `H0+SE`) | STEP_END → next PC |
| AX marker               |      4 | `NEXT_AX`                  | PC-byte3 row (`H1+PC ∧ ¬H0+PC`) |
| SP marker               |      9 | `NEXT_SP`                  | AX-byte3 row                 |
| BP marker               |     14 | `NEXT_BP`                  | SP-byte3 row                 |
| MEM marker              |     19 | `NEXT_MEM`                 | BP-byte3 row (`H1+BP`)       |
| STEP_END                |     28 | `NEXT_SE`                  | MEM-done row (`H3+MEM`)      |

The LM head (`model_ops._head_bake_rules`, model_ops.py:2071) is DESIGNED to make
the marker win its row:

```
head.weight[marker_tok, NEXT_<REG>] = +20 ,  head.bias[marker_tok] = -10   # marker
for every byte b:  head.weight[b, OUTPUT_LO+(b&0xF)] = +5
                   head.weight[b, OUTPUT_HI+(b>>4)]  = +5
                   head.weight[b, NEXT_<REG>]        = -80   # byte can't win a marker row
                   head.bias[b] = -5   (byte 0 -> -4)
```

So on a marker row, with `NEXT_<REG> ≈ 1.0`, the marker logit is `+20 - 10 = +10`
and every byte carries `-80` from the flag — the marker wins, IF OUTPUT is a
normal byte magnitude (≤255 → byte logit ≤ ~1275).

**The misfire:** OUTPUT at a marker row is SEMANTICALLY DEAD (the only token the
head emits there is the marker), so nothing clears it — and the L20 (block 34)
tail FFN + the L25 (block 41) tail corrector SPRAY the just-computed
ALU/store/frame result into `OUTPUT_LO/HI` at ~`1e14..3e20` on exactly the
marker rows (measured, l0_ops.py:552-561 and :750-761). Then

```
byte_logit(row) = 5*OUTPUT_LO[b&0xF] + 5*OUTPUT_HI[b>>4] - 80*NEXT_<REG> - 5
                ≈ 5 * (~1e15)  =  ~5e15    >>    +10 (marker)
```

so a STRAY value byte wins the marker row instead of the marker. That inserts /
drops a token and desyncs the fixed-stride decode. This is `docs/semantic_spec_
EMIT_FRAMING.md` §G5 spelled out with the algebra: the frame MODEL is correct
(30/35), the R-BYTE decode LEAKS a stray byte at a marker row because the
`-80*NEXT_*` suppression is out-voted by unbounded OUTPUT garbage.

### 1.c The dims involved

* **Trigger flag (per row):** one of `NEXT_PC` / `NEXT_AX` / `NEXT_SP` /
  `NEXT_BP` / `NEXT_MEM` / `NEXT_SE` (registry `dim_registry.py:678-692`), plus
  the bounded STEP_END one-hot `MARK_SE_ONLY` (`dim_registry.py:586`). Exactly
  one is lit per marker-predicting row; all are ~0 on value-byte rows and on the
  teacher-forced MEM addr/val rows.
* **The corrupted bus:** `OUTPUT_LO+[0..15]` / `OUTPUT_HI+[0..15]` (the canonical
  32 nibble one-hots the LM head reads). The garbage is written by the L20/L25
  tail (block 34 / block 41 = `tail_bit32_result_correction`, l10_ops.py).
* **The head columns that decide the race:** `head.weight[byte, OUTPUT_LO/HI+k]
  = +5` vs `head.weight[byte, NEXT_*] = -80` vs `head.weight[marker, NEXT_*] =
  +20` (model_ops.py:2071-2190).

### 1.d AR-oracle confirmation (`tools/_probe_clean_emitter_drift.py`, campaign, CPU)

The AR oracle is `FaithfulAutoregressiveRunner._run_fail_fast` (opaque_skipped ==
[]), the byte-exact CPU autoregressive decode that REPRODUCES the 30/31 miscount
(it feeds emitted tokens back as the next step's input — the fixed-stride
cumulative desync the re-anchoring `interp_oracle_gate` cannot see). The probe
re-segments the emitted tape by REG_PC markers and flags any step whose emitted
length ≠ STEP_TOKENS.

**`if_gt_0` (id 350), campaign config, both existing correctors ON (default):**

```
seg0 (len=30)  <REG_PC> 10.. <REG_AX> 35.. <REG_SP> 0,0,1,0 <REG_BP> 0,0,1,0 <MEM> ..  STEP_END
seg1 (len=30)  <REG_PC> 18.. <REG_AX> 35.. <REG_SP> 248,255,8,0 <REG_BP> .. <MEM> 224.. STEP_END
seg2 (len=30)  <REG_PC> 26.. <REG_AX> 43.. ...                                          STEP_END
seg3 (len=30)  <REG_PC> 34.. <REG_AX> 0,0,0,0 ...   <-- AX WRONG (downstream value bug)
```

Every segment is `len=30` — the FRAME is clean. This is the crux: **the two
already-landed point-fix correctors (`no_stack0_se_output_clear` on the
STEP_END/PC row, `no_stack0_mem_marker_output_clear` on the MEM row) are what
hold the frame at 30 for the PC/MEM/SE rows.** They are DEFAULT-ON in the
campaign config; with them the drift at those three rows does not occur. (The
`if_gt` AX=0 at seg3 is a separate DOWNSTREAM value bug, not framing — the frame
is intact.)

**Complementary evidence — same cluster with the MEM-row corrector OFF**
(`C4_MEM_MARKER_OUTPUT_CLEAR=0`, `var_update_0` id 325, campaign, max-steps 6):

```
seg0..seg7 (len=30)  frames but VALUES corrupted: AX = 232,255 (=0xFFE8 garbage)
                     at seg2/3/6/7 where a clean value was due (0xFF-leak family)
seg8 (len=23)  <<< emits 23 != 30   frame LENGTH breaks
```

Turning the corrector ON removes exactly this: the store step's MEM-marker row
then wins the marker (the `if_gt` `len=30` dump above), the register values are
clean, and the frame holds. The in-code root cause (l0_ops.py:737-761, spec_k=0
GPU) gives the algebra for the OFF case: at the MEM-marker row (off19) the model
emits a STRAY ALU-result byte instead of the MEM marker —
`head_logit(byte 57) = 5*OUTPUT_LO[9] + 5*OUTPUT_HI[3] - 80*NEXT_MEM ≈ +8e15
>> +17.4 (MEM marker)` — so the `NEXT_MEM → NEXT_SE → NEXT_PC` chain never fires
and the frame desyncs. This is the direct proof that the correctors are per-row
SUPPRESSORS of ONE shared class — the ≠STEP_TOKENS drift of §G5 — differing only
in WHICH `NEXT_*` row they gate. (Reproduce:
`tools/_probe_clean_emitter_drift.py --ids 325 --campaign --max-steps 6`, with
and without `C4_MEM_MARKER_OUTPUT_CLEAR=0`.)

---

## 2. Why the current fix is a point-fix FAMILY (the LOC lever)

The two landed correctors (`l0_ops.py`) each sink OUTPUT hugely negative at ONE
marker row so the `-80*NEXT_*` suppression is restored and the marker wins:

| op                                  | gate (row)         | rules | sink   |
|-------------------------------------|--------------------|------:|--------|
| `no_stack0_se_output_clear`         | `MARK_SE_ONLY` (STEP_END/PC row) | 32 | `-1e20` |
| `no_stack0_mem_marker_output_clear` | `NEXT_MEM` (MEM row) | 32 | `-1e20` |

They cover only **2 of the 6 marker rows**. The AX / SP / BP marker rows
(`NEXT_AX` / `NEXT_SP` / `NEXT_BP`) have NO such corrector — so any program whose
ALU/frame spray lands on an AX/SP/BP marker-predicting row STILL drifts, and the
"next" natural fix is to add THREE MORE copies of the same 32-rule point-fix
(~140 LOC each). That is the point-fix family the megaroot spawns.

**The clean generic emitter subsumes the whole family.** One op gated on the
DISJUNCTION `(NEXT_PC ∨ NEXT_AX ∨ NEXT_SP ∨ NEXT_BP ∨ NEXT_MEM ∨ NEXT_SE)` sinks
OUTPUT at EVERY marker-predicting row, so the marker wins its row by
construction and the ≠STEP_TOKENS class cannot occur for ANY register. This is
the R-FRAME marker-dominance invariant (`semantic_spec_EMIT_FRAMING.md` §2.b):

```
INVARIANT (marker-dominance):  on any row where some NEXT_<REG> is lit,
    OUTPUT_LO/HI are driven far below the marker baseline
    => head argmax = REG_<REG>  (never a byte)   =>  the frame is exactly N tokens.
```

### Corrector families this megaroot-fix would DELETE (LOC estimate)

| family (all `l0_ops.py`, all `C4_NO_STACK0_EMIT`-gated) | rules | ~LOC |
|----------------------------------------------------------|------:|-----:|
| `no_stack0_se_output_clear` (STEP_END/PC row)            |    32 | ~140 |
| `no_stack0_mem_marker_output_clear` (MEM row)            |    32 | ~140 |
| the 3 NOT-yet-written AX/SP/BP-row point-fixes it PRE-EMPTS | 3×32 | ~420 |
| **subtotal (marker-dominance point-fixes)**              | **160** | **~700** |

The generic emitter replaces all of that with ONE ~34-rule op (32 nibble sinks
gated on the 6-way NEXT_* OR). The `no_stack0_pc_highbyte_clear` family (94
rules, ~240 LOC) is a DIFFERENT concern (a VALUE-cleanup of the PC high bytes /
0x01-leak, not marker-dominance) and is NOT subsumed by this increment — it is
noted here only to bound the scope: CLEAN_EMITTER deletes the marker-dominance
point-fixes, not the value-cleanup ones. Verdict-neutral geometry (each point
fix is behaviorally covered by the generic op), landed behind a flag so the
golden count is preserved until the collapse is proven.

---

## 3. The first increment — `C4_CLEAN_EMITTER` (default OFF)

**What it is (increment 0):** a standalone `PureFFN` post_op on the L25 tail
block (appended AFTER `tail_bit32_result_correction`, so it is the LAST writer of
OUTPUT before the LM head — same slot as the existing correctors), that sinks
`OUTPUT_LO/HI[0..15]` hugely negative on EVERY marker-predicting row, gated on
the 6-way OR of the `NEXT_*` marker-schedule flags. 32 AND-rules (one per OUTPUT
nibble), each firing when ANY `NEXT_*` is lit. This is the marker-dominance
invariant of §2 realized as one op — the same mechanism as the two existing
point-fixes but covering all six rows at once.

**Gating (byte-identity):**
* `C4_CLEAN_EMITTER` DEFAULT OFF (env-unset). The op registers as a NO-OP
  Operation when off (the exact pattern the existing correctors use for their
  flag-off path), so the op list / dep graph is stable but no weights change.
* Because `tools/_isa_golden_hash.py` runs in a BARE env (`C4_CLEAN_EMITTER`
  unset → OFF), the golden `91f55411` is byte-identical with this op registered.
* When ON (`C4_CLEAN_EMITTER=1`) it additionally requires `_no_stack0_emit()`
  (the 30-token frame); in the 35-token golden the marker rows have STACK0-block
  decode slack that absorbs a +1 drift, so the emitter is a 30-token-frame op.

**Why increment 0 is the right first step:** the generic emitter must (a) prove
the 6-way NEXT_* OR gate fires cleanly at every marker row and (b) prove sinking
OUTPUT there is verdict-neutral (never sinks a real value-byte row, since NEXT_*
is 0 on value rows). Increment 0 lands the op flag-OFF (byte-identical golden)
with the gate + sink in place; the FOLLOW-UP increments then (1) turn it ON in
the campaign config and A/B it against the two existing correctors via
`cpu_full_trace` + `flag_regression_gate`, and (2) once ON-parity is proven,
DELETE the two point-fixes (and never write the AX/SP/BP ones), collapsing the
~700 LOC of §2.

---

## 4. Verification performed
* Golden baseline: `tools/_isa_golden_hash.py` (CPU, bare env) =
  `91f5541100d9a9ec7081105a6bde83f9de5c65cc62317a62bf108bdc93aa56c1` (`91f55411`);
  confirmed the bare-env golden is the 30-token campaign build (`_no_stack0_emit()
  == True`, `_mem_marker_output_clear_enabled() == True`).
* AR oracle (`tools/_probe_clean_emitter_drift.py --ids 350 --campaign
  --max-steps 6`, CPU, spec_k=0): `if_gt_0` frame is clean `len=30` at every
  segment with the correctors ON — the frame-suppression is real and per-row.
* AR oracle, MEM-row corrector OFF (`C4_MEM_MARKER_OUTPUT_CLEAR=0`, `var_update_0`
  id 325): frame VALUES corrupt (AX=0xFFE8) at seg2/3/6/7 and frame length breaks
  (seg8 `len=23`) — the SAME class the corrector suppresses when ON.
* Head-bake algebra read from `model_ops._head_bake_rules` (model_ops.py:2071):
  `+20` marker vs `-80*NEXT_*` byte vs `+5*OUTPUT_*` byte — the exact race the
  sink restores.
* L0 marker chain read from `l0_ops.py:93-160` (the 6 `NEXT_*` transitions) and
  the two existing correctors' root-cause comments (l0_ops.py:543-934).

## 5. Feeds
`docs/semantic_spec_EMIT_FRAMING.md` §G5 (drift is LOWERING misfire) + §2.b
(R-FRAME marker-dominance). This is the FRAMING half of generic-lowering pilot
#391. Next: increment 1 (campaign-ON A/B via `cpu_full_trace` +
`flag_regression_gate`), then increment 2 (collapse the point-fixes).
