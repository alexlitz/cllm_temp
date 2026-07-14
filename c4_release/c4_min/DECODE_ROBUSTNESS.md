# c4_min — Framing-Robust Emission + Decode (design study)

**Status:** DESIGN INPUT for the green-field substrate. This document is consumed
by `DESIGN.md` (owned by the `greenfield-substrate` agent); where the two differ,
**`DESIGN.md` is authoritative**. This file does NOT edit DESIGN.md or any
compiler code — it is the *why* behind the emission/decode contract §(c) that
DESIGN.md already sketches, plus the alternatives that were considered and the
recommendation.

**Scope.** The single biggest verdict-flipping failure mode of the OLD 48k model
is the **FRAMING DESYNC**: a fixed-stride emission+decode contract where any
single value-byte failure changes the token *count* of a step, which desyncs the
fixed-stride slicer for that step **and every step after it** (whole-trace
cascade). The green-field must be *correct-by-construction* on this — a value
error must stay a **local, single-step, single-register** error and must never
move the decode frame. This doc (1) dissects the old mechanism precisely, (2)
proposes four framing-robust-by-construction contracts, and (3) recommends one.

---

## 1. ROOT — why the old fixed-stride contract desyncs

### 1.1 The contract

The old model emits a **fixed stride of `Token.STEP_TOKENS` tokens per VM step**
(35 golden / 30 campaign) and the verdict decodes each step by **fixed byte
offset** inside that stride. Authoritative sources:

- **Layout** (`neural_vm/token_layout.py`): each step is a fixed sequence of
  *register blocks*, each block = `[marker, b0, b1, b2, b3]` (little-endian
  32-bit), in the order PC, AX, SP, BP, (STACK0 only in 35-token), then a MEM
  block `[MEM, addr0..3, val0..3]`, then a single `STEP_END`/`HALT`:

  ```
  35-token: [PC m,b0..3][AX m,b0..3][SP m,b0..3][BP m,b0..3][STACK0 m,b0..3]
            [MEM m, addr0..3, val0..3][STEP_END]
  offsets:   0 1..4      5 6..9      10 11..14  15 16..19   20 21..24
            25 26..29 30..33          34
  ```

- **The decode is fixed-offset** (`batched_pure_neural.py`):
  - `_ff_check_new_steps` (the verdict, full_trace): `completed = token_pos //
    STEP; start = prefix_len + step_idx*STEP; step_tokens = context[start:start+STEP]`
    — it literally chops the stream into `STEP`-sized slices by arithmetic, then
    `_decode_step_register(step_tokens, REG_PC/REG_AX)` marker-scans **inside that
    fixed slice** for the 4 value bytes after the marker.
  - `_step_one` only advances the VM (`_dispatch_pure_neural`) when the emitted
    token is `STEP_END`/`HALT`. So the *state machine* is delimited by `STEP_END`,
    but the *verifier slicer* is delimited by fixed arithmetic (`step_idx*STEP`).
    **These two delimiters disagree the instant a step's token count ≠ STEP.**

- **How a token wins its row** (from `docs/semantic_spec_EMIT_FRAMING.md`,
  `model_ops._head_bake_rules`): the LM head gives each row two ways to argmax:
  - a **marker** row: whichever `NEXT_<REG>` schedule flag the L0 frame
    state-machine (`phase_a_ffn`) lit → the `REG_<REG>` token wins (`+20` vs every
    byte's `-80`);
  - a **value-byte** row: the byte whose low/high nibble one-hots are lit in
    `OUTPUT_LO/OUTPUT_HI` wins (`+10` vs `-5` bias).

  The frame length is therefore *emergent*, produced by the L0 marker
  state-machine racing the value-byte decoder at each position — it is **not a
  hard structural constant**. That is the whole vulnerability.

### 1.2 The exact desync mechanism (the marker/count trigger)

A step emits exactly `STEP` tokens **only if, at every position, the intended
token wins its row**: markers win the 5 marker positions, value bytes win the 20
value positions. The count breaks in two directions, both observed and localized
in the campaign docs:

**(A) OVER-emit → step is LONG (+k tokens).**
`docs/FUNC_LEV_IS_LI_FROM_FRAME_37TOKEN_DESYNC_2026_06_14.md` and EMIT_FRAMING
§G5: a value byte at a marker position fails to *yield* to the next marker. The
L0 STEP_END-row stray-byte suppression (`−80·NEXT_PC`) is out-voted by a
sentinel-magnitude value logit (a `0xFF` emitter fires at **logit 4.4e10**), so a
spurious byte token is emitted **before** the next marker. The post-ENT step emits
**37 tokens** (two leading stray `0xFF`) instead of 35. Per-step counts for
`func_identity_0`: `[35,35,37,35,35,35,37,35,35]` — the two 37s are exactly the
instructions right after an `ENT`.

**(B) UNDER-emit → step is SHORT (−k tokens).**
`docs/POST_ENT_DESYNC_BYTE3_SHORT_STEP_2026_06_16.md`: the inverse. A value byte's
row gets **crushed** (its `OUTPUT_LO/HI` nibble one-hots are driven strongly
negative by a relayed darkening — e.g. a `NEXT_SP` OUTPUT-darkening at −218
relayed ×3 = −654 onto the STACK0 byte-3 row), so **the marker token wins a row
that should have been a value byte**. The value byte is dropped and the next
marker fires one position early → the step emits **34 tokens**. Same cluster,
opposite sign, same fixed-slicer break.

**The trigger, stated once:** *a per-register value byte and a marker share the
same argmax competition at every row; whenever a value byte's decode is
mis-magnituded (either strong-enough to beat a marker suppression, or weak-enough
to lose to a marker), the boundary between "value" and "marker" moves by ±1
position, changing the step's token count.*

### 1.3 Why one byte cascades to a whole-trace failure

This is the fatal amplification, and it is purely a property of the fixed-stride
*decode* (not of the model):

1. Step `t` emits `STEP ± k` tokens instead of `STEP`.
2. The verifier computes `start = prefix_len + t*STEP` — but the actual step-`t`
   payload no longer occupies `[t*STEP, (t+1)*STEP)`. **All bytes from the (t)th
   boundary onward are shifted by `∓k`.**
3. `_decode_step_register(step_tokens, REG_PC)` now marker-scans a **mis-aligned
   window** that straddles two real steps. It finds a `REG_PC` marker at the
   wrong place (or `+4 ≥ len` and returns `None`), so `got pc = None` or a
   garbage value — reported at a step whose *value was actually correct*.
4. Every subsequent step inherits the same `∓k` offset. Because the slicer index
   is `t*STEP` (absolute arithmetic), the misalignment **never re-syncs** — one
   bad byte at step 2 fails steps 2..N.

`_extract_register` (the dispatch-time reader) has the same disease at smaller
radius: it scans back only `STEP + 5` tokens, so a short/long step makes it read
the *previous* step's marker (`_decode_step_register`'s docstring even calls out
"a scan-back can return a stale prior-step value").

**Net:** the model may have produced the correct AX for every step, but a single
mis-magnituded value byte at step 2 desyncs the *frame* and the verdict reads the
wrong PC/AX for steps 2..N. This is the megaroot the memory notes describe as
"the comparison step emits 57 tokens not 35 … fixed-35 slicer misreads PC for
that step AND all following (whole-step desync)" and "func/nested/var diverge at
the first frame-local LI because the post-ENT step is 37 tokens."

### 1.4 The three structural sins (what the green-field must NOT inherit)

1. **Frame length is emergent, not structural.** It is produced by a marker
   state-machine (`NEXT_*`) racing a value decoder at every position. A value
   error can move a marker boundary.
2. **Value and delimiter share one argmax channel.** Markers and value bytes
   compete on the *same* rows via the *same* head; magnitude bugs convert one to
   the other.
3. **Decode position is derived by cumulative arithmetic (`t*STEP`), with no
   re-sync.** Any local count error is integrated forward forever.

A robust contract must break **at least sin #3** (make decode position
independent of prior steps' realized lengths) and preferably #1 and #2 (make the
frame length structural and the delimiter channel-disjoint from values).

---

## 2. ALTERNATIVES — framing-robust-by-construction contracts

Each is evaluated on: **decode**, **locality** (why a byte failure stays local),
**compile-time cost**, and **fit to the minimal 8-bit transformer** (ALiBi
additive-mask attention, bare additive residual, SwiGLU FFN, softmax attention —
per `c4_min/model.py`; scalar-per-register residual per `DESIGN.md` §b).

Throughout, note the green-field's decisive structural advantages over the old
model that make robustness *cheap*:

- **8-bit only** → every register value is **one token**, not four
  little-endian bytes. There are no per-byte rows to individually mis-magnitude,
  so the entire byte-level over/under-emit family (§1.2) *cannot even be
  expressed*.
- **Scalar-per-register residual** → a register lives in one residual dim; decode
  can read a *fixed residual position*, not a scanned token slice.
- **"Step ruler" input** (`DESIGN.md` §c/§e): the model is run over a
  fixed-length positional input where position `t` = VM step `t`, with state
  carried in the residual by a carry-forward attention head. **The emitted token
  stream is an *observation*, never fed back to reconstruct state.** This is the
  single most important robustness lever: decode cannot desync execution because
  decode is downstream of a positionally-fixed compute, not an autoregressive
  re-parse.

### Alternative A — One token per step (position-indexed, self-synchronizing)

**Contract.** Emit exactly **one token per VM step**: `result[t] =
argmax(logits[t])` = the low byte of the step's `OUTPUT` (= AX), from vocabulary
`{0..255} ∪ {HALT=256}`. Step `t`'s output is at sequence position `t`, full
stop. (This is DESIGN.md §c's committed contract; included here as the baseline
because it is *maximally* robust and the others are only interesting if they buy
something it lacks.)

- **Decode.** `for t: out[t] = argmax(logits[t])`. Stride = 1. There is no slice,
  no marker scan, no `t*STEP` arithmetic. PC/AX/SP/BP are read from the residual
  at position `t` if the verdict wants them (position-addressed, see below), or
  the single emitted token is AX and that is the whole verdict for value-trace
  criteria.
- **Locality.** A wrong AX at step `t` corrupts **exactly `out[t]`** and nothing
  else: position `t+1`'s logits are a function of the *residual* at `t+1` (carried
  forward by attention over the true VM state), **not** of the token emitted at
  `t`. One byte failure = one wrong entry. **Zero cross-step cascade by
  construction** — there is no count to change and no cumulative index to drift.
- **Compile cost.** *Negative* vs the old model: no L0 marker state-machine, no
  `NEXT_*` schedule, no per-byte nibble banks, no STEP_END emitter, no tail
  correction bank. One `emit` FFN (`OUTPUT ← AX`) + a 257-row LM head
  (`byte → logit`). Roughly the `emit FFN` + `lm_head()` lines already in
  DESIGN.md §e.
- **Transformer fit.** Ideal. ALiBi/softmax carry-forward head reads register
  bands from position `t-1`; a single SwiGLU emit rule copies `AX → OUTPUT`; the
  LM head is a 257×D readout of the `OUTPUT` scalar (thermometer/one-hot of the
  byte value). No multi-token framing to schedule.

**Cost / limitation.** The emitted stream shows **only AX**. If the *verdict*
needs PC/SP/BP per step (the old `full_trace` compares `(PC, AX)`), one token per
step does not carry it in the token stream. Two clean resolutions, both robust:
(i) the verdict reads PC/SP/BP by **position-addressed residual probe** (read the
`PC`/`SP`/`BP` scalar dim at sequence position `t` — a fixed dim at a fixed
position, no scan; this is exactly Alternative D applied to the verifier only);
or (ii) if a *token-stream* PC trace is required, use Alternative B. In the
green-field the oracle harness (`c4_min/oracle.py`) already compares against a
reference interpreter, so a residual probe for PC is the natural, robust fit.

### Alternative B — Fixed small tuple per step (position-indexed, structural stride)

**Contract.** Emit a **fixed, tiny** tuple per step — e.g. `[PC, AX]` or
`[PC, AX, SP, BP]` — **one token per register** (8-bit → one token each), with
`STRIDE = len(tuple)` a hard structural constant. Crucially, **no markers and no
value/marker sharing**: position `t*STRIDE + r` *is* register `r` of step `t`,
addressed by absolute position, and each register slot is emitted by a
*dedicated, position-gated* emit rule.

- **Decode.** `reg[r] at step t = argmax(logits[t*STRIDE + r])`. Fixed stride,
  but — unlike the old model — the stride is **enforced structurally**: register
  `r`'s slot is written by a rule gated on `POS % STRIDE == r` (a positional
  window on the `POS` band), so exactly `STRIDE` tokens are emitted per step **by
  construction of the position gate**, regardless of any value.
- **Locality.** A wrong value at slot `(t, r)` corrupts **only that slot**. It
  **cannot change the count**, because the count is not a race between markers and
  values — it is `STRIDE` positions, each owned by a positional gate that fires
  independently of value magnitude. There is no `NEXT_*` boundary to move, so the
  §1.2 trigger is structurally absent. A byte error stays a single-slot error;
  the next slot's logits depend on `POS`, not on the emitted token.
- **Compile cost.** `STRIDE` emit rules (one per register), each a positional-gated
  `OUTPUT ← <reg>` SwiGLU rule + a shared LM head. No marker state-machine, no
  STEP_END. Slightly more than A (STRIDE emit rules vs 1), still a fraction of the
  old L0+tail machinery.
- **Transformer fit.** Good. The `POS` band (already in `layout.py`) drives a
  positional gate `silu(sharp·(POS − expected_slot))`; ALiBi is not even needed
  for the emit gate (it is content/position on the same token). The LM head reads
  whichever register the active slot selected into `OUTPUT`.

**Why this is *not* the old model:** the old stride was 35 with **markers inside
the stride** and value bytes competing with markers on shared rows, and the
decoder scanned for markers *inside a slice* — so a value could impersonate a
marker and the slice could straddle steps. Here the stride is small, **markerless**
(position *is* the index), each slot is a **whole 8-bit register in one token**,
and the emit gate is **position-driven, not value-driven**. The count cannot move.

**Cost / limitation.** The decode still uses `t*STRIDE` arithmetic, so it is only
robust *because* the position gate guarantees exactly `STRIDE` emissions. If a
future extension re-introduced a value-dependent emission (e.g. variable-length
MEM), the guarantee would weaken — keep every slot fixed-width (the 8-bit
constraint makes this free).

### Alternative C — Self-delimiting / length-tagged step (separator + re-sync)

**Contract.** Emit a variable payload per step terminated by a **reserved
separator token `SEP`** that **no value token can ever equal** (vocabulary
partition: values `0..255`, `SEP=256`, `HALT=257`). Decode splits the stream on
`SEP` and re-syncs at each separator. Optionally length-prefix each step
(`[N, tok_0..tok_{N-1}]`).

- **Decode.** `steps = split(stream, SEP)`. `step[t]` = the tokens between the
  `t`-th and `(t+1)`-th `SEP`. The verdict reads registers positionally *within*
  a split group.
- **Locality — PARTIAL, and this is the key finding.** A separator makes the frame
  **self-re-synchronizing**: even if step `t` emits the wrong *number* of value
  tokens, decode re-anchors at the next `SEP`, so the damage is bounded to step
  `t` and does **not** cascade to `t+1..N` (unlike §1.3). **BUT** locality holds
  **only if the separator channel is disjoint from the value channel** — i.e. a
  value byte can never be mis-magnituded into a `SEP`, and a `SEP` can never be
  crushed into a value. This is precisely the sin (§1.4 #2) that broke the old
  model: there, the "separator" (STEP_END/marker) shared the argmax channel with
  values, so a value became a marker (short step) or a marker was out-voted (long
  step). A robust separator contract **must** reserve a dedicated residual band
  (`IS_SEP`) and a dedicated head-row so the separator wins its position by a
  structural margin independent of any value magnitude.
- **Compile cost.** A separator-emit rule + a re-sync-aware decoder. Cheaper than
  the old marker state-machine (one `SEP` vs six `NEXT_*` edges) but strictly more
  than A/B (needs the variable-length split logic and the disjoint-channel
  guarantee).
- **Transformer fit.** Workable: reserve `IS_SEP` band; an emit rule sets
  `IS_SEP` at the last slot of a step; the LM head gives `SEP` a `+BIG·IS_SEP`
  column and `−BIG` on all value columns (and vice-versa), so the channels are
  argmax-disjoint. ALiBi carry-forward unaffected.

**Cost / limitation.** Self-delimiting *tolerates* a count error rather than
*preventing* it — the step whose count is wrong still decodes wrong (its
registers are mis-positioned within its own group), you just don't cascade. It
also re-introduces a *variable* frame length, which is exactly the surface the
old model failed on; the only reason it is safe here is the disjoint channel +
8-bit single-token registers. Strictly dominated by B for the fixed-register
case (B prevents the count error entirely at the same channel-disjointness cost).

### Alternative D — Position-addressed residual decode (no token framing at all)

**Contract.** Do not decode registers from the **token stream** at all. Run the
"step ruler" (positions `0..T`), and read each register `R` of step `t`
**directly from its dedicated residual dim at sequence position `t`**:
`R[t] = quantize(residual[t, dim_R])`. The token stream (Alt A's one-token AX)
remains as the human-facing output, but the *verdict* reads the residual.

- **Decode.** For the verifier: `for t: (PC,AX,SP,BP)[t] = round(final_residual[t,
  [PC,AX,SP,BP]])`. A fixed dim at a fixed position. No slicing, no markers, no
  arithmetic over prior steps.
- **Locality.** Total. Register `R` at step `t` is a single scalar at
  `(t, dim_R)`; it depends only on step `t`'s compute (carried forward from `t-1`
  by attention). A wrong value at `(t, R)` is exactly one wrong scalar. There is
  no count, no delimiter, no slice — **the three structural sins of §1.4 are all
  absent**: the decode "position" is the sequence position itself (given by the
  ruler input), which is fixed regardless of any value.
- **Compile cost.** ~Zero *emission* cost (no emit framing needed for the verdict
  path). The cost moves to the verifier: it must read the model's *residual*
  (an activation probe), which the green-field oracle harness already does at the
  build level. Requires the registers to live in stable, named dims across the
  whole forward (they do — `layout.py` gives each register a fixed scalar dim).
- **Transformer fit.** Native to the scalar-per-register design: the registers
  already occupy fixed dims (`AX/SP/BP/PC` in `layout.py`); the carry-forward head
  already keeps them live at every position. Reading them is a `state_dict`-free
  activation read.

**Cost / limitation.** The residual value is a **float**; decoding it to an exact
8-bit integer needs a stable quantization (round-to-nearest, or a one-hot/
thermometer band per register so the read is an argmax, not a fragile float
compare). The green-field's exact-integer SwiGLU identity (`silu(S)·(v/S)` gives
exact `+v`, DESIGN.md §d) makes the scalar exact to fp32, so `round()` is safe;
using a one-hot register band would make it argmax-exact at the cost of `256·k`
dims per probed register. Also: a residual probe is a *verifier* mechanism — for a
deployed model that must *emit* an answer sequence, you still want Alt A's
one-token stream as the observable. D and A compose: **A for the observable
stream, D for the verdict.**

---

## 3. RECOMMENDATION

**Adopt Alternative A (one token per step) as the emitted contract, with
Alternative D (position-addressed residual read) as the verdict-side decode for
any register beyond AX.** This is exactly what `DESIGN.md` §c already commits to
for the stream; this study endorses it and adds the D verdict path so the
green-field can reproduce the old `full_trace` `(PC, AX)` comparison *without*
re-introducing multi-token framing.

**Why A+D over B and C:**

- **A+D removes all three structural sins (§1.4) at once.** There is no emergent
  frame length (each step is exactly one observable token *and* one residual
  column), no value/delimiter channel sharing (the token is a value; the verdict
  reads a fixed dim), and no cumulative decode index (position `t` is given by the
  ruler input, not integrated from prior step lengths). The §1.2 count-trigger
  **cannot be expressed** because 8-bit registers are one token each and the
  verdict never slices.

- **B is the right fallback if a *token-stream* multi-register trace is a hard
  requirement.** B is genuinely framing-robust (markerless, position-gated,
  fixed small stride, whole-register-per-token) and strictly better than the old
  contract. But it costs `STRIDE` emit rules and re-introduces `t*STRIDE`
  arithmetic whose safety *depends on* the position gate holding — a slightly
  larger correctness surface than A+D's "read the dim." Choose B only if the
  deployment consumer parses a token stream (not a residual) and needs PC/SP/BP in
  that stream.

- **C is not recommended.** Self-delimiting only *tolerates* a count error (stops
  the cascade) instead of *preventing* it, and its safety rests on the exact
  channel-disjointness property whose violation was the old model's root. It buys
  cascade-immunity that A/B/D already have structurally, while re-introducing
  variable-length steps — the very surface that failed. Keep `SEP`
  channel-disjointness as a *lint invariant* (see below), not as the primary
  contract.

**The load-bearing invariants the compiler must enforce (regardless of A/B):**

1. **Registers live in fixed named dims for the entire forward** (`layout.py`
   already guarantees this) → position-addressed read (D) and carry-forward are
   always valid.
2. **8-bit registers are exactly one token / one slot each** → no per-byte
   over/under-emit family can exist. *Do not* re-introduce little-endian
   multi-byte value blocks; if 16-bit is ever needed, make each byte its own
   *fixed* slot (B-style), never a variable run.
3. **Value channel and any delimiter channel must be argmax-disjoint** — if a
   `SEP`/marker/HALT token is ever introduced, give it a dedicated residual band
   and an LM-head column that beats every value column by a fixed margin
   (`+BIG·IS_SEP`, `−BIG` on values), so no value magnitude can flip a delimiter
   and vice-versa. Add a **compile-time lint** that asserts this margin.
4. **Decode position is the ruler position `t`, never a cumulative sum of realized
   step lengths.** The verdict indexes by `t` directly (A/D) or by
   `t*STRIDE + r` under a *position gate that guarantees the stride* (B) — never
   by scanning for the next marker in a slice.
5. **HALT is a distinct top-of-vocabulary token** (`256` under A) so "did it
   halt" is a value-disjoint signal, and "ran too long / too few steps" is decided
   by the fixed ruler length, not by counting emitted tokens.

**One-line verdict:** the green-field is framing-robust *by construction* the
moment (a) each 8-bit register is one token / one fixed residual dim, and (b) the
verdict indexes by the ruler position `t` instead of a marker-scanned
fixed-stride slice. A+D deliver both with the least machinery; the old model's
whole-trace desync **cannot occur** because there is no per-byte marker race and
no cumulative decode index to drift.

---

### Sources consulted (read-only)

- `neural_vm/batched_pure_neural.py` — `_decode_step_register` (fixed-slice marker
  scan), `_ff_check_new_steps` / `_ff_check_new_steps_strict` (the `t*STEP`
  verdict slicer), `_extract_register` (`STEP+5` scan-back), `_step_one` /
  `_dispatch_pure_neural` (STEP_END-delimited state advance), `_UNSAFE_OFFSETS`.
- `neural_vm/token_layout.py` — the 35/30-token fixed layout + offset map.
- `neural_vm/constants.py` — `STEP_TOKENS == (30 if C4_NO_STACK0_EMIT else 35)`.
- `docs/semantic_spec_EMIT_FRAMING.md` — R-FRAME (marker state-machine) + R-BYTE
  (nibble→token) primitives; §G5 "≠STEP_TOKENS drift is an emission MISFIRE …
  vanishes under a clean lowering where the marker always wins its row."
- `docs/FUNC_LEV_IS_LI_FROM_FRAME_37TOKEN_DESYNC_2026_06_14.md` — the 37-token
  OVER-emit (sentinel `0xFF` @ 4.4e10), per-step count trace, whole-trace cascade.
- `docs/POST_ENT_DESYNC_BYTE3_SHORT_STEP_2026_06_16.md` — the inverse 34-token
  SHORT step (value byte crushed −654 → marker wins its row early).
- `c4_min/DESIGN.md` §(b) scalar-per-register layout, §(c) one-token contract,
  §(e) step-ruler compile stages; `c4_min/model.py` (ALiBi/bare-residual/SwiGLU);
  `c4_min/layout.py` (fixed named register dims).
