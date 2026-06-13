# AX byte-1 DUMP carry — the binding wall is the H1-WRITE scheduler cycle, not the L13 ALiBi host (2026-06-13)

**Status:** NOT LANDED. This supersedes the *host-choice* framing in
`AX_BYTE1_DUMP_LATE_CARRY_BLOCKER_2026_06_13.md`. That doc concluded the
blocker was "L13 is a build-fragile host; find a build-safe one that reads
block-12-or-later." A build-safe host **was** found and built on
(block 13 = logical L12, an empty attention block with the model-default
geometric ALiBi — no imperative `alibi_slopes.fill_()` to perturb). It
**still does not compile**, and the reason is a *deeper* architectural
wall that no host choice can avoid:

> **A late-layer attention head that writes the `H1` band cannot be
> scheduled, because `H1` is read fresh (same-step) by 54 ops across
> L1–L16, and the carried-vs-fresh GATE the head needs (`AX_CARRY`) is
> produced by ops that themselves read `H1` — forming a 2-cycle the
> dim-only scheduler's cycle-breaker is structurally unable to break.**

The carry-head *design* (back-attend to the prior step's byte-1 predictor
row, V-copy its `H1` one-hot forward, positive-ALiBi recency, AX_CARRY
gate) remains correct. What's blocked is **depositing the result into
`H1`** at a late layer. This is the multi-session re-architecture the root
doc (`AX_HIGH_BYTE_DUMP_ROOT_IS_H1_ONEHOT_2026_06_13.md` §"genuine
multi-component re-architecture") already anticipated — now with the exact
mechanism proven, not suspected.

## What was attempted and the exact failure

A complete `make_layer12_ax_byte1_dump_carry_op` (L12 attn head 0,
declarative `DeclarativeAttentionHeadSpec`, host = empty block[12].attn)
was built with the verified gate/signature. It compiles to a
**`ValueError: Dependency cycle detected`** in
`layer_compiler._topological_sort`, stuck-ops list spanning
`layer1_ffn, layer2_mem_byte_flags, layer3_carry_forward_attn,
layer6_attn, layer8_multibyte_fetch, layer10_*, layer14_*, layer15_*,
layer16_lev_routing` — i.e. the whole H1-reader / AX_CARRY-producer SCC.
(The op is preserved as a patch at `/tmp` during the session; it is NOT in
the tree because it does not build.)

## The 2-cycle, exactly

The carry head, to gate carried-vs-fresh, must read the only clean
discriminator: the `AX_CARRY` band (`Σ AX_CARRY ≈ −988` on a FRESH-AX step
vs `≈ +2.7` on a CARRIED step at block 12 — re-confirmed this session,
spec_k=0; see the separability dump below). It writes `H1` (the LM head's
only byte-1 emission band). Then:

* `layer3_carry_forward_attn` **writes** `AX_CARRY_LO/HI` → carry head
  **reads** it ⇒ edge `L3 → carry_head` (correct: keep).
* carry head **writes** `H1` → `layer3_carry_forward_attn` **reads** `H1`
  (l3_ops.py:130, the L0 fine-threshold marker-distance, a legitimate
  *same-step* read of L0's output) ⇒ edge `carry_head → L3`.

Two opposing edges between the same pair ⇒ cycle. The same 2-cycle exists
with `layer8_multibyte_fetch` and `layer6_attn_bake` (both write `AX_CARRY`
**and** read `H1`):

```
AX_CARRY writers that ALSO read H1 (each forms the 2-cycle):
  layer3_carry_forward_attn, layer6_attn_bake,
  layer8_multibyte_fetch, layer8_multibyte_fetch_bake
```

## Why the scheduler's cycle-breaker cannot break it

`layer_compiler._topological_sort` has exactly three cycle-breakers, and
**all three suppress edges on the side of the op that READS** a dim
cross-step — none can drop an edge created because a LATE op's *write* is
read by an EARLY op:

1. **Phase pruning** (`if u.phase is not None and v.phase is not None and
   u.phase > v.phase: continue`). Requires BOTH ops to carry a `phase`.
   The binding H1-readers all dropped their phase in Phase 11.A r3
   (`layer3_carry_forward_attn`, `layer8_multibyte_fetch`, `layer1_ffn`,
   `layer2_mem_byte_flags`, `nibble_copy_ffn` are all `phase=None`). So no
   `phase=12` on the carry op can prune `carry_head → L3`: the guard
   short-circuits on `v.phase is None`. (The prior L12 op's `phase=12.0`
   was therefore inert — confirmed.)
2. **`requires["next_step_after"]`** — suppresses edges from the
   *referenced* op's WRITES into the *declaring* op's READS. The declaring
   op is the reader. Useless for a late WRITE read by an early op.
3. **`requires["after"]` B9 cross-step exception** — same direction: drops
   the edge from a later-phase op whose WRITES the *declaring* op READS.
   Again reader-side.

To decouple `carry_head → L3`, **`layer3_carry_forward_attn` itself** would
have to declare its `H1` read as cross-step relative to the carry op (the
`OUTPUT_HI` / `_PREV_STEP` SSA pattern). But L3's `H1` read is genuinely a
*same-step* read of L0's value — it is not a prev-step carry — so that
declaration would be semantically false, and it would have to be repeated
on **all 54** same-step `H1` readers (L1–L16). That is the B9-style `H1`
SSA decomposition, not a single-head landing.

### The 54 same-step H1 readers (why no late H1 write fits)

`H1` is declared in `writes` by **only** `l0_ops.py` (L0 head 1). Every
other touch is a *read*. 54 ops read `H1` fresh, spanning L1→L16
(`layer1_ffn`, `layer2_mem_byte_flags`, `layer3_carry_forward_attn`,
`layer4_*`, `layer6_*`, `layer7_memory_heads`, `layer8_*`, `layer10_*`
(×12), `layer14_*` (×11), `layer15_*`, `layer16_lev_routing`, …). There is
**no precedent for a late `H1` write** because the dim-only dependency
model treats any `H1` writer as an upstream of all of them. A late writer
that also reads any dim produced *between* it and L16 by an H1-reader
cycles. The AX_CARRY gate read is the unavoidable instance.

## The carried-vs-fresh gate is unavoidably AX_CARRY

Re-measured this session (spec_k=0, block 12 = L13 input, 4 add programs,
byte-1 predictor rows). The dims that cleanly separate FRESH vs CARRIED
(non-overlapping ranges, gap > 5) are **dominated by `AX_CARRY_LO/HI`**
(`≈ −35` per slot fresh / `≈ 0` carried; sum `≈ −988` / `≈ +2.7`), plus
`ALU_LO` and a weak `H3+4`. Every strong separator is in a band written by
an `H1`-reader (`AX_CARRY` ← L3/L6/L8; `ALU_LO` ← ALU ops that read H1).
There is **no clean carried/fresh discriminator in a band written only by
non-H1-readers**, so the gate read cannot avoid the 2-cycle.

(The head cannot drop the gate either: without it, on a FRESH step the head
would self-copy the current row's own `H1` one-hot back into `H1`, doubling
`H1+4` 12.14→24.28 and breaking byte-identity. The gate is what keeps the
fresh-AX path untouched.)

## The two real unblocking paths — both multi-session

1. **`H1` SSA split (the B9 pattern applied to `H1`).** Decompose `H1`
   into `H1` (L0's same-step output, read by the 54 L1–L16 readers) and a
   distinct late-write alias the LM head reads cross-step. Then the carry
   head writes the alias and the early readers keep reading L0's `H1`.
   This is a census + rename touching all 54 readers + a head-bake
   re-point — exactly the `OUTPUT_HI`-split sub-unit shape
   (`docs/B9_OUTPUT_HI_SPLIT_SPEC.md`, 40 ops). Multi-session.

2. **Re-point the dump to a fresh band + additive LM-head bake.** Have the
   carry head write a NEW residual band `AX_DUMP_B1` (no early readers ⇒
   reading `AX_CARRY` no longer cycles, because no `AX_CARRY` writer reads
   `AX_DUMP_B1`), and add `head.weight[byte1_token, AX_DUMP_B1+k]`
   columns via a `TokenEmbeddingRule` head bake so the LM head emits from
   it (additive: `AX_DUMP_B1=0` on fresh steps ⇒ fresh byte-1 stays
   byte-identical via the existing `H1` path). **Blocker:** there is **no
   free residual band**. The dynamic registry's `d_model=920` but the
   compiled model is `872`; the 872→920 span is the SE_* speculative dims,
   and the sub-872 region has a single 1-dim gap (at 733). A new
   value-band needs ≥ 7 dims (to mirror the `H1+(v+2)` one-hot) or 16 (a
   real nibble band), forcing a `d_model` widen — which memory note
   `project_mul_div_mod_arch_blocked` records regresses `bnz` (the 32-dim
   `MUL_RESULT_HI` widen 920→952 regressed with only 9 free dims). So this
   path is widen-gated + multi-component (populate `AX_DUMP_B1` on BOTH
   fresh and carried steps, re-point + re-gate, each byte-identity).
   Multi-session.

## What is now DEFINITELY known (don't re-derive)

* The LM head reads the AX byte-1 emission **exclusively** from `H1`
  (dims 67..73); `logit[0x02]−logit[0x00]` at the byte-1 predictor row is
  `H1+4·(+5.0) + H1+2·(−5.0)`, everything else ~0
  (`tools/probe_ax_logit_attrib.py`, re-run this session).
* The one-hot is BORN at block 10 (= logical L9) on the fresh step and is
  byte-identical block 10→39; it is NEVER born on the carried step
  (`H1` band trace, re-run: block 9 `[_,_,0.94,0,0,0,0]` both steps;
  block 10 fresh `[_,_,2.82,0,12.14,0,1.54]`, carried unchanged).
* The carried/fresh gate is `Σ AX_CARRY` and ONLY separable at block 12+
  (block 10 is +62/+2.7 = not separable, per the prior doc — still true).
* The host search is **exhausted**: an empty (ALiBi-clean) block 13 host
  removes the L13 build-perturbation blocker entirely and STILL fails on
  the H1-write cycle. The cycle is independent of host choice — it is a
  property of *writing `H1`*, not *where* you write it.

## Bottom line

This is not a one-head landing under the current dim graph. The carry
mechanism is designed and verified; the only remaining work is the
**emission-target re-architecture** (H1 SSA split OR fresh-band +
LM-head re-point + d_model widen), each multi-session and byte-identity
gated. The prior "find a build-safe block-12 host" recommendation is a
dead end: no host avoids the H1-write cycle.

## Tools (kept, reusable, spec_k=0)

* `tools/probe_carry_host_survey.py` — per-block carried-vs-fresh
  `Σ AX_CARRY` separability + row-signature survey.
* `tools/probe_carry_rowsig.py` — byte-1 predictor row-signature
  candidate dump at a chosen block.
* `tools/probe_ax_logit_attrib.py`, `tools/probe_h1_carry_design.py`,
  `tools/probe_ax_carry.py` — as documented in the prior two AX docs.

## Baselines (HEAD bf2ce4f7, GPU 0, spec_k=0)

* add (ids 0-49) full_trace: **12/50** (unchanged — no weights changed).
* sub (ids 50-99) full_trace: **5/50** (unchanged).
* smoke: **49 passed / 2 failed** (simple_function / mul_overflow) —
  unchanged; this commit touches docs only.
