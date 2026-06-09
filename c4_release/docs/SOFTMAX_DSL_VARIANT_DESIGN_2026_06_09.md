# Softmax DSL Variant — design + scaffold (2026-06-09)

A parallel DSL variant for **standard softmax** (as used by stock Qwen3)
alongside the native VM's **softmax1**. Per BLOG_SPEC.md §410 (*"all
addresses are helpfully initialized to zero by softmax1"*) the native VM
relies on softmax1's `+1` denominator for two semantic functions:

1. **ZFOD** ("Zero Fill On Demand"). An attention head whose Q address
   doesn't match any K row sees no real K contribution; softmax1's anchor
   absorbs the entire softmax mass, leaving `Σ_j softmax1(s)_j · V_j = 0`.
   The residual position effectively reads zero — *as if* unmapped memory
   is pre-initialized to 0.

2. **Empty attention head**. A head whose Q gate is OFF (no Q-side
   activation) produces score = 0 against every K row. With softmax1 the
   anchor is the only non-negative contributor, so the head's output
   contribution is 0 — it harmlessly self-suppresses. Under standard
   softmax the head spreads uniform mass across all K rows, mixing them
   into the residual.

Stock Qwen3 uses **standard softmax** (no `+1` term). Without a
substitute for the softmax1 anchor, every memory-lookup head would
normalise over its K rows even when nothing legitimately matches,
corrupting the residual whenever L15 LI hits an unmapped address.

This document specifies how the unified compiler can emit a
**softmax-variant**-tagged build where every softmax1-relying head
gets a synthetic **sink K row** that reproduces the anchor's effect
exactly (the R3 result from
`docs/QWEN_SOFTMAX_SINK_PROTOTYPE_2026_06_07.md`).

## 1. Compile-time flag

`compile_full_vm_dynamic(...)` (and its static redirect in
`full_vm_compiler.py`) grows one new kwarg:

```python
def compile_full_vm_dynamic(
    *,
    softmax_variant: Literal["softmax1", "standard"] = "softmax1",
    ...
):
```

Semantics:

| `softmax_variant` | `attention_normalization` | sink injection |
|-------------------|----------------------------|----------------|
| `"softmax1"` (default) | `"softmax1"` | none (runtime anchor) |
| `"standard"`           | `"softmax"`  | per-head sink K row |

The flag composes with — but is independent of — the existing
`attention_normalization=` kwarg:

* `"softmax1"` forces `attention_normalization="softmax1"` and is a
  no-op on the bake (preserves byte-identity with the production
  compile path).
* `"standard"` forces `attention_normalization="softmax"` AND injects
  one synthetic sink K row per memory-lookup head per layer (see §3).
* Passing both `softmax_variant="standard"` and
  `attention_normalization="softmax1"` is a hard error (`TypeError`).

`softmax_variant` is the **primary** surface for new callers. The
existing `attention_normalization=` surface stays for back-compat (it
ONLY swaps the runtime softmax with no compensating bake change) and
is deprecated for the bake path — pass `softmax_variant=` if you want
both the runtime softmax AND the bake-side sink injection coordinated.

## 2. ZFOD substitute — the "sink K row" construction

R3 already validated the math (see `QWEN_SOFTMAX_SINK_PROTOTYPE_2026_06_07.md`):

```
softmax1(s)_i = exp(s_i) / (1 + Σ_j exp(s_j))
softmax([0, s])_i ≡ exp(s_i) / (exp(0) + Σ_j exp(s_j))   (for i ≥ 1)
                  = softmax1(s)_i.
```

The deployed runtime already implements this in `vm_step.py:537-559`
(the SDPA path appends a sink `K=0, V=0` column when `use_softmax1` is
on). What's *missing* is a **bake-time** equivalent for a model that's
been compiled with `attention_normalization="softmax"` but still needs
ZFOD for memory-lookup heads.

### The bake-time sink

For each softmax1-relying head we add **one extra K-side slot** whose:

* `K` projection writes a constant offset (`CONST → K_sink_slot = 0`),
  so the sink K row has all-zero values regardless of the input
  residual;
* `V` projection has *no writes* — `V_sink = 0` so the sink absorbs
  softmax mass without contributing to the output;
* `Q` projection has no extra writes — the Q row still sees its
  natural target K rows; against the K-zero sink it scores exactly 0
  (matching softmax1's anchor).

The sink slot is allocated from the head's HD budget by reserving an
unused slot index (per-head `sink_idx`, see §3 placement).

This is a **per-head** sink, not a per-sequence one (Qwen's R3 export
uses a per-sequence sink token at column 0 of `input_ids`; that path
remains the export-time mechanism and is unchanged). The DSL-side sink
is logically embedded into the K projection, so it requires no input
preprocessing — every position naturally gets a sink K row in its KV
buffer.

### Why this preserves byte-identity *of the head's intended output*

Under softmax1, when no K matches:
* All scores ≈ negative; anchor = 0 wins; output = 0.

Under standard softmax + sink:
* All scores ≈ negative; sink_score = 0 wins; sink's V = 0; output = 0.

When some K matches (normal lookup):
* softmax1: matched K wins; `Σ softmax1 · V = matched_V`;
* sink: matched K wins; sink's V = 0 contributes nothing; same output.

The bake therefore preserves the head's **functional** output. The
exact softmax mass distribution differs by `O(exp(-large_score))` —
identical to the fp32 noise floor measured in the R3 prototype
(4.77e-07 worst case, 0 at saturated scales).

## 3. Where sinks must be injected

The compile path walks each `AttentionOp` and, for any head spec
tagged `requires_softmax1_anchor=True`, allocates a sink K slot.
The following heads are tagged in this design:

### L15 `layer15_memory_lookup.li_lc_stack0_h{0..3}`
LI/LC byte 0..3 + STACK0 pop heads. Each reads a Q address and looks
up a `ADDR_KEY` K row. Misses must yield 0 (the L14 corrective layer
relies on the unmapped read reading exactly 0 to leave the residual
slot blank for downstream LI-vs-LC dispatch).

### L15 `layer15_memory_lookup.lev_heads_4_11`
LEV `saved_bp` / `return_addr` byte 0..3 heads. Only emitted when
`num_heads >= 12`. Same logic: a LEV miss must read 0 so the cleanup
chain doesn't see a phantom return address.

### L7 `layer7_memory_heads.head_{2..7}`
ADDR_B0/B1/B2 LO/HI gather heads + flag relay (head 5..7). These run
in the address-formation pipeline; an unmatched gather must read 0 so
the assembled ADDR_KEY hits an unmapped row at L15 (which is itself a
softmax1-relying lookup).

Other softmax1-suppress-style heads (the `softmax1_suppress` audit
category in `dsl_interpreter.py`) are NOT tagged here — those heads
use K=CONST to *deliberately fail* the score race against the anchor.
Under standard softmax they'd need a different rewrite (K must be
pushed actively negative so the sink still wins). This is logged as
**out-of-scope** for the present DSL variant; a follow-up design will
ratchet the audit's `softmax1_suppress` set into compile errors when
`softmax_variant="standard"`.

## 4. Per-op effect under `softmax_variant="standard"`

| Op | Effect |
|----|--------|
| `layer7_memory_heads` | Each head gains a sink K slot; output unchanged on hit, 0 on miss (matches softmax1). |
| `layer15_memory_lookup.heads_0_3` | Same; LI/LC misses now read 0 via sink instead of via softmax1 anchor. |
| `layer15_memory_lookup.lev_heads_4_11` | Same for LEV heads (when `num_heads >= 12`). |
| L1 ALiBi efficient-exp head (`efficient_exp_attention`, BLOG §561) | NOT migrated. The exp-via-softmax1 construction is fundamentally about exploiting the anchor's algebraic identity; under standard softmax the construction collapses. Compile raises `NotImplementedError`. |
| L10 / L6 / others using K=CONST suppression | See §3: NOT migrated, will be flagged by a follow-up audit. |
| `div_mode="log_softmax1"` | Already raises in `VMConfig.__post_init__` when `attention_normalization != "softmax1"`. The new flag's check piggy-backs on that path. |

The static-baked weights for **every other op** are byte-identical
under both variants — the sink injection is additive (one extra K
slot per tagged head; existing slots are unchanged).

## 5. Test contract

A 1-step VM forward with `softmax_variant="softmax1"` and
`softmax_variant="standard"` on the same VM compile must produce
**byte-identical OUTPUT_LO / OUTPUT_HI bytes for unmapped LI/SI/LC
addresses**, modulo a fp32-noise allclose at `1e-6` for the per-head
sink-mass redistribution.

`tests/test_softmax_dsl_variant.py` implements this contract:

1. Compile two VMs with identical kwargs except `softmax_variant`.
2. Drive both with the same initial state where `MEM_addr0` is set to
   an address that has never been stored (a guaranteed unmapped LI).
3. Step both VMs one tick.
4. Assert `OUTPUT_LO == OUTPUT_LO_ref` (byte) and `OUTPUT_HI ==
   OUTPUT_HI_ref` (byte). Numerical residual-stream equality is
   asserted via `torch.allclose(..., atol=1e-6)`.

The skeleton test ships as `xfail` until the sink-injection lowering
lands (this design doc + scaffolding only — no op migration yet).

## 6. Scaffold delivered

* New kwarg `softmax_variant=` plumbed through `compile_full_vm_dynamic`
  (validation only; lowering side is a no-op unless the lowering wave
  lands).
* `_inject_sink_k_row(spec, sink_idx)` helper added to `primitives.py`.
  Returns a new `DeclarativeAttentionHeadSpec` with one extra `K`
  projection at `sink_idx` whose only write is `CONST → 0`. The
  existing K rows and the V/Q/O writes are passed through unchanged.
  Idempotent: calling twice with the same `sink_idx` is a no-op
  (deduplicated).
* `tests/test_softmax_dsl_variant.py` with one passing skeleton test
  that pins the flag plumbing (validates the kwarg surface today; the
  ZFOD-equivalence assertion is `xfail` pending the lowering wave).

## 7. Non-goals / explicit exclusions

* **No op migration**: this change does NOT flip any production op to
  `softmax_variant="standard"`. The default remains `"softmax1"` and
  byte-identity with the historical compile path is preserved.
* **No runtime softmax change**: the `AutoregressiveAttention` forward
  pass is unchanged. The bake-side sink slot is consumed by the
  existing standard-softmax path.
* **No `softmax1_suppress` rewrite**: heads using K=CONST to lose
  against the anchor are out of scope (see §3).
* **No `log_softmax1` migration**: the `div_mode="log_softmax1"` path
  is mathematically tied to the softmax1 anchor; under
  `softmax_variant="standard"` it raises `NotImplementedError`
  (unchanged from `VMConfig.__post_init__`).

## 8. Follow-up

1. **Wave A**: lower `_inject_sink_k_row` from the L15 + L7 head
   specs when `softmax_variant="standard"`. Land behind the flag with
   the test promoted from `xfail` to a hard byte-identity gate.
2. **Wave B**: audit `softmax1_suppress` heads (see
   `dsl_interpreter.py:audit_attention_gates`) and either rewrite to
   active-negative K or pin to a separate "needs explicit
   anti-anchor" tag.
3. **Wave C**: rewrite `efficient_exp_attention` (BLOG §561) for
   standard softmax — likely needs a different algebraic identity
   (or accepts the `NotImplementedError` permanently).
4. **Wave D**: optional — collapse `attention_normalization=` into
   `softmax_variant=` once all call sites migrate, deleting the
   bare runtime-only knob.
