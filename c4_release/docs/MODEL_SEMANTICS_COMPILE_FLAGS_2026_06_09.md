# Model Semantics Compile Flags — Umbrella Design (2026-06-09)

Companion design notes:
- `QWEN_RMS_IDENTITY_PROTOTYPE_2026_06_07.md` — `normalization` variant.
- `QWEN_SOFTMAX_SINK_PROTOTYPE_2026_06_07.md` — `softmax_variant`.
- `QWEN_SWIGLU_REPACK_PROTOTYPE_2026_06_07.md` — `ffn_variant`.
- `A3_9_ALIBI_DIAGNOSTIC_2026_06_09.md` — `positional_encoding` divergence.
- `QWEN_STRUCTURAL_ADAPTER_PLAN_2026_06_07.md` — overall Qwen migration plan.
- `TOGGLE_INVENTORY_2026_06_09.md` — full env-toggle landscape these new
  flags integrate into.

This document is the **umbrella**: how the six per-axis variant choices
compose into a single `compile_full_vm_dynamic(...)` call, how the two
presets snap onto them, and how the byte-identity parity test proves the
backward-compatibility default is wired correctly. Each per-axis design
doc (above) owns the *implementation* of that axis. This doc owns the
*API surface* that switches between them.

## 1. The compile API

Six independent flags, each with a `"native"` value (today's behavior)
and a `"qwen"` value (the Qwen-compatible alternative), plus a `preset`
shortcut that expands to a coherent pair of six.

```python
from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
    compile_full_vm_dynamic,
)

model, layout = compile_full_vm_dynamic(
    # Existing kwargs (S=, n_heads=, ffn_hidden=, ...) unchanged.
    preset=None,                          # None | "native" | "qwen"
    positional_encoding=None,             # None | "alibi" | "rope"
    softmax_variant=None,                 # None | "softmax1" | "standard"
    normalization=None,                   # None | "identity" | "rmsnorm"
    ffn_variant=None,                     # None | "native" | "swiglu"
    per_head_qk_norm=None,                # None | "none" | "qwen"
    ffn_routing=None,                     # None | "single" | "composite"
)
```

All six flags default to `None`. `None` means "use the per-axis default",
which is the **native** value on every axis. This guarantees backward
compatibility: every existing call site keeps the same behavior because
no positional / kwarg signature changed and every default is unchanged.

### Per-axis values

| Flag | Native value | Qwen value | Owner doc |
|---|---|---|---|
| `positional_encoding` | `"alibi"` | `"rope"` | `A3_9_ALIBI_DIAGNOSTIC_2026_06_09.md` |
| `softmax_variant` | `"softmax1"` | `"standard"` | `QWEN_SOFTMAX_SINK_PROTOTYPE_2026_06_07.md` |
| `normalization` | `"identity"` | `"rmsnorm"` | `QWEN_RMS_IDENTITY_PROTOTYPE_2026_06_07.md` |
| `ffn_variant` | `"native"` | `"swiglu"` | `QWEN_SWIGLU_REPACK_PROTOTYPE_2026_06_07.md` |
| `per_head_qk_norm` | `"none"` | `"qwen"` | (new — see §6) |
| `ffn_routing` | `"single"` | `"composite"` | already exists (`enable_moe_routing`) |

### Bridging to existing kwargs

Two of the new flags simply rename (and slightly retype) existing kwargs;
the umbrella shims them so the legacy surface still works:

- `positional_encoding` already exists on the compile signature.
  The new value space (`"alibi" | "rope"`) is a strict subset of the
  existing one — the legacy `"hybrid"` continues to work for callers
  that pass it explicitly.
- `softmax_variant="softmax1"|"standard"` maps onto the existing
  `attention_normalization="softmax1"|"softmax"` kwarg.
- `normalization="identity"|"rmsnorm"` maps onto the existing
  `use_rms_norm=False|True` kwarg.
- `ffn_variant="native"|"swiglu"` maps onto a new
  `ffn_activation_kind="relu"|"swiglu"` knob (today only `"relu"` is
  realized; the SwiGLU prototype is sketched in
  `QWEN_SWIGLU_REPACK_PROTOTYPE_2026_06_07.md`).
- `per_head_qk_norm` is fresh — no legacy kwarg.
- `ffn_routing="single"|"composite"` maps onto the existing
  `enable_moe_routing=False|True` kwarg (composite = MoE-routed FFN
  cluster).

Mixing the new and legacy kwargs in the same call is an error
(`TypeError`, mirroring the existing `arch=` vs individual-kwarg gate).
The umbrella picks one surface per call site.

## 2. The two presets

```python
PRESETS = {
    "native": {
        "positional_encoding": "alibi",
        "softmax_variant":     "softmax1",
        "normalization":       "identity",
        "ffn_variant":         "native",
        "per_head_qk_norm":    "none",
        "ffn_routing":         "single",
    },
    "qwen": {
        "positional_encoding": "rope",
        "softmax_variant":     "standard",
        "normalization":       "rmsnorm",
        "ffn_variant":         "swiglu",
        "per_head_qk_norm":    "qwen",
        "ffn_routing":         "composite",
    },
}
```

`preset=` is *expanded then merged*: the preset's six values become
defaults, and any explicit per-flag kwarg overrides them. Example:

```python
compile_full_vm_dynamic(
    preset="qwen",
    per_head_qk_norm="none",   # ablate the per-head qk norm only
)
```

…compiles a Qwen-shaped VM with RoPE + standard softmax + RMSNorm +
SwiGLU + composite routing, but without the per-head q/k norm.
This is the per-axis ablation surface the diagnostic harnesses
(`QWEN_R8_LAYER_DIVERGENCE_2026_06_09.md`) consume.

Passing `preset=` together with `arch=ModelArchitectureSpec(...)`
is rejected the same way the existing
`arch=` vs individual-kwarg gate works.

## 3. Composition rules

> Every independent combination of the six flags is a valid compile.

Concretely: there are 2⁶ = 64 reachable points and the umbrella commits
to admitting all of them at compile time. **No combination is reserved**
— each per-axis design doc is responsible for ensuring its variant is a
drop-in replacement for the native value at every other axis's setting,
and the per-op tests (one per variant doc) guarantee the cross product
holds.

Justification:

- **No shared state between axes.** The five non-routing axes touch
  disjoint pieces of the bake: positional encoding lives in
  `base_layers.py` attention modules, softmax variant in the per-head
  decode lowering, normalization in the pre-attn / pre-ffn norm layer,
  FFN variant in the FFN block construction, per-head qk-norm in the
  attention head module. There is no path where one axis's variant
  forces another's value.
- **`ffn_routing` is upstream.** Composite routing wraps the entire FFN
  block in a routing layer, so it sees the chosen `ffn_variant` as its
  expert recipe. Native + composite (a routed bank of native FFNs) and
  Qwen-FFN + single (a single SwiGLU block) are both meaningful
  ablations.
- **Per-op tests guard the cross product.** Each per-axis design doc
  ships a smoke test that the variant works under both native and Qwen
  values of every *other* axis. The umbrella's parity test (§4) only
  proves the **byte-identity backward-compat** corner.

There is one **hard error**: passing the legacy `arch=` kwarg
*together* with any of the six new flags. The existing surface already
rejects `arch=` + individual-arch-kwargs; the umbrella extends the same
rule to the six new ones. See §5.

## 4. The parity test

Two byte-identity claims gate the design. Only the first lands this
round; the second is deferred until the variant implementations exist.

### 4.1 Backward-compat parity (this round)

> `compile_full_vm_dynamic(preset="native")` is byte-identical to
> `compile_full_vm_dynamic()` (every default).

Implemented as `tests/test_compile_flag_parity.py::test_preset_native_byte_identical_to_default`.
The test compiles both VMs with `disk_cache=False` and walks every
attention block, asserting `W_q`, `W_k`, `W_v`, `W_o`,
`mask`, FFN `W_in` / `W_out`, and pre/post norm parameters are
bytewise equal via `torch.equal(...)`. It also asserts the
`layout.dim_positions` dict matches one-for-one.

This is the load-bearing guarantee for the design: existing callers see
no diff, the umbrella is a pure-additive change.

### 4.2 Semantic parity (deferred — written but xfailed until §6)

> A mini-VM compiled with `preset="native"` and a mini-VM compiled with
> `preset="qwen"` produce byte-identical `OUTPUT_LO` / `OUTPUT_HI` on a
> single-token program.

The premise: the semantic dim bands are the same regardless of which
positional / softmax / norm / FFN representation the compute layers use.
W_q / W_k / W_v / W_o storage *shapes* differ (SwiGLU repacks the FFN
hidden into 2*hidden, per-head qk-norm adds two scalar buffers per head,
RoPE swaps the alibi-slope buffer for cos/sin caches), so the test
projects to the read-only `OUTPUT_LO` / `OUTPUT_HI` dims rather than
diffing internal storage. The semantic-parity claim is what makes the
six axes worth shipping: every observable VM output is invariant under
the axis choice, modulo intentional model-quality differences.

The deferred test is sketched as
`tests/test_compile_flag_parity.py::test_preset_qwen_output_parity`
guarded by `pytest.skip("variants not yet implemented")` so the
scaffolding is visible. Each per-axis variant doc owns flipping the
skip to a passing assertion as its acceptance criterion.

## 5. Migration path

Long pole first, byte-identity gate at every step.

### 5.1 Token embedding (L0)
Already done via the `I3` wrapper / `TokenEmbeddingRule` path
(`docs/AUTOREGRESSIVE_IO_PLAN.md`). No work.

### 5.2 Register layers (L1-L5)
Stateful register / decode ops. The relevant axes here are
`normalization` (pre-attn / pre-ffn norm chosen by the IR spec) and
`positional_encoding` (RoPE caches initialised when `kind="rope"`).
The base-layer modules already accept both kinds; the work is wiring
the new flags down through `_bake_from_scheduled_ops` to the
`positional_encoding=` / `use_rms_norm=` constructor args (already in
place — see `full_vm_compiler_dynamic.py:1856-1858`).

### 5.3 Compute layers (L6+)
Attention + FFN + routing. The bulk of the variant surface lives here:
softmax variant, per-head q/k norm, FFN variant, FFN routing. Each
per-axis variant doc owns the lowering changes for its axis. The
umbrella's only job is plumbing the flag through.

### 5.4 Memory lookup (L15) — long pole
`memory_lookup` is still partially imperative (per
`CLAUDE.md` Phase status note). Several axes (per-head qk-norm,
SwiGLU FFN) require declarative coverage of the L15 attention head
before they can be applied uniformly. The migration order is therefore:

1. Land §4.1 byte-identity gate (this round).
2. Each per-axis variant doc lands its own native-side declarative
   coverage (no behavior change, just IR migration).
3. Each per-axis variant doc lands its Qwen-side value behind the
   flag, with a per-op test under §3's composition matrix.
4. The deferred §4.2 byte-identity gate flips from xfail to passing
   once all six axes are wired.
5. The full `preset="qwen"` end-to-end (the existing
   `QWEN_R8_PRODUCTION_STATUS_2026_06_09.md` campaign) flips to green
   as the downstream consumer.

## 6. Open question: `per_head_qk_norm`

The other five axes have prior art (env toggles, prototype docs, or
existing kwargs). `per_head_qk_norm="qwen"` is the only axis without a
landed prototype. The Qwen value adds two `nn.LayerNorm` modules per
attention head (one each on the Q and K projections, *before* the
RoPE / ALiBi step) with `head_dim`-shaped scale parameters.

For this round the flag is **defined but not implemented**: passing
`per_head_qk_norm="qwen"` raises `NotImplementedError` from the compile
path. The flag exists on the signature so callers (and the deferred
parity test) can name it; the wiring lands with the per-axis design
agent's implementation.

## 7. Coordination with the per-axis design agents

This doc owns:
- The compile-flag signature and preset table (§1, §2).
- The composition gate (§3).
- The backward-compat parity test (§4.1) and the deferred semantic
  parity test (§4.2 scaffolding).
- The plumbing through `compile_full_vm_dynamic` → `_bake_from_scheduled_ops`.

The per-axis design docs own:
- The IR / lowering changes for their axis.
- The per-op tests that prove their axis composes under every other
  axis's setting.
- Flipping §4.2 from xfail to passing for their slice.

If a per-axis design doc renames its flag value (e.g. `"none"` →
`"identity"` for `normalization`), this doc and the preset table are
the single point of change.
