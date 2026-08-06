# HF-model fit — the parameterized c4-VM build

How does `qwen_full_vm.build(subset, code_size, …)` (and the vanilla-loop analog
`qwen_vanilla_vm.build`) scale the model geometry `(n_layers, hidden, intermediate)` to
the op-set it must cover, which stock HF (Qwen2) configs can host which subset, and is it
Qwen2-only or could it target other HF architectures?

Generated 2026-08-06. Golden `069cc32f` re-confirmed intact. **Docs-only.** The shape
table below was produced by MATERIALIZING / accounting the REAL builders on this branch
(`f5f31bd8`), not copied from an older doc.

Companion: [`MODEL_FIT_CONFIGURATOR_2026_07_21.md`](MODEL_FIT_CONFIGURATOR_2026_07_21.md)
(the `qwen_fit_solver.py` solver + cost-axis model),
[`DOOM_VS_GENERAL_BOUNDARY.md`](DOOM_VS_GENERAL_BOUNDARY.md) (this is the general core).

---

## TL;DR

- **Parameterized: YES, and TESTED end-to-end.** `build(subset, …)` derives
  `(n_layers, hidden, intermediate)` from the op-set's block specs + the residual width
  `D_used`. Building the **base** subset materializes a genuine `Qwen2Model` with
  `n_layers=6, hidden=896, intermediate=896`, 14 query / 2 KV heads, head_dim 64,
  25.7M params — `fits_stock=True` (the released Qwen2.5-0.5B width). The base build was
  confirmed by actual materialization; the wider subsets by the memory-safe spec
  accounting (both reproduce the same builder geometry).
- **`recurrence folds DEPTH`: confirmed.** `recurrent_divmod=True` weight-ties the
  long-division body → `+muldiv` STORED layers drop 102 → 39 while APPLIED depth stays
  102 (Universal-Transformer-style). Width is unchanged.
- **Which stock config fits which subset:** only **base** fits stock 0.5B. `mem+cmp`
  already overflows hidden (1152 > 896). `+muldiv` / FULL need ~1.5B-to-7B-class WIDTH
  (hidden 2944, intermediate 7920) plus far more depth than any stock Qwen2 ships — they
  do not fit an off-the-shelf checkpoint shape (see the honest scope note).
- **Qwen2-family sizing, not arbitrary architecture (today).** The build constructs a
  literal `Qwen2Model`/`Qwen2Config`. The SEMANTIC requirements (additive norm-free
  residual, softmax-with-sink = softmax1, GQA `repeat_kv`, ALiBi-slope-0 + no-RoPE
  position invariance) are architecture-generic and could target Llama/Mistral/Gemma
  with an arch-adapter — but that is unbuilt. See "Arbitrary-architecture scope".

---

## The parameterization (how the dims are derived)

`c4_min/qwen_full_vm.py::build(code_size=24, subset=SUBSET_BASE, arch=QWEN2_5_ARCH,
efficient_alu=True, recurrent_divmod=False, pad_to_stock=False, …)`:

1. `QwenFullLayout(code_size, subset, …)` lays out the residual bands the op-set needs →
   `D_used` (used residual width).
2. `block_specs = _block_specs(L, code_size, subset, …)` builds the per-block FFN/attn
   specs for exactly the opcodes the subset covers.
3. **`hidden = arch.hidden_for(D_used+1)`** = `D_used` rounded UP to a multiple of
   `head_dim` (64), floored at `num_attention_heads*head_dim` (= 14·64 = 896). This keeps
   the Qwen GQA head partition valid.
4. **`intermediate = max(FFN W_up rows over all blocks, qheads·head_dim, 8)`** = the
   widest block's FFN hidden.
5. **`n_layers = len(block_specs)`** (STORED distinct physical layers). With
   `recurrent_divmod`, the divmod iterations become repeated module references, so
   `n_layers` (STORED) < `n_applied` (APPLIED per forward).
6. `fits_stock = hidden ≤ 896 and intermediate ≤ 4864 and n_layers ≤ 24` (the stock
   Qwen2.5-0.5B budget). `pad_to_stock=True` grows a `fits_stock` subset to the EXACT
   released shape (896/4864/24) with the extra layers identity + residual past `D_used`
   held at 0 — proving the SAME weights run byte-exact in a stock-shaped config.

`qwen_vanilla_vm.build(subset, …)` reuses the same `_block_specs` compute blocks and
adds the vanilla emit/ingest bands, then builds a stock `Qwen2ForCausalLM` with a real
`embed_tokens` + `lm_head` so the STANDARD autoregressive `generate` loop runs the VM
with discrete-token registers. Same subset→dims scaling, wider `D_used` for the vanilla
bands.

### The four cost axes (from the fit solver)

| lever | axis | effect |
|---|---|---|
| op-SET (mem/cmp/bitwise/muldiv) | WIDTH + DEPTH | lights the subset flags → more bands + more blocks |
| `efficient_alu` (the ONLY muldiv path now) | DEPTH | `nibble_alu32` gadgets: MUL ~+10 layers, DIV/MOD ~+the long-division iterations |
| `recurrent_divmod` | DEPTH (STORED) | weight-tie the div body: STORED shrinks, APPLIED unchanged |
| precision 8/16/32 | scales DEPTH | nibble granularity scales the ALU iteration depth (32-bit is the built/verified path) |

> The historical 256×256×3 MUL/DIV/MOD **lookup table** (intermediate ~160465 → ~45 GB,
> the WIDTH-blowup path) has been **removed** from the production build; `efficient_alu`
> (trade table WIDTH for DEPTH) is the only MUL/DIV/MOD path.

---

## Shape table (subset → dims → which HF config fits) — MEASURED on `f5f31bd8`

`code_size=24`, `arch=QWEN2_5_ARCH` (14 q-heads / 2 kv-heads / head_dim 64),
`efficient_alu=True`. STORED = `num_hidden_layers`; APPLIED = layers run per forward
(> STORED only under recurrence). Stock 0.5B budget: hidden ≤ 896, inter ≤ 4864,
layers ≤ 24.

| subset | hidden | intermediate | STORED | APPLIED | D_used | fits stock 0.5B? | smallest stock Qwen2 shape that fits |
|---|---|---|---|---|---|---|---|
| **base** (ISA core) | **896** | **896** | **6** | 6 | 270 | **YES** | Qwen2.5-**0.5B** (896 / 4864 / 24) |
| mem+cmp | 1152 | 896 | 10 | 10 | 1121 | no (hidden 1152 > 896) | needs ≥ **1.5B**-class width (hidden 1536) |
| +bitwise | 1152 | 896 | 30 | 30 | 1121 | no (hidden + depth) | width 1.5B-class; depth > any stock 0.5B/1.5B (30 > 28) |
| +muldiv (unrolled) | 2944 | 7920 | 102 | 102 | 2906 | no | width ~3B-class (hidden 3072); depth **far** past any stock |
| +muldiv (**recurrent**) | 2944 | 7920 | **39** | 102 | 2906 | no | recurrence folds STORED 102→39; width still ~3B-class |
| FULL (unrolled) | 2944 | 7920 | 107 | 107 | 2927 | no | width ~3B-class; depth past any stock |
| FULL (**recurrent**) | 2944 | 7920 | **44** | 107 | 2927 | no | recurrence folds STORED 107→44; width still ~3B-class |

**Materialization check (base):** `build(subset=SUBSET_BASE)` → real `Qwen2Model`,
`Qwen2Config(num_hidden_layers=6, hidden_size=896, intermediate_size=896,
num_attention_heads=14, num_key_value_heads=2, head_dim=64, vocab=267)`, 25.7M params,
`fits_stock=True`. The parameterization works through to a live model.

### Reading the table (honest points)

- **Only `base` fits an off-the-shelf checkpoint shape.** It matches the released
  Qwen2.5-0.5B WIDTH (896 / 4864 / 24) exactly — that is the "≈ released 0.5B" claim,
  and `pad_to_stock=True` grows the 6 real layers to the full 24 (rest identity) so it
  loads into the literal 0.5B config.
- **`recurrence folds depth, not width.** The STORED-layer drop (102→39, 107→44) is the
  Universal-Transformer weight-tie of the divmod loop; APPLIED depth (what actually runs)
  is unchanged. It buys checkpoint SIZE, not a smaller forward.
- **Numbers differ from older docs.** The `MODEL_FIT_CONFIGURATOR_2026_07_21.md` doc
  (and the task's rough hints of "~14 layers/896", "~111/3136") predate this branch. The
  base build here is 6 STORED layers / hidden 896 / inter 896; FULL is hidden 2944 /
  inter 7920. Trust THIS table for `f5f31bd8` — it was measured, not carried over. The
  older doc's hidden-960 base and inter-160465 table rows reflect the pre-removal
  lookup-table build.

---

## Is it Qwen2-only, or could it target other HF architectures?

### Today: **Qwen2-family sizing.**

`build` constructs a literal `transformers.models.qwen2.Qwen2Model` from a `Qwen2Config`
(`rope_theta=1e6`, `rms_norm_eps=1e-6`, `hidden_act="silu"`, GQA 14/2, eager attn). The
`QwenArch` dataclass parameterizes the head geometry `(num_attention_heads,
num_key_value_heads, head_dim)` and the `hidden_for` rounding, but is bound to
`Qwen2Config`. All the shape scaling above is WITHIN the Qwen2 family (0.5B width up to
~3B/7B-class width for FULL).

### Why Qwen2 was chosen — the load-bearing arch features

The VM step maps onto exactly these `Qwen2DecoderLayer` properties (per the module
docstring):

- **additive norm-free residual** — `x + attn(RMSNorm(x))` then `x + mlp(RMSNorm(x))`;
  the RMSNorm is made ≈ identity by a K-compensator lane (`K/√H` γ), so the two additive
  sub-layers ARE the VM's `ffn(attn(x))`.
- **softmax attention with a content-free sink row = softmax1 / ZFOD** — plain softmax +
  a BOS sink row reproduces the blog-spec's softmax1.
- **GQA `repeat_kv`** — the VM's read heads live in KV-group 0 and are broadcast.
- **position invariance** — ALiBi-slope-0 + no-RoPE (the fast-lane distance decay picks
  the latest frame); the build zeroes the RoPE contribution on the VM lanes.

### Could it target Llama / Mistral / Gemma / …?

**In principle yes; unbuilt today.** Those architectures share the load-bearing features
— additive residual, RMSNorm (Gemma: RMSNorm; Llama/Mistral: RMSNorm), SwiGLU MLP, GQA.
Porting would need an **arch-adapter** that:
1. emits the target `*Config` + `*Model` instead of `Qwen2Config`/`Qwen2Model`;
2. reproduces the softmax1 SINK (a content-free sink row, architecture-generic);
3. neutralizes the target's positional scheme to position-invariant on the VM lanes
   (RoPE-zeroing as done for Qwen; a pure-ALiBi target like older MPT would need the
   inverse handling).

**The known hard blocker is the POSITIONAL scheme, not the block shape.** The memory note
`project_qwen_r8_alibi_rope_arch_blocked.md` records that an ALiBi-vs-RoPE mismatch caps
in-place conversion (~25% argmax) and is mathematically unsolvable without a RoPE
recompile or a position-0-only gate — i.e. the port must build the target model FRESH
with the VM weights (as `build` does for Qwen2), not convert a pretrained non-Qwen
checkpoint. So: **arbitrary-architecture is a fresh-build arch-adapter away, not a config
flag today.** The current scope is honestly "Qwen2-family sizing".

---

## How to reproduce this table (memory-safe, CPU)

```python
# Accounting only — reproduces the REAL builder geometry, NO model materialised:
from c4_min import qwen_full_vm as Q
QL = Q.QwenFullLayout(24, Q.SUBSET_FULL, efficient_alu=True, recurrent_divmod=True,
                      code_from_memory=True, shift_via_mul=True, div_logsink=False)
specs = Q._block_specs(QL.L, 24, Q.SUBSET_FULL, efficient_alu=True, recurrent_divmod=True,
                       code_from_memory=True, shift_via_mul=QL.shift_via_mul,
                       div_logsink=QL.div_logsink)
hidden = Q.QWEN2_5_ARCH.hidden_for(QL.D_used + 1)
inter  = max(int(s["W_up"].shape[0]) for _, s in specs)
stored = len(specs)                       # num_hidden_layers

# Full materialization + read-back geometry (base is the memory-cheap one):
vm = Q.build(code_size=24, subset=Q.SUBSET_BASE)     # -> genuine Qwen2Model
print(vm.n_layers, vm.hidden_size, vm.intermediate_size, vm.fits_stock)

# The solver front-door (accounting + feasibility vs a named stock target):
from c4_min import qwen_fit_solver as S
res = S.fit(target="stock-0.5b", ops=S.FULL, minimize="depth")   # honestly INFEASIBLE
```

---

## Golden safety

Golden `069cc32fa7cecfbceae448a7dbf6e2140b3db6cf6857c8accec5639b9c55c0ca` re-confirmed
via `python -m c4_min._fingerprint_build`. The base-subset build + spec accounting above
materialize/inspect models but write no repo state. This doc adds no build/weight change.
