# Qwen Compatibility Blocker Inventory — Q1 Plan

Date: 2026-06-07
Base commit: `be93fabc` (off `main`)
Author: Q1 deployment-plan inventory pass

## Scope and method

The deployment plan calls for running
`analyze_qwen_compatibility(model)` and `build_qwen2_dense_mapping_plan(model)`
from `c4_release/neural_vm/qwen_compat.py`. **That file does not exist on
`main` at `be93fabc`** (verified via `find . -iname '*qwen*'` — no hits in the
repo). The compile-and-run step is therefore a no-op until the helper lands.

In place of the tooling output, this doc inventories blockers via direct
structural inspection of the VM's transformer (`AutoregressiveVM` in
`c4_release/neural_vm/vm_step.py`, `AutoregressiveAttention` same file,
`PureFFN` / `PureAttention` in `c4_release/neural_vm/base_layers.py`) and
config (`c4_release/neural_vm/config.py`, `c4_release/neural_vm/constants.py`).

## Observed VM architecture

| Knob | VM value | Source |
| --- | --- | --- |
| `d_model` | **512** (not 784) | `vm_step.py:1306`, `constants.py:134` |
| `n_layers` | **17** | `vm_step.py:1307` (was 16, +L16 routing) |
| `n_heads` | **8** | `vm_step.py:1308` |
| `head_dim` | 64 (= 512/8) | `vm_step.py:94`, `constants.py:136` |
| `ffn_hidden` | 4096 default (per-block dict allowed) | `vm_step.py:1309`, `1342-1347` |
| Per-position embed | `E.DIM = 169`, `NUM_POSITIONS = 8` | `embedding.py:90-91` |
| Norm | **NONE** — block is `attn → ffn → post_ops`, no pre/post norm | `vm_step.py:1261-1279` |
| FFN | SwiGLU: `down(silu(up·x + b_up) * (gate·x + b_gate)) + b_down` | `base_layers.py:57-87` |
| Attention | softmax1 sink (`anchor=0`), per-head ALiBi or RoPE | `vm_step.py:486-537`, `428-434` |
| Positional | Configurable: ALiBi / RoPE / hybrid (L0-L2 ALiBi, L3+ RoPE) | `vm_step.py:118-161`, `config.py:26` |
| ALiBi slopes | `2.0 ** (-8.0/n_heads * (i+1))` per-head | `vm_step.py:136-138` |
| RoPE base | 10000.0 default (configurable) | `config.py:29`, `vm_step.py:153` |
| Max seq len | 1024 (extendable) | `vm_step.py:1310`, `_extend_rope_cache` |
| Sliding window | unbounded (causal mask only) | `vm_step.py:441-444` |
| Post-ops chain | per-block `nn.ModuleList` of PureFFN post-ops | `vm_step.py:1272-1278` |
| L15 head expansion | `H*HD > D` (12×64=768→W_o:512) | `vm_step.py:544-549` |
| Compact heads | live head pruning via `_is_compact` | `vm_step.py:213-266` |

## Blocker table (vs Qwen2-Dense, Qwen2.5, Qwen3-Dense)

| # | Delta | Qwen2.5 expects | VM has | Class | Effort |
| --- | --- | --- | --- | --- | --- |
| 1 | **No normalization** in `TransformerBlock` | Pre-norm RMSNorm before attn and FFN (`input_layernorm`, `post_attention_layernorm`) plus final `model.norm` | block = `attn(x) → ffn(x)` only (`vm_step.py:1274-1278`) | **Architectural** | High — wrapper or retrain: identity-init RMSNorm (γ=1) preserves logits but the VM relies on residual magnitudes baked relative to no-norm. Surgical wrapper at export time. |
| 2 | **softmax1 sink** (anchor=0) | Standard softmax | sink K/V column appended (`vm_step.py:486-537`) | **Lossy** | Med — drop sink at export changes attention weights at low-energy positions; need accuracy probe. Some ops (e.g. ZFOD relays) depend on softmax1. |
| 3 | **ALiBi (default)** vs RoPE | RoPE base=1e6 (Qwen2.5) / 1e6 (Qwen3) | ALiBi default; RoPE available but `rope_base=10000` | **Lossy** | Med — switch to `VMConfig.rope_mode(rope_base=1e6)` and rebake ALiBi-dependent layers (L0-L2 in hybrid). |
| 4 | **Hybrid ALiBi+RoPE** (L0-L2 vs L3+) | All-layer RoPE | mixed per-layer | **Architectural** | High if L0-L2 weights bake ALiBi slopes — re-bake L0-L2 against RoPE or insert ALiBi shim. |
| 5 | **MHA 8/8** vs **GQA 14Q/2KV** (Qwen2.5-7B) | 14 query heads, 2 KV heads (group=7) | 8 Q, 8 KV (MHA) | **Lossy** | Med-High — head merge to GQA: cluster K/V heads (k-means on baked weights) or pad to 14Q/2KV by repeat-interleave. Distinct head behaviour (per-head ALiBi slopes, L15's 12 heads) makes merging non-trivial. |
| 6 | **head_dim=64** vs Qwen2.5-7B `head_dim=128` | 128 | 64 | **Lossy** | High — concat-pair heads or upscale W_q/k/v columns; breaks per-head budget assumptions in `vm_step.py:103-108`. |
| 7 | **L15 wide head** (`H*HD > D`, W_o:768→512) | Standard `H*HD == D` | L15 only | **Architectural** | Med — split L15 into two equivalent layers or absorb into W_o (already happens). |
| 8 | **Compact heads + per-head soft-eviction mask** | n/a | runtime mask `kv_cache.per_head_keep_mask` (`vm_step.py:457-469`) | **Cosmetic** | Low — disable at export. |
| 9 | **Sliding window** unbounded | Qwen2 sliding=4096 (Qwen2.5 disables) | unbounded causal | **Cosmetic** | Low — pick Qwen2.5/Qwen3 (no SWA) or set `use_sliding_window=False`. |
| 10 | **Tied embed/head** | Qwen3-0.6B ties, larger don't | `self.head = nn.Linear` separate (`vm_step.py:1362`) | **Cosmetic** | Low — rename / tie at export. |
| 11 | **Per-block `post_ops` chain** | Single FFN per block | 0-N PureFFN post-ops per block (`vm_step.py:1272-1278`) | **Architectural** | Med — expand into extra "passthrough-attn + post_op FFN" blocks (already done in `_split_post_ops_into_blocks` near line 2168/2187/2194); raises layer count well past Qwen norms (17 → 30-50+). |
| 12 | **Vocabulary** = `Token.VOCAB_SIZE` (≪ 152k) | 152064 (Qwen2.5) / 151936 (Qwen3) | `Token.VOCAB_SIZE` ≈ 50 | **Architectural** | High for general LLM use; **Cosmetic** if shipping a VM-tokenizer fork. |
| 13 | **Embedding** is `NeuralVMEmbedding` (state-encoder with positional injection at `ADDR_KEY`/`MEM_*`) | nn.Embedding(vocab, d_model) | Token-format-aware encoder (`vm_step.py:1331`) | **Architectural** | High — VM embedding bakes positional/dim allocation; cannot be a plain lookup. |
| 14 | **FFN bias terms** present (`b_up`, `b_gate`, `b_down`) | Qwen FFN: `down(silu(gate(x)) * up(x))`, **no bias**, gate-before-up multiplication order | bias on all three; order `silu(up) * gate` (`base_layers.py:83-86`) | **Lossy** | Med — swap multiplication ordering (silu(gate) * up); absorb biases into W (where rank allows) or zero-init. |
| 15 | **per-block FFN width dict** | Uniform `intermediate_size` | per-block widths (`ffn_widths` dict, post-`compact`) | **Cosmetic** | Low — pad to max width with zero rows. |

## Recommendation

**Target variant: Qwen3-Dense** (smaller, no SWA, modern RoPE base 1e6,
optional tied embeddings on small models). Qwen2.5 adds GQA which forces
Blocker #5 (lossy head merge) immediately; Qwen2 adds sliding window
(#9). Qwen3-dense minimises lossy steps to #2, #14.

A minimum-effort path:

1. Build the missing `qwen_compat.py` (Q0 prereq) — auto-emit the table above
   from `model.blocks`.
2. Move to `VMConfig.rope_mode(rope_base=1e6)`, re-bake L0-L2 against RoPE,
   retire ALiBi path (#3, #4).
3. Insert identity-init RMSNorm wrappers around each `TransformerBlock`'s
   attn and FFN, plus `model.norm` (#1).
4. Drop softmax1 sink behind a config flag and accuracy-probe (#2).
5. Re-order FFN multiplication to `silu(gate)*up`, zero biases (#14).
6. Materialise `post_ops` as full transformer blocks (#11) so the export
   has one (attn, ffn) per block.
7. Ship a VM-tokenizer fork to side-step (#12)/(#13).

After steps 1-7 the model is mappable to Qwen3-Dense via state-dict rename
+ identity-init norms; remaining accuracy loss is bounded by #2 and #14.

## Effort summary

| Class | Count |
| --- | --- |
| Cosmetic | 4 (#8, #9, #10, #15) |
| Lossy | 5 (#2, #3, #5, #6, #14) |
| Architectural | 6 (#1, #4, #7, #11, #12, #13) |

## Verification artifacts (TODO once `qwen_compat.py` lands)

- [ ] `analyze_qwen_compatibility(model)` JSON report committed under
      `c4_release/docs/qwen/`.
- [ ] `build_qwen2_dense_mapping_plan(model)` state-dict diff committed.
- [ ] Logit-parity probe between VM and shimmed Qwen-format export.
