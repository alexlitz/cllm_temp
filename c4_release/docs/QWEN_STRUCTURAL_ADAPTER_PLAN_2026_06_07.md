# Qwen structural adapter plan (no retraining)

The weights are hand-set per BLOG_SPEC — they ARE the model. The goal is to make Qwen's architecture act like ours on our weights via **compensating dims + padding + key renames**. No retraining.

## Compensating-dim foundation

The trick: add a residual-stream dim that holds a known constant `K`. Then upstream architectural operations (RMSNorm, normalized softmax, biases) act as effective identities when combined with carefully-set weight scales.

### RMSNorm compensation

RMSNorm: `x' = x / sqrt(mean(x²) + ε) * γ`

Add a dim where `x[norm_compensator_idx] = K` always. Then for `K² >> sum(real_dims²)`:

```
sqrt(mean(x²) + ε) ≈ K / sqrt(d_model)
x' ≈ x * sqrt(d_model) / K * γ
```

Choose `γ_i = K / sqrt(d_model)` for every dim → `x' = x`. **RMSNorm becomes identity.**

The norm_compensator dim:
- Width: 1 per residual stream
- Cost: +1 to d_model (~0.13% overhead at d=784)
- Inserted by: a model-level bake that sets `embed.embed.weight[:, norm_compensator_idx] = K`
- Preserved by: ensuring every block's `W_o`, `W_down`, and `head_bake` writes 0 to this slot (or that K is preserved structurally)

### softmax1 → standard softmax compensation

softmax1: `softmax1(s)_i = exp(s_i) / (1 + sum_j(exp(s_j)))`
Standard softmax: `softmax(s)_i = exp(s_i) / sum_j(exp(s_j))`

The `+1` in softmax1 is a sink position with implicit score 0. Replicate in standard softmax by **adding a virtual K position** at attention input where K[virtual] = 0 for all heads. Standard softmax then naturally includes `exp(0)=1` in its denominator.

Implementation: when exporting to Qwen, prepend a single token-position to the sequence at compile-time with K=0 for all heads, V=0. The position is invisible to downstream tokens.

Caveat: positional encoding shifts. Need to adjust RoPE/ALiBi to account for the inserted index.

### SwiGLU repacking

Qwen: `down(silu(gate(x)) * up(x))`
Ours: `down(silu(up(x)) * gate(x))` — order flipped

Math: `silu(a) * b == b * silu(a)`. The factors commute. **Pure key rename:**
- Our `W_up.weight` → Qwen's `gate_proj.weight`
- Our `W_gate.weight` → Qwen's `up_proj.weight`
- Our `W_down.weight` → Qwen's `down_proj.weight`

### SwiGLU bias folding

Qwen's MLP has no bias. Ours has `b_up`, `b_gate`, `b_down`.

Folding: assume the residual stream has a `bias_compensator` dim = 1 always. Then `W_up @ x + b_up == W_up' @ x` where `W_up'[:, bias_compensator_idx] += b_up`. Same for `b_gate`. For `b_down` (output bias), we'd need a final bias_compensator dim on the post-FFN side.

Implementation: pre-fold all biases into weights using the existing `CONST=1` residual dim (which our model already uses). On export, set biases to 0 and absorb their values into the corresponding W matrices' bias_compensator column.

### Per-block post_ops flattening

Our `TransformerBlock` has `post_ops: List[nn.Module]`. Each post_op is an extra forward step after `attn → ffn`.

Qwen doesn't support post_ops. Flatten:
- A block with N post_ops becomes 1 + N Qwen blocks
- The base block exports normally
- Each post_op becomes a Qwen block where:
  - `self_attn.W_q/K/V/O` = identity-like (or skip-pass — pass `attn_output = hidden_states`)
  - `mlp` runs the post_op's transformation

Effective layer count: `sum(1 + len(block.post_ops) for block in model.blocks)`. The audit suggests this is ~26-30 layers after flattening.

### d_model padding for L15 wide heads

Qwen requires `H * HD == D`. Our L15 has `H * HD > D` (wide heads via `W_o` projection).

Pad d_model to `H * HD`:
- Add `H * HD - D` compensating dims all carrying 0
- Pad every weight matrix accordingly (rows/columns zero-extended)
- W_o discards the padding dims after attention

### GQA / sliding window

Choose **Qwen3-Dense** target — no GQA, no enforced SWA.

## Phased implementation plan

### Phase R1 — Compensating-dim infrastructure (1 session)
- Add `norm_compensator` to dim registry
- Token embedding bake: `embed_weight[:, norm_compensator_idx] = K` for all token IDs
- Verify residual stream preserves K through one block forward (probe at end of L0)
- Verify byte-identity on smoke (the new dim should be invisible)

### Phase R2 — RMS norm identity validation (1 session)
- Synthetic test: wrap one of our blocks with a leading RMSNorm using compensating-dim trick
- Compare wrapped output to our base output via `torch.equal`
- Tune K and γ to maximize byte-identity (likely K = 100-1000, γ scales accordingly)

### Phase R3 — Softmax1 → standard softmax sink (1 session)
- Implement virtual-position sink at attention input
- Verify attention weights byte-identity for representative test programs
- Adjust positional encoding offsets

### Phase R4 — SwiGLU repack + bias folding (1 session)
- Repacking: pure key-rename layer in `qwen_compat.export_qwen2_dense_state_dict`
- Bias folding: extend `bias_compensator` dim (already exists as CONST=1); pre-fold all linear biases at export time

### Phase R5 — Post-ops flattening (1-2 sessions)
- Count flattened layer count
- Build per-post-op block with skip-pass attention
- Verify forward equivalence on a single token

### Phase R6 — Full export + load (1 session)
- Wire all phases through `export_qwen3_dense(model, output_dir)`
- Test `AutoModelForCausalLM.from_pretrained(output_dir)` succeeds
- Token-by-token forward equivalence on 5 smoke programs

### Phase R7 — Tokenizer (1 session, Option B byte-wrapper)
- Add a tokenizer-wrapping layer
- Translate Qwen vocab tokens → our byte tokens on input
- Translate our byte tokens → Qwen-readable string on output

### Phase R8 — E2E validation (1 session)
- `int main(){return 42;}` round-trips through Qwen-exported model
- Diff vs native runner output
- Confirm `≥99%` argmax match (since CSR causes ~0.1% noise; dense should be 100%)

### Cumulative: ~8-10 sessions

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Compensating-dim K too small → RMS norm not identity | Add a static check at export that verifies the assumption (compute `K² / sum(real²)` ratio at each block and assert it's > 1000) |
| Adding norm_compensator breaks byte-identity in current smoke | Make it opt-in via env var `C4_QWEN_EXPORT_COMPAT=1` |
| Bias folding produces NaN under bf16 due to magnitude | Use fp32 export path; let HF cast post-load if user wants bf16 |
| Post_ops flattening produces too many layers for Qwen's max_position_embeddings | Cap at Qwen3-Dense's max (~131k) — our 35 expanded layers fit easily |
| Tokenizer Option B expands byte sequences to use too much context | Cap programs at the byte/token ratio Qwen can handle (~4× BPE expansion typically) |

## Verification matrix

| Phase | Test | Acceptance |
|---|---|---|
| R1 | `K` preserved through block | `residual[norm_compensator] == K ± 1e-6` |
| R2 | RMSNorm wraps to identity | `torch.equal(wrapped_block(x), base_block(x))` |
| R3 | Softmax sink replicates softmax1 | Attention weights match to 1e-5 |
| R4 | SwiGLU export round-trip | FFN output byte-identical |
| R5 | Flatten produces equivalent forward | Final residual byte-identical |
| R6 | HF AutoModel load | No errors, all keys consumed |
| R7 | Tokenizer round-trips program | `decode(encode(prog)) == prog` byte-equal |
| R8 | Full inference parity | Final exit code byte-equal to native runner |
