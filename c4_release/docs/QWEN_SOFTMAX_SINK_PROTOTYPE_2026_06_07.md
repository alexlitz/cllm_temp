# Qwen R3 — softmax1 via virtual K-position sink (prototype validation)

Validates the Phase R3 claim in `QWEN_STRUCTURAL_ADAPTER_PLAN_2026_06_07.md`
(commit `b79374b5`): softmax1, used by our attention, can be replaced by
standard softmax in Qwen by prepending a single "sink" key/value position
with `K_sink = 0` and `V_sink = 0`.

Script: `c4_release/tools/qwen_softmax_sink_prototype.py`.

## Math

Our normalized attention uses **softmax1**:

```
softmax1(s)_i = exp(s_i) / (1 + sum_j exp(s_j))
```

The `+1` in the denominator is the "sink": an implicit position with score 0,
which never contributes to the value-weighted sum because it has no V.

Take an augmented score vector with a virtual position prepended:
`s' = [0, s_1, …, s_T]`. Standard softmax over `s'` is:

```
softmax(s')_v = exp(0)   / (exp(0) + sum_j exp(s_j)) = 1       / (1 + Z)
softmax(s')_i = exp(s_i) / (exp(0) + sum_j exp(s_j)) = exp(s_i)/ (1 + Z)
              = softmax1(s)_i                                          ✓
```

where `Z = sum_j exp(s_j)`. So **standard softmax over `[0, s]` equals
softmax1 over `s` on the real positions**.

For the V-weighted output, set `V_sink = 0`:
```
attn(real_i) = sum_t softmax(s')_t · V'_t
             = softmax(s')_v · 0  +  sum_j softmax(s')_j · V_j
             = sum_j softmax1(s)_j · V_j
```

So the recipe is:
1. Compute scores against augmented keys `K' = concat([0_row, K], dim=T)`.
2. Standard softmax over the T+1 scores.
3. Multiply by `V' = concat([0_row, V], dim=T)`.

The Q tensor is **unchanged**: queries only need to attend to the sink, not
be attended to from it. Qwen's standard MHA layer naturally produces the
correct softmax1 outputs once K and V are augmented.

## Equivalence test results

All shapes `(B=1, H=8, T=64, D=64)` unless noted; fp32 unless noted.

| Case                         | Byte-identical | Max abs err | Max rel err |
|------------------------------|----------------|-------------|-------------|
| random_fp32                  | no             | 2.09e-07    | 1.52e-03    |
| random_fp64                  | no             | 5.55e-16    | 2.55e-12    |
| large_scores (Q,K × 100)     | **yes**        | 0           | 0           |
| huge_scores (Q,K × 1000)     | **yes**        | 0           | 0           |
| all_zero_K (uniform softmax) | **yes**        | 0           | 0           |
| all_zero_Q                   | **yes**        | 0           | 0           |
| all_zero_QKV                 | **yes**        | 0           | 0           |
| very_negative_scores (-50)   | no             | 4.77e-07    | 5.04e-06    |
| weights_real_positions       | no             | 2.98e-08    | 3.72e-07    |
| sink_mass = 1 − Σ softmax1   | no             | 2.22e-07    | 3.09e-05    |
| seqlen_T = 1                 | no             | 1.19e-07    | 1.23e-07    |
| seqlen_T = 8                 | no             | 4.77e-07    | 8.60e-05    |
| seqlen_T = 64                | no             | 2.98e-07    | 9.36e-03    |
| seqlen_T = 512               | no             | 1.49e-07    | 4.35e-02    |

**Worst-case abs error across all cases: 4.77e-07** (fp32). The fp64 case
collapses to 5.55e-16, confirming the residual error is pure floating-point
rounding — algebraically the two formulations are *identical*.

The byte-identical saturated cases (×100, ×1000, all-zero) are the most
informative: when scores are large or zero, both branches take the exact
same paths through `exp` and division and produce identical bits.

Plateau of max rel error at sequence length ~10⁻² is **not** a softmax1
defect — it reflects that some output components are ~10⁻⁵ in magnitude
while the absolute error is ~10⁻⁷, so relative error inflates. Output
correctness is bounded by abs error.

### Acceptance vs the plan's R3 row

| R3 acceptance criterion (plan)              | Result                  |
|---------------------------------------------|-------------------------|
| Attention weights match to 1e-5             | **PASS** (max 3e-8 fp32)|
| Output attention match (added by prototype) | **PASS** (max 5e-7 fp32)|

## Caveats

### Positional encoding (RoPE) **does** shift

Qwen uses RoPE: position-dependent rotation of Q and K *before* the
`Q·K^T` product. If we prepend a virtual position to K, the real K at
original index `0` would be rotated as if it sat at position `1`, which
breaks attention to it. **Two clean fixes**:

1. **Assign the sink RoPE index −1** (or any value outside the program's
   active range) and keep real keys at their original indices. This is
   the trick used by HF's StreamingLLM / attention-sink Llama variants —
   the sink is positionless because its key is zero anyway, so the RoPE
   rotation on it is also zero (`R(θ) · 0 = 0`). Concretely: skip the
   RoPE multiply for the sink slot, or just multiply it; it stays zero.

2. **Bake the sink into the K projection itself**, not the sequence. Add
   a `1` to the residual stream's `bias_compensator` dim, then set
   `W_K[bias_compensator_idx, :]` to a row that produces a "virtual key"
   accessible by *every* Q via a shared additive softmax denominator.
   This is more invasive and requires modifying attention kernels — not
   compatible with stock Qwen MHA.

**Decision for R3**: option (1). The sink K and V are literally zero, so
RoPE on the sink is a no-op regardless of which position id is assigned.
But the **real** tokens' positions must NOT shift by 1. Implementation
detail: the export must keep real-position RoPE indices unchanged.
HF attention layers compute RoPE from `position_ids` rather than slot
index, so we control this via the `position_ids` tensor at inference
time, or by patching the rotary cache in the exported model.

### Attention mask (causal)

Causal mask blocks Q at position `t` from attending to K at positions
`> t`. The sink must be visible to **every** Q. If we prepend the sink
at slot 0, the standard causal mask already permits this (row `t` of the
causal mask allows columns `0…t`). The sink is therefore mask-safe.

### KV cache layout

If R6 exports use HF's KV cache, the cache must be prepended with the
sink K and V on first forward, and the cache offset shifted by 1 for all
subsequent autoregressive steps. This is a wiring detail, not a math
issue.

### Numerical stability under bf16

Qwen3 typically loads in bf16. The plan calls for fp32 export. bf16 has
~3 decimal digits of mantissa; the sink trick adds `exp(0) = 1` to the
denominator which has magnitude 1 — same order as a typical
post-softmax-scaling exp term. No new instability versus baseline
softmax. Recommend fp32 attention compute regardless.

### Sequence length cost

One extra K/V row per layer per sequence. Negligible (`T+1` vs `T`).

## Verdict

The math is correct, the prototype is byte-identical in saturated /
degenerate regimes and fp32-noise-level across all 13 tests. **R3 is
mechanically simple**; the only thing the implementer must get right is
the RoPE position id for the real tokens (must NOT shift by 1 just
because a sink was prepended).
