# Qwen R2: RMSNorm-as-identity prototype (2026-06-07)

Validates the math behind Phase R2 of
[`QWEN_STRUCTURAL_ADAPTER_PLAN_2026_06_07.md`](QWEN_STRUCTURAL_ADAPTER_PLAN_2026_06_07.md)
on top of commit `b79374b5`. Script:
[`c4_release/tools/qwen_rms_identity_prototype.py`](../tools/qwen_rms_identity_prototype.py).

## Math derivation

RMSNorm on a `d_model` residual `x`:

```
rms = sqrt( mean(x^2) + eps )
    = sqrt( (K^2 + S) / d_model + eps )
x'  = x / rms * gamma
```

Slot 0 is the **norm_compensator**: `x[..., 0] = K` (a constant). The other
`d_model - 1` "real" dims have sum of squares `S = sum_{i>=1} x_i^2`.

Set `gamma_i = K / sqrt(d_model)` for every `i`. Then for any dim:

```
x'_i = x_i * (K / sqrt(d_model)) / rms
     = x_i * (K / sqrt(d_model)) / sqrt( (K^2 + S) / d_model + eps )
```

Define the universal multiplier

```
m(K, S) = (K / sqrt(d_model)) / sqrt( (K^2 + S) / d_model + eps )
        = K / sqrt( K^2 + S + d_model * eps )
        = 1 / sqrt( 1 + S/K^2 + d_model*eps/K^2 )
```

So `m -> 1` exactly when `K^2 >> S` (and `K^2 >> d_model * eps`, which is
trivially satisfied for any reasonable `K`).

For the compensator slot itself: `x'_0 = K * m(K, S)`. Same multiplier, so the
absolute deviation is `K * (1 - m) ~ K * S / (2 K^2) = S / (2K)`, which **shrinks
as K grows** in relative terms but grows linearly in absolute terms relative to
the small real values. The relative deviation `(1 - m) ~ S / (2 K^2)` matches
the real-dim relative error.

## Synthetic results

`d_model = 784`, residual `x ~ N(0, 5^2)` per dim, eps = 1e-6, seed = 0. fp32:

|       K | max_rel_err (real dims) |     multiplier `m` | `1 - m` | comp_dev_abs (slot 0) |       K^2 / S |
|--------:|------------------------:|-------------------:|--------:|----------------------:|--------------:|
|      10 |                  0.931  |             0.069  |  0.931  |               9.305   |        0.005  |
|     100 |                  0.428  |             0.572  |  0.428  |              42.848   |        0.485  |
|    1000 |                  0.0102 |             0.9898 | 0.0102  |              10.151   |       48.5    |
|  10000  |                **1.03e-4** |          0.99990  | 1.03e-4 |               1.029   |     4850.7    |
| 100000  |               **1.02e-6** |          0.999999 | 1.03e-6 |               0.102   |     4.85e+5   |

fp64 matches fp32 to within a ulp at every K — the error is **algebraic, not
numerical**. Byte-identity reachable today depends only on K, not on the
working precision.

Target: `max_rel_err < 1e-4`.

- **Smallest K in the sweep meeting the target: `K = 10000`** (max_rel_err =
  1.03e-4 — right at the edge).
- Comfortable margin (~100x headroom): **`K = 100000`** (max_rel_err = 1.02e-6).

## Recommended K

**Use `K = 1e5` for the initial adapter bake.**

Rationale:

- Gives ~3 orders of magnitude headroom below the 1e-4 byte-identity threshold,
  so noise from other Qwen-side surgery (attention skip-pass, FFN identity) has
  room before it pushes us over.
- Absolute compensator drift through one RMSNorm is ~0.1 (out of 100k); easy to
  re-clamp downstream if needed, but well below any reasonable downstream
  threshold.
- Stays well clear of fp16/bf16 overflow if Qwen ever runs in mixed precision
  (bf16 max ~3.4e38; K=1e5 in `embed[:, 0]` produces no overflow even after
  large matmul fan-in).

`K = 1e4` is the floor — workable but no margin against the stress modes below.

## Caveats / failure modes

1. **Large real-dim magnitudes shrink the safety margin.** Stress sweep at
   `K = 10000` (fp32):

   | residual_scale | max_rel_err |   K^2 / S |
   |---------------:|------------:|----------:|
   |            1.0 |     4.2e-6  |  1.21e+5  |
   |            5.0 |    1.03e-4  |  4.85e+3  |
   |           20.0 |     1.6e-3  |  3.03e+2  |
   |          100.0 |     3.9e-2  |    12.1   |

   `S` scales like `d_model * scale^2`. Any block that pushes residual magnitudes
   above ~5 (typical post-MLP magnitudes can hit 20+ in untuned models) will
   break identity at K=1e4. K=1e5 absorbs scale up to ~50; K=1e6 absorbs ~500.

2. **Per-step drift accumulates.** This prototype validates *one* RMSNorm.
   Each block in Qwen has two (pre-attn, pre-ffn). Errors compound — N
   norms in series scale max_rel_err by ~N (linear, since the multiplier is
   deterministic per-block and applied in series). For Qwen 28L * 2 = 56
   RMSNorms, a 1e-6 per-norm error becomes ~5.6e-5, still under 1e-4. K=1e4
   would compound to ~5.8e-3 — well over threshold.

3. **Compensator slot replenishment.** Each RMSNorm shrinks the compensator by
   `S / (2 K)`. Over 56 norms this drift accumulates linearly. At K=1e5 the
   per-norm drift is ~0.1, so 56 norms drift ~5.6 (relative ~5.6e-5). The
   plan's Phase R1 ("K preserved through block") relies on this staying small,
   or on the embed/output paths re-clamping it. Validation will need an
   end-to-end check that the compensator stays in band, not just per-norm.

4. **Distribution shape doesn't matter much.** The math depends only on
   `S = sum(real^2)`, not on the distribution; Gaussian was used here for
   convenience. Outlier-heavy distributions with the same `S` give identical
   error.

5. **`eps` is irrelevant at scale.** `d_model * eps = 7.84e-4`; once `K^2 > 10`
   it's invisible. Don't worry about Qwen's eps choice (typically 1e-6).

## Verdict

The trick works as stated in the plan. With `K = 1e5` and
`gamma = K / sqrt(d_model)`, RMSNorm is byte-equivalent to identity within
1e-4 across the whole 56-norm Qwen stack, with comfortable margin against
typical residual magnitudes up to ~50. R2 is green; the riskiest piece of the
Qwen adapter is validated. Proceed to R3 (wrap a real block in this RMSNorm
and confirm `torch.equal`).
