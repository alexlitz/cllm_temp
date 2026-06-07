"""Validate Phase R3: softmax1 via virtual K-position sink.

Math:
  softmax1(s)_i = exp(s_i) / (1 + sum_j exp(s_j))

  With a virtual K position s_v = 0 prepended (so s' = [0, s_1, ..., s_T]),
  standard softmax gives:
    softmax(s')_v = exp(0)  / (exp(0) + sum_j exp(s_j)) = 1       / (1 + Z)
    softmax(s')_i = exp(s_i)/ (exp(0) + sum_j exp(s_j)) = exp(s_i)/ (1 + Z)
                  = softmax1(s)_i  ✓

If V_v = 0 as well, the virtual position contributes nothing to the
attention output for the real positions: attn = sum_i softmax(s')_i * V'_i
where V'_0 = 0.

This prototype constructs Q, K, V and compares:
  attn_sm1 = softmax1(QK^T/sqrt(D)) @ V
  attn_std = softmax(Q (K')^T/sqrt(D)) @ V'    with K'=cat([0_row, K]), V'=cat([0_row, V])

It also exercises adversarial inputs (large scores, all-zero K, all-zero
scores), and reports max-abs and max-rel error.

Run:  python -m c4_release.tools.qwen_softmax_sink_prototype
"""

from __future__ import annotations

import math
import sys

import torch


def softmax1(scores: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """softmax1(s)_i = exp(s_i) / (1 + sum_j exp(s_j))

    Implemented numerically-stable: subtract max(s, 0) so neither exp(s_i - m)
    nor exp(0 - m) = exp(-m) overflows.
    """
    zero = torch.zeros_like(scores.select(dim, 0).unsqueeze(dim))
    s_aug = torch.cat([zero, scores], dim=dim)  # prepend a 0 along reduced dim
    m, _ = s_aug.max(dim=dim, keepdim=True)
    e = torch.exp(scores - m)
    e_zero = torch.exp(-m).squeeze(dim)  # exp(0 - m), squeezed back
    denom = e.sum(dim=dim) + e_zero  # = sum_j exp(s_j - m) + exp(-m)
    return e / denom.unsqueeze(dim)


def attention_softmax1(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Compute attention with softmax1 in the score normalization."""
    d = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d)
    weights = softmax1(scores, dim=-1)
    return torch.matmul(weights, v)


def attention_std_with_sink(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Standard softmax attention with K, V prepended by a zero row (the sink).

    Returns the attention output for the real positions (sink row dropped if
    present in Q, but here Q is unchanged — only K/V are augmented).
    """
    d = q.shape[-1]
    # K shape: (B, H, T, D). Prepend a zero row along T (dim=-2).
    zero_k = torch.zeros_like(k.select(-2, 0).unsqueeze(-2))  # (B, H, 1, D)
    zero_v = torch.zeros_like(v.select(-2, 0).unsqueeze(-2))
    k_aug = torch.cat([zero_k, k], dim=-2)  # (B, H, T+1, D)
    v_aug = torch.cat([zero_v, v], dim=-2)
    scores = torch.matmul(q, k_aug.transpose(-1, -2)) / math.sqrt(d)
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v_aug)


def compare(name: str, a: torch.Tensor, b: torch.Tensor) -> dict:
    diff = (a - b).abs()
    max_abs = float(diff.max())
    # Avoid div-by-zero in rel error
    denom = a.abs().clamp_min(1e-12)
    max_rel = float((diff / denom).max())
    byte_eq = bool(torch.equal(a, b))
    print(f"[{name}]")
    print(f"  byte-identical : {byte_eq}")
    print(f"  max abs diff   : {max_abs:.3e}")
    print(f"  max rel diff   : {max_rel:.3e}")
    print(f"  allclose 1e-6  : {bool(torch.allclose(a, b, atol=1e-6, rtol=1e-5))}")
    print(f"  allclose 1e-5  : {bool(torch.allclose(a, b, atol=1e-5, rtol=1e-4))}")
    return {
        "name": name,
        "byte_identical": byte_eq,
        "max_abs": max_abs,
        "max_rel": max_rel,
    }


def run_case(name: str, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> dict:
    attn_sm1 = attention_softmax1(q, k, v)
    attn_std = attention_std_with_sink(q, k, v)
    return compare(name, attn_sm1, attn_std)


def main() -> int:
    torch.manual_seed(0xC4)
    B, H, T, D = 1, 8, 64, 64

    results: list[dict] = []

    # 1. Random fp32 inputs (the canonical case).
    q = torch.randn(B, H, T, D, dtype=torch.float32)
    k = torch.randn(B, H, T, D, dtype=torch.float32)
    v = torch.randn(B, H, T, D, dtype=torch.float32)
    results.append(run_case("random_fp32", q, k, v))

    # 2. Random fp64 — tighter bound on numerical noise.
    q64 = q.double()
    k64 = k.double()
    v64 = v.double()
    results.append(run_case("random_fp64", q64, k64, v64))

    # 3. Large scores (attention saturation): scale Q and K up by 100x.
    results.append(run_case("large_scores_100x", q * 100, k * 100, v))

    # 4. Very large scores (1000x) — risk of inf/nan if not stable.
    results.append(run_case("huge_scores_1000x", q * 1000, k * 1000, v))

    # 5. All-zero K — every score is 0, softmax becomes uniform.
    #    softmax1: each weight = exp(0) / (1 + T*exp(0)) = 1/(T+1).
    #    sink std: each weight (for real positions) = 1/(T+1). Match expected.
    k_zero = torch.zeros_like(k)
    results.append(run_case("all_zero_K", q, k_zero, v))

    # 6. All-zero Q — every score is 0, same uniform behavior.
    q_zero = torch.zeros_like(q)
    results.append(run_case("all_zero_Q", q_zero, k, v))

    # 7. All-zero everything — degenerate.
    results.append(run_case("all_zero_QKV", q_zero, k_zero, torch.zeros_like(v)))

    # 8. Negative-heavy scores — sink term dominates (softmax1 -> ~0 output).
    #    Verify both versions agree that real positions have small weight.
    results.append(run_case("very_negative_scores", q - 50, k, v))

    # 9. Sanity probe: confirm softmax1 weights and standard-with-sink weights
    #    agree on the real positions (not just on the V-weighted output).
    d = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d)
    w_sm1 = softmax1(scores, dim=-1)  # (B, H, T, T)
    zero_k = torch.zeros_like(k.select(-2, 0).unsqueeze(-2))
    k_aug = torch.cat([zero_k, k], dim=-2)
    scores_aug = torch.matmul(q, k_aug.transpose(-1, -2)) / math.sqrt(d)
    w_std = torch.softmax(scores_aug, dim=-1)  # (B, H, T, T+1)
    # Drop the sink column (index 0) from std weights to compare.
    w_std_real = w_std[..., 1:]
    compare("weights_real_positions", w_sm1, w_std_real)

    # Also confirm the sink absorbs exactly the right mass:
    # w_std[..., 0] should equal 1 - w_sm1.sum(-1).
    sink_mass = w_std[..., 0]
    expected_sink = 1.0 - w_sm1.sum(dim=-1)
    compare("sink_mass_vs_1_minus_sum_softmax1", sink_mass, expected_sink)

    # 10. Sequence-length sweep: T in {1, 8, 64, 512}.
    for t in (1, 8, 64, 512):
        qt = torch.randn(B, H, t, D)
        kt = torch.randn(B, H, t, D)
        vt = torch.randn(B, H, t, D)
        results.append(run_case(f"seqlen_T={t}", qt, kt, vt))

    print()
    print("=" * 60)
    print("Summary (max abs error per case):")
    print("=" * 60)
    worst = 0.0
    for r in results:
        print(f"  {r['name']:30s}  abs={r['max_abs']:.3e}  rel={r['max_rel']:.3e}")
        worst = max(worst, r["max_abs"])
    print()
    print(f"Worst-case max abs error: {worst:.3e}")

    # Acceptance: fp32 noise floor is ~1e-6 on this size; the sink trick adds
    # one extra term to the denominator, so we accept up to 1e-5.
    threshold = 1e-5
    ok = worst < threshold
    print(f"Threshold {threshold:.0e}: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
