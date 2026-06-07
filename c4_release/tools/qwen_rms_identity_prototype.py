"""Synthetic validation of the RMSNorm-as-identity trick for Qwen structural adapter.

Phase R2 of c4_release/docs/QWEN_STRUCTURAL_ADAPTER_PLAN_2026_06_07.md.

Claim:
  Add a "norm_compensator" dim to the residual stream holding a large constant K.
  With gamma_i = K / sqrt(d_model) for every dim, RMSNorm becomes an effective
  identity on the real dims, provided K^2 >> sum(real_dims^2).

This script sweeps K and reports:
  (a) the max relative error on the "real" dims after RMSNorm
  (b) the absolute deviation of the norm_compensator slot from K after RMSNorm
"""

from __future__ import annotations

import math

import torch


def rmsnorm(x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Standard RMSNorm: x' = x / sqrt(mean(x^2) + eps) * gamma."""
    # x: (..., d_model). Compute mean over the last dim.
    mean_sq = x.pow(2).mean(dim=-1, keepdim=True)
    rms = torch.sqrt(mean_sq + eps)
    return x / rms * gamma


def run_one(K: float, d_model: int, residual_scale: float, seed: int,
            dtype: torch.dtype, eps: float) -> dict:
    """Run a single RMSNorm-identity trial and return diagnostics."""
    torch.manual_seed(seed)
    # Real residual values: random Gaussian * residual_scale
    x = (torch.randn(1, 1, d_model) * residual_scale).to(dtype)
    # Slot 0 is the norm_compensator: holds K
    x[0, 0, 0] = K

    # Proposed gamma: K / sqrt(d_model) for every dim
    gamma = torch.full((d_model,), K / math.sqrt(d_model), dtype=dtype)

    x_out = rmsnorm(x, gamma, eps=eps)

    # Real-dim error excludes slot 0
    real_in = x[0, 0, 1:]
    real_out = x_out[0, 0, 1:]
    abs_err = (real_out - real_in).abs()
    max_abs_in = real_in.abs().max().item() or 1.0
    max_rel_err = (abs_err.max().item()) / max_abs_in

    # Compensator slot deviation
    comp_out = x_out[0, 0, 0].item()
    comp_dev = abs(comp_out - K)

    # Stats on the ratio of K^2 to S = sum of squared real values
    S = real_in.pow(2).sum().item()
    K_sq = K * K
    ratio = K_sq / max(S, 1e-30)

    # The theoretical multiplier on real dims is (K / sqrt(d_model)) / rms.
    rms_actual = math.sqrt((K_sq + S) / d_model + eps)
    multiplier = (K / math.sqrt(d_model)) / rms_actual
    multiplier_err = abs(multiplier - 1.0)

    return {
        "K": K,
        "max_rel_err": max_rel_err,
        "comp_out": comp_out,
        "comp_dev_abs": comp_dev,
        "comp_dev_rel": comp_dev / K,
        "K2_over_S": ratio,
        "multiplier": multiplier,
        "multiplier_err": multiplier_err,
    }


def fmt(x: float) -> str:
    if x == 0.0:
        return "0"
    if abs(x) < 1e-3 or abs(x) > 1e4:
        return f"{x:.3e}"
    return f"{x:.6f}"


def main() -> None:
    d_model = 784
    residual_scale = 5.0
    seed = 0
    eps = 1e-6
    K_sweep = [10.0, 100.0, 1000.0, 10000.0, 100000.0]

    for dtype, label in [(torch.float32, "fp32"), (torch.float64, "fp64")]:
        print(f"\n=== d_model={d_model}, residual_scale={residual_scale}, "
              f"seed={seed}, eps={eps}, dtype={label} ===")
        print(f"{'K':>10} | {'max_rel_err':>14} | {'multiplier':>14} | "
              f"{'mult_err':>12} | {'comp_dev_abs':>14} | {'K^2/S':>14}")
        print("-" * 100)
        results = []
        for K in K_sweep:
            r = run_one(K, d_model, residual_scale, seed, dtype, eps)
            results.append(r)
            print(f"{r['K']:>10.0f} | {fmt(r['max_rel_err']):>14} | "
                  f"{fmt(r['multiplier']):>14} | {fmt(r['multiplier_err']):>12} | "
                  f"{fmt(r['comp_dev_abs']):>14} | {fmt(r['K2_over_S']):>14}")

        target = 1e-4
        passing = [r for r in results if r["max_rel_err"] < target]
        if passing:
            smallest = min(passing, key=lambda r: r["K"])
            print(f"\n  smallest K with max_rel_err < {target}: K={smallest['K']:.0f}  "
                  f"(max_rel_err={fmt(smallest['max_rel_err'])})")
        else:
            print(f"\n  no K in sweep meets max_rel_err < {target}")

    # Also probe sensitivity to larger residual_scale (stress test)
    print("\n=== stress: large residual_scale at K=10000 (fp32) ===")
    print(f"{'scale':>8} | {'max_rel_err':>14} | {'K^2/S':>14}")
    print("-" * 50)
    for scale in [1.0, 5.0, 20.0, 100.0]:
        r = run_one(10000.0, d_model, scale, seed, torch.float32, eps)
        print(f"{scale:>8.1f} | {fmt(r['max_rel_err']):>14} | {fmt(r['K2_over_S']):>14}")


if __name__ == "__main__":
    main()
