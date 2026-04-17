"""Analyze how different configurations affect attention behavior."""

import torch
import torch.nn.functional as F
from neural_vm.kv_cache_eviction import softmax1

def compare_softmax_variants():
    """Compare softmax1 vs F.softmax with null key."""
    print("="*70)
    print("Comparing softmax1 vs F.softmax with null key")
    print("="*70)

    # Test with different score patterns
    test_cases = [
        ("All negative (ZFOD case)", torch.tensor([-5.0, -10.0, -8.0, -12.0])),
        ("Mixed scores", torch.tensor([2.0, -1.0, 0.0, -3.0])),
        ("All positive", torch.tensor([1.0, 2.0, 3.0, 4.0])),
        ("One dominant", torch.tensor([10.0, 0.0, 0.0, 0.0])),
    ]

    for name, scores in test_cases:
        print(f"\n{name}: {scores.tolist()}")

        # softmax1
        attn_softmax1 = softmax1(scores.unsqueeze(0), dim=-1)[0]

        # F.softmax with null key
        scores_with_null = torch.cat([scores, torch.tensor([0.0])])
        attn_with_null = F.softmax(scores_with_null.unsqueeze(0), dim=-1)[0]
        attn_fsoftmax = attn_with_null[:-1]  # Remove null attention
        null_attn = attn_with_null[-1]

        print(f"  softmax1:     {attn_softmax1.tolist()}")
        print(f"  F.softmax:    {attn_fsoftmax.tolist()}")
        print(f"  Null attn:    {null_attn.item():.4f}")
        print(f"  Difference:   {(attn_softmax1 - attn_fsoftmax).abs().max().item():.6f}")
        print(f"  Sum softmax1: {attn_softmax1.sum().item():.4f}")
        print(f"  Sum F.soft:   {attn_fsoftmax.sum().item():.4f}")

        # Check if mathematically equivalent
        if (attn_softmax1 - attn_fsoftmax).abs().max() < 1e-6:
            print(f"  ✓ Mathematically equivalent!")
        else:
            print(f"  ✗ NOT equivalent - weights need adjustment!")


def analyze_rope_effect():
    """Analyze how RoPE rotation affects Q-K similarity."""
    print("\n" + "="*70)
    print("Analyzing RoPE effect on Q-K similarity")
    print("="*70)

    # Simple test vectors
    dim = 8
    Q = torch.randn(1, 2, 4, dim)  # [B, H, S, D]
    K = torch.randn(1, 2, 4, dim)

    # Standard attention scores
    scores_standard = torch.matmul(Q, K.transpose(-2, -1)) / (dim ** 0.5)

    # Apply simple RoPE (just rotation by position)
    def apply_simple_rope(x, seq_len):
        B, H, S, D = x.shape
        positions = torch.arange(seq_len).unsqueeze(1)
        freqs = 1.0 / (10000.0 ** (torch.arange(0, D, 2).float() / D))
        angles = positions * freqs
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        rotated = torch.zeros_like(x)
        rotated[..., 0::2] = rotated_x1
        rotated[..., 1::2] = rotated_x2
        return rotated

    Q_rope = apply_simple_rope(Q, 4)
    K_rope = apply_simple_rope(K, 4)
    scores_rope = torch.matmul(Q_rope, K_rope.transpose(-2, -1)) / (dim ** 0.5)

    print(f"\nStandard scores:\n{scores_standard[0, 0]}")
    print(f"\nRoPE scores:\n{scores_rope[0, 0]}")
    print(f"\nDifference (abs):\n{(scores_standard - scores_rope).abs()[0, 0]}")
    print(f"\nMax difference: {(scores_standard - scores_rope).abs().max().item():.4f}")

    # Check correlation
    correlation = F.cosine_similarity(
        scores_standard.flatten(),
        scores_rope.flatten(),
        dim=0
    )
    print(f"\nCorrelation between standard and RoPE scores: {correlation.item():.4f}")

    if correlation < 0.9:
        print("⚠️  RoPE significantly changes attention patterns!")
        print("   Weights designed for standard attention may not work well.")
    else:
        print("✓ RoPE preserves attention patterns reasonably well.")


def analyze_alibi_vs_rope_recency():
    """Analyze difference between pure ALiBi and RoPE + recency bias."""
    print("\n" + "="*70)
    print("Analyzing ALiBi vs RoPE+recency")
    print("="*70)

    S = 4
    slope = 1.0

    # ALiBi bias
    positions = torch.arange(S)
    dist = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs().float()
    alibi = -slope * dist

    print(f"\nALiBi bias matrix (slope={slope}):")
    print(alibi)

    # With RoPE, we have rotation + same recency bias
    # The effective bias is the same, but the base scores (QK^T) are different
    # So the final attention distribution will be different

    # Example: compare two scenarios
    base_scores = torch.randn(S, S)

    # ALiBi
    scores_alibi = base_scores + alibi
    attn_alibi = F.softmax(scores_alibi, dim=-1)

    # RoPE + recency (assume RoPE changes base scores by some amount)
    rope_delta = torch.randn(S, S) * 0.5  # RoPE effect
    scores_rope = (base_scores + rope_delta) + alibi  # Same recency, different base
    attn_rope = F.softmax(scores_rope, dim=-1)

    print(f"\nAttention with ALiBi:\n{attn_alibi}")
    print(f"\nAttention with RoPE+recency:\n{attn_rope}")
    print(f"\nDifference:\n{(attn_alibi - attn_rope).abs()}")
    print(f"\nMax difference: {(attn_alibi - attn_rope).abs().max().item():.4f}")


def analyze_slope_sensitivity():
    """Analyze how sensitive attention is to slope values."""
    print("\n" + "="*70)
    print("Analyzing slope sensitivity")
    print("="*70)

    S = 8
    base_scores = torch.randn(S, S)

    slopes = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    print(f"\nBase scores:\n{base_scores}")

    for slope in slopes:
        positions = torch.arange(S)
        dist = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs().float()
        bias = -slope * dist
        scores = base_scores + bias
        attn = F.softmax(scores, dim=-1)

        # Compute average attention to most recent vs distant tokens
        recent_attn = attn.diagonal(1).mean()  # i attends to i-1
        distant_attn = attn[:-4, :4].mean()  # late tokens attend to early tokens

        print(f"\nSlope {slope:5.1f}: recent={recent_attn:.4f}, distant={distant_attn:.4f}, ratio={recent_attn/distant_attn:.2f}")


if __name__ == "__main__":
    compare_softmax_variants()
    analyze_rope_effect()
    analyze_alibi_vs_rope_recency()
    analyze_slope_sensitivity()

    print("\n" + "="*70)
    print("CONCLUSIONS")
    print("="*70)
    print("""
1. softmax1 vs F.softmax with null key:
   - Mathematically IDENTICAL when null key has score 0 and value 0
   - No weight adjustment needed for F.softmax variant

2. RoPE effect:
   - RoPE rotation changes the base QK^T scores
   - Hand-crafted Q/K weights may not produce intended patterns with RoPE
   - Weight adjustment likely needed for RoPE configurations

3. Recency bias:
   - Adding same bias to RoPE doesn't fully compensate for different base scores
   - The interaction of RoPE rotation + recency is different from pure ALiBi
   - Slope values may need adjustment for RoPE configurations

4. Slope sensitivity:
   - Attention is highly sensitive to slope values
   - Different slopes produce dramatically different attention patterns
   - Slope adjustment is critical for configuration compatibility
""")
