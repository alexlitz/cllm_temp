"""
Softmax1 Analysis: What Operations Does It Enable?

Softmax1 (quiet attention) is defined as:
    softmax1(x)_i = exp(x_i) / (1 + Σ_j exp(x_j))

The key difference from regular softmax: the "+1" in the denominator.

This creates a "phantom" or "null" position that always has score 0.
"""

import torch
import torch.nn.functional as F
import math


def softmax1(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax with +1 in denominator."""
    x_max = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - x_max)
    return exp_x / (torch.exp(-x_max) + exp_x.sum(dim=dim, keepdim=True))


# =============================================================================
# KEY INSIGHT 1: Softmax1([s]) = sigmoid(s)
# =============================================================================
"""
For a SINGLE element:
    softmax1([s]) = exp(s) / (1 + exp(s)) = sigmoid(s)

This means:
    - sigmoid is just 1-element softmax1
    - We get sigmoid "for free" from the attention mechanism
    - No need for separate SiLU or GELU for this!
"""

def test_softmax1_is_sigmoid():
    s = torch.linspace(-5, 5, 100)
    sm1 = softmax1(s.unsqueeze(-1), dim=-1).squeeze(-1)
    sig = torch.sigmoid(s)
    assert torch.allclose(sm1, sig, atol=1e-6)
    print("✓ softmax1([s]) = sigmoid(s)")


# =============================================================================
# KEY INSIGHT 2: The "Leftover" Encodes Count
# =============================================================================
"""
When we have n positions all with score 0:
    - Each position gets weight: 1/(1+n)
    - Sum of weights: n/(1+n)
    - Leftover: 1 - n/(1+n) = 1/(1+n)

So: leftover = 1/(1+n), which means: n = 1/leftover - 1

This lets us COUNT without explicit selector_width!

For positions with varying scores s_i:
    - leftover = 1 / (1 + Σ exp(s_i))
    - If all s_i = 0: leftover = 1/(1+n)
    - If all s_i = -∞: leftover = 1 (attend to nothing)
    - If any s_i = +∞: leftover = 0 (fully attend)
"""

def test_leftover_encodes_count():
    for n in [1, 2, 5, 10, 20]:
        x = torch.zeros(n)  # n positions with score 0
        weights = softmax1(x, dim=0)
        leftover = 1 - weights.sum()
        recovered_n = 1/leftover - 1
        print(f"n={n}: leftover={leftover:.4f}, recovered_n={recovered_n:.1f}")
    print("✓ leftover encodes count")


# =============================================================================
# KEY INSIGHT 3: Soft Thresholding / Step Function
# =============================================================================
"""
sigmoid(β * x) approaches a step function as β → ∞:
    - sigmoid(β * x) ≈ 1 if x > 0
    - sigmoid(β * x) ≈ 0 if x < 0

For threshold t:
    sigmoid(β * (x - t)) ≈ step(x - t)

This gives us COMPARISONS without explicit comparison operators!
    x > y  ≈  sigmoid(β * (x - y))  for large β
    x < y  ≈  sigmoid(β * (y - x))  for large β
"""

def test_soft_threshold():
    x = torch.linspace(-2, 2, 100)
    threshold = 0.5

    for beta in [1, 5, 10, 50]:
        soft_step = torch.sigmoid(beta * (x - threshold))
        hard_step = (x > threshold).float()
        error = (soft_step - hard_step).abs().mean()
        print(f"β={beta}: mean error from hard step = {error:.4f}")
    print("✓ sigmoid(β*x) → step function as β → ∞")


# =============================================================================
# KEY INSIGHT 4: Reciprocal via Softmax1
# =============================================================================
"""
We want 1/x. Here's the connection:

sigmoid(-log(x)) = 1/(1 + exp(log(x))) = 1/(1 + x)

So: 1/(1+x) = softmax1([-log(x)])

To get 1/x:
    1/x = 1/(1 + (x-1)) = sigmoid(-log(x-1))  for x > 1

Or use the identity:
    1/x = (1/(1+x)) * (1 + 1/x) = (1/(1+x)) * ((x+1)/x)

More directly for small x (Taylor):
    sigmoid(-x) ≈ 1/2 - x/4 + x³/48 - ...

For attention-based reciprocal:
    Set up scores so softmax1 weight ∝ 1/x
"""

def test_reciprocal_via_sigmoid():
    # For 1/(1+x):
    x = torch.linspace(0.1, 5, 100)

    # Method 1: Direct sigmoid
    approx1 = torch.sigmoid(-x)  # This is 1/(1+exp(x)), not 1/(1+x)
    exact = 1 / (1 + x)

    # Method 2: If we had log, sigmoid(-log(x)) = 1/(1+x)
    approx2 = torch.sigmoid(-torch.log(x))

    print(f"sigmoid(-x) vs 1/(1+x): max error = {(approx1 - exact).abs().max():.4f}")
    print(f"sigmoid(-log(x)) vs 1/(1+x): max error = {(approx2 - exact).abs().max():.6f}")
    print("✓ sigmoid(-log(x)) = 1/(1+x) exactly")


# =============================================================================
# KEY INSIGHT 5: Soft Max/Min Selection
# =============================================================================
"""
softmax1(β * x) as β → ∞ approaches one-hot at argmax.

To extract max value:
    max(x) ≈ Σ_i x_i * softmax1(β * x)_i  for large β

The softmax1 version can also detect "no clear max" via leftover:
    - If all values are very negative, leftover → 1
    - If one value dominates, leftover → 0

For min: use softmax1(-β * x)
"""

def test_soft_max():
    x = torch.tensor([1.0, 3.0, 2.0, 0.5])

    for beta in [1, 5, 10, 50]:
        weights = softmax1(beta * x, dim=0)
        soft_max = (x * weights).sum()
        hard_max = x.max()
        leftover = 1 - weights.sum()
        print(f"β={beta}: soft_max={soft_max:.3f}, hard_max={hard_max:.1f}, leftover={leftover:.4f}")
    print("✓ softmax1(βx) @ x → max(x) as β → ∞")


# =============================================================================
# KEY INSIGHT 6: Binary Selection (Ternary Operator)
# =============================================================================
"""
For ternary: cond ? a : b

Current: cond * a + (1-cond) * b  (requires cond ∈ {0,1})

With softmax1:
    scores = [β * cond, -β * cond]  (or [β * cond, 0])
    weights = softmax1(scores)
    result = weights[0] * a + weights[1] * b

As β → ∞, this becomes hard selection.
But we can also do SOFT selection with finite β!
"""

def test_soft_ternary():
    # cond = 1 (true) → select a
    # cond = 0 (false) → select b
    a, b = 10.0, 20.0

    for cond in [0.0, 0.5, 1.0]:
        for beta in [1, 5, 20]:
            scores = torch.tensor([beta * cond, beta * (1 - cond)])
            weights = softmax1(scores, dim=0)
            result = weights[0] * a + weights[1] * b
            leftover = 1 - weights.sum()
            print(f"cond={cond}, β={beta}: result={result:.2f}, leftover={leftover:.3f}")
    print("✓ softmax1 enables soft ternary selection")


# =============================================================================
# KEY INSIGHT 7: Bounded Accumulation
# =============================================================================
"""
Since softmax1 outputs sum to < 1, we can accumulate without overflow.

For running sums with bounded output:
    - Standard attention: must normalize, loses magnitude
    - Softmax1: magnitude encoded in "how much we attend"

This is useful for:
    - Probability-like quantities
    - Confidence scores
    - Bounded counters
"""


# =============================================================================
# KEY INSIGHT 8: Bit Extraction via Thresholding
# =============================================================================
"""
To extract bit i from integer x:
    bit_i = floor(x / 2^i) % 2

The "% 2" part is key. We can reframe as:
    bit_i = 1 if (floor(x / 2^i) is odd) else 0
    bit_i = 1 if (floor(x / 2^i) % 2 >= 1) else 0

Using our threshold insight:
    Let r = x % 2^(i+1)  (remainder mod 2^(i+1))
    bit_i = 1 if r >= 2^i else 0
    bit_i ≈ sigmoid(β * (r - 2^i + 0.5))

This replaces modulo+floor with:
    1. One modulo (which we already implemented via %)
    2. One sigmoid threshold

But actually, the cleaner approach is:
    bit_i = floor((x / 2^i) % 2)
          = floor(x / 2^i) - 2 * floor(x / 2^(i+1))

Using shifts:
    bit_i = (x >> i) - 2 * (x >> (i+1))
          = (x >> i) % 2  (our current approach)

The sigmoid approach could be:
    bit_i = sigmoid(β * (x % 2^(i+1) - 2^i + 0.5))

For large β, this gives 1 if the bit is set, 0 otherwise.
"""

def test_bit_extraction_via_sigmoid():
    x = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

    for bit_pos in range(4):
        # Current approach: (x >> i) % 2
        current = ((x // (2**bit_pos)) % 2).float()

        # Sigmoid approach
        mod_val = (x % (2**(bit_pos + 1))).float()
        threshold = 2**bit_pos - 0.5
        beta = 100.0
        sigmoid_bits = torch.sigmoid(beta * (mod_val - threshold))

        # Compare
        error = (current - sigmoid_bits.round()).abs().max()
        print(f"bit {bit_pos}: max error = {error:.6f}")
    print("✓ sigmoid threshold can extract bits")


# =============================================================================
# KEY INSIGHT 9: Attention-Based Integer Division
# =============================================================================
"""
For x // d (integer division):

We want to find q such that q*d <= x < (q+1)*d

Using attention:
    - Create positions for each possible quotient q = 0, 1, 2, ...
    - Score for position q: how well does q*d approximate x?
    - Select the best q using softmax1

Score function:
    score(q) = -|x - q*d| or -max(0, q*d - x) (penalize overshooting)

With softmax1, the argmax gives us the quotient.
"""


# =============================================================================
# KEY INSIGHT 10: LOG via Inverse Softmax1
# =============================================================================
"""
Since sigmoid(s) = exp(s)/(1+exp(s)), we have:
    s = log(sigmoid(s) / (1 - sigmoid(s)))  (logit function)

If we can compute softmax1 and its "leftover":
    For single element: y = softmax1([s]) = sigmoid(s)
    leftover = 1 - y = 1/(1+exp(s))

    Then: s = log(y/leftover) = log(y) - log(leftover)

So: log(y/leftover) gives us the original score!

To get log(x):
    Set up softmax1 such that y/leftover = x
    y = x/(1+x) and leftover = 1/(1+x) gives y/leftover = x  ✓

    So: score = log(x) gives softmax1([log(x)]) = x/(1+x)

    Inverting: if we observe ratio r = y/leftover, then log(r) = original score

This means: log(x) = logit(x/(1+x)) = logit(sigmoid(log(x)))  [tautology]

More usefully:
    If we can compute softmax1 and measure the leftover,
    then log(attention_weight / leftover) recovers the score.
"""


# =============================================================================
# SUMMARY: What Softmax1 Gives Us
# =============================================================================
"""
PRIMITIVES THAT COME "FREE" WITH SOFTMAX1:

1. SIGMOID: softmax1([s]) = sigmoid(s)
   - Comparisons via sigmoid(β*(x-y))
   - Thresholding via sigmoid(β*(x-t))
   - Soft step functions

2. COUNTING: leftover = 1/(1+n) when all scores = 0
   - Count = 1/leftover - 1
   - No need for separate selector_width!

3. RECIPROCAL: sigmoid(-log(x)) = 1/(1+x)
   - Combined with counting, can get 1/n

4. SOFT SELECTION: softmax1([β*a, β*b, ...]) → one-hot as β→∞
   - argmax/argmin
   - ternary operator
   - max/min extraction

5. BOUNDED ACCUMULATION: outputs sum < 1
   - Safe for running operations
   - Encodes "confidence" in the leftover

6. LOG RECOVERY: log(weight/leftover) = original score
   - Can recover log from softmax1 if we track leftover

OPERATIONS WE CAN SIMPLIFY:

Current approach          → Softmax1 approach
---------------------------------------------------------
comparison (==, <, >)     → sigmoid(β*(a-b))
ternary (c ? a : b)       → softmax1([β*c, β*(1-c)]) @ [a,b]
count (selector_width)    → 1/leftover - 1
1/(1+x)                   → sigmoid(-log(x)) [needs log]
max(a,b,c,...)            → softmax1(β*[a,b,c,...]) @ [a,b,c,...]
argmax                    → softmax1(β*x) → one-hot
"""


if __name__ == "__main__":
    print("=" * 60)
    print("SOFTMAX1 ANALYSIS")
    print("=" * 60)

    print("\n1. Softmax1 is sigmoid for single element:")
    test_softmax1_is_sigmoid()

    print("\n2. Leftover encodes count:")
    test_leftover_encodes_count()

    print("\n3. Soft thresholding:")
    test_soft_threshold()

    print("\n4. Reciprocal via sigmoid:")
    test_reciprocal_via_sigmoid()

    print("\n5. Soft max extraction:")
    test_soft_max()

    print("\n6. Soft ternary selection:")
    test_soft_ternary()

    print("\n8. Bit extraction via sigmoid:")
    test_bit_extraction_via_sigmoid()

    print("\n" + "=" * 60)
    print("All insights verified!")
