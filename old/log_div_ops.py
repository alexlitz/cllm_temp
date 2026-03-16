"""
Log, reciprocal, division, and mod using standard transformer primitives.

Pipeline:
1. log(x) via SwiGLU FFN (one layer)
2. 1/x via attention with score -log(x) or [0, log(x-1)]
3. a/b = a * (1/b) via SwiGLU
4. a % b = a - floor(a/b) * b
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def silu(x):
    return x * torch.sigmoid(x)


def swiglu_mul(a, b):
    """a * b using SwiGLU identity"""
    a, b = a.float(), b.float()
    return a * silu(b) - a * silu(-b)


class Log2SwiGLUEnumerated(nn.Module):
    """
    Compute log2(x) using enumeration for the fractional part.

    For each bit position k and mantissa bucket b:
        gate_{k,b} = 1 when 2^k * (1 + b/B) <= x < 2^k * (1 + (b+1)/B)
        value_{k,b} = k + log2(1 + (b+0.5)/B)  (precomputed)

    output = sum_{k,b} gate_{k,b} * value_{k,b}

    With B buckets per octave and K bit positions:
        hidden_dim = 4 * K * B (4 SiLU terms per gate)
    """

    def __init__(self, num_bits=16, buckets_per_octave=16, scale=20.0):
        super().__init__()
        self.num_bits = num_bits
        self.buckets = buckets_per_octave
        self.scale = scale

        # Precompute bucket boundaries and log2 values
        import math
        log2_values = []
        for k in range(num_bits):
            for b in range(buckets_per_octave):
                # Bucket b covers mantissa from (1 + b/B) to (1 + (b+1)/B)
                # Use midpoint for value
                m_mid = 1.0 + (b + 0.5) / buckets_per_octave
                log2_values.append(k + math.log2(m_mid))

        self.register_buffer('log2_values', torch.tensor(log2_values))

    def forward(self, x):
        x = x.float()
        B = self.buckets

        result = torch.zeros_like(x)

        idx = 0
        for k in range(self.num_bits):
            pow_k = 2.0 ** k
            bucket_width = pow_k / B  # Width of each bucket at this octave

            # Scale inversely with bucket width for sharp gating
            s = self.scale / bucket_width

            for b in range(B):
                # Lower bound: 2^k * (1 + b/B)
                lower = pow_k * (1.0 + b / B)
                upper = pow_k * (1.0 + (b + 1) / B)

                # gate = threshold(x - lower + 0.5) - threshold(x - upper + 0.5)
                # With adaptive scaling, +/-0.5 in bucket-width units
                half_bw = bucket_width * 0.5

                t1 = x - lower + half_bw
                th1 = (silu(s * (t1 + half_bw)) - silu(s * (t1 - half_bw))) / (s * bucket_width)

                t2 = x - upper + half_bw
                th2 = (silu(s * (t2 + half_bw)) - silu(s * (t2 - half_bw))) / (s * bucket_width)

                gate = th1 - th2
                result = result + gate * self.log2_values[idx]
                idx += 1

        return result


class Log2SwiGLU(nn.Module):
    """
    Compute log2(x) using a single SwiGLU FFN layer.

    For each bit position k:
        gate_k = threshold(x - 2^k) - threshold(x - 2^(k+1))
               = 1 when 2^k <= x < 2^(k+1), 0 otherwise

        value_k = k + 1.4427 * (x/2^k - 1)  (linear approx of fractional part)

        output = sum_k gate_k * value_k

    Each threshold is (SiLU(s*(t+0.5)) - SiLU(s*(t-0.5)))/s
    So gate_k needs 4 SiLU terms, each multiplied by value_k.

    This fits standard SwiGLU: sum of SiLU(gate_i) * up_i
    """

    def __init__(self, num_bits=16, scale=20.0):
        super().__init__()
        self.num_bits = num_bits
        self.scale = scale

        # Precompute powers of 2
        self.register_buffer('powers', 2.0 ** torch.arange(num_bits + 1))

        # Build W_gate and W_up matrices
        # For each k, we have 4 hidden units
        # hidden_dim = 4 * num_bits
        hidden_dim = 4 * num_bits

        # W_gate: (hidden_dim,) - bias terms for gate
        # W_up: (hidden_dim,) - coefficients for up (depends on x)
        # But up = value_k = k + 1.4427 * (x * 2^(-k) - 1) is linear in x
        #        = (k - 1.4427) + 1.4427 * 2^(-k) * x

        # We'll compute this explicitly in forward for clarity
        # In a real implementation, this would be wired into weight matrices

    def forward(self, x):
        """
        Compute log2(x) for x > 0.
        """
        x = x.float()
        s = self.scale

        result = torch.zeros_like(x)

        for k in range(self.num_bits):
            # value_k = k + 1.4427 * (x / 2^k - 1)
            #         = k - 1.4427 + 1.4427 * x / 2^k
            value_k = k - 1.4427 + 1.4427 * x / self.powers[k]

            # gate_k = threshold(x - 2^k + 0.5) - threshold(x - 2^(k+1) + 0.5)
            # threshold(t) = (SiLU(s*(t+0.5)) - SiLU(s*(t-0.5))) / s
            # The +0.5 offset ensures x = 2^k gives threshold ≈ 1

            # For threshold(x - 2^k + 0.5): want 1 when x >= 2^k
            t1 = x - self.powers[k] + 0.5
            th1_plus = silu(s * (t1 + 0.5))
            th1_minus = silu(s * (t1 - 0.5))
            threshold1 = (th1_plus - th1_minus) / s

            # For threshold(x - 2^(k+1) + 0.5): want 1 when x >= 2^(k+1)
            t2 = x - self.powers[k + 1] + 0.5
            th2_plus = silu(s * (t2 + 0.5))
            th2_minus = silu(s * (t2 - 0.5))
            threshold2 = (th2_plus - th2_minus) / s

            gate_k = threshold1 - threshold2

            # In standard SwiGLU, this would be:
            # hidden = SiLU(gate) * up, summed via W_down
            # Here gate_k is already computed as SiLU differences
            # and we multiply by value_k

            result = result + gate_k * value_k

        return result


class Log2SwiGLUPure(nn.Module):
    """
    Pure SwiGLU implementation with explicit weight matrices.

    hidden = up * SiLU(gate)
    output = hidden @ W_down

    For log2, we have 4 hidden units per bit position k.
    """

    def __init__(self, num_bits=16, scale=20.0):
        super().__init__()
        self.num_bits = num_bits
        self.scale = scale
        hidden_dim = 4 * num_bits

        # W_gate: maps x -> gate values for each hidden unit
        # W_up: maps x -> up values for each hidden unit
        # W_down: maps hidden -> output

        # For hidden unit (k, j) where j in {0,1,2,3}:
        # j=0: gate = s*(x - 2^k + 0.5), up = +value_k
        # j=1: gate = s*(x - 2^k - 0.5), up = -value_k
        # j=2: gate = s*(x - 2^(k+1) + 0.5), up = -value_k
        # j=3: gate = s*(x - 2^(k+1) - 0.5), up = +value_k

        # gate = W_gate_w * x + W_gate_b
        W_gate_w = torch.zeros(hidden_dim)
        W_gate_b = torch.zeros(hidden_dim)

        # up = W_up_w * x + W_up_b
        # value_k = (k - 1.4427) + 1.4427 * 2^(-k) * x
        W_up_w = torch.zeros(hidden_dim)
        W_up_b = torch.zeros(hidden_dim)

        W_down = torch.zeros(hidden_dim)

        for k in range(num_bits):
            base_idx = 4 * k
            pow_k = 2.0 ** k
            pow_k1 = 2.0 ** (k + 1)

            # value_k coefficients
            val_w = 1.4427 / pow_k  # coefficient of x
            val_b = k - 1.4427       # constant term

            # threshold(x - 2^k + 0.5) = (SiLU(s*(x - 2^k + 1)) - SiLU(s*(x - 2^k))) / s
            # Unit 0: gate = s*(x - 2^k + 1), up = +value_k
            W_gate_w[base_idx + 0] = scale
            W_gate_b[base_idx + 0] = scale * (-pow_k + 1.0)
            W_up_w[base_idx + 0] = val_w
            W_up_b[base_idx + 0] = val_b
            W_down[base_idx + 0] = 1.0 / scale

            # Unit 1: gate = s*(x - 2^k), up = -value_k
            W_gate_w[base_idx + 1] = scale
            W_gate_b[base_idx + 1] = scale * (-pow_k)
            W_up_w[base_idx + 1] = -val_w
            W_up_b[base_idx + 1] = -val_b
            W_down[base_idx + 1] = 1.0 / scale

            # threshold(x - 2^(k+1) + 0.5) = (SiLU(s*(x - 2^(k+1) + 1)) - SiLU(s*(x - 2^(k+1)))) / s
            # Unit 2: gate = s*(x - 2^(k+1) + 1), up = -value_k
            W_gate_w[base_idx + 2] = scale
            W_gate_b[base_idx + 2] = scale * (-pow_k1 + 1.0)
            W_up_w[base_idx + 2] = -val_w
            W_up_b[base_idx + 2] = -val_b
            W_down[base_idx + 2] = 1.0 / scale

            # Unit 3: gate = s*(x - 2^(k+1)), up = +value_k
            W_gate_w[base_idx + 3] = scale
            W_gate_b[base_idx + 3] = scale * (-pow_k1)
            W_up_w[base_idx + 3] = val_w
            W_up_b[base_idx + 3] = val_b
            W_down[base_idx + 3] = 1.0 / scale

        self.register_buffer('W_gate_w', W_gate_w)
        self.register_buffer('W_gate_b', W_gate_b)
        self.register_buffer('W_up_w', W_up_w)
        self.register_buffer('W_up_b', W_up_b)
        self.register_buffer('W_down', W_down)

    def forward(self, x):
        """Compute log2(x) using pure SwiGLU structure."""
        x = x.float()

        # gate = W_gate_w * x + W_gate_b
        gate = self.W_gate_w * x + self.W_gate_b

        # up = W_up_w * x + W_up_b
        up = self.W_up_w * x + self.W_up_b

        # hidden = up * SiLU(gate)
        hidden = up * silu(gate)

        # output = hidden @ W_down
        output = (hidden * self.W_down).sum()

        return output


class ReciprocalViaAttention(nn.Module):
    """
    Compute 1/x using attention with log.

    softmax([0, log(x-1)]) = [1/x, (x-1)/x]

    Position 0 with value=1 gives output = 1/x
    """

    def __init__(self, num_bits=16, use_enumerated=True):
        super().__init__()
        if use_enumerated:
            self.log_op = Log2SwiGLUEnumerated(num_bits, buckets_per_octave=16)
        else:
            self.log_op = Log2SwiGLU(num_bits)
        self.ln2 = 0.693147  # ln(2)

    def forward(self, x):
        """Compute 1/x for x > 1."""
        x = x.float()

        # log(x-1) = log2(x-1) * ln(2)
        log_x_minus_1 = self.log_op(x - 1) * self.ln2

        # Attention scores: [0, log(x-1)]
        scores = torch.stack([torch.zeros_like(x), log_x_minus_1])

        # Softmax
        weights = F.softmax(scores, dim=0)

        # Values: [1, 0]
        values = torch.tensor([1.0, 0.0])

        # Output = weighted sum = 1/x
        output = weights[0] * values[0] + weights[1] * values[1]

        return output


class ReciprocalViaAttentionV2(nn.Module):
    """
    Alternative: use softmax1 with score = -log(x)

    For score s, softmax1([s]) gives:
    - weight = exp(s) / (1 + exp(s))
    - leftover = 1 / (1 + exp(s))

    With s = -log(x):
    - exp(s) = 1/x
    - weight = (1/x) / (1 + 1/x) = 1/(x+1)
    - leftover = 1 / (1 + 1/x) = x/(x+1)

    Neither is 1/x directly, but we can use:
    leftover * (1 + 1/x) = 1... not helpful

    Actually, use: weight / leftover = (1/x) / (x/(x+1)) * ((x+1)/x) = 1/x * (x+1)/x...

    Simpler: weight * x = (1/(x+1)) * x = x/(x+1), leftover = x/(x+1)
    So weight * x = leftover... interesting but not 1/x.

    Let's stick with the 2-position version.
    """
    pass


class DivisionViaSwiGLU(nn.Module):
    """
    Compute a / b using:
    1. log(b) via SwiGLU
    2. 1/b via attention
    3. a * (1/b) via SwiGLU multiplication
    """

    def __init__(self, num_bits=16):
        super().__init__()
        self.reciprocal = ReciprocalViaAttention(num_bits)

    def forward(self, a, b):
        """Compute a / b (floating point)."""
        inv_b = self.reciprocal(b)
        result = swiglu_mul(a, inv_b)
        return result

    def integer_div(self, a, b):
        """Compute floor(a / b)."""
        return torch.floor(self.forward(a, b)).long()


class DivisionViaEnumeration(nn.Module):
    """
    Compute a / b by enumerating possible quotients.

    For each possible quotient q:
        gate_q = 1 when q*b <= a < (q+1)*b
        output = sum_q (q * gate_q)

    The gate uses: threshold(a - q*b) - threshold(a - (q+1)*b)
    """

    def __init__(self, max_quotient=256, scale=20.0):
        super().__init__()
        self.max_quotient = max_quotient
        self.scale = scale

    def forward(self, a, b):
        """Compute floor(a / b) for positive integers."""
        a, b = a.float(), b.float()
        s = self.scale

        result = torch.zeros_like(a)

        for q in range(self.max_quotient):
            # Gate fires when q*b <= a < (q+1)*b
            lower = q * b
            upper = (q + 1) * b

            # threshold(a - lower + 0.5): ~1 when a >= lower
            t1 = a - lower + 0.5
            th1 = (silu(s * (t1 + 0.5)) - silu(s * (t1 - 0.5))) / s

            # threshold(a - upper + 0.5): ~1 when a >= upper
            t2 = a - upper + 0.5
            th2 = (silu(s * (t2 + 0.5)) - silu(s * (t2 - 0.5))) / s

            gate_q = th1 - th2
            result = result + gate_q * q

        return result.round().long()


class ModViaEnumeration(nn.Module):
    """
    Compute a % b = a - floor(a/b) * b
    """

    def __init__(self, max_quotient=256):
        super().__init__()
        self.div = DivisionViaEnumeration(max_quotient)

    def forward(self, a, b):
        a, b = a.float(), b.float()
        q = self.div(a, b).float()
        result = a - swiglu_mul(q, b)
        return result.round().long()


class ModViaSwiGLU(nn.Module):
    """
    Compute a % b using:
    a % b = a - floor(a/b) * b
    """

    def __init__(self, num_bits=16):
        super().__init__()
        self.div = DivisionViaSwiGLU(num_bits)

    def forward(self, a, b):
        """Compute a % b."""
        a, b = a.float(), b.float()

        # floor(a/b)
        q = torch.floor(self.div.forward(a, b))

        # a - q * b
        qb = swiglu_mul(q, b)
        result = a - qb

        return result.round().long()


def test_log2():
    print("LOG2 via SwiGLU FFN")
    print("=" * 50)
    print()

    import math

    tests = [1, 2, 3, 4, 5, 7, 8, 10, 16, 32, 64, 100, 1000]

    print("Linear approximation (fast but ~10% error):")
    log_linear = Log2SwiGLU(num_bits=16)
    for x in tests:
        result = log_linear(torch.tensor(float(x))).item()
        expected = math.log2(x)
        error = abs(result - expected)
        status = "✓" if error < 0.1 else "✗"
        print(f"  {status} log2({x:5d}) = {result:7.3f}  (expected {expected:7.3f}, error {error:.3f})")

    print()
    print("Enumerated (16 buckets, high accuracy):")
    log_enum = Log2SwiGLUEnumerated(num_bits=16, buckets_per_octave=16)
    for x in tests:
        result = log_enum(torch.tensor(float(x))).item()
        expected = math.log2(x)
        error = abs(result - expected)
        status = "✓" if error < 0.05 else "✗"
        print(f"  {status} log2({x:5d}) = {result:7.3f}  (expected {expected:7.3f}, error {error:.3f})")


def test_reciprocal():
    print()
    print("RECIPROCAL via Attention")
    print("=" * 50)
    print()
    print("Using: softmax([0, log(x-1)]) -> weight[0] = 1/x")
    print()

    recip = ReciprocalViaAttention(num_bits=16)

    tests = [2, 3, 4, 5, 8, 10, 16, 100]

    for x in tests:
        result = recip(torch.tensor(float(x))).item()
        expected = 1.0 / x
        error = abs(result - expected)
        rel_error = error / expected * 100
        status = "✓" if rel_error < 10 else "✗"
        print(f"  {status} 1/{x:3d} = {result:.6f}  (expected {expected:.6f}, error {rel_error:.1f}%)")


def test_division():
    print()
    print("DIVISION via SwiGLU + Attention")
    print("=" * 50)
    print()

    div = DivisionViaSwiGLU(num_bits=16)

    tests = [
        (10, 2, 5),
        (15, 3, 5),
        (100, 10, 10),
        (49, 7, 7),
        (100, 7, 14),
        (1000, 8, 125),
        (255, 16, 15),
    ]

    print("Integer division floor(a/b):")
    for a, b, expected in tests:
        result = div.integer_div(torch.tensor(float(a)), torch.tensor(float(b))).item()
        status = "✓" if result == expected else "✗"
        print(f"  {status} {a:4d} / {b:3d} = {result:4d}  (expected {expected})")


def test_mod():
    print()
    print("MOD via SwiGLU + Attention")
    print("=" * 50)
    print()

    mod = ModViaSwiGLU(num_bits=16)

    tests = [
        (10, 3, 1),
        (17, 5, 2),
        (100, 7, 2),
        (49, 7, 0),
        (255, 16, 15),
        (1000, 37, 1),
        (64, 10, 4),
    ]

    for a, b, expected in tests:
        result = mod(torch.tensor(a), torch.tensor(b)).item()
        status = "✓" if result == expected else "✗"
        print(f"  {status} {a:4d} % {b:3d} = {result:3d}  (expected {expected})")


def test_division_enum():
    print()
    print("DIVISION via Enumeration (direct)")
    print("=" * 50)
    print()

    div = DivisionViaEnumeration(max_quotient=256)

    tests = [
        (10, 2, 5),
        (15, 3, 5),
        (100, 10, 10),
        (49, 7, 7),
        (100, 7, 14),
        (1000, 8, 125),
        (255, 16, 15),
        (17, 5, 3),
        (100, 100, 1),
        (50, 100, 0),
    ]

    print("Integer division floor(a/b):")
    passed = 0
    for a, b, expected in tests:
        result = div(torch.tensor(float(a)), torch.tensor(float(b))).item()
        status = "✓" if result == expected else "✗"
        passed += (result == expected)
        print(f"  {status} {a:4d} / {b:3d} = {result:4d}  (expected {expected})")
    print(f"\nPassed: {passed}/{len(tests)}")


def test_mod_enum():
    print()
    print("MOD via Enumeration")
    print("=" * 50)
    print()

    mod = ModViaEnumeration(max_quotient=256)

    tests = [
        (10, 3, 1),
        (17, 5, 2),
        (100, 7, 2),
        (49, 7, 0),
        (255, 16, 15),
        (1000, 37, 1),
        (64, 10, 4),
        (100, 100, 0),
        (50, 100, 50),
    ]

    passed = 0
    for a, b, expected in tests:
        result = mod(torch.tensor(a), torch.tensor(b)).item()
        status = "✓" if result == expected else "✗"
        passed += (result == expected)
        print(f"  {status} {a:4d} % {b:3d} = {result:3d}  (expected {expected})")
    print(f"\nPassed: {passed}/{len(tests)}")


if __name__ == "__main__":
    test_log2()
    # test_reciprocal()
    # test_division()
    # test_mod()
    test_division_enum()
    test_mod_enum()
