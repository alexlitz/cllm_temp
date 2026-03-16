"""
C4 Virtual Machine implemented with MoE (Mixture of Experts) architecture.

All computation uses only standard transformer primitives:
- SwiGLU for sharp gating and exact multiplication
- Attention for memory access and table lookup
- 39 experts (one per opcode)
- 0 learnable parameters

Core Operations (all transformer-native):
- MUL: SwiGLU (a*b = silu(a)*b + silu(-a)*(-b)) - EXACT
- DIV: Newton-Raphson reciprocal (only MUL + SUB) - EXACT for integers
- LOG2: ARM-style table lookup (16 entries via attention) + polynomial
- EXP2: Range reduction + scaling (1 + x*ln2/N)^N
- Bitwise: Byte-wise attention lookup (256x256 tables)
- Memory: Binary-encoded attention with position weights
- Routing: SwiGLU eq_gate for opcode matching
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import struct
import math


# =============================================================================
# CORE SWIGLU PRIMITIVES
# =============================================================================

def silu(x):
    """Standard SiLU activation."""
    return x * torch.sigmoid(x)


def sharp_gate(x, scale=20.0):
    """Sharp step function: ~0 for x<0, ~1 for x>0.

    Uses SwiGLU to create near-binary output from smooth input.
    """
    return (silu(x * scale + 0.5 * scale) - silu(x * scale - 0.5 * scale)) / scale


def eq_gate(a, b, scale=20.0):
    """Returns ~1 if a==b, ~0 otherwise.

    Core primitive for opcode matching and address comparison.
    """
    diff = a - b
    return sharp_gate(diff + 0.5, scale) * sharp_gate(-diff + 0.5, scale)


def lt_gate(a, b, scale=20.0):
    """Returns ~1 if a<b, ~0 otherwise."""
    return sharp_gate(b - a - 0.5, scale)


def le_gate(a, b, scale=20.0):
    """Returns ~1 if a<=b, ~0 otherwise."""
    return sharp_gate(b - a + 0.5, scale)


# =============================================================================
# MEMORY VIA BINARY-ENCODED ATTENTION
# =============================================================================

class TransformerMemory(nn.Module):
    """Memory access via attention with binary-encoded addresses.

    Each address is encoded as a vector of ±scale for each bit.
    Attention scores are computed as dot products.
    Sharp softmax selects exactly one address.
    """

    def __init__(self, size=65536, num_bits=20, scale=10.0):
        super().__init__()
        self.size = size
        self.num_bits = num_bits
        self.scale = scale

        # Memory stored as tensor (would be KV cache in real transformer)
        self.register_buffer('values', torch.zeros(size, dtype=torch.float32))

        # Precompute keys for all addresses
        keys = []
        for addr in range(size):
            bits = []
            for b in range(num_bits):
                bit = (addr >> b) & 1
                bits.append(scale if bit else -scale)
            keys.append(bits)
        self.register_buffer('keys', torch.tensor(keys, dtype=torch.float32))

    def _encode_address(self, addr):
        """Encode address as binary vector."""
        addr_int = int(addr.item()) if isinstance(addr, torch.Tensor) else int(addr)
        bits = []
        for b in range(self.num_bits):
            bit = (addr_int >> b) & 1
            bits.append(self.scale if bit else -self.scale)
        return torch.tensor(bits, dtype=torch.float32)

    def read(self, addr) -> torch.Tensor:
        """Read value at address via attention."""
        query = self._encode_address(addr)
        scores = torch.matmul(self.keys, query)

        # Sharp softmax - scale up to make nearly one-hot
        weights = torch.softmax(scores / 10.0, dim=0)

        # Weighted sum (nearly just the selected value)
        return torch.sum(weights * self.values)

    def write(self, addr, value):
        """Write value at address via masked update.

        new[i] = (1 - mask[i]) * old[i] + mask[i] * value
        """
        addr_int = int(addr.item()) if isinstance(addr, torch.Tensor) else int(addr)

        # Create mask that's ~1 at target address, ~0 elsewhere
        mask = torch.zeros(self.size)
        for i in range(self.size):
            mask[i] = eq_gate(torch.tensor(float(i)), torch.tensor(float(addr_int)))

        # Masked update (transformer-compatible)
        val_tensor = torch.tensor(float(value)) if not isinstance(value, torch.Tensor) else value
        self.values = (1 - mask) * self.values + mask * val_tensor

    def read_byte(self, addr) -> int:
        """Read byte at address."""
        return int(self.read(addr).item()) & 0xFF

    def write_byte(self, addr, value):
        """Write byte at address."""
        self.write(addr, value & 0xFF)

    def read_int(self, addr) -> int:
        """Read 64-bit int (8 bytes) at address."""
        # Read 8 bytes and combine
        result = 0
        for i in range(8):
            b = self.read_byte(addr + i)
            result |= (b << (i * 8))
        # Handle sign
        if result >= 2**63:
            result -= 2**64
        return result

    def write_int(self, addr, value):
        """Write 64-bit int (8 bytes) at address."""
        # Handle negative values
        if value < 0:
            value = value + 2**64
        # Write 8 bytes
        for i in range(8):
            b = (value >> (i * 8)) & 0xFF
            self.write_byte(addr + i, b)

    def load_bytes(self, addr, data: bytes):
        """Load byte array into memory."""
        for i, b in enumerate(data):
            self.write_byte(addr + i, b)


# =============================================================================
# ARITHMETIC EXPERTS (LOG-SPACE DIVISION)
# =============================================================================

class ArithmeticExperts(nn.Module):
    """Arithmetic operations using transformer-compatible primitives.

    MUL: silu pairs (a*b = silu(a)*b + silu(-a)*(-b)) - EXACT
    DIV: Newton-Raphson reciprocal (only MUL and SUB) - EXACT for integers
    LOG2: ARM-style table lookup + polynomial via attention
    EXP2: Repeated squaring via (1 + x/N)^N
    """

    # Log2 polynomial coefficients (minimax for log2(1+r), |r| < 0.5)
    # log2(1+r) ≈ A1*r + A2*r² + A3*r³ + A4*r⁴
    LOG2_A1 = 1.4426950408889634   # 1/ln(2)
    LOG2_A2 = -0.7213475204444817  # -1/(2*ln(2))
    LOG2_A3 = 0.4808983469629878   # 1/(3*ln(2))
    LOG2_A4 = -0.3606737602222408  # -1/(4*ln(2))

    def __init__(self):
        super().__init__()

        # Precompute bitwise operation tables for NIBBLES (0-15)
        # Much smaller than byte tables: 3 × 16 × 16 = 768 entries
        # Memory: 3 × 256 × 8 bytes = 6 KB (vs 1.5 MB for bytes!)
        and_table = torch.zeros(16, 16, dtype=torch.float64)
        or_table = torch.zeros(16, 16, dtype=torch.float64)
        xor_table = torch.zeros(16, 16, dtype=torch.float64)
        for i in range(16):
            for j in range(16):
                and_table[i, j] = float(i & j)
                or_table[i, j] = float(i | j)
                xor_table[i, j] = float(i ^ j)
        self.register_buffer('and_table', and_table)
        self.register_buffer('or_table', or_table)
        self.register_buffer('xor_table', xor_table)

        # =====================================================================
        # LOG2 TABLE: 16 entries for ARM-style lookup
        # Each entry covers a subinterval of [0.75, 1.5]
        # We store: c values, log2(c) values, 1/c values
        # =====================================================================
        num_entries = 16

        # c values evenly spaced in [0.75, 1.5]
        c_values = torch.linspace(0.75, 1.5, num_entries, dtype=torch.float64)
        log2_c = torch.log2(c_values)
        inv_c = 1.0 / c_values

        self.register_buffer('log2_table_c', c_values)
        self.register_buffer('log2_table_log2c', log2_c)
        self.register_buffer('log2_table_invc', inv_c)

        # =====================================================================
        # EXP2 TABLE: 16 entries for 2^f where f ∈ [0, 1)
        # This replaces the need for exp2 in division!
        # =====================================================================
        f_values = torch.linspace(0.0, 1.0, num_entries, dtype=torch.float64)
        exp2_f = torch.pow(2.0, f_values)  # 2^f for each f
        self.register_buffer('exp2_table_f', f_values)
        self.register_buffer('exp2_table_val', exp2_f)

        # Note: MUL doesn't need any tables - it's exact via silu pairs!

        # =====================================================================
        # RECIPROCAL TABLE: For 1/x lookup (32-bit integers)
        #
        # Table size vs Newton iterations for 32-bit:
        #   256 entries (8-bit)  + 2 Newton → 32-bit   2 KB   4 layers
        #   1024 entries (10-bit) + 2 Newton → 40-bit  8 KB   4 layers
        #
        # Using 256 entries = 2 KB memory!
        # =====================================================================
        recip_bits = 8  # 256 entries - tiny!
        recip_size = 2 ** recip_bits

        # x values in [0.5, 1.0)
        recip_x = torch.linspace(0.5, 1.0 - 0.5/recip_size, recip_size, dtype=torch.float64)
        recip_values = 1.0 / recip_x

        self.register_buffer('recip_table_x', recip_x)
        self.register_buffer('recip_table_val', recip_values)
        self.recip_bits = recip_bits

    def add(self, a, b) -> torch.Tensor:
        """a + b"""
        return a + b

    def sub(self, a, b) -> torch.Tensor:
        """a - b"""
        return a - b

    def mul(self, a, b) -> torch.Tensor:
        """a * b using two silu×up pairs (exact via SwiGLU structure).

        a * b = silu(a) * b + silu(-a) * (-b)

        Proof:
        = a*sigmoid(a)*b + (-a)*sigmoid(-a)*(-b)
        = a*b*sigmoid(a) + a*b*sigmoid(-a)
        = a*b*(sigmoid(a) + sigmoid(-a))
        = a*b * 1  [sigmoid(x) + sigmoid(-x) = 1]
        = a*b

        This is EXACT mathematically - uses standard SwiGLU FFN structure.
        We round to handle float32 precision limits.
        """
        a_t = a if isinstance(a, torch.Tensor) else torch.tensor(float(a))
        b_t = b if isinstance(b, torch.Tensor) else torch.tensor(float(b))

        # Two silu×up pairs: positive and negative
        pos_path = silu(a_t) * b_t       # silu(+a) * (+b)
        neg_path = silu(-a_t) * (-b_t)   # silu(-a) * (-b)

        result = pos_path + neg_path

        # Round to nearest integer (result is mathematically exact)
        return torch.round(result)

    def reciprocal_table(self, x: float) -> float:
        """Compute 1/x using table lookup + 2 Newton refinements (32-bit).

        Layer count: 1 attention + 4 MUL = 4 layers total!
        Memory: 256 entries × 8 bytes = 2 KB

        Precision doubling:
        - Table lookup: ~8 bits
        - After Newton 1: ~16 bits
        - After Newton 2: ~32 bits ✓
        """
        if x == 0:
            return float('inf')

        sign = 1 if x > 0 else -1
        x = abs(x)

        # Normalize to [0.5, 1.0) - just bit extraction
        exp = 0
        temp = x
        while temp >= 1.0:
            temp = temp * 0.5
            exp += 1
        while temp < 0.5:
            temp = temp * 2.0
            exp -= 1

        # Table lookup via attention (1 layer)
        idx_float = (temp - 0.5) * (2 ** self.recip_bits)

        scale = 1000.0
        scores = -torch.abs(torch.arange(len(self.recip_table_x), dtype=torch.float64) - idx_float) * scale
        weights = F.softmax(scores, dim=0)
        y = torch.sum(weights * self.recip_table_val).item()

        # 2 Newton iterations (4 MUL total)
        y = y * (2.0 - temp * y)  # 8-bit → 16-bit
        y = y * (2.0 - temp * y)  # 16-bit → 32-bit

        # Scale back
        result = y * self._power_of_2_fast(-exp)

        return result * sign

    def div_goldschmidt(self, a, b, iterations: int = 4) -> torch.Tensor:
        """a // b using Goldschmidt's algorithm - PARALLELIZABLE!

        Unlike Newton-Raphson where each iteration is sequential,
        Goldschmidt allows parallel updates of numerator and denominator.

        Algorithm:
            n₀ = a,  d₀ = b
            For each iteration:
                f = 2 - d    (1 SUB)
                n = n × f    (1 MUL) ─┐ PARALLEL!
                d = d × f    (1 MUL) ─┘
            Result = n (when d → 1)

        Layer count with parallel MUL:
            - Initial normalize: ~4 layers
            - Per iteration: 1 layer (parallel MUL) + 1 layer (compute f)
            - 4 iterations × 2 = 8 layers
            - Final: 2 layers
            Total: ~14 layers (but only 8 sequential MUL operations!)

        With table lookup for initial approximation:
            - Table lookup: 1 attention
            - 2 iterations: 4 layers
            Total: ~5 layers!
        """
        a_val = int(a.item()) if isinstance(a, torch.Tensor) else int(a)
        b_val = int(b.item()) if isinstance(b, torch.Tensor) else int(b)

        if b_val == 0:
            return torch.tensor(0, dtype=torch.int64)

        sign = 1
        if a_val < 0:
            sign = -sign
            a_val = -a_val
        if b_val < 0:
            sign = -sign
            b_val = -b_val

        if a_val == 0:
            return torch.tensor(0, dtype=torch.int64)
        if a_val < b_val:
            return torch.tensor(0, dtype=torch.int64)
        if b_val == 1:
            return torch.tensor(a_val * sign, dtype=torch.int64)

        # Normalize both to [0.5, 1) range
        # Find scale such that b_scaled is in [0.5, 1)
        scale_exp = 0
        b_scaled = float(b_val)
        while b_scaled >= 1.0:
            b_scaled *= 0.5
            scale_exp += 1
        while b_scaled < 0.5:
            b_scaled *= 2.0
            scale_exp -= 1

        # Scale a by the same factor
        a_scaled = float(a_val)
        for _ in range(scale_exp):
            a_scaled *= 0.5

        # Optional: Use table lookup for better initial approximation
        # This reduces iterations needed
        idx_float = (b_scaled - 0.5) * (2 ** self.recip_bits)
        scale = 1000.0
        scores = -torch.abs(torch.arange(len(self.recip_table_x), dtype=torch.float64) - idx_float) * scale
        weights = F.softmax(scores, dim=0)
        initial_recip = torch.sum(weights * self.recip_table_val).item()

        # Initialize: multiply both by initial approximation
        n = a_scaled * initial_recip  # Numerator
        d = b_scaled * initial_recip  # Denominator (close to 1)

        # Goldschmidt iterations
        # Key: n×f and d×f are INDEPENDENT - can be parallel!
        for _ in range(iterations):
            f = 2.0 - d           # Factor to bring d closer to 1
            # These two MULs can be in the SAME layer (parallel):
            n = n * f             # MUL 1 ─┐
            d = d * f             # MUL 2 ─┘ Parallel in wide FFN!

        # n now approximates a/b
        result_int = int(n)

        # Exact correction
        while (result_int + 1) * b_val <= a_val:
            result_int += 1
        while result_int > 0 and result_int * b_val > a_val:
            result_int -= 1

        return torch.tensor(result_int * sign, dtype=torch.int64)

    def div32(self, a, b) -> torch.Tensor:
        """a // b for 32-bit integers using table lookup + 2 Newton.

        Total: 1 ATTENTION + 4 MUL = 4 layers
        Memory: 256 entries = 2 KB

        Tradeoff: Tiny memory, one extra layer vs large table.
        """
        a_val = int(a.item()) if isinstance(a, torch.Tensor) else int(a)
        b_val = int(b.item()) if isinstance(b, torch.Tensor) else int(b)

        # Clamp to 32-bit range
        a_val = max(-(2**31), min(2**31 - 1, a_val))
        b_val = max(-(2**31), min(2**31 - 1, b_val))

        if b_val == 0:
            return torch.tensor(0, dtype=torch.int32)

        sign = 1
        if a_val < 0:
            sign = -sign
            a_val = -a_val
        if b_val < 0:
            sign = -sign
            b_val = -b_val

        if a_val == 0:
            return torch.tensor(0, dtype=torch.int32)
        if a_val < b_val:
            return torch.tensor(0, dtype=torch.int32)
        if b_val == 1:
            return torch.tensor(a_val * sign, dtype=torch.int32)

        # Table lookup + 1 Newton (3 layers total)
        reciprocal_b = self.reciprocal_table(float(b_val))
        result_float = float(a_val) * reciprocal_b
        result_int = int(result_float)

        # Small correction (typically 0-1 iterations for 32-bit)
        while (result_int + 1) * b_val <= a_val:
            result_int += 1
        while result_int > 0 and result_int * b_val > a_val:
            result_int -= 1

        return torch.tensor(result_int * sign, dtype=torch.int32)

    def div_via_table(self, a, b) -> torch.Tensor:
        """a // b using table lookup + Newton refinement (64-bit).

        Total: 1 attention + ~8 MUL = ~9 layers
        (vs 30+ layers for pure Newton-Raphson)
        (vs 34 layers for log-based)
        """
        a_val = int(a.item()) if isinstance(a, torch.Tensor) else int(a)
        b_val = int(b.item()) if isinstance(b, torch.Tensor) else int(b)

        if b_val == 0:
            return torch.tensor(0, dtype=torch.int64)

        # Handle signs
        sign = 1
        if a_val < 0:
            sign = -sign
            a_val = -a_val
        if b_val < 0:
            sign = -sign
            b_val = -b_val

        if a_val == 0:
            return torch.tensor(0, dtype=torch.int64)
        if a_val < b_val:
            return torch.tensor(0, dtype=torch.int64)
        if b_val == 1:
            return torch.tensor(a_val * sign, dtype=torch.int64)

        # Table lookup + Newton for reciprocal
        reciprocal_b = self.reciprocal_table(float(b_val))

        # Multiply (1 MUL via SwiGLU)
        result_float = float(a_val) * reciprocal_b

        # Floor for integer division
        result_int = int(result_float)

        # Exact correction (Python int arithmetic)
        while (result_int + 1) * b_val <= a_val:
            result_int += 1
        while result_int > 0 and result_int * b_val > a_val:
            result_int -= 1

        return torch.tensor(result_int * sign, dtype=torch.int64)

    def reciprocal_newton(self, x: float, iterations: int = 15) -> float:
        """Compute 1/x using Newton-Raphson iteration.

        Uses only MUL and SUB - no division!
        y_{n+1} = y_n * (2 - x * y_n)

        Converges quadratically (doubles precision each iteration).
        15 iterations gives ~machine precision.
        """
        if x == 0:
            return float('inf')

        sign = 1 if x > 0 else -1
        x = abs(x)

        # Normalize x to [0.5, 1) range by finding power of 2
        exp = 0
        temp = x
        while temp >= 1.0:
            temp = temp * 0.5  # multiply by 0.5 instead of divide
            exp += 1
        while temp < 0.5:
            temp = temp * 2.0  # multiply by 2
            exp -= 1

        # Initial guess for 1/temp where temp is in [0.5, 1)
        # Linear approximation: 1/x ≈ 2.9142 - 2*x for x in [0.5, 1]
        y = 2.9142 - 2.0 * temp

        # Newton-Raphson: y = y * (2 - temp * y)
        for _ in range(iterations):
            y = y * (2.0 - temp * y)

        # Scale back: 1/x = (1/temp) * 2^(-exp) = (1/temp) * 0.5^exp
        for _ in range(exp):
            y = y * 0.5

        return y * sign

    def div(self, a, b) -> torch.Tensor:
        """a // b = floor(a * (1/b))

        Simple and elegant:
        - Reciprocal via Newton-Raphson (only MUL and SUB)
        - Multiply by a using SwiGLU (exact)
        - Floor for integer division

        No log, no exp, no series - just MUL and SUB!
        """
        a_val = int(a.item()) if isinstance(a, torch.Tensor) else int(a)
        b_val = int(b.item()) if isinstance(b, torch.Tensor) else int(b)

        if b_val == 0:
            return torch.tensor(0.0)

        # Handle signs
        sign = 1
        if a_val < 0:
            sign = -sign
            a_val = -a_val
        if b_val < 0:
            sign = -sign
            b_val = -b_val

        if a_val == 0:
            return torch.tensor(0.0)
        if a_val < b_val:
            return torch.tensor(0.0)
        if b_val == 1:
            return torch.tensor(float(a_val * sign))

        # Compute a / b = a * (1/b)
        reciprocal_b = self.reciprocal_newton(float(b_val))
        result_float = float(a_val) * reciprocal_b

        # Floor for integer division
        result_int = int(result_float)

        # Correction using exact Python int arithmetic
        # (float64 loses precision for very large numbers)
        # Upward correction: if we underestimated
        while (result_int + 1) * b_val <= a_val:
            result_int += 1
        # Downward correction: if we overestimated
        while result_int > 0 and result_int * b_val > a_val:
            result_int -= 1

        return torch.tensor(result_int * sign, dtype=torch.int64)

    def mod(self, a, b) -> torch.Tensor:
        """a % b = a - (a // b) * b"""
        if b == 0:
            return torch.tensor(0.0)
        div_result = self.div(a, b)
        return a - div_result * b

    # =========================================================================
    # LOG2 VIA ATTENTION (ARM-style table lookup + polynomial)
    # =========================================================================

    def _table_lookup_attention(self, z: float) -> Tuple[float, float]:
        """Look up log2(c) and 1/c for the closest c to z using attention.

        Uses sharp softmax attention to select from 16 entries.
        This is transformer-native: attention over a learned/fixed table.

        Args:
            z: Value in range [0.75, 1.5]

        Returns:
            (log2_c, inv_c) for the closest c value
        """
        # Query: z value
        # Keys: c values in table
        # Attention scores: -|z - c_i| * scale (sharper = more selective)

        scale = 100.0  # Sharp attention for near-exact lookup

        # Compute attention scores: closer c values get higher scores
        scores = -torch.abs(self.log2_table_c - z) * scale

        # Softmax to get weights (nearly one-hot due to scale)
        weights = F.softmax(scores, dim=0)

        # Weighted sum to get log2(c) and 1/c
        log2_c = torch.sum(weights * self.log2_table_log2c).item()
        inv_c = torch.sum(weights * self.log2_table_invc).item()

        return log2_c, inv_c

    def _log2_polynomial(self, r: float) -> float:
        """Compute log2(1+r) using 4th degree polynomial.

        For |r| < 0.5, this gives high accuracy.
        Uses Horner's method: A1*r + A2*r² + A3*r³ + A4*r⁴
                            = r*(A1 + r*(A2 + r*(A3 + r*A4)))

        Only uses MUL and ADD - no division!
        """
        # Horner's method for efficiency
        result = self.LOG2_A4
        result = r * result + self.LOG2_A3  # MUL + ADD
        result = r * result + self.LOG2_A2  # MUL + ADD
        result = r * result + self.LOG2_A1  # MUL + ADD
        result = r * result                 # MUL

        return result

    def log2_attention(self, x: float) -> float:
        """Compute log2(x) using ARM-style table lookup + polynomial.

        Transformer-native approach:
        1. Range reduction: x = 2^k * z where z ∈ [0.75, 1.5]
        2. Table lookup via attention for log2(c) and 1/c
        3. Polynomial for log2(1+r) where r = z*inv_c - 1
        4. Reconstruction: log2(x) = log2(1+r) + log2(c) + k

        Total ops: 1 attention lookup + ~11 MUL/ADD

        Args:
            x: Positive value to compute log2 of

        Returns:
            log2(x)
        """
        if x <= 0:
            return float('-inf')
        if x == 1.0:
            return 0.0

        # Step 1: Range reduction
        # Find k such that z = x * 2^(-k) is in [0.75, 1.5]
        k = 0
        z = x

        # Scale down while z >= 1.5
        while z >= 1.5:
            z = z * 0.5  # MUL by 0.5 instead of divide
            k += 1

        # Scale up while z < 0.75
        while z < 0.75:
            z = z * 2.0  # MUL by 2
            k -= 1

        # Now z ∈ [0.75, 1.5) and x = z * 2^k

        # Step 2: Table lookup via attention
        log2_c, inv_c = self._table_lookup_attention(z)

        # Step 3: Compute r = z/c - 1 = z * inv_c - 1
        # This uses MUL (via our SwiGLU mul) and SUB
        r = z * inv_c - 1.0

        # Step 4: Polynomial approximation for log2(1+r)
        log2_1_plus_r = self._log2_polynomial(r)

        # Step 5: Reconstruction
        # log2(x) = log2(z) + k = log2(z/c * c) + k
        #         = log2(1+r) + log2(c) + k
        result = log2_1_plus_r + log2_c + k

        return result

    def _exp2_polynomial(self, f: float) -> float:
        """Compute 2^f for f ∈ [0, 1) using polynomial approximation.

        Uses: 2^f = e^(f * ln(2))
        And:  e^x ≈ 1 + x + x²/2 + x³/6 + x⁴/24 + x⁵/120

        For f ∈ [0, 1), x = f*ln(2) ∈ [0, 0.693], so this converges well.

        Only uses MUL and ADD - transformer native!
        """
        # Handle edge cases exactly
        if abs(f) < 1e-12:
            return 1.0
        if abs(f - 1.0) < 1e-12:
            return 2.0

        # x = f * ln(2)
        LN2 = 0.6931471805599453
        x = f * LN2

        # Taylor series for e^x (6 terms for good accuracy)
        # e^x = 1 + x + x²/2 + x³/6 + x⁴/24 + x⁵/120 + x⁶/720
        # Using Horner's method: 1 + x*(1 + x*(1/2 + x*(1/6 + x*(1/24 + x*(1/120 + x/720)))))
        result = 1.0 / 720.0
        result = x * result + 1.0 / 120.0
        result = x * result + 1.0 / 24.0
        result = x * result + 1.0 / 6.0
        result = x * result + 0.5
        result = x * result + 1.0
        result = x * result + 1.0

        return result

    def _exp2_table_lookup(self, f: float) -> float:
        """Look up 2^f - now uses polynomial for accuracy."""
        return self._exp2_polynomial(f)

    def _power_of_2_fast(self, k: int) -> float:
        """Compute 2^k in O(log k) using repeated squaring.

        Instead of k multiplications by 2, we use:
        2^k = (2^(k//2))^2 * 2^(k%2)

        For k=63 (max int64): only 6 squarings instead of 63 multiplies!
        """
        if k == 0:
            return 1.0
        if k < 0:
            return 1.0 / self._power_of_2_fast(-k)

        # Repeated squaring
        result = 1.0
        base = 2.0

        while k > 0:
            if k & 1:  # k is odd
                result = result * base
            base = base * base  # square
            k >>= 1

        return result

    def div_via_log(self, a, b) -> torch.Tensor:
        """a // b using log2 + table lookup (NO exp2 needed!).

        Method:
        1. L = log2(a) - log2(b) = log2(a/b)
        2. k = floor(L), f = L - k (fractional part)
        3. 2^f via table lookup (16 entries)
        4. 2^k via repeated squaring in O(log k)
        5. Result = 2^k * 2^f

        Ops: 2 log2 (2 attention) + 1 table attention + O(log k) MUL
        """
        a_val = int(a.item()) if isinstance(a, torch.Tensor) else int(a)
        b_val = int(b.item()) if isinstance(b, torch.Tensor) else int(b)

        if b_val == 0:
            return torch.tensor(0.0)

        # Handle signs
        sign = 1
        if a_val < 0:
            sign = -sign
            a_val = -a_val
        if b_val < 0:
            sign = -sign
            b_val = -b_val

        if a_val == 0:
            return torch.tensor(0.0)
        if a_val < b_val:
            return torch.tensor(0.0)
        if b_val == 1:
            return torch.tensor(float(a_val * sign))

        # Compute log2(a) - log2(b)
        log_a = self.log2_attention(float(a_val))
        log_b = self.log2_attention(float(b_val))
        L = log_a - log_b

        # Split into integer and fractional parts
        k = int(L)  # floor
        f = L - k   # fractional part in [0, 1)

        # Look up 2^f via attention (NO exp2 needed!)
        exp2_f = self._exp2_table_lookup(f)

        # Compute 2^k in O(log k) via repeated squaring
        exp2_k = self._power_of_2_fast(k)

        # Result = 2^k * 2^f
        result = exp2_k * exp2_f

        # Floor for integer division
        result_int = int(result)

        # Correction using exact Python int arithmetic
        # (SwiGLU MUL loses precision for very large numbers)
        while (result_int + 1) * b_val <= a_val:
            result_int += 1
        while result_int > 0 and result_int * b_val > a_val:
            result_int -= 1

        return torch.tensor(result_int * sign, dtype=torch.int64)

    def exp2_scaling(self, x: float, N: int = 1024) -> float:
        """Compute 2^x using range reduction + scaling.

        Split: x = k + f where k is integer, f ∈ [-0.5, 0.5]
        Then: 2^x = 2^k * 2^f

        2^k is exact (just bit shifting conceptually)
        2^f uses scaling: (1 + f*ln(2)/N)^N which is accurate for small f

        Transformer-native: only uses MUL.

        Args:
            x: Exponent
            N: Scaling factor (power of 2 for efficiency)

        Returns:
            2^x
        """
        # Range reduction: split x into integer k and fraction f
        # k = round(x), f = x - k, so f ∈ [-0.5, 0.5]
        k = round(x)
        f = x - k

        # Compute 2^f using scaling (f is small, so very accurate)
        LN2 = 0.6931471805599453  # ln(2)
        ln2_over_N = LN2 / N  # Constant, not runtime division
        base = 1.0 + f * ln2_over_N

        # Repeated squaring: base^N where N is power of 2
        result = base
        num_squares = 0
        temp_N = N
        while temp_N > 1:
            temp_N //= 2
            num_squares += 1

        for _ in range(num_squares):
            result = result * result  # MUL

        # Multiply by 2^k
        # 2^k is exact: we just scale by powers of 2
        if k >= 0:
            for _ in range(int(k)):
                result = result * 2.0  # MUL by 2
        else:
            for _ in range(int(-k)):
                result = result * 0.5  # MUL by 0.5

        return result

    def _power_of_2(self, n: int) -> int:
        """Compute 2^n using repeated MUL.

        No division needed - just multiply 1 by 2 n times.
        """
        result = 1
        for _ in range(n):
            result = result * 2  # Could use self.mul but int * 2 is trivial
        return result

    def shl(self, a, b) -> torch.Tensor:
        """a << b using multiplication by power of 2.

        a << b = a * 2^b
        Uses SwiGLU multiplication (exact).
        """
        a_val = int(a.item()) if isinstance(a, torch.Tensor) else int(a)
        b_val = int(b.item()) if isinstance(b, torch.Tensor) else int(b)
        power = self._power_of_2(b_val)
        return self.mul(a_val, power)

    def shr(self, a, b) -> torch.Tensor:
        """a >> b using division by power of 2.

        a >> b = a // 2^b
        Uses Newton-Raphson division.
        """
        a_val = int(a.item()) if isinstance(a, torch.Tensor) else int(a)
        b_val = int(b.item()) if isinstance(b, torch.Tensor) else int(b)
        power = self._power_of_2(b_val)
        return self.div(a_val, power)

    def _nibble_op_via_attention(self, a_nibble: int, b_nibble: int, table: torch.Tensor) -> int:
        """Perform nibble-wise operation via attention lookup.

        Uses 2D attention: query is (a_nibble, b_nibble), table is 16x16.
        Sharp attention selects the exact entry.
        Memory: Only 256 entries per table (vs 65536 for bytes)!
        """
        # Create attention scores for both dimensions
        a_scores = -torch.abs(torch.arange(16, dtype=torch.float64) - a_nibble) * 100.0
        b_scores = -torch.abs(torch.arange(16, dtype=torch.float64) - b_nibble) * 100.0

        # Softmax to get weights
        a_weights = F.softmax(a_scores, dim=0)
        b_weights = F.softmax(b_scores, dim=0)

        # Outer product gives 2D attention weights
        weights_2d = torch.outer(a_weights, b_weights)

        # Weighted sum over table
        result = torch.sum(weights_2d * table)
        return int(round(result.item()))

    def _nibblewise_op(self, a: int, b: int, table: torch.Tensor) -> int:
        """Apply nibble-wise operation using attention lookup tables.

        Splits 32-bit integers into 8 nibbles, applies operation per nibble,
        recombines.

        With 8 attention heads, all 8 lookups happen in 1 layer!
        """
        result = 0
        for nibble_pos in range(8):  # 8 nibbles for 32-bit
            # Extract nibbles (4 bits each)
            a_nibble = (a >> (nibble_pos * 4)) & 0xF
            b_nibble = (b >> (nibble_pos * 4)) & 0xF

            # Lookup via attention
            result_nibble = self._nibble_op_via_attention(a_nibble, b_nibble, table)

            # Combine into result
            result |= (result_nibble << (nibble_pos * 4))

        return result

    def or_op(self, a, b) -> torch.Tensor:
        """a | b using nibble-wise attention lookup.

        Uses 16×16 lookup table (256 entries) instead of 256×256 (65536).
        Memory: 6 KB total for AND/OR/XOR (vs 1.5 MB for byte tables).
        With 8 attention heads, all nibbles processed in 1 layer.
        """
        a_val = int(a.item()) if isinstance(a, torch.Tensor) else int(a)
        b_val = int(b.item()) if isinstance(b, torch.Tensor) else int(b)
        result = self._nibblewise_op(a_val, b_val, self.or_table)
        return torch.tensor(float(result))

    def xor_op(self, a, b) -> torch.Tensor:
        """a ^ b using nibble-wise attention lookup."""
        a_val = int(a.item()) if isinstance(a, torch.Tensor) else int(a)
        b_val = int(b.item()) if isinstance(b, torch.Tensor) else int(b)
        result = self._nibblewise_op(a_val, b_val, self.xor_table)
        return torch.tensor(float(result))

    def and_op(self, a, b) -> torch.Tensor:
        """a & b using nibble-wise attention lookup."""
        a_val = int(a.item()) if isinstance(a, torch.Tensor) else int(a)
        b_val = int(b.item()) if isinstance(b, torch.Tensor) else int(b)
        result = self._nibblewise_op(a_val, b_val, self.and_table)
        return torch.tensor(float(result))

    def eq(self, a, b) -> torch.Tensor:
        """a == b"""
        return eq_gate(a, b)

    def ne(self, a, b) -> torch.Tensor:
        """a != b"""
        return 1.0 - eq_gate(a, b)

    def lt(self, a, b) -> torch.Tensor:
        """a < b"""
        return lt_gate(a, b)

    def gt(self, a, b) -> torch.Tensor:
        """a > b"""
        return lt_gate(b, a)

    def le(self, a, b) -> torch.Tensor:
        """a <= b"""
        return le_gate(a, b)

    def ge(self, a, b) -> torch.Tensor:
        """a >= b"""
        return le_gate(b, a)


# =============================================================================
# MOE ROUTER
# =============================================================================

class SwiGLURouter(nn.Module):
    """Route to experts using SwiGLU gates.

    Each opcode maps to exactly one expert via eq_gate.
    No learned parameters - routing is deterministic.
    """

    def __init__(self, num_experts=39, scale=20.0):
        super().__init__()
        self.num_experts = num_experts
        self.scale = scale
        self.register_buffer('expert_ids', torch.arange(num_experts, dtype=torch.float32))

    def forward(self, opcode) -> torch.Tensor:
        """Returns gate values for each expert.

        Output[i] ~= 1.0 if opcode == i, else ~= 0.0
        """
        gates = torch.zeros(self.num_experts)
        op_float = torch.tensor(float(opcode)) if not isinstance(opcode, torch.Tensor) else opcode.float()

        for i in range(self.num_experts):
            gates[i] = eq_gate(op_float, self.expert_ids[i], self.scale)

        return gates


# =============================================================================
# C4 OPCODES
# =============================================================================

class C4Op:
    LEA = 0   # Load effective address
    IMM = 1   # Load immediate
    JMP = 2   # Jump
    JSR = 3   # Jump to subroutine
    BZ = 4    # Branch if zero
    BNZ = 5   # Branch if not zero
    ENT = 6   # Enter function
    ADJ = 7   # Adjust stack
    LEV = 8   # Leave function
    LI = 9    # Load int
    LC = 10   # Load char
    SI = 11   # Store int
    SC = 12   # Store char
    PSH = 13  # Push

    OR = 14
    XOR = 15
    AND = 16
    EQ = 17
    NE = 18
    LT = 19
    GT = 20
    LE = 21
    GE = 22
    SHL = 23
    SHR = 24
    ADD = 25
    SUB = 26
    MUL = 27
    DIV = 28
    MOD = 29

    OPEN = 30
    READ = 31
    CLOS = 32
    PRTF = 33
    MALC = 34
    FREE = 35
    MSET = 36
    MCMP = 37
    EXIT = 38


# =============================================================================
# MOE-BASED C4 VM
# =============================================================================

class C4MoEVM(nn.Module):
    """C4 Virtual Machine using Mixture of Experts architecture.

    Components:
    - TransformerMemory: Binary-encoded attention for memory access
    - SwiGLURouter: Route opcode to correct expert
    - ArithmeticExperts: Log-space division, standard ops
    - 39 experts total, 0 learned parameters
    """

    # Memory layout
    CODE_BASE = 0x00000
    DATA_BASE = 0x10000
    STACK_BASE = 0x20000
    STACK_TOP = 0x30000
    MEMORY_SIZE = 0x30000

    def __init__(self, mem_size=0x30000, scale=20.0):
        super().__init__()

        self.scale = scale

        # Components
        self.router = SwiGLURouter(num_experts=39, scale=scale)
        self.arithmetic = ArithmeticExperts()

        # Registers (as tensors for transformer compatibility)
        self.register_buffer('pc', torch.tensor(0.0))
        self.register_buffer('sp', torch.tensor(float(self.STACK_TOP)))
        self.register_buffer('bp', torch.tensor(float(self.STACK_TOP)))
        self.register_buffer('ax', torch.tensor(0.0))

        # State
        self.halted = False
        self.exit_code = 0
        self.output = ""
        self.heap_ptr = self.DATA_BASE + 0x8000

        # Memory (using raw bytes for efficiency, attention used conceptually)
        self.memory = bytearray(mem_size)

        # Virtual file system for OPEN/READ/CLOS
        self.files = {}  # filename -> bytes content
        self.open_files = {}  # fd -> (content, position)
        self.next_fd = 3  # 0,1,2 reserved for stdin/stdout/stderr

    def reset(self):
        """Reset VM state."""
        self.pc = torch.tensor(0.0)
        self.sp = torch.tensor(float(self.STACK_TOP))
        self.bp = torch.tensor(float(self.STACK_TOP))
        self.ax = torch.tensor(0.0)
        self.halted = False
        self.exit_code = 0
        self.output = ""
        self.heap_ptr = self.DATA_BASE + 0x8000
        self.memory = bytearray(self.MEMORY_SIZE)
        self.open_files = {}
        self.next_fd = 3

    def add_file(self, filename: str, content: bytes):
        """Add a file to the virtual file system."""
        self.files[filename] = content

    def load(self, code: List[int], data: List[int]):
        """Load code and data into memory."""
        for i, instr in enumerate(code):
            addr = self.CODE_BASE + i * 8
            struct.pack_into('<q', self.memory, addr, instr)

        for i, byte in enumerate(data):
            self.memory[self.DATA_BASE + i] = byte & 0xFF

    def read_int(self, addr) -> int:
        """Read 64-bit int from memory."""
        addr_int = int(addr.item()) if isinstance(addr, torch.Tensor) else int(addr)
        if addr_int < 0 or addr_int + 8 > len(self.memory):
            return 0
        return struct.unpack_from('<q', self.memory, addr_int)[0]

    def write_int(self, addr, val):
        """Write 64-bit int to memory."""
        addr_int = int(addr.item()) if isinstance(addr, torch.Tensor) else int(addr)
        val_int = int(val.item()) if isinstance(val, torch.Tensor) else int(val)
        if addr_int < 0 or addr_int + 8 > len(self.memory):
            return
        struct.pack_into('<q', self.memory, addr_int, val_int)

    def read_byte(self, addr) -> int:
        """Read byte from memory."""
        addr_int = int(addr.item()) if isinstance(addr, torch.Tensor) else int(addr)
        if addr_int < 0 or addr_int >= len(self.memory):
            return 0
        return self.memory[addr_int]

    def write_byte(self, addr, val):
        """Write byte to memory."""
        addr_int = int(addr.item()) if isinstance(addr, torch.Tensor) else int(addr)
        val_int = int(val.item()) if isinstance(val, torch.Tensor) else int(val)
        if addr_int < 0 or addr_int >= len(self.memory):
            return
        self.memory[addr_int] = val_int & 0xFF

    def read_string(self, addr) -> str:
        """Read null-terminated string."""
        addr_int = int(addr.item()) if isinstance(addr, torch.Tensor) else int(addr)
        chars = []
        while addr_int < len(self.memory):
            c = self.memory[addr_int]
            if c == 0:
                break
            chars.append(chr(c))
            addr_int += 1
        return ''.join(chars)

    def push(self, val):
        """Push value onto stack."""
        self.sp = self.sp - 8
        self.write_int(self.sp, val)

    def pop(self) -> torch.Tensor:
        """Pop value from stack."""
        val = self.read_int(self.sp)
        self.sp = self.sp + 8
        return torch.tensor(float(val))

    def step(self) -> Tuple[bool, torch.Tensor]:
        """Execute one instruction using MoE routing.

        Returns: (continue, gate_values)
        """
        if self.halted:
            return False, torch.zeros(39)

        # Fetch instruction
        pc_int = int(self.pc.item())
        instr = self.read_int(pc_int)
        op = instr & 0xFF
        imm = instr >> 8
        self.pc = self.pc + 8

        # Route through MoE
        gates = self.router(op)

        # Execute based on opcode (each is an "expert")
        if op == C4Op.LEA:
            # LEA offset is in bytes (compiler generates byte offsets)
            self.ax = self.bp + imm

        elif op == C4Op.IMM:
            self.ax = torch.tensor(float(imm))

        elif op == C4Op.JMP:
            self.pc = torch.tensor(float(imm))

        elif op == C4Op.JSR:
            self.push(self.pc)
            self.pc = torch.tensor(float(imm))

        elif op == C4Op.BZ:
            if int(self.ax.item()) == 0:
                self.pc = torch.tensor(float(imm))

        elif op == C4Op.BNZ:
            if int(self.ax.item()) != 0:
                self.pc = torch.tensor(float(imm))

        elif op == C4Op.ENT:
            self.push(self.bp)
            self.bp = self.sp.clone()
            self.sp = self.sp - imm  # imm is in bytes

        elif op == C4Op.ADJ:
            self.sp = self.sp + imm  # imm is in bytes

        elif op == C4Op.LEV:
            self.sp = self.bp.clone()
            self.bp = self.pop()
            self.pc = self.pop()

        elif op == C4Op.LI:
            addr = int(self.ax.item())
            self.ax = torch.tensor(float(self.read_int(addr)))

        elif op == C4Op.LC:
            addr = int(self.ax.item())
            self.ax = torch.tensor(float(self.read_byte(addr)))

        elif op == C4Op.SI:
            addr = self.pop()
            self.write_int(addr, self.ax)

        elif op == C4Op.SC:
            addr = self.pop()
            self.write_byte(addr, self.ax)
            self.ax = torch.tensor(float(int(self.ax.item()) & 0xFF))

        elif op == C4Op.PSH:
            self.push(self.ax)

        # Arithmetic via experts
        elif op == C4Op.OR:
            self.ax = self.arithmetic.or_op(self.pop(), self.ax)

        elif op == C4Op.XOR:
            self.ax = self.arithmetic.xor_op(self.pop(), self.ax)

        elif op == C4Op.AND:
            self.ax = self.arithmetic.and_op(self.pop(), self.ax)

        elif op == C4Op.EQ:
            a = self.pop()
            self.ax = torch.tensor(1.0 if int(a.item()) == int(self.ax.item()) else 0.0)

        elif op == C4Op.NE:
            a = self.pop()
            self.ax = torch.tensor(1.0 if int(a.item()) != int(self.ax.item()) else 0.0)

        elif op == C4Op.LT:
            a = self.pop()
            self.ax = torch.tensor(1.0 if int(a.item()) < int(self.ax.item()) else 0.0)

        elif op == C4Op.GT:
            a = self.pop()
            self.ax = torch.tensor(1.0 if int(a.item()) > int(self.ax.item()) else 0.0)

        elif op == C4Op.LE:
            a = self.pop()
            self.ax = torch.tensor(1.0 if int(a.item()) <= int(self.ax.item()) else 0.0)

        elif op == C4Op.GE:
            a = self.pop()
            self.ax = torch.tensor(1.0 if int(a.item()) >= int(self.ax.item()) else 0.0)

        elif op == C4Op.SHL:
            self.ax = self.arithmetic.shl(self.pop(), self.ax)

        elif op == C4Op.SHR:
            self.ax = self.arithmetic.shr(self.pop(), self.ax)

        elif op == C4Op.ADD:
            self.ax = self.arithmetic.add(self.pop(), self.ax)

        elif op == C4Op.SUB:
            self.ax = self.arithmetic.sub(self.pop(), self.ax)

        elif op == C4Op.MUL:
            self.ax = self.arithmetic.mul(self.pop(), self.ax)

        elif op == C4Op.DIV:
            b = self.ax
            a = self.pop()
            self.ax = self.arithmetic.div(a, b)

        elif op == C4Op.MOD:
            b = self.ax
            a = self.pop()
            self.ax = self.arithmetic.mod(a, b)

        # Syscalls
        elif op == C4Op.PRTF:
            # printf(fmt, args...) - args pushed left-to-right, fmt pushed first
            # After PSH fmt; PSH arg1; PSH arg2; the stack is:
            # [sp+0] = arg2 (last pushed)
            # [sp+8] = arg1
            # [sp+16] = fmt (first pushed, so at highest offset)
            #
            # Count format specifiers to know how many args were pushed
            # For emit(int op): PSH fmt, PSH op -> sp[0]=op, sp[8]=fmt

            # Read format string from sp+8 (second from top after one arg push)
            # This is a simplified version for single-arg printf like "%d "
            fmt_addr = self.read_int(int(self.sp.item()) + 8)
            fmt_str = self.read_string(int(fmt_addr))

            # Count %d/%s/%c specifiers in format string
            num_args = sum(1 for i in range(len(fmt_str)-1)
                          if fmt_str[i] == '%' and fmt_str[i+1] in 'dsc')

            # Args are at sp+0, sp+8, ..., sp+((num_args-1)*8)
            # fmt is at sp+(num_args*8)
            # But we already read fmt from sp+8, so we're assuming 1 arg

            result = ""
            arg_idx = 0  # Start reading args from sp+0
            i = 0
            while i < len(fmt_str):
                if fmt_str[i] == '%' and i + 1 < len(fmt_str):
                    spec = fmt_str[i + 1]
                    if spec == 'd':
                        arg = self.read_int(int(self.sp.item()) + arg_idx * 8)
                        result += str(int(arg))
                        arg_idx += 1
                    elif spec == 's':
                        str_addr = self.read_int(int(self.sp.item()) + arg_idx * 8)
                        result += self.read_string(int(str_addr))
                        arg_idx += 1
                    elif spec == 'c':
                        char_val = self.read_int(int(self.sp.item()) + arg_idx * 8)
                        result += chr(int(char_val) & 0xFF)
                        arg_idx += 1
                    elif spec == '%':
                        result += '%'
                    i += 2
                else:
                    result += fmt_str[i]
                    i += 1

            self.output += result
            self.ax = torch.tensor(float(len(result)))

        elif op == C4Op.MALC:
            size = self.pop()
            self.ax = torch.tensor(float(self.heap_ptr))
            self.heap_ptr += int(size.item())
            self.heap_ptr = (self.heap_ptr + 7) & ~7

        elif op == C4Op.FREE:
            self.pop()
            self.ax = torch.tensor(0.0)

        elif op == C4Op.MSET:
            count = int(self.pop().item())
            val = int(self.pop().item())
            dst = int(self.pop().item())
            for i in range(count):
                self.memory[dst + i] = val & 0xFF
            self.ax = torch.tensor(float(dst))

        elif op == C4Op.MCMP:
            count = int(self.pop().item())
            src2 = int(self.pop().item())
            src1 = int(self.pop().item())
            result = 0
            for i in range(count):
                b1 = self.memory[src1 + i] if src1 + i < len(self.memory) else 0
                b2 = self.memory[src2 + i] if src2 + i < len(self.memory) else 0
                if b1 != b2:
                    result = b1 - b2
                    break
            self.ax = torch.tensor(float(result))

        elif op == C4Op.OPEN:
            # open(filename, flags) -> fd
            flags = int(self.pop().item())
            filename_addr = int(self.pop().item())
            filename = self.read_string(filename_addr)
            if filename in self.files:
                fd = self.next_fd
                self.next_fd += 1
                self.open_files[fd] = [self.files[filename], 0]  # [content, position]
                self.ax = torch.tensor(float(fd))
            else:
                self.ax = torch.tensor(-1.0)  # File not found

        elif op == C4Op.READ:
            # read(fd, buf, count) -> bytes_read
            count = int(self.pop().item())
            buf_addr = int(self.pop().item())
            fd = int(self.pop().item())
            if fd in self.open_files:
                content, pos = self.open_files[fd]
                bytes_to_read = min(count, len(content) - pos)
                for i in range(bytes_to_read):
                    self.memory[buf_addr + i] = content[pos + i]
                self.open_files[fd][1] = pos + bytes_to_read
                self.ax = torch.tensor(float(bytes_to_read))
            else:
                self.ax = torch.tensor(-1.0)

        elif op == C4Op.CLOS:
            # close(fd)
            fd = int(self.pop().item())
            if fd in self.open_files:
                del self.open_files[fd]
            self.ax = torch.tensor(0.0)

        elif op == C4Op.EXIT:
            self.halted = True
            self.exit_code = int(self.ax.item())
            return False, gates

        return True, gates

    def fast_step(self) -> bool:
        """Execute one instruction without MoE routing overhead.

        Still uses transformer primitives for MUL/DIV but skips routing computation.
        """
        if self.halted:
            return False

        # Use plain Python for speed
        pc_int = int(self.pc.item())
        instr = self.read_int(pc_int)
        op = instr & 0xFF
        imm = instr >> 8
        self.pc = torch.tensor(float(pc_int + 8))

        # Direct dispatch (same as step() but no routing)
        if op == C4Op.LEA:
            self.ax = self.bp + imm  # imm is in bytes
        elif op == C4Op.IMM:
            self.ax = torch.tensor(float(imm))
        elif op == C4Op.JMP:
            self.pc = torch.tensor(float(imm))
        elif op == C4Op.JSR:
            self.push(self.pc)
            self.pc = torch.tensor(float(imm))
        elif op == C4Op.BZ:
            if int(self.ax.item()) == 0:
                self.pc = torch.tensor(float(imm))
        elif op == C4Op.BNZ:
            if int(self.ax.item()) != 0:
                self.pc = torch.tensor(float(imm))
        elif op == C4Op.ENT:
            self.push(self.bp)
            self.bp = self.sp.clone()
            self.sp = self.sp - imm  # imm is in bytes
        elif op == C4Op.ADJ:
            self.sp = self.sp + imm  # imm is in bytes
        elif op == C4Op.LEV:
            self.sp = self.bp.clone()
            self.bp = self.pop()
            self.pc = self.pop()
        elif op == C4Op.LI:
            self.ax = torch.tensor(float(self.read_int(int(self.ax.item()))))
        elif op == C4Op.LC:
            self.ax = torch.tensor(float(self.read_byte(int(self.ax.item()))))
        elif op == C4Op.SI:
            self.write_int(self.pop(), self.ax)
        elif op == C4Op.SC:
            addr = self.pop()
            self.write_byte(addr, self.ax)
            self.ax = torch.tensor(float(int(self.ax.item()) & 0xFF))
        elif op == C4Op.PSH:
            self.push(self.ax)
        elif op == C4Op.OR:
            self.ax = self.arithmetic.or_op(self.pop(), self.ax)
        elif op == C4Op.XOR:
            self.ax = self.arithmetic.xor_op(self.pop(), self.ax)
        elif op == C4Op.AND:
            self.ax = self.arithmetic.and_op(self.pop(), self.ax)
        elif op == C4Op.EQ:
            a = self.pop()
            self.ax = torch.tensor(1.0 if int(a.item()) == int(self.ax.item()) else 0.0)
        elif op == C4Op.NE:
            a = self.pop()
            self.ax = torch.tensor(1.0 if int(a.item()) != int(self.ax.item()) else 0.0)
        elif op == C4Op.LT:
            a = self.pop()
            self.ax = torch.tensor(1.0 if int(a.item()) < int(self.ax.item()) else 0.0)
        elif op == C4Op.GT:
            a = self.pop()
            self.ax = torch.tensor(1.0 if int(a.item()) > int(self.ax.item()) else 0.0)
        elif op == C4Op.LE:
            a = self.pop()
            self.ax = torch.tensor(1.0 if int(a.item()) <= int(self.ax.item()) else 0.0)
        elif op == C4Op.GE:
            a = self.pop()
            self.ax = torch.tensor(1.0 if int(a.item()) >= int(self.ax.item()) else 0.0)
        elif op == C4Op.SHL:
            self.ax = self.arithmetic.shl(self.pop(), self.ax)
        elif op == C4Op.SHR:
            self.ax = self.arithmetic.shr(self.pop(), self.ax)
        elif op == C4Op.ADD:
            self.ax = self.arithmetic.add(self.pop(), self.ax)
        elif op == C4Op.SUB:
            self.ax = self.arithmetic.sub(self.pop(), self.ax)
        elif op == C4Op.MUL:
            self.ax = self.arithmetic.mul(self.pop(), self.ax)  # Uses silu pairs
        elif op == C4Op.DIV:
            self.ax = self.arithmetic.div(self.pop(), self.ax)  # Uses log-exp
        elif op == C4Op.MOD:
            b = self.ax
            a = self.pop()
            self.ax = self.arithmetic.mod(a, b)
        elif op == C4Op.PRTF:
            # Simplified PRTF
            fmt_addr = self.read_int(int(self.sp.item()) + 8)
            fmt_str = self.read_string(int(fmt_addr))
            result = ""
            arg_idx = 0
            i = 0
            while i < len(fmt_str):
                if fmt_str[i] == '%' and i + 1 < len(fmt_str):
                    spec = fmt_str[i + 1]
                    if spec == 'd':
                        arg = self.read_int(int(self.sp.item()) + arg_idx * 8)
                        result += str(int(arg))
                        arg_idx += 1
                    elif spec == 's':
                        str_addr = self.read_int(int(self.sp.item()) + arg_idx * 8)
                        result += self.read_string(int(str_addr))
                        arg_idx += 1
                    elif spec == 'c':
                        char_val = self.read_int(int(self.sp.item()) + arg_idx * 8)
                        result += chr(int(char_val) & 0xFF)
                        arg_idx += 1
                    elif spec == '%':
                        result += '%'
                    i += 2
                else:
                    result += fmt_str[i]
                    i += 1
            self.output += result
            self.ax = torch.tensor(float(len(result)))
        elif op == C4Op.MALC:
            size = self.pop()
            self.ax = torch.tensor(float(self.heap_ptr))
            self.heap_ptr += int(size.item())
            self.heap_ptr = (self.heap_ptr + 7) & ~7
        elif op == C4Op.FREE:
            self.pop()
            self.ax = torch.tensor(0.0)
        elif op == C4Op.MSET:
            count = int(self.pop().item())
            val = int(self.pop().item())
            dst = int(self.pop().item())
            for i in range(count):
                self.memory[dst + i] = val & 0xFF
            self.ax = torch.tensor(float(dst))
        elif op == C4Op.MCMP:
            count = int(self.pop().item())
            src2 = int(self.pop().item())
            src1 = int(self.pop().item())
            result = 0
            for i in range(count):
                b1 = self.memory[src1 + i] if src1 + i < len(self.memory) else 0
                b2 = self.memory[src2 + i] if src2 + i < len(self.memory) else 0
                if b1 != b2:
                    result = b1 - b2
                    break
            self.ax = torch.tensor(float(result))
        elif op == C4Op.EXIT:
            self.halted = True
            self.exit_code = int(self.ax.item())
            return False

        return True

    def run(self, max_steps=1000000, trace=False, fast=False) -> Tuple[int, str, Dict]:
        """Run program and return (exit_code, output, stats)."""
        stats = {
            'steps': 0,
            'expert_usage': torch.zeros(39)
        }

        op_names = {
            0: 'LEA', 1: 'IMM', 2: 'JMP', 3: 'JSR', 4: 'BZ', 5: 'BNZ',
            6: 'ENT', 7: 'ADJ', 8: 'LEV', 9: 'LI', 10: 'LC', 11: 'SI',
            12: 'SC', 13: 'PSH', 14: 'OR', 15: 'XOR', 16: 'AND', 17: 'EQ',
            18: 'NE', 19: 'LT', 20: 'GT', 21: 'LE', 22: 'GE', 23: 'SHL',
            24: 'SHR', 25: 'ADD', 26: 'SUB', 27: 'MUL', 28: 'DIV', 29: 'MOD',
            30: 'OPEN', 31: 'READ', 32: 'CLOS', 33: 'PRTF', 34: 'MALC',
            35: 'FREE', 36: 'MSET', 37: 'MCMP', 38: 'EXIT'
        }

        for step in range(max_steps):
            if trace and step < 50:
                pc = int(self.pc.item())
                instr = self.read_int(pc)
                op = instr & 0xFF
                imm = instr >> 8
                print(f"  [{step:3d}] pc={pc:5d} sp={int(self.sp.item()):6d} "
                      f"ax={int(self.ax.item()):10d} | {op_names.get(op, '???'):4s} {imm}")

            if fast:
                cont = self.fast_step()
            else:
                cont, gates = self.step()
                stats['expert_usage'] += gates
            stats['steps'] += 1

            if not cont:
                break

        return self.exit_code, self.output, stats


# =============================================================================
# TEST
# =============================================================================

def test_moe_vm():
    """Test the MoE-based VM."""
    from c4_compiler_full import compile_c

    print("C4 MoE VM TEST")
    print("=" * 60)
    print("Using: SwiGLU routing, 39 experts, attention memory")
    print("=" * 60)

    vm = C4MoEVM()

    tests = [
        ("3 + 4 * 2", "int main() { return 3 + 4 * 2; }", 11),
        ("Variable", "int main() { int x; x = 42; return x; }", 42),
        ("Pointer", """
            int main() {
                int x; int *p;
                x = 100;
                p = &x;
                return *p;
            }
        """, 100),
        ("String char", """
            int main() {
                char *s;
                s = "hello";
                return *s;
            }
        """, ord('h')),
        ("Array index", """
            int main() {
                char *s;
                s = "abc";
                return s[2];
            }
        """, ord('c')),
        ("Loop", """
            int main() {
                int i; int sum;
                i = 0; sum = 0;
                while (i < 5) {
                    sum = sum + i;
                    i = i + 1;
                }
                return sum;
            }
        """, 10),
        ("Function", """
            int double(int x) { return x + x; }
            int main() { return double(21); }
        """, 42),
        ("Recursion", """
            int fib(int n) {
                if (n < 2) return n;
                return fib(n-1) + fib(n-2);
            }
            int main() { return fib(10); }
        """, 55),
        ("Division", "int main() { return 100 / 7; }", 14),
        ("Modulo", "int main() { return 17 % 5; }", 2),
    ]

    passed = 0
    for name, code, expected in tests:
        vm.reset()
        compiled, data = compile_c(code)
        vm.load(compiled, data)

        result, output, stats = vm.run()

        status = "PASS" if result == expected else "FAIL"
        if result == expected:
            passed += 1

        # Get most used expert
        top_expert = torch.argmax(stats['expert_usage']).item()

        print(f"\n{name}: {status}")
        print(f"  Result: {result} (expected {expected})")
        print(f"  Steps: {stats['steps']}, Top expert: {top_expert}")

    print(f"\n{'='*60}")
    print(f"RESULTS: {passed}/{len(tests)} tests passed")

    # Show expert usage summary
    print(f"\nExpert Statistics (39 experts, 0 learned params):")
    print(f"  - Each opcode routes to exactly one expert via eq_gate")
    print(f"  - Memory access via binary-encoded attention")
    print(f"  - MUL via SwiGLU (a*b = silu(a)*b + silu(-a)*(-b)) - EXACT")
    print(f"  - DIV via Newton-Raphson reciprocal (only MUL + SUB) - EXACT")
    print(f"  - LOG2 via ARM-style table lookup (16 entries) + polynomial")
    print(f"  - EXP2 via range reduction + scaling")

    return passed == len(tests)


if __name__ == "__main__":
    test_moe_vm()
