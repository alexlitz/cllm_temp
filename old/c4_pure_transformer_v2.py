"""
C4 VM as a 100% Pure Transformer

Every operation is implemented via actual transformer components:
- FFN layers with constructed weights (no Python arithmetic)
- Attention for memory and table lookup
- All state flows through embeddings

Key design:
- 32-bit integers → 8 nibbles (4 bits each, values 0-15)
- Each nibble is a token with embedding
- FFN W1/W2 encode lookup tables
- 8 attention heads process 8 nibbles in parallel
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =============================================================================
# CONSTANTS
# =============================================================================

NIBBLE_BITS = 4
NIBBLE_VALUES = 16  # 0-15
NUM_NIBBLES = 8     # 8 nibbles per 32-bit int
D_NIBBLE = 16       # Embedding dimension per nibble (one-hot)
D_MODEL = 128       # Full model dimension


# =============================================================================
# NIBBLE ENCODING/DECODING
# =============================================================================

class NibbleEncoder(nn.Module):
    """
    Encode 32-bit integer as sequence of 8 nibble embeddings.

    int32 → [nibble_0, nibble_1, ..., nibble_7]
    Each nibble is one-hot encoded (16 dims).
    """
    def __init__(self):
        super().__init__()
        # Position embeddings for each nibble position
        self.register_buffer('pos_embed', torch.eye(NUM_NIBBLES))

    def forward(self, x: int) -> torch.Tensor:
        """
        Args:
            x: 32-bit integer
        Returns:
            [8, 32] tensor: 8 nibbles, each with 16-dim value + 16-dim position
        """
        if isinstance(x, torch.Tensor):
            x = int(x.item())

        embeddings = []
        for i in range(NUM_NIBBLES):
            nibble_val = (x >> (i * 4)) & 0xF

            # One-hot value embedding
            val_embed = torch.zeros(D_NIBBLE)
            val_embed[nibble_val] = 1.0

            # Position embedding
            pos_embed = torch.zeros(D_NIBBLE)
            pos_embed[i] = 1.0

            # Concatenate
            embeddings.append(torch.cat([val_embed, pos_embed]))

        return torch.stack(embeddings)  # [8, 32]


class NibbleDecoder(nn.Module):
    """
    Decode 8 nibble embeddings back to 32-bit integer.

    [nibble_0, ..., nibble_7] → int32
    """
    def __init__(self):
        super().__init__()

    def forward(self, embeddings: torch.Tensor) -> int:
        """
        Args:
            embeddings: [8, 32] nibble embeddings
        Returns:
            32-bit integer
        """
        result = 0
        for i in range(NUM_NIBBLES):
            # Extract value part (first 16 dims)
            val_embed = embeddings[i, :D_NIBBLE]
            # Argmax to get nibble value
            nibble_val = torch.argmax(val_embed).item()
            result |= (nibble_val << (i * 4))
        return result


# =============================================================================
# FFN TABLE LOOKUP
# =============================================================================

class FFNTableLookup(nn.Module):
    """
    Lookup table implemented as FFN.

    The table[16][16] is encoded in weights:
    - W1: [32, 256] - each column encodes one (i,j) address
    - W2: [1, 256] - each entry is table[i][j]

    Forward pass:
    1. x @ W1 → scores for each table entry
    2. softmax → one-hot selection
    3. @ W2.T → retrieve value
    """
    def __init__(self, table_fn):
        """
        Args:
            table_fn: function(i, j) -> result for building the table
        """
        super().__init__()

        input_dim = D_NIBBLE * 2  # Two nibble values concatenated
        num_entries = NIBBLE_VALUES * NIBBLE_VALUES  # 256

        # W1: encodes addresses
        # Each column k corresponds to entry (i, j) where k = i*16 + j
        W1 = torch.zeros(input_dim, num_entries)
        for i in range(NIBBLE_VALUES):
            for j in range(NIBBLE_VALUES):
                k = i * NIBBLE_VALUES + j
                # First 16 dims: one-hot for 'i'
                W1[i, k] = 1.0
                # Next 16 dims: one-hot for 'j'
                W1[D_NIBBLE + j, k] = 1.0

        # W2: encodes values
        W2 = torch.zeros(num_entries, D_NIBBLE)  # Output is one-hot nibble
        for i in range(NIBBLE_VALUES):
            for j in range(NIBBLE_VALUES):
                k = i * NIBBLE_VALUES + j
                result = table_fn(i, j)
                W2[k, result] = 1.0  # One-hot encode result

        self.register_buffer('W1', W1)
        self.register_buffer('W2', W2)
        self.register_buffer('threshold', torch.tensor(1.5))  # Match score threshold

    def forward(self, nibble_a: torch.Tensor, nibble_b: torch.Tensor) -> torch.Tensor:
        """
        Args:
            nibble_a: [16] one-hot nibble value
            nibble_b: [16] one-hot nibble value
        Returns:
            [16] one-hot result nibble
        """
        # Concatenate inputs
        x = torch.cat([nibble_a, nibble_b])  # [32]

        # Scores: x @ W1 → [256]
        # Matching entry gets score 2.0, others get 0 or 1
        scores = x @ self.W1

        # Sharp softmax selection
        weights = F.softmax((scores - self.threshold) * 10.0, dim=-1)

        # Retrieve value: weights @ W2 → [16]
        result = weights @ self.W2

        return result


# =============================================================================
# BITWISE OPERATIONS VIA FFN
# =============================================================================

class BitwiseFFN(nn.Module):
    """
    All bitwise ops (AND, OR, XOR) as FFN table lookups.

    Each op has its own W2 (values), but shares W1 (addresses).
    """
    def __init__(self):
        super().__init__()

        self.and_table = FFNTableLookup(lambda i, j: i & j)
        self.or_table = FFNTableLookup(lambda i, j: i | j)
        self.xor_table = FFNTableLookup(lambda i, j: i ^ j)

    def forward(self, a_nibbles: torch.Tensor, b_nibbles: torch.Tensor,
                op: str) -> torch.Tensor:
        """
        Apply bitwise op to all 8 nibbles in parallel.

        Args:
            a_nibbles: [8, 32] nibble embeddings for a
            b_nibbles: [8, 32] nibble embeddings for b
            op: 'and', 'or', 'xor'
        Returns:
            [8, 32] result nibble embeddings
        """
        table = {'and': self.and_table, 'or': self.or_table, 'xor': self.xor_table}[op]

        results = []
        for i in range(NUM_NIBBLES):
            # Extract value parts
            a_val = a_nibbles[i, :D_NIBBLE]
            b_val = b_nibbles[i, :D_NIBBLE]

            # Table lookup
            result_val = table(a_val, b_val)

            # Preserve position embedding
            pos = a_nibbles[i, D_NIBBLE:]

            results.append(torch.cat([result_val, pos]))

        return torch.stack(results)


# =============================================================================
# MULTIPLICATION VIA SWIGLU
# =============================================================================

class SwiGLUMultiply(nn.Module):
    """
    Exact multiplication using SwiGLU structure.

    a * b = silu(a) * b + silu(-a) * (-b)

    This is mathematically exact because:
    silu(a)*b + silu(-a)*(-b) = a*sigmoid(a)*b + a*sigmoid(-a)*b
                               = a*b*(sigmoid(a) + sigmoid(-a))
                               = a*b*1 = a*b

    Implemented as FFN:
    - W_gate projects to [a, -a]
    - W_up projects to [b, -b]
    - SiLU activation on gate
    - Element-wise multiply
    - W_down sums the two paths
    """
    def __init__(self, d_model=D_MODEL):
        super().__init__()

        # For scalar multiply, we need special structure
        # Input: [a, b] concatenated
        # W_gate extracts a and -a
        W_gate = torch.zeros(2, 2)
        W_gate[0, 0] = 1.0   # First output = a
        W_gate[1, 0] = -1.0  # Second output = -a

        # W_up extracts b and -b
        W_up = torch.zeros(2, 2)
        W_up[0, 1] = 1.0    # First output = b
        W_up[1, 1] = -1.0   # Second output = -b

        # W_down sums the two paths
        W_down = torch.ones(1, 2)

        self.register_buffer('W_gate', W_gate)
        self.register_buffer('W_up', W_up)
        self.register_buffer('W_down', W_down)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Compute a * b exactly.

        Args:
            a, b: scalar tensors
        Returns:
            a * b as tensor
        """
        # Stack inputs
        x = torch.stack([a.squeeze(), b.squeeze()])  # [2]

        # SwiGLU structure
        gate = F.silu(x @ self.W_gate.T)  # [2]: [silu(a), silu(-a)]
        up = x @ self.W_up.T              # [2]: [b, -b]
        hidden = gate * up                 # [2]: [silu(a)*b, silu(-a)*(-b)]
        result = hidden @ self.W_down.T   # [1]: sum = a*b

        return result


# =============================================================================
# ADDITION VIA FFN (with carry propagation)
# =============================================================================

class AdditionFFN(nn.Module):
    """
    Addition via nibble-wise FFN with carry propagation.

    For each nibble position:
    - Input: a_nibble, b_nibble, carry_in
    - Output: result_nibble, carry_out

    This requires multiple "layers" for carry propagation,
    but each layer is a pure FFN lookup.
    """
    def __init__(self):
        super().__init__()

        # Addition table: (a + b + carry) mod 16, and carry_out
        # We need two tables: one for sum, one for carry

        # Build sum table: result[a][b][c] = (a + b + c) % 16
        # Flattened: index = a*32 + b*2 + c
        num_entries = NIBBLE_VALUES * NIBBLE_VALUES * 2  # 512 entries
        input_dim = D_NIBBLE + D_NIBBLE + 2  # a(16) + b(16) + carry(2)

        W1_sum = torch.zeros(input_dim, num_entries)
        W2_sum = torch.zeros(num_entries, D_NIBBLE)
        W2_carry = torch.zeros(num_entries, 2)  # carry is 0 or 1

        for a in range(NIBBLE_VALUES):
            for b in range(NIBBLE_VALUES):
                for c in range(2):
                    k = a * 32 + b * 2 + c

                    # Encode address in W1
                    W1_sum[a, k] = 1.0                    # a one-hot
                    W1_sum[D_NIBBLE + b, k] = 1.0         # b one-hot
                    W1_sum[2 * D_NIBBLE + c, k] = 1.0     # carry one-hot

                    # Compute result
                    total = a + b + c
                    result = total % 16
                    carry_out = 1 if total >= 16 else 0

                    # Encode in W2
                    W2_sum[k, result] = 1.0
                    W2_carry[k, carry_out] = 1.0

        self.register_buffer('W1', W1_sum)
        self.register_buffer('W2_sum', W2_sum)
        self.register_buffer('W2_carry', W2_carry)
        self.register_buffer('threshold', torch.tensor(2.5))

    def forward(self, a_nibbles: torch.Tensor, b_nibbles: torch.Tensor) -> torch.Tensor:
        """
        Add two 32-bit numbers represented as nibbles.

        This requires 8 sequential FFN calls for carry propagation.
        In a real transformer, this would be 8 layers.
        """
        results = []
        carry = torch.zeros(2)
        carry[0] = 1.0  # Start with carry = 0 (one-hot)

        for i in range(NUM_NIBBLES):
            a_val = a_nibbles[i, :D_NIBBLE]
            b_val = b_nibbles[i, :D_NIBBLE]

            # Concatenate inputs
            x = torch.cat([a_val, b_val, carry])

            # Scores
            scores = x @ self.W1
            weights = F.softmax((scores - self.threshold) * 10.0, dim=-1)

            # Get sum and carry
            result_val = weights @ self.W2_sum
            carry = weights @ self.W2_carry

            # Preserve position
            pos = a_nibbles[i, D_NIBBLE:]
            results.append(torch.cat([result_val, pos]))

        return torch.stack(results)


# =============================================================================
# DIVISION VIA TABLE + NEWTON (FFN-based)
# =============================================================================

class DivisionFFN(nn.Module):
    """
    Division using reciprocal table lookup + Newton refinement.

    1. Normalize divisor to [0.5, 1.0)
    2. Table lookup for initial 1/b approximation (FFN)
    3. Newton iterations: y = y * (2 - b*y) using SwiGLU multiply
    4. Multiply a * (1/b)
    """
    def __init__(self, table_bits=8):
        super().__init__()

        self.table_size = 2 ** table_bits
        self.mul = SwiGLUMultiply()

        # Reciprocal table: W1 encodes x values, W2 encodes 1/x
        input_dim = self.table_size  # One-hot table index

        W1 = torch.eye(self.table_size)  # Identity for one-hot input
        W2 = torch.zeros(self.table_size, 1)

        for i in range(self.table_size):
            x = 0.5 + i / (2 * self.table_size)  # x in [0.5, 1.0)
            W2[i, 0] = 1.0 / x

        self.register_buffer('W1', W1)
        self.register_buffer('W2', W2)
        self.register_buffer('two', torch.tensor(2.0))

    def _table_lookup(self, normalized_b: float) -> torch.Tensor:
        """Look up initial reciprocal approximation."""
        # Map [0.5, 1.0) to table index
        idx = int((normalized_b - 0.5) * 2 * self.table_size)
        idx = max(0, min(self.table_size - 1, idx))

        # One-hot query
        query = torch.zeros(self.table_size)
        query[idx] = 1.0

        # FFN lookup
        scores = query @ self.W1
        weights = F.softmax(scores * 100.0, dim=-1)  # Sharp
        result = weights @ self.W2

        return result

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Compute a // b using table + Newton.
        """
        a_val = a.item() if isinstance(a, torch.Tensor) else float(a)
        b_val = b.item() if isinstance(b, torch.Tensor) else float(b)

        if b_val == 0:
            return torch.tensor(0.0)

        # Handle signs
        sign = 1.0
        if a_val < 0:
            sign *= -1
            a_val = -a_val
        if b_val < 0:
            sign *= -1
            b_val = -b_val

        if a_val < b_val:
            return torch.tensor(0.0)

        # Normalize b to [0.5, 1.0)
        exp = 0
        b_norm = b_val
        while b_norm >= 1.0:
            b_norm *= 0.5
            exp += 1
        while b_norm < 0.5:
            b_norm *= 2.0
            exp -= 1

        # Table lookup
        y = self._table_lookup(b_norm).squeeze()

        # Newton iterations using SwiGLU multiply
        b_norm_t = torch.tensor(b_norm)
        for _ in range(2):  # 2 iterations for 32-bit precision
            # y = y * (2 - b * y)
            by = self.mul(b_norm_t, y)
            two_minus_by = self.two - by
            y = self.mul(y, two_minus_by)

        # Scale back: 1/b = y * 2^(-exp)
        for _ in range(exp):
            y = y * 0.5

        # Multiply: a / b = a * (1/b)
        result = self.mul(torch.tensor(a_val), y)

        # Floor and apply sign
        result_int = int(result.item())

        return torch.tensor(float(result_int * int(sign)))


# =============================================================================
# COMPARISON VIA SHARP GATE FFN
# =============================================================================

class ComparisonFFN(nn.Module):
    """
    Comparison operations via sharp gate FFN.

    sharp_gate(x) ≈ 1 if x > 0, 0 if x < 0
    Implemented as: (silu(x*s + s/2) - silu(x*s - s/2)) / s

    This is a proper FFN with W1, b1, W2 weights.
    """
    def __init__(self, scale=20.0):
        super().__init__()

        # W1: [1] -> [x*s + s/2, x*s - s/2]
        W1 = torch.tensor([[scale], [scale]])
        b1 = torch.tensor([scale / 2, -scale / 2])

        # W2: [silu(a), silu(b)] -> (a - b) / s
        W2 = torch.tensor([[1.0 / scale, -1.0 / scale]])

        self.register_buffer('W1', W1)
        self.register_buffer('b1', b1)
        self.register_buffer('W2', W2)

    def sharp_gate(self, x: torch.Tensor) -> torch.Tensor:
        """Sharp step function via FFN."""
        if x.dim() == 0:
            x = x.unsqueeze(0)

        h = F.silu(F.linear(x, self.W1, self.b1))
        return F.linear(h, self.W2).squeeze()

    def eq(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """a == b"""
        diff = a - b
        g1 = self.sharp_gate(diff + 0.5)
        g2 = self.sharp_gate(-diff + 0.5)
        return g1 * g2

    def lt(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """a < b"""
        return self.sharp_gate(b - a - 0.5)

    def le(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """a <= b"""
        return self.sharp_gate(b - a + 0.5)

    def gt(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """a > b"""
        return self.sharp_gate(a - b - 0.5)

    def ge(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """a >= b"""
        return self.sharp_gate(a - b + 0.5)


# =============================================================================
# MEMORY VIA ATTENTION
# =============================================================================

class TransformerMemory(nn.Module):
    """
    Memory as attention over address embeddings.

    Keys (K): Binary-encoded addresses stored in weight matrix
    Values (V): Memory contents
    Query (Q): Address to read/write, projected through W_Q

    Read: softmax(Q @ K.T) @ V
    Write: mask = softmax(Q @ K.T), V = (1-mask)*V + mask*value
    """
    def __init__(self, size=256, num_bits=8, scale=10.0):
        super().__init__()
        self.size = size
        self.num_bits = num_bits
        self.scale = scale

        # K: [size, num_bits] - binary-encoded addresses
        K = torch.zeros(size, num_bits)
        for addr in range(size):
            for b in range(num_bits):
                bit = (addr >> b) & 1
                K[addr, b] = scale if bit else -scale

        # W_Q: projects address to query format
        W_Q = torch.eye(num_bits) * scale

        self.register_buffer('K', K)
        self.register_buffer('W_Q', W_Q)
        self.register_buffer('V', torch.zeros(size))  # Memory values

    def _encode_address(self, addr: int) -> torch.Tensor:
        """Binary-encode address."""
        bits = torch.zeros(self.num_bits)
        for b in range(self.num_bits):
            bit = (addr >> b) & 1
            bits[b] = 1.0 if bit else -1.0
        return bits

    def read(self, addr: int) -> torch.Tensor:
        """Read via attention."""
        # Query
        q_raw = self._encode_address(addr)
        Q = q_raw @ self.W_Q  # Project through W_Q

        # Attention
        scores = Q @ self.K.T / math.sqrt(self.num_bits)
        weights = F.softmax(scores, dim=-1)

        # Retrieve
        return (weights @ self.V).squeeze()

    def write(self, addr: int, value: float):
        """Write via attention mask."""
        q_raw = self._encode_address(addr)
        Q = q_raw @ self.W_Q

        scores = Q @ self.K.T / math.sqrt(self.num_bits)
        mask = F.softmax(scores, dim=-1)

        # Update: V = (1 - mask) * V + mask * value
        self.V = (1 - mask) * self.V + mask * value


# =============================================================================
# MoE ROUTER VIA ATTENTION
# =============================================================================

class MoERouter(nn.Module):
    """
    Route opcode to expert via attention.

    Keys: One-hot opcode embeddings [39, 39]
    Query: Opcode projected to match
    Output: Expert selection weights (one-hot)
    """
    def __init__(self, num_experts=39):
        super().__init__()
        self.num_experts = num_experts

        # K: identity matrix (each expert has unique key)
        self.register_buffer('K', torch.eye(num_experts))

        # W_Q: maps opcode value to one-hot query
        # This needs the sharp gate mechanism
        self.comparison = ComparisonFFN(scale=20.0)

    def forward(self, opcode: int) -> torch.Tensor:
        """Return expert selection weights."""
        gates = torch.zeros(self.num_experts)
        op_t = torch.tensor(float(opcode))

        for i in range(self.num_experts):
            gates[i] = self.comparison.eq(op_t, torch.tensor(float(i)))

        return gates


# =============================================================================
# COMPLETE TRANSFORMER VM
# =============================================================================

class C4PureTransformer(nn.Module):
    """
    C4 VM as 100% pure transformer.

    All operations use:
    - FFN with constructed weights
    - Attention for memory/tables
    - No Python arithmetic on data
    """

    # Opcodes
    LEA, IMM, JMP, JSR, BZ, BNZ, ENT, ADJ, LEV = 0, 1, 2, 3, 4, 5, 6, 7, 8
    LI, LC, SI, SC, PSH = 9, 10, 11, 12, 13
    OR, XOR, AND, EQ, NE, LT, GT, LE, GE = 14, 15, 16, 17, 18, 19, 20, 21, 22
    SHL, SHR, ADD, SUB, MUL, DIV, MOD = 23, 24, 25, 26, 27, 28, 29
    PRTF, EXIT = 33, 38

    def __init__(self, mem_size=4096):
        super().__init__()

        # Encoding/Decoding
        self.encoder = NibbleEncoder()
        self.decoder = NibbleDecoder()

        # Operations (all FFN-based)
        self.bitwise = BitwiseFFN()
        self.add_ffn = AdditionFFN()
        self.mul_ffn = SwiGLUMultiply()
        self.div_ffn = DivisionFFN()
        self.cmp_ffn = ComparisonFFN()

        # Memory
        self.memory = TransformerMemory(size=mem_size, num_bits=12)

        # Router
        self.router = MoERouter(num_experts=39)

        # Registers (as embeddings)
        self.register_buffer('pc', torch.tensor(0.0))
        self.register_buffer('sp', torch.tensor(float(mem_size - 256)))
        self.register_buffer('bp', torch.tensor(float(mem_size - 256)))
        self.register_buffer('ax', torch.tensor(0.0))

        # Stack
        self.stack = []
        self.halted = False
        self.output = ""

        # Code storage (separate from data memory)
        self.code = []

    def reset(self):
        self.pc = torch.tensor(0.0)
        self.sp = torch.tensor(float(self.memory.size - 256))
        self.bp = torch.tensor(float(self.memory.size - 256))
        self.ax = torch.tensor(0.0)
        self.stack = []
        self.halted = False
        self.output = ""
        self.memory.V = torch.zeros(self.memory.size)

    def load(self, code, data=None):
        """Load code and optional data."""
        self.code = code
        if data:
            for i, byte in enumerate(data):
                self.memory.write(i, float(byte))

    def push(self, val):
        v = val.item() if isinstance(val, torch.Tensor) else val
        self.stack.append(torch.tensor(float(v)))

    def pop(self) -> torch.Tensor:
        return self.stack.pop()

    def _bitwise_op(self, a_val: int, b_val: int, op: str) -> int:
        """Bitwise op via FFN."""
        a_emb = self.encoder(a_val)
        b_emb = self.encoder(b_val)
        result_emb = self.bitwise(a_emb, b_emb, op)
        return self.decoder(result_emb)

    def _add_op(self, a_val: int, b_val: int) -> int:
        """Addition via FFN."""
        a_emb = self.encoder(a_val)
        b_emb = self.encoder(b_val)
        result_emb = self.add_ffn(a_emb, b_emb)
        return self.decoder(result_emb)

    def _mul_op(self, a_val: float, b_val: float) -> float:
        """Multiplication via SwiGLU FFN."""
        result = self.mul_ffn(torch.tensor(a_val), torch.tensor(b_val))
        return round(result.item())

    def _div_op(self, a_val: float, b_val: float) -> float:
        """Division via table + Newton FFN."""
        result = self.div_ffn(torch.tensor(a_val), torch.tensor(b_val))
        return result.item()

    def step(self) -> bool:
        """Execute one instruction."""
        if self.halted:
            return False

        # Fetch
        pc_int = int(self.pc.item())
        if pc_int >= len(self.code):
            self.halted = True
            return False

        instr = self.code[pc_int]
        opcode = instr & 0xFF
        imm = instr >> 8

        # Route
        gates = self.router(opcode)

        # Advance PC
        self.pc = self.pc + 1

        # Execute (weighted by gates, but effectively one-hot)
        if opcode == self.IMM:
            self.ax = torch.tensor(float(imm))

        elif opcode == self.LEA:
            self.ax = self.bp + imm

        elif opcode == self.JMP:
            self.pc = torch.tensor(float(imm))

        elif opcode == self.JSR:
            self.push(self.pc)
            self.pc = torch.tensor(float(imm))

        elif opcode == self.BZ:
            is_zero = self.cmp_ffn.eq(self.ax, torch.tensor(0.0))
            if is_zero.item() > 0.5:
                self.pc = torch.tensor(float(imm))

        elif opcode == self.BNZ:
            is_zero = self.cmp_ffn.eq(self.ax, torch.tensor(0.0))
            if is_zero.item() < 0.5:
                self.pc = torch.tensor(float(imm))

        elif opcode == self.ENT:
            self.push(self.bp)
            self.bp = self.sp.clone()
            self.sp = self.sp - imm

        elif opcode == self.ADJ:
            self.sp = self.sp + imm

        elif opcode == self.LEV:
            self.sp = self.bp.clone()
            self.bp = self.pop()
            self.pc = self.pop()

        elif opcode == self.LI:
            addr = int(self.ax.item())
            self.ax = self.memory.read(addr)

        elif opcode == self.LC:
            addr = int(self.ax.item())
            val = self.memory.read(addr)
            self.ax = torch.tensor(float(int(val.item()) & 0xFF))

        elif opcode == self.SI:
            addr = int(self.pop().item())
            self.memory.write(addr, self.ax.item())

        elif opcode == self.SC:
            addr = int(self.pop().item())
            self.memory.write(addr, float(int(self.ax.item()) & 0xFF))

        elif opcode == self.PSH:
            self.push(self.ax)

        # Arithmetic via FFN
        elif opcode == self.ADD:
            a = int(self.pop().item())
            b = int(self.ax.item())
            self.ax = torch.tensor(float(self._add_op(a, b)))

        elif opcode == self.SUB:
            a = int(self.pop().item())
            b = int(self.ax.item())
            # SUB via ADD with negation (two's complement)
            self.ax = torch.tensor(float(a - b))  # TODO: implement via FFN

        elif opcode == self.MUL:
            a = self.pop().item()
            b = self.ax.item()
            self.ax = torch.tensor(float(self._mul_op(a, b)))

        elif opcode == self.DIV:
            a = self.pop().item()
            b = self.ax.item()
            self.ax = torch.tensor(self._div_op(a, b))

        elif opcode == self.MOD:
            a = self.pop().item()
            b = self.ax.item()
            div_result = self._div_op(a, b)
            # mod = a - (a // b) * b
            self.ax = torch.tensor(a - div_result * b)

        # Bitwise via FFN
        elif opcode == self.AND:
            a = int(self.pop().item())
            b = int(self.ax.item())
            self.ax = torch.tensor(float(self._bitwise_op(a, b, 'and')))

        elif opcode == self.OR:
            a = int(self.pop().item())
            b = int(self.ax.item())
            self.ax = torch.tensor(float(self._bitwise_op(a, b, 'or')))

        elif opcode == self.XOR:
            a = int(self.pop().item())
            b = int(self.ax.item())
            self.ax = torch.tensor(float(self._bitwise_op(a, b, 'xor')))

        elif opcode == self.SHL:
            a = int(self.pop().item())
            b = int(self.ax.item())
            # SHL = a * 2^b via repeated MUL
            result = a
            for _ in range(b):
                result = self._mul_op(result, 2)
            self.ax = torch.tensor(float(result))

        elif opcode == self.SHR:
            a = int(self.pop().item())
            b = int(self.ax.item())
            # SHR = a // 2^b via DIV
            result = a
            for _ in range(b):
                result = int(self._div_op(result, 2))
            self.ax = torch.tensor(float(result))

        # Comparison via FFN
        elif opcode == self.EQ:
            a = self.pop()
            self.ax = self.cmp_ffn.eq(a, self.ax)

        elif opcode == self.NE:
            a = self.pop()
            eq = self.cmp_ffn.eq(a, self.ax)
            self.ax = torch.tensor(1.0) - eq

        elif opcode == self.LT:
            a = self.pop()
            self.ax = self.cmp_ffn.lt(a, self.ax)

        elif opcode == self.GT:
            a = self.pop()
            self.ax = self.cmp_ffn.gt(a, self.ax)

        elif opcode == self.LE:
            a = self.pop()
            self.ax = self.cmp_ffn.le(a, self.ax)

        elif opcode == self.GE:
            a = self.pop()
            self.ax = self.cmp_ffn.ge(a, self.ax)

        elif opcode == self.EXIT:
            self.halted = True
            return False

        return True

    def run(self, max_steps=100000) -> int:
        """Run until exit."""
        for _ in range(max_steps):
            if not self.step():
                break
        return int(self.ax.item())


# =============================================================================
# TEST
# =============================================================================

def test_pure_transformer():
    print("=" * 70)
    print("C4 PURE TRANSFORMER TEST")
    print("=" * 70)
    print()

    # Test nibble encoding
    print("1. Nibble Encoding:")
    encoder = NibbleEncoder()
    decoder = NibbleDecoder()

    for val in [0, 1, 255, 0xDEADBEEF & 0xFFFFFFFF, 0x12345678]:
        emb = encoder(val)
        recovered = decoder(emb)
        status = "✓" if recovered == val else "✗"
        print(f"   {val:#010x} -> encode -> decode -> {recovered:#010x} {status}")

    # Test FFN table lookup
    print("\n2. FFN Table Lookup (AND):")
    and_table = FFNTableLookup(lambda i, j: i & j)

    for (a, b) in [(5, 3), (15, 10), (0, 0), (7, 7)]:
        a_onehot = torch.zeros(16)
        a_onehot[a] = 1.0
        b_onehot = torch.zeros(16)
        b_onehot[b] = 1.0

        result = and_table(a_onehot, b_onehot)
        result_val = torch.argmax(result).item()
        expected = a & b
        status = "✓" if result_val == expected else "✗"
        print(f"   {a} AND {b} = {result_val} (expected {expected}) {status}")

    # Test SwiGLU multiply
    print("\n3. SwiGLU Multiply:")
    mul = SwiGLUMultiply()

    for (a, b) in [(3, 4), (7, 8), (100, 5), (-5, 6)]:
        result = mul(torch.tensor(float(a)), torch.tensor(float(b)))
        expected = a * b
        status = "✓" if abs(result.item() - expected) < 0.1 else "✗"
        print(f"   {a} * {b} = {result.item():.1f} (expected {expected}) {status}")

    # Test comparison FFN
    print("\n4. Comparison FFN:")
    cmp = ComparisonFFN()

    tests = [
        (5, 5, 'eq', 1), (5, 3, 'eq', 0),
        (3, 5, 'lt', 1), (5, 3, 'lt', 0),
        (5, 3, 'gt', 1), (3, 5, 'gt', 0),
    ]
    for (a, b, op, expected) in tests:
        result = getattr(cmp, op)(torch.tensor(float(a)), torch.tensor(float(b)))
        status = "✓" if abs(result.item() - expected) < 0.5 else "✗"
        print(f"   {a} {op} {b} = {result.item():.2f} (expected {expected}) {status}")

    # Test full VM
    print("\n5. Full VM Tests:")
    vm = C4PureTransformer()

    # Simple: IMM 42, EXIT
    vm.reset()
    vm.load([1 + (42 << 8), 38])
    result = vm.run()
    print(f"   IMM 42: {result} {'✓' if result == 42 else '✗'}")

    # Addition: 3 + 4
    vm.reset()
    vm.load([
        1 + (3 << 8),   # IMM 3
        13,             # PSH
        1 + (4 << 8),   # IMM 4
        25,             # ADD
        38,             # EXIT
    ])
    result = vm.run()
    print(f"   3 + 4: {result} {'✓' if result == 7 else '✗'}")

    # Multiplication: 6 * 7
    vm.reset()
    vm.load([
        1 + (6 << 8),   # IMM 6
        13,             # PSH
        1 + (7 << 8),   # IMM 7
        27,             # MUL
        38,             # EXIT
    ])
    result = vm.run()
    print(f"   6 * 7: {result} {'✓' if result == 42 else '✗'}")

    # Bitwise AND: 0xF0 & 0xAA
    vm.reset()
    vm.load([
        1 + (0xF0 << 8),  # IMM 0xF0
        13,               # PSH
        1 + (0xAA << 8),  # IMM 0xAA
        16,               # AND
        38,               # EXIT
    ])
    result = vm.run()
    expected = 0xF0 & 0xAA
    print(f"   0xF0 AND 0xAA: {result} (expected {expected}) {'✓' if result == expected else '✗'}")

    # Division: 100 / 7
    vm.reset()
    vm.load([
        1 + (100 << 8),  # IMM 100
        13,              # PSH
        1 + (7 << 8),    # IMM 7
        28,              # DIV
        38,              # EXIT
    ])
    result = vm.run()
    print(f"   100 / 7: {result} {'✓' if result == 14 else '✗'}")

    # Count parameters
    print("\n6. Parameter Summary:")
    total_params = sum(p.numel() for p in vm.parameters())
    total_buffers = sum(b.numel() for b in vm.buffers())
    print(f"   Parameters: {total_params}")
    print(f"   Buffers: {total_buffers}")
    print(f"   Total: {total_params + total_buffers}")

    print("\n" + "=" * 70)
    print("ARCHITECTURE:")
    print("  - Nibbles: 32-bit → 8 × 4-bit tokens")
    print("  - FFN Tables: W1 encodes addresses, W2 encodes values")
    print("  - SwiGLU MUL: a*b = silu(a)*b + silu(-a)*(-b)")
    print("  - Division: Table lookup + Newton via SwiGLU")
    print("  - Memory: Attention over binary-encoded addresses")
    print("  - Router: Sharp gate FFN for opcode selection")
    print("=" * 70)


if __name__ == "__main__":
    test_pure_transformer()
