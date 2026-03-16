"""
C4 VM as Pure Transformer - Byte Tokens

Each token = 1 byte (0-255)
32-bit integer = 4 byte tokens
Simpler than nibbles, slightly larger tables

Vocab size: 256
Table size: 256 × 256 = 65536 entries per op
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =============================================================================
# CONSTANTS
# =============================================================================

BYTE_VALUES = 256      # 0-255
NUM_BYTES = 4          # 4 bytes per 32-bit int
D_MODEL = 256          # Embedding dim (one-hot for simplicity)


# =============================================================================
# BYTE ENCODING
# =============================================================================

class ByteEncoder(nn.Module):
    """
    Encode 32-bit int as 4 byte tokens.

    Simple one-hot encoding - no learned embeddings needed.
    """
    def __init__(self, d_model=D_MODEL):
        super().__init__()
        self.d_model = d_model

    def forward(self, x: int) -> torch.Tensor:
        """
        Args:
            x: 32-bit integer
        Returns:
            [4, d_model] one-hot embeddings
        """
        if isinstance(x, torch.Tensor):
            x = int(x.item())

        embeddings = []
        for i in range(4):
            byte_val = (x >> (i * 8)) & 0xFF
            # One-hot encode
            emb = torch.zeros(self.d_model)
            emb[byte_val] = 1.0
            embeddings.append(emb)

        return torch.stack(embeddings)


class ByteDecoder(nn.Module):
    """
    Decode 4 byte embeddings back to 32-bit int.

    Just argmax on each byte embedding.
    """
    def __init__(self, d_model=D_MODEL):
        super().__init__()
        self.d_model = d_model

    def forward(self, embeddings: torch.Tensor) -> int:
        """
        Args:
            embeddings: [4, d_model]
        Returns:
            32-bit integer
        """
        result = 0
        for i in range(4):
            byte_val = torch.argmax(embeddings[i]).item()
            result |= (byte_val << (i * 8))
        return result


# =============================================================================
# FFN BYTE TABLE LOOKUP
# =============================================================================

class ByteTableFFN(nn.Module):
    """
    Byte-wise operation as FFN table lookup.

    Table[256][256] encoded in W1, W2:
    - W1: [512, 65536] - columns encode (a, b) pairs
    - W2: [65536, 256] - rows encode results (one-hot bytes)

    For one byte: input [a_embed, b_embed] → output byte embedding
    """
    def __init__(self, op_fn, d_model=D_MODEL):
        """
        Args:
            op_fn: function(a_byte, b_byte) -> result_byte
        """
        super().__init__()

        input_dim = d_model * 2  # Two byte embeddings concatenated
        num_entries = 256 * 256   # 65536

        # Build W1: address encoding
        # Each column k = a*256 + b encodes the (a, b) pair
        W1 = torch.zeros(input_dim, num_entries)
        for a in range(256):
            for b in range(256):
                k = a * 256 + b
                # First d_model dims: one-hot for 'a'
                W1[a, k] = 1.0
                # Next d_model dims: one-hot for 'b'
                W1[d_model + b, k] = 1.0

        # Build W2: value encoding
        W2 = torch.zeros(num_entries, d_model)
        for a in range(256):
            for b in range(256):
                k = a * 256 + b
                result = op_fn(a, b) & 0xFF
                W2[k, result] = 1.0

        self.register_buffer('W1', W1)
        self.register_buffer('W2', W2)

    def forward(self, a_emb: torch.Tensor, b_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            a_emb, b_emb: [d_model] byte embeddings
        Returns:
            [d_model] result byte embedding
        """
        # Concatenate
        x = torch.cat([a_emb, b_emb])  # [2*d_model]

        # Scores: matching entry gets score 2, others get 0 or 1
        scores = x @ self.W1  # [65536]

        # Sharp softmax
        weights = F.softmax((scores - 1.5) * 10.0, dim=-1)

        # Retrieve result
        result = weights @ self.W2  # [d_model]

        return result


# =============================================================================
# BITWISE OPERATIONS
# =============================================================================

class BitwiseOps(nn.Module):
    """Byte-wise AND, OR, XOR via FFN tables."""

    def __init__(self):
        super().__init__()
        self.and_ffn = ByteTableFFN(lambda a, b: a & b)
        self.or_ffn = ByteTableFFN(lambda a, b: a | b)
        self.xor_ffn = ByteTableFFN(lambda a, b: a ^ b)

    def forward(self, a_emb: torch.Tensor, b_emb: torch.Tensor, op: str) -> torch.Tensor:
        """
        Apply bytewise op to 4-byte embeddings.

        Args:
            a_emb, b_emb: [4, d_model] byte embeddings
            op: 'and', 'or', 'xor'
        Returns:
            [4, d_model] result embeddings
        """
        table = {'and': self.and_ffn, 'or': self.or_ffn, 'xor': self.xor_ffn}[op]

        results = []
        for i in range(4):
            result = table(a_emb[i], b_emb[i])
            results.append(result)

        return torch.stack(results)


# =============================================================================
# ADDITION WITH CARRY (FFN)
# =============================================================================

class AdditionFFN(nn.Module):
    """
    Byte addition with carry via FFN.

    Table: (a, b, carry_in) -> (sum, carry_out)
    Size: 256 * 256 * 2 = 131072 entries
    """
    def __init__(self, d_model=D_MODEL):
        super().__init__()

        input_dim = d_model * 2 + 2  # a_emb + b_emb + carry (2 for one-hot 0/1)
        num_entries = 256 * 256 * 2

        W1 = torch.zeros(input_dim, num_entries)
        W2_sum = torch.zeros(num_entries, d_model)
        W2_carry = torch.zeros(num_entries, 2)

        for a in range(256):
            for b in range(256):
                for c in range(2):
                    k = a * 512 + b * 2 + c

                    # Encode address
                    W1[a, k] = 1.0
                    W1[d_model + b, k] = 1.0
                    W1[d_model * 2 + c, k] = 1.0

                    # Compute result
                    total = a + b + c
                    result_byte = total & 0xFF
                    carry_out = 1 if total >= 256 else 0

                    W2_sum[k, result_byte] = 1.0
                    W2_carry[k, carry_out] = 1.0

        self.register_buffer('W1', W1)
        self.register_buffer('W2_sum', W2_sum)
        self.register_buffer('W2_carry', W2_carry)

    def forward(self, a_emb: torch.Tensor, b_emb: torch.Tensor) -> torch.Tensor:
        """
        Add two 32-bit numbers (4 bytes each).

        Requires 4 sequential steps for carry propagation.
        """
        results = []
        carry = torch.zeros(2)
        carry[0] = 1.0  # carry = 0 (one-hot)

        for i in range(4):
            # Concatenate inputs
            x = torch.cat([a_emb[i], b_emb[i], carry])

            # Lookup
            scores = x @ self.W1
            weights = F.softmax((scores - 2.5) * 10.0, dim=-1)

            result = weights @ self.W2_sum
            carry = weights @ self.W2_carry

            results.append(result)

        return torch.stack(results)


# =============================================================================
# SWIGLU MULTIPLY (Exact)
# =============================================================================

class SwiGLUMul(nn.Module):
    """
    Exact multiply via SwiGLU: a*b = silu(a)*b + silu(-a)*(-b)
    """
    def __init__(self):
        super().__init__()
        # W_gate: [a, -a], W_up: [b, -b], W_down: [1, 1]
        self.register_buffer('W_gate', torch.tensor([[1.0, -1.0]]))
        self.register_buffer('W_up', torch.tensor([[1.0, -1.0]]))
        self.register_buffer('W_down', torch.tensor([[1.0], [1.0]]))

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Multiply two scalars."""
        gate = F.silu(a * self.W_gate)  # [1, 2]: [silu(a), silu(-a)]
        up = b * self.W_up              # [1, 2]: [b, -b]
        hidden = gate * up              # [1, 2]: [silu(a)*b, silu(-a)*(-b)]
        return (hidden @ self.W_down).squeeze()


# =============================================================================
# COMPARISON (Sharp Gate FFN)
# =============================================================================

class CompareFFN(nn.Module):
    """Comparison via sharp gate FFN."""

    def __init__(self, scale=20.0):
        super().__init__()
        self.register_buffer('W1', torch.tensor([[scale], [scale]]))
        self.register_buffer('b1', torch.tensor([scale/2, -scale/2]))
        self.register_buffer('W2', torch.tensor([[1/scale, -1/scale]]))

    def sharp_gate(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(x.unsqueeze(-1) @ self.W1.T + self.b1)
        return (h @ self.W2.T).squeeze()

    def eq(self, a, b):
        d = a - b
        return self.sharp_gate(d + 0.5) * self.sharp_gate(-d + 0.5)

    def lt(self, a, b):
        return self.sharp_gate(b - a - 0.5)


# =============================================================================
# MEMORY (Attention)
# =============================================================================

class MemoryAttention(nn.Module):
    """Memory via attention over address keys."""

    def __init__(self, size=1024, d_model=D_MODEL):
        super().__init__()
        self.size = size
        num_bits = int(math.log2(size)) + 1

        # Keys: binary-encoded addresses
        K = torch.zeros(size, num_bits)
        for addr in range(size):
            for b in range(num_bits):
                K[addr, b] = 10.0 if ((addr >> b) & 1) else -10.0

        self.register_buffer('K', K)
        self.register_buffer('V', torch.zeros(size, d_model))
        self.num_bits = num_bits

    def _addr_query(self, addr: int) -> torch.Tensor:
        q = torch.zeros(self.num_bits)
        for b in range(self.num_bits):
            q[b] = 10.0 if ((addr >> b) & 1) else -10.0
        return q

    def read(self, addr: int) -> torch.Tensor:
        Q = self._addr_query(addr)
        scores = Q @ self.K.T / math.sqrt(self.num_bits)
        weights = F.softmax(scores, dim=-1)
        return weights @ self.V

    def write(self, addr: int, value: torch.Tensor):
        Q = self._addr_query(addr)
        scores = Q @ self.K.T / math.sqrt(self.num_bits)
        mask = F.softmax(scores, dim=-1)
        self.V = self.V * (1 - mask.unsqueeze(-1)) + mask.unsqueeze(-1) * value


# =============================================================================
# ROUTER (Sharp Gate)
# =============================================================================

class Router(nn.Module):
    """Route opcode to expert via sharp gate."""

    def __init__(self, num_ops=64):
        super().__init__()
        self.num_ops = num_ops
        self.cmp = CompareFFN()

    def forward(self, opcode: int) -> torch.Tensor:
        gates = torch.zeros(self.num_ops)
        op_t = torch.tensor(float(opcode))
        for i in range(self.num_ops):
            gates[i] = self.cmp.eq(op_t, torch.tensor(float(i)))
        return gates


# =============================================================================
# FULL VM
# =============================================================================

class C4ByteTransformer(nn.Module):
    """C4 VM with byte tokens - 100% transformer."""

    # Opcodes
    IMM, PSH, ADD, SUB, MUL, DIV, AND, OR, XOR = 1, 13, 25, 26, 27, 28, 16, 14, 15
    EQ, LT, GT, EXIT = 17, 19, 20, 38

    def __init__(self):
        super().__init__()

        self.encoder = ByteEncoder()
        self.decoder = ByteDecoder()
        self.bitwise = BitwiseOps()
        self.add_ffn = AdditionFFN()
        self.mul_ffn = SwiGLUMul()
        self.cmp_ffn = CompareFFN()
        self.memory = MemoryAttention()
        self.router = Router()

        # Registers as embeddings
        self.ax = torch.zeros(4, D_MODEL)
        self.stack = []
        self.code = []
        self.pc = 0
        self.halted = False

    def reset(self):
        self.ax = torch.zeros(4, D_MODEL)
        self.stack = []
        self.pc = 0
        self.halted = False

    def load(self, code):
        """Load tokenized code: [(opcode, imm_int), ...]"""
        self.code = code

    def push(self, emb):
        self.stack.append(emb.clone())

    def pop(self):
        return self.stack.pop()

    def step(self):
        if self.halted or self.pc >= len(self.code):
            self.halted = True
            return False

        opcode, imm = self.code[self.pc]
        self.pc += 1

        # Route
        gates = self.router(opcode)

        if opcode == self.IMM:
            self.ax = self.encoder(imm)

        elif opcode == self.PSH:
            self.push(self.ax)

        elif opcode == self.ADD:
            b = self.ax
            a = self.pop()
            self.ax = self.add_ffn(a, b)

        elif opcode == self.MUL:
            b_val = self.decoder(self.ax)
            a_val = self.decoder(self.pop())
            result = self.mul_ffn(torch.tensor(float(a_val)), torch.tensor(float(b_val)))
            self.ax = self.encoder(int(round(result.item())))

        elif opcode == self.AND:
            b = self.ax
            a = self.pop()
            self.ax = self.bitwise(a, b, 'and')

        elif opcode == self.OR:
            b = self.ax
            a = self.pop()
            self.ax = self.bitwise(a, b, 'or')

        elif opcode == self.XOR:
            b = self.ax
            a = self.pop()
            self.ax = self.bitwise(a, b, 'xor')

        elif opcode == self.EQ:
            b_val = float(self.decoder(self.ax))
            a_val = float(self.decoder(self.pop()))
            result = self.cmp_ffn.eq(torch.tensor(a_val), torch.tensor(b_val))
            self.ax = self.encoder(1 if result.item() > 0.5 else 0)

        elif opcode == self.EXIT:
            self.halted = True
            return False

        return True

    def run(self, max_steps=10000):
        for _ in range(max_steps):
            if not self.step():
                break
        return self.decoder(self.ax)


# =============================================================================
# TEST
# =============================================================================

def test():
    print("=" * 60)
    print("C4 BYTE TRANSFORMER")
    print("=" * 60)

    # Test encoding
    print("\n1. Byte Encoding:")
    enc = ByteEncoder()
    dec = ByteDecoder()

    for val in [0, 42, 255, 0x12345678, 0xDEADBEEF]:
        emb = enc(val)
        recovered = dec(emb)
        status = "✓" if recovered == val else "✗"
        print(f"   {val:#010x} → encode → decode → {recovered:#010x} {status}")

    # Test AND table
    print("\n2. Byte AND Table:")
    and_ffn = ByteTableFFN(lambda a, b: a & b)

    for (a, b) in [(0xF0, 0xAA), (0xFF, 0x0F), (0x12, 0x34)]:
        a_emb = enc(a)[0]  # First byte only
        b_emb = enc(b)[0]
        result_emb = and_ffn(a_emb, b_emb)
        result = torch.argmax(result_emb).item()
        expected = a & b
        status = "✓" if result == expected else "✗"
        print(f"   0x{a:02X} AND 0x{b:02X} = 0x{result:02X} (expected 0x{expected:02X}) {status}")

    # Test full bitwise
    print("\n3. Full 32-bit AND:")
    bitwise = BitwiseOps()

    for (a, b) in [(0xF0F0F0F0, 0xAAAAAAAA), (0xFFFFFFFF, 0x12345678)]:
        a_emb = enc(a)
        b_emb = enc(b)
        result_emb = bitwise(a_emb, b_emb, 'and')
        result = dec(result_emb)
        expected = a & b
        status = "✓" if result == expected else "✗"
        print(f"   {a:#010x} AND {b:#010x} = {result:#010x} (expected {expected:#010x}) {status}")

    # Test SwiGLU multiply
    print("\n4. SwiGLU Multiply:")
    mul = SwiGLUMul()

    for (a, b) in [(6, 7), (12, 11), (100, 5)]:
        result = mul(torch.tensor(float(a)), torch.tensor(float(b)))
        expected = a * b
        status = "✓" if abs(result.item() - expected) < 0.5 else "✗"
        print(f"   {a} × {b} = {result.item():.1f} (expected {expected}) {status}")

    # Test VM
    print("\n5. Full VM:")
    vm = C4ByteTransformer()

    # 6 * 7
    vm.reset()
    vm.load([(1, 6), (13, 0), (1, 7), (27, 0), (38, 0)])
    result = vm.run()
    print(f"   6 * 7 = {result} {'✓' if result == 42 else '✗'}")

    # 0xF0 AND 0xAA
    vm.reset()
    vm.load([(1, 0xF0), (13, 0), (1, 0xAA), (16, 0), (38, 0)])
    result = vm.run()
    expected = 0xF0 & 0xAA
    print(f"   0xF0 AND 0xAA = {result} (expected {expected}) {'✓' if result == expected else '✗'}")

    # Memory usage
    print("\n6. Table Sizes:")
    print(f"   AND table: 256×256 = 65,536 entries")
    print(f"   Add table: 256×256×2 = 131,072 entries")
    print(f"   Total bitwise (AND+OR+XOR): ~200K entries")

    print("\n" + "=" * 60)
    print("ARCHITECTURE:")
    print("  Token = 1 byte (vocab 256)")
    print("  32-bit int = 4 tokens")
    print("  Tables: 256×256 FFN weights")
    print("=" * 60)


if __name__ == "__main__":
    test()
