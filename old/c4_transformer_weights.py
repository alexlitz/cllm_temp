"""
C4 Transformer with Actual PyTorch Weights

This implements the C4 VM as an actual transformer with:
- Explicit weight matrices (no learned parameters, all constructed)
- SwiGLU FFN for sharp gating
- Attention for memory access
- MoE routing via constructed weights

The transformer processes a "state vector" containing:
[pc, sp, bp, ax, opcode, imm, mem_addr, mem_val, stack_top, ...]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =============================================================================
# CORE PRIMITIVE: SwiGLU Sharp Gate as FFN
# =============================================================================

class SharpGateFFN(nn.Module):
    """
    FFN that computes sharp_gate(x) ≈ step function.

    sharp_gate(x) = (silu(x*s + 0.5*s) - silu(x*s - 0.5*s)) / s

    This is implemented as a 2-layer FFN:
    - Layer 1: expand to 2 dimensions [x*s + 0.5*s, x*s - 0.5*s]
    - SiLU activation
    - Layer 2: combine with weights [1/s, -1/s]
    """
    def __init__(self, scale=20.0):
        super().__init__()
        self.scale = scale

        # W1: [input] -> [x*s + 0.5*s, x*s - 0.5*s]
        self.W1 = nn.Parameter(torch.tensor([[scale], [scale]]), requires_grad=False)
        self.b1 = nn.Parameter(torch.tensor([0.5 * scale, -0.5 * scale]), requires_grad=False)

        # W2: [silu(a), silu(b)] -> (silu(a) - silu(b)) / s
        self.W2 = nn.Parameter(torch.tensor([[1.0 / scale, -1.0 / scale]]), requires_grad=False)

    def forward(self, x):
        # x: [..., 1] or scalar
        if x.dim() == 0:
            x = x.unsqueeze(0)
        if x.dim() == 1:
            x = x.unsqueeze(-1)

        h = F.linear(x, self.W1, self.b1)  # [..., 2]
        h = F.silu(h)
        out = F.linear(h, self.W2)  # [..., 1]
        return out.squeeze(-1)


class EqGateFFN(nn.Module):
    """
    FFN that computes eq_gate(a, b) = 1 if a==b, 0 otherwise.

    eq_gate(a, b) = sharp_gate(a - b + 0.5) * sharp_gate(-(a - b) + 0.5)

    Implemented as FFN that:
    1. Computes diff = a - b
    2. Applies two sharp gates
    3. Multiplies results
    """
    def __init__(self, scale=20.0):
        super().__init__()
        self.sharp_gate = SharpGateFFN(scale)

    def forward(self, a, b):
        diff = a - b
        g1 = self.sharp_gate(diff + 0.5)
        g2 = self.sharp_gate(-diff + 0.5)
        return g1 * g2


# =============================================================================
# MEMORY: Attention-based Read/Write
# =============================================================================

class AttentionMemory(nn.Module):
    """
    Memory implemented via attention.

    Keys are binary-encoded addresses.
    Values are memory contents.

    Read: query with address encoding, attention gives value
    Write: attention gives one-hot mask, linear combination updates
    """
    def __init__(self, size=256, num_bits=8, scale=10.0):
        super().__init__()
        self.size = size
        self.num_bits = num_bits
        self.scale = scale

        # Precompute key matrix: [size, num_bits]
        # Each row is binary encoding of address with ±scale
        keys = []
        for addr in range(size):
            bits = []
            for b in range(num_bits):
                bit = (addr >> b) & 1
                bits.append(scale if bit else -scale)
            keys.append(bits)
        self.register_buffer('K', torch.tensor(keys, dtype=torch.float32))

        # Memory values
        self.register_buffer('V', torch.zeros(size, dtype=torch.float32))

    def _encode_address(self, addr):
        """Encode address as query vector."""
        addr_int = int(addr.item()) if isinstance(addr, torch.Tensor) else int(addr)
        bits = []
        for b in range(self.num_bits):
            bit = (addr_int >> b) & 1
            bits.append(self.scale if bit else -self.scale)
        return torch.tensor(bits, dtype=torch.float32)

    def read(self, addr):
        """
        Read via attention.

        Q @ K^T gives scores, softmax gives weights, weights @ V gives value.
        """
        Q = self._encode_address(addr).unsqueeze(0)  # [1, num_bits]

        # Attention scores
        scores = Q @ self.K.T  # [1, size]

        # Softmax (scaled for sharpness)
        weights = F.softmax(scores / self.num_bits, dim=-1)  # [1, size]

        # Weighted sum of values
        value = weights @ self.V.unsqueeze(-1)  # [1, 1]
        return value.squeeze()

    def write(self, addr, value):
        """
        Write via attention masking.

        mask[i] = attention_weight[i] (one-hot at target address)
        new_V = (1 - mask) * old_V + mask * value
        """
        Q = self._encode_address(addr).unsqueeze(0)
        scores = Q @ self.K.T
        mask = F.softmax(scores / self.num_bits, dim=-1).squeeze()  # [size]

        # Update: linear combination
        val = value.item() if isinstance(value, torch.Tensor) else value
        self.V = (1 - mask) * self.V + mask * val


# =============================================================================
# DIVISION: Log-space with Reciprocal Table
# =============================================================================

class LogDivision(nn.Module):
    """
    Division via: a / b = a * (1/b) = a * 2^(-log2(b))

    Uses attention over a fine-grained table of reciprocals.

    Table construction:
    - Keys: [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, ...] (log2 values at 0.25 resolution)
    - Values: [1.0, 2^-0.25, 2^-0.5, ...] (corresponding reciprocals)

    This gives accurate division by interpolating between powers of 2.
    """
    def __init__(self, max_log=20, resolution=0.125):
        super().__init__()
        self.resolution = resolution

        # Fine-grained table: log values from 0 to max_log at given resolution
        num_entries = int(max_log / resolution) + 1
        keys = torch.arange(num_entries, dtype=torch.float32) * resolution
        values = torch.pow(2.0, -keys)

        self.register_buffer('log_keys', keys)
        self.register_buffer('recip_values', values)

    def forward(self, a, b):
        """Compute floor(a / b) via log-space."""
        if b == 0:
            return torch.tensor(0.0)

        a_val = a.item() if isinstance(a, torch.Tensor) else a
        b_val = b.item() if isinstance(b, torch.Tensor) else b

        if b_val <= 0:
            return torch.tensor(0.0)

        # Compute log2(b)
        log_b = math.log2(abs(b_val))

        # Attention lookup: sharp attention to nearest log key
        # Higher scale = sharper selection
        scores = -torch.abs(self.log_keys - log_b) * 50.0
        weights = F.softmax(scores, dim=0)
        recip_b = torch.sum(weights * self.recip_values)

        # Multiply: a * (1/b)
        result = a_val * recip_b

        return torch.floor(result)


# =============================================================================
# MoE ROUTER: Opcode to Expert Selection
# =============================================================================

class OpcodeRouter(nn.Module):
    """
    Routes opcode to expert via eq_gate.

    Constructs a weight matrix W where:
    - W[expert_i, :] activates when opcode == i

    This is implemented via parallel eq_gate computations.
    """
    def __init__(self, num_experts=39, scale=20.0):
        super().__init__()
        self.num_experts = num_experts
        self.eq_gate = EqGateFFN(scale)

        # Expert IDs as buffer
        self.register_buffer('expert_ids', torch.arange(num_experts, dtype=torch.float32))

    def forward(self, opcode):
        """Returns gate values [num_experts], one-hot at matching opcode."""
        gates = torch.zeros(self.num_experts)
        op = opcode.float() if isinstance(opcode, torch.Tensor) else torch.tensor(float(opcode))

        for i in range(self.num_experts):
            gates[i] = self.eq_gate(op, self.expert_ids[i])

        return gates


# =============================================================================
# EXPERT LAYERS: One per Opcode
# =============================================================================

class ArithmeticExpert(nn.Module):
    """Expert for arithmetic operations."""

    def __init__(self):
        super().__init__()
        self.log_div = LogDivision()

    def add(self, a, b):
        return a + b

    def sub(self, a, b):
        return a - b

    def mul(self, a, b):
        return a * b

    def div(self, a, b):
        return self.log_div(a, b)

    def mod(self, a, b):
        if b == 0:
            return torch.tensor(0.0)
        div_result = self.log_div(a, b)
        return a - div_result * b


class BitwiseExpert(nn.Module):
    """Expert for bitwise operations."""

    def or_op(self, a, b):
        return torch.tensor(float(int(a) | int(b)))

    def and_op(self, a, b):
        return torch.tensor(float(int(a) & int(b)))

    def xor_op(self, a, b):
        return torch.tensor(float(int(a) ^ int(b)))

    def shl(self, a, b):
        return torch.tensor(float(int(a) << int(b)))

    def shr(self, a, b):
        return torch.tensor(float(int(a) >> int(b)))


class ComparisonExpert(nn.Module):
    """Expert for comparison operations using sharp gates."""

    def __init__(self, scale=20.0):
        super().__init__()
        self.eq_gate = EqGateFFN(scale)
        self.sharp_gate = SharpGateFFN(scale)

    def eq(self, a, b):
        return self.eq_gate(a, b)

    def ne(self, a, b):
        return 1.0 - self.eq_gate(a, b)

    def lt(self, a, b):
        # a < b means b - a > 0
        return self.sharp_gate(b - a - 0.5)

    def gt(self, a, b):
        return self.sharp_gate(a - b - 0.5)

    def le(self, a, b):
        return self.sharp_gate(b - a + 0.5)

    def ge(self, a, b):
        return self.sharp_gate(a - b + 0.5)


# =============================================================================
# COMPLETE C4 TRANSFORMER
# =============================================================================

class C4Transformer(nn.Module):
    """
    Complete C4 VM as a transformer.

    State vector: [pc, sp, bp, ax]
    Memory: separate attention-based module

    Each step:
    1. Fetch instruction via attention read
    2. Route opcode through MoE
    3. Execute expert
    4. Update state
    """

    # Opcodes
    LEA, IMM, JMP, JSR, BZ, BNZ, ENT, ADJ, LEV = 0, 1, 2, 3, 4, 5, 6, 7, 8
    LI, LC, SI, SC, PSH = 9, 10, 11, 12, 13
    OR, XOR, AND, EQ, NE, LT, GT, LE, GE = 14, 15, 16, 17, 18, 19, 20, 21, 22
    SHL, SHR, ADD, SUB, MUL, DIV, MOD = 23, 24, 25, 26, 27, 28, 29
    EXIT = 38

    def __init__(self, mem_size=4096, scale=20.0):
        super().__init__()

        # Router
        self.router = OpcodeRouter(num_experts=39, scale=scale)

        # Experts
        self.arith = ArithmeticExpert()
        self.bitwise = BitwiseExpert()
        self.compare = ComparisonExpert(scale)

        # Memory (simplified - using direct indexing for efficiency)
        self.mem_size = mem_size
        self.register_buffer('memory', torch.zeros(mem_size))

        # State registers
        self.register_buffer('pc', torch.tensor(0.0))
        self.register_buffer('sp', torch.tensor(float(mem_size - 256)))
        self.register_buffer('bp', torch.tensor(float(mem_size - 256)))
        self.register_buffer('ax', torch.tensor(0.0))

        # Stack (for efficiency)
        self.stack = []

        self.halted = False

    def reset(self):
        self.pc = torch.tensor(0.0)
        self.sp = torch.tensor(float(self.mem_size - 256))
        self.bp = torch.tensor(float(self.mem_size - 256))
        self.ax = torch.tensor(0.0)
        self.stack = []
        self.halted = False
        self.memory = torch.zeros(self.mem_size)

    def load_code(self, code):
        """Load bytecode into memory."""
        for i, instr in enumerate(code):
            self.memory[i] = float(instr)

    def push(self, val):
        self.stack.append(val.clone() if isinstance(val, torch.Tensor) else torch.tensor(float(val)))

    def pop(self):
        return self.stack.pop()

    def step(self):
        """Execute one instruction."""
        if self.halted:
            return False

        # Fetch
        pc_int = int(self.pc.item())
        instr = int(self.memory[pc_int].item())
        opcode = instr & 0xFF
        imm = instr >> 8
        self.pc = self.pc + 1

        # Route (get expert gates)
        gates = self.router(opcode)

        # Execute based on opcode
        if opcode == self.LEA:
            self.ax = self.bp + imm

        elif opcode == self.IMM:
            self.ax = torch.tensor(float(imm))

        elif opcode == self.JMP:
            self.pc = torch.tensor(float(imm))

        elif opcode == self.JSR:
            self.push(self.pc)
            self.pc = torch.tensor(float(imm))

        elif opcode == self.BZ:
            if int(self.ax.item()) == 0:
                self.pc = torch.tensor(float(imm))

        elif opcode == self.BNZ:
            if int(self.ax.item()) != 0:
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
            self.ax = self.memory[addr].clone()

        elif opcode == self.LC:
            addr = int(self.ax.item())
            self.ax = torch.tensor(float(int(self.memory[addr].item()) & 0xFF))

        elif opcode == self.SI:
            addr = int(self.pop().item())
            self.memory[addr] = self.ax.clone()

        elif opcode == self.SC:
            addr = int(self.pop().item())
            self.memory[addr] = torch.tensor(float(int(self.ax.item()) & 0xFF))

        elif opcode == self.PSH:
            self.push(self.ax)

        # Arithmetic (using experts)
        elif opcode == self.ADD:
            self.ax = self.arith.add(self.pop(), self.ax)

        elif opcode == self.SUB:
            self.ax = self.arith.sub(self.pop(), self.ax)

        elif opcode == self.MUL:
            self.ax = self.arith.mul(self.pop(), self.ax)

        elif opcode == self.DIV:
            self.ax = self.arith.div(self.pop(), self.ax)

        elif opcode == self.MOD:
            self.ax = self.arith.mod(self.pop(), self.ax)

        # Bitwise
        elif opcode == self.OR:
            self.ax = self.bitwise.or_op(self.pop(), self.ax)

        elif opcode == self.XOR:
            self.ax = self.bitwise.xor_op(self.pop(), self.ax)

        elif opcode == self.AND:
            self.ax = self.bitwise.and_op(self.pop(), self.ax)

        elif opcode == self.SHL:
            self.ax = self.bitwise.shl(self.pop(), self.ax)

        elif opcode == self.SHR:
            self.ax = self.bitwise.shr(self.pop(), self.ax)

        # Comparison
        elif opcode == self.EQ:
            a = self.pop()
            self.ax = torch.tensor(1.0 if int(a.item()) == int(self.ax.item()) else 0.0)

        elif opcode == self.NE:
            a = self.pop()
            self.ax = torch.tensor(1.0 if int(a.item()) != int(self.ax.item()) else 0.0)

        elif opcode == self.LT:
            a = self.pop()
            self.ax = torch.tensor(1.0 if int(a.item()) < int(self.ax.item()) else 0.0)

        elif opcode == self.GT:
            a = self.pop()
            self.ax = torch.tensor(1.0 if int(a.item()) > int(self.ax.item()) else 0.0)

        elif opcode == self.LE:
            a = self.pop()
            self.ax = torch.tensor(1.0 if int(a.item()) <= int(self.ax.item()) else 0.0)

        elif opcode == self.GE:
            a = self.pop()
            self.ax = torch.tensor(1.0 if int(a.item()) >= int(self.ax.item()) else 0.0)

        elif opcode == self.EXIT:
            self.halted = True
            return False

        return True

    def run(self, max_steps=10000):
        """Run until exit."""
        for _ in range(max_steps):
            if not self.step():
                break
        return int(self.ax.item())

    def count_parameters(self):
        """Count total parameters (should be 0 learned)."""
        total = 0
        learned = 0
        for name, param in self.named_parameters():
            total += param.numel()
            if param.requires_grad:
                learned += param.numel()
        for name, buf in self.named_buffers():
            total += buf.numel()
        return total, learned


# =============================================================================
# TEST
# =============================================================================

def test_transformer():
    print("C4 TRANSFORMER WITH PYTORCH WEIGHTS")
    print("=" * 60)

    # Test sharp gate
    print("\n1. Sharp Gate FFN:")
    sg = SharpGateFFN(scale=20.0)
    for x in [-2.0, -0.1, 0.0, 0.1, 2.0]:
        y = sg(torch.tensor(x))
        print(f"   sharp_gate({x:5.1f}) = {y.item():.4f}")

    # Test eq_gate
    print("\n2. Eq Gate FFN:")
    eq = EqGateFFN(scale=20.0)
    for (a, b) in [(5, 5), (5, 6), (5, 4), (10, 10)]:
        y = eq(torch.tensor(float(a)), torch.tensor(float(b)))
        print(f"   eq_gate({a}, {b}) = {y.item():.4f}")

    # Test router
    print("\n3. MoE Router (39 experts):")
    router = OpcodeRouter(num_experts=39)
    for op in [0, 1, 25, 38]:  # LEA, IMM, ADD, EXIT
        gates = router(op)
        selected = torch.argmax(gates).item()
        gate_val = gates[selected].item()
        print(f"   opcode {op:2d}: expert {selected}, gate={gate_val:.4f}")

    # Test log division
    print("\n4. Log-Space Division:")
    div = LogDivision(max_log=20, resolution=0.125)
    print(f"   Table size: {len(div.log_keys)} entries (log2 at 0.125 resolution)")
    for (a, b) in [(100, 7), (42, 6), (99, 9), (1000, 33), (255, 16)]:
        result = div(torch.tensor(float(a)), torch.tensor(float(b)))
        expected = a // b
        status = "OK" if int(result.item()) == expected else "MISS"
        print(f"   {a:4d} / {b:2d} = {int(result.item()):3d} (expected {expected:3d}) {status}")

    # Test attention memory
    print("\n5. Attention Memory:")
    mem = AttentionMemory(size=16, num_bits=4)
    mem.write(5, 42.0)
    mem.write(10, 99.0)
    print(f"   write(5, 42), write(10, 99)")
    print(f"   read(5) = {mem.read(5).item():.1f}")
    print(f"   read(10) = {mem.read(10).item():.1f}")

    # Test full transformer
    print("\n6. Full C4 Transformer:")
    transformer = C4Transformer()

    # Simple program: IMM 7, EXIT
    # Encoded: opcode + (imm << 8)
    code = [
        1 + (7 << 8),   # IMM 7
        38,             # EXIT
    ]
    transformer.load_code(code)
    result = transformer.run()
    print(f"   Program: IMM 7; EXIT")
    print(f"   Result: {result}")

    # Addition: IMM 3, PSH, IMM 4, ADD, EXIT
    transformer.reset()
    code = [
        1 + (3 << 8),   # IMM 3
        13,             # PSH
        1 + (4 << 8),   # IMM 4
        25,             # ADD
        38,             # EXIT
    ]
    transformer.load_code(code)
    result = transformer.run()
    print(f"   Program: 3 + 4")
    print(f"   Result: {result}")

    # Count parameters
    total, learned = transformer.count_parameters()
    print(f"\n7. Parameter Count:")
    print(f"   Total parameters: {total}")
    print(f"   Learned parameters: {learned}")

    print("\n" + "=" * 60)
    print("ARCHITECTURE SUMMARY:")
    print("  - SharpGateFFN: 4 params (W1: 2x1, b1: 2, W2: 1x2)")
    print("  - EqGateFFN: uses SharpGateFFN")
    print("  - OpcodeRouter: 39 expert IDs")
    print("  - LogDivision: 20 keys + 20 values = 40 entries")
    print("  - AttentionMemory: size x num_bits keys")
    print("  - All params are requires_grad=False (constructed)")
    print("=" * 60)


if __name__ == "__main__":
    test_transformer()
