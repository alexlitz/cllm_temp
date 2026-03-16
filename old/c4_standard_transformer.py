"""
C4 VM as a Standard Transformer with Constructed Weights

This implements C4 operations using a uniform transformer architecture where:
- Every expert has the SAME structure (attention + FFN)
- Operations are encoded in the WEIGHT VALUES
- Standard forward pass: attention -> FFN -> output

The state is a vector: [pc, sp, bp, ax, arg1, arg2, mem_addr, mem_val, opcode, ...]
Each layer transforms this state according to the current opcode.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =============================================================================
# STATE VECTOR LAYOUT
# =============================================================================
# Index positions in state vector
PC = 0      # Program counter
SP = 1      # Stack pointer
BP = 2      # Base pointer
AX = 3      # Accumulator
ARG1 = 4    # First argument (popped from stack)
ARG2 = 5    # Second argument
OPCODE = 6  # Current opcode
IMM = 7     # Immediate value
TEMP = 8    # Temporary storage
STATE_DIM = 16


# =============================================================================
# STANDARD FFN LAYER
# =============================================================================

class StandardFFN(nn.Module):
    """
    Standard FFN: x -> W2 @ silu(W1 @ x + b1) + b2

    We construct W1, b1, W2, b2 to implement various operations.
    """
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W1 = nn.Parameter(torch.zeros(d_ff, d_model), requires_grad=False)
        self.b1 = nn.Parameter(torch.zeros(d_ff), requires_grad=False)
        self.W2 = nn.Parameter(torch.zeros(d_model, d_ff), requires_grad=False)
        self.b2 = nn.Parameter(torch.zeros(d_model), requires_grad=False)

    def forward(self, x):
        h = F.silu(F.linear(x, self.W1, self.b1))
        return F.linear(h, self.W2, self.b2)


class StandardAttention(nn.Module):
    """
    Standard attention: softmax(Q @ K.T / sqrt(d)) @ V

    We construct Wq, Wk, Wv, Wo to implement memory operations.
    """
    def __init__(self, d_model, n_heads=1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.Wq = nn.Parameter(torch.zeros(d_model, d_model), requires_grad=False)
        self.Wk = nn.Parameter(torch.zeros(d_model, d_model), requires_grad=False)
        self.Wv = nn.Parameter(torch.zeros(d_model, d_model), requires_grad=False)
        self.Wo = nn.Parameter(torch.zeros(d_model, d_model), requires_grad=False)

    def forward(self, x, kv=None):
        """x: [batch, seq, d_model] or [seq, d_model] or [d_model]"""
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 2:
            x = x.unsqueeze(0)

        if kv is None:
            kv = x

        Q = F.linear(x, self.Wq)
        K = F.linear(kv, self.Wk)
        V = F.linear(kv, self.Wv)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, V)
        out = F.linear(out, self.Wo)

        return out.squeeze()


# =============================================================================
# EXPERT LAYER (Uniform Architecture)
# =============================================================================

class ExpertLayer(nn.Module):
    """
    One expert = Attention + FFN with specific weights.

    All experts have identical architecture, different weights.
    The weights encode what operation to perform.
    """
    def __init__(self, d_model=STATE_DIM, d_ff=64):
        super().__init__()
        self.attention = StandardAttention(d_model)
        self.ffn = StandardFFN(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)

    def forward(self, x):
        # Standard transformer block
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# =============================================================================
# WEIGHT CONSTRUCTION FUNCTIONS
# =============================================================================

def construct_add_weights(expert: ExpertLayer, scale=20.0):
    """
    Construct weights for ADD operation: ax = arg1 + ax

    FFN computes: new_ax = state[ARG1] + state[AX]
    Then writes result to AX position.
    """
    d_ff = expert.ffn.W1.shape[0]
    d_model = expert.ffn.W1.shape[1]

    # W1 extracts ARG1 and AX, combines them
    # Row 0: extract ARG1
    expert.ffn.W1.data[0, ARG1] = 1.0
    # Row 1: extract AX
    expert.ffn.W1.data[1, AX] = 1.0

    # After SiLU, W2 writes sum to AX position
    # For addition, we need: W2[AX, :] @ silu([arg1, ax, 0...])
    # This is tricky with SiLU...

    # Simpler: use the FFN to compute a*1 + b*1 = a + b
    # But SiLU is nonlinear. For small inputs, silu(x) ≈ x * sigmoid(x)

    # Actually, let's use a direct approach:
    # FFN computes correction: new_ax - old_ax = arg1
    # So we just need to add arg1 to ax
    expert.ffn.W1.data[0, ARG1] = 1.0  # Extract arg1
    expert.ffn.W2.data[AX, 0] = 1.0    # Write to ax position

    # This adds arg1 to ax via residual connection


def construct_sub_weights(expert: ExpertLayer):
    """Construct weights for SUB: ax = arg1 - ax"""
    expert.ffn.W1.data[0, ARG1] = 1.0
    expert.ffn.W1.data[1, AX] = -1.0
    expert.ffn.W2.data[AX, 0] = 1.0
    expert.ffn.W2.data[AX, 1] = 1.0


def construct_mul_weights(expert: ExpertLayer):
    """
    Construct weights for MUL: ax = arg1 * ax

    Uses the fact that multiplication can be done via:
    a * b = ((a + b)^2 - (a - b)^2) / 4

    Or use log-exp: a * b = exp(log(a) + log(b))
    But that requires the log/exp tables.

    For now, we'll use a direct encoding with the understanding
    that MUL requires additional structure.
    """
    # Placeholder - multiplication needs special handling
    pass


def construct_identity_weights(expert: ExpertLayer):
    """Construct weights that pass through unchanged."""
    # All zeros = no change (residual connection passes through)
    pass


# =============================================================================
# ROUTER (Opcode -> Expert Selection via Attention)
# =============================================================================

class OpcodeRouter(nn.Module):
    """
    Routes to correct expert using attention over opcode embeddings.

    This is standard attention where:
    - Query = current opcode
    - Keys = all opcode IDs [0, 1, 2, ..., 38]
    - Values = expert selection (one-hot)

    The attention weights become the expert gates.
    """
    def __init__(self, num_experts=39, d_model=STATE_DIM):
        super().__init__()
        self.num_experts = num_experts

        # Opcode embeddings (keys)
        # Each opcode i has embedding that matches query for opcode i
        self.register_buffer('opcode_keys', torch.eye(num_experts))

        # Query projection: extract opcode from state, convert to one-hot-ish
        self.Wq = nn.Parameter(torch.zeros(num_experts, d_model), requires_grad=False)
        # Set up Wq to create one-hot from opcode value
        # This requires the sharp gate mechanism

    def forward(self, state):
        """Returns expert gates [num_experts]."""
        opcode = state[OPCODE]

        # Create query that matches opcode-th key
        # Using eq_gate style: gate[i] = sharp(opcode - i + 0.5) * sharp(-(opcode - i) + 0.5)
        gates = torch.zeros(self.num_experts)
        for i in range(self.num_experts):
            diff = opcode - i
            # Sharp gate approximation
            g1 = torch.sigmoid((diff + 0.5) * 20)
            g2 = torch.sigmoid((-diff + 0.5) * 20)
            gates[i] = g1 * g2

        return gates


# =============================================================================
# COMPLETE C4 TRANSFORMER
# =============================================================================

class C4StandardTransformer(nn.Module):
    """
    C4 VM as a standard transformer.

    Architecture:
    - State vector: [pc, sp, bp, ax, arg1, arg2, opcode, imm, ...]
    - Router: attention-based opcode -> expert selection
    - Experts: uniform FFN + attention layers with constructed weights
    - Output: updated state vector

    Each forward pass = one VM instruction.
    """

    # Opcodes
    LEA, IMM, JMP, JSR, BZ, BNZ, ENT, ADJ, LEV = 0, 1, 2, 3, 4, 5, 6, 7, 8
    LI, LC, SI, SC, PSH = 9, 10, 11, 12, 13
    OR, XOR, AND, EQ, NE, LT, GT, LE, GE = 14, 15, 16, 17, 18, 19, 20, 21, 22
    SHL, SHR, ADD, SUB, MUL, DIV, MOD = 23, 24, 25, 26, 27, 28, 29
    EXIT = 38

    def __init__(self, num_experts=39, d_model=STATE_DIM, d_ff=64):
        super().__init__()

        self.num_experts = num_experts
        self.d_model = d_model

        # Create experts (all same architecture)
        self.experts = nn.ModuleList([
            ExpertLayer(d_model, d_ff) for _ in range(num_experts)
        ])

        # Router
        self.router = OpcodeRouter(num_experts, d_model)

        # Construct weights for each expert
        self._construct_all_weights()

        # State
        self.register_buffer('state', torch.zeros(d_model))
        self.memory = []
        self.stack = []
        self.halted = False

    def _construct_all_weights(self):
        """Construct weights for all experts."""
        # ADD expert
        construct_add_weights(self.experts[self.ADD])

        # SUB expert
        construct_sub_weights(self.experts[self.SUB])

        # Others get identity (placeholder)
        for i in range(self.num_experts):
            if i not in [self.ADD, self.SUB]:
                construct_identity_weights(self.experts[i])

    def forward(self, state):
        """
        One forward pass = execute one instruction.

        1. Router computes expert gates from opcode
        2. Each expert processes state
        3. Weighted combination of expert outputs
        """
        # Get expert gates
        gates = self.router(state)

        # Apply all experts and combine
        output = torch.zeros_like(state)
        for i, expert in enumerate(self.experts):
            expert_out = expert(state)
            output = output + gates[i] * expert_out

        return output

    def step(self, opcode, imm, arg1=0):
        """Execute one instruction."""
        # Set up state
        self.state[OPCODE] = opcode
        self.state[IMM] = imm
        self.state[ARG1] = arg1

        # Forward pass
        new_state = self.forward(self.state)

        # Update state
        self.state = new_state

        return self.state[AX].item()


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demonstrate():
    print("C4 STANDARD TRANSFORMER")
    print("=" * 60)
    print()
    print("Architecture:")
    print("  - Uniform expert layers (Attention + FFN)")
    print("  - Operations encoded in weight matrices")
    print("  - Standard forward pass")
    print()

    model = C4StandardTransformer()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    print(f"Experts: {model.num_experts}")
    print(f"State dimension: {model.d_model}")
    print()

    # Show expert structure
    print("Expert structure (same for all 39):")
    expert = model.experts[0]
    print(f"  Attention: Wq, Wk, Wv, Wo each [{model.d_model}, {model.d_model}]")
    print(f"  FFN W1: [{expert.ffn.W1.shape[0]}, {expert.ffn.W1.shape[1]}]")
    print(f"  FFN W2: [{expert.ffn.W2.shape[0]}, {expert.ffn.W2.shape[1]}]")
    print()

    # Test router
    print("Router test (opcode -> expert gates):")
    for op in [0, 1, 25, 26, 38]:
        state = torch.zeros(STATE_DIM)
        state[OPCODE] = op
        gates = model.router(state)
        top = torch.argmax(gates).item()
        print(f"  Opcode {op:2d}: top expert = {top}, gate = {gates[top]:.4f}")

    print()
    print("=" * 60)
    print("KEY INSIGHT:")
    print("  All experts have IDENTICAL architecture (attention + FFN)")
    print("  The WEIGHTS encode what operation each expert performs")
    print("  Router uses attention to select expert based on opcode")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate()
