"""
Mixture of Experts (MoE) Layer with Opcode-Based Routing

Standard MoE implementation where each operation is an expert and the opcode
deterministically selects which expert to use.

Architecture:
- Router: Opcode one-hot encoding directly determines expert selection
- Experts: Individual FFN weights for each operation
- Output: Weighted sum of expert outputs (in practice, only selected expert contributes)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ExpertConfig:
    """Configuration for a single expert."""
    opcode: int           # Which opcode this expert handles
    name: str            # Human-readable name
    ffn_hidden: int      # Hidden dimension for this expert's FFN
    weights: Optional[Dict[str, torch.Tensor]] = None  # Precomputed weights


class OpcodeMoELayer(nn.Module):
    """MoE layer where opcode deterministically selects the expert.

    Standard MoE formulation:
        output = sum_i router_weight[i] * expert[i](input)

    In our case:
        router_weight[i] = opcode_onehot[i]  (deterministic, not learned)
        expert[i] = operation-specific FFN

    This provides:
    - Clean separation: each operation has its own expert
    - No interference: only selected expert activates
    - Standard MoE semantics: well-understood, debuggable
    - Efficient: expert FFNs only as wide as needed
    """

    def __init__(self, d_model: int, experts: List[ExpertConfig]):
        """
        Args:
            d_model: Model dimension
            experts: List of expert configurations
        """
        super().__init__()
        self.d_model = d_model
        self.experts = experts
        self.num_experts = len(experts)

        # Create a mapping from opcode to expert index
        self.opcode_to_expert = {}
        for idx, expert in enumerate(experts):
            self.opcode_to_expert[expert.opcode] = idx

        # Create FFN for each expert
        self.expert_ffns = nn.ModuleList()
        for expert in experts:
            ffn = self._create_expert_ffn(expert)
            self.expert_ffns.append(ffn)

    def _create_expert_ffn(self, expert: ExpertConfig) -> nn.Module:
        """Create FFN for a single expert."""
        hidden_dim = expert.ffn_hidden

        # SwiGLU FFN: output = down(silu(up(x)) * gate(x))
        class ExpertFFN(nn.Module):
            def __init__(self, d_model, hidden_dim):
                super().__init__()
                self.W_up = nn.Parameter(torch.zeros(hidden_dim, d_model))
                self.b_up = nn.Parameter(torch.zeros(hidden_dim))
                self.W_gate = nn.Parameter(torch.zeros(hidden_dim, d_model))
                self.b_gate = nn.Parameter(torch.zeros(hidden_dim))
                # W_down: [d_model, hidden_dim] for F.linear
                self.W_down = nn.Parameter(torch.zeros(d_model, hidden_dim))
                self.b_down = nn.Parameter(torch.zeros(d_model))

            def forward(self, x):
                # x: [batch, seq, d_model]
                up = torch.nn.functional.linear(x, self.W_up, self.b_up)
                gate = torch.nn.functional.linear(x, self.W_gate, self.b_gate)
                hidden = torch.nn.functional.silu(up) * gate
                # W_down is [hidden, d_model], use it directly (F.linear will transpose)
                output = torch.nn.functional.linear(hidden, self.W_down, self.b_down)
                return output

        ffn = ExpertFFN(self.d_model, hidden_dim)

        # Load precomputed weights if available
        if expert.weights is not None:
            with torch.no_grad():
                # Stored weights: W_up [hidden, d_model], W_down [d_model, hidden]
                # Parameter shapes: W_up [hidden, d_model], W_down [d_model, hidden]
                ffn.W_up.data[:] = expert.weights['W_up']      # [hidden, d_model] ✓
                ffn.b_up.data[:] = expert.weights['b_up']
                ffn.W_gate.data[:] = expert.weights['W_gate']  # [hidden, d_model] ✓
                ffn.b_gate.data[:] = expert.weights['b_gate']
                ffn.W_down.data[:] = expert.weights['W_down']  # [d_model, hidden] ✓
                ffn.b_down.data[:] = expert.weights['b_down']

        return ffn

    def forward(self, x: torch.Tensor, opcode_onehot: torch.Tensor) -> torch.Tensor:
        """Forward pass with opcode-based routing.

        Args:
            x: Input tensor [batch, seq, d_model]
            opcode_onehot: One-hot opcode encoding [batch, num_opcodes]
                          This determines which expert to use

        Returns:
            Output tensor [batch, seq, d_model]
        """
        batch_size, seq_len, _ = x.shape

        # For efficiency, we can extract which opcode is active
        # Since opcode_onehot is one-hot, argmax gives us the active opcode
        active_opcodes = opcode_onehot.argmax(dim=-1)  # [batch]

        # Initialize output
        output = torch.zeros_like(x)

        # Process each unique opcode in the batch
        unique_opcodes = active_opcodes.unique()

        for opcode in unique_opcodes:
            opcode_val = opcode.item()

            # Skip if this opcode doesn't have an expert
            if opcode_val not in self.opcode_to_expert:
                continue

            # Get expert index
            expert_idx = self.opcode_to_expert[opcode_val]

            # Find samples with this opcode
            mask = (active_opcodes == opcode_val)

            # Run expert on those samples
            x_subset = x[mask]  # [num_samples, seq, d_model]
            expert_output = self.expert_ffns[expert_idx](x_subset)

            # Write back to output
            output[mask] = expert_output

        return output


class OpcodeMoEBlock(nn.Module):
    """Transformer block with MoE FFN layer (opcode-routed).

    Standard transformer block:
        x = x + attention(norm(x))
        x = x + ffn(norm(x))

    Here FFN is replaced with MoE layer.
    """

    def __init__(self, attn: nn.Module, moe: OpcodeMoELayer):
        super().__init__()
        self.attn = attn
        self.moe = moe
        self.norm1 = nn.LayerNorm(moe.d_model)
        self.norm2 = nn.LayerNorm(moe.d_model)

    def forward(self, x: torch.Tensor, opcode_onehot: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input [batch, seq, d_model]
            opcode_onehot: One-hot opcode [batch, num_opcodes]
        """
        # Attention
        x = x + self.attn(self.norm1(x))

        # MoE FFN
        x = x + self.moe(self.norm2(x), opcode_onehot)

        return x


def extract_opcode_onehot(x: torch.Tensor, opcode_start: int, num_opcodes: int) -> torch.Tensor:
    """Extract opcode one-hot encoding from embedding.

    Args:
        x: Input embedding [batch, seq, d_model]
        opcode_start: Starting dimension of opcode one-hot (E.OP_START)
        num_opcodes: Number of opcodes (E.NUM_OPS)

    Returns:
        One-hot opcode [batch, num_opcodes]
    """
    # Opcode is shared across positions, so we can take it from any position
    # Take from position 0
    batch_size, seq_len, d_model = x.shape

    # Extract opcode region from position 0
    opcode_region = x[:, 0, opcode_start:opcode_start + num_opcodes]  # [batch, num_opcodes]

    return opcode_region
