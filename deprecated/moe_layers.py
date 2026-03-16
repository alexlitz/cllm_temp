"""
Mixture of Experts layers for Neural VM V7.

MoELayer: Sparse routing based on opcode
MultiExpertMoELayer: Multiple experts per opcode
TransformerBlock: Attention + MoE
"""

import torch
import torch.nn as nn

from .embedding import E


class MoELayer(nn.Module):
    """
    Mixture of Experts layer with sparse routing based on opcode.

    Each expert is an FFN specialized for a specific opcode.
    The router extracts the opcode from the input and selects
    only the matching expert(s) to run.
    """

    def __init__(self, expert_dict: dict):
        """
        Args:
            expert_dict: Dict mapping opcode -> expert FFN
        """
        super().__init__()
        self.expert_dict = nn.ModuleDict({str(k): v for k, v in expert_dict.items()})
        self.opcodes = list(expert_dict.keys())

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Route to correct expert based on opcode in input."""
        B, S, D = x.shape

        opcode_vec = x[0, 0, E.OP_START:E.OP_START + E.NUM_OPS]
        active_opcode = torch.argmax(opcode_vec).item()

        if active_opcode in self.opcodes:
            expert = self.expert_dict[str(active_opcode)]
            x = expert(x)

        return x


class MultiExpertMoELayer(nn.Module):
    """
    MoE layer that runs multiple related experts for an operation.

    Some operations require multiple FFN stages.
    This layer bundles related experts and runs them all for matching opcodes.
    """

    def __init__(self, opcode_experts: dict):
        """
        Args:
            opcode_experts: Dict mapping opcode -> list of expert FFNs
        """
        super().__init__()
        self.opcode_to_experts = {}
        all_experts = []
        for opcode, experts in opcode_experts.items():
            self.opcode_to_experts[opcode] = []
            for expert in experts:
                idx = len(all_experts)
                all_experts.append(expert)
                self.opcode_to_experts[opcode].append(idx)

        self.experts = nn.ModuleList(all_experts)
        self.opcodes = list(opcode_experts.keys())

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Route to correct experts based on opcode."""
        opcode_vec = x[0, 0, E.OP_START:E.OP_START + E.NUM_OPS]
        active_opcode = torch.argmax(opcode_vec).item()

        if active_opcode in self.opcode_to_experts:
            for expert_idx in self.opcode_to_experts[active_opcode]:
                x = self.experts[expert_idx](x)

        return x


class TransformerBlock(nn.Module):
    """
    Standard transformer block: Attention + MoE.

    Alternates between attention (for position-to-position communication)
    and MoE layers (for operation-specific computation).
    """

    def __init__(self, attention, moe: nn.Module):
        super().__init__()
        self.attention = attention
        self.moe = moe

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run attention then MoE."""
        x = self.attention(x)
        x = self.moe(x)
        return x
