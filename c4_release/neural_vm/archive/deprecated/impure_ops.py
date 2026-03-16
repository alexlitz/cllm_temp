"""
DEPRECATED: Impure Operations.

These implementations use non-pure operations like .sum(), .clone(), .expand()
and should NOT be used. They are kept only for historical reference.

WHY DEPRECATED:
- .sum() is reduction, not a neural network operation
- .clone() creates a copy, not a learned transformation
- .expand() is broadcasting, not attention or FFN

USE INSTEAD:
- MultiHeadGatherAttention → NibbleGatherFFN (from reduce_ffn.py)
- GatherAttentionPure → NibbleGatherFFN (from reduce_ffn.py)
"""

import torch
import torch.nn as nn

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_vm.embedding import E, Opcode
from neural_vm.base_layers import PureAttention


class MultiHeadGatherAttention(nn.Module):
    """
    DEPRECATED: Uses .sum(), .clone(), .expand() - NOT PURE.

    Use NibbleGatherFFN instead.

    This gathers 8 nibbles into a 32-bit value, but uses Python operations
    that cannot be implemented in a real transformer.
    """

    def __init__(self, num_positions: int = 8, gather_a: bool = True, gather_b: bool = True):
        super().__init__()
        self.num_positions = num_positions
        self.gather_a = gather_a
        self.gather_b = gather_b
        self.num_heads = num_positions

        scales = torch.tensor([16.0 ** k for k in range(num_positions)])
        self.register_buffer('scales', scales)

        self.A_FULL = E.TEMP + 10
        self.B_FULL = E.TEMP + 11

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        DEPRECATED: Uses .sum(), .clone(), .expand().

        These are NOT neural network operations and cannot be implemented
        in a real transformer forward pass.
        """
        batch_size = x.shape[0]

        nib_a = x[:, :, E.NIB_A]
        nib_b = x[:, :, E.NIB_B]

        if self.gather_a:
            # IMPURE: .sum() is a reduction, not a neural op
            a_full = (nib_a * self.scales.unsqueeze(0)).sum(dim=1, keepdim=True)
        if self.gather_b:
            b_full = (nib_b * self.scales.unsqueeze(0)).sum(dim=1, keepdim=True)

        # IMPURE: .clone() creates a copy
        y = x.clone()
        if self.gather_a:
            # IMPURE: .expand() is broadcasting
            y[:, :, self.A_FULL] = a_full.expand(-1, self.num_positions)
        if self.gather_b:
            y[:, :, self.B_FULL] = b_full.expand(-1, self.num_positions)

        return y


class GatherAttentionPure(PureAttention):
    """
    DEPRECATED: Attention-based gather.

    Use NibbleGatherFFN instead (flattened FFN approach is simpler
    and doesn't require Q/K matching).
    """

    def __init__(self, is_b: bool = False, output_slot: int = None):
        self.is_b = is_b
        self.output_slot = output_slot or (E.TEMP + 11 if is_b else E.TEMP + 10)
        super().__init__(E.DIM, num_heads=8, causal=False)

    def _bake_weights(self):
        with torch.no_grad():
            src_slot = E.NIB_B if self.is_b else E.NIB_A

            for head in range(8):
                scale = 16.0 ** head
                self.W_q[head, E.POS] = 1.0
                self.b_q[head] = -float(head)
                self.W_k[head, E.POS] = 1.0
                self.W_v[head, src_slot] = scale
                self.W_o[self.output_slot, head] = 1.0
