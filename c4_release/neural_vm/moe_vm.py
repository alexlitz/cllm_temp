"""
MoE-based VM with Opcode-Routed Experts

Each layer contains multiple experts (one per operation at that depth),
and the opcode deterministically routes to the appropriate expert.
"""

import torch
import torch.nn as nn
from typing import List
from .moe_layer import OpcodeMoELayer, OpcodeMoEBlock, ExpertConfig, extract_opcode_onehot
from .embedding import E
from .nibble_embedding import NibbleVMEmbedding

# Import AutoregressiveAttention and Token from vm_step
from .vm_step import AutoregressiveAttention, Token


class MoEAutoregressiveVM(nn.Module):
    """Autoregressive VM with MoE layers for opcode-based routing.

    Architecture:
    - Each layer has multiple experts (operations at same pipeline depth)
    - Opcode one-hot determines which expert activates
    - Standard MoE semantics with deterministic routing
    """

    def __init__(
        self,
        vocab_size=None,
        d_model=1352,
        n_layers=7,
        n_heads=8,
        experts_per_layer: List[List[ExpertConfig]] = None,
        max_seq_len=4096,
    ):
        """
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            n_layers: Number of transformer layers
            n_heads: Number of attention heads
            experts_per_layer: List of expert configs for each layer
                              experts_per_layer[i] = list of experts for layer i
            max_seq_len: Maximum sequence length
        """
        super().__init__()

        if vocab_size is None:
            vocab_size = Token.VOCAB_SIZE

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

        # Embedding
        self.embed = NibbleVMEmbedding(d_model)

        # Create MoE blocks
        if experts_per_layer is None:
            raise ValueError("Must provide experts_per_layer configuration")

        if len(experts_per_layer) != n_layers:
            raise ValueError(f"experts_per_layer must have {n_layers} entries, got {len(experts_per_layer)}")

        self.blocks = nn.ModuleList()
        for layer_idx in range(n_layers):
            # Create attention
            attn = AutoregressiveAttention(d_model, num_heads=n_heads, max_seq_len=max_seq_len)

            # Create MoE layer with experts for this layer
            moe = OpcodeMoELayer(d_model, experts_per_layer[layer_idx])

            # Create block
            block = OpcodeMoEBlock(attn, moe)
            self.blocks.append(block)

        # Output head (not used in this architecture, kept for compatibility)
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input embedding [batch, seq, d_model]
               Must contain opcode one-hot encoding in dimensions [E.OP_START:E.OP_START+E.NUM_OPS]

        Returns:
            Output [batch, seq, d_model]
        """
        # Extract opcode one-hot from input
        opcode_onehot = extract_opcode_onehot(x, E.OP_START, E.NUM_OPS)

        # Run through MoE blocks
        for block in self.blocks:
            x = block(x, opcode_onehot)

        return x
