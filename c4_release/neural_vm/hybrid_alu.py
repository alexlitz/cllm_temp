#!/usr/bin/env python3
"""Hybrid ALU: combines efficient neural ALU with lookup FFN for non-core ops.

The efficient ALU handles ADD/SUB/AND/OR/XOR/MUL/SHL/SHR/DIV/MOD using
structural multi-layer FFN pipelines (carry lookahead, schoolbook mul, etc.)
instead of full lookup tables.

The lookup FFN handles LEA, ADJ, ENT, CMP, AX passthrough, carry propagation,
and other VM-specific operations that the efficient ALU doesn't cover.

The efficient module runs as a post_op after the lookup FFN. For opcodes it
handles, it overwrites the lookup's OUTPUT with its own result.
"""

import torch
import torch.nn as nn

from .efficient_alu_neural import PureNeuralALU


class HybridALUBlock(nn.Module):
    """Combined FFN that runs lookup weights first, then efficient ALU on top.

    Usage: model.blocks[i].ffn = HybridALUBlock(original_ffn, efficient_alu)
    """

    def __init__(self, lookup_ffn, efficient_alu):
        super().__init__()
        self.lookup_ffn = lookup_ffn
        self.efficient_alu = efficient_alu

    def forward(self, x):
        x = self.lookup_ffn(x)
        x = self.efficient_alu(x)
        return x

    def compact(self, block_size=1):
        if hasattr(self.lookup_ffn, 'compact'):
            self.lookup_ffn.compact(block_size)

    def sparsify(self):
        if hasattr(self.lookup_ffn, 'sparsify'):
            self.lookup_ffn.sparsify()

    def compact_moe(self, opcode_range=None, relay_map=None):
        if hasattr(self.lookup_ffn, 'compact_moe'):
            self.lookup_ffn.compact_moe(opcode_range, relay_map)
