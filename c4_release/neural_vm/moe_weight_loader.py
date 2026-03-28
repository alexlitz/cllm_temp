"""
Weight Loader for MoE VM

Creates expert configurations for each layer based on which operations
are at each pipeline depth.
"""

import torch
from typing import Dict, List
from .layer_allocator import LayerAllocator
from .alu_weight_extractor import ALUWeightExtractor
from .opcode_nibble_integration import OpcodeNibbleCompiler
from .embedding import Opcode, E
from .moe_layer import ExpertConfig


class MoEWeightLoader:
    """Load operation weights into MoE VM structure."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.allocator = LayerAllocator(use_sharing=True)
        self.n_layers = self.allocator.total_layers
        self.alu_extractor = ALUWeightExtractor()
        self.compiler = OpcodeNibbleCompiler(num_positions=8, ffn_hidden=10000)  # Temp size

        if verbose:
            self.allocator.print_allocation()

    def create_expert_configs(self) -> List[List[ExpertConfig]]:
        """Create expert configurations for all layers.

        Returns:
            List of expert configs per layer: experts_per_layer[layer_idx] = [ExpertConfig, ...]
        """
        # Initialize empty list for each layer
        experts_per_layer = [[] for _ in range(self.n_layers)]

        # Load multi-layer operations
        self._add_mul_experts(experts_per_layer)
        self._add_shl_experts(experts_per_layer)
        self._add_shr_experts(experts_per_layer)
        self._add_add_experts(experts_per_layer)
        self._add_sub_experts(experts_per_layer)

        # Load single-layer bitwise operations
        self._add_bitwise_experts(experts_per_layer)

        if self.verbose:
            print()
            print("=" * 70)
            print("EXPERT CONFIGURATION")
            print("=" * 70)
            for layer_idx, experts in enumerate(experts_per_layer):
                if experts:
                    expert_names = ", ".join(f"{e.name}({e.ffn_hidden})" for e in experts)
                    print(f"Layer {layer_idx}: {expert_names}")
            print("=" * 70)

        return experts_per_layer

    def _extract_nonzero_units(self, weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Extract only non-zero units from compiled weights."""
        W_up = weights['W_up']
        b_up = weights['b_up']
        W_gate = weights['W_gate']
        b_gate = weights['b_gate']
        W_down = weights['W_down']
        b_down = weights['b_down']

        # Find non-zero units
        nonzero_mask = (W_up.abs().sum(dim=1) > 1e-9) | (W_gate.abs().sum(dim=1) > 1e-9)
        nonzero_indices = torch.where(nonzero_mask)[0]

        if len(nonzero_indices) == 0:
            raise ValueError("No non-zero units found!")

        return {
            'W_up': W_up[nonzero_indices],
            'b_up': b_up[nonzero_indices],
            'W_gate': W_gate[nonzero_indices],
            'b_gate': b_gate[nonzero_indices],
            'W_down': W_down[:, nonzero_indices],
            'b_down': b_down,
        }

    def _add_mul_experts(self, experts_per_layer):
        """Add MUL experts to layers."""
        mul_weights = self.alu_extractor.extract_mul_weights(Opcode.MUL)
        layer_range = self.allocator.get_layer_range('MUL')

        for i, layer_weights in enumerate(mul_weights.layers):
            if layer_weights is None:
                continue

            layer_idx = layer_range.start + i
            ffn_hidden = layer_weights['W_up'].shape[0]

            expert = ExpertConfig(
                opcode=Opcode.MUL,
                name=f"MUL-{i}",
                ffn_hidden=ffn_hidden,
                weights=layer_weights
            )
            experts_per_layer[layer_idx].append(expert)

    def _add_shl_experts(self, experts_per_layer):
        """Add SHL experts to layers."""
        shl_weights = self.alu_extractor.extract_shl_weights(Opcode.SHL)
        layer_range = self.allocator.get_layer_range('SHL')

        for i, layer_weights in enumerate(shl_weights.layers):
            if layer_weights is None:
                continue

            layer_idx = layer_range.start + i
            ffn_hidden = layer_weights['W_up'].shape[0]

            expert = ExpertConfig(
                opcode=Opcode.SHL,
                name=f"SHL-{i}",
                ffn_hidden=ffn_hidden,
                weights=layer_weights
            )
            experts_per_layer[layer_idx].append(expert)

    def _add_shr_experts(self, experts_per_layer):
        """Add SHR experts to layers."""
        shr_weights = self.alu_extractor.extract_shr_weights(Opcode.SHR)
        layer_range = self.allocator.get_layer_range('SHR')

        for i, layer_weights in enumerate(shr_weights.layers):
            if layer_weights is None:
                continue

            layer_idx = layer_range.start + i
            ffn_hidden = layer_weights['W_up'].shape[0]

            expert = ExpertConfig(
                opcode=Opcode.SHR,
                name=f"SHR-{i}",
                ffn_hidden=ffn_hidden,
                weights=layer_weights
            )
            experts_per_layer[layer_idx].append(expert)

    def _add_add_experts(self, experts_per_layer):
        """Add ADD experts to layers."""
        add_weights = self.alu_extractor.extract_add_weights(Opcode.ADD)
        layer_range = self.allocator.get_layer_range('ADD')

        # Layer 0: RawGen
        layer_idx = layer_range.start
        ffn_hidden = add_weights.layer1['W_up'].shape[0]
        expert = ExpertConfig(
            opcode=Opcode.ADD,
            name="ADD-RawGen",
            ffn_hidden=ffn_hidden,
            weights=add_weights.layer1
        )
        experts_per_layer[layer_idx].append(expert)

        # Layer 1: CarryLA
        layer_idx = layer_range.start + 1
        ffn_hidden = add_weights.layer2['W_up'].shape[0]
        expert = ExpertConfig(
            opcode=Opcode.ADD,
            name="ADD-CarryLA",
            ffn_hidden=ffn_hidden,
            weights=add_weights.layer2
        )
        experts_per_layer[layer_idx].append(expert)

        # Layer 2: Finalize
        layer_idx = layer_range.start + 2
        ffn_hidden = add_weights.layer3['W_up'].shape[0]
        expert = ExpertConfig(
            opcode=Opcode.ADD,
            name="ADD-Finalize",
            ffn_hidden=ffn_hidden,
            weights=add_weights.layer3
        )
        experts_per_layer[layer_idx].append(expert)

    def _add_sub_experts(self, experts_per_layer):
        """Add SUB experts to layers."""
        sub_weights = self.alu_extractor.extract_sub_weights(Opcode.SUB)
        layer_range = self.allocator.get_layer_range('SUB')

        # Layer 0: RawGen
        layer_idx = layer_range.start
        ffn_hidden = sub_weights.layer1['W_up'].shape[0]
        expert = ExpertConfig(
            opcode=Opcode.SUB,
            name="SUB-RawGen",
            ffn_hidden=ffn_hidden,
            weights=sub_weights.layer1
        )
        experts_per_layer[layer_idx].append(expert)

        # Layer 1: BorrowLA
        layer_idx = layer_range.start + 1
        ffn_hidden = sub_weights.layer2['W_up'].shape[0]
        expert = ExpertConfig(
            opcode=Opcode.SUB,
            name="SUB-BorrowLA",
            ffn_hidden=ffn_hidden,
            weights=sub_weights.layer2
        )
        experts_per_layer[layer_idx].append(expert)

        # Layer 2: Finalize
        layer_idx = layer_range.start + 2
        ffn_hidden = sub_weights.layer3['W_up'].shape[0]
        expert = ExpertConfig(
            opcode=Opcode.SUB,
            name="SUB-Finalize",
            ffn_hidden=ffn_hidden,
            weights=sub_weights.layer3
        )
        experts_per_layer[layer_idx].append(expert)

    def _add_bitwise_experts(self, experts_per_layer):
        """Add bitwise operation experts to layer 0."""
        for opcode, name in [(Opcode.OR, "OR"), (Opcode.XOR, "XOR"), (Opcode.AND, "AND")]:
            # Compile and extract non-zero units
            full_weights = self.compiler.compile_opcode(opcode, unit_offset=0)
            weights = self._extract_nonzero_units(full_weights)

            ffn_hidden = weights['W_up'].shape[0]

            expert = ExpertConfig(
                opcode=opcode,
                name=name,
                ffn_hidden=ffn_hidden,
                weights=weights
            )

            # All bitwise ops go in layer 0
            experts_per_layer[0].append(expert)
