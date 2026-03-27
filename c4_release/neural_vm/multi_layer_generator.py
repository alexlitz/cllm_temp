"""
Multi-Layer Weight Generator

Generates FFN weights for opcodes that span multiple transformer layers
(e.g., memory operations like LI, SI, PSH).

Pipeline:
    MultiLayerWeights → Compile each FFN layer → Layer weights dictionary

Usage:
    from neural_vm.multi_layer_generator import MultiLayerWeightGenerator
    from neural_vm.full_opcode_mapper import FullOpcodeMapper
    from neural_vm.embedding import Opcode

    mapper = FullOpcodeMapper()
    generator = MultiLayerWeightGenerator()

    # Get multi-layer structure for LI
    multilayer = mapper.map_opcode_multilayer(Opcode.LI)

    # Generate weights for each FFN layer
    layer_weights = generator.generate_weights(multilayer, Opcode.LI)

    # Load into VM
    vm.blocks[9].ffn.load_state_dict(layer_weights[0])   # Setup layer
    vm.blocks[11].ffn.load_state_dict(layer_weights[2])  # Copy layer
"""

import torch
from typing import Dict, Optional
from dataclasses import dataclass

from .multi_operation_compiler import MultiOperationCompiler
from .graph_weight_compiler import OpType


class MultiLayerWeightGenerator:
    """Generate FFN weights for multi-layer operations.

    Handles operations like:
    - LI (3 layers): FFN setup → Attention → FFN copy
    - PSH (2 layers): FFN setup → Attention write
    - JSR (3 layers): FFN setup → Attention write → FFN PC update
    """

    def __init__(self, num_positions: int = 8):
        """Initialize multi-layer weight generator.

        Args:
            num_positions: Number of nibble positions (8 for 32-bit)
        """
        self.num_positions = num_positions
        self.multi_op_compiler = MultiOperationCompiler(num_positions)

    def generate_weights(
        self,
        multilayer,  # MultiLayerWeights from full_opcode_mapper
        opcode: int
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """Generate weights for each FFN layer.

        Args:
            multilayer: MultiLayerWeights with (layer_type, graph) tuples
            opcode: C4 opcode for gating

        Returns:
            Dictionary mapping layer_index → weight_dict
            {
                0: {'W_up': ..., 'b_up': ..., ...},  # Layer 9 FFN
                2: {'W_up': ..., 'b_up': ..., ...},  # Layer 11 FFN
            }

        Example:
            For LI (3 layers):
            - Layer 0 (FFN): Setup MEM_READ request
            - Layer 1 (Attention): Memory lookup (skip)
            - Layer 2 (FFN): Copy MEM_DATA to AX
        """
        layer_weights = {}

        for layer_idx, (layer_type, graph) in enumerate(multilayer.layers):
            if layer_type == "ffn":
                # Compile FFN graph to weights
                if graph is None:
                    raise ValueError(f"Layer {layer_idx} is FFN but graph is None")

                weights = self._compile_ffn_layer(graph, opcode)
                layer_weights[layer_idx] = weights

            elif layer_type == "attention":
                # Skip - attention uses existing mechanism
                # Attention is handled by the transformer's native attention layer
                layer_weights[layer_idx] = None

            else:
                raise ValueError(f"Unknown layer type: {layer_type}")

        return layer_weights

    def _compile_ffn_layer(
        self,
        graph,  # ComputationGraph
        opcode: int
    ) -> Dict[str, torch.Tensor]:
        """Compile a single FFN layer graph to weights.

        Args:
            graph: ComputationGraph for this layer
            opcode: C4 opcode for gating

        Returns:
            Weight dictionary compatible with PureFFN
        """
        # Check if single operation or multi-operation
        ops = [n for n in graph.nodes.values() if n.op != OpType.CONST]

        if len(ops) == 0:
            # Empty graph - return zero weights (no-op)
            return self._generate_zero_weights()

        elif len(ops) == 1 and ops[0].op == OpType.MOVE:
            # Simple MOVE - can use optimized single-operation path
            from .nibble_weight_compiler import NibbleWeightCompiler
            compiler = NibbleWeightCompiler(self.num_positions)
            return compiler.compile_operation(OpType.MOVE, opcode)

        else:
            # Multi-operation or complex graph - use multi-op compiler
            return self.multi_op_compiler.compile_graph(graph, opcode)

    def _generate_zero_weights(self) -> Dict[str, torch.Tensor]:
        """Generate zero weights for no-op FFN layer.

        Returns:
            Weight dictionary with all zeros
        """
        d_model = self.num_positions * 160  # 8 positions × 160 dims = 1280
        d_ff = 4096

        return {
            'W_up': torch.zeros(d_ff, d_model),
            'b_up': torch.zeros(d_ff),
            'W_gate': torch.zeros(d_ff, d_model),
            'b_gate': torch.zeros(d_ff),
            'W_down': torch.zeros(d_model, d_ff),
            'b_down': torch.zeros(d_model),
        }

    def compile_opcode(self, opcode: int) -> Dict[int, Dict[str, torch.Tensor]]:
        """Convenience method to map and compile an opcode in one call.

        Args:
            opcode: C4 opcode to compile

        Returns:
            Dictionary mapping layer_index → weights

        Example:
            generator = MultiLayerWeightGenerator()
            weights = generator.compile_opcode(Opcode.LI)
            # weights = {0: {...}, 2: {...}}
        """
        from .full_opcode_mapper import FullOpcodeMapper

        mapper = FullOpcodeMapper()
        multilayer = mapper.map_opcode_multilayer(opcode)
        return self.generate_weights(multilayer, opcode)

    def print_layer_summary(
        self,
        layer_weights: Dict[int, Dict[str, torch.Tensor]]
    ):
        """Print summary of generated weights.

        Args:
            layer_weights: Output from generate_weights()
        """
        print("="*70)
        print("Multi-Layer Weight Summary")
        print("="*70)

        total_params = 0

        for layer_idx in sorted(layer_weights.keys()):
            weights = layer_weights[layer_idx]

            if weights is None:
                print(f"\nLayer {layer_idx}: Attention (no FFN weights)")
                continue

            # Count non-zero parameters
            nonzero = 0
            total = 0
            for name, tensor in weights.items():
                n = (tensor.abs() > 1e-9).sum().item()
                t = tensor.numel()
                nonzero += n
                total += t
                total_params += t

            sparsity = 100.0 * (1 - nonzero / total) if total > 0 else 100.0

            print(f"\nLayer {layer_idx}: FFN")
            print(f"  Non-zero params: {nonzero:,} / {total:,}")
            print(f"  Sparsity: {sparsity:.2f}%")

        print()
        print("="*70)
        print(f"Total parameters: {total_params:,}")
        print("="*70)
