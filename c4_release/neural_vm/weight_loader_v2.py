"""
Weight Loader V2 - With Graph Coloring and Multi-Layer Operations

Loads compiled opcode weights into AutoregressiveVM with layer sharing.
Uses LayerAllocator for efficient layer assignment via graph coloring.
"""

import torch
from typing import Dict, Optional
from dataclasses import dataclass

from .layer_allocator import LayerAllocator
from .alu_weight_extractor import ALUWeightExtractor, MultiLayerWeights, ThreeLayerWeights
from .opcode_nibble_integration import OpcodeNibbleCompiler
from .embedding import Opcode


class CompiledWeightLoader:
    """Load all compiled opcode weights into AutoregressiveVM with layer sharing."""

    def __init__(self, use_layer_sharing: bool = True, verbose: bool = False):
        """
        Args:
            use_layer_sharing: If True, use graph coloring to share layers (7 layers).
                              If False, use sequential allocation (34 layers).
            verbose: Print detailed loading information
        """
        self.use_layer_sharing = use_layer_sharing
        self.verbose = verbose

        # Initialize layer allocator
        self.allocator = LayerAllocator(use_sharing=use_layer_sharing)
        self.n_layers = self.allocator.total_layers

        # Initialize extractors/compilers
        self.alu_extractor = ALUWeightExtractor()
        self.compiler = OpcodeNibbleCompiler(reg_map=None)  # For single-layer ops

        # Calculate FFN size
        self.ffn_hidden = self._calculate_min_ffn_size()

        # Track unit usage per layer (for shared layers)
        self.layer_unit_usage = {i: 0 for i in range(self.n_layers)}

        if verbose:
            self.allocator.print_allocation()
            print()
            print(f"FFN hidden size: {self.ffn_hidden}")
            print()

    def _calculate_min_ffn_size(self) -> int:
        """Calculate minimum FFN size needed for all operations."""
        # Rough estimates per operation:
        # - Single-layer ops: 32-96 units each
        # - Multi-layer ops: distributed across layers
        #
        # With layer sharing, multiple ops share each layer, so we need
        # to sum the units for all ops that share the same layer.
        #
        # Layer 0 (deepest sharing): MUL, DIV, MOD, SHL, SHR, ADD, SUB, OR, XOR, AND, CMP, REG, CTRL
        # That's a lot of operations! Estimate ~3000 units for layer 0.
        #
        # Conservative estimate: 5000 units * 1.2 safety margin = 6000
        return 6000

    def _load_weights_into_layer(self, vm, layer_idx: int, weights: Dict[str, torch.Tensor],
                                 unit_offset: int, op_name: str):
        """Load weights into a specific layer with unit offset.

        Args:
            vm: AutoregressiveVM instance
            layer_idx: Layer index to load into
            weights: Dictionary with W_up, b_up, W_gate, b_gate, W_down, b_down
            unit_offset: Offset for hidden units (for shared layers)
            op_name: Name of operation (for logging)
        """
        W_up = weights['W_up']
        b_up = weights['b_up']
        W_gate = weights['W_gate']
        b_gate = weights['b_gate']
        W_down = weights['W_down']
        b_down = weights['b_down']

        num_units = W_up.shape[0]
        d_model = W_up.shape[1]

        if d_model != vm.d_model:
            raise ValueError(f"Weight d_model {d_model} doesn't match VM d_model {vm.d_model}")

        # Get target layer
        layer = vm.blocks[layer_idx]

        # Load with unit offset
        h_start = unit_offset
        h_end = unit_offset + num_units

        # Check if we have enough space
        if h_end > vm.ffn_hidden:
            raise ValueError(f"Layer {layer_idx}: {op_name} needs {h_end} units, but FFN only has {vm.ffn_hidden}")

        # Copy weights
        with torch.no_grad():
            layer.ffn.W_up.data[h_start:h_end, :] = W_up
            layer.ffn.b_up.data[h_start:h_end] = b_up
            layer.ffn.W_gate.data[h_start:h_end, :] = W_gate
            layer.ffn.b_gate.data[h_start:h_end] = b_gate
            layer.ffn.W_down.data[:, h_start:h_end] = W_down
            layer.ffn.b_down.data[:] += b_down  # Accumulate (multiple ops may write to same output)

        if self.verbose:
            nonzero = sum((w.abs() > 1e-9).sum().item() for w in weights.values())
            print(f"  Layer {layer_idx:2d} [{h_start:5d}:{h_end:5d}] {op_name:15s} ({num_units:4d} units, {nonzero:7d} params)")

        return h_end - h_start

    def load_all_weights(self, vm):
        """Load all operation weights into VM.

        Args:
            vm: AutoregressiveVM instance (must have d_model=1352, n_layers=self.n_layers)
        """
        if vm.d_model != 1352:
            raise ValueError(f"VM must have d_model=1352, got {vm.d_model}")

        if vm.n_layers != self.n_layers:
            raise ValueError(f"VM must have n_layers={self.n_layers}, got {vm.n_layers}")

        if self.verbose:
            print("=" * 70)
            print("LOADING COMPILED WEIGHTS")
            print("=" * 70)
            print()

        # Load multi-layer operations
        self._load_mul(vm)
        self._load_shl(vm)
        self._load_shr(vm)
        self._load_add(vm)
        self._load_sub(vm)

        # Load single-layer bitwise operations
        self._load_bitwise(vm, Opcode.OR, "OR")
        self._load_bitwise(vm, Opcode.XOR, "XOR")
        self._load_bitwise(vm, Opcode.AND, "AND")

        if self.verbose:
            print()
            print("=" * 70)
            print("LOADING COMPLETE")
            print("=" * 70)

    def _load_mul(self, vm):
        """Load MUL operation (7 layers)."""
        if self.verbose:
            print(f"Loading MUL (7 layers)...")

        mul_weights = self.alu_extractor.extract_mul_weights(Opcode.MUL)
        layer_range = self.allocator.get_layer_range('MUL')

        for i, layer_weights in enumerate(mul_weights.layers):
            if layer_weights is None:
                if self.verbose:
                    print(f"  Layer {layer_range.start + i:2d}: MUL-{i} (non-FFN, skipped)")
                continue

            layer_idx = layer_range.start + i
            unit_offset = self.layer_unit_usage[layer_idx]

            units_used = self._load_weights_into_layer(
                vm, layer_idx, layer_weights, unit_offset, f"MUL-{i}"
            )

            self.layer_unit_usage[layer_idx] += units_used

    def _load_shl(self, vm):
        """Load SHL operation (2 layers)."""
        if self.verbose:
            print(f"\nLoading SHL (2 layers)...")

        shl_weights = self.alu_extractor.extract_shl_weights(Opcode.SHL)
        layer_range = self.allocator.get_layer_range('SHL')

        for i, layer_weights in enumerate(shl_weights.layers):
            if layer_weights is None:
                if self.verbose:
                    print(f"  Layer {layer_range.start + i:2d}: SHL-{i} (non-FFN, skipped)")
                continue

            layer_idx = layer_range.start + i
            unit_offset = self.layer_unit_usage[layer_idx]

            units_used = self._load_weights_into_layer(
                vm, layer_idx, layer_weights, unit_offset, f"SHL-{i}"
            )

            self.layer_unit_usage[layer_idx] += units_used

    def _load_shr(self, vm):
        """Load SHR operation (2 layers)."""
        if self.verbose:
            print(f"\nLoading SHR (2 layers)...")

        shr_weights = self.alu_extractor.extract_shr_weights(Opcode.SHR)
        layer_range = self.allocator.get_layer_range('SHR')

        for i, layer_weights in enumerate(shr_weights.layers):
            if layer_weights is None:
                if self.verbose:
                    print(f"  Layer {layer_range.start + i:2d}: SHR-{i} (non-FFN, skipped)")
                continue

            layer_idx = layer_range.start + i
            unit_offset = self.layer_unit_usage[layer_idx]

            units_used = self._load_weights_into_layer(
                vm, layer_idx, layer_weights, unit_offset, f"SHR-{i}"
            )

            self.layer_unit_usage[layer_idx] += units_used

    def _load_add(self, vm):
        """Load ADD operation (3 layers)."""
        if self.verbose:
            print(f"\nLoading ADD (3 layers)...")

        add_weights = self.alu_extractor.extract_add_weights(Opcode.ADD)
        layer_range = self.allocator.get_layer_range('ADD')

        # Layer 0: RawAndGen
        layer_idx = layer_range.start
        unit_offset = self.layer_unit_usage[layer_idx]
        units_used = self._load_weights_into_layer(
            vm, layer_idx, add_weights.layer1, unit_offset, "ADD-RawGen"
        )
        self.layer_unit_usage[layer_idx] += units_used

        # Layer 1: CarryLookahead
        layer_idx = layer_range.start + 1
        unit_offset = self.layer_unit_usage[layer_idx]
        units_used = self._load_weights_into_layer(
            vm, layer_idx, add_weights.layer2, unit_offset, "ADD-CarryLA"
        )
        self.layer_unit_usage[layer_idx] += units_used

        # Layer 2: Finalize
        layer_idx = layer_range.start + 2
        unit_offset = self.layer_unit_usage[layer_idx]
        units_used = self._load_weights_into_layer(
            vm, layer_idx, add_weights.layer3, unit_offset, "ADD-Finalize"
        )
        self.layer_unit_usage[layer_idx] += units_used

    def _load_sub(self, vm):
        """Load SUB operation (3 layers)."""
        if self.verbose:
            print(f"\nLoading SUB (3 layers)...")

        sub_weights = self.alu_extractor.extract_sub_weights(Opcode.SUB)
        layer_range = self.allocator.get_layer_range('SUB')

        # Layer 0: RawAndGen
        layer_idx = layer_range.start
        unit_offset = self.layer_unit_usage[layer_idx]
        units_used = self._load_weights_into_layer(
            vm, layer_idx, sub_weights.layer1, unit_offset, "SUB-RawGen"
        )
        self.layer_unit_usage[layer_idx] += units_used

        # Layer 1: BorrowLookahead
        layer_idx = layer_range.start + 1
        unit_offset = self.layer_unit_usage[layer_idx]
        units_used = self._load_weights_into_layer(
            vm, layer_idx, sub_weights.layer2, unit_offset, "SUB-BorrowLA"
        )
        self.layer_unit_usage[layer_idx] += units_used

        # Layer 2: Finalize
        layer_idx = layer_range.start + 2
        unit_offset = self.layer_unit_usage[layer_idx]
        units_used = self._load_weights_into_layer(
            vm, layer_idx, sub_weights.layer3, unit_offset, "SUB-Finalize"
        )
        self.layer_unit_usage[layer_idx] += units_used

    def _load_bitwise(self, vm, opcode: int, name: str):
        """Load a bitwise operation (OR/XOR/AND) into layer 0."""
        if self.verbose:
            print(f"\nLoading {name} (1 layer)...")

        # Compile using nibble compiler
        weights = self.compiler.compile_opcode(opcode, unit_offset=0)

        # All single-layer ops go into layer 0
        layer_idx = 0
        unit_offset = self.layer_unit_usage[layer_idx]

        units_used = self._load_weights_into_layer(
            vm, layer_idx, weights, unit_offset, name
        )

        self.layer_unit_usage[layer_idx] += units_used
