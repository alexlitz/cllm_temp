"""
Weight Loader for Compiled Opcodes

Loads all compiled opcode weights into an AutoregressiveVM instance.

Usage:
    from neural_vm.weight_loader import CompiledWeightLoader
    from neural_vm.vm_step import AutoregressiveVM

    vm = AutoregressiveVM(d_model=1280, n_layers=16, ffn_hidden=70000)
    loader = CompiledWeightLoader()
    loader.load_all_weights(vm)

    # Now VM has all 32 opcode weights loaded
    # Note: ffn_hidden=70000 required for lookup-table operations with 4-unit inclusion-exclusion
"""

import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .opcode_nibble_integration import OpcodeNibbleCompiler
from .alu_weight_extractor import ALUWeightExtractor
from .embedding import Opcode


@dataclass
class OpcodeLayerMapping:
    """Maps opcodes to transformer layer indices."""
    
    # Layer allocation strategy:
    # - Layers 0-5: Instruction fetch, decode, register read
    # - Layers 6-8: Pre-ALU setup
    # - Layer 9: Primary ALU (arithmetic, comparison, bitwise)
    # - Layer 10: Secondary ALU (shift, control flow)
    # - Layer 11: Memory/Stack setup (for multi-layer ops)
    # - Layer 12: Attention layer for memory access
    # - Layer 13: Memory/Stack result copy
    # - Layer 14: PC update, post-processing
    # - Layer 15: Output generation
    
    # Single-operation ALU opcodes → Layer 9
    PRIMARY_ALU = 9
    
    # Multi-operation control flow → Layer 10
    CONTROL_FLOW = 10
    
    # Multi-layer memory operations
    MEMORY_SETUP = 11      # FFN layer before attention
    MEMORY_ATTENTION = 12  # Attention layer
    MEMORY_RESULT = 13     # FFN layer after attention


class CompiledWeightLoader:
    """Load all compiled opcode weights into AutoregressiveVM."""

    def __init__(self, ffn_hidden: Optional[int] = None):
        # Calculate minimum required FFN size if not specified
        if ffn_hidden is None:
            ffn_hidden = self._calculate_min_ffn_size()

        self.ffn_hidden = ffn_hidden
        self.compiler = OpcodeNibbleCompiler(ffn_hidden=ffn_hidden)
        self.alu_extractor = ALUWeightExtractor()
        self.layer_map = OpcodeLayerMapping()
        self.unit_allocations = self._calculate_unit_allocation()

    def _calculate_min_ffn_size(self) -> int:
        """Calculate minimum FFN size needed for all operations.

        Returns:
            Minimum ffn_hidden value required
        """
        # Rough estimate based on current implementation:
        # - ADD/SUB: 64 each = 128
        # - MUL/DIV/MOD: 2048 each = 6144
        # - Comparisons (6 ops): 64 each = 384
        # - AND: 32
        # - OR/XOR: 96 each = 192
        # - SHL/SHR: 2048 each = 4096
        # - Control flow/register (14 ops): 64 each = 896
        # Total: ~11,872

        # Add 20% margin for safety
        return int(12000 * 1.2)

    def _calculate_unit_allocation(self) -> Dict[int, int]:
        """Calculate non-overlapping hidden unit ranges for each opcode.

        Returns:
            Dictionary mapping opcode -> unit_offset

        Strategy:
            - Algorithmic ops (ADD, SUB, comparisons): ~64 units
            - Lookup table ops with binary bit matching: 2048 units each
              (256 units per position × 8 positions, where 256 = 16×16 pairs)
            - Allocate sequentially to avoid collisions
        """
        allocations = {}
        current_offset = 0

        # Arithmetic operations (64 units each - algorithmic with carry propagation)
        for op in [Opcode.ADD, Opcode.SUB]:
            allocations[op] = current_offset
            current_offset += 64

        # Multiply with binary bit matching (2048 units)
        # 256 units per position × 8 positions
        allocations[Opcode.MUL] = current_offset
        current_offset += 2048

        # Division operations with binary bit matching (2048 units each)
        for op in [Opcode.DIV, Opcode.MOD]:
            allocations[op] = current_offset
            current_offset += 2048

        # Comparison operations (64 units each - algorithmic threshold-based)
        for op in [Opcode.EQ, Opcode.NE, Opcode.LT, Opcode.GT, Opcode.LE, Opcode.GE]:
            allocations[op] = current_offset
            current_offset += 64

        # Bitwise operations with bit-level primitives (32-96 units each)
        # AND: 4 units/position × 8 = 32 units
        # OR/XOR: 12 units/position × 8 = 96 units
        allocations[Opcode.AND] = current_offset
        current_offset += 32

        for op in [Opcode.OR, Opcode.XOR]:
            allocations[op] = current_offset
            current_offset += 96

        # Shift operations with binary bit matching (2048 units each)
        for op in [Opcode.SHL, Opcode.SHR]:
            allocations[op] = current_offset
            current_offset += 2048

        # Register operations (64 units each)
        for op in [Opcode.LEA, Opcode.IMM]:
            allocations[op] = current_offset
            current_offset += 64

        # Control flow (64 units each)
        for op in [Opcode.JMP, Opcode.BZ, Opcode.BNZ, Opcode.ADJ]:
            allocations[op] = current_offset
            current_offset += 64

        # Heap management (64 units each)
        for op in [Opcode.MALC, Opcode.FREE]:
            allocations[op] = current_offset
            current_offset += 64

        # Multi-layer ops (64 units each for setup/result layers)
        for op in [Opcode.LI, Opcode.LC, Opcode.SI, Opcode.SC,
                   Opcode.PSH, Opcode.JSR, Opcode.ENT, Opcode.LEV]:
            allocations[op] = current_offset
            current_offset += 64

        # Verify we haven't exceeded available units
        # With bit-level primitives:
        # - Bitwise (OR, XOR, AND): 224 units
        # - Shifts (SHL, SHR): 4096 units (still using lookup)
        # - MUL/DIV/MOD: 6144 units (still using lookup)
        # - Other ops: ~2000 units
        # Total: ~12,500 units
        # We dynamically allocate based on actual needs
        if current_offset > self.ffn_hidden:
            raise ValueError(
                f"Unit allocation {current_offset} exceeds ffn_hidden={self.ffn_hidden}. "
                f"Increase ffn_hidden to at least {current_offset}."
            )

        return allocations
        
    def load_all_weights(self, vm, verbose: bool = True):
        """Load all 32 compilable opcode weights into VM.

        Args:
            vm: AutoregressiveVM instance (must have d_model=1352)
            verbose: Print loading progress

        Returns:
            Dictionary with loading statistics
        """
        if vm.d_model != 1352:
            raise ValueError(f"VM must have d_model=1352, got {vm.d_model}")
        
        if verbose:
            print("="*70)
            print("LOADING COMPILED OPCODE WEIGHTS")
            print("="*70)
            print()
        
        stats = {
            'three_layer_loaded': 0,
            'single_op_loaded': 0,
            'multi_op_loaded': 0,
            'multi_layer_loaded': 0,
            'total_params': 0,
            'failed': [],
        }

        # 3-layer arithmetic operations (ADD, SUB) → Layers 9-11
        three_layer_ops = [
            (Opcode.ADD, "ADD"),
            (Opcode.SUB, "SUB"),
        ]

        if verbose:
            print(f"Loading 3-Layer Arithmetic Opcodes → Layers 9-11")
            print("-" * 70)

        # Track unit usage for each layer to allocate non-overlapping ranges
        layer_unit_usage = {9: 0, 10: 0, 11: 0}

        for opcode, name in three_layer_ops:
            try:
                # Extract 3-layer weights from real ALU implementation
                if name == "ADD":
                    three_layer = self.alu_extractor.extract_add_weights(opcode)
                elif name == "SUB":
                    three_layer = self.alu_extractor.extract_sub_weights(opcode)
                else:
                    if verbose:
                        print(f"  ⚠️  {name}: 3-layer extraction not available")
                    continue

                # Load Layer 9 (raw + generate) with unit offset
                unit_offset_l1 = layer_unit_usage[9]
                self._load_into_layer_with_offset(vm, 9, three_layer.layer1, f"{name}-L1", unit_offset_l1, verbose)
                layer_unit_usage[9] += three_layer.layer1['W_up'].shape[0]

                # Load Layer 10 (carry lookahead) with unit offset
                unit_offset_l2 = layer_unit_usage[10]
                self._load_into_layer_with_offset(vm, 10, three_layer.layer2, f"{name}-L2", unit_offset_l2, verbose)
                layer_unit_usage[10] += three_layer.layer2['W_up'].shape[0]

                # Load Layer 11 (finalize) with unit offset
                unit_offset_l3 = layer_unit_usage[11]
                self._load_into_layer_with_offset(vm, 11, three_layer.layer3, f"{name}-L3", unit_offset_l3, verbose)
                layer_unit_usage[11] += three_layer.layer3['W_up'].shape[0]

                stats['three_layer_loaded'] += 1
                stats['total_params'] += sum((w.abs() > 1e-9).sum().item() for w in three_layer.layer1.values())
                stats['total_params'] += sum((w.abs() > 1e-9).sum().item() for w in three_layer.layer2.values())
                stats['total_params'] += sum((w.abs() > 1e-9).sum().item() for w in three_layer.layer3.values())
            except Exception as e:
                stats['failed'].append((name, str(e)))
                if verbose:
                    print(f"  ❌ {name}: {e}")

        print()

        # Split operations across layers to avoid interference
        # Without one-hot opcode encoding, each operation needs its own layer

        # Load each bitwise operation into a separate layer
        if verbose:
            print(f"Loading OR → Layer 6")
            print("-" * 70)
        try:
            unit_offset = self.unit_allocations.get(Opcode.OR, 0)
            weights = self.compiler.compile_opcode(Opcode.OR, unit_offset=unit_offset)
            self._load_into_layer(vm, 6, weights, "OR", verbose)
            stats['single_op_loaded'] += 1
            stats['total_params'] += sum((w.abs() > 1e-9).sum().item() for w in weights.values())
        except Exception as e:
            stats['failed'].append(("OR", str(e)))
            if verbose:
                print(f"  ❌ OR: {e}")

        print()

        if verbose:
            print(f"Loading XOR → Layer 7")
            print("-" * 70)
        try:
            unit_offset = self.unit_allocations.get(Opcode.XOR, 0)
            weights = self.compiler.compile_opcode(Opcode.XOR, unit_offset=unit_offset)
            self._load_into_layer(vm, 7, weights, "XOR", verbose)
            stats['single_op_loaded'] += 1
            stats['total_params'] += sum((w.abs() > 1e-9).sum().item() for w in weights.values())
        except Exception as e:
            stats['failed'].append(("XOR", str(e)))
            if verbose:
                print(f"  ❌ XOR: {e}")

        print()

        if verbose:
            print(f"Loading AND → Layer 8")
            print("-" * 70)
        try:
            unit_offset = self.unit_allocations.get(Opcode.AND, 0)
            weights = self.compiler.compile_opcode(Opcode.AND, unit_offset=unit_offset)
            self._load_into_layer(vm, 8, weights, "AND", verbose)
            stats['single_op_loaded'] += 1
            stats['total_params'] += sum((w.abs() > 1e-9).sum().item() for w in weights.values())
        except Exception as e:
            stats['failed'].append(("AND", str(e)))
            if verbose:
                print(f"  ❌ AND: {e}")

        print()

        # Layer allocation for remaining operations (VM has layers 0-15)
        # Use early layers (0-5) for lookup operations that don't need prior context

        # Layer 0: DIV
        if verbose:
            print(f"Loading DIV → Layer 0")
            print("-" * 70)
        try:
            unit_offset = self.unit_allocations.get(Opcode.DIV, 0)
            weights = self.compiler.compile_opcode(Opcode.DIV, unit_offset=unit_offset)
            self._load_into_layer(vm, 0, weights, "DIV", verbose)
            stats['single_op_loaded'] += 1
            stats['total_params'] += sum((w.abs() > 1e-9).sum().item() for w in weights.values())
        except Exception as e:
            stats['failed'].append(("DIV", str(e)))
            if verbose:
                print(f"  ❌ DIV: {e}")

        print()

        # Layer 1: MOD
        if verbose:
            print(f"Loading MOD → Layer 1")
            print("-" * 70)
        try:
            unit_offset = self.unit_allocations.get(Opcode.MOD, 0)
            weights = self.compiler.compile_opcode(Opcode.MOD, unit_offset=unit_offset)
            self._load_into_layer(vm, 1, weights, "MOD", verbose)
            stats['single_op_loaded'] += 1
            stats['total_params'] += sum((w.abs() > 1e-9).sum().item() for w in weights.values())
        except Exception as e:
            stats['failed'].append(("MOD", str(e)))
            if verbose:
                print(f"  ❌ MOD: {e}")

        print()

        # Layer 2: MUL
        if verbose:
            print(f"Loading MUL → Layer 2")
            print("-" * 70)
        try:
            unit_offset = self.unit_allocations.get(Opcode.MUL, 0)
            weights = self.compiler.compile_opcode(Opcode.MUL, unit_offset=unit_offset)
            self._load_into_layer(vm, 2, weights, "MUL", verbose)
            stats['single_op_loaded'] += 1
            stats['total_params'] += sum((w.abs() > 1e-9).sum().item() for w in weights.values())
        except Exception as e:
            stats['failed'].append(("MUL", str(e)))
            if verbose:
                print(f"  ❌ MUL: {e}")

        print()

        # Layer 3: SHL
        if verbose:
            print(f"Loading SHL → Layer 3")
            print("-" * 70)
        try:
            unit_offset = self.unit_allocations.get(Opcode.SHL, 0)
            weights = self.compiler.compile_opcode(Opcode.SHL, unit_offset=unit_offset)
            self._load_into_layer(vm, 3, weights, "SHL", verbose)
            stats['single_op_loaded'] += 1
            stats['total_params'] += sum((w.abs() > 1e-9).sum().item() for w in weights.values())
        except Exception as e:
            stats['failed'].append(("SHL", str(e)))
            if verbose:
                print(f"  ❌ SHL: {e}")

        print()

        # Layer 4: SHR
        if verbose:
            print(f"Loading SHR → Layer 4")
            print("-" * 70)
        try:
            unit_offset = self.unit_allocations.get(Opcode.SHR, 0)
            weights = self.compiler.compile_opcode(Opcode.SHR, unit_offset=unit_offset)
            self._load_into_layer(vm, 4, weights, "SHR", verbose)
            stats['single_op_loaded'] += 1
            stats['total_params'] += sum((w.abs() > 1e-9).sum().item() for w in weights.values())
        except Exception as e:
            stats['failed'].append(("SHR", str(e)))
            if verbose:
                print(f"  ❌ SHR: {e}")

        print()

        # Layer 5: Register operations (LEA, IMM) - can share since they use different units
        register_ops = [
            (Opcode.LEA, "LEA"),
            (Opcode.IMM, "IMM"),
        ]

        if verbose:
            print(f"Loading Register Opcodes → Layer 5")
            print("-" * 70)

        for opcode, name in register_ops:
            try:
                unit_offset = self.unit_allocations.get(opcode, 0)
                weights = self.compiler.compile_opcode(opcode, unit_offset=unit_offset)
                self._load_into_layer(vm, 5, weights, name, verbose)
                stats['single_op_loaded'] += 1
                stats['total_params'] += sum((w.abs() > 1e-9).sum().item() for w in weights.values())
            except Exception as e:
                stats['failed'].append((name, str(e)))
                if verbose:
                    print(f"  ❌ {name}: {e}")

        print()

        # Layer 12: Comparisons (can share since they use algorithmic approach with small unit counts)
        comparison_ops = [
            (Opcode.EQ, "EQ"),
            (Opcode.NE, "NE"),
            (Opcode.LT, "LT"),
            (Opcode.GT, "GT"),
            (Opcode.LE, "LE"),
            (Opcode.GE, "GE"),
        ]

        if verbose:
            print(f"Loading Comparison Opcodes → Layer 12")
            print("-" * 70)

        for opcode, name in comparison_ops:
            try:
                unit_offset = self.unit_allocations.get(opcode, 0)
                weights = self.compiler.compile_opcode(opcode, unit_offset=unit_offset)
                self._load_into_layer(vm, 12, weights, name, verbose)
                stats['single_op_loaded'] += 1
                stats['total_params'] += sum((w.abs() > 1e-9).sum().item() for w in weights.values())
            except Exception as e:
                stats['failed'].append((name, str(e)))
                if verbose:
                    print(f"  ❌ {name}: {e}")

        print()

        # Multi-operation control flow → Layer 13
        multi_ops = [
            (Opcode.JMP, "JMP"),
            (Opcode.BZ, "BZ"),
            (Opcode.BNZ, "BNZ"),
            (Opcode.ADJ, "ADJ"),
            (Opcode.MALC, "MALC"),
            (Opcode.FREE, "FREE"),
        ]

        if verbose:
            print(f"Loading Multi-Operation Opcodes → Layer 13")
            print("-" * 70)

        for opcode, name in multi_ops:
            try:
                unit_offset = self.unit_allocations.get(opcode, 0)
                weights = self.compiler.compile_opcode(opcode, unit_offset=unit_offset)
                self._load_into_layer(vm, 13, weights, name, verbose)
                stats['multi_op_loaded'] += 1
                stats['total_params'] += sum((w.abs() > 1e-9).sum().item() for w in weights.values())
            except Exception as e:
                stats['failed'].append((name, str(e)))
                if verbose:
                    print(f"  ❌ {name}: {e}")

        print()

        # Multi-layer memory/stack operations → Layers 14-15
        multi_layer_ops = [
            (Opcode.LI, "LI"),
            (Opcode.LC, "LC"),
            (Opcode.SI, "SI"),
            (Opcode.SC, "SC"),
            (Opcode.PSH, "PSH"),
            (Opcode.JSR, "JSR"),
            (Opcode.ENT, "ENT"),
            (Opcode.LEV, "LEV"),
        ]

        if verbose:
            print(f"Loading Multi-Layer Opcodes → Layers 14-15")
            print("-" * 70)

        for opcode, name in multi_layer_ops:
            try:
                layer_weights = self.compiler.compile_multilayer_opcode(opcode)

                # Load setup layer (typically layer 0) → Layer 14
                if 0 in layer_weights and layer_weights[0]:
                    self._load_into_layer(vm, 14,
                                        layer_weights[0], f"{name}-setup", verbose)

                # Load result layer (typically layer 2 or last FFN layer) → Layer 15
                result_layer_idx = max(k for k in layer_weights.keys() if layer_weights[k] is not None)
                if result_layer_idx != 0 and layer_weights[result_layer_idx]:
                    self._load_into_layer(vm, 15,
                                        layer_weights[result_layer_idx], f"{name}-result", verbose)
                
                stats['multi_layer_loaded'] += 1
                for weights in layer_weights.values():
                    if weights:
                        stats['total_params'] += sum((w.abs() > 1e-9).sum().item() for w in weights.values())
            except Exception as e:
                stats['failed'].append((name, str(e)))
                if verbose:
                    print(f"  ❌ {name}: {e}")
        
        print()
        
        # Summary
        if verbose:
            print("="*70)
            print("LOADING SUMMARY")
            print("="*70)
            print(f"  3-layer arithmetic:       {stats['three_layer_loaded']}/2 (ADD, SUB)")
            print(f"  Single-layer opcodes:     {stats['single_op_loaded']}/13")
            print(f"  Multi-operation opcodes:  {stats['multi_op_loaded']}/6")
            print(f"  Multi-layer opcodes:      {stats['multi_layer_loaded']}/8")
            total_loaded = stats['three_layer_loaded'] + stats['single_op_loaded'] + stats['multi_op_loaded'] + stats['multi_layer_loaded']
            print(f"  Total opcodes loaded:     {total_loaded}/29")
            print(f"  Total non-zero params:    {stats['total_params']:,}")

            if stats['failed']:
                print(f"  Failed:                   {len(stats['failed'])}")
                for name, error in stats['failed']:
                    print(f"    - {name}: {error}")

            print("="*70)
        
        return stats
    
    def _load_into_layer_with_offset(self, vm, layer_idx: int, weights: Dict[str, torch.Tensor],
                                    name: str, unit_offset: int, verbose: bool):
        """Load weights into a specific FFN layer with unit offset.

        Args:
            vm: AutoregressiveVM instance
            layer_idx: Layer index
            weights: Weight dictionary
            name: Name for logging
            unit_offset: Hidden unit offset to place weights at
            verbose: Print progress
        """
        layer = vm.blocks[layer_idx].ffn

        weights_hidden = weights['W_up'].shape[0]
        weights_d_model = weights['W_up'].shape[1]

        expected_hidden = layer.W_up.shape[0]
        expected_d_model = layer.W_up.shape[1]

        if weights_d_model != expected_d_model:
            raise ValueError(
                f"Weight d_model mismatch for {name}: "
                f"expected {expected_d_model}, got {weights_d_model}"
            )

        if unit_offset + weights_hidden > expected_hidden:
            raise ValueError(
                f"Unit offset {unit_offset} + {weights_hidden} exceeds layer capacity {expected_hidden}"
            )

        # Place weights at the specified unit offset
        layer.W_up.data[unit_offset:unit_offset+weights_hidden, :] += weights['W_up']
        layer.b_up.data[unit_offset:unit_offset+weights_hidden] += weights['b_up']
        layer.W_gate.data[unit_offset:unit_offset+weights_hidden, :] += weights['W_gate']
        layer.b_gate.data[unit_offset:unit_offset+weights_hidden] += weights['b_gate']
        layer.W_down.data[:, unit_offset:unit_offset+weights_hidden] += weights['W_down']
        # b_down is per d_model, not per hidden unit, so accumulate fully
        layer.b_down.data += weights['b_down']

        if verbose:
            nonzero = sum((w.abs() > 1e-9).sum().item() for w in weights.values())
            print(f"  ✅ {name:15s} → Layer {layer_idx} (units {unit_offset}-{unit_offset+weights_hidden-1}): {nonzero:,} params")

    def _load_into_layer(self, vm, layer_idx: int, weights: Dict[str, torch.Tensor],
                        name: str, verbose: bool):
        """Load weights into a specific FFN layer.

        For layers 9-11 (3-layer arithmetic), weights are replaced since they use
        the full hidden dimension.

        For other layers (12+), weights are accumulated since multiple opcodes
        use non-overlapping hidden unit ranges via unit_offset.
        """
        layer = vm.blocks[layer_idx].ffn

        # Check if weights match layer dimensions
        expected_hidden = layer.W_up.shape[0]
        expected_d_model = layer.W_up.shape[1]
        weights_hidden = weights['W_up'].shape[0]
        weights_d_model = weights['W_up'].shape[1]

        if weights_hidden == expected_hidden and weights_d_model == expected_d_model:
            # Exact match - always accumulate (multiple opcodes can coexist via gating)
            layer.W_up.data += weights['W_up']
            layer.b_up.data += weights['b_up']
            layer.W_gate.data += weights['W_gate']
            layer.b_gate.data += weights['b_gate']
            layer.W_down.data += weights['W_down']
            layer.b_down.data += weights['b_down']
        elif weights_d_model == expected_d_model:
            # d_model matches but hidden dims differ - pad and accumulate
            # This happens when extracted weights have fewer hidden units
            W_up_padded = torch.zeros_like(layer.W_up.data)
            b_up_padded = torch.zeros_like(layer.b_up.data)
            W_gate_padded = torch.zeros_like(layer.W_gate.data)
            b_gate_padded = torch.zeros_like(layer.b_gate.data)
            W_down_padded = torch.zeros_like(layer.W_down.data)
            b_down_padded = torch.zeros_like(layer.b_down.data)

            # Copy extracted weights into padded tensors
            W_up_padded[:weights_hidden, :] = weights['W_up']
            b_up_padded[:weights_hidden] = weights['b_up']
            W_gate_padded[:weights_hidden, :] = weights['W_gate']
            b_gate_padded[:weights_hidden] = weights['b_gate']
            W_down_padded[:, :weights_hidden] = weights['W_down']
            b_down_padded[:] = weights['b_down']

            if layer_idx in [9, 10, 11]:
                # Layers 9-11: Replace
                layer.W_up.data = W_up_padded
                layer.b_up.data = b_up_padded
                layer.W_gate.data = W_gate_padded
                layer.b_gate.data = b_gate_padded
                layer.W_down.data = W_down_padded
                layer.b_down.data = b_down_padded
            else:
                # Other layers: Accumulate
                layer.W_up.data += W_up_padded
                layer.b_up.data += b_up_padded
                layer.W_gate.data += W_gate_padded
                layer.b_gate.data += b_gate_padded
                layer.W_down.data += W_down_padded
                layer.b_down.data += b_down_padded
        else:
            raise ValueError(
                f"Weight dimension mismatch for {name}: "
                f"expected d_model={expected_d_model}, got {weights_d_model}"
            )

        if verbose:
            nonzero = sum((w.abs() > 1e-9).sum().item() for w in weights.values())
            print(f"  ✅ {name:15s} → Layer {layer_idx}: {nonzero:,} params")
    
    def get_layer_mapping(self) -> Dict[str, int]:
        """Get the opcode-to-layer mapping used by this loader.

        New 3-layer architecture:
        - Layers 9-11: 3-layer arithmetic pipeline (ADD, SUB)
        - Layer 12: Single-layer operations (comparisons, bitwise)
        - Layer 13: Control flow operations
        - Layers 14-15: Memory/stack operations
        """
        return {
            'ARITHMETIC_L1': 9,    # Raw + generate
            'ARITHMETIC_L2': 10,   # Carry lookahead
            'ARITHMETIC_L3': 11,   # Finalize
            'SINGLE_LAYER': 12,    # Comparisons, bitwise, etc.
            'CONTROL_FLOW': 13,    # JMP, BZ, BNZ, ADJ, MALC, FREE
            'MEMORY_SETUP': 14,    # Memory/stack setup
            'MEMORY_RESULT': 15,   # Memory/stack result
        }
