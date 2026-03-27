"""
Weight Loader for Compiled Opcodes

Loads all compiled opcode weights into an AutoregressiveVM instance.

Usage:
    from neural_vm.weight_loader import CompiledWeightLoader
    from neural_vm.vm_step import AutoregressiveVM
    
    vm = AutoregressiveVM(d_model=1280, n_layers=16, ffn_hidden=4096)
    loader = CompiledWeightLoader()
    loader.load_all_weights(vm)
    
    # Now VM has all 32 opcode weights loaded
"""

import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .opcode_nibble_integration import OpcodeNibbleCompiler
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
    
    def __init__(self):
        self.compiler = OpcodeNibbleCompiler()
        self.layer_map = OpcodeLayerMapping()
        
    def load_all_weights(self, vm, verbose: bool = True):
        """Load all 32 compilable opcode weights into VM.
        
        Args:
            vm: AutoregressiveVM instance (must have d_model=1280)
            verbose: Print loading progress
            
        Returns:
            Dictionary with loading statistics
        """
        if vm.d_model != 1280:
            raise ValueError(f"VM must have d_model=1280, got {vm.d_model}")
        
        if verbose:
            print("="*70)
            print("LOADING COMPILED OPCODE WEIGHTS")
            print("="*70)
            print()
        
        stats = {
            'single_op_loaded': 0,
            'multi_op_loaded': 0,
            'multi_layer_loaded': 0,
            'total_params': 0,
            'failed': [],
        }
        
        # Single-operation opcodes → Layer 9
        single_ops = [
            (Opcode.ADD, "ADD"),
            (Opcode.SUB, "SUB"),
            (Opcode.MUL, "MUL"),
            (Opcode.DIV, "DIV"),
            (Opcode.MOD, "MOD"),
            (Opcode.EQ, "EQ"),
            (Opcode.NE, "NE"),
            (Opcode.LT, "LT"),
            (Opcode.GT, "GT"),
            (Opcode.LE, "LE"),
            (Opcode.GE, "GE"),
            (Opcode.OR, "OR"),
            (Opcode.XOR, "XOR"),
            (Opcode.AND, "AND"),
            (Opcode.SHL, "SHL"),
            (Opcode.SHR, "SHR"),
            (Opcode.LEA, "LEA"),
            (Opcode.IMM, "IMM"),
        ]
        
        if verbose:
            print(f"Loading Single-Operation Opcodes → Layer {self.layer_map.PRIMARY_ALU}")
            print("-" * 70)
        
        for opcode, name in single_ops:
            try:
                weights = self.compiler.compile_opcode(opcode)
                self._load_into_layer(vm, self.layer_map.PRIMARY_ALU, weights, name, verbose)
                stats['single_op_loaded'] += 1
                stats['total_params'] += sum((w.abs() > 1e-9).sum().item() for w in weights.values())
            except Exception as e:
                stats['failed'].append((name, str(e)))
                if verbose:
                    print(f"  ❌ {name}: {e}")
        
        print()
        
        # Multi-operation control flow → Layer 10
        multi_ops = [
            (Opcode.JMP, "JMP"),
            (Opcode.BZ, "BZ"),
            (Opcode.BNZ, "BNZ"),
            (Opcode.ADJ, "ADJ"),
            (Opcode.MALC, "MALC"),
            (Opcode.FREE, "FREE"),
        ]
        
        if verbose:
            print(f"Loading Multi-Operation Opcodes → Layer {self.layer_map.CONTROL_FLOW}")
            print("-" * 70)
        
        for opcode, name in multi_ops:
            try:
                weights = self.compiler.compile_opcode(opcode)
                self._load_into_layer(vm, self.layer_map.CONTROL_FLOW, weights, name, verbose)
                stats['multi_op_loaded'] += 1
                stats['total_params'] += sum((w.abs() > 1e-9).sum().item() for w in weights.values())
            except Exception as e:
                stats['failed'].append((name, str(e)))
                if verbose:
                    print(f"  ❌ {name}: {e}")
        
        print()
        
        # Multi-layer memory/stack operations
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
            print(f"Loading Multi-Layer Opcodes → Layers {self.layer_map.MEMORY_SETUP}, {self.layer_map.MEMORY_RESULT}")
            print("-" * 70)
        
        for opcode, name in multi_layer_ops:
            try:
                layer_weights = self.compiler.compile_multilayer_opcode(opcode)
                
                # Load setup layer (typically layer 0)
                if 0 in layer_weights and layer_weights[0]:
                    self._load_into_layer(vm, self.layer_map.MEMORY_SETUP, 
                                        layer_weights[0], f"{name}-setup", verbose)
                
                # Load result layer (typically layer 2 or last FFN layer)
                result_layer_idx = max(k for k in layer_weights.keys() if layer_weights[k] is not None)
                if result_layer_idx != 0 and layer_weights[result_layer_idx]:
                    self._load_into_layer(vm, self.layer_map.MEMORY_RESULT,
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
            print(f"  Single-operation opcodes: {stats['single_op_loaded']}/18")
            print(f"  Multi-operation opcodes:  {stats['multi_op_loaded']}/6")
            print(f"  Multi-layer opcodes:      {stats['multi_layer_loaded']}/8")
            print(f"  Total opcodes loaded:     {stats['single_op_loaded'] + stats['multi_op_loaded'] + stats['multi_layer_loaded']}/32")
            print(f"  Total non-zero params:    {stats['total_params']:,}")
            
            if stats['failed']:
                print(f"  Failed:                   {len(stats['failed'])}")
                for name, error in stats['failed']:
                    print(f"    - {name}: {error}")
            
            print("="*70)
        
        return stats
    
    def _load_into_layer(self, vm, layer_idx: int, weights: Dict[str, torch.Tensor],
                        name: str, verbose: bool):
        """Load weights into a specific FFN layer.

        Weights are accumulated (added) rather than replaced, since each opcode
        uses different hidden units (selected by opcode gating in W_gate).
        """
        layer = vm.blocks[layer_idx].ffn

        # Accumulate weights (opcode-gated, so they don't interfere)
        layer.W_up.data += weights['W_up']
        layer.b_up.data += weights['b_up']
        layer.W_gate.data += weights['W_gate']
        layer.b_gate.data += weights['b_gate']
        layer.W_down.data += weights['W_down']
        layer.b_down.data += weights['b_down']

        if verbose:
            nonzero = sum((w.abs() > 1e-9).sum().item() for w in weights.values())
            print(f"  ✅ {name:15s} → Layer {layer_idx}: {nonzero:,} params")
    
    def get_layer_mapping(self) -> Dict[str, int]:
        """Get the opcode-to-layer mapping used by this loader."""
        return {
            'PRIMARY_ALU': self.layer_map.PRIMARY_ALU,
            'CONTROL_FLOW': self.layer_map.CONTROL_FLOW,
            'MEMORY_SETUP': self.layer_map.MEMORY_SETUP,
            'MEMORY_ATTENTION': self.layer_map.MEMORY_ATTENTION,
            'MEMORY_RESULT': self.layer_map.MEMORY_RESULT,
        }
