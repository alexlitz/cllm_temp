"""
Single-Operation Executor

Tests individual operations end-to-end with compiled weights.

This is the minimal viable executor - it runs one operation through
the neural VM and checks if the result is correct.
"""

import torch
from typing import Dict

from .vm_step import AutoregressiveVM
from .weight_loader import CompiledWeightLoader
from .nibble_embedding import NibbleVMEmbedding
from .embedding import E, Opcode


class SingleOperationExecutor:
    """
    Execute single operations using compiled weights.
    
    This bypasses the full bytecode execution and tests individual
    operations in isolation.
    """
    
    def __init__(self):
        """Initialize executor with compiled weights."""
        # Create VM
        self.vm = AutoregressiveVM(
            d_model=1280,
            n_layers=16,
            n_heads=8,
            ffn_hidden=4096,
        )
        self.vm.eval()
        
        # Load compiled weights
        self.loader = CompiledWeightLoader()
        self.loader.load_all_weights(self.vm, verbose=False)
        
        # Create embedding
        self.embed = NibbleVMEmbedding(d_model=1280)
        
        # Get layer mapping
        self.layer_map = self.loader.get_layer_mapping()
        
    def execute_alu_op(
        self,
        opcode: int,
        operand_a: int,
        operand_b: int,
    ) -> int:
        """
        Execute a single ALU operation.
        
        Args:
            opcode: Opcode to execute (e.g., Opcode.ADD)
            operand_a: First operand
            operand_b: Second operand
            
        Returns:
            Result as 32-bit integer
        """
        # Encode input state
        input_embedding = self.embed.encode_vm_state(
            pc=0,
            ax=operand_a,
            sp=4096,
            bp=4096,
            opcode=opcode,
            stack_top=operand_b,
            batch_size=1,
        )
        
        # Get the layer that handles this opcode
        # Most ALU ops are in PRIMARY_ALU (layer 9)
        layer_idx = self.layer_map['PRIMARY_ALU']
        
        # Run through just the FFN layer
        # (We're bypassing attention and other layers for this test)
        layer = self.vm.blocks[layer_idx]
        
        with torch.no_grad():
            # FFN forward with residual: output = input + delta
            # Where delta = W_down @ (silu(W_up @ x) * W_gate @ x)

            # Up projection
            up = layer.ffn.W_up @ input_embedding.T + layer.ffn.b_up.unsqueeze(1)  # [4096, 1]

            # Gate projection
            gate = layer.ffn.W_gate @ input_embedding.T + layer.ffn.b_gate.unsqueeze(1)  # [4096, 1]

            # SwiGLU activation
            hidden = torch.nn.functional.silu(up) * gate  # [4096, 1]

            # Down projection (delta)
            delta = layer.ffn.W_down @ hidden + layer.ffn.b_down.unsqueeze(1)  # [1280, 1]

            # IMPORTANT: Add residual connection
            output = input_embedding.T + delta  # [1280, 1]

            output_embedding = output.T  # [1, 1280]
        
        # Decode result
        result = self.embed.decode_result_nibbles(output_embedding)
        
        return result
    
    def test_operation(
        self,
        opcode: int,
        op_name: str,
        test_cases: list,  # List of (a, b, expected) tuples
    ) -> Dict:
        """
        Test an operation with multiple test cases.
        
        Args:
            opcode: Opcode to test
            op_name: Name for display
            test_cases: List of (operand_a, operand_b, expected_result)
            
        Returns:
            Dictionary with test results
        """
        results = {
            'opcode': opcode,
            'name': op_name,
            'passed': 0,
            'failed': 0,
            'errors': 0,
            'cases': [],
        }
        
        for a, b, expected in test_cases:
            try:
                result = self.execute_alu_op(opcode, a, b)
                
                if result == expected:
                    results['passed'] += 1
                    results['cases'].append((a, b, expected, result, 'PASS'))
                else:
                    results['failed'] += 1
                    results['cases'].append((a, b, expected, result, 'FAIL'))
            except Exception as e:
                results['errors'] += 1
                results['cases'].append((a, b, expected, None, f'ERROR: {e}'))
        
        return results
