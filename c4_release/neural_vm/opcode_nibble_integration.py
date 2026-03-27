"""
C4 Opcode → Nibble Weight Integration

Bridges the opcode mapper with the nibble weight compiler to generate
AutoregressiveVM-compatible FFN weights for C4 opcodes.

Pipeline:
    C4 Opcode → Computation Graph → Nibble-based FFN Weights → AutoregressiveVM

Usage:
    from neural_vm.opcode_nibble_integration import OpcodeNibbleCompiler
    from neural_vm.embedding import Opcode

    compiler = OpcodeNibbleCompiler()

    # Generate nibble-based weights for ADD opcode
    weights = compiler.compile_opcode(Opcode.ADD)

    # Load into AutoregressiveVM
    vm.blocks[9].ffn.W_up.data = weights['W_up']
    # ... etc
"""

import torch
from typing import Dict, Optional
from dataclasses import dataclass

from .nibble_weight_compiler import NibbleWeightCompiler, NibbleRegisterMap
from .graph_weight_compiler import OpType
from .opcode_mapper import OpcodeMapper, C4Opcode, OpcodeSupport
from .embedding import Opcode
from .multi_operation_compiler import MultiOperationCompiler
from .multi_layer_generator import MultiLayerWeightGenerator


@dataclass
class NibbleOpcodeSupport:
    """Classification of C4 opcodes by nibble compilation support."""

    # Directly compilable to nibble FFN (ALU operations)
    FFN_DIRECT = "ffn_direct"

    # Requires attention (memory operations)
    ATTENTION_NEEDED = "attention_needed"

    # Not supported yet
    NOT_SUPPORTED = "not_supported"


# Map C4 opcodes to nibble compilation support
_NIBBLE_OPCODE_SUPPORT = {
    # Arithmetic (direct nibble FFN)
    Opcode.ADD: (NibbleOpcodeSupport.FFN_DIRECT, OpType.ADD),
    Opcode.SUB: (NibbleOpcodeSupport.FFN_DIRECT, OpType.SUB),
    Opcode.MUL: (NibbleOpcodeSupport.FFN_DIRECT, OpType.MUL),
    Opcode.DIV: (NibbleOpcodeSupport.FFN_DIRECT, OpType.DIV),
    Opcode.MOD: (NibbleOpcodeSupport.FFN_DIRECT, OpType.MOD),

    # Comparison (direct nibble FFN)
    Opcode.EQ: (NibbleOpcodeSupport.FFN_DIRECT, OpType.CMP_EQ),
    Opcode.NE: (NibbleOpcodeSupport.FFN_DIRECT, OpType.CMP_NE),
    Opcode.LT: (NibbleOpcodeSupport.FFN_DIRECT, OpType.CMP_LT),
    Opcode.GT: (NibbleOpcodeSupport.FFN_DIRECT, OpType.CMP_GT),
    Opcode.LE: (NibbleOpcodeSupport.FFN_DIRECT, OpType.CMP_LE),
    Opcode.GE: (NibbleOpcodeSupport.FFN_DIRECT, OpType.CMP_GE),

    # Bitwise (direct nibble FFN)
    Opcode.OR: (NibbleOpcodeSupport.FFN_DIRECT, OpType.BIT_OR),
    Opcode.XOR: (NibbleOpcodeSupport.FFN_DIRECT, OpType.BIT_XOR),
    Opcode.AND: (NibbleOpcodeSupport.FFN_DIRECT, OpType.BIT_AND),
    Opcode.SHL: (NibbleOpcodeSupport.FFN_DIRECT, OpType.SHL),
    Opcode.SHR: (NibbleOpcodeSupport.FFN_DIRECT, OpType.SHR),

    # Stack/Register (direct nibble FFN - simple operations)
    Opcode.IMM: (NibbleOpcodeSupport.FFN_DIRECT, OpType.MOVE),  # AX = imm
    Opcode.LEA: (NibbleOpcodeSupport.FFN_DIRECT, OpType.ADD),   # AX = BP + imm

    # Memory (require attention)
    Opcode.LI: (NibbleOpcodeSupport.ATTENTION_NEEDED, None),
    Opcode.LC: (NibbleOpcodeSupport.ATTENTION_NEEDED, None),
    Opcode.SI: (NibbleOpcodeSupport.ATTENTION_NEEDED, None),
    Opcode.SC: (NibbleOpcodeSupport.ATTENTION_NEEDED, None),
    Opcode.PSH: (NibbleOpcodeSupport.ATTENTION_NEEDED, None),

    # Function calls (require attention for stack ops)
    Opcode.JSR: (NibbleOpcodeSupport.ATTENTION_NEEDED, None),
    Opcode.ENT: (NibbleOpcodeSupport.ATTENTION_NEEDED, None),
    Opcode.LEV: (NibbleOpcodeSupport.ATTENTION_NEEDED, None),

    # Control flow (multi-operation FFN)
    Opcode.JMP: (NibbleOpcodeSupport.FFN_DIRECT, None),  # Multi-op: PC_SET
    Opcode.BZ: (NibbleOpcodeSupport.FFN_DIRECT, None),   # Multi-op: CMP + ADD + PC_CONDITIONAL
    Opcode.BNZ: (NibbleOpcodeSupport.FFN_DIRECT, None),  # Multi-op: CMP + ADD + PC_CONDITIONAL
    Opcode.ADJ: (NibbleOpcodeSupport.FFN_DIRECT, None),  # Multi-op: ADD (SP adjustment)

    # Heap management (multi-operation FFN)
    Opcode.MALC: (NibbleOpcodeSupport.FFN_DIRECT, None),  # Multi-op: ADD + CMP + SELECT
    Opcode.FREE: (NibbleOpcodeSupport.FFN_DIRECT, None),  # No-op (bump allocator)
}


class OpcodeNibbleCompiler:
    """Compile C4 opcodes to nibble-based FFN weights.

    This is the high-level interface that bridges:
    - C4 Opcode enum (from embedding.py)
    - OpType enum (from graph_weight_compiler.py)
    - Nibble-based weight generation (from nibble_weight_compiler.py)
    - AutoregressiveVM FFN format

    Example:
        compiler = OpcodeNibbleCompiler()

        # Check if opcode is supported
        if compiler.is_compilable(Opcode.ADD):
            # Generate weights
            weights = compiler.compile_opcode(Opcode.ADD)

            # Load into AutoregressiveVM layer 9 (ALU)
            vm.blocks[9].ffn.W_up.data = weights['W_up']
            vm.blocks[9].ffn.b_up.data = weights['b_up']
            vm.blocks[9].ffn.W_gate.data = weights['W_gate']
            vm.blocks[9].ffn.b_gate.data = weights['b_gate']
            vm.blocks[9].ffn.W_down.data = weights['W_down']
            vm.blocks[9].ffn.b_down.data = weights['b_down']
    """

    def __init__(self, num_positions: int = 8):
        """Initialize opcode nibble compiler.

        Args:
            num_positions: Number of nibble positions (8 for 32-bit)
        """
        self.num_positions = num_positions
        self.nibble_compiler = NibbleWeightCompiler(num_positions)
        self.multi_op_compiler = MultiOperationCompiler(num_positions)
        self.multi_layer_gen = MultiLayerWeightGenerator(num_positions)
        self.reg_map = NibbleRegisterMap()

    def is_compilable(self, opcode: int) -> bool:
        """Check if opcode can be compiled to nibble FFN.

        Args:
            opcode: C4 opcode (from Opcode class)

        Returns:
            True if opcode can be compiled to pure FFN weights
        """
        if opcode not in _NIBBLE_OPCODE_SUPPORT:
            return False

        support, _ = _NIBBLE_OPCODE_SUPPORT[opcode]
        return support == NibbleOpcodeSupport.FFN_DIRECT

    def get_support_level(self, opcode: int) -> str:
        """Get support level for an opcode.

        Args:
            opcode: C4 opcode

        Returns:
            Support level string ("ffn_direct", "attention_needed", "not_supported")
        """
        if opcode not in _NIBBLE_OPCODE_SUPPORT:
            return NibbleOpcodeSupport.NOT_SUPPORTED

        support, _ = _NIBBLE_OPCODE_SUPPORT[opcode]
        return support

    def compile_opcode(self, opcode: int) -> Dict[str, torch.Tensor]:
        """Compile C4 opcode to nibble-based FFN weights.

        Handles both single-operation and multi-operation graphs.

        Args:
            opcode: C4 opcode to compile (from Opcode class)

        Returns:
            Dictionary of weight matrices for PureFFN:
            {
                'W_up': [4096, 1280],
                'b_up': [4096],
                'W_gate': [4096, 1280],
                'b_gate': [4096],
                'W_down': [1280, 4096],
                'b_down': [1280]
            }

        Raises:
            ValueError: If opcode is not supported for nibble compilation
        """
        if opcode not in _NIBBLE_OPCODE_SUPPORT:
            raise ValueError(f"Opcode {opcode} not in support table")

        support, op_type = _NIBBLE_OPCODE_SUPPORT[opcode]

        if support != NibbleOpcodeSupport.FFN_DIRECT:
            raise ValueError(
                f"Opcode {opcode} requires {support}, cannot compile to pure FFN"
            )

        # Check if this is a multi-operation opcode
        if op_type is None:
            # Multi-operation - need to compile from graph
            from .full_opcode_mapper import FullOpcodeMapper
            mapper = FullOpcodeMapper()

            # Get graph (with immediate if needed)
            if opcode in [Opcode.JMP, Opcode.BZ, Opcode.BNZ]:
                graph = mapper.map_opcode(opcode, imm=0)
            elif opcode in [Opcode.ADJ, Opcode.LEA]:
                graph = mapper.map_opcode(opcode, imm=8)
            else:
                graph = mapper.map_opcode(opcode)

            # Check if single or multi-operation
            from .graph_weight_compiler import OpType as GOpType
            ops = [n for n in graph.nodes.values() if n.op != GOpType.CONST]

            if len(ops) == 1 and ops[0].op in [GOpType.ADD, GOpType.SUB, GOpType.MOVE]:
                # Single operation - use existing compiler
                return self.nibble_compiler.compile_operation(ops[0].op, opcode)
            else:
                # Multi-operation - use new compiler
                return self.multi_op_compiler.compile_graph(graph, opcode)
        else:
            # Single operation - use existing compiler
            return self.nibble_compiler.compile_operation(op_type, opcode)

    def compile_multilayer_opcode(self, opcode: int) -> Dict[int, Dict[str, torch.Tensor]]:
        """Compile C4 opcode that requires multiple layers (FFN + Attention).

        This handles opcodes like LI, SI, PSH, JSR that need:
        - Layer N (FFN): Setup operation
        - Layer N+1 (Attention): Memory read/write
        - Layer N+2 (FFN): Copy result

        Args:
            opcode: C4 opcode to compile (from Opcode class)

        Returns:
            Dictionary mapping layer_index → weight_dict
            {
                0: {'W_up': ..., 'b_up': ..., ...},  # Setup FFN
                1: None,                              # Attention (skip)
                2: {'W_up': ..., 'b_up': ..., ...},  # Copy FFN
            }

        Raises:
            ValueError: If opcode is not a multi-layer opcode

        Example:
            compiler = OpcodeNibbleCompiler()
            weights = compiler.compile_multilayer_opcode(Opcode.LI)

            # Load into VM layers
            vm.blocks[9].ffn.load_state_dict(weights[0])   # Setup
            vm.blocks[11].ffn.load_state_dict(weights[2])  # Copy
        """
        if opcode not in _NIBBLE_OPCODE_SUPPORT:
            raise ValueError(f"Opcode {opcode} not in support table")

        support, _ = _NIBBLE_OPCODE_SUPPORT[opcode]

        if support != NibbleOpcodeSupport.ATTENTION_NEEDED:
            raise ValueError(
                f"Opcode {opcode} does not require multi-layer compilation (support={support})"
            )

        # Use multi-layer generator
        return self.multi_layer_gen.compile_opcode(opcode)

    def print_support_summary(self):
        """Print summary of opcode compilation support."""
        print("="*70)
        print("C4 Opcode → Nibble FFN Compilation Support")
        print("="*70)

        # Group by support level
        ffn_direct = []
        attention_needed = []
        not_supported = []

        for opcode in range(72):  # All possible opcodes
            if opcode not in _NIBBLE_OPCODE_SUPPORT:
                continue

            support, op_type = _NIBBLE_OPCODE_SUPPORT[opcode]

            # Get opcode name
            opcode_name = None
            for name, value in vars(Opcode).items():
                if name.isupper() and value == opcode:
                    opcode_name = name
                    break

            if opcode_name is None:
                opcode_name = f"OPCODE_{opcode}"

            entry = (opcode, opcode_name, op_type)

            if support == NibbleOpcodeSupport.FFN_DIRECT:
                ffn_direct.append(entry)
            elif support == NibbleOpcodeSupport.ATTENTION_NEEDED:
                attention_needed.append(entry)
            else:
                not_supported.append(entry)

        # Print FFN Direct
        print(f"\n✅ FFN Direct ({len(ffn_direct)} opcodes):")
        print("  Can be compiled to pure nibble-based FFN weights")
        print()
        for opcode, name, op_type in sorted(ffn_direct):
            op_str = op_type.value if op_type else "N/A"
            print(f"    {opcode:2d} {name:8s} → {op_str}")

        # Print Attention Needed
        print(f"\n⚠️  Attention Needed ({len(attention_needed)} opcodes):")
        print("  Require attention mechanism for memory operations")
        print()
        for opcode, name, op_type in sorted(attention_needed):
            print(f"    {opcode:2d} {name:8s}")

        # Print Not Supported
        print(f"\n❌ Not Supported ({len(not_supported)} opcodes):")
        print("  Require additional implementation")
        print()
        for opcode, name, op_type in sorted(not_supported):
            print(f"    {opcode:2d} {name:8s}")

        # Summary
        total = len(ffn_direct) + len(attention_needed) + len(not_supported)
        ffn_pct = 100 * len(ffn_direct) / total if total > 0 else 0
        print()
        print("="*70)
        print(f"Total: {len(ffn_direct)}/{total} opcodes compilable to FFN ({ffn_pct:.1f}%)")
        print("="*70)

    def load_weights_into_vm(self, vm, layer_idx: int, weights: Dict[str, torch.Tensor]):
        """Load compiled weights into an AutoregressiveVM layer.

        Args:
            vm: AutoregressiveVM instance
            layer_idx: Layer index (typically 9-12 for ALU)
            weights: Compiled weight dictionary

        Example:
            weights = compiler.compile_opcode(Opcode.ADD)
            compiler.load_weights_into_vm(vm, layer_idx=9, weights=weights)
        """
        layer = vm.blocks[layer_idx]

        # Check dimensions match
        expected_dim = self.num_positions * self.reg_map.DIM  # 1280
        expected_hidden = 4096

        if layer.ffn.W_up.shape != (expected_hidden, expected_dim):
            raise ValueError(
                f"Layer {layer_idx} FFN dimension mismatch. "
                f"Expected ({expected_hidden}, {expected_dim}), "
                f"got {layer.ffn.W_up.shape}"
            )

        # Load weights
        layer.ffn.W_up.data = weights['W_up']
        layer.ffn.b_up.data = weights['b_up']
        layer.ffn.W_gate.data = weights['W_gate']
        layer.ffn.b_gate.data = weights['b_gate']
        layer.ffn.W_down.data = weights['W_down']
        layer.ffn.b_down.data = weights['b_down']

        print(f"✅ Loaded weights into layer {layer_idx}")
        print(f"   Non-zero params: {(weights['W_up'].abs() > 1e-9).sum().item():,}")

    def compile_all_ffn_opcodes(self) -> Dict[int, Dict[str, torch.Tensor]]:
        """Compile all FFN-compilable opcodes.

        Returns:
            Dictionary mapping opcode → weights
        """
        compiled = {}

        for opcode in range(72):
            if self.is_compilable(opcode):
                try:
                    weights = self.compile_opcode(opcode)
                    compiled[opcode] = weights
                except Exception as e:
                    print(f"⚠️  Failed to compile opcode {opcode}: {e}")

        return compiled
