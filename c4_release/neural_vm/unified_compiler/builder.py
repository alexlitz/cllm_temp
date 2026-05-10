"""
IR Builder for Neural VM Compiler.

Constructs CompilerIR from opcode specifications.
This is the "frontend" that translates high-level specs into IR.
"""

from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field

from .ir import (
    CompilerIR, LayerSpec, AttentionOp, FFNOp,
    AttentionOpType, FFNOpType, DimensionAlloc
)
from .opcodes import OPCODES, OpcodeSpec, OpcodeCategory
from .allocator import create_standard_allocator, DimensionAllocator


@dataclass
class PruningConfig:
    """Configuration for weight pruning.

    Allows specifying which opcodes to include in the compiled model,
    reducing weight count for specialized use cases.

    Attributes:
        enabled_opcodes: Set of opcode names to include (e.g., {"ADD", "SUB", "IMM"}).
                        If None, all opcodes are included.
        include_memory: Include memory operations (LI, LC, SI, SC). Default True.
        include_stack: Include stack operations (PSH, ENT, ADJ, LEV). Default True.
        include_control: Include control flow (JMP, JSR, BZ, BNZ). Default True.
        include_syscalls: Include syscall ops (PUTCHAR, GETCHAR, EXIT). Default True.
    """
    enabled_opcodes: Optional[Set[str]] = None
    include_memory: bool = True
    include_stack: bool = True
    include_control: bool = True
    include_syscalls: bool = True

    def should_include(self, opcode_name: str) -> bool:
        """Check if an opcode should be included based on config."""
        # Explicit list takes precedence
        if self.enabled_opcodes is not None:
            return opcode_name in self.enabled_opcodes

        # Category-based filtering
        memory_ops = {"LI", "LC", "SI", "SC", "LEA"}
        stack_ops = {"PSH", "ENT", "ADJ", "LEV"}
        control_ops = {"JMP", "JSR", "BZ", "BNZ"}
        syscall_ops = {"PUTCHAR", "GETCHAR", "EXIT", "NOP"}

        if opcode_name in memory_ops and not self.include_memory:
            return False
        if opcode_name in stack_ops and not self.include_stack:
            return False
        if opcode_name in control_ops and not self.include_control:
            return False
        if opcode_name in syscall_ops and not self.include_syscalls:
            return False

        return True


@dataclass
class BuilderConfig:
    """Configuration for IR building."""
    d_model: int = 512
    n_layers: int = 17
    n_heads: int = 8
    head_dim: int = 64
    scale: float = 100.0
    pruning: Optional[PruningConfig] = None


class IRBuilder:
    """Builds CompilerIR from opcode specifications.

    Takes opcode specs and generates the IR operations needed
    to implement them in the transformer.
    """

    def __init__(self, config: Optional[BuilderConfig] = None):
        self.config = config or BuilderConfig()
        self.ir: Optional[CompilerIR] = None
        self.allocator: Optional[DimensionAllocator] = None
        self._unit_counters: Dict[int, int] = {}  # layer -> next unit

    def build(self, opcodes: Optional[List[OpcodeSpec]] = None) -> CompilerIR:
        """Build complete IR for given opcodes.

        Args:
            opcodes: List of opcode specs to implement.
                    If None, uses all registered opcodes (subject to pruning config).

        Returns:
            Complete CompilerIR ready for code generation.
        """
        if opcodes is None:
            opcodes = list(OPCODES.values())

        # Apply pruning if configured
        if self.config.pruning is not None:
            opcodes = [op for op in opcodes if self.config.pruning.should_include(op.name)]
            self.ir_metadata_pruned = [op.name for op in opcodes]

        # Initialize
        self.allocator = create_standard_allocator()
        self.ir = CompilerIR(
            d_model=self.config.d_model,
            n_layers=self.config.n_layers,
            n_heads=self.config.n_heads,
        )
        self.ir.dimensions = self.allocator.to_ir()
        self._unit_counters = {i: 0 for i in range(self.config.n_layers)}

        # Build layers
        self._build_embedding_layer()
        self._build_threshold_layers()      # L0-L2
        self._build_carry_forward_layer()   # L3
        self._build_fetch_layers()          # L4-L5
        self._build_alu_layers(opcodes)     # L6-L12
        self._build_memory_layers()         # L13-L15
        self._build_writeback_layer()       # L16

        return self.ir

    def _get_dim(self, name: str) -> int:
        """Get starting dimension index."""
        return self.allocator.get_start(name)

    def _get_dims(self, name: str) -> List[int]:
        """Get list of dimension indices."""
        return list(self.allocator.get_range(name))

    def _next_unit(self, layer: int, count: int = 1) -> int:
        """Get next available FFN unit for a layer."""
        unit = self._unit_counters[layer]
        self._unit_counters[layer] += count
        return unit

    def _build_embedding_layer(self) -> None:
        """Build embedding layer operations.

        Embedding sets markers and nibble encodings.
        """
        # Embedding is handled by model initialization
        # Just mark dimensions as written by embedding (-1)
        pass

    def _build_threshold_layers(self) -> None:
        """Build L0-L2 threshold detection layers.

        L0: 8 threshold heads for marker distance detection
        L1: Fine thresholds (0.5, 1.5, 2.5, 6.5) + HAS_SE detection
        L2: Threshold 5.5 for MEM byte position detection
        """
        # L0: Coarse threshold attention heads
        l0 = self.ir.get_layer(0)

        # 8 heads for distances - must match vm_step.py
        l0_thresholds = [3.5, 4.5, 7.5, 8.5, 9.5, 14.5, 19.5, 24.5]
        l0_outputs = ["H0", "H1", "H2", "H3", "H4", "H5", "H6", "H7"]

        # 7 marker type dimensions for V - must match vm_step._SetDim.MARKS
        # MARKS = [MARK_PC, MARK_AX, MARK_SP, MARK_BP, MARK_MEM, MARK_SE, MARK_CS]
        marker_dims = [
            self._get_dim("MARK_PC"),
            self._get_dim("MARK_AX"),
            self._get_dim("MARK_SP"),
            self._get_dim("MARK_BP"),
            self._get_dim("MARK_MEM"),
            self._get_dim("MARK_SE"),
            self._get_dim("MARK_CS"),
        ]

        for head_idx, (threshold, out_name) in enumerate(zip(l0_thresholds, l0_outputs)):
            l0.attention_ops.append(AttentionOp(
                op_type=AttentionOpType.THRESHOLD,
                head_idx=head_idx,
                q_dims=[self._get_dim("CONST")],
                k_dims=[self._get_dim("IS_MARK")],
                v_dims=marker_dims,  # 7 marker types
                o_dims=self._get_dims(out_name),
                params={'threshold': threshold, 'slope': 10.0},  # ALIBI_S = 10.0
            ))

        l0.reads.update(["CONST", "IS_MARK"] + [f"MARK_{m}" for m in ["PC", "AX", "SP", "BP", "MEM", "CS", "STACK0"]])
        l0.writes.update(l0_outputs)

        # L0 FFN: Marker transition detection (NEXT_PC, NEXT_AX, etc.)
        self._build_l0_ffn_ops(l0)

        # L1: Fine threshold heads + HAS_SE detection
        l1 = self.ir.get_layer(1)

        # L1H0-L1H2: Fine thresholds (0.5, 1.5, 2.5) at heads 0-2
        l1_thresholds = [0.5, 1.5, 2.5]
        l1_outputs = ["L1H0", "L1H1", "L1H2"]

        for head_idx, (threshold, out_name) in enumerate(zip(l1_thresholds, l1_outputs)):
            l1.attention_ops.append(AttentionOp(
                op_type=AttentionOpType.THRESHOLD,
                head_idx=head_idx,
                q_dims=[self._get_dim("CONST")],
                k_dims=[self._get_dim("IS_MARK")],
                v_dims=marker_dims,  # 7 marker types (reuse from L0)
                o_dims=self._get_dims(out_name),
                params={'threshold': threshold, 'slope': 10.0},  # ALIBI_S = 10.0
            ))

        # L1 Head 3: HAS_SE detection via global attention to MARK_SE_ONLY
        # Q/K use scale 10.0, ALiBi slope is 0.0 (global attention)
        l1.attention_ops.append(AttentionOp(
            op_type=AttentionOpType.DETECT,
            head_idx=3,
            q_dims=[self._get_dim("CONST")],
            k_dims=[self._get_dim("MARK_SE_ONLY")],
            v_dims=[self._get_dim("MARK_SE_ONLY")],
            o_dims=[self._get_dim("HAS_SE")],
            params={'scale': 10.0, 'alibi_slope': 0.0, 'output_scale': 1.0},
        ))

        # L1 Head 4: Threshold 6.5 -> L1H4
        l1.attention_ops.append(AttentionOp(
            op_type=AttentionOpType.THRESHOLD,
            head_idx=4,
            q_dims=[self._get_dim("CONST")],
            k_dims=[self._get_dim("IS_MARK")],
            v_dims=marker_dims,
            o_dims=self._get_dims("L1H4"),
            params={'threshold': 6.5, 'slope': 10.0},  # ALIBI_S = 10.0
        ))

        l1.reads.update(["CONST", "IS_MARK", "IS_SE"])
        l1.writes.update(["L1H0", "L1H1", "L1H2", "HAS_SE", "L1H4"])

        # L1 FFN: BYTE_INDEX flags
        self._build_l1_ffn_ops(l1)

        # L2: Threshold 5.5 for MEM byte detection
        l2 = self.ir.get_layer(2)

        l2.attention_ops.append(AttentionOp(
            op_type=AttentionOpType.THRESHOLD,
            head_idx=0,
            q_dims=[self._get_dim("CONST")],
            k_dims=[self._get_dim("IS_MARK")],
            v_dims=marker_dims,
            o_dims=self._get_dims("L2H0"),
            params={'threshold': 5.5, 'slope': 10.0},  # ALIBI_S = 10.0
        ))

        l2.reads.update(["CONST", "IS_MARK"])
        l2.writes.update(["L2H0"])

        # L2 FFN: MEM byte position flags
        self._build_l2_ffn_ops(l2)

    def _build_l0_ffn_ops(self, l0: LayerSpec) -> None:
        """Build L0 FFN operations for marker transitions.

        Detects transitions between markers in 35-token step structure:
        SE→PC, PC→AX, AX→SP, SP→BP, BP→STACK0, STACK0→MEM, MEM→SE
        """
        PC_I, AX_I, SP_I, BP_I, MEM_I, SE_I = 0, 1, 2, 3, 4, 5

        # Transitions: (up_dim_name, up_offset, gate_dim_name, gate_offset, out_dim_name)
        transitions = [
            # SE → PC: SE is nearest marker at d<=3.5
            ("H0", SE_I, None, None, "NEXT_PC"),
            # PC → AX: PC at d<=4.5 (H1) but not d<=3.5 (H0)
            ("H1", PC_I, "H0", PC_I, "NEXT_AX"),
            # AX → SP
            ("H1", AX_I, "H0", AX_I, "NEXT_SP"),
            # SP → BP
            ("H1", SP_I, "H0", SP_I, "NEXT_BP"),
            # BP → STACK0
            ("H1", BP_I, "H0", BP_I, "NEXT_STACK0"),
            # STACK0 → MEM: BP at d=9 (H4 - H3)
            ("H4", BP_I, "H3", BP_I, "NEXT_MEM"),
            # MEM → SE: MEM at d=8 (H3 - H2)
            ("H3", MEM_I, "H2", MEM_I, "NEXT_SE"),
        ]

        for up_name, up_off, gate_name, gate_off, out_name in transitions:
            unit = self._next_unit(0, 1)
            gate_dims = [self._get_dim(gate_name) + gate_off] if gate_name else []
            l0.ffn_ops.append(FFNOp(
                op_type=FFNOpType.THRESHOLD_FLAG,
                unit_start=unit,
                unit_count=1,
                gate_dims=gate_dims,
                input_dims=[self._get_dim(up_name) + up_off],
                output_dims=[self._get_dim(out_name)],
                params={'threshold': 0.3, 'negate_gate': True, 'out_scale': 1.0},
            ))

        l0.reads.update(["H0", "H1", "H2", "H3", "H4"])
        l0.writes.update(["NEXT_PC", "NEXT_AX", "NEXT_SP", "NEXT_BP", "NEXT_STACK0", "NEXT_MEM", "NEXT_SE"])

    def _build_l1_ffn_ops(self, l1: LayerSpec) -> None:
        """Build L1 FFN operations for BYTE_INDEX flags."""
        # STACK0_BYTE0: L1H4[BP] AND NOT H1[BP] AND IS_BYTE
        unit = self._next_unit(1, 1)
        l1.ffn_ops.append(FFNOp(
            op_type=FFNOpType.THRESHOLD_FLAG,
            unit_start=unit,
            unit_count=1,
            gate_dims=[self._get_dim("H1") + 3],  # H1[BP] as suppression
            input_dims=[self._get_dim("L1H4") + 3, self._get_dim("IS_BYTE")],  # L1H4[BP], IS_BYTE
            output_dims=[self._get_dim("STACK0_BYTE0")],
            params={'threshold': 1.5, 'negate_gate': True},
        ))

        # BYTE_INDEX_0: IS_BYTE AND any(L1H1[i]) AND NOT any(L1H0[i])
        unit = self._next_unit(1, 1)
        l1.ffn_ops.append(FFNOp(
            op_type=FFNOpType.THRESHOLD_FLAG,
            unit_start=unit,
            unit_count=1,
            gate_dims=self._get_dims("L1H0"),  # Suppression
            input_dims=[self._get_dim("IS_BYTE")] + self._get_dims("L1H1"),
            output_dims=[self._get_dim("BYTE_INDEX_0")],
            params={'threshold': 1.5, 'negate_gate': True},
        ))

        # BYTE_INDEX_1: IS_BYTE AND any(L1H2[i]) AND NOT any(L1H1[i])
        unit = self._next_unit(1, 1)
        l1.ffn_ops.append(FFNOp(
            op_type=FFNOpType.THRESHOLD_FLAG,
            unit_start=unit,
            unit_count=1,
            gate_dims=self._get_dims("L1H1"),
            input_dims=[self._get_dim("IS_BYTE")] + self._get_dims("L1H2"),
            output_dims=[self._get_dim("BYTE_INDEX_1")],
            params={'threshold': 1.5, 'negate_gate': True},
        ))

        # BYTE_INDEX_2: IS_BYTE AND any(H0[i]) AND NOT any(L1H2[i])
        unit = self._next_unit(1, 1)
        l1.ffn_ops.append(FFNOp(
            op_type=FFNOpType.THRESHOLD_FLAG,
            unit_start=unit,
            unit_count=1,
            gate_dims=self._get_dims("L1H2"),
            input_dims=[self._get_dim("IS_BYTE")] + self._get_dims("H0"),
            output_dims=[self._get_dim("BYTE_INDEX_2")],
            params={'threshold': 1.5, 'negate_gate': True},
        ))

        # BYTE_INDEX_3: IS_BYTE AND any(H1[i]) AND NOT any(H0[i])
        unit = self._next_unit(1, 1)
        l1.ffn_ops.append(FFNOp(
            op_type=FFNOpType.THRESHOLD_FLAG,
            unit_start=unit,
            unit_count=1,
            gate_dims=self._get_dims("H0"),
            input_dims=[self._get_dim("IS_BYTE")] + self._get_dims("H1"),
            output_dims=[self._get_dim("BYTE_INDEX_3")],
            params={'threshold': 1.5, 'negate_gate': True},
        ))

        l1.reads.update(["IS_BYTE", "L1H0", "L1H1", "L1H2", "L1H4", "H0", "H1"])
        l1.writes.update(["STACK0_BYTE0", "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3"])

    def _build_l2_ffn_ops(self, l2: LayerSpec) -> None:
        """Build L2 FFN operations for MEM byte position flags."""
        MEM_I = 4  # MEM marker index
        BP_I = 3   # BP marker index

        # MEM_VAL_B0: H1[MEM]=1, H0[MEM]=0
        unit = self._next_unit(2, 1)
        l2.ffn_ops.append(FFNOp(
            op_type=FFNOpType.THRESHOLD_FLAG,
            unit_start=unit,
            unit_count=1,
            gate_dims=[self._get_dim("H0") + MEM_I],
            input_dims=[self._get_dim("H1") + MEM_I, self._get_dim("IS_BYTE")],
            output_dims=[self._get_dim("MEM_VAL_B0")],
            params={'threshold': 1.5, 'negate_gate': True},
        ))

        # MEM_VAL_B1: L2H0[MEM]=1, H1[MEM]=0
        unit = self._next_unit(2, 1)
        l2.ffn_ops.append(FFNOp(
            op_type=FFNOpType.THRESHOLD_FLAG,
            unit_start=unit,
            unit_count=1,
            gate_dims=[self._get_dim("H1") + MEM_I],
            input_dims=[self._get_dim("L2H0") + MEM_I, self._get_dim("IS_BYTE")],
            output_dims=[self._get_dim("MEM_VAL_B1")],
            params={'threshold': 1.5, 'negate_gate': True},
        ))

        # MEM_VAL_B2: L1H4[MEM]=1, L2H0[MEM]=0
        unit = self._next_unit(2, 1)
        l2.ffn_ops.append(FFNOp(
            op_type=FFNOpType.THRESHOLD_FLAG,
            unit_start=unit,
            unit_count=1,
            gate_dims=[self._get_dim("L2H0") + MEM_I],
            input_dims=[self._get_dim("L1H4") + MEM_I, self._get_dim("IS_BYTE")],
            output_dims=[self._get_dim("MEM_VAL_B2")],
            params={'threshold': 1.5, 'negate_gate': True},
        ))

        # MEM_VAL_B3: H2[MEM]=1, L1H4[MEM]=0
        unit = self._next_unit(2, 1)
        l2.ffn_ops.append(FFNOp(
            op_type=FFNOpType.THRESHOLD_FLAG,
            unit_start=unit,
            unit_count=1,
            gate_dims=[self._get_dim("L1H4") + MEM_I],
            input_dims=[self._get_dim("H2") + MEM_I, self._get_dim("IS_BYTE")],
            output_dims=[self._get_dim("MEM_VAL_B3")],
            params={'threshold': 1.5, 'negate_gate': True},
        ))

        # Extended BYTE_INDEX for STACK0 byte positions (d=6,7,8,9 from BP)
        # BYTE_INDEX_0 at STACK0: d=6 from BP → L1H4[BP]=1, H1[BP]=0
        unit = self._next_unit(2, 1)
        l2.ffn_ops.append(FFNOp(
            op_type=FFNOpType.THRESHOLD_FLAG,
            unit_start=unit,
            unit_count=1,
            gate_dims=[self._get_dim("H1") + BP_I],
            input_dims=[self._get_dim("L1H4") + BP_I, self._get_dim("IS_BYTE")],
            output_dims=[self._get_dim("BYTE_INDEX_0")],
            params={'threshold': 1.5, 'negate_gate': True},
        ))

        # BYTE_INDEX_1 at STACK0: d=7 from BP → H2[BP]=1, L1H4[BP]=0
        unit = self._next_unit(2, 1)
        l2.ffn_ops.append(FFNOp(
            op_type=FFNOpType.THRESHOLD_FLAG,
            unit_start=unit,
            unit_count=1,
            gate_dims=[self._get_dim("L1H4") + BP_I],
            input_dims=[self._get_dim("H2") + BP_I, self._get_dim("IS_BYTE")],
            output_dims=[self._get_dim("BYTE_INDEX_1")],
            params={'threshold': 1.5, 'negate_gate': True},
        ))

        # BYTE_INDEX_2 at STACK0: d=8 from BP → H3[BP]=1, H2[BP]=0
        unit = self._next_unit(2, 1)
        l2.ffn_ops.append(FFNOp(
            op_type=FFNOpType.THRESHOLD_FLAG,
            unit_start=unit,
            unit_count=1,
            gate_dims=[self._get_dim("H2") + BP_I],
            input_dims=[self._get_dim("H3") + BP_I, self._get_dim("IS_BYTE")],
            output_dims=[self._get_dim("BYTE_INDEX_2")],
            params={'threshold': 1.5, 'negate_gate': True},
        ))

        # BYTE_INDEX_3 at STACK0: d=9 from BP → H4[BP]=1, H3[BP]=0
        unit = self._next_unit(2, 1)
        l2.ffn_ops.append(FFNOp(
            op_type=FFNOpType.THRESHOLD_FLAG,
            unit_start=unit,
            unit_count=1,
            gate_dims=[self._get_dim("H3") + BP_I],
            input_dims=[self._get_dim("H4") + BP_I, self._get_dim("IS_BYTE")],
            output_dims=[self._get_dim("BYTE_INDEX_3")],
            params={'threshold': 1.5, 'negate_gate': True},
        ))

        l2.reads.update(["IS_BYTE", "H0", "H1", "H2", "H3", "H4", "L1H4", "L2H0"])
        l2.writes.update(["MEM_VAL_B0", "MEM_VAL_B1", "MEM_VAL_B2", "MEM_VAL_B3",
                          "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3"])

    def _build_l3_ffn_ops(self, l3: LayerSpec) -> None:
        """Build L3 FFN operations for defaults and PC increment.

        Operations:
        1. PC first-step default (PC = PC_OFFSET + INSTR_WIDTH = 10)
        2. SP/BP byte defaults (STACK_INIT = 0x10000)
        3. PC/AX upper byte defaults (bytes 1-3 = 0)
        4. MEM/STACK0 defaults
        5. PC increment for subsequent steps
        """
        from ..constants import PC_OFFSET, INSTR_WIDTH, STACK_INIT

        PC_I, AX_I, SP_I, BP_I, MEM_I = 0, 1, 2, 3, 4

        # === PC FIRST-STEP DEFAULT ===
        # When MARK_PC AND NOT HAS_SE, set PC=PC_OFFSET+INSTR_WIDTH=10 (0x0A)
        first_pc = PC_OFFSET + INSTR_WIDTH  # 10
        pc_lo = first_pc & 0xF  # 10
        pc_hi = (first_pc >> 4) & 0xF  # 0

        # Lo nibble unit
        unit = self._next_unit(3, 1)
        l3.ffn_ops.append(FFNOp(
            op_type=FFNOpType.CONDITIONAL_OUTPUT,
            unit_start=unit,
            unit_count=1,
            gate_dims=[],
            input_dims=[self._get_dim("MARK_PC")],
            output_dims=self._get_dims("OUTPUT_LO") + self._get_dims("OUTPUT_HI"),
            params={
                'threshold': 0.5,
                'nibble_lo': pc_lo,
                'also_write': [self._get_dim("EMBED_LO") + pc_lo],
            },
        ))

        # Lo undo when HAS_SE
        unit = self._next_unit(3, 1)
        l3.ffn_ops.append(FFNOp(
            op_type=FFNOpType.CONDITIONAL_OUTPUT,
            unit_start=unit,
            unit_count=1,
            gate_dims=[self._get_dim("MARK_PC")],  # Used as W_gate here
            input_dims=[self._get_dim("HAS_SE")],
            output_dims=self._get_dims("OUTPUT_LO") + self._get_dims("OUTPUT_HI"),
            params={
                'threshold': 0.5,
                'nibble_lo': pc_lo,
                'out_scale': -2.0,  # Negative to cancel
                'also_write': [self._get_dim("EMBED_LO") + pc_lo],
            },
        ))

        # Hi nibble unit
        unit = self._next_unit(3, 1)
        l3.ffn_ops.append(FFNOp(
            op_type=FFNOpType.CONDITIONAL_OUTPUT,
            unit_start=unit,
            unit_count=1,
            gate_dims=[],
            input_dims=[self._get_dim("MARK_PC")],
            output_dims=self._get_dims("OUTPUT_LO") + self._get_dims("OUTPUT_HI"),
            params={
                'threshold': 0.5,
                'nibble_hi': pc_hi,
                'also_write': [self._get_dim("EMBED_HI") + pc_hi],
            },
        ))

        # Hi undo when HAS_SE
        unit = self._next_unit(3, 1)
        l3.ffn_ops.append(FFNOp(
            op_type=FFNOpType.CONDITIONAL_OUTPUT,
            unit_start=unit,
            unit_count=1,
            gate_dims=[self._get_dim("MARK_PC")],
            input_dims=[self._get_dim("HAS_SE")],
            output_dims=self._get_dims("OUTPUT_LO") + self._get_dims("OUTPUT_HI"),
            params={
                'threshold': 0.5,
                'nibble_hi': pc_hi,
                'out_scale': -2.0,
                'also_write': [self._get_dim("EMBED_HI") + pc_hi],
            },
        ))

        # === SP/BP BYTE DEFAULTS (STACK_INIT = 0x10000) ===
        # Bytes: 0x00, 0x00, 0x01, 0x00
        # SP/BP bytes 0, 2 = 0 (byte 1 = 0x01 handled separately)
        for marker_i, marker_name in [(SP_I, "H1"), (BP_I, "H1")]:
            for byte_idx_dim in [self._get_dim("BYTE_INDEX_0"), self._get_dim("BYTE_INDEX_2")]:
                # Lo nibble = 0
                unit = self._next_unit(3, 1)
                l3.ffn_ops.append(FFNOp(
                    op_type=FFNOpType.CONDITIONAL_OUTPUT,
                    unit_start=unit,
                    unit_count=1,
                    gate_dims=[],
                    input_dims=[self._get_dim(marker_name) + marker_i, byte_idx_dim],
                    output_dims=self._get_dims("OUTPUT_LO") + self._get_dims("OUTPUT_HI"),
                    params={'threshold': 1.5, 'nibble_lo': 0},
                ))
                # Hi nibble = 0
                unit = self._next_unit(3, 1)
                l3.ffn_ops.append(FFNOp(
                    op_type=FFNOpType.CONDITIONAL_OUTPUT,
                    unit_start=unit,
                    unit_count=1,
                    gate_dims=[],
                    input_dims=[self._get_dim(marker_name) + marker_i, byte_idx_dim],
                    output_dims=self._get_dims("OUTPUT_LO") + self._get_dims("OUTPUT_HI"),
                    params={'threshold': 1.5, 'nibble_hi': 0},
                ))

        # SP/BP byte 2 = 0x01 (lo=1, hi=0) - FIRST STEP ONLY
        for marker_i in [SP_I, BP_I]:
            # Lo nibble = 1 (first step only)
            unit = self._next_unit(3, 1)
            l3.ffn_ops.append(FFNOp(
                op_type=FFNOpType.CONDITIONAL_OUTPUT,
                unit_start=unit,
                unit_count=1,
                gate_dims=[],
                input_dims=[self._get_dim("H1") + marker_i, self._get_dim("BYTE_INDEX_1")],
                output_dims=self._get_dims("OUTPUT_LO") + self._get_dims("OUTPUT_HI"),
                params={
                    'threshold': 1.5,
                    'nibble_lo': 1,
                    'neg_dims': [self._get_dim("HAS_SE")],
                },
            ))
            # Hi nibble = 0 (first step only)
            unit = self._next_unit(3, 1)
            l3.ffn_ops.append(FFNOp(
                op_type=FFNOpType.CONDITIONAL_OUTPUT,
                unit_start=unit,
                unit_count=1,
                gate_dims=[],
                input_dims=[self._get_dim("H1") + marker_i, self._get_dim("BYTE_INDEX_1")],
                output_dims=self._get_dims("OUTPUT_LO") + self._get_dims("OUTPUT_HI"),
                params={
                    'threshold': 1.5,
                    'nibble_hi': 0,
                    'neg_dims': [self._get_dim("HAS_SE")],
                },
            ))

        # === PC/AX UPPER BYTE DEFAULTS (bytes 1-3 = 0) ===
        for marker_i in [PC_I, AX_I]:
            for byte_idx_dim in [self._get_dim("BYTE_INDEX_0"),
                                 self._get_dim("BYTE_INDEX_1"),
                                 self._get_dim("BYTE_INDEX_2")]:
                # Lo = 0
                unit = self._next_unit(3, 1)
                l3.ffn_ops.append(FFNOp(
                    op_type=FFNOpType.CONDITIONAL_OUTPUT,
                    unit_start=unit,
                    unit_count=1,
                    gate_dims=[],
                    input_dims=[self._get_dim("H1") + marker_i, byte_idx_dim],
                    output_dims=self._get_dims("OUTPUT_LO") + self._get_dims("OUTPUT_HI"),
                    params={'threshold': 1.5, 'nibble_lo': 0},
                ))
                # Hi = 0
                unit = self._next_unit(3, 1)
                l3.ffn_ops.append(FFNOp(
                    op_type=FFNOpType.CONDITIONAL_OUTPUT,
                    unit_start=unit,
                    unit_count=1,
                    gate_dims=[],
                    input_dims=[self._get_dim("H1") + marker_i, byte_idx_dim],
                    output_dims=self._get_dims("OUTPUT_LO") + self._get_dims("OUTPUT_HI"),
                    params={'threshold': 1.5, 'nibble_hi': 0},
                ))

        # === MEM DEFAULT (all bytes = 0) ===
        # At MEM marker, default to 0
        unit = self._next_unit(3, 1)
        l3.ffn_ops.append(FFNOp(
            op_type=FFNOpType.CONDITIONAL_OUTPUT,
            unit_start=unit,
            unit_count=1,
            gate_dims=[],
            input_dims=[self._get_dim("MARK_MEM")],
            output_dims=self._get_dims("OUTPUT_LO") + self._get_dims("OUTPUT_HI"),
            params={'threshold': 0.5, 'nibble_lo': 0},
        ))
        unit = self._next_unit(3, 1)
        l3.ffn_ops.append(FFNOp(
            op_type=FFNOpType.CONDITIONAL_OUTPUT,
            unit_start=unit,
            unit_count=1,
            gate_dims=[],
            input_dims=[self._get_dim("MARK_MEM")],
            output_dims=self._get_dims("OUTPUT_LO") + self._get_dims("OUTPUT_HI"),
            params={'threshold': 0.5, 'nibble_hi': 0},
        ))

        # MEM addr bytes 1-3 default to 0
        for byte_idx_dim in [self._get_dim("BYTE_INDEX_0"),
                             self._get_dim("BYTE_INDEX_1"),
                             self._get_dim("BYTE_INDEX_2")]:
            unit = self._next_unit(3, 1)
            l3.ffn_ops.append(FFNOp(
                op_type=FFNOpType.CONDITIONAL_OUTPUT,
                unit_start=unit,
                unit_count=1,
                gate_dims=[],
                input_dims=[self._get_dim("H1") + MEM_I, byte_idx_dim],
                output_dims=self._get_dims("OUTPUT_LO") + self._get_dims("OUTPUT_HI"),
                params={'threshold': 1.5, 'nibble_lo': 0},
            ))
            unit = self._next_unit(3, 1)
            l3.ffn_ops.append(FFNOp(
                op_type=FFNOpType.CONDITIONAL_OUTPUT,
                unit_start=unit,
                unit_count=1,
                gate_dims=[],
                input_dims=[self._get_dim("H1") + MEM_I, byte_idx_dim],
                output_dims=self._get_dims("OUTPUT_LO") + self._get_dims("OUTPUT_HI"),
                params={'threshold': 1.5, 'nibble_hi': 0},
            ))

        # === STACK0 UPPER BYTE DEFAULTS ===
        for byte_idx_dim in [self._get_dim("BYTE_INDEX_0"),
                             self._get_dim("BYTE_INDEX_1"),
                             self._get_dim("BYTE_INDEX_2")]:
            unit = self._next_unit(3, 1)
            l3.ffn_ops.append(FFNOp(
                op_type=FFNOpType.CONDITIONAL_OUTPUT,
                unit_start=unit,
                unit_count=1,
                gate_dims=[],
                input_dims=[self._get_dim("H4") + BP_I, byte_idx_dim],
                output_dims=self._get_dims("OUTPUT_LO") + self._get_dims("OUTPUT_HI"),
                params={
                    'threshold': 1.5,
                    'nibble_lo': 0,
                    'neg_dims': [self._get_dim("H1") + BP_I],  # Exclude BP area
                },
            ))
            unit = self._next_unit(3, 1)
            l3.ffn_ops.append(FFNOp(
                op_type=FFNOpType.CONDITIONAL_OUTPUT,
                unit_start=unit,
                unit_count=1,
                gate_dims=[],
                input_dims=[self._get_dim("H4") + BP_I, byte_idx_dim],
                output_dims=self._get_dims("OUTPUT_LO") + self._get_dims("OUTPUT_HI"),
                params={
                    'threshold': 1.5,
                    'nibble_hi': 0,
                    'neg_dims': [self._get_dim("H1") + BP_I],
                },
            ))

        # STACK0 first-step default: byte 0 = 0
        unit = self._next_unit(3, 1)
        l3.ffn_ops.append(FFNOp(
            op_type=FFNOpType.CONDITIONAL_OUTPUT,
            unit_start=unit,
            unit_count=1,
            gate_dims=[],
            input_dims=[self._get_dim("MARK_STACK0")],
            output_dims=self._get_dims("OUTPUT_LO") + self._get_dims("OUTPUT_HI"),
            params={
                'threshold': 0.5,
                'nibble_lo': 0,
                'neg_dims': [self._get_dim("HAS_SE")],
            },
        ))
        unit = self._next_unit(3, 1)
        l3.ffn_ops.append(FFNOp(
            op_type=FFNOpType.CONDITIONAL_OUTPUT,
            unit_start=unit,
            unit_count=1,
            gate_dims=[],
            input_dims=[self._get_dim("MARK_STACK0")],
            output_dims=self._get_dims("OUTPUT_LO") + self._get_dims("OUTPUT_HI"),
            params={
                'threshold': 0.5,
                'nibble_hi': 0,
                'neg_dims': [self._get_dim("HAS_SE")],
            },
        ))

        # === PC INCREMENT ===
        # When MARK_PC AND HAS_SE AND NOT OP_LEV, add INSTR_WIDTH to carried-forward value
        unit = self._next_unit(3, 48)  # 16 + 16 + 16 for lo/hi/carry
        l3.ffn_ops.append(FFNOp(
            op_type=FFNOpType.PC_INCREMENT,
            unit_start=unit,
            unit_count=48,
            gate_dims=[],
            input_dims=self._get_dims("EMBED_LO") + self._get_dims("EMBED_HI"),
            output_dims=self._get_dims("OUTPUT_LO") + self._get_dims("OUTPUT_HI"),
            params={
                'instr_width': INSTR_WIDTH,
                'marker_dim': self._get_dim("MARK_PC"),
                'has_se_dim': self._get_dim("HAS_SE"),
                'op_lev_dim': self._get_dim("OP_LEV"),
            },
        ))

        l3.writes.update(["OUTPUT_LO", "OUTPUT_HI"])

    def _build_l4_ffn_ops(self, l4: LayerSpec) -> None:
        """Build L4 FFN operations for PC+1 computation.

        Operations:
        1. PC_PLUS1_LO: rotate EMBED_LO by +1 at MARK_AX → TEMP[0:16]
        2. PC_PLUS1_HI: default copy at MARK_AX → TEMP[16:32]
        3. PC_PLUS1_HI: carry correction when EMBED_LO[15]==1
        4. TEMP clearing at PC marker (prevent leakage)
        5. PC_PLUS1 at PC marker for L5 dynamic FETCH
        """
        # === PC_PLUS1_LO: rotate EMBED_LO by +1 at MARK_AX ===
        # Pattern: for each k, if MARK_AX and EMBED_LO[(k-1)%16], output to TEMP[k]
        unit = self._next_unit(4, 16)
        l4.ffn_ops.append(FFNOp(
            op_type=FFNOpType.GATED_NIBBLE_COPY,
            unit_start=unit,
            unit_count=16,
            gate_dims=[self._get_dim("MARK_AX")],
            input_dims=self._get_dims("EMBED_LO"),
            output_dims=self._get_dims("TEMP")[:16],
            params={
                'threshold': 0.5,  # Single condition
                'rotate': 1,  # Output at (k+1)%16
            },
        ))

        # === PC_PLUS1_HI: default copy (no carry) at MARK_AX ===
        unit = self._next_unit(4, 16)
        l4.ffn_ops.append(FFNOp(
            op_type=FFNOpType.GATED_NIBBLE_COPY,
            unit_start=unit,
            unit_count=16,
            gate_dims=[self._get_dim("MARK_AX")],
            input_dims=self._get_dims("EMBED_HI"),
            output_dims=self._get_dims("TEMP")[16:32],
            params={
                'threshold': 0.5,
                'rotate': 0,  # Direct copy
            },
        ))

        # === PC_PLUS1_HI: carry correction when EMBED_LO[15]==1 ===
        # Cancel default copy and write rotated (+1) hi nibble
        # Pattern: when MARK_AX AND EMBED_LO[15], cancel HI[k] and add HI[(k-1)%16]

        # Cancel: MARK_AX AND LO[15] with W_gate[HI[k]] = -1.0
        unit = self._next_unit(4, 16)
        l4.ffn_ops.append(FFNOp(
            op_type=FFNOpType.GATED_NIBBLE_COPY,
            unit_start=unit,
            unit_count=16,
            gate_dims=[self._get_dim("MARK_AX"), self._get_dim("EMBED_LO") + 15],
            input_dims=self._get_dims("EMBED_HI"),
            output_dims=self._get_dims("TEMP")[16:32],
            params={
                'threshold': 1.5,
                'gate_weight': -1.0,  # Negative to cancel
                'rotate': 0,
            },
        ))

        # Add rotated: MARK_AX AND LO[15] with W_gate[HI[(k-1)%16]] = +1.0
        unit = self._next_unit(4, 16)
        l4.ffn_ops.append(FFNOp(
            op_type=FFNOpType.GATED_NIBBLE_COPY,
            unit_start=unit,
            unit_count=16,
            gate_dims=[self._get_dim("MARK_AX"), self._get_dim("EMBED_LO") + 15],
            input_dims=self._get_dims("EMBED_HI"),
            output_dims=self._get_dims("TEMP")[16:32],
            params={
                'threshold': 1.5,
                'gate_weight': 1.0,  # Positive to add
                'rotate': 1,  # Rotated input
            },
        ))

        # === TEMP clearing at PC marker ===
        # Clear TEMP dims at PC marker to prevent leakage to L6
        # Skip TEMP[0] - used for IS_JSR flag
        for k in range(1, 32):
            unit = self._next_unit(4, 1)
            l4.ffn_ops.append(FFNOp(
                op_type=FFNOpType.CLEAR_DIMS,
                unit_start=unit,
                unit_count=1,
                gate_dims=[self._get_dim("MARK_PC")],
                input_dims=[self._get_dim("TEMP") + k],
                output_dims=[self._get_dim("TEMP") + k],
                params={'threshold': 0.5},
            ))

        # === PC_PLUS1 at PC marker for L5 dynamic FETCH ===
        # At PC marker, compute PC+1 for L5 head 3 attention query
        # PC_PLUS1 uses positions 471-502 (aliases AX_FULL)
        pc_plus1_lo_start = 471
        pc_plus1_hi_start = 487
        if True:  # Always enabled, removed try/except

            # PC_PLUS1_LO: rotate EMBED_LO by +1 at PC marker
            unit = self._next_unit(4, 16)
            l4.ffn_ops.append(FFNOp(
                op_type=FFNOpType.GATED_NIBBLE_COPY,
                unit_start=unit,
                unit_count=16,
                gate_dims=[self._get_dim("MARK_PC")],
                input_dims=self._get_dims("EMBED_LO"),
                output_dims=list(range(pc_plus1_lo_start, pc_plus1_lo_start + 16)),
                params={
                    'threshold': 0.5,
                    'rotate': 1,
                },
            ))

            # PC_PLUS1_HI: default copy at PC marker
            unit = self._next_unit(4, 16)
            l4.ffn_ops.append(FFNOp(
                op_type=FFNOpType.GATED_NIBBLE_COPY,
                unit_start=unit,
                unit_count=16,
                gate_dims=[self._get_dim("MARK_PC")],
                input_dims=self._get_dims("EMBED_HI"),
                output_dims=list(range(pc_plus1_hi_start, pc_plus1_hi_start + 16)),
                params={
                    'threshold': 0.5,
                    'rotate': 0,
                },
            ))

            # PC_PLUS1_HI: carry correction at PC marker
            # Cancel: MARK_PC AND LO[15] with W_gate[HI[k]] = -1.0
            unit = self._next_unit(4, 16)
            l4.ffn_ops.append(FFNOp(
                op_type=FFNOpType.GATED_NIBBLE_COPY,
                unit_start=unit,
                unit_count=16,
                gate_dims=[self._get_dim("MARK_PC"), self._get_dim("EMBED_LO") + 15],
                input_dims=self._get_dims("EMBED_HI"),
                output_dims=list(range(pc_plus1_hi_start, pc_plus1_hi_start + 16)),
                params={
                    'threshold': 1.5,
                    'gate_weight': -1.0,
                    'rotate': 0,
                },
            ))

            # Add rotated: MARK_PC AND LO[15] with W_gate[HI[(k-1)%16]] = +1.0
            unit = self._next_unit(4, 16)
            l4.ffn_ops.append(FFNOp(
                op_type=FFNOpType.GATED_NIBBLE_COPY,
                unit_start=unit,
                unit_count=16,
                gate_dims=[self._get_dim("MARK_PC"), self._get_dim("EMBED_LO") + 15],
                input_dims=self._get_dims("EMBED_HI"),
                output_dims=list(range(pc_plus1_hi_start, pc_plus1_hi_start + 16)),
                params={
                    'threshold': 1.5,
                    'gate_weight': 1.0,
                    'rotate': 1,
                },
            ))

        l4.writes.update(["TEMP"])
        l4.reads.update(["MARK_AX", "MARK_PC", "EMBED_LO", "EMBED_HI"])

    def _build_carry_forward_layer(self) -> None:
        """Build L3 carry-forward layer.

        Copies register values (PC, AX, SP, BP, STACK0) to next step positions.
        Each head attends to previous step's byte 0 using L1H1/L1H0 pattern.
        """
        l3 = self.ir.get_layer(3)
        L = 15.0
        PC_I, AX_I, SP_I, BP_I = 0, 1, 2, 3

        # Head 0: PC carry (prev step PC byte 0 → EMBED at PC marker)
        l3.attention_ops.append(AttentionOp(
            op_type=AttentionOpType.CARRY_FORWARD,
            head_idx=0,
            q_dims=[self._get_dim("MARK_PC")],
            k_dims=[self._get_dim("L1H1") + PC_I, self._get_dim("L1H0") + PC_I],
            v_dims=self._get_dims("EMBED_LO") + self._get_dims("EMBED_HI"),
            o_dims=self._get_dims("EMBED_LO") + self._get_dims("EMBED_HI"),
            params={'scale': L, 'marker_idx': PC_I, 'l1h1_idx': PC_I, 'l1h0_idx': PC_I},
        ))

        # Head 1: AX carry (prev step AX byte 0 EMBED → AX_CARRY staging)
        l3.attention_ops.append(AttentionOp(
            op_type=AttentionOpType.CARRY_FORWARD,
            head_idx=1,
            q_dims=[self._get_dim("MARK_AX")],
            k_dims=[self._get_dim("L1H1") + AX_I, self._get_dim("L1H0") + AX_I],
            v_dims=self._get_dims("EMBED_LO") + self._get_dims("EMBED_HI"),
            o_dims=self._get_dims("AX_CARRY_LO") + self._get_dims("AX_CARRY_HI"),
            params={'scale': L, 'marker_idx': AX_I, 'l1h1_idx': AX_I, 'l1h0_idx': AX_I},
        ))

        # Head 2: SP carry (prev step SP byte 0 → EMBED at SP marker)
        l3.attention_ops.append(AttentionOp(
            op_type=AttentionOpType.CARRY_FORWARD,
            head_idx=2,
            q_dims=[self._get_dim("MARK_SP")],
            k_dims=[self._get_dim("L1H1") + SP_I, self._get_dim("L1H0") + SP_I],
            v_dims=self._get_dims("EMBED_LO") + self._get_dims("EMBED_HI"),
            o_dims=self._get_dims("EMBED_LO") + self._get_dims("EMBED_HI"),
            params={'scale': L, 'marker_idx': SP_I, 'l1h1_idx': SP_I, 'l1h0_idx': SP_I},
        ))

        # Head 3: BP carry (prev step BP byte 0 → EMBED at BP marker)
        l3.attention_ops.append(AttentionOp(
            op_type=AttentionOpType.CARRY_FORWARD,
            head_idx=3,
            q_dims=[self._get_dim("MARK_BP")],
            k_dims=[self._get_dim("L1H1") + BP_I, self._get_dim("L1H0") + BP_I],
            v_dims=self._get_dims("EMBED_LO") + self._get_dims("EMBED_HI"),
            o_dims=self._get_dims("EMBED_LO") + self._get_dims("EMBED_HI"),
            params={'scale': L, 'marker_idx': BP_I, 'l1h1_idx': BP_I, 'l1h0_idx': BP_I},
        ))

        # Head 4: STACK0 carry (uses STACK0_BYTE0 flag as key instead of L1H1/L1H0)
        # This uses a different pattern - key is STACK0_BYTE0 flag
        l3.attention_ops.append(AttentionOp(
            op_type=AttentionOpType.RELAY,  # Different pattern from carry-forward
            head_idx=4,
            q_dims=[self._get_dim("MARK_STACK0")],
            k_dims=[self._get_dim("STACK0_BYTE0")],
            v_dims=self._get_dims("EMBED_LO") + self._get_dims("EMBED_HI"),
            o_dims=self._get_dims("EMBED_LO") + self._get_dims("EMBED_HI"),
            params={'scale': L},
        ))

        # Head 5: AX full value relay (prev AX marker OUTPUT → current AX marker AX_FULL)
        # Q fires at AX marker with HAS_SE condition (subsequent steps only)
        # AX_FULL aliases with TEMP region (471-502)
        AX_FULL_LO = 471
        AX_FULL_HI = 487
        l3.attention_ops.append(AttentionOp(
            op_type=AttentionOpType.RELAY,
            head_idx=5,
            q_dims=[self._get_dim("MARK_AX"), self._get_dim("HAS_SE")],
            k_dims=[self._get_dim("MARK_AX")],
            v_dims=self._get_dims("OUTPUT_LO") + self._get_dims("OUTPUT_HI"),
            o_dims=list(range(AX_FULL_LO, AX_FULL_LO + 16)) + list(range(AX_FULL_HI, AX_FULL_HI + 16)),
            params={'scale': L, 'q_threshold': -L * 1.5},
        ))

        # Head 6: BP carry to PC marker for LEV return_addr lookup
        # Q fires at PC marker when OP_LEV active
        # NOTE: This head does NOT write to W_o (fixed to avoid corrupting output)
        l3.attention_ops.append(AttentionOp(
            op_type=AttentionOpType.CARRY_FORWARD,
            head_idx=6,
            q_dims=[self._get_dim("MARK_PC")],  # Extra dims handled in params
            k_dims=[self._get_dim("L1H1") + BP_I, self._get_dim("L1H0") + BP_I],
            v_dims=self._get_dims("CLEAN_EMBED_LO") + self._get_dims("CLEAN_EMBED_HI"),
            o_dims=[],  # No output writes - fixed in baseline
            params={'scale': L, 'marker_idx': BP_I, 'l1h1_idx': BP_I, 'l1h0_idx': BP_I,
                    'extra_q_dim': self._get_dim("OP_LEV"), 'extra_q_scale': L / 5,
                    'q_threshold': -L * 1.5, 'with_output': False},
        ))

        l3.reads.update(["MARK_PC", "MARK_AX", "MARK_SP", "MARK_BP", "MARK_STACK0",
                         "L1H0", "L1H1", "STACK0_BYTE0", "HAS_SE", "OP_LEV",
                         "EMBED_LO", "EMBED_HI", "OUTPUT_LO", "OUTPUT_HI",
                         "CLEAN_EMBED_LO", "CLEAN_EMBED_HI"])
        l3.writes.update(["EMBED_LO", "EMBED_HI", "AX_CARRY_LO", "AX_CARRY_HI",
                          "CLEAN_EMBED_LO", "CLEAN_EMBED_HI"])

        # L3 FFN: Defaults and PC increment
        self._build_l3_ffn_ops(l3)

    def _build_fetch_layers(self) -> None:
        """Build L4-L5 instruction fetch layers.

        L4: PC relay + PC+1 computation
        L5: Opcode fetch + decode (8 heads)
        """
        from ..constants import PC_OFFSET, INSTR_WIDTH

        # L4: PC relay to AX position for fetch
        l4 = self.ir.get_layer(4)

        l4.attention_ops.append(AttentionOp(
            op_type=AttentionOpType.RELAY,
            head_idx=0,
            q_dims=[self._get_dim("MARK_AX")],
            k_dims=[self._get_dim("MARK_PC")],
            v_dims=self._get_dims("EMBED_LO") + self._get_dims("EMBED_HI"),
            o_dims=self._get_dims("EMBED_LO") + self._get_dims("EMBED_HI"),
            params={'scale': 15.0},
        ))

        l4.reads.update(["MARK_AX", "MARK_PC", "EMBED_LO", "EMBED_HI"])
        l4.writes.update(["EMBED_LO", "EMBED_HI"])

        # L4 FFN: PC+1 computation and TEMP clearing
        self._build_l4_ffn_ops(l4)

        # L5: Instruction fetch (8 heads)
        l5 = self.ir.get_layer(5)
        L = 15.0  # Query/Key scale

        # === Head 0: Immediate fetch at PC+1 (non-first steps) ===
        # Q: TEMP[0:32] (PC+1) at AX marker, gated by HAS_SE
        l5.attention_ops.append(AttentionOp(
            op_type=AttentionOpType.FETCH,
            head_idx=0,
            q_dims=self._get_dims("TEMP")[:32] + [self._get_dim("MARK_AX")],
            k_dims=self._get_dims("ADDR_KEY")[:33],
            v_dims=self._get_dims("CLEAN_EMBED_LO") + self._get_dims("CLEAN_EMBED_HI"),
            o_dims=self._get_dims("FETCH_LO") + self._get_dims("FETCH_HI"),
            params={
                'scale': L,
                'has_se_gate': True,  # Only non-first steps
                'anti_leak_marker': 'MARK_AX',
            },
        ))

        # === Head 1: Opcode fetch at PC (non-first steps) ===
        # Q: EMBED_LO/HI (PC value relayed by L4) at AX marker, gated by HAS_SE
        l5.attention_ops.append(AttentionOp(
            op_type=AttentionOpType.FETCH,
            head_idx=1,
            q_dims=self._get_dims("EMBED_LO") + self._get_dims("EMBED_HI")[:16] + [self._get_dim("MARK_AX")],
            k_dims=self._get_dims("ADDR_KEY")[:33],
            v_dims=self._get_dims("CLEAN_EMBED_LO") + self._get_dims("CLEAN_EMBED_HI"),
            o_dims=self._get_dims("OPCODE_BYTE_LO")[:16] + self._get_dims("OPCODE_BYTE_HI")[:16],
            params={
                'scale': L,
                'has_se_gate': True,
                'anti_leak_marker': 'MARK_AX',
            },
        ))

        # === Head 2: First-step opcode fetch to PC marker ===
        # Q: Fixed address PC_OFFSET at PC marker, NOT HAS_SE
        l5.attention_ops.append(AttentionOp(
            op_type=AttentionOpType.FETCH,
            head_idx=2,
            q_dims=[self._get_dim("MARK_PC")],
            k_dims=self._get_dims("ADDR_KEY")[:33],
            v_dims=self._get_dims("CLEAN_EMBED_LO") + self._get_dims("CLEAN_EMBED_HI"),
            o_dims=self._get_dims("OPCODE_BYTE_LO")[:16] + self._get_dims("OPCODE_BYTE_HI")[:16],
            params={
                'scale': L,
                'fixed_addr': PC_OFFSET,
                'not_has_se': True,  # Only first step
                'anti_leak_marker': 'MARK_PC',
            },
        ))

        # === Head 3: Fetch immediate at dynamic PC+1 (all steps) ===
        # Q: PC_PLUS1_LO/HI at PC marker (for JMP support at any step)
        pc_plus1_lo = 471
        pc_plus1_hi = 487
        l5.attention_ops.append(AttentionOp(
            op_type=AttentionOpType.FETCH,
            head_idx=3,
            q_dims=list(range(pc_plus1_lo, pc_plus1_lo + 16)) + list(range(pc_plus1_hi, pc_plus1_hi + 16)) + [self._get_dim("MARK_PC")],
            k_dims=self._get_dims("ADDR_KEY")[:33],
            v_dims=self._get_dims("CLEAN_EMBED_LO") + self._get_dims("CLEAN_EMBED_HI"),
            o_dims=self._get_dims("FETCH_LO") + self._get_dims("FETCH_HI"),
            params={
                'scale': L,
                'anti_leak_marker': 'MARK_PC',
                'out_scale': 40.0,  # Amplified for relay to AX
            },
        ))

        # === Head 4: First-step opcode fetch to AX marker ===
        # Q: Fixed address PC_OFFSET at AX marker, NOT HAS_SE
        l5.attention_ops.append(AttentionOp(
            op_type=AttentionOpType.FETCH,
            head_idx=4,
            q_dims=[self._get_dim("MARK_AX")],
            k_dims=self._get_dims("ADDR_KEY")[:33],
            v_dims=self._get_dims("CLEAN_EMBED_LO") + self._get_dims("CLEAN_EMBED_HI"),
            o_dims=self._get_dims("OPCODE_BYTE_LO")[:16] + self._get_dims("OPCODE_BYTE_HI")[:16],
            params={
                'scale': L,
                'fixed_addr': PC_OFFSET,
                'not_has_se': True,  # Only first step
                'anti_leak_marker': 'MARK_AX',
            },
        ))

        # === Head 5: unused ===

        # === Head 6: OP_* flag relay from CODE to PC marker (non-first steps) ===
        op_flags = [
            "OP_IMM", "OP_LEA", "OP_EXIT", "OP_JMP", "OP_JSR",
            "OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV", "OP_MOD",
            "OP_OR", "OP_XOR", "OP_AND", "OP_EQ", "OP_LT", "OP_SHL", "OP_SHR"
        ]
        v_dims = [self._get_dim(f) for f in op_flags if f in self.allocator.allocations]
        o_dims = v_dims  # Write to same dimensions
        l5.attention_ops.append(AttentionOp(
            op_type=AttentionOpType.OP_FLAG_RELAY,
            head_idx=6,
            q_dims=self._get_dims("EMBED_LO") + self._get_dims("EMBED_HI")[:16] + [self._get_dim("MARK_PC")],
            k_dims=self._get_dims("ADDR_KEY")[:33],
            v_dims=v_dims,
            o_dims=o_dims,
            params={
                'scale': L,
                'has_se_gate': True,  # Only non-first steps
                'anti_leak_marker': 'MARK_PC',
            },
        ))

        # === Head 7: OP_* flag relay from CODE to PC marker (first step) ===
        l5.attention_ops.append(AttentionOp(
            op_type=AttentionOpType.OP_FLAG_RELAY,
            head_idx=7,
            q_dims=[self._get_dim("MARK_PC")],
            k_dims=self._get_dims("ADDR_KEY")[:33],
            v_dims=v_dims,
            o_dims=o_dims,
            params={
                'scale': L,
                'fixed_addr': PC_OFFSET,
                'not_has_se': True,  # Only first step
                'anti_leak_marker': 'MARK_PC',
            },
        ))

        l5.reads.update(["TEMP", "ADDR_KEY", "EMBED_LO", "EMBED_HI", "MARK_AX", "MARK_PC",
                         "HAS_SE", "CLEAN_EMBED_LO", "CLEAN_EMBED_HI", "CONST"])
        l5.writes.update(["FETCH_LO", "FETCH_HI", "OPCODE_BYTE_LO", "OPCODE_BYTE_HI"] + op_flags)

        # L5 FFN: Opcode decode
        self._build_l5_ffn_ops(l5)

    def _build_l5_ffn_ops(self, l5: LayerSpec) -> None:
        """Build L5 FFN operations for opcode decode.

        Operations:
        1. Standard opcode decode at AX marker (34 opcodes)
        2. First-step opcode decode at PC marker (key opcodes for routing)
        3. TEMP clearing at PC marker (skip TEMP[0] for IS_JSR flag)
        """
        # === Standard opcode decode at AX marker ===
        # Full opcode list with their (lo_nibble, hi_nibble) values
        opcodes = [
            ("OP_LEA", 0, 0), ("OP_IMM", 1, 0), ("OP_JMP", 2, 0), ("OP_JSR", 3, 0),
            ("OP_BZ", 4, 0), ("OP_BNZ", 5, 0), ("OP_ENT", 6, 0), ("OP_ADJ", 7, 0),
            ("OP_LEV", 8, 0), ("OP_LI", 9, 0), ("OP_LC", 10, 0), ("OP_SI", 11, 0),
            ("OP_SC", 12, 0), ("OP_PSH", 13, 0), ("OP_OR", 14, 0), ("OP_XOR", 15, 0),
            ("OP_AND", 0, 1), ("OP_EQ", 1, 1), ("OP_NE", 2, 1), ("OP_LT", 3, 1),
            ("OP_GT", 4, 1), ("OP_LE", 5, 1), ("OP_GE", 6, 1), ("OP_SHL", 7, 1),
            ("OP_SHR", 8, 1), ("OP_ADD", 9, 1), ("OP_SUB", 10, 1), ("OP_MUL", 11, 1),
            ("OP_DIV", 12, 1), ("OP_MOD", 13, 1),
            ("OP_EXIT", 6, 2), ("OP_NOP", 7, 2),
            ("OP_PUTCHAR", 1, 4), ("OP_GETCHAR", 0, 4),
        ]

        for dim_name, lo_nibble, hi_nibble in opcodes:
            try:
                out_dim = self._get_dim(dim_name)
            except KeyError:
                continue  # Skip if dimension not allocated

            unit = self._next_unit(5, 1)
            # Use OPCODE_BYTE dimensions (8 each for lo/hi nibbles)
            l5.ffn_ops.append(FFNOp(
                op_type=FFNOpType.OPCODE_DECODE,
                unit_start=unit,
                unit_count=1,
                gate_dims=[self._get_dim("MARK_AX")],
                input_dims=self._get_dims("OPCODE_BYTE_LO") + self._get_dims("OPCODE_BYTE_HI"),
                output_dims=[out_dim],
                params={
                    'lo_nibble': lo_nibble,
                    'hi_nibble': hi_nibble,
                    'out_scale': 10.0,  # Match baseline W_down scale
                },
            ))

        # === First-step opcode decode at PC marker ===
        # These need to be decoded at PC marker for routing to work
        first_step_opcodes = [
            ("OP_JMP", 2, 0), ("OP_JSR", 3, 0), ("OP_IMM", 1, 0), ("OP_LEA", 0, 0),
            ("OP_EXIT", 6, 2), ("OP_NOP", 7, 2),
            ("OP_ADD", 9, 1), ("OP_SUB", 10, 1), ("OP_MUL", 11, 1), ("OP_DIV", 12, 1),
            ("OP_MOD", 13, 1), ("OP_OR", 14, 0), ("OP_XOR", 15, 0), ("OP_AND", 0, 1),
            ("OP_EQ", 1, 1), ("OP_LT", 3, 1), ("OP_SHL", 7, 1), ("OP_SHR", 8, 1),
        ]

        for dim_name, lo_nibble, hi_nibble in first_step_opcodes:
            try:
                out_dim = self._get_dim(dim_name)
            except KeyError:
                continue

            unit = self._next_unit(5, 1)
            # JSR writes to TEMP[0] instead of OP_JSR
            if dim_name == "OP_JSR":
                output_dims = [self._get_dim("TEMP")]  # TEMP[0]
            else:
                output_dims = [out_dim]

            l5.ffn_ops.append(FFNOp(
                op_type=FFNOpType.OPCODE_DECODE,
                unit_start=unit,
                unit_count=1,
                gate_dims=[],  # Uses W_up gating pattern
                input_dims=self._get_dims("OPCODE_BYTE_LO") + self._get_dims("OPCODE_BYTE_HI"),
                output_dims=output_dims,
                params={
                    'lo_nibble': lo_nibble,
                    'hi_nibble': hi_nibble,
                    'pc_marker_gate': True,  # Include MARK_PC in W_up
                    'not_has_se': True,  # Include -HAS_SE in W_up
                    'out_scale': 10.0,
                },
            ))

        # === TEMP clearing at PC marker ===
        # Clear TEMP dims at PC marker to prevent leakage
        # Skip TEMP[0] - used for IS_JSR flag
        for k in range(1, 32):
            unit = self._next_unit(5, 1)
            l5.ffn_ops.append(FFNOp(
                op_type=FFNOpType.CLEAR_DIMS,
                unit_start=unit,
                unit_count=1,
                gate_dims=[self._get_dim("MARK_PC")],
                input_dims=[self._get_dim("TEMP") + k],
                output_dims=[self._get_dim("TEMP") + k],
                params={'threshold': 0.5},
            ))

        l5.writes.update([op[0] for op in opcodes if op[0] in self.allocator.allocations])

    def _build_alu_layers(self, opcodes: List[OpcodeSpec]) -> None:
        """Build L6-L12 ALU layers.

        L6: JMP/EXIT/JSR relay + OP flag relay + FETCH relay
        L7: Operand gather from stack
        L8-L9: ALU computation
        L10: Carry propagation
        L11-L12: Multi-byte operations (MUL/DIV)
        """
        self._build_l6_attention(opcodes)

    def _build_l6_attention(self, opcodes: List[OpcodeSpec]) -> None:
        """Build L6 attention for control flow relays.

        Head 0: JMP relay (PC marker ← prev step's AX marker)
        Head 1: EXIT relay (NEXT_SE ← current step's AX marker)
        Head 2: First-step JMP relay (PC marker self-attention)
        Head 3: JSR relay (PC marker ← AX marker, first step only)
        Head 4: Reserved for BZ/BNZ relay
        Head 5: First-step OP flag + FETCH relay (AX marker ← PC marker)
        """
        l6 = self.ir.get_layer(6)
        L = 50.0  # Large scale for strong gating

        # === Head 0: JMP relay (PC marker ← prev step's AX marker) ===
        # Fires at step 1+ (HAS_SE > 0), copies OP_JMP and FETCH to AX_CARRY
        l6.attention_ops.append(AttentionOp(
            op_type=AttentionOpType.RELAY,
            head_idx=0,
            q_dims=[self._get_dim("MARK_PC")],
            k_dims=[self._get_dim("MARK_AX")],
            v_dims=[self._get_dim("OP_JMP")] + self._get_dims("FETCH_LO") + self._get_dims("FETCH_HI"),
            o_dims=[self._get_dim("CMP")] + self._get_dims("AX_CARRY_LO") + self._get_dims("AX_CARRY_HI"),
            params={
                'scale': L,
                'has_se_gate': True,  # Only step 1+
                'block_at_ax': True,  # Block at AX marker
                'strong_gate': True,  # Use ±1000 gating
            },
        ))

        # === Head 1: EXIT relay (NEXT_SE ← current step's AX marker) ===
        # Fires at NEXT_SE positions, copies OP_EXIT to CMP[1]
        l6.attention_ops.append(AttentionOp(
            op_type=AttentionOpType.RELAY,
            head_idx=1,
            q_dims=[self._get_dim("NEXT_SE")],
            k_dims=[self._get_dim("MARK_AX")],
            v_dims=[self._get_dim("OP_EXIT")],
            o_dims=[self._get_dim("CMP") + 1],  # CMP[1] = IS_EXIT
            params={
                'scale': L,
                'block_at_ax': True,
                'v_scale': 0.2,  # Scaled for EXIT/NOP separation
            },
        ))

        # === Head 2: First-step JMP relay (PC marker self-attention) ===
        # Fires at PC marker when NOT HAS_SE AND OP_JMP active
        l6.attention_ops.append(AttentionOp(
            op_type=AttentionOpType.RELAY,
            head_idx=2,
            q_dims=[self._get_dim("MARK_PC")],
            k_dims=[self._get_dim("MARK_PC")],
            v_dims=[self._get_dim("OP_JMP")] + self._get_dims("FETCH_LO") + self._get_dims("FETCH_HI"),
            o_dims=[self._get_dim("CMP")] + self._get_dims("AX_CARRY_LO") + self._get_dims("AX_CARRY_HI"),
            params={
                'scale': L,
                'not_has_se': True,  # Only first step
                'block_at_ax': True,
                'op_jmp_gate': True,  # Only fire when OP_JMP active
                'strong_gate': True,
            },
        ))

        # === Head 3: JSR relay (PC marker ← AX marker, first step only) ===
        # Copies OP_JSR flag to TEMP[0] for L6 FFN
        l6.attention_ops.append(AttentionOp(
            op_type=AttentionOpType.RELAY,
            head_idx=3,
            q_dims=[self._get_dim("MARK_PC")],
            k_dims=[self._get_dim("MARK_AX")],
            v_dims=[self._get_dim("OP_JSR")],
            o_dims=[self._get_dim("TEMP")],  # TEMP[0]
            params={
                'scale': L,
                'not_has_se': True,  # Only first step
                'block_at_ax': True,
            },
        ))

        # === Head 5: First-step OP flag + FETCH relay (AX marker ← PC marker) ===
        # Copies OP flags and FETCH from PC marker to AX marker on first step
        op_flags = [
            "OP_IMM", "OP_LEA", "OP_JMP", "OP_JSR", "OP_EXIT", "OP_NOP",
            "OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV", "OP_MOD",
            "OP_OR", "OP_XOR", "OP_AND", "OP_EQ", "OP_LT", "OP_SHL", "OP_SHR"
        ]
        v_dims = [self._get_dim(f) for f in op_flags if f in self.allocator.allocations]
        v_dims += self._get_dims("FETCH_LO") + self._get_dims("FETCH_HI")
        o_dims = [self._get_dim(f) for f in op_flags if f in self.allocator.allocations]
        o_dims += self._get_dims("FETCH_LO") + self._get_dims("FETCH_HI")

        l6.attention_ops.append(AttentionOp(
            op_type=AttentionOpType.RELAY,
            head_idx=5,
            q_dims=[self._get_dim("MARK_AX")],
            k_dims=[self._get_dim("MARK_PC")],
            v_dims=v_dims,
            o_dims=o_dims,
            params={
                'scale': L,
                'not_has_se': True,  # Only first step
                'has_se_block': True,  # Block when HAS_SE = 1
            },
        ))

        l6.reads.update(["MARK_PC", "MARK_AX", "NEXT_SE", "HAS_SE", "CONST",
                         "OP_JMP", "OP_JSR", "OP_EXIT", "FETCH_LO", "FETCH_HI"] + op_flags)
        l6.writes.update(["CMP", "AX_CARRY_LO", "AX_CARRY_HI", "TEMP",
                          "FETCH_LO", "FETCH_HI"] + op_flags)

        # L7: Gather operand from stack
        l7 = self.ir.get_layer(7)

        l7.attention_ops.append(AttentionOp(
            op_type=AttentionOpType.RELAY,
            head_idx=0,
            q_dims=[self._get_dim("MARK_AX")],
            k_dims=[self._get_dim("MARK_STACK0")],
            v_dims=self._get_dims("EMBED_LO") + self._get_dims("EMBED_HI"),
            o_dims=self._get_dims("ALU_LO") + self._get_dims("ALU_HI"),
            params={'scale': 15.0},
        ))

        l7.reads.update(["MARK_AX", "MARK_STACK0", "EMBED_LO", "EMBED_HI"])
        l7.writes.update(["ALU_LO", "ALU_HI"])

        # L9: ALU computation
        l9 = self.ir.get_layer(9)
        self._build_alu_operations(l9, opcodes)

        # L10: Carry propagation
        l10 = self.ir.get_layer(10)
        l10.reads.update(["ALU_LO", "ALU_HI", "CARRY"])
        l10.writes.update(["OUTPUT_LO", "OUTPUT_HI"])

    def _build_alu_operations(self, l9: LayerSpec, opcodes: List[OpcodeSpec]) -> None:
        """Build ALU lookup tables for arithmetic/bitwise ops."""
        alu_opcodes = [op for op in opcodes if op.alu_op is not None]

        for opcode in alu_opcodes:
            gate_dim_name = f"OP_{opcode.name}"
            try:
                gate_dim = self._get_dim(gate_dim_name)
            except KeyError:
                continue

            unit = self._next_unit(9, 256)  # 16x16 lookup table
            l9.ffn_ops.append(FFNOp(
                op_type=FFNOpType.ALU_LOOKUP,
                unit_start=unit,
                unit_count=256,
                gate_dims=[gate_dim],
                input_dims=self._get_dims("ALU_LO") + self._get_dims("ALU_HI"),
                output_dims=self._get_dims("OUTPUT_LO"),
                params={
                    'alu_fn': opcode.alu_op,
                    'scale': 100.0,
                },
            ))

        l9.reads.update(["ALU_LO", "ALU_HI"] + [f"OP_{op.name}" for op in alu_opcodes])
        l9.writes.add("OUTPUT_LO")

    def _build_memory_layers(self) -> None:
        """Build L13-L15 memory layers.

        L13: Address computation
        L14: Memory fetch
        L15: Memory writeback
        """
        l13 = self.ir.get_layer(13)
        l13.reads.update(["MARK_AX", "EMBED_LO", "EMBED_HI"])

        l14 = self.ir.get_layer(14)
        l14.reads.update(["MARK_MEM", "ADDR_KEY"])

        l15 = self.ir.get_layer(15)
        l15.reads.update(["OUTPUT_LO", "OUTPUT_HI"])

    def _build_writeback_layer(self) -> None:
        """Build L16 writeback layer.

        Final state update for next step.
        """
        l16 = self.ir.get_layer(16)
        l16.reads.update(["OUTPUT_LO", "OUTPUT_HI", "MARK_PC", "MARK_AX"])


def build_standard_ir() -> CompilerIR:
    """Build IR with standard configuration and all opcodes."""
    builder = IRBuilder()
    return builder.build()
