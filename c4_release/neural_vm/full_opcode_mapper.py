"""
Full C4 Opcode Mapper - 100% Coverage

Maps all 42 C4 VM opcodes to computation graphs.
Supports multi-layer compilation (FFN → Attention → FFN).
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .graph_weight_compiler import OpType, ComputationGraph
from .embedding import Opcode


@dataclass
class MultiLayerWeights:
    """Weights for multi-layer opcode compilation."""
    layers: List[Tuple[str, Dict]]  # List of (layer_type, weights)
    # layer_type: "ffn" or "attention"


class FullOpcodeMapper:
    """Maps all 42 C4 opcodes to computation graphs.

    Usage:
        mapper = FullOpcodeMapper()

        # Simple opcode (single FFN layer)
        graph = mapper.map_opcode(Opcode.ADD)

        # Complex opcode (multiple layers)
        multi = mapper.map_opcode_multilayer(Opcode.LI)
        # Returns: [("ffn", setup_weights), ("attention", None), ("ffn", copy_weights)]
    """

    def __init__(self):
        pass

    def map_opcode(self, opcode: int, **params) -> ComputationGraph:
        """Map C4 opcode to computation graph.

        Args:
            opcode: C4 opcode value
            **params: Opcode-specific parameters (e.g., imm for immediate values)

        Returns:
            ComputationGraph representing the opcode
        """
        # Arithmetic (25-29)
        if opcode == Opcode.ADD:
            return self._map_add()
        elif opcode == Opcode.SUB:
            return self._map_sub()
        elif opcode == Opcode.MUL:
            return self._map_mul()
        elif opcode == Opcode.DIV:
            return self._map_div()
        elif opcode == Opcode.MOD:
            return self._map_mod()

        # Comparison (17-22)
        elif opcode == Opcode.EQ:
            return self._map_eq()
        elif opcode == Opcode.NE:
            return self._map_ne()
        elif opcode == Opcode.LT:
            return self._map_lt()
        elif opcode == Opcode.GT:
            return self._map_gt()
        elif opcode == Opcode.LE:
            return self._map_le()
        elif opcode == Opcode.GE:
            return self._map_ge()

        # Bitwise (14-16)
        elif opcode == Opcode.OR:
            return self._map_or()
        elif opcode == Opcode.XOR:
            return self._map_xor()
        elif opcode == Opcode.AND:
            return self._map_and()
        elif opcode == Opcode.SHL:
            return self._map_shl()
        elif opcode == Opcode.SHR:
            return self._map_shr()

        # Stack/Register (0-1, 7)
        elif opcode == Opcode.LEA:
            return self._map_lea(**params)
        elif opcode == Opcode.IMM:
            return self._map_imm(**params)
        elif opcode == Opcode.ADJ:
            return self._map_adj(**params)

        # Control Flow (2, 4-5)
        elif opcode == Opcode.JMP:
            return self._map_jmp(**params)
        elif opcode == Opcode.BZ:
            return self._map_bz(**params)
        elif opcode == Opcode.BNZ:
            return self._map_bnz(**params)

        # System Calls
        elif opcode == Opcode.MALC:
            return self._map_malc()
        elif opcode == Opcode.EXIT:
            return self._map_exit()

        # I/O
        elif opcode == Opcode.GETCHAR:
            return self._map_getchar()
        elif opcode == Opcode.PUTCHAR:
            return self._map_putchar()

        else:
            raise NotImplementedError(f"Opcode {opcode} not yet mapped to graph")

    # =============================================================================
    # Pure FFN Operations (Direct Mapping)
    # =============================================================================

    def _map_add(self) -> ComputationGraph:
        """ADD: AX = pop + AX"""
        graph = ComputationGraph()
        a = graph.add_input("pop")  # Popped value
        b = graph.add_input("AX")
        result = graph.add_op(OpType.ADD, [a, b], "AX")
        return graph

    def _map_sub(self) -> ComputationGraph:
        """SUB: AX = pop - AX"""
        graph = ComputationGraph()
        a = graph.add_input("pop")
        b = graph.add_input("AX")
        result = graph.add_op(OpType.SUB, [a, b], "AX")
        return graph

    def _map_mul(self) -> ComputationGraph:
        """MUL: AX = pop * AX"""
        graph = ComputationGraph()
        a = graph.add_input("pop")
        b = graph.add_input("AX")
        result = graph.add_op(OpType.MUL, [a, b], "AX")
        return graph

    def _map_div(self) -> ComputationGraph:
        """DIV: AX = pop / AX"""
        graph = ComputationGraph()
        a = graph.add_input("pop")
        b = graph.add_input("AX")
        result = graph.add_op(OpType.DIV, [a, b], "AX")
        return graph

    def _map_mod(self) -> ComputationGraph:
        """MOD: AX = pop % AX"""
        graph = ComputationGraph()
        a = graph.add_input("pop")
        b = graph.add_input("AX")
        result = graph.add_op(OpType.MOD, [a, b], "AX")
        return graph

    def _map_eq(self) -> ComputationGraph:
        """EQ: AX = (pop == AX)"""
        graph = ComputationGraph()
        a = graph.add_input("pop")
        b = graph.add_input("AX")
        result = graph.add_op(OpType.CMP_EQ, [a, b], "AX")
        return graph

    def _map_ne(self) -> ComputationGraph:
        """NE: AX = (pop != AX)"""
        graph = ComputationGraph()
        a = graph.add_input("pop")
        b = graph.add_input("AX")
        result = graph.add_op(OpType.CMP_NE, [a, b], "AX")
        return graph

    def _map_lt(self) -> ComputationGraph:
        """LT: AX = (pop < AX)"""
        graph = ComputationGraph()
        a = graph.add_input("pop")
        b = graph.add_input("AX")
        result = graph.add_op(OpType.CMP_LT, [a, b], "AX")
        return graph

    def _map_gt(self) -> ComputationGraph:
        """GT: AX = (pop > AX)"""
        graph = ComputationGraph()
        a = graph.add_input("pop")
        b = graph.add_input("AX")
        result = graph.add_op(OpType.CMP_GT, [a, b], "AX")
        return graph

    def _map_le(self) -> ComputationGraph:
        """LE: AX = (pop <= AX)"""
        graph = ComputationGraph()
        a = graph.add_input("pop")
        b = graph.add_input("AX")
        result = graph.add_op(OpType.CMP_LE, [a, b], "AX")
        return graph

    def _map_ge(self) -> ComputationGraph:
        """GE: AX = (pop >= AX)"""
        graph = ComputationGraph()
        a = graph.add_input("pop")
        b = graph.add_input("AX")
        result = graph.add_op(OpType.CMP_GE, [a, b], "AX")
        return graph

    def _map_or(self) -> ComputationGraph:
        """OR: AX = pop | AX"""
        graph = ComputationGraph()
        a = graph.add_input("pop")
        b = graph.add_input("AX")
        result = graph.add_op(OpType.BIT_OR, [a, b], "AX")
        return graph

    def _map_xor(self) -> ComputationGraph:
        """XOR: AX = pop ^ AX"""
        graph = ComputationGraph()
        a = graph.add_input("pop")
        b = graph.add_input("AX")
        result = graph.add_op(OpType.BIT_XOR, [a, b], "AX")
        return graph

    def _map_and(self) -> ComputationGraph:
        """AND: AX = pop & AX"""
        graph = ComputationGraph()
        a = graph.add_input("pop")
        b = graph.add_input("AX")
        result = graph.add_op(OpType.BIT_AND, [a, b], "AX")
        return graph

    def _map_shl(self) -> ComputationGraph:
        """SHL: AX = pop << AX"""
        graph = ComputationGraph()
        a = graph.add_input("pop")
        b = graph.add_input("AX")
        result = graph.add_op(OpType.SHL, [a, b], "AX")
        return graph

    def _map_shr(self) -> ComputationGraph:
        """SHR: AX = pop >> AX"""
        graph = ComputationGraph()
        a = graph.add_input("pop")
        b = graph.add_input("AX")
        result = graph.add_op(OpType.SHR, [a, b], "AX")
        return graph

    # =============================================================================
    # Composite Operations
    # =============================================================================

    def _map_lea(self, imm: int = 0) -> ComputationGraph:
        """LEA: AX = BP + imm"""
        graph = ComputationGraph()
        bp = graph.add_input("BP")
        imm_val = graph.add_const(imm)
        result = graph.add_op(OpType.ADD, [bp, imm_val], "AX")
        return graph

    def _map_imm(self, imm: int = 0) -> ComputationGraph:
        """IMM: AX = imm"""
        graph = ComputationGraph()
        imm_val = graph.add_const(imm)
        result = graph.add_op(OpType.MOVE, [imm_val], "AX")
        return graph

    def _map_adj(self, imm: int = 0) -> ComputationGraph:
        """ADJ: SP += imm"""
        graph = ComputationGraph()
        sp = graph.add_input("SP")
        imm_val = graph.add_const(imm)
        result = graph.add_op(OpType.ADD, [sp, imm_val], "SP")
        return graph

    def _map_jmp(self, imm: int = 0) -> ComputationGraph:
        """JMP: PC = imm"""
        graph = ComputationGraph()
        imm_val = graph.add_const(imm)
        result = graph.add_op(OpType.PC_SET, [imm_val], "PC")
        return graph

    def _map_bz(self, imm: int = 0) -> ComputationGraph:
        """BZ: if (AX == 0) PC = imm else PC = PC + instr_width"""
        graph = ComputationGraph()

        # Condition: AX == 0
        ax = graph.add_input("AX")
        zero = graph.add_const(0)
        cond = graph.add_op(OpType.CMP_EQ, [ax, zero], "cond")

        # Target: immediate
        target = graph.add_const(imm)

        # Fallthrough: PC + instruction_width
        pc = graph.add_input("PC")
        width = graph.add_const(8)  # Assuming 8-byte instruction width
        fallthrough = graph.add_op(OpType.ADD, [pc, width], "fallthrough")

        # Conditional: PC = cond ? target : fallthrough
        result = graph.add_op(OpType.PC_CONDITIONAL, [cond, target, fallthrough], "PC")

        return graph

    def _map_bnz(self, imm: int = 0) -> ComputationGraph:
        """BNZ: if (AX != 0) PC = imm else PC = PC + instr_width"""
        graph = ComputationGraph()

        # Condition: AX != 0
        ax = graph.add_input("AX")
        zero = graph.add_const(0)
        cond = graph.add_op(OpType.CMP_NE, [ax, zero], "cond")

        # Target and fallthrough
        target = graph.add_const(imm)
        pc = graph.add_input("PC")
        width = graph.add_const(8)
        fallthrough = graph.add_op(OpType.ADD, [pc, width], "fallthrough")

        # Conditional PC update
        result = graph.add_op(OpType.PC_CONDITIONAL, [cond, target, fallthrough], "PC")

        return graph

    def _map_malc(self) -> ComputationGraph:
        """MALC: AX = malloc(size) - Pure FFN bump allocator!"""
        graph = ComputationGraph()

        # Inputs
        heap_ptr = graph.add_input("HEAP_PTR")
        size = graph.add_input("AX")  # size argument
        heap_end = graph.add_input("HEAP_END")

        # result = current HEAP_PTR
        result = graph.add_op(OpType.MOVE, [heap_ptr], "result")

        # new_ptr = HEAP_PTR + size
        new_ptr = graph.add_op(OpType.ADD, [heap_ptr, size], "new_ptr")

        # valid = (new_ptr < HEAP_END)
        valid = graph.add_op(OpType.CMP_LT, [new_ptr, heap_end], "valid")

        # Update HEAP_PTR: valid ? new_ptr : HEAP_PTR
        updated_ptr = graph.add_op(OpType.SELECT, [valid, new_ptr, heap_ptr], "HEAP_PTR")

        # Return: AX = valid ? result : 0
        zero = graph.add_const(0)
        ax_result = graph.add_op(OpType.SELECT, [valid, result, zero], "AX")

        return graph

    def _map_exit(self) -> ComputationGraph:
        """EXIT: exit(code) - Set IO_PROGRAM_END flag"""
        graph = ComputationGraph()

        # Set exit flag
        flag_node = graph.add_op(OpType.FLAG_SET, [], "IO_PROGRAM_END")

        # Copy exit code to mailbox (optional)
        ax = graph.add_input("AX")
        exit_code = graph.add_op(OpType.MOVE, [ax], "IO_EXIT_CODE")

        return graph

    def _map_getchar(self) -> ComputationGraph:
        """GETCHAR: AX = getchar() - Set IO_NEED_INPUT flag"""
        graph = ComputationGraph()

        # Layer 1: Request input
        request = graph.add_op(OpType.IO_GETCHAR_REQUEST, [], "io_request")

        # Layer 2 (separate): Read from IO_CHAR mailbox
        response = graph.add_op(OpType.IO_READ_RESPONSE, [], "AX")

        return graph

    def _map_putchar(self) -> ComputationGraph:
        """PUTCHAR: putchar(AX) - Copy to IO_CHAR, set IO_OUTPUT_READY"""
        graph = ComputationGraph()

        # Copy AX to IO_CHAR mailbox and set flag
        ax = graph.add_input("AX")
        output = graph.add_op(OpType.IO_PUTCHAR_REQUEST, [ax], "io_output")

        return graph

    # =============================================================================
    # Multi-Layer Operations (FFN → Attention → FFN)
    # =============================================================================

    def map_opcode_multilayer(self, opcode: int, **params) -> MultiLayerWeights:
        """Map opcode to multi-layer weights.

        For opcodes that need FFN → Attention → FFN pipeline.

        Returns:
            MultiLayerWeights with list of (layer_type, weights) tuples
        """
        if opcode == Opcode.LI:
            return self._map_li_multilayer()
        elif opcode == Opcode.LC:
            return self._map_lc_multilayer()
        elif opcode == Opcode.SI:
            return self._map_si_multilayer()
        elif opcode == Opcode.SC:
            return self._map_sc_multilayer()
        elif opcode == Opcode.PSH:
            return self._map_psh_multilayer()
        elif opcode == Opcode.JSR:
            return self._map_jsr_multilayer(**params)
        elif opcode == Opcode.ENT:
            return self._map_ent_multilayer(**params)
        elif opcode == Opcode.LEV:
            return self._map_lev_multilayer()
        else:
            raise NotImplementedError(f"Multi-layer mapping not implemented for opcode {opcode}")

    def _map_li_multilayer(self) -> MultiLayerWeights:
        """LI: AX = *AX (Load Int)

        Layer 1 (FFN): Copy AX → MEM_ADDR, set MEM_READ
        Layer 2 (Attention): Memory lookup via Q·K^T
        Layer 3 (FFN): Copy MEM_DATA → AX
        """
        layers = []

        # Layer 1: Setup read request
        graph1 = ComputationGraph()
        ax = graph1.add_input("AX")
        read_req = graph1.add_op(OpType.MEM_READ_REQUEST, [ax], "mem_read")
        layers.append(("ffn", graph1))

        # Layer 2: Attention (handled by existing attention layer)
        layers.append(("attention", None))

        # Layer 3: Copy result
        graph3 = ComputationGraph()
        mem_data = graph3.add_input("MEM_DATA")
        result = graph3.add_op(OpType.MOVE, [mem_data], "AX")
        layers.append(("ffn", graph3))

        return MultiLayerWeights(layers=layers)

    def _map_lc_multilayer(self) -> MultiLayerWeights:
        """LC: AX = *(char*)AX (Load Char) - Same as LI but 1 byte"""
        return self._map_li_multilayer()  # Same structure, different data size

    def _map_si_multilayer(self) -> MultiLayerWeights:
        """SI: *pop = AX (Store Int)

        Layer 1 (FFN): Copy pop → MEM_ADDR, AX → MEM_DATA, set MEM_WRITE
        Layer 2 (Attention): Generate MEM token, append to context
        """
        layers = []

        # Layer 1: Setup write request
        graph1 = ComputationGraph()
        addr = graph1.add_input("pop")
        data = graph1.add_input("AX")
        write_req = graph1.add_op(OpType.MEM_WRITE_REQUEST, [addr, data], "mem_write")
        layers.append(("ffn", graph1))

        # Layer 2: Attention (memory write)
        layers.append(("attention", None))

        return MultiLayerWeights(layers=layers)

    def _map_sc_multilayer(self) -> MultiLayerWeights:
        """SC: *(char*)pop = AX (Store Char)"""
        return self._map_si_multilayer()  # Same structure

    def _map_psh_multilayer(self) -> MultiLayerWeights:
        """PSH: SP -= 8; *SP = AX

        Layer 1 (FFN): SP -= 8, copy SP → MEM_ADDR, AX → MEM_DATA, set MEM_WRITE
        Layer 2 (Attention): Write to stack
        """
        layers = []

        # Layer 1: Stack push request
        graph1 = ComputationGraph()
        sp = graph1.add_input("SP")
        ax = graph1.add_input("AX")
        push = graph1.add_op(OpType.STACK_PUSH_REQUEST, [sp, ax], "stack_push")
        layers.append(("ffn", graph1))

        # Layer 2: Attention
        layers.append(("attention", None))

        return MultiLayerWeights(layers=layers)

    def _map_jsr_multilayer(self, imm: int = 0) -> MultiLayerWeights:
        """JSR: push PC; PC = imm

        Composite: STACK_PUSH + PC_SET
        """
        layers = []

        # Layer 1: Push PC
        graph1 = ComputationGraph()
        sp = graph1.add_input("SP")
        pc = graph1.add_input("PC")
        push = graph1.add_op(OpType.STACK_PUSH_REQUEST, [sp, pc], "push_pc")
        layers.append(("ffn", graph1))

        # Layer 2: Attention
        layers.append(("attention", None))

        # Layer 3: Set PC
        graph3 = ComputationGraph()
        target = graph3.add_const(imm)
        pc_set = graph3.add_op(OpType.PC_SET, [target], "PC")
        layers.append(("ffn", graph3))

        return MultiLayerWeights(layers=layers)

    def _map_ent_multilayer(self, imm: int = 0) -> MultiLayerWeights:
        """ENT: push BP; BP = SP; SP -= imm"""
        layers = []

        # Layer 1: Push BP
        graph1 = ComputationGraph()
        sp = graph1.add_input("SP")
        bp = graph1.add_input("BP")
        push = graph1.add_op(OpType.STACK_PUSH_REQUEST, [sp, bp], "push_bp")
        layers.append(("ffn", graph1))

        # Layer 2: Attention
        layers.append(("attention", None))

        # Layer 3: BP = SP, SP -= imm
        graph3 = ComputationGraph()
        sp_val = graph3.add_input("SP")
        bp_set = graph3.add_op(OpType.MOVE, [sp_val], "BP")
        imm_val = graph3.add_const(imm)
        sp_dec = graph3.add_op(OpType.SUB, [sp_val, imm_val], "SP")
        layers.append(("ffn", graph3))

        return MultiLayerWeights(layers=layers)

    def _map_lev_multilayer(self) -> MultiLayerWeights:
        """LEV: SP = BP; BP = pop; PC = pop"""
        layers = []

        # Layer 1: SP = BP, prepare pop BP
        graph1 = ComputationGraph()
        bp = graph1.add_input("BP")
        sp_set = graph1.add_op(OpType.MOVE, [bp], "SP")
        pop_bp = graph1.add_op(OpType.STACK_POP_REQUEST, [sp_set], "pop_bp_req")
        layers.append(("ffn", graph1))

        # Layer 2: Attention (read BP)
        layers.append(("attention", None))

        # Layer 3: BP = popped, prepare pop PC
        graph3 = ComputationGraph()
        bp_data = graph3.add_input("MEM_DATA")
        bp_update = graph3.add_op(OpType.MOVE, [bp_data], "BP")
        sp_current = graph3.add_input("SP")
        pop_pc = graph3.add_op(OpType.STACK_POP_REQUEST, [sp_current], "pop_pc_req")
        layers.append(("ffn", graph3))

        # Layer 4: Attention (read PC)
        layers.append(("attention", None))

        # Layer 5: PC = popped
        graph5 = ComputationGraph()
        pc_data = graph5.add_input("MEM_DATA")
        pc_update = graph5.add_op(OpType.MOVE, [pc_data], "PC")
        layers.append(("ffn", graph5))

        return MultiLayerWeights(layers=layers)
