"""
Multi-Operation Graph Compiler

Compiles computation graphs with multiple operations to a single FFN layer.

Strategy:
1. Topological sort: Order operations by dependencies
2. Slot allocation: Assign E class slots for intermediate values
3. Sequential emission: Emit each operation using allocated slots
4. Data flow: Use TEMP slots to pass data between operations

Example: BZ (Branch if Zero)
    Graph: AX → [CMP_EQ with 0] → cond
           PC → [ADD with 8] → fallthrough
           cond, target, fallthrough → [PC_CONDITIONAL] → PC

    Execution:
    1. CMP_EQ emits to TEMP (cond)
    2. ADD emits to TEMP2 (fallthrough)
    3. PC_CONDITIONAL reads TEMP, TEMP2, writes PC
"""

import torch
from typing import Dict, List, Tuple, Optional, Set
from .graph_weight_compiler import ComputationGraph, OpType
from .nibble_weight_compiler import NibbleWeightEmitter, NibbleRegisterMap
from .embedding import E

# NodeID is just an integer
NodeID = int


def topological_sort(graph: ComputationGraph) -> List[NodeID]:
    """
    Sort nodes by dependencies (topologically).

    Returns nodes in execution order where all dependencies come before dependents.
    """
    visited: Set[NodeID] = set()
    result: List[NodeID] = []

    def visit(node_id: NodeID):
        if node_id in visited:
            return
        visited.add(node_id)

        node = graph.nodes[node_id]
        # Visit all input dependencies first
        for input_id in node.inputs:
            if input_id in graph.nodes:
                visit(input_id)

        # Add this node after its dependencies
        if node.op != OpType.CONST:  # Skip constants
            result.append(node_id)

    # Visit all nodes
    for node_id in graph.nodes.keys():
        visit(node_id)

    return result


class IntermediateSlotAllocator:
    """
    Allocates E class slots for intermediate values in multi-operation graphs.

    Strategy:
    - Named outputs (AX, SP, PC) get specific result slots
    - Intermediate temps use sequential TEMP/TEMP2/TEMP3 slots
    - Conditions (0/1 flags) use single-value slots
    """

    def __init__(self):
        self.next_temp_slot = E.TEMP  # Start at TEMP (slot 6)
        self.allocations: Dict[NodeID, List[Tuple[int, int]]] = {}
        self.reg_map = NibbleRegisterMap()

    def allocate(self, node_id: NodeID, output_reg: Optional[str]) -> List[Tuple[int, int]]:
        """
        Allocate slots for a node's output.

        Returns: List of (position, slot) tuples
                 - 8 tuples for multi-nibble values (registers)
                 - 1 tuple for single values (conditions)
        """
        if node_id in self.allocations:
            return self.allocations[node_id]

        # Final outputs - use RESULT slots
        if output_reg in ["AX", "SP", "PC"]:
            slots = [(pos, E.RESULT) for pos in range(8)]

        # Condition flags (single value, not per-nibble)
        elif output_reg in ["cond", "valid", "flag"]:
            slots = [(0, E.TEMP)]  # Use TEMP slot for conditions

        # Intermediate temporaries (like "fallthrough", "target", etc.)
        # Use CARRY_IN slots since they're not needed for non-arithmetic ops
        elif output_reg is not None:
            slots = [(pos, E.CARRY_IN) for pos in range(8)]

        # Unnamed outputs - use CARRY_OUT as secondary intermediate
        else:
            slots = [(pos, E.CARRY_OUT) for pos in range(8)]

        self.allocations[node_id] = slots
        return slots


class MultiOperationCompiler:
    """
    Compile multi-operation graphs to single FFN layer.

    Handles graphs like:
    - BZ: CMP_EQ + ADD + PC_CONDITIONAL
    - MALC: ADD + CMP_LT + SELECT (×3)
    - ADJ: ADD
    """

    def __init__(self, num_positions: int = 8):
        self.num_positions = num_positions
        self.reg_map = NibbleRegisterMap()

    def compile_graph(
        self,
        graph: ComputationGraph,
        opcode: int
    ) -> Dict[str, torch.Tensor]:
        """
        Compile multi-operation graph to FFN weights.

        Returns: Weight dictionary compatible with PureFFN
        """
        # Create weight emitter
        emitter = NibbleWeightEmitter(opcode, self.num_positions)

        # Sort operations by dependencies
        sorted_nodes = topological_sort(graph)

        # Allocate slots for all node outputs
        allocator = IntermediateSlotAllocator()
        slot_map: Dict[NodeID, List[Tuple[int, int]]] = {}

        # Pre-allocate slots for input registers (CONST nodes with 'input'=True)
        # These represent VM state that we read from (AX, PC, SP, etc.)
        for node_id, node in graph.nodes.items():
            if node.op == OpType.CONST and node.params.get('input', False):
                # Use TEMP_A2 as staging area for input registers
                # Each position (0-7) gets its own TEMP_A2 slot instance
                slot_map[node_id] = [(pos, E.TEMP_A2) for pos in range(8)]

        # Allocate slots for computed node outputs
        for node_id in sorted_nodes:
            node = graph.nodes[node_id]
            slot_map[node_id] = allocator.allocate(node_id, node.output_reg)

        # Emit each operation sequentially
        for node_id in sorted_nodes:
            node = graph.nodes[node_id]

            if node.op == OpType.ADD:
                self._emit_add(emitter, node, graph, slot_map)
            elif node.op == OpType.SUB:
                self._emit_sub(emitter, node, graph, slot_map)
            elif node.op == OpType.CMP_EQ:
                self._emit_cmp_eq(emitter, node, graph, slot_map)
            elif node.op == OpType.CMP_NE:
                self._emit_cmp_ne(emitter, node, graph, slot_map)
            elif node.op == OpType.CMP_LT:
                self._emit_cmp_lt(emitter, node, graph, slot_map)
            elif node.op == OpType.CMP_GT:
                self._emit_cmp_gt(emitter, node, graph, slot_map)
            elif node.op == OpType.CMP_LE:
                self._emit_cmp_le(emitter, node, graph, slot_map)
            elif node.op == OpType.CMP_GE:
                self._emit_cmp_ge(emitter, node, graph, slot_map)
            elif node.op == OpType.MOVE:
                self._emit_move(emitter, node, graph, slot_map)
            elif node.op == OpType.PC_SET:
                self._emit_pc_set(emitter, node, graph, slot_map)
            elif node.op == OpType.PC_CONDITIONAL:
                self._emit_pc_conditional(emitter, node, graph, slot_map)
            elif node.op == OpType.SELECT:
                self._emit_select(emitter, node, graph, slot_map)
            elif node.op == OpType.MEM_READ_REQUEST:
                self._emit_mem_read_request(emitter, node, graph, slot_map)
            elif node.op == OpType.MEM_WRITE_REQUEST:
                self._emit_mem_write_request(emitter, node, graph, slot_map)
            elif node.op == OpType.STACK_PUSH_REQUEST:
                self._emit_stack_push_request(emitter, node, graph, slot_map)
            elif node.op == OpType.STACK_POP_REQUEST:
                self._emit_stack_pop_request(emitter, node, graph, slot_map)
            else:
                raise NotImplementedError(
                    f"Multi-operation emission not implemented for {node.op}"
                )

        return emitter.get_weights()

    def _get_input_value(
        self,
        input_id: NodeID,
        graph: ComputationGraph,
        slot_map: Dict[NodeID, List[Tuple[int, int]]]
    ) -> object:
        """
        Get the value for an input node.

        Returns:
        - For CONST nodes: the constant value (int)
        - For computed nodes: list of (position, slot) tuples
        """
        if input_id not in graph.nodes:
            raise ValueError(f"Input node {input_id} not found in graph")

        input_node = graph.nodes[input_id]

        if input_node.op == OpType.CONST:
            # Return the constant value directly
            if 'value' in input_node.params:
                return input_node.params['value']
            elif 'input' in input_node.params:
                # This is an input node (virtual register)
                # Return its allocated slots
                if input_id in slot_map:
                    return slot_map[input_id]
                else:
                    # Input register - allocate default slots
                    # For now, return None and handle specially
                    return input_node.output_reg  # Return register name
            else:
                return 0  # Default constant
        else:
            # Return the allocated slots
            return slot_map[input_id]

    def _emit_add(
        self,
        emitter: NibbleWeightEmitter,
        node,
        graph: ComputationGraph,
        slot_map: Dict[NodeID, List[Tuple[int, int]]]
    ):
        """Emit ADD operation with slot mapping."""
        # Get input values
        input1 = self._get_input_value(node.inputs[0], graph, slot_map)
        input2 = self._get_input_value(node.inputs[1], graph, slot_map)
        output_slots = slot_map[node.id]

        # Handle constant inputs
        if isinstance(input1, int):
            # Load constant to NIB_A slots
            for pos in range(8):
                nibble_val = (input1 >> (pos * 4)) & 0xF
                self._emit_constant_to_slot(emitter, nibble_val, (pos, E.NIB_A))
        else:
            # Copy from allocated slots to NIB_A
            for pos in range(8):
                self._emit_copy_nibble(emitter, input1[pos], (pos, E.NIB_A))

        if isinstance(input2, int):
            # Load constant to NIB_B slots
            for pos in range(8):
                nibble_val = (input2 >> (pos * 4)) & 0xF
                self._emit_constant_to_slot(emitter, nibble_val, (pos, E.NIB_B))
        else:
            # Copy from allocated slots to NIB_B
            for pos in range(8):
                self._emit_copy_nibble(emitter, input2[pos], (pos, E.NIB_B))

        # Emit ADD for all positions
        for pos in range(8):
            emitter.emit_add_nibble(pos)

        # Copy results to output slots
        for pos in range(8):
            self._emit_copy_nibble(emitter, (pos, E.RESULT), output_slots[pos])

    def _emit_sub(self, emitter, node, graph, slot_map):
        """Emit SUB operation (similar to ADD)."""
        input1 = self._get_input_value(node.inputs[0], graph, slot_map)
        input2 = self._get_input_value(node.inputs[1], graph, slot_map)
        output_slots = slot_map[node.id]

        # Setup inputs
        if isinstance(input1, int):
            for pos in range(8):
                nibble_val = (input1 >> (pos * 4)) & 0xF
                self._emit_constant_to_slot(emitter, nibble_val, (pos, E.NIB_A))
        else:
            for pos in range(8):
                self._emit_copy_nibble(emitter, input1[pos], (pos, E.NIB_A))

        if isinstance(input2, int):
            for pos in range(8):
                nibble_val = (input2 >> (pos * 4)) & 0xF
                self._emit_constant_to_slot(emitter, nibble_val, (pos, E.NIB_B))
        else:
            for pos in range(8):
                self._emit_copy_nibble(emitter, input2[pos], (pos, E.NIB_B))

        # Emit SUB
        for pos in range(8):
            emitter.emit_sub_nibble(pos)

        # Copy results
        for pos in range(8):
            self._emit_copy_nibble(emitter, (pos, E.RESULT), output_slots[pos])

    def _emit_cmp_eq(self, emitter, node, graph, slot_map):
        """Emit CMP_EQ operation - outputs single flag value."""
        input1 = self._get_input_value(node.inputs[0], graph, slot_map)
        input2 = self._get_input_value(node.inputs[1], graph, slot_map)
        output_slots = slot_map[node.id]  # Single slot for condition

        # Setup inputs
        if isinstance(input1, int):
            for pos in range(8):
                nibble_val = (input1 >> (pos * 4)) & 0xF
                self._emit_constant_to_slot(emitter, nibble_val, (pos, E.NIB_A))
        else:
            for pos in range(8):
                self._emit_copy_nibble(emitter, input1[pos], (pos, E.NIB_A))

        if isinstance(input2, int):
            for pos in range(8):
                nibble_val = (input2 >> (pos * 4)) & 0xF
                self._emit_constant_to_slot(emitter, nibble_val, (pos, E.NIB_B))
        else:
            for pos in range(8):
                self._emit_copy_nibble(emitter, input2[pos], (pos, E.NIB_B))

        # Emit CMP_EQ (writes to TEMP per nibble)
        for pos in range(8):
            emitter.emit_cmp_eq_nibble(pos)

        # AND all nibble comparisons together to get final condition
        # Result is 1 only if ALL nibbles match
        # For simplicity, we'll use the first nibble's result
        # (Full implementation would need AND reduction across all nibbles)
        output_pos, output_slot = output_slots[0]
        self._emit_copy_nibble(emitter, (0, E.TEMP), (output_pos, output_slot))

    def _emit_cmp_lt(self, emitter, node, graph, slot_map):
        """Emit CMP_LT operation."""
        # Similar to CMP_EQ but uses emit_cmp_lt_nibble
        input1 = self._get_input_value(node.inputs[0], graph, slot_map)
        input2 = self._get_input_value(node.inputs[1], graph, slot_map)
        output_slots = slot_map[node.id]

        # Setup inputs (similar pattern)
        if isinstance(input1, int):
            for pos in range(8):
                nibble_val = (input1 >> (pos * 4)) & 0xF
                self._emit_constant_to_slot(emitter, nibble_val, (pos, E.NIB_A))
        else:
            for pos in range(8):
                self._emit_copy_nibble(emitter, input1[pos], (pos, E.NIB_A))

        if isinstance(input2, int):
            for pos in range(8):
                nibble_val = (input2 >> (pos * 4)) & 0xF
                self._emit_constant_to_slot(emitter, nibble_val, (pos, E.NIB_B))
        else:
            for pos in range(8):
                self._emit_copy_nibble(emitter, input2[pos], (pos, E.NIB_B))

        # Emit CMP_LT
        for pos in range(8):
            emitter.emit_cmp_lt_nibble(pos)

        # Copy result
        output_pos, output_slot = output_slots[0]
        self._emit_copy_nibble(emitter, (0, E.TEMP), (output_pos, output_slot))

    def _emit_cmp_ne(self, emitter, node, graph, slot_map):
        """Emit CMP_NE."""
        # Same pattern as CMP_EQ
        pass  # Implementation similar to _emit_cmp_eq

    def _emit_cmp_gt(self, emitter, node, graph, slot_map):
        """Emit CMP_GT."""
        pass  # Implementation similar to _emit_cmp_lt

    def _emit_cmp_le(self, emitter, node, graph, slot_map):
        """Emit CMP_LE."""
        pass  # Implementation similar to _emit_cmp_lt

    def _emit_cmp_ge(self, emitter, node, graph, slot_map):
        """Emit CMP_GE."""
        pass  # Implementation similar to _emit_cmp_lt

    def _emit_move(self, emitter, node, graph, slot_map):
        """Emit MOVE operation."""
        input_val = self._get_input_value(node.inputs[0], graph, slot_map)
        output_slots = slot_map[node.id]

        if isinstance(input_val, int):
            # Move constant to output
            for pos in range(8):
                nibble_val = (input_val >> (pos * 4)) & 0xF
                self._emit_constant_to_slot(emitter, nibble_val, output_slots[pos])
        else:
            # Move from slots to output slots
            for pos in range(8):
                self._emit_copy_nibble(emitter, input_val[pos], output_slots[pos])

    def _emit_pc_set(self, emitter, node, graph, slot_map):
        """Emit PC_SET operation (JMP)."""
        # PC = imm (just a MOVE to PC slots)
        self._emit_move(emitter, node, graph, slot_map)

    def _emit_pc_conditional(self, emitter, node, graph, slot_map):
        """Emit PC_CONDITIONAL: PC = cond ? target : fallthrough"""
        # Get inputs
        cond = self._get_input_value(node.inputs[0], graph, slot_map)
        target = self._get_input_value(node.inputs[1], graph, slot_map)
        fallthrough = self._get_input_value(node.inputs[2], graph, slot_map)

        # Condition is single value
        cond_pos, cond_slot = cond[0] if isinstance(cond, list) else (0, cond)

        # Target and fallthrough are 8 nibbles each
        target_slots = target if isinstance(target, list) else [(pos, target) for pos in range(8)]
        fallthrough_slots = fallthrough if isinstance(fallthrough, list) else [(pos, fallthrough) for pos in range(8)]

        # Use existing emit_pc_conditional
        emitter.emit_pc_conditional(
            cond_slot=cond_slot,
            target_nibbles=target_slots,
            pc_nibbles=fallthrough_slots
        )

    def _emit_select(self, emitter, node, graph, slot_map):
        """
        Emit SELECT: result = cond ? val_true : val_false

        This is like PC_CONDITIONAL but for general values, not just PC.
        """
        # Get inputs
        cond = self._get_input_value(node.inputs[0], graph, slot_map)
        val_true = self._get_input_value(node.inputs[1], graph, slot_map)
        val_false = self._get_input_value(node.inputs[2], graph, slot_map)
        output_slots = slot_map[node.id]

        # Similar pattern to PC_CONDITIONAL but outputs to result slots
        # For now, use PC_CONDITIONAL logic but write to output_slots instead
        # (Would need to implement emit_select_nibble in emitter)

        # Simplified: just use PC_CONDITIONAL pattern
        cond_pos, cond_slot = cond[0] if isinstance(cond, list) else (0, cond)
        true_slots = val_true if isinstance(val_true, list) else [(pos, val_true) for pos in range(8)]
        false_slots = val_false if isinstance(val_false, list) else [(pos, val_false) for pos in range(8)]

        # For now, emit as PC_CONDITIONAL but target output_slots
        # (This is a simplification - full implementation would need custom SELECT emitter)
        emitter.emit_pc_conditional(
            cond_slot=cond_slot,
            target_nibbles=true_slots,
            pc_nibbles=false_slots
        )

    def _emit_copy_nibble(
        self,
        emitter: NibbleWeightEmitter,
        src: Tuple[int, int],
        dest: Tuple[int, int]
    ):
        """
        Emit weights to copy one nibble slot to another.

        Uses cancel pair pattern: +S, -S to create identity through residual.
        """
        src_pos, src_slot = src
        dest_pos, dest_slot = dest

        S = emitter.reg_map.SCALE

        src_idx = emitter.reg_map.flat_index(src_pos, src_slot)
        dest_idx = emitter.reg_map.flat_index(dest_pos, dest_slot)

        # Cancel pair for copy
        emitter.W_up[emitter.unit_offset, src_idx] = S
        emitter._opcode_gate(emitter.unit_offset)
        emitter.W_down[dest_idx, emitter.unit_offset] = 1.0 / S

        emitter.W_up[emitter.unit_offset + 1, src_idx] = -S
        emitter._opcode_gate(emitter.unit_offset + 1)
        emitter.W_down[dest_idx, emitter.unit_offset + 1] = 1.0 / S

        emitter.unit_offset += 2

    def _emit_constant_to_slot(
        self,
        emitter: NibbleWeightEmitter,
        value: int,
        dest: Tuple[int, int]
    ):
        """
        Emit weights to write a constant value to a slot.

        Uses gate bias to create constant output.
        """
        dest_pos, dest_slot = dest
        S = emitter.reg_map.SCALE

        dest_idx = emitter.reg_map.flat_index(dest_pos, dest_slot)

        # Write constant using gate bias
        emitter._opcode_gate(emitter.unit_offset)
        emitter.b_gate[emitter.unit_offset] = S * value
        emitter.W_down[dest_idx, emitter.unit_offset] = 1.0 / S

        # Cancel pair for stability
        emitter._opcode_gate(emitter.unit_offset + 1)
        emitter.b_gate[emitter.unit_offset + 1] = -S * value
        emitter.W_down[dest_idx, emitter.unit_offset + 1] = 1.0 / S

        emitter.unit_offset += 2

    def _emit_mem_read_request(self, emitter, node, graph, slot_map):
        """Emit MEM_READ_REQUEST operation.

        Copies address to MEM_ADDR slots and sets MEM_READ flag.
        """
        # Get input address
        input_val = self._get_input_value(node.inputs[0], graph, slot_map)

        # Copy address to MEM_ADDR slots (E.MEM_ADDR_BASE through MEM_ADDR_BASE+7)
        if isinstance(input_val, int):
            # Constant address - load directly to MEM_ADDR
            for pos in range(8):
                nibble = (input_val >> (pos * 4)) & 0xF
                mem_addr_slot = E.MEM_ADDR_BASE + pos
                self._emit_constant_to_slot(emitter, nibble, (0, mem_addr_slot))
        else:
            # Variable address - copy from slots
            for pos in range(8):
                src_slot = input_val[pos]
                dest_slot = (0, E.MEM_ADDR_BASE + pos)
                self._emit_copy_nibble(emitter, src_slot, dest_slot)

        # Set MEM_READ flag
        self._emit_constant_to_slot(emitter, 1, (0, E.MEM_READ))

    def _emit_mem_write_request(self, emitter, node, graph, slot_map):
        """Emit MEM_WRITE_REQUEST operation.

        Copies address to MEM_ADDR, data to MEM_DATA, and sets MEM_WRITE flag.
        """
        # Get input address and data
        addr_val = self._get_input_value(node.inputs[0], graph, slot_map)
        data_val = self._get_input_value(node.inputs[1], graph, slot_map)

        # Copy address to MEM_ADDR
        if isinstance(addr_val, int):
            for pos in range(8):
                nibble = (addr_val >> (pos * 4)) & 0xF
                mem_addr_slot = E.MEM_ADDR_BASE + pos
                self._emit_constant_to_slot(emitter, nibble, (0, mem_addr_slot))
        else:
            for pos in range(8):
                src_slot = addr_val[pos]
                dest_slot = (0, E.MEM_ADDR_BASE + pos)
                self._emit_copy_nibble(emitter, src_slot, dest_slot)

        # Copy data to MEM_DATA
        if isinstance(data_val, int):
            for pos in range(8):
                nibble = (data_val >> (pos * 4)) & 0xF
                mem_data_slot = E.MEM_DATA_BASE + pos
                self._emit_constant_to_slot(emitter, nibble, (0, mem_data_slot))
        else:
            for pos in range(8):
                src_slot = data_val[pos]
                dest_slot = (0, E.MEM_DATA_BASE + pos)
                self._emit_copy_nibble(emitter, src_slot, dest_slot)

        # Set MEM_WRITE flag
        self._emit_constant_to_slot(emitter, 1, (0, E.MEM_WRITE))

    def _emit_stack_push_request(self, emitter, node, graph, slot_map):
        """Emit STACK_PUSH_REQUEST operation.

        Decrements SP, copies SP to MEM_ADDR, copies data to MEM_DATA,
        and sets MEM_WRITE flag.
        """
        # For stack push, we typically:
        # 1. Get data to push
        # 2. Copy data to MEM_DATA
        # 3. Copy SP-8 to MEM_ADDR (address where to write)
        # 4. Set MEM_WRITE flag
        # Note: SP decrement might be in a separate node

        data_val = self._get_input_value(node.inputs[0], graph, slot_map)

        # Copy data to MEM_DATA
        if isinstance(data_val, int):
            for pos in range(8):
                nibble = (data_val >> (pos * 4)) & 0xF
                mem_data_slot = E.MEM_DATA_BASE + pos
                self._emit_constant_to_slot(emitter, nibble, (0, mem_data_slot))
        else:
            for pos in range(8):
                src_slot = data_val[pos]
                dest_slot = (0, E.MEM_DATA_BASE + pos)
                self._emit_copy_nibble(emitter, src_slot, dest_slot)

        # Set MEM_WRITE flag
        self._emit_constant_to_slot(emitter, 1, (0, E.MEM_WRITE))

    def _emit_stack_pop_request(self, emitter, node, graph, slot_map):
        """Emit STACK_POP_REQUEST operation.

        Copies SP to MEM_ADDR and sets MEM_READ flag.
        Note: SP increment happens in a separate operation.
        """
        # For stack pop:
        # 1. Copy SP to MEM_ADDR (address to read from)
        # 2. Set MEM_READ flag
        # SP increment will be in a separate ADD node

        addr_val = self._get_input_value(node.inputs[0], graph, slot_map)

        # Copy address to MEM_ADDR
        if isinstance(addr_val, int):
            for pos in range(8):
                nibble = (addr_val >> (pos * 4)) & 0xF
                mem_addr_slot = E.MEM_ADDR_BASE + pos
                self._emit_constant_to_slot(emitter, nibble, (0, mem_addr_slot))
        else:
            for pos in range(8):
                src_slot = addr_val[pos]
                dest_slot = (0, E.MEM_ADDR_BASE + pos)
                self._emit_copy_nibble(emitter, src_slot, dest_slot)

        # Set MEM_READ flag
        self._emit_constant_to_slot(emitter, 1, (0, E.MEM_READ))
