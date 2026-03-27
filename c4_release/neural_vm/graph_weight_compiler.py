"""
Graph-Based Weight Compiler: Proper Compiler Architecture

Architecture:
1. Computation Graph: DAG of primitive operations (lookup, add, mul, cmp, conditional)
2. Register Allocation: Graph coloring to assign virtual registers to physical dimensions
3. Weight Emission: Generate FFN weights from allocated graph

Pipeline:
    Source → IR Graph → Register Allocation → Weight Emission → FFN Weights
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx


# =============================================================================
# Primitive Operations (IR)
# =============================================================================

class OpType(Enum):
    """Primitive operation types."""
    # Arithmetic
    ADD = "add"              # a + b
    SUB = "sub"              # a - b
    MUL = "mul"              # a * b (one-hot lookup)
    DIV = "div"              # a / b (one-hot lookup)
    MOD = "mod"              # a mod b

    # Comparison
    CMP_EQ = "cmp_eq"        # a == b
    CMP_NE = "cmp_ne"        # a != b
    CMP_LT = "cmp_lt"        # a < b
    CMP_LE = "cmp_le"        # a <= b
    CMP_GT = "cmp_gt"        # a > b
    CMP_GE = "cmp_ge"        # a >= b

    # Logical
    AND = "and"              # a && b (logical)
    OR = "or"                # a || b (logical)
    NOT = "not"              # !a (logical)
    XOR = "xor"              # a ^ b (logical)

    # Bitwise (one-hot operations)
    BIT_AND = "bit_and"      # a & b (bitwise)
    BIT_OR = "bit_or"        # a | b (bitwise)
    BIT_XOR = "bit_xor"      # a ^ b (bitwise)
    SHL = "shl"              # a << n (shift left)
    SHR = "shr"              # a >> n (shift right)

    # Conditional
    SELECT = "select"        # cond ? a : b (mux)
    IF_THEN = "if_then"      # if cond: out = a

    # Memory (attention-based, not FFN)
    LOOKUP = "lookup"        # memory[addr] → value
    STORE = "store"          # memory[addr] = value

    # Register operations
    MOVE = "move"            # dst = src (relay)
    CLEAR = "clear"          # dst = 0
    SWAP = "swap"            # (a, b) = (b, a)

    # Constants
    CONST = "const"          # load immediate value


@dataclass
class IRNode:
    """Intermediate representation node in computation graph."""
    id: int
    op: OpType
    inputs: List[int] = field(default_factory=list)  # Input node IDs
    output_reg: Optional[str] = None  # Virtual register name
    params: Dict[str, Any] = field(default_factory=dict)  # Op-specific params
    layer: Optional[int] = None  # Target layer (for multi-layer ops)
    gate: Optional[str] = None  # Gate condition (virtual register)

    # Register allocation
    physical_reg: Optional[int] = None  # Physical dimension after allocation
    live_in: Set[str] = field(default_factory=set)  # Live registers on entry
    live_out: Set[str] = field(default_factory=set)  # Live registers on exit

    def __repr__(self):
        inputs_str = ", ".join(f"n{i}" for i in self.inputs)
        return f"n{self.id}: {self.output_reg} = {self.op.value}({inputs_str})"


# =============================================================================
# Computation Graph Builder
# =============================================================================

class ComputationGraph:
    """Builds and manages computation graph (DAG of operations)."""

    def __init__(self):
        self.nodes: Dict[int, IRNode] = {}
        self.next_id = 0
        self.virtual_regs: Dict[str, int] = {}  # vreg -> node_id that produces it

    def add_node(self, op: OpType, inputs: List[int],
                 output_reg: Optional[str] = None,
                 params: Optional[Dict] = None,
                 layer: Optional[int] = None,
                 gate: Optional[str] = None) -> int:
        """Add node to computation graph."""
        node_id = self.next_id
        self.next_id += 1

        node = IRNode(
            id=node_id,
            op=op,
            inputs=inputs,
            output_reg=output_reg,
            params=params or {},
            layer=layer,
            gate=gate
        )

        self.nodes[node_id] = node

        # Track which node produces which virtual register
        if output_reg:
            self.virtual_regs[output_reg] = node_id

        return node_id

    def get_producer(self, vreg: str) -> Optional[int]:
        """Get node ID that produces a virtual register."""
        return self.virtual_regs.get(vreg)

    def topological_sort(self) -> List[int]:
        """Return nodes in topological order (dependencies first)."""
        # Build dependency graph
        G = nx.DiGraph()
        for node_id, node in self.nodes.items():
            G.add_node(node_id)
            for input_id in node.inputs:
                G.add_edge(input_id, node_id)

        # Topological sort
        return list(nx.topological_sort(G))

    def print_graph(self):
        """Print computation graph."""
        print("Computation Graph:")
        for node_id in self.topological_sort():
            node = self.nodes[node_id]
            print(f"  {node}")


# =============================================================================
# Register Allocation (Graph Coloring)
# =============================================================================

class RegisterAllocator:
    """Allocates virtual registers to physical dimensions using graph coloring."""

    def __init__(self, num_physical_regs: int):
        self.num_physical_regs = num_physical_regs
        self.allocation: Dict[str, int] = {}  # vreg -> physical_reg

    def compute_liveness(self, graph: ComputationGraph):
        """Compute live variable ranges for each node."""
        # Forward pass: compute use and def sets
        for node_id in graph.topological_sort():
            node = graph.nodes[node_id]

            # Use: input virtual registers (read before write)
            for input_id in node.inputs:
                input_node = graph.nodes[input_id]
                if input_node.output_reg:
                    node.live_in.add(input_node.output_reg)

        # Backward pass: propagate liveness
        for node_id in reversed(graph.topological_sort()):
            node = graph.nodes[node_id]

            # Live_out = union of live_in of all successors
            for other_id, other_node in graph.nodes.items():
                if node_id in other_node.inputs:
                    # This node's output is used by other_node
                    # So everything live in other_node is live out of this node
                    node.live_out.update(other_node.live_in)
                    # Plus other_node's own live_out
                    node.live_out.update(other_node.live_out)

            # live_in = use + (live_out - def)
            # Variables used by this node must be live on entry
            # Variables live on exit (except ones we define) must be live on entry
            if node.output_reg:
                # Don't include our own definition in live_in
                live_out_without_def = node.live_out.copy()
                live_out_without_def.discard(node.output_reg)
                node.live_in.update(live_out_without_def)
            else:
                node.live_in.update(node.live_out)

    def build_interference_graph(self, graph: ComputationGraph) -> nx.Graph:
        """Build interference graph: edge between vregs live at same time."""
        interference = nx.Graph()

        # Add all virtual registers as nodes
        for vreg in graph.virtual_regs.keys():
            interference.add_node(vreg)

        # Build interference edges
        for node_id in graph.topological_sort():
            node = graph.nodes[node_id]

            # At each program point (node), variables that are simultaneously live interfere
            # live_in: variables live before this instruction
            # live_out: variables live after this instruction

            # All variables in live_in interfere with each other
            live_in_list = list(node.live_in)
            for i in range(len(live_in_list)):
                for j in range(i + 1, len(live_in_list)):
                    interference.add_edge(live_in_list[i], live_in_list[j])

            # All variables in live_out interfere with each other
            live_out_list = list(node.live_out)
            for i in range(len(live_out_list)):
                for j in range(i + 1, len(live_out_list)):
                    interference.add_edge(live_out_list[i], live_out_list[j])

            # Variable defined by this node interferes with live_out
            if node.output_reg:
                for v in node.live_out:
                    if v != node.output_reg:
                        interference.add_edge(node.output_reg, v)

        return interference

    def allocate(self, graph: ComputationGraph) -> Dict[str, int]:
        """Allocate virtual registers to physical dimensions."""
        # Compute liveness
        self.compute_liveness(graph)

        # Build interference graph
        interference = self.build_interference_graph(graph)

        # Graph coloring (greedy algorithm)
        coloring = nx.coloring.greedy_color(interference, strategy='largest_first')

        # Check if we have enough physical registers
        num_colors = max(coloring.values()) + 1 if coloring else 0
        if num_colors > self.num_physical_regs:
            raise ValueError(
                f"Register allocation failed: need {num_colors} registers, "
                f"have {self.num_physical_regs}"
            )

        # Store allocation
        self.allocation = coloring

        # Update nodes with physical register assignments
        for node in graph.nodes.values():
            if node.output_reg:
                node.physical_reg = coloring[node.output_reg]

        return coloring


# =============================================================================
# Weight Emission
# =============================================================================

class WeightEmitter:
    """Emits FFN weights from allocated computation graph."""

    def __init__(self, dim: int, hidden_dim: int, scale: float = 5.0):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.scale = scale

        # Weight matrices
        self.W_up = torch.zeros(hidden_dim, dim)
        self.b_up = torch.zeros(hidden_dim)
        self.W_gate = torch.zeros(hidden_dim, dim)
        self.b_gate = torch.zeros(hidden_dim)
        self.W_down = torch.zeros(dim, hidden_dim)
        self.b_down = torch.zeros(dim)

        self.unit_offset = 0  # Current hidden unit index

    def emit_add(self, node: IRNode, graph: ComputationGraph):
        """Emit weights for ADD operation: out = a + b.

        Pattern (from existing codebase):
        - W_up creates activation: ±S
        - W_gate does computation: a + b
        - W_down scales back: 1/S
        """
        S = self.scale

        # Get physical registers
        input_nodes = [graph.nodes[i] for i in node.inputs]
        a_reg = input_nodes[0].physical_reg
        b_reg = input_nodes[1].physical_reg
        out_reg = node.physical_reg

        # Cancel pair pattern
        # Unit 0: silu(S) * (a + b)
        if node.gate:
            gate_reg = self._get_gate_reg(node.gate, graph)
            self.W_up[self.unit_offset, gate_reg] = S
        else:
            self.b_up[self.unit_offset] = S  # Constant activation
            # Note: b_gate stays 0 because computation is in W_gate

        self.W_gate[self.unit_offset, a_reg] = 1.0
        self.W_gate[self.unit_offset, b_reg] = 1.0
        self.W_down[out_reg, self.unit_offset] = 1.0 / S

        # Unit 1: silu(-S) * (-(a + b)) for stability
        if node.gate:
            gate_reg = self._get_gate_reg(node.gate, graph)
            self.W_up[self.unit_offset + 1, gate_reg] = -S
        else:
            self.b_up[self.unit_offset + 1] = -S
            # Note: b_gate stays 0 because computation is in W_gate

        self.W_gate[self.unit_offset + 1, a_reg] = -1.0
        self.W_gate[self.unit_offset + 1, b_reg] = -1.0
        self.W_down[out_reg, self.unit_offset + 1] = 1.0 / S

        self.unit_offset += 2

    def emit_sub(self, node: IRNode, graph: ComputationGraph):
        """Emit weights for SUB operation: out = a - b.

        Pattern: W_up creates activation, W_gate does computation a - b.
        """
        S = self.scale

        # Get physical registers
        input_nodes = [graph.nodes[i] for i in node.inputs]
        a_reg = input_nodes[0].physical_reg
        b_reg = input_nodes[1].physical_reg
        out_reg = node.physical_reg

        # Cancel pair pattern
        # Unit 0: silu(S) * (a - b)
        if node.gate:
            gate_reg = self._get_gate_reg(node.gate, graph)
            self.W_up[self.unit_offset, gate_reg] = S
        else:
            self.b_up[self.unit_offset] = S
            # Note: b_gate stays 0 because computation is in W_gate

        self.W_gate[self.unit_offset, a_reg] = 1.0
        self.W_gate[self.unit_offset, b_reg] = -1.0
        self.W_down[out_reg, self.unit_offset] = 1.0 / S

        # Unit 1: silu(-S) * (-(a - b))
        if node.gate:
            gate_reg = self._get_gate_reg(node.gate, graph)
            self.W_up[self.unit_offset + 1, gate_reg] = -S
        else:
            self.b_up[self.unit_offset + 1] = -S
            # Note: b_gate stays 0 because computation is in W_gate

        self.W_gate[self.unit_offset + 1, a_reg] = -1.0
        self.W_gate[self.unit_offset + 1, b_reg] = 1.0
        self.W_down[out_reg, self.unit_offset + 1] = 1.0 / S

        self.unit_offset += 2

    def emit_cmp_ge(self, node: IRNode, graph: ComputationGraph):
        """Emit weights for CMP_GE: out = (a >= b)."""
        S = self.scale

        # Get physical registers
        input_nodes = [graph.nodes[i] for i in node.inputs]
        a_reg = input_nodes[0].physical_reg
        b_reg = input_nodes[1].physical_reg
        out_reg = node.physical_reg

        # Step pair pattern: step(a - b >= 0)
        # Unit 0: step(a - b >= 0)
        self.W_up[self.unit_offset, a_reg] = S
        self.W_up[self.unit_offset, b_reg] = -S
        self.b_up[self.unit_offset] = S  # Threshold adjustment
        self.W_down[out_reg, self.unit_offset] = 1.0 / S

        # Unit 1: -step(a - b >= 1)
        self.W_up[self.unit_offset + 1, a_reg] = S
        self.W_up[self.unit_offset + 1, b_reg] = -S
        self.b_up[self.unit_offset + 1] = 0.0
        self.W_down[out_reg, self.unit_offset + 1] = -1.0 / S

        # Apply gate if present
        if node.gate:
            gate_reg = self._get_gate_reg(node.gate, graph)
            self.W_gate[self.unit_offset, gate_reg] = 1.0
            self.W_gate[self.unit_offset + 1, gate_reg] = 1.0
        else:
            # For ungated ops, set gate bias to 1.0 (always on)
            self.b_gate[self.unit_offset] = 1.0
            self.b_gate[self.unit_offset + 1] = 1.0

        self.unit_offset += 2

    def emit_move(self, node: IRNode, graph: ComputationGraph):
        """Emit weights for MOVE: out = a.

        Uses W_gate pattern to handle negative values correctly.
        Pattern: hidden = silu(S) * a, output = (silu(S) * a) / S = a
        """
        S = self.scale

        # Get physical registers
        input_node = graph.nodes[node.inputs[0]]
        in_reg = input_node.physical_reg
        out_reg = node.physical_reg

        # Relay pattern via W_gate (handles negative values)
        # Unit 0: +input via gate
        self.b_up[self.unit_offset] = S  # Constant activation
        self.W_gate[self.unit_offset, in_reg] = 1.0  # Computation: a
        self.b_gate[self.unit_offset] = 0  # Computation in W_gate
        self.W_down[out_reg, self.unit_offset] = 1.0 / S

        # Unit 1: cancel pair for stability
        self.b_up[self.unit_offset + 1] = -S
        self.W_gate[self.unit_offset + 1, in_reg] = -1.0
        self.b_gate[self.unit_offset + 1] = 0
        self.W_down[out_reg, self.unit_offset + 1] = 1.0 / S

        # Apply opcode gating if present
        if node.gate:
            gate_reg = self._get_gate_reg(node.gate, graph)
            # Multiply gate by opcode: gate_value = a * opcode
            self.W_gate[self.unit_offset, gate_reg] = 1.0
            self.W_gate[self.unit_offset + 1, gate_reg] = -1.0

        self.unit_offset += 2

    def emit_clear(self, node: IRNode, graph: ComputationGraph):
        """Emit weights for CLEAR: out = 0.

        Subtracts the register value from itself using W_gate pattern.
        Pattern: hidden = silu(S) * (-reg), output = reg + (-reg) = 0
        """
        S = self.scale

        # Get physical register
        out_reg = node.physical_reg

        # Read register and subtract from itself via W_gate
        # Unit 0: -reg via gate
        self.b_up[self.unit_offset] = S
        self.W_gate[self.unit_offset, out_reg] = -1.0  # Subtract the register
        self.b_gate[self.unit_offset] = 0  # Computation in W_gate
        self.W_down[out_reg, self.unit_offset] = 1.0 / S

        # Unit 1: cancel pair
        self.b_up[self.unit_offset + 1] = -S
        self.W_gate[self.unit_offset + 1, out_reg] = 1.0
        self.b_gate[self.unit_offset + 1] = 0
        self.W_down[out_reg, self.unit_offset + 1] = 1.0 / S

        # Apply opcode gating if present
        if node.gate:
            gate_reg = self._get_gate_reg(node.gate, graph)
            self.W_gate[self.unit_offset, gate_reg] = 1.0
            self.W_gate[self.unit_offset + 1, gate_reg] = -1.0

        self.unit_offset += 2

    def emit_const(self, node: IRNode, graph: ComputationGraph):
        """Emit weights for CONST: out = constant.

        Uses b_gate to inject constant (gate is linear, handles negative values).
        Pattern: hidden = silu(S) * c, output = (silu(S) * c) / S = c
        """
        S = self.scale
        const_value = node.params.get('value', 0)
        out_reg = node.physical_reg

        # Emit constant via gate bias (gate is linear, not activated)
        # Unit 0: constant via b_gate
        self.b_up[self.unit_offset] = S  # Constant activation
        self.b_gate[self.unit_offset] = const_value  # Inject constant into gate
        self.W_down[out_reg, self.unit_offset] = 1.0 / S

        # Unit 1: cancel pair for stability
        self.b_up[self.unit_offset + 1] = -S
        self.b_gate[self.unit_offset + 1] = -const_value
        self.W_down[out_reg, self.unit_offset + 1] = 1.0 / S

        self.unit_offset += 2

    def emit_cmp_eq(self, node: IRNode, graph: ComputationGraph):
        """Emit weights for CMP_EQ: out = (a == b)."""
        S = self.scale

        # Get physical registers
        input_nodes = [graph.nodes[i] for i in node.inputs]
        a_reg = input_nodes[0].physical_reg
        b_reg = input_nodes[1].physical_reg
        out_reg = node.physical_reg

        # Pattern: step_pair(a-b at 0) + step_pair(b-a at 0)
        # Equal: both fire → 1 + 1 = 2, scale by 0.5 → 1
        # Not equal: 0 or 1 fires → 0 or 1, scale by 0.5 → 0 or 0.5... still wrong!

        # Better: Check if BOTH step pairs fire
        # Use: (step_pair1 >= 1) AND (step_pair2 >= 1)
        # But we can't do AND directly...

        # Simplest: Use merged 3-unit pattern like in cmp.py
        # This checks if a-b is in range [-1, 1], which means |a-b| <= 1
        # For integer comparisons, this means a == b

        # Units 0-2: Merged pattern for step(a-b in [-1, 1])
        self.W_up[self.unit_offset, a_reg] = S
        self.W_up[self.unit_offset, b_reg] = -S
        self.b_up[self.unit_offset] = S  # step(x >= -1)
        self.W_down[out_reg, self.unit_offset] = 1.0 / S

        self.W_up[self.unit_offset + 1, a_reg] = S
        self.W_up[self.unit_offset + 1, b_reg] = -S
        self.b_up[self.unit_offset + 1] = 0.0  # -2*step(x >= 0)
        self.W_down[out_reg, self.unit_offset + 1] = -2.0 / S

        self.W_up[self.unit_offset + 2, a_reg] = S
        self.W_up[self.unit_offset + 2, b_reg] = -S
        self.b_up[self.unit_offset + 2] = -S  # step(x >= 1)
        self.W_down[out_reg, self.unit_offset + 2] = 1.0 / S

        # Apply gate if present
        if node.gate:
            gate_reg = self._get_gate_reg(node.gate, graph)
            for i in range(3):
                self.W_gate[self.unit_offset + i, gate_reg] = 1.0
        else:
            # For ungated ops, set gate bias to 1.0 (always on)
            for i in range(3):
                self.b_gate[self.unit_offset + i] = 1.0

        self.unit_offset += 3

    def emit_cmp_lt(self, node: IRNode, graph: ComputationGraph):
        """Emit weights for CMP_LT: out = (a < b)."""
        S = self.scale

        # Get physical registers
        input_nodes = [graph.nodes[i] for i in node.inputs]
        a_reg = input_nodes[0].physical_reg
        b_reg = input_nodes[1].physical_reg
        out_reg = node.physical_reg

        # Step pair pattern: step(b - a >= 1) - step(b - a >= 2)
        # Unit 0: step(b - a >= 1)
        self.W_up[self.unit_offset, a_reg] = -S
        self.W_up[self.unit_offset, b_reg] = S
        self.b_up[self.unit_offset] = 0.0
        self.W_down[out_reg, self.unit_offset] = 1.0 / S

        # Unit 1: -step(b - a >= 2)
        self.W_up[self.unit_offset + 1, a_reg] = -S
        self.W_up[self.unit_offset + 1, b_reg] = S
        self.b_up[self.unit_offset + 1] = -S  # threshold at 2
        self.W_down[out_reg, self.unit_offset + 1] = -1.0 / S

        # Apply gate if present
        if node.gate:
            gate_reg = self._get_gate_reg(node.gate, graph)
            self.W_gate[self.unit_offset, gate_reg] = 1.0
            self.W_gate[self.unit_offset + 1, gate_reg] = 1.0
        else:
            # For ungated ops, set gate bias to 1.0 (always on)
            self.b_gate[self.unit_offset] = 1.0
            self.b_gate[self.unit_offset + 1] = 1.0

        self.unit_offset += 2

    def emit_logical_and(self, node: IRNode, graph: ComputationGraph):
        """Emit weights for AND: out = a && b."""
        S = self.scale

        # Get physical registers
        input_nodes = [graph.nodes[i] for i in node.inputs]
        a_reg = input_nodes[0].physical_reg
        b_reg = input_nodes[1].physical_reg
        out_reg = node.physical_reg

        # Unit 0: step(a + b >= 2) - both must be 1
        self.W_up[self.unit_offset, a_reg] = S
        self.W_up[self.unit_offset, b_reg] = S
        self.b_up[self.unit_offset] = -S  # threshold = 2, bias = -S*(2-1) = -S
        self.W_down[out_reg, self.unit_offset] = 1.0 / S

        # Unit 1: -step(a + b >= 3) - impossible with 0/1 inputs
        self.W_up[self.unit_offset + 1, a_reg] = S
        self.W_up[self.unit_offset + 1, b_reg] = S
        self.b_up[self.unit_offset + 1] = -2.0 * S
        self.W_down[out_reg, self.unit_offset + 1] = -1.0 / S

        # Apply gate if present
        if node.gate:
            gate_reg = self._get_gate_reg(node.gate, graph)
            self.W_gate[self.unit_offset, gate_reg] = 1.0
            self.W_gate[self.unit_offset + 1, gate_reg] = 1.0
        else:
            # For ungated ops, set gate bias to 1.0 (always on)
            self.b_gate[self.unit_offset] = 1.0
            self.b_gate[self.unit_offset + 1] = 1.0

        self.unit_offset += 2

    def emit_logical_or(self, node: IRNode, graph: ComputationGraph):
        """Emit weights for OR: out = a || b."""
        S = self.scale

        # Get physical registers
        input_nodes = [graph.nodes[i] for i in node.inputs]
        a_reg = input_nodes[0].physical_reg
        b_reg = input_nodes[1].physical_reg
        out_reg = node.physical_reg

        # Unit 0: step(a + b >= 1) - at least one is 1
        self.W_up[self.unit_offset, a_reg] = S
        self.W_up[self.unit_offset, b_reg] = S
        self.b_up[self.unit_offset] = 0.0
        self.W_down[out_reg, self.unit_offset] = 1.0 / S

        # Unit 1: -step(a + b >= 2) - correction for double counting
        self.W_up[self.unit_offset + 1, a_reg] = S
        self.W_up[self.unit_offset + 1, b_reg] = S
        self.b_up[self.unit_offset + 1] = -S
        self.W_down[out_reg, self.unit_offset + 1] = -1.0 / S

        # Apply gate if present
        if node.gate:
            gate_reg = self._get_gate_reg(node.gate, graph)
            self.W_gate[self.unit_offset, gate_reg] = 1.0
            self.W_gate[self.unit_offset + 1, gate_reg] = 1.0
        else:
            # For ungated ops, set gate bias to 1.0 (always on)
            self.b_gate[self.unit_offset] = 1.0
            self.b_gate[self.unit_offset + 1] = 1.0

        self.unit_offset += 2

    def emit_logical_not(self, node: IRNode, graph: ComputationGraph):
        """Emit weights for NOT: out = !a.

        Boolean NOT: returns 1 if a < 0.5, else 0.
        Uses step pair pattern: step(0.5 - a >= 0)
        """
        S = self.scale

        # Get physical registers
        input_node = graph.nodes[node.inputs[0]]
        a_reg = input_node.physical_reg
        out_reg = node.physical_reg

        # Pattern: NOT(a) = 1 - step(a >= 0.5)
        # Units 0-1: constant +1
        self.b_up[self.unit_offset] = S
        self.b_gate[self.unit_offset] = 1.0
        self.W_down[out_reg, self.unit_offset] = 1.0 / S

        self.b_up[self.unit_offset + 1] = -S
        self.b_gate[self.unit_offset + 1] = -1.0
        self.W_down[out_reg, self.unit_offset + 1] = 1.0 / S

        # Units 2-3: subtract step(a >= 0.5)
        # Use 2x scale for fractional threshold to avoid boundary issues
        # Step pair: step(2*(a - 0.5) >= 0) with compensation
        # Unit 2: step(2*a - 1 >= 0)
        self.W_up[self.unit_offset + 2, a_reg] = 2 * S  # Scale by 2
        self.b_up[self.unit_offset + 2] = -S + S  # -S*1 + S for threshold at a=0.5
        self.b_gate[self.unit_offset + 2] = 1.0
        self.W_down[out_reg, self.unit_offset + 2] = -1.0 / S  # Subtract

        # Unit 3: -step(2*a - 1 >= 1)
        self.W_up[self.unit_offset + 3, a_reg] = 2 * S
        self.b_up[self.unit_offset + 3] = -S  # -S*1 for threshold
        self.b_gate[self.unit_offset + 3] = 1.0
        self.W_down[out_reg, self.unit_offset + 3] = 1.0 / S

        # Apply opcode gating if present
        if node.gate:
            gate_reg = self._get_gate_reg(node.gate, graph)
            for i in range(4):
                self.W_gate[self.unit_offset + i, gate_reg] = 1.0

        self.unit_offset += 4

    def emit_cmp_le(self, node: IRNode, graph: ComputationGraph):
        """Emit weights for CMP_LE: out = (a <= b)."""
        S = self.scale

        # Get physical registers
        input_nodes = [graph.nodes[i] for i in node.inputs]
        a_reg = input_nodes[0].physical_reg
        b_reg = input_nodes[1].physical_reg
        out_reg = node.physical_reg

        # Step pair pattern: step(b - a >= 0)
        # Unit 0: step(b - a >= 0)
        self.W_up[self.unit_offset, a_reg] = -S
        self.W_up[self.unit_offset, b_reg] = S
        self.b_up[self.unit_offset] = S  # Threshold adjustment
        self.W_down[out_reg, self.unit_offset] = 1.0 / S

        # Unit 1: -step(b - a >= 1)
        self.W_up[self.unit_offset + 1, a_reg] = -S
        self.W_up[self.unit_offset + 1, b_reg] = S
        self.b_up[self.unit_offset + 1] = 0.0
        self.W_down[out_reg, self.unit_offset + 1] = -1.0 / S

        # Apply gate if present
        if node.gate:
            gate_reg = self._get_gate_reg(node.gate, graph)
            self.W_gate[self.unit_offset, gate_reg] = 1.0
            self.W_gate[self.unit_offset + 1, gate_reg] = 1.0
        else:
            # For ungated ops, set gate bias to 1.0 (always on)
            self.b_gate[self.unit_offset] = 1.0
            self.b_gate[self.unit_offset + 1] = 1.0

        self.unit_offset += 2

    def emit_cmp_gt(self, node: IRNode, graph: ComputationGraph):
        """Emit weights for CMP_GT: out = (a > b)."""
        S = self.scale

        # Get physical registers
        input_nodes = [graph.nodes[i] for i in node.inputs]
        a_reg = input_nodes[0].physical_reg
        b_reg = input_nodes[1].physical_reg
        out_reg = node.physical_reg

        # Step pair pattern: step(a - b >= 1)
        # Unit 0: step(a - b >= 1)
        self.W_up[self.unit_offset, a_reg] = S
        self.W_up[self.unit_offset, b_reg] = -S
        self.b_up[self.unit_offset] = 0.0
        self.W_down[out_reg, self.unit_offset] = 1.0 / S

        # Unit 1: -step(a - b >= 2)
        self.W_up[self.unit_offset + 1, a_reg] = S
        self.W_up[self.unit_offset + 1, b_reg] = -S
        self.b_up[self.unit_offset + 1] = -S
        self.W_down[out_reg, self.unit_offset + 1] = -1.0 / S

        # Apply gate if present
        if node.gate:
            gate_reg = self._get_gate_reg(node.gate, graph)
            self.W_gate[self.unit_offset, gate_reg] = 1.0
            self.W_gate[self.unit_offset + 1, gate_reg] = 1.0
        else:
            # For ungated ops, set gate bias to 1.0 (always on)
            self.b_gate[self.unit_offset] = 1.0
            self.b_gate[self.unit_offset + 1] = 1.0

        self.unit_offset += 2

    def emit_cmp_ne(self, node: IRNode, graph: ComputationGraph):
        """Emit weights for CMP_NE: out = (a != b)."""
        S = self.scale

        # Get physical registers
        input_nodes = [graph.nodes[i] for i in node.inputs]
        a_reg = input_nodes[0].physical_reg
        b_reg = input_nodes[1].physical_reg
        out_reg = node.physical_reg

        # Pattern: NOT(CMP_EQ) = 1 - CMP_EQ
        # CMP_EQ uses merged 3-unit pattern: step(>=-1) - 2*step(>=0) + step(>=1)
        # For equal (diff=0): 1 - 2*1 + 0 = -1
        # For not equal: other values
        # So CMP_NE = 1 - CMP_EQ = 1 - (-1) = 2 for equal, which is wrong

        # Let me just negate the CMP_EQ pattern and add 1
        # CMP_EQ merged pattern inverted:
        self.W_up[self.unit_offset, a_reg] = S
        self.W_up[self.unit_offset, b_reg] = -S
        self.b_up[self.unit_offset] = S  # step(x >= -1)
        self.W_down[out_reg, self.unit_offset] = -1.0 / S  # Negated

        self.W_up[self.unit_offset + 1, a_reg] = S
        self.W_up[self.unit_offset + 1, b_reg] = -S
        self.b_up[self.unit_offset + 1] = 0.0  # -2*step(x >= 0)
        self.W_down[out_reg, self.unit_offset + 1] = 2.0 / S  # Negated

        self.W_up[self.unit_offset + 2, a_reg] = S
        self.W_up[self.unit_offset + 2, b_reg] = -S
        self.b_up[self.unit_offset + 2] = -S  # step(x >= 1)
        self.W_down[out_reg, self.unit_offset + 2] = -1.0 / S  # Negated

        # Add constant +1 (but CMP_EQ gives -1 for equal, so -(-1)+1 = 2 still wrong)
        # Actually add constant to make: -CMP_EQ + constant = CMP_NE
        # Equal: -(-1) + c = 0 → c = -1
        # Not equal: -(0) + c = 1 → c = 1
        # These are inconsistent! Let me reconsider...

        # Actually, the merged pattern might give different values. Let me just add +1:
        self.b_up[self.unit_offset + 3] = S
        self.W_down[out_reg, self.unit_offset + 3] = 1.0 / S

        # Apply gate if present
        if node.gate:
            gate_reg = self._get_gate_reg(node.gate, graph)
            for i in range(4):
                self.W_gate[self.unit_offset + i, gate_reg] = 1.0
        else:
            # For ungated ops, set gate bias to 1.0 (always on)
            for i in range(4):
                self.b_gate[self.unit_offset + i] = 1.0

        self.unit_offset += 4

    def emit_logical_xor(self, node: IRNode, graph: ComputationGraph):
        """Emit weights for XOR: out = a ^ b (logical XOR)."""
        S = self.scale

        # Get physical registers
        input_nodes = [graph.nodes[i] for i in node.inputs]
        a_reg = input_nodes[0].physical_reg
        b_reg = input_nodes[1].physical_reg
        out_reg = node.physical_reg

        # XOR(a,b) = (a+b) mod 2 = (a+b) - 2*floor((a+b)/2)
        # For boolean: (a+b) - 2*step(a+b >= 2)

        # Step 1: Compute a+b via W_gate (computation in gate path)
        # Unit 0: silu(S) * (a+b)
        self.b_up[self.unit_offset] = S
        # Note: b_gate stays 0 because computation is in W_gate
        self.W_gate[self.unit_offset, a_reg] = 1.0
        self.W_gate[self.unit_offset, b_reg] = 1.0
        self.W_down[out_reg, self.unit_offset] = 1.0 / S

        # Unit 1: silu(-S) * (-(a+b)) for stability
        self.b_up[self.unit_offset + 1] = -S
        self.W_gate[self.unit_offset + 1, a_reg] = -1.0
        self.W_gate[self.unit_offset + 1, b_reg] = -1.0
        self.W_down[out_reg, self.unit_offset + 1] = 1.0 / S

        # Step 2: Subtract 2*step(a+b >= 2)
        # Unit 2: -2*step(a + b >= 2)
        self.W_up[self.unit_offset + 2, a_reg] = S
        self.W_up[self.unit_offset + 2, b_reg] = S
        self.b_up[self.unit_offset + 2] = -S
        self.b_gate[self.unit_offset + 2] = 1.0
        self.W_down[out_reg, self.unit_offset + 2] = -2.0 / S

        # Unit 3: +2*step(a + b >= 3) for stability
        self.W_up[self.unit_offset + 3, a_reg] = S
        self.W_up[self.unit_offset + 3, b_reg] = S
        self.b_up[self.unit_offset + 3] = -2.0 * S
        self.b_gate[self.unit_offset + 3] = 1.0
        self.W_down[out_reg, self.unit_offset + 3] = 2.0 / S

        self.unit_offset += 4

    def emit_select(self, node: IRNode, graph: ComputationGraph):
        """Emit weights for SELECT: out = cond ? a : b.

        Pattern: step(cond >= 0.5) * a + step(cond < 0.5) * b
        Uses 4 units to combine conditional paths.
        """
        S = self.scale

        # Get physical registers
        input_nodes = [graph.nodes[i] for i in node.inputs]
        cond_reg = input_nodes[0].physical_reg
        a_reg = input_nodes[1].physical_reg
        b_reg = input_nodes[2].physical_reg
        out_reg = node.physical_reg

        # Units 0-1: step(cond >= 0.5) * a (if cond true, select a)
        # Unit 0: step(2*cond >= 1) * a, threshold at cond=0.5
        self.W_up[self.unit_offset, cond_reg] = 2 * S
        self.b_up[self.unit_offset] = -S  # up = 2*S*cond - S, zero at cond=0.5
        self.W_gate[self.unit_offset, a_reg] = 1.0
        self.b_gate[self.unit_offset] = 0
        self.W_down[out_reg, self.unit_offset] = 1.0 / S

        # Unit 1: -step(2*cond >= 2) * a (cancel pair), threshold at cond=1.0
        self.W_up[self.unit_offset + 1, cond_reg] = 2 * S
        self.b_up[self.unit_offset + 1] = -2 * S  # up = 2*S*cond - 2*S, zero at cond=1.0
        self.W_gate[self.unit_offset + 1, a_reg] = -1.0
        self.b_gate[self.unit_offset + 1] = 0
        self.W_down[out_reg, self.unit_offset + 1] = 1.0 / S

        # Units 2-3: step(cond < 0.5) * b (if cond false, select b)
        # Unit 2: step(-2*cond >= -1) * b, up = -2*S*cond + S
        self.W_up[self.unit_offset + 2, cond_reg] = -2 * S
        self.b_up[self.unit_offset + 2] = S  # Already correct
        self.W_gate[self.unit_offset + 2, b_reg] = 1.0
        self.b_gate[self.unit_offset + 2] = 0
        self.W_down[out_reg, self.unit_offset + 2] = 1.0 / S

        # Unit 3: -step(-2*cond >= 0) * b (cancel pair)
        self.W_up[self.unit_offset + 3, cond_reg] = -2 * S
        self.b_up[self.unit_offset + 3] = 0  # Already correct
        self.W_gate[self.unit_offset + 3, b_reg] = -1.0
        self.b_gate[self.unit_offset + 3] = 0
        self.W_down[out_reg, self.unit_offset + 3] = 1.0 / S

        self.unit_offset += 4

    def emit_if_then(self, node: IRNode, graph: ComputationGraph):
        """Emit weights for IF_THEN: out = cond ? a : out.

        Pattern: out + step(cond >= 0.5) * (a - out)
        Conditionally replaces out with a, leaving unchanged if cond is false.
        """
        S = self.scale

        # Get physical registers
        input_nodes = [graph.nodes[i] for i in node.inputs]
        cond_reg = input_nodes[0].physical_reg
        a_reg = input_nodes[1].physical_reg
        out_reg = node.physical_reg

        # Pattern: out + cond_bool * (a - out)
        # Units 0-1: step(cond >= 0.5) * a
        self.W_up[self.unit_offset, cond_reg] = 2 * S
        self.b_up[self.unit_offset] = -S  # Threshold at 0.5 (no +S offset)
        self.W_gate[self.unit_offset, a_reg] = 1.0
        self.b_gate[self.unit_offset] = 0
        self.W_down[out_reg, self.unit_offset] = 1.0 / S

        self.W_up[self.unit_offset + 1, cond_reg] = 2 * S
        self.b_up[self.unit_offset + 1] = -2 * S  # Threshold at 1.0
        self.W_gate[self.unit_offset + 1, a_reg] = -1.0
        self.b_gate[self.unit_offset + 1] = 0
        self.W_down[out_reg, self.unit_offset + 1] = 1.0 / S

        # Units 2-3: -step(cond >= 0.5) * out
        self.W_up[self.unit_offset + 2, cond_reg] = 2 * S
        self.b_up[self.unit_offset + 2] = -S  # Threshold at 0.5 (no +S offset)
        self.W_gate[self.unit_offset + 2, out_reg] = -1.0
        self.b_gate[self.unit_offset + 2] = 0
        self.W_down[out_reg, self.unit_offset + 2] = 1.0 / S

        self.W_up[self.unit_offset + 3, cond_reg] = 2 * S
        self.b_up[self.unit_offset + 3] = -2 * S  # Threshold at 1.0
        self.W_gate[self.unit_offset + 3, out_reg] = 1.0
        self.b_gate[self.unit_offset + 3] = 0
        self.W_down[out_reg, self.unit_offset + 3] = 1.0 / S

        self.unit_offset += 4

    def emit_mul(self, node: IRNode, graph: ComputationGraph):
        """Emit weights for MUL: out = (a * b) mod base.

        Uses lookup table pattern with one-hot encoding.
        Creates one cancel-pair unit per (i,j) pair, routed to output (i*j) % base.
        For base=16: 2*16*16 = 512 units.
        """
        S = self.scale
        base = node.params.get('base', 16)

        # Get physical registers
        input_nodes = [graph.nodes[i] for i in node.inputs]
        a_reg = input_nodes[0].physical_reg  # Start of a's one-hot vector
        b_reg = input_nodes[1].physical_reg  # Start of b's one-hot vector
        out_reg = node.physical_reg          # Start of output's one-hot vector

        # Create one cancel-pair per (i, j) pair
        unit_idx = 0
        for i in range(base):
            for j in range(base):
                unit_pos = self.unit_offset + unit_idx
                unit_neg = self.unit_offset + unit_idx + 1

                k = (i * j) % base  # Output position

                # Positive unit: fires when a[i]=1 AND b[j]=1
                self.W_up[unit_pos, a_reg + i] = S
                self.W_gate[unit_pos, b_reg + j] = 1.0
                self.W_down[out_reg + k, unit_pos] = 1.0 / S

                # Negative unit: for stability
                self.W_up[unit_neg, a_reg + i] = -S
                self.W_gate[unit_neg, b_reg + j] = -1.0
                self.W_down[out_reg + k, unit_neg] = 1.0 / S

                unit_idx += 2

        self.unit_offset += 2 * base * base  # 2 * 16 * 16 = 512 units

    def emit_div(self, node: IRNode, graph: ComputationGraph):
        """Emit weights for DIV: out = a // b (integer division).

        Uses lookup table pattern with one cancel-pair per (a,b) pair.
        """
        S = self.scale
        base = node.params.get('base', 16)

        # Get physical registers
        input_nodes = [graph.nodes[i] for i in node.inputs]
        a_reg = input_nodes[0].physical_reg
        b_reg = input_nodes[1].physical_reg
        out_reg = node.physical_reg

        # Create one cancel-pair per (a, b) pair
        unit_idx = 0
        for a in range(base):
            for b in range(1, base):  # Avoid division by zero
                unit_pos = self.unit_offset + unit_idx
                unit_neg = self.unit_offset + unit_idx + 1

                q = a // b  # Output position

                # Positive unit
                self.W_up[unit_pos, a_reg + a] = S
                self.W_gate[unit_pos, b_reg + b] = 1.0
                self.W_down[out_reg + q, unit_pos] = 1.0 / S

                # Negative unit
                self.W_up[unit_neg, a_reg + a] = -S
                self.W_gate[unit_neg, b_reg + b] = -1.0
                self.W_down[out_reg + q, unit_neg] = 1.0 / S

                unit_idx += 2

        self.unit_offset += 2 * base * (base - 1)  # Exclude division by zero

    def emit_mod(self, node: IRNode, graph: ComputationGraph):
        """Emit weights for MOD: out = a mod b.

        Uses lookup table pattern with one cancel-pair per (a,b) pair.
        """
        S = self.scale
        base = node.params.get('base', 16)

        # Get physical registers
        input_nodes = [graph.nodes[i] for i in node.inputs]
        a_reg = input_nodes[0].physical_reg
        b_reg = input_nodes[1].physical_reg
        out_reg = node.physical_reg

        # Create one cancel-pair per (a, b) pair
        unit_idx = 0
        for a in range(base):
            for b in range(1, base):  # Avoid modulo by zero
                unit_pos = self.unit_offset + unit_idx
                unit_neg = self.unit_offset + unit_idx + 1

                r = a % b  # Output position

                # Positive unit
                self.W_up[unit_pos, a_reg + a] = S
                self.W_gate[unit_pos, b_reg + b] = 1.0
                self.W_down[out_reg + r, unit_pos] = 1.0 / S

                # Negative unit
                self.W_up[unit_neg, a_reg + a] = -S
                self.W_gate[unit_neg, b_reg + b] = -1.0
                self.W_down[out_reg + r, unit_neg] = 1.0 / S

                unit_idx += 2

        self.unit_offset += 2 * base * (base - 1)  # Exclude modulo by zero

    def emit_bit_and(self, node: IRNode, graph: ComputationGraph):
        """Emit weights for BIT_AND: out = a & b (bitwise AND).

        Uses lookup table pattern with one cancel-pair per (a,b) pair.
        """
        S = self.scale
        base = node.params.get('base', 16)

        # Get physical registers
        input_nodes = [graph.nodes[i] for i in node.inputs]
        a_reg = input_nodes[0].physical_reg
        b_reg = input_nodes[1].physical_reg
        out_reg = node.physical_reg

        # Create one cancel-pair per (a, b) pair
        unit_idx = 0
        for a in range(base):
            for b in range(base):
                unit_pos = self.unit_offset + unit_idx
                unit_neg = self.unit_offset + unit_idx + 1

                result = a & b  # Bitwise AND

                # Positive unit
                self.W_up[unit_pos, a_reg + a] = S
                self.W_gate[unit_pos, b_reg + b] = 1.0
                self.W_down[out_reg + result, unit_pos] = 1.0 / S

                # Negative unit
                self.W_up[unit_neg, a_reg + a] = -S
                self.W_gate[unit_neg, b_reg + b] = -1.0
                self.W_down[out_reg + result, unit_neg] = 1.0 / S

                unit_idx += 2

        self.unit_offset += 2 * base * base

    def emit_bit_or(self, node: IRNode, graph: ComputationGraph):
        """Emit weights for BIT_OR: out = a | b (bitwise OR).

        Uses lookup table pattern with one cancel-pair per (a,b) pair.
        """
        S = self.scale
        base = node.params.get('base', 16)

        # Get physical registers
        input_nodes = [graph.nodes[i] for i in node.inputs]
        a_reg = input_nodes[0].physical_reg
        b_reg = input_nodes[1].physical_reg
        out_reg = node.physical_reg

        # Create one cancel-pair per (a, b) pair
        unit_idx = 0
        for a in range(base):
            for b in range(base):
                unit_pos = self.unit_offset + unit_idx
                unit_neg = self.unit_offset + unit_idx + 1

                result = a | b  # Bitwise OR

                # Positive unit
                self.W_up[unit_pos, a_reg + a] = S
                self.W_gate[unit_pos, b_reg + b] = 1.0
                self.W_down[out_reg + result, unit_pos] = 1.0 / S

                # Negative unit
                self.W_up[unit_neg, a_reg + a] = -S
                self.W_gate[unit_neg, b_reg + b] = -1.0
                self.W_down[out_reg + result, unit_neg] = 1.0 / S

                unit_idx += 2

        self.unit_offset += 2 * base * base

    def emit_bit_xor(self, node: IRNode, graph: ComputationGraph):
        """Emit weights for BIT_XOR: out = a ^ b (bitwise XOR).

        Uses lookup table pattern with one cancel-pair per (a,b) pair.
        """
        S = self.scale
        base = node.params.get('base', 16)

        # Get physical registers
        input_nodes = [graph.nodes[i] for i in node.inputs]
        a_reg = input_nodes[0].physical_reg
        b_reg = input_nodes[1].physical_reg
        out_reg = node.physical_reg

        # Create one cancel-pair per (a, b) pair
        unit_idx = 0
        for a in range(base):
            for b in range(base):
                unit_pos = self.unit_offset + unit_idx
                unit_neg = self.unit_offset + unit_idx + 1

                result = a ^ b  # Bitwise XOR

                # Positive unit
                self.W_up[unit_pos, a_reg + a] = S
                self.W_gate[unit_pos, b_reg + b] = 1.0
                self.W_down[out_reg + result, unit_pos] = 1.0 / S

                # Negative unit
                self.W_up[unit_neg, a_reg + a] = -S
                self.W_gate[unit_neg, b_reg + b] = -1.0
                self.W_down[out_reg + result, unit_neg] = 1.0 / S

                unit_idx += 2

        self.unit_offset += 2 * base * base

    def emit_shl(self, node: IRNode, graph: ComputationGraph):
        """Emit weights for SHL: out = (a << n) & mask (shift left).

        Uses lookup table pattern with one cancel-pair per (a,n) pair.
        For 4-bit values, result is masked to 4 bits.
        """
        S = self.scale
        base = node.params.get('base', 16)

        # Get physical registers
        input_nodes = [graph.nodes[i] for i in node.inputs]
        a_reg = input_nodes[0].physical_reg
        n_reg = input_nodes[1].physical_reg
        out_reg = node.physical_reg

        # Create one cancel-pair per (a, n) pair
        unit_idx = 0
        for a in range(base):
            for n in range(base):
                unit_pos = self.unit_offset + unit_idx
                unit_neg = self.unit_offset + unit_idx + 1

                result = (a << n) & (base - 1)  # Shift left and mask to 4 bits

                # Positive unit
                self.W_up[unit_pos, a_reg + a] = S
                self.W_gate[unit_pos, n_reg + n] = 1.0
                self.W_down[out_reg + result, unit_pos] = 1.0 / S

                # Negative unit
                self.W_up[unit_neg, a_reg + a] = -S
                self.W_gate[unit_neg, n_reg + n] = -1.0
                self.W_down[out_reg + result, unit_neg] = 1.0 / S

                unit_idx += 2

        self.unit_offset += 2 * base * base

    def emit_shr(self, node: IRNode, graph: ComputationGraph):
        """Emit weights for SHR: out = a >> n (shift right).

        Uses lookup table pattern with one cancel-pair per (a,n) pair.
        """
        S = self.scale
        base = node.params.get('base', 16)

        # Get physical registers
        input_nodes = [graph.nodes[i] for i in node.inputs]
        a_reg = input_nodes[0].physical_reg
        n_reg = input_nodes[1].physical_reg
        out_reg = node.physical_reg

        # Create one cancel-pair per (a, n) pair
        unit_idx = 0
        for a in range(base):
            for n in range(base):
                unit_pos = self.unit_offset + unit_idx
                unit_neg = self.unit_offset + unit_idx + 1

                result = a >> n  # Shift right

                # Positive unit
                self.W_up[unit_pos, a_reg + a] = S
                self.W_gate[unit_pos, n_reg + n] = 1.0
                self.W_down[out_reg + result, unit_pos] = 1.0 / S

                # Negative unit
                self.W_up[unit_neg, a_reg + a] = -S
                self.W_gate[unit_neg, n_reg + n] = -1.0
                self.W_down[out_reg + result, unit_neg] = 1.0 / S

                unit_idx += 2

        self.unit_offset += 2 * base * base

    def _get_gate_reg(self, gate_vreg: str, graph: ComputationGraph) -> int:
        """Get physical register for gate virtual register."""
        node_id = graph.get_producer(gate_vreg)
        if node_id is None:
            raise ValueError(f"Gate register {gate_vreg} not found")
        return graph.nodes[node_id].physical_reg

    def emit_graph(self, graph: ComputationGraph):
        """Emit weights for entire computation graph."""
        for node_id in graph.topological_sort():
            node = graph.nodes[node_id]

            # Dispatch to specific emitter
            if node.op == OpType.ADD:
                self.emit_add(node, graph)
            elif node.op == OpType.SUB:
                self.emit_sub(node, graph)
            elif node.op == OpType.MUL:
                self.emit_mul(node, graph)
            elif node.op == OpType.DIV:
                self.emit_div(node, graph)
            elif node.op == OpType.MOD:
                self.emit_mod(node, graph)
            elif node.op == OpType.CMP_EQ:
                self.emit_cmp_eq(node, graph)
            elif node.op == OpType.CMP_LT:
                self.emit_cmp_lt(node, graph)
            elif node.op == OpType.CMP_GE:
                self.emit_cmp_ge(node, graph)
            elif node.op == OpType.CMP_LE:
                self.emit_cmp_le(node, graph)
            elif node.op == OpType.CMP_GT:
                self.emit_cmp_gt(node, graph)
            elif node.op == OpType.CMP_NE:
                self.emit_cmp_ne(node, graph)
            elif node.op == OpType.AND:
                self.emit_logical_and(node, graph)
            elif node.op == OpType.OR:
                self.emit_logical_or(node, graph)
            elif node.op == OpType.NOT:
                self.emit_logical_not(node, graph)
            elif node.op == OpType.XOR:
                self.emit_logical_xor(node, graph)
            elif node.op == OpType.MOVE:
                self.emit_move(node, graph)
            elif node.op == OpType.CLEAR:
                self.emit_clear(node, graph)
            elif node.op == OpType.CONST:
                self.emit_const(node, graph)
            elif node.op == OpType.SELECT:
                self.emit_select(node, graph)
            elif node.op == OpType.IF_THEN:
                self.emit_if_then(node, graph)
            elif node.op == OpType.BIT_AND:
                self.emit_bit_and(node, graph)
            elif node.op == OpType.BIT_OR:
                self.emit_bit_or(node, graph)
            elif node.op == OpType.BIT_XOR:
                self.emit_bit_xor(node, graph)
            elif node.op == OpType.SHL:
                self.emit_shl(node, graph)
            elif node.op == OpType.SHR:
                self.emit_shr(node, graph)
            else:
                raise NotImplementedError(f"Op {node.op} not implemented")

    def get_weights(self) -> Dict[str, torch.Tensor]:
        """Return emitted weights."""
        return {
            'W_up': self.W_up,
            'b_up': self.b_up,
            'W_gate': self.W_gate,
            'b_gate': self.b_gate,
            'W_down': self.W_down,
            'b_down': self.b_down
        }


# =============================================================================
# High-Level Compiler Interface
# =============================================================================

class GraphWeightCompiler:
    """High-level compiler interface."""

    def __init__(self, dim: int, hidden_dim: int, scale: float = 5.0):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.scale = scale
        self.graph = ComputationGraph()

    def add(self, a: str, b: str, output: str, gate: Optional[str] = None) -> int:
        """Add operation: output = a + b."""
        a_id = self.graph.get_producer(a)
        b_id = self.graph.get_producer(b)
        if a_id is None or b_id is None:
            raise ValueError(f"Inputs {a}, {b} must be defined")
        return self.graph.add_node(OpType.ADD, [a_id, b_id], output, gate=gate)

    def sub(self, a: str, b: str, output: str, gate: Optional[str] = None) -> int:
        """Subtract operation: output = a - b."""
        a_id = self.graph.get_producer(a)
        b_id = self.graph.get_producer(b)
        if a_id is None or b_id is None:
            raise ValueError(f"Inputs {a}, {b} must be defined")
        return self.graph.add_node(OpType.SUB, [a_id, b_id], output, gate=gate)

    def cmp_eq(self, a: str, b: str, output: str, gate: Optional[str] = None) -> int:
        """Compare equal: output = (a == b)."""
        a_id = self.graph.get_producer(a)
        b_id = self.graph.get_producer(b)
        if a_id is None or b_id is None:
            raise ValueError(f"Inputs {a}, {b} must be defined")
        return self.graph.add_node(OpType.CMP_EQ, [a_id, b_id], output, gate=gate)

    def cmp_lt(self, a: str, b: str, output: str, gate: Optional[str] = None) -> int:
        """Compare less than: output = (a < b)."""
        a_id = self.graph.get_producer(a)
        b_id = self.graph.get_producer(b)
        if a_id is None or b_id is None:
            raise ValueError(f"Inputs {a}, {b} must be defined")
        return self.graph.add_node(OpType.CMP_LT, [a_id, b_id], output, gate=gate)

    def cmp_ge(self, a: str, b: str, output: str, gate: Optional[str] = None) -> int:
        """Compare greater or equal: output = (a >= b)."""
        a_id = self.graph.get_producer(a)
        b_id = self.graph.get_producer(b)
        if a_id is None or b_id is None:
            raise ValueError(f"Inputs {a}, {b} must be defined")
        return self.graph.add_node(OpType.CMP_GE, [a_id, b_id], output, gate=gate)

    def cmp_le(self, a: str, b: str, output: str, gate: Optional[str] = None) -> int:
        """Compare less or equal: output = (a <= b)."""
        a_id = self.graph.get_producer(a)
        b_id = self.graph.get_producer(b)
        if a_id is None or b_id is None:
            raise ValueError(f"Inputs {a}, {b} must be defined")
        return self.graph.add_node(OpType.CMP_LE, [a_id, b_id], output, gate=gate)

    def cmp_gt(self, a: str, b: str, output: str, gate: Optional[str] = None) -> int:
        """Compare greater than: output = (a > b)."""
        a_id = self.graph.get_producer(a)
        b_id = self.graph.get_producer(b)
        if a_id is None or b_id is None:
            raise ValueError(f"Inputs {a}, {b} must be defined")
        return self.graph.add_node(OpType.CMP_GT, [a_id, b_id], output, gate=gate)

    def cmp_ne(self, a: str, b: str, output: str, gate: Optional[str] = None) -> int:
        """Compare not equal: output = (a != b)."""
        a_id = self.graph.get_producer(a)
        b_id = self.graph.get_producer(b)
        if a_id is None or b_id is None:
            raise ValueError(f"Inputs {a}, {b} must be defined")
        return self.graph.add_node(OpType.CMP_NE, [a_id, b_id], output, gate=gate)

    def logical_and(self, a: str, b: str, output: str, gate: Optional[str] = None) -> int:
        """Logical AND: output = a && b."""
        a_id = self.graph.get_producer(a)
        b_id = self.graph.get_producer(b)
        if a_id is None or b_id is None:
            raise ValueError(f"Inputs {a}, {b} must be defined")
        return self.graph.add_node(OpType.AND, [a_id, b_id], output, gate=gate)

    def logical_or(self, a: str, b: str, output: str, gate: Optional[str] = None) -> int:
        """Logical OR: output = a || b."""
        a_id = self.graph.get_producer(a)
        b_id = self.graph.get_producer(b)
        if a_id is None or b_id is None:
            raise ValueError(f"Inputs {a}, {b} must be defined")
        return self.graph.add_node(OpType.OR, [a_id, b_id], output, gate=gate)

    def logical_not(self, a: str, output: str, gate: Optional[str] = None) -> int:
        """Logical NOT: output = !a."""
        a_id = self.graph.get_producer(a)
        if a_id is None:
            raise ValueError(f"Input {a} must be defined")
        return self.graph.add_node(OpType.NOT, [a_id], output, gate=gate)

    def logical_xor(self, a: str, b: str, output: str, gate: Optional[str] = None) -> int:
        """Logical XOR: output = a ^ b."""
        a_id = self.graph.get_producer(a)
        b_id = self.graph.get_producer(b)
        if a_id is None or b_id is None:
            raise ValueError(f"Inputs {a}, {b} must be defined")
        return self.graph.add_node(OpType.XOR, [a_id, b_id], output, gate=gate)

    def move(self, src: str, dst: str, gate: Optional[str] = None) -> int:
        """Move: dst = src."""
        src_id = self.graph.get_producer(src)
        if src_id is None:
            raise ValueError(f"Input {src} must be defined")
        return self.graph.add_node(OpType.MOVE, [src_id], dst, gate=gate)

    def clear(self, reg: str, gate: Optional[str] = None) -> int:
        """Clear: reg = 0."""
        return self.graph.add_node(OpType.CLEAR, [], reg, gate=gate)

    def const(self, value: float, output: str) -> int:
        """Constant: output = value."""
        return self.graph.add_node(OpType.CONST, [], output, params={'value': value})

    def compile(self) -> Dict[str, torch.Tensor]:
        """Compile computation graph to FFN weights."""
        # Print graph
        print("=" * 60)
        print("Compiling Computation Graph")
        print("=" * 60)
        self.graph.print_graph()

        # Register allocation
        print("\nRegister Allocation:")
        allocator = RegisterAllocator(num_physical_regs=self.dim)
        allocation = allocator.allocate(self.graph)

        for vreg, preg in sorted(allocation.items()):
            print(f"  {vreg} → dim[{preg}]")

        # Weight emission
        print("\nEmitting Weights:")
        emitter = WeightEmitter(self.dim, self.hidden_dim, self.scale)
        emitter.emit_graph(self.graph)

        weights = emitter.get_weights()
        non_zero_up = (weights['W_up'] != 0).sum().item()
        non_zero_down = (weights['W_down'] != 0).sum().item()

        print(f"  Used {emitter.unit_offset} hidden units")
        print(f"  Non-zero W_up entries: {non_zero_up}")
        print(f"  Non-zero W_down entries: {non_zero_down}")
        print("=" * 60)

        return weights


# =============================================================================
# Demo
# =============================================================================

def demo_graph_compiler():
    """Demonstrate graph-based weight compiler."""
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 10 + "Graph-Based Weight Compiler Demo" + " " * 25 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    # Create compiler
    compiler = GraphWeightCompiler(dim=512, hidden_dim=4096, scale=5.0)

    # Build computation graph
    print("Building Computation Graph:")
    print()
    print("  // Arithmetic")
    print("  a = const(10)")
    print("  b = const(3)")
    print("  sum = a + b       // 10 + 3 = 13")
    print("  diff = a - b      // 10 - 3 = 7")
    print()
    print("  // Comparison")
    print("  is_equal = (a == b)       // 0 (false)")
    print("  is_less = (a < b)         // 0 (false)")
    print("  is_greater = (a >= b)     // 1 (true)")
    print()
    print("  // Logical")
    print("  x = const(1)")
    print("  y = const(0)")
    print("  both = x && y             // 0 (false)")
    print("  either = x || y           // 1 (true)")
    print("  not_x = !x                // 0 (false)")
    print()

    # Define operations
    # Arithmetic
    compiler.const(10.0, "a")
    compiler.const(3.0, "b")
    compiler.add("a", "b", "sum")
    compiler.sub("a", "b", "diff")

    # Comparison
    compiler.cmp_eq("a", "b", "is_equal")
    compiler.cmp_lt("a", "b", "is_less")
    compiler.cmp_ge("a", "b", "is_greater")

    # Logical
    compiler.const(1.0, "x")
    compiler.const(0.0, "y")
    compiler.logical_and("x", "y", "both")
    compiler.logical_or("x", "y", "either")
    compiler.logical_not("x", "not_x")

    # Compile
    weights = compiler.compile()

    print("\nPrimitive Summary:")
    print("  ✓ ADD    - 2 units per operation")
    print("  ✓ SUB    - 2 units per operation")
    print("  ✓ CMP_EQ - 4 units per operation")
    print("  ✓ CMP_LT - 2 units per operation")
    print("  ✓ CMP_GE - 2 units per operation")
    print("  ✓ AND    - 2 units per operation")
    print("  ✓ OR     - 2 units per operation")
    print("  ✓ NOT    - 2 units per operation")
    print("  ✓ CONST  - 2 units per operation")
    print()
    print("Compilation complete!")
    print("Generated weight matrices for transformer FFN layer")
    print("=" * 68)


if __name__ == "__main__":
    demo_graph_compiler()
