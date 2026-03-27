"""
Weight Compiler: High-Level Abstractions for Neural VM Weight Setting

Transforms manual weight manipulation into declarative operation specs.

Example:
    # Instead of 40 lines of manual weight setting:
    W_up[0, ge.NIB_A] = S
    W_up[0, ge.NIB_B] = S
    ...

    # Write this:
    compiler.cancel_pair(["NIB_A", "NIB_B"], output="RAW_SUM", gate="OPCODE_ADD")
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from enum import Enum


# =============================================================================
# Core Data Structures
# =============================================================================

@dataclass
class Register:
    """Named dimension in embedding space."""
    name: str
    index: int  # Base dimension index
    width: int = 1  # Number of dimensions (1 for scalar, 16 for one-hot nibble)

    def __repr__(self):
        return f"Register({self.name}@{self.index}, width={self.width})"


class RegisterSet:
    """Collection of registers."""

    def __init__(self):
        self.registers: Dict[str, Register] = {}

    def add(self, name: str, index: int, width: int = 1) -> Register:
        """Add a register."""
        reg = Register(name, index, width)
        self.registers[name] = reg
        return reg

    def __getitem__(self, name: str) -> Register:
        """Get register by name."""
        return self.registers[name]

    def __contains__(self, name: str) -> bool:
        """Check if register exists."""
        return name in self.registers


@dataclass
class WeightValues:
    """Container for weight matrix values."""
    W_up: Optional[torch.Tensor] = None
    b_up: Optional[torch.Tensor] = None
    W_gate: Optional[torch.Tensor] = None
    b_gate: Optional[torch.Tensor] = None
    W_down: Optional[torch.Tensor] = None
    b_down: Optional[torch.Tensor] = None

    def merge(self, other: 'WeightValues'):
        """Merge another WeightValues into this one."""
        if other.W_up is not None:
            if self.W_up is None:
                self.W_up = other.W_up
            else:
                self.W_up += other.W_up

        if other.b_up is not None:
            if self.b_up is None:
                self.b_up = other.b_up
            else:
                self.b_up += other.b_up

        # Same for other matrices...


class ComputeMethod(Enum):
    """Available computation methods."""
    CANCEL_PAIR = "cancel_pair"
    STEP_PAIR = "step_pair"
    CLEAR_PAIR = "clear_pair"
    EQUALITY_STEP = "equality_step"
    MODULAR_ADD = "modular_add"
    RELAY = "relay"
    LOOKUP = "lookup"  # Not FFN-based, uses attention


@dataclass
class ComputeNode:
    """Intermediate representation of a computation."""
    layer: int
    output: str  # Output register name
    method: ComputeMethod
    inputs: List[str]  # Input register names
    gate: Optional[str] = None  # Gate register name
    params: Dict[str, Any] = field(default_factory=dict)  # Method-specific params

    def allocate_units(self) -> int:
        """Return number of hidden units needed."""
        method_units = {
            ComputeMethod.CANCEL_PAIR: 2,
            ComputeMethod.STEP_PAIR: 2,
            ComputeMethod.CLEAR_PAIR: 2,
            ComputeMethod.EQUALITY_STEP: 4,
            ComputeMethod.MODULAR_ADD: 6,
            ComputeMethod.LOOKUP: 0,  # Attention, not FFN
        }

        # RELAY depends on register width
        if self.method == ComputeMethod.RELAY:
            # Will be computed by compiler based on register width
            # Scalar: 2 units, Vector (width=N): 2*N units
            return 2  # Default for scalar

        return method_units.get(self.method, 1)


# =============================================================================
# Method Compilers
# =============================================================================

class MethodCompiler:
    """Base class for method-specific weight compilers."""

    def compile(self, node: ComputeNode, unit_offset: int,
                registers: RegisterSet, scale: float,
                dim: int, hidden_dim: int) -> WeightValues:
        """Compile node to weight values."""
        raise NotImplementedError


class CancelPairCompiler(MethodCompiler):
    """Compiles cancel pair: output = (+sum) + (-sum) for signed addition."""

    def compile(self, node: ComputeNode, unit_offset: int,
                registers: RegisterSet, S: float,
                dim: int, hidden_dim: int) -> WeightValues:
        weights = WeightValues()
        weights.W_up = torch.zeros(hidden_dim, dim)
        weights.b_up = torch.zeros(hidden_dim)
        weights.W_gate = torch.zeros(hidden_dim, dim)
        weights.b_gate = torch.zeros(hidden_dim)
        weights.W_down = torch.zeros(dim, hidden_dim)
        weights.b_down = torch.zeros(dim)

        output_reg = registers[node.output]
        gate_reg = registers[node.gate] if node.gate else None

        # Unit 0: Positive sum
        for inp_name in node.inputs:
            inp_reg = registers[inp_name]
            weights.W_up[unit_offset, inp_reg.index] = S

        if gate_reg:
            weights.W_gate[unit_offset, gate_reg.index] = S

        weights.W_down[output_reg.index, unit_offset] = 1.0 / (S * S)

        # Unit 1: Negative sum
        for inp_name in node.inputs:
            inp_reg = registers[inp_name]
            weights.W_up[unit_offset + 1, inp_reg.index] = -S

        if gate_reg:
            weights.W_gate[unit_offset + 1, gate_reg.index] = -S

        weights.W_down[output_reg.index, unit_offset + 1] = 1.0 / (S * S)

        return weights


class StepPairCompiler(MethodCompiler):
    """Compiles step pair: output = step(sum >= threshold)."""

    def compile(self, node: ComputeNode, unit_offset: int,
                registers: RegisterSet, S: float,
                dim: int, hidden_dim: int) -> WeightValues:
        weights = WeightValues()
        weights.W_up = torch.zeros(hidden_dim, dim)
        weights.b_up = torch.zeros(hidden_dim)
        weights.W_gate = torch.zeros(hidden_dim, dim)
        weights.b_gate = torch.zeros(hidden_dim)
        weights.W_down = torch.zeros(dim, hidden_dim)
        weights.b_down = torch.zeros(dim)

        threshold = node.params.get('threshold', 0.0)
        output_reg = registers[node.output]
        gate_reg = registers[node.gate] if node.gate else None

        # Unit 0: step(sum >= threshold)
        for inp_name in node.inputs:
            inp_reg = registers[inp_name]
            weights.W_up[unit_offset, inp_reg.index] = S

        weights.b_up[unit_offset] = -S * (threshold - 1.0)

        if gate_reg:
            weights.W_gate[unit_offset, gate_reg.index] = 1.0

        weights.W_down[output_reg.index, unit_offset] = 1.0 / S

        # Unit 1: -step(sum >= threshold+1)
        for inp_name in node.inputs:
            inp_reg = registers[inp_name]
            weights.W_up[unit_offset + 1, inp_reg.index] = S

        weights.b_up[unit_offset + 1] = -S * threshold

        if gate_reg:
            weights.W_gate[unit_offset + 1, gate_reg.index] = 1.0

        weights.W_down[output_reg.index, unit_offset + 1] = -1.0 / S

        return weights


class ClearPairCompiler(MethodCompiler):
    """Compiles clear pair: output = 0 (clears a register)."""

    def compile(self, node: ComputeNode, unit_offset: int,
                registers: RegisterSet, S: float,
                dim: int, hidden_dim: int) -> WeightValues:
        weights = WeightValues()
        weights.W_up = torch.zeros(hidden_dim, dim)
        weights.b_up = torch.zeros(hidden_dim)
        weights.W_gate = torch.zeros(hidden_dim, dim)
        weights.b_gate = torch.zeros(hidden_dim)
        weights.W_down = torch.zeros(dim, hidden_dim)
        weights.b_down = torch.zeros(dim)

        inp_reg = registers[node.inputs[0]]  # Register to clear
        gate_reg = registers[node.gate] if node.gate else None

        # Unit 0: +input
        weights.W_up[unit_offset, inp_reg.index] = S
        if gate_reg:
            weights.W_gate[unit_offset, gate_reg.index] = S
        weights.W_down[inp_reg.index, unit_offset] = -1.0 / S

        # Unit 1: -input
        weights.W_up[unit_offset + 1, inp_reg.index] = -S
        if gate_reg:
            weights.W_gate[unit_offset + 1, gate_reg.index] = -S
        weights.W_down[inp_reg.index, unit_offset + 1] = -1.0 / S

        return weights


class EqualityStepCompiler(MethodCompiler):
    """Compiles equality check: output = step(A == B)."""

    def compile(self, node: ComputeNode, unit_offset: int,
                registers: RegisterSet, S: float,
                dim: int, hidden_dim: int) -> WeightValues:
        weights = WeightValues()
        weights.W_up = torch.zeros(hidden_dim, dim)
        weights.b_up = torch.zeros(hidden_dim)
        weights.W_gate = torch.zeros(hidden_dim, dim)
        weights.b_gate = torch.zeros(hidden_dim)
        weights.W_down = torch.zeros(dim, hidden_dim)
        weights.b_down = torch.zeros(dim)

        if len(node.inputs) != 2:
            raise ValueError("EqualityStep requires exactly 2 inputs")

        inp_a = registers[node.inputs[0]]
        inp_b = registers[node.inputs[1]]
        output_reg = registers[node.output]
        gate_reg = registers[node.gate] if node.gate else None

        # Unit 0: step(A - B >= 0)
        weights.W_up[unit_offset, inp_a.index] = S
        weights.W_up[unit_offset, inp_b.index] = -S
        weights.b_up[unit_offset] = S

        if gate_reg:
            weights.W_gate[unit_offset, gate_reg.index] = 1.0

        weights.W_down[output_reg.index, unit_offset] = 1.0 / S

        # Unit 1: step(B - A >= 0)
        weights.W_up[unit_offset + 1, inp_a.index] = -S
        weights.W_up[unit_offset + 1, inp_b.index] = S
        weights.b_up[unit_offset + 1] = S

        if gate_reg:
            weights.W_gate[unit_offset + 1, gate_reg.index] = 1.0

        weights.W_down[output_reg.index, unit_offset + 1] = 1.0 / S

        # Unit 2: -step(A - B >= 1) [A > B]
        weights.W_up[unit_offset + 2, inp_a.index] = S
        weights.W_up[unit_offset + 2, inp_b.index] = -S
        weights.b_up[unit_offset + 2] = 0.0

        if gate_reg:
            weights.W_gate[unit_offset + 2, gate_reg.index] = 1.0

        weights.W_down[output_reg.index, unit_offset + 2] = -1.0 / S

        # Unit 3: -step(B - A >= 1) [B > A]
        weights.W_up[unit_offset + 3, inp_a.index] = -S
        weights.W_up[unit_offset + 3, inp_b.index] = S
        weights.b_up[unit_offset + 3] = 0.0

        if gate_reg:
            weights.W_gate[unit_offset + 3, gate_reg.index] = 1.0

        weights.W_down[output_reg.index, unit_offset + 3] = -1.0 / S

        return weights


class ModularAddCompiler(MethodCompiler):
    """Compiles modular addition: (A + B + C) mod base."""

    def compile(self, node: ComputeNode, unit_offset: int,
                registers: RegisterSet, S: float,
                dim: int, hidden_dim: int) -> WeightValues:
        weights = WeightValues()
        weights.W_up = torch.zeros(hidden_dim, dim)
        weights.b_up = torch.zeros(hidden_dim)
        weights.W_gate = torch.zeros(hidden_dim, dim)
        weights.b_gate = torch.zeros(hidden_dim)
        weights.W_down = torch.zeros(dim, hidden_dim)
        weights.b_down = torch.zeros(dim)

        base = node.params.get('base', 16)
        output_reg = registers[node.output]
        gate_reg = registers[node.gate] if node.gate else None

        # Units 0-1: raw_sum = A + B + C (cancel pair)
        for inp_name in node.inputs:
            inp_reg = registers[inp_name]
            weights.W_up[unit_offset, inp_reg.index] = S
            weights.W_up[unit_offset + 1, inp_reg.index] = -S

        if gate_reg:
            weights.W_gate[unit_offset, gate_reg.index] = S
            weights.W_gate[unit_offset + 1, gate_reg.index] = -S

        # Intermediate register for raw sum (assumed to exist)
        raw_sum_idx = output_reg.index - 1  # Convention: raw sum stored before output

        weights.W_down[raw_sum_idx, unit_offset] = 1.0 / (S * S)
        weights.W_down[raw_sum_idx, unit_offset + 1] = 1.0 / (S * S)

        # Units 2-3: output = raw_sum - base * step(raw_sum >= base)
        # First relay raw_sum
        weights.W_up[unit_offset + 2, raw_sum_idx] = S
        if gate_reg:
            weights.W_gate[unit_offset + 2, gate_reg.index] = 1.0
        weights.W_down[output_reg.index, unit_offset + 2] = 1.0 / S

        # Subtract base if carry
        for inp_name in node.inputs:
            inp_reg = registers[inp_name]
            weights.W_up[unit_offset + 3, inp_reg.index] = S

        weights.b_up[unit_offset + 3] = -S * (base - 1.0)
        if gate_reg:
            weights.W_gate[unit_offset + 3, gate_reg.index] = 1.0
        weights.W_down[output_reg.index, unit_offset + 3] = -base / S

        # Units 4-5: carry_out = step(A + B + C >= base)
        carry_out_idx = output_reg.index + 1  # Convention: carry stored after output

        for inp_name in node.inputs:
            inp_reg = registers[inp_name]
            weights.W_up[unit_offset + 4, inp_reg.index] = S
            weights.W_up[unit_offset + 5, inp_reg.index] = S

        weights.b_up[unit_offset + 4] = -S * (base - 1.0)
        weights.b_up[unit_offset + 5] = -S * base

        if gate_reg:
            weights.W_gate[unit_offset + 4, gate_reg.index] = 1.0
            weights.W_gate[unit_offset + 5, gate_reg.index] = 1.0

        weights.W_down[carry_out_idx, unit_offset + 4] = 1.0 / S
        weights.W_down[carry_out_idx, unit_offset + 5] = -1.0 / S

        return weights


class RelayCompiler(MethodCompiler):
    """Compiles register relay: output = input (with optional gating)."""

    def compile(self, node: ComputeNode, unit_offset: int,
                registers: RegisterSet, S: float,
                dim: int, hidden_dim: int) -> WeightValues:
        weights = WeightValues()
        weights.W_up = torch.zeros(hidden_dim, dim)
        weights.b_up = torch.zeros(hidden_dim)
        weights.W_gate = torch.zeros(hidden_dim, dim)
        weights.b_gate = torch.zeros(hidden_dim)
        weights.W_down = torch.zeros(dim, hidden_dim)
        weights.b_down = torch.zeros(dim)

        if len(node.inputs) != 1:
            raise ValueError("Relay requires exactly 1 input")

        inp_reg = registers[node.inputs[0]]
        output_reg = registers[node.output]
        gate_reg = registers[node.gate] if node.gate else None

        # Handle scalar vs vector registers
        if inp_reg.width == 1:
            # Scalar relay: use cancel pair for numerical stability
            # Unit 0: +input
            weights.W_up[unit_offset, inp_reg.index] = S
            if gate_reg:
                weights.W_gate[unit_offset, gate_reg.index] = S
            weights.W_down[output_reg.index, unit_offset] = 1.0 / S

            # Unit 1: -(-input)
            weights.W_up[unit_offset + 1, inp_reg.index] = -S
            if gate_reg:
                weights.W_gate[unit_offset + 1, gate_reg.index] = -S
            weights.W_down[output_reg.index, unit_offset + 1] = 1.0 / S

        else:
            # Vector relay (e.g., one-hot nibble): relay each dimension
            for i in range(inp_reg.width):
                # Each dimension gets a cancel pair
                u0 = unit_offset + 2 * i
                u1 = unit_offset + 2 * i + 1

                weights.W_up[u0, inp_reg.index + i] = S
                weights.W_up[u1, inp_reg.index + i] = -S

                if gate_reg:
                    weights.W_gate[u0, gate_reg.index] = S
                    weights.W_gate[u1, gate_reg.index] = -S

                weights.W_down[output_reg.index + i, u0] = 1.0 / S
                weights.W_down[output_reg.index + i, u1] = 1.0 / S

        return weights


# Method compiler registry
METHOD_COMPILERS = {
    ComputeMethod.CANCEL_PAIR: CancelPairCompiler(),
    ComputeMethod.STEP_PAIR: StepPairCompiler(),
    ComputeMethod.CLEAR_PAIR: ClearPairCompiler(),
    ComputeMethod.EQUALITY_STEP: EqualityStepCompiler(),
    ComputeMethod.MODULAR_ADD: ModularAddCompiler(),
    ComputeMethod.RELAY: RelayCompiler(),
}


# =============================================================================
# Operation Builder
# =============================================================================

class Operation:
    """High-level operation builder."""

    def __init__(self, name: str, compiler: 'WeightCompiler'):
        self.name = name
        self.compiler = compiler
        self.current_layer: Optional[int] = None
        self.nodes: List[ComputeNode] = []

    def layer(self, index: int):
        """Enter layer context."""
        return LayerContext(self, index)

    def cancel_pair(self, inputs: List[str], output: str,
                    gate: Optional[str] = None):
        """Add cancel pair computation: output = (+sum) + (-sum)."""
        node = ComputeNode(
            layer=self.current_layer,
            output=output,
            method=ComputeMethod.CANCEL_PAIR,
            inputs=inputs,
            gate=gate
        )
        self.nodes.append(node)

    def step(self, inputs: List[str], output: str, threshold: float,
             gate: Optional[str] = None):
        """Add step pair: output = step(sum >= threshold)."""
        node = ComputeNode(
            layer=self.current_layer,
            output=output,
            method=ComputeMethod.STEP_PAIR,
            inputs=inputs,
            gate=gate,
            params={'threshold': threshold}
        )
        self.nodes.append(node)

    def clear(self, register: str, gate: Optional[str] = None):
        """Add clear pair: register = 0."""
        node = ComputeNode(
            layer=self.current_layer,
            output=register,
            method=ComputeMethod.CLEAR_PAIR,
            inputs=[register],
            gate=gate
        )
        self.nodes.append(node)

    def equality_check(self, input_a: str, input_b: str, output: str,
                       gate: Optional[str] = None):
        """Add equality check: output = (input_a == input_b)."""
        node = ComputeNode(
            layer=self.current_layer,
            output=output,
            method=ComputeMethod.EQUALITY_STEP,
            inputs=[input_a, input_b],
            gate=gate
        )
        self.nodes.append(node)

    def modular_add(self, inputs: List[str], output: str, base: int = 16,
                    gate: Optional[str] = None):
        """Add modular addition: output = (sum of inputs) mod base."""
        node = ComputeNode(
            layer=self.current_layer,
            output=output,
            method=ComputeMethod.MODULAR_ADD,
            inputs=inputs,
            gate=gate,
            params={'base': base}
        )
        self.nodes.append(node)

    def relay(self, input_reg: str, output: str, gate: Optional[str] = None):
        """Add register relay: output = input."""
        node = ComputeNode(
            layer=self.current_layer,
            output=output,
            method=ComputeMethod.RELAY,
            inputs=[input_reg],
            gate=gate
        )
        self.nodes.append(node)


class LayerContext:
    """Context manager for layer operations."""

    def __init__(self, operation: Operation, layer_index: int):
        self.operation = operation
        self.layer_index = layer_index

    def __enter__(self):
        self.operation.current_layer = self.layer_index
        return self.operation

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.operation.current_layer = None


# =============================================================================
# Weight Compiler
# =============================================================================

class WeightCompiler:
    """Main weight compiler."""

    def __init__(self, dim: int, hidden_dim: int, scale: float = 5.0):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.scale = scale
        self.registers = RegisterSet()
        self.operations: Dict[str, Operation] = {}

    def operation(self, name: str) -> Operation:
        """Create or get an operation."""
        if name not in self.operations:
            self.operations[name] = Operation(name, self)
        return self.operations[name]

    def compile_operation(self, op: Operation) -> WeightValues:
        """Compile an operation to weight values."""
        # Group nodes by layer
        layers = {}
        for node in op.nodes:
            if node.layer not in layers:
                layers[node.layer] = []
            layers[node.layer].append(node)

        # Compile each layer
        all_weights = WeightValues()
        all_weights.W_up = torch.zeros(self.hidden_dim, self.dim)
        all_weights.b_up = torch.zeros(self.hidden_dim)
        all_weights.W_gate = torch.zeros(self.hidden_dim, self.dim)
        all_weights.b_gate = torch.zeros(self.hidden_dim)
        all_weights.W_down = torch.zeros(self.dim, self.hidden_dim)
        all_weights.b_down = torch.zeros(self.dim)

        unit_offset = 0

        for layer_idx in sorted(layers.keys()):
            layer_nodes = layers[layer_idx]

            for node in layer_nodes:
                # Get method compiler
                method_compiler = METHOD_COMPILERS[node.method]

                # Compile node
                weights = method_compiler.compile(
                    node, unit_offset, self.registers,
                    self.scale, self.dim, self.hidden_dim
                )

                # Merge weights
                all_weights.W_up += weights.W_up
                all_weights.b_up += weights.b_up
                all_weights.W_gate += weights.W_gate
                all_weights.b_gate += weights.b_gate
                all_weights.W_down += weights.W_down
                all_weights.b_down += weights.b_down

                # Advance unit offset
                unit_offset += node.allocate_units()

        return all_weights


# =============================================================================
# Demo
# =============================================================================

def demo_weight_compiler():
    """Demonstrate weight compiler usage."""
    print("=" * 70)
    print("Weight Compiler Demo - All Method Compilers")
    print("=" * 70)

    # Create compiler
    compiler = WeightCompiler(dim=512, hidden_dim=4096, scale=5.0)

    # Define registers
    compiler.registers.add("NIB_A", 0, width=16)
    compiler.registers.add("NIB_B", 16, width=16)
    compiler.registers.add("NIB_C", 32, width=16)
    compiler.registers.add("RAW_SUM", 100, width=1)
    compiler.registers.add("CARRY_OUT", 101, width=1)
    compiler.registers.add("CARRY_IN", 102, width=1)
    compiler.registers.add("IS_EQUAL", 103, width=1)
    compiler.registers.add("RESULT", 110, width=16)
    compiler.registers.add("TEMP", 120, width=1)
    compiler.registers.add("OPCODE_ADD", 200, width=1)
    compiler.registers.add("OPCODE_CMP", 201, width=1)

    print("\nRegisters:")
    for name, reg in compiler.registers.registers.items():
        print(f"  {name}: index={reg.index}, width={reg.width}")

    print("\n" + "=" * 70)
    print("DEMO 1: Basic ADD Operation (cancel_pair + step)")
    print("=" * 70)

    # Define ADD operation
    add = compiler.operation("ADD")
    with add.layer(1):
        add.cancel_pair(["NIB_A", "NIB_B"], output="RAW_SUM", gate="OPCODE_ADD")
        add.step(["NIB_A", "NIB_B"], output="CARRY_OUT",
                threshold=16.0, gate="OPCODE_ADD")

    print(f"Created {len(add.nodes)} computation nodes")
    weights_add = compiler.compile_operation(add)
    print(f"  W_up shape: {weights_add.W_up.shape}")
    print(f"  Non-zero W_up entries: {(weights_add.W_up != 0).sum().item()}")
    print(f"  Non-zero W_down entries: {(weights_add.W_down != 0).sum().item()}")

    print("\n" + "=" * 70)
    print("DEMO 2: Equality Check (equality_check)")
    print("=" * 70)

    # Define CMP operation
    cmp = compiler.operation("CMP")
    with cmp.layer(1):
        cmp.equality_check("NIB_A", "NIB_B", output="IS_EQUAL", gate="OPCODE_CMP")

    print(f"Created {len(cmp.nodes)} computation nodes")
    weights_cmp = compiler.compile_operation(cmp)
    print(f"  Uses 4 hidden units for equality check")
    print(f"  Non-zero W_up entries: {(weights_cmp.W_up != 0).sum().item()}")
    print(f"  Non-zero W_down entries: {(weights_cmp.W_down != 0).sum().item()}")

    print("\n" + "=" * 70)
    print("DEMO 3: Modular Addition (modular_add)")
    print("=" * 70)

    # Define MODULAR_ADD operation
    mod_add = compiler.operation("MODULAR_ADD")
    with mod_add.layer(1):
        mod_add.modular_add(["NIB_A", "NIB_B", "CARRY_IN"],
                           output="RESULT", base=16, gate="OPCODE_ADD")

    print(f"Created {len(mod_add.nodes)} computation nodes")
    print(f"  Computes: (NIB_A + NIB_B + CARRY_IN) mod 16")
    print(f"  Also produces carry_out flag")

    weights_mod = compiler.compile_operation(mod_add)
    print(f"  Uses 6 hidden units")
    print(f"  Non-zero W_up entries: {(weights_mod.W_up != 0).sum().item()}")

    print("\n" + "=" * 70)
    print("DEMO 4: Register Relay (relay)")
    print("=" * 70)

    # Define RELAY operation
    relay = compiler.operation("RELAY")
    with relay.layer(1):
        relay.relay("RAW_SUM", output="TEMP", gate="OPCODE_ADD")

    print(f"Created {len(relay.nodes)} computation nodes")
    weights_relay = compiler.compile_operation(relay)
    print(f"  Scalar relay uses 2 hidden units")
    print(f"  Non-zero W_up entries: {(weights_relay.W_up != 0).sum().item()}")

    print("\n" + "=" * 70)
    print("DEMO 5: Clear Register (clear)")
    print("=" * 70)

    # Define CLEAR operation
    clear_op = compiler.operation("CLEAR")
    with clear_op.layer(1):
        clear_op.clear("TEMP", gate="OPCODE_ADD")

    print(f"Created {len(clear_op.nodes)} computation nodes")
    weights_clear = compiler.compile_operation(clear_op)
    print(f"  Uses 2 hidden units (cancel pair that zeros output)")
    print(f"  Non-zero W_up entries: {(weights_clear.W_up != 0).sum().item()}")

    print("\n" + "=" * 70)
    print("DEMO 6: Multi-Layer Operation")
    print("=" * 70)

    # Define complex multi-layer operation
    complex_op = compiler.operation("COMPLEX")

    with complex_op.layer(1):
        print("  Layer 1: Compute raw sum")
        complex_op.cancel_pair(["NIB_A", "NIB_B"], output="RAW_SUM", gate="OPCODE_ADD")

    with complex_op.layer(2):
        print("  Layer 2: Check if sum >= 16")
        complex_op.step(["RAW_SUM"], output="CARRY_OUT", threshold=16.0, gate="OPCODE_ADD")

    with complex_op.layer(3):
        print("  Layer 3: Relay to result")
        complex_op.relay("RAW_SUM", output="TEMP", gate="OPCODE_ADD")

    print(f"Created {len(complex_op.nodes)} computation nodes across 3 layers")
    weights_complex = compiler.compile_operation(complex_op)
    print(f"  Total non-zero W_up entries: {(weights_complex.W_up != 0).sum().item()}")
    print(f"  Total non-zero W_down entries: {(weights_complex.W_down != 0).sum().item()}")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("\nAvailable method compilers:")
    print("  ✓ cancel_pair    - Signed arithmetic (A + B)")
    print("  ✓ step           - Threshold detection (step(x >= t))")
    print("  ✓ clear          - Zero register")
    print("  ✓ equality_check - Equality test (A == B)")
    print("  ✓ modular_add    - Modular arithmetic (A + B) mod N")
    print("  ✓ relay          - Copy register (output = input)")
    print("\nCode reduction example:")
    print("  Manual weight setting: ~40 lines per operation")
    print("  High-level API:        ~4 lines per operation")
    print("  Reduction:             10x fewer lines")

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    demo_weight_compiler()
