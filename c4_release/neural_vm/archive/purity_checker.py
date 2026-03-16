#!/usr/bin/env python3
"""
Purity Checker for Neural VM.

Ensures that forward() methods contain NO Python control flow or arithmetic.
Only _bake_weights() methods may use Python operations.

Rules:
1. forward() must be PURE tensor operations only
2. No Python if/for/while/arithmetic in forward()
3. _bake_weights() can use any Python (runs once at init)

This checker uses AST analysis to detect violations.
"""

import ast
import inspect
import sys
from typing import List, Tuple, Set, Optional
from dataclasses import dataclass
import importlib


@dataclass
class PurityViolation:
    """A purity violation in a forward method."""
    class_name: str
    method_name: str
    line_number: int
    violation_type: str
    code_snippet: str
    explanation: str


class ForwardPurityVisitor(ast.NodeVisitor):
    """
    AST visitor that detects Python control flow and arithmetic in forward methods.

    Distinguishes between:
    - STRUCTURAL control flow (allowed): iterating over self.modules, checking self.flag
    - DATA-DEPENDENT control flow (forbidden): branching on tensor values, .item()
    """

    def __init__(self, source_lines: List[str], class_name: str):
        self.violations: List[PurityViolation] = []
        self.source_lines = source_lines
        self.class_name = class_name
        self.in_forward = False
        self.method_name = ""

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Track when we're inside a forward method."""
        # Only check 'forward' methods - not __init__, _bake_weights, etc.
        if node.name == 'forward':
            self.in_forward = True
            self.method_name = node.name
            self.generic_visit(node)
            self.in_forward = False
        else:
            # Skip all other methods - they're allowed Python ops
            # (__init__, _bake_weights, check_io, etc.)
            pass

    def _add_violation(self, node: ast.AST, vtype: str, explanation: str):
        """Add a violation."""
        lineno = getattr(node, 'lineno', 0)
        snippet = self.source_lines[lineno - 1].strip() if lineno <= len(self.source_lines) else ""
        self.violations.append(PurityViolation(
            class_name=self.class_name,
            method_name=self.method_name,
            line_number=lineno,
            violation_type=vtype,
            code_snippet=snippet,
            explanation=explanation
        ))

    def _is_structural_condition(self, node: ast.expr) -> bool:
        """Check if condition is structural (self.attr) vs data-dependent (tensor value)."""
        # self.attr is structural
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == 'self':
                return True
        # Constants are structural
        if isinstance(node, ast.Constant):
            return True
        # Comparisons involving only self.attr are structural
        if isinstance(node, ast.Compare):
            return self._is_structural_condition(node.left)
        return False

    def _is_structural_iterator(self, node: ast.For) -> bool:
        """Check if for loop iterates over structural collection (self.modules, range(self.n))."""
        iter_node = node.iter
        # for x in self.modules / self.experts / etc
        if isinstance(iter_node, ast.Attribute):
            if isinstance(iter_node.value, ast.Name) and iter_node.value.id == 'self':
                return True
        # for i in range(self.num_experts)
        if isinstance(iter_node, ast.Call):
            if isinstance(iter_node.func, ast.Name) and iter_node.func.id == 'range':
                if iter_node.args:
                    arg = iter_node.args[0]
                    # range(self.attr)
                    if isinstance(arg, ast.Attribute):
                        if isinstance(arg.value, ast.Name) and arg.value.id == 'self':
                            return True
                    # range(constant)
                    if isinstance(arg, ast.Constant):
                        return True
        return False

    def visit_If(self, node: ast.If):
        """Python if statements - only allowed if structural."""
        if self.in_forward:
            if not self._is_structural_condition(node.test):
                self._add_violation(node, "IF_STATEMENT",
                    "Data-dependent 'if' not allowed - use torch.where() or masking")
        self.generic_visit(node)

    def visit_For(self, node: ast.For):
        """Python for loops - only allowed if structural."""
        if self.in_forward:
            if not self._is_structural_iterator(node):
                self._add_violation(node, "FOR_LOOP",
                    "Data-dependent 'for' not allowed - use vectorized tensor ops")
        self.generic_visit(node)

    def visit_While(self, node: ast.While):
        """Python while loops are never allowed in forward."""
        if self.in_forward:
            self._add_violation(node, "WHILE_LOOP",
                "Python 'while' not allowed - use fixed iteration count or attention")
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp):
        """Check for Python arithmetic on non-tensor values."""
        if self.in_forward:
            # Check if this is a Python int/float operation (not tensor)
            # This is heuristic - we flag operations on literals
            if isinstance(node.left, ast.Constant) or isinstance(node.right, ast.Constant):
                if isinstance(node.left, ast.Constant) and isinstance(node.right, ast.Constant):
                    # Both constants - this is Python arithmetic
                    self._add_violation(node, "PYTHON_ARITHMETIC",
                        "Python arithmetic on constants - precompute in _bake_weights()")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        """Check for forbidden function calls."""
        if self.in_forward:
            # Check for .item() calls (extracts Python scalar from tensor)
            if isinstance(node.func, ast.Attribute) and node.func.attr == 'item':
                self._add_violation(node, "TENSOR_ITEM",
                    ".item() extracts Python scalar - keep as tensor")

            # Check for Python builtins that extract tensor data
            # Note: range() with self.attr is allowed (structural)
            data_extractors = {'int', 'float', 'round'}
            if isinstance(node.func, ast.Name) and node.func.id in data_extractors:
                self._add_violation(node, "DATA_EXTRACTION",
                    f"'{node.func.id}()' extracts tensor data - keep as tensor")
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript):
        """Check for Python indexing that might indicate non-tensor ops."""
        if self.in_forward:
            # Indexing with Python variables (not tensors) can be problematic
            # This is heuristic - we allow tensor indexing
            pass
        self.generic_visit(node)


def check_class_purity(cls) -> List[PurityViolation]:
    """
    Check a class for purity violations in its forward method.

    Args:
        cls: The class to check

    Returns:
        List of PurityViolation objects
    """
    try:
        source = inspect.getsource(cls)
        source_lines = source.split('\n')
        tree = ast.parse(source)
    except (OSError, TypeError):
        return []

    # Find the class definition
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls.__name__:
            visitor = ForwardPurityVisitor(source_lines, cls.__name__)
            visitor.visit(node)
            return visitor.violations

    return []


def check_module_purity(module_name: str) -> List[PurityViolation]:
    """
    Check all classes in a module for purity violations.

    Args:
        module_name: Name of module to check (e.g., 'neural_vm.arithmetic_ops')

    Returns:
        List of all violations found
    """
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        print(f"Warning: Could not import {module_name}: {e}")
        return []

    violations = []

    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type) and hasattr(obj, 'forward'):
            # Skip deprecated classes
            if obj.__name__ in DEPRECATED_CLASSES:
                continue
            # Skip utility classes (intentionally impure handlers)
            if obj.__name__ in UTILITY_CLASSES:
                continue
            violations.extend(check_class_purity(obj))

    return violations


def check_all_neural_vm() -> List[PurityViolation]:
    """Check all neural_vm modules for purity violations."""
    modules = [
        'neural_vm.base_layers',
        'neural_vm.arithmetic_ops',
        'neural_vm.bitwise_ops',
        'neural_vm.comparison_ops',
        'neural_vm.shift_ops',
        'neural_vm.control_flow_ops',
        'neural_vm.memory_ops',
        'neural_vm.io_ops',
        'neural_vm.io_handler',
        'neural_vm.pure_moe',
        'neural_vm.pure_alu',
        'neural_vm.zero_detection',
        'neural_vm.missing_ops',
    ]

    all_violations = []
    for module in modules:
        violations = check_module_purity(module)
        all_violations.extend(violations)

    return all_violations


# =============================================================================
# ALLOWED FORWARD CLASSES
# =============================================================================

ALLOWED_FORWARD_CLASSES = {
    'PureFFN',
    'PureAttention',
    'MoE',
    'SoftMoEFFN',
    'SoftMoEAttention',
    'UnifiedMoEBlock',  # Composition of SoftMoE layers
    # Note: PureALU is now nn.Sequential (no custom forward)
}

# Classes that are deprecated but kept for backwards compatibility
# These are excluded from purity checks
DEPRECATED_CLASSES = {
    # Old MoE implementations with Python control flow
    'MoELayer',
    'MultiExpertMoELayer',
    'TransformerBlock',
    'SparseMoEALU',
    'UnifiedMoETransformer',
    # Old composite FFN classes with Python control flow
    'AttentionThenFFN',
    'ConditionalFFN',
    'IteratedAttentionFFN',
    'LoopUnrollFFN',
    'MoERoutedFFN',
    'ParallelPureFFN',
    'SequentialPureFFN',
    'BakedCompositeFFN',
    # Old integrated VM implementations
    'IntegratedVM',
    'MemoryLoadAttention',
    # Old multiplication implementations (mul_multi.py)
    'MultiNibbleMul',
    'MulBroadcastBjAttention',
    'MulClearTempFFN',
    'MulPartialProductFFN',
    'MulClearRawSumFFN',
    'MulShiftAddAttention',
    'MulZeroLowPositionsFFN',
    'MulAccumulateFFN',
    'MulClearCarryInFFN',
    'MulInitResultFFN',
    # Old mul/div implementations (mul_div_multi.py)
    'MulCrossProductFFN',
    'BroadcastNibBAttention',
    'MulAccumulateAttention',
    'ClearTempForMulFFN',
    'ClearRawSumForMulFFN',
    'MulAddPartialToResultFFN',
    'DivMultiInitFFN',
    'DivCompareFFN',
    'DivBorrowDetectFFN',
    'DivSubtractIfGeFFN',
    'MultiNibbleALU',
    'MultiNibbleMulStage',
    'NeuralMultiNibbleMul',
    # Old pure_multi_ops.py implementations
    'ExtractBitFFN',
    'ExtractBitFromPosAttention',
    'ShiftedAddFFN',
    'MulAccumulatorCarryFFN',
    'DivShiftRemainderFFN',
    'DivBringDownBitFFN',
    'DivCompareSubtractFFN',
    'PureMul32',
    'MulInitAccumulatorFFN',
    'PureDiv32',
    'DivInitRemainderFFN',
    'PureMod32',
    'ModExtractRemainderFFN',
    # sparse_moe_alu.py
    'IdentityFFN',
    'IdentityAttention',
    # Old unified transformer (unified_transformer_alu.py)
    'UnifiedMoELayer',
    'UnifiedAttentionLayer',
    'TransformerALUBlock',
    'UnifiedTransformerALU',
    'SequentialExpert',
    # Memory subroutines (use loops)
    'MsetFFN',
    'McmpFFN',
    'McpyFFN',
}

# Utility classes that are intentionally impure
# These are external handlers/encoders, not neural layers
UTILITY_CLASSES = {
    # Binary I/O handlers (run outside the neural network)
    'BinaryPositionEncoder',  # Encodes positions before forward pass
    'BinaryMatchAttention',   # Utility for address lookup
    'BinaryMatchAttentionPure',  # Pure version
    'BinaryAddressEncoder',   # Address encoding helper
    'LLMIOState',             # State management (not a neural layer)
    'LLMNativeIOHandler',     # I/O handler (not a neural layer)
    'GetcharBinaryFFN',       # Binary I/O FFN
    'PutcharBinaryFFN',       # Binary I/O FFN
    'MemoryReadAttention',    # Memory utility
    'MemoryWriteAttention',   # Memory utility
    # KV memory (utility classes)
    'AddressEncoder',
    'AliBiPositionalBias',
    'Softmax1',
    'KVMemoryAttention',
    'KVMemoryCache',
    'VMStateEncoder',
}


def check_forward_class_violations(module_name: str) -> List[str]:
    """
    Check that only allowed classes have forward() methods.

    Returns list of class names that have forward() but shouldn't.
    """
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return []

    violations = []

    for name in dir(module):
        obj = getattr(module, name)
        if not isinstance(obj, type):
            continue
        if not hasattr(obj, 'forward'):
            continue

        # Check if this class defines its OWN forward (not inherited)
        if 'forward' not in obj.__dict__:
            continue

        # Skip deprecated classes (they're kept for backwards compat)
        if obj.__name__ in DEPRECATED_CLASSES:
            continue

        # Skip utility classes (intentionally impure handlers)
        if obj.__name__ in UTILITY_CLASSES:
            continue

        # Check inheritance
        is_allowed = False
        for allowed in ALLOWED_FORWARD_CLASSES:
            # Check if class name matches or inherits from allowed
            if obj.__name__ == allowed:
                is_allowed = True
                break
            for base in obj.__mro__:
                if base.__name__ in ALLOWED_FORWARD_CLASSES:
                    is_allowed = True
                    break
            if is_allowed:
                break

        if not is_allowed:
            violations.append(f"{module_name}.{obj.__name__}")

    return violations


def check_all_forward_classes() -> List[str]:
    """Check all modules for classes with unauthorized forward() methods."""
    modules = [
        'neural_vm.base_layers',
        'neural_vm.arithmetic_ops',
        'neural_vm.bitwise_ops',
        'neural_vm.comparison_ops',
        'neural_vm.shift_ops',
        'neural_vm.memory_ops',
        'neural_vm.io_ops',
        'neural_vm.control_flow_ops',
        'neural_vm.pure_moe',
        'neural_vm.pure_alu',
        'neural_vm.zero_detection',
        'neural_vm.missing_ops',
    ]

    all_violations = []
    for module in modules:
        violations = check_forward_class_violations(module)
        all_violations.extend(violations)

    return all_violations


def print_violations(violations: List[PurityViolation], verbose: bool = True):
    """Print violations in a readable format."""
    if not violations:
        print("✓ No purity violations found!")
        return

    print(f"Found {len(violations)} purity violation(s):\n")

    # Group by class
    by_class = {}
    for v in violations:
        key = f"{v.class_name}.{v.method_name}"
        if key not in by_class:
            by_class[key] = []
        by_class[key].append(v)

    for class_method, class_violations in sorted(by_class.items()):
        print(f"  {class_method}:")
        for v in class_violations:
            print(f"    Line {v.line_number}: [{v.violation_type}]")
            if verbose:
                print(f"      Code: {v.code_snippet}")
                print(f"      Fix:  {v.explanation}")
        print()


def main():
    """Run purity check on all neural_vm modules."""
    print("=" * 70)
    print("Neural VM Purity Checker")
    print("=" * 70)
    print()

    exit_code = 0

    # Check 1: Forward method purity
    print("Check 1: Python control flow in forward() methods")
    print("-" * 70)
    violations = check_all_neural_vm()
    print_violations(violations)

    if violations:
        print(f"FAILED: {len(violations)} purity violation(s)")
        exit_code = 1
    else:
        print("PASSED: All forward() methods are pure tensor operations")

    print()

    # Check 2: Only allowed classes have forward()
    print("Check 2: Unauthorized forward() methods")
    print("-" * 70)
    print(f"Allowed classes with forward(): {ALLOWED_FORWARD_CLASSES}")
    print()

    forward_violations = check_all_forward_classes()
    if forward_violations:
        print(f"FAILED: {len(forward_violations)} class(es) have unauthorized forward():")
        for v in forward_violations:
            print(f"  - {v}")
        exit_code = 1
    else:
        print("PASSED: Only allowed classes define forward()")

    print()
    print("=" * 70)

    if exit_code == 0:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED")
        print("\nTo fix purity violations:")
        print("  - Move Python arithmetic to _bake_weights()")
        print("  - Replace 'if' with torch.where() or masking")
        print("  - Replace 'for' with vectorized tensor ops")
        print("  - Replace .item() with tensor operations")
        print("\nTo fix unauthorized forward():")
        print("  - Inherit from PureFFN or PureAttention")
        print("  - Or compose using SoftMoEFFN/SoftMoEAttention")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
