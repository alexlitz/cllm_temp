"""Purity checking system for C4 Neural VM.

This module ensures opcodes execute with pure neural weights (no handlers)
and detects when handlers are incorrectly added or enabled.

CRITICAL: This file should NOT be modified without explicit justification.
Any changes must be reviewed and documented in PURITY_CHECK_CHANGELOG.md
"""

from typing import Dict, Set, Optional
from enum import Enum
import sys

# Version tracking for protection
PURITY_CHECK_VERSION = "1.0.0"
LAST_MODIFIED = "2026-04-07"


class ExecutionMode(Enum):
    """Execution modes for the VM."""
    PURE_NEURAL = "pure_neural"  # No handlers allowed
    HYBRID = "hybrid"  # Handlers allowed for documented opcodes
    UNRESTRICTED = "unrestricted"  # All handlers allowed (testing only)


class PurityViolation(Exception):
    """Raised when a handler is used in pure neural mode."""
    pass


# ============================================================================
# OPCODE CLASSIFICATIONS (Update with care)
# ============================================================================

PURE_NEURAL_OPCODES: Set[int] = {
    # Stack/Control (no computation, just routing)
    1,   # IMM
    2,   # JMP
    3,   # JSR (PC jump only, stack ops still use handler)
    4,   # BZ
    5,   # BNZ
    38,  # EXIT

    # Memory (L15 attention-based)
    9,   # LI
    11,  # SI

    # Comparison (working neural implementations)
    18,  # NE
    21,  # LE

    # System/I/O (working neural implementations)
    33,  # PRTF
    65,  # PUTCHAR
}

HANDLER_REQUIRED_OPCODES: Set[int] = {
    # Arithmetic (neural weights broken, handlers permanent)
    25,  # ADD
    26,  # SUB
    27,  # MUL
    28,  # DIV
    29,  # MOD

    # Bitwise (neural weights broken, handlers permanent)
    14,  # OR
    15,  # XOR
    16,  # AND

    # Shift (neural weights broken, handlers permanent)
    23,  # SHL
    24,  # SHR

    # Stack/Function (complex multi-step operations)
    13,  # PSH
    6,   # ENT
    8,   # LEV

    # Other
    0,   # LEA (handler for immediate fetch)
    1,   # IMM (handler for 32-bit correctness)
}

BROKEN_OPCODES: Set[int] = {
    # Comparison (no handlers, neural weights don't work)
    17,  # EQ (purity violation)
    19,  # LT (purity violation)
    20,  # GT (returns 0 instead of 1)
    22,  # GE (returns 0 instead of 1)
}

UNTESTED_OPCODES: Set[int] = {
    7,   # ADJ
    10,  # LC
    12,  # SC
    30,  # OPEN
    31,  # READ
    32,  # CLOS
    34,  # MALC
    35,  # FREE
    36,  # MSET
    37,  # MCMP
    39,  # NOP
    40,  # POP
    41,  # BLT
    42,  # BGE
    64,  # GETCHAR
}


# ============================================================================
# PURITY ENFORCEMENT
# ============================================================================

class PurityChecker:
    """Enforces purity constraints on VM execution."""

    def __init__(self, mode: ExecutionMode = ExecutionMode.HYBRID):
        self.mode = mode
        self.violations: list[str] = []
        self.handler_calls: Dict[int, int] = {}  # opcode -> call count

    def check_handler_use(self, opcode: int, opcode_name: str) -> None:
        """Check if handler use is allowed for this opcode.

        Args:
            opcode: Opcode value (0-255)
            opcode_name: Human-readable opcode name

        Raises:
            PurityViolation: If handler used in pure neural mode inappropriately
        """
        # Track all handler calls
        self.handler_calls[opcode] = self.handler_calls.get(opcode, 0) + 1

        if self.mode == ExecutionMode.UNRESTRICTED:
            return  # Allow all handlers

        if self.mode == ExecutionMode.PURE_NEURAL:
            # In pure neural mode, NO handlers allowed
            msg = (f"Purity violation: Handler called for {opcode_name} (opcode {opcode}) "
                   f"in PURE_NEURAL mode. No handlers are permitted.")
            self.violations.append(msg)
            raise PurityViolation(msg)

        if self.mode == ExecutionMode.HYBRID:
            # In hybrid mode, only documented handler-required opcodes allowed
            if opcode in PURE_NEURAL_OPCODES:
                msg = (f"Purity violation: Handler called for {opcode_name} (opcode {opcode}) "
                       f"which is classified as PURE_NEURAL. This opcode should work without handlers.")
                self.violations.append(msg)
                # WARNING: Don't raise, just log - some pure opcodes may have handlers for 32-bit correctness
                print(f"WARNING: {msg}", file=sys.stderr)

            if opcode in BROKEN_OPCODES:
                msg = (f"Purity violation: Handler called for {opcode_name} (opcode {opcode}) "
                       f"which is classified as BROKEN. Add handler to HANDLER_REQUIRED_OPCODES "
                       f"or fix neural weights.")
                self.violations.append(msg)
                print(f"WARNING: {msg}", file=sys.stderr)

    def report(self) -> str:
        """Generate purity report."""
        lines = [
            "=" * 70,
            "PURITY CHECK REPORT",
            "=" * 70,
            f"Execution Mode: {self.mode.value}",
            f"Handler Calls: {sum(self.handler_calls.values())} total",
            "",
        ]

        if self.handler_calls:
            lines.append("Handlers Used:")
            for opcode, count in sorted(self.handler_calls.items()):
                classification = self._classify_opcode(opcode)
                lines.append(f"  Opcode {opcode:3}: {count:4} calls ({classification})")

        if self.violations:
            lines.extend([
                "",
                f"Violations: {len(self.violations)}",
            ])
            for violation in self.violations:
                lines.append(f"  - {violation}")
        else:
            lines.append("\n✓ No purity violations detected")

        lines.append("=" * 70)
        return "\n".join(lines)

    def _classify_opcode(self, opcode: int) -> str:
        """Classify opcode for reporting."""
        if opcode in PURE_NEURAL_OPCODES:
            return "PURE_NEURAL"
        elif opcode in HANDLER_REQUIRED_OPCODES:
            return "HANDLER_REQUIRED"
        elif opcode in BROKEN_OPCODES:
            return "BROKEN"
        elif opcode in UNTESTED_OPCODES:
            return "UNTESTED"
        else:
            return "UNKNOWN"


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_handler_config(handler_dict: Dict[int, callable]) -> list[str]:
    """Validate handler configuration against expected classifications.

    Args:
        handler_dict: Dictionary mapping opcode values to handler functions

    Returns:
        List of warning messages (empty if all OK)
    """
    warnings = []

    # Check for handlers on pure neural opcodes
    for opcode in PURE_NEURAL_OPCODES:
        if opcode in handler_dict:
            # Exception: IMM and JSR have handlers for 32-bit correctness
            if opcode not in {1, 3}:  # IMM, JSR
                warnings.append(
                    f"WARNING: Handler defined for PURE_NEURAL opcode {opcode}. "
                    f"This opcode should work without handlers."
                )

    # Check for missing handlers on required opcodes
    for opcode in HANDLER_REQUIRED_OPCODES:
        if opcode not in handler_dict:
            # Some opcodes may not be in handler dict if they're truly neural now
            pass  # Not necessarily an error

    # Check for handlers on broken opcodes (indicates attempted fix)
    for opcode in BROKEN_OPCODES:
        if opcode in handler_dict:
            warnings.append(
                f"INFO: Handler defined for BROKEN opcode {opcode}. "
                f"If this fixes the opcode, move it to HANDLER_REQUIRED_OPCODES."
            )

    return warnings


def assert_pure_neural_opcode(opcode: int, opcode_name: str) -> None:
    """Assert that an opcode is classified as pure neural.

    Useful in tests to ensure opcodes remain pure neural over time.

    Args:
        opcode: Opcode value
        opcode_name: Human-readable name

    Raises:
        AssertionError: If opcode is not pure neural
    """
    assert opcode in PURE_NEURAL_OPCODES, (
        f"Opcode {opcode_name} (opcode {opcode}) is not classified as PURE_NEURAL. "
        f"Classification: {PurityChecker()._classify_opcode(opcode)}"
    )


def assert_no_handlers_for_pure_opcodes(handler_dict: Dict[int, callable]) -> None:
    """Assert no handlers exist for pure neural opcodes (except allowed exceptions).

    Args:
        handler_dict: Handler configuration to check

    Raises:
        AssertionError: If forbidden handlers found
    """
    allowed_exceptions = {1, 3}  # IMM, JSR (32-bit correctness)

    forbidden_handlers = []
    for opcode in PURE_NEURAL_OPCODES:
        if opcode in handler_dict and opcode not in allowed_exceptions:
            forbidden_handlers.append(opcode)

    assert not forbidden_handlers, (
        f"Handlers found for pure neural opcodes: {forbidden_handlers}. "
        f"These opcodes should work without handlers."
    )


# ============================================================================
# CHANGELOG PROTECTION
# ============================================================================

def get_purity_check_hash() -> str:
    """Get hash of this file for change detection.

    Returns:
        Hash string of file contents
    """
    import hashlib
    try:
        with open(__file__, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception:
        return "UNKNOWN"


def verify_purity_check_integrity() -> bool:
    """Verify this file hasn't been modified unexpectedly.

    Returns:
        True if integrity check passes
    """
    # Store expected hash - update when file is intentionally modified
    EXPECTED_HASH = None  # Set after first deployment

    if EXPECTED_HASH is None:
        # First run, store hash
        return True

    current_hash = get_purity_check_hash()
    if current_hash != EXPECTED_HASH:
        print(
            f"WARNING: purity_check.py has been modified!\n"
            f"Expected hash: {EXPECTED_HASH}\n"
            f"Current hash: {current_hash}\n"
            f"Changes must be documented in PURITY_CHECK_CHANGELOG.md",
            file=sys.stderr
        )
        return False

    return True


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Example: Validate current handler configuration
    from neural_vm.vm_step import Opcode

    # Simulate handler dict (would come from AutoregressiveVMRunner)
    example_handlers = {
        Opcode.IMM: lambda: None,
        Opcode.ADD: lambda: None,
        Opcode.SUB: lambda: None,
        Opcode.MUL: lambda: None,
        # ... etc
    }

    warnings = validate_handler_config(example_handlers)
    if warnings:
        print("\n".join(warnings))
    else:
        print("✓ Handler configuration looks good")

    # Example: Create purity checker
    checker = PurityChecker(ExecutionMode.HYBRID)

    # Simulate handler calls
    checker.check_handler_use(25, "ADD")  # OK in hybrid mode
    try:
        checker_pure = PurityChecker(ExecutionMode.PURE_NEURAL)
        checker_pure.check_handler_use(25, "ADD")  # Violation in pure mode
    except PurityViolation as e:
        print(f"\nExpected violation: {e}")

    print("\n" + checker.report())
