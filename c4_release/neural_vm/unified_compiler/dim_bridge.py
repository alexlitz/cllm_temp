"""
Dimension Bridge - Connects auto-allocator to vm_step.py.

Provides a drop-in replacement for _SetDim that uses the auto-allocator
for dimension lookups while maintaining backward compatibility.
"""

from typing import Dict, Optional
from .auto_allocator import create_standard_allocator, AutoAllocator


class AllocatorBackedDim:
    """Drop-in replacement for _SetDim using auto-allocator.

    Usage:
        # In vm_step.py, replace:
        #   BD = _SetDim
        # with:
        #   from neural_vm.unified_compiler.dim_bridge import get_dim_class
        #   BD = get_dim_class()

    All dimension accesses (e.g., BD.MARK_PC, BD.ALU_LO) work the same way,
    but values come from the auto-allocator instead of hardcoded constants.
    """

    _alloc: Optional[AutoAllocator] = None
    _positions: Optional[Dict[str, int]] = None

    @classmethod
    def _ensure_allocated(cls):
        if cls._alloc is None:
            cls._alloc = create_standard_allocator(pinned=True)
            cls._positions = cls._alloc.allocate()

    def __class_getitem__(cls, name: str) -> int:
        """Support BD["MARK_PC"] style access."""
        cls._ensure_allocated()
        if name in cls._positions:
            return cls._positions[name]
        raise KeyError(f"Unknown dimension: {name}")

    @classmethod
    def get(cls, name: str) -> int:
        """Get dimension position by name."""
        cls._ensure_allocated()
        return cls._alloc.get(name)

    @classmethod
    def get_range(cls, name: str) -> range:
        """Get dimension range by name."""
        cls._ensure_allocated()
        return cls._alloc.get_range(name)


def _create_dim_class() -> type:
    """Create a dimension class with attributes from auto-allocator.

    Returns a class that can be used as a drop-in replacement for _SetDim.
    All known dimensions become class attributes.
    """
    alloc = create_standard_allocator(pinned=True)
    positions = alloc.allocate()

    # Create class attributes for all dimensions
    attrs = dict(positions)

    # Add convenience constants that _SetDim has
    attrs['MARKS'] = [
        positions['MARK_PC'], positions['MARK_AX'], positions['MARK_SP'],
        positions['MARK_BP'], positions['MARK_MEM'], positions['MARK_SE'],
        positions['MARK_CS']
    ]
    attrs['NUM_MARKERS'] = 7
    attrs['OPCODE_BASE'] = positions.get('OP_LEA', 262)
    attrs['NUM_OPCODES'] = 34

    # Add opcode_dim method
    def opcode_dim(opcode_value: int) -> int:
        """Return dimension for opcode flag given opcode int value."""
        from ..embedding import Opcode
        opcode_names = {
            Opcode.LEA: 'OP_LEA', Opcode.IMM: 'OP_IMM', Opcode.JMP: 'OP_JMP',
            Opcode.JSR: 'OP_JSR', Opcode.BZ: 'OP_BZ', Opcode.BNZ: 'OP_BNZ',
            Opcode.ENT: 'OP_ENT', Opcode.ADJ: 'OP_ADJ', Opcode.LEV: 'OP_LEV',
            Opcode.LI: 'OP_LI', Opcode.LC: 'OP_LC', Opcode.SI: 'OP_SI',
            Opcode.SC: 'OP_SC', Opcode.PSH: 'OP_PSH', Opcode.OR: 'OP_OR',
            Opcode.XOR: 'OP_XOR', Opcode.AND: 'OP_AND', Opcode.EQ: 'OP_EQ',
            Opcode.NE: 'OP_NE', Opcode.LT: 'OP_LT', Opcode.GT: 'OP_GT',
            Opcode.LE: 'OP_LE', Opcode.GE: 'OP_GE', Opcode.SHL: 'OP_SHL',
            Opcode.SHR: 'OP_SHR', Opcode.ADD: 'OP_ADD', Opcode.SUB: 'OP_SUB',
            Opcode.MUL: 'OP_MUL', Opcode.DIV: 'OP_DIV', Opcode.MOD: 'OP_MOD',
            Opcode.EXIT: 'OP_EXIT',
        }
        name = opcode_names.get(opcode_value)
        if name and name in positions:
            return positions[name]
        # Fall back to offset-based calculation
        return positions.get('OP_LEA', 262) + (opcode_value - Opcode.LEA)

    attrs['opcode_dim'] = staticmethod(opcode_dim)

    # Add helper methods
    def get(name: str) -> int:
        return positions[name]

    def get_range(name: str) -> range:
        return alloc.get_range(name)

    attrs['get'] = staticmethod(get)
    attrs['get_range'] = staticmethod(get_range)
    attrs['_alloc'] = alloc
    attrs['_positions'] = positions

    return type('AllocDim', (), attrs)


# Singleton instance
_dim_class = None

def get_dim_class() -> type:
    """Get the allocator-backed dimension class (singleton)."""
    global _dim_class
    if _dim_class is None:
        _dim_class = _create_dim_class()
    return _dim_class


def verify_compatibility():
    """Verify that allocator dimensions match _SetDim hardcoded values.

    Returns (matches, mismatches) tuple.
    """
    from ..vm_step import _SetDim

    AllocDim = get_dim_class()
    matches = []
    mismatches = []

    # Check all attributes that are integers
    for name in dir(_SetDim):
        if name.startswith('_'):
            continue
        orig_val = getattr(_SetDim, name, None)
        if not isinstance(orig_val, int):
            continue

        alloc_val = getattr(AllocDim, name, None)
        if alloc_val is None:
            mismatches.append((name, orig_val, None, "missing in allocator"))
        elif orig_val != alloc_val:
            mismatches.append((name, orig_val, alloc_val, "value mismatch"))
        else:
            matches.append(name)

    return matches, mismatches


if __name__ == "__main__":
    # Run verification
    matches, mismatches = verify_compatibility()
    print(f"Matches: {len(matches)}")
    print(f"Mismatches: {len(mismatches)}")
    for name, orig, alloc, reason in mismatches[:20]:
        print(f"  {name}: _SetDim={orig}, AllocDim={alloc} ({reason})")
