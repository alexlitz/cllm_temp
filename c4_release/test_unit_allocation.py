"""
Debug Unit Allocation Strategy

Prints the unit offset allocation for each opcode to verify non-overlapping ranges.
"""

from neural_vm.weight_loader import CompiledWeightLoader
from neural_vm.embedding import Opcode

def test_unit_allocation():
    """Print unit allocation for all opcodes."""

    print("="*70)
    print("UNIT ALLOCATION DEBUG")
    print("="*70)
    print()

    loader = CompiledWeightLoader()
    allocations = loader.unit_allocations

    print(f"Total hidden units available: 4096")
    print()
    print("Opcode allocations:")
    print("-" * 70)

    # Get opcode names
    opcode_names = {}
    for name, value in vars(Opcode).items():
        if name.isupper() and isinstance(value, int):
            opcode_names[value] = name

    # Sort by offset
    sorted_allocs = sorted(allocations.items(), key=lambda x: x[1])

    max_offset = 0
    for opcode, offset in sorted_allocs:
        name = opcode_names.get(opcode, f"OPCODE_{opcode}")

        # Estimate unit count (next offset - current offset)
        idx = [i for i, (op, _) in enumerate(sorted_allocs) if op == opcode][0]
        if idx < len(sorted_allocs) - 1:
            next_offset = sorted_allocs[idx + 1][1]
            unit_count = next_offset - offset
        else:
            # Last opcode - assume 64 units
            unit_count = 64

        print(f"  {opcode:2d} {name:8s} → units {offset:4d}-{offset+unit_count-1:4d} ({unit_count:3d} units)")
        max_offset = max(max_offset, offset + unit_count)

    print()
    print("-" * 70)
    print(f"Total units allocated: {max_offset}/4096")
    print(f"Utilization: {100*max_offset/4096:.1f}%")
    print("="*70)

if __name__ == "__main__":
    test_unit_allocation()
