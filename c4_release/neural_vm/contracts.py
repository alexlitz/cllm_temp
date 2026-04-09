"""
Dimension Contract System - Validate that layers respect dimension reservations.

Usage:
    from neural_vm.contracts import DimensionContract

    # Validate model configuration
    violations = DimensionContract.validate_model(model)
    if violations:
        print("Contract violations found!")
        for v in violations:
            print(f"  - {v}")

    # Print dimension usage map
    DimensionContract.print_dimension_map(model)
"""

from typing import List, Dict, Set
from .vm_step import _SetDim, AutoregressiveVM

BD = _SetDim


class DimensionContract:
    """Track and validate which layers read/write which dimensions."""

    # Define dimension contracts
    # Format: 'dimension_name': {'writers': [...], 'readers': [...], 'reserved': bool}
    CONTRACTS = {
        'AX_CARRY_LO': {
            'description': 'Previous AX value for binary operations',
            'writers': ['Layer3_Attn_Head1'],  # Only Layer 3 should write
            'readers': ['Layer8_FFN', 'Layer9_FFN'],  # Who needs to read
            'reserved': True,  # No other layer should touch
            'dim_start': BD.AX_CARRY_LO,
            'dim_count': 16,
        },
        'AX_CARRY_HI': {
            'description': 'Previous AX value (high nibble) for binary operations',
            'writers': ['Layer3_Attn_Head1'],
            'readers': ['Layer8_FFN', 'Layer9_FFN'],
            'reserved': True,
            'dim_start': BD.AX_CARRY_HI,
            'dim_count': 16,
        },
        'ALU_LO': {
            'description': 'First operand for ALU operations',
            'writers': ['Layer7_Attn_Head0', 'Layer7_Attn_Head1', 'Layer6_Attn_Head2', 'Layer6_Attn_Head3'],
            'readers': ['Layer8_FFN', 'Layer9_FFN', 'Layer10_FFN'],
            'reserved': False,  # Can be overwritten
            'dim_start': BD.ALU_LO,
            'dim_count': 16,
        },
        'ALU_HI': {
            'description': 'First operand (high nibble) for ALU operations',
            'writers': ['Layer7_Attn_Head0', 'Layer7_Attn_Head1', 'Layer6_Attn_Head2', 'Layer6_Attn_Head3'],
            'readers': ['Layer8_FFN', 'Layer9_FFN', 'Layer10_FFN'],
            'reserved': False,
            'dim_start': BD.ALU_HI,
            'dim_count': 16,
        },
    }

    @classmethod
    def validate_model(cls, model: AutoregressiveVM) -> List[str]:
        """Check if model weights violate dimension contracts."""
        violations = []

        for dim_name, contract in cls.CONTRACTS.items():
            dim_start = contract['dim_start']
            dim_count = contract['dim_count']
            dim_end = dim_start + dim_count

            # Check each layer for unauthorized writes
            for layer_idx in range(16):
                layer = model.blocks[layer_idx]

                # Check attention W_o for writes
                writes_attn = cls._check_writes_attn(layer.attn, dim_start, dim_end)
                if writes_attn['writes']:
                    for head in writes_attn['heads']:
                        writer_name = f"Layer{layer_idx}_Attn_Head{head}"
                        if contract['reserved'] and writer_name not in contract['writers']:
                            violations.append({
                                'type': 'unauthorized_write',
                                'layer': layer_idx,
                                'component': f'Attn_Head{head}',
                                'dimension': dim_name,
                                'severity': 'error' if contract['reserved'] else 'warning',
                                'message': f"Layer {layer_idx} Attention Head {head} writes to reserved dimension {dim_name}",
                            })

                # Check FFN W_down for writes
                writes_ffn = cls._check_writes_ffn(layer.ffn, dim_start, dim_end)
                if writes_ffn:
                    writer_name = f"Layer{layer_idx}_FFN"
                    if contract['reserved'] and writer_name not in contract['writers']:
                        violations.append({
                            'type': 'unauthorized_write',
                            'layer': layer_idx,
                            'component': 'FFN',
                            'dimension': dim_name,
                            'severity': 'error' if contract['reserved'] else 'warning',
                            'message': f"Layer {layer_idx} FFN writes to reserved dimension {dim_name}",
                        })

        return violations

    @classmethod
    def _check_writes_attn(cls, attn, dim_start: int, dim_end: int) -> Dict:
        """Check if attention layer writes to dimension range."""
        w_o = attn.W_o[dim_start:dim_end, :]
        has_writes = (w_o.abs() > 1e-6).any().item()

        if not has_writes:
            return {'writes': False, 'heads': []}

        # Find which heads write
        heads = []
        head_dim = attn.head_dim
        num_heads = attn.num_heads

        for head in range(num_heads):
            head_start = head * head_dim
            head_end = head_start + head_dim
            head_writes = w_o[:, head_start:head_end]

            if (head_writes.abs() > 1e-6).any().item():
                heads.append(head)

        return {'writes': True, 'heads': heads}

    @classmethod
    def _check_writes_ffn(cls, ffn, dim_start: int, dim_end: int) -> bool:
        """Check if FFN writes to dimension range."""
        w_down = ffn.W_down[dim_start:dim_end, :]
        return (w_down.abs() > 1e-6).any().item()

    @classmethod
    def print_dimension_map(cls, model: AutoregressiveVM):
        """Print comprehensive map of dimension usage."""
        print("\n" + "=" * 80)
        print("DIMENSION USAGE MAP")
        print("=" * 80 + "\n")

        for dim_name, contract in cls.CONTRACTS.items():
            print(f"{dim_name} (dims {contract['dim_start']}-{contract['dim_start'] + contract['dim_count'] - 1}):")
            print(f"  Description: {contract['description']}")
            print(f"  Reserved: {'Yes' if contract['reserved'] else 'No'}")

            # Find actual writers
            actual_writers = []
            for layer_idx in range(16):
                layer = model.blocks[layer_idx]
                dim_start = contract['dim_start']
                dim_end = dim_start + contract['dim_count']

                # Check attention
                writes_attn = cls._check_writes_attn(layer.attn, dim_start, dim_end)
                if writes_attn['writes']:
                    for head in writes_attn['heads']:
                        actual_writers.append(f"Layer{layer_idx}_Attn_Head{head}")

                # Check FFN
                if cls._check_writes_ffn(layer.ffn, dim_start, dim_end):
                    actual_writers.append(f"Layer{layer_idx}_FFN")

            print(f"  Expected writers: {', '.join(contract['writers'])}")
            print(f"  Actual writers: {', '.join(actual_writers) if actual_writers else '(none)'}")

            # Check for violations
            if contract['reserved']:
                unauthorized = [w for w in actual_writers if w not in contract['writers']]
                if unauthorized:
                    print(f"  ❌ UNAUTHORIZED: {', '.join(unauthorized)}")
                elif set(actual_writers) == set(contract['writers']):
                    print(f"  ✓ Configuration correct")
                else:
                    missing = [w for w in contract['writers'] if w not in actual_writers]
                    if missing:
                        print(f"  ⚠️  MISSING: {', '.join(missing)}")

            print()

    @classmethod
    def print_violations_verbose(cls, violations: List[Dict]):
        """Print violations in user-friendly format."""
        if not violations:
            print("\n" + "=" * 80)
            print("✓ NO CONTRACT VIOLATIONS FOUND")
            print("=" * 80 + "\n")
            return

        print("\n" + "=" * 80)
        print("❌ DIMENSION CONTRACT VIOLATIONS DETECTED")
        print("=" * 80 + "\n")

        errors = [v for v in violations if v['severity'] == 'error']
        warnings = [v for v in violations if v['severity'] == 'warning']

        if errors:
            print(f"ERRORS ({len(errors)}):")
            for v in errors:
                print(f"  ❌ {v['message']}")
            print()

        if warnings:
            print(f"WARNINGS ({len(warnings)}):")
            for v in warnings:
                print(f"  ⚠️  {v['message']}")
            print()

        print("=" * 80 + "\n")

        # Print impact
        print("IMPACT:")
        affected_dims = set(v['dimension'] for v in violations)
        for dim in affected_dims:
            contract = cls.CONTRACTS.get(dim)
            if contract:
                print(f"  {dim}: {contract['description']}")
                print(f"    → This may cause: Binary operations to fail")
        print()


def validate_and_print(model: AutoregressiveVM):
    """Convenience function to validate and print results."""
    violations = DimensionContract.validate_model(model)
    DimensionContract.print_violations_verbose(violations)
    DimensionContract.print_dimension_map(model)
    return violations


if __name__ == "__main__":
    from .vm_step import AutoregressiveVM, set_vm_weights

    print("Validating Neural VM dimension contracts...\n")

    model = AutoregressiveVM()
    set_vm_weights(model)

    validate_and_print(model)
