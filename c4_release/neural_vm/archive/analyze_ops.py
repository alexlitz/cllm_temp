#!/usr/bin/env python3
"""
Opcode Analysis Tool for Neural VM.

Reports for each opcode:
1. Number of layers (FFNs/attention) used
2. Number of non-zero parameters
3. Total parameter count
4. Sparsity metrics
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

from .embedding import E, Opcode
from .base_layers import PureFFN, PureAttention


@dataclass
class OpcodeStats:
    """Statistics for a single opcode."""
    opcode: int
    name: str
    num_layers: int
    num_ffns: int
    num_attention: int
    total_params: int
    nonzero_params: int
    sparsity: float  # Percentage of zeros
    layer_names: List[str]


def count_nonzero_params(module: nn.Module) -> Tuple[int, int]:
    """
    Count total and non-zero parameters in a module.

    Returns:
        (total_params, nonzero_params)
    """
    total = 0
    nonzero = 0

    for param in module.parameters():
        total += param.numel()
        nonzero += torch.count_nonzero(param).item()

    return total, nonzero


def count_layers(module: nn.Module) -> Tuple[int, int, List[str]]:
    """
    Count FFN and attention layers in a module.

    Returns:
        (num_ffns, num_attention, layer_names)
    """
    ffn_count = 0
    attn_count = 0
    names = []

    for name, child in module.named_modules():
        if isinstance(child, PureFFN) or (
            hasattr(child, 'W_up') and hasattr(child, 'W_gate') and hasattr(child, 'W_down')
        ):
            ffn_count += 1
            names.append(f"FFN:{name}" if name else child.__class__.__name__)
        elif isinstance(child, PureAttention) or (
            hasattr(child, 'W_Q') and hasattr(child, 'W_K') and hasattr(child, 'W_V')
        ):
            attn_count += 1
            names.append(f"Attn:{name}" if name else child.__class__.__name__)

    return ffn_count, attn_count, names


def analyze_moe_layer(moe_layer) -> Dict[int, Tuple[int, int, int, int, List[str]]]:
    """
    Analyze a MultiExpertMoELayer to get per-opcode expert stats.

    Returns:
        Dict mapping opcode to (num_ffns, num_attn, total_params, nonzero_params, layer_names)
    """
    results = {}

    if not hasattr(moe_layer, 'opcode_to_experts') or not hasattr(moe_layer, 'experts'):
        return results

    for opcode, expert_indices in moe_layer.opcode_to_experts.items():
        num_ffns = 0
        num_attn = 0
        total_params = 0
        nonzero_params = 0
        layer_names = []

        for idx in expert_indices:
            if idx < len(moe_layer.experts):
                expert = moe_layer.experts[idx]
                f, a, names = count_layers(expert)
                num_ffns += f
                num_attn += a
                layer_names.extend(names)
                t, n = count_nonzero_params(expert)
                total_params += t
                nonzero_params += n

        results[opcode] = (num_ffns, num_attn, total_params, nonzero_params, layer_names)

    return results


def analyze_sparse_moe_alu(alu) -> Dict[int, OpcodeStats]:
    """
    Analyze a SparseMoEALU to get per-opcode statistics.

    Args:
        alu: SparseMoEALU instance

    Returns:
        Dict mapping opcode number to OpcodeStats
    """
    stats = {}

    # Get opcode names
    opcode_names = {
        getattr(Opcode, name): name
        for name in dir(Opcode)
        if not name.startswith('_') and isinstance(getattr(Opcode, name), int)
    }

    # Collect all opcodes and their experts from various MoE layers
    all_opcode_data = defaultdict(lambda: {
        'ffns': 0, 'attn': 0, 'total': 0, 'nonzero': 0, 'names': []
    })

    # Analyze MoE layers
    moe_attrs = ['initial_experts', 'carry_experts', 'final_experts', 'post_final_experts']
    for attr in moe_attrs:
        if hasattr(alu, attr):
            moe_layer = getattr(alu, attr)
            layer_stats = analyze_moe_layer(moe_layer)
            for opcode, (f, a, t, n, names) in layer_stats.items():
                all_opcode_data[opcode]['ffns'] += f
                all_opcode_data[opcode]['attn'] += a
                all_opcode_data[opcode]['total'] += t
                all_opcode_data[opcode]['nonzero'] += n
                all_opcode_data[opcode]['names'].extend(names)

    # Analyze shared attention layers that may be used by multiple ops
    shared_attn_attrs = [
        'carry_attn', 'eq_reduce_attn', 'ne_reduce_attn', 'cmp_broadcast_attn',
        'shl_attn', 'shr_attn', 'branch_cond_attn'
    ]
    shared_attn_stats = {'total': 0, 'nonzero': 0, 'count': 0}
    for attr in shared_attn_attrs:
        if hasattr(alu, attr):
            attn = getattr(alu, attr)
            t, n = count_nonzero_params(attn)
            shared_attn_stats['total'] += t
            shared_attn_stats['nonzero'] += n
            shared_attn_stats['count'] += 1

    # Map ops to their attention layers
    ops_using_carry_attn = [Opcode.ADD, Opcode.SUB, Opcode.MUL, Opcode.DIV, Opcode.MOD]
    ops_using_shift_attn = [Opcode.SHL, Opcode.SHR]
    ops_using_cmp_attn = [Opcode.LT, Opcode.GT, Opcode.LE, Opcode.GE, Opcode.EQ, Opcode.NE]

    # Add shared attention to relevant ops
    if hasattr(alu, 'carry_attn'):
        t, n = count_nonzero_params(alu.carry_attn)
        for op in ops_using_carry_attn:
            if op in all_opcode_data:
                all_opcode_data[op]['attn'] += 1
                all_opcode_data[op]['total'] += t
                all_opcode_data[op]['nonzero'] += n
                all_opcode_data[op]['names'].append('CarryPropagateAttention')

    if hasattr(alu, 'shl_attn'):
        t, n = count_nonzero_params(alu.shl_attn)
        if Opcode.SHL in all_opcode_data:
            all_opcode_data[Opcode.SHL]['attn'] += 1
            all_opcode_data[Opcode.SHL]['total'] += t
            all_opcode_data[Opcode.SHL]['nonzero'] += n

    if hasattr(alu, 'shr_attn'):
        t, n = count_nonzero_params(alu.shr_attn)
        if Opcode.SHR in all_opcode_data:
            all_opcode_data[Opcode.SHR]['attn'] += 1
            all_opcode_data[Opcode.SHR]['total'] += t
            all_opcode_data[Opcode.SHR]['nonzero'] += n

    # Convert to OpcodeStats
    for opcode, data in all_opcode_data.items():
        op_name = opcode_names.get(opcode, f"OP_{opcode}")
        total = data['total']
        nonzero = data['nonzero']
        sparsity = 100.0 * (1 - nonzero / max(total, 1))

        stats[opcode] = OpcodeStats(
            opcode=opcode,
            name=op_name,
            num_layers=data['ffns'] + data['attn'],
            num_ffns=data['ffns'],
            num_attention=data['attn'],
            total_params=total,
            nonzero_params=nonzero,
            sparsity=sparsity,
            layer_names=data['names']
        )

    return stats


def analyze_module_recursively(module: nn.Module) -> Tuple[int, int, int, int, List[str]]:
    """
    Recursively analyze a module for layers and parameters.

    Returns:
        (num_ffns, num_attn, total_params, nonzero_params, layer_names)
    """
    num_ffns = 0
    num_attn = 0
    total_params = 0
    nonzero_params = 0
    layer_names = []

    # Check if this module itself is an FFN or Attention
    is_ffn = isinstance(module, PureFFN) or (
        hasattr(module, 'W_up') and hasattr(module, 'W_gate') and hasattr(module, 'W_down')
    )
    is_attn = isinstance(module, PureAttention) or (
        hasattr(module, 'W_Q') and hasattr(module, 'W_K') and hasattr(module, 'W_V')
    )

    if is_ffn:
        num_ffns = 1
        layer_names.append(module.__class__.__name__)
        t, n = count_nonzero_params(module)
        total_params = t
        nonzero_params = n
    elif is_attn:
        num_attn = 1
        layer_names.append(module.__class__.__name__)
        t, n = count_nonzero_params(module)
        total_params = t
        nonzero_params = n
    else:
        # Recurse into children
        for name, child in module.named_children():
            f, a, t, n, names = analyze_module_recursively(child)
            num_ffns += f
            num_attn += a
            total_params += t
            nonzero_params += n
            layer_names.extend(names)

    return num_ffns, num_attn, total_params, nonzero_params, layer_names


def analyze_any_alu(alu) -> Dict[int, OpcodeStats]:
    """
    Analyze any ALU type to get per-opcode statistics.

    Works with SparseMoEALU, UnifiedTransformerALU, etc.
    """
    stats = {}

    # Get opcode names
    opcode_names = {
        getattr(Opcode, name): name
        for name in dir(Opcode)
        if not name.startswith('_') and isinstance(getattr(Opcode, name), int)
    }

    # Check if it's a SparseMoEALU with experts
    if hasattr(alu, 'initial_experts'):
        return analyze_sparse_moe_alu(alu)

    # Check for unified transformer style
    if hasattr(alu, 'blocks'):
        # All ops share same layers, just count total
        total_ffns = 0
        total_attn = 0
        total_params = 0
        total_nonzero = 0
        all_names = []

        for block in alu.blocks:
            f, a, t, n, names = analyze_module_recursively(block)
            total_ffns += f
            total_attn += a
            total_params += t
            total_nonzero += n
            all_names.extend(names)

        sparsity = 100.0 * (1 - total_nonzero / max(total_params, 1))

        # Create stats for all implemented opcodes
        for op_num, op_name in opcode_names.items():
            if op_num < 70:  # Only implemented opcodes
                stats[op_num] = OpcodeStats(
                    opcode=op_num,
                    name=op_name,
                    num_layers=total_ffns + total_attn,
                    num_ffns=total_ffns,
                    num_attention=total_attn,
                    total_params=total_params,
                    nonzero_params=total_nonzero,
                    sparsity=sparsity,
                    layer_names=all_names[:10] + ['...'] if len(all_names) > 10 else all_names
                )

    return stats


def print_opcode_report(stats: Dict[int, OpcodeStats], verbose: bool = False):
    """Print formatted report of opcode statistics."""

    print("=" * 80)
    print("OPCODE ANALYSIS REPORT")
    print("=" * 80)
    print()
    print(f"{'Opcode':<8} {'Name':<12} {'Layers':<8} {'FFNs':<6} {'Attn':<6} {'Params':<12} {'Non-Zero':<12} {'Sparsity':<10}")
    print("-" * 80)

    # Sort by opcode number
    for op_num in sorted(stats.keys()):
        s = stats[op_num]
        print(f"{s.opcode:<8} {s.name:<12} {s.num_layers:<8} {s.num_ffns:<6} {s.num_attention:<6} "
              f"{s.total_params:<12,} {s.nonzero_params:<12,} {s.sparsity:>8.1f}%")

        if verbose and s.layer_names:
            for name in s.layer_names[:5]:
                print(f"         -> {name}")
            if len(s.layer_names) > 5:
                print(f"         ... and {len(s.layer_names) - 5} more")

    print("-" * 80)

    # Summary
    total_ops = len(stats)
    total_params = sum(s.total_params for s in stats.values())
    total_nonzero = sum(s.nonzero_params for s in stats.values())
    avg_layers = sum(s.num_layers for s in stats.values()) / max(total_ops, 1)
    avg_sparsity = sum(s.sparsity for s in stats.values()) / max(total_ops, 1)

    print(f"\nSummary:")
    print(f"  Total opcodes analyzed: {total_ops}")
    print(f"  Average layers per op: {avg_layers:.1f}")
    print(f"  Average sparsity: {avg_sparsity:.1f}%")

    # Group by operation category
    categories = {
        'Arithmetic': [Opcode.ADD, Opcode.SUB, Opcode.MUL, Opcode.DIV, Opcode.MOD],
        'Bitwise': [Opcode.OR, Opcode.XOR, Opcode.AND],
        'Comparison': [Opcode.EQ, Opcode.NE, Opcode.LT, Opcode.GT, Opcode.LE, Opcode.GE],
        'Shift': [Opcode.SHL, Opcode.SHR],
        'Memory': [Opcode.LI, Opcode.LC, Opcode.SI, Opcode.SC, Opcode.PSH],
        'I/O': [Opcode.GETCHAR, Opcode.PUTCHAR, Opcode.PRTF, Opcode.EXIT],
    }

    print("\nBy Category:")
    for cat_name, opcodes in categories.items():
        cat_stats = [stats[op] for op in opcodes if op in stats]
        if cat_stats:
            avg_layers = sum(s.num_layers for s in cat_stats) / len(cat_stats)
            avg_params = sum(s.nonzero_params for s in cat_stats) / len(cat_stats)
            print(f"  {cat_name}: {len(cat_stats)} ops, avg {avg_layers:.1f} layers, avg {avg_params:,.0f} non-zero params")

    print("=" * 80)


def main():
    """Run opcode analysis on SparseMoEALU."""
    print("Loading SparseMoEALU...")

    try:
        from .sparse_moe_alu import SparseMoEALU
        alu = SparseMoEALU()
        print("ALU loaded successfully\n")

        stats = analyze_sparse_moe_alu(alu)
        print_opcode_report(stats, verbose=True)

    except Exception as e:
        print(f"Error loading ALU: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
