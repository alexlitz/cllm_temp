#!/usr/bin/env python3
"""
Transformer FLOP Analysis for Neural VM

Based on: https://kipp.ly/transformer-inference-arithmetic/

Key formulas:
- Attention (QKV projections): 2 × 3 × d_model² = 6 × d_model²
- Attention output: 2 × d_model²
- FFN: 2 × 8 × d_model² = 16 × d_model²  (for d_ff = 4 × d_model)
- Total per layer: 24 × d_model²

For our neural VM primitives, we count FLOPs for each operation:
1. SwiGLU multiplication
2. Newton-Raphson division
3. Attention-based memory access
"""

import math


class FLOPCounter:
    """Count FLOPs for transformer VM primitives."""

    def __init__(self, memory_bits: int = 20, memory_entries: int = 460):
        self.memory_bits = memory_bits
        self.memory_entries = memory_entries

        # FLOP counts per operation
        self.flops = {
            'swiglu_mul': 0,
            'newton_div': 0,
            'memory_read': 0,
            'memory_write': 0,
            'softmax': 0,
            'attention': 0,
        }

    @staticmethod
    def sigmoid_flops() -> int:
        """FLOPs for sigmoid(x) = 1 / (1 + exp(-x))."""
        # exp: ~20 FLOPs (Taylor series or lookup)
        # add: 1 FLOP
        # divide: ~20 FLOPs (Newton-Raphson)
        return 41

    @staticmethod
    def silu_flops() -> int:
        """FLOPs for silu(x) = x * sigmoid(x)."""
        # sigmoid: ~41 FLOPs
        # multiply: 1 FLOP
        return 42

    def swiglu_mul_flops(self) -> int:
        """
        FLOPs for SwiGLU multiplication: silu(a)*b + silu(-a)*(-b).

        Breakdown:
        - silu(a): 42 FLOPs
        - silu(-a): 1 (negate) + 42 = 43 FLOPs
        - silu(a) * b: 1 FLOP
        - -b: 1 FLOP
        - silu(-a) * (-b): 1 FLOP
        - sum: 1 FLOP
        Total: ~89 FLOPs

        Note: This is the "neural" cost. Integer multiply is 1 FLOP.
        """
        return 42 + 43 + 1 + 1 + 1 + 1  # = 89

    def newton_raphson_div_flops(self) -> int:
        """
        FLOPs for Newton-Raphson division.

        Steps:
        1. Normalization to [0.5, 1.0): ~15 FLOPs (bit shifts, compares)
        2. Table lookup via attention (256 entries):
           - Query encoding: memory_bits FLOPs
           - Dot products: 256 * memory_bits FLOPs
           - Softmax over 256: 256 * (exp + add + div) = 256 * 41 FLOPs
           - Weighted sum: 256 FLOPs
        3. Newton iteration 1: y = y * (2 - b*y)
           - b*y: 89 FLOPs (SwiGLU mul)
           - 2 - (b*y): 1 FLOP
           - y * result: 89 FLOPs
        4. Newton iteration 2: same = 179 FLOPs
        5. Scale back: ~5 FLOPs

        Total: ~15 + (20 + 256*20 + 256*41 + 256) + 179 + 179 + 5
             = ~16,000 FLOPs
        """
        normalization = 15
        query_encode = self.memory_bits
        dot_products = 256 * self.memory_bits  # 5,120
        softmax_256 = 256 * 41  # 10,496
        weighted_sum = 256

        table_lookup = query_encode + dot_products + softmax_256 + weighted_sum  # ~15,892

        newton_iter = 89 + 1 + 89  # 179 FLOPs per iteration
        newton_total = 2 * newton_iter  # 358

        scale_back = 5

        return normalization + table_lookup + newton_total + scale_back

    def memory_read_flops(self) -> int:
        """
        FLOPs for attention-based memory read.

        Steps:
        1. Query encoding (address to binary key): memory_bits FLOPs
        2. Dot products with all keys: n_entries * memory_bits FLOPs
        3. Softmax over n_entries: n_entries * 41 FLOPs
        4. Weighted sum of values: n_entries FLOPs

        For n=460 entries, bits=20:
        Total = 20 + 460*20 + 460*41 + 460 = 28,100 FLOPs
        """
        n = self.memory_entries
        query_encode = self.memory_bits
        dot_products = n * self.memory_bits
        softmax_n = n * 41
        weighted_sum = n
        return query_encode + dot_products + softmax_n + weighted_sum

    def memory_write_flops(self) -> int:
        """
        FLOPs for attention-based memory write.

        Steps:
        1. Query encoding: memory_bits FLOPs
        2. Check if address exists (attention query): memory_read_flops
        3. If exists, update value: 1 FLOP
        4. If new, append: ~memory_bits FLOPs (key encoding)

        Average: ~memory_read_flops + memory_bits
        """
        return self.memory_read_flops() + self.memory_bits

    def analyze_vm_execution(self, mul_count: int, div_count: int,
                              add_count: int, memory_reads: int,
                              memory_writes: int) -> dict:
        """Analyze total FLOPs for a VM execution."""

        swiglu_flops = mul_count * self.swiglu_mul_flops()
        newton_flops = div_count * self.newton_raphson_div_flops()
        add_flops = add_count * 1  # Direct addition is 1 FLOP
        mem_read_flops = memory_reads * self.memory_read_flops()
        mem_write_flops = memory_writes * self.memory_write_flops()

        total = swiglu_flops + newton_flops + add_flops + mem_read_flops + mem_write_flops

        return {
            'swiglu_mul': {
                'count': mul_count,
                'flops_per_op': self.swiglu_mul_flops(),
                'total_flops': swiglu_flops,
            },
            'newton_div': {
                'count': div_count,
                'flops_per_op': self.newton_raphson_div_flops(),
                'total_flops': newton_flops,
            },
            'add_sub': {
                'count': add_count,
                'flops_per_op': 1,
                'total_flops': add_flops,
            },
            'memory_read': {
                'count': memory_reads,
                'flops_per_op': self.memory_read_flops(),
                'total_flops': mem_read_flops,
            },
            'memory_write': {
                'count': memory_writes,
                'flops_per_op': self.memory_write_flops(),
                'total_flops': mem_write_flops,
            },
            'total_flops': total,
            'total_gflops': total / 1e9,
        }


def compare_with_transformer(analysis: dict, model_params: dict) -> dict:
    """
    Compare VM FLOPs with equivalent transformer inference.

    Transformer FLOPs per token per layer = 24 × d_model²
    """
    d_model = model_params.get('d_model', 768)  # GPT-2 small
    n_layers = model_params.get('n_layers', 12)
    d_ff = model_params.get('d_ff', 4 * d_model)

    # FLOPs per token for full transformer
    attention_flops = 8 * d_model ** 2  # QKV + output = 8 × d_model²
    ffn_flops = 2 * 2 * d_model * d_ff  # Two matrix muls = 4 × d_model × d_ff
    layer_flops = attention_flops + ffn_flops
    total_transformer_flops = n_layers * layer_flops

    # How many "transformer token equivalents" is our VM?
    vm_flops = analysis['total_flops']
    token_equivalents = vm_flops / total_transformer_flops

    return {
        'transformer': {
            'd_model': d_model,
            'n_layers': n_layers,
            'd_ff': d_ff,
            'flops_per_token': total_transformer_flops,
            'gflops_per_token': total_transformer_flops / 1e9,
        },
        'vm_total_flops': vm_flops,
        'vm_gflops': vm_flops / 1e9,
        'token_equivalents': token_equivalents,
    }


def print_analysis(analysis: dict, comparison: dict = None):
    """Pretty print the analysis."""
    print("=" * 70)
    print("  TRANSFORMER FLOP ANALYSIS FOR NEURAL VM")
    print("=" * 70)
    print()

    print("OPERATION BREAKDOWN:")
    print("-" * 70)
    print(f"{'Operation':<20} {'Count':>12} {'FLOPs/op':>12} {'Total FLOPs':>18}")
    print("-" * 70)

    for op_name in ['swiglu_mul', 'newton_div', 'add_sub', 'memory_read', 'memory_write']:
        op = analysis[op_name]
        print(f"{op_name:<20} {op['count']:>12,} {op['flops_per_op']:>12,} {op['total_flops']:>18,}")

    print("-" * 70)
    print(f"{'TOTAL':<20} {'':<12} {'':<12} {analysis['total_flops']:>18,}")
    print(f"{'':20} {'':<12} {'':<12} {analysis['total_gflops']:>15.3f} GFLOPs")
    print()

    if comparison:
        print("COMPARISON WITH STANDARD TRANSFORMER:")
        print("-" * 70)
        t = comparison['transformer']
        print(f"  Model: d_model={t['d_model']}, n_layers={t['n_layers']}, d_ff={t['d_ff']}")
        print(f"  FLOPs per token: {t['flops_per_token']:,} ({t['gflops_per_token']:.4f} GFLOPs)")
        print()
        print(f"  VM total: {comparison['vm_gflops']:.3f} GFLOPs")
        print(f"  Equivalent to: {comparison['token_equivalents']:,.1f} transformer tokens")
        print()


def main():
    # Example: 256x256 Mandelbrot results
    print("ANALYZING 256x256 MANDELBROT (from pruned VM run):")
    print()

    counter = FLOPCounter(memory_bits=20, memory_entries=460)

    # From the 256x256 run output:
    # Neural MUL (SwiGLU): 7,735,616
    # Neural DIV (Newton): 4,098,286
    # Total writes: ~52,739,421
    # Total reads: ~52,739,421 (approx similar)

    analysis = counter.analyze_vm_execution(
        mul_count=7_735_616,
        div_count=4_098_286,
        add_count=20_000_000,  # Estimate based on op counts
        memory_reads=52_739_421,
        memory_writes=52_739_421,
    )

    # Compare with GPT-2 small
    comparison = compare_with_transformer(analysis, {
        'd_model': 768,
        'n_layers': 12,
        'd_ff': 3072,
    })

    print_analysis(analysis, comparison)

    # Also show breakdown percentages
    print("FLOP BREAKDOWN BY PERCENTAGE:")
    print("-" * 70)
    total = analysis['total_flops']
    for op_name in ['swiglu_mul', 'newton_div', 'add_sub', 'memory_read', 'memory_write']:
        op = analysis[op_name]
        pct = 100 * op['total_flops'] / total
        print(f"  {op_name:<20}: {pct:>6.2f}%")

    print()
    print("=" * 70)

    # Compare different model sizes
    print()
    print("COMPARISON WITH DIFFERENT TRANSFORMER SIZES:")
    print("-" * 70)
    models = [
        ('GPT-2 Small', {'d_model': 768, 'n_layers': 12, 'd_ff': 3072}),
        ('GPT-2 Medium', {'d_model': 1024, 'n_layers': 24, 'd_ff': 4096}),
        ('GPT-2 Large', {'d_model': 1280, 'n_layers': 36, 'd_ff': 5120}),
        ('GPT-3 175B', {'d_model': 12288, 'n_layers': 96, 'd_ff': 49152}),
        ('LLaMA 7B', {'d_model': 4096, 'n_layers': 32, 'd_ff': 11008}),
        ('LLaMA 70B', {'d_model': 8192, 'n_layers': 80, 'd_ff': 28672}),
    ]

    for name, params in models:
        comp = compare_with_transformer(analysis, params)
        t = comp['transformer']
        print(f"  {name:<15}: {comp['token_equivalents']:>12,.1f} token equivalents "
              f"({t['gflops_per_token']:.4f} GFLOPs/token)")

    print()


if __name__ == "__main__":
    main()
