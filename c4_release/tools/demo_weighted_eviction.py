#!/usr/bin/env python3
"""
Demo: Weighted Scoring Eviction

Shows how the weighted scoring algorithm works and can be tuned.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neural_vm.weighted_eviction import WeightConfig

def main():
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 12 + "WEIGHTED SCORING EVICTION DEMO" + " " * 16 + "║")
    print("╚" + "═" * 58 + "╝")
    print()

    print("Score-based eviction reimplemented as a clean weighted scoring")
    print("algorithm, similar to traditional cache algorithms (ARC, LIRS).")
    print()

    # Show default configuration
    config = WeightConfig()

    print("=" * 60)
    print("DEFAULT WEIGHT CONFIGURATION")
    print("=" * 60)

    print("\n1. Layer 3 (Register Carry-Forward) - Recency-Based")
    print("   Similar to LRU (Least Recently Used)")
    print()
    for k, v in config.l3_weights.items():
        print(f"   {k:30s} {v:8}")

    print("\n2. Layer 5 (Code Fetch) - Importance-Based")
    print("   Similar to Priority-Based Caching")
    print()
    for k, v in config.l5_weights.items():
        print(f"   {k:30s} {v:8}")

    print("\n3. Layer 15 (Memory Lookup) - Validity + Importance")
    print("   Semantic understanding (knows about overwrites)")
    print()
    for k, v in config.l15_weights.items():
        print(f"   {k:30s} {v:8}")

    print("\n4. Eviction Threshold")
    print(f"   threshold: {config.threshold}")
    print(f"   Decision: score < {config.threshold} → EVICT")

    # Show score calculations
    print("\n" + "=" * 60)
    print("SCORE CALCULATIONS")
    print("=" * 60)

    print("\nExample 1: Valid Memory Entry")
    print("  Formula: base + mem_store_valid + zfod_offset + addr_match")
    print("  Calculation: 0 + 312.5*1 + (-600)*1 + 300*1")
    print(f"  Score: {0 + 312.5 - 600 + 300}")
    print(f"  Decision: {0 + 312.5 - 600 + 300} >= {config.threshold} → KEEP")

    print("\nExample 2: Overwritten Memory Entry")
    print("  Formula: base + mem_store_invalid + zfod_offset + addr_match")
    print("  Calculation: 0 + (-312.5)*1 + (-600)*1 + 300*1")
    print(f"  Score: {0 - 312.5 - 600 + 300}")
    print(f"  Decision: {0 - 312.5 - 600 + 300} < {config.threshold} → EVICT")

    print("\nExample 3: Most Recent PC Marker")
    print("  Formula: is_most_recent_marker")
    print("  Calculation: 50.0*1")
    print(f"  Score: 50.0")
    print(f"  Decision: 50.0 >= {config.threshold} → KEEP")

    print("\nExample 4: Old PC Marker")
    print("  Formula: is_old_marker")
    print("  Calculation: (-inf)*1")
    print(f"  Score: -inf")
    print(f"  Decision: -inf < {config.threshold} → EVICT")

    # Show how to tune weights
    print("\n" + "=" * 60)
    print("TUNING WEIGHTS FOR DIFFERENT WORKLOADS")
    print("=" * 60)

    print("\n1. Memory-Constrained Environment")
    print("   Goal: More aggressive eviction")
    print()
    print("   # Reduce weights to evict more:")
    print("   config.l15_weights['mem_store_valid'] = 150.0  # Was 312.5")
    print("   config.threshold = -5.0  # Was -10.0 (higher = more eviction)")

    print("\n2. Latency-Sensitive Environment")
    print("   Goal: Keep more entries (avoid cache misses)")
    print()
    print("   # Increase weights to keep more:")
    print("   config.l15_weights['mem_store_valid'] = 500.0  # Was 312.5")
    print("   config.threshold = -15.0  # Was -10.0 (lower = less eviction)")

    print("\n3. Custom Feature Weights")
    print("   Goal: Prioritize specific patterns")
    print()
    print("   # Add custom weights:")
    print("   config.l3_weights['is_loop_counter'] = 100.0  # NEW")
    print("   config.l5_weights['is_function_entry'] = 200.0  # NEW")

    # Compare with traditional algorithms
    print("\n" + "=" * 60)
    print("COMPARISON WITH TRADITIONAL CACHE ALGORITHMS")
    print("=" * 60)

    comparisons = [
        ("LRU", "Recency", "L3: is_most_recent_marker"),
        ("LFU", "Frequency", "Would need access counters"),
        ("ARC", "Adaptive", "Mix of L3 (recency) + custom"),
        ("LIRS", "Reuse distance", "Could add reuse_distance feature"),
        ("Priority", "Importance", "L5: has_addr_key"),
    ]

    print("\n{:<10s} {:<20s} {:<30s}".format("Algorithm", "Strategy", "Weighted Eviction"))
    print("-" * 60)
    for algo, strategy, weighted in comparisons:
        print(f"{algo:<10s} {strategy:<20s} {weighted:<30s}")

    # Show advantages
    print("\n" + "=" * 60)
    print("ADVANTAGES OF WEIGHTED SCORING APPROACH")
    print("=" * 60)

    advantages = [
        ("Interpretable", "Clear what each weight does"),
        ("Tunable", "Easy to adjust for different workloads"),
        ("Extensible", "Add new features without changing algorithm"),
        ("Provable", "Weights learned from transformer guarantee correctness"),
        ("Semantic", "Understands validity (overwrite), not just usage"),
        ("Multi-Layer", "Different weights per layer, take max"),
    ]

    print()
    for advantage, explanation in advantages:
        print(f"  ✓ {advantage}: {explanation}")

    # Usage example
    print("\n" + "=" * 60)
    print("USAGE EXAMPLE")
    print("=" * 60)

    print("""
from neural_vm.weighted_eviction import WeightedEviction, WeightConfig
from src.transformer_vm import C4TransformerVM

# Option 1: Use default weights
vm = C4TransformerVM()  # Uses weighted eviction internally

# Option 2: Customize weights
config = WeightConfig()
config.l15_weights['mem_store_valid'] = 500.0  # More conservative
config.threshold = -15.0  # Keep more entries

# Apply to eviction system
eviction = WeightedEviction(vm._runner.model, config)

# Get statistics
stats = eviction.get_stats(token_ids, embeddings)
print(f"Eviction rate: {stats['eviction_rate']:.1%}")
print(f"Layer contributions: {stats['layer_contributions']}")
""")

    print("\n" + "═" * 60)
    print()


if __name__ == "__main__":
    main()
