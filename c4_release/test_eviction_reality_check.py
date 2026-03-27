#!/usr/bin/env python3
"""
Reality Check: Does key similarity eviction actually work?

Tests whether the model's actual weights produce similar keys
for similar tokens (which is required for eviction to work).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from neural_vm.key_similarity_eviction import KeySimilarityEviction
from neural_vm.vm_step import AutoregressiveVM, Token


def test_actual_key_similarity():
    """Test if actual model produces similar keys for similar tokens."""
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "REALITY CHECK: Key Similarity" + " " * 18 + "║")
    print("╚" + "═" * 58 + "╝")

    print("\nTesting if model produces similar keys for similar tokens...")

    # Create model
    model = AutoregressiveVM(d_model=512, n_layers=16)
    model.eval()

    eviction = KeySimilarityEviction(model, similarity_threshold=0.99)

    # Test 1: Same token repeated
    print("\n" + "=" * 60)
    print("TEST 1: Same Token Repeated (REG_PC)")
    print("=" * 60)

    # Create context with same token repeated
    context = [Token.REG_PC] * 5 + [100, 0, 0, 0]
    token_ids = torch.tensor([context], dtype=torch.long)

    keys = eviction.compute_keys(token_ids, layer_idx=0)
    similarity = eviction.compute_pairwise_similarity(keys)

    # Check similarity between first and last REG_PC tokens
    sim_same_token = similarity[0, 4].item()
    print(f"\nContext: 5x REG_PC marker, then 4 bytes")
    print(f"Similarity between REG_PC[0] and REG_PC[4]: {sim_same_token:.6f}")
    print(f"Expected for eviction: > 0.99")
    print(f"Result: {'✓ SIMILAR' if sim_same_token > 0.99 else '✗ NOT SIMILAR'}")

    # Test 2: Different tokens
    print("\n" + "=" * 60)
    print("TEST 2: Different Tokens (REG_PC vs REG_AX)")
    print("=" * 60)

    context2 = [Token.REG_PC, Token.REG_AX]
    token_ids2 = torch.tensor([context2], dtype=torch.long)

    keys2 = eviction.compute_keys(token_ids2, layer_idx=0)
    similarity2 = eviction.compute_pairwise_similarity(keys2)

    sim_diff_tokens = similarity2[0, 1].item()
    print(f"\nContext: REG_PC, REG_AX")
    print(f"Similarity between REG_PC and REG_AX: {sim_diff_tokens:.6f}")
    print(f"Expected: < 0.50 (different registers)")
    print(f"Result: {'✓ DIFFERENT' if sim_diff_tokens < 0.50 else '✗ TOO SIMILAR'}")

    # Test 3: Realistic VM step pattern
    print("\n" + "=" * 60)
    print("TEST 3: Realistic VM Steps (Multiple PC Writes)")
    print("=" * 60)

    # Simulate 3 VM steps writing PC register
    step1 = [Token.REG_PC, 100, 0, 0, 0]  # PC = 100
    step2 = [Token.REG_PC, 105, 0, 0, 0]  # PC = 105
    step3 = [Token.REG_PC, 110, 0, 0, 0]  # PC = 110

    context3 = step1 + step2 + step3
    token_ids3 = torch.tensor([context3], dtype=torch.long)

    keys3 = eviction.compute_keys(token_ids3, layer_idx=0)
    similarity3 = eviction.compute_pairwise_similarity(keys3)

    # Check similarity between PC markers
    pc_idx_1 = 0
    pc_idx_2 = 5
    pc_idx_3 = 10

    sim_pc_1_2 = similarity3[pc_idx_1, pc_idx_2].item()
    sim_pc_1_3 = similarity3[pc_idx_1, pc_idx_3].item()
    sim_pc_2_3 = similarity3[pc_idx_2, pc_idx_3].item()

    print(f"\nContext: 3 PC writes (100, 105, 110)")
    print(f"Similarity between PC markers:")
    print(f"  PC[0] vs PC[5]:  {sim_pc_1_2:.6f}")
    print(f"  PC[0] vs PC[10]: {sim_pc_1_3:.6f}")
    print(f"  PC[5] vs PC[10]: {sim_pc_2_3:.6f}")
    print(f"Expected for eviction: > 0.99")

    max_sim = max(sim_pc_1_2, sim_pc_1_3, sim_pc_2_3)
    print(f"\nMax similarity: {max_sim:.6f}")
    print(f"Result: {'✓ WOULD EVICT' if max_sim > 0.99 else '✗ WOULD NOT EVICT'}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_tests = [
        ("Same token similarity", sim_same_token, "> 0.99", sim_same_token > 0.99),
        ("Different tokens", sim_diff_tokens, "< 0.50", sim_diff_tokens < 0.50),
        ("VM step PC markers", max_sim, "> 0.99", max_sim > 0.99),
    ]

    passing = sum(1 for _, _, _, passed in all_tests if passed)
    total = len(all_tests)

    print(f"\nTests: {passing}/{total} passing")
    for test_name, value, expected, passed in all_tests:
        status = "✓" if passed else "✗"
        print(f"  {status} {test_name}: {value:.4f} (expected {expected})")

    # Diagnosis
    print("\n" + "=" * 60)
    print("DIAGNOSIS")
    print("=" * 60)

    if passing == total:
        print("\n✅ Eviction is working correctly!")
        print("   Model produces similar keys for similar tokens.")
        print("   Latest-write-wins will function as designed.")
    else:
        print("\n⚠️  Eviction will NOT work as designed.")
        print("\nReason:")
        print("  • Model weights are untrained/random")
        print("  • Similar tokens produce dissimilar keys")
        print("  • Similarity threshold (0.99) never reached")
        print("  • No entries will be evicted")

        print("\nTo fix:")
        print("  1. Train the model so similar tokens cluster in key space")
        print("  2. OR use trained weights from checkpoint")
        print("  3. OR lower threshold temporarily (not recommended)")

        print("\nExpected after training:")
        print("  • PC marker tokens → very similar keys (>0.99)")
        print("  • Memory at same address → similar keys (>0.99)")
        print("  • Different registers → dissimilar keys (<0.50)")

    print("\n" + "═" * 60)

    return passing == total


if __name__ == "__main__":
    success = test_actual_key_similarity()
    sys.exit(0 if success else 1)
