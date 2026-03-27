#!/usr/bin/env python3
"""
Unit tests for score-based eviction - test just the score computation logic.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from neural_vm.score_based_eviction import ScoreBasedEviction
from neural_vm.vm_step import Token, _SetDim as BD


def test_score_computation_basic():
    """Test basic score computation without full VM."""
    print("=" * 60)
    print("TEST: Basic Score Computation")
    print("=" * 60)

    # Create a mock model with blocks attribute
    class MockModel:
        def __init__(self):
            # Create 16 empty blocks (n_layers=16)
            self.blocks = [None] * 16

    model = MockModel()
    eviction = ScoreBasedEviction(model, eviction_threshold=-10.0)

    print(f"Number of layers: {eviction.n_layers}")
    assert eviction.n_layers == 16, f"Expected 16 layers, got {eviction.n_layers}"
    print("✓ Layer count correct")

    # Test L15 score computation (memory)
    print("\n--- Testing L15 (Memory) Scores ---")

    # Create test tokens: MEM token
    token_ids = torch.tensor([[Token.MEM]])

    # Create embeddings with MEM_STORE=1 (valid entry)
    embeddings = torch.zeros(1, 1, 512)
    embeddings[0, 0, BD.MEM_STORE] = 1.0  # Set MEM_STORE flag

    scores_valid = eviction._compute_l15_max_scores(token_ids, embeddings)
    print(f"MEM with MEM_STORE=1 score: {scores_valid[0, 0].item()}")
    assert scores_valid[0, 0].item() == 12.5, "Expected score +12.5 for valid MEM"
    print("✓ Valid MEM entry gets score +12.5 (KEEP)")

    # Test with MEM_STORE=0 (overwritten entry)
    embeddings[0, 0, BD.MEM_STORE] = 0.0
    scores_invalid = eviction._compute_l15_max_scores(token_ids, embeddings)
    print(f"MEM with MEM_STORE=0 score: {scores_invalid[0, 0].item()}")
    assert scores_invalid[0, 0].item() == -612.5, "Expected score -612.5 for overwritten MEM"
    print("✓ Overwritten MEM entry gets score -612.5 (EVICT)")

    # Test L3 score computation (register carry-forward)
    print("\n--- Testing L3 (Register Carry) Scores ---")

    # Create sequence: old PC, new PC
    token_ids = torch.tensor([[Token.REG_PC, Token.REG_PC]])
    embeddings = torch.zeros(1, 2, 512)

    scores_l3 = eviction._compute_l3_max_scores(token_ids, embeddings)
    old_score = scores_l3[0, 0].item()
    latest_score = scores_l3[0, 1].item()
    print(f"Old PC marker score: {old_score}")
    print(f"Latest PC marker score: {latest_score}")

    assert old_score == -float('inf'), "Old PC marker should have score -inf (evicted)"
    assert latest_score == 50.0, "Latest PC marker should have score 50"
    print("✓ Only most recent register markers get positive scores")

    # Test L5 score computation (code fetch)
    print("\n--- Testing L5 (Code Fetch) Scores ---")

    # Create bytecode position with ADDR_KEY
    token_ids = torch.tensor([[5, 10]])  # Arbitrary byte values
    embeddings = torch.zeros(1, 2, 512)

    # First position has ADDR_KEY (bytecode)
    embeddings[0, 0, BD.ADDR_KEY] = 1.0

    scores_l5 = eviction._compute_l5_max_scores(token_ids, embeddings)
    with_addr = scores_l5[0, 0].item()
    without_addr = scores_l5[0, 1].item()
    print(f"Position with ADDR_KEY score: {with_addr}")
    print(f"Position without ADDR_KEY score: {without_addr}")

    assert with_addr == 300.0, "Position with ADDR_KEY should score 300"
    assert without_addr == -float('inf'), "Position without ADDR_KEY should score -inf"
    print("✓ Bytecode positions with ADDR_KEY get score +300.0")

    # Test retention mask
    print("\n--- Testing Retention Mask ---")

    # Create context with mix of scores
    token_ids = torch.tensor([[Token.MEM, Token.MEM, Token.REG_PC]])
    embeddings = torch.zeros(1, 3, 512)

    # MEM[0]: MEM_STORE=1 (keep)
    embeddings[0, 0, BD.MEM_STORE] = 1.0
    # MEM[1]: MEM_STORE=0 (evict)
    embeddings[0, 1, BD.MEM_STORE] = 0.0
    # PC: most recent (keep)

    mask = eviction.get_retention_mask(token_ids, embeddings)
    print(f"Retention mask: {mask[0].tolist()}")

    assert mask[0, 0].item() == True, "Valid MEM should be retained"
    assert mask[0, 1].item() == False, "Invalid MEM should be evicted"
    assert mask[0, 2].item() == True, "Latest PC should be retained"
    print("✓ Retention mask correctly identifies positions to keep/evict")

    # Test eviction decision
    print("\n--- Testing Eviction Decision ---")

    max_scores = torch.tensor([[15.0, -5.0, -15.0]])

    should_keep_0 = not eviction.should_evict(max_scores, 0)
    should_keep_1 = not eviction.should_evict(max_scores, 1)
    should_keep_2 = not eviction.should_evict(max_scores, 2)

    print(f"Score +15.0: keep={should_keep_0}")
    print(f"Score -5.0:  keep={should_keep_1}")
    print(f"Score -15.0: keep={should_keep_2}")

    assert should_keep_0 == True, "Score > -10 should be kept"
    assert should_keep_1 == True, "Score > -10 should be kept"
    assert should_keep_2 == False, "Score < -10 should be evicted"
    print("✓ Eviction threshold (-10.0) works correctly")

    print("\n" + "=" * 60)
    print("✓ ALL UNIT TESTS PASSED")
    print("=" * 60)
    return True


def main():
    try:
        test_score_computation_basic()
        print("\n✅ Score-based eviction logic is working correctly!")
        print("\nNext step: Run full integration tests with neural VM")
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
