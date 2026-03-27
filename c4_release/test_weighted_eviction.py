#!/usr/bin/env python3
"""
Test weighted scoring eviction implementation.

Verifies that the clean weighted scoring approach produces the same
results as the original score-based eviction.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from neural_vm.weighted_eviction import WeightedEviction, WeightConfig
from neural_vm.score_based_eviction import ScoreBasedEviction
from neural_vm.vm_step import Token, _SetDim as BD


def test_weight_config():
    """Test weight configuration."""
    print("=" * 60)
    print("TEST: Weight Configuration")
    print("=" * 60)

    config = WeightConfig(threshold=-10.0)

    print("\nChecking weight values...")
    assert config.l3_weights['is_most_recent_marker'] == 50.0
    assert config.l5_weights['has_addr_key'] == 300.0
    assert config.l15_weights['mem_store_valid'] == 312.5
    assert config.l15_weights['mem_store_invalid'] == -312.5
    assert config.threshold == -10.0

    print("✓ All weights configured correctly")
    return True


def test_feature_extraction():
    """Test feature extraction from embeddings."""
    print("\n" + "=" * 60)
    print("TEST: Feature Extraction")
    print("=" * 60)

    # Create mock model
    class MockModel:
        def __init__(self):
            self.blocks = [None] * 16

    model = MockModel()
    eviction = WeightedEviction(model)

    # Test MEM token with MEM_STORE=1
    print("\n--- Test 1: Valid MEM entry ---")
    token_ids = torch.tensor([[Token.MEM]])
    embeddings = torch.zeros(1, 1, 512)
    embeddings[0, 0, BD.MEM_STORE] = 1.0  # Valid

    features = eviction.extract_features(token_ids, embeddings, 0)

    print(f"MEM_STORE value: {embeddings[0, 0, BD.MEM_STORE].item()}")
    print(f"Features extracted:")
    print(f"  mem_store_valid: {features.get('mem_store_valid', 0)}")
    print(f"  mem_store_invalid: {features.get('mem_store_invalid', 0)}")

    assert features['mem_store_valid'] == 1.0
    assert features['mem_store_invalid'] == 0.0
    print("✓ Valid MEM features correct")

    # Test MEM token with MEM_STORE=0
    print("\n--- Test 2: Overwritten MEM entry ---")
    embeddings[0, 0, BD.MEM_STORE] = 0.0  # Overwritten

    features = eviction.extract_features(token_ids, embeddings, 0)

    print(f"MEM_STORE value: {embeddings[0, 0, BD.MEM_STORE].item()}")
    print(f"Features extracted:")
    print(f"  mem_store_valid: {features.get('mem_store_valid', 0)}")
    print(f"  mem_store_invalid: {features.get('mem_store_invalid', 0)}")

    assert features['mem_store_valid'] == 0.0
    assert features['mem_store_invalid'] == 1.0
    print("✓ Overwritten MEM features correct")

    # Test register marker recency
    print("\n--- Test 3: Register marker recency ---")
    token_ids = torch.tensor([[Token.REG_PC, Token.REG_PC]])
    embeddings = torch.zeros(1, 2, 512)

    # First PC (old)
    features_old = eviction.extract_features(token_ids, embeddings, 0)
    print(f"Old PC marker:")
    print(f"  is_most_recent_marker: {features_old.get('is_most_recent_marker', 0)}")
    print(f"  is_old_marker: {features_old.get('is_old_marker', 0)}")

    assert features_old['is_most_recent_marker'] == 0.0
    assert features_old['is_old_marker'] == 1.0

    # Second PC (most recent)
    features_recent = eviction.extract_features(token_ids, embeddings, 1)
    print(f"Recent PC marker:")
    print(f"  is_most_recent_marker: {features_recent.get('is_most_recent_marker', 0)}")
    print(f"  is_old_marker: {features_recent.get('is_old_marker', 0)}")

    assert features_recent['is_most_recent_marker'] == 1.0
    assert features_recent['is_old_marker'] == 0.0
    print("✓ Register recency features correct")

    return True


def test_score_computation():
    """Test weighted score computation."""
    print("\n" + "=" * 60)
    print("TEST: Weighted Score Computation")
    print("=" * 60)

    class MockModel:
        def __init__(self):
            self.blocks = [None] * 16

    model = MockModel()
    config = WeightConfig()
    eviction = WeightedEviction(model, config)

    # Test 1: Valid MEM entry
    print("\n--- Test 1: Valid MEM entry score ---")
    token_ids = torch.tensor([[Token.MEM]])
    embeddings = torch.zeros(1, 1, 512)
    embeddings[0, 0, BD.MEM_STORE] = 1.0  # Valid

    score = eviction.compute_score(token_ids, embeddings, 0)
    print(f"Score: {score}")
    print(f"Calculation: 0 + 312.5*1 + (-600)*1 + 300*1 = {0 + 312.5 - 600 + 300}")

    expected = 12.5
    assert abs(score - expected) < 0.1, f"Expected {expected}, got {score}"
    print(f"✓ Valid MEM score = {score} (expected {expected})")

    # Test 2: Overwritten MEM entry
    print("\n--- Test 2: Overwritten MEM entry score ---")
    embeddings[0, 0, BD.MEM_STORE] = 0.0  # Overwritten

    score = eviction.compute_score(token_ids, embeddings, 0)
    print(f"Score: {score}")
    print(f"Calculation: 0 + (-312.5)*1 + (-600)*1 + 300*1 = {0 - 312.5 - 600 + 300}")

    expected = -612.5
    assert abs(score - expected) < 0.1, f"Expected {expected}, got {score}"
    print(f"✓ Overwritten MEM score = {score} (expected {expected})")

    # Test 3: Most recent PC marker
    print("\n--- Test 3: Most recent PC marker score ---")
    token_ids = torch.tensor([[Token.REG_PC]])
    embeddings = torch.zeros(1, 1, 512)

    score = eviction.compute_score(token_ids, embeddings, 0)
    print(f"Score: {score}")

    expected = 50.0
    assert abs(score - expected) < 0.1, f"Expected {expected}, got {score}"
    print(f"✓ Most recent PC score = {score} (expected {expected})")

    # Test 4: Position with ADDR_KEY
    print("\n--- Test 4: Bytecode position (ADDR_KEY) score ---")
    token_ids = torch.tensor([[5]])  # Arbitrary byte
    embeddings = torch.zeros(1, 1, 512)
    embeddings[0, 0, BD.ADDR_KEY] = 1.0  # Has ADDR_KEY

    score = eviction.compute_score(token_ids, embeddings, 0)
    print(f"Score: {score}")

    expected = 300.0
    assert abs(score - expected) < 0.1, f"Expected {expected}, got {score}"
    print(f"✓ Bytecode position score = {score} (expected {expected})")

    return True


def test_eviction_decisions():
    """Test eviction decisions based on scores."""
    print("\n" + "=" * 60)
    print("TEST: Eviction Decisions")
    print("=" * 60)

    class MockModel:
        def __init__(self):
            self.blocks = [None] * 16

    model = MockModel()
    config = WeightConfig(threshold=-10.0)
    eviction = WeightedEviction(model, config)

    test_cases = [
        ("Valid MEM", Token.MEM, 1.0, 12.5, False),      # Should KEEP
        ("Overwritten MEM", Token.MEM, 0.0, -612.5, True),  # Should EVICT
        ("Recent PC", Token.REG_PC, 0.0, 50.0, False),   # Should KEEP
    ]

    print(f"\nThreshold: {config.threshold}")
    print(f"Decision: score < {config.threshold} → EVICT\n")

    for name, token, mem_store, expected_score, should_evict in test_cases:
        token_ids = torch.tensor([[token]])
        embeddings = torch.zeros(1, 1, 512)
        if token == Token.MEM:
            embeddings[0, 0, BD.MEM_STORE] = mem_store

        score = eviction.compute_score(token_ids, embeddings, 0)
        decision = eviction.should_evict(score)

        status = "✓" if decision == should_evict else "✗"
        action = "EVICT" if decision else "KEEP"
        print(f"{status} {name:20s} score={score:8.1f} → {action}")

        assert abs(score - expected_score) < 0.1, f"Score mismatch for {name}"
        assert decision == should_evict, f"Decision mismatch for {name}"

    print("\n✓ All eviction decisions correct")
    return True


def test_retention_mask():
    """Test retention mask generation."""
    print("\n" + "=" * 60)
    print("TEST: Retention Mask Generation")
    print("=" * 60)

    class MockModel:
        def __init__(self):
            self.blocks = [None] * 16

    model = MockModel()
    eviction = WeightedEviction(model)

    # Create context with mix of scores
    token_ids = torch.tensor([[Token.MEM, Token.MEM, Token.REG_PC]])
    embeddings = torch.zeros(1, 3, 512)

    # MEM[0]: valid (KEEP)
    embeddings[0, 0, BD.MEM_STORE] = 1.0
    # MEM[1]: overwritten (EVICT)
    embeddings[0, 1, BD.MEM_STORE] = 0.0
    # PC: most recent (KEEP)

    mask = eviction.get_retention_mask(token_ids, embeddings)

    print("\nToken IDs:", token_ids[0].tolist())
    print("MEM_STORE values:", [embeddings[0, i, BD.MEM_STORE].item() for i in range(3)])
    print("Retention mask:", mask[0].tolist())

    assert mask[0, 0].item() == True, "Valid MEM should be retained"
    assert mask[0, 1].item() == False, "Overwritten MEM should be evicted"
    assert mask[0, 2].item() == True, "Most recent PC should be retained"

    print("\n✓ Retention mask correct")
    return True


def compare_with_original():
    """Compare weighted eviction with original score-based eviction."""
    print("\n" + "=" * 60)
    print("TEST: Compare Weighted vs Original Implementation")
    print("=" * 60)

    class MockModel:
        def __init__(self):
            self.blocks = [None] * 16

    model = MockModel()

    # Create both implementations
    weighted = WeightedEviction(model)
    original = ScoreBasedEviction(model)

    # Test cases
    test_cases = [
        ("Valid MEM", Token.MEM, 1.0, None),
        ("Overwritten MEM", Token.MEM, 0.0, None),
        ("Most recent PC", Token.REG_PC, 0.0, None),
    ]

    print("\nComparing scores:\n")

    all_match = True
    for name, token, mem_store, _ in test_cases:
        token_ids = torch.tensor([[token]])
        embeddings = torch.zeros(1, 1, 512)
        if token == Token.MEM:
            embeddings[0, 0, BD.MEM_STORE] = mem_store

        # Compute scores with both methods
        score_weighted = weighted.compute_score(token_ids, embeddings, 0)

        # For original, we need to compute layer-specific scores
        if token == Token.MEM:
            score_original = original._compute_l15_max_scores(token_ids, embeddings)[0, 0].item()
        elif token == Token.REG_PC:
            score_original = original._compute_l3_max_scores(token_ids, embeddings)[0, 0].item()
        else:
            score_original = 0.0

        match = abs(score_weighted - score_original) < 0.1
        status = "✓" if match else "✗"

        print(f"{status} {name:20s} weighted={score_weighted:8.1f}  original={score_original:8.1f}")

        if not match:
            all_match = False

    if all_match:
        print("\n✅ Weighted implementation matches original!")
    else:
        print("\n⚠️  Some scores differ")

    return all_match


def main():
    print("\n╔" + "═" * 58 + "╗")
    print("║" + " " * 12 + "WEIGHTED EVICTION TEST SUITE" + " " * 18 + "║")
    print("╚" + "═" * 58 + "╝")

    results = []

    try:
        # Test 1: Configuration
        results.append(("Weight Config", test_weight_config()))

        # Test 2: Feature extraction
        results.append(("Feature Extraction", test_feature_extraction()))

        # Test 3: Score computation
        results.append(("Score Computation", test_score_computation()))

        # Test 4: Eviction decisions
        results.append(("Eviction Decisions", test_eviction_decisions()))

        # Test 5: Retention mask
        results.append(("Retention Mask", test_retention_mask()))

        # Test 6: Compare with original
        results.append(("Compare vs Original", compare_with_original()))

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        for name, passed in results:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {name:25s} {status}")

        all_passed = all(passed for _, passed in results)

        if all_passed:
            print("\n╔" + "═" * 58 + "╗")
            print("║" + " " * 15 + "ALL TESTS PASSED ✓" + " " * 25 + "║")
            print("╚" + "═" * 58 + "╝")
            print()
            print("Summary:")
            print("  ✓ Weighted eviction implemented correctly")
            print("  ✓ Produces same scores as original implementation")
            print("  ✓ Feature extraction works")
            print("  ✓ Eviction decisions correct")
            print("  ✓ Clean, interpretable weighted scoring algorithm")
            print()
        else:
            print("\n❌ SOME TESTS FAILED")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ TEST FAILED WITH ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
