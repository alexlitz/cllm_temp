#!/usr/bin/env python3
"""
Test suite for standard eviction policy interface.

Verifies that the eviction policies follow standard cache algorithm signatures:
- on_insert(key, metadata)
- on_access(key)
- get_score(key)
- select_victims(budget)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from neural_vm.eviction_policy import (
    EvictionPolicy,
    LRUEviction,
    LFUEviction,
    WeightedScoringEviction,
    AdaptiveEviction,
    TransformerEviction,
    create_policy,
)
from neural_vm.vm_step import Token, _SetDim as BD


def test_lru_eviction():
    """Test LRU (Least Recently Used) eviction policy."""
    print("\n" + "=" * 60)
    print("TEST: LRU Eviction Policy")
    print("=" * 60)

    policy = LRUEviction()

    # Insert entries
    policy.on_insert(0, {'token': Token.REG_PC})
    policy.on_insert(1, {'token': Token.REG_AX})
    policy.on_insert(2, {'token': Token.MEM})

    print("\nAfter insertion:")
    print(f"  Entry 0 score (access time): {policy.get_score(0)}")
    print(f"  Entry 1 score (access time): {policy.get_score(1)}")
    print(f"  Entry 2 score (access time): {policy.get_score(2)}")

    # Access entry 0 (make it most recent)
    policy.on_access(0)

    print("\nAfter accessing entry 0:")
    print(f"  Entry 0 score (access time): {policy.get_score(0)}")
    print(f"  Entry 1 score (access time): {policy.get_score(1)}")
    print(f"  Entry 2 score (access time): {policy.get_score(2)}")

    # Select victims (should evict least recently used)
    victims = policy.select_victims(budget=1)

    print(f"\nEviction (budget=1): {victims}")
    print(f"  Expected: [1] (least recently used)")

    assert victims == [1], f"Expected to evict entry 1, got {victims}"

    print("  ✓ LRU eviction correct")

    # Select more victims
    victims = policy.select_victims(budget=2)
    print(f"\nEviction (budget=2): {victims}")
    print(f"  Expected: [1, 2] (least recently used first)")

    assert victims == [1, 2], f"Expected to evict [1, 2], got {victims}"

    print("  ✓ LRU multi-eviction correct")
    print("\n✅ LRU Eviction Policy PASSED")


def test_lfu_eviction():
    """Test LFU (Least Frequently Used) eviction policy."""
    print("\n" + "=" * 60)
    print("TEST: LFU Eviction Policy")
    print("=" * 60)

    policy = LFUEviction()

    # Insert entries
    policy.on_insert(0, {'token': Token.REG_PC})
    policy.on_insert(1, {'token': Token.REG_AX})
    policy.on_insert(2, {'token': Token.MEM})

    print("\nAfter insertion (all frequency=1):")
    print(f"  Entry 0 frequency: {policy.get_score(0)}")
    print(f"  Entry 1 frequency: {policy.get_score(1)}")
    print(f"  Entry 2 frequency: {policy.get_score(2)}")

    # Access entry 0 multiple times
    policy.on_access(0)
    policy.on_access(0)
    policy.on_access(0)

    # Access entry 1 once
    policy.on_access(1)

    print("\nAfter accesses (0→4, 1→2, 2→1):")
    print(f"  Entry 0 frequency: {policy.get_score(0)}")
    print(f"  Entry 1 frequency: {policy.get_score(1)}")
    print(f"  Entry 2 frequency: {policy.get_score(2)}")

    # Select victims (should evict least frequently used)
    victims = policy.select_victims(budget=1)

    print(f"\nEviction (budget=1): {victims}")
    print(f"  Expected: [2] (least frequently used)")

    assert victims == [2], f"Expected to evict entry 2, got {victims}"

    print("  ✓ LFU eviction correct")

    # Select more victims
    victims = policy.select_victims(budget=2)
    print(f"\nEviction (budget=2): {victims}")
    print(f"  Expected: [2, 1] (least frequent first)")

    assert victims == [2, 1], f"Expected to evict [2, 1], got {victims}"

    print("  ✓ LFU multi-eviction correct")
    print("\n✅ LFU Eviction Policy PASSED")


def test_weighted_scoring_eviction():
    """Test weighted scoring eviction policy."""
    print("\n" + "=" * 60)
    print("TEST: Weighted Scoring Eviction Policy")
    print("=" * 60)

    policy = WeightedScoringEviction()

    # Create test embeddings
    embedding_valid = torch.zeros(512)
    embedding_valid[BD.MEM_STORE] = 1.0  # Valid memory

    embedding_invalid = torch.zeros(512)
    embedding_invalid[BD.MEM_STORE] = 0.0  # Overwritten memory

    embedding_pc = torch.zeros(512)

    # Insert entries
    policy.on_insert(0, {
        'token': Token.MEM,
        'embedding': embedding_valid,
        'position': 0
    })

    policy.on_insert(1, {
        'token': Token.MEM,
        'embedding': embedding_invalid,
        'position': 1
    })

    policy.on_insert(2, {
        'token': Token.REG_PC,
        'embedding': embedding_pc,
        'position': 2
    })

    print("\nComputing scores:")

    # Valid MEM: 0 + 312.5 - 600 + 300 = +12.5
    score_0 = policy.get_score(0)
    print(f"  Entry 0 (valid MEM): {score_0:.1f}")
    print(f"    Expected: +12.5 (KEEP)")

    # Overwritten MEM: 0 - 312.5 - 600 + 300 = -612.5
    score_1 = policy.get_score(1)
    print(f"  Entry 1 (invalid MEM): {score_1:.1f}")
    print(f"    Expected: -612.5 (EVICT)")

    # Most recent PC: +50.0
    score_2 = policy.get_score(2)
    print(f"  Entry 2 (PC marker): {score_2:.1f}")
    print(f"    Expected: +50.0 (KEEP)")

    # Verify scores
    assert abs(score_0 - 12.5) < 1.0, f"Valid MEM score wrong: {score_0}"
    assert score_1 < -600, f"Invalid MEM score wrong: {score_1}"
    assert abs(score_2 - 50.0) < 1.0, f"PC marker score wrong: {score_2}"

    print("\n  ✓ Score computation correct")

    # Select victims (threshold = -10.0)
    victims = policy.select_victims(budget=3)

    print(f"\nEviction (threshold=-10.0, budget=3): {victims}")
    print(f"  Expected: [1] (only invalid MEM has score < -10)")

    assert victims == [1], f"Expected to evict only entry 1, got {victims}"

    print("  ✓ Victim selection correct")
    print("\n✅ Weighted Scoring Eviction Policy PASSED")


def test_adaptive_eviction():
    """Test adaptive eviction (combines LRU and LFU)."""
    print("\n" + "=" * 60)
    print("TEST: Adaptive Eviction Policy (ARC-like)")
    print("=" * 60)

    policy = AdaptiveEviction()

    # Insert entries
    policy.on_insert(0, {'token': Token.REG_PC})
    policy.on_insert(1, {'token': Token.REG_AX})
    policy.on_insert(2, {'token': Token.MEM})

    # Entry 0: Access once (recent but not frequent)
    policy.on_access(0)

    # Entry 1: Access multiple times (both recent and frequent)
    policy.on_access(1)
    policy.on_access(1)
    policy.on_access(1)

    # Entry 2: Never accessed (neither recent nor frequent)

    print("\nScores (50% LRU + 50% LFU):")
    print(f"  Entry 0 (recent=4, freq=2): {policy.get_score(0):.1f}")
    print(f"  Entry 1 (recent=7, freq=4): {policy.get_score(1):.1f}")
    print(f"  Entry 2 (recent=3, freq=1): {policy.get_score(2):.1f}")

    # Select victims
    victims = policy.select_victims(budget=1)

    print(f"\nEviction (budget=1): {victims}")
    print(f"  Expected: [2] (lowest combined score)")

    assert victims == [2], f"Expected to evict entry 2, got {victims}"

    print("  ✓ Adaptive eviction correct")
    print("\n✅ Adaptive Eviction Policy PASSED")


def test_policy_factory():
    """Test create_policy factory function."""
    print("\n" + "=" * 60)
    print("TEST: Policy Factory Function")
    print("=" * 60)

    # Test LRU creation
    lru = create_policy("lru")
    assert isinstance(lru, LRUEviction), "LRU policy creation failed"
    print("  ✓ LRU policy created")

    # Test LFU creation
    lfu = create_policy("lfu")
    assert isinstance(lfu, LFUEviction), "LFU policy creation failed"
    print("  ✓ LFU policy created")

    # Test weighted creation
    weighted = create_policy("weighted")
    assert isinstance(weighted, WeightedScoringEviction), "Weighted policy creation failed"
    print("  ✓ Weighted policy created")

    # Test transformer alias
    transformer = create_policy("transformer")
    assert isinstance(transformer, TransformerEviction), "Transformer policy creation failed"
    print("  ✓ Transformer policy created")

    # Test adaptive creation
    adaptive = create_policy("adaptive")
    assert isinstance(adaptive, AdaptiveEviction), "Adaptive policy creation failed"
    print("  ✓ Adaptive policy created")

    # Test with parameters
    weighted_custom = create_policy("weighted", threshold=-15.0)
    assert weighted_custom.threshold == -15.0, "Custom threshold not applied"
    print("  ✓ Custom parameters work")

    # Test invalid policy type
    try:
        create_policy("invalid_type")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unknown policy type" in str(e)
        print("  ✓ Invalid policy type raises error")

    print("\n✅ Policy Factory PASSED")


def test_standard_interface_compliance():
    """Test that all policies implement the standard interface."""
    print("\n" + "=" * 60)
    print("TEST: Standard Interface Compliance")
    print("=" * 60)

    policies = [
        ("LRU", LRUEviction()),
        ("LFU", LFUEviction()),
        ("Weighted", WeightedScoringEviction()),
        ("Adaptive", AdaptiveEviction()),
    ]

    for name, policy in policies:
        print(f"\nTesting {name}:")

        # Check methods exist
        assert hasattr(policy, 'on_insert'), f"{name}: missing on_insert()"
        assert hasattr(policy, 'on_access'), f"{name}: missing on_access()"
        assert hasattr(policy, 'get_score'), f"{name}: missing get_score()"
        assert hasattr(policy, 'select_victims'), f"{name}: missing select_victims()"

        print(f"  ✓ Has all required methods")

        # Check methods are callable
        assert callable(policy.on_insert), f"{name}: on_insert not callable"
        assert callable(policy.on_access), f"{name}: on_access not callable"
        assert callable(policy.get_score), f"{name}: get_score not callable"
        assert callable(policy.select_victims), f"{name}: select_victims not callable"

        print(f"  ✓ All methods are callable")

        # Test basic usage
        if name != "Weighted":  # Weighted requires special metadata
            policy.on_insert(0, {'token': Token.REG_PC})
            policy.on_access(0)
            score = policy.get_score(0)
            victims = policy.select_victims(budget=0)

            assert isinstance(score, (int, float)), f"{name}: score not numeric"
            assert isinstance(victims, list), f"{name}: victims not list"

            print(f"  ✓ Basic operations work")

    print("\n✅ Standard Interface Compliance PASSED")


def test_weighted_scoring_features():
    """Test feature extraction in weighted scoring."""
    print("\n" + "=" * 60)
    print("TEST: Weighted Scoring Feature Extraction")
    print("=" * 60)

    policy = WeightedScoringEviction()

    # Test 1: Most recent marker
    print("\n1. Testing most recent marker detection:")

    embedding = torch.zeros(512)

    # Insert two PC markers
    policy.on_insert(0, {
        'token': Token.REG_PC,
        'embedding': embedding,
        'position': 0
    })

    policy.on_insert(1, {
        'token': Token.REG_PC,
        'embedding': embedding,
        'position': 1
    })

    # Extract features
    features_0 = policy.extract_features(0)
    features_1 = policy.extract_features(1)

    print(f"  Entry 0 (old PC): is_most_recent={features_0.get('is_most_recent', 0)}")
    print(f"  Entry 1 (new PC): is_most_recent={features_1.get('is_most_recent', 0)}")

    # Entry 1 should be most recent
    assert features_1.get('is_most_recent', 0) == 1.0, "Entry 1 should be most recent"
    assert features_0.get('is_old', 0) == 1.0, "Entry 0 should be old"

    print("  ✓ Most recent marker detection works")

    # Test 2: MEM_STORE validity
    print("\n2. Testing MEM_STORE validity:")

    valid_embedding = torch.zeros(512)
    valid_embedding[BD.MEM_STORE] = 1.0

    policy.on_insert(10, {
        'token': Token.MEM,
        'embedding': valid_embedding,
        'position': 10
    })

    features = policy.extract_features(10)

    print(f"  mem_store_valid: {features.get('mem_store_valid', 0)}")
    print(f"  mem_store_invalid: {features.get('mem_store_invalid', 0)}")

    assert features.get('mem_store_valid', 0) == 1.0, "Should detect valid memory"
    assert features.get('mem_store_invalid', 0) == 0.0, "Should not be invalid"

    print("  ✓ MEM_STORE validity detection works")

    # Test 3: ADDR_KEY importance
    print("\n3. Testing ADDR_KEY importance:")

    addr_embedding = torch.zeros(512)
    addr_embedding[BD.ADDR_KEY + 5] = 0.5  # Non-zero addr key

    policy.on_insert(20, {
        'token': Token.MEM,
        'embedding': addr_embedding,
        'position': 20
    })

    features = policy.extract_features(20)

    print(f"  has_addr_key: {features.get('has_addr_key', 0)}")

    assert features.get('has_addr_key', 0) == 1.0, "Should detect addr key"

    print("  ✓ ADDR_KEY detection works")

    print("\n✅ Feature Extraction PASSED")


def run_all_tests():
    """Run all test cases."""
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "EVICTION POLICY TEST SUITE" + " " * 22 + "║")
    print("╚" + "═" * 58 + "╝")

    tests = [
        ("LRU Eviction", test_lru_eviction),
        ("LFU Eviction", test_lfu_eviction),
        ("Weighted Scoring", test_weighted_scoring_eviction),
        ("Adaptive Eviction", test_adaptive_eviction),
        ("Policy Factory", test_policy_factory),
        ("Interface Compliance", test_standard_interface_compliance),
        ("Feature Extraction", test_weighted_scoring_features),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n❌ {name} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"\n❌ {name} ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"  Passed: {passed}/{len(tests)}")
    print(f"  Failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\n" + "╔" + "═" * 58 + "╗")
        print("║" + " " * 17 + "ALL TESTS PASSED ✓" + " " * 22 + "║")
        print("╚" + "═" * 58 + "╝")
    else:
        print(f"\n⚠️  {failed} test(s) failed")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
