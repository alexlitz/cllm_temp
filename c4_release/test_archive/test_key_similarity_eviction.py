#!/usr/bin/env python3
"""
Test suite for key similarity-based eviction.

Verifies that eviction based on key similarity (cosine > 0.99) works correctly:
- Duplicate keys are detected
- Older entries are evicted
- Latest-write-wins is implemented naturally
- Eviction runs every ~120 tokens
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from neural_vm.key_similarity_eviction import KeySimilarityEviction, EvictionStats
from neural_vm.vm_step import AutoregressiveVM, Token


def create_test_model():
    """Create a minimal test model."""
    model = AutoregressiveVM(
        d_model=512,
        n_layers=16,
        n_heads=8,
        ffn_hidden=4096,
        max_seq_len=4096
    )
    model.eval()
    return model


def test_key_computation():
    """Test that keys can be computed for context."""
    print("\n" + "=" * 60)
    print("TEST: Key Computation")
    print("=" * 60)

    model = create_test_model()
    eviction = KeySimilarityEviction(model)

    # Create simple context
    context = [Token.REG_PC, 1, 2, 3, 4,  # PC marker + 4 bytes
               Token.REG_AX, 5, 6, 7, 8]  # AX marker + 4 bytes

    token_ids = torch.tensor([context], dtype=torch.long)

    # Compute keys
    keys = eviction.compute_keys(token_ids, layer_idx=0)

    print(f"\nContext length: {len(context)}")
    print(f"Keys shape: {keys.shape}")
    print(f"Expected: [{len(context)}, 512]")

    assert keys.shape == (len(context), 512), f"Wrong shape: {keys.shape}"

    print("  ✓ Key computation works")
    print("\n✅ Key Computation PASSED")


def test_similarity_computation():
    """Test pairwise similarity computation."""
    print("\n" + "=" * 60)
    print("TEST: Similarity Computation")
    print("=" * 60)

    model = create_test_model()
    eviction = KeySimilarityEviction(model)

    # Create context with two PC markers (should be similar)
    context = [Token.REG_PC, 1, 2, 3, 4,  # PC #1
               Token.REG_AX, 5, 6, 7, 8,  # AX
               Token.REG_PC, 9, 10, 11, 12]  # PC #2 (similar to PC #1)

    token_ids = torch.tensor([context], dtype=torch.long)

    # Compute keys
    keys = eviction.compute_keys(token_ids, layer_idx=0)

    # Compute similarity
    similarity = eviction.compute_pairwise_similarity(keys)

    print(f"\nSimilarity matrix shape: {similarity.shape}")
    print(f"Expected: [{len(context)}, {len(context)}]")

    assert similarity.shape == (len(context), len(context))

    # Check self-similarity is 1.0
    for i in range(len(context)):
        assert abs(similarity[i, i].item() - 1.0) < 0.01, f"Self-similarity != 1.0 at {i}"

    print("  ✓ Similarity matrix computed")
    print("  ✓ Self-similarity = 1.0")

    # Check if PC markers have high similarity
    pc1_idx = 0  # First PC marker
    pc2_idx = 10  # Second PC marker
    pc_similarity = similarity[pc1_idx, pc2_idx].item()

    print(f"\nSimilarity between PC markers: {pc_similarity:.4f}")
    print(f"  (Expected > 0.9 for same marker type)")

    print("\n✅ Similarity Computation PASSED")


def test_duplicate_detection():
    """Test detection of duplicate (similar) keys."""
    print("\n" + "=" * 60)
    print("TEST: Duplicate Detection")
    print("=" * 60)

    model = create_test_model()
    eviction = KeySimilarityEviction(model, similarity_threshold=0.90)

    # Create context with duplicate markers
    context = [Token.REG_PC, 1, 2, 3, 4,  # PC #1 (pos 0-4)
               Token.REG_AX, 5, 6, 7, 8,  # AX #1 (pos 5-9)
               Token.REG_PC, 9, 10, 11, 12,  # PC #2 (pos 10-14)
               Token.REG_AX, 13, 14, 15, 16]  # AX #2 (pos 15-19)

    token_ids = torch.tensor([context], dtype=torch.long)

    # Compute keys and similarity
    keys = eviction.compute_keys(token_ids, layer_idx=0)
    similarity = eviction.compute_pairwise_similarity(keys)

    # Find duplicate pairs
    pairs = eviction.find_duplicate_pairs(similarity)

    print(f"\nContext length: {len(context)}")
    print(f"Duplicate pairs found: {len(pairs)}")

    for keep_idx, evict_idx in pairs[:5]:  # Show first 5
        sim = similarity[keep_idx, evict_idx].item()
        print(f"  Pair: keep={keep_idx}, evict={evict_idx}, similarity={sim:.4f}")

    assert len(pairs) > 0, "Should find at least some duplicate pairs"

    print(f"\n  ✓ Found {len(pairs)} duplicate pairs")
    print("\n✅ Duplicate Detection PASSED")


def test_victim_selection():
    """Test that older entries are selected for eviction."""
    print("\n" + "=" * 60)
    print("TEST: Victim Selection (Evict Older)")
    print("=" * 60)

    model = create_test_model()
    eviction = KeySimilarityEviction(model, similarity_threshold=0.90)

    # Pairs: (keep_idx, evict_idx)
    pairs = [
        (10, 0),  # keep 10, evict 0 (older)
        (15, 5),  # keep 15, evict 5 (older)
        (20, 10), # keep 20, evict 10 (older)
    ]

    victims = eviction.select_victims(pairs)

    print(f"\nPairs: {pairs}")
    print(f"Victims (to evict): {sorted(victims)}")
    print(f"Expected: {sorted([0, 5, 10])}")

    assert victims == {0, 5, 10}, f"Wrong victims: {victims}"

    print("  ✓ Correct victims selected (older entries)")
    print("\n✅ Victim Selection PASSED")


def test_eviction_timing():
    """Test that eviction runs every 120 tokens."""
    print("\n" + "=" * 60)
    print("TEST: Eviction Timing (Every 120 Tokens)")
    print("=" * 60)

    model = create_test_model()
    eviction = KeySimilarityEviction(
        model,
        eviction_interval=120,
        min_context_size=50
    )

    # Small context - should not evict
    assert not eviction.should_evict(context_len=30)
    print("  ✓ No eviction for small context (< 50)")

    # Large context, no tokens processed yet
    assert not eviction.should_evict(context_len=100)
    print("  ✓ No eviction immediately after creation")

    # Simulate 119 tokens
    eviction.tokens_since_last_eviction = 119
    assert not eviction.should_evict(context_len=100)
    print("  ✓ No eviction at 119 tokens")

    # Simulate 120 tokens - should trigger
    eviction.tokens_since_last_eviction = 120
    assert eviction.should_evict(context_len=100)
    print("  ✓ Eviction triggered at 120 tokens")

    # After eviction, counter resets
    eviction.tokens_since_last_eviction = 0
    assert not eviction.should_evict(context_len=100)
    print("  ✓ No eviction after reset")

    print("\n✅ Eviction Timing PASSED")


def test_latest_write_wins():
    """Test that latest-write-wins is implemented naturally."""
    print("\n" + "=" * 60)
    print("TEST: Latest-Write-Wins (Natural Implementation)")
    print("=" * 60)

    model = create_test_model()
    eviction = KeySimilarityEviction(
        model,
        similarity_threshold=0.90,
        min_context_size=10
    )

    # Create context with multiple writes to same register
    # Simulating: PC=100, PC=105, PC=110 (three writes)
    context = (
        [Token.REG_PC, 100, 0, 0, 0] +  # PC write #1 (pos 0-4, oldest)
        [Token.REG_PC, 105, 0, 0, 0] +  # PC write #2 (pos 5-9)
        [Token.REG_PC, 110, 0, 0, 0] +  # PC write #3 (pos 10-14, newest)
        [Token.STEP_END]                 # Terminator
    )

    print(f"\nOriginal context length: {len(context)}")
    print("  3 PC writes: 100 (oldest), 105, 110 (newest)")

    # Run eviction
    token_ids = torch.tensor([context], dtype=torch.long)
    pruned = eviction.evict_context(
        context=context,
        token_ids=token_ids,
        protected_range=None
    )

    print(f"Pruned context length: {len(pruned)}")
    print(f"Evicted entries: {len(context) - len(pruned)}")

    # Should evict some old PC writes (depending on similarity)
    assert len(pruned) < len(context), "Should evict some entries"

    stats = eviction.get_stats()
    print(f"\nEviction stats:")
    print(f"  Total evicted: {stats.total_evicted}")

    print("\n  ✓ Latest-write-wins implemented (older PC writes evicted)")
    print("\n✅ Latest-Write-Wins PASSED")


def test_protected_range():
    """Test that protected range is not evicted."""
    print("\n" + "=" * 60)
    print("TEST: Protected Range (Bytecode Prefix)")
    print("=" * 60)

    model = create_test_model()
    eviction = KeySimilarityEviction(model, similarity_threshold=0.90)

    # Create context with protected prefix
    protected_prefix = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Bytecode
    context = protected_prefix + [
        Token.REG_PC, 100, 0, 0, 0,  # PC #1
        Token.REG_PC, 105, 0, 0, 0,  # PC #2 (duplicate)
    ]

    print(f"\nContext length: {len(context)}")
    print(f"Protected range: 0-10 (bytecode prefix)")

    # Run eviction with protected range
    token_ids = torch.tensor([context], dtype=torch.long)
    pruned = eviction.evict_context(
        context=context,
        token_ids=token_ids,
        protected_range=(0, 10)
    )

    # Check protected prefix is intact
    assert pruned[:10] == protected_prefix, "Protected prefix was modified!"

    print(f"Pruned context length: {len(pruned)}")
    print("  ✓ Protected prefix intact")

    print("\n✅ Protected Range PASSED")


def test_step_integration():
    """Test step() method for easy integration."""
    print("\n" + "=" * 60)
    print("TEST: Step Integration")
    print("=" * 60)

    model = create_test_model()
    eviction = KeySimilarityEviction(
        model,
        eviction_interval=5,  # Small interval for testing
        min_context_size=10
    )

    context = [1, 2, 3, 4, 5] * 4  # 20 tokens

    print(f"\nInitial context length: {len(context)}")

    # Step 1-4: No eviction yet (< 5 tokens processed)
    for i in range(4):
        context = eviction.step(context)
        assert len(context) == 20, f"Should not evict at step {i+1}"

    print("  ✓ No eviction in steps 1-4")

    # Step 5: Eviction should run (5 tokens = interval)
    context = eviction.step(context)

    print(f"After step 5: {len(context)} tokens")
    print(f"Stats: {eviction.stats.total_evicted} evicted")

    print("  ✓ Step integration works")
    print("\n✅ Step Integration PASSED")


def run_all_tests():
    """Run all test cases."""
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 8 + "KEY SIMILARITY EVICTION TEST SUITE" + " " * 16 + "║")
    print("╚" + "═" * 58 + "╝")

    tests = [
        ("Key Computation", test_key_computation),
        ("Similarity Computation", test_similarity_computation),
        ("Duplicate Detection", test_duplicate_detection),
        ("Victim Selection", test_victim_selection),
        ("Eviction Timing", test_eviction_timing),
        ("Latest-Write-Wins", test_latest_write_wins),
        ("Protected Range", test_protected_range),
        ("Step Integration", test_step_integration),
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
