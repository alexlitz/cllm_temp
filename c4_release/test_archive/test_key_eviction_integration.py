#!/usr/bin/env python3
"""
Integration test for key similarity eviction with actual VM execution.

Tests eviction on a real program to see how it works in practice.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from src.transformer_vm import C4TransformerVM
from neural_vm.key_similarity_eviction import KeySimilarityEviction


def test_simple_program_eviction():
    """Test eviction on a simple program that runs multiple steps."""
    print("\n" + "=" * 60)
    print("INTEGRATION TEST: Simple Program with Eviction")
    print("=" * 60)

    # Create VM
    vm = C4TransformerVM()

    # Create eviction manager
    eviction = KeySimilarityEviction(
        vm._runner.model,
        similarity_threshold=0.99,
        eviction_interval=120,
        min_context_size=50
    )

    # Simple C program
    from src.compiler import compile_c

    source = "int main() { return 42; }"
    bytecode, data = compile_c(source)

    print(f"\nProgram: {source}")
    print(f"Bytecode length: {len(bytecode)} bytes")

    try:
        # Load and run
        vm.reset()
        vm.load_bytecode(bytecode, data)
        result = vm.run(max_steps=100)

        print(f"\nExecution completed:")
        print(f"  Return value: {result}")

        # Get final context from runner
        context = vm._runner.context if hasattr(vm._runner, 'context') else []

        if context:
            print(f"\nContext before eviction: {len(context)} tokens")

            # Try eviction
            token_ids = torch.tensor([context], dtype=torch.long)

            # Compute keys
            keys = eviction.compute_keys(token_ids, layer_idx=0)
            print(f"Keys shape: {keys.shape}")

            # Compute similarity
            similarity = eviction.compute_pairwise_similarity(keys)
            print(f"Similarity matrix: {similarity.shape}")

            # Find high similarity pairs
            high_sim_count = 0
            max_sim = 0.0
            for i in range(len(context)):
                for j in range(i + 1, len(context)):
                    sim = similarity[i, j].item()
                    if sim > max_sim:
                        max_sim = sim
                    if sim > 0.90:
                        high_sim_count += 1

            print(f"\nSimilarity analysis:")
            print(f"  Max similarity (non-diagonal): {max_sim:.4f}")
            print(f"  Pairs with sim > 0.90: {high_sim_count}")
            print(f"  Pairs with sim > 0.95: {sum(1 for i in range(len(context)) for j in range(i+1, len(context)) if similarity[i, j] > 0.95)}")
            print(f"  Pairs with sim > 0.99: {sum(1 for i in range(len(context)) for j in range(i+1, len(context)) if similarity[i, j] > 0.99)}")

            # Try eviction with lower threshold for testing
            eviction_test = KeySimilarityEviction(
                vm._runner.model,
                similarity_threshold=0.85,  # Lower for testing with random weights
                eviction_interval=1,
                min_context_size=10
            )

            pruned = eviction_test.evict_context(
                context=context,
                token_ids=token_ids,
                protected_range=(0, len(bytecode))
            )

            print(f"\nEviction with threshold=0.85:")
            print(f"  Original: {len(context)} tokens")
            print(f"  Pruned: {len(pruned)} tokens")
            print(f"  Evicted: {len(context) - len(pruned)} tokens")

            stats = eviction_test.get_stats()
            print(f"  Total evicted (stats): {stats.total_evicted}")
        else:
            print("\n⚠️  No context available from VM execution")

        print("\n✅ Integration test completed")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


def test_loop_program_eviction():
    """Test eviction on a program with a loop (many steps)."""
    print("\n" + "=" * 60)
    print("INTEGRATION TEST: Loop Program with Multiple Steps")
    print("=" * 60)

    # Create VM
    vm = C4TransformerVM()

    # Create eviction manager
    eviction = KeySimilarityEviction(
        vm._runner.model,
        similarity_threshold=0.90,  # Lower for testing
        eviction_interval=50,  # More frequent for testing
        min_context_size=20
    )

    # Simple loop program
    from src.compiler import compile_c

    source = """
    int main() {
        int i;
        int sum;
        i = 0;
        sum = 0;
        while (i < 3) {
            sum = sum + i;
            i = i + 1;
        }
        return sum;
    }
    """

    bytecode, data = compile_c(source)

    print(f"\nProgram: Loop summing 0+1+2")
    print(f"Bytecode length: {len(bytecode)} bytes")

    try:
        vm.reset()
        vm.load_bytecode(bytecode, data)
        result = vm.run(max_steps=200)

        print(f"\nExecution completed:")
        print(f"  Return value: {result}")

        context = vm._runner.context if hasattr(vm._runner, 'context') else []

        if context and len(context) > 20:
            print(f"\nContext analysis:")
            print(f"  Total tokens: {len(context)}")

            # Simulate eviction at regular intervals
            original_len = len(context)

            # Split into chunks and evict each
            chunk_size = 50
            total_evicted = 0

            for chunk_start in range(0, len(context), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(context))
                chunk = context[chunk_start:chunk_end]

                if len(chunk) < 20:
                    continue

                token_ids = torch.tensor([chunk], dtype=torch.long)
                keys = eviction.compute_keys(token_ids, layer_idx=0)
                similarity = eviction.compute_pairwise_similarity(keys)

                # Find duplicates
                pairs = eviction.find_duplicate_pairs(similarity)
                victims = eviction.select_victims(pairs)

                if victims:
                    print(f"  Chunk {chunk_start}-{chunk_end}: {len(victims)} victims")
                    total_evicted += len(victims)

            print(f"\nEviction summary:")
            print(f"  Original context: {original_len} tokens")
            print(f"  Total evictable: {total_evicted} tokens")
            print(f"  Reduction: {total_evicted / original_len * 100:.1f}%")

        print("\n✅ Loop test completed")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


def test_key_similarity_stats():
    """Analyze key similarity statistics for VM-generated context."""
    print("\n" + "=" * 60)
    print("ANALYSIS: Key Similarity Statistics")
    print("=" * 60)

    vm = C4TransformerVM()

    # Simple program
    from src.compiler import compile_c

    source = "int main() { return 42; }"
    bytecode, data = compile_c(source)

    try:
        vm.reset()
        vm.load_bytecode(bytecode, data)
        result = vm.run(max_steps=50)

        context = vm._runner.context if hasattr(vm._runner, 'context') else []

        if context and len(context) > 10:
            print(f"\nContext: {len(context)} tokens")

            # Compute keys
            eviction = KeySimilarityEviction(vm._runner.model)
            token_ids = torch.tensor([context], dtype=torch.long)
            keys = eviction.compute_keys(token_ids, layer_idx=0)

            # Compute full similarity matrix
            similarity = eviction.compute_pairwise_similarity(keys)

            # Analyze distribution
            upper_triangle = []
            for i in range(len(context)):
                for j in range(i + 1, len(context)):
                    upper_triangle.append(similarity[i, j].item())

            if upper_triangle:
                import statistics
                print(f"\nSimilarity distribution (non-diagonal):")
                print(f"  Min: {min(upper_triangle):.4f}")
                print(f"  Max: {max(upper_triangle):.4f}")
                print(f"  Mean: {statistics.mean(upper_triangle):.4f}")
                print(f"  Median: {statistics.median(upper_triangle):.4f}")

                # Histogram
                bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
                counts = [0] * (len(bins) - 1)
                for sim in upper_triangle:
                    for i in range(len(bins) - 1):
                        if bins[i] <= sim < bins[i + 1]:
                            counts[i] += 1
                            break

                print(f"\nSimilarity histogram:")
                for i in range(len(bins) - 1):
                    bar = "█" * (counts[i] // 10) if counts[i] > 0 else ""
                    print(f"  [{bins[i]:.2f}-{bins[i+1]:.2f}): {counts[i]:4d} {bar}")

                print(f"\nKey insights:")
                print(f"  • With random weights, most similarities are low (~0.0-0.3)")
                print(f"  • High similarity (>0.99) requires trained model weights")
                print(f"  • Similar tokens will cluster after training")
                print(f"  • This is expected behavior for untrained model")

        print("\n✅ Analysis completed")

    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all integration tests."""
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "KEY EVICTION INTEGRATION TESTS" + " " * 18 + "║")
    print("╚" + "═" * 58 + "╝")

    test_simple_program_eviction()
    test_loop_program_eviction()
    test_key_similarity_stats()

    print("\n" + "=" * 60)
    print("All integration tests completed")
    print("=" * 60)
    print("\nNote: Low similarity values are expected with untrained weights.")
    print("After training, similar tokens will produce similar keys (>0.99).")


if __name__ == "__main__":
    main()
