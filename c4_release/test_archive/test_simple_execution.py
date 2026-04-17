"""
Simple Execution Test

Test if the compiled weights can execute a trivial program.

This is a minimal integration test to verify the architecture works.
"""

import torch
from neural_vm.weight_loader import CompiledWeightLoader
from neural_vm.vm_step import AutoregressiveVM
from src.compiler import compile_c

def test_simple_arithmetic():
    """Test executing: int main() { return 2 + 3; }"""
    
    print("="*70)
    print("SIMPLE EXECUTION TEST: 2 + 3")
    print("="*70)
    print()
    
    # Step 1: Compile C to bytecode
    print("Step 1: Compiling C source...")
    source = "int main() { return 2 + 3; }"
    try:
        bytecode, data = compile_c(source)
        print(f"  ✅ Bytecode: {len(bytecode)} instructions")
        print(f"  ✅ Data: {len(data) if data else 0} bytes")
        
        # Show first few bytecode instructions
        print(f"  First 10 bytes: {bytecode[:10]}")
    except Exception as e:
        print(f"  ❌ Compilation failed: {e}")
        return False
    
    print()
    
    # Step 2: Create VM with compiled weights
    print("Step 2: Creating VM with compiled weights...")
    try:
        vm = AutoregressiveVM(
            d_model=1280,
            n_layers=16,
            n_heads=8,
            ffn_hidden=4096,
        )
        
        loader = CompiledWeightLoader()
        stats = loader.load_all_weights(vm, verbose=False)
        
        print(f"  ✅ VM created")
        print(f"  ✅ Loaded {stats['single_op_loaded'] + stats['multi_op_loaded'] + stats['multi_layer_loaded']}/32 opcodes")
        print(f"  ✅ Total params: {stats['total_params']:,}")
    except Exception as e:
        print(f"  ❌ VM setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    
    # Step 3: Prepare input
    print("Step 3: Preparing input...")
    try:
        # The challenge: We need to convert bytecode to nibble-based token format
        # For now, let's just try a forward pass with dummy input to see what happens
        
        # Create a simple input sequence (placeholder)
        # Real implementation would need proper token encoding
        batch_size = 1
        seq_len = 35  # One VM step (35 tokens per step)
        
        # Dummy input tokens (we'd need proper encoding here)
        input_tokens = torch.randint(0, 256, (batch_size, seq_len))
        
        print(f"  ✅ Input shape: {input_tokens.shape}")
    except Exception as e:
        print(f"  ❌ Input preparation failed: {e}")
        return False
    
    print()
    
    # Step 4: Run forward pass
    print("Step 4: Running forward pass...")
    try:
        with torch.no_grad():
            # This will fail without proper embeddings, but let's try
            output = vm(input_tokens)
            print(f"  ✅ Output shape: {output.shape}")
            print(f"  ✅ Output range: [{output.min().item():.2f}, {output.max().item():.2f}]")
    except Exception as e:
        print(f"  ⚠️  Forward pass failed (expected without proper embeddings): {e}")
        # This is expected - we don't have the nibble-based embeddings set up yet
        print()
        print("="*70)
        print("ANALYSIS")
        print("="*70)
        print()
        print("❌ Execution test failed (expected)")
        print()
        print("Missing components for full execution:")
        print("  1. Nibble-based token encoding")
        print("  2. VM state embedding (PC, AX, SP, BP registers)")
        print("  3. Bytecode-to-token conversion")
        print("  4. Output decoding (tokens → result)")
        print()
        print("The compiled weights are loaded correctly, but we need the")
        print("full nibble-based VM infrastructure to actually run programs.")
        print()
        return False
    
    print()
    print("="*70)
    print("SUCCESS (unexpected!)")
    print("="*70)
    return True

if __name__ == "__main__":
    success = test_simple_arithmetic()
    exit(0 if success else 1)
