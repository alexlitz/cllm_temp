#!/usr/bin/env python3
"""
Save trained model weights for C4 Transformer VM.

This script saves:
1. Transformer VM weights (all FFN tables)
2. Speculator (FastLogicalVM doesn't need weights)
3. HuggingFace-compatible model package
"""

import sys
import os
import torch
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.transformer_vm import C4TransformerVM, C4Config
from src.speculator import FastLogicalVM
from src.huggingface import C4HFModel, C4HFConfig


def save_models(models_dir: str = "models"):
    """Save all model weights."""
    print("=" * 60)
    print("SAVING C4 TRANSFORMER VM MODELS")
    print("=" * 60)

    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    # 1. Save raw transformer VM weights
    print("\n1. Saving Transformer VM weights...")
    vm = C4TransformerVM()

    # Get parameter count
    num_buffers = sum(b.numel() for b in vm.buffers())
    print(f"   Buffers: {num_buffers:,}")
    print(f"   Memory: {num_buffers * 4 / 1024:.1f} KB (float32)")

    # Save state dict
    state_dict_path = models_path / "transformer_vm_state_dict.pt"
    torch.save(vm.state_dict(), state_dict_path)
    print(f"   Saved: {state_dict_path}")

    # Save as single file
    full_model_path = models_path / "transformer_vm.pt"
    torch.save(vm, full_model_path)
    print(f"   Saved: {full_model_path}")

    # 2. Save HuggingFace-compatible model
    print("\n2. Saving HuggingFace-compatible model...")
    hf_model = C4HFModel()
    hf_path = models_path / "huggingface"
    hf_model.save_pretrained(hf_path)
    print(f"   Saved: {hf_path}")

    # Verify we can load it back
    loaded_model = C4HFModel.from_pretrained(hf_path)
    print(f"   Verified: Load successful")

    # 3. Save configuration
    print("\n3. Saving configuration...")
    config = {
        "model_type": "c4-transformer-vm",
        "version": "1.0.0",
        "architecture": {
            "vocab_size": 256,
            "hidden_dim": 256,
            "word_size": 8,
            "table_bits": 8,
            "newton_iterations": 2,
        },
        "components": {
            "byte_encoder": "ByteEncoder - int to 4 one-hot bytes",
            "byte_decoder": "ByteDecoder - 4 one-hot bytes to int",
            "byte_to_nibble": "ByteToNibbleFFN - byte to 2 nibbles via FFN",
            "nibble_to_byte": "NibbleToByteFFN - 2 nibbles to byte via FFN",
            "bitwise_ops": "BitwiseOps - AND/OR/XOR via nibble tables",
            "addition": "ByteAddFFN - ripple carry via nibble add tables",
            "multiply": "SwiGLUMul - exact via silu(a)*b + silu(-a)*(-b)",
            "division": "DivisionFFN - reciprocal table + Newton iterations",
            "compare": "CompareFFN - sharp gate comparisons",
        },
        "statistics": {
            "num_buffers": num_buffers,
            "memory_kb_fp32": num_buffers * 4 / 1024,
            "memory_kb_fp16": num_buffers * 2 / 1024,
        }
    }

    config_path = models_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"   Saved: {config_path}")

    # 4. Test the saved model
    print("\n4. Testing saved model...")
    from src.compiler import compile_c

    # Load the model (use weights_only=False for full model)
    loaded_vm = torch.load(full_model_path, weights_only=False)

    # Test Fibonacci
    fib_source = """
    int fib(int n) {
        if (n < 2) return n;
        return fib(n-1) + fib(n-2);
    }
    int main() { return fib(10); }
    """

    bytecode, data = compile_c(fib_source)
    loaded_vm.reset()
    loaded_vm.load_bytecode(bytecode, data)
    result = loaded_vm.run()

    if result == 55:
        print(f"   fib(10) = {result} PASS")
    else:
        print(f"   fib(10) = {result} FAIL (expected 55)")

    # Summary
    print("\n" + "=" * 60)
    print("SAVED FILES:")
    print("=" * 60)
    for f in models_path.rglob("*"):
        if f.is_file():
            size = f.stat().st_size
            if size > 1024*1024:
                size_str = f"{size/1024/1024:.1f} MB"
            elif size > 1024:
                size_str = f"{size/1024:.1f} KB"
            else:
                size_str = f"{size} B"
            print(f"  {f.relative_to(models_path)}: {size_str}")


if __name__ == "__main__":
    save_models()
