#!/usr/bin/env python3
"""
HuggingFace-compatible interface for C4 Transformer VM.

This wraps the transformer VM in a format compatible with HuggingFace's
model interfaces and provides a simple "chat" interface for running C programs.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from dataclasses import dataclass

from c4_byte_to_nibble import (
    C4ByteNibbleVM, ByteEncoder, ByteDecoder,
    SwiGLUMul, DivisionFFN, BitwiseOps, ByteAddFFN
)
from c4_compiler_full import compile_c


@dataclass
class C4Config:
    """Configuration for C4 Transformer VM (HuggingFace style)."""
    model_type: str = "c4_transformer_vm"
    vocab_size: int = 256  # Byte tokens
    hidden_size: int = 256  # 4 bytes × 256 one-hot
    num_hidden_layers: int = 1
    intermediate_size: int = 256
    max_position_embeddings: int = 1024

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_type,
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "intermediate_size": self.intermediate_size,
            "max_position_embeddings": self.max_position_embeddings,
        }


class C4TransformerForExecution(nn.Module):
    """
    HuggingFace-style model wrapper for C4 Transformer VM.

    This model takes C source code as input and returns the execution result.
    All arithmetic is performed by transformer components (FFN, SwiGLU).
    """

    def __init__(self, config: Optional[C4Config] = None):
        super().__init__()
        self.config = config or C4Config()

        # Core transformer components
        self.vm = C4ByteNibbleVM()

        # Expose components for inspection
        self.encoder = self.vm.enc
        self.decoder = self.vm.dec
        self.multiply = self.vm.mul_ffn
        self.divide = self.vm.div_ffn
        self.add = self.vm.add_ffn
        self.bitwise = self.vm.bitwise

    def forward(self, source_code: str, max_steps: int = 100000) -> Dict[str, Any]:
        """
        Execute C source code and return result.

        Args:
            source_code: C program as string
            max_steps: Maximum execution steps

        Returns:
            Dict with 'result', 'bytecode_length', 'steps_executed'
        """
        # Compile
        try:
            bytecode, data = compile_c(source_code)
        except Exception as e:
            return {"error": str(e), "result": None}

        # Execute on transformer VM
        self.vm.reset()
        self.vm.load_bytecode(bytecode, data)

        steps = 0
        while steps < max_steps:
            if not self.vm.step():
                break
            steps += 1

        result = self.vm.ax_int()

        return {
            "result": result,
            "bytecode_length": len(bytecode),
            "steps_executed": steps,
        }

    def compute(self, a: int, b: int, op: str) -> int:
        """Direct computation without compilation."""
        if op == "add":
            a_emb = self.encoder(a)
            b_emb = self.encoder(b)
            r_emb = self.add(a_emb, b_emb)
            return self.decoder(r_emb)
        elif op == "mul":
            return int(self.multiply(
                torch.tensor(float(a)),
                torch.tensor(float(b))
            ).round().item())
        elif op == "div":
            return self.divide(a, b)
        elif op == "and":
            a_emb = self.encoder(a)
            b_emb = self.encoder(b)
            r_emb = self.bitwise(a_emb, b_emb, 'and')
            return self.decoder(r_emb)
        else:
            raise ValueError(f"Unknown op: {op}")

    def save_pretrained(self, path: str):
        """Save model (HuggingFace style)."""
        import json
        import os
        os.makedirs(path, exist_ok=True)

        # Save config
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # Save weights
        torch.save(self.state_dict(), os.path.join(path, "pytorch_model.bin"))

    @classmethod
    def from_pretrained(cls, path: str) -> "C4TransformerForExecution":
        """Load model (HuggingFace style)."""
        import json
        import os

        with open(os.path.join(path, "config.json")) as f:
            config_dict = json.load(f)

        config = C4Config(**config_dict)
        model = cls(config)

        state_dict = torch.load(os.path.join(path, "pytorch_model.bin"))
        model.load_state_dict(state_dict)

        return model


def chat_interface():
    """Simple chat interface for running C programs."""

    print("=" * 60)
    print("  C4 Transformer VM - Chat Interface")
    print("=" * 60)
    print()
    print("Enter C programs to execute. The transformer will run them.")
    print("Type 'quit' to exit, 'demo' for examples.")
    print()

    model = C4TransformerForExecution()

    demos = [
        ("6 * 7", "int main() { return 6 * 7; }"),
        ("Fibonacci(10)", """
            int fib(int n) {
                if (n < 2) return n;
                return fib(n-1) + fib(n-2);
            }
            int main() { return fib(10); }
        """),
        ("100 / 7", "int main() { return 100 / 7; }"),
        ("Loop sum", """
            int main() {
                int i, sum;
                sum = 0; i = 0;
                while (i < 10) { sum = sum + i; i = i + 1; }
                return sum;
            }
        """),
    ]

    while True:
        try:
            user_input = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'demo':
            print("\nRunning demos:")
            for name, code in demos:
                result = model(code)
                print(f"  {name}: {result['result']}")
            continue
        elif not user_input:
            continue

        # Try to parse as simple expression
        if not user_input.startswith('int'):
            # Wrap in main()
            user_input = f"int main() {{ return {user_input}; }}"

        result = model(user_input)

        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Result: {result['result']}")
            print(f"  ({result['bytecode_length']} instructions, {result['steps_executed']} steps)")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--chat":
        chat_interface()
    else:
        # Demo
        print("=" * 60)
        print("  C4 Transformer VM - HuggingFace Interface Demo")
        print("=" * 60)

        model = C4TransformerForExecution()

        print(f"\nModel config: {model.config.to_dict()}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
        print(f"Buffers: {sum(b.numel() for b in model.buffers())}")

        # Test forward pass
        print("\n--- Execution Tests ---")

        tests = [
            ("6 * 7", "int main() { return 6 * 7; }", 42),
            ("100 / 7", "int main() { return 100 / 7; }", 14),
            ("Fibonacci(10)", """
                int fib(int n) {
                    if (n < 2) return n;
                    return fib(n-1) + fib(n-2);
                }
                int main() { return fib(10); }
            """, 55),
        ]

        for name, code, expected in tests:
            result = model(code)
            status = "✓" if result["result"] == expected else "✗"
            print(f"  {name}: {result['result']} (expected {expected}) {status}")

        # Test direct compute
        print("\n--- Direct Compute Tests ---")
        print(f"  6 * 7 = {model.compute(6, 7, 'mul')}")
        print(f"  100 + 50 = {model.compute(100, 50, 'add')}")
        print(f"  100 / 7 = {model.compute(100, 7, 'div')}")
        print(f"  0xFF & 0x0F = {model.compute(0xFF, 0x0F, 'and')}")

        # Save/load test
        print("\n--- Save/Load Test ---")
        model.save_pretrained("/tmp/c4_model")
        print("  Saved to /tmp/c4_model")

        loaded = C4TransformerForExecution.from_pretrained("/tmp/c4_model")
        result = loaded("int main() { return 123 * 456; }")
        print(f"  Loaded model: 123 * 456 = {result['result']}")

        print("\n--- Chat Interface ---")
        print("Run with --chat flag for interactive mode:")
        print("  python c4_hf_interface.py --chat")
