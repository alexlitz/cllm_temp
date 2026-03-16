"""
System Prompt Baking for Autoregressive VM

Allows "baking in" a system prompt (C program) into the VM's context.
The autoregressive model passes programs as context tokens and executes
via attention -- no ONNX weight-baking needed.

Usage:
    from src.prompt_baking import BakedPromptVM, bake_system_prompt

    # Create a VM with baked system functions
    vm = bake_system_prompt('''
        int square(int x) { return x * x; }
        int cube(int x) { return x * square(x); }
    ''')

    # Run user code that uses baked functions
    output, exit_code = vm.run_user_code('int main() { return cube(3); }')
    # exit_code == 27

Previous implementation (C4TransformerVM/ONNX pipeline) archived to
src/archive/prompt_baking_v1.py.
"""

import re
import json
from typing import Optional, List, Tuple
from pathlib import Path

from .compiler import compile_c


def _apply_arvm_weights(model, weights):
    """Apply load_arvm() weight dict onto an AutoregressiveVM model.

    Args:
        model: AutoregressiveVM instance
        weights: dict from load_arvm() with 'embed_weight', 'layers',
                 'head_weight', 'head_bias' numpy arrays
    """
    import torch
    import numpy as np

    with torch.no_grad():
        model.embed.weight.copy_(torch.from_numpy(weights['embed_weight']))

        for i, layer_w in enumerate(weights['layers']):
            block = model.blocks[i]
            attn = block.attn
            ffn = block.ffn

            attn.alibi_slopes.copy_(torch.from_numpy(layer_w['alibi_slopes']))
            attn.W_q.copy_(torch.from_numpy(layer_w['W_q']))
            attn.W_k.copy_(torch.from_numpy(layer_w['W_k']))
            attn.W_v.copy_(torch.from_numpy(layer_w['W_v']))
            attn.W_o.copy_(torch.from_numpy(layer_w['W_o']))

            ffn.W_up.copy_(torch.from_numpy(layer_w['W_up']))
            ffn.b_up.copy_(torch.from_numpy(layer_w['b_up']))
            ffn.W_gate.copy_(torch.from_numpy(layer_w['W_gate']))
            ffn.b_gate.copy_(torch.from_numpy(layer_w['b_gate']))
            ffn.W_down.copy_(torch.from_numpy(layer_w['W_down']))
            ffn.b_down.copy_(torch.from_numpy(layer_w['b_down']))

        model.head.weight.copy_(torch.from_numpy(weights['head_weight']))
        model.head.bias.copy_(torch.from_numpy(weights['head_bias']))


class BakedPromptVM:
    """
    VM with a baked-in system prompt.

    The system prompt (C code) is stored as source and prepended to user
    code at compile time. The autoregressive model executes the combined
    program as context tokens via attention.
    """

    def __init__(
        self,
        system_source: str,
        d_model: int = 512,
        n_layers: int = 16,
        n_heads: int = 8,
        ffn_hidden: int = 4096,
        bake_weights: bool = True,
    ):
        """
        Create a VM with baked system prompt.

        Args:
            system_source: C source code for system functions
            d_model: Model embedding dimension
            n_layers: Number of transformer layers
            n_heads: Number of attention heads
            ffn_hidden: FFN hidden dimension
            bake_weights: If True, call set_vm_weights() to bake all
                         non-syscall opcodes into the model
        """
        self.system_source = system_source
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.ffn_hidden = ffn_hidden

        # Extract function names from system source
        self.available_functions = self._extract_function_names(system_source)

        # Lazy-create runner (imports torch + neural_vm)
        from neural_vm.run_vm import AutoregressiveVMRunner
        self.runner = AutoregressiveVMRunner(
            d_model=d_model, n_layers=n_layers,
            n_heads=n_heads, ffn_hidden=ffn_hidden,
        )

        if bake_weights:
            from neural_vm.vm_step import set_vm_weights
            set_vm_weights(self.runner.model)

    def _extract_function_names(self, source: str) -> List[str]:
        """Extract function names from C source."""
        pattern = r'\b(int|void|char)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        matches = re.findall(pattern, source)
        return [name for _, name in matches]

    def run_user_code(
        self,
        user_source: str,
        argv: Optional[List[str]] = None,
        stdin: str = "",
        max_steps: int = 100000,
    ) -> Tuple[str, int]:
        """
        Run user code with access to system functions.

        The user code's main() is compiled together with system functions.

        Args:
            user_source: C source code with main()
            argv: Optional argument list
            stdin: Optional stdin string
            max_steps: Maximum execution steps

        Returns:
            (output_string, exit_code) tuple
        """
        combined_source = self.system_source + "\n" + user_source
        bytecode, data = compile_c(combined_source)
        return self.runner.run(bytecode, data, argv, stdin, max_steps)

    def run_bytecode(
        self,
        bytecode: List[int],
        data: Optional[bytes] = None,
        argv: Optional[List[str]] = None,
        stdin: str = "",
        max_steps: int = 100000,
    ) -> Tuple[str, int]:
        """Run pre-compiled bytecode."""
        return self.runner.run(bytecode, data or b"", argv, stdin, max_steps)

    def save_pretrained(self, path: str):
        """Save the baked prompt VM to a directory.

        Saves prompt_config.json and model.arvm.
        """
        from tools.export_autoregressive import export_autoregressive

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        config = {
            'system_source': self.system_source,
            'available_functions': self.available_functions,
            'd_model': self.d_model,
            'n_layers': self.n_layers,
            'n_heads': self.n_heads,
            'ffn_hidden': self.ffn_hidden,
        }
        with open(path / 'prompt_config.json', 'w') as f:
            json.dump(config, f, indent=2)

        export_autoregressive(self.runner.model, str(path / 'model.arvm'))

    @classmethod
    def from_pretrained(cls, path: str) -> 'BakedPromptVM':
        """Load a baked prompt VM from a directory."""
        from tools.export_autoregressive import load_arvm

        path = Path(path)

        with open(path / 'prompt_config.json', 'r') as f:
            config = json.load(f)

        vm = cls(
            system_source=config['system_source'],
            d_model=config.get('d_model', 512),
            n_layers=config.get('n_layers', 16),
            n_heads=config.get('n_heads', 8),
            ffn_hidden=config.get('ffn_hidden', 4096),
            bake_weights=False,
        )

        arvm_path = path / 'model.arvm'
        if arvm_path.exists():
            weights = load_arvm(str(arvm_path))
            _apply_arvm_weights(vm.runner.model, weights)

        return vm


def bake_system_prompt(system_source: str, **kwargs) -> BakedPromptVM:
    """
    Convenience function to create a BakedPromptVM.

    Args:
        system_source: C source code for system functions
        **kwargs: Passed to BakedPromptVM (d_model, n_layers, etc.)

    Returns:
        BakedPromptVM instance

    Example:
        vm = bake_system_prompt('''
            int factorial(int n) {
                if (n < 2) return 1;
                return n * factorial(n - 1);
            }
        ''')
        output, exit_code = vm.run_user_code('int main() { return factorial(5); }')
        # exit_code == 120
    """
    return BakedPromptVM(system_source=system_source, **kwargs)


# Prebuilt system prompts for common use cases

MATH_SYSTEM_PROMPT = """
int abs(int x) {
    if (x < 0) return 0 - x;
    return x;
}

int min(int a, int b) {
    if (a < b) return a;
    return b;
}

int max(int a, int b) {
    if (a > b) return a;
    return b;
}

int factorial(int n) {
    if (n < 2) return 1;
    return n * factorial(n - 1);
}

int mypow(int base, int exp) {
    int result;
    result = 1;
    while (exp > 0) {
        result = result * base;
        exp = exp - 1;
    }
    return result;
}

int gcd(int a, int b) {
    int temp;
    while (b != 0) {
        temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}
"""

ARRAY_SYSTEM_PROMPT = """
int sum_range(int start, int end) {
    int total;
    int i;
    total = 0;
    i = start;
    while (i <= end) {
        total = total + i;
        i = i + 1;
    }
    return total;
}

int count_up(int n) {
    int i;
    i = 0;
    while (i < n) {
        i = i + 1;
    }
    return i;
}
"""


__all__ = [
    'BakedPromptVM',
    'bake_system_prompt',
    'MATH_SYSTEM_PROMPT',
    'ARRAY_SYSTEM_PROMPT',
]
