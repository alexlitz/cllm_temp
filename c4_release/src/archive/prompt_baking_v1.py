"""
System Prompt Baking for C4 Transformer VM

Allows "baking in" a system prompt (C program) into the VM's weights/state.
The baked prompt becomes part of the model's initial configuration, enabling:

1. Pre-configured VMs with specific behaviors
2. System programs that run before user programs
3. Library functions available by default
4. Fast execution via speculator for common patterns

Usage:
    from src.prompt_baking import BakedPromptVM, bake_system_prompt

    # Create a VM with baked system functions
    vm = bake_system_prompt('''
        int square(int x) { return x * x; }
        int cube(int x) { return x * square(x); }
    ''')

    # Run user code that uses baked functions
    result = vm.run_user_code('int main() { return cube(3); }')
    # Returns 27
"""

import torch
import json
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict

from .transformer_vm import C4TransformerVM
from .speculator import FastLogicalVM, SpeculativeVM
from .compiler import compile_c


@dataclass
class BakedPromptConfig:
    """Configuration for a baked system prompt."""
    system_source: str
    system_bytecode: List[int]
    system_data: Optional[bytes]
    entry_point: int  # Where user code should jump to
    available_functions: List[str]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['system_data'] = list(d['system_data']) if d['system_data'] else None
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'BakedPromptConfig':
        if d.get('system_data'):
            d['system_data'] = bytes(d['system_data'])
        return cls(**d)


class BakedPromptVM:
    """
    VM with a baked-in system prompt.

    The system prompt (C code) is pre-compiled and embedded in the VM's
    initial state. User code can call system functions directly.
    """

    def __init__(
        self,
        system_source: str,
        use_speculator: bool = True,
        validate_ratio: float = 0.0,
    ):
        """
        Create a VM with baked system prompt.

        Args:
            system_source: C source code for system functions
            use_speculator: Use fast speculative execution
            validate_ratio: Fraction of runs to validate with transformer VM
        """
        self.system_source = system_source
        self.use_speculator = use_speculator

        # Compile system code
        self._compile_system()

        # Create VMs
        self.transformer_vm = C4TransformerVM()
        self.fast_vm = FastLogicalVM()

        if use_speculator:
            self.speculator = SpeculativeVM(
                transformer_vm=self.transformer_vm if validate_ratio > 0 else None,
                validate_ratio=validate_ratio,
            )
        else:
            self.speculator = None

    def _compile_system(self):
        """Compile system source to bytecode."""
        # Parse system source to find function names
        self.available_functions = self._extract_function_names(self.system_source)

        # We'll create a wrapper that allows calling system functions
        # For now, system source must include main() which we'll redirect
        self.system_bytecode, self.system_data = compile_c(self.system_source)

    def _extract_function_names(self, source: str) -> List[str]:
        """Extract function names from C source."""
        import re
        # Simple regex to find function definitions
        pattern = r'\b(int|void|char)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        matches = re.findall(pattern, source)
        return [name for _, name in matches]

    def run_user_code(
        self,
        user_source: str,
        max_steps: int = 100000,
    ) -> int:
        """
        Run user code with access to system functions.

        The user code's main() is compiled and runs after system setup.
        System functions are available to call.

        Args:
            user_source: C source code with main()
            max_steps: Maximum execution steps

        Returns:
            Return value from user's main()
        """
        # Combine system and user source
        combined_source = self.system_source + "\n" + user_source

        # Compile combined code
        bytecode, data = compile_c(combined_source)

        # Run with speculator for speed
        if self.speculator:
            return self.speculator.run(bytecode, data)
        else:
            self.transformer_vm.reset()
            self.transformer_vm.load_bytecode(bytecode, data)
            return self.transformer_vm.run(max_steps=max_steps)

    def run_bytecode(
        self,
        bytecode: List[int],
        data: Optional[bytes] = None,
        max_steps: int = 100000,
    ) -> int:
        """Run pre-compiled bytecode."""
        if self.speculator:
            return self.speculator.run(bytecode, data)
        else:
            self.transformer_vm.reset()
            self.transformer_vm.load_bytecode(bytecode, data)
            return self.transformer_vm.run(max_steps=max_steps)

    def get_config(self) -> BakedPromptConfig:
        """Get the baked prompt configuration."""
        return BakedPromptConfig(
            system_source=self.system_source,
            system_bytecode=self.system_bytecode,
            system_data=self.system_data,
            entry_point=0,
            available_functions=self.available_functions,
        )

    def save_pretrained(self, path: str):
        """Save the baked prompt VM to a directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save config
        config = self.get_config()
        with open(path / 'prompt_config.json', 'w') as f:
            json.dump(config.to_dict(), f, indent=2)

        # Save transformer VM weights
        torch.save(
            self.transformer_vm.state_dict(),
            path / 'transformer_weights.pt'
        )

    @classmethod
    def from_pretrained(cls, path: str) -> 'BakedPromptVM':
        """Load a baked prompt VM from a directory."""
        path = Path(path)

        # Load config
        with open(path / 'prompt_config.json', 'r') as f:
            config_dict = json.load(f)
        config = BakedPromptConfig.from_dict(config_dict)

        # Create VM with system source
        vm = cls(system_source=config.system_source)

        # Load transformer weights if they exist
        weights_path = path / 'transformer_weights.pt'
        if weights_path.exists():
            vm.transformer_vm.load_state_dict(
                torch.load(weights_path, map_location='cpu')
            )

        return vm


def bake_system_prompt(
    system_source: str,
    use_speculator: bool = True,
    validate_ratio: float = 0.0,
) -> BakedPromptVM:
    """
    Convenience function to create a BakedPromptVM.

    Args:
        system_source: C source code for system functions
        use_speculator: Use fast speculative execution
        validate_ratio: Fraction of runs to validate with transformer VM

    Returns:
        BakedPromptVM instance

    Example:
        vm = bake_system_prompt('''
            int factorial(int n) {
                if (n < 2) return 1;
                return n * factorial(n - 1);
            }
        ''')
        result = vm.run_user_code('int main() { return factorial(5); }')
        # Returns 120
    """
    return BakedPromptVM(
        system_source=system_source,
        use_speculator=use_speculator,
        validate_ratio=validate_ratio,
    )


class WeightBaker:
    """
    Advanced weight baking for embedding computation patterns.

    Goes beyond just pre-loading bytecode - this modifies the FFN
    lookup tables to encode specific computation shortcuts.

    For example, if the system prompt frequently computes factorial,
    we could add a factorial lookup table for small inputs.
    """

    def __init__(self, vm: C4TransformerVM):
        self.vm = vm
        self.custom_tables = {}

    def add_lookup_table(
        self,
        name: str,
        func: callable,
        input_range: range,
    ):
        """
        Add a custom lookup table for a function.

        Args:
            name: Table name for reference
            func: Python function to encode
            input_range: Range of inputs to precompute
        """
        table = {}
        for x in input_range:
            try:
                table[x] = func(x)
            except:
                table[x] = 0
        self.custom_tables[name] = table

    def create_table_tensor(self, name: str) -> torch.Tensor:
        """Create a tensor lookup table from precomputed values."""
        table = self.custom_tables[name]
        max_key = max(table.keys())
        tensor = torch.zeros(max_key + 1, dtype=torch.float32)
        for k, v in table.items():
            tensor[k] = float(v)
        return tensor

    def get_baked_weights(self) -> Dict[str, torch.Tensor]:
        """Get all custom lookup tables as tensors."""
        return {
            name: self.create_table_tensor(name)
            for name in self.custom_tables
        }


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
    'BakedPromptConfig',
    'WeightBaker',
    'bake_system_prompt',
    'MATH_SYSTEM_PROMPT',
    'ARRAY_SYSTEM_PROMPT',
]
