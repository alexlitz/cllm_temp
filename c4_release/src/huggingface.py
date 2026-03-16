"""
HuggingFace Integration for C4 Transformer VM

Provides save_pretrained and from_pretrained methods compatible
with the HuggingFace model hub pattern.
"""

import torch
import json
import os
from typing import Optional, Dict, Any, Union
from pathlib import Path
from dataclasses import dataclass, asdict

from .transformer_vm import C4TransformerVM, C4Config


@dataclass
class C4HFConfig:
    """
    Configuration for HuggingFace-style model saving.

    Extends C4Config with HF-specific fields.
    """
    # Model architecture
    hidden_dim: int = 256
    table_bits: int = 8
    newton_iterations: int = 2

    # VM parameters
    vocab_size: int = 256
    word_size: int = 8
    data_base: int = 0x10000

    # HuggingFace fields
    model_type: str = "c4-transformer-vm"
    architectures: list = None

    def __post_init__(self):
        if self.architectures is None:
            self.architectures = ["C4TransformerVM"]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "C4HFConfig":
        # Filter out unknown keys
        valid_keys = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered)

    def save_pretrained(self, save_directory: Union[str, Path]):
        """Save configuration to directory."""
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        config_file = save_directory / "config.json"
        with open(config_file, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_pretrained(cls, pretrained_path: Union[str, Path]) -> "C4HFConfig":
        """Load configuration from directory."""
        pretrained_path = Path(pretrained_path)
        config_file = pretrained_path / "config.json"

        with open(config_file, "r") as f:
            config_dict = json.load(f)

        return cls.from_dict(config_dict)


class C4HFModel(torch.nn.Module):
    """
    HuggingFace-compatible wrapper for C4TransformerVM.

    Usage:
        # Create and save
        model = C4HFModel()
        model.save_pretrained("./my-c4-model")

        # Load
        model = C4HFModel.from_pretrained("./my-c4-model")

        # Run program
        result = model.run_c("int main() { return 42; }")
    """

    def __init__(self, config: Optional[C4HFConfig] = None):
        super().__init__()
        self.config = config or C4HFConfig()
        self.vm = C4TransformerVM()

    def forward(self, bytecode: list, data: Optional[bytes] = None) -> int:
        """
        Run bytecode on the transformer VM.

        Args:
            bytecode: List of instruction integers
            data: Optional data segment bytes

        Returns:
            Result of execution (value in AX register)
        """
        self.vm.reset()
        self.vm.load_bytecode(bytecode, data)
        return self.vm.run()

    def run_c(self, source: str, max_steps: int = 100000) -> int:
        """
        Compile and run C source code.

        Args:
            source: C source code
            max_steps: Maximum execution steps

        Returns:
            Result of execution
        """
        from .compiler import compile_c

        bytecode, data = compile_c(source)
        self.vm.reset()
        self.vm.load_bytecode(bytecode, data)
        return self.vm.run(max_steps=max_steps)

    def save_pretrained(
        self,
        save_directory: Union[str, Path],
        safe_serialization: bool = True,
    ):
        """
        Save model to directory in HuggingFace format.

        Args:
            save_directory: Directory to save to
            safe_serialization: If True, save in safetensors format (if available)
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        # Save config
        self.config.save_pretrained(save_directory)

        # Save model weights
        model_file = save_directory / "pytorch_model.bin"
        state_dict = self.vm.state_dict()
        torch.save(state_dict, model_file)

        # Try safetensors if requested
        if safe_serialization:
            try:
                from safetensors.torch import save_file
                safetensors_file = save_directory / "model.safetensors"
                # Convert to flat dict for safetensors
                flat_dict = {}
                for key, value in state_dict.items():
                    if isinstance(value, torch.Tensor):
                        flat_dict[key] = value
                save_file(flat_dict, safetensors_file)
            except ImportError:
                pass  # safetensors not available

        # Save model card
        model_card = save_directory / "README.md"
        if not model_card.exists():
            with open(model_card, "w") as f:
                f.write(self._generate_model_card())

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: Union[str, Path],
        **kwargs,
    ) -> "C4HFModel":
        """
        Load model from directory.

        Args:
            pretrained_path: Path to saved model directory

        Returns:
            Loaded C4HFModel instance
        """
        pretrained_path = Path(pretrained_path)

        # Load config
        config = C4HFConfig.from_pretrained(pretrained_path)

        # Create model
        model = cls(config=config)

        # Load weights
        # Try safetensors first
        safetensors_file = pretrained_path / "model.safetensors"
        pytorch_file = pretrained_path / "pytorch_model.bin"

        if safetensors_file.exists():
            try:
                from safetensors.torch import load_file
                state_dict = load_file(safetensors_file)
                model.vm.load_state_dict(state_dict)
            except ImportError:
                if pytorch_file.exists():
                    state_dict = torch.load(pytorch_file, map_location="cpu")
                    model.vm.load_state_dict(state_dict)
        elif pytorch_file.exists():
            state_dict = torch.load(pytorch_file, map_location="cpu")
            model.vm.load_state_dict(state_dict)

        return model

    def _generate_model_card(self) -> str:
        """Generate a README model card."""
        return """---
language: en
tags:
- neural-vm
- transformer
- arithmetic
- byte-tokens
license: mit
---

# C4 Transformer VM

A pure transformer virtual machine where all arithmetic operations
(multiply, divide, add, compare) are implemented using neural network
operations (nn.Module).

## Features

- **Byte-level tokenization**: Vocab size of 256 (one token per byte)
- **Nibble tables**: 16x16 lookup tables for efficient computation
- **SwiGLU multiplication**: Exact integer multiply via silu activations
- **FFN division**: Reciprocal table + Newton-Raphson refinement
- **Full C4 VM**: Runs complete C programs via compiled bytecode

## Usage

```python
from c4_release.src import C4HFModel

# Load model
model = C4HFModel.from_pretrained("./my-c4-model")

# Run C code
result = model.run_c("int main() { return 6 * 7; }")
print(result)  # 42
```

## Architecture

- **Parameters**: ~653K (buffer weights in FFN tables)
- **Tables**: Byte-to-nibble conversion, nibble operations, nibble-to-byte
- **Arithmetic**: SwiGLU for multiply, 256-entry reciprocal + Newton for divide

## Citation

```bibtex
@software{c4_transformer_vm,
  title={C4 Transformer VM},
  year={2024},
  description={Pure transformer virtual machine with byte tokens}
}
```
"""

    def get_num_parameters(self) -> int:
        """Get total number of parameters (buffer weights)."""
        return sum(b.numel() for b in self.vm.buffers())

    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        num_params = self.get_num_parameters()
        return {
            "num_buffers": num_params,
            "memory_mb_fp32": num_params * 4 / (1024 * 1024),
            "memory_mb_fp16": num_params * 2 / (1024 * 1024),
        }


def push_to_hub(
    model: C4HFModel,
    repo_id: str,
    private: bool = False,
    token: Optional[str] = None,
):
    """
    Push model to HuggingFace Hub.

    Args:
        model: C4HFModel instance
        repo_id: Repository ID (e.g., "username/my-c4-model")
        private: Whether repo should be private
        token: HuggingFace auth token
    """
    try:
        from huggingface_hub import HfApi, create_repo

        # Create repo
        api = HfApi(token=token)
        create_repo(repo_id, private=private, token=token, exist_ok=True)

        # Save locally first
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            api.upload_folder(
                folder_path=tmpdir,
                repo_id=repo_id,
                token=token,
            )

    except ImportError:
        raise ImportError("Please install huggingface_hub: pip install huggingface_hub")


__all__ = ['C4HFModel', 'C4HFConfig', 'push_to_hub']
