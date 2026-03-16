---
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
