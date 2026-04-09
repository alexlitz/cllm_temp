"""
MOVED TO src/archive/baked_c4.py

BakedC4Transformer is a legacy wrapper around C4TransformerVM.
The current implementation is AutoregressiveVMRunner in neural_vm/run_vm.py.

For backward compatibility, import from archive:
    from src.archive.baked_c4 import BakedC4Transformer
"""

from src.archive.baked_c4 import BakedC4Transformer

__all__ = ['BakedC4Transformer']
