"""
C4 Transformer VM - 100% Autoregressive Neural Network Virtual Machine

Every step() is a single differentiable forward pass with no Python
control flow branching on data values.

Components:
    - C4TransformerVM: 100% autoregressive VM (soft-blend opcode dispatch)
    - FastLogicalVM: Fast reference VM for speculative execution
    - SpeculativeVM: Combines fast and transformer VMs
    - C4Tokenizer: Byte-level tokenizer with special tokens
    - compile_c: C4 compiler for C source code
"""

from .compiler import compile_c, Compiler

try:
    from .transformer_vm import (
        C4TransformerVM,
        C4Config,
        NeuralALU,
        TransformerState,
    )
except ImportError:
    pass

try:
    from .speculator import (
        FastLogicalVM,
        SpeculativeVM,
        ParallelSpeculator,
        TracingVM,
        TraceSpeculator,
    )
except ImportError:
    pass

try:
    from .io_support import (
        InteractiveVM,
        StreamingVM,
        IOExtendedVM,
    )
except ImportError:
    pass

try:
    from .tokenizer import (
        C4Tokenizer,
        SpecialToken,
        TokenizerConfig,
        TOKEN_NAMES,
    )
except ImportError:
    pass

try:
    from .prompt_baking import (
        BakedPromptVM,
        bake_system_prompt,
        MATH_SYSTEM_PROMPT,
        ARRAY_SYSTEM_PROMPT,
    )
except ImportError:
    pass

try:
    from .baked_c4 import (
        BakedC4Transformer,
        BakedC4Config,
        BytecodeBakedTransformer,
        QuineTransformer,
        create_quine_transformer,
        create_self_baking_transformer,
        generate_bytecode_baking_c_program,
    )
except ImportError:
    pass

__version__ = "1.0.0"

__all__ = [
    # VM
    'C4TransformerVM',
    'C4Config',
    'NeuralALU',
    'TransformerState',
    # Speculator
    'FastLogicalVM',
    'SpeculativeVM',
    'ParallelSpeculator',
    'TracingVM',
    'TraceSpeculator',
    # I/O
    'InteractiveVM',
    'StreamingVM',
    'IOExtendedVM',
    # Tokenizer
    'C4Tokenizer',
    'SpecialToken',
    'TokenizerConfig',
    'TOKEN_NAMES',
    # Compiler
    'compile_c',
    'Compiler',
    # Prompt Baking
    'BakedPromptVM',
    'bake_system_prompt',
    'MATH_SYSTEM_PROMPT',
    'ARRAY_SYSTEM_PROMPT',
    # Baked C4
    'BakedC4Transformer',
    'BakedC4Config',
    'BytecodeBakedTransformer',
    'QuineTransformer',
    'create_quine_transformer',
    'create_self_baking_transformer',
    'generate_bytecode_baking_c_program',
]
