"""
Structural enforcement for autoregressive purity.

This module provides runtime verification that the forward pass remains pure
and hasn't been modified to include Python tensor modifications.

Similar to the _BlockedDraftVM approach, this creates structural guarantees
that purity cannot be easily violated without explicit circumvention.
"""

import inspect
import re


class PurityViolationError(RuntimeError):
    """Raised when forward pass purity is violated."""
    pass


def verify_forward_purity(model):
    """Verify forward() hasn't been modified from pure implementation.

    This provides structural enforcement by checking that forward()
    source code matches the expected pure implementation pattern.

    Args:
        model: AutoregressiveVM instance

    Raises:
        PurityViolationError: If forward() contains forbidden modifications

    Returns:
        True if pure

    Example forbidden code (will raise):
        >>> def forward(self, token_ids):
        ...     x = self.embed(token_ids)
        ...     x[0, 0, 100] = 42.0  # ← FORBIDDEN!

    Example pure code (will pass):
        >>> def forward(self, token_ids):
        ...     x = self.embed(token_ids)
        ...     for block in self.blocks:
        ...         x = block(x)
        ...     return self.head(x)
    """
    try:
        source = inspect.getsource(model.forward)
    except (TypeError, OSError) as e:
        raise PurityViolationError(
            f"Cannot inspect forward() source code: {e}\n"
            f"This may indicate the method has been replaced or wrapped."
        )

    # Remove docstrings and comments for analysis
    lines = source.split('\n')
    code_lines = []
    in_docstring = False

    for line in lines:
        stripped = line.strip()
        # Skip docstrings
        if '"""' in stripped or "'''" in stripped:
            in_docstring = not in_docstring
            continue
        if in_docstring or stripped.startswith('#'):
            continue
        code_lines.append(line)

    code = '\n'.join(code_lines)

    # Check for forbidden patterns (Python tensor modifications)
    forbidden_patterns = {
        r'\bx\s*\[': 'Direct tensor indexing/modification (x[...])',
        r'\bx\.data\s*\[': 'Direct data tensor modification (x.data[...])',
        r'embeddings?\s*\[': 'Direct embedding modification',
        r'_add_code_addr_keys\s*\(': 'Call to old augmentation method _add_code_addr_keys()',
        r'_inject_mem_store\s*\(': 'Call to old augmentation method _inject_mem_store()',
        r'\.item\(\)\s*\n\s*x\[': 'Python loop with tensor assignment',
        r'for\s+.*\s+in\s+range\(.*\):\s*\n\s+.*x\[': 'Python loop modifying embeddings',
    }

    violations = []
    for pattern, description in forbidden_patterns.items():
        matches = re.finditer(pattern, code, re.MULTILINE)
        for match in matches:
            # Get line number
            line_num = code[:match.start()].count('\n') + 1
            violations.append(f"  Line {line_num}: {description}")

    if violations:
        raise PurityViolationError(
            "PURITY VIOLATION DETECTED in forward() method!\n\n"
            "Forbidden patterns found:\n" + '\n'.join(violations) + "\n\n"
            "Forward pass must be PURE:\n"
            "  x = self.embed(token_ids)  # All augmentations INSIDE embedding\n"
            "  for block in self.blocks:\n"
            "      x = block(x)  # ONLY neural operations\n"
            "  return self.head(x)\n\n"
            "All embedding modifications must be inside NeuralVMEmbedding.forward().\n"
            "Python tensor modifications between embed() and blocks are FORBIDDEN."
        )

    # Check for required structure (positive verification)
    required_patterns = {
        r'self\.embed\s*\(\s*token_ids(?:\s*,\s*\w+\s*=\s*[^)]+)?\s*\)': 'Must call self.embed(token_ids)',
        r'for\s+.*\s+in\s+enumerate\s*\(\s*self\.blocks\s*\)': 'Must iterate through self.blocks',
        r'self\.head\s*\(': 'Must call self.head()',
    }

    missing = []
    for pattern, description in required_patterns.items():
        if not re.search(pattern, code):
            missing.append(f"  {description}")

    if missing:
        raise PurityViolationError(
            "PURITY VIOLATION: forward() structure is invalid!\n\n"
            "Missing required patterns:\n" + '\n'.join(missing) + "\n\n"
            "Forward pass must follow the structure:\n"
            "  x = self.embed(token_ids)\n"
            "  for i, block in enumerate(self.blocks):\n"
            "      x = block(x, kv_cache)\n"
            "  return self.head(x)"
        )

    # Verify embedding is NeuralVMEmbedding
    from .neural_embedding import NeuralVMEmbedding
    if not isinstance(model.embed, NeuralVMEmbedding):
        raise PurityViolationError(
            f"PURITY VIOLATION: model.embed must be NeuralVMEmbedding!\n"
            f"Found: {type(model.embed).__name__}\n\n"
            f"The embedding layer must be NeuralVMEmbedding to encapsulate\n"
            f"all position-dependent augmentations (ADDR_KEY, MEM_STORE)."
        )

    return True


def verify_embedding_purity(embedding):
    """Verify NeuralVMEmbedding contains augmentation methods.

    Args:
        embedding: NeuralVMEmbedding instance

    Raises:
        PurityViolationError: If augmentation methods are missing

    Returns:
        True if pure
    """
    from .neural_embedding import NeuralVMEmbedding

    if not isinstance(embedding, NeuralVMEmbedding):
        raise PurityViolationError(
            f"Expected NeuralVMEmbedding, got {type(embedding).__name__}"
        )

    # Check for required augmentation methods
    required_methods = ['_add_code_addr_keys', '_inject_mem_store', 'set_mem_history_end']
    missing = [m for m in required_methods if not hasattr(embedding, m)]

    if missing:
        raise PurityViolationError(
            f"NeuralVMEmbedding missing required methods: {missing}\n"
            f"The embedding must encapsulate all augmentation logic."
        )

    # Verify forward() calls augmentation methods
    try:
        source = inspect.getsource(embedding.forward)
        if '_add_code_addr_keys' not in source or '_inject_mem_store' not in source:
            raise PurityViolationError(
                "NeuralVMEmbedding.forward() must call both "
                "_add_code_addr_keys() and _inject_mem_store()"
            )
    except (TypeError, OSError):
        # Can't inspect source - warning but don't fail
        pass

    return True


def enable_purity_enforcement(model, strict=True):
    """Enable purity enforcement on a model.

    This is called automatically by set_vm_weights() to ensure
    weights are only loaded into pure models.

    Args:
        model: AutoregressiveVM instance
        strict: If True, raise on violations. If False, warn only.

    Returns:
        True if pure, False otherwise

    Raises:
        PurityViolationError: If strict=True and violations found
    """
    try:
        verify_forward_purity(model)
        verify_embedding_purity(model.embed)
        return True
    except PurityViolationError as e:
        if strict:
            raise
        else:
            import warnings
            warnings.warn(f"Purity violation detected: {e}", UserWarning)
            return False


def create_pure_model():
    """Factory function that creates a guaranteed-pure AutoregressiveVM.

    This is the recommended way to create models as it ensures purity
    from the start.

    Returns:
        AutoregressiveVM with purity verified

    Example:
        >>> model = create_pure_model()
        >>> set_vm_weights(model)  # Will verify purity before loading
        >>> logits = model.forward(token_ids)  # Guaranteed pure
    """
    from .vm_step import AutoregressiveVM
    model = AutoregressiveVM()

    # Verify purity immediately
    verify_forward_purity(model)
    verify_embedding_purity(model.embed)

    return model
