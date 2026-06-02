"""
Embedding Weight Module.

Sets token embedding weights for the Neural VM.

Phase 7.D.3 migration: the imperative per-token writes were replaced by
``TokenEmbeddingRule``s lowered via :meth:`CompilerIR.lower_token_embeddings`.
The rule list is the same one used by the active production bake op
``make_embedding_bake_op`` in ``unified_compiler/ops/model_ops.py`` --
single source of truth.
"""

from typing import List
from .base import WeightModule, WeightConfig, get_dimension_registry


def _setdim_to_positions(BD) -> dict:
    """Build a ``dim_positions`` dict from a ``_SetDim``-like class.

    Mirrors the contract expected by ``CompilerIR.lower_token_embeddings``:
    a ``Mapping[str, int]``. Walks every public class attribute that resolves
    to an ``int`` (the dim-position constants on ``_SetDim``). Lookups for
    missing dims fall back to ``getattr(_SetDim, name)`` inside the rule
    lowering only if the dim name is referenced; here we copy every
    integer attribute up-front so the mapping is closed.
    """
    dim_positions = {}
    for name in dir(BD):
        if name.startswith("_"):
            continue
        val = getattr(BD, name, None)
        if isinstance(val, int) and not isinstance(val, bool):
            dim_positions[name] = val
    return dim_positions


class EmbeddingWeights(WeightModule):
    """Weight module for token embeddings."""

    @property
    def name(self) -> str:
        return "embedding"

    @property
    def layers(self) -> List[int]:
        return []  # Embedding is not in transformer layers

    @property
    def dimensions(self) -> List[int]:
        return list(range(512))  # Uses all dimensions

    def set_weights(self, model) -> None:
        """Set embedding weights via declarative ``TokenEmbeddingRule`` IR.

        Phase 7.D.3 migration: replaced per-token imperative writes with a
        call to ``CompilerIR.lower_token_embeddings`` carrying the same rule
        list that ``make_embedding_bake_op`` uses. The bake_fn first zeroes
        ``model.embed.embed.weight`` (the IR has no zeroing primitive) and
        then lowers the rules.
        """
        import torch
        from neural_vm.unified_compiler.ir import CompilerIR
        from neural_vm.unified_compiler.ops.model_ops import _embedding_bake_rules

        BD = get_dimension_registry()
        dim_positions = _setdim_to_positions(BD)

        with torch.no_grad():
            model.embed.embed.weight.zero_()

        ir = CompilerIR()
        ir.embeddings.extend(_embedding_bake_rules(model.vocab_size))
        ir.lower_token_embeddings(model, dim_positions)
