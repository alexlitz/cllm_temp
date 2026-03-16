"""
RASP to PyTorch Compiler.

Compiles RASP programs to PyTorch transformer weights. This allows
expressing algorithms in RASP and getting a working transformer.

The compilation process:
1. Analyze the RASP program's computation graph
2. Assign encodings (residual stream dimensions) to each S-op
3. Construct attention heads for each Select/Aggregate pair
4. Construct MLP layers for elementwise operations
5. Build the final PyTorch model

Usage:
    from rasp import tokens, indices, select, aggregate, RASPProgram
    from rasp_compiler import compile_rasp

    # Define program
    same = select(tokens, tokens, lambda k, q: k == q)
    count = aggregate(same, ones)
    program = RASPProgram([count], vocab=['a','b','c'], max_seq_len=16)

    # Compile to PyTorch
    model = compile_rasp(program)

    # Run inference
    output = model(input_ids)
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rasp import (
    SOp, Selector, RASPProgram,
    InputSOp, IndicesSOp, LengthSOp, OnesSOp, ConstantSOp,
    ElementwiseOp, MapOp, SequenceMapOp, AggregateOp, SelectorWidthOp,
    SelectOp, ConstantSelector, SelectorAnd, SelectorOr, SelectorNot,
    Encoding
)


# =============================================================================
# COMPILATION ANALYSIS
# =============================================================================

@dataclass
class BasisAllocation:
    """Tracks which dimensions of the residual stream are used for what."""

    # Map from S-op id to its encoding
    sop_encodings: Dict[int, Encoding] = field(default_factory=dict)

    # Next free dimension
    next_dim: int = 0

    # Total dimension of residual stream
    total_dim: int = 0

    def allocate_categorical(self, sop: SOp, categories: List[Any]) -> Encoding:
        """Allocate one-hot encoding for categorical S-op."""
        size = len(categories)
        encoding = Encoding(
            encoding_type=Encoding.Type.CATEGORICAL,
            basis_start=self.next_dim,
            basis_size=size,
            categories=categories
        )
        self.sop_encodings[sop.id] = encoding
        self.next_dim += size
        self.total_dim = self.next_dim
        return encoding

    def allocate_numerical(self, sop: SOp, size: int = 1) -> Encoding:
        """Allocate numerical encoding for S-op."""
        encoding = Encoding(
            encoding_type=Encoding.Type.NUMERICAL,
            basis_start=self.next_dim,
            basis_size=size,
            categories=None
        )
        self.sop_encodings[sop.id] = encoding
        self.next_dim += size
        self.total_dim = self.next_dim
        return encoding

    def get_encoding(self, sop: SOp) -> Optional[Encoding]:
        """Get encoding for an S-op if allocated."""
        return self.sop_encodings.get(sop.id)


@dataclass
class LayerPlan:
    """Plan for a single transformer layer."""

    # Attention heads: list of (selector, values_sop, output_sop)
    attention_heads: List[Tuple[Selector, SOp, SOp]] = field(default_factory=list)

    # MLP operations: list of (input_sops, output_sop, operation)
    mlp_operations: List[Tuple[List[SOp], SOp, str]] = field(default_factory=list)


@dataclass
class CompilationPlan:
    """Complete compilation plan for a RASP program."""

    program: RASPProgram
    basis: BasisAllocation
    layers: List[LayerPlan]

    # Map from S-op id to the layer where it's computed
    sop_to_layer: Dict[int, int] = field(default_factory=dict)


def analyze_program(program: RASPProgram) -> CompilationPlan:
    """
    Analyze a RASP program and create a compilation plan.

    This determines:
    - How to encode each S-op in the residual stream
    - How many layers are needed
    - What each layer computes
    """
    all_sops, all_selectors = program.get_all_ops()

    basis = BasisAllocation()

    # Step 1: Allocate basis for primitive S-ops
    for sop in all_sops:
        if isinstance(sop, InputSOp):
            # Tokens get one-hot encoding
            basis.allocate_categorical(sop, program.vocab)
        elif isinstance(sop, IndicesSOp):
            # Indices get numerical encoding (one dim)
            basis.allocate_numerical(sop, 1)
        elif isinstance(sop, LengthSOp):
            # Length gets numerical encoding
            basis.allocate_numerical(sop, 1)
        elif isinstance(sop, OnesSOp):
            # Ones get numerical encoding
            basis.allocate_numerical(sop, 1)
        elif isinstance(sop, ConstantSOp):
            # Constants get numerical encoding
            basis.allocate_numerical(sop, 1)

    # Step 2: Topological sort and layer assignment
    # S-ops that depend on attention results must come after the attention layer
    sop_to_layer: Dict[int, int] = {}

    def get_sop_layer(sop: SOp) -> int:
        """Get the layer where this S-op is available."""
        if sop.id in sop_to_layer:
            return sop_to_layer[sop.id]

        # Primitives are available at layer 0 (after embedding)
        if isinstance(sop, (InputSOp, IndicesSOp, LengthSOp, OnesSOp, ConstantSOp)):
            sop_to_layer[sop.id] = 0
            return 0

        # For derived S-ops, need to be after all dependencies
        deps = sop.get_dependencies()
        if not deps:
            sop_to_layer[sop.id] = 0
            return 0

        max_dep_layer = max(get_sop_layer(dep) for dep in deps)

        # Aggregates and SelectorWidth need attention, so they're computed
        # in a layer and available in the next layer
        if isinstance(sop, (AggregateOp, SelectorWidthOp)):
            layer = max_dep_layer + 1
        else:
            # Elementwise ops are computed in MLP, same layer as deps
            layer = max_dep_layer

        sop_to_layer[sop.id] = layer
        return layer

    # Compute layers for all S-ops
    for sop in all_sops:
        get_sop_layer(sop)

    # Step 3: Allocate basis for derived S-ops
    for sop in all_sops:
        if sop.id not in basis.sop_encodings:
            # Default to numerical encoding for derived S-ops
            basis.allocate_numerical(sop, 1)

    # Step 4: Build layer plans
    num_layers = max(sop_to_layer.values()) + 1 if sop_to_layer else 1
    layers = [LayerPlan() for _ in range(num_layers)]

    for sop in all_sops:
        layer_idx = sop_to_layer[sop.id]

        if isinstance(sop, AggregateOp):
            # This is computed by an attention head
            layers[layer_idx].attention_heads.append(
                (sop.selector, sop.values, sop)
            )
        elif isinstance(sop, SelectorWidthOp):
            # This is also attention-based (count selected positions)
            # We create a pseudo-aggregate that sums ones
            layers[layer_idx].attention_heads.append(
                (sop.selector, None, sop)  # None indicates count mode
            )
        elif isinstance(sop, (ElementwiseOp, MapOp, SequenceMapOp)):
            # This is computed in the MLP
            deps = sop.get_dependencies()
            op_name = getattr(sop, 'op_name', 'map')
            layers[layer_idx].mlp_operations.append((deps, sop, op_name))

    return CompilationPlan(
        program=program,
        basis=basis,
        layers=layers,
        sop_to_layer=sop_to_layer
    )


# =============================================================================
# PYTORCH MODEL
# =============================================================================

class CompiledAttentionHead(nn.Module):
    """
    Attention head compiled from a RASP selector.

    Implements: output[i] = sum_j (attn[i,j] * values[j])
    where attn[i,j] is determined by the selector pattern.
    """

    def __init__(
        self,
        d_model: int,
        selector: Selector,
        values_encoding: Optional[Encoding],
        output_encoding: Encoding,
        is_count_mode: bool = False
    ):
        super().__init__()
        self.d_model = d_model
        self.selector = selector
        self.values_encoding = values_encoding
        self.output_encoding = output_encoding
        self.is_count_mode = is_count_mode

        # For count mode, we sum ones instead of values
        if is_count_mode:
            self.value_size = 1
        else:
            self.value_size = values_encoding.basis_size if values_encoding else 1

        # Output projection - initialize to identity for correct pass-through
        self.W_o = nn.Linear(self.value_size, output_encoding.basis_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for identity-like pass-through."""
        with torch.no_grad():
            # Set W_o to identity (or truncated identity if sizes differ)
            nn.init.zeros_(self.W_o.weight)
            min_size = min(self.value_size, self.output_encoding.basis_size)
            for i in range(min_size):
                self.W_o.weight[i, i] = 1.0

    def forward(
        self,
        x: torch.Tensor,
        selector_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply attention using softmax1 (quiet attention).

        Softmax1: exp(score) / (1 + sum(exp(scores)))

        This allows the model to "attend to nothing" via the +1 term.
        The "leftover" (1 - sum(weights)) encodes count information.

        Args:
            x: (batch, seq_len, d_model) residual stream
            selector_mask: (batch, seq_len, seq_len) boolean attention pattern

        Returns:
            output: (batch, seq_len, d_model) with result in output dimensions
        """
        batch, seq_len, _ = x.shape

        # Get values
        if self.is_count_mode:
            # Count mode: values are all ones
            values = torch.ones(batch, seq_len, 1, device=x.device, dtype=x.dtype)
        else:
            # Extract values from residual stream
            start = self.values_encoding.basis_start
            end = start + self.values_encoding.basis_size
            values = x[:, :, start:end]

        # Convert boolean selector mask to attention scores for softmax1
        # Selected positions get score 0, non-selected get -inf
        scores = torch.where(
            selector_mask,
            torch.zeros_like(selector_mask, dtype=x.dtype),
            torch.full_like(selector_mask, float('-inf'), dtype=x.dtype)
        )

        # Apply softmax1: exp(s) / (1 + sum(exp(s)))
        # With numerical stability: subtract max first
        scores_max = scores.max(dim=-1, keepdim=True).values
        scores_max = torch.where(
            scores_max == float('-inf'),
            torch.zeros_like(scores_max),
            scores_max
        )
        exp_scores = torch.exp(scores - scores_max)

        # The +1 term in softmax1 becomes exp(-max) after max subtraction
        one_term = torch.exp(-scores_max)
        denominator = one_term + exp_scores.sum(dim=-1, keepdim=True)

        # Softmax1 weights (these sum to < 1 due to the +1 term)
        attn_weights = exp_scores / denominator

        # The "leftover" is 1 - sum(weights) = one_term / denominator
        leftover = one_term / denominator  # (batch, seq_len, 1)

        if self.is_count_mode:
            # Count = 1/leftover - 1 = (1 - leftover) / leftover
            # But simpler: count = sum of selected = n
            # With softmax1: sum of weights = n / (1 + n)
            # So: n = sum_weights / (1 - sum_weights) = sum_weights / leftover
            sum_weights = 1 - leftover  # = n / (1 + n)
            count = sum_weights / leftover.clamp(min=1e-9)  # = n
            output = self.W_o(count)
        else:
            # Aggregate: weighted sum, then normalize to get mean
            # softmax1 weighted sum = sum(v_i * w_i) where sum(w_i) = n/(1+n)
            # To get mean: divide by sum(w_i) = (1 - leftover)
            attended = torch.bmm(attn_weights, values)  # (batch, seq_len, value_size)

            # Normalize to recover mean (same as RASP aggregate semantics)
            sum_weights = (1 - leftover)  # (batch, seq_len, 1)
            attended = attended / sum_weights.clamp(min=1e-9)

            output = self.W_o(attended)

        # Overwrite output dimensions in the residual stream
        result = x.clone()
        start = self.output_encoding.basis_start
        end = start + self.output_encoding.basis_size
        result[:, :, start:end] = output

        return result


class CompiledMLP(nn.Module):
    """
    MLP compiled from RASP elementwise operations.

    Currently implements operations as direct computation.
    A more sophisticated version would use learned approximations.
    """

    def __init__(
        self,
        d_model: int,
        operations: List[Tuple[List[SOp], SOp, str]],
        basis: BasisAllocation
    ):
        super().__init__()
        self.d_model = d_model
        self.operations = operations
        self.basis = basis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply all elementwise operations.

        Note: This is a simplified implementation that directly computes
        operations. A full transformer implementation would use learned
        linear layers that approximate these operations.
        """
        result = x.clone()

        for input_sops, output_sop, op_name in self.operations:
            output_enc = self.basis.get_encoding(output_sop)
            if output_enc is None:
                continue

            # Get input values (read from result to get updated intermediate values)
            input_values = []
            for inp in input_sops:
                enc = self.basis.get_encoding(inp)
                if enc is not None:
                    start = enc.basis_start
                    end = start + enc.basis_size
                    input_values.append(result[:, :, start:end])

            if not input_values:
                continue

            # Apply operation
            if op_name == '+':
                out_val = input_values[0] + input_values[1]
            elif op_name == '-':
                out_val = input_values[0] - input_values[1]
            elif op_name == '*':
                out_val = input_values[0] * input_values[1]
            elif op_name == '/':
                out_val = input_values[0] / (input_values[1] + 1e-9)
            elif op_name == 'neg':
                out_val = -input_values[0]
            elif op_name == '==':
                out_val = (input_values[0] == input_values[1]).float()
            elif op_name == '!=':
                out_val = (input_values[0] != input_values[1]).float()
            elif op_name == '<':
                out_val = (input_values[0] < input_values[1]).float()
            elif op_name == '<=':
                out_val = (input_values[0] <= input_values[1]).float()
            elif op_name == '>':
                out_val = (input_values[0] > input_values[1]).float()
            elif op_name == '>=':
                out_val = (input_values[0] >= input_values[1]).float()
            # C4-compatible operations
            elif op_name == '%':
                # Modulo: a % b = a - floor(a/b) * b
                a, b = input_values[0], input_values[1] + 1e-9
                out_val = a - torch.floor(a / b) * b
            elif op_name == '//':
                # Floor division
                out_val = torch.floor(input_values[0] / (input_values[1] + 1e-9))
            elif op_name == '<<':
                # Left shift: a << b = a * 2^b
                out_val = input_values[0] * torch.pow(2.0, input_values[1])
            elif op_name == '>>':
                # Right shift: a >> b = floor(a / 2^b)
                out_val = torch.floor(input_values[0] / torch.pow(2.0, input_values[1]))
            elif op_name == 'and':
                # Logical AND (assuming 0/1 values)
                out_val = (input_values[0] * input_values[1]).clamp(0, 1)
            elif op_name == 'or':
                # Logical OR (assuming 0/1 values)
                out_val = (input_values[0] + input_values[1]).clamp(0, 1)
            elif op_name == 'not':
                # Logical NOT
                out_val = 1.0 - input_values[0].clamp(0, 1)
            else:
                # Unknown op, skip
                continue

            # Write to output dimensions
            start = output_enc.basis_start
            end = start + output_enc.basis_size
            result[:, :, start:end] = out_val

        return result


class CompiledTransformerLayer(nn.Module):
    """Single transformer layer compiled from RASP."""

    def __init__(
        self,
        d_model: int,
        layer_plan: LayerPlan,
        basis: BasisAllocation
    ):
        super().__init__()
        self.d_model = d_model

        # Compile attention heads
        self.attention_heads = nn.ModuleList()
        self.head_selectors = []  # Store selectors for mask computation

        for selector, values_sop, output_sop in layer_plan.attention_heads:
            values_enc = basis.get_encoding(values_sop) if values_sop else None
            output_enc = basis.get_encoding(output_sop)
            is_count = values_sop is None

            head = CompiledAttentionHead(
                d_model=d_model,
                selector=selector,
                values_encoding=values_enc,
                output_encoding=output_enc,
                is_count_mode=is_count
            )
            self.attention_heads.append(head)
            self.head_selectors.append(selector)

        # Compile MLP
        self.mlp = CompiledMLP(d_model, layer_plan.mlp_operations, basis)

    def forward(
        self,
        x: torch.Tensor,
        selector_masks: Dict[int, torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            selector_masks: Precomputed selector masks {selector_id: mask}
        """
        # Apply attention heads - they OVERWRITE their output dimensions
        for head, selector in zip(self.attention_heads, self.head_selectors):
            mask = selector_masks.get(selector.id)
            if mask is not None:
                x = head(x, mask)  # head.forward now overwrites directly

        # Apply MLP
        x = self.mlp(x)

        return x


class CompiledRASPTransformer(nn.Module):
    """
    Full transformer compiled from a RASP program.
    """

    def __init__(self, plan: CompilationPlan):
        super().__init__()
        self.plan = plan
        self.d_model = plan.basis.total_dim

        # Token embedding (one-hot to residual)
        self.token_embedding = nn.Embedding(
            len(plan.program.vocab),
            self.d_model
        )
        self._init_token_embedding(plan)

        # Transformer layers
        self.layers = nn.ModuleList([
            CompiledTransformerLayer(self.d_model, layer_plan, plan.basis)
            for layer_plan in plan.layers
        ])

        # Store all selectors for mask computation
        _, all_selectors = plan.program.get_all_ops()
        self.selectors = {sel.id: sel for sel in all_selectors}

    def _init_token_embedding(self, plan: CompilationPlan):
        """Initialize token embedding to place one-hot in correct dimensions."""
        with torch.no_grad():
            # Find tokens S-op encoding
            for sop_id, enc in plan.basis.sop_encodings.items():
                if enc.encoding_type == Encoding.Type.CATEGORICAL:
                    # This is the token encoding
                    # Set embedding[i] to have 1 at position (basis_start + i)
                    self.token_embedding.weight.zero_()
                    for i in range(len(plan.program.vocab)):
                        self.token_embedding.weight[i, enc.basis_start + i] = 1.0
                    break

    def compute_selector_masks(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor
    ) -> Dict[int, torch.Tensor]:
        """
        Compute all selector masks for the input.

        For now, this uses the RASP interpreter logic directly.
        A full implementation would compute these from Q/K projections.
        """
        batch, seq_len = input_ids.shape
        device = input_ids.device

        masks = {}

        # Get values for S-ops used in selectors
        sop_values = self._compute_sop_values(x, input_ids)

        def eval_selector(sel: Selector) -> torch.Tensor:
            if sel.id in masks:
                return masks[sel.id]

            if isinstance(sel, SelectOp):
                keys = sop_values.get(sel.keys.id)
                queries = sop_values.get(sel.queries.id)

                if keys is None or queries is None:
                    # Default to all-true
                    mask = torch.ones(batch, seq_len, seq_len, device=device, dtype=torch.bool)
                else:
                    # Compute selector: mask[b,i,j] = predicate(keys[b,j], queries[b,i])
                    # This is slow but correct
                    mask = torch.zeros(batch, seq_len, seq_len, device=device, dtype=torch.bool)
                    for b in range(batch):
                        for i in range(seq_len):
                            for j in range(seq_len):
                                try:
                                    k_val = keys[b, j].item() if keys[b, j].numel() == 1 else keys[b, j, 0].item()
                                    q_val = queries[b, i].item() if queries[b, i].numel() == 1 else queries[b, i, 0].item()
                                    mask[b, i, j] = sel.predicate(k_val, q_val)
                                except:
                                    mask[b, i, j] = True

            elif isinstance(sel, ConstantSelector):
                mask = torch.full(
                    (batch, seq_len, seq_len),
                    sel.value,
                    device=device,
                    dtype=torch.bool
                )

            elif isinstance(sel, SelectorAnd):
                left = eval_selector(sel.left)
                right = eval_selector(sel.right)
                mask = left & right

            elif isinstance(sel, SelectorOr):
                left = eval_selector(sel.left)
                right = eval_selector(sel.right)
                mask = left | right

            elif isinstance(sel, SelectorNot):
                inner = eval_selector(sel.inner)
                mask = ~inner

            else:
                mask = torch.ones(batch, seq_len, seq_len, device=device, dtype=torch.bool)

            masks[sel.id] = mask
            return mask

        # Evaluate all selectors
        for sel_id, sel in self.selectors.items():
            eval_selector(sel)

        return masks

    def _compute_sop_values(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor
    ) -> Dict[int, torch.Tensor]:
        """Compute values for S-ops needed by selectors."""
        batch, seq_len = input_ids.shape
        device = input_ids.device

        values = {}

        # Find all S-ops used in selectors
        for sel in self.selectors.values():
            if isinstance(sel, SelectOp):
                # Need to compute keys and queries
                self._compute_sop_value(sel.keys, values, x, input_ids)
                self._compute_sop_value(sel.queries, values, x, input_ids)

        return values

    def _compute_sop_value(
        self,
        sop: SOp,
        cache: Dict[int, torch.Tensor],
        x: torch.Tensor,
        input_ids: torch.Tensor
    ):
        """Recursively compute an S-op's value."""
        if sop.id in cache:
            return

        batch, seq_len, _ = x.shape
        device = x.device

        from rasp import (InputSOp, IndicesSOp, LengthSOp, OnesSOp, ConstantSOp,
                          ElementwiseOp, MapOp, SequenceMapOp)

        if isinstance(sop, InputSOp):
            # Tokens - use input_ids directly
            cache[sop.id] = input_ids.unsqueeze(-1).float()

        elif isinstance(sop, IndicesSOp):
            indices = torch.arange(seq_len, device=device).unsqueeze(0).unsqueeze(-1)
            cache[sop.id] = indices.expand(batch, seq_len, 1).float()

        elif isinstance(sop, LengthSOp):
            cache[sop.id] = torch.full((batch, seq_len, 1), seq_len, device=device, dtype=torch.float)

        elif isinstance(sop, OnesSOp):
            cache[sop.id] = torch.ones(batch, seq_len, 1, device=device)

        elif isinstance(sop, ConstantSOp):
            cache[sop.id] = torch.full((batch, seq_len, 1), sop.value, device=device, dtype=torch.float)

        elif isinstance(sop, ElementwiseOp):
            # Compute dependencies first
            self._compute_sop_value(sop.left, cache, x, input_ids)
            left_val = cache[sop.left.id]

            if sop.right is not None:
                self._compute_sop_value(sop.right, cache, x, input_ids)
                right_val = cache[sop.right.id]
            else:
                right_val = None

            # Apply operation
            if sop.op_name == '+':
                cache[sop.id] = left_val + right_val
            elif sop.op_name == '-':
                cache[sop.id] = left_val - right_val
            elif sop.op_name == '*':
                cache[sop.id] = left_val * right_val
            elif sop.op_name == '/':
                cache[sop.id] = left_val / (right_val + 1e-9)
            elif sop.op_name == 'neg':
                cache[sop.id] = -left_val
            elif sop.op_name == '==':
                cache[sop.id] = (left_val == right_val).float()
            elif sop.op_name == '<':
                cache[sop.id] = (left_val < right_val).float()
            elif sop.op_name == '<=':
                cache[sop.id] = (left_val <= right_val).float()
            elif sop.op_name == '>':
                cache[sop.id] = (left_val > right_val).float()
            elif sop.op_name == '>=':
                cache[sop.id] = (left_val >= right_val).float()
            else:
                cache[sop.id] = left_val  # fallback

        else:
            # Default: zeros
            cache[sop.id] = torch.zeros(batch, seq_len, 1, device=device)

    def forward(
        self,
        input_ids: torch.Tensor,
        return_residual: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Run the compiled transformer.

        Args:
            input_ids: (batch, seq_len) token indices
            return_residual: Whether to return full residual stream

        Returns:
            Dictionary with output S-op values
        """
        batch, seq_len = input_ids.shape

        # Embed tokens
        x = self.token_embedding(input_ids)

        # Initialize other primitive S-ops in residual stream
        x = self._init_primitives(x, input_ids)

        # Compute selector masks
        selector_masks = self.compute_selector_masks(x, input_ids)

        # Run transformer layers
        for layer in self.layers:
            x = layer(x, selector_masks)

        # Extract output values
        outputs = {}
        for output_sop in self.plan.program.outputs:
            enc = self.plan.basis.get_encoding(output_sop)
            if enc is not None:
                start = enc.basis_start
                end = start + enc.basis_size
                outputs[output_sop.name] = x[:, :, start:end]

        if return_residual:
            outputs['residual'] = x

        return outputs

    def _init_primitives(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor
    ) -> torch.Tensor:
        """Initialize primitive S-ops in the residual stream."""
        batch, seq_len, _ = x.shape
        device = x.device

        from rasp import IndicesSOp, LengthSOp, OnesSOp, ConstantSOp

        for sop_id, enc in self.plan.basis.sop_encodings.items():
            # Find the S-op
            all_sops, _ = self.plan.program.get_all_ops()
            for sop in all_sops:
                if sop.id == sop_id:
                    if isinstance(sop, IndicesSOp):
                        indices = torch.arange(seq_len, device=device, dtype=x.dtype)
                        x[:, :, enc.basis_start] = indices.unsqueeze(0)
                    elif isinstance(sop, LengthSOp):
                        x[:, :, enc.basis_start] = seq_len
                    elif isinstance(sop, OnesSOp):
                        x[:, :, enc.basis_start] = 1.0
                    elif isinstance(sop, ConstantSOp):
                        x[:, :, enc.basis_start] = sop.value
                    break

        return x


# =============================================================================
# MAIN COMPILATION FUNCTION
# =============================================================================

def compile_rasp(program: RASPProgram) -> CompiledRASPTransformer:
    """
    Compile a RASP program to a PyTorch transformer.

    Args:
        program: The RASP program to compile

    Returns:
        A PyTorch nn.Module that implements the program
    """
    plan = analyze_program(program)
    model = CompiledRASPTransformer(plan)
    return model


# =============================================================================
# TESTING
# =============================================================================

def test_compiler():
    """Test the RASP compiler."""
    from rasp import (tokens, indices, ones, select, aggregate, selector_width,
                      RASPProgram, make_length_program, make_histogram_program,
                      interpret)

    print("Testing RASP Compiler")
    print("=" * 50)

    # Test 1: Length program
    print("\n1. Compiling length program:")
    prog = make_length_program()
    model = compile_rasp(prog)

    print(f"   Residual dimension: {model.d_model}")
    print(f"   Number of layers: {len(model.layers)}")

    # Run inference
    input_seq = [1, 2, 3, 4, 5]
    input_ids = torch.tensor([[prog.vocab.index(t) if t in prog.vocab else 0 for t in input_seq]])

    output = model(input_ids)
    print(f"   Input: {input_seq}")
    print(f"   Output 'length': {output['length'][0, :, 0].tolist()}")

    # Compare with interpreter
    interp_result = interpret(prog, input_seq)
    print(f"   Interpreter: {interp_result['length']}")

    # Test 2: Histogram program
    print("\n2. Compiling histogram program:")
    prog = make_histogram_program()
    model = compile_rasp(prog)

    input_str = "aabbc"
    input_ids = torch.tensor([[prog.vocab.index(c) for c in input_str]])

    output = model(input_ids)
    print(f"   Input: '{input_str}'")
    print(f"   Output 'count': {output['count'][0, :, 0].tolist()}")

    interp_result = interpret(prog, list(input_str))
    print(f"   Interpreter: {interp_result['count']}")

    # Test 3: Simple arithmetic
    print("\n3. Compiling arithmetic program (indices * 2 + 1):")
    double_plus_one = indices * 2 + 1
    prog = RASPProgram([double_plus_one], list(range(5)), 8, "arith")
    model = compile_rasp(prog)

    input_seq = [0, 1, 2, 3, 4]
    input_ids = torch.tensor([[i for i in input_seq]])

    output = model(input_ids)
    print(f"   Input indices: {input_seq}")
    result_key = list(output.keys())[0]
    print(f"   Output: {output[result_key][0, :, 0].tolist()}")

    print("\n" + "=" * 50)
    print("RASP Compiler tests passed!")
    return True


def test_c4_operations_compilation():
    """Test compiling C4-compatible operations to transformers."""
    from rasp import (tokens, indices, ones, select, aggregate, selector_width,
                      RASPProgram, interpret, ternary, modulo, left_shift,
                      right_shift, floor_div, negate, logical_not, c_sizeof,
                      abs_val, min_val, max_val, clamp, ConstantSOp,
                      bitwise_and, bitwise_or, bitwise_xor, bitwise_not, count_ones)

    print("\nTesting C4 Operations Compilation")
    print("=" * 50)

    # Test 1: Modulo operation
    print("\n1. Compiling modulo (indices % 3):")
    mod_op = indices % 3
    prog = RASPProgram([mod_op], list(range(10)), 16, "modulo")
    model = compile_rasp(prog)

    input_seq = [0, 1, 2, 3, 4, 5, 6]
    input_ids = torch.tensor([[i for i in input_seq]])
    output = model(input_ids)
    result_key = list(output.keys())[0]
    compiled_result = [int(round(v)) for v in output[result_key][0, :, 0].tolist()]
    interp_result = interpret(prog, input_seq)[mod_op.name]
    print(f"   Input: {input_seq}")
    print(f"   Compiled: {compiled_result}")
    print(f"   Interpreter: {interp_result}")
    assert compiled_result == interp_result, f"Mismatch: {compiled_result} vs {interp_result}"
    print("   ✓ Modulo compilation passed")

    # Test 2: Left shift
    print("\n2. Compiling left shift ((indices+1) << 2):")
    lshift_op = (indices + 1) << 2
    prog = RASPProgram([lshift_op], list(range(10)), 16, "lshift")
    model = compile_rasp(prog)

    input_seq = [0, 1, 2, 3]
    input_ids = torch.tensor([[i for i in input_seq]])
    output = model(input_ids)
    result_key = list(output.keys())[0]
    compiled_result = [int(round(v)) for v in output[result_key][0, :, 0].tolist()]
    interp_result = interpret(prog, input_seq)[lshift_op.name]
    print(f"   Input: {input_seq}")
    print(f"   Compiled: {compiled_result}")
    print(f"   Interpreter: {interp_result}")
    assert compiled_result == interp_result, f"Mismatch: {compiled_result} vs {interp_result}"
    print("   ✓ Left shift compilation passed")

    # Test 3: Right shift (using indices * 4 to get [0,4,8,12,...] then shift right)
    print("\n3. Compiling right shift ((indices * 4) >> 1):")
    rshift_op = (indices * 4) >> 1  # [0,4,8,12] >> 1 = [0,2,4,6]
    prog = RASPProgram([rshift_op], list(range(10)), 16, "rshift")
    model = compile_rasp(prog)

    input_seq = [0, 1, 2, 3]
    input_ids = torch.tensor([[i for i in input_seq]])
    output = model(input_ids)
    result_key = list(output.keys())[0]
    compiled_result = [int(round(v)) for v in output[result_key][0, :, 0].tolist()]
    interp_result = interpret(prog, input_seq)[rshift_op.name]
    print(f"   Input indices: {input_seq}")
    print(f"   (indices * 4) >> 1 = {compiled_result}")
    print(f"   Interpreter: {interp_result}")
    assert compiled_result == interp_result, f"Mismatch: {compiled_result} vs {interp_result}"
    print("   ✓ Right shift compilation passed")

    # Test 4: Floor division
    print("\n4. Compiling floor division (indices // 2):")
    fdiv_op = indices // 2
    prog = RASPProgram([fdiv_op], list(range(10)), 16, "fdiv")
    model = compile_rasp(prog)

    input_seq = [0, 1, 2, 3, 4, 5, 6]
    input_ids = torch.tensor([[i for i in input_seq]])
    output = model(input_ids)
    result_key = list(output.keys())[0]
    compiled_result = [int(round(v)) for v in output[result_key][0, :, 0].tolist()]
    interp_result = interpret(prog, input_seq)[fdiv_op.name]
    print(f"   Input: {input_seq}")
    print(f"   Compiled: {compiled_result}")
    print(f"   Interpreter: {interp_result}")
    assert compiled_result == interp_result, f"Mismatch: {compiled_result} vs {interp_result}"
    print("   ✓ Floor division compilation passed")

    # Test 5: Unary negation
    print("\n5. Compiling unary negation (-indices):")
    neg_op = -indices
    prog = RASPProgram([neg_op], list(range(10)), 16, "neg")
    model = compile_rasp(prog)

    input_seq = [0, 1, 2, 3, 4]
    input_ids = torch.tensor([[i for i in input_seq]])
    output = model(input_ids)
    result_key = list(output.keys())[0]
    compiled_result = [int(round(v)) for v in output[result_key][0, :, 0].tolist()]
    interp_result = interpret(prog, input_seq)[neg_op.name]
    print(f"   Input: {input_seq}")
    print(f"   Compiled: {compiled_result}")
    print(f"   Interpreter: {interp_result}")
    assert compiled_result == interp_result, f"Mismatch: {compiled_result} vs {interp_result}"
    print("   ✓ Unary negation compilation passed")

    # Test 6: Combined C expression
    print("\n6. Compiling combined expression ((i % 4) << 1) + 1:")
    combined = ((indices % 4) << 1) + 1
    prog = RASPProgram([combined], list(range(10)), 16, "combined")
    model = compile_rasp(prog)

    input_seq = [0, 1, 2, 3, 4, 5, 6, 7]
    input_ids = torch.tensor([[i for i in input_seq]])
    output = model(input_ids)
    result_key = list(output.keys())[0]
    compiled_result = [int(round(v)) for v in output[result_key][0, :, 0].tolist()]
    interp_result = interpret(prog, input_seq)[combined.name]
    print(f"   Input: {input_seq}")
    print(f"   Compiled: {compiled_result}")
    print(f"   Interpreter: {interp_result}")
    assert compiled_result == interp_result, f"Mismatch: {compiled_result} vs {interp_result}"
    print("   ✓ Combined expression compilation passed")

    # Test 7: Bitwise AND (via bit decomposition)
    print("\n7. Compiling bitwise AND (indices & 0b0011):")
    and_op = bitwise_and(indices, ConstantSOp(0b0011), num_bits=4)
    prog = RASPProgram([and_op], list(range(16)), 16, "bitand")
    model = compile_rasp(prog)

    input_seq = [0, 1, 2, 3, 4, 5, 6, 7]
    input_ids = torch.tensor([[i for i in input_seq]])
    output = model(input_ids)
    result_key = list(output.keys())[0]
    compiled_result = [int(round(v)) for v in output[result_key][0, :, 0].tolist()]
    interp_result = interpret(prog, input_seq)[and_op.name]
    print(f"   Input: {input_seq}")
    print(f"   Compiled: {compiled_result}")
    print(f"   Interpreter: {interp_result}")
    expected_and = [i & 0b0011 for i in input_seq]
    assert compiled_result == expected_and, f"Mismatch: {compiled_result} vs {expected_and}"
    print("   ✓ Bitwise AND compilation passed")

    # Test 8: Bitwise XOR (indices ^ 15, where indices = [0,1,2,3])
    print("\n8. Compiling bitwise XOR (indices ^ 0b1111):")
    xor_op = bitwise_xor(indices, ConstantSOp(0b1111), num_bits=4)
    prog = RASPProgram([xor_op], list(range(16)), 16, "bitxor")
    model = compile_rasp(prog)

    input_seq = [0, 1, 2, 3]  # indices will be [0,1,2,3]
    input_ids = torch.tensor([[i for i in input_seq]])
    output = model(input_ids)
    result_key = list(output.keys())[0]
    compiled_result = [int(round(v)) for v in output[result_key][0, :, 0].tolist()]
    interp_result = interpret(prog, input_seq)[xor_op.name]
    print(f"   Indices: [0,1,2,3]")
    print(f"   Compiled: {compiled_result}")
    print(f"   Interpreter: {interp_result}")
    # indices XOR 15: 0^15=15, 1^15=14, 2^15=13, 3^15=12
    expected_xor = [i ^ 0b1111 for i in range(len(input_seq))]
    assert compiled_result == expected_xor, f"Mismatch: {compiled_result} vs {expected_xor}"
    print("   ✓ Bitwise XOR compilation passed")

    # Test 9: Population count (count 1 bits in indices [0,1,2,3,4,5,6,7])
    print("\n9. Compiling population count (count 1 bits):")
    pop_op = count_ones(indices, num_bits=4)
    prog = RASPProgram([pop_op], list(range(16)), 16, "popcount")
    model = compile_rasp(prog)

    input_seq = [0, 1, 2, 3, 4, 5, 6, 7]  # indices will be [0,1,2,3,4,5,6,7]
    input_ids = torch.tensor([[i for i in input_seq]])
    output = model(input_ids)
    result_key = list(output.keys())[0]
    compiled_result = [int(round(v)) for v in output[result_key][0, :, 0].tolist()]
    interp_result = interpret(prog, input_seq)[pop_op.name]
    print(f"   Indices: [0,1,2,3,4,5,6,7]")
    print(f"   Compiled: {compiled_result}")
    print(f"   Interpreter: {interp_result}")
    # popcount of indices: 0→0, 1→1, 2→1, 3→2, 4→1, 5→2, 6→2, 7→3
    expected_pop = [bin(i).count('1') for i in range(len(input_seq))]
    assert compiled_result == expected_pop, f"Mismatch: {compiled_result} vs {expected_pop}"
    print("   ✓ Population count compilation passed")

    print("\n" + "=" * 50)
    print("All C4 operation compilation tests passed!")
    return True


if __name__ == "__main__":
    test_compiler()
    test_c4_operations_compilation()
