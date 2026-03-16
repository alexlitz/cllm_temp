"""
Tracr to PyTorch adapter.

Tracr (https://github.com/google-deepmind/tracr) compiles RASP programs to
transformer weights in JAX/Haiku format. This module provides:

1. A PyTorch transformer implementation compatible with tracr's architecture
2. Conversion functions from tracr Haiku params to PyTorch
3. Integration with hidden_dim_ops for adding custom operations

Usage:
    from tracr.compiler import compiling
    from tracr_pytorch import TrackedTransformer, convert_tracr_to_pytorch

    # Compile RASP program with tracr
    model = compiling.compile_rasp_to_model(program, vocab, max_seq_len)

    # Convert to PyTorch
    pt_model = convert_tracr_to_pytorch(model)

    # Run inference
    output = pt_model(input_tokens)
"""

import math
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TrackerConfig:
    """Configuration matching tracr's transformer setup."""
    num_layers: int
    num_heads: int
    d_model: int  # hidden dimension
    key_size: int  # attention dimension per head
    mlp_hidden_size: int
    vocab_size: int
    max_seq_len: int
    layer_norm: bool = True
    causal: bool = False
    dropout_rate: float = 0.0
    activation: str = 'gelu'  # tracr default


class MultiHeadAttention(nn.Module):
    """Multi-head attention compatible with tracr's implementation."""

    def __init__(self, config: TrackerConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.key_size = config.key_size
        self.d_model = config.d_model
        self.causal = config.causal

        # Q, K, V projections (per head)
        self.W_q = nn.Linear(config.d_model, config.num_heads * config.key_size, bias=True)
        self.W_k = nn.Linear(config.d_model, config.num_heads * config.key_size, bias=True)
        self.W_v = nn.Linear(config.d_model, config.num_heads * config.key_size, bias=True)

        # Output projection
        self.W_o = nn.Linear(config.num_heads * config.key_size, config.d_model, bias=True)

        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: Optional attention mask
            return_attention: Whether to return attention weights

        Returns:
            output: (batch, seq_len, d_model)
            attention: Optional attention weights (batch, heads, seq_len, seq_len)
        """
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.W_q(x)  # (batch, seq_len, heads * key_size)
        k = self.W_k(x)
        v = self.W_v(x)

        # Reshape to (batch, heads, seq_len, key_size)
        q = q.view(batch, seq_len, self.num_heads, self.key_size).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.key_size).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.key_size).transpose(1, 2)

        # Scaled dot-product attention
        scale = math.sqrt(self.key_size)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # (batch, heads, seq, seq)

        # Apply causal mask if needed
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1
            )
            scores = scores.masked_fill(causal_mask, float('-inf'))

        # Apply external mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)  # (batch, heads, seq_len, key_size)

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        out = self.W_o(out)

        if return_attention:
            return out, attn
        return out, None


class MLP(nn.Module):
    """Feed-forward network compatible with tracr."""

    def __init__(self, config: TrackerConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.mlp_hidden_size)
        self.fc2 = nn.Linear(config.mlp_hidden_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout_rate)

        if config.activation == 'gelu':
            self.activation = F.gelu
        elif config.activation == 'relu':
            self.activation = F.relu
        elif config.activation == 'silu':
            self.activation = F.silu
        else:
            raise ValueError(f"Unknown activation: {config.activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerLayer(nn.Module):
    """Single transformer layer compatible with tracr."""

    def __init__(self, config: TrackerConfig):
        super().__init__()
        self.config = config

        self.attention = MultiHeadAttention(config)
        self.mlp = MLP(config)

        if config.layer_norm:
            self.ln1 = nn.LayerNorm(config.d_model)
            self.ln2 = nn.LayerNorm(config.d_model)
        else:
            self.ln1 = nn.Identity()
            self.ln2 = nn.Identity()

        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Pre-norm attention
        normed = self.ln1(x)
        attn_out, attn_weights = self.attention(normed, mask, return_attention)
        x = x + self.dropout(attn_out)

        # Pre-norm MLP
        normed = self.ln2(x)
        mlp_out = self.mlp(normed)
        x = x + self.dropout(mlp_out)

        return x, attn_weights


class TrackedTransformer(nn.Module):
    """
    PyTorch transformer compatible with tracr's architecture.

    This model can:
    1. Load weights converted from tracr's Haiku format
    2. Run inference matching tracr's output
    3. Be extended with custom operations from hidden_dim_ops
    """

    def __init__(self, config: TrackerConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Positional embedding
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.num_layers)
        ])

        # Output projection (unembedding)
        self.unembed = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Final layer norm (optional, some tracr models use this)
        if config.layer_norm:
            self.final_ln = nn.LayerNorm(config.d_model)
        else:
            self.final_ln = nn.Identity()

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_intermediates: bool = False
    ) -> Dict[str, Any]:
        """
        Args:
            input_ids: (batch, seq_len) token indices
            mask: Optional attention mask
            return_intermediates: Whether to return layer-by-layer outputs

        Returns:
            Dictionary with:
                - 'output': final embeddings (batch, seq_len, d_model)
                - 'logits': vocabulary logits (batch, seq_len, vocab_size)
                - 'residuals': list of residual states (if return_intermediates)
                - 'attentions': list of attention weights (if return_intermediates)
        """
        batch, seq_len = input_ids.shape
        device = input_ids.device

        # Embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.pos_embedding(positions)

        # Track intermediates
        residuals = [x] if return_intermediates else None
        attentions = [] if return_intermediates else None

        # Transformer layers
        for layer in self.layers:
            x, attn = layer(x, mask, return_attention=return_intermediates)
            if return_intermediates:
                residuals.append(x)
                attentions.append(attn)

        # Final layer norm and unembedding
        x = self.final_ln(x)
        logits = self.unembed(x)

        result = {
            'output': x,
            'logits': logits,
        }

        if return_intermediates:
            result['residuals'] = residuals
            result['attentions'] = attentions

        return result

    def get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get just the final embeddings (before unembedding)."""
        return self.forward(input_ids)['output']

    def get_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get vocabulary logits."""
        return self.forward(input_ids)['logits']


def convert_tracr_to_pytorch(
    tracr_model,
    return_config: bool = False
) -> TrackedTransformer:
    """
    Convert a tracr compiled model to PyTorch.

    Args:
        tracr_model: The tracr AssembledTransformerModel
        return_config: Whether to also return the config

    Returns:
        PyTorch TrackedTransformer (and optionally config)
    """
    # Extract config from tracr model
    # tracr models have .model_config attribute
    tc = tracr_model.model_config

    config = TrackerConfig(
        num_layers=tc.num_layers,
        num_heads=tc.num_heads,
        d_model=tc.key_size * tc.num_heads,  # tracr uses key_size per head
        key_size=tc.key_size,
        mlp_hidden_size=tc.mlp_hidden_size,
        vocab_size=len(tracr_model.input_encoder.encoding),
        max_seq_len=tc.max_seq_len if hasattr(tc, 'max_seq_len') else 128,
        layer_norm=tc.layer_norm if hasattr(tc, 'layer_norm') else True,
        causal=tc.causal if hasattr(tc, 'causal') else False,
    )

    # Create PyTorch model
    pt_model = TrackedTransformer(config)

    # Get tracr params (Haiku format)
    params = tracr_model.params

    # Convert parameters
    _convert_params(pt_model, params, config)

    if return_config:
        return pt_model, config
    return pt_model


def _convert_params(pt_model: TrackedTransformer, haiku_params: Dict, config: TrackerConfig):
    """
    Convert Haiku parameters to PyTorch.

    Haiku params structure (typical):
        {
            'transformer/embed_tokens': {'embeddings': ...},
            'transformer/positional_embeddings': {'embeddings': ...},
            'transformer/layer_0/attention/query': {'w': ..., 'b': ...},
            'transformer/layer_0/attention/key': {'w': ..., 'b': ...},
            'transformer/layer_0/attention/value': {'w': ..., 'b': ...},
            'transformer/layer_0/attention/linear': {'w': ..., 'b': ...},
            'transformer/layer_0/mlp/linear_1': {'w': ..., 'b': ...},
            'transformer/layer_0/mlp/linear_2': {'w': ..., 'b': ...},
            'transformer/layer_0/layer_norm_1': {'scale': ..., 'offset': ...},
            'transformer/layer_0/layer_norm_2': {'scale': ..., 'offset': ...},
            ...
            'transformer/unembed': {'w': ...},
        }
    """
    with torch.no_grad():
        # Helper to convert JAX array to PyTorch
        def to_torch(arr):
            import numpy as np
            if hasattr(arr, 'numpy'):
                arr = arr.numpy()  # JAX array
            return torch.from_numpy(np.array(arr))

        # Token embeddings
        if 'transformer/embed_tokens' in haiku_params:
            emb = haiku_params['transformer/embed_tokens']['embeddings']
            pt_model.token_embedding.weight.copy_(to_torch(emb))

        # Positional embeddings
        if 'transformer/positional_embeddings' in haiku_params:
            pos = haiku_params['transformer/positional_embeddings']['embeddings']
            pt_model.pos_embedding.weight.copy_(to_torch(pos))

        # Transformer layers
        for i in range(config.num_layers):
            prefix = f'transformer/layer_{i}'
            layer = pt_model.layers[i]

            # Attention
            if f'{prefix}/attention/query' in haiku_params:
                layer.attention.W_q.weight.copy_(
                    to_torch(haiku_params[f'{prefix}/attention/query']['w']).T
                )
                if 'b' in haiku_params[f'{prefix}/attention/query']:
                    layer.attention.W_q.bias.copy_(
                        to_torch(haiku_params[f'{prefix}/attention/query']['b']).flatten()
                    )

            if f'{prefix}/attention/key' in haiku_params:
                layer.attention.W_k.weight.copy_(
                    to_torch(haiku_params[f'{prefix}/attention/key']['w']).T
                )
                if 'b' in haiku_params[f'{prefix}/attention/key']:
                    layer.attention.W_k.bias.copy_(
                        to_torch(haiku_params[f'{prefix}/attention/key']['b']).flatten()
                    )

            if f'{prefix}/attention/value' in haiku_params:
                layer.attention.W_v.weight.copy_(
                    to_torch(haiku_params[f'{prefix}/attention/value']['w']).T
                )
                if 'b' in haiku_params[f'{prefix}/attention/value']:
                    layer.attention.W_v.bias.copy_(
                        to_torch(haiku_params[f'{prefix}/attention/value']['b']).flatten()
                    )

            if f'{prefix}/attention/linear' in haiku_params:
                layer.attention.W_o.weight.copy_(
                    to_torch(haiku_params[f'{prefix}/attention/linear']['w']).T
                )
                if 'b' in haiku_params[f'{prefix}/attention/linear']:
                    layer.attention.W_o.bias.copy_(
                        to_torch(haiku_params[f'{prefix}/attention/linear']['b'])
                    )

            # MLP
            if f'{prefix}/mlp/linear_1' in haiku_params:
                layer.mlp.fc1.weight.copy_(
                    to_torch(haiku_params[f'{prefix}/mlp/linear_1']['w']).T
                )
                if 'b' in haiku_params[f'{prefix}/mlp/linear_1']:
                    layer.mlp.fc1.bias.copy_(
                        to_torch(haiku_params[f'{prefix}/mlp/linear_1']['b'])
                    )

            if f'{prefix}/mlp/linear_2' in haiku_params:
                layer.mlp.fc2.weight.copy_(
                    to_torch(haiku_params[f'{prefix}/mlp/linear_2']['w']).T
                )
                if 'b' in haiku_params[f'{prefix}/mlp/linear_2']:
                    layer.mlp.fc2.bias.copy_(
                        to_torch(haiku_params[f'{prefix}/mlp/linear_2']['b'])
                    )

            # Layer norms
            if config.layer_norm:
                if f'{prefix}/layer_norm_1' in haiku_params:
                    layer.ln1.weight.copy_(
                        to_torch(haiku_params[f'{prefix}/layer_norm_1']['scale'])
                    )
                    layer.ln1.bias.copy_(
                        to_torch(haiku_params[f'{prefix}/layer_norm_1']['offset'])
                    )
                if f'{prefix}/layer_norm_2' in haiku_params:
                    layer.ln2.weight.copy_(
                        to_torch(haiku_params[f'{prefix}/layer_norm_2']['scale'])
                    )
                    layer.ln2.bias.copy_(
                        to_torch(haiku_params[f'{prefix}/layer_norm_2']['offset'])
                    )

        # Unembedding
        if 'transformer/unembed' in haiku_params:
            pt_model.unembed.weight.copy_(
                to_torch(haiku_params['transformer/unembed']['w']).T
            )


# =============================================================================
# INTEGRATION WITH HIDDEN_DIM_OPS
# =============================================================================

class ExtendedTransformer(TrackedTransformer):
    """
    TrackedTransformer extended with custom operations from hidden_dim_ops.

    This allows adding new computational primitives that aren't in the
    original tracr-compiled model.
    """

    def __init__(self, config: TrackerConfig):
        super().__init__(config)
        self.custom_ops = nn.ModuleDict()

    def add_operation(self, name: str, op: nn.Module):
        """Add a custom operation that can be applied to hidden states."""
        self.custom_ops[name] = op

    def apply_operation(
        self,
        x: torch.Tensor,
        op_name: str,
        **kwargs
    ) -> torch.Tensor:
        """Apply a custom operation to hidden states."""
        if op_name not in self.custom_ops:
            raise ValueError(f"Unknown operation: {op_name}")
        return self.custom_ops[op_name](x, **kwargs)

    def forward_with_custom_ops(
        self,
        input_ids: torch.Tensor,
        ops_after_layer: Optional[Dict[int, List[str]]] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass with custom operations inserted after specified layers.

        Args:
            input_ids: (batch, seq_len) token indices
            ops_after_layer: Dict mapping layer index to list of op names to apply
                             e.g., {2: ['windowed_mean'], 4: ['reciprocal']}
            mask: Optional attention mask

        Returns:
            Same as forward(), but with custom ops applied at specified points
        """
        batch, seq_len = input_ids.shape
        device = input_ids.device

        # Embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.pos_embedding(positions)

        ops_after_layer = ops_after_layer or {}

        # Transformer layers with custom ops
        for i, layer in enumerate(self.layers):
            x, _ = layer(x, mask, return_attention=False)

            # Apply custom ops after this layer
            if i in ops_after_layer:
                for op_name in ops_after_layer[i]:
                    x = self.apply_operation(x, op_name)

        # Final layer norm and unembedding
        x = self.final_ln(x)
        logits = self.unembed(x)

        return {
            'output': x,
            'logits': logits,
        }


def create_extended_model_from_tracr(tracr_model) -> ExtendedTransformer:
    """
    Create an ExtendedTransformer from a tracr model.

    Example usage:
        from tracr.compiler import compiling
        from hidden_dim_ops import WindowedMean, ReciprocalNewton

        # Compile RASP program
        tracr_model = compiling.compile_rasp_to_model(program, vocab, max_seq)

        # Convert to extended PyTorch model
        model = create_extended_model_from_tracr(tracr_model)

        # Add custom operations
        model.add_operation('windowed_mean', WindowedMean(d_model, window_size=4))
        model.add_operation('reciprocal', ReciprocalNewton(d_model, n_iter=5))

        # Run with custom ops after layers 2 and 4
        output = model.forward_with_custom_ops(
            input_ids,
            ops_after_layer={2: ['windowed_mean'], 4: ['reciprocal']}
        )
    """
    # Get config from tracr model
    tc = tracr_model.model_config

    config = TrackerConfig(
        num_layers=tc.num_layers,
        num_heads=tc.num_heads,
        d_model=tc.key_size * tc.num_heads,
        key_size=tc.key_size,
        mlp_hidden_size=tc.mlp_hidden_size,
        vocab_size=len(tracr_model.input_encoder.encoding),
        max_seq_len=tc.max_seq_len if hasattr(tc, 'max_seq_len') else 128,
        layer_norm=tc.layer_norm if hasattr(tc, 'layer_norm') else True,
        causal=tc.causal if hasattr(tc, 'causal') else False,
    )

    # Create extended model
    model = ExtendedTransformer(config)

    # Convert weights
    _convert_params(model, tracr_model.params, config)

    return model


# =============================================================================
# TESTING
# =============================================================================

def test_pytorch_transformer():
    """Test the PyTorch transformer implementation."""
    print("Testing PyTorch Transformer (standalone)")
    print("=" * 50)

    config = TrackerConfig(
        num_layers=2,
        num_heads=4,
        d_model=64,
        key_size=16,
        mlp_hidden_size=128,
        vocab_size=100,
        max_seq_len=32,
    )

    model = TrackedTransformer(config)

    # Test forward pass
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    output = model(input_ids)
    print(f"Input shape: {input_ids.shape}")
    print(f"Output embedding shape: {output['output'].shape}")
    print(f"Logits shape: {output['logits'].shape}")

    # Test with intermediates
    output_full = model(input_ids, return_intermediates=True)
    print(f"Number of residual states: {len(output_full['residuals'])}")
    print(f"Number of attention matrices: {len(output_full['attentions'])}")

    print("\nPASS: PyTorch Transformer")
    return True


def test_extended_transformer():
    """Test the extended transformer with custom ops."""
    print("\nTesting Extended Transformer with Custom Ops")
    print("=" * 50)

    # Import operations from hidden_dim_ops
    try:
        from hidden_dim_ops import WindowedMean, RunningSum, ReciprocalNewton
        ops_available = True
    except ImportError:
        print("hidden_dim_ops not available, skipping custom ops test")
        ops_available = False

    config = TrackerConfig(
        num_layers=4,
        num_heads=4,
        d_model=64,
        key_size=16,
        mlp_hidden_size=128,
        vocab_size=100,
        max_seq_len=32,
    )

    model = ExtendedTransformer(config)

    if ops_available:
        # Add custom operations
        model.add_operation('windowed_mean', WindowedMean(config.d_model, window_size=4))
        model.add_operation('running_sum', RunningSum(config.d_model, method='direct'))

        print(f"Added operations: {list(model.custom_ops.keys())}")

        # Test forward with custom ops
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        # Standard forward
        output_std = model(input_ids)

        # Forward with custom ops
        output_custom = model.forward_with_custom_ops(
            input_ids,
            ops_after_layer={1: ['windowed_mean'], 3: ['running_sum']}
        )

        print(f"Standard output shape: {output_std['output'].shape}")
        print(f"Custom ops output shape: {output_custom['output'].shape}")

        # Outputs should be different due to custom ops
        diff = (output_std['output'] - output_custom['output']).abs().mean().item()
        print(f"Mean difference (should be > 0): {diff:.6f}")

    print("\nPASS: Extended Transformer")
    return True


if __name__ == "__main__":
    test_pytorch_transformer()
    test_extended_transformer()
    print("\nAll tracr_pytorch tests passed!")
