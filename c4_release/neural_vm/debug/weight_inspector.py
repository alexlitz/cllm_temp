"""
Weight Inspector for Neural VM.

Provides tools to inspect model weights, embeddings, and activations.
"""

import torch
from typing import Optional, List, Tuple, Dict, Any


class WeightInspector:
    """Inspect model weights and activations for debugging.

    Usage:
        inspector = WeightInspector(model)

        # Inspect embeddings
        inspector.show_embedding(token_id=256)  # REG_PC token

        # Inspect FFN
        inspector.show_ffn_weights(layer=5, unit=100)
        inspector.show_ffn_activations(layer=5, input_hidden=hidden_state)

        # Inspect attention
        inspector.show_attention_weights(layer=5, head=0)
    """

    def __init__(self, model):
        """Initialize inspector with a model.

        Args:
            model: AutoregressiveVM model instance
        """
        self.model = model
        self.d_model = model.d_model
        self.n_layers = len(model.blocks)
        self.n_heads = model.blocks[0].attn.num_heads

    def show_embedding(
        self,
        token_id: int,
        top_k: int = 10,
        threshold: float = 0.1,
    ) -> Dict[str, Any]:
        """Show embedding vector for a token.

        Args:
            token_id: Token ID to inspect
            top_k: Number of top dimensions to show
            threshold: Minimum absolute value to show

        Returns:
            Dict with embedding info
        """
        embed = self.model.embed.embed.weight[token_id].detach().cpu()

        # Find significant dimensions
        abs_embed = embed.abs()
        top_indices = abs_embed.argsort(descending=True)[:top_k]

        result = {
            "token_id": token_id,
            "norm": embed.norm().item(),
            "max": embed.max().item(),
            "min": embed.min().item(),
            "nonzero": (embed.abs() > threshold).sum().item(),
            "top_dimensions": [
                {"dim": idx.item(), "value": embed[idx].item()}
                for idx in top_indices
                if abs(embed[idx].item()) > threshold
            ],
        }

        print(f"\nEmbedding for token {token_id}:")
        print(f"  Norm: {result['norm']:.4f}")
        print(f"  Range: [{result['min']:.4f}, {result['max']:.4f}]")
        print(f"  Nonzero (>{threshold}): {result['nonzero']}")
        print(f"  Top dimensions:")
        for d in result["top_dimensions"][:5]:
            print(f"    dim {d['dim']:3d}: {d['value']:+.4f}")

        return result

    def show_ffn_weights(
        self,
        layer: int,
        unit: int,
        show_gate: bool = True,
    ) -> Dict[str, Any]:
        """Show FFN weights for a specific unit.

        Args:
            layer: Layer index (0-indexed)
            unit: FFN unit index
            show_gate: Also show gate projection weights

        Returns:
            Dict with weight info
        """
        ffn = self.model.blocks[layer].ffn

        W_up = ffn.W_up.weight.detach().cpu()
        W_down = ffn.W_down.weight.detach().cpu()

        result = {
            "layer": layer,
            "unit": unit,
            "W_up_row": W_up[unit].tolist(),
            "W_down_col": W_down[:, unit].tolist(),
            "W_up_norm": W_up[unit].norm().item(),
            "W_down_norm": W_down[:, unit].norm().item(),
        }

        if show_gate and hasattr(ffn, "W_gate"):
            W_gate = ffn.W_gate.weight.detach().cpu()
            result["W_gate_row"] = W_gate[unit].tolist()
            result["W_gate_norm"] = W_gate[unit].norm().item()

        print(f"\nFFN weights for layer {layer}, unit {unit}:")
        print(f"  W_up norm: {result['W_up_norm']:.4f}")
        print(f"  W_down norm: {result['W_down_norm']:.4f}")
        if "W_gate_norm" in result:
            print(f"  W_gate norm: {result['W_gate_norm']:.4f}")

        # Show top input dimensions
        top_inputs = W_up[unit].abs().argsort(descending=True)[:5]
        print(f"  Top input dims: {top_inputs.tolist()}")

        # Show top output dimensions
        top_outputs = W_down[:, unit].abs().argsort(descending=True)[:5]
        print(f"  Top output dims: {top_outputs.tolist()}")

        return result

    def show_ffn_activations(
        self,
        layer: int,
        input_hidden: torch.Tensor,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """Show FFN unit activations for given input.

        Args:
            layer: Layer index
            input_hidden: Input hidden state [batch, seq, d_model]
            top_k: Number of top units to show

        Returns:
            Dict with activation info
        """
        ffn = self.model.blocks[layer].ffn

        with torch.no_grad():
            # Get activations before SwiGLU
            up = ffn.W_up(input_hidden)
            gate = ffn.W_gate(input_hidden) if hasattr(ffn, "W_gate") else up

            # SwiGLU activation
            activated = torch.nn.functional.silu(up) * gate

        # Analyze last position
        acts = activated[0, -1].detach().cpu()

        top_indices = acts.abs().argsort(descending=True)[:top_k]

        result = {
            "layer": layer,
            "max_activation": acts.max().item(),
            "min_activation": acts.min().item(),
            "mean_activation": acts.mean().item(),
            "active_units": (acts.abs() > 0.1).sum().item(),
            "top_units": [
                {"unit": idx.item(), "activation": acts[idx].item()}
                for idx in top_indices
            ],
        }

        print(f"\nFFN activations for layer {layer}:")
        print(f"  Range: [{result['min_activation']:.4f}, {result['max_activation']:.4f}]")
        print(f"  Mean: {result['mean_activation']:.4f}")
        print(f"  Active units (>0.1): {result['active_units']}")
        print(f"  Top units:")
        for u in result["top_units"][:5]:
            print(f"    unit {u['unit']:4d}: {u['activation']:+.4f}")

        return result

    def show_attention_weights(
        self,
        layer: int,
        head: int,
        query_dim: Optional[int] = None,
        key_dim: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Show attention projection weights.

        Args:
            layer: Layer index
            head: Head index
            query_dim: Specific query dimension to inspect
            key_dim: Specific key dimension to inspect

        Returns:
            Dict with attention weight info
        """
        attn = self.model.blocks[layer].attn
        head_dim = self.d_model // self.n_heads

        # Extract head-specific weights
        W_q = attn.W_q.weight.detach().cpu()
        W_k = attn.W_k.weight.detach().cpu()
        W_v = attn.W_v.weight.detach().cpu()

        start = head * head_dim
        end = start + head_dim

        W_q_head = W_q[start:end]
        W_k_head = W_k[start:end]
        W_v_head = W_v[start:end]

        result = {
            "layer": layer,
            "head": head,
            "head_dim": head_dim,
            "W_q_norm": W_q_head.norm().item(),
            "W_k_norm": W_k_head.norm().item(),
            "W_v_norm": W_v_head.norm().item(),
        }

        print(f"\nAttention weights for layer {layer}, head {head}:")
        print(f"  Head dim: {head_dim}")
        print(f"  W_q norm: {result['W_q_norm']:.4f}")
        print(f"  W_k norm: {result['W_k_norm']:.4f}")
        print(f"  W_v norm: {result['W_v_norm']:.4f}")

        return result

    def show_attention_scores(
        self,
        layer: int,
        head: int,
        hidden_states: torch.Tensor,
        query_pos: int = -1,
    ) -> Dict[str, Any]:
        """Compute and show attention scores for a specific head.

        Args:
            layer: Layer index
            head: Head index
            hidden_states: Input hidden states [batch, seq, d_model]
            query_pos: Query position to analyze (-1 for last)

        Returns:
            Dict with attention score info
        """
        attn = self.model.blocks[layer].attn
        head_dim = self.d_model // self.n_heads

        with torch.no_grad():
            # Project to Q, K
            Q = attn.W_q(hidden_states)
            K = attn.W_k(hidden_states)

            # Extract head
            start = head * head_dim
            end = start + head_dim

            q = Q[:, query_pos, start:end]  # [batch, head_dim]
            k = K[:, :, start:end]  # [batch, seq, head_dim]

            # Compute scores
            scores = torch.matmul(q.unsqueeze(1), k.transpose(-2, -1))
            scores = scores / (head_dim ** 0.5)
            scores = scores[0, 0].detach().cpu()  # [seq]

        top_indices = scores.argsort(descending=True)[:5]

        result = {
            "layer": layer,
            "head": head,
            "query_pos": query_pos,
            "max_score": scores.max().item(),
            "min_score": scores.min().item(),
            "top_positions": [
                {"pos": idx.item(), "score": scores[idx].item()}
                for idx in top_indices
            ],
        }

        print(f"\nAttention scores for layer {layer}, head {head}, query_pos {query_pos}:")
        print(f"  Score range: [{result['min_score']:.4f}, {result['max_score']:.4f}]")
        print(f"  Top attended positions:")
        for p in result["top_positions"]:
            print(f"    pos {p['pos']:3d}: {p['score']:+.4f}")

        return result

    def compare_embeddings(
        self,
        token_ids: List[int],
        dim: Optional[int] = None,
    ) -> None:
        """Compare embeddings across multiple tokens.

        Args:
            token_ids: List of token IDs to compare
            dim: Specific dimension to compare (None for all)
        """
        print(f"\nComparing embeddings for tokens: {token_ids}")

        embeds = [
            self.model.embed.embed.weight[tid].detach().cpu()
            for tid in token_ids
        ]

        if dim is not None:
            print(f"\nDimension {dim}:")
            for tid, emb in zip(token_ids, embeds):
                print(f"  Token {tid:3d}: {emb[dim]:+.4f}")
        else:
            # Compare norms and key dimensions
            print("\nNorms:")
            for tid, emb in zip(token_ids, embeds):
                print(f"  Token {tid:3d}: {emb.norm():.4f}")

            # Find dimensions that differ most
            if len(embeds) >= 2:
                diff = (embeds[0] - embeds[1]).abs()
                top_diff = diff.argsort(descending=True)[:5]
                print(f"\nTop differing dimensions (tokens {token_ids[0]} vs {token_ids[1]}):")
                for d in top_diff:
                    print(
                        f"  dim {d.item():3d}: "
                        f"{embeds[0][d]:+.4f} vs {embeds[1][d]:+.4f} "
                        f"(diff: {diff[d]:.4f})"
                    )
