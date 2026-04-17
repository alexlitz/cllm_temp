"""
Attention Visualizer for Neural VM.

Provides text-based attention pattern visualization suitable for
terminal output and agent consumption.
"""

import torch
from typing import Optional, List, Dict, Any, Tuple


class AttentionVisualizer:
    """Visualize attention patterns in text format.

    Designed for terminal output and consumption by AI agents.
    All output is text-based, no graphics libraries required.

    Usage:
        viz = AttentionVisualizer(model)

        # Show attention pattern for a specific layer/head
        viz.show_attention_pattern(
            hidden_states=hidden,
            layer=5,
            head=0,
            query_pos=-1,  # Last position
        )

        # Show which positions attend to markers
        viz.show_marker_attention(hidden_states=hidden, layer=5)

        # Summarize attention across all heads
        viz.summarize_layer_attention(hidden_states=hidden, layer=5)
    """

    # Token type markers for visualization
    MARKERS = {
        256: "PC",   # REG_PC
        257: "AX",   # REG_AX
        258: "SP",   # REG_SP
        259: "BP",   # REG_BP
        260: "MEM",  # MEM marker
        261: "CS",   # CODE_START
    }

    def __init__(self, model):
        """Initialize visualizer with model.

        Args:
            model: AutoregressiveVM model instance
        """
        self.model = model
        self.d_model = model.d_model
        self.n_layers = len(model.blocks)
        self.n_heads = model.blocks[0].attn.num_heads
        self.head_dim = self.d_model // self.n_heads

    def _get_attention_scores(
        self,
        hidden_states: torch.Tensor,
        layer: int,
        head: int,
    ) -> torch.Tensor:
        """Compute raw attention scores for a specific head.

        Args:
            hidden_states: [batch, seq, d_model]
            layer: Layer index
            head: Head index

        Returns:
            Attention scores [seq, seq] (before softmax)
        """
        attn = self.model.blocks[layer].attn

        with torch.no_grad():
            Q = attn.W_q(hidden_states)
            K = attn.W_k(hidden_states)

            start = head * self.head_dim
            end = start + self.head_dim

            q = Q[0, :, start:end]  # [seq, head_dim]
            k = K[0, :, start:end]  # [seq, head_dim]

            scores = torch.matmul(q, k.transpose(-2, -1))
            scores = scores / (self.head_dim ** 0.5)

        return scores.detach().cpu()

    def _get_attention_weights(
        self,
        hidden_states: torch.Tensor,
        layer: int,
        head: int,
    ) -> torch.Tensor:
        """Compute attention weights (after softmax) for a specific head.

        Args:
            hidden_states: [batch, seq, d_model]
            layer: Layer index
            head: Head index

        Returns:
            Attention weights [seq, seq] (after softmax, with causal mask)
        """
        scores = self._get_attention_scores(hidden_states, layer, head)
        seq_len = scores.shape[0]

        # Apply causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        scores.masked_fill_(mask, float("-inf"))

        # Softmax
        weights = torch.softmax(scores, dim=-1)
        return weights

    def show_attention_pattern(
        self,
        hidden_states: torch.Tensor,
        layer: int,
        head: int,
        query_pos: int = -1,
        top_k: int = 10,
        token_ids: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Show attention pattern for a query position.

        Args:
            hidden_states: Input hidden states [batch, seq, d_model]
            layer: Layer index
            head: Head index
            query_pos: Query position (-1 for last)
            top_k: Number of top attended positions to show
            token_ids: Optional token IDs for labeling

        Returns:
            Dict with attention info
        """
        weights = self._get_attention_weights(hidden_states, layer, head)
        seq_len = weights.shape[0]

        if query_pos < 0:
            query_pos = seq_len + query_pos

        attn = weights[query_pos]  # [seq]

        # Get top attended positions
        top_indices = attn.argsort(descending=True)[:top_k]

        result = {
            "layer": layer,
            "head": head,
            "query_pos": query_pos,
            "seq_len": seq_len,
            "top_attended": [],
        }

        print(f"\n{'='*60}")
        print(f"ATTENTION PATTERN: Layer {layer}, Head {head}, Query {query_pos}")
        print(f"{'='*60}")
        print(f"Sequence length: {seq_len}")
        print(f"\nTop {top_k} attended positions:")
        print(f"{'Pos':>5} {'Weight':>8} {'Token':>8} {'Label':>10}")
        print("-" * 35)

        for idx in top_indices:
            pos = idx.item()
            weight = attn[idx].item()

            # Get token label
            if token_ids is not None and pos < len(token_ids):
                tok = token_ids[pos]
                label = self.MARKERS.get(tok, str(tok))
            else:
                label = "-"

            result["top_attended"].append({
                "position": pos,
                "weight": weight,
                "token_id": token_ids[pos] if token_ids and pos < len(token_ids) else None,
                "label": label,
            })

            print(f"{pos:5d} {weight:8.4f} {token_ids[pos] if token_ids and pos < len(token_ids) else '-':>8} {label:>10}")

        print(f"{'='*60}\n")

        return result

    def show_marker_attention(
        self,
        hidden_states: torch.Tensor,
        layer: int,
        token_ids: List[int],
        query_pos: int = -1,
    ) -> Dict[str, Any]:
        """Show how much attention each head pays to marker tokens.

        Args:
            hidden_states: Input hidden states
            layer: Layer index
            token_ids: Token IDs in the sequence
            query_pos: Query position to analyze

        Returns:
            Dict mapping marker -> head -> attention weight
        """
        seq_len = hidden_states.shape[1]
        if query_pos < 0:
            query_pos = seq_len + query_pos

        # Find marker positions
        marker_positions = {}
        for pos, tok in enumerate(token_ids):
            if tok in self.MARKERS:
                marker_name = self.MARKERS[tok]
                marker_positions[marker_name] = pos

        result = {
            "layer": layer,
            "query_pos": query_pos,
            "markers": {},
        }

        print(f"\n{'='*60}")
        print(f"MARKER ATTENTION: Layer {layer}, Query {query_pos}")
        print(f"{'='*60}")

        # Header
        header = f"{'Marker':>6} {'Pos':>4}"
        for h in range(self.n_heads):
            header += f" {'H'+str(h):>6}"
        print(header)
        print("-" * (12 + 7 * self.n_heads))

        for marker_name, marker_pos in marker_positions.items():
            row = f"{marker_name:>6} {marker_pos:4d}"
            result["markers"][marker_name] = {"pos": marker_pos, "heads": {}}

            for head in range(self.n_heads):
                weights = self._get_attention_weights(hidden_states, layer, head)
                w = weights[query_pos, marker_pos].item()
                result["markers"][marker_name]["heads"][head] = w
                row += f" {w:6.3f}"

            print(row)

        print(f"{'='*60}\n")

        return result

    def summarize_layer_attention(
        self,
        hidden_states: torch.Tensor,
        layer: int,
        query_pos: int = -1,
    ) -> Dict[str, Any]:
        """Summarize attention pattern across all heads in a layer.

        Args:
            hidden_states: Input hidden states
            layer: Layer index
            query_pos: Query position

        Returns:
            Summary dict with per-head statistics
        """
        seq_len = hidden_states.shape[1]
        if query_pos < 0:
            query_pos = seq_len + query_pos

        result = {
            "layer": layer,
            "query_pos": query_pos,
            "heads": [],
        }

        print(f"\n{'='*60}")
        print(f"ATTENTION SUMMARY: Layer {layer}, Query {query_pos}")
        print(f"{'='*60}")
        print(f"{'Head':>4} {'MaxPos':>6} {'MaxWt':>7} {'Entropy':>8} {'Spread':>7}")
        print("-" * 36)

        for head in range(self.n_heads):
            weights = self._get_attention_weights(hidden_states, layer, head)
            attn = weights[query_pos]

            max_pos = attn.argmax().item()
            max_wt = attn.max().item()

            # Entropy (lower = more focused)
            entropy = -(attn * (attn + 1e-10).log()).sum().item()

            # Spread (how many positions have >1% attention)
            spread = (attn > 0.01).sum().item()

            result["heads"].append({
                "head": head,
                "max_position": max_pos,
                "max_weight": max_wt,
                "entropy": entropy,
                "spread": spread,
            })

            print(f"{head:4d} {max_pos:6d} {max_wt:7.4f} {entropy:8.3f} {spread:7d}")

        print(f"{'='*60}\n")

        return result

    def show_attention_heatmap(
        self,
        hidden_states: torch.Tensor,
        layer: int,
        head: int,
        max_seq: int = 30,
        token_ids: Optional[List[int]] = None,
    ) -> str:
        """Generate text-based attention heatmap.

        Args:
            hidden_states: Input hidden states
            layer: Layer index
            head: Head index
            max_seq: Maximum sequence length to display
            token_ids: Optional token IDs for labels

        Returns:
            Text heatmap string
        """
        weights = self._get_attention_weights(hidden_states, layer, head)
        seq_len = min(weights.shape[0], max_seq)

        # Characters for different attention levels
        chars = " .:-=+*#@"

        output = []
        output.append(f"\nATTENTION HEATMAP: Layer {layer}, Head {head}")
        output.append("Q\\K " + "".join(f"{i%10}" for i in range(seq_len)))
        output.append("-" * (seq_len + 4))

        for q in range(seq_len):
            row = f"{q:3d} "
            for k in range(seq_len):
                if k > q:  # Causal mask
                    row += " "
                else:
                    w = weights[q, k].item()
                    # Map weight to character
                    idx = min(int(w * len(chars)), len(chars) - 1)
                    row += chars[idx]
            output.append(row)

        output.append("\nLegend: ' '=0  '.'<.1  '-'<.2  ':'<.3  '='<.4  '+'<.5  '*'<.7  '#'<.9  '@'>=.9")

        result = "\n".join(output)
        print(result)
        return result

    def compare_heads(
        self,
        hidden_states: torch.Tensor,
        layer: int,
        query_pos: int = -1,
        key_pos: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Compare attention across all heads for specific positions.

        Args:
            hidden_states: Input hidden states
            layer: Layer index
            query_pos: Query position
            key_pos: Specific key position to compare (None for all)

        Returns:
            Comparison dict
        """
        seq_len = hidden_states.shape[1]
        if query_pos < 0:
            query_pos = seq_len + query_pos

        result = {"layer": layer, "query_pos": query_pos, "heads": {}}

        print(f"\n{'='*60}")
        print(f"HEAD COMPARISON: Layer {layer}, Query {query_pos}")
        print(f"{'='*60}")

        if key_pos is not None:
            # Compare specific key position across heads
            print(f"Attention to position {key_pos}:")
            for head in range(self.n_heads):
                weights = self._get_attention_weights(hidden_states, layer, head)
                w = weights[query_pos, key_pos].item()
                result["heads"][head] = w
                bar = "#" * int(w * 40)
                print(f"  Head {head}: {w:.4f} |{bar}")
        else:
            # Show top position for each head
            print(f"{'Head':>4} {'TopPos':>6} {'Weight':>7}")
            print("-" * 20)
            for head in range(self.n_heads):
                weights = self._get_attention_weights(hidden_states, layer, head)
                attn = weights[query_pos]
                top_pos = attn.argmax().item()
                top_wt = attn.max().item()
                result["heads"][head] = {"top_pos": top_pos, "weight": top_wt}
                print(f"{head:4d} {top_pos:6d} {top_wt:7.4f}")

        print(f"{'='*60}\n")

        return result
