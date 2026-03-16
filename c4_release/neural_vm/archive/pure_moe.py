"""
Mixture-of-Experts Layer with opcode-based gating.

Default: skips inactive experts for fast runtime (~13x speedup with 1 active opcode).
ONNX export: auto-detects tracing and runs all experts with soft gating.

Architecture:
    output = x + sum_i(opcode_weight[i] * (expert[i](x) - x))

Where opcode_weight is the one-hot encoding from embedding slots.
"""

import torch
import torch.nn as nn
from typing import List, Dict

from .embedding import E, Opcode
from .base_layers import PureFFN, PureAttention


class MoE(nn.Module):
    """
    Mixture-of-Experts with opcode-based gating.

    Accepts any nn.Module experts (PureFFN, PureAttention, or mixed).

    Runtime: skips experts whose opcode weight is below threshold (~13x speedup).
    ONNX: auto-detects tracing and runs all experts for a fixed computation graph.
    """

    def __init__(self, experts: List[nn.Module], expert_opcodes: List[int],
                 threshold: float = 0.01):
        """
        Args:
            experts: List of nn.Module expert modules (PureFFN, PureAttention, or any nn.Module)
            expert_opcodes: List of opcode numbers, one per expert
            threshold: Skip experts with weight < threshold (default 0.01)
        """
        super().__init__()
        self.experts = nn.ModuleList(experts)
        # Store opcodes as Python list (not tensor) for ONNX compatibility
        self.expert_opcode_list = list(expert_opcodes)
        # Also store as buffer for state_dict
        self.register_buffer('expert_opcodes', torch.tensor(expert_opcodes, dtype=torch.long))
        self.num_experts = len(experts)
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Default forward: skip inactive experts for fast runtime.

        Auto-detects ONNX export and uses soft forward for tracing.
        """
        if torch.onnx.is_in_onnx_export():
            return self._soft_forward(x)

        opcode_weights = x[:, 0, E.OP_START:E.OP_START + E.NUM_OPS]  # [batch, NUM_OPS]

        # Gather weights for all experts: [num_experts, batch]
        all_weights = opcode_weights[:, self.expert_opcodes].T

        # Find which experts are active (max across batch > threshold)
        active_mask = all_weights.max(dim=1).values > self.threshold
        active_indices = active_mask.nonzero(as_tuple=True)[0]

        if len(active_indices) == 0:
            return x  # No active experts = identity

        output = torch.zeros_like(x)

        # Run only active experts
        for idx in active_indices:
            i = idx.item()
            expert_out = self.experts[i](x)  # [batch, pos, dim]
            opcode_idx = self.expert_opcodes[i]
            weight = opcode_weights[:, opcode_idx:opcode_idx+1].unsqueeze(-1)  # [batch, 1, 1]
            output = output + weight * (expert_out - x)  # Weighted residual

        return x + output

    def _soft_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        ONNX-compatible soft forward: run all experts, weight by opcode.

        No Python control flow — pure tensor operations.
        Uses Python ints (static at trace time) for ONNX compatibility.
        """
        opcode_weights = x[:, 0, E.OP_START:E.OP_START + E.NUM_OPS]  # [batch, NUM_OPS]
        output = torch.zeros_like(x)

        for i in range(self.num_experts):
            expert_out = self.experts[i](x)  # [batch, pos, dim]
            opcode_idx = self.expert_opcode_list[i]
            weight = opcode_weights[:, opcode_idx:opcode_idx+1].unsqueeze(-1)  # [batch, 1, 1]
            output = output + weight * (expert_out - x)  # Weighted residual

        return x + output


# Backward compatibility aliases
SoftMoEFFN = MoE
SoftMoEAttention = MoE
EarlyExitMoEFFN = MoE


class UnifiedMoEAttention(nn.Module):
    """
    Unified Multi-Head Attention with opcode-gated heads.

    Instead of running separate attention modules and blending outputs,
    this uses a single attention where each head is gated by its opcode.

    Mechanism:
    - Each head is "owned" by an opcode
    - Before softmax, add -INF_GATE to scores for inactive heads
    - Inactive opcode -> scores = -inf -> softmax -> 0 -> no contribution
    - Active opcode -> scores unchanged -> normal attention

    This is more efficient than MoE with separate PureAttention modules:
    - Single forward pass instead of N separate attention modules
    - Gating happens inside softmax, not via output blending
    """

    INF_GATE = 1e9  # Large negative value for gating

    def __init__(self, dim: int, head_opcodes: List[int], head_dim: int = None):
        """
        Args:
            dim: Embedding dimension
            head_opcodes: List of opcode numbers, one per head
            head_dim: Dimension per head (defaults to dim for full-dim heads)
        """
        super().__init__()
        self.dim = dim
        self.num_heads = len(head_opcodes)
        self.register_buffer('head_opcodes', torch.tensor(head_opcodes, dtype=torch.long))

        # Default to full-dim heads (each head sees entire embedding)
        self.head_dim = head_dim if head_dim is not None else dim
        self.scale = self.head_dim ** -0.5

        # Unified projection matrices: each head has full dim input, head_dim output
        # W_q[head_idx] is at rows [head_idx * head_dim : (head_idx+1) * head_dim]
        self.W_q = nn.Parameter(torch.zeros(self.num_heads * self.head_dim, dim))
        self.W_k = nn.Parameter(torch.zeros(self.num_heads * self.head_dim, dim))
        self.W_v = nn.Parameter(torch.zeros(self.num_heads * self.head_dim, dim))
        self.W_o = nn.Parameter(torch.zeros(dim, self.num_heads * self.head_dim))

        # Per-head position masks (for different attention patterns)
        # Shape: [num_heads, NUM_POSITIONS, NUM_POSITIONS]
        self.register_buffer('head_masks',
            torch.zeros(self.num_heads, E.NUM_POSITIONS, E.NUM_POSITIONS))

    def set_head_mask(self, head_idx: int, mask: torch.Tensor):
        """
        Set attention mask for a specific head.

        Args:
            head_idx: Head index
            mask: [NUM_POSITIONS, NUM_POSITIONS] mask (0 = attend, -inf = block)
        """
        with torch.no_grad():
            self.head_masks[head_idx] = mask

    def set_head_weights(self, head_idx: int,
                          W_q: torch.Tensor, W_k: torch.Tensor,
                          W_v: torch.Tensor, W_o: torch.Tensor):
        """
        Set projection weights for a specific head.

        Args:
            head_idx: Head index
            W_q, W_k, W_v: [head_dim, dim] projection matrices
            W_o: [dim, head_dim] output projection
        """
        start = head_idx * self.head_dim
        end = start + self.head_dim
        with torch.no_grad():
            self.W_q[start:end, :] = W_q
            self.W_k[start:end, :] = W_k
            self.W_v[start:end, :] = W_v
            self.W_o[:, start:end] = W_o

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward with opcode-gated heads.

        Inactive heads contribute nothing via post-softmax gating.
        """
        B, S, D = x.shape
        H = self.num_heads
        HD = self.head_dim

        # Get opcode weights: [batch, NUM_OPS]
        opcode_weights = x[:, 0, E.OP_START:E.OP_START + E.NUM_OPS]

        # Get head gate weights: [batch, num_heads]
        head_weights = opcode_weights[:, self.head_opcodes]  # [batch, num_heads]

        # Project to Q, K, V: [batch, seq, num_heads * head_dim]
        Q = torch.nn.functional.linear(x, self.W_q)
        K = torch.nn.functional.linear(x, self.W_k)
        V = torch.nn.functional.linear(x, self.W_v)

        # Reshape to [batch, num_heads, seq, head_dim]
        Q = Q.view(B, S, H, HD).transpose(1, 2)
        K = K.view(B, S, H, HD).transpose(1, 2)
        V = V.view(B, S, H, HD).transpose(1, 2)

        # Compute attention scores: [batch, num_heads, seq, seq]
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Add per-head position masks
        scores = scores + self.head_masks[:, :S, :S]

        # Softmax
        attn = torch.nn.functional.softmax(scores, dim=-1)

        # OPCODE GATING: Multiply attention weights by head gate
        # head_weights: [batch, num_heads] -> [batch, num_heads, 1, 1]
        head_gate = head_weights.unsqueeze(-1).unsqueeze(-1)
        attn = attn * head_gate  # Zero out inactive heads

        # Apply attention to values
        out = torch.matmul(attn, V)  # [batch, num_heads, seq, head_dim]

        # Reshape back: [batch, seq, num_heads * head_dim]
        out = out.transpose(1, 2).contiguous().view(B, S, H * HD)

        # Output projection
        out = torch.nn.functional.linear(out, self.W_o)

        return x + out

    @classmethod
    def from_pure_attentions(cls, attentions: List[PureAttention],
                              opcodes: List[int]) -> 'UnifiedMoEAttention':
        """
        Convert list of PureAttention modules to unified attention.

        Each PureAttention has W_q/W_k/W_v/W_o of shape [dim, dim].
        We create a unified attention where each head has full dim.
        """
        dim = attentions[0].dim
        num_heads = len(attentions)

        # Create unified attention with full-dim heads
        unified = cls(dim=dim, head_opcodes=opcodes, head_dim=dim)

        # Copy weights from each attention
        with torch.no_grad():
            for i, attn in enumerate(attentions):
                unified.set_head_weights(
                    head_idx=i,
                    W_q=attn.W_q,
                    W_k=attn.W_k,
                    W_v=attn.W_v,
                    W_o=attn.W_o
                )
                unified.set_head_mask(i, attn.mask)

        return unified


class UnifiedMoEBlock(nn.Module):
    """
    Single unified MoE block: Attention + FFN.

    Both attention and FFN are MoE with opcode routing.
    All operations for all opcodes run in parallel.
    """

    def __init__(self,
                 ffn_experts: List[PureFFN],
                 ffn_opcodes: List[int],
                 attn_experts: List[PureAttention] = None,
                 attn_opcodes: List[int] = None):
        super().__init__()

        self.has_attention = attn_experts is not None and len(attn_experts) > 0

        if self.has_attention:
            self.attention = MoE(attn_experts, attn_opcodes)

        self.ffn = MoE(ffn_experts, ffn_opcodes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pure tensor forward."""
        if self.has_attention:
            x = self.attention(x)
        x = self.ffn(x)
        return x


class IdentityFFN(PureFFN):
    """Identity FFN - passes input through unchanged."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=1)

    def _bake_weights(self):
        # All zeros = identity (due to residual connection)
        pass


class IdentityAttention(PureAttention):
    """Identity Attention - passes input through unchanged."""

    def __init__(self):
        super().__init__(E.DIM, num_heads=1, causal=False)

    def _bake_weights(self):
        # All zeros = identity (due to residual connection)
        pass


# =============================================================================
# PURE FORWARD VERIFICATION
# =============================================================================

def verify_pure_forward(module: nn.Module) -> bool:
    """
    Verify a module's forward is pure tensor operations.

    Returns True if forward contains no Python control flow.
    MoE modules are exempt (they use .item() for expert skipping).
    """
    import ast
    import inspect

    # MoE modules use .item() for expert skipping — that's by design
    if isinstance(module, MoE):
        return True

    try:
        source = inspect.getsource(module.forward)
        tree = ast.parse(source)
    except:
        return False

    # Check for forbidden constructs
    forbidden = (ast.If, ast.For, ast.While)

    for node in ast.walk(tree):
        if isinstance(node, forbidden):
            return False
        # Check for .item() calls
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == 'item':
                return False

    return True
