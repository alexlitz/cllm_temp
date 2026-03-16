"""
C4 Transformer - Transformer for simulating C4 (C in 4 functions) execution.

Key innovations:
1. Top-K Sorted Attention with d_head=1 (scalar QK attention)
   - Sort K once: O(n log n)
   - Top-k lookup: O(1) per query
   - Total: O(n log n + nk) instead of O(n²)

2. Sparse representation
   - Only process non-zero memory blocks
   - C4 uses ~24KB of 1MB actively

3. SwiGLU FFN with RMSNorm

Architecture:
  State:      1MB (1,048,576 bytes)
  Tokens:     1 byte = 1 token
  dim:        128
  heads:      128 (d_head_qk=1, d_head_v=1)
  top_k:      32
  layers:     4
  FFN:        SwiGLU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math


class SwiGLU(nn.Module):
    """SwiGLU activation: (x @ W_up) * SiLU(x @ W_gate) @ W_down"""
    def __init__(self, dim, hidden_mult=8/3):
        super().__init__()
        hidden = int(dim * hidden_mult)
        self.w_gate = nn.Linear(dim, hidden, bias=False)
        self.w_up = nn.Linear(dim, hidden, bias=False)
        self.w_down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class TopKSortedAttention(nn.Module):
    """
    Top-K Sorted Attention with d_head_qk=1 (scalar attention).

    With d_head=1: score[i,j] = q[i] * k[j]

    Algorithm:
    1. Sort K once: O(n log n)
    2. For q > 0: top-k are largest K (end of sorted)
    3. For q < 0: top-k are smallest K (start of sorted)
    4. Softmax over k elements, weighted sum of V

    Complexity: O(n log n + n*k) instead of O(n²)
    """
    def __init__(self, dim, num_heads=128, top_k=32, chunk_size=16384):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.d_head_v = dim // num_heads

        # Q, K: d_head=1 (scalar per head)
        self.to_q = nn.Linear(dim, num_heads, bias=False)
        self.to_k = nn.Linear(dim, num_heads, bias=False)
        # V: full dimension
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        batch, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype
        k = self.top_k

        # Project Q, K, V
        q_all = self.to_q(x).transpose(1, 2)  # (batch, heads, seq)
        keys = self.to_k(x).transpose(1, 2)   # (batch, heads, seq)
        v = self.to_v(x).view(batch, seq_len, self.num_heads, self.d_head_v)
        v = v.permute(0, 2, 1, 3)  # (batch, heads, seq, d_head_v)

        # Sort keys once (reused for all queries)
        k_sorted, k_indices = keys.sort(dim=-1)

        # Reorder values by key sort order
        v_sorted = torch.gather(
            v, 2,
            k_indices.unsqueeze(-1).expand(-1, -1, -1, self.d_head_v)
        )

        # Extract top-k (largest) and bottom-k (smallest)
        top_k_keys = k_sorted[:, :, -k:]      # (batch, heads, k)
        bot_k_keys = k_sorted[:, :, :k]
        top_k_vals = v_sorted[:, :, -k:, :]   # (batch, heads, k, d_head_v)
        bot_k_vals = v_sorted[:, :, :k, :]

        # Process queries in chunks for memory efficiency
        output = torch.zeros(batch, self.num_heads, seq_len, self.d_head_v,
                           device=device, dtype=dtype)

        for i in range(0, seq_len, self.chunk_size):
            i_end = min(i + self.chunk_size, seq_len)
            q_chunk = q_all[:, :, i:i_end]  # (batch, heads, chunk)

            # Compute scores with top-k and bottom-k
            # (batch, heads, chunk, 1) * (batch, heads, 1, k) -> (batch, heads, chunk, k)
            scores_top = q_chunk.unsqueeze(-1) * top_k_keys.unsqueeze(2)
            scores_bot = q_chunk.unsqueeze(-1) * bot_k_keys.unsqueeze(2)

            # Softmax over k dimension
            attn_top = F.softmax(scores_top, dim=-1)
            attn_bot = F.softmax(scores_bot, dim=-1)

            # Weighted sum of values
            out_top = torch.einsum('bhck,bhkd->bhcd', attn_top, top_k_vals)
            out_bot = torch.einsum('bhck,bhkd->bhcd', attn_bot, bot_k_vals)

            # Select based on query sign
            q_positive = (q_chunk > 0).float().unsqueeze(-1)
            output[:, :, i:i_end, :] = q_positive * out_top + (1 - q_positive) * out_bot

        # Reshape and project output
        output = output.permute(0, 2, 1, 3).reshape(batch, seq_len, self.dim)
        return self.to_out(output)


class TransformerBlock(nn.Module):
    """Transformer block with Top-K Sorted Attention + SwiGLU."""
    def __init__(self, dim, num_heads=128, top_k=32, chunk_size=16384):
        super().__init__()
        self.attn = TopKSortedAttention(dim, num_heads, top_k, chunk_size)
        self.ffn = SwiGLU(dim)
        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class C4Transformer(nn.Module):
    """
    C4 State Transformer.

    Processes 1MB C4 state and predicts next state.
    Uses sparse representation to only process non-zero blocks.

    Args:
        state_size: Total state size in bytes (default 1MB)
        block_size: Size of each block for sparse processing (default 256)
        dim: Hidden dimension (default 128)
        depth: Number of transformer layers (default 4)
        num_heads: Number of attention heads (default 128, d_head=1)
        top_k: Number of keys to attend to (default 32)
    """
    def __init__(self,
                 state_size=1024*1024,
                 block_size=256,
                 dim=128,
                 depth=4,
                 num_heads=128,
                 top_k=32,
                 chunk_size=16384):
        super().__init__()
        self.state_size = state_size
        self.block_size = block_size
        self.num_blocks = state_size // block_size
        self.dim = dim
        self.num_heads = num_heads
        self.top_k = top_k

        # Embeddings
        self.byte_embed = nn.Embedding(256, dim)
        self.block_pos_embed = nn.Embedding(self.num_blocks, dim)
        self.local_pos_embed = nn.Embedding(block_size, dim)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads, top_k, chunk_size)
            for _ in range(depth)
        ])

        self.final_norm = nn.RMSNorm(dim)
        self.out = nn.Linear(dim, 256, bias=False)

    def forward(self, state_bytes, sparse=True):
        """
        Forward pass.

        Args:
            state_bytes: (batch, state_size) byte values 0-255
            sparse: If True, only process non-zero blocks

        Returns:
            logits: (batch, state_size, 256) predictions for each byte
        """
        batch = state_bytes.shape[0]
        device = state_bytes.device
        dtype = next(self.parameters()).dtype

        if sparse:
            return self._forward_sparse(state_bytes)
        else:
            return self._forward_dense(state_bytes)

    def _forward_dense(self, state_bytes):
        """Process all positions (no sparsity)."""
        batch, seq_len = state_bytes.shape
        device = state_bytes.device

        # Embed bytes
        x = self.byte_embed(state_bytes)  # (batch, seq, dim)

        # Add position embeddings
        # Block positions
        block_idx = torch.arange(self.num_blocks, device=device)
        block_pos = self.block_pos_embed(block_idx)  # (num_blocks, dim)
        block_pos = block_pos.unsqueeze(1).expand(-1, self.block_size, -1)
        block_pos = block_pos.reshape(1, seq_len, self.dim)

        # Local positions within block
        local_idx = torch.arange(self.block_size, device=device)
        local_pos = self.local_pos_embed(local_idx)  # (block_size, dim)
        local_pos = local_pos.unsqueeze(0).expand(self.num_blocks, -1, -1)
        local_pos = local_pos.reshape(1, seq_len, self.dim)

        x = x + block_pos + local_pos

        # Transformer layers
        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)
        return self.out(x)

    def _forward_sparse(self, state_bytes):
        """Process only non-zero blocks."""
        batch = state_bytes.shape[0]
        device = state_bytes.device
        dtype = next(self.parameters()).dtype

        # Find non-zero blocks
        blocks = state_bytes.view(batch, self.num_blocks, self.block_size)
        active_mask = (blocks != 0).any(dim=-1)  # (batch, num_blocks)

        # Get active block indices (assume batch=1 for simplicity)
        active_indices = active_mask[0].nonzero().squeeze(-1)
        num_active = len(active_indices)

        if num_active == 0:
            # All zeros - return uniform predictions
            return torch.zeros(batch, self.state_size, 256, device=device, dtype=dtype)

        # Extract active blocks
        active_blocks = blocks[0, active_indices]  # (num_active, block_size)

        # Embed
        x = self.byte_embed(active_blocks)  # (num_active, block_size, dim)

        # Add position embeddings
        block_pos = self.block_pos_embed(active_indices)  # (num_active, dim)
        x = x + block_pos.unsqueeze(1)

        local_pos = self.local_pos_embed(torch.arange(self.block_size, device=device))
        x = x + local_pos.unsqueeze(0)

        # Flatten to single sequence
        seq_len = num_active * self.block_size
        x = x.view(1, seq_len, self.dim)

        # Transformer layers
        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)
        active_logits = self.out(x)  # (1, seq_len, 256)

        # Scatter back to full state
        full_logits = torch.zeros(batch, self.state_size, 256, device=device, dtype=active_logits.dtype)
        for i, block_idx in enumerate(active_indices):
            start = block_idx * self.block_size
            end = start + self.block_size
            full_logits[0, start:end] = active_logits[0, i*self.block_size:(i+1)*self.block_size]

        return full_logits

    def count_active_blocks(self, state_bytes):
        """Count non-zero blocks in state."""
        blocks = state_bytes.view(-1, self.num_blocks, self.block_size)
        active = (blocks != 0).any(dim=-1).sum().item()
        return active, self.num_blocks


def create_c4_state(device, code_kb=16, data_kb=4, stack_kb=4):
    """Create a realistic C4 state with active regions."""
    state = torch.zeros(1, 1024*1024, dtype=torch.long, device=device)

    # Code segment: 0 to code_kb
    code_size = code_kb * 1024
    state[0, :code_size] = torch.randint(1, 256, (code_size,), device=device)

    # Data segment: 256KB to 256KB + data_kb
    data_start = 256 * 1024
    data_size = data_kb * 1024
    state[0, data_start:data_start+data_size] = torch.randint(1, 256, (data_size,), device=device)

    # Stack: 768KB to 768KB + stack_kb
    stack_start = 768 * 1024
    stack_size = stack_kb * 1024
    state[0, stack_start:stack_start+stack_size] = torch.randint(1, 256, (stack_size,), device=device)

    return state


def test_model():
    """Test the C4 Transformer."""
    import gc

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    print()

    # Create model
    model = C4Transformer(
        state_size=1024*1024,
        block_size=256,
        dim=128,
        depth=4,
        num_heads=128,
        top_k=32,
        chunk_size=16384
    )
    model = model.to(device).half()

    params = sum(p.numel() for p in model.parameters())

    print("=" * 60)
    print("C4 TRANSFORMER")
    print("=" * 60)
    print(f"""
Architecture:
  State size:     1MB ({model.state_size:,} bytes)
  Block size:     {model.block_size} bytes
  Num blocks:     {model.num_blocks}

  dim:            {model.dim}
  num_heads:      {model.num_heads}
  d_head_qk:      1 (scalar attention)
  d_head_v:       {model.dim // model.num_heads}
  top_k:          {model.top_k}
  layers:         {len(model.layers)}
  FFN:            SwiGLU

  Parameters:     {params:,} ({params*2/1e6:.1f} MB in float16)
""")

    # Test with realistic C4 state (sparse)
    print("Creating realistic C4 state...")
    state = create_c4_state(device, code_kb=16, data_kb=4, stack_kb=4)

    active, total = model.count_active_blocks(state)
    print(f"Active blocks: {active}/{total} ({100*active/total:.1f}%)")
    print(f"Active bytes: {active * model.block_size:,}")
    print()

    # Sparse forward pass
    print("Testing SPARSE forward pass...")
    gc.collect()
    if device.type == 'mps':
        torch.mps.empty_cache()

    start = time.time()
    with torch.no_grad():
        logits = model(state, sparse=True)
    if device.type == 'mps':
        torch.mps.synchronize()
    sparse_time = time.time() - start

    print(f"  Time: {sparse_time:.2f}s")
    print(f"  Output shape: {logits.shape}")
    print()

    # Dense forward pass (full 1MB)
    print("Testing DENSE forward pass (full 1MB)...")
    gc.collect()
    if device.type == 'mps':
        torch.mps.empty_cache()

    start = time.time()
    with torch.no_grad():
        logits_dense = model(state, sparse=False)
    if device.type == 'mps':
        torch.mps.synchronize()
    dense_time = time.time() - start

    print(f"  Time: {dense_time:.2f}s")
    print(f"  Throughput: {model.state_size/dense_time:,.0f} tokens/sec")
    print()

    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Sparse (24KB active): {sparse_time:.2f}s")
    print(f"  Dense (1MB):          {dense_time:.2f}s")
    print(f"  Sparse speedup:       {dense_time/sparse_time:.1f}x")


if __name__ == "__main__":
    test_model()
