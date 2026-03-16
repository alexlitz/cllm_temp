"""
Test: Can we run a transformer with 1M context for C4 simulation?

C4 state: 1MB = 1,048,576 bytes
Goal: Process this state with a small transformer

RESULTS (16GB Mac):
- Linear attention: 1M tokens in 0.5s ✓
- Full C4 transformer (1MB state, 262K tokens): 0.69s ✓
- Flash attention (PyTorch SDPA): works to ~100K tokens
- Standard attention: OOM at ~50K tokens

Key insight: We don't need full O(n²) attention for C4 simulation.
Each instruction only accesses a small portion of memory, so we can use:
1. Linear attention (O(n) memory) - RECOMMENDED
2. Sliding window attention
3. Flash attention (chunked for very long)

Architecture for 1MB C4 state:
- 4 bytes per token → 262,144 sequence length
- dim=64, depth=4, heads=4
- ~17M parameters (68MB)
- Processes full state in <1 second
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import gc
import math

# =============================================================================
# Memory utilities
# =============================================================================

def get_memory_usage():
    """Get current memory usage in GB."""
    if torch.backends.mps.is_available():
        # MPS doesn't have direct memory query, estimate from tensors
        return 0
    elif torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9
    return 0

def estimate_tensor_memory(shape, dtype=torch.float32):
    """Estimate memory for a tensor in GB."""
    numel = 1
    for s in shape:
        numel *= s
    bytes_per_elem = torch.tensor([], dtype=dtype).element_size()
    return numel * bytes_per_elem / 1e9

# =============================================================================
# Linear Attention (O(n) memory)
# =============================================================================

class LinearAttention(nn.Module):
    """
    Linear attention using kernel feature maps.

    Instead of softmax(QK^T)V which is O(n²),
    we compute: (φ(Q) @ (φ(K)^T @ V)) which is O(n).

    Using ELU+1 as feature map (simple and effective).
    """
    def __init__(self, dim, heads=4):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

    def feature_map(self, x):
        """ELU + 1 feature map (always positive)."""
        return F.elu(x) + 1

    def forward(self, x):
        batch, seq_len, _ = x.shape

        # Get Q, K, V
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(batch, seq_len, self.heads, self.head_dim).transpose(1, 2)
                   for t in qkv]

        # Apply feature map
        q = self.feature_map(q)
        k = self.feature_map(k)

        # Linear attention: O(n) instead of O(n²)
        # Compute K^T @ V first: (heads, head_dim, head_dim)
        kv = torch.einsum('bhnd,bhne->bhde', k, v)

        # Then Q @ (K^T @ V): (batch, heads, seq, head_dim)
        out = torch.einsum('bhnd,bhde->bhne', q, kv)

        # Normalize by sum of attention weights
        k_sum = k.sum(dim=2, keepdim=True)  # (batch, heads, 1, head_dim)
        normalizer = torch.einsum('bhnd,bhkd->bhnk', q, k_sum).squeeze(-1)
        out = out / (normalizer.unsqueeze(-1) + 1e-6)

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.dim)
        return self.to_out(out)


class CausalLinearAttention(nn.Module):
    """
    Causal linear attention using cumulative sums.

    For causal masking, we compute running sums:
    - S_i = Σ_{j≤i} φ(k_j) ⊗ v_j  (cumulative KV)
    - z_i = Σ_{j≤i} φ(k_j)         (cumulative normalizer)
    - output_i = (φ(q_i) @ S_i) / (φ(q_i) @ z_i)
    """
    def __init__(self, dim, heads=4):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

    def feature_map(self, x):
        return F.elu(x) + 1

    def forward(self, x):
        batch, seq_len, _ = x.shape

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(batch, seq_len, self.heads, self.head_dim).transpose(1, 2)
                   for t in qkv]

        q = self.feature_map(q)
        k = self.feature_map(k)

        # Causal linear attention via cumsum
        # kv[i] = k[i] ⊗ v[i], then cumsum
        kv = torch.einsum('bhnd,bhne->bhnde', k, v)  # outer product
        kv_cumsum = kv.cumsum(dim=2)  # cumulative KV

        # Normalizer
        k_cumsum = k.cumsum(dim=2)

        # Output
        out = torch.einsum('bhnd,bhnde->bhne', q, kv_cumsum)
        normalizer = torch.einsum('bhnd,bhnd->bhn', q, k_cumsum)
        out = out / (normalizer.unsqueeze(-1) + 1e-6)

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.dim)
        return self.to_out(out)


# =============================================================================
# Sliding Window Attention (for local patterns)
# =============================================================================

class SlidingWindowAttention(nn.Module):
    """
    Attention with a fixed window size.
    Memory: O(n * window_size) instead of O(n²)
    """
    def __init__(self, dim, heads=4, window_size=512):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.window_size = window_size

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        batch, seq_len, _ = x.shape

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(batch, seq_len, self.heads, self.head_dim).transpose(1, 2)
                   for t in qkv]

        # Process in windows
        outputs = []
        for i in range(0, seq_len, self.window_size):
            end = min(i + self.window_size, seq_len)
            start_k = max(0, i - self.window_size)  # look back one window

            q_chunk = q[:, :, i:end, :]
            k_chunk = k[:, :, start_k:end, :]
            v_chunk = v[:, :, start_k:end, :]

            # Standard attention within window
            scores = torch.einsum('bhqd,bhkd->bhqk', q_chunk, k_chunk) / (self.head_dim ** 0.5)

            # Causal mask within window
            q_pos = torch.arange(i, end, device=x.device)
            k_pos = torch.arange(start_k, end, device=x.device)
            mask = q_pos.unsqueeze(1) >= k_pos.unsqueeze(0)
            scores = scores.masked_fill(~mask, float('-inf'))

            attn = F.softmax(scores, dim=-1)
            out_chunk = torch.einsum('bhqk,bhkd->bhqd', attn, v_chunk)
            outputs.append(out_chunk)

        out = torch.cat(outputs, dim=2)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.dim)
        return self.to_out(out)


# =============================================================================
# Flash Attention (via PyTorch 2.0+ scaled_dot_product_attention)
# =============================================================================

class FlashAttention(nn.Module):
    """
    Flash Attention using PyTorch 2.0+ scaled_dot_product_attention.

    This automatically uses:
    - Flash Attention (CUDA)
    - Memory-efficient attention (all backends)
    - Math fallback when needed

    Key benefit: O(n) memory instead of O(n²) by not materializing attention matrix.
    """
    def __init__(self, dim, heads=4, causal=True):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.causal = causal

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        batch, seq_len, _ = x.shape

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(batch, seq_len, self.heads, self.head_dim).transpose(1, 2)
                   for t in qkv]

        # Use PyTorch's optimized attention
        # This automatically selects best implementation (flash, mem-efficient, or math)
        # On CPU/MPS, it uses memory-efficient or math backend
        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=self.causal,
            dropout_p=0.0
        )

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.dim)
        return self.to_out(out)


class TiledFlashAttention(nn.Module):
    """
    Doubly-Tiled Flash Attention - tiles both Q and KV dimensions.

    This is the actual flash attention algorithm:
    - Process queries in chunks
    - For each query chunk, process KV in chunks
    - Use online softmax to combine results

    Memory: O(q_chunk * kv_chunk) - CONSTANT regardless of sequence length!
    Time: O(n²) but with excellent memory efficiency

    Works on MPS/CPU by avoiding full attention matrix materialization.
    """
    def __init__(self, dim, heads=4, q_chunk_size=2048, kv_chunk_size=2048, causal=True):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.q_chunk_size = q_chunk_size
        self.kv_chunk_size = kv_chunk_size
        self.causal = causal
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        batch, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(batch, seq_len, self.heads, self.head_dim).transpose(1, 2)
                   for t in qkv]

        output = torch.zeros_like(q)

        # Process queries in chunks
        for i in range(0, seq_len, self.q_chunk_size):
            i_end = min(i + self.q_chunk_size, seq_len)
            q_chunk = q[:, :, i:i_end, :]
            q_len = i_end - i

            # Accumulators for online softmax
            o_chunk = torch.zeros((batch, self.heads, q_len, self.head_dim), device=device, dtype=dtype)
            l_chunk = torch.zeros((batch, self.heads, q_len, 1), device=device, dtype=dtype)
            m_chunk = torch.full((batch, self.heads, q_len, 1), float('-inf'), device=device, dtype=dtype)

            kv_end_limit = i_end if self.causal else seq_len

            # Process KV in chunks
            for j in range(0, kv_end_limit, self.kv_chunk_size):
                j_end = min(j + self.kv_chunk_size, kv_end_limit)

                k_chunk = k[:, :, j:j_end, :]
                v_chunk = v[:, :, j:j_end, :]

                # Compute attention scores
                scores = torch.einsum('bhqd,bhkd->bhqk', q_chunk, k_chunk) * self.scale

                # Apply causal mask
                if self.causal:
                    q_pos = torch.arange(i, i_end, device=device).unsqueeze(1)
                    k_pos = torch.arange(j, j_end, device=device).unsqueeze(0)
                    mask = q_pos >= k_pos
                    scores = scores.masked_fill(~mask, float('-inf'))

                # Online softmax update
                m_new = torch.maximum(m_chunk, scores.max(dim=-1, keepdim=True).values)
                alpha = torch.exp(m_chunk - m_new)
                l_chunk = l_chunk * alpha
                o_chunk = o_chunk * alpha

                p = torch.exp(scores - m_new)
                l_chunk = l_chunk + p.sum(dim=-1, keepdim=True)
                o_chunk = o_chunk + torch.einsum('bhqk,bhkd->bhqd', p, v_chunk)
                m_chunk = m_new

            output[:, :, i:i_end, :] = o_chunk / l_chunk.clamp(min=1e-6)

        out = output.transpose(1, 2).contiguous().view(batch, seq_len, self.dim)
        return self.to_out(out)


# =============================================================================
# Minimal Transformer for C4 Simulation
# =============================================================================

class C4SimulatorTransformer(nn.Module):
    """
    Minimal transformer for simulating C4 execution.

    State encoding:
    - 1MB C4 state encoded as sequence of tokens
    - Each token represents a memory cell or register

    Uses linear attention for O(n) complexity.
    """
    def __init__(self,
                 state_size=1024*1024,  # 1MB
                 dim=64,                 # small hidden dim
                 depth=4,                # few layers
                 heads=4,
                 bytes_per_token=4,      # pack 4 bytes per position
                 attention_type='linear'):
        super().__init__()

        self.state_size = state_size
        self.dim = dim
        self.bytes_per_token = bytes_per_token
        self.seq_len = state_size // bytes_per_token

        # Embedding: byte values (0-255) to hidden dim
        self.embed = nn.Embedding(256, dim // bytes_per_token)
        self.pos_embed = nn.Embedding(self.seq_len, dim)

        # Attention layers
        self.layers = nn.ModuleList()
        for _ in range(depth):
            if attention_type == 'linear':
                attn = LinearAttention(dim, heads)
            elif attention_type == 'causal_linear':
                attn = CausalLinearAttention(dim, heads)
            elif attention_type == 'window':
                attn = SlidingWindowAttention(dim, heads, window_size=1024)
            elif attention_type == 'flash':
                attn = FlashAttention(dim, heads, causal=True)
            elif attention_type == 'tiled_flash':
                attn = TiledFlashAttention(dim, heads, q_chunk_size=2048, kv_chunk_size=2048, causal=True)
            else:
                raise ValueError(f"Unknown attention type: {attention_type}")

            self.layers.append(nn.ModuleDict({
                'attn': attn,
                'ff': nn.Sequential(
                    nn.Linear(dim, dim * 2),
                    nn.GELU(),
                    nn.Linear(dim * 2, dim)
                ),
                'norm1': nn.LayerNorm(dim),
                'norm2': nn.LayerNorm(dim)
            }))

        # Output: predict next memory state
        self.out = nn.Linear(dim, 256 * bytes_per_token)

    def forward(self, state_bytes):
        """
        state_bytes: (batch, state_size) - raw bytes of C4 state
        Returns: (batch, seq_len, 256 * bytes_per_token) - predictions
        """
        batch = state_bytes.shape[0]

        # Reshape to (batch, seq_len, bytes_per_token)
        x = state_bytes.view(batch, self.seq_len, self.bytes_per_token)

        # Embed each byte and concat
        x = self.embed(x)  # (batch, seq_len, bytes_per_token, dim//bytes_per_token)
        x = x.view(batch, self.seq_len, self.dim)

        # Add positional embedding
        positions = torch.arange(self.seq_len, device=x.device)
        x = x + self.pos_embed(positions)

        # Transformer layers
        for layer in self.layers:
            x = x + layer['attn'](layer['norm1'](x))
            x = x + layer['ff'](layer['norm2'](x))

        return self.out(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def estimate_memory(self, batch_size=1):
        """Estimate memory usage in GB."""
        # Parameters
        param_mem = self.count_parameters() * 4 / 1e9

        # Activations (rough estimate)
        # Main activations: batch * seq_len * dim * num_layers * 2
        act_mem = batch_size * self.seq_len * self.dim * len(self.layers) * 2 * 4 / 1e9

        # For linear attention, no O(n²) attention matrix!
        # KV cache: batch * heads * head_dim * head_dim * num_layers
        head_dim = self.dim // 4
        kv_mem = batch_size * 4 * head_dim * head_dim * len(self.layers) * 4 / 1e9

        return {
            'parameters': param_mem,
            'activations': act_mem,
            'kv_cache': kv_mem,
            'total': param_mem + act_mem + kv_mem
        }


# =============================================================================
# Tests
# =============================================================================

def test_memory_estimates():
    """Estimate memory for different configurations."""
    print("=" * 60)
    print("MEMORY ESTIMATES FOR C4 TRANSFORMER")
    print("=" * 60)

    configs = [
        {'state_size': 1024*1024, 'dim': 64, 'depth': 4, 'bytes_per_token': 4},
        {'state_size': 1024*1024, 'dim': 128, 'depth': 4, 'bytes_per_token': 4},
        {'state_size': 1024*1024, 'dim': 64, 'depth': 8, 'bytes_per_token': 4},
        {'state_size': 1024*1024, 'dim': 64, 'depth': 4, 'bytes_per_token': 8},
    ]

    for cfg in configs:
        model = C4SimulatorTransformer(**cfg, attention_type='linear')
        mem = model.estimate_memory(batch_size=1)
        seq_len = cfg['state_size'] // cfg['bytes_per_token']

        print(f"\nConfig: dim={cfg['dim']}, depth={cfg['depth']}, bytes/tok={cfg['bytes_per_token']}")
        print(f"  Sequence length: {seq_len:,}")
        print(f"  Parameters: {model.count_parameters():,} ({mem['parameters']*1000:.1f} MB)")
        print(f"  Est. activations: {mem['activations']*1000:.1f} MB")
        print(f"  Est. total: {mem['total']*1000:.1f} MB")


def test_linear_attention_scaling():
    """Test that linear attention scales to 1M tokens."""
    print("\n" + "=" * 60)
    print("LINEAR ATTENTION SCALING TEST")
    print("=" * 60)

    dim = 64
    heads = 4
    attn = LinearAttention(dim, heads)

    for seq_len in [1000, 10000, 100000, 262144, 500000, 1000000]:
        gc.collect()

        try:
            x = torch.randn(1, seq_len, dim)

            start = time.time()
            with torch.no_grad():
                out = attn(x)
            elapsed = time.time() - start

            mem_est = seq_len * dim * 4 * 3 / 1e6  # Q, K, V

            print(f"  seq_len={seq_len:>10,}: {elapsed:.3f}s, ~{mem_est:.0f}MB tensors")

            del x, out
            gc.collect()

        except RuntimeError as e:
            print(f"  seq_len={seq_len:>10,}: FAILED - {e}")
            break


def test_full_c4_simulation():
    """Test full C4 state simulation."""
    print("\n" + "=" * 60)
    print("FULL C4 STATE SIMULATION TEST")
    print("=" * 60)

    # 1MB state, packed into 262,144 positions (4 bytes each)
    model = C4SimulatorTransformer(
        state_size=1024*1024,
        dim=64,
        depth=4,
        heads=4,
        bytes_per_token=4,
        attention_type='linear'
    )

    print(f"Model created:")
    print(f"  Sequence length: {model.seq_len:,}")
    print(f"  Parameters: {model.count_parameters():,}")

    mem = model.estimate_memory(batch_size=1)
    print(f"  Estimated memory: {mem['total']*1000:.1f} MB")

    # Test forward pass
    print("\nTesting forward pass...")
    gc.collect()

    # Create random C4 state (1MB of bytes)
    state = torch.randint(0, 256, (1, 1024*1024), dtype=torch.long)

    try:
        start = time.time()
        with torch.no_grad():
            output = model(state)
        elapsed = time.time() - start

        print(f"  Forward pass: {elapsed:.2f}s")
        print(f"  Output shape: {output.shape}")
        print(f"  SUCCESS: Can process 1MB C4 state!")

    except RuntimeError as e:
        print(f"  FAILED: {e}")


def test_comparison_attention_types():
    """Compare different attention mechanisms."""
    print("\n" + "=" * 60)
    print("ATTENTION TYPE COMPARISON")
    print("=" * 60)

    seq_len = 100000  # 100K for quick test
    dim = 64

    attentions = {
        'linear': LinearAttention(dim, 4),
        'causal_linear': CausalLinearAttention(dim, 4),
        'window_1024': SlidingWindowAttention(dim, 4, window_size=1024),
        'flash': FlashAttention(dim, 4, causal=True),
        'chunked_flash_64k': ChunkedFlashAttention(dim, 4, chunk_size=65536, causal=True),
    }

    x = torch.randn(1, seq_len, dim)

    for name, attn in attentions.items():
        gc.collect()
        try:
            start = time.time()
            with torch.no_grad():
                out = attn(x)
            elapsed = time.time() - start
            print(f"  {name:20s}: {elapsed:.3f}s, output shape {out.shape}")
        except RuntimeError as e:
            print(f"  {name:20s}: FAILED - {e}")


def test_flash_attention_scaling():
    """Test flash attention scaling to large sequences."""
    print("\n" + "=" * 60)
    print("FLASH ATTENTION SCALING TEST")
    print("=" * 60)

    dim = 64
    heads = 4

    # Test regular flash attention
    print("\nRegular Flash Attention:")
    attn = FlashAttention(dim, heads, causal=True)

    for seq_len in [10000, 50000, 100000, 200000, 262144]:
        gc.collect()
        try:
            x = torch.randn(1, seq_len, dim)
            start = time.time()
            with torch.no_grad():
                out = attn(x)
            elapsed = time.time() - start
            print(f"  seq_len={seq_len:>10,}: {elapsed:.3f}s")
            del x, out
        except RuntimeError as e:
            print(f"  seq_len={seq_len:>10,}: FAILED - {str(e)[:50]}")
            break

    # Test chunked flash attention for 1M
    print("\nChunked Flash Attention (for 1M+):")
    attn_chunked = ChunkedFlashAttention(dim, heads, chunk_size=65536, causal=True)

    for seq_len in [100000, 262144, 500000, 1000000]:
        gc.collect()
        try:
            x = torch.randn(1, seq_len, dim)
            start = time.time()
            with torch.no_grad():
                out = attn_chunked(x)
            elapsed = time.time() - start
            print(f"  seq_len={seq_len:>10,}: {elapsed:.3f}s")
            del x, out
        except RuntimeError as e:
            print(f"  seq_len={seq_len:>10,}: FAILED - {str(e)[:50]}")
            break


if __name__ == '__main__':
    test_memory_estimates()
    test_linear_attention_scaling()
    test_flash_attention_scaling()
    test_comparison_attention_types()
    test_full_c4_simulation()
