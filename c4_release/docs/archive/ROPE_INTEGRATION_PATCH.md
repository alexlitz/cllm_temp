# RoPE Integration Patch for vm_step.py

This document contains all code changes needed to complete the RoPE/ALiBi integration in `neural_vm/vm_step.py`.

---

## Change 1: Update AutoregressiveAttention.__init__

**Location**: Line 90, `AutoregressiveAttention.__init__` method

**Find this:**
```python
def __init__(self, dim, num_heads=4, max_seq_len=4096):
    super().__init__()
    self.dim = dim
    self.num_heads = num_heads
    self.head_dim = dim // num_heads
    self.scale = self.head_dim**-0.5
    self.max_seq_len = max_seq_len

    self.W_q = nn.Parameter(torch.zeros(dim, dim))
    self.W_k = nn.Parameter(torch.zeros(dim, dim))
    self.W_v = nn.Parameter(torch.zeros(dim, dim))
    self.W_o = nn.Parameter(torch.zeros(dim, dim))

    # ALiBi slopes: geometric sequence 2^(-8/n * (i+1)) for each head
    slopes = torch.tensor(
        [2.0 ** (-8.0 / num_heads * (i + 1)) for i in range(num_heads)]
    )
    self.register_buffer("alibi_slopes", slopes)  # [H]
```

**Replace with:**
```python
def __init__(self, dim, num_heads=4, max_seq_len=4096, layer_idx=None):
    super().__init__()
    self.dim = dim
    self.num_heads = num_heads
    self.head_dim = dim // num_heads
    self.scale = self.head_dim**-0.5
    self.max_seq_len = max_seq_len
    self.layer_idx = layer_idx

    self.W_q = nn.Parameter(torch.zeros(dim, dim))
    self.W_k = nn.Parameter(torch.zeros(dim, dim))
    self.W_v = nn.Parameter(torch.zeros(dim, dim))
    self.W_o = nn.Parameter(torch.zeros(dim, dim))

    # Determine positional encoding for this layer
    try:
        from .config import get_config
        config = get_config()

        # Hybrid mode: L0-L2 use ALiBi, rest use RoPE
        if config.positional_encoding == "hybrid" and layer_idx is not None and layer_idx < 3:
            self._positional_encoding = "alibi"
        else:
            self._positional_encoding = config.positional_encoding
    except ImportError:
        # Fallback if config not available (backwards compatibility)
        self._positional_encoding = "alibi"

    # Initialize ALiBi slopes if using ALiBi (or hybrid mode with layer < 3)
    use_alibi = (self._positional_encoding == "alibi" or
                 (self._positional_encoding == "hybrid" and layer_idx is not None and layer_idx < 3))
    if use_alibi:
        slopes = torch.tensor(
            [2.0 ** (-8.0 / num_heads * (i + 1)) for i in range(num_heads)]
        )
        self.register_buffer("alibi_slopes", slopes)  # [H]
    else:
        self.alibi_slopes = None

    # Initialize RoPE cache if using RoPE (or hybrid mode with layer >= 3)
    use_rope = (self._positional_encoding == "rope" or
                (self._positional_encoding == "hybrid" and layer_idx is not None and layer_idx >= 3))
    if use_rope:
        try:
            from .config import get_config
            from .base_layers import precompute_rope_cache
            config = get_config()
            rope_base = config.rope_base
        except (ImportError, AttributeError):
            rope_base = 10000.0

        from .base_layers import precompute_rope_cache
        cos, sin = precompute_rope_cache(self.head_dim, max_seq_len, base=rope_base)
        self.register_buffer("_rope_cos", cos)
        self.register_buffer("_rope_sin", sin)
    else:
        self._rope_cos = None
        self._rope_sin = None
```

---

## Change 2: Update AutoregressiveAttention.forward - Add RoPE Application

**Location**: Line ~238 in `forward` method, after computing Q and K but before computing scores

**Find this:**
```python
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # ALiBi bias: -slope * |i - j|, computed on-the-fly
        # Note: With KV cache, query length S and key length S_kv may differ
        S_q = Q.shape[2]  # Query sequence length
        S_kv = K.shape[2]  # Cached key/value sequence length
        q_positions = torch.arange(S_q, device=x.device).unsqueeze(1)  # [S_q, 1]
        k_positions = torch.arange(S_kv, device=x.device).unsqueeze(0)  # [1, S_kv]
        dist = (q_positions - k_positions).abs().float()  # [S_q, S_kv]
        alibi = -self.alibi_slopes.view(1, H, 1, 1) * dist  # [1, H, S_q, S_kv]
        scores = scores + alibi
```

**Replace with:**
```python
        # Apply RoPE if enabled (check for RoPE cache presence)
        if self._rope_cos is not None:
            # Q and K are [B, H, S_q/S_kv, HD]
            S_q = Q.shape[2]
            S_kv = K.shape[2]

            # For cached scenarios, queries are at positions [S_kv - S_q, S_kv)
            # For non-cached scenarios, S_q == S_kv and queries are at [0, S_q)
            q_offset = S_kv - S_q

            # Apply RoPE to Q and K
            from .base_layers import rotate_half
            cos_q = self._rope_cos[q_offset:q_offset + S_q].unsqueeze(0).unsqueeze(0)  # [1, 1, S_q, HD]
            sin_q = self._rope_sin[q_offset:q_offset + S_q].unsqueeze(0).unsqueeze(0)  # [1, 1, S_q, HD]
            cos_k = self._rope_cos[0:S_kv].unsqueeze(0).unsqueeze(0)  # [1, 1, S_kv, HD]
            sin_k = self._rope_sin[0:S_kv].unsqueeze(0).unsqueeze(0)  # [1, 1, S_kv, HD]

            Q = (Q * cos_q) + (rotate_half(Q) * sin_q)
            K = (K * cos_k) + (rotate_half(K) * sin_k)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # ALiBi bias (only if using ALiBi): -slope * |i - j|, computed on-the-fly
        # Note: With KV cache, query length S and key length S_kv may differ
        # Check for alibi_slopes presence rather than _positional_encoding string
        if self.alibi_slopes is not None:
            S_q = Q.shape[2]  # Query sequence length
            S_kv = K.shape[2]  # Cached key/value sequence length
            q_positions = torch.arange(S_q, device=x.device).unsqueeze(1)  # [S_q, 1]
            k_positions = torch.arange(S_kv, device=x.device).unsqueeze(0)  # [1, S_kv]
            dist = (q_positions - k_positions).abs().float()  # [S_q, S_kv]
            alibi = -self.alibi_slopes.view(1, H, 1, 1) * dist  # [1, H, S_q, S_kv]
            scores = scores + alibi
```

---

## Change 3: Update compact() method to handle alibi_slopes conditionally

**Location**: Line ~204 in `compact` method

**Find this:**
```python
        # head_dim stays the same; alibi_slopes shrinks to active heads
        self.alibi_slopes = self.alibi_slopes[active_heads]
```

**Replace with:**
```python
        # head_dim stays the same; alibi_slopes shrinks to active heads (if present)
        if self.alibi_slopes is not None:
            self.alibi_slopes = self.alibi_slopes[active_heads]
```

---

## Change 4: Update AutoregressiveVM layer instantiation

**Location**: Line ~641 in `AutoregressiveVM.__init__`

**Find this:**
```python
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    attn=AutoregressiveAttention(
                        d_model, num_heads=n_heads, max_seq_len=max_seq_len
                    ),
                    ffn=PureFFN(d_model, ffn_hidden),
                )
                for _ in range(n_layers)
            ]
        )
```

**Replace with:**
```python
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    attn=AutoregressiveAttention(
                        d_model, num_heads=n_heads, max_seq_len=max_seq_len, layer_idx=i
                    ),
                    ffn=PureFFN(d_model, ffn_hidden),
                )
                for i in range(n_layers)
            ]
        )
```

---

## Change 5: Guard ALiBi-specific weight setting in bake_all_vm_weights

**Location 1**: Line ~1481 in `bake_all_vm_weights` function

**Find this:**
```python
    attn0 = model.blocks[0].attn
    attn0.alibi_slopes.fill_(ALIBI_S)
```

**Replace with:**
```python
    attn0 = model.blocks[0].attn
    # ALiBi-specific: set slopes for threshold heads
    if hasattr(attn0, 'alibi_slopes') and attn0.alibi_slopes is not None:
        attn0.alibi_slopes.fill_(ALIBI_S)
```

**Location 2**: Line ~1495

**Find this:**
```python
    attn1 = model.blocks[1].attn
    attn1.alibi_slopes.fill_(ALIBI_S)
    attn1.alibi_slopes[3] = 0.0  # Head 3: global attention for SE detection
```

**Replace with:**
```python
    attn1 = model.blocks[1].attn
    # ALiBi-specific: set slopes for threshold heads
    if hasattr(attn1, 'alibi_slopes') and attn1.alibi_slopes is not None:
        attn1.alibi_slopes.fill_(ALIBI_S)
        attn1.alibi_slopes[3] = 0.0  # Head 3: global attention for SE detection
```

---

## Change 6: Update class docstring

**Location**: Line 83

**Find this:**
```python
    """Multi-head attention with softmax1 (ZFOD) and ALiBi positional bias.

    NOT a PureAttention subclass — PureAttention.forward() is FINAL and uses
    F.softmax. This class uses softmax1 for zero-fill-on-demand semantics
    and adds ALiBi bias for recency/latest-write-wins.
    """
```

**Replace with:**
```python
    """Multi-head attention with softmax1 (ZFOD) and ALiBi/RoPE positional encoding.

    NOT a PureAttention subclass — PureAttention.forward() is FINAL and uses
    F.softmax. This class uses softmax1 for zero-fill-on-demand semantics
    and supports both ALiBi and RoPE positional encodings via config.
    """
```

---

## Testing After Integration

Once all changes are applied:

### 1. Run Unit Tests
```bash
python -m pytest tests/test_positional_encoding.py -v
```

Expected: All 20 tests passing

### 2. Test with ALiBi (default - backwards compatibility)
```bash
python tests/run_1000_tests.py
```

Expected: 1096/1096 passing (same as before)

### 3. Test with RoPE
```bash
NEURAL_VM_POS_ENCODING=rope python tests/run_1000_tests.py
```

Expected: Tests may fail initially (threshold heads trained for ALiBi patterns)

### 4. Test with Hybrid
```bash
NEURAL_VM_POS_ENCODING=hybrid python tests/run_1000_tests.py
```

Expected: Should pass (threshold layers use ALiBi, content layers use RoPE)

---

## Summary of Changes

- **Lines changed**: ~120 lines
- **Functions modified**: 4 (`__init__`, `forward`, `compact`, `bake_all_vm_weights`)
- **Backwards compatible**: Yes (default ALiBi mode)
- **New dependencies**: `neural_vm/config.py` (already created)

All changes are non-breaking and maintain full backwards compatibility through the default ALiBi mode.
