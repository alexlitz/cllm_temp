# RoPE/ALiBi Positional Encoding Implementation Status

## Overview

Implementation of configurable positional encoding toggle between RoPE (Rotary Position Embeddings) and ALiBi (Attention with Linear Biases) for the Neural VM.

---

## ✅ Completed Components

### 1. Configuration System (`neural_vm/config.py`)
- **Status**: ✅ Complete
- **Features**:
  - `VMConfig` dataclass with three modes:
    - `alibi` - 100% ALiBi (default, backwards compatible)
    - `rope` - 100% RoPE
    - `hybrid` - ALiBi for L0-L2, RoPE for L3+
  - Factory methods: `VMConfig.alibi_mode()`, `rope_mode()`, `hybrid_mode()`
  - Environment variable support: `NEURAL_VM_POS_ENCODING`
  - Global config management: `get_config()`, `set_config()`, `reset_config()`

### 2. RoPE Helper Functions (`neural_vm/base_layers.py`)
- **Status**: ✅ Complete
- **Functions**:
  - `rotate_half()` - Rotate half the hidden dims
  - `apply_rotary_emb()` - Apply RoPE to Q/K tensors
  - `precompute_rope_cache()` - Precompute cos/sin cache
- **Implementation**: Standard RoPE with `theta_k = 10000^(-2k/d)`

### 3. Unit Tests (`tests/test_positional_encoding.py`)
- **Status**: ✅ Complete (framework ready)
- **Test Coverage**:
  - Config system (8 tests) ✅ All passing
  - RoPE helpers (3 tests) ✅ All passing
  - AutoregressiveAttention (awaiting integration)
  - Full VM integration (awaiting integration)

---

## 🔄 Remaining Work

### 1. AutoregressiveAttention Integration (`neural_vm/vm_step.py`)

Need to modify `AutoregressiveAttention` class:

**A. Update `__init__` to accept `layer_idx` and initialize based on config:**

```python
def __init__(self, dim, num_heads=4, max_seq_len=4096, layer_idx=None):
    super().__init__()
    # ... existing initialization ...

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
        self._positional_encoding = "alibi"

    # Initialize ALiBi slopes if using ALiBi (or hybrid mode with layer < 3)
    use_alibi = (self._positional_encoding == "alibi" or
                 (self._positional_encoding == "hybrid" and layer_idx is not None and layer_idx < 3))
    if use_alibi:
        slopes = torch.tensor([2.0 ** (-8.0 / num_heads * (i + 1)) for i in range(num_heads)])
        self.register_buffer("alibi_slopes", slopes)
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

**B. Update `forward()` to apply RoPE and conditionally apply ALiBi:**

In the forward method, after computing Q and K but before computing scores:

```python
# Apply RoPE if enabled (check for RoPE cache presence)
if self._rope_cos is not None:
    S_q = Q.shape[2]
    S_kv = K.shape[2]
    q_offset = S_kv - S_q  # For KV cache support

    from .base_layers import rotate_half
    cos_q = self._rope_cos[q_offset:q_offset + S_q].unsqueeze(0).unsqueeze(0)
    sin_q = self._rope_sin[q_offset:q_offset + S_q].unsqueeze(0).unsqueeze(0)
    cos_k = self._rope_cos[0:S_kv].unsqueeze(0).unsqueeze(0)
    sin_k = self._rope_sin[0:S_kv].unsqueeze(0).unsqueeze(0)

    Q = (Q * cos_q) + (rotate_half(Q) * sin_q)
    K = (K * cos_k) + (rotate_half(K) * sin_k)

scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

# ALiBi bias (only if using ALiBi) - check for alibi_slopes presence
if self.alibi_slopes is not None:
    S_q = Q.shape[2]
    S_kv = K.shape[2]
    q_positions = torch.arange(S_q, device=x.device).unsqueeze(1)
    k_positions = torch.arange(S_kv, device=x.device).unsqueeze(0)
    dist = (q_positions - k_positions).abs().float()
    alibi = -self.alibi_slopes.view(1, H, 1, 1) * dist
    scores = scores + alibi
```

### 2. Layer Creation Update

Update layer instantiation in `AutoregressiveVM.__init__` (around line 699-709):

```python
self.blocks = nn.ModuleList(
    [
        TransformerBlock(
            attn=AutoregressiveAttention(
                d_model, num_heads=n_heads, max_seq_len=max_seq_len, layer_idx=i
            ),
            ffn=PureFFN(d_model, ffn_hidden),
        )
        for i in range(n_layers)  # Use i instead of _
    ]
)
```

### 3. ALiBi Weight Guards

Guard ALiBi-specific weight setting in `bake_all_vm_weights()`:

At line ~1481 and ~1495:

```python
# ===== LAYER 0: Step structure via threshold attention =====
attn0 = model.blocks[0].attn
# ALiBi-specific: set slopes for threshold heads
if hasattr(attn0, 'alibi_slopes') and attn0.alibi_slopes is not None:
    attn0.alibi_slopes.fill_(ALIBI_S)

# ===== LAYER 1: Fine thresholds + STEP_END detection =====
attn1 = model.blocks[1].attn
# ALiBi-specific: set slopes for threshold heads
if hasattr(attn1, 'alibi_slopes') and attn1.alibi_slopes is not None:
    attn1.alibi_slopes.fill_(ALIBI_S)
    attn1.alibi_slopes[3] = 0.0  # Head 3: global attention for SE detection
```

---

## 📋 Testing Plan

Once integration is complete:

### 1. Unit Tests
```bash
python -m pytest tests/test_positional_encoding.py -v
```

Expected: All 20 tests passing

### 2. Integration Tests with Full VM

**Test ALiBi mode (default, backwards compatible):**
```bash
python tests/test_suite_1000.py
```
Expected: 1096/1096 tests passing (same as current)

**Test RoPE mode:**
```bash
NEURAL_VM_POS_ENCODING=rope python tests/test_suite_1000.py
```
Expected: All tests passing (may need attention pattern adjustments)

**Test Hybrid mode:**
```bash
NEURAL_VM_POS_ENCODING=hybrid python tests/test_suite_1000.py
```
Expected: All tests passing

---

## 🔧 Implementation Notes

### Why RoPE?

**Advantages over ALiBi:**
- Better length extrapolation
- Multiplicative vs additive bias (more principled)
- Widely adopted standard (LLaMA, GPT-NeoX, etc.)
- Proven at scale

**Why keep ALiBi?**
- Existing weights trained with ALiBi
- Backwards compatibility
- Simpler threshold heads (distance-based attention)

### Hybrid Mode Rationale

- **L0-L2 use ALiBi**: Threshold layers rely on distance-based attention patterns
- **L3+ use RoPE**: Content layers benefit from RoPE's better position encoding

This provides a smooth transition path and allows testing both mechanisms.

---

## 📊 Expected Outcomes

### Performance

- **ALiBi mode**: Identical to current (100% backwards compatible)
- **RoPE mode**: May require attention weight retraining for threshold heads
- **Hybrid mode**: Best of both worlds - proven threshold patterns + better content encoding

### Compatibility

- **Existing checkpoints**: Continue working with ALiBi mode (default)
- **New models**: Can be trained with RoPE or hybrid
- **ONNX export**: Should work with all modes (static position embeddings)

---

## 🚀 Quick Start (Once Complete)

### Using RoPE Mode

```python
from neural_vm.config import set_config, VMConfig

# Set RoPE mode globally
set_config(VMConfig.rope_mode())

# Or use environment variable
# NEURAL_VM_POS_ENCODING=rope python your_script.py
```

### Using Hybrid Mode

```python
from neural_vm.config import set_config, VMConfig

set_config(VMConfig.hybrid_mode())
```

---

## 📝 Files Modified/Created

### Created
- ✅ `neural_vm/config.py` (91 lines)
- ✅ `tests/test_positional_encoding.py` (273 lines)
- ✅ RoPE helpers in `neural_vm/base_layers.py` (+62 lines)
- ✅ This status document

### To Modify
- 🔄 `neural_vm/vm_step.py`:
  - `AutoregressiveAttention.__init__` (~40 lines added)
  - `AutoregressiveAttention.forward` (~15 lines added)
  - Layer instantiation (1 line change)
  - Weight baking guards (2 locations, ~4 lines)

**Total Code Added**: ~350 lines
**Core Logic Changes**: ~60 lines (clean, modular)

---

## ✅ Next Steps

1. Apply the three changes to `neural_vm/vm_step.py` listed above
2. Run unit tests: `pytest tests/test_positional_encoding.py -v`
3. Run integration tests with all three modes
4. If RoPE mode fails tests, consider retraining threshold heads or using hybrid mode
5. Update main documentation with configuration options

---

**Status**: Foundation complete (config + helpers + tests). Integration pending (~1 hour estimated).
