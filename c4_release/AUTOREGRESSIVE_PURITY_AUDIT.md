# Autoregressive Purity Audit

## Question

**Are we ensuring**:
1. ✅/❌ All computation in FFN, MoE, attention forwards?
2. ✅/❌ WITHOUT modification?
3. ✅/❌ All tokens 100% autoregressive?
4. ✅/❌ Except prompt?

## Findings

### ❌ VIOLATION 1: Embedding Modifications

**File**: `neural_vm/vm_step.py` lines 827-875

```python
def forward(self, token_ids, kv_cache=None):
    x = self.embed(token_ids)

    # MODIFICATION 1: Add code address keys
    self._add_code_addr_keys(token_ids, x)  # ← MODIFIES embeddings!

    # MODIFICATION 2: Inject mem store flags
    self._inject_mem_store(token_ids, x)    # ← MODIFIES embeddings!

    # THEN run through layers
    for i, block in enumerate(self.blocks):
        x = block(x, kv_cache=layer_cache)
    return self.head(x)
```

**What These Do**:

#### `_add_code_addr_keys()` (lines 837-875)
- Loops through token_ids in **Python**
- Computes addresses from position: `addr = i - cs_pos - 1`
- Writes to embedding: `x[b, i, BD.ADDR_KEY + lo] = 1.0`
- **Python arithmetic**, not neural computation

#### `_inject_mem_store()` (lines 877-900+)
- Loops through historical MEM markers in **Python**
- Sets flag: `x[b, i, BD.MEM_STORE] = 1.0`
- **Python modification**, not neural computation

### ❌ VIOLATION 2: Not Pure Autoregressive

**Issue**: The forward pass runs on the ENTIRE sequence at once:

```python
def forward(self, token_ids, kv_cache=None):
    # token_ids: [batch, seq] ← ENTIRE sequence!
    x = self.embed(token_ids)  # Embed ALL tokens
    # ... modifications to ALL tokens ...
    for block in self.blocks:
        x = block(x)  # Process ALL tokens at once
    return self.head(x)  # Get logits for ALL positions
```

**This is NOT token-by-token autoregressive generation!**

It's:
- ✅ Autoregressive attention (causal mask)
- ❌ NOT autoregressive generation (processes whole sequence)

### ❌ VIOLATION 3: Speculative Verification Not Pure

**File**: `neural_vm/vm_step.py` lines 958-983

```python
def verify_speculative_batch(self, contexts_with_draft, draft_lens, kv_cache=None):
    token_ids = torch.tensor(contexts_with_draft)
    logits = self.forward(token_ids, kv_cache=kv_cache)  # Full forward on context+draft

    # Then CHECK against draft tokens
    for i in range(draft_lens[b]):
        pred = logits[b, ctx_len - 1 + i, :].argmax(-1).item()
        if contexts_with_draft[b][ctx_len + i] == pred:
            accepted += 1
        else:
            break
```

**Issue**: This runs forward pass on `context + draft_tokens` ALL AT ONCE, not token-by-token.

## What SHOULD Happen (Pure Autoregressive)

### True Autoregressive Generation

```python
def generate_autoregressive(self, context, max_tokens):
    """Pure autoregressive: generate ONE token at a time."""
    tokens = context.copy()

    for _ in range(max_tokens):
        # Forward pass on ENTIRE context so far
        logits = self.forward(tokens)  # [1, len(tokens), vocab]

        # Predict NEXT token (only the last position)
        next_token = logits[0, -1, :].argmax(-1).item()

        # Append and continue
        tokens.append(next_token)

        if next_token == Token.HALT:
            break

    return tokens
```

**Key**: Each new token generated INDIVIDUALLY, depends on all previous.

### Pure Neural Computation (No Modifications)

```python
def forward(self, token_ids):
    """Pure forward: NO modifications outside layers."""
    x = self.embed(token_ids)  # ONLY embedding lookup

    # NO _add_code_addr_keys()
    # NO _inject_mem_store()
    # NO Python modifications

    # ALL computation through layers
    for block in self.blocks:
        x = block(x)  # FFN, MoE, Attention ONLY

    return self.head(x)
```

**Key**: ZERO Python arithmetic between embed and layers.

## Current Implementation Analysis

### What's Actually Happening

```
1. Embed tokens
   ↓
2. PYTHON: Add address keys ← VIOLATION
   ↓
3. PYTHON: Inject mem flags ← VIOLATION
   ↓
4. Run through layers (FFN, MoE, Attention) ← OK
   ↓
5. Output head ← OK
```

**28.6% of forward pass** (2/7 steps) **is non-neural Python code**.

### Speculative Decoding Flow

```
DraftVM generates tokens:
   [tok1, tok2, tok3, tok4, tok5]

Transformer:
   forward(context + [tok1, tok2, tok3, tok4, tok5])  ← Batch forward, not autoregressive!
   ↓
   Compare predictions vs draft
   ↓
   Accept/reject
```

**NOT token-by-token generation!**

## Why This Matters

### Principle Violations

1. **Not Pure Neural**: Python code does computation that SHOULD be learned
   - Address calculation should be learned by FFN layers
   - MEM_STORE flags should be learned by attention

2. **Not Truly Autoregressive**: Batch processing entire sequences
   - Should generate ONE token at a time
   - Each token should depend on forward pass through ALL previous

3. **Shortcuts Prevent Learning**: Hardcoded logic can't be improved
   - `_add_code_addr_keys` is fixed algorithm
   - Can't learn better address computation
   - Can't adapt to different code patterns

### Correctness Concerns

1. **Python Bugs**: Modifications could have bugs that neural model wouldn't
2. **Non-Differentiable**: Can't backprop through Python modifications
3. **Training Mismatch**: If trained without these modifications, inference is different

## Recommendations

### Option 1: Move to Embedding

Make these modifications part of the **embedding layer**, not forward pass:

```python
class NeuralVMEmbedding(nn.Module):
    def forward(self, token_ids):
        x = self.token_embed(token_ids)

        # Add positional info (like RoPE/ALiBi)
        x = self.add_positional_encoding(x, token_ids)

        # These become LEARNED, not hardcoded
        return x
```

### Option 2: Remove Modifications Entirely

Let the model **learn** these patterns through weights:

```python
def forward(self, token_ids):
    x = self.embed(token_ids)
    # NO modifications - layers learn everything
    for block in self.blocks:
        x = block(x)
    return self.head(x)
```

### Option 3: True Autoregressive Generation

Generate token-by-token:

```python
def generate_next_token(self, context):
    """Generate ONE token autoregressively."""
    logits = self.forward(context)  # Full forward
    return logits[0, -1, :].argmax(-1).item()

# Usage:
for step in range(max_steps):
    next_token = model.generate_next_token(context)
    context.append(next_token)
```

## Current Status

| Requirement | Status | Notes |
|-------------|--------|-------|
| All computation in FFN/MoE/Attention | ❌ NO | 2 Python modifications |
| WITHOUT modification | ❌ NO | Embeddings modified pre-layers |
| 100% autoregressive | ⚠️ PARTIAL | Causal attention, but batch forward |
| Except prompt | ✅ YES | Prompt treated as system input |

## Answer to Your Question

**"Are we ensuring all computation is done in FFN, MOE, attention forwards without modification?"**

**NO**. There are 2 Python modifications (`_add_code_addr_keys`, `_inject_mem_store`) that happen BEFORE the layers.

**"All tokens produced 100% autoregressively?"**

**PARTIALLY**. Causal attention is autoregressive, but generation is batched (processes multiple tokens per forward pass), not token-by-token.

## Fixing This

To achieve TRUE purity:

1. **Remove `_add_code_addr_keys()`** - Let model learn from position
2. **Remove `_inject_mem_store()`** - Let attention learn to mark memory
3. **Generate token-by-token** - One forward pass per token
4. **No Python arithmetic** - Only embedding → layers → head

This would be **100% pure neural computation** with **100% autoregressive generation**.
