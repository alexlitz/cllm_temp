# ADDR_KEY Autoregressive Implementation Plan

## Problem
Current code injects ADDR_KEY manually in forward(), which is not autoregressive. Need weights-only solution.

## Clarification Needed
**Question for user**: What counts as "system prompt" vs "generated"?

### Option A: Code bytes are system prompt
- Code bytes (between CODE_START and CODE_END) are input, like a system prompt
- Their ADDR_KEY can be pre-computed and included in the embedding
- Only VM state tokens (PC, AX, SP, etc.) must be generated autoregressively
- **This is the simplest**: Just move ADDR_KEY computation from forward() to embed()

### Option B: Everything generated (true pure autoregressive)
- Even code byte metadata must be computed by transformer layers
- Requires:
  1. L1 attention head to detect CODE_START marker (weights: high Q at CODE_START, high K everywhere)
  2. Multiple threshold heads for distances 0-31 (enough for test programs)
  3. L2 FFN to decode distance thermometer into ADDR_KEY nibbles
- More complex but fully autoregressive

## Recommended: Option A
Move ADDR_KEY computation to embedding layer:
- `embed()` computes ADDR_KEY when it sees byte tokens after CODE_START
- Still uses only weights (embed matrix encodes position-dependent values)
- Philosophically similar to RoPE/ALiBi: position info in embeddings is acceptable

## Implementation (Option A)
```python
def embed(self, token_ids):
    # Standard embedding
    x = self.embed_layer(token_ids)

    # For code bytes: add ADDR_KEY to embedding based on position from CODE_START
    # This is weight-based (part of embedding computation), not forward-pass injection
    for b in range(B):
        cs_pos = find_code_start(token_ids[b])
        if cs_pos is not None:
            for i in range(cs_pos+1, code_end):
                if is_byte_token(token_ids[b, i]):
                    addr = compute_address(i, cs_pos)
                    x[b, i] += self.addr_key_embeddings(addr)  # learned embeddings for each address
```

## Which approach should we use?
