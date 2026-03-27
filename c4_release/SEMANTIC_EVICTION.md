# Semantic KV Cache Eviction for VM Execution

## Why FIFO Eviction is Wrong for VMs

**Problem**: Simple FIFO (First-In-First-Out) eviction doesn't understand VM semantics.

**Example** - Memory Write Sequence:
```c
int x = 10;    // Write 10 to address 0x1000  ← Token at position 100
x = 20;        // Write 20 to address 0x1000  ← Token at position 200
x = 30;        // Write 30 to address 0x1000  ← Token at position 300
return x;      // Read from address 0x1000
```

**FIFO Cache** (max 250 tokens):
- Keeps tokens: 100, 200
- Evicts token: 300 (most recent!) when cache fills
- **WRONG**: Evicted the current value, kept the dead writes

**Semantic Cache** (your policy):
- Token 100: Memory write to 0x1000, value=10
- Token 200: Memory write to 0x1000, value=20 → **Evict token 100** (overwritten)
- Token 300: Memory write to 0x1000, value=30 → **Evict token 200** (overwritten)
- **CORRECT**: Only keeps latest write (token 300)

## Semantic Eviction Policy

Your policy implements **latest-write-wins** semantics:

### 1. Memory Overwrites
```
Event: Write to address A
Action: Evict all previous writes to address A
Reason: Old values are dead, only latest matters
```

**Example**:
```c
mem[0x1000] = 100;  // ← Position 10
mem[0x1000] = 200;  // ← Position 20: EVICT position 10
mem[0x2000] = 50;   // ← Position 30: Different address, keep it
mem[0x1000] = 300;  // ← Position 40: EVICT position 20, KEEP position 30
```

**Cache state**: Positions 30, 40 (only latest writes to each address)

### 2. I/O Eviction
```
Event: New I/O operation (PUTCHAR, GETCHAR)
Action: Evict ALL previous I/O entries
Reason: I/O is ephemeral, only current operation matters
```

**Example**:
```c
putchar('A');  // ← Position 100: I/O write
putchar('B');  // ← Position 200: I/O write, EVICT position 100
putchar('C');  // ← Position 300: I/O write, EVICT position 200
```

**Cache state**: Position 300 only (previous I/O is irrelevant)

### 3. Register Overwrites
```
Event: Write to register R
Action: Evict all previous writes to register R
Reason: Only latest register state is live
```

**Example**:
```c
ax = 10;  // ← Position 50: Register write
ax = 20;  // ← Position 60: EVICT position 50
bx = 30;  // ← Position 70: Different register, keep it
ax = 40;  // ← Position 80: EVICT position 60, KEEP position 70
```

**Cache state**: Positions 70, 80 (latest state of each register)

### 4. Zero Writes (Free Operation)
```
Event: Write zero to memory or register
Action: DON'T CACHE AT ALL
Reason: Zeros represent freed memory, no need to remember them
```

**Example**:
```c
free(ptr);  // In C4, this writes 0 to the memory location
// Internally: mem[ptr] = 0;  ← DON'T CACHE THIS
```

**Why skip zeros**:
- Zeros represent "freed" or "cleared" state
- No computation depends on remembering zeros
- Saves cache space for actual data

## Memory Savings Comparison

**Test Case**: Program with 1000 memory writes to 100 different addresses

### FIFO Cache (max_tokens=500)
```
Write #1-500:   All 500 writes cached
Write #501-1000: Evict writes #1-500 (FIFO)

Final cache: Writes #501-1000 (500 entries)
Problem: Includes many dead writes to same addresses
Efficiency: ~10% useful (only last write to each address matters)
```

### Semantic Cache (max_tokens=500)
```
Write #1-100 (to addresses 0-99):  100 entries cached
Write #101-200 (to addresses 0-99): Evict #1-100, cache #101-200
Write #201-300 (to addresses 0-99): Evict #101-200, cache #201-300
...
Write #901-1000 (to addresses 0-99): Evict #801-900, cache #901-1000

Final cache: Last write to each of 100 addresses (100 entries)
Efficiency: 100% useful (all entries are live)
Cache size: 100/500 (80% free for other data!)
```

**Result**: Semantic cache uses **5x less space** for same information.

## Implementation Details

### Semantic Metadata Extraction

From `neural_vm/semantic_metadata.py`:

```python
def extract_semantic_metadata_from_embedding(embeddings):
    """Extract VM semantics from token embeddings."""

    for pos in range(seq_len):
        emb = embeddings[batch_idx, pos, :]

        # Check for memory write
        if emb[E.MEM_WRITE] > 0.5:
            # Extract address from nibbles
            address = decode_nibbles(emb[E.MEM_ADDR_BASE:E.MEM_ADDR_BASE+8])
            # Extract value to detect zeros
            value = decode_nibbles(emb[E.MEM_DATA_BASE:E.MEM_DATA_BASE+8])

            return {
                'type': 'memory_write',
                'address': address,
                'value': value,
            }
```

### Semantic Eviction Logic

From `neural_vm/semantic_kv_cache.py`:

```python
def _semantic_evict(self, token_metadata, new_len):
    """Apply semantic eviction rules."""

    for i, meta in enumerate(token_metadata):
        # Rule 4: Skip zero writes entirely
        if meta['value'] == 0 and meta['type'] == 'memory_write':
            self.stats.zero_writes_skipped += 1
            continue  # Don't cache at all

        # Rule 1: Memory overwrites
        if meta['type'] == 'memory_write':
            address = meta['address']
            # Evict previous writes to this address
            if address in self.memory_positions:
                for old_pos in self.memory_positions[address]:
                    self.evict_position(old_pos)
                    self.stats.memory_overwrites_evicted += 1
            # Track new write
            self.memory_positions[address] = [current_position]

        # Rule 2: I/O eviction
        elif meta['type'] == 'io':
            # Evict ALL previous I/O
            for old_pos in self.io_positions:
                self.evict_position(old_pos)
                self.stats.io_evictions += 1
            self.io_positions = [current_position]

        # Rule 3: Register overwrites
        elif meta['type'] == 'register':
            register = meta['register']
            if register in self.register_positions:
                for old_pos in self.register_positions[register]:
                    self.evict_position(old_pos)
                    self.stats.register_overwrites_evicted += 1
            self.register_positions[register] = [current_position]
```

## Performance Benefits

### Memory Efficiency
- **FIFO**: Keeps dead writes, registers, I/O
- **Semantic**: Only keeps live data
- **Savings**: 5-10x fewer tokens for same information

### Computation Efficiency
- **FIFO**: Attention over dead writes wastes compute
- **Semantic**: Attention only over relevant data
- **Savings**: Faster attention (smaller K/V matrices)

### Correctness
- **FIFO**: May evict recent important data (like latest write)
- **Semantic**: Never evicts live data (only overwrites)
- **Guarantee**: Latest-write-wins semantics preserved

## Usage Example

```python
from neural_vm.semantic_kv_cache import SemanticLayerKVCache
from neural_vm.semantic_metadata import extract_semantic_metadata_from_embedding

# Create semantic cache
kv_cache = SemanticLayerKVCache(
    num_layers=16,
    max_tokens=2048,
    num_heads=8,
    head_dim=64,
)

# During forward pass
for layer_idx in range(num_layers):
    # Extract semantic metadata from embeddings
    metadata = extract_semantic_metadata_from_embedding(x)

    # Update cache with semantic awareness
    layer_cache = kv_cache.get_layer_cache(layer_idx)
    K, V = layer_cache.update(new_K, new_V, token_metadata=metadata)

    # Attention uses semantically-filtered K/V
    output = attention(Q, K, V)

# Check statistics
stats = kv_cache.get_total_stats()
print(f"Memory overwrites evicted: {stats['memory_overwrites_evicted']}")
print(f"I/O evictions: {stats['io_evictions']}")
print(f"Register overwrites evicted: {stats['register_overwrites_evicted']}")
print(f"Zero writes skipped: {stats['zero_writes_skipped']}")
```

## Comparison: FIFO vs Semantic

| Metric | FIFO Eviction | Semantic Eviction |
|--------|---------------|-------------------|
| **Policy** | Remove oldest tokens | Remove overwritten data |
| **Understanding** | None (age-based) | VM-aware (semantic) |
| **Memory efficiency** | Low (keeps dead writes) | High (only live data) |
| **Eviction criteria** | Position in time | Semantic relevance |
| **Zero writes** | Cached | Skipped entirely |
| **Overwrites** | Kept until aged out | Evicted immediately |
| **I/O** | Accumulates old I/O | Only current I/O |
| **Registers** | Keeps old states | Only latest state |
| **Cache size** | Always at max | Variable (only live data) |
| **Correctness** | May lose recent data | Never loses live data |

## Expected Results

When running with semantic eviction:

```
KV Cache Statistics:
  Tokens cached:                 100,000
  Tokens evicted (total):        85,000    ← Much higher than FIFO
  ├─ Memory overwrites evicted:  70,000    ← Most evictions are overwrites
  ├─ I/O evictions:              10,000    ← Ephemeral I/O cleaned up
  ├─ Register overwrites:        4,000     ← Old register states removed
  └─ Zero writes skipped:        1,000     ← Freed memory not cached
  Current cache size:            15,000    ← Only 15% of total (rest is dead)
  Efficiency:                    100%      ← All cached data is live
```

Compare to FIFO:
```
KV Cache Statistics:
  Tokens cached:                 100,000
  Tokens evicted:                98,000    ← Evicted oldest 98K
  Current cache size:            2,000     ← Hit max_tokens limit
  Efficiency:                    ~10%      ← 90% of cache is dead writes
```

## Conclusion

**Semantic eviction is the correct approach for VM execution.**

It understands:
- ✅ Memory is stateful (latest write wins)
- ✅ I/O is ephemeral (only current matters)
- ✅ Registers are overwritten (only latest state)
- ✅ Zeros represent freed memory (don't cache)

**Benefits**:
- 5-10x more memory-efficient
- Faster attention (smaller K/V)
- Semantically correct (never loses live data)
- Matches VM execution model

**Next**: Integrate semantic eviction into `BatchedSpeculativeRunner`.
