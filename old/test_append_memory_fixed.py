"""
Append-based memory with CONDITIONAL position bias.

The key insight: position bias should only apply to entries that MATCH the address.
We achieve this by gating the position bias with a sharp sigmoid of the base score.
"""
import torch
import torch.nn.functional as F

# Configuration
NUM_BITS = 4
SCALE = 10.0
POSITION_BIAS = 5.0
THRESHOLD = 350.0  # Base score threshold for "matching address"
SHARPNESS = 0.1    # How sharp the gating function is

def make_key(addr, scale=SCALE):
    """Binary-encode address into key vector."""
    bits = []
    for b in range(NUM_BITS):
        bit = (int(addr) >> b) & 1
        bits.append(scale if bit else -scale)
    return torch.tensor(bits, dtype=torch.float32)

# Initialize memory with 8 slots
K = torch.stack([make_key(i) for i in range(8)])  # [8, 4]
V = torch.tensor([100., 200., 300., 400., 500., 600., 700., 800.])
positions = torch.arange(8, dtype=torch.float32)  # Initial positions 0-7

print("APPEND MEMORY WITH CONDITIONAL POSITION BIAS")
print("=" * 60)
print(f"\nConfig: NUM_BITS={NUM_BITS}, SCALE={SCALE}")
print(f"POSITION_BIAS={POSITION_BIAS}, THRESHOLD={THRESHOLD}")
print(f"\nInitial memory: {V.tolist()}")
print(f"Initial positions: {positions.tolist()}")

def read(addr):
    """Read with conditional position bias."""
    Q = make_key(addr).unsqueeze(0)  # [1, 4]
    base_scores = (Q @ K.T).squeeze()  # Raw address matching scores
    
    # Gate: only apply position bias when base score is high (address matches)
    # sigmoid((score - threshold) * sharpness) → ~1 for matches, ~0 for non-matches
    gate = torch.sigmoid((base_scores - THRESHOLD) * SHARPNESS)
    
    # Position bias only applies to matching entries
    gated_position_bias = gate * positions * POSITION_BIAS
    biased_scores = base_scores + gated_position_bias
    
    weights = F.softmax(biased_scores, dim=-1)
    value = (weights * V).sum()
    
    winner_idx = torch.argmax(weights).item()
    return value.item(), winner_idx, base_scores, gate, biased_scores, weights

def write(addr, val):
    """Append new entry with incremented position."""
    global K, V, positions
    new_pos = positions.max() + 1
    K = torch.cat([K, make_key(addr).unsqueeze(0)], dim=0)
    V = torch.cat([V, torch.tensor([val])])
    positions = torch.cat([positions, torch.tensor([new_pos])])

# Initial reads
print("\n--- Initial reads (before any writes) ---")
for addr in range(4):
    val, winner, base, gate, biased, weights = read(addr)
    expected = (addr + 1) * 100
    status = "✓" if abs(val - expected) < 1 else "✗"
    print(f"Read[{addr}]: {val:.1f} (expected {expected}) {status}")
    print(f"  Base scores[0:4]: {base[:4].tolist()}")
    print(f"  Gate[0:4]: {[f'{g:.3f}' for g in gate[:4].tolist()]}")

# Now write to addresses 0, 1, 2, 3
print("\n--- Writing new values ---")
writes = [(0, 111), (1, 222), (2, 333), (0, 999)]  # Note: addr 0 written TWICE
for addr, val in writes:
    write(addr, val)
    print(f"Write[{addr}] = {val}  (positions now has {len(positions)} entries)")

# Read back
print("\n--- Reads after writes ---")
expected_values = {0: 999, 1: 222, 2: 333, 3: 400}  # addr 0 should be 999 (latest)

for addr in [0, 1, 2, 3]:
    val, winner, base, gate, biased, weights = read(addr)
    expected = expected_values[addr]
    status = "✓" if abs(val - expected) < 1 else "✗"
    
    print(f"\nRead[{addr}]: {val:.1f} (expected {expected}) {status}")
    print(f"  Winner: pos[{winner}] = {V[winner]:.0f}")
    
    # Show entries for this address
    matching = []
    for i, (b, g, w, v) in enumerate(zip(base, gate, weights, V)):
        if b > 300:  # Likely matches
            matching.append(f"pos[{i}]:v={v:.0f},gate={g:.2f},w={w:.3f}")
    print(f"  Matching entries: {matching}")

print("\n" + "=" * 60)
print("Key insight: gate suppresses position bias for non-matching addresses")
print("So wrong-address entries don't get boosted by high position numbers")
