"""
Position as part of the key, not a separate bias.

Key = [addr_bits..., position * pos_scale]
Query = [query_bits..., pos_query]

The challenge: position should only matter for MATCHING addresses.
We achieve this by making address scale >> position scale.
"""
import torch
import torch.nn.functional as F

NUM_BITS = 4
ADDR_SCALE = 50.0   # High scale for address bits
POS_SCALE = 1.0     # Lower scale for position

def make_key(addr, position):
    """Key includes both address and position."""
    bits = []
    for b in range(NUM_BITS):
        bit = (int(addr) >> b) & 1
        bits.append(ADDR_SCALE if bit else -ADDR_SCALE)
    # Append position as part of key
    bits.append(position * POS_SCALE)
    return torch.tensor(bits, dtype=torch.float32)

def make_query(addr):
    """Query for address, with position component that favors high positions."""
    bits = []
    for b in range(NUM_BITS):
        bit = (int(addr) >> b) & 1
        bits.append(ADDR_SCALE if bit else -ADDR_SCALE)
    # Query wants HIGH position (newer entries)
    bits.append(POS_SCALE)  # Dot with key's position gives position * POS_SCALE^2
    return torch.tensor(bits, dtype=torch.float32)

# Initialize
K = torch.stack([make_key(i, i) for i in range(8)])  # positions 0-7
V = torch.tensor([100., 200., 300., 400., 500., 600., 700., 800.])

print("POSITION ENCODED IN KEY")
print("=" * 60)
print(f"Key = [addr_bits (scale={ADDR_SCALE}), position (scale={POS_SCALE})]")
print(f"Query = [addr_bits, {POS_SCALE}]  (to favor high positions)")
print()

print("Initial keys (showing addr 0,1,2):")
for i in range(3):
    print(f"  K[{i}] = {K[i].tolist()}")

# Test reads
print("\nInitial reads:")
for addr in range(3):
    Q = make_query(addr).unsqueeze(0)
    scores = (Q @ K.T).squeeze()
    weights = F.softmax(scores, dim=-1)
    val = (weights * V).sum().item()
    print(f"  Read[{addr}] = {val:.1f} (expected {(addr+1)*100})")

# Now append writes
print("\n--- Appending writes ---")
writes = [(0, 111, 8), (0, 999, 9), (1, 222, 10)]  # (addr, val, position)

for addr, val, pos in writes:
    new_key = make_key(addr, pos).unsqueeze(0)
    K = torch.cat([K, new_key], dim=0)
    V = torch.cat([V, torch.tensor([val])])
    print(f"  Append: addr={addr}, val={val}, pos={pos}")

print(f"\nTotal entries: {len(V)}")

# Read back
print("\nReads after appends:")
expected = {0: 999, 1: 222, 2: 300}

for addr in range(3):
    Q = make_query(addr).unsqueeze(0)
    scores = (Q @ K.T).squeeze()
    weights = F.softmax(scores, dim=-1)
    val = (weights * V).sum().item()
    exp = expected[addr]
    
    # Debug: show scores for relevant entries
    print(f"\n  Read[{addr}] = {val:.1f} (expected {exp}) {'✓' if abs(val-exp)<1 else '✗'}")
    
    # Find entries for this address
    for i, (s, w, v) in enumerate(zip(scores, weights, V)):
        if w > 0.001:  # Show significant weights
            print(f"    Entry {i}: score={s:.1f}, weight={w:.4f}, val={v:.0f}")

print("\n" + "=" * 60)
print("Analysis: Does address scale dominate position?")
print(f"  Matching addr score: ~{NUM_BITS * ADDR_SCALE**2:.0f}")
print(f"  Off-by-1 addr score: ~{(NUM_BITS-2) * ADDR_SCALE**2:.0f}")
print(f"  Position contribution: position * {POS_SCALE**2}")
print(f"  Max position diff effect: ~10 * {POS_SCALE**2} = {10 * POS_SCALE**2}")
