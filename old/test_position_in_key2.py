"""
Position in key with higher position scale.
Need position diff to dominate in softmax.
"""
import torch
import torch.nn.functional as F

NUM_BITS = 4
ADDR_SCALE = 50.0
POS_SCALE = 10.0   # Increased! Each position step = 100 in score

def make_key(addr, position):
    bits = []
    for b in range(NUM_BITS):
        bit = (int(addr) >> b) & 1
        bits.append(ADDR_SCALE if bit else -ADDR_SCALE)
    bits.append(position * POS_SCALE)
    return torch.tensor(bits, dtype=torch.float32)

def make_query(addr):
    bits = []
    for b in range(NUM_BITS):
        bit = (int(addr) >> b) & 1
        bits.append(ADDR_SCALE if bit else -ADDR_SCALE)
    bits.append(POS_SCALE)  # Query component
    return torch.tensor(bits, dtype=torch.float32)

K = torch.stack([make_key(i, i) for i in range(8)])
V = torch.tensor([100., 200., 300., 400., 500., 600., 700., 800.])

print("POSITION IN KEY - Higher scale")
print("=" * 60)
print(f"Position scale = {POS_SCALE}")
print(f"Position diff of 1 → score diff of {POS_SCALE**2}")
print()

# Append writes
writes = [(0, 111, 8), (0, 999, 9), (1, 222, 10)]
for addr, val, pos in writes:
    K = torch.cat([K, make_key(addr, pos).unsqueeze(0)])
    V = torch.cat([V, torch.tensor([val])])

print("Reads after appends:")
expected = {0: 999, 1: 222, 2: 300}

all_pass = True
for addr in range(3):
    Q = make_query(addr).unsqueeze(0)
    scores = (Q @ K.T).squeeze()
    weights = F.softmax(scores, dim=-1)
    val = (weights * V).sum().item()
    exp = expected[addr]
    ok = abs(val - exp) < 1
    all_pass = all_pass and ok
    
    print(f"\n  Read[{addr}] = {val:.1f} (expected {exp}) {'✓' if ok else '✗'}")
    for i, (s, w, v) in enumerate(zip(scores, weights, V)):
        if w > 0.0001:
            print(f"    Entry {i}: score={s:.0f}, weight={w:.6f}, val={v:.0f}")

print(f"\n{'ALL PASS' if all_pass else 'SOME FAIL'}")
print()

# Check cross-address interference
print("Checking cross-address interference:")
print("  Matching addr score: ~10000 + pos * 100")
print("  Off-by-1 addr score: ~5000 + pos * 100")
print()
print("  For addr=0, newest entry at pos=9: score ≈ 10000 + 900 = 10900")
print("  For addr=1 entry at pos=10: score ≈ 5000 + 1000 = 6000")
print("  → Matching address still wins (10900 >> 6000)")
