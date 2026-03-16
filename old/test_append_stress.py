"""
Stress test: Many writes to same addresses, verify latest-wins semantics.
"""
import torch
import torch.nn.functional as F

NUM_BITS = 4
SCALE = 10.0
POSITION_BIAS = 5.0
THRESHOLD = 350.0
SHARPNESS = 0.1

def make_key(addr):
    bits = []
    for b in range(NUM_BITS):
        bit = (int(addr) >> b) & 1
        bits.append(SCALE if bit else -SCALE)
    return torch.tensor(bits, dtype=torch.float32)

class AppendMemory:
    def __init__(self, size=8):
        self.K = torch.stack([make_key(i) for i in range(size)])
        self.V = torch.zeros(size)  # Start with zeros
        self.positions = torch.arange(size, dtype=torch.float32)
    
    def write(self, addr, val):
        new_pos = self.positions.max() + 1
        self.K = torch.cat([self.K, make_key(addr).unsqueeze(0)])
        self.V = torch.cat([self.V, torch.tensor([val])])
        self.positions = torch.cat([self.positions, torch.tensor([new_pos])])
    
    def read(self, addr):
        Q = make_key(addr).unsqueeze(0)
        base_scores = (Q @ self.K.T).squeeze()
        gate = torch.sigmoid((base_scores - THRESHOLD) * SHARPNESS)
        biased_scores = base_scores + gate * self.positions * POSITION_BIAS
        weights = F.softmax(biased_scores, dim=-1)
        return (weights * self.V).sum().item()

print("STRESS TEST: Multiple writes to same addresses")
print("=" * 60)

mem = AppendMemory(8)

# Write sequence
writes = [
    (0, 10), (1, 20), (2, 30),  # Initial
    (0, 100), (0, 200), (0, 300),  # Triple rewrite addr 0
    (1, 111), (1, 222),  # Double rewrite addr 1
    (5, 555),  # Single write addr 5
    (0, 999),  # Final write to addr 0
]

print("\nWrite sequence:")
for addr, val in writes:
    mem.write(addr, val)
    print(f"  Write[{addr}] = {val}")

print(f"\nTotal entries in memory: {len(mem.V)}")

# Expected final values
expected = {0: 999, 1: 222, 2: 30, 3: 0, 4: 0, 5: 555, 6: 0, 7: 0}

print("\nRead all addresses:")
all_pass = True
for addr in range(8):
    val = mem.read(addr)
    exp = expected[addr]
    ok = abs(val - exp) < 0.1
    all_pass = all_pass and ok
    status = "✓" if ok else "✗"
    print(f"  Read[{addr}] = {val:.1f} (expected {exp}) {status}")

print(f"\n{'ALL TESTS PASS!' if all_pass else 'SOME TESTS FAILED'}")
print("=" * 60)

# Show how position bias helps
print("\nDEMO: Why conditional gating matters")
print("-" * 40)
Q = make_key(0).unsqueeze(0)
base = (Q @ mem.K.T).squeeze()
gate = torch.sigmoid((base - THRESHOLD) * SHARPNESS)

print(f"Reading addr 0:")
print(f"  Base scores (first 12): {[f'{b:.0f}' for b in base[:12]]}")
print(f"  Gates (first 12): {[f'{g:.2f}' for g in gate[:12]]}")
print(f"  Positions (first 12): {[f'{p:.0f}' for p in mem.positions[:12]]}")

# Without gate, position bias would boost non-matching entries
ungated_bias = mem.positions * POSITION_BIAS
gated_bias = gate * mem.positions * POSITION_BIAS
print(f"\n  Ungated pos bias (first 12): {[f'{b:.0f}' for b in ungated_bias[:12]]}")
print(f"  Gated pos bias (first 12): {[f'{b:.0f}' for b in gated_bias[:12]]}")
