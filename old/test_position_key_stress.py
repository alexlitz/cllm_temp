"""
Stress test: position-in-key approach with many writes.
"""
import torch
import torch.nn.functional as F

NUM_BITS = 4
ADDR_SCALE = 50.0
POS_SCALE = 10.0

def make_key(addr, position):
    bits = [(ADDR_SCALE if (int(addr) >> b) & 1 else -ADDR_SCALE) for b in range(NUM_BITS)]
    bits.append(position * POS_SCALE)
    return torch.tensor(bits, dtype=torch.float32)

def make_query(addr):
    bits = [(ADDR_SCALE if (int(addr) >> b) & 1 else -ADDR_SCALE) for b in range(NUM_BITS)]
    bits.append(POS_SCALE)
    return torch.tensor(bits, dtype=torch.float32)

class Memory:
    def __init__(self, size=8):
        self.pos_counter = size
        self.K = torch.stack([make_key(i, i) for i in range(size)])
        self.V = torch.zeros(size)
    
    def write(self, addr, val):
        self.K = torch.cat([self.K, make_key(addr, self.pos_counter).unsqueeze(0)])
        self.V = torch.cat([self.V, torch.tensor([val])])
        self.pos_counter += 1
    
    def read(self, addr):
        Q = make_query(addr).unsqueeze(0)
        scores = (Q @ self.K.T).squeeze()
        weights = F.softmax(scores, dim=-1)
        return (weights * self.V).sum().item()

print("STRESS TEST - Position in Key")
print("=" * 60)

mem = Memory(8)

# Many writes to same addresses
writes = [
    (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),  # 5 writes to addr 0
    (1, 10), (1, 20), (1, 30),  # 3 writes to addr 1
    (7, 700), (7, 777),  # 2 writes to addr 7
    (0, 999),  # Final write to addr 0
]

for addr, val in writes:
    mem.write(addr, val)

print(f"Total entries: {len(mem.V)}")
print()

expected = {0: 999, 1: 30, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 777}

print("Reads:")
all_pass = True
for addr in range(8):
    val = mem.read(addr)
    exp = expected[addr]
    ok = abs(val - exp) < 0.5
    all_pass = all_pass and ok
    print(f"  Read[{addr}] = {val:.1f} (expected {exp}) {'✓' if ok else '✗'}")

print(f"\n{'ALL PASS!' if all_pass else 'SOME FAIL'}")

print("\n" + "=" * 60)
print("SUMMARY: Position-in-key approach")
print("=" * 60)
print("""
Key structure:   K = [addr_bits, position * POS_SCALE]
Query structure: Q = [addr_bits, POS_SCALE]
Score: Q · K = addr_match + position * POS_SCALE²

Requirements:
  - ADDR_SCALE² * NUM_BITS >> max_position * POS_SCALE²
  - This ensures address matching dominates
  - POS_SCALE² large enough that 1 position step dominates in softmax

This is SIMPLER than conditional gating:
  - Just one attention layer
  - Position is part of the key embedding
  - No separate FFN for gating
""")
