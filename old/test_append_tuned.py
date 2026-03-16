"""
Tuned parameters for clean latest-wins semantics.
"""
import torch
import torch.nn.functional as F

NUM_BITS = 4
SCALE = 10.0
POSITION_BIAS = 20.0  # Increased from 5 to 20
THRESHOLD = 350.0
SHARPNESS = 0.5  # Sharper gating

def make_key(addr):
    bits = []
    for b in range(NUM_BITS):
        bit = (int(addr) >> b) & 1
        bits.append(SCALE if bit else -SCALE)
    return torch.tensor(bits, dtype=torch.float32)

class AppendMemory:
    def __init__(self, size=8):
        self.K = torch.stack([make_key(i) for i in range(size)])
        self.V = torch.zeros(size)
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

print("APPEND MEMORY - TUNED PARAMETERS")
print("=" * 60)
print(f"POSITION_BIAS={POSITION_BIAS}, SHARPNESS={SHARPNESS}")

mem = AppendMemory(8)

writes = [
    (0, 10), (1, 20), (2, 30),
    (0, 100), (0, 200), (0, 300),
    (1, 111), (1, 222),
    (5, 555),
    (0, 999),
]

for addr, val in writes:
    mem.write(addr, val)

expected = {0: 999, 1: 222, 2: 30, 3: 0, 4: 0, 5: 555, 6: 0, 7: 0}

print("\nRead all addresses:")
all_pass = True
for addr in range(8):
    val = mem.read(addr)
    exp = expected[addr]
    ok = abs(val - exp) < 0.5
    all_pass = all_pass and ok
    status = "✓" if ok else "✗"
    print(f"  Read[{addr}] = {val:.2f} (expected {exp}) {status}")

print(f"\n{'ALL TESTS PASS!' if all_pass else 'SOME TESTS FAILED'}")

# Now show this works in pure attention form
print("\n" + "=" * 60)
print("ENCODING IN STANDARD ATTENTION")
print("=" * 60)
print("""
The conditional position bias can be encoded as:

1. Base attention: Q @ K.T computes address matching scores
2. Gate: sigmoid((scores - threshold) * sharpness)  
3. Position bias: gate * positions * POSITION_BIAS
4. Final: softmax(base_scores + position_bias) @ V

In standard transformer terms:
- Wk encodes binary address → key
- Wq encodes address query
- We need a "scoring head" that applies the sigmoid gate
- This can be done with an FFN between attention layers:
  
  FFN computes: gate = sigmoid((base_score - threshold) * sharpness)
  Then modifies scores: biased = base + gate * position_embed

Or more elegantly, use TWO attention heads:
- Head 1: standard address matching
- Head 2: position-aware, masked by head 1's output
""")
