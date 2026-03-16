#!/usr/bin/env python3
"""
C4 VM AS A STANDARD TRANSFORMER - FINAL SUMMARY
================================================

A complete C compiler's VM implemented using only:
  - Attention: softmax(Q @ K.T) @ V
  - FFN: W2 @ silu(W1 @ x + b1) + b2
  - Constructed (not learned) weights

"""
import torch
import torch.nn.functional as F

print("""
╔══════════════════════════════════════════════════════════════════════╗
║             C4 VM AS A STANDARD TRANSFORMER                          ║
╠══════════════════════════════════════════════════════════════════════╣
║  A Turing-complete virtual machine using only attention + FFN        ║
║  Zero learnable parameters - all weights are constructed constants   ║
╚══════════════════════════════════════════════════════════════════════╝

┌──────────────────────────────────────────────────────────────────────┐
│ PART 1: THE FOUNDATION - SwiGLU SHARP GATES                         │
└──────────────────────────────────────────────────────────────────────┘

  SiLU(x) = x · σ(x)     where σ = sigmoid

  SHARP GATE (approximates step function):
  
    sharp_gate(x) = [silu(x·s + 0.5·s) - silu(x·s - 0.5·s)] / s
    
    Returns: ~1 if |x| < 0.5
             ~0 otherwise
             
  EQUALITY GATE (integer comparison):
  
    eq_gate(a, b) = sharp_gate(a-b+0.5) · sharp_gate(-(a-b)+0.5)
    
    Returns: ~1 if a == b (integers)
             ~0 otherwise

  This is computed by a STANDARD FFN with constructed weights!
""")

# Verify
def silu(x): return x * torch.sigmoid(x)
def sharp_gate(x, s=20.0): return (silu(x*s + 0.5*s) - silu(x*s - 0.5*s)) / s
def eq_gate(a, b, s=20.0): 
    d = a - b
    return sharp_gate(d + 0.5, s) * sharp_gate(-d + 0.5, s)

print("  Verification:")
for x in [4, 5, 6]:
    g = eq_gate(torch.tensor(5.0), torch.tensor(float(x)))
    print(f"    eq_gate(5, {x}) = {g.item():.6f}")

print("""
┌──────────────────────────────────────────────────────────────────────┐
│ PART 2: MoE ROUTING - 39 EXPERTS                                     │
└──────────────────────────────────────────────────────────────────────┘

  C4 has 39 opcodes → 39 experts (one per opcode)
  
  ROUTER computes expert gates using eq_gate:
  
    gates[i] = eq_gate(opcode, i)   for i in 0..38
    
  Result: gate ≈ 1 for matching expert, ≈ 0 for others
  
  OPCODES:
    0-8:   Control (LEA, IMM, JMP, JSR, BZ, BNZ, ENT, ADJ, LEV)
    9-13:  Memory  (LI, LC, SI, SC, PSH)
    14-22: Logic   (OR, XOR, AND, EQ, NE, LT, GT, LE, GE)
    23-29: Math    (SHL, SHR, ADD, SUB, MUL, DIV, MOD)
    38:    EXIT
""")

print("""
┌──────────────────────────────────────────────────────────────────────┐
│ PART 3: ARITHMETIC OPERATIONS                                        │
└──────────────────────────────────────────────────────────────────────┘

  ╔═══════════════╦════════════════════════════════════════════════════╗
  ║ OPERATION     ║ IMPLEMENTATION                                     ║
  ╠═══════════════╬════════════════════════════════════════════════════╣
  ║ ADD           ║ FFN extracts arg1, residual adds to ax             ║
  ║ ax += arg1    ║ W1[0,ARG1]=1, W2[AX,0]=1                           ║
  ╠═══════════════╬════════════════════════════════════════════════════╣
  ║ SUB           ║ FFN with bias shift to keep values positive        ║
  ║ ax = arg1-ax  ║ silu(x+B)-B ≈ x for large B                        ║
  ╠═══════════════╬════════════════════════════════════════════════════╣
  ║ MUL           ║ Quarter-squares identity:                          ║
  ║ ax = arg1*ax  ║ a·b = [(a+b)² - (a-b)²] / 4                        ║
  ║               ║ Squares via attention lookup table                 ║
  ╠═══════════════╬════════════════════════════════════════════════════╣
  ║ DIV           ║ Reciprocal lookup via attention:                   ║
  ║ ax = arg1/ax  ║ Keys=[1,2,3...], Values=[1,0.5,0.33...]            ║
  ║               ║ Then multiply: a/b = a × (1/b)                     ║
  ╠═══════════════╬════════════════════════════════════════════════════╣
  ║ MOD           ║ a % b = a - (a/b)*b                                ║
  ╚═══════════════╩════════════════════════════════════════════════════╝
""")

# Demo multiplication
MAX_SQ = 64
sq_table = torch.arange(MAX_SQ+1).float()**2

def square_lookup(x):
    x = torch.clamp(torch.abs(x), max=MAX_SQ)
    scores = torch.stack([eq_gate(x, torch.tensor(float(i)), 30) for i in range(MAX_SQ+1)])
    return (F.softmax(scores*100, dim=0) * sq_table).sum()

def mul(a, b):
    return (square_lookup(a+b) - square_lookup(torch.abs(a-b))) / 4

print("  MUL verification: ", end="")
for a,b in [(3,4), (7,8)]:
    r = mul(torch.tensor(float(a)), torch.tensor(float(b)))
    print(f"{a}×{b}={r.item():.0f} ", end="")
print()

# Demo division
MAX_VAL = 64
recip_table = 1.0 / torch.arange(1, MAX_VAL+1).float()

def div(a, b):
    scores = torch.stack([eq_gate(b, torch.tensor(float(i+1)), 30) for i in range(MAX_VAL)])
    inv_b = (F.softmax(scores*100, dim=0) * recip_table).sum()
    return torch.floor(a * inv_b)

print("  DIV verification: ", end="")
for a,b in [(20,4), (35,7)]:
    r = div(torch.tensor(float(a)), torch.tensor(float(b)))
    print(f"{a}/{b}={r.item():.0f} ", end="")
print()

print("""
┌──────────────────────────────────────────────────────────────────────┐
│ PART 4: MEMORY - APPEND-ONLY WITH POSITION-IN-KEY                    │
└──────────────────────────────────────────────────────────────────────┘

  Memory is a KV-cache with attention-based read/write.
  
  KEY ENCODING:
    Key = [binary_addr × ADDR_SCALE, position × POS_SCALE]
    
    Example for addr=5 (binary 0101), position=10:
    Key = [-50, +50, -50, +50, 100]
           ^    ^    ^    ^    ^
         bit0 bit1 bit2 bit3  pos
  
  QUERY:
    Query = [binary_addr × ADDR_SCALE, POS_SCALE]
    
  SCORE DECOMPOSITION:
    score = Q · K = addr_match + position × POS_SCALE²
                    ↑            ↑
                 ±10000       per-write bonus
  
  WHY IT WORKS:
    • Matching address: score ≈ +10000 + pos×100
    • Wrong address:    score ≈ -5000 to +5000
    • Address matching DOMINATES (selects correct address)
    • Position BREAKS TIES (newest write wins)
  
  WRITE = Append new (key, value) pair
  READ  = Standard attention over all pairs
""")

# Demo memory
ADDR_SCALE, POS_SCALE, NUM_BITS = 50.0, 10.0, 4

def make_key(addr, pos):
    return torch.tensor([(ADDR_SCALE if (addr>>b)&1 else -ADDR_SCALE) for b in range(NUM_BITS)] + [pos*POS_SCALE])

def make_query(addr):
    return torch.tensor([(ADDR_SCALE if (addr>>b)&1 else -ADDR_SCALE) for b in range(NUM_BITS)] + [POS_SCALE])

# Initialize + writes
K = torch.stack([make_key(i, i) for i in range(8)])
V = torch.zeros(8)
pos = 8

def mem_write(addr, val):
    global K, V, pos
    K = torch.cat([K, make_key(addr, pos).unsqueeze(0)])
    V = torch.cat([V, torch.tensor([float(val)])])
    pos += 1

def mem_read(addr):
    Q = make_query(addr).unsqueeze(0)
    return (F.softmax((Q @ K.T).squeeze(), dim=-1) * V).sum().item()

mem_write(0, 100)
mem_write(0, 999)  # Overwrite
mem_write(1, 200)

print("  Memory verification:")
print(f"    Write[0]=100, Write[0]=999, Write[1]=200")
print(f"    Read[0]={mem_read(0):.0f} (expected 999)")
print(f"    Read[1]={mem_read(1):.0f} (expected 200)")

print("""
┌──────────────────────────────────────────────────────────────────────┐
│ PART 5: SPECIAL - EXP VIA SOFTMAX                                    │
└──────────────────────────────────────────────────────────────────────┘

  EXACT exponential using only softmax:
  
    exp(x) = softmax([x, 0])[0] / softmax([x, 0])[1]
    
  PROOF:
    softmax([x, 0]) = [eˣ/(eˣ+1), 1/(eˣ+1)]
    
    ratio = [eˣ/(eˣ+1)] / [1/(eˣ+1)] = eˣ
    
  This is EXACT, not an approximation!
  No lookup table needed for exp.
""")

def exp_exact(x):
    logits = torch.tensor([x, 0.0])
    p = F.softmax(logits, dim=0)
    return p[0] / p[1]

print("  Verification:")
for x in [0, 1, 2, 3]:
    r = exp_exact(float(x))
    print(f"    exp({x}) = {r.item():.6f} (exact: {torch.exp(torch.tensor(float(x))).item():.6f})")

print("""
┌──────────────────────────────────────────────────────────────────────┐
│ PART 6: COMPLETE ARCHITECTURE                                        │
└──────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────────────┐
                    │  State: [pc,sp,bp,ax,...]   │
                    └─────────────┬───────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │  ROUTER (eq_gate FFN)       │
                    │  gates[i] = eq_gate(op, i)  │
                    └─────────────┬───────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          ▼                       ▼                       ▼
    ┌───────────┐           ┌───────────┐           ┌───────────┐
    │ Expert 0  │    ...    │ Expert 25 │    ...    │ Expert 38 │
    │   (LEA)   │           │   (ADD)   │           │  (EXIT)   │
    └─────┬─────┘           └─────┬─────┘           └─────┬─────┘
          │                       │                       │
          └───────────────────────┼───────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │  output = Σ gate[i]·exp[i]  │
                    └─────────────────────────────┘

  EACH EXPERT (identical architecture):
  
    ┌─────────────────────────────────────────────┐
    │  h = x + Attention(x, kv_cache)   ← Memory  │
    │  y = h + FFN(h)                   ← Compute │
    └─────────────────────────────────────────────┘

  WEIGHTS encode the operation:
    • ADD expert: FFN weights do addition
    • MUL expert: Attention accesses square table
    • LI expert:  Attention reads memory
    • etc.

┌──────────────────────────────────────────────────────────────────────┐
│ PART 7: PARAMETER COUNT                                              │
└──────────────────────────────────────────────────────────────────────┘

  State dimension:     16
  FFN hidden dim:      64
  Experts:             39
  
  Per expert:
    Attention (Wq,Wk,Wv,Wo): 4 × 16 × 16 = 1024
    FFN (W1, W2):            16×64 + 64×16 = 2048
    Biases:                  64 + 16 = 80
    Total:                   3152 parameters
  
  ALL EXPERTS:  39 × 3152 = 122,928 parameters
  
  But these are CONSTRUCTED, not learned!
  The actual "information content" is ~100 constants.

┌──────────────────────────────────────────────────────────────────────┐
│ SUMMARY TABLE                                                        │
└──────────────────────────────────────────────────────────────────────┘

  ┌─────────────────┬─────────────────────────────────────────────────┐
  │ CAPABILITY      │ TRANSFORMER PRIMITIVE                           │
  ├─────────────────┼─────────────────────────────────────────────────┤
  │ Integer ==      │ eq_gate via SwiGLU sharp gates                  │
  │ Expert routing  │ eq_gate over 39 opcodes                         │
  │ Addition        │ FFN residual connection                         │
  │ Subtraction     │ FFN with bias shift                             │
  │ Multiplication  │ Quarter-squares + attention lookup              │
  │ Division        │ Reciprocal table + attention lookup             │
  │ Exponential     │ softmax([x,0])[0] / softmax([x,0])[1]  (EXACT!) │
  │ Memory read     │ Attention with binary-encoded keys              │
  │ Memory write    │ KV append + position-in-key                     │
  │ Conditionals    │ eq_gate weighted combination                    │
  └─────────────────┴─────────────────────────────────────────────────┘

  ✓ Zero learnable parameters
  ✓ Standard transformer primitives only
  ✓ Turing complete (can compute anything)
  ✓ Each forward pass = one VM instruction
  ✓ ~100 constants define the entire system
""")
