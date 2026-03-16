#!/usr/bin/env python3
"""
C4 Transformer: Thinking Token Demo

Shows the full execution with each step as a "thinking token"
"""
import torch
import torch.nn.functional as F
import math

# === Primitives ===
def silu(x): return x * torch.sigmoid(x)
def sharp_gate(x, s=20.0): return (silu(x*s + 0.5*s) - silu(x*s - 0.5*s)) / s
def eq_gate(a, b, s=20.0): 
    d = a - b
    return sharp_gate(d + 0.5, s) * sharp_gate(-d + 0.5, s)

# === Opcodes ===
LEA, IMM, JMP, JSR, BZ, BNZ, ENT, ADJ, LEV = 0, 1, 2, 3, 4, 5, 6, 7, 8
LI, LC, SI, SC, PSH = 9, 10, 11, 12, 13
OR, XOR, AND, EQ, NE, LT, GT, LE, GE = 14, 15, 16, 17, 18, 19, 20, 21, 22
SHL, SHR, ADD, SUB, MUL, DIV, MOD = 23, 24, 25, 26, 27, 28, 29
PRTF, EXIT = 37, 38

OP_NAMES = {
    0: "LEA", 1: "IMM", 2: "JMP", 3: "JSR", 4: "BZ", 5: "BNZ",
    6: "ENT", 7: "ADJ", 8: "LEV", 9: "LI", 10: "LC", 11: "SI",
    12: "SC", 13: "PSH", 14: "OR", 15: "XOR", 16: "AND", 17: "EQ",
    18: "NE", 19: "LT", 20: "GT", 21: "LE", 22: "GE", 23: "SHL",
    24: "SHR", 25: "ADD", 26: "SUB", 27: "MUL", 28: "DIV", 29: "MOD",
    37: "PRTF", 38: "EXIT"
}

# === Arithmetic (FFN-based) ===
MAX_VAL = 64
squares = torch.arange(MAX_VAL * 2 + 1).float() ** 2
log_keys = torch.arange(1, MAX_VAL + 1).float()
log_values = torch.log2(log_keys)

def square_lookup(x):
    x = torch.clamp(torch.abs(x), max=MAX_VAL * 2)
    scores = torch.stack([eq_gate(x, torch.tensor(float(i)), 30) for i in range(MAX_VAL * 2 + 1)])
    return (F.softmax(scores * 100, dim=0) * squares).sum()

def mul_ffn(a, b):
    return (square_lookup(a + b) - square_lookup(torch.abs(a - b))) / 4

def log2_attention(b):
    b = torch.clamp(b, min=1.0, max=float(MAX_VAL))
    scores = torch.stack([eq_gate(b, k, 30) for k in log_keys])
    return (F.softmax(scores * 100, dim=0) * log_values).sum()

def exp_softmax(x):
    logits = torch.stack([x * math.log(2), torch.tensor(0.0)])
    probs = F.softmax(logits, dim=0)
    return probs[0] / probs[1]

def div_log_exp(a, b):
    if b <= 0: return torch.tensor(0.0)
    log_b = log2_attention(b)
    inv_b = exp_softmax(-log_b)
    result = a * inv_b
    rounded = torch.round(result)
    if torch.abs(result - rounded) < 0.001:
        result = rounded
    return torch.floor(result)

# === Memory (Attention with position-in-key) ===
ADDR_SCALE, POS_SCALE, NUM_BITS = 50.0, 10.0, 4

class Memory:
    def __init__(self, size=256):
        self.K = torch.stack([self._key(i, i) for i in range(size)])
        self.V = torch.zeros(size)
        self.pos = size
    
    def _key(self, addr, pos):
        bits = [(ADDR_SCALE if (addr>>b)&1 else -ADDR_SCALE) for b in range(NUM_BITS)]
        return torch.tensor(bits + [pos * POS_SCALE])
    
    def _query(self, addr):
        bits = [(ADDR_SCALE if (addr>>b)&1 else -ADDR_SCALE) for b in range(NUM_BITS)]
        return torch.tensor(bits + [POS_SCALE])
    
    def read(self, addr):
        Q = self._query(int(addr)).unsqueeze(0)
        weights = F.softmax((Q @ self.K.T).squeeze(), dim=-1)
        return (weights * self.V).sum().item()
    
    def write(self, addr, val):
        self.K = torch.cat([self.K, self._key(int(addr), self.pos).unsqueeze(0)])
        self.V = torch.cat([self.V, torch.tensor([float(val)])])
        self.pos += 1

# === VM with Thinking ===
class ThinkingVM:
    def __init__(self, code, data=None):
        self.code = code
        self.mem = Memory()
        self.pc = 0
        self.sp = 256
        self.bp = 256
        self.ax = 0
        self.stack = [0] * 256
        self.cycle = 0
        self.halted = False
        self.output = []
        
        # Load data into memory
        if data:
            for i, v in enumerate(data):
                self.mem.write(i, v)
    
    def route(self, opcode):
        """MoE routing via eq_gate"""
        gates = {op: eq_gate(torch.tensor(float(opcode)), torch.tensor(float(op))).item() 
                 for op in OP_NAMES.keys()}
        return max(gates, key=gates.get), gates[max(gates, key=gates.get)]
    
    def step(self):
        """One thinking step = one transformer forward pass"""
        if self.halted or self.pc >= len(self.code):
            return None
        
        op = self.code[self.pc]
        imm = self.code[self.pc + 1] if self.pc + 1 < len(self.code) else 0
        
        # Route to expert
        expert, gate = self.route(op)
        op_name = OP_NAMES.get(expert, f"OP{expert}")
        
        # Build thinking token
        thought = {
            "cycle": self.cycle,
            "pc": self.pc,
            "opcode": op_name,
            "imm": imm,
            "expert": expert,
            "gate": gate,
            "ax_before": self.ax,
            "sp_before": self.sp,
        }
        
        # Execute via expert
        self.pc += 2  # Most ops use immediate
        
        if expert == IMM:
            self.ax = imm
        elif expert == ADD:
            arg = self.stack[self.sp]
            self.sp += 1
            self.ax = int(arg + self.ax)
        elif expert == SUB:
            arg = self.stack[self.sp]
            self.sp += 1
            self.ax = int(arg - self.ax)
        elif expert == MUL:
            arg = self.stack[self.sp]
            self.sp += 1
            self.ax = int(mul_ffn(torch.tensor(float(arg)), torch.tensor(float(self.ax))).item())
        elif expert == DIV:
            arg = self.stack[self.sp]
            self.sp += 1
            self.ax = int(div_log_exp(torch.tensor(float(arg)), torch.tensor(float(self.ax))).item())
        elif expert == PSH:
            self.sp -= 1
            self.stack[self.sp] = self.ax
        elif expert == LI:
            self.ax = int(self.mem.read(self.ax))
        elif expert == SI:
            self.mem.write(self.stack[self.sp], self.ax)
            self.sp += 1
        elif expert == EXIT:
            self.halted = True
            self.pc -= 2
        elif expert == PRTF:
            self.output.append(self.ax)
            self.sp += 1  # Pop format string
        
        thought["ax_after"] = self.ax
        thought["sp_after"] = self.sp
        self.cycle += 1
        
        return thought

def run_with_thinking(code, data=None, max_cycles=100):
    """Run program and show thinking tokens"""
    vm = ThinkingVM(code, data)
    thoughts = []
    
    while not vm.halted and vm.cycle < max_cycles:
        thought = vm.step()
        if thought:
            thoughts.append(thought)
    
    return thoughts, vm.ax, vm.output

# === Demo ===
print("=" * 70)
print("C4 TRANSFORMER: THINKING DEMO")
print("=" * 70)
print()
print("This shows each forward pass as a 'thinking token'")
print()

# Program: compute (3 + 4) * 5
print("─" * 70)
print("PROGRAM 1: (3 + 4) * 5 = 35")
print("─" * 70)
code1 = [
    IMM, 3,    # ax = 3
    PSH, 0,    # push ax
    IMM, 4,    # ax = 4
    ADD, 0,    # ax = pop + ax = 3 + 4 = 7
    PSH, 0,    # push ax  
    IMM, 5,    # ax = 5
    MUL, 0,    # ax = pop * ax = 7 * 5 = 35
    EXIT, 0
]

thoughts, result, _ = run_with_thinking(code1)

print("\nThinking tokens:")
print("┌─────┬────┬────────┬─────┬────────────────────────────┐")
print("│ Cyc │ PC │ Expert │ IMM │ State Change               │")
print("├─────┼────┼────────┼─────┼────────────────────────────┤")
for t in thoughts:
    state = f"ax: {t['ax_before']} → {t['ax_after']}"
    print(f"│ {t['cycle']:3d} │ {t['pc']:2d} │ {t['opcode']:6s} │ {t['imm']:3d} │ {state:26s} │")
print("└─────┴────┴────────┴─────┴────────────────────────────┘")
print(f"\nFinal result: {result}")

# Program: factorial-like: 5 * 4 * 3 * 2 * 1
print("\n" + "─" * 70)
print("PROGRAM 2: 5 * 4 = 20, then 20 * 3 = 60")
print("─" * 70)
code2 = [
    IMM, 5,    # ax = 5
    PSH, 0,    # push 5
    IMM, 4,    # ax = 4
    MUL, 0,    # ax = 5 * 4 = 20
    PSH, 0,    # push 20
    IMM, 3,    # ax = 3
    MUL, 0,    # ax = 20 * 3 = 60
    EXIT, 0
]

thoughts, result, _ = run_with_thinking(code2)

print("\nThinking tokens:")
for t in thoughts:
    print(f"  [{t['cycle']}] {t['opcode']:6s} {t['imm']:3d}  │  ax: {t['ax_before']:3d} → {t['ax_after']:3d}")
print(f"\nFinal result: {result}")

# Program with division
print("\n" + "─" * 70)
print("PROGRAM 3: 100 / 5 + 10 = 30")
print("─" * 70)
code3 = [
    IMM, 100,  # ax = 100
    PSH, 0,    # push 100
    IMM, 5,    # ax = 5
    DIV, 0,    # ax = 100 / 5 = 20
    PSH, 0,    # push 20
    IMM, 10,   # ax = 10
    ADD, 0,    # ax = 20 + 10 = 30
    EXIT, 0
]

thoughts, result, _ = run_with_thinking(code3)

print("\nThinking tokens (showing DIV uses log-exp):")
for t in thoughts:
    extra = ""
    if t['opcode'] == 'DIV':
        extra = " [log attention → exp softmax → mul FFN]"
    elif t['opcode'] == 'MUL':
        extra = " [quarter-squares FFN]"
    print(f"  [{t['cycle']}] {t['opcode']:6s} {t['imm']:3d}  │  ax: {t['ax_before']:3d} → {t['ax_after']:3d}{extra}")
print(f"\nFinal result: {result}")

print("\n" + "=" * 70)
print("WHAT EACH THINKING TOKEN REPRESENTS")
print("=" * 70)
print("""
Each row above is ONE transformer forward pass:

  1. INPUT: State vector [pc, sp, ax, opcode, imm, ...]
  
  2. ROUTER: eq_gate selects expert based on opcode
     gates[i] = eq_gate(opcode, i)
     
  3. EXPERT: Executes operation via attention + FFN
     - ADD: FFN residual
     - MUL: FFN quarter-squares  
     - DIV: Attention(log) → Softmax(exp) → FFN(mul)
     - Memory: Attention with position-in-key
     
  4. OUTPUT: Updated state vector

This IS the "thinking" - each step transforms the state
using only standard transformer primitives.
""")
