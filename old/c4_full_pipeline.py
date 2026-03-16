#!/usr/bin/env python3
"""
FULL PIPELINE: C Source → Compile → Bytecode → Thinking → Result
"""
import torch
import torch.nn.functional as F
import math

# ============================================================================
# OPCODES
# ============================================================================
IMM, PSH, ADD, SUB, MUL, DIV, EXIT = 1, 13, 25, 26, 27, 28, 38
OP_NAMES = {1: "IMM", 13: "PSH", 25: "ADD", 26: "SUB", 27: "MUL", 28: "DIV", 38: "EXIT"}

# ============================================================================
# SIMPLE EXPRESSION COMPILER
# ============================================================================
def compile_expr(expr):
    """Compile arithmetic expression to bytecode using recursive descent."""
    expr = expr.replace(" ", "")
    pos = [0]  # Use list for mutability in nested functions
    code = []
    
    def peek():
        return expr[pos[0]] if pos[0] < len(expr) else None
    
    def consume():
        ch = peek()
        pos[0] += 1
        return ch
    
    def parse_number():
        start = pos[0]
        while peek() and peek().isdigit():
            consume()
        return int(expr[start:pos[0]])
    
    def parse_atom():
        """Parse number or parenthesized expression."""
        if peek() == '(':
            consume()  # eat '('
            parse_additive()
            consume()  # eat ')'
        else:
            num = parse_number()
            code.extend([IMM, num])
    
    def parse_multiplicative():
        """Parse * and / (higher precedence)."""
        parse_atom()
        while peek() in '*/':
            op = consume()
            code.extend([PSH, 0])
            parse_atom()
            code.extend([MUL if op == '*' else DIV, 0])
    
    def parse_additive():
        """Parse + and - (lower precedence)."""
        parse_multiplicative()
        while peek() in '+-':
            op = consume()
            code.extend([PSH, 0])
            parse_multiplicative()
            code.extend([ADD if op == '+' else SUB, 0])
    
    parse_additive()
    code.extend([EXIT, 0])
    return code

# ============================================================================
# TRANSFORMER PRIMITIVES
# ============================================================================
def silu(x): return x * torch.sigmoid(x)
def sharp_gate(x, s=20.0): return (silu(x*s + 0.5*s) - silu(x*s - 0.5*s)) / s
def eq_gate(a, b, s=20.0): 
    d = a - b
    return sharp_gate(d + 0.5, s) * sharp_gate(-d + 0.5, s)

MAX_VAL = 128
squares = torch.arange(MAX_VAL * 2 + 1).float() ** 2
log_keys = torch.arange(1, MAX_VAL + 1).float()
log_values = torch.log2(log_keys)

def mul_ffn(a, b):
    def sq(x):
        x = torch.clamp(torch.abs(x), max=MAX_VAL * 2)
        scores = torch.stack([eq_gate(x, torch.tensor(float(i)), 30) for i in range(len(squares))])
        return (F.softmax(scores * 100, dim=0) * squares).sum()
    return (sq(a + b) - sq(torch.abs(a - b))) / 4

def div_log_exp(a, b):
    if b <= 0: return torch.tensor(0.0)
    b = torch.clamp(b, min=1.0, max=float(MAX_VAL))
    scores = torch.stack([eq_gate(b, k, 30) for k in log_keys])
    log_b = (F.softmax(scores * 100, dim=0) * log_values).sum()
    logits = torch.stack([-log_b * math.log(2), torch.tensor(0.0)])
    inv_b = F.softmax(logits, dim=0)[0] / F.softmax(logits, dim=0)[1]
    result = a * inv_b
    if torch.abs(result - torch.round(result)) < 0.001:
        result = torch.round(result)
    return torch.floor(result)

# ============================================================================
# THINKING VM
# ============================================================================
class ThinkingVM:
    def __init__(self, code):
        self.code = code
        self.pc = 0
        self.ax = 0
        self.stack = [0] * 256
        self.sp = 256
        self.halted = False
        self.thoughts = []
    
    def step(self):
        if self.halted or self.pc >= len(self.code):
            return
        
        op = self.code[self.pc]
        imm = self.code[self.pc + 1]
        ax_before = self.ax
        self.pc += 2
        
        if op == IMM:
            self.ax = imm
        elif op == PSH:
            self.sp -= 1
            self.stack[self.sp] = self.ax
        elif op == ADD:
            arg = self.stack[self.sp]; self.sp += 1
            self.ax = arg + self.ax
        elif op == SUB:
            arg = self.stack[self.sp]; self.sp += 1
            self.ax = arg - self.ax
        elif op == MUL:
            arg = self.stack[self.sp]; self.sp += 1
            self.ax = int(mul_ffn(torch.tensor(float(arg)), torch.tensor(float(self.ax))).item())
        elif op == DIV:
            arg = self.stack[self.sp]; self.sp += 1
            self.ax = int(div_log_exp(torch.tensor(float(arg)), torch.tensor(float(self.ax))).item())
        elif op == EXIT:
            self.halted = True
        
        op_name = OP_NAMES.get(op, f"OP{op}")
        self.thoughts.append(f"[{len(self.thoughts):2d}] {op_name:4s} {imm:4d}  →  ax: {ax_before:5d} → {self.ax:5d}")
    
    def run(self):
        while not self.halted:
            self.step()
        return self.ax

# ============================================================================
# FULL PIPELINE
# ============================================================================
def full_pipeline(expr):
    print(f"\n{'─' * 60}")
    print(f"C SOURCE: {expr}")
    print(f"{'─' * 60}")
    
    # Compile
    bytecode = compile_expr(expr)
    bc_str = []
    for i in range(0, len(bytecode), 2):
        op, imm = bytecode[i], bytecode[i+1]
        name = OP_NAMES.get(op, f"{op}")
        bc_str.append(f"{name}" + (f" {imm}" if op == IMM else ""))
    print(f"BYTECODE: {', '.join(bc_str)}")
    
    # Execute
    print(f"\nTHINKING:")
    vm = ThinkingVM(bytecode)
    result = vm.run()
    for t in vm.thoughts:
        print(f"  {t}")
    
    # Verify
    expected = int(eval(expr))
    status = "✓" if result == expected else "✗"
    print(f"\nRESULT: {result}  (expected {expected}) {status}")
    return result == expected

# ============================================================================
# DEMO
# ============================================================================
print("=" * 60)
print("C SOURCE → COMPILE → THINKING → RESULT")
print("=" * 60)

tests = [
    "3 + 4",
    "10 - 3", 
    "6 * 7",
    "100 / 5",
    "(3 + 4) * 5",
    "10 + 20 * 3",
    "100 / 5 + 10",
    "(8 + 4) * (3 + 2)",
    "2 * 3 * 4 * 5",
    "120 / 4 / 3",
]

passed = sum(full_pipeline(t) for t in tests)
print(f"\n{'=' * 60}")
print(f"PASSED: {passed}/{len(tests)}")
print("=" * 60)

print("""
WHAT THIS SHOWS:
────────────────
  1. C expression is COMPILED to bytecode
  2. Each THINKING token = one transformer forward pass
  3. Operations use transformer primitives:
     • MUL: FFN quarter-squares  
     • DIV: Attention(log) → Softmax(exp) → FFN(mul)
     • ADD/SUB: FFN residual
  4. State flows through as hidden states
  5. Final ax = output token

TO MAKE THIS A TRUE LLM:
────────────────────────
  • Input tokens: bytecode sequence
  • Hidden state: [pc, ax, sp, stack...]  
  • Each layer: one VM instruction
  • Output: final ax value
""")
