#!/usr/bin/env python3
"""
FULL PIPELINE: C Source → Compile → Bytecode → Thinking → Result
"""
import torch
import torch.nn.functional as F
import math

# Opcodes
IMM, PSH, ADD, SUB, MUL, DIV, EXIT = 1, 13, 25, 26, 27, 28, 38
OP_NAMES = {1: "IMM", 13: "PSH", 25: "ADD", 26: "SUB", 27: "MUL", 28: "DIV", 38: "EXIT"}

def compile_expr(expr):
    """Compile arithmetic expression to bytecode."""
    tokens = []
    i = 0
    expr = expr.replace(" ", "")
    while i < len(expr):
        if expr[i].isdigit():
            j = i
            while j < len(expr) and expr[j].isdigit():
                j += 1
            tokens.append(int(expr[i:j]))
            i = j
        else:
            tokens.append(expr[i])
            i += 1
    
    pos = [0]
    code = []
    
    def peek():
        return tokens[pos[0]] if pos[0] < len(tokens) else None
    
    def consume():
        t = peek()
        pos[0] += 1
        return t
    
    def parse_atom():
        t = peek()
        if t == '(':
            consume()
            parse_additive()
            consume()  # ')'
        elif isinstance(t, int):
            code.extend([IMM, consume()])
    
    def parse_mult():
        parse_atom()
        while peek() in ('*', '/'):
            op = consume()
            code.extend([PSH, 0])
            parse_atom()
            code.extend([MUL if op == '*' else DIV, 0])
    
    def parse_additive():
        parse_mult()
        while peek() in ('+', '-'):
            op = consume()
            code.extend([PSH, 0])
            parse_mult()
            code.extend([ADD if op == '+' else SUB, 0])
    
    parse_additive()
    code.extend([EXIT, 0])
    return code

# Transformer primitives
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
    b_t = torch.clamp(torch.tensor(float(b)), min=1.0, max=float(MAX_VAL))
    scores = torch.stack([eq_gate(b_t, k, 30) for k in log_keys])
    log_b = (F.softmax(scores * 100, dim=0) * log_values).sum()
    logits = torch.stack([-log_b * math.log(2), torch.tensor(0.0)])
    probs = F.softmax(logits, dim=0)
    inv_b = probs[0] / probs[1]
    result = a * inv_b
    if torch.abs(result - torch.round(result)) < 0.001:
        result = torch.round(result)
    return torch.floor(result)

class ThinkingVM:
    def __init__(self, code):
        self.code = code
        self.pc = self.ax = 0
        self.stack = [0] * 256
        self.sp = 256
        self.halted = False
        self.thoughts = []
    
    def run(self):
        while not self.halted and self.pc < len(self.code):
            op, imm = self.code[self.pc], self.code[self.pc + 1]
            ax_before = self.ax
            self.pc += 2
            
            if op == IMM: self.ax = imm
            elif op == PSH: self.sp -= 1; self.stack[self.sp] = self.ax
            elif op == ADD: self.ax = self.stack[self.sp] + self.ax; self.sp += 1
            elif op == SUB: self.ax = self.stack[self.sp] - self.ax; self.sp += 1
            elif op == MUL: 
                self.ax = int(mul_ffn(torch.tensor(float(self.stack[self.sp])), torch.tensor(float(self.ax))).item())
                self.sp += 1
            elif op == DIV:
                self.ax = int(div_log_exp(torch.tensor(float(self.stack[self.sp])), torch.tensor(float(self.ax))).item())
                self.sp += 1
            elif op == EXIT: self.halted = True
            
            self.thoughts.append((OP_NAMES.get(op, "?"), imm, ax_before, self.ax))
        return self.ax

def demo(expr):
    print(f"\n{'─'*50}\nC: {expr}")
    bc = compile_expr(expr)
    print(f"BC: {[OP_NAMES.get(bc[i], bc[i]) + (f' {bc[i+1]}' if bc[i]==IMM else '') for i in range(0, len(bc), 2)]}")
    vm = ThinkingVM(bc)
    result = vm.run()
    print("Thinking:", end=" ")
    for op, imm, before, after in vm.thoughts:
        if op == "IMM": print(f"[{op} {imm}→ax={after}]", end=" ")
        elif op in ("ADD","SUB","MUL","DIV"): print(f"[{op}→{after}]", end=" ")
    print(f"\nResult: {result} (expected {int(eval(expr))}) {'✓' if result == int(eval(expr)) else '✗'}")

print("=" * 50)
print("C → BYTECODE → THINKING → RESULT")
print("=" * 50)
for e in ["3+4", "6*7", "100/5", "(3+4)*5", "2*3*4", "120/4/3", "(8+4)*(3+2)"]:
    demo(e)
