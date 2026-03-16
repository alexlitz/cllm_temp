"""
C4 Transformer System

Architecture:
- "System Prompt": C4 compiler (as Python function generating bytecode)
- Input: C source code
- Output: Program bytecode
- Transformer VM executes bytecode using only attention + SwiGLU

This demonstrates the full pipeline:
  C source → Compiler → Bytecode → Transformer VM → Result
"""

import torch
from c4_moe_vm import C4MoEVM, C4Op


# =============================================================================
# "SYSTEM PROMPT": C4 Compiler (generates bytecode for simple expressions)
# =============================================================================

class C4Compiler:
    """
    Minimal C4 compiler that compiles:
        int main() { return <expr>; }
    where <expr> is numbers, +, -, *, /, (), and produces bytecode.

    This is the "system prompt" - the compiler that runs conceptually
    as bytecode in the transformer VM.
    """

    # Opcodes
    IMM = 1
    PSH = 13
    ADD = 25
    SUB = 26
    MUL = 27
    DIV = 28
    MOD = 29
    EXIT = 38

    def __init__(self):
        self.src = ""
        self.pos = 0
        self.token = None
        self.token_value = 0
        self.bytecode = []

    def emit(self, op, imm=None):
        """Emit bytecode instruction."""
        if imm is not None:
            self.bytecode.append(op | (imm << 8))
        else:
            self.bytecode.append(op)

    def peek(self):
        """Peek at current character."""
        while self.pos < len(self.src) and self.src[self.pos] in ' \t\n\r':
            self.pos += 1
        if self.pos >= len(self.src):
            return '\0'
        return self.src[self.pos]

    def advance(self):
        """Advance to next character."""
        self.pos += 1

    def next_token(self):
        """Get next token."""
        while self.pos < len(self.src) and self.src[self.pos] in ' \t\n\r':
            self.pos += 1

        if self.pos >= len(self.src):
            self.token = 'EOF'
            return

        c = self.src[self.pos]

        # Number
        if c.isdigit():
            start = self.pos
            while self.pos < len(self.src) and self.src[self.pos].isdigit():
                self.pos += 1
            self.token_value = int(self.src[start:self.pos])
            self.token = 'NUM'
            return

        # Operators and punctuation
        self.pos += 1
        if c == '+': self.token = '+'
        elif c == '-': self.token = '-'
        elif c == '*': self.token = '*'
        elif c == '/': self.token = '/'
        elif c == '%': self.token = '%'
        elif c == '(': self.token = '('
        elif c == ')': self.token = ')'
        elif c == '{': self.token = '{'
        elif c == '}': self.token = '}'
        elif c == ';': self.token = ';'
        # Keywords (simplified)
        elif c == 'i':  # int
            self.pos += 2
            self.token = 'INT'
        elif c == 'm':  # main
            self.pos += 3
            self.token = 'MAIN'
        elif c == 'r':  # return
            self.pos += 5
            self.token = 'RETURN'
        else:
            self.token = c

    def factor(self):
        """Parse factor: NUM | '(' expr ')' | '-' factor"""
        if self.token == 'NUM':
            self.emit(self.IMM, self.token_value)
            self.next_token()
        elif self.token == '(':
            self.next_token()
            self.expr()
            self.next_token()  # skip ')'
        elif self.token == '-':
            self.next_token()
            self.emit(self.IMM, 0)
            self.emit(self.PSH)
            self.factor()
            self.emit(self.SUB)

    def term(self):
        """Parse term: factor (('*' | '/' | '%') factor)*"""
        self.factor()
        while self.token in ('*', '/', '%'):
            op = self.token
            self.emit(self.PSH)
            self.next_token()
            self.factor()
            if op == '*':
                self.emit(self.MUL)
            elif op == '/':
                self.emit(self.DIV)
            else:
                self.emit(self.MOD)

    def expr(self):
        """Parse expression: term (('+' | '-') term)*"""
        self.term()
        while self.token in ('+', '-'):
            op = self.token
            self.emit(self.PSH)
            self.next_token()
            self.term()
            if op == '+':
                self.emit(self.ADD)
            else:
                self.emit(self.SUB)

    def compile(self, source: str) -> list:
        """Compile C source to bytecode."""
        self.src = source
        self.pos = 0
        self.bytecode = []

        # Parse: int main() { return <expr>; }
        self.next_token()  # int
        self.next_token()  # main
        self.next_token()  # (
        self.next_token()  # )
        self.next_token()  # {
        self.next_token()  # return
        self.next_token()  # first token of expr

        self.expr()
        self.emit(self.EXIT)

        return self.bytecode


# =============================================================================
# FULL PIPELINE
# =============================================================================

def run_c4_pipeline(source: str, trace: bool = False) -> int:
    """
    Full C4 pipeline:
    1. Compile C source to bytecode (the "system prompt")
    2. Execute bytecode in transformer VM

    Args:
        source: C source code (e.g., "int main() { return 6 * 7; }")
        trace: Whether to print execution trace

    Returns:
        Program result (exit code)
    """
    # Step 1: Compile (the "system prompt" in action)
    compiler = C4Compiler()
    bytecode = compiler.compile(source)

    if trace:
        print(f"Source: {source}")
        print(f"Bytecode: {bytecode}")

    # Step 2: Execute in transformer VM
    vm = C4MoEVM()
    vm.load(bytecode, [])
    result, output, stats = vm.run(trace=trace)

    return result


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demonstrate the full C4 transformer pipeline."""
    print("=" * 70)
    print("C4 TRANSFORMER SYSTEM")
    print("=" * 70)
    print()
    print("Architecture:")
    print("  'System Prompt': C4 compiler (generates bytecode)")
    print("  Transformer VM: Executes bytecode using only attention + SwiGLU")
    print()
    print("Key operations (all transformer-compatible):")
    print("  MUL: silu(a)*b + silu(-a)*(-b) = a*b  [exact via sigmoid identity]")
    print("  DIV: log2 via attention, exp via softmax ratio  [exact]")
    print("  Memory: Binary-encoded attention with position-in-key")
    print("  Routing: 39 experts via eq_gate (SwiGLU)")
    print()
    print("=" * 70)
    print()

    tests = [
        ("int main() { return 42; }", 42),
        ("int main() { return 6 * 7; }", 42),
        ("int main() { return 100 / 5; }", 20),
        ("int main() { return 3 + 4; }", 7),
        ("int main() { return 10 - 3; }", 7),
        ("int main() { return (3 + 4) * 5; }", 35),
        ("int main() { return 100 / 5 + 10 * 3; }", 50),
        ("int main() { return 17 % 5; }", 2),
        ("int main() { return (100 - 20) / 4; }", 20),
        ("int main() { return 2 * 3 * 4 * 5; }", 120),
    ]

    print("Running programs through C4 transformer pipeline:")
    print()

    passed = 0
    for source, expected in tests:
        result = run_c4_pipeline(source)
        status = "✓" if result == expected else "✗"
        if result == expected:
            passed += 1

        # Extract expression
        expr = source.split("return")[1].split(";")[0].strip()
        print(f"  {expr:25s} = {result:4d}  (expected {expected:4d}) {status}")

    print()
    print(f"Results: {passed}/{len(tests)} passed")
    print()
    print("=" * 70)
    print("DETAILED EXAMPLE: 6 * 7")
    print("=" * 70)
    print()

    result = run_c4_pipeline("int main() { return 6 * 7; }", trace=True)
    print(f"\nFinal result: {result}")


if __name__ == "__main__":
    demo()
