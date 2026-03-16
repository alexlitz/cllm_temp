#!/usr/bin/env python3
"""
ASCII Byte-Tokenized VM

Source code tokenized as raw ASCII bytes.
Tokenizer = ord() for each character.

This is exactly how transformers see text!
"""

from c4_byte_to_nibble import C4ByteNibbleVM
from c4_compiler_full import compile_c

# =============================================================================
# ASCII BYTE TOKENIZATION
# =============================================================================

def tokenize(s: str) -> list:
    """Tokenize string as ASCII bytes. Just ord() for each char!"""
    return [ord(c) for c in s]

# =============================================================================
# BYTE-PARSING EXPRESSION EVALUATOR
# =============================================================================

# This C code reads ASCII bytes and evaluates expressions
# It handles precedence correctly for 3-term expressions

BYTE_EVAL = '''
// Evaluate expression from ASCII bytes
// Supports: single/double digit numbers, +, -, *, /
// Handles precedence for a op1 b op2 c

int parse_num(int d0, int d1) {
    // Parse 1-2 digit number from ASCII
    int n;
    n = 0;
    if (d0 >= 48) {
        if (d0 <= 57) {
            n = d0 - 48;
        }
    }
    if (d1 >= 48) {
        if (d1 <= 57) {
            n = n * 10 + d1 - 48;
        }
    }
    return n;
}

int is_op(int c) {
    if (c == 43) { return 1; }  // +
    if (c == 45) { return 1; }  // -
    if (c == 42) { return 1; }  // *
    if (c == 47) { return 1; }  // /
    return 0;
}

int apply_op(int a, int op, int b) {
    if (op == 43) { return a + b; }
    if (op == 45) { return a - b; }
    if (op == 42) { return a * b; }
    if (op == 47) { return a / b; }
    return 0;
}

int is_high_prec(int op) {
    // * and / have higher precedence
    if (op == 42) { return 1; }
    if (op == 47) { return 1; }
    return 0;
}

int eval_ascii(int c0, int c1, int c2, int c3, int c4, int c5, int c6, int c7) {
    // Parse and evaluate expression from 8 ASCII bytes
    // Format: num1 op1 num2 [op2 num3]

    int a, b, c;
    int op1, op2;
    int pos;
    int result;

    // Parse first number (1-2 digits)
    a = 0;
    pos = 0;
    if (c0 >= 48) {
        if (c0 <= 57) {
            a = c0 - 48;
            pos = 1;
            if (c1 >= 48) {
                if (c1 <= 57) {
                    a = a * 10 + c1 - 48;
                    pos = 2;
                }
            }
        }
    }

    // Get operator 1
    op1 = 0;
    if (pos == 1) { op1 = c1; pos = 2; }
    if (pos == 2) {
        if (is_op(c2)) { op1 = c2; pos = 3; }
    }

    // If no operator, return first number
    if (op1 == 0) {
        return a;
    }

    // Parse second number
    b = 0;
    if (pos == 2) {
        if (c2 >= 48) {
            if (c2 <= 57) {
                b = c2 - 48;
                pos = 3;
                if (c3 >= 48) {
                    if (c3 <= 57) {
                        b = b * 10 + c3 - 48;
                        pos = 4;
                    }
                }
            }
        }
    }
    if (pos == 3) {
        if (c3 >= 48) {
            if (c3 <= 57) {
                b = c3 - 48;
                pos = 4;
                if (c4 >= 48) {
                    if (c4 <= 57) {
                        b = b * 10 + c4 - 48;
                        pos = 5;
                    }
                }
            }
        }
    }

    // Get operator 2
    op2 = 0;
    if (pos == 3) { if (is_op(c3)) { op2 = c3; pos = 4; } }
    if (pos == 4) { if (is_op(c4)) { op2 = c4; pos = 5; } }
    if (pos == 5) { if (is_op(c5)) { op2 = c5; pos = 6; } }

    // If no second operator, evaluate a op1 b
    if (op2 == 0) {
        return apply_op(a, op1, b);
    }

    // Parse third number
    c = 0;
    if (pos == 4) { c = c4 - 48; if (c5 >= 48) { if (c5 <= 57) { c = c * 10 + c5 - 48; } } }
    if (pos == 5) { c = c5 - 48; if (c6 >= 48) { if (c6 <= 57) { c = c * 10 + c6 - 48; } } }
    if (pos == 6) { c = c6 - 48; if (c7 >= 48) { if (c7 <= 57) { c = c * 10 + c7 - 48; } } }

    // Handle precedence: a op1 b op2 c
    if (is_high_prec(op2)) {
        // Do op2 first: a op1 (b op2 c)
        result = apply_op(b, op2, c);
        return apply_op(a, op1, result);
    }
    if (is_high_prec(op1)) {
        // Do op1 first: (a op1 b) op2 c
        result = apply_op(a, op1, b);
        return apply_op(result, op2, c);
    }
    // Same precedence, left to right
    result = apply_op(a, op1, b);
    return apply_op(result, op2, c);
}

int main() {
    // Test: "3+4*2" = [51, 43, 52, 42, 50, 0, 0, 0]
    return eval_ascii(51, 43, 52, 42, 50, 0, 0, 0);
}
'''

# =============================================================================
# ASCII VM CLASS
# =============================================================================

class AsciiVM:
    """VM that takes ASCII string and evaluates it."""

    def __init__(self):
        self.vm = C4ByteNibbleVM()

    def eval(self, expr: str) -> int:
        """Evaluate expression string."""
        # Tokenize: just ord() for each character!
        tokens = tokenize(expr)

        # Pad to 8 bytes
        while len(tokens) < 8:
            tokens.append(0)
        tokens = tokens[:8]

        # Generate code with these byte values
        code = BYTE_EVAL.replace(
            'return eval_ascii(51, 43, 52, 42, 50, 0, 0, 0);',
            f'return eval_ascii({tokens[0]}, {tokens[1]}, {tokens[2]}, {tokens[3]}, {tokens[4]}, {tokens[5]}, {tokens[6]}, {tokens[7]});'
        )

        # Compile and run
        bytecode, data = compile_c(code)
        self.vm.reset()
        self.vm.load_bytecode(bytecode, data)
        return self.vm.run(max_steps=100000)


# =============================================================================
# DEMO
# =============================================================================

def demo():
    print("=" * 70)
    print("  ASCII BYTE-TOKENIZED VM")
    print("=" * 70)
    print()
    print("Tokenization: Just ord() for each character!")
    print()

    # Show tokenization examples
    examples = ["3+4", "6*7", "3+4*2", "10-3", "42"]
    print("Examples:")
    for s in examples:
        tokens = tokenize(s)
        print(f"  '{s}' → {tokens}")
    print()

    # Compile the ASCII evaluator
    print("Compiling ASCII byte evaluator...")
    bytecode, _ = compile_c(BYTE_EVAL)
    print(f"  Bytecode: {len(bytecode)} instructions")
    print()

    # Test
    print("Evaluating expressions:")
    print("-" * 50)

    avm = AsciiVM()

    tests = [
        ("3+4", 7),
        ("6*7", 42),
        ("9-5", 4),
        ("8/2", 4),
        ("3+4*2", 11),    # Precedence: 3 + (4*2) = 11
        ("10-2*3", 4),    # Precedence: 10 - (2*3) = 4
        ("2*3+4", 10),    # Precedence: (2*3) + 4 = 10
        ("12+34", 46),    # Multi-digit
        ("99/9", 11),     # Multi-digit division
    ]

    all_pass = True
    for expr, expected in tests:
        tokens = tokenize(expr)
        result = avm.eval(expr)
        status = "✓" if result == expected else "✗"
        if result != expected:
            all_pass = False
        print(f"  '{expr:8}' → {str(tokens)[:20]:20} = {result:4} (expected {expected:4}) {status}")

    print()
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"""
All tests: {'PASSED' if all_pass else 'FAILED'}

Architecture:
  "3+4*2" (string)
      │
      ▼ tokenize = ord()
  [51, 43, 52, 42, 50] (ASCII bytes)
      │
      ▼ Compiler bytecode
  Bytecode (runs on transformer)
      │
      ▼ SwiGLU multiply, FFN divide
  Result: 11

The input tokenization is just ord() - exactly like byte-level LLMs!
Vocab size = 256 (one token per byte)
No BPE, no special tokenizer needed.
""")


if __name__ == "__main__":
    demo()
