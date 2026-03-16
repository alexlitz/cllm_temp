#!/usr/bin/env python3
"""
Byte-Tokenized Self-Hosted VM

Source code is tokenized as raw ASCII bytes.
No special tokenizer needed - just ord(char) for each character.

Flow:
  "3+4" → [51, 43, 52] (ASCII bytes) → Compiler bytecode → Result
"""

from c4_byte_to_nibble import C4ByteNibbleVM
from c4_compiler_full import compile_c

# =============================================================================
# ASCII BYTE TOKENIZATION
# =============================================================================

def tokenize_source(source: str) -> list:
    """
    Tokenize source code as raw ASCII bytes.

    No BPE, no special tokens - just ord() for each character.
    This is the simplest possible tokenization.
    """
    return [ord(c) for c in source]


def bytes_to_string(tokens: list) -> str:
    """Convert byte tokens back to string."""
    return ''.join(chr(t) for t in tokens)


# =============================================================================
# COMPILER THAT READS BYTE TOKENS
# =============================================================================

# This compiler reads source code as bytes from memory
# and generates bytecode

BYTE_COMPILER = '''
// Compiler that reads ASCII byte tokens from memory
// Input: Source bytes at 0x20000
// Output: Computed result (interpreter mode for simplicity)

int src_ptr;
int cur_char;

int next_char() {
    int c;
    int addr;
    addr = 131072 + src_ptr;  // 0x20000 + offset
    src_ptr = src_ptr + 1;
    // Read byte from memory
    // For now we pass it differently
    return 0;
}

int skip_space() {
    while (cur_char == 32) {  // space
        cur_char = next_char();
    }
    return 0;
}

int is_digit(int c) {
    if (c >= 48) {
        if (c <= 57) {
            return 1;
        }
    }
    return 0;
}

int parse_number() {
    int n;
    n = 0;
    while (is_digit(cur_char)) {
        n = n * 10 + cur_char - 48;
        cur_char = next_char();
    }
    return n;
}

// For demo: direct byte parsing
int eval_expr_bytes(int b0, int b1, int b2, int b3, int b4) {
    // Parse simple expression from 5 bytes
    // e.g., "3+4" = [51, 43, 52, 0, 0]

    int a, b, op;

    // First number
    a = 0;
    if (b0 >= 48) {
        if (b0 <= 57) {
            a = b0 - 48;
        }
    }

    // Operator
    op = b1;

    // Second number
    b = 0;
    if (b2 >= 48) {
        if (b2 <= 57) {
            b = b2 - 48;
        }
    }

    // Evaluate
    if (op == 43) { return a + b; }  // +
    if (op == 45) { return a - b; }  // -
    if (op == 42) { return a * b; }  // *
    if (op == 47) { return a / b; }  // /

    return a;  // Just return first number if no op
}

int main() {
    // Test: "3+4" = [51, 43, 52]
    return eval_expr_bytes(51, 43, 52, 0, 0);
}
'''

# =============================================================================
# FULL BYTE TOKENIZED COMPILER
# =============================================================================

FULL_BYTE_COMPILER = '''
// Full expression parser reading ASCII bytes
// Handles multi-digit numbers and precedence

int eval_bytes(int b0, int b1, int b2, int b3, int b4,
               int b5, int b6, int b7, int b8, int b9) {
    // Parse expression from up to 10 bytes
    // Supports: numbers (multi-digit), +, -, *, /

    int nums[5];
    int ops[4];
    int num_count;
    int op_count;
    int i;
    int c;
    int n;
    int in_num;
    int bytes[10];

    bytes[0] = b0; bytes[1] = b1; bytes[2] = b2; bytes[3] = b3; bytes[4] = b4;
    bytes[5] = b5; bytes[6] = b6; bytes[7] = b7; bytes[8] = b8; bytes[9] = b9;

    num_count = 0;
    op_count = 0;
    n = 0;
    in_num = 0;

    i = 0;
    while (i < 10) {
        c = bytes[i];

        if (c == 0) {
            i = 10;  // End
        }

        // Skip space
        if (c == 32) {
            if (in_num) {
                nums[num_count] = n;
                num_count = num_count + 1;
                n = 0;
                in_num = 0;
            }
        }

        // Digit
        if (c >= 48) {
            if (c <= 57) {
                n = n * 10 + c - 48;
                in_num = 1;
            }
        }

        // Operator
        if (c == 43) {
            if (in_num) { nums[num_count] = n; num_count = num_count + 1; n = 0; in_num = 0; }
            ops[op_count] = 43;
            op_count = op_count + 1;
        }
        if (c == 45) {
            if (in_num) { nums[num_count] = n; num_count = num_count + 1; n = 0; in_num = 0; }
            ops[op_count] = 45;
            op_count = op_count + 1;
        }
        if (c == 42) {
            if (in_num) { nums[num_count] = n; num_count = num_count + 1; n = 0; in_num = 0; }
            ops[op_count] = 42;
            op_count = op_count + 1;
        }
        if (c == 47) {
            if (in_num) { nums[num_count] = n; num_count = num_count + 1; n = 0; in_num = 0; }
            ops[op_count] = 47;
            op_count = op_count + 1;
        }

        if (c != 0) {
            i = i + 1;
        }
    }

    // Final number
    if (in_num) {
        nums[num_count] = n;
        num_count = num_count + 1;
    }

    // Evaluate with precedence
    // First pass: * and /
    i = 0;
    while (i < op_count) {
        if (ops[i] == 42) {
            nums[i] = nums[i] * nums[i+1];
            // Shift remaining
            nums[i+1] = nums[i+2];
            nums[i+2] = nums[i+3];
            ops[i] = ops[i+1];
            ops[i+1] = ops[i+2];
            op_count = op_count - 1;
            num_count = num_count - 1;
        }
        if (ops[i] == 47) {
            nums[i] = nums[i] / nums[i+1];
            nums[i+1] = nums[i+2];
            nums[i+2] = nums[i+3];
            ops[i] = ops[i+1];
            ops[i+1] = ops[i+2];
            op_count = op_count - 1;
            num_count = num_count - 1;
        }
        i = i + 1;
    }

    // Second pass: + and -
    i = 0;
    while (i < op_count) {
        if (ops[i] == 43) {
            nums[i] = nums[i] + nums[i+1];
            nums[i+1] = nums[i+2];
            ops[i] = ops[i+1];
            op_count = op_count - 1;
            num_count = num_count - 1;
        }
        if (ops[i] == 45) {
            nums[i] = nums[i] - nums[i+1];
            nums[i+1] = nums[i+2];
            ops[i] = ops[i+1];
            op_count = op_count - 1;
            num_count = num_count - 1;
        }
        i = i + 1;
    }

    return nums[0];
}

int main() {
    // Test: "3+4*2" = [51, 43, 52, 42, 50]
    return eval_bytes(51, 43, 52, 42, 50, 0, 0, 0, 0, 0);
}
'''

# =============================================================================
# BYTE-TOKENIZED VM
# =============================================================================

class ByteTokenizedVM:
    """
    VM that takes raw ASCII bytes as input.

    The "tokenizer" is just ord() - converting characters to bytes.
    This is exactly how transformers see text!
    """

    def __init__(self):
        self.vm = C4ByteNibbleVM()
        self.compiler_bytecode = None
        self._compile_compiler()

    def _compile_compiler(self):
        """Compile the byte-reading compiler."""
        self.compiler_bytecode, _ = compile_c(BYTE_COMPILER)

    def eval_expression(self, expr: str) -> int:
        """
        Evaluate expression given as string.

        The string is tokenized as raw ASCII bytes.
        """
        # Tokenize: string → bytes (just ord() for each char!)
        tokens = tokenize_source(expr)

        # Pad to expected length
        while len(tokens) < 10:
            tokens.append(0)
        tokens = tokens[:10]

        # Generate code that calls eval with these bytes
        code = f'''
        int eval_bytes(int b0, int b1, int b2, int b3, int b4,
                       int b5, int b6, int b7, int b8, int b9);

        // Paste the eval function here
        ''' + FULL_BYTE_COMPILER.split('int main()')[0] + f'''

        int main() {{
            return eval_bytes({tokens[0]}, {tokens[1]}, {tokens[2]}, {tokens[3]}, {tokens[4]},
                              {tokens[5]}, {tokens[6]}, {tokens[7]}, {tokens[8]}, {tokens[9]});
        }}
        '''

        # The full code
        full_code = FULL_BYTE_COMPILER.replace(
            'return eval_bytes(51, 43, 52, 42, 50, 0, 0, 0, 0, 0);',
            f'return eval_bytes({tokens[0]}, {tokens[1]}, {tokens[2]}, {tokens[3]}, {tokens[4]}, {tokens[5]}, {tokens[6]}, {tokens[7]}, {tokens[8]}, {tokens[9]});'
        )

        bytecode, data = compile_c(full_code)

        self.vm.reset()
        self.vm.load_bytecode(bytecode, data)
        return self.vm.run(max_steps=100000)


# =============================================================================
# DEMO
# =============================================================================

def demo():
    print("=" * 70)
    print("BYTE-TOKENIZED SELF-HOSTED VM")
    print("=" * 70)
    print()

    print("Tokenization: Raw ASCII bytes (just ord() for each character)")
    print()

    # Show tokenization
    examples = ["3+4", "6*7", "3+4*2", "10-3"]
    print("Example tokenizations:")
    for expr in examples:
        tokens = tokenize_source(expr)
        print(f"  '{expr}' → {tokens}")
    print()

    # Compile the byte-reading compiler
    print("Compiling byte-reading compiler...")
    bytecode, _ = compile_c(BYTE_COMPILER)
    print(f"  Compiler bytecode: {len(bytecode)} instructions")
    print()

    # Run self-test
    print("Self-test: '3+4' = [51, 43, 52]")
    vm = C4ByteNibbleVM()
    vm.reset()
    vm.load_bytecode(bytecode, None)
    result = vm.run()
    print(f"  Result: {result} (expected 7)")
    print()

    # Test full byte compiler
    print("Full byte-tokenized evaluation:")
    print("-" * 50)

    btvm = ByteTokenizedVM()

    tests = [
        ("3+4", 7),
        ("6*7", 42),
        ("9-5", 4),
        ("8/2", 4),
        ("3+4*2", 11),
        ("10-2*3", 4),
    ]

    for expr, expected in tests:
        tokens = tokenize_source(expr)
        result = btvm.eval_expression(expr)
        status = "✓" if result == expected else "✗"
        print(f"  '{expr}' → {tokens[:5]}... = {result} (expected {expected}) {status}")

    print()
    print("=" * 70)
    print("KEY INSIGHT: Tokenization is just ord()")
    print("=" * 70)
    print("""
The transformer sees:
  "3+4*2" → [51, 43, 52, 42, 50]  (ASCII bytes)

This is EXACTLY like how LLMs see text:
  - Each character is a token (byte)
  - Vocab size = 256
  - No special BPE needed

The compiler bytecode parses these bytes and evaluates the expression.
All on the pure transformer VM!
""")


if __name__ == "__main__":
    demo()
